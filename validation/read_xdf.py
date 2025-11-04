import io
import time
import urllib
import warnings
from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyxdf
import scipy
import scipy.interpolate
import scipy.signal

# ========================================================================================
# Helper Functions for Resampling
# ========================================================================================


# def _create_timestamps(stream_data, target_fs):
#     """
#     Creates a new, regularly spaced timestamp vector based on the global
#     min/max time of all streams and the target sampling rate.
#     """
#     if target_fs <= 0:
#         raise ValueError("target_fs must be positive.")

#     dt = 1.0 / target_fs

#     # Find the global time range
#     global_min_ts = min([s["timestamps"].min() for s in stream_data])
#     global_max_ts = max([s["timestamps"].max() for s in stream_data])

#     # Create the new timestamp vector
#     new_timestamps = np.arange(global_min_ts, global_max_ts + dt, dt)
#     return new_timestamps


def _create_timestamps(stream_data, target_fs):
    """
    Creates a new, regularly spaced timestamp vector "anchored" to the
    stream with the highest effective sampling rate.

    This minimizes interpolation error for the fastest stream by aligning
    the new grid's phase with its existing timestamps. The grid is
    guaranteed to cover the global min/max time of all streams.
    """
    if target_fs <= 0:
        raise ValueError("target_fs must be positive.")

    dt = 1.0 / target_fs

    # 1. Find the global time range (still needed)
    global_min_ts = min([s["timestamps"].min() for s in stream_data])
    global_max_ts = max([s["timestamps"].max() for s in stream_data])

    # 2. Find the "reference" stream (highest effective srate)
    #    We check for len > 1 to avoid divide-by-zero on effective_srate
    #    for single-sample streams.
    try:
        ref_stream = max(
            [s for s in stream_data if len(s["timestamps"]) > 1],
            key=lambda s: s["effective_srate"],
        )
        anchor_ts = ref_stream["timestamps"][0]

    except ValueError:
        # Fallback: No streams have > 1 sample.
        # Revert to the original "ignorant" grid behavior.
        warnings.warn("Could not find a reference stream with > 1 sample. " "Reverting to un-anchored grid.")
        anchor_ts = global_min_ts

    # 3. Calculate the new start time based on the anchor
    #    We need to find a t_start that is <= global_min_ts
    #    AND is an integer number of steps (dt) away from the anchor.

    # Calculate how far back from the anchor we need to go
    time_before_anchor = anchor_ts - global_min_ts

    # Calculate how many steps (dt) this requires, rounding *up*
    # to ensure we at least cover the global_min_ts.
    # We add a small epsilon to handle potential float precision issues
    # where (time_before_anchor / dt) is *exactly* an integer.
    epsilon = 1e-9
    steps_back = np.ceil((time_before_anchor / dt) + epsilon)

    # Calculate the new, aligned start time
    t_start = anchor_ts - (steps_back * dt)

    # 4. Create the new timestamp vector
    #    The 'stop' condition (global_max_ts + dt) ensures the
    #    last point is >= global_max_ts.
    new_timestamps = np.arange(t_start, global_max_ts + dt, dt)

    return new_timestamps


# ========================================================================================
# NOTE: _interpolate_splat_ffill has been removed as it is no longer
# required. All streams are numeric and are handled by _interpolate_streams.
# ========================================================================================


def _interpolate_streams(stream_data, new_timestamps, all_columns, col_to_idx, dtype):
    """
    Performs efficient interpolation for all numeric streams.

    It dynamically chooses the interpolation method per stream:
    - 'linear': For streams with > 2 unique values (continuous data).
    - 'previous': For streams with <= 2 unique values (event markers).

    Args:
        stream_data (list): List of stream dictionaries (numeric only).
        new_timestamps (np.ndarray): The target regular timestamp grid.
        all_columns (list): List of all column names for the output array.
        col_to_idx (dict): Map of column name to output array index.
        dtype (type): The dtype for the output. Must be a floating type.

    Returns:
        np.ndarray: A (len(new_timestamps), len(all_columns)) array with
                    interpolated data.
    """
    # This function is only intended for numeric data.
    if not np.issubdtype(dtype, np.floating):
        raise ValueError("_interpolate_streams can only be used for numeric (float) dtypes.")

    # 1. Create the empty (NaN-filled) data grid
    resampled_data = np.full((len(new_timestamps), len(all_columns)), np.nan, dtype=np.float64)

    # 2. Iterate over each *original* stream and interpolate
    for s in stream_data:
        original_ts = s["timestamps"]
        original_data = s["data"]
        col_indices = [col_to_idx[c] for c in s["columns"]]

        # Handle edge case: stream with 0 or 1 samples
        if len(original_ts) < 2:
            if len(original_ts) == 1:
                # Fallback to "splat" (nearest) for single points
                insertion_idx = np.searchsorted(new_timestamps, original_ts[0], side="left")
                left_idx = np.clip(insertion_idx - 1, 0, len(new_timestamps) - 1)
                right_idx = np.clip(insertion_idx, 0, len(new_timestamps) - 1)
                is_left_closer = (original_ts[0] - new_timestamps[left_idx]) <= (
                    new_timestamps[right_idx] - original_ts[0]
                )
                closest_idx = left_idx if is_left_closer else right_idx
                resampled_data[closest_idx, col_indices] = original_data[0]
            continue  # Skip to next stream

        # --- Determine interpolation kind based on unique values ---
        # This implements the "event marker" logic.
        num_unique = np.unique(s["data"]).size
        if num_unique > 2:
            interp_kind = "linear"
        else:
            # Use 'previous' (forward-fill / zero-order hold)
            # for streams with <= 2 unique values (e.g., event markers)
            interp_kind = "previous"

        # --- Use scipy.interpolate.interp1d for efficiency ---
        # This interpolates all channels in the stream at once (axis=0).
        # bounds_error=False and fill_value=np.nan ensures that
        # new_timestamps outside the range of original_ts become NaN.
        try:
            interpolator = scipy.interpolate.interp1d(
                original_ts,
                original_data,
                axis=0,
                kind=interp_kind,  # <-- Use dynamic kind
                bounds_error=False,
                fill_value=np.nan,
            )

            # Apply the interpolator to the new timestamps
            interpolated_data_block = interpolator(new_timestamps)

            # Place the interpolated data block into the final grid
            resampled_data[:, col_indices] = interpolated_data_block

        except ValueError as e:
            # This can happen if timestamps are not monotonically increasing
            warnings.warn(
                f"Interpolation failed for stream '{s['name']}'. "
                f"Timestamps may not be monotonic. Skipping. Error: {e}"
            )
            continue

    return resampled_data


def _fill_missing_data(resampled_df, fill_method="ffill", fill_value=0):
    """
    Fills NaN values in the resampled DataFrame.

    'fill_method':
        - 'ffill': Forward fill
        - 'bfill': Backward fill
        - None: Do not time-based fill
    'fill_value':
        - Value to fill any remaining NaNs (e.g., at the start)
    """
    if fill_method == "ffill":
        resampled_df = resampled_df.ffill()
    elif fill_method == "bfill":
        resampled_df = resampled_df.bfill()

    # Fill any remaining NaNs (e.g., at the very beginning)
    if fill_value is not None:
        resampled_df = resampled_df.fillna(fill_value)

    # After filling, infer the best possible dtypes to silence FutureWarning
    # copy=False modifies the df in place if possible
    resampled_df = resampled_df.infer_objects(copy=False)

    return resampled_df


# ========================================================================================
# Main Resampling Function
# ========================================================================================


def _resample_streams(stream_data, target_fs, fill_method="ffill", fill_value=0):
    """
    Resamples and merges multiple XDF streams into a single DataFrame using
    dynamic interpolation (linear or 'previous') and forward-filling.

    Args:
        stream_data (list): List of stream dictionaries from the loading phase.
        target_fs (float): The target sampling rate in Hz.
        fill_method (str): Method for filling NaNs ('ffill', 'bfill', None).
        fill_value (any): Value to fill remaining NaNs (e.g., 0 or np.nan).

    Returns:
        pd.DataFrame: A single DataFrame with all streams resampled and merged.
    """
    # Unpack column names
    cols = [col for s in stream_data for col in s["columns"]]

    # Create name-to-index mappings for each type
    col_to_idx = {name: i for i, name in enumerate(cols)}

    # Create the target *regular* timestamp grid (once)
    new_ts = _create_timestamps(stream_data, target_fs)

    # Process all streams using the dynamic interpolation function
    data = _interpolate_streams(
        stream_data,
        new_ts,
        cols,
        col_to_idx,
        dtype=np.float64,
    )

    # Create DataFrame with specific dtype to save memory
    resampled_df = pd.DataFrame(data, index=new_ts, columns=cols, dtype=np.float64)

    # Fill NaNs (e.g., at the beginning) and return
    resampled_df = _fill_missing_data(resampled_df, fill_method, fill_value)

    return resampled_df


# ========================================================================================
# Quality Control Functions
# ========================================================================================


def _visual_control(original, resampled, window_start=None, window_length=2.0, ax=None):
    """
    Helper for plotting a window of original vs. resampled data.
    If 'ax' is provided, it plots onto that axis.
    Otherwise, it creates a new figure and axis.
    """
    # If no axis is provided, create a new figure and axis
    # This maintains old behavior
    show_plot = False
    if ax is None:
        plt.figure(figsize=(15, 5))
        ax = plt.gca()  # Get current axis
        show_plot = True  # We are responsible for plt.show()

    if window_start is None:
        # Default to plotting a window in the middle of the data
        window_start = original.index[int(len(original) / 2)]
    window_end = window_start + window_length

    # Select the time window
    signal = original[(original.index >= window_start) & (original.index <= window_end)]
    resampled = resampled[(resampled.index >= window_start) & (resampled.index <= window_end)]

    ax.plot(signal.index, signal, "o-", label="original", alpha=0.7, markersize=3)
    ax.plot(resampled.index, resampled, ".-", label="resampled", alpha=0.7, markersize=2)
    ax.legend()
    ax.set_title(f"Visual Control: {original.name}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)

    # If we created the figure, we must show it.
    if show_plot:
        plt.show()


def _quality_control(stream, resampled_df, show=False):
    """
    Performs QC by calculating a closeness score (MAE) for all channels
    in the stream and optionally plotting the first channel.

    Args:
        stream (dict): A single stream dictionary from the stream_data list.
        resampled_df (pd.DataFrame): The output of resample_streams.
        show (bool): If True, generate and show the plot for the first channel.

    Returns:
        dict: A dictionary mapping channel names to their MAE scores.
    """
    scores = {}

    for i, col_name in enumerate(stream["columns"]):
        # 1. Get original data for this channel
        original_ts = stream["timestamps"]
        original_data = stream["data"][:, i]

        # 2. Find the corresponding resampled column
        if col_name not in resampled_df.columns:
            warnings.warn(f"Could not find column '{col_name}' in resampled data for QC.")
            continue

        resampled_series = resampled_df[col_name]
        resampled_ts = resampled_series.index
        resampled_data = resampled_series.values

        # 3. Calculate score: Interpolate original onto resampled time axis
        # This creates a 1-to-1 comparison for calculating error

        # --- Determine interpolation kind for QC comparison ---
        # We must use the *same* interpolation method for the QC check
        # as was used in the resampling step.
        num_unique = np.unique(stream["data"]).size
        if num_unique > 2:
            # For linear, we np.interp (which is linear)
            original_interp = np.interp(
                resampled_ts,
                original_ts,
                original_data,
                left=np.nan,  # Use NaN for areas where original doesn't cover
                right=np.nan,
            )
        else:
            # For 'previous', we must use 'previous'
            interp_func = scipy.interpolate.interp1d(
                original_ts,
                original_data,
                kind="previous",
                bounds_error=False,
                fill_value=np.nan,
            )
            original_interp = interp_func(resampled_ts)

        # 4. Calculate Mean Absolute Error, ignoring NaNs
        diff = (original_interp - resampled_data) ** 2
        mae = np.nanmean(diff)

        # 5. Calculate Normalized MAE (as a ratio)
        signal_range = np.nanmax(original_data) - np.nanmin(original_data)

        if signal_range > 1e-9:  # Avoid division by zero for constant signals
            normalized_mae = mae / signal_range
        else:
            # If range is zero, MAE should also be zero (or near-zero)
            normalized_mae = 0.0 if np.allclose(mae, 0) else np.inf

        scores[col_name] = normalized_mae

        # 5. Plot the first channel if show=True
        if i == 0 and show:
            # Create a pandas Series for the original data (for plotting)
            original_series = pd.Series(original_data, index=original_ts, name=col_name)
            original_series.index.name = "timestamps"

            # This will create its own figure by default
            _visual_control(original_series, resampled_series, window_start=None, window_length=2.0)

    return scores


# ========================================================================================
# Main function
# ========================================================================================
def synchronize_streams(
    stream_data,
    upsample_factor=2.0,
    fill_method="ffill",
    fill_value=0,
    show=None,
    window_start=None,
):
    """
    - upsample_factor: Factor to multiply max nominal srate by.
    - fill_method: 'ffill', 'bfill', or None
    - fill_value: Value for remaining NaNs
    - show (list or None): List of channel names to plot on a single figure.
                           If None, no plots are generated.
    """
    # --- Compute Target Sampling Rate ---
    target_fs = int(np.max([s["nominal_srate"] for s in stream_data]) * upsample_factor)

    print(f"Target sampling rate: {target_fs} Hz")

    # --- Run Resampling ---
    start_time = time.time()
    resampled_df = _resample_streams(stream_data, target_fs=target_fs, fill_method=fill_method, fill_value=fill_value)
    duration = time.time() - start_time

    print(f"Resampling complete in {duration:.2f} seconds.")

    # --- Run Quality Control ---
    # Get scores for each stream, but disable plotting from here (show=False)
    print("\n--- Quality Control ---")
    for stream in stream_data:
        print(f"Running QC for stream: {stream['name']}")
        # Call QC to get scores, but NOT to plot (show=False)
        scores = _quality_control(stream, resampled_df, show=False)

        # Calculate and print only the average score
        all_scores = list(scores.values())
        # Calculate average, ignoring potential NaNs/Infs
        avg_score = np.nanmean([s for s in all_scores if np.isfinite(s)])

        print(f"  - Average N-MAE: {avg_score}\n")

    # --- Custom Subplot Generation ---
    if show is not None and isinstance(show, list) and len(show) > 0:
        print(f"\nGenerating custom plot for {len(show)} specified channels...")

        if window_start is None:
            window_start = resampled_df.index[int(len(resampled_df) / 2)]

        n_plots = len(show)
        # Create a figure with N subplots, sharing the X-axis
        fig, axes = plt.subplots(n_plots, 1, figsize=(15, 4 * n_plots), sharex=True)

        # Ensure 'axes' is always an iterable array, even if n_plots=1
        if n_plots == 1:
            axes = [axes]

        # Build a lookup map for original stream data (more efficient)
        original_data_map = {}
        for s in stream_data:
            for i, col_name in enumerate(s["columns"]):
                original_data_map[col_name] = {
                    "timestamps": s["timestamps"],
                    "data": s["data"][:, i],
                }

        # Plot each requested channel on its subplot
        for ax, channel_name in zip(axes, show):
            if channel_name not in original_data_map or channel_name not in resampled_df.columns:
                warnings.warn(f"Channel '{channel_name}' not found in data. Skipping plot.")
                ax.set_title(f"Channel '{channel_name}' - NOT FOUND")
                ax.grid(True)
                continue

            # Get original data and create Series
            original_info = original_data_map[channel_name]
            original_series = pd.Series(
                original_info["data"],
                index=original_info["timestamps"],
                name=channel_name,
            )
            original_series.index.name = "timestamps"

            # Get resampled data (it's already a Series)
            resampled_series = resampled_df[channel_name]

            # Call the visual control helper, passing the specific axis
            _visual_control(original_series, resampled_series, ax=ax, window_start=window_start)

        # Tidy up the figure
        fig.tight_layout()
        plt.show()

    return resampled_df


# ========================================================================================
# Main Script Execution
# ========================================================================================

# --- Check raw data ---
# import OpenMuse


# --- Configuration ---
filename = "./test-18-dev.xdf"
dejitter_timestamps = ["OpenSignals"]

# --- Load Data ---
streams, header = pyxdf.load_xdf(
    filename,
    synchronize_clocks=True,
    handle_clock_resets=True,
    dejitter_timestamps=False,
)

# De-jitter timestamps for specified streams
streams_to_dejitter = []
for i, s in enumerate(streams):
    name = s["info"].get("name", ["Unnamed"])[0]
    if name in dejitter_timestamps:
        streams_to_dejitter.append(i)
if len(streams_to_dejitter) > 0:
    streams2, _ = pyxdf.load_xdf(
        filename,
        synchronize_clocks=True,
        handle_clock_resets=True,
        dejitter_timestamps=True,
    )
    for i in streams_to_dejitter:
        streams[i] = streams2[i]


# TEMPORARY HOTFIX FOR tests with gaps
for i, s in enumerate(streams):
    name = s["info"].get("name", ["Unnamed"])[0]
    if "Muse_" in name:
        ts = s["time_stamps"]
        diffs = np.diff(s["time_stamps"])
        if max(diffs) > 100 * np.median(diffs):
            print(f"Hotfix: Adjusting timestamps for stream {i} - {name}")
            # Find indices where large jumps occur
            jump_indices = np.where(diffs > 1000 * np.median(diffs))[0]
            # Replace with median
            for idx in jump_indices:
                streams[i]["time_stamps"][idx + 1 :] = ts[idx + 1 :] - diffs[idx] + np.median(diffs)
streams.pop(1)


# --- Pre-processing & Sanity Checks ---

# Warn if any stream has no time_stamps
for i, stream in enumerate(streams):
    name = stream["info"].get("name", ["Unnamed"])[0]
    if len(stream["time_stamps"]) == 0:
        warnings.warn(f"Stream {i} - {name} has no time_stamps. Dropping it.")
# Drop streams with no timestamps
streams = [s for s in streams if len(s["time_stamps"]) > 0]


# Get smaller time stamp to later use as offset (zero point)
min_ts = min([min(s["time_stamps"]) for s in streams])

# Make sure the length of all streams are roughly within the same range
ts_mins = np.array([stream["time_stamps"].min() for stream in streams])
ts_maxs = np.array([stream["time_stamps"].max() for stream in streams])
ts_durations = ts_maxs - ts_mins
duration_diffs = np.abs(ts_durations[:, np.newaxis] - ts_durations[np.newaxis, :])
if np.any(duration_diffs > 7200):  # 2 hours
    warnings.warn("Some streams differ in duration by more than 2 hours. This might be indicative of an issue.")

# --- Convert to common format (list of dicts) ---
stream_data = []
for stream in streams:
    # Get columns names
    try:
        channels_info = stream["info"]["desc"][0]["channels"][0]["channel"]
        cols = [channels_info[i]["label"][0] for i in range(len(channels_info))]
    except (KeyError, TypeError, IndexError):
        cols = [f"CHANNEL_{i}" for i in range(np.array(stream["time_series"]).shape[1])]
        warnings.warn(f"Using default channel names for stream: {stream['info'].get('name', ['Unnamed'])[0]}")

    name = stream["info"].get("name", ["Unnamed"])[0]
    timestamps = stream["time_stamps"] - min_ts  # Offset to zero
    data = np.array(stream["time_series"])

    # If duplicate timestamps exist, take last occurrence
    unique_ts, unique_indices = np.unique(timestamps, return_index=True)
    # Use efficient numpy indexing instead of list comprehension
    data = data[unique_indices]
    timestamps = unique_ts

    # Ensure data is 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # --- Ensure data is numeric ---
    if not np.issubdtype(data.dtype, np.number):
        # Data is not numeric (e.g., object type with strings). Process column by column.
        processed_cols = []
        for col_idx in range(data.shape[1]):
            column_data = data[:, col_idx]
            try:
                # 1. Try to convert to numeric (e.g., "0", "1.5")
                processed_cols.append(column_data.astype(float))
            except (ValueError, TypeError):
                # 2. Failed: map unique strings to integers
                warnings.warn(
                    f"Stream '{name}', column {col_idx} has non-numeric strings. "
                    f"Converting to integers by alphabetical order."
                )
                # Find unique, sorted strings
                unique_strings = sorted(np.unique(column_data.astype(str)))
                # Create mapping
                string_to_int_map = {s: i for i, s in enumerate(unique_strings)}
                # Apply mapping
                mapped_col = np.array([string_to_int_map[s] for s in column_data])
                processed_cols.append(mapped_col)

        # Recombine columns into a 2D numpy array
        data = np.stack(processed_cols, axis=1)
    # --- End of new block ---

    if data.shape[0] != len(timestamps):
        warnings.warn(f"Data shape mismatch for stream {name} after unique. Skipping.")
        continue

    stream_data.append(
        {
            "timestamps": timestamps,
            "data": data,
            "columns": cols,
            "name": name,
            "nominal_srate": float(stream["info"]["nominal_srate"][0]),
            "effective_srate": (len(timestamps) / (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0),
        }
    )

# --- Handle Duplicate Column Names ---
all_cols = [col for s in stream_data for col in s["columns"]]
duplicate_cols = set([col for col in all_cols if all_cols.count(col) > 1])

if duplicate_cols:
    warnings.warn(f"Duplicate column names found: {duplicate_cols}. Prefixing with stream names.")
    for s in stream_data:
        # Check if any of this stream's columns are duplicates
        if any(col in duplicate_cols for col in s["columns"]):
            s["columns"] = [f"{s['name']}_{col}" for col in s["columns"]]

# --- Synchronize Streams ---
df = synchronize_streams(
    stream_data,
    upsample_factor=2.0,
    fill_method="ffill",
    fill_value=0,
    show=[
        "CHANNEL_0",
        "LUX2",
        "ACC_X",
        "EEG_AF8",
        "OPTICS_RI_RED",
        "ECGBIT1",
        "PULSEOXI3",
    ],  # Pass the list of channels to plot
    window_start=835.5,  # df.index[int(len(df) / 2)]
)


# Find events
df.iloc[200000:300000].plot(y=["LUX2", "CHANNEL_0", "OPTICS_RI_RED"], figsize=(15, 5), subplots=True)
lux_events = nk.events_find(df["LUX2"], threshold_keep="below", duration_min=100, duration_max=8000)
jspsych_events = nk.events_find(df["CHANNEL_0"], threshold_keep="above", duration_min=100, duration_max=8000)
heartbeats = nk.ecg_findpeaks(df["ECGBIT1"], sampling_rate=2000)["ECG_R_Peaks"]

# Investigate events
print(f"len jspsych onsets: {len(jspsych_events['onset'])}, len lux onsets: {len(lux_events['onset'])}")

delays = jspsych_ts_onset - lux_ts_onset
_ = plt.hist(delays, bins=50)

# Preprocess
df["EEG_AF8"] = df["EEG_AF8"] - df[["EEG_TP9", "EEG_TP10"]].mean(axis=1).values
df["EEG_AF7"] = df["EEG_AF7"] - df[["EEG_TP9", "EEG_TP10"]].mean(axis=1).values
df["EEG_AF8"] = nk.signal_filter(df["EEG_AF8"], sampling_rate=2000, lowcut=1.0, highcut=40.0)
df["EEG_AF7"] = nk.signal_filter(df["EEG_AF7"], sampling_rate=2000, lowcut=1.0, highcut=40.0)
df["OPTICS_RI_RED"] = nk.signal_filter(df["OPTICS_RI_RED"], sampling_rate=2000, lowcut=0.2, highcut=40.0)

df["EEG_AF"] = df[["EEG_AF7", "EEG_AF8"]].mean(axis=1).values

# Create epochs
epochs = nk.epochs_create(
    df,
    events=lux_events,
    sampling_rate=2000,
    epochs_start=-1,
    epochs_end=3,
)
grand_av = nk.epochs_average(
    epochs,
    which=["EEG_AF8", "ECGBIT1", "PULSEOXI3", "OPTICS_LI_AMB"],
    indices=["mean", "std", "ci"],
    show=False,
)
grand_av.plot(
    x="Time",
    y=["EEG_AF8_Mean", "ECGBIT1_Mean", "PULSEOXI3_Mean", "OPTICS_LI_AMB_Mean"],
    figsize=(10, 5),
    subplots=True,
)

# df.iloc[200000:210000].plot(y=["OPTICS_RI_RED"], figsize=(15, 5), subplots=True)
df.columns
