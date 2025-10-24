import io
import urllib
import warnings
from fractions import Fraction

import numpy as np
import pandas as pd
import pyxdf
import scipy

import scipy.interpolate
import scipy.signal
import time
import matplotlib.pyplot as plt

# ========================================================================================
# Helper Functions for Resampling
# ========================================================================================


def _create_target_timestamps(stream_data, target_fs):
    """
    Creates a new, regularly spaced timestamp vector based on the global
    min/max time of all streams and the target sampling rate.
    """
    if target_fs <= 0:
        raise ValueError("target_fs must be positive.")

    dt = 1.0 / target_fs

    # Find the global time range
    global_min_ts = min([s["timestamps"].min() for s in stream_data])
    global_max_ts = max([s["timestamps"].max() for s in stream_data])

    # Create the new timestamp vector
    new_timestamps = np.arange(global_min_ts, global_max_ts + dt, dt)
    return new_timestamps


def _interpolate_splat_ffill(stream_data, new_timestamps, all_columns, col_to_idx):
    """
    Performs a fast "splat" resampling.

    Each sample in the original streams is "splatted" onto its nearest
    timestamp in the new_timestamps grid. This is very fast and vectorized.
    The resulting grid is sparse (mostly NaN) and is intended to be
    forward-filled.
    """
    # 1. Create the empty (NaN-filled) data grid
    resampled_data = np.full(
        (len(new_timestamps), len(all_columns)), np.nan, dtype=object
    )

    # 2. Iterate over each *original* stream and "splat" its data
    for s in stream_data:
        # Find the nearest index in new_timestamps for each original timestamp
        insertion_indices = np.searchsorted(
            new_timestamps, s["timestamps"], side="left"
        )

        # Get indices of new timestamps to the left and right
        left_indices = np.clip(insertion_indices - 1, 0, len(new_timestamps) - 1)
        right_indices = np.clip(insertion_indices, 0, len(new_timestamps) - 1)

        # Get the actual new timestamps
        left_ts = new_timestamps[left_indices]
        right_ts = new_timestamps[right_indices]

        # Find which of the two is closer
        is_left_closer = (s["timestamps"] - left_ts) <= (right_ts - s["timestamps"])
        closest_indices = np.where(is_left_closer, left_indices, right_indices)

        # Get the column indices for this stream
        col_indices = [col_to_idx[c] for c in s["columns"]]

        # "Splat" the data into the grid at the nearest timestamps.
        # This is a many-to-one mapping; if multiple original samples map
        # to the same new timestamp, the last one wins.
        resampled_data[closest_indices[:, np.newaxis], col_indices] = s["data"]

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


def resample_streams(stream_data, target_fs, fill_method="ffill", fill_value=0):
    """
    Resamples and merges multiple XDF streams into a single DataFrame using
    a fast "nearest-neighbor assignment" and "forward-fill" strategy.

    Args:
        stream_data (list): List of stream dictionaries from the loading phase.
        target_fs (float): The target sampling rate in Hz.
        fill_method (str): Method for filling NaNs ('ffill', 'bfill', None).
        fill_value (any): Value to fill remaining NaNs (e.g., 0 or np.nan).

    Returns:
        pd.DataFrame: A single DataFrame with all streams resampled and merged.
    """

    # 1. Get all column names and create a name-to-index mapping
    all_columns = []
    for s in stream_data:
        all_columns.extend(s["columns"])
    col_to_idx = {name: i for i, name in enumerate(all_columns)}

    # 2. Create the target *regular* timestamp grid
    new_ts = _create_target_timestamps(stream_data, target_fs)

    # 3. Splat original data onto the new grid
    resampled_data = _interpolate_splat_ffill(
        stream_data, new_ts, all_columns, col_to_idx
    )

    # 4. Create the DataFrame
    # This DataFrame is sparse, containing original values at their
    # nearest new timestamp, and NaN everywhere else.
    resampled_df = pd.DataFrame(resampled_data, index=new_ts, columns=all_columns)
    resampled_df.index.name = "timestamps"

    # 5. Fill missing values (this is the forward-padding step)
    resampled_df = _fill_missing_data(resampled_df, fill_method, fill_value)

    return resampled_df


# ========================================================================================
# Quality Control Functions
# ========================================================================================


def _visual_control(original, resampled, window_start=None, window_length=2.0):
    """Helper for plotting a window of original vs. resampled data."""
    if window_start is None:
        # Default to plotting a window in the middle of the data
        window_start = original.index[int(len(original) / 2)]
    window_end = window_start + window_length

    # Select the time window
    signal = original[(original.index >= window_start) & (original.index <= window_end)]
    resampled = resampled[
        (resampled.index >= window_start) & (resampled.index <= window_end)
    ]

    plt.figure(figsize=(15, 5))
    plt.plot(signal.index, signal, "o-", label="original", alpha=0.7, markersize=3)
    # Changed to '.-' to show individual new samples
    plt.plot(
        resampled.index, resampled, ".-", label="resampled", alpha=0.7, markersize=2
    )
    plt.legend()
    plt.title(f"Visual Control: {original.name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()


def quality_control(stream, resampled_df, show=False):
    """
    Performs a quality control check by plotting the first channel
    of a given stream against its resampled version.

    Args:
        stream (dict): A single stream dictionary from the stream_data list.
        resampled_df (pd.DataFrame): The output of resample_streams.
        show (bool): If True, generate and show the plot.
    """
    if not show:
        return

    # Don't plot non-numeric streams
    if not stream["is_numeric"]:
        warnings.warn(f"Skipping QC plot for non-numeric stream: {stream['name']}")
        return

    # Get the name of the first column in this stream
    col_to_plot = stream["columns"][0]

    # Create a pandas Series for the original data of the first channel
    original = pd.Series(
        stream["data"][:, 0], index=stream["timestamps"], name=col_to_plot
    )
    original.index.name = "timestamps"

    # Find the corresponding column in resampled_df
    resampled_col_name = None
    if col_to_plot in resampled_df.columns:
        resampled_col_name = col_to_plot
    # Note: The prefixing logic ensures col_to_plot *already* has the prefix
    # if it was needed. So we just need to check for its existence.

    if resampled_col_name is None:
        warnings.warn(
            f"Could not find column '{col_to_plot}' in resampled data for QC plot."
        )
        return

    resampled = resampled_df[resampled_col_name]

    _visual_control(original, resampled, window_start=None, window_length=2.0)


# ========================================================================================
# Main Script Execution
# ========================================================================================

# --- Configuration ---
filename = "./test-09.xdf"
upsample_factor = 2.0  # Factor to multiply max nominal srate by
# Note: Interpolation method is now fixed to 'splat_ffill'
fill_method = "ffill"  # 'ffill', 'bfill', or None
fill_value = 0  # Value for remaining NaNs

# --- Load Data ---
streams, header = pyxdf.load_xdf(
    filename,
    synchronize_clocks=True,
    handle_clock_resets=True,
    dejitter_timestamps=False,
)

# --- Pre-processing & Sanity Checks ---

# Warn if any stream has no time_stamps
for i, stream in enumerate(streams):
    name = stream["info"].get("name", ["Unnamed"])[0]
    if len(stream["time_stamps"]) == 0:
        warnings.warn(f"Stream {i} - {name} has no time_stamps. Dropping it.")
# Drop streams with no timestamps
streams = [s for s in streams if len(s["time_stamps"]) > 0]

if not streams:
    raise RuntimeError("No valid streams found in the file.")

# Get smaller time stamp to later use as offset (zero point)
min_ts = min([min(s["time_stamps"]) for s in streams])

# Make sure the length of all streams are roughly within the same range
ts_mins = np.array([stream["time_stamps"].min() for stream in streams])
ts_maxs = np.array([stream["time_stamps"].max() for stream in streams])
ts_durations = ts_maxs - ts_mins
duration_diffs = np.abs(ts_durations[:, np.newaxis] - ts_durations[np.newaxis, :])
if np.any(duration_diffs > 7200):  # 2 hours
    warnings.warn(
        "Some streams differ in duration by more than 2 hours. This might be indicative of an issue."
    )

# --- Convert to common format (list of dicts) ---
stream_data = []
for stream in streams:
    # Get columns names
    try:
        channels_info = stream["info"]["desc"][0]["channels"][0]["channel"]
        cols = [channels_info[i]["label"][0] for i in range(len(channels_info))]
    except (KeyError, TypeError, IndexError):
        cols = [f"CHANNEL_{i}" for i in range(np.array(stream["time_series"]).shape[1])]
        warnings.warn(
            f"Using default channel names for stream: {stream['info'].get('name', ['Unnamed'])[0]}"
        )

    name = stream["info"].get("name", ["Unnamed"])[0]
    timestamps = stream["time_stamps"] - min_ts  # Offset to zero
    data = np.array(stream["time_series"])

    # If duplicate timestamps exist, take last occurrence
    unique_ts, unique_indices = np.unique(timestamps, return_index=True)
    data = np.array([data[i] for i in unique_indices])
    timestamps = unique_ts

    # Ensure data is 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)

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
            "effective_srate": (
                len(timestamps) / (timestamps[-1] - timestamps[0])
                if len(timestamps) > 1
                else 0
            ),
            "is_numeric": np.issubdtype(data.dtype, np.number),
        }
    )

# --- Handle Duplicate Column Names ---
all_cols = [col for s in stream_data for col in s["columns"]]
duplicate_cols = set([col for col in all_cols if all_cols.count(col) > 1])

if duplicate_cols:
    warnings.warn(
        f"Duplicate column names found: {duplicate_cols}. Prefixing with stream names."
    )
    for s in stream_data:
        # Check if any of this stream's columns are duplicates
        if any(col in duplicate_cols for col in s["columns"]):
            s["columns"] = [f"{s['name']}_{col}" for col in s["columns"]]

# --- Compute Target Sampling Rate ---
max_nominal_srate = np.max([s["nominal_srate"] for s in stream_data])
target_fs = int(max_nominal_srate * upsample_factor)

print(f"Max nominal sampling rate: {max_nominal_srate} Hz")
print(f"Target sampling rate: {target_fs} Hz")

# --- Run Resampling ---
start_time = time.time()
resampled_df = resample_streams(
    stream_data, target_fs=target_fs, fill_method=fill_method, fill_value=fill_value
)
end_time = time.time()

print(f"Resampling complete in {end_time - start_time:.2f} seconds.")


# --- Run Quality Control ---
quality_control(stream_data[0], resampled_df, show=True)
