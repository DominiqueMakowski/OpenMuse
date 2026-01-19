"""
Find heartbeats (R-peaks) in the EEG signal using PPG as a reference.

Typical Heartbeat Evoked Potentials (HEPs) show a characteristic pattern.
The R-peak in the ECG has a corresponding cardiac field artifact (CFA), typically showing as a sharp negative deflection (100ms), followed by a second slow negative deflection peaking between 200-300ms post CFA peak.

The goal of these functions is to leverage the signals available in the Muse S Athena, namely the Optics channels from which we can derive a PPG signal (pulse), to help identifying heartbeats locations in the EEG signal (AF7 and AF8).

The assumptions are:
- Each PPG peak has a corresponding R-peak-related feature in the EEG, occurring some time before the PPG peak. The resulting number of detected heartbeats should equal the number of PPG peaks (minus one if the first PPG peak is too close to the start of the recording).
- The R-peak always precedes the PPG trough (start of the rise) for the corresponding cardiac cycle, as the electrical signal (ECG) initiates the mechanical pulse wave that reaches the forehead after a delay (PAT). The rise time (from trough to peak) begins only after the pulse wave arrives, so the R-peak occurs ~50-200 ms before the trough.
- Under rest or controlled conditions, the delay between the R-Peak and the PPG peak varies minimally (e.g., standard deviation ~18-24 ms across beats in healthy adults).

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
import pywt
from sklearn.decomposition import FastICA
import scipy.stats
import scipy.interpolate
import warnings


# =======================================================================
# Utilities
# =======================================================================
def bandpass(data, fs, low=1, high=20, order=3):
    sos = scipy.signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return scipy.signal.sosfiltfilt(sos, data)


def find_ppg_peaks(ppg, fs):
    # Basic PPG peak detection
    distance = int(0.4 * fs)  # min 150 bpm
    peaks, _ = scipy.signal.find_peaks(
        ppg, distance=distance, prominence=np.std(ppg) * 0.3
    )
    return peaks


def find_ppg_troughs(ppg, fs):
    # Detect PPG troughs (feet) as minima
    distance = int(0.4 * fs)  # min 150 bpm
    troughs, _ = scipy.signal.find_peaks(
        -ppg, distance=distance, prominence=np.std(ppg) * 0.3
    )
    return troughs


# =======================================================================
# Plotting
# =======================================================================


def _plot_hep_on_axis(
    ax1,
    eeg_signal,
    rpeaks,
    fs,
    ppg=None,
    window=(-0.3, 0.5),
    baseline_window=(-0.2, -0.05),
    n_traces=None,
    random_state=None,
    channel_name="",
):
    """
    Internal function to plot HEP data on a *provided* matplotlib axis.
    """
    pre, post = window
    n_pre = int(np.abs(pre * fs))
    n_post = int(post * fs)
    n_samples = n_pre + n_post
    t = np.linspace(pre, post, n_samples)

    # --- Extract EEG epochs ---
    epochs = []
    for rp in rpeaks:
        start = rp - n_pre
        end = rp + n_post
        epoch = np.full(n_samples, np.nan)
        src_start = max(0, start)
        src_end = min(len(eeg_signal), end)
        tgt_start = max(0, -start)
        tgt_end = tgt_start + (src_end - src_start)
        if src_start < src_end:
            epoch[tgt_start:tgt_end] = eeg_signal[src_start:src_end]
        epochs.append(epoch)

    epochs = np.array(epochs)
    if epochs.size == 0:
        ax1.set_title(f"{channel_name}: No valid epochs found")
        return

    # --- Baseline Correction ---
    if baseline_window is not None:
        try:
            bl_pre_idx = int((baseline_window[0] - pre) * fs)
            bl_post_idx = int((baseline_window[1] - pre) * fs)
            bl_pre_idx = max(0, bl_pre_idx)
            bl_post_idx = min(n_samples, bl_post_idx)
            if bl_pre_idx < bl_post_idx:
                baseline_values = epochs[:, bl_pre_idx:bl_post_idx]
                baseline_mean = np.nanmean(baseline_values, axis=1, keepdims=True)
                epochs = epochs - baseline_mean
            else:
                warnings.warn(
                    f"Baseline window {baseline_window} is invalid or outside epoch. Skipping."
                )
        except Exception as e:
            warnings.warn(f"Error during baseline correction: {e}. Skipping.")

    # Use nanmean to calculate average, ignoring NaNs
    avg = np.nanmean(epochs, axis=0)

    # --- Extract PPG epochs (if provided) ---
    avg_ppg = None
    if ppg is not None:
        epochs_ppg = []
        for rp in rpeaks:
            start = rp - n_pre
            end = rp + n_post
            epoch_ppg = np.full(n_samples, np.nan)
            src_start = max(0, start)
            src_end = min(len(ppg), end)
            tgt_start = max(0, -start)
            tgt_end = tgt_start + (src_end - src_start)
            if src_start < src_end:
                epoch_ppg[tgt_start:tgt_end] = ppg[src_start:src_end]
            epochs_ppg.append(epoch_ppg)
        if epochs_ppg:
            avg_ppg = np.nanmean(np.array(epochs_ppg), axis=0)

    # Determine which traces to plot
    if n_traces is not None and n_traces < len(epochs):
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(epochs), size=n_traces, replace=False)
        epochs_to_plot = epochs[idx]
    else:
        epochs_to_plot = epochs

    # --- Plot on the provided axis 'ax1' ---
    # Plot individual EEG traces
    ax1.plot(t * 1000, epochs_to_plot.T, color="grey", alpha=0.3, lw=0.8)
    # Plot average EEG
    line1 = ax1.plot(t * 1000, avg, color="red", lw=2, label="Average EEG")
    ax1.axvline(0, color="k", linestyle="--", lw=1)
    ax1.axhline(0, color="k", linestyle=":", lw=0.5)  # Add horizontal zero line

    ax1.set_xlabel("Time (ms, relative to R-peak)")
    ax1.set_ylabel("EEG amplitude (a.u.)")

    n_valid_epochs = np.sum(~np.isnan(epochs[:, n_pre]))  # Check at R-peak
    ax1.set_title(f"{channel_name} ({n_valid_epochs} epochs)")

    # Compute y-limits for EEG
    y_min, y_max = np.nanmin(avg), np.nanmax(avg)
    y_range = y_max - y_min
    y_pad = 0.25 * y_range if y_range > 0 else 1
    if np.isfinite(y_min) and np.isfinite(y_max):
        ax1.set_ylim(y_min - y_pad, y_max + y_pad)

    lines, labels = ax1.get_legend_handles_labels()

    # Plot PPG on second y-axis if available
    if avg_ppg is not None and not np.all(np.isnan(avg_ppg)):
        ax2 = ax1.twinx()
        line2 = ax2.plot(t * 1000, avg_ppg, color="purple", lw=2, label="Average PPG")
        ax2.set_ylabel("PPG amplitude (a.u.)", color="purple")
        ax2.tick_params(axis="y", labelcolor="purple")
        y_min_ppg, y_max_ppg = np.nanmin(avg_ppg), np.nanmax(avg_ppg)
        y_range_ppg = y_max_ppg - y_min_ppg
        y_pad_ppg = 0.1 * y_range_ppg if y_range_ppg > 0 else 1
        if np.isfinite(y_min_ppg) and np.isfinite(y_max_ppg):
            ax2.set_ylim(y_min_ppg - y_pad_ppg, y_max_ppg + y_pad_ppg)
        lines.extend(line2)
        labels.extend([l.get_label() for l in line2])

    ax1.legend(lines, labels, loc=0)


def plot_heartbeatevoked(
    eeg,
    rpeaks,
    fs,
    ppg=None,
    window=(-0.3, 0.5),
    baseline_window=(-0.2, -0.05),
    n_traces=None,
    random_state=None,
):
    """
    Plot Heartbeat-Evoked Potentials (HEPs) for one or more EEG channels.

    Parameters
    ----------
    eeg : np.ndarray or list
        - If np.ndarray: A 1D EEG signal.
        - If list: A list of dictionaries, e.g.,
          [{'AF7': af7_signal}, {'AF8': af8_signal}, ...]
    rpeaks : array-like
        Indices of detected R-peaks.
    fs : float
        Sampling rate in Hz.
    ppg : np.ndarray, optional
        1D PPG signal. If provided, its average epoch is plotted on each subplot.
    window : tuple (pre, post), optional
        Time window around R-peak in seconds.
    baseline_window : tuple (pre, post) or None, optional
        Time window for baseline correction (relative to R-peak).
        Set to None to disable.
    n_traces : int or None, optional
        Number of individual traces to display.
    random_state : int or None, optional
        Random seed for sampling traces.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : matplotlib.axes.Axes or array of Axes
        The axes object(s).
    """

    # Case 1: eeg is a single np.ndarray
    if isinstance(eeg, np.ndarray):
        fig, ax1 = plt.subplots(figsize=(8, 5))
        _plot_hep_on_axis(
            ax1,
            eeg_signal=eeg,
            rpeaks=rpeaks,
            fs=fs,
            ppg=ppg,
            window=window,
            baseline_window=baseline_window,
            n_traces=n_traces,
            random_state=random_state,
            channel_name="Heartbeat-evoked Potential",
        )
        plt.tight_layout()
        plt.show()
        return fig, ax1

    # Case 2: eeg is a list of dicts
    elif isinstance(eeg, list):
        n_signals = len(eeg)
        if n_signals == 0:
            print("Warning: 'eeg' argument is an empty list. Nothing to plot.")
            return

        fig, axes = plt.subplots(n_signals, 1, figsize=(10, 5 * n_signals), sharex=True)

        # Ensure 'axes' is always an array, even if n_signals=1
        if n_signals == 1:
            axes = [axes]

        fig.suptitle("Heartbeat-evoked Potentials", fontsize=16, y=1.02)

        for i, eeg_dict in enumerate(eeg):
            ax = axes[i]

            # Validate format
            if not (isinstance(eeg_dict, dict) and len(eeg_dict) == 1):
                warnings.warn(
                    f"Skipping item {i}: Invalid format. "
                    f"Expected {{'name': signal_array}}, got {eeg_dict}"
                )
                ax.set_title(f"Plot {i}: Invalid data format")
                continue

            channel_name = list(eeg_dict.keys())[0]
            eeg_signal = eeg_dict[channel_name]

            if not isinstance(eeg_signal, np.ndarray):
                warnings.warn(
                    f"Skipping item {i} ('{channel_name}'): "
                    f"Signal is not a numpy array."
                )
                ax.set_title(f"Plot {i} ('{channel_name}'): Invalid signal type")
                continue

            _plot_hep_on_axis(
                ax,
                eeg_signal=eeg_signal,
                rpeaks=rpeaks,
                fs=fs,
                ppg=ppg,
                window=window,
                baseline_window=baseline_window,
                n_traces=n_traces,
                random_state=random_state,
                channel_name=channel_name,
            )

        plt.tight_layout(rect=[0, 0.03, 1, 0.98])  # Adjust for suptitle
        plt.show()
        return fig, axes

    # Case 3: Invalid type
    else:
        raise TypeError(
            "eeg argument must be a 1D np.ndarray or a "
            "list of dicts (e.g., [{'AF7': signal}, ...])"
        )


# =======================================================================
# Template-Correlation
# =======================================================================
def find_rpeaks_template(
    eeg,
    ppg,
    fs,
    window=0.3,
    template_width=0.2,
    search_margin=0.1,
):
    """
    Estimate R-peaks from EEG using PPG-guided template correlation.
    (Robust version)

    Parameters:
    -----------
    eeg : np.ndarray
        1D EEG signal.
    ppg : np.ndarray
        1D PPG signal.
    fs : float
        Sampling rate in Hz.
    window : float, optional
        The search window (in seconds) *before* each PPG trough to look
        for the R-peak (default: 0.3s).
    template_width : float, optional
        The width (in seconds) of the CFA/R-peak template to
        build (default: 0.2s).
    search_margin : float, optional
        The margin (in seconds) to search around the globally-estimated
        R-peak location for per-beat refinement (default: 0.1s).

    Returns:
    --------
    rpeaks : np.ndarray
        Estimated R-peak indices.
    info : dict
        Dictionary with debugging and supplementary info.
        On failure, 'error' key will be present.
        On success, 'status': 'success' will be present.
    """

    # --- 1. Validation ---
    if any(v <= 0 for v in [fs, window, template_width]):
        raise ValueError("fs, window, and template_width must be positive values.")
    if search_margin < 0:
        raise ValueError("search_margin cannot be negative.")

    # Convert times to samples
    window_samples = int(window * fs)
    template_width_samples = int(template_width * fs)
    search_margin_samples = int(search_margin * fs)

    if template_width_samples == 0 or window_samples == 0:
        return np.array([]), {
            "error": "Resulting sample sizes are zero.",
            "window_samples": window_samples,
            "template_width_samples": template_width_samples,
            "fs": fs,
            "window_s": window,
            "template_width_s": template_width,
        }

    # CRITICAL CHECK 1: Template *must* be smaller than the window.
    if template_width_samples >= window_samples:
        return np.array([]), {
            "error": "template_width must be smaller than window.",
            "window_s": window,
            "template_width_s": template_width,
            "window_samples": window_samples,
            "template_width_samples": template_width_samples,
        }

    # --- 2. Signal Preparation ---
    # Apply filters as defined in the original script's context
    eeg_filt = bandpass(eeg, fs, low=0.5, high=40)
    ppg_filt = bandpass(ppg, fs, low=0.5, high=12)

    ppg_troughs = find_ppg_troughs(ppg_filt, fs)
    if len(ppg_troughs) < 10:  # Need a few beats for a stable template
        return np.array([]), {
            "error": "Not enough PPG troughs found to build a stable template.",
            "troughs_found": len(ppg_troughs),
            "min_required": 10,
        }

    # Extract EEG windows preceding each PPG trough
    segments = []
    for trough in ppg_troughs:
        start = trough - window_samples
        end = trough
        if start < 0 or end > len(eeg_filt):
            continue  # Segment out of bounds

        # Ensure segments are all the same length
        if len(eeg_filt[start:end]) == window_samples:
            segments.append(eeg_filt[start:end])

    segments = np.array(segments)

    if segments.shape[0] < 5:  # Need at least a few valid segments
        return np.array([]), {
            "error": "Not enough valid segments to build template.",
            "info": "PPG troughs may be too close to data edges.",
            "total_troughs": len(ppg_troughs),
            "valid_segments_found": segments.shape[0],
            "min_required": 5,
        }

    # --- 3. Find consistent lag (R-peak) ---
    # We find the lag with the highest *variance* across segments,
    # as the R-peak/CFA is a high-amplitude, consistent event.
    variance_curve = np.var(segments, axis=0)

    # --- 4. Build CFA template (Robustly) ---

    # Define template padding
    template_half_samples_pre = template_width_samples // 2
    template_half_samples_post = template_width_samples - template_half_samples_pre

    # **CRITICAL FIX**: Define a "safe" search area *within* the window.
    # The *center* of the template (best_lag_idx) must be
    # at least half_pre from the start and half_post from the end.
    search_start = template_half_samples_pre
    search_end = window_samples - template_half_samples_post

    # CRITICAL CHECK 2: Is this search area valid?
    if search_start >= search_end:
        return np.array([]), {
            "error": "Template width is too large for the search window. "
            "Cannot define a valid search region.",
            "info": "Reduce template_width or increase window.",
            "window_samples": window_samples,
            "template_width_samples": template_width_samples,
            "safe_search_start": search_start,
            "safe_search_end": search_end,
        }

    # Search *only* within the "safe" region of the variance curve
    variance_curve_focus = variance_curve[search_start:search_end]

    if len(variance_curve_focus) == 0:
        return np.array([]), {
            "error": "Safe search region (variance_curve_focus) is empty. "
            "This is an unexpected state.",
            "search_start": search_start,
            "search_end": search_end,
        }

    # Find the index of max variance *within the safe region*
    best_lag_idx_relative_to_focus = np.argmax(variance_curve_focus)

    # Convert back to index *relative to the window start*
    best_lag_idx_global = search_start + best_lag_idx_relative_to_focus

    # This is the global lag (in samples) from the PPG trough
    best_lag = best_lag_idx_global - window_samples  # (will be negative)

    # Now, extract all template slices.
    # Because we searched in the safe region, we are *guaranteed*
    # that the slices are valid and won't go out of bounds.
    aligned_for_template = []
    for seg in segments:
        start_t = best_lag_idx_global - template_half_samples_pre
        end_t = best_lag_idx_global + template_half_samples_post
        aligned_for_template.append(seg[start_t:end_t])

    # This check is now just a formality, as the list *should* be full.
    if not aligned_for_template:
        return np.array([]), {
            "error": "Failed to build template array, even after robust search. "
            "This is unexpected.",
            "info": "This might happen if 'segments' array was empty, "
            "but that should have been caught earlier.",
        }

    template = np.mean(aligned_for_template, axis=0)

    # Check template validity before standardization
    template_std = np.std(template)
    if template_std < 1e-6:
        return np.array([]), {
            "error": "Template has near-zero variance. Cannot proceed.",
            "info": "This might mean the EEG signal is flat or "
            "the detected lag is in a quiet spot.",
            "template_std": template_std,
        }

    template_norm = (template - np.mean(template)) / template_std

    # --- 5. Per-beat refinement using template cross-correlation ---
    refined_rpeaks = []
    individual_lags_samples = []

    # Iterate over the *original* troughs to find all possible heartbeats
    for trough in ppg_troughs:
        # Estimate R-peak location based on *global* lag
        expected_rpeak_loc = trough + best_lag

        # Define search segment in the EEG signal, wide enough for
        # the template to slide by search_margin_samples
        start = expected_rpeak_loc - search_margin_samples - template_half_samples_pre
        end = expected_rpeak_loc + search_margin_samples + template_half_samples_post

        if start < 0 or end >= len(eeg_filt):
            continue  # This beat is too close to the edge

        segment = eeg_filt[start:end]

        # The segment must be long enough to correlate with the template
        if len(segment) < len(template_norm):
            continue

        # Standardize segment
        segment_std = np.std(segment)
        if segment_std < 1e-6:
            segment_norm = segment - np.mean(segment)  # Avoid division by zero
        else:
            segment_norm = (segment - np.mean(segment)) / segment_std

        # Correlate
        # 'valid' mode results in array of length: len(segment) - len(template) + 1
        # This should be exactly (2 * search_margin_samples) + 1
        corr = scipy.signal.correlate(segment_norm, template_norm, mode="valid")

        if len(corr) == 0:
            continue

        # Find offset from the *center* of the search window
        # Center of corr is at index search_margin_samples
        # (assuming len(corr) == 2 * search_margin_samples + 1)
        center_idx = search_margin_samples

        # Ensure center_idx is valid for corr array
        if center_idx >= len(corr):
            # This can happen if segment length was off
            center_idx = len(corr) // 2

        best_corr_idx = np.argmax(corr)
        offset_from_center = best_corr_idx - center_idx

        # Calculate final R-peak location
        refined_peak = expected_rpeak_loc + offset_from_center
        refined_rpeaks.append(refined_peak)

        actual_lag = refined_peak - trough
        individual_lags_samples.append(actual_lag)

    rpeaks_arr = np.array(refined_rpeaks, dtype=int)
    lags_arr = np.array(individual_lags_samples)

    info = {
        "status": "success",
        "ppg_troughs": ppg_troughs,
        "rpeaks_found": rpeaks_arr,
        "n_rpeaks_found": len(rpeaks_arr),
        "n_troughs_found": len(ppg_troughs),
        "individual_lags_samples": lags_arr,
        "individual_lags_s": lags_arr / fs if len(lags_arr) > 0 else np.array([]),
        "mean_lag_s": np.mean(lags_arr) / fs if len(lags_arr) > 0 else np.nan,
        "std_lag_s": np.std(lags_arr) / fs if len(lags_arr) > 0 else np.nan,
        "global_best_lag_samples": best_lag,
        "global_best_lag_s": best_lag / fs,
        "template": template_norm,  # Return the standardized template
        "lag_consistency_scores (variance)": variance_curve,
        "lag_consistency_samples": np.arange(-window_samples, 0),
    }

    return rpeaks_arr, info


# =======================================================================
# TEST
# =======================================================================
# Note: df has been obtained by running another script
df["Muse_PPG"], _ = preprocess_ppg(
    df,
    sampling_rate=2000,
    hp_cutoff=0.5,
    lp_cutoff=8.0,
    verbose=False,
)
df.columns
nk.standardize(df.iloc[60000:75000]).plot(
    y=["Muse_PPG", "OPTICS_RI_RED", "OPTICS_RI_NIR", "OPTICS_RI_AMB"], figsize=(15, 5)
)

# PREPROCESS EEG SIGNALS
af7 = bandpass(df["EEG_AF7"].values, fs=2000, low=1, high=40, order=3)
af8 = bandpass(df["EEG_AF8"].values, fs=2000, low=1, high=40, order=3)
tp9 = bandpass(df["EEG_TP9"].values, fs=2000, low=1, high=40, order=3)
tp10 = bandpass(df["EEG_TP10"].values, fs=2000, low=1, high=40, order=3)
ref = (tp9 + tp10) / 2.0
af7ref = af7 - ref
af8ref = af8 - ref
eeg = (af7ref + af8ref) / 2.0

rpeaks, info = find_rpeaks_template(
    af8ref,
    df["Muse_PPG"].values,
    fs=2000,
    window=0.5,
    template_width=0.2,
    search_margin=0.1,
)

print(f"Found {len(rpeaks)} R-peaks")

# Plot the result
plot_heartbeatevoked(
    [
        {"AF7": af7},
        {"AF8": af8},
        {"TP9": tp9},
        {"TP10": tp10},
        {"AF7 - rereferenced": af7ref},
        {"AF8 - rereferenced": af8ref},
    ],
    rpeaks,
    ppg=df["Muse_PPG"].values,
    fs=2000,
    n_traces=100,
    window=(-0.4, 1.0),
)
