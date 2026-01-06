"""
EEG Blink Analysis: Validate both devices record consistent blinks
==================================================================

During the recording:
- 1 min rest (0-60s)
- 1 min blinks: 10 both eyes, 10 left, 10 right (60-120s)
- 1 min head movements (120-180s)

This script analyzes the blink period to:
1. Detect blink events in both devices
2. Compare blink timing/correlation
3. Verify frontal channels (AF7, AF8) show proper blink artifacts
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pyxdf
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d

# Configuration
XDF_FILE = "test1_eeg.xdf"
OLD_FIRMWARE_MAC = "00:55:DA:B9:FA:20"
NEW_FIRMWARE_MAC = "00:55:DA:BB:CD:CD"


def load_xdf(filename):
    streams, header = pyxdf.load_xdf(
        filename,
        synchronize_clocks=True,
        handle_clock_resets=True,
        dejitter_timestamps=False,
    )
    return streams, header


def get_stream_by_name(streams, name_substring):
    for stream in streams:
        name = stream["info"].get("name", ["Unnamed"])[0]
        if name_substring in name:
            return stream
    return None


def get_channel_labels(stream):
    try:
        return [d["label"][0] for d in stream["info"]["desc"][0]["channels"][0]["channel"]]
    except (KeyError, TypeError, IndexError):
        n_ch = stream["time_series"].shape[1] if len(stream["time_series"].shape) > 1 else 1
        return [f"Ch_{i}" for i in range(n_ch)]


def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """Apply bandpass filter."""
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = scipy_signal.butter(order, [low, high], btype="band")
    return scipy_signal.filtfilt(b, a, signal)


def detect_blinks(signal, ts, fs, threshold_std=3):
    """
    Detect blinks as large negative deflections in frontal EEG.
    Returns indices and times of detected blinks.
    """
    # Filter 0.5-10 Hz to capture blink artifacts
    filtered = bandpass_filter(signal, 0.5, 10, fs)

    # Blinks appear as large negative deflections
    threshold = -threshold_std * np.std(filtered)

    # Find peaks (inverted signal for negative peaks)
    peaks, properties = scipy_signal.find_peaks(
        -filtered, height=-threshold, distance=int(0.3 * fs)
    )  # Min 300ms between blinks

    return peaks, ts[peaks], filtered


def analyze_blinks(streams):
    """Analyze blink detection in both devices."""

    old_stream = get_stream_by_name(streams, f"EEG ({OLD_FIRMWARE_MAC})")
    new_stream = get_stream_by_name(streams, f"EEG ({NEW_FIRMWARE_MAC})")

    old_data = np.array(old_stream["time_series"])
    old_ts = old_stream["time_stamps"]
    new_data = np.array(new_stream["time_series"])
    new_ts = new_stream["time_stamps"]

    old_channels = get_channel_labels(old_stream)
    new_channels = get_channel_labels(new_stream)

    fs = 256  # Sampling rate

    # Common time range
    t_start = max(old_ts.min(), new_ts.min())

    print("=" * 80)
    print("EEG BLINK ANALYSIS")
    print("=" * 80)

    # Focus on blink period (60-120s)
    blink_start = 60
    blink_end = 120

    old_blink_mask = ((old_ts - t_start) >= blink_start) & ((old_ts - t_start) <= blink_end)
    new_blink_mask = ((new_ts - t_start) >= blink_start) & ((new_ts - t_start) <= blink_end)

    # Use AF7 and AF8 (frontal channels, best for blinks)
    af7_idx_old = old_channels.index("EEG_AF7")
    af8_idx_old = old_channels.index("EEG_AF8")
    af7_idx_new = new_channels.index("EEG_AF7")
    af8_idx_new = new_channels.index("EEG_AF8")

    print(f"\nAnalyzing blink period: {blink_start}-{blink_end}s")
    print(f"OLD firmware samples in range: {np.sum(old_blink_mask)}")
    print(f"NEW firmware samples in range: {np.sum(new_blink_mask)}")

    # Detect blinks in each device
    print("\n--- Blink Detection (AF7) ---")

    old_af7 = old_data[old_blink_mask, af7_idx_old]
    old_af7_ts = old_ts[old_blink_mask]
    new_af7 = new_data[new_blink_mask, af7_idx_new]
    new_af7_ts = new_ts[new_blink_mask]

    old_peaks, old_peak_times, old_filtered = detect_blinks(old_af7, old_af7_ts, fs)
    new_peaks, new_peak_times, new_filtered = detect_blinks(new_af7, new_af7_ts, fs)

    print(f"OLD firmware: detected {len(old_peaks)} blinks")
    print(f"NEW firmware: detected {len(new_peaks)} blinks")

    # Detect blinks in AF8 as well
    print("\n--- Blink Detection (AF8) ---")

    old_af8 = old_data[old_blink_mask, af8_idx_old]
    new_af8 = new_data[new_blink_mask, af8_idx_new]

    old_peaks_af8, _, old_filtered_af8 = detect_blinks(old_af8, old_af7_ts, fs)
    new_peaks_af8, _, new_filtered_af8 = detect_blinks(new_af8, new_af7_ts, fs)

    print(f"OLD firmware: detected {len(old_peaks_af8)} blinks")
    print(f"NEW firmware: detected {len(new_peaks_af8)} blinks")

    # --- Visualization ---
    fig, axes = plt.subplots(4, 1, figsize=(16, 14))
    fig.suptitle("Blink Period EEG Analysis (60-120s)", fontsize=14)

    # Plot time relative to blink start
    old_t_rel = old_af7_ts - old_af7_ts[0]
    new_t_rel = new_af7_ts - new_af7_ts[0]

    # Raw AF7
    ax = axes[0]
    ax.plot(old_t_rel, old_af7, label="Old firmware", alpha=0.7, linewidth=0.5)
    ax.plot(new_t_rel, new_af7, label="New firmware", alpha=0.7, linewidth=0.5)
    ax.set_title("Raw EEG_AF7")
    ax.set_ylabel("µV")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Filtered AF7 with detected blinks
    ax = axes[1]
    ax.plot(old_t_rel, old_filtered, label="Old (filtered)", alpha=0.7, linewidth=0.5)
    ax.plot(old_t_rel[old_peaks], old_filtered[old_peaks], "ro", markersize=5, label="Old blinks")
    ax.plot(new_t_rel, new_filtered, label="New (filtered)", alpha=0.7, linewidth=0.5)
    ax.plot(new_t_rel[new_peaks], new_filtered[new_peaks], "g^", markersize=5, label="New blinks")
    ax.set_title("Filtered EEG_AF7 (0.5-10 Hz) with detected blinks")
    ax.set_ylabel("µV")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Raw AF8
    ax = axes[2]
    ax.plot(old_t_rel, old_af8, label="Old firmware", alpha=0.7, linewidth=0.5)
    ax.plot(new_t_rel, new_af8, label="New firmware", alpha=0.7, linewidth=0.5)
    ax.set_title("Raw EEG_AF8")
    ax.set_ylabel("µV")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Filtered AF8
    ax = axes[3]
    ax.plot(old_t_rel, old_filtered_af8, label="Old (filtered)", alpha=0.7, linewidth=0.5)
    ax.plot(old_t_rel[old_peaks_af8], old_filtered_af8[old_peaks_af8], "ro", markersize=5, label="Old blinks")
    ax.plot(new_t_rel, new_filtered_af8, label="New (filtered)", alpha=0.7, linewidth=0.5)
    ax.plot(new_t_rel[new_peaks_af8], new_filtered_af8[new_peaks_af8], "g^", markersize=5, label="New blinks")
    ax.set_title("Filtered EEG_AF8 (0.5-10 Hz) with detected blinks")
    ax.set_ylabel("µV")
    ax.set_xlabel("Time since blink period start (s)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("eeg_blink_analysis.png", dpi=150)
    print("\n→ Saved: eeg_blink_analysis.png")

    # --- Zoom in on individual blinks ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Individual Blink Examples (zoomed)", fontsize=14)

    # Find 3 blinks that are detected in both devices (within 100ms)
    matched_blinks = []
    for old_t in old_peak_times:
        for new_t in new_peak_times:
            if abs(old_t - new_t) < 0.1:  # Within 100ms
                matched_blinks.append((old_t, new_t))
                break
        if len(matched_blinks) >= 3:
            break

    print(f"\nMatched blinks (within 100ms): {len(matched_blinks)}")

    for i, (old_t, new_t) in enumerate(matched_blinks[:3]):
        # Get 1-second window around blink
        window = 0.5

        old_window_mask = (old_af7_ts >= old_t - window) & (old_af7_ts <= old_t + window)
        new_window_mask = (new_af7_ts >= new_t - window) & (new_af7_ts <= new_t + window)

        # AF7
        ax = axes[0, i]
        ax.plot((old_af7_ts[old_window_mask] - old_t) * 1000, old_af7[old_window_mask], label="Old", alpha=0.8)
        ax.plot((new_af7_ts[new_window_mask] - new_t) * 1000, new_af7[new_window_mask], label="New", alpha=0.8)
        ax.axvline(0, color="r", linestyle="--", alpha=0.5)
        ax.set_title(f"Blink {i+1} - AF7")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("µV")
        if i == 0:
            ax.legend()
        ax.grid(True, alpha=0.3)

        # AF8
        old_window_mask = (old_af7_ts >= old_t - window) & (old_af7_ts <= old_t + window)
        new_window_mask = (new_af7_ts >= new_t - window) & (new_af7_ts <= new_t + window)

        ax = axes[1, i]
        ax.plot((old_af7_ts[old_window_mask] - old_t) * 1000, old_af8[old_window_mask], label="Old", alpha=0.8)
        ax.plot((new_af7_ts[new_window_mask] - new_t) * 1000, new_af8[new_window_mask], label="New", alpha=0.8)
        ax.axvline(0, color="r", linestyle="--", alpha=0.5)
        ax.set_title(f"Blink {i+1} - AF8")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("µV")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("eeg_blink_examples.png", dpi=150)
    print("→ Saved: eeg_blink_examples.png")

    # --- Cross-correlation analysis ---
    print("\n--- Cross-correlation Analysis ---")

    # Resample to common timestamps for correlation
    t_common_start = max(old_af7_ts[0], new_af7_ts[0])
    t_common_end = min(old_af7_ts[-1], new_af7_ts[-1])
    n_samples = int((t_common_end - t_common_start) * fs)
    common_ts = np.linspace(t_common_start, t_common_end, n_samples)

    interp_old = interp1d(old_af7_ts, old_filtered, kind="linear", bounds_error=False)
    interp_new = interp1d(new_af7_ts, new_filtered, kind="linear", bounds_error=False)

    old_resampled = interp_old(common_ts)
    new_resampled = interp_new(common_ts)

    # Remove NaN
    valid = ~np.isnan(old_resampled) & ~np.isnan(new_resampled)
    old_valid = old_resampled[valid]
    new_valid = new_resampled[valid]

    # Cross-correlation
    correlation = np.correlate(old_valid - old_valid.mean(), new_valid - new_valid.mean(), mode="full")
    correlation = correlation / (len(old_valid) * np.std(old_valid) * np.std(new_valid))
    lags = np.arange(-len(old_valid) + 1, len(old_valid)) / fs * 1000  # in ms

    max_corr_idx = np.argmax(correlation)
    max_corr = correlation[max_corr_idx]
    optimal_lag = lags[max_corr_idx]

    print(f"Maximum cross-correlation: {max_corr:.4f}")
    print(f"Optimal lag: {optimal_lag:.1f} ms")

    # Plot cross-correlation
    fig, ax = plt.subplots(figsize=(12, 4))
    lag_range = 500  # Show ±500ms
    lag_mask = (lags >= -lag_range) & (lags <= lag_range)
    ax.plot(lags[lag_mask], correlation[lag_mask])
    ax.axvline(optimal_lag, color="r", linestyle="--", label=f"Peak: {optimal_lag:.1f}ms, r={max_corr:.3f}")
    ax.axvline(0, color="k", linestyle=":", alpha=0.5)
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Cross-correlation")
    ax.set_title("Cross-correlation of filtered AF7 between devices")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("eeg_crosscorrelation.png", dpi=150)
    print("→ Saved: eeg_crosscorrelation.png")

    plt.show()

    return {
        "old_blinks_af7": len(old_peaks),
        "new_blinks_af7": len(new_peaks),
        "old_blinks_af8": len(old_peaks_af8),
        "new_blinks_af8": len(new_peaks_af8),
        "matched_blinks": len(matched_blinks),
        "max_correlation": max_corr,
        "optimal_lag_ms": optimal_lag,
    }


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    streams, _ = load_xdf(XDF_FILE)
    results = analyze_blinks(streams)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Blinks detected (AF7): Old={results['old_blinks_af7']}, New={results['new_blinks_af7']}")
    print(f"Blinks detected (AF8): Old={results['old_blinks_af8']}, New={results['new_blinks_af8']}")
    print(f"Matched blinks (within 100ms): {results['matched_blinks']}")
    print(f"Cross-correlation: {results['max_correlation']:.4f} at {results['optimal_lag_ms']:.1f}ms lag")


if __name__ == "__main__":
    main()
