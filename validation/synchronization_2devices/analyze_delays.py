"""
Analyze the actual delay distribution to understand the scatter.
"""

import matplotlib.pyplot as plt
import numpy as np
import pyxdf
from scipy import ndimage

NEW_FIRMWARE_MAC = "00:55:DA:BB:CD:CD"
OLD_FIRMWARE_MAC = "00:55:DA:B9:FA:20"


def rolling_normalize(signal_data, window_sec=5.0, fs=64.0):
    window_samples = max(int(window_sec * fs), 1)
    rolling_min = ndimage.minimum_filter1d(signal_data, size=window_samples, mode="nearest")
    rolling_max = ndimage.maximum_filter1d(signal_data, size=window_samples, mode="nearest")
    range_vals = rolling_max - rolling_min
    range_vals[range_vals < 1e-10] = 1e-10
    return (signal_data - rolling_min) / range_vals


def median_filter(signal_data, kernel_size=3):
    return ndimage.median_filter(signal_data, size=kernel_size)


def find_channel(streams, stream_name, channel_name):
    for s in streams:
        name = s["info"].get("name", ["Unnamed"])[0]
        if name == stream_name:
            try:
                channels = [d["label"][0] for d in s["info"]["desc"][0]["channels"][0]["channel"]]
            except:
                return None, None
            matching = [ch for ch in channels if channel_name in ch]
            if not matching:
                return None, None
            ch_name = matching[0]
            signal_data = np.array(s["time_series"])[:, channels.index(ch_name)].astype(float)
            ts = s["time_stamps"]
            return signal_data, ts
    return None, None


def find_events_midpoint(signal_data, timestamps, hysteresis=0.15):
    if len(signal_data) < 10:
        return np.array([])

    fs = len(timestamps) / (timestamps[-1] - timestamps[0])
    min_gap_samples = int(0.15 * fs)

    high_thresh = 0.5 + hysteresis
    low_thresh = 0.5 - hysteresis

    events = []
    state = None

    if signal_data[0] > high_thresh:
        state = "high"
    elif signal_data[0] < low_thresh:
        state = "low"

    for i in range(1, len(signal_data)):
        if state == "low" and signal_data[i] > high_thresh:
            for j in range(i, max(0, i - 20), -1):
                if signal_data[j] <= 0.5:
                    events.append(j)
                    break
            state = "high"
        elif state == "high" and signal_data[i] < low_thresh:
            for j in range(i, max(0, i - 20), -1):
                if signal_data[j] >= 0.5:
                    events.append(j)
                    break
            state = "low"
        elif state is None:
            if signal_data[i] > high_thresh:
                state = "high"
            elif signal_data[i] < low_thresh:
                state = "low"

    filtered = []
    for e in sorted(events):
        if len(filtered) == 0 or e - filtered[-1] > min_gap_samples:
            filtered.append(e)

    return np.array(filtered)


def match_events(ref_onsets, target_onsets, max_diff=0.25):
    if len(target_onsets) == 0:
        return np.full(len(ref_onsets), np.nan)
    matched = []
    for ref_t in ref_onsets:
        diffs = np.abs(target_onsets - ref_t)
        min_diff = np.min(diffs)
        if min_diff < max_diff:
            matched.append(target_onsets[np.argmin(diffs)])
        else:
            matched.append(np.nan)
    return np.array(matched)


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Delay Distribution Analysis", fontsize=14)

for col, filename in enumerate(["test1_robust.xdf", "test1_windowed.xdf"]):
    print(f"\n{'='*70}")
    print(f"FILE: {filename}")
    print("=" * 70)

    streams, _ = pyxdf.load_xdf(filename, synchronize_clocks=True, handle_clock_resets=True, dejitter_timestamps=False)

    # Get and process signals
    lux_raw, lux_ts = find_channel(streams, "OpenSignals", "LUX")
    new_raw, new_ts = find_channel(streams, f"Muse-OPTICS ({NEW_FIRMWARE_MAC})", "OPTICS_RI_IR")
    old_raw, old_ts = find_channel(streams, f"Muse-OPTICS ({OLD_FIRMWARE_MAC})", "OPTICS_RI_IR")

    fs_lux = len(lux_ts) / (lux_ts[-1] - lux_ts[0])
    fs_optics = len(new_ts) / (new_ts[-1] - new_ts[0])

    lux_norm = rolling_normalize(median_filter(lux_raw, 3), 5.0, fs_lux)
    new_norm = rolling_normalize(median_filter(new_raw, 3), 5.0, fs_optics)
    old_norm = rolling_normalize(median_filter(old_raw, 3), 5.0, fs_optics)

    # Find events
    lux_events = find_events_midpoint(lux_norm, lux_ts)
    new_events = find_events_midpoint(new_norm, new_ts)
    old_events = find_events_midpoint(old_norm, old_ts)

    lux_onsets = lux_ts[lux_events]
    new_onsets = new_ts[new_events]
    old_onsets = old_ts[old_events]

    # Match and compute delays
    matched_new = match_events(lux_onsets, new_onsets)
    matched_old = match_events(lux_onsets, old_onsets)

    delays_new = (lux_onsets - matched_new) * 1000  # ms
    delays_old = (lux_onsets - matched_old) * 1000

    valid_new = ~np.isnan(delays_new) & (np.abs(delays_new) < 250)
    valid_old = ~np.isnan(delays_old) & (np.abs(delays_old) < 250)

    print(f"\nOLD delays: mean={np.nanmean(delays_old[valid_old]):.1f}ms, std={np.nanstd(delays_old[valid_old]):.1f}ms")
    print(f"NEW delays: mean={np.nanmean(delays_new[valid_new]):.1f}ms, std={np.nanstd(delays_new[valid_new]):.1f}ms")

    # Histogram of delays
    ax = axes[0, col]
    ax.hist(delays_old[valid_old], bins=100, alpha=0.7, color="red", label="OLD", density=True)
    ax.hist(delays_new[valid_new], bins=100, alpha=0.5, color="green", label="NEW", density=True)
    ax.set_xlabel("LUX - OPTICS delay (ms)")
    ax.set_ylabel("Density")
    ax.set_title(f"{filename.replace('.xdf', '')}")
    ax.legend()
    ax.set_xlim(-200, 200)

    # Time series of delays
    ax = axes[1, col]
    time_min = (lux_onsets - lux_onsets[0]) / 60
    ax.scatter(time_min[valid_old], delays_old[valid_old], s=5, alpha=0.5, c="red", label="OLD")
    ax.scatter(time_min[valid_new], delays_new[valid_new], s=5, alpha=0.5, c="green", label="NEW")
    ax.axhline(0, color="black", ls="--", alpha=0.5)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Delay (ms)")
    ax.set_ylim(-200, 200)
    ax.legend()

    # Check if delays are bimodal
    print(f"\nNEW delay distribution:")
    print(f"  10th percentile: {np.percentile(delays_new[valid_new], 10):.1f}ms")
    print(f"  25th percentile: {np.percentile(delays_new[valid_new], 25):.1f}ms")
    print(f"  50th percentile: {np.percentile(delays_new[valid_new], 50):.1f}ms")
    print(f"  75th percentile: {np.percentile(delays_new[valid_new], 75):.1f}ms")
    print(f"  90th percentile: {np.percentile(delays_new[valid_new], 90):.1f}ms")

plt.tight_layout()
plt.savefig("delay_analysis.png", dpi=150, bbox_inches="tight")
print("\nSaved: delay_analysis.png")
plt.show()
