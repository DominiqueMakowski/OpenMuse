"""
Debug script to understand why NEW device delays are scattered despite
seemingly aligned signals in the signal plot.
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
    """Find channel and return raw signal."""
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
    """Schmitt trigger event detection."""
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


# Compare robust (bad NEW) vs windowed (good NEW)
for filename in ["test1_robust.xdf", "test1_windowed.xdf"]:
    print(f"\n{'='*70}")
    print(f"Analyzing: {filename}")
    print("=" * 70)

    streams, _ = pyxdf.load_xdf(filename, synchronize_clocks=True, handle_clock_resets=True, dejitter_timestamps=False)

    # Get LUX
    lux_raw, lux_ts = find_channel(streams, "OpenSignals", "LUX")
    fs_lux = len(lux_ts) / (lux_ts[-1] - lux_ts[0])
    lux_filt = median_filter(lux_raw, kernel_size=3)
    lux_norm = rolling_normalize(lux_filt, window_sec=5.0, fs=fs_lux)

    # Get OPTICS
    new_raw, new_ts = find_channel(streams, f"Muse-OPTICS ({NEW_FIRMWARE_MAC})", "OPTICS_RI_IR")
    old_raw, old_ts = find_channel(streams, f"Muse-OPTICS ({OLD_FIRMWARE_MAC})", "OPTICS_RI_IR")

    fs_optics = len(new_ts) / (new_ts[-1] - new_ts[0]) if new_ts is not None else 64

    new_filt = median_filter(new_raw, kernel_size=3)
    new_norm = rolling_normalize(new_filt, window_sec=5.0, fs=fs_optics)

    old_filt = median_filter(old_raw, kernel_size=3)
    old_norm = rolling_normalize(old_filt, window_sec=5.0, fs=fs_optics)

    # Find events
    lux_events = find_events_midpoint(lux_norm, lux_ts)
    new_events = find_events_midpoint(new_norm, new_ts)
    old_events = find_events_midpoint(old_norm, old_ts)

    print(f"LUX events: {len(lux_events)}")
    print(f"OLD events: {len(old_events)}")
    print(f"NEW events: {len(new_events)}")

    # Calculate event density (events per minute)
    duration_min = (lux_ts[-1] - lux_ts[0]) / 60
    print(f"Duration: {duration_min:.1f} minutes")
    print(f"LUX rate: {len(lux_events)/duration_min:.1f} events/min")
    print(f"OLD rate: {len(old_events)/duration_min:.1f} events/min")
    print(f"NEW rate: {len(new_events)/duration_min:.1f} events/min")

    # Check signal quality
    # Calculate how often signal is in "transition zone" vs stable high/low
    in_transition = np.sum((new_norm > 0.35) & (new_norm < 0.65)) / len(new_norm) * 100
    print(f"\nNEW signal in transition zone (0.35-0.65): {in_transition:.1f}%")

    in_transition_old = np.sum((old_norm > 0.35) & (old_norm < 0.65)) / len(old_norm) * 100
    print(f"OLD signal in transition zone (0.35-0.65): {in_transition_old:.1f}%")

    # Plot detailed view of 30 seconds
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(f"{filename}: Signal Quality Analysis", fontsize=14)

    t_start = lux_ts[0] + 60  # Start 1 minute in
    t_end = t_start + 30  # 30 second window

    # LUX
    ax = axes[0]
    mask = (lux_ts >= t_start) & (lux_ts <= t_end)
    ax.plot(lux_ts[mask] - t_start, lux_norm[mask], "b-", lw=1, label="LUX normalized")
    ax.axhline(0.5, color="gray", ls="--", alpha=0.5)
    ax.axhline(0.65, color="red", ls=":", alpha=0.5)
    ax.axhline(0.35, color="red", ls=":", alpha=0.5)
    # Mark events
    for e in lux_events:
        if t_start <= lux_ts[e] <= t_end:
            ax.axvline(lux_ts[e] - t_start, color="blue", alpha=0.3, lw=1)
    ax.set_ylabel("LUX")
    ax.legend()
    ax.set_ylim(-0.1, 1.1)

    # OLD
    ax = axes[1]
    mask = (old_ts >= t_start) & (old_ts <= t_end)
    ax.plot(old_ts[mask] - t_start, old_norm[mask], "r-", lw=1, label="OLD IR normalized")
    ax.axhline(0.5, color="gray", ls="--", alpha=0.5)
    ax.axhline(0.65, color="red", ls=":", alpha=0.5)
    ax.axhline(0.35, color="red", ls=":", alpha=0.5)
    for e in old_events:
        if t_start <= old_ts[e] <= t_end:
            ax.axvline(old_ts[e] - t_start, color="red", alpha=0.3, lw=1)
    ax.set_ylabel("OLD")
    ax.legend()
    ax.set_ylim(-0.1, 1.1)

    # NEW
    ax = axes[2]
    mask = (new_ts >= t_start) & (new_ts <= t_end)
    ax.plot(new_ts[mask] - t_start, new_norm[mask], "g-", lw=1, label="NEW IR normalized")
    ax.axhline(0.5, color="gray", ls="--", alpha=0.5)
    ax.axhline(0.65, color="red", ls=":", alpha=0.5, label="Hysteresis thresholds")
    ax.axhline(0.35, color="red", ls=":", alpha=0.5)
    for e in new_events:
        if t_start <= new_ts[e] <= t_end:
            ax.axvline(new_ts[e] - t_start, color="green", alpha=0.3, lw=1)
    ax.set_ylabel("NEW")
    ax.set_xlabel("Time (s)")
    ax.legend()
    ax.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.savefig(f"debug_{filename.replace('.xdf', '')}.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: debug_{filename.replace('.xdf', '')}.png")
    plt.close()
