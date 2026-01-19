"""
Diagnostic script to analyze OPTICS channels and understand their behavior.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pyxdf
from scipy import ndimage

# Muse device MACs
OLD_FIRMWARE_MAC = "00:55:DA:B9:FA:20"
NEW_FIRMWARE_MAC = "00:55:DA:BB:CD:CD"

OPTICS_CHANNELS = ["OPTICS_RI_AMB", "OPTICS_RI_RED", "OPTICS_RI_IR", "OPTICS_LI_AMB", "OPTICS_LI_RED", "OPTICS_LI_IR"]


def rolling_normalize(signal_data, window_sec=10.0, fs=64.0):
    """Apply rolling min-max normalization."""
    window_samples = max(int(window_sec * fs), 1)
    rolling_min = ndimage.minimum_filter1d(signal_data, size=window_samples, mode="nearest")
    rolling_max = ndimage.maximum_filter1d(signal_data, size=window_samples, mode="nearest")
    range_vals = rolling_max - rolling_min
    range_vals[range_vals < 1e-10] = 1e-10
    return (signal_data - rolling_min) / range_vals


def find_channel_raw(streams, stream_name, channel_name):
    """Find channel and return RAW signal (no normalization)."""
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
            signal_data = np.array(s["time_series"])[:, channels.index(ch_name)]
            ts = s["time_stamps"]
            return signal_data, ts
    return None, None


def analyze_file(filename):
    """Analyze OPTICS channels in detail for one file."""
    print(f"\n{'='*70}")
    print(f"Analyzing: {filename}")
    print("=" * 70)

    streams, _ = pyxdf.load_xdf(filename, synchronize_clocks=True, handle_clock_resets=True, dejitter_timestamps=False)

    # Get LUX as reference
    lux, lux_ts = find_channel_raw(streams, "OpenSignals", "LUX")
    if lux is None:
        print("  No LUX found!")
        return

    # Get timestamps info
    print(f"\nLUX: {len(lux)} samples, range [{lux_ts[0]:.1f}, {lux_ts[-1]:.1f}]")

    # Find when experiment actually ends (last marker)
    for s in streams:
        name = s["info"].get("name", ["Unnamed"])[0]
        if "jsPsych" in name.lower() or "marker" in name.lower():
            markers = np.array(s["time_series"]).flatten()
            marker_ts = s["time_stamps"]
            last_marker_time = marker_ts[-1]
            print(f"Last marker at: {last_marker_time:.1f}s (recording ends at {lux_ts[-1]:.1f}s)")
            print(f"  Extra recording time after experiment: {lux_ts[-1] - last_marker_time:.1f}s")
            break

    # Create figure for this file
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle(f"OPTICS Channel Analysis: {filename}", fontsize=14)

    # Time window for detailed view (start of recording)
    t_start = lux_ts[0] + 6
    t_end = t_start + 10  # 10 second window

    # Plot LUX in first row
    ax_lux = axes[0, 0]
    mask = (lux_ts >= t_start) & (lux_ts <= t_end)
    lux_norm = rolling_normalize(lux, window_sec=10.0, fs=len(lux_ts) / (lux_ts[-1] - lux_ts[0]))
    ax_lux.plot(lux_ts[mask] - t_start, lux_norm[mask], "b-", lw=1.5, label="LUX (normalized)")
    ax_lux.set_title("LUX (Reference)")
    ax_lux.set_ylabel("Normalized")
    ax_lux.legend()
    ax_lux.grid(True, alpha=0.3)

    # Analyze each OPTICS channel for both devices
    for device_idx, (device_label, device_mac) in enumerate([("OLD", OLD_FIRMWARE_MAC), ("NEW", NEW_FIRMWARE_MAC)]):
        stream_name = f"Muse-OPTICS ({device_mac})"

        print(f"\n{device_label} Device ({device_mac}):")

        all_channels = [
            "OPTICS_RI_AMB",
            "OPTICS_RI_RED",
            "OPTICS_RI_IR",
            "OPTICS_LI_AMB",
            "OPTICS_LI_RED",
            "OPTICS_LI_IR",
        ]
        for ch_idx, ch_name in enumerate(all_channels):
            signal, ts = find_channel_raw(streams, stream_name, ch_name)

            row = 1 + device_idx
            col = ch_idx % 3  # Wrap to 3 columns
            if ch_idx >= 3:
                continue  # Skip LI channels in plot but still analyze
            ax = axes[row, col]

            if signal is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_title(f"{device_label} - {ch_name}")
                continue

            # Get time-aligned mask
            mask = (ts >= t_start) & (ts <= t_end)

            if not np.any(mask):
                ax.text(0.5, 0.5, "No data in window", ha="center", va="center")
                ax.set_title(f"{device_label} - {ch_name}")
                continue

            # Raw signal stats
            raw_min, raw_max = np.min(signal), np.max(signal)
            raw_range = raw_max - raw_min

            # Normalize for plotting
            fs = len(ts) / (ts[-1] - ts[0])
            signal_norm = rolling_normalize(signal, window_sec=10.0, fs=fs)

            # Plot
            ax.plot(
                ts[mask] - t_start,
                signal_norm[mask],
                "-",
                lw=1,
                color="red" if device_label == "OLD" else "green",
                label=f'{ch_name.split("_")[-1]}',
            )

            # Add LUX for reference
            lux_mask = (lux_ts >= t_start) & (lux_ts <= t_end)
            ax.plot(lux_ts[lux_mask] - t_start, lux_norm[lux_mask], "b-", lw=0.5, alpha=0.5)

            ax.set_title(f"{device_label} - {ch_name}\nRange: {raw_range:.0f}")
            ax.set_ylabel("Normalized")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

            # Calculate correlation with LUX
            # Resample to common timestamps
            from scipy.interpolate import interp1d

            common_ts = np.linspace(t_start, t_end, 500)

            lux_interp = interp1d(lux_ts, lux_norm, bounds_error=False, fill_value=np.nan)
            optics_interp = interp1d(ts, signal_norm, bounds_error=False, fill_value=np.nan)

            lux_resampled = lux_interp(common_ts)
            optics_resampled = optics_interp(common_ts)

            valid = ~np.isnan(lux_resampled) & ~np.isnan(optics_resampled)
            if np.sum(valid) > 10:
                corr = np.corrcoef(lux_resampled[valid], optics_resampled[valid])[0, 1]
                print(f"  {ch_name}: range={raw_range:.0f}, corr with LUX = {corr:.3f}")

    # Bottom row: Show derivative/transitions
    for ch_idx, ch_name in enumerate(["OPTICS_RI_AMB", "OPTICS_RI_RED", "OPTICS_RI_IR"]):
        ax = axes[3, ch_idx]

        # Get NEW device signal
        signal, ts = find_channel_raw(streams, f"Muse-OPTICS ({NEW_FIRMWARE_MAC})", ch_name)
        if signal is None:
            continue

        mask = (ts >= t_start) & (ts <= t_end)
        if not np.any(mask):
            continue

        fs = len(ts) / (ts[-1] - ts[0])
        signal_norm = rolling_normalize(signal, window_sec=10.0, fs=fs)

        # Calculate derivative
        derivative = np.gradient(signal_norm)

        ax.plot(ts[mask] - t_start, derivative[mask], "g-", lw=1, label="Derivative")
        ax.axhline(0, color="black", ls="--", alpha=0.5)
        ax.set_title(f"NEW - {ch_name} Derivative")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("d/dt")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"optics_analysis_{filename.replace('.xdf', '')}.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: optics_analysis_{filename.replace('.xdf', '')}.png")
    plt.show()


def main():
    import sys

    xdf_files = sorted([f for f in os.listdir(".") if f.endswith(".xdf")])

    if not xdf_files:
        print("No XDF files found!")
        return

    # If a specific file is requested, analyze that
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if target in xdf_files:
            analyze_file(target)
        else:
            print(f"File not found: {target}")
            print(f"Available: {xdf_files}")
    else:
        # Analyze all files
        print(f"Found {len(xdf_files)} XDF files: {xdf_files}")
        for f in xdf_files:
            analyze_file(f)


if __name__ == "__main__":
    main()
