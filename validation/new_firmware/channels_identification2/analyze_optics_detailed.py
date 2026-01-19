"""
Detailed OPTICS Investigation: Looking for Channel Inversion
=============================================================

Based on the initial analysis, we found that BOTH devices show POSITIVE
INNER-OUTER correlation in this recording. This is different from the
summary.txt findings where the NEW firmware showed -0.97 correlation.

Let's investigate more carefully:
1. Time-varying correlation analysis
2. Cross-correlation with lag analysis
3. Raw signal visualization
4. Individual wavelength analysis
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyxdf
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

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


def analyze_optics_detailed(streams):
    """Detailed OPTICS channel analysis."""

    old_stream = get_stream_by_name(streams, f"OPTICS ({OLD_FIRMWARE_MAC})")
    new_stream = get_stream_by_name(streams, f"OPTICS ({NEW_FIRMWARE_MAC})")

    old_data = np.array(old_stream["time_series"])
    old_ts = old_stream["time_stamps"]
    new_data = np.array(new_stream["time_series"])
    new_ts = new_stream["time_stamps"]

    old_channels = get_channel_labels(old_stream)
    new_channels = get_channel_labels(new_stream)

    # Common time range
    t_start = max(old_ts.min(), new_ts.min())
    t_end = min(old_ts.max(), new_ts.max())

    print("=" * 80)
    print("DETAILED OPTICS ANALYSIS")
    print("=" * 80)

    # --- 1. Full Channel Correlation Matrix (within each device) ---
    print("\n--- WITHIN-DEVICE CORRELATION ANALYSIS ---")

    # Group channels by type
    channel_groups = {
        "NIR": ["OPTICS_LO_NIR", "OPTICS_RO_NIR", "OPTICS_LI_NIR", "OPTICS_RI_NIR"],
        "IR": ["OPTICS_LO_IR", "OPTICS_RO_IR", "OPTICS_LI_IR", "OPTICS_RI_IR"],
        "RED": ["OPTICS_LO_RED", "OPTICS_RO_RED", "OPTICS_LI_RED", "OPTICS_RI_RED"],
        "AMB": ["OPTICS_LO_AMB", "OPTICS_RO_AMB", "OPTICS_LI_AMB", "OPTICS_RI_AMB"],
    }

    for wavelength, channels in channel_groups.items():
        print(f"\n{wavelength} channels:")

        # OLD device
        print(f"  OLD firmware ({OLD_FIRMWARE_MAC}):")
        for i in range(len(channels)):
            for j in range(i + 1, len(channels)):
                ch1, ch2 = channels[i], channels[j]
                if ch1 in old_channels and ch2 in old_channels:
                    idx1, idx2 = old_channels.index(ch1), old_channels.index(ch2)
                    corr = np.corrcoef(old_data[:, idx1], old_data[:, idx2])[0, 1]
                    print(f"    {ch1.split('_')[1]}-{ch2.split('_')[1]}: {corr:+.4f}")

        # NEW device
        print(f"  NEW firmware ({NEW_FIRMWARE_MAC}):")
        for i in range(len(channels)):
            for j in range(i + 1, len(channels)):
                ch1, ch2 = channels[i], channels[j]
                if ch1 in new_channels and ch2 in new_channels:
                    idx1, idx2 = new_channels.index(ch1), new_channels.index(ch2)
                    corr = np.corrcoef(new_data[:, idx1], new_data[:, idx2])[0, 1]
                    print(f"    {ch1.split('_')[1]}-{ch2.split('_')[1]}: {corr:+.4f}")

    # --- 2. Cross-device correlation for each channel ---
    print("\n" + "=" * 80)
    print("CROSS-DEVICE CORRELATION ANALYSIS")
    print("=" * 80)

    def resample_to_common(data1, ts1, data2, ts2, ch_idx):
        t_start = max(ts1.min(), ts2.min())
        t_end = min(ts1.max(), ts2.max())
        n_samples = int((t_end - t_start) * 64)
        common_ts = np.linspace(t_start, t_end, n_samples)

        interp1 = interp1d(ts1, data1[:, ch_idx], kind="linear", bounds_error=False, fill_value=np.nan)
        interp2 = interp1d(ts2, data2[:, ch_idx], kind="linear", bounds_error=False, fill_value=np.nan)

        sig1 = interp1(common_ts)
        sig2 = interp2(common_ts)
        valid = ~np.isnan(sig1) & ~np.isnan(sig2)
        return sig1[valid], sig2[valid], common_ts[valid]

    print("\nCross-device correlation by channel type:")
    for wavelength, channels in channel_groups.items():
        print(f"\n{wavelength}:")
        correlations = []
        for ch in channels:
            if ch in old_channels and ch in new_channels:
                idx_old = old_channels.index(ch)
                idx_new = new_channels.index(ch)
                sig_old, sig_new, _ = resample_to_common(old_data, old_ts, new_data, new_ts, idx_old)
                if len(sig_old) > 10:
                    corr = np.corrcoef(sig_old, sig_new)[0, 1]
                    correlations.append(corr)
                    location = ch.split("_")[1]  # LO, RO, LI, RI
                    print(f"  {location}: {corr:+.4f}")

        if correlations:
            print(f"  Mean: {np.mean(correlations):+.4f}")

    # --- 3. Visualize raw signals ---
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    # Plot all 16 channels for both devices
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    fig.suptitle("All OPTICS Channels: 16-channel comparison (first 30s)", fontsize=14)

    # Time window for plotting
    old_mask = old_ts - t_start <= 30
    new_mask = new_ts - t_start <= 30

    for i, ch in enumerate(old_channels[:16]):
        ax = axes[i // 4, i % 4]

        # Normalize signals for comparison
        old_sig = old_data[old_mask, i]
        new_sig = new_data[new_mask, i]

        old_norm = (old_sig - old_sig.mean()) / (old_sig.std() + 1e-10)
        new_norm = (new_sig - new_sig.mean()) / (new_sig.std() + 1e-10)

        ax.plot(old_ts[old_mask] - t_start, old_norm, label="Old", alpha=0.7, linewidth=0.5)
        ax.plot(new_ts[new_mask] - t_start, new_norm + 3, label="New (+3)", alpha=0.7, linewidth=0.5)
        ax.set_title(ch.replace("OPTICS_", ""), fontsize=10)
        ax.set_ylim([-4, 8])
        if i == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("optics_all_channels_comparison.png", dpi=150)
    print("→ Saved: optics_all_channels_comparison.png")

    # --- 4. Time-varying correlation for AMB channels ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Time-varying INNER-OUTER AMB correlation (30s windows)", fontsize=14)

    window_size = 30 * 64  # 30 seconds at 64 Hz
    step_size = 5 * 64  # 5 second steps

    pairs = [
        ("OLD", old_data, old_ts, old_channels, OLD_FIRMWARE_MAC),
        ("NEW", new_data, new_ts, new_channels, NEW_FIRMWARE_MAC),
    ]

    for row, (label, data, ts, channels, mac) in enumerate(pairs):
        lo_idx = channels.index("OPTICS_LO_AMB")
        li_idx = channels.index("OPTICS_LI_AMB")
        ro_idx = channels.index("OPTICS_RO_AMB")
        ri_idx = channels.index("OPTICS_RI_AMB")

        # Left side
        correlations_left = []
        times = []
        for start in range(0, len(data) - window_size, step_size):
            window_lo = data[start : start + window_size, lo_idx]
            window_li = data[start : start + window_size, li_idx]
            corr = np.corrcoef(window_lo, window_li)[0, 1]
            correlations_left.append(corr)
            times.append(ts[start] - ts[0])

        axes[row, 0].plot(times, correlations_left, "b-", marker="o", markersize=3)
        axes[row, 0].axhline(y=0, color="r", linestyle="--", alpha=0.5)
        axes[row, 0].set_title(f"{label} ({mac}) - LEFT OUTER-INNER AMB")
        axes[row, 0].set_ylabel("Correlation")
        axes[row, 0].set_ylim([-1.1, 1.1])
        axes[row, 0].grid(True, alpha=0.3)

        # Right side
        correlations_right = []
        times = []
        for start in range(0, len(data) - window_size, step_size):
            window_ro = data[start : start + window_size, ro_idx]
            window_ri = data[start : start + window_size, ri_idx]
            corr = np.corrcoef(window_ro, window_ri)[0, 1]
            correlations_right.append(corr)
            times.append(ts[start] - ts[0])

        axes[row, 1].plot(times, correlations_right, "g-", marker="o", markersize=3)
        axes[row, 1].axhline(y=0, color="r", linestyle="--", alpha=0.5)
        axes[row, 1].set_title(f"{label} ({mac}) - RIGHT OUTER-INNER AMB")
        axes[row, 1].set_ylim([-1.1, 1.1])
        axes[row, 1].grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig("optics_timevarying_correlation.png", dpi=150)
    print("→ Saved: optics_timevarying_correlation.png")

    # --- 5. Scatter plots for INNER vs OUTER AMB ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("INNER vs OUTER AMB Scatter Plots", fontsize=14)

    for row, (label, data, ts, channels, mac) in enumerate(pairs):
        lo_idx = channels.index("OPTICS_LO_AMB")
        li_idx = channels.index("OPTICS_LI_AMB")
        ro_idx = channels.index("OPTICS_RO_AMB")
        ri_idx = channels.index("OPTICS_RI_AMB")

        # Sample every 10th point for visibility
        sample_idx = slice(None, None, 10)

        # Left side
        ax = axes[row, 0]
        ax.scatter(data[sample_idx, lo_idx], data[sample_idx, li_idx], alpha=0.3, s=1, c="blue")
        corr = np.corrcoef(data[:, lo_idx], data[:, li_idx])[0, 1]
        ax.set_title(f"{label} - LEFT: r={corr:.4f}")
        ax.set_xlabel("OUTER AMB")
        ax.set_ylabel("INNER AMB")

        # Right side
        ax = axes[row, 1]
        ax.scatter(data[sample_idx, ro_idx], data[sample_idx, ri_idx], alpha=0.3, s=1, c="green")
        corr = np.corrcoef(data[:, ro_idx], data[:, ri_idx])[0, 1]
        ax.set_title(f"{label} - RIGHT: r={corr:.4f}")
        ax.set_xlabel("OUTER AMB")
        ax.set_ylabel("INNER AMB")

    plt.tight_layout()
    plt.savefig("optics_scatter_plots.png", dpi=150)
    print("→ Saved: optics_scatter_plots.png")

    plt.show()


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    streams, _ = load_xdf(XDF_FILE)
    analyze_optics_detailed(streams)
    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
