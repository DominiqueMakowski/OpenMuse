"""
Dual Device Analysis: Old Firmware vs New Firmware
==================================================

Recording Setup:
- Old Muse: Bottom of forehead (closer to eyebrows)
- New Muse: Top of forehead

Recording Protocol:
- 1 min rest
- 1 min strong blinks (10 both eyes, 10 left, 10 right)
- 1 min head movements in various directions

Analysis Goals:
1. ACCGYRO correlation (validate channel mapping for new firmware)
2. EEG correlation + power spectrum comparison
3. OPTICS channel investigation (reversed channels issue)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyxdf
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

# --- Configuration ---
XDF_FILE = "test1_eeg.xdf"

# Device identifiers (based on summary.txt)
OLD_FIRMWARE_MAC = "00:55:DA:B9:FA:20"  # Old firmware
NEW_FIRMWARE_MAC = "00:55:DA:BB:CD:CD"  # New firmware


def load_xdf(filename):
    """Load XDF file and return streams with metadata."""
    print(f"Loading {filename}...")
    streams, header = pyxdf.load_xdf(
        filename,
        synchronize_clocks=True,
        handle_clock_resets=True,
        dejitter_timestamps=False,  # Use actual timestamps for quality analysis
    )
    return streams, header


def summarize_streams(streams):
    """Print summary of all streams in the XDF file."""
    print("\n" + "=" * 80)
    print("STREAM SUMMARY")
    print("=" * 80)

    data = []
    for i, stream in enumerate(streams):
        name = stream["info"].get("name", ["Unnamed"])[0]
        n_samples = len(stream["time_stamps"])

        if n_samples == 0:
            continue

        ts_min = stream["time_stamps"].min()
        ts_max = stream["time_stamps"].max()
        duration = ts_max - ts_min
        nominal_srate = float(stream["info"]["nominal_srate"][0])
        effective_srate = n_samples / duration if duration > 0 else np.nan

        # Get channel info
        try:
            channels = [d["label"][0] for d in stream["info"]["desc"][0]["channels"][0]["channel"]]
            n_channels = len(channels)
        except (KeyError, TypeError, IndexError):
            n_channels = stream["time_series"].shape[1] if len(stream["time_series"].shape) > 1 else 1
            channels = []

        data.append(
            {
                "Index": i,
                "Stream": name,
                "Samples": n_samples,
                "Channels": n_channels,
                "Duration (s)": f"{duration:.1f}",
                "Nominal SR": f"{nominal_srate:.1f}",
                "Effective SR": f"{effective_srate:.2f}",
                "SR Error %": f"{100*(effective_srate/nominal_srate - 1):.2f}" if nominal_srate > 0 else "N/A",
            }
        )

    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    print()
    return df


def get_stream_by_name(streams, name_substring):
    """Find a stream by partial name match."""
    for stream in streams:
        name = stream["info"].get("name", ["Unnamed"])[0]
        if name_substring in name:
            return stream
    return None


def get_channel_labels(stream):
    """Extract channel labels from stream info."""
    try:
        return [d["label"][0] for d in stream["info"]["desc"][0]["channels"][0]["channel"]]
    except (KeyError, TypeError, IndexError):
        n_ch = stream["time_series"].shape[1] if len(stream["time_series"].shape) > 1 else 1
        return [f"Ch_{i}" for i in range(n_ch)]


def resample_to_common_time(sig1, ts1, sig2, ts2, target_fs=None):
    """Resample two signals to common timestamps for correlation analysis."""
    t_start = max(ts1.min(), ts2.min())
    t_end = min(ts1.max(), ts2.max())

    if target_fs is None:
        # Use higher of the two effective sampling rates
        fs1 = len(ts1) / (ts1.max() - ts1.min())
        fs2 = len(ts2) / (ts2.max() - ts2.min())
        target_fs = max(fs1, fs2)

    n_samples = int((t_end - t_start) * target_fs)
    common_ts = np.linspace(t_start, t_end, n_samples)

    interp1 = interp1d(ts1, sig1, kind="linear", bounds_error=False, fill_value=np.nan)
    interp2 = interp1d(ts2, sig2, kind="linear", bounds_error=False, fill_value=np.nan)

    sig1_r = interp1(common_ts)
    sig2_r = interp2(common_ts)

    valid = ~np.isnan(sig1_r) & ~np.isnan(sig2_r)
    return sig1_r[valid], sig2_r[valid], common_ts[valid]


def analyze_accgyro_correlation(streams):
    """
    Analyze ACCGYRO correlation between old and new firmware devices.

    This validates that our channel mapping is correct for the new firmware.
    If both devices were on the same head, their accelerometer and gyroscope
    readings should be highly correlated (same head movements).
    """
    print("\n" + "=" * 80)
    print("1. ACCGYRO CORRELATION ANALYSIS")
    print("=" * 80)

    # Find ACCGYRO streams
    old_stream = get_stream_by_name(streams, f"ACCGYRO ({OLD_FIRMWARE_MAC})")
    new_stream = get_stream_by_name(streams, f"ACCGYRO ({NEW_FIRMWARE_MAC})")

    if old_stream is None or new_stream is None:
        print("ERROR: Could not find both ACCGYRO streams!")
        print("Available streams:")
        for s in streams:
            print(f"  - {s['info'].get('name', ['?'])[0]}")
        return None

    print(f"\nOld firmware stream: {old_stream['info']['name'][0]}")
    print(f"New firmware stream: {new_stream['info']['name'][0]}")

    old_data = np.array(old_stream["time_series"])
    old_ts = old_stream["time_stamps"]
    new_data = np.array(new_stream["time_series"])
    new_ts = new_stream["time_stamps"]

    old_channels = get_channel_labels(old_stream)
    new_channels = get_channel_labels(new_stream)

    print(f"\nOld channels: {old_channels}")
    print(f"New channels: {new_channels}")

    # Compute channel-by-channel correlation
    print("\n--- Channel Correlation (Same Channel Names) ---")
    print(f"{'Channel':<12} {'Correlation':>12} {'p-value':>12}")
    print("-" * 40)

    correlations = {}
    for i, ch in enumerate(old_channels):
        sig1, sig2, _ = resample_to_common_time(old_data[:, i], old_ts, new_data[:, i], new_ts, target_fs=52)
        if len(sig1) > 10:
            corr, pval = pearsonr(sig1, sig2)
            correlations[ch] = corr
            print(f"{ch:<12} {corr:>12.4f} {pval:>12.2e}")
        else:
            print(f"{ch:<12} {'N/A':>12} {'N/A':>12}")

    # Summary
    acc_corr = [correlations[ch] for ch in correlations if ch.startswith("ACC")]
    gyro_corr = [correlations[ch] for ch in correlations if ch.startswith("GYRO")]

    print(f"\nACCELEROMETER mean |correlation|: {np.mean(np.abs(acc_corr)):.4f}")
    print(f"GYROSCOPE mean |correlation|: {np.mean(np.abs(gyro_corr)):.4f}")

    # Check for sign inversions (which would indicate channel mapping issues)
    print("\n--- Sign Check ---")
    sign_issues = []
    for ch, corr in correlations.items():
        if corr < 0:
            sign_issues.append(ch)
            print(f"  WARNING: {ch} has NEGATIVE correlation ({corr:.4f})")

    if not sign_issues:
        print("  All channels have POSITIVE correlation (✓ consistent mapping)")
    else:
        print(f"\n  {len(sign_issues)} channel(s) may have inverted polarity!")

    return correlations


def analyze_eeg_correlation(streams):
    """
    Analyze EEG correlation and power spectrum comparison.

    Even though the devices were at different forehead positions,
    we expect some correlation in low-frequency artifacts (blinks, movements)
    and similar overall power spectra.
    """
    print("\n" + "=" * 80)
    print("2. EEG CORRELATION & POWER SPECTRUM ANALYSIS")
    print("=" * 80)

    # Find EEG streams
    old_stream = get_stream_by_name(streams, f"EEG ({OLD_FIRMWARE_MAC})")
    new_stream = get_stream_by_name(streams, f"EEG ({NEW_FIRMWARE_MAC})")

    if old_stream is None or new_stream is None:
        print("ERROR: Could not find both EEG streams!")
        return None

    print(f"\nOld firmware stream: {old_stream['info']['name'][0]}")
    print(f"New firmware stream: {new_stream['info']['name'][0]}")

    old_data = np.array(old_stream["time_series"])
    old_ts = old_stream["time_stamps"]
    new_data = np.array(new_stream["time_series"])
    new_ts = new_stream["time_stamps"]

    old_channels = get_channel_labels(old_stream)
    new_channels = get_channel_labels(new_stream)

    print(f"\nOld channels ({len(old_channels)}): {old_channels[:4]}")  # First 4 are EEG
    print(f"New channels ({len(new_channels)}): {new_channels[:4]}")

    # --- Sampling Rate Analysis ---
    print("\n--- Effective Sampling Rate ---")
    old_duration = old_ts.max() - old_ts.min()
    new_duration = new_ts.max() - new_ts.min()
    old_effective_sr = len(old_ts) / old_duration
    new_effective_sr = len(new_ts) / new_duration

    print(f"Old firmware: {old_effective_sr:.2f} Hz (nominal 256 Hz, error: {100*(old_effective_sr/256-1):.2f}%)")
    print(f"New firmware: {new_effective_sr:.2f} Hz (nominal 256 Hz, error: {100*(new_effective_sr/256-1):.2f}%)")

    # --- Channel Correlation (first 4 channels = EEG) ---
    print("\n--- EEG Channel Correlation (Same Channel Names) ---")
    print("Note: Devices at different head positions, so correlation may be modest")
    print(f"{'Channel':<12} {'Correlation':>12} {'p-value':>12}")
    print("-" * 40)

    correlations = {}
    for i in range(min(4, len(old_channels), len(new_channels))):
        ch = old_channels[i]
        sig1, sig2, _ = resample_to_common_time(old_data[:, i], old_ts, new_data[:, i], new_ts, target_fs=256)
        if len(sig1) > 10:
            corr, pval = pearsonr(sig1, sig2)
            correlations[ch] = corr
            print(f"{ch:<12} {corr:>12.4f} {pval:>12.2e}")

    # --- Power Spectrum Analysis ---
    print("\n--- Power Spectrum Comparison ---")

    # Use first EEG channel (TP9) for spectrum comparison
    fs = 256  # Nominal sampling rate

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("EEG Power Spectrum Comparison: Old vs New Firmware", fontsize=14)

    eeg_channels = ["EEG_TP9", "EEG_AF7", "EEG_AF8", "EEG_TP10"]

    for idx, ch in enumerate(eeg_channels[:4]):
        ax = axes[idx // 2, idx % 2]

        ch_idx = old_channels.index(ch) if ch in old_channels else idx

        # Compute PSD for both devices
        old_sig = old_data[:, ch_idx]
        new_sig = new_data[:, ch_idx]

        # Use Welch's method
        nperseg = min(1024, len(old_sig) // 4)
        f_old, psd_old = scipy_signal.welch(old_sig, fs=fs, nperseg=nperseg)
        f_new, psd_new = scipy_signal.welch(new_sig, fs=fs, nperseg=nperseg)

        # Plot in dB
        ax.semilogy(f_old, psd_old, label="Old firmware", alpha=0.8)
        ax.semilogy(f_new, psd_new, label="New firmware", alpha=0.8)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (µV²/Hz)")
        ax.set_title(f"{ch}")
        ax.set_xlim([0, 50])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Print key band powers
        delta_mask = (f_old >= 1) & (f_old <= 4)
        alpha_mask = (f_old >= 8) & (f_old <= 12)

        delta_old = np.mean(psd_old[delta_mask]) if np.any(delta_mask) else 0
        delta_new = np.mean(psd_new[delta_mask]) if np.any(delta_mask) else 0
        alpha_old = np.mean(psd_old[alpha_mask]) if np.any(alpha_mask) else 0
        alpha_new = np.mean(psd_new[alpha_mask]) if np.any(alpha_mask) else 0

        print(f"\n{ch}:")
        print(f"  Delta (1-4 Hz): Old={delta_old:.2e}, New={delta_new:.2e}, Ratio={delta_new/delta_old:.2f}")
        print(f"  Alpha (8-12 Hz): Old={alpha_old:.2e}, New={alpha_new:.2e}, Ratio={alpha_new/alpha_old:.2f}")

    plt.tight_layout()
    plt.savefig("eeg_power_spectrum_comparison.png", dpi=150)
    print("\n→ Saved: eeg_power_spectrum_comparison.png")
    plt.show()

    return correlations


def analyze_optics_channels(streams):
    """
    Analyze OPTICS channels to investigate the reversed channel issue.

    Key investigation:
    - INNER vs OUTER correlation
    - AMB channel polarity (suspected inversion in new firmware)
    """
    print("\n" + "=" * 80)
    print("3. OPTICS CHANNEL ANALYSIS")
    print("=" * 80)

    # Find OPTICS streams
    old_stream = get_stream_by_name(streams, f"OPTICS ({OLD_FIRMWARE_MAC})")
    new_stream = get_stream_by_name(streams, f"OPTICS ({NEW_FIRMWARE_MAC})")

    if old_stream is None or new_stream is None:
        print("ERROR: Could not find both OPTICS streams!")
        return None

    print(f"\nOld firmware stream: {old_stream['info']['name'][0]}")
    print(f"New firmware stream: {new_stream['info']['name'][0]}")

    old_data = np.array(old_stream["time_series"])
    old_ts = old_stream["time_stamps"]
    new_data = np.array(new_stream["time_series"])
    new_ts = new_stream["time_stamps"]

    old_channels = get_channel_labels(old_stream)
    new_channels = get_channel_labels(new_stream)

    print(f"\nOld channels ({len(old_channels)}): {old_channels}")
    print(f"New channels ({len(new_channels)}): {new_channels}")

    # --- Cross-device correlation for each channel ---
    print("\n--- Cross-Device Correlation (Same Channel Names) ---")
    print(f"{'Channel':<18} {'Correlation':>12} {'|r|':>8}")
    print("-" * 42)

    correlations = {}
    for i, ch in enumerate(old_channels):
        if i >= len(new_channels):
            break
        sig1, sig2, _ = resample_to_common_time(old_data[:, i], old_ts, new_data[:, i], new_ts, target_fs=64)
        if len(sig1) > 10:
            corr, _ = pearsonr(sig1, sig2)
            correlations[ch] = corr
            print(f"{ch:<18} {corr:>12.4f} {abs(corr):>8.4f}")

    # --- Analyze INNER-OUTER correlation within each device ---
    print("\n--- INNER-OUTER AMB Correlation (within device) ---")
    print("Physical expectation: POSITIVE correlation (same ambient light source)")

    amb_channels = {
        "LEFT_OUTER": "OPTICS_LO_AMB",
        "LEFT_INNER": "OPTICS_LI_AMB",
        "RIGHT_OUTER": "OPTICS_RO_AMB",
        "RIGHT_INNER": "OPTICS_RI_AMB",
    }

    # Check if we have AMB channels
    if all(ch in old_channels for ch in amb_channels.values()):
        # Old device: OUTER vs INNER correlation
        lo_idx = old_channels.index("OPTICS_LO_AMB")
        li_idx = old_channels.index("OPTICS_LI_AMB")
        ro_idx = old_channels.index("OPTICS_RO_AMB")
        ri_idx = old_channels.index("OPTICS_RI_AMB")

        old_left_io_corr = np.corrcoef(old_data[:, lo_idx], old_data[:, li_idx])[0, 1]
        old_right_io_corr = np.corrcoef(old_data[:, ro_idx], old_data[:, ri_idx])[0, 1]

        print(f"\nOLD firmware ({OLD_FIRMWARE_MAC}):")
        print(f"  LEFT OUTER-INNER AMB correlation:  {old_left_io_corr:+.4f}")
        print(f"  RIGHT OUTER-INNER AMB correlation: {old_right_io_corr:+.4f}")

        # New device: OUTER vs INNER correlation
        lo_idx = new_channels.index("OPTICS_LO_AMB")
        li_idx = new_channels.index("OPTICS_LI_AMB")
        ro_idx = new_channels.index("OPTICS_RO_AMB")
        ri_idx = new_channels.index("OPTICS_RI_AMB")

        new_left_io_corr = np.corrcoef(new_data[:, lo_idx], new_data[:, li_idx])[0, 1]
        new_right_io_corr = np.corrcoef(new_data[:, ro_idx], new_data[:, ri_idx])[0, 1]

        print(f"\nNEW firmware ({NEW_FIRMWARE_MAC}):")
        print(f"  LEFT OUTER-INNER AMB correlation:  {new_left_io_corr:+.4f}")
        print(f"  RIGHT OUTER-INNER AMB correlation: {new_right_io_corr:+.4f}")

        # Summary
        print("\n--- Summary ---")
        if new_left_io_corr < 0 or new_right_io_corr < 0:
            print("⚠ NEW firmware shows NEGATIVE INNER-OUTER correlation!")
            print("  This confirms the suspected INNER AMB channel inversion issue.")
        else:
            print("✓ Both devices show POSITIVE INNER-OUTER correlation")
    else:
        print("  Could not find all AMB channels for analysis")

    # --- Plot time series comparison ---
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle("OPTICS AMB Channel Comparison: Old vs New Firmware", fontsize=14)

    # Use common time range
    t_start = max(old_ts.min(), new_ts.min())
    t_end = min(old_ts.max(), new_ts.max())

    # Only plot first 30 seconds for visibility
    plot_duration = min(30, t_end - t_start)
    t_plot_end = t_start + plot_duration

    old_mask = (old_ts >= t_start) & (old_ts <= t_plot_end)
    new_mask = (new_ts >= t_start) & (new_ts <= t_plot_end)

    amb_pairs = [
        ("OPTICS_LO_AMB", "OPTICS_LI_AMB", "LEFT"),
        ("OPTICS_RO_AMB", "OPTICS_RI_AMB", "RIGHT"),
    ]

    for ax_idx, (outer_ch, inner_ch, side) in enumerate(amb_pairs):
        if outer_ch in old_channels and inner_ch in old_channels:
            outer_idx = old_channels.index(outer_ch)
            inner_idx = old_channels.index(inner_ch)

            # Old device
            ax = axes[ax_idx * 2]
            ax.plot(old_ts[old_mask] - t_start, old_data[old_mask, outer_idx], label=f"{outer_ch} (OUTER)", alpha=0.8)
            ax.plot(old_ts[old_mask] - t_start, old_data[old_mask, inner_idx], label=f"{inner_ch} (INNER)", alpha=0.8)
            ax.set_title(f"OLD Firmware - {side} AMB Channels")
            ax.set_ylabel("Value")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)

        if outer_ch in new_channels and inner_ch in new_channels:
            outer_idx = new_channels.index(outer_ch)
            inner_idx = new_channels.index(inner_ch)

            # New device
            ax = axes[ax_idx * 2 + 1]
            ax.plot(new_ts[new_mask] - t_start, new_data[new_mask, outer_idx], label=f"{outer_ch} (OUTER)", alpha=0.8)
            ax.plot(new_ts[new_mask] - t_start, new_data[new_mask, inner_idx], label=f"{inner_ch} (INNER)", alpha=0.8)
            ax.set_title(f"NEW Firmware - {side} AMB Channels")
            ax.set_ylabel("Value")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig("optics_amb_comparison.png", dpi=150)
    print("\n→ Saved: optics_amb_comparison.png")
    plt.show()

    return correlations


def plot_accgyro_timeseries(streams):
    """Plot ACCGYRO time series to visualize head movements."""
    print("\n--- Plotting ACCGYRO time series ---")

    old_stream = get_stream_by_name(streams, f"ACCGYRO ({OLD_FIRMWARE_MAC})")
    new_stream = get_stream_by_name(streams, f"ACCGYRO ({NEW_FIRMWARE_MAC})")

    if old_stream is None or new_stream is None:
        print("ERROR: Could not find both ACCGYRO streams!")
        return

    old_data = np.array(old_stream["time_series"])
    old_ts = old_stream["time_stamps"]
    new_data = np.array(new_stream["time_series"])
    new_ts = new_stream["time_stamps"]

    old_channels = get_channel_labels(old_stream)

    # Common time range
    t_start = max(old_ts.min(), new_ts.min())
    t_end = min(old_ts.max(), new_ts.max())

    fig, axes = plt.subplots(6, 1, figsize=(14, 16))
    fig.suptitle("ACCGYRO Channel Comparison: Old vs New Firmware", fontsize=14)

    for i, ch in enumerate(old_channels):
        ax = axes[i]

        # Old device (relative time)
        ax.plot(old_ts - t_start, old_data[:, i], label="Old firmware", alpha=0.7)
        ax.plot(new_ts - t_start, new_data[:, i], label="New firmware", alpha=0.7)

        ax.set_title(ch)
        ax.set_ylabel("Value")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, t_end - t_start])

    axes[-1].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig("accgyro_comparison.png", dpi=150)
    print("→ Saved: accgyro_comparison.png")
    plt.show()


def plot_eeg_timeseries(streams):
    """Plot EEG time series focusing on the blink period."""
    print("\n--- Plotting EEG time series ---")

    old_stream = get_stream_by_name(streams, f"EEG ({OLD_FIRMWARE_MAC})")
    new_stream = get_stream_by_name(streams, f"EEG ({NEW_FIRMWARE_MAC})")

    if old_stream is None or new_stream is None:
        print("ERROR: Could not find both EEG streams!")
        return

    old_data = np.array(old_stream["time_series"])
    old_ts = old_stream["time_stamps"]
    new_data = np.array(new_stream["time_series"])
    new_ts = new_stream["time_stamps"]

    old_channels = get_channel_labels(old_stream)

    # Common time range
    t_start = max(old_ts.min(), new_ts.min())
    t_end = min(old_ts.max(), new_ts.max())

    # Plot first 60s (rest) and second 60s (blinks)
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    fig.suptitle("EEG Channel Comparison: Old vs New Firmware", fontsize=14)

    time_segments = [
        (0, 30, "Rest (0-30s)"),
        (60, 90, "Blinks (60-90s)"),
    ]

    for col, (t0, t1, label) in enumerate(time_segments):
        for row, ch in enumerate(old_channels[:4]):  # First 4 are EEG channels
            ax = axes[row, col]

            old_mask = (old_ts - t_start >= t0) & (old_ts - t_start <= t1)
            new_mask = (new_ts - t_start >= t0) & (new_ts - t_start <= t1)

            ax.plot(old_ts[old_mask] - t_start, old_data[old_mask, row], label="Old", alpha=0.7, linewidth=0.5)
            ax.plot(new_ts[new_mask] - t_start, new_data[new_mask, row], label="New", alpha=0.7, linewidth=0.5)

            ax.set_title(f"{ch} - {label}")
            ax.set_ylabel("µV")
            if row == 0:
                ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig("eeg_timeseries_comparison.png", dpi=150)
    print("→ Saved: eeg_timeseries_comparison.png")
    plt.show()


def main():
    """Main analysis function."""
    print("=" * 80)
    print("DUAL DEVICE ANALYSIS: OLD vs NEW FIRMWARE")
    print("=" * 80)

    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Load data
    streams, header = load_xdf(XDF_FILE)

    # Summary
    summarize_streams(streams)

    # 1. ACCGYRO correlation
    accgyro_corr = analyze_accgyro_correlation(streams)

    # 2. EEG analysis
    eeg_corr = analyze_eeg_correlation(streams)

    # 3. OPTICS analysis
    optics_corr = analyze_optics_channels(streams)

    # Additional visualizations
    plot_accgyro_timeseries(streams)
    plot_eeg_timeseries(streams)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - eeg_power_spectrum_comparison.png")
    print("  - optics_amb_comparison.png")
    print("  - accgyro_comparison.png")
    print("  - eeg_timeseries_comparison.png")


if __name__ == "__main__":
    main()
