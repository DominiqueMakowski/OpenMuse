"""
PPG Pulsatile Pattern Analysis: Heart Rate Signal Validation
=============================================================

This script validates that the OPTICS channels show pulsatile blood flow
patterns suitable for heart rate extraction in both OLD and NEW firmware.

Expected patterns:
- RED, IR, NIR channels should show cardiac pulse waves (~60-100 bpm = 1-1.7 Hz)
- AMB (ambient) channels should NOT show strong pulsatile patterns
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pyxdf
from scipy import signal as scipy_signal

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


def bandpass_filter(signal, lowcut, highcut, fs, order=3):
    """Apply bandpass filter."""
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    # Clamp to valid range
    low = max(0.001, min(low, 0.99))
    high = max(low + 0.01, min(high, 0.99))
    b, a = scipy_signal.butter(order, [low, high], btype="band")
    return scipy_signal.filtfilt(b, a, signal)


def detect_peaks_in_ppg(signal, fs, min_hr=40, max_hr=180):
    """
    Detect cardiac peaks in PPG signal.

    Returns peak indices and estimated heart rate.
    """
    # Filter to cardiac frequency range
    lowcut = min_hr / 60  # Convert BPM to Hz
    highcut = max_hr / 60

    filtered = bandpass_filter(signal, lowcut, highcut, fs)

    # Find peaks with minimum distance based on max heart rate
    min_distance = int(fs * 60 / max_hr)  # Minimum samples between peaks

    peaks, properties = scipy_signal.find_peaks(filtered, distance=min_distance, prominence=np.std(filtered) * 0.3)

    if len(peaks) > 1:
        # Calculate heart rate from peak intervals
        intervals = np.diff(peaks) / fs  # in seconds
        hr = 60 / np.mean(intervals)  # in BPM
    else:
        hr = np.nan

    return peaks, hr, filtered


def compute_power_spectrum(signal, fs):
    """Compute power spectrum using Welch's method."""
    nperseg = min(512, len(signal) // 4)
    f, psd = scipy_signal.welch(signal, fs=fs, nperseg=nperseg)
    return f, psd


def analyze_ppg_channel(data, ts, ch_name, fs=64):
    """Analyze a single PPG channel for cardiac pulsatility."""

    # Use middle 60 seconds of recording (after settling)
    duration = ts[-1] - ts[0]
    mid_start = ts[0] + max(30, duration * 0.25)
    mid_end = mid_start + min(60, duration * 0.5)

    mask = (ts >= mid_start) & (ts <= mid_end)
    signal = data[mask]
    signal_ts = ts[mask]

    if len(signal) < fs * 10:  # Need at least 10 seconds
        return None

    # Detect peaks
    peaks, hr, filtered = detect_peaks_in_ppg(signal, fs)

    # Power spectrum
    f, psd = compute_power_spectrum(signal, fs)

    # Find peak frequency in cardiac range (0.5-3 Hz = 30-180 BPM)
    cardiac_mask = (f >= 0.5) & (f <= 3.0)
    if np.any(cardiac_mask):
        cardiac_f = f[cardiac_mask]
        cardiac_psd = psd[cardiac_mask]
        peak_freq = cardiac_f[np.argmax(cardiac_psd)]
        peak_power = np.max(cardiac_psd)

        # Signal-to-noise: cardiac power vs total power
        total_power = np.sum(psd)
        cardiac_power = np.sum(cardiac_psd)
        snr = cardiac_power / total_power if total_power > 0 else 0
    else:
        peak_freq = np.nan
        peak_power = np.nan
        snr = np.nan

    return {
        "channel": ch_name,
        "n_peaks": len(peaks),
        "estimated_hr": hr,
        "peak_freq_hz": peak_freq,
        "peak_freq_bpm": peak_freq * 60 if not np.isnan(peak_freq) else np.nan,
        "cardiac_snr": snr,
        "signal": signal,
        "filtered": filtered,
        "timestamps": signal_ts,
        "peaks": peaks,
        "psd_f": f,
        "psd": psd,
    }


def analyze_device_ppg(streams, device_mac, device_name):
    """Analyze all OPTICS channels for a device."""

    stream = get_stream_by_name(streams, f"OPTICS ({device_mac})")
    if stream is None:
        print(f"ERROR: Could not find OPTICS stream for {device_name}")
        return None

    data = np.array(stream["time_series"])
    ts = stream["time_stamps"]
    channels = get_channel_labels(stream)

    print(f"\n{'='*70}")
    print(f"{device_name} FIRMWARE ({device_mac})")
    print(f"{'='*70}")

    results = {}

    # Group channels by type
    channel_types = {
        "NIR": [ch for ch in channels if "NIR" in ch],
        "IR": [ch for ch in channels if "IR" in ch and "NIR" not in ch],
        "RED": [ch for ch in channels if "RED" in ch],
        "AMB": [ch for ch in channels if "AMB" in ch],
    }

    print(f"\n{'Channel':<20} {'Peaks':<8} {'HR (BPM)':<12} {'Peak Freq':<12} {'Cardiac SNR':<12}")
    print("-" * 70)

    for ch_type, ch_list in channel_types.items():
        for ch in ch_list:
            if ch in channels:
                ch_idx = channels.index(ch)
                result = analyze_ppg_channel(data[:, ch_idx], ts, ch)
                if result:
                    results[ch] = result
                    print(
                        f"{ch:<20} {result['n_peaks']:<8} {result['estimated_hr']:<12.1f} "
                        f"{result['peak_freq_bpm']:<12.1f} {result['cardiac_snr']:<12.4f}"
                    )

    # Summary by channel type
    print(f"\n--- Summary by Channel Type ---")
    for ch_type, ch_list in channel_types.items():
        hrs = [
            results[ch]["estimated_hr"] for ch in ch_list if ch in results and not np.isnan(results[ch]["estimated_hr"])
        ]
        snrs = [
            results[ch]["cardiac_snr"] for ch in ch_list if ch in results and not np.isnan(results[ch]["cardiac_snr"])
        ]
        if hrs:
            print(f"{ch_type:<6}: Mean HR = {np.mean(hrs):.1f} BPM, Mean SNR = {np.mean(snrs):.4f}")

    return results


def plot_ppg_comparison(old_results, new_results):
    """Create comparison plots for PPG analysis."""

    # Select representative channels
    channels_to_plot = ["OPTICS_LI_IR", "OPTICS_RI_IR", "OPTICS_LI_RED", "OPTICS_RI_RED"]

    fig, axes = plt.subplots(4, 4, figsize=(18, 14))
    fig.suptitle("PPG Pulsatile Pattern Comparison: OLD vs NEW Firmware", fontsize=14)

    for row, ch in enumerate(channels_to_plot):
        if ch not in old_results or ch not in new_results:
            continue

        old = old_results[ch]
        new = new_results[ch]

        # Plot 10 seconds of raw signal
        plot_samples = int(64 * 10)  # 10 seconds at 64 Hz

        # OLD - Raw signal with peaks
        ax = axes[row, 0]
        t_old = old["timestamps"][:plot_samples] - old["timestamps"][0]
        sig_old = old["signal"][:plot_samples]
        ax.plot(t_old, sig_old, "b-", linewidth=0.5, label="Raw")
        # Mark peaks within plot range
        peaks_in_range = old["peaks"][old["peaks"] < plot_samples]
        ax.plot(t_old[peaks_in_range], sig_old[peaks_in_range], "ro", markersize=4, label="Peaks")
        ax.set_title(f"OLD - {ch.replace('OPTICS_', '')}\nHR={old['estimated_hr']:.0f} BPM")
        ax.set_ylabel("Amplitude")
        if row == 0:
            ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # NEW - Raw signal with peaks
        ax = axes[row, 1]
        t_new = new["timestamps"][:plot_samples] - new["timestamps"][0]
        sig_new = new["signal"][:plot_samples]
        ax.plot(t_new, sig_new, "g-", linewidth=0.5, label="Raw")
        peaks_in_range = new["peaks"][new["peaks"] < plot_samples]
        ax.plot(t_new[peaks_in_range], sig_new[peaks_in_range], "ro", markersize=4, label="Peaks")
        ax.set_title(f"NEW - {ch.replace('OPTICS_', '')}\nHR={new['estimated_hr']:.0f} BPM")
        if row == 0:
            ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # OLD - Power spectrum
        ax = axes[row, 2]
        cardiac_mask = (old["psd_f"] >= 0.5) & (old["psd_f"] <= 3.0)
        ax.semilogy(old["psd_f"], old["psd"], "b-", alpha=0.7)
        ax.axvspan(0.5, 3.0, alpha=0.2, color="red", label="Cardiac band")
        ax.set_xlim([0, 5])
        ax.set_title(f"OLD PSD\nPeak: {old['peak_freq_bpm']:.0f} BPM")
        ax.set_ylabel("Power")
        ax.grid(True, alpha=0.3)

        # NEW - Power spectrum
        ax = axes[row, 3]
        ax.semilogy(new["psd_f"], new["psd"], "g-", alpha=0.7)
        ax.axvspan(0.5, 3.0, alpha=0.2, color="red", label="Cardiac band")
        ax.set_xlim([0, 5])
        ax.set_title(f"NEW PSD\nPeak: {new['peak_freq_bpm']:.0f} BPM")
        ax.grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)" if axes[-1, :].tolist().index(ax) < 2 else "Frequency (Hz)")

    plt.tight_layout()
    plt.savefig("ppg_pulsatile_comparison.png", dpi=150)
    print("\n→ Saved: ppg_pulsatile_comparison.png")

    # Plot filtered signals overlay
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8))
    fig2.suptitle("Filtered PPG Signals (0.5-3 Hz cardiac band)", fontsize=14)

    for idx, ch in enumerate(["OPTICS_LI_IR", "OPTICS_RI_IR"]):
        if ch not in old_results or ch not in new_results:
            continue

        old = old_results[ch]
        new = new_results[ch]

        plot_samples = int(64 * 10)

        ax = axes2[idx, 0]
        t_old = old["timestamps"][:plot_samples] - old["timestamps"][0]
        ax.plot(t_old, old["filtered"][:plot_samples], "b-", linewidth=1, label="OLD")
        ax.set_title(f"{ch.replace('OPTICS_', '')} - OLD Filtered")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)

        ax = axes2[idx, 1]
        t_new = new["timestamps"][:plot_samples] - new["timestamps"][0]
        ax.plot(t_new, new["filtered"][:plot_samples], "g-", linewidth=1, label="NEW")
        ax.set_title(f"{ch.replace('OPTICS_', '')} - NEW Filtered")
        ax.grid(True, alpha=0.3)

    axes2[-1, 0].set_xlabel("Time (s)")
    axes2[-1, 1].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig("ppg_filtered_comparison.png", dpi=150)
    print("→ Saved: ppg_filtered_comparison.png")

    plt.show()


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 70)
    print("PPG PULSATILE PATTERN ANALYSIS")
    print("=" * 70)
    print("Validating cardiac pulse detection in OPTICS channels")

    streams, _ = load_xdf(XDF_FILE)

    # Analyze both devices
    old_results = analyze_device_ppg(streams, OLD_FIRMWARE_MAC, "OLD")
    new_results = analyze_device_ppg(streams, NEW_FIRMWARE_MAC, "NEW")

    # Summary comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    if old_results and new_results:
        # Compare heart rates
        old_hrs = [r["estimated_hr"] for r in old_results.values() if not np.isnan(r["estimated_hr"])]
        new_hrs = [r["estimated_hr"] for r in new_results.values() if not np.isnan(r["estimated_hr"])]

        print(f"\nMean estimated HR:")
        print(f"  OLD firmware: {np.mean(old_hrs):.1f} ± {np.std(old_hrs):.1f} BPM")
        print(f"  NEW firmware: {np.mean(new_hrs):.1f} ± {np.std(new_hrs):.1f} BPM")

        hr_diff = abs(np.mean(old_hrs) - np.mean(new_hrs))
        if hr_diff < 5:
            print(f"\n✓ Heart rate estimates are CONSISTENT between devices (diff: {hr_diff:.1f} BPM)")
        else:
            print(f"\n⚠ Heart rate estimates DIFFER between devices (diff: {hr_diff:.1f} BPM)")

        # Check for pulsatility
        old_snrs = [
            r["cardiac_snr"] for ch, r in old_results.items() if "AMB" not in ch and not np.isnan(r["cardiac_snr"])
        ]
        new_snrs = [
            r["cardiac_snr"] for ch, r in new_results.items() if "AMB" not in ch and not np.isnan(r["cardiac_snr"])
        ]

        print(f"\nMean cardiac SNR (excluding AMB):")
        print(f"  OLD firmware: {np.mean(old_snrs):.4f}")
        print(f"  NEW firmware: {np.mean(new_snrs):.4f}")

        if np.mean(old_snrs) > 0.01 and np.mean(new_snrs) > 0.01:
            print("\n✓ BOTH devices show cardiac pulsatility in PPG channels")
            print("✓ OPTICS decoding is CORRECT for heart rate extraction")

        # Generate comparison plots
        plot_ppg_comparison(old_results, new_results)

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("If both devices show similar heart rates and clear pulsatile patterns,")
    print("the OPTICS decoding is correct and suitable for heart rate extraction.")


if __name__ == "__main__":
    main()
