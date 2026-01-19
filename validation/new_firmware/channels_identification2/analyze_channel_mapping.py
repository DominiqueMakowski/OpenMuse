"""
Cross-Correlation Matrix Analysis: Channel Order Verification
=============================================================

This script computes the full cross-correlation matrix between OLD and NEW
device channels to check if any channel mapping might be swapped.

If channel order is correct:
- OLD_ACC_X should correlate most strongly with NEW_ACC_X
- Diagonal of correlation matrix should have highest values

If channel order is wrong:
- OLD_ACC_X might correlate better with NEW_ACC_Y (off-diagonal peak)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pyxdf
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


def resample_to_common(data1, ts1, data2, ts2, target_fs):
    """Resample both datasets to common timestamps."""
    t_start = max(ts1.min(), ts2.min())
    t_end = min(ts1.max(), ts2.max())
    n_samples = int((t_end - t_start) * target_fs)
    common_ts = np.linspace(t_start, t_end, n_samples)

    # Resample data1
    data1_resampled = np.zeros((n_samples, data1.shape[1]))
    for i in range(data1.shape[1]):
        interp = interp1d(ts1, data1[:, i], kind="linear", bounds_error=False, fill_value=np.nan)
        data1_resampled[:, i] = interp(common_ts)

    # Resample data2
    data2_resampled = np.zeros((n_samples, data2.shape[1]))
    for i in range(data2.shape[1]):
        interp = interp1d(ts2, data2[:, i], kind="linear", bounds_error=False, fill_value=np.nan)
        data2_resampled[:, i] = interp(common_ts)

    # Remove rows with NaN
    valid = ~np.any(np.isnan(data1_resampled), axis=1) & ~np.any(np.isnan(data2_resampled), axis=1)

    return data1_resampled[valid], data2_resampled[valid], common_ts[valid]


def compute_cross_correlation_matrix(data_old, data_new, old_channels, new_channels):
    """Compute correlation between all pairs of channels."""
    n_old = len(old_channels)
    n_new = len(new_channels)

    corr_matrix = np.zeros((n_old, n_new))

    for i in range(n_old):
        for j in range(n_new):
            corr_matrix[i, j] = np.corrcoef(data_old[:, i], data_new[:, j])[0, 1]

    return corr_matrix


def analyze_channel_mapping(streams, sensor_type, target_fs):
    """Analyze channel mapping for a specific sensor type."""

    old_stream = get_stream_by_name(streams, f"{sensor_type} ({OLD_FIRMWARE_MAC})")
    new_stream = get_stream_by_name(streams, f"{sensor_type} ({NEW_FIRMWARE_MAC})")

    if old_stream is None or new_stream is None:
        print(f"ERROR: Could not find both {sensor_type} streams!")
        return None, None, None

    old_data = np.array(old_stream["time_series"])
    old_ts = old_stream["time_stamps"]
    new_data = np.array(new_stream["time_series"])
    new_ts = new_stream["time_stamps"]

    old_channels = get_channel_labels(old_stream)
    new_channels = get_channel_labels(new_stream)

    # Resample to common timestamps
    old_resampled, new_resampled, _ = resample_to_common(old_data, old_ts, new_data, new_ts, target_fs)

    # Compute cross-correlation matrix
    corr_matrix = compute_cross_correlation_matrix(old_resampled, new_resampled, old_channels, new_channels)

    return corr_matrix, old_channels, new_channels


def print_correlation_analysis(corr_matrix, old_channels, new_channels, sensor_name):
    """Print detailed correlation analysis."""

    print(f"\n{'='*70}")
    print(f"{sensor_name} CROSS-CORRELATION MATRIX")
    print(f"{'='*70}")

    # Print matrix
    print(f"\n{'OLD \\ NEW':<15}", end="")
    for ch in new_channels:
        short = ch.split("_")[-1] if "_" in ch else ch[-6:]
        print(f"{short:>10}", end="")
    print()
    print("-" * (15 + 10 * len(new_channels)))

    for i, old_ch in enumerate(old_channels):
        short_old = old_ch.split("_")[-1] if "_" in old_ch else old_ch[-6:]
        print(f"{short_old:<15}", end="")
        for j in range(len(new_channels)):
            corr = corr_matrix[i, j]
            # Highlight the highest correlation in each row
            if j == np.argmax(np.abs(corr_matrix[i, :])):
                print(f"{corr:>10.4f}*", end="")
            else:
                print(f"{corr:>10.4f}", end="")
        print()

    print("\n* = highest |correlation| in row")

    # Check if diagonal is maximum
    print(f"\n--- Channel Mapping Verification ---")
    mapping_correct = True
    for i, old_ch in enumerate(old_channels):
        if i < len(new_channels):
            max_j = np.argmax(np.abs(corr_matrix[i, :]))
            diag_corr = corr_matrix[i, i] if i < corr_matrix.shape[1] else None
            max_corr = corr_matrix[i, max_j]

            if max_j != i:
                mapping_correct = False
                print(
                    f"  ⚠ {old_ch}: Best match is {new_channels[max_j]} (r={max_corr:.4f}), not {new_channels[i]} (r={diag_corr:.4f})"
                )
            else:
                print(f"  ✓ {old_ch}: Best match is diagonal ({new_channels[i]}, r={max_corr:.4f})")

    if mapping_correct:
        print(f"\n✓ All {sensor_name} channels correctly mapped!")
    else:
        print(f"\n⚠ Potential channel mapping issues detected!")

    return mapping_correct


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 70)
    print("CROSS-CORRELATION CHANNEL MAPPING VERIFICATION")
    print("=" * 70)

    streams, _ = load_xdf(XDF_FILE)

    results = {}

    # 1. ACCGYRO Analysis
    corr_matrix, old_ch, new_ch = analyze_channel_mapping(streams, "ACCGYRO", 52)
    if corr_matrix is not None:
        results["ACCGYRO"] = print_correlation_analysis(corr_matrix, old_ch, new_ch, "ACCGYRO")

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(new_ch)))
        ax.set_yticks(range(len(old_ch)))
        ax.set_xticklabels([c.split("_")[-1] for c in new_ch], rotation=45)
        ax.set_yticklabels([c.split("_")[-1] for c in old_ch])
        ax.set_xlabel("NEW Firmware Channels")
        ax.set_ylabel("OLD Firmware Channels")
        ax.set_title("ACCGYRO Cross-Correlation Matrix")
        plt.colorbar(im, label="Correlation")

        # Add correlation values as text
        for i in range(len(old_ch)):
            for j in range(len(new_ch)):
                text = ax.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center", color="black", fontsize=9)

        plt.tight_layout()
        plt.savefig("accgyro_crosscorr_matrix.png", dpi=150)
        print("\n→ Saved: accgyro_crosscorr_matrix.png")

    # 2. EEG Analysis (first 4 channels)
    corr_matrix, old_ch, new_ch = analyze_channel_mapping(streams, "EEG", 256)
    if corr_matrix is not None:
        # Only analyze first 4 EEG channels
        corr_matrix = corr_matrix[:4, :4]
        old_ch = old_ch[:4]
        new_ch = new_ch[:4]
        results["EEG"] = print_correlation_analysis(corr_matrix, old_ch, new_ch, "EEG")

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(new_ch)))
        ax.set_yticks(range(len(old_ch)))
        ax.set_xticklabels([c.replace("EEG_", "") for c in new_ch], rotation=45)
        ax.set_yticklabels([c.replace("EEG_", "") for c in old_ch])
        ax.set_xlabel("NEW Firmware Channels")
        ax.set_ylabel("OLD Firmware Channels")
        ax.set_title("EEG Cross-Correlation Matrix")
        plt.colorbar(im, label="Correlation")

        for i in range(len(old_ch)):
            for j in range(len(new_ch)):
                text = ax.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center", color="black", fontsize=10)

        plt.tight_layout()
        plt.savefig("eeg_crosscorr_matrix.png", dpi=150)
        print("\n→ Saved: eeg_crosscorr_matrix.png")

    # 3. OPTICS Analysis
    corr_matrix, old_ch, new_ch = analyze_channel_mapping(streams, "OPTICS", 64)
    if corr_matrix is not None:
        results["OPTICS"] = print_correlation_analysis(corr_matrix, old_ch, new_ch, "OPTICS")

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(new_ch)))
        ax.set_yticks(range(len(old_ch)))
        ax.set_xticklabels([c.replace("OPTICS_", "") for c in new_ch], rotation=90, fontsize=8)
        ax.set_yticklabels([c.replace("OPTICS_", "") for c in old_ch], fontsize=8)
        ax.set_xlabel("NEW Firmware Channels")
        ax.set_ylabel("OLD Firmware Channels")
        ax.set_title("OPTICS Cross-Correlation Matrix")
        plt.colorbar(im, label="Correlation")

        for i in range(len(old_ch)):
            for j in range(len(new_ch)):
                text = ax.text(j, i, f"{corr_matrix[i, j]:.1f}", ha="center", va="center", color="black", fontsize=6)

        plt.tight_layout()
        plt.savefig("optics_crosscorr_matrix.png", dpi=150)
        print("\n→ Saved: optics_crosscorr_matrix.png")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for sensor, correct in results.items():
        status = "✓ Correct" if correct else "⚠ Issues found"
        print(f"  {sensor}: {status}")

    plt.show()


if __name__ == "__main__":
    main()
