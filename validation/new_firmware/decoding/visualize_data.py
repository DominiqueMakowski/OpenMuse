"""
Visualization script for comparing decoded Muse data across firmware versions.

This script creates plots to help verify that decoded data looks correct:
- EEG signals should show physiological patterns
- OPTICS (PPG) should show pulsatile waveforms when worn on forehead
- ACCGYRO should show physical motion patterns
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from OpenMuse.decode import decode_rawdata


def load_messages(filepath: str, max_messages: int = None) -> list:
    """Load messages from a raw data file."""
    messages = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                messages.append(line)
                if max_messages and len(messages) >= max_messages:
                    break
    return messages


def plot_eeg_comparison(results: dict, device_names: list, output_path: str = None):
    """Plot EEG data from multiple devices for comparison."""
    fig, axes = plt.subplots(
        len(device_names), 1, figsize=(14, 4 * len(device_names)), sharex=False
    )
    if len(device_names) == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, 8))

    for ax, name in zip(axes, device_names):
        df = results.get(name, {}).get("EEG")
        if df is None or df.empty:
            ax.text(
                0.5,
                0.5,
                f"{name}: No EEG data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(name)
            continue

        # Plot first 10 seconds of EEG
        time_mask = df["time"] <= 10.0
        df_subset = df[time_mask]

        time = df_subset["time"].values
        channels = ["EEG_TP9", "EEG_AF7", "EEG_AF8", "EEG_TP10"]

        for i, ch in enumerate(channels):
            if ch in df_subset.columns:
                ax.plot(
                    time,
                    df_subset[ch].values,
                    label=ch,
                    color=colors[i],
                    alpha=0.8,
                    linewidth=0.5,
                )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("EEG (µV)")
        ax.set_title(f"{name} - EEG Channels (first 10s)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved EEG plot to {output_path}")
    plt.show()


def plot_optics_comparison(results: dict, device_names: list, output_path: str = None):
    """Plot OPTICS (PPG) data from multiple devices for comparison."""
    fig, axes = plt.subplots(
        len(device_names), 1, figsize=(14, 4 * len(device_names)), sharex=False
    )
    if len(device_names) == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, 8))

    for ax, name in zip(axes, device_names):
        df = results.get(name, {}).get("OPTICS")
        if df is None or df.empty:
            ax.text(
                0.5,
                0.5,
                f"{name}: No OPTICS data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(name)
            continue

        # Plot first 15 seconds of PPG (enough to see several heartbeats)
        time_mask = df["time"] <= 15.0
        df_subset = df[time_mask]

        time = df_subset["time"].values

        # Plot NIR channels (typically best for PPG)
        nir_channels = [
            "OPTICS_LI_NIR",
            "OPTICS_RI_NIR",
            "OPTICS_LO_NIR",
            "OPTICS_RO_NIR",
        ]
        for i, ch in enumerate(nir_channels):
            if ch in df_subset.columns:
                ax.plot(
                    time,
                    df_subset[ch].values,
                    label=ch,
                    color=colors[i],
                    alpha=0.8,
                    linewidth=1,
                )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("PPG (a.u.)")
        ax.set_title(f"{name} - NIR PPG Channels (first 15s)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved OPTICS plot to {output_path}")
    plt.show()


def plot_accgyro_comparison(results: dict, device_names: list, output_path: str = None):
    """Plot ACCGYRO data from multiple devices for comparison."""
    fig, axes = plt.subplots(
        len(device_names), 2, figsize=(14, 3 * len(device_names)), sharex=False
    )
    if len(device_names) == 1:
        axes = axes.reshape(1, -1)

    acc_colors = ["r", "g", "b"]
    gyro_colors = ["orange", "purple", "cyan"]

    for row, name in enumerate(device_names):
        df = results.get(name, {}).get("ACCGYRO")
        if df is None or df.empty:
            axes[row, 0].text(
                0.5,
                0.5,
                f"{name}: No ACCGYRO data",
                ha="center",
                va="center",
                transform=axes[row, 0].transAxes,
            )
            axes[row, 1].text(
                0.5,
                0.5,
                f"{name}: No ACCGYRO data",
                ha="center",
                va="center",
                transform=axes[row, 1].transAxes,
            )
            continue

        # Plot first 20 seconds
        time_mask = df["time"] <= 20.0
        df_subset = df[time_mask]
        time = df_subset["time"].values

        # Accelerometer
        for i, ch in enumerate(["ACC_X", "ACC_Y", "ACC_Z"]):
            if ch in df_subset.columns:
                axes[row, 0].plot(
                    time, df_subset[ch].values, label=ch, color=acc_colors[i], alpha=0.8
                )
        axes[row, 0].set_xlabel("Time (s)")
        axes[row, 0].set_ylabel("Acceleration (g)")
        axes[row, 0].set_title(f"{name} - Accelerometer")
        axes[row, 0].legend(loc="upper right", fontsize=8)
        axes[row, 0].grid(True, alpha=0.3)

        # Gyroscope
        for i, ch in enumerate(["GYRO_X", "GYRO_Y", "GYRO_Z"]):
            if ch in df_subset.columns:
                axes[row, 1].plot(
                    time,
                    df_subset[ch].values,
                    label=ch,
                    color=gyro_colors[i],
                    alpha=0.8,
                )
        axes[row, 1].set_xlabel("Time (s)")
        axes[row, 1].set_ylabel("Angular velocity (°/s)")
        axes[row, 1].set_title(f"{name} - Gyroscope")
        axes[row, 1].legend(loc="upper right", fontsize=8)
        axes[row, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved ACCGYRO plot to {output_path}")
    plt.show()


def plot_battery(results: dict, device_names: list, output_path: str = None):
    """Plot battery data from multiple devices."""
    fig, ax = plt.subplots(figsize=(10, 4))

    for name in device_names:
        df = results.get(name, {}).get("BATTERY")
        if df is not None and not df.empty:
            ax.plot(
                df["time"].values,
                df["battery_percent"].values,
                label=name,
                marker="o",
                markersize=2,
            )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Battery (%)")
    ax.set_title("Battery Level Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved Battery plot to {output_path}")
    plt.show()


def main():
    base_path = Path(__file__).parent

    files = {
        "device1 (Old FW)": base_path / "device1.txt",
        "device2 (New FW)": base_path / "device2.txt",
        "device3 (Unknown)": base_path / "device3.txt",
    }

    # Load and decode data
    print("Loading and decoding data...")
    results = {}

    for name, fpath in files.items():
        if fpath.exists():
            print(f"  Processing {name}...")
            messages = load_messages(str(fpath), max_messages=5000)
            results[name] = decode_rawdata(messages)

    device_names = list(results.keys())

    # Create visualizations
    print("\nGenerating plots...")

    # EEG comparison
    plot_eeg_comparison(results, device_names, str(base_path / "eeg_comparison.png"))

    # OPTICS comparison
    plot_optics_comparison(
        results, device_names, str(base_path / "optics_comparison.png")
    )

    # ACCGYRO comparison
    plot_accgyro_comparison(
        results, device_names, str(base_path / "accgyro_comparison.png")
    )

    # Battery (only old FW has it)
    plot_battery(results, device_names, str(base_path / "battery_comparison.png"))

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
