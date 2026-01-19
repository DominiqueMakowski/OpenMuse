"""
Deep dive into NEW device OPTICS signals to understand why they're noisy.
"""

import matplotlib.pyplot as plt
import numpy as np
import pyxdf
from scipy import ndimage

NEW_FIRMWARE_MAC = "00:55:DA:BB:CD:CD"
OLD_FIRMWARE_MAC = "00:55:DA:B9:FA:20"


def find_channel_raw(streams, stream_name, channel_name):
    """Find channel and return RAW signal."""
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


def rolling_normalize(signal_data, window_sec=5.0, fs=64.0):
    window_samples = max(int(window_sec * fs), 1)
    rolling_min = ndimage.minimum_filter1d(signal_data, size=window_samples, mode="nearest")
    rolling_max = ndimage.maximum_filter1d(signal_data, size=window_samples, mode="nearest")
    range_vals = rolling_max - rolling_min
    range_vals[range_vals < 1e-10] = 1e-10
    return (signal_data - rolling_min) / range_vals


filename = "test1_robust.xdf"
print(f"Loading {filename}...")
streams, _ = pyxdf.load_xdf(filename, synchronize_clocks=True, handle_clock_resets=True, dejitter_timestamps=False)

# Get LUX
lux_raw, lux_ts = find_channel_raw(streams, "OpenSignals", "LUX")
fs_lux = len(lux_ts) / (lux_ts[-1] - lux_ts[0])
lux_norm = rolling_normalize(lux_raw, window_sec=5.0, fs=fs_lux)

# Get OPTICS from both devices
old_ir, old_ts = find_channel_raw(streams, f"Muse-OPTICS ({OLD_FIRMWARE_MAC})", "OPTICS_RI_IR")
new_ir, new_ts = find_channel_raw(streams, f"Muse-OPTICS ({NEW_FIRMWARE_MAC})", "OPTICS_RI_IR")

fs_optics = len(old_ts) / (old_ts[-1] - old_ts[0]) if old_ts is not None else 64

if old_ir is not None:
    old_norm = rolling_normalize(old_ir, window_sec=5.0, fs=fs_optics)
if new_ir is not None:
    new_norm = rolling_normalize(new_ir, window_sec=5.0, fs=fs_optics)

# Create multi-panel figure showing different time windows
fig, axes = plt.subplots(4, 3, figsize=(18, 14))
fig.suptitle(f"Signal Analysis: {filename}", fontsize=14)

# Different time windows to check
t_start = lux_ts[0]
windows = [
    (t_start + 10, "Start of recording (+10s)"),
    (t_start + 300, "5 minutes in"),
    (t_start + 900, "15 minutes in"),
    (t_start + 1500, "25 minutes in"),
]

for row, (t0, title) in enumerate(windows):
    t1 = t0 + 10  # 10 second window

    # Raw signals
    ax = axes[row, 0]

    # LUX
    mask = (lux_ts >= t0) & (lux_ts <= t1)
    if np.any(mask):
        ax.plot(lux_ts[mask] - t0, lux_norm[mask], "b-", lw=1, label="LUX", alpha=0.8)

    # OLD
    if old_ts is not None:
        mask = (old_ts >= t0) & (old_ts <= t1)
        if np.any(mask):
            ax.plot(old_ts[mask] - t0, old_norm[mask], "r-", lw=1, label="OLD IR", alpha=0.7)

    ax.set_title(title)
    ax.set_ylabel("Normalized")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    # NEW signal alone
    ax2 = axes[row, 1]
    if new_ts is not None:
        mask = (new_ts >= t0) & (new_ts <= t1)
        if np.any(mask):
            ax2.plot(new_ts[mask] - t0, new_norm[mask], "g-", lw=1, label="NEW IR", alpha=0.8)
            # Also show raw values
            ax2_twin = ax2.twinx()
            mask_raw = (new_ts >= t0) & (new_ts <= t1)
            ax2_twin.plot(new_ts[mask_raw] - t0, new_ir[mask_raw], "g:", lw=0.5, alpha=0.3)
            ax2_twin.set_ylabel("Raw", color="gray", fontsize=8)

    ax2.set_title(f"NEW IR alone")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.1)

    # Histogram of signal values
    ax3 = axes[row, 2]
    if new_ts is not None:
        mask = (new_ts >= t0) & (new_ts <= t1)
        if np.any(mask):
            ax3.hist(new_norm[mask], bins=50, alpha=0.7, color="green", label="NEW IR")
    if old_ts is not None:
        mask = (old_ts >= t0) & (old_ts <= t1)
        if np.any(mask):
            ax3.hist(old_norm[mask], bins=50, alpha=0.5, color="red", label="OLD IR")

    ax3.set_title("Distribution")
    ax3.legend(fontsize=8)
    ax3.set_xlabel("Normalized value")

plt.tight_layout()
plt.savefig("new_device_analysis.png", dpi=150, bbox_inches="tight")
print("Saved: new_device_analysis.png")
plt.show()

# Print raw value statistics
print("\n" + "=" * 50)
print("RAW SIGNAL STATISTICS")
print("=" * 50)
if old_ir is not None:
    print(f"\nOLD IR: min={np.min(old_ir):.0f}, max={np.max(old_ir):.0f}, range={np.max(old_ir)-np.min(old_ir):.0f}")
    print(f"  Typical values: {np.percentile(old_ir, [5, 50, 95])}")

if new_ir is not None:
    print(f"\nNEW IR: min={np.min(new_ir):.0f}, max={np.max(new_ir):.0f}, range={np.max(new_ir)-np.min(new_ir):.0f}")
    print(f"  Typical values: {np.percentile(new_ir, [5, 50, 95])}")
