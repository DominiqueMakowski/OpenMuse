"""
Check if timestamp jitter differs between clocks.
"""

import numpy as np
import pyxdf

NEW_FIRMWARE_MAC = "00:55:DA:BB:CD:CD"
OLD_FIRMWARE_MAC = "00:55:DA:B9:FA:20"


def find_stream(streams, stream_name):
    for s in streams:
        name = s["info"].get("name", ["Unnamed"])[0]
        if name == stream_name:
            return s["time_stamps"]
    return None


for filename in ["test1_robust.xdf", "test1_windowed.xdf"]:
    print(f"\n{'='*70}")
    print(f"FILE: {filename}")
    print("=" * 70)

    streams, _ = pyxdf.load_xdf(filename, synchronize_clocks=True, handle_clock_resets=True, dejitter_timestamps=False)

    # Get timestamps for each OPTICS stream
    new_ts = find_stream(streams, f"Muse-OPTICS ({NEW_FIRMWARE_MAC})")
    old_ts = find_stream(streams, f"Muse-OPTICS ({OLD_FIRMWARE_MAC})")
    lux_ts = find_stream(streams, "OpenSignals")

    # Calculate inter-sample intervals
    if new_ts is not None:
        new_isi = np.diff(new_ts) * 1000  # ms
        print(f"\nNEW OPTICS inter-sample interval:")
        print(f"  Mean: {np.mean(new_isi):.3f} ms (expected ~15.625 ms for 64 Hz)")
        print(f"  Std:  {np.std(new_isi):.3f} ms")
        print(f"  Min:  {np.min(new_isi):.3f} ms")
        print(f"  Max:  {np.max(new_isi):.3f} ms")
        # Count large jumps
        large_jumps = np.sum(new_isi > 30)
        print(f"  Large jumps (>30ms): {large_jumps}")

    if old_ts is not None:
        old_isi = np.diff(old_ts) * 1000
        print(f"\nOLD OPTICS inter-sample interval:")
        print(f"  Mean: {np.mean(old_isi):.3f} ms")
        print(f"  Std:  {np.std(old_isi):.3f} ms")
        print(f"  Min:  {np.min(old_isi):.3f} ms")
        print(f"  Max:  {np.max(old_isi):.3f} ms")
        large_jumps = np.sum(old_isi > 30)
        print(f"  Large jumps (>30ms): {large_jumps}")

    if lux_ts is not None:
        lux_isi = np.diff(lux_ts) * 1000
        print(f"\nLUX (OpenSignals) inter-sample interval:")
        print(f"  Mean: {np.mean(lux_isi):.3f} ms")
        print(f"  Std:  {np.std(lux_isi):.3f} ms")
