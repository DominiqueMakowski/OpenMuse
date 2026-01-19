"""
Verification Script for Decoding Correctness
============================================

This script validates:
1. AMB channel correlation patterns (INNER vs OUTER) between devices
2. Sampling rate verification for all channels to confirm no data loss
3. Whether inverting old device AMB channels matches new device patterns

Run from: validation/new_firmware/
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent to path for OpenMuse imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from OpenMuse.decode import decode_rawdata, SENSORS


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


def analyze_sampling_rates(filepath: str, name: str = "Device"):
    """
    Analyze effective sampling rates for all sensor types.

    This verifies that decoding is working correctly by checking:
    1. Sample counts match expected rates
    2. Timestamps are monotonic
    3. No gaps or missing data
    """
    print(f"\n{'='*80}")
    print(f"SAMPLING RATE ANALYSIS: {name}")
    print(f"File: {filepath}")
    print(f"{'='*80}")

    messages = load_messages(filepath)
    print(f"Loaded {len(messages)} messages")

    # Decode all data
    data = decode_rawdata(messages)

    results = {}

    for sensor_type in ["EEG", "ACCGYRO", "OPTICS", "BATTERY"]:
        df = data.get(sensor_type)
        if df is None or df.empty:
            print(f"\n{sensor_type}: No data")
            continue

        n_samples = len(df)
        if n_samples < 2:
            print(f"\n{sensor_type}: Only {n_samples} sample(s)")
            continue

        times = df["time"].values
        duration = times[-1] - times[0]

        if duration <= 0:
            print(f"\n{sensor_type}: Invalid duration ({duration:.3f}s)")
            continue

        effective_rate = (n_samples - 1) / duration

        # Get expected rate from SENSORS config
        expected_rates = {
            "EEG": 256.0,
            "ACCGYRO": 52.0,
            "OPTICS": 64.0,
            "BATTERY": 1.0,  # Old firmware; new firmware ~0.2 Hz
        }
        expected = expected_rates.get(sensor_type, 0)

        # Check monotonicity
        diffs = np.diff(times)
        n_non_mono = np.sum(diffs < 0)
        pct_non_mono = 100.0 * n_non_mono / len(diffs)

        # Check for gaps (> 2x expected interval)
        expected_interval = 1.0 / expected if expected > 0 else 1.0
        n_gaps = np.sum(diffs > 2 * expected_interval)

        # Statistics
        mean_interval = np.mean(diffs)
        std_interval = np.std(diffs)

        results[sensor_type] = {
            "samples": n_samples,
            "duration": duration,
            "effective_rate": effective_rate,
            "expected_rate": expected,
            "rate_error_pct": (
                100 * (effective_rate - expected) / expected if expected > 0 else 0
            ),
            "non_monotonic": n_non_mono,
            "gaps": n_gaps,
            "mean_interval_ms": mean_interval * 1000,
            "std_interval_ms": std_interval * 1000,
        }

        print(f"\n{sensor_type}:")
        print(f"  Samples: {n_samples:,}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Effective Rate: {effective_rate:.2f} Hz (expected {expected:.1f} Hz)")
        print(f"  Rate Error: {results[sensor_type]['rate_error_pct']:+.2f}%")
        print(f"  Non-monotonic: {n_non_mono} ({pct_non_mono:.2f}%)")
        print(f"  Gaps (>2x interval): {n_gaps}")
        print(
            f"  Mean interval: {mean_interval*1000:.3f}ms ± {std_interval*1000:.3f}ms"
        )

    return data, results


def analyze_amb_correlations(data: dict, device_name: str = "Device"):
    """
    Analyze correlations between INNER and OUTER ambient channels.

    If the sensors are just placed at different locations measuring the same thing,
    they should be highly correlated (with the same sign).
    """
    print(f"\n{'='*80}")
    print(f"AMB CHANNEL CORRELATION ANALYSIS: {device_name}")
    print(f"{'='*80}")

    df = data.get("OPTICS")
    if df is None or df.empty:
        print("No OPTICS data!")
        return None

    # Check which columns we have
    amb_channels = [c for c in df.columns if "AMB" in c]
    print(f"AMB channels found: {amb_channels}")

    if len(amb_channels) < 4:
        print(f"Not enough AMB channels (need 4, found {len(amb_channels)})")
        return None

    # Compute correlation matrix for AMB channels
    amb_data = df[amb_channels]
    corr = amb_data.corr()

    print("\nAMB Channel Correlation Matrix:")
    print(corr.to_string())

    # Check INNER vs OUTER correlation
    inner_amb = ["OPTICS_LI_AMB", "OPTICS_RI_AMB"]
    outer_amb = ["OPTICS_LO_AMB", "OPTICS_RO_AMB"]

    # Inner-Inner correlation
    if all(c in df.columns for c in inner_amb):
        inner_inner = df["OPTICS_LI_AMB"].corr(df["OPTICS_RI_AMB"])
        print(f"\nINNER-INNER (LI_AMB vs RI_AMB): {inner_inner:.3f}")

    # Outer-Outer correlation
    if all(c in df.columns for c in outer_amb):
        outer_outer = df["OPTICS_LO_AMB"].corr(df["OPTICS_RO_AMB"])
        print(f"OUTER-OUTER (LO_AMB vs RO_AMB): {outer_outer:.3f}")

    # Inner-Outer cross-correlations
    print("\nINNER-OUTER Cross-correlations:")
    for inner in inner_amb:
        for outer in outer_amb:
            if inner in df.columns and outer in df.columns:
                corr_val = df[inner].corr(df[outer])
                print(f"  {inner} vs {outer}: {corr_val:.3f}")

    return corr


def test_inversion_hypothesis(old_data: dict, new_data: dict):
    """
    Test if inverting old device's INNER AMB channels makes them match the new device.

    Hypothesis: Old firmware had inverted INNER AMB channels, new firmware fixed this.
    """
    print(f"\n{'='*80}")
    print("INVERSION HYPOTHESIS TEST")
    print("Testing: Do inverted old INNER AMB channels match new device patterns?")
    print(f"{'='*80}")

    old_optics = old_data.get("OPTICS")
    new_optics = new_data.get("OPTICS")

    if old_optics is None or old_optics.empty:
        print("No old device OPTICS data!")
        return
    if new_optics is None or new_optics.empty:
        print("No new device OPTICS data!")
        return

    inner_amb = ["OPTICS_LI_AMB", "OPTICS_RI_AMB"]
    outer_amb = ["OPTICS_LO_AMB", "OPTICS_RO_AMB"]

    # For old device: compute INNER-OUTER correlation with and without inversion
    print("\n--- OLD DEVICE ---")
    print("Original INNER-OUTER correlations:")

    old_inner_outer_orig = {}
    for inner in inner_amb:
        for outer in outer_amb:
            if inner in old_optics.columns and outer in old_optics.columns:
                corr = old_optics[inner].corr(old_optics[outer])
                old_inner_outer_orig[f"{inner}_vs_{outer}"] = corr
                print(f"  {inner} vs {outer}: {corr:.3f}")

    print("\nInverted INNER channels INNER-OUTER correlations:")
    old_inner_outer_inv = {}
    for inner in inner_amb:
        for outer in outer_amb:
            if inner in old_optics.columns and outer in old_optics.columns:
                # Invert the inner channel (multiply by -1 after centering)
                inner_vals = old_optics[inner].values
                inner_centered = inner_vals - np.mean(inner_vals)
                inner_inverted = -inner_centered + np.mean(inner_vals)

                outer_vals = old_optics[outer].values
                corr = np.corrcoef(inner_inverted, outer_vals)[0, 1]
                old_inner_outer_inv[f"{inner}_vs_{outer}"] = corr
                print(f"  {inner}(inverted) vs {outer}: {corr:.3f}")

    # For new device: compute INNER-OUTER correlation
    print("\n--- NEW DEVICE ---")
    print("INNER-OUTER correlations:")
    new_inner_outer = {}
    for inner in inner_amb:
        for outer in outer_amb:
            if inner in new_optics.columns and outer in new_optics.columns:
                corr = new_optics[inner].corr(new_optics[outer])
                new_inner_outer[f"{inner}_vs_{outer}"] = corr
                print(f"  {inner} vs {outer}: {corr:.3f}")

    # Summary
    print("\n--- SUMMARY ---")
    print("If inversion hypothesis is correct:")
    print("  - Old inverted INNER-OUTER should have SAME SIGN as New INNER-OUTER")
    print("  - Old original INNER-OUTER should have OPPOSITE SIGN to New INNER-OUTER")

    matches_orig = 0
    matches_inv = 0
    total = 0

    for key in old_inner_outer_orig:
        if key in new_inner_outer:
            total += 1
            old_orig = old_inner_outer_orig[key]
            old_inv = old_inner_outer_inv.get(key, 0)
            new_val = new_inner_outer[key]

            if old_orig * new_val > 0:
                matches_orig += 1
            if old_inv * new_val > 0:
                matches_inv += 1

    if total > 0:
        print(f"\nSign agreement with NEW device:")
        print(
            f"  Original old INNER: {matches_orig}/{total} ({100*matches_orig/total:.0f}%)"
        )
        print(
            f"  Inverted old INNER: {matches_inv}/{total} ({100*matches_inv/total:.0f}%)"
        )

        if matches_inv > matches_orig:
            print("\n✓ INVERSION HYPOTHESIS SUPPORTED")
            print(
                "  Inverting old device's INNER AMB channels improves agreement with new device"
            )
        else:
            print("\n✗ INVERSION HYPOTHESIS NOT SUPPORTED")
            print("  Inverting doesn't improve agreement - issue may be more complex")


def analyze_packet_structure(filepath: str, name: str = "Device"):
    """
    Analyze raw packet structure to check for potential data loss.

    Specifically check:
    1. Number of each packet type (0x11, 0x12, 0x36, 0x47, 0x88, 0x98)
    2. Whether 0x88 packets could be "eating" other packet types
    """
    import struct

    print(f"\n{'='*80}")
    print(f"PACKET STRUCTURE ANALYSIS: {name}")
    print(f"{'='*80}")

    messages = load_messages(filepath)

    packet_counts = {}
    packet_data_lens = {}
    total_bytes = 0

    for msg in messages:
        parts = msg.strip().split("\t", 2)
        if len(parts) < 3:
            continue
        payload = bytes.fromhex(parts[2].strip())
        total_bytes += len(payload)

        offset = 0
        while offset + 14 <= len(payload):
            pkt_len = payload[offset]
            if pkt_len < 14 or offset + pkt_len > len(payload):
                break

            pkt_id = payload[offset + 9]
            data_len = pkt_len - 14

            packet_counts[pkt_id] = packet_counts.get(pkt_id, 0) + 1
            if pkt_id not in packet_data_lens:
                packet_data_lens[pkt_id] = []
            packet_data_lens[pkt_id].append(data_len)

            offset += pkt_len

    print(f"\nTotal messages: {len(messages)}")
    print(f"Total payload bytes: {total_bytes:,}")

    print("\nPacket type distribution:")
    print(f"{'Tag':<10} {'Count':>10} {'Data Len (min-max)':>20} {'Sensor Type':<15}")
    print("-" * 60)

    for tag in sorted(packet_counts.keys()):
        count = packet_counts[tag]
        lens = packet_data_lens[tag]
        min_len = min(lens)
        max_len = max(lens)

        sensor_info = SENSORS.get(tag, {"type": "Unknown"})
        sensor_type = sensor_info["type"]
        expected_len = sensor_info.get("data_len", "?")

        len_str = f"{min_len}-{max_len}" if min_len != max_len else str(min_len)
        print(
            f"0x{tag:02X}       {count:>10} {len_str:>20} {sensor_type:<15} (expected: {expected_len})"
        )

    # Check for 0x88 specifically
    if 0x88 in packet_counts:
        print(f"\n⚠ 0x88 packets found: {packet_counts[0x88]}")
        print(
            "  This is a new firmware status packet with variable length (188-230 bytes)"
        )
        print("  It should NOT interfere with other packet types.")

        lens_88 = packet_data_lens[0x88]
        print(
            f"  Data lengths: min={min(lens_88)}, max={max(lens_88)}, unique={sorted(set(lens_88))[:5]}..."
        )

    return packet_counts, packet_data_lens


def main():
    """Run all validation tests."""

    # Files to test
    decoding_dir = Path(__file__).parent / "decoding"

    old_file = decoding_dir / "device1.txt"  # Old firmware
    new_file = decoding_dir / "device2.txt"  # New firmware
    new_file_long = decoding_dir / "device3a.txt"  # Longer new firmware recording

    # ========================================
    # Test 1: Sampling Rate Verification
    # ========================================
    print("\n" + "=" * 80)
    print("TEST 1: SAMPLING RATE VERIFICATION")
    print("=" * 80)

    if old_file.exists():
        old_data, old_rates = analyze_sampling_rates(
            str(old_file), "OLD Firmware (device1)"
        )
        analyze_packet_structure(str(old_file), "OLD Firmware (device1)")
    else:
        print(f"File not found: {old_file}")
        old_data = None

    if new_file.exists():
        new_data, new_rates = analyze_sampling_rates(
            str(new_file), "NEW Firmware (device2)"
        )
        analyze_packet_structure(str(new_file), "NEW Firmware (device2)")
    else:
        print(f"File not found: {new_file}")
        new_data = None

    if new_file_long.exists():
        new_long_data, new_long_rates = analyze_sampling_rates(
            str(new_file_long), "NEW Firmware Long (device3a)"
        )
        analyze_packet_structure(str(new_file_long), "NEW Firmware Long (device3a)")

    # ========================================
    # Test 2: AMB Channel Correlations
    # ========================================
    print("\n" + "=" * 80)
    print("TEST 2: AMB CHANNEL CORRELATION ANALYSIS")
    print("=" * 80)

    if old_data:
        analyze_amb_correlations(old_data, "OLD Firmware")
    if new_data:
        analyze_amb_correlations(new_data, "NEW Firmware")

    # ========================================
    # Test 3: Inversion Hypothesis
    # ========================================
    if old_data and new_data:
        test_inversion_hypothesis(old_data, new_data)

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nKey Questions Answered:")
    print("1. Are sampling rates correct? Check the 'Rate Error' values above.")
    print("   - Should be within ±1% for EEG, ACCGYRO, OPTICS")
    print("   - BATTERY rate varies by firmware (1Hz old, ~0.2Hz new)")
    print()
    print("2. Is 0x88 packet causing data loss?")
    print("   - Compare packet counts between old and new firmware")
    print("   - EEG/ACCGYRO/OPTICS counts should be proportional to recording duration")
    print()
    print("3. Do INNER and OUTER AMB channels correlate?")
    print("   - Check the correlation matrices above")
    print("   - If inversion hypothesis is correct, old INNER AMB was inverted")


if __name__ == "__main__":
    main()
