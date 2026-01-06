"""
Deep analysis of 0x88 packets to understand their structure.

Goals:
1. Determine effective sampling rate of 0x88 packets
2. Explore different unpacking strategies to infer channel counts
3. Look for embedded sensor data patterns
"""

import struct
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from OpenMuse.decode import SENSORS, PACKET_HEADER_SIZE, DEVICE_CLOCK_HZ


def load_messages(filepath: str) -> list:
    """Load messages from a raw data file."""
    messages = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                messages.append(line)
    return messages


def extract_0x88_packets(filepath: str):
    """Extract all 0x88 packets from file."""
    messages = load_messages(filepath)
    packets_0x88 = []

    for msg in messages:
        parts = msg.strip().split("\t", 2)
        if len(parts) < 3:
            continue
        ts, uuid, hexstring = parts
        payload = bytes.fromhex(hexstring.strip())

        # Parse packets
        offset = 0
        while offset < len(payload):
            if offset + PACKET_HEADER_SIZE > len(payload):
                break

            pkt_len = payload[offset]
            if offset + pkt_len > len(payload):
                break

            pkt_bytes = payload[offset : offset + pkt_len]
            pkt_id = pkt_bytes[9]

            if pkt_id == 0x88:
                pkt_index = pkt_bytes[1]
                pkt_time_raw = struct.unpack_from("<I", pkt_bytes, 2)[0]
                pkt_time = pkt_time_raw / DEVICE_CLOCK_HZ
                pkt_data = pkt_bytes[PACKET_HEADER_SIZE:]
                byte_13 = pkt_bytes[13]

                packets_0x88.append(
                    {
                        "timestamp": ts,
                        "pkt_index": pkt_index,
                        "pkt_time": pkt_time,
                        "pkt_time_raw": pkt_time_raw,
                        "byte_13": byte_13,
                        "data_len": len(pkt_data),
                        "data": pkt_data,
                    }
                )

            offset += pkt_len

    return packets_0x88


def analyze_sampling_rate(packets):
    """Analyze the effective sampling rate of 0x88 packets."""
    if len(packets) < 2:
        return None

    times = np.array([p["pkt_time"] for p in packets])
    diffs = np.diff(times)

    # Filter out outliers (e.g., gaps > 1 second)
    valid_diffs = diffs[(diffs > 0) & (diffs < 1.0)]

    if len(valid_diffs) == 0:
        return None

    mean_interval = np.mean(valid_diffs)
    std_interval = np.std(valid_diffs)
    median_interval = np.median(valid_diffs)

    return {
        "count": len(packets),
        "duration": times[-1] - times[0],
        "mean_interval_ms": mean_interval * 1000,
        "std_interval_ms": std_interval * 1000,
        "median_interval_ms": median_interval * 1000,
        "effective_rate_hz": 1.0 / mean_interval if mean_interval > 0 else 0,
        "min_interval_ms": np.min(valid_diffs) * 1000,
        "max_interval_ms": np.max(valid_diffs) * 1000,
    }


def analyze_channel_possibilities(packets):
    """
    Explore different unpacking strategies to infer what data might be inside.

    Known Muse data formats:
    - EEG: 14-bit packed (28 bytes = 16 values)
    - OPTICS: 20-bit packed (30/40 bytes = 12/16 values)
    - ACCGYRO: 16-bit int (36 bytes = 18 values = 3 samples x 6 channels)
    - Battery: 16-bit int (first 2 bytes)
    """
    if not packets:
        return {}

    # Get the most common data length
    data_lens = [p["data_len"] for p in packets]
    common_len = max(set(data_lens), key=data_lens.count)

    # Filter to packets with common length
    filtered = [p for p in packets if p["data_len"] == common_len]
    if not filtered:
        return {}

    results = {
        "common_data_len": common_len,
        "packet_count": len(filtered),
        "interpretations": [],
    }

    # Stack all data for analysis
    all_data = np.array(
        [list(p["data"][:common_len]) for p in filtered], dtype=np.uint8
    )

    # Interpretation 1: As 16-bit signed integers (like ACCGYRO)
    n_int16 = common_len // 2
    usable_bytes = n_int16 * 2  # Ensure we use an even number of bytes
    if n_int16 > 0:
        # Truncate to even bytes
        truncated_data = all_data[:, :usable_bytes]
        data_int16 = np.frombuffer(truncated_data.tobytes(), dtype="<i2").reshape(
            len(filtered), n_int16
        )

        # Calculate statistics per "channel"
        means = np.mean(data_int16, axis=0)
        stds = np.std(data_int16, axis=0)

        # Find channels with significant variation (likely real data)
        significant = np.where(stds > 10)[0]

        results["interpretations"].append(
            {
                "format": "int16",
                "n_values": n_int16,
                "significant_channels": len(significant),
                "channel_stds": stds.tolist()[:20],  # First 20
            }
        )

    # Interpretation 2: As 32-bit floats
    n_float32 = common_len // 4
    if n_float32 > 0:
        try:
            data_float = np.frombuffer(all_data.tobytes(), dtype="<f4").reshape(
                len(filtered), n_float32
            )
            # Check if values are in reasonable range
            valid = np.isfinite(data_float).all()
            if valid:
                stds = np.std(data_float, axis=0)
                results["interpretations"].append(
                    {
                        "format": "float32",
                        "n_values": n_float32,
                        "significant_channels": np.sum(stds > 0.001),
                    }
                )
        except:
            pass

    # Interpretation 3: Check for 14-bit packed EEG-like data
    # 14 bits per value, so common_len * 8 / 14 values
    n_14bit = (common_len * 8) // 14
    if n_14bit > 0:
        results["interpretations"].append(
            {
                "format": "14-bit packed (like EEG)",
                "n_values": n_14bit,
                "possible_configs": [
                    f"{n_14bit} values total",
                    (
                        f"{n_14bit // 4} samples x 4 channels"
                        if n_14bit % 4 == 0
                        else None
                    ),
                    (
                        f"{n_14bit // 8} samples x 8 channels"
                        if n_14bit % 8 == 0
                        else None
                    ),
                ],
            }
        )

    # Interpretation 4: Check for 20-bit packed OPTICS-like data
    n_20bit = (common_len * 8) // 20
    if n_20bit > 0:
        results["interpretations"].append(
            {
                "format": "20-bit packed (like OPTICS)",
                "n_values": n_20bit,
                "possible_configs": [
                    f"{n_20bit} values total",
                    (
                        f"{n_20bit // 4} samples x 4 channels"
                        if n_20bit % 4 == 0
                        else None
                    ),
                    (
                        f"{n_20bit // 8} samples x 8 channels"
                        if n_20bit % 8 == 0
                        else None
                    ),
                    (
                        f"{n_20bit // 16} samples x 16 channels"
                        if n_20bit % 16 == 0
                        else None
                    ),
                ],
            }
        )

    # Look for battery-like pattern in first 2 bytes
    battery_values = [
        struct.unpack("<H", p["data"][0:2])[0] / 256.0 for p in filtered[:20]
    ]
    battery_stable = np.std(battery_values) < 5.0  # Battery should be relatively stable
    results["battery_in_first_2_bytes"] = {
        "likely": battery_stable and 0 < np.mean(battery_values) < 110,
        "values": [f"{v:.1f}%" for v in battery_values[:5]],
        "mean": np.mean(battery_values),
        "std": np.std(battery_values),
    }

    return results


def analyze_byte_patterns(packets):
    """Look for constant bytes (headers/markers) vs variable bytes (data)."""
    if not packets:
        return {}

    # Get common length
    data_lens = [p["data_len"] for p in packets]
    common_len = min(data_lens)

    # Stack data
    all_data = np.array([list(p["data"][:common_len]) for p in packets], dtype=np.uint8)

    # Calculate variability per byte position
    byte_stds = np.std(all_data.astype(float), axis=0)
    byte_means = np.mean(all_data.astype(float), axis=0)

    # Find constant bytes (likely headers/markers)
    constant_positions = np.where(byte_stds < 1.0)[0]

    # Find highly variable bytes (likely data)
    variable_positions = np.where(byte_stds > 30)[0]

    return {
        "analyzed_length": common_len,
        "constant_byte_positions": constant_positions.tolist()[:20],
        "constant_byte_values": [int(byte_means[i]) for i in constant_positions[:20]],
        "variable_regions": _find_contiguous_regions(variable_positions),
        "byte_variability": byte_stds.tolist()[:50],  # First 50 bytes
    }


def _find_contiguous_regions(positions):
    """Find contiguous regions in a list of positions."""
    if len(positions) == 0:
        return []

    regions = []
    start = positions[0]
    end = positions[0]

    for pos in positions[1:]:
        if pos == end + 1:
            end = pos
        else:
            regions.append((int(start), int(end)))
            start = pos
            end = pos

    regions.append((int(start), int(end)))
    return regions


def analyze_file(filepath: str):
    """Complete analysis of 0x88 packets in a file."""
    print(f"\n{'='*80}")
    print(f"ANALYZING: {Path(filepath).name}")
    print(f"{'='*80}")

    packets = extract_0x88_packets(filepath)

    if not packets:
        print("No 0x88 packets found")
        return None

    # Sampling rate analysis
    print(f"\n--- SAMPLING RATE ANALYSIS ---")
    rate_info = analyze_sampling_rate(packets)
    if rate_info:
        print(f"  Packet count: {rate_info['count']}")
        print(f"  Duration: {rate_info['duration']:.2f} s")
        print(f"  Effective rate: {rate_info['effective_rate_hz']:.2f} Hz")
        print(f"  Mean interval: {rate_info['mean_interval_ms']:.2f} ms")
        print(f"  Std interval: {rate_info['std_interval_ms']:.2f} ms")
        print(
            f"  Interval range: {rate_info['min_interval_ms']:.2f} - {rate_info['max_interval_ms']:.2f} ms"
        )

    # Data length distribution
    data_lens = [p["data_len"] for p in packets]
    unique_lens = sorted(set(data_lens))
    print(f"\n--- DATA LENGTH DISTRIBUTION ---")
    print(f"  Unique lengths: {unique_lens}")
    for length in unique_lens:
        count = data_lens.count(length)
        print(f"    {length} bytes: {count} packets ({100*count/len(packets):.1f}%)")

    # Channel possibilities
    print(f"\n--- CHANNEL INTERPRETATION ANALYSIS ---")
    channel_info = analyze_channel_possibilities(packets)
    if channel_info:
        print(f"  Most common data length: {channel_info['common_data_len']} bytes")
        for interp in channel_info["interpretations"]:
            print(f"\n  Format: {interp['format']}")
            print(f"    Values per packet: {interp['n_values']}")
            if "significant_channels" in interp:
                print(f"    Significant channels: {interp['significant_channels']}")
            if "possible_configs" in interp:
                for cfg in interp["possible_configs"]:
                    if cfg:
                        print(f"    Possible: {cfg}")

        batt = channel_info.get("battery_in_first_2_bytes", {})
        if batt:
            print(f"\n  Battery in first 2 bytes:")
            print(f"    Likely: {batt['likely']}")
            print(f"    Sample values: {batt['values']}")
            print(f"    Mean: {batt['mean']:.1f}%, Std: {batt['std']:.2f}")

    # Byte patterns
    print(f"\n--- BYTE PATTERN ANALYSIS ---")
    patterns = analyze_byte_patterns(packets)
    if patterns:
        print(f"  Analyzed length: {patterns['analyzed_length']} bytes")
        print(f"  Constant byte positions: {patterns['constant_byte_positions']}")
        print(f"  Variable regions (likely data): {patterns['variable_regions']}")

    return {
        "rate_info": rate_info,
        "channel_info": channel_info,
        "patterns": patterns,
    }


def compare_all_files():
    """Compare 0x88 analysis across all device files."""
    base_path = Path(__file__).parent

    files = [
        "device1.txt",
        "device2.txt",
        "device3.txt",
        "device3b.txt",
        "device3c.txt",
        "device3d.txt",
    ]

    print(f"\n{'#'*80}")
    print(f"# 0x88 PACKET ANALYSIS - ALL DEVICES")
    print(f"{'#'*80}")

    results = {}
    for fname in files:
        fpath = base_path / fname
        if fpath.exists():
            results[fname] = analyze_file(str(fpath))

    # Summary comparison
    print(f"\n\n{'#'*80}")
    print(f"# SUMMARY COMPARISON")
    print(f"{'#'*80}")

    print(
        f"\n{'Device':<15} {'Packets':>10} {'Rate (Hz)':>12} {'Data Len':>15} {'Battery':>10}"
    )
    print("-" * 70)

    for fname, data in results.items():
        if data is None:
            print(f"{fname:<15} {'No 0x88':>10}")
            continue

        rate = data["rate_info"]["effective_rate_hz"] if data["rate_info"] else 0
        n_pkts = data["rate_info"]["count"] if data["rate_info"] else 0
        data_len = (
            data["channel_info"]["common_data_len"] if data["channel_info"] else "N/A"
        )
        batt = (
            data["channel_info"].get("battery_in_first_2_bytes", {}).get("mean", 0)
            if data["channel_info"]
            else 0
        )

        print(
            f"{fname:<15} {n_pkts:>10} {rate:>12.2f} {str(data_len):>15} {batt:>9.1f}%"
        )

    return results


if __name__ == "__main__":
    compare_all_files()
