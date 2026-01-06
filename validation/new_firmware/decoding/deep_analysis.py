"""
Deep analysis of new firmware tags, especially 0x88 and battery changes.
Also test decoding of all three device files and compute effective sampling rates.
"""

import struct
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from OpenMuse.decode import (
    SENSORS,
    PACKET_HEADER_SIZE,
    SUBPACKET_HEADER_SIZE,
    decode_rawdata,
    parse_message,
    make_timestamps,
)


def load_messages(filepath: str) -> list:
    """Load messages from a raw data file."""
    messages = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                messages.append(line)
    return messages


def find_tag_0x88_packets(filepath: str):
    """Find and analyze all packets containing 0x88 tag."""
    print(f"\n{'='*80}")
    print(f"Searching for 0x88 tag in: {filepath}")
    print(f"{'='*80}")

    messages = load_messages(filepath)
    found = []

    for i, msg in enumerate(messages):
        parts = msg.strip().split("\t", 2)
        if len(parts) < 3:
            continue
        ts, uuid, hexstring = parts
        payload = bytes.fromhex(hexstring.strip())

        # Look for 0x88 anywhere in payload
        positions = [j for j, b in enumerate(payload) if b == 0x88]
        if positions:
            found.append(
                {
                    "msg_index": i,
                    "timestamp": ts,
                    "positions": positions,
                    "payload_hex": hexstring[:200],
                    "payload_len": len(payload),
                    "payload": payload,
                }
            )

    print(f"Found {len(found)} messages containing 0x88")

    for item in found[:5]:
        print(f"\n--- Message {item['msg_index']} ---")
        print(f"Timestamp: {item['timestamp']}")
        print(f"0x88 positions: {item['positions']}")

        # Analyze context around 0x88
        payload = item["payload"]
        for pos in item["positions"]:
            print(f"\n  Context around position {pos}:")
            start = max(0, pos - 10)
            end = min(len(payload), pos + 30)
            context = payload[start:end]
            print(f"    Bytes: {context.hex()}")
            print(f"    Position in context: {pos - start}")

            # Check if this looks like a subpacket header
            if pos + 5 < len(payload):
                potential_header = payload[pos : pos + 5]
                print(f"    Potential subpacket header: {potential_header.hex()}")
                print(
                    f"    Tag: 0x88, Index: {potential_header[1]}, Unknown: {potential_header[2:5].hex()}"
                )

    return found


def analyze_tag_0x53(filepath: str):
    """Analyze the 0x53 (Unknown) tag to understand its structure."""
    print(f"\n{'='*80}")
    print(f"Analyzing 0x53 tag in: {filepath}")
    print(f"{'='*80}")

    messages = load_messages(filepath)
    samples_0x53 = []

    for msg in messages[:1000]:
        parts = msg.strip().split("\t", 2)
        if len(parts) < 3:
            continue
        payload = bytes.fromhex(parts[2].strip())

        # Find 0x53 in payload
        for i in range(len(payload)):
            if payload[i] == 0x53:
                if i + SUBPACKET_HEADER_SIZE + 24 <= len(payload):
                    # Extract potential 0x53 data
                    data = payload[
                        i + SUBPACKET_HEADER_SIZE : i + SUBPACKET_HEADER_SIZE + 24
                    ]
                    samples_0x53.append(
                        {
                            "position": i,
                            "index": payload[i + 1] if i + 1 < len(payload) else None,
                            "data": data.hex(),
                            "data_bytes": data,
                        }
                    )

    print(f"Found {len(samples_0x53)} potential 0x53 subpackets")

    # Look for patterns
    if samples_0x53:
        print("\nSample 0x53 data patterns:")
        for sample in samples_0x53[:10]:
            data = sample["data_bytes"]
            print(f"  Index {sample['index']}: {sample['data']}")
            # Try different interpretations
            if len(data) >= 12:
                # As 6 uint16 values
                vals_u16 = struct.unpack("<6H", data[:12])
                print(f"    As 6x uint16: {vals_u16}")
                # As 6 int16 values
                vals_i16 = struct.unpack("<6h", data[:12])
                print(f"    As 6x int16: {vals_i16}")


def test_decoding(filepath: str, max_messages: int = None):
    """Test the current decoder on a file and report results."""
    print(f"\n{'='*80}")
    print(f"TESTING DECODER: {filepath}")
    print(f"{'='*80}")

    messages = load_messages(filepath)
    if max_messages:
        messages = messages[:max_messages]

    print(f"Processing {len(messages)} messages...")

    # Decode using existing function
    try:
        result = decode_rawdata(messages)

        for sensor_type, df in result.items():
            if df.empty:
                print(f"\n{sensor_type}: No data")
                continue

            print(f"\n{sensor_type}:")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")

            if "time" in df.columns and len(df) > 1:
                times = df["time"].values
                diffs = np.diff(times)

                # Compute effective sampling rate
                mean_diff = np.mean(diffs)
                effective_rate = 1.0 / mean_diff if mean_diff > 0 else 0

                print(f"  Time range: {times[0]:.3f} - {times[-1]:.3f} s")
                print(f"  Duration: {times[-1] - times[0]:.3f} s")
                print(f"  Effective sampling rate: {effective_rate:.2f} Hz")

                # Check expected rate
                expected_rate = {
                    "EEG": 256.0,
                    "ACCGYRO": 52.0,
                    "OPTICS": 64.0,
                    "BATTERY": 1.0,
                }.get(sensor_type, 0)
                if expected_rate > 0:
                    rate_error = (
                        abs(effective_rate - expected_rate) / expected_rate * 100
                    )
                    print(
                        f"  Expected rate: {expected_rate} Hz, Error: {rate_error:.1f}%"
                    )

                # Check monotonicity
                non_mono = np.sum(diffs < 0)
                print(
                    f"  Non-monotonic samples: {non_mono} ({100*non_mono/len(diffs):.2f}%)"
                )

            # Show data summary
            data_cols = [c for c in df.columns if c != "time"]
            if data_cols:
                print(f"  Data range per channel:")
                for col in data_cols[:4]:  # First 4 channels
                    print(
                        f"    {col}: min={df[col].min():.4f}, max={df[col].max():.4f}, mean={df[col].mean():.4f}"
                    )
                if len(data_cols) > 4:
                    print(f"    ... and {len(data_cols) - 4} more channels")

    except Exception as e:
        print(f"Error during decoding: {e}")
        import traceback

        traceback.print_exc()

    return result


def compare_decoding_results():
    """Compare decoding results across all device files."""
    base_path = Path(__file__).parent

    files = [
        ("device1.txt", "Old FW"),
        ("device2.txt", "New FW"),
        ("device3.txt", "Unknown FW"),
        ("device3b.txt", "Unknown FW 2"),
        ("device3c.txt", "Unknown FW 3"),
        ("device3d.txt", "Unknown FW 4"),
    ]

    results = {}
    for fname, label in files:
        fpath = base_path / fname
        if fpath.exists():
            print(f"\n{'#'*80}")
            print(f"# {label}: {fname}")
            print(f"{'#'*80}")
            results[fname] = test_decoding(str(fpath), max_messages=2000)

    return results


def visualize_data(filepath: str, sensor_type: str = "EEG", max_samples: int = 2000):
    """Create a simple text visualization of decoded data."""
    print(f"\n{'='*80}")
    print(f"DATA VISUALIZATION: {filepath} - {sensor_type}")
    print(f"{'='*80}")

    messages = load_messages(filepath)
    result = decode_rawdata(messages[:1000])

    df = result.get(sensor_type)
    if df is None or df.empty:
        print(f"No {sensor_type} data found")
        return

    # Sample data for display
    df = df.head(max_samples)

    print(f"\nFirst 20 samples:")
    print(df.head(20).to_string())

    print(f"\nStatistics:")
    print(df.describe())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--find-0x88", action="store_true", help="Find 0x88 tag")
    parser.add_argument("--analyze-0x53", action="store_true", help="Analyze 0x53 tag")
    parser.add_argument("--test-decode", action="store_true", help="Test decoder")
    parser.add_argument("--compare", action="store_true", help="Compare all files")
    parser.add_argument("--visualize", action="store_true", help="Visualize data")
    parser.add_argument("--file", type=str, help="Specific file")
    parser.add_argument(
        "--sensor", type=str, default="EEG", help="Sensor type for visualization"
    )

    args = parser.parse_args()

    base_path = Path(__file__).parent

    if args.compare or (
        not any([args.find_0x88, args.analyze_0x53, args.test_decode, args.visualize])
    ):
        compare_decoding_results()

    if args.find_0x88:
        for fname in ["device1.txt", "device2.txt", "device3.txt"]:
            fpath = base_path / fname
            if fpath.exists():
                find_tag_0x88_packets(str(fpath))

    if args.analyze_0x53:
        for fname in ["device1.txt", "device2.txt", "device3.txt"]:
            fpath = base_path / fname
            if fpath.exists():
                analyze_tag_0x53(str(fpath))

    if args.test_decode and args.file:
        test_decoding(args.file)

    if args.visualize and args.file:
        visualize_data(args.file, args.sensor)
