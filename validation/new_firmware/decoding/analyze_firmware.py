"""
Firmware Analysis Script
========================

This script analyzes raw Muse data files to understand differences between
device firmwares and identify parsing issues.

Key differences to investigate:
1. Battery (0x98): Old FW sends as standalone packet (20 bytes), New FW as subpacket (14 bytes)
2. EEG (0x12): Old FW has fixed 28-byte payload, New FW has variable length
3. New tag 0x88: Appears in newer firmware, possibly PPG/ECG related
"""

import struct
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import sys

# Add parent to path to import decode module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from OpenMuse.decode import SENSORS, PACKET_HEADER_SIZE, SUBPACKET_HEADER_SIZE


def load_messages(filepath: str) -> list:
    """Load messages from a raw data file."""
    messages = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                messages.append(line)
    return messages


def parse_raw_packet(payload: bytes) -> dict:
    """
    Parse a raw BLE packet with detailed structure analysis.
    Returns comprehensive information about packet structure.
    """
    result = {
        "raw_bytes": payload.hex(),
        "length": len(payload),
        "packets": [],
        "errors": [],
    }

    offset = 0
    packet_num = 0

    while offset < len(payload):
        if offset + PACKET_HEADER_SIZE > len(payload):
            result["errors"].append(f"Incomplete header at offset {offset}")
            break

        pkt_len = payload[offset]

        if pkt_len == 0:
            result["errors"].append(f"Zero length packet at offset {offset}")
            break

        if offset + pkt_len > len(payload):
            result["errors"].append(
                f"Packet extends beyond payload at offset {offset}: declared {pkt_len} bytes"
            )
            break

        pkt_bytes = payload[offset : offset + pkt_len]

        # Parse packet header
        pkt_info = {
            "offset": offset,
            "declared_len": pkt_len,
            "actual_len": len(pkt_bytes),
            "pkt_index": pkt_bytes[1],
            "pkt_time_raw": struct.unpack_from("<I", pkt_bytes, 2)[0],
            "unknown1": pkt_bytes[6:9].hex(),
            "pkt_id": pkt_bytes[9],
            "pkt_id_hex": hex(pkt_bytes[9]),
            "unknown2": pkt_bytes[10:13].hex(),
            "byte_13": pkt_bytes[13],
            "header_bytes": pkt_bytes[:PACKET_HEADER_SIZE].hex(),
            "data_bytes": (
                pkt_bytes[PACKET_HEADER_SIZE:].hex()
                if len(pkt_bytes) > PACKET_HEADER_SIZE
                else ""
            ),
            "data_len": len(pkt_bytes) - PACKET_HEADER_SIZE,
            "subpackets": [],
        }

        # Identify packet type
        pkt_config = SENSORS.get(pkt_bytes[9])
        pkt_info["sensor_type"] = pkt_config["type"] if pkt_config else "Unknown"
        pkt_info["expected_data_len"] = pkt_config["data_len"] if pkt_config else None

        # Parse data section for subpackets
        data_section = pkt_bytes[PACKET_HEADER_SIZE:]
        _parse_data_section(data_section, pkt_info)

        result["packets"].append(pkt_info)
        offset += pkt_len
        packet_num += 1

    return result


def _parse_data_section(data: bytes, pkt_info: dict):
    """Analyze data section for subpackets."""
    offset = 0
    subpkt_num = 0
    pkt_id = pkt_info["pkt_id"]

    # First data block (no TAG, matches pkt_id type)
    first_config = SENSORS.get(pkt_id)
    if first_config and first_config["data_len"] > 0:
        expected_first_len = first_config["data_len"]
        if offset + expected_first_len <= len(data):
            first_subpkt = {
                "index": subpkt_num,
                "tag": pkt_id,
                "tag_hex": hex(pkt_id),
                "type": "primary",
                "expected_len": expected_first_len,
                "actual_len": min(expected_first_len, len(data) - offset),
                "data": data[offset : offset + expected_first_len].hex(),
                "has_header": False,
            }
            pkt_info["subpackets"].append(first_subpkt)
            offset += expected_first_len
            subpkt_num += 1

    # Remaining subpackets (with TAG + 4-byte header)
    while offset < len(data):
        if offset + 1 > len(data):
            break

        tag = data[offset]
        tag_config = SENSORS.get(tag)

        subpkt = {
            "index": subpkt_num,
            "tag": tag,
            "tag_hex": hex(tag),
            "type": "secondary",
            "offset_in_data": offset,
            "has_header": True,
        }

        if tag_config:
            subpkt["sensor_type"] = tag_config["type"]
            subpkt["expected_data_len"] = tag_config["data_len"]

            if offset + SUBPACKET_HEADER_SIZE > len(data):
                subpkt["error"] = "Incomplete header"
                pkt_info["subpackets"].append(subpkt)
                break

            subpkt["subpkt_index"] = data[offset + 1]
            subpkt["unknown_bytes"] = data[
                offset + 2 : offset + SUBPACKET_HEADER_SIZE
            ].hex()

            full_len = SUBPACKET_HEADER_SIZE + tag_config["data_len"]
            if offset + full_len <= len(data):
                subpkt["data"] = data[
                    offset + SUBPACKET_HEADER_SIZE : offset + full_len
                ].hex()
                subpkt["actual_len"] = tag_config["data_len"]
                offset += full_len
            else:
                remaining = len(data) - offset - SUBPACKET_HEADER_SIZE
                subpkt["data"] = data[offset + SUBPACKET_HEADER_SIZE :].hex()
                subpkt["actual_len"] = remaining
                subpkt["error"] = (
                    f'Truncated: expected {tag_config["data_len"]}, got {remaining}'
                )
                pkt_info["subpackets"].append(subpkt)
                break
        else:
            # Unknown tag - try to show remaining bytes
            subpkt["sensor_type"] = "Unknown"
            subpkt["remaining_bytes"] = data[offset:].hex()
            subpkt["remaining_len"] = len(data) - offset
            pkt_info["subpackets"].append(subpkt)
            break

        pkt_info["subpackets"].append(subpkt)
        subpkt_num += 1


def scan_for_tags(payload: bytes) -> dict:
    """Scan payload for all potential TAG bytes and their positions."""
    known_tags = set(SENSORS.keys())
    found = defaultdict(list)

    for i, byte in enumerate(payload):
        if byte in known_tags:
            found[byte].append(i)

    # Also look for unknown potential tags (common values that might be new)
    potential_new_tags = [0x88, 0x99, 0x33, 0x44, 0x55, 0x66, 0x77]
    for tag in potential_new_tags:
        if tag not in known_tags:
            positions = [i for i, b in enumerate(payload) if b == tag]
            if positions:
                found[tag] = positions

    return dict(found)


def analyze_file(filepath: str, max_messages: int = 100):
    """Analyze a raw data file and print statistics."""
    print(f"\n{'='*80}")
    print(f"ANALYZING: {filepath}")
    print(f"{'='*80}")

    messages = load_messages(filepath)
    print(f"Total messages: {len(messages)}")

    # Statistics
    packet_id_counts = defaultdict(int)
    subpacket_tag_counts = defaultdict(int)
    data_lengths = defaultdict(list)
    unknown_tags_found = defaultdict(int)
    battery_as_subpacket = 0
    battery_as_packet = 0
    tag_0x88_count = 0
    errors = []

    # Analyze messages
    for i, msg in enumerate(messages[:max_messages]):
        try:
            parts = msg.strip().split("\t", 2)
            if len(parts) < 3:
                continue
            ts, uuid, hexstring = parts
            payload = bytes.fromhex(hexstring.strip())

            # Detailed packet analysis
            result = parse_raw_packet(payload)

            for pkt in result["packets"]:
                pkt_id = pkt["pkt_id"]
                packet_id_counts[hex(pkt_id)] += 1
                data_lengths[hex(pkt_id)].append(pkt["data_len"])

                if pkt_id == 0x98:
                    battery_as_packet += 1

                for subpkt in pkt["subpackets"]:
                    if subpkt["type"] == "secondary":
                        tag = subpkt["tag"]
                        subpacket_tag_counts[hex(tag)] += 1

                        if tag == 0x98:
                            battery_as_subpacket += 1
                        if tag == 0x88:
                            tag_0x88_count += 1

                        if subpkt.get("sensor_type") == "Unknown":
                            unknown_tags_found[hex(tag)] += 1

            if result["errors"]:
                errors.extend(result["errors"])

        except Exception as e:
            errors.append(f"Message {i}: {str(e)}")

    # Print results
    print(f"\n--- Packet IDs (Primary Packet Types) ---")
    for tag, count in sorted(packet_id_counts.items(), key=lambda x: -x[1]):
        sensor_config = SENSORS.get(int(tag, 16))
        sensor_name = sensor_config["type"] if sensor_config else "UNKNOWN"
        lengths = data_lengths[tag]
        min_len = min(lengths) if lengths else 0
        max_len = max(lengths) if lengths else 0
        expected = sensor_config["data_len"] if sensor_config else "?"
        print(
            f"  {tag} ({sensor_name}): {count} packets, data_len: {min_len}-{max_len} (expected: {expected})"
        )

    print(f"\n--- Subpacket Tags (Secondary Data Types) ---")
    for tag, count in sorted(subpacket_tag_counts.items(), key=lambda x: -x[1]):
        sensor_config = SENSORS.get(int(tag, 16))
        sensor_name = sensor_config["type"] if sensor_config else "UNKNOWN"
        print(f"  {tag} ({sensor_name}): {count} subpackets")

    print(f"\n--- Key Observations ---")
    print(f"  Battery as standalone packet (0x98): {battery_as_packet}")
    print(f"  Battery as subpacket: {battery_as_subpacket}")
    print(f"  Tag 0x88 occurrences: {tag_0x88_count}")

    if unknown_tags_found:
        print(f"\n--- Unknown Tags Found ---")
        for tag, count in unknown_tags_found.items():
            print(f"  {tag}: {count}")

    if errors:
        print(f"\n--- Parsing Errors ({len(errors)}) ---")
        for err in errors[:10]:
            print(f"  {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    return {
        "packet_ids": packet_id_counts,
        "subpacket_tags": subpacket_tag_counts,
        "battery_as_packet": battery_as_packet,
        "battery_as_subpacket": battery_as_subpacket,
        "tag_0x88": tag_0x88_count,
        "errors": len(errors),
    }


def detailed_packet_dump(filepath: str, num_messages: int = 5):
    """Print detailed packet structure for first few messages."""
    print(f"\n{'='*80}")
    print(f"DETAILED PACKET DUMP: {filepath}")
    print(f"{'='*80}")

    messages = load_messages(filepath)

    for i, msg in enumerate(messages[:num_messages]):
        print(f"\n--- Message {i+1} ---")
        try:
            parts = msg.strip().split("\t", 2)
            if len(parts) < 3:
                continue
            ts, uuid, hexstring = parts
            payload = bytes.fromhex(hexstring.strip())

            print(f"Timestamp: {ts}")
            print(f"Payload length: {len(payload)} bytes")
            print(f"Raw: {hexstring[:100]}...")

            result = parse_raw_packet(payload)

            for j, pkt in enumerate(result["packets"]):
                print(f"\n  Packet {j+1}:")
                print(f"    Offset: {pkt['offset']}, Length: {pkt['declared_len']}")
                print(f"    Index: {pkt['pkt_index']}, Time: {pkt['pkt_time_raw']}")
                print(f"    ID: {pkt['pkt_id_hex']} ({pkt['sensor_type']})")
                print(
                    f"    Data section: {pkt['data_len']} bytes (expected: {pkt['expected_data_len']})"
                )
                print(f"    Header: {pkt['header_bytes']}")

                for k, subpkt in enumerate(pkt["subpackets"]):
                    print(f"\n    Subpacket {k+1}:")
                    print(
                        f"      Tag: {subpkt['tag_hex']} ({subpkt.get('sensor_type', 'N/A')})"
                    )
                    print(f"      Type: {subpkt['type']}")
                    if subpkt.get("subpkt_index") is not None:
                        print(f"      Subpkt Index: {subpkt['subpkt_index']}")
                    if subpkt.get("data"):
                        print(f"      Data: {subpkt['data'][:60]}...")
                    if subpkt.get("error"):
                        print(f"      ERROR: {subpkt['error']}")
                    if subpkt.get("remaining_bytes"):
                        print(f"      Remaining: {subpkt['remaining_bytes'][:60]}...")

        except Exception as e:
            print(f"  Error: {e}")


def compare_files():
    """Compare all device files."""
    base_path = Path(__file__).parent

    files = [
        base_path / "device1.txt",  # Old firmware (reference)
        base_path / "device2.txt",  # New firmware
        base_path / "device3.txt",  # Unknown firmware
    ]

    results = {}
    for filepath in files:
        if filepath.exists():
            results[filepath.name] = analyze_file(str(filepath), max_messages=500)

    # Summary comparison
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")

    print(
        "\nFeature                    | Device 1 (Old FW) | Device 2 (New FW) | Device 3"
    )
    print("-" * 80)

    for feature in ["battery_as_packet", "battery_as_subpacket", "tag_0x88", "errors"]:
        row = f"{feature:26} |"
        for name in ["device1.txt", "device2.txt", "device3.txt"]:
            val = results.get(name, {}).get(feature, "N/A")
            row += f" {str(val):17} |"
        print(row)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Muse firmware data files")
    parser.add_argument("--file", type=str, help="Specific file to analyze")
    parser.add_argument("--dump", action="store_true", help="Show detailed packet dump")
    parser.add_argument(
        "--compare", action="store_true", help="Compare all device files"
    )
    parser.add_argument(
        "--messages", type=int, default=500, help="Number of messages to analyze"
    )

    args = parser.parse_args()

    if args.compare or (not args.file and not args.dump):
        compare_files()

    if args.file:
        analyze_file(args.file, max_messages=args.messages)
        if args.dump:
            detailed_packet_dump(args.file, num_messages=5)

    if args.dump and not args.file:
        # Dump first few packets from each file
        base_path = Path(__file__).parent
        for fname in ["device1.txt", "device2.txt", "device3.txt"]:
            fpath = base_path / fname
            if fpath.exists():
                detailed_packet_dump(str(fpath), num_messages=3)
