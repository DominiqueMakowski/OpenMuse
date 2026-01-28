"""
Detailed analysis of reorder magnitude distribution.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import struct
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

DEVICE_CLOCK_HZ = 256000.0
PACKET_HEADER_SIZE = 14


def parse_raw_file(filepath: str) -> list:
    """Parse raw BLE messages from a .txt file."""
    messages = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                parts = line.split('\t')
                if len(parts) < 3:
                    continue

                ts_str, uuid, hexdata = parts[0], parts[1], parts[2]
                arrival_time = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                payload = bytes.fromhex(hexdata.strip())

                device_times = []
                offset = 0

                while offset < len(payload):
                    if offset + PACKET_HEADER_SIZE > len(payload):
                        break
                    pkt_len = payload[offset]
                    if offset + pkt_len > len(payload):
                        break
                    pkt_time_raw = struct.unpack_from('<I', payload, offset + 2)[0]
                    pkt_time = pkt_time_raw / DEVICE_CLOCK_HZ
                    device_times.append(pkt_time)
                    offset += pkt_len

                if device_times:
                    messages.append({
                        'arrival_ts': arrival_time.timestamp(),
                        'min_device_time': min(device_times),
                        'max_device_time': max(device_times),
                    })

            except Exception:
                continue

    return messages


def get_reorder_magnitudes(messages):
    """Get all reorder magnitudes in ms."""
    max_seen = messages[0]['max_device_time']
    magnitudes = []

    for msg in messages[1:]:
        if msg['min_device_time'] < max_seen:
            mag_ms = (max_seen - msg['min_device_time']) * 1000
            magnitudes.append(mag_ms)
        max_seen = max(max_seen, msg['max_device_time'])

    return np.array(magnitudes)


def main():
    txt_files = sorted([f for f in os.listdir('.') if f.endswith('.txt') and f != 'prompt.txt'])

    print("Detailed Reorder Magnitude Analysis")
    print("=" * 80)

    all_magnitudes = []
    file_magnitudes = {}

    for filename in txt_files:
        print(f"\n{filename}:")
        messages = parse_raw_file(filename)
        if len(messages) < 10:
            print("  Too few messages")
            continue

        mags = get_reorder_magnitudes(messages)
        file_magnitudes[filename] = mags
        all_magnitudes.extend(mags)

        if len(mags) > 0:
            print(f"  Total out-of-order: {len(mags)}")
            print(f"  Percentiles (ms):")
            for p in [50, 75, 90, 95, 99, 99.9, 100]:
                val = np.percentile(mags, p)
                print(f"    p{p:5.1f}: {val:8.1f}ms")

            # Count by magnitude buckets
            buckets = [0, 5, 10, 25, 50, 100, 150, 200, 500, 1000, float('inf')]
            print(f"  Distribution:")
            for i in range(len(buckets)-1):
                count = np.sum((mags >= buckets[i]) & (mags < buckets[i+1]))
                pct = 100 * count / len(mags)
                if count > 0:
                    label = f"    {buckets[i]:4.0f}-{buckets[i+1]:4.0f}ms" if buckets[i+1] != float('inf') else f"    {buckets[i]:4.0f}ms+"
                    print(f"{label}: {count:5d} ({pct:5.1f}%)")

    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    mags = np.array(all_magnitudes)
    print(f"\nTotal out-of-order messages across all files: {len(mags)}")
    print(f"\nOverall percentiles (ms):")
    for p in [50, 75, 90, 95, 99, 99.5, 99.9, 100]:
        val = np.percentile(mags, p)
        print(f"  p{p:5.1f}: {val:8.1f}ms")

    print(f"\n--- BUFFER COVERAGE ANALYSIS ---")
    for buffer_ms in [50, 75, 100, 125, 150, 175, 200, 250, 300]:
        coverage = 100 * np.sum(mags <= buffer_ms) / len(mags)
        print(f"  {buffer_ms:3d}ms buffer: captures {coverage:5.1f}% of reordered messages")

    # Excluding the extreme outlier file
    print(f"\n--- EXCLUDING test2_standard2_unknow.txt (has extreme 1s+ outliers) ---")
    mags_filtered = []
    for fname, m in file_magnitudes.items():
        if 'standard2_unknow' not in fname:
            mags_filtered.extend(m)
    mags_filtered = np.array(mags_filtered)

    print(f"Filtered out-of-order messages: {len(mags_filtered)}")
    print(f"Filtered percentiles (ms):")
    for p in [50, 75, 90, 95, 99, 99.5, 99.9, 100]:
        val = np.percentile(mags_filtered, p)
        print(f"  p{p:5.1f}: {val:8.1f}ms")

    print(f"\n--- BUFFER COVERAGE (filtered) ---")
    for buffer_ms in [50, 75, 100, 125, 150, 175, 200]:
        coverage = 100 * np.sum(mags_filtered <= buffer_ms) / len(mags_filtered)
        print(f"  {buffer_ms:3d}ms buffer: captures {coverage:5.1f}% of reordered messages")


if __name__ == '__main__':
    main()
