"""
Analyze arrival timing and understand what the '1ms reordering' actually means.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import struct
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

DEVICE_CLOCK_HZ = 256000.0
PACKET_HEADER_SIZE = 14


def parse_raw_file_detailed(filepath: str, max_msgs=50000) -> list:
    """Parse with more detail."""
    messages = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_msgs:
                break
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

                packets = []
                offset = 0

                while offset < len(payload):
                    if offset + PACKET_HEADER_SIZE > len(payload):
                        break
                    pkt_len = payload[offset]
                    if offset + pkt_len > len(payload):
                        break
                    pkt_index = payload[offset + 1]
                    pkt_time_raw = struct.unpack_from('<I', payload, offset + 2)[0]
                    pkt_time = pkt_time_raw / DEVICE_CLOCK_HZ
                    packets.append({
                        'pkt_index': pkt_index,
                        'pkt_time': pkt_time,
                        'pkt_time_raw': pkt_time_raw,
                    })
                    offset += pkt_len

                if packets:
                    messages.append({
                        'arrival_ts': arrival_time.timestamp(),
                        'packets': packets,
                        'min_device_time': min(p['pkt_time'] for p in packets),
                        'max_device_time': max(p['pkt_time'] for p in packets),
                    })

            except Exception:
                continue

    return messages


def analyze_file(filename: str):
    """Detailed analysis of one file."""
    print(f"\n{'='*70}")
    print(f"FILE: {filename}")
    print('='*70)

    messages = parse_raw_file_detailed(filename)
    if len(messages) < 100:
        print("Too few messages")
        return

    # Analyze arrival intervals
    arrival_intervals_ms = []
    for i in range(1, len(messages)):
        interval = (messages[i]['arrival_ts'] - messages[i-1]['arrival_ts']) * 1000
        arrival_intervals_ms.append(interval)

    intervals = np.array(arrival_intervals_ms)

    print(f"\n--- ARRIVAL INTERVAL DISTRIBUTION ---")
    print(f"Mean: {np.mean(intervals):.1f}ms")
    print(f"Std:  {np.std(intervals):.1f}ms")
    print(f"Min:  {np.min(intervals):.1f}ms")
    print(f"Max:  {np.max(intervals):.1f}ms")
    print(f"\nPercentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  p{p:2d}: {np.percentile(intervals, p):6.1f}ms")

    # Count messages per arrival interval bucket
    print(f"\nArrival interval distribution:")
    for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 30), (30, 50), (50, 100), (100, 200), (200, 500)]:
        count = np.sum((intervals >= lo) & (intervals < hi))
        pct = 100 * count / len(intervals)
        if count > 0:
            print(f"  {lo:3d}-{hi:3d}ms: {count:6d} ({pct:5.1f}%)")

    # Analyze device time progression
    print(f"\n--- DEVICE TIME ANALYSIS ---")

    # Flatten all device times in arrival order
    all_device_times = []
    for msg in messages:
        all_device_times.append(msg['min_device_time'])

    device_times = np.array(all_device_times)
    diffs = np.diff(device_times) * 1000  # to ms

    print(f"Device time differences between consecutive messages:")
    print(f"  Mean: {np.mean(diffs):.2f}ms")
    print(f"  Negative (inversions): {np.sum(diffs < 0)} ({100*np.sum(diffs<0)/len(diffs):.2f}%)")

    negative_diffs = diffs[diffs < 0]
    if len(negative_diffs) > 0:
        print(f"\n  Inversion magnitudes (negative diffs):")
        print(f"    Mean: {-np.mean(negative_diffs):.2f}ms")
        print(f"    Max:  {-np.min(negative_diffs):.2f}ms")
        for p in [50, 90, 95, 99, 100]:
            print(f"    p{p:3d}: {-np.percentile(negative_diffs, 100-p):.2f}ms")

    # Check packets within messages
    print(f"\n--- PACKETS PER MESSAGE ---")
    pkts_per_msg = [len(m['packets']) for m in messages]
    print(f"  Min: {min(pkts_per_msg)}, Max: {max(pkts_per_msg)}, Mean: {np.mean(pkts_per_msg):.1f}")

    # Check if reordering is just 1 sample (3.9µs clock resolution)
    clock_tick_ms = 1000 / DEVICE_CLOCK_HZ  # ~0.0039ms per tick
    print(f"\n--- CLOCK RESOLUTION ---")
    print(f"  Device clock tick: {clock_tick_ms*1000:.2f}µs ({clock_tick_ms:.4f}ms)")
    print(f"  256Hz sample period: {1000/256:.2f}ms")

    # Look at actual raw tick values for inversions
    print(f"\n--- INVERSION ANALYSIS (detailed) ---")
    max_seen = messages[0]['max_device_time']
    inversion_samples = []

    for i, msg in enumerate(messages[1:], 1):
        if msg['min_device_time'] < max_seen:
            diff_ticks = int((max_seen - msg['min_device_time']) * DEVICE_CLOCK_HZ)
            inversion_samples.append(diff_ticks)
        max_seen = max(max_seen, msg['max_device_time'])

    if inversion_samples:
        inv = np.array(inversion_samples)
        print(f"  Inversions in clock ticks (256kHz):")
        unique, counts = np.unique(inv, return_counts=True)
        for u, c in sorted(zip(unique, counts), key=lambda x: -x[1])[:10]:
            pct = 100 * c / len(inv)
            ms = u / DEVICE_CLOCK_HZ * 1000
            print(f"    {u:6d} ticks ({ms:6.2f}ms): {c:5d} times ({pct:5.1f}%)")


def main():
    txt_files = sorted([f for f in os.listdir('.') if f.endswith('.txt') and f != 'prompt.txt'])

    # Analyze a few representative files
    for filename in txt_files[:3]:  # First 3 files
        analyze_file(filename)

    # Also check the anomalous file
    if 'test2_standard2_unknow.txt' in txt_files:
        analyze_file('test2_standard2_unknow.txt')


if __name__ == '__main__':
    main()
