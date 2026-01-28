"""
Analyze raw BLE packet timing from recorded .txt files.

This script examines:
1. Out-of-order packet arrival (message reordering)
2. Timing inversions within device timestamps
3. Inter-packet arrival time distribution
4. Reordering magnitude (how far back in time do late packets go?)

Results help determine optimal buffer/flush parameters for stream.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import struct

# Add parent to path to import decode module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from collections import defaultdict

# Constants from decode.py
DEVICE_CLOCK_HZ = 256000.0
PACKET_HEADER_SIZE = 14


def parse_raw_file(filepath: str, max_messages: int = None) -> List[Dict]:
    """Parse raw BLE messages from a .txt file."""
    messages = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_messages and i >= max_messages:
                break

            line = line.strip()
            if not line:
                continue

            try:
                parts = line.split('\t')
                if len(parts) < 3:
                    continue

                ts_str, uuid, hexdata = parts[0], parts[1], parts[2]

                # Parse arrival timestamp
                arrival_time = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))

                # Parse payload
                payload = bytes.fromhex(hexdata.strip())

                # Extract device timestamps from packets in payload
                device_times = []
                pkt_indices = []
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

                    device_times.append(pkt_time)
                    pkt_indices.append(pkt_index)

                    offset += pkt_len

                if device_times:
                    messages.append({
                        'arrival_time': arrival_time,
                        'arrival_ts': arrival_time.timestamp(),
                        'device_times': device_times,
                        'pkt_indices': pkt_indices,
                        'min_device_time': min(device_times),
                        'max_device_time': max(device_times),
                        'n_packets': len(device_times),
                    })

            except Exception as e:
                continue

    return messages


def analyze_reordering(messages: List[Dict]) -> Dict:
    """Analyze packet reordering statistics."""

    if len(messages) < 2:
        return {}

    # Track the maximum device time seen so far
    max_device_time_seen = messages[0]['max_device_time']

    # Statistics
    total_messages = len(messages)
    out_of_order_messages = 0
    reorder_magnitudes_ms = []  # How far back (in ms) did the late packet go?

    arrival_intervals_ms = []
    device_time_gaps_ms = []

    prev_arrival = messages[0]['arrival_ts']
    prev_device_time = messages[0]['max_device_time']

    for i, msg in enumerate(messages[1:], 1):
        # Check arrival interval
        arrival_interval = (msg['arrival_ts'] - prev_arrival) * 1000  # ms
        arrival_intervals_ms.append(arrival_interval)

        # Check device time progression
        device_time_gap = (msg['min_device_time'] - prev_device_time) * 1000  # ms
        device_time_gaps_ms.append(device_time_gap)

        # Check for out-of-order: does this message have device times earlier than what we've seen?
        if msg['min_device_time'] < max_device_time_seen:
            out_of_order_messages += 1
            # How far back in time?
            reorder_mag = (max_device_time_seen - msg['min_device_time']) * 1000  # ms
            reorder_magnitudes_ms.append(reorder_mag)

        # Update tracking
        max_device_time_seen = max(max_device_time_seen, msg['max_device_time'])
        prev_arrival = msg['arrival_ts']
        prev_device_time = msg['max_device_time']

    arrival_intervals = np.array(arrival_intervals_ms)
    device_gaps = np.array(device_time_gaps_ms)
    reorder_mags = np.array(reorder_magnitudes_ms) if reorder_magnitudes_ms else np.array([0])

    return {
        'total_messages': total_messages,
        'out_of_order_messages': out_of_order_messages,
        'out_of_order_rate': 100 * out_of_order_messages / total_messages,
        'reorder_magnitude_mean_ms': np.mean(reorder_mags) if len(reorder_mags) > 0 else 0,
        'reorder_magnitude_max_ms': np.max(reorder_mags) if len(reorder_mags) > 0 else 0,
        'reorder_magnitude_p95_ms': np.percentile(reorder_mags, 95) if len(reorder_mags) > 0 else 0,
        'reorder_magnitude_p99_ms': np.percentile(reorder_mags, 99) if len(reorder_mags) > 0 else 0,
        'arrival_interval_mean_ms': np.mean(arrival_intervals),
        'arrival_interval_std_ms': np.std(arrival_intervals),
        'arrival_interval_max_ms': np.max(arrival_intervals),
        'device_gap_mean_ms': np.mean(device_gaps),
        'device_gap_negative_count': np.sum(device_gaps < 0),
        'device_gap_negative_rate': 100 * np.sum(device_gaps < 0) / len(device_gaps),
    }


def analyze_timing_inversions(messages: List[Dict]) -> Dict:
    """Analyze device timestamp inversions (non-monotonic device times)."""

    all_device_times = []
    all_pkt_indices = []

    for msg in messages:
        all_device_times.extend(msg['device_times'])
        all_pkt_indices.extend(msg['pkt_indices'])

    device_times = np.array(all_device_times)

    if len(device_times) < 2:
        return {}

    # Check for inversions
    diffs = np.diff(device_times)
    inversions = diffs < 0

    inversion_magnitudes_ms = -diffs[inversions] * 1000  # Convert to positive ms

    return {
        'total_packets': len(device_times),
        'inversions_count': np.sum(inversions),
        'inversions_rate': 100 * np.sum(inversions) / len(diffs),
        'inversion_magnitude_mean_ms': np.mean(inversion_magnitudes_ms) if len(inversion_magnitudes_ms) > 0 else 0,
        'inversion_magnitude_max_ms': np.max(inversion_magnitudes_ms) if len(inversion_magnitudes_ms) > 0 else 0,
        'inversion_magnitude_p95_ms': np.percentile(inversion_magnitudes_ms, 95) if len(inversion_magnitudes_ms) > 0 else 0,
    }


def main():
    """Analyze all .txt files in the current directory."""

    txt_files = sorted([f for f in os.listdir('.') if f.endswith('.txt') and f != 'prompt.txt'])

    if not txt_files:
        print("No .txt files found!")
        return

    print(f"Found {len(txt_files)} raw data files")
    print("=" * 80)

    all_results = []

    for filename in txt_files:
        print(f"\nAnalyzing: {filename}")
        print("-" * 60)

        try:
            messages = parse_raw_file(filename)

            if len(messages) < 10:
                print(f"  Too few messages ({len(messages)}), skipping")
                continue

            # Duration
            duration_s = messages[-1]['arrival_ts'] - messages[0]['arrival_ts']

            reorder_stats = analyze_reordering(messages)
            inversion_stats = analyze_timing_inversions(messages)

            result = {
                'file': filename,
                'duration_s': duration_s,
                'n_messages': len(messages),
                **reorder_stats,
                **inversion_stats,
            }
            all_results.append(result)

            print(f"  Duration: {duration_s:.1f}s, Messages: {len(messages)}")
            print(f"  Out-of-order messages: {reorder_stats['out_of_order_messages']} ({reorder_stats['out_of_order_rate']:.2f}%)")
            print(f"  Reorder magnitude: mean={reorder_stats['reorder_magnitude_mean_ms']:.1f}ms, "
                  f"max={reorder_stats['reorder_magnitude_max_ms']:.1f}ms, "
                  f"p95={reorder_stats['reorder_magnitude_p95_ms']:.1f}ms, "
                  f"p99={reorder_stats['reorder_magnitude_p99_ms']:.1f}ms")
            print(f"  Device time inversions: {inversion_stats['inversions_count']} ({inversion_stats['inversions_rate']:.2f}%)")
            print(f"  Arrival interval: mean={reorder_stats['arrival_interval_mean_ms']:.1f}ms, "
                  f"std={reorder_stats['arrival_interval_std_ms']:.1f}ms, "
                  f"max={reorder_stats['arrival_interval_max_ms']:.1f}ms")

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    if all_results:
        print("\n" + "=" * 80)
        print("SUMMARY ACROSS ALL FILES")
        print("=" * 80)

        df = pd.DataFrame(all_results)

        print(f"\nTotal recordings analyzed: {len(df)}")
        print(f"Total duration: {df['duration_s'].sum()/60:.1f} minutes")
        print(f"Total messages: {df['n_messages'].sum():,}")

        print("\n--- OUT-OF-ORDER MESSAGE STATISTICS ---")
        print(f"Average out-of-order rate: {df['out_of_order_rate'].mean():.2f}%")
        print(f"Max out-of-order rate: {df['out_of_order_rate'].max():.2f}%")

        print("\n--- REORDER MAGNITUDE (how far back late messages go) ---")
        print(f"Average mean magnitude: {df['reorder_magnitude_mean_ms'].mean():.1f}ms")
        print(f"Average max magnitude: {df['reorder_magnitude_max_ms'].mean():.1f}ms")
        print(f"Overall max magnitude: {df['reorder_magnitude_max_ms'].max():.1f}ms")
        print(f"Average p95 magnitude: {df['reorder_magnitude_p95_ms'].mean():.1f}ms")
        print(f"Average p99 magnitude: {df['reorder_magnitude_p99_ms'].mean():.1f}ms")

        print("\n--- DEVICE TIMESTAMP INVERSIONS ---")
        print(f"Average inversion rate: {df['inversions_rate'].mean():.2f}%")

        print("\n--- BUFFER SIZE RECOMMENDATION ---")
        p99_mag = df['reorder_magnitude_p99_ms'].max()
        max_mag = df['reorder_magnitude_max_ms'].max()
        print(f"To capture 99% of reordered messages: buffer >= {p99_mag:.0f}ms")
        print(f"To capture 100% of reordered messages: buffer >= {max_mag:.0f}ms")
        print(f"Current buffer (150ms): would capture messages reordered up to 150ms")

        # Check what percentage of reordering would be captured at different buffer sizes
        print("\n--- BUFFER SIZE TRADE-OFFS ---")
        for buffer_ms in [50, 75, 100, 125, 150, 175, 200, 250]:
            # This is approximate - we'd need the full distribution
            coverage = f"covers reordering up to {buffer_ms}ms"
            print(f"  {buffer_ms}ms buffer: {coverage}")


if __name__ == '__main__':
    main()
