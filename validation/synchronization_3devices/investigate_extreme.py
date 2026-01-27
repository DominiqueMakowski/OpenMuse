"""
Investigate the extreme reordering in test2_standard2_unknow.txt
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


def parse_raw_file(filepath: str) -> list:
    """Parse with full detail."""
    messages = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
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
                    pkt_id = payload[offset + 9]
                    packets.append({
                        'pkt_index': pkt_index,
                        'pkt_time': pkt_time,
                        'pkt_time_raw': pkt_time_raw,
                        'pkt_id': pkt_id,
                    })
                    offset += pkt_len

                if packets:
                    messages.append({
                        'msg_idx': i,
                        'arrival_time': arrival_time,
                        'arrival_ts': arrival_time.timestamp(),
                        'uuid': uuid,
                        'packets': packets,
                        'min_device_time': min(p['pkt_time'] for p in packets),
                        'max_device_time': max(p['pkt_time'] for p in packets),
                    })

            except Exception:
                continue

    return messages


def main():
    filename = 'test2_standard2_unknow.txt'
    print(f"Analyzing extreme reordering in {filename}")
    print("=" * 70)

    messages = parse_raw_file(filename)
    print(f"Total messages: {len(messages)}")

    # Find all cases of extreme reordering (> 100ms)
    max_seen = messages[0]['max_device_time']
    extreme_events = []

    for i, msg in enumerate(messages[1:], 1):
        if msg['min_device_time'] < max_seen:
            reorder_ms = (max_seen - msg['min_device_time']) * 1000
            if reorder_ms > 100:  # Only extreme cases
                extreme_events.append({
                    'msg_idx': i,
                    'arrival_time': msg['arrival_time'],
                    'reorder_ms': reorder_ms,
                    'device_time': msg['min_device_time'],
                    'max_seen_before': max_seen,
                    'uuid': msg['uuid'],
                    'prev_uuid': messages[i-1]['uuid'],
                    'pkt_id': msg['packets'][0]['pkt_id'] if msg['packets'] else None,
                })
        max_seen = max(max_seen, msg['max_device_time'])

    print(f"\nExtreme reordering events (>100ms): {len(extreme_events)}")

    if extreme_events:
        print("\n--- First 20 extreme events ---")
        for e in extreme_events[:20]:
            print(f"  Msg {e['msg_idx']}: {e['reorder_ms']:.0f}ms behind, "
                  f"pkt_id=0x{e['pkt_id']:02x}, "
                  f"UUID={e['uuid'][-8:]}")

        # Analyze what's special about these
        print("\n--- UUID distribution in extreme events ---")
        uuid_counts = {}
        for e in extreme_events:
            uuid = e['uuid']
            uuid_counts[uuid] = uuid_counts.get(uuid, 0) + 1

        for uuid, count in sorted(uuid_counts.items(), key=lambda x: -x[1]):
            print(f"  {uuid}: {count} events ({100*count/len(extreme_events):.1f}%)")

        # Check pkt_id distribution
        print("\n--- Packet ID distribution in extreme events ---")
        pkt_id_counts = {}
        for e in extreme_events:
            pid = e['pkt_id']
            pkt_id_counts[pid] = pkt_id_counts.get(pid, 0) + 1

        for pid, count in sorted(pkt_id_counts.items(), key=lambda x: -x[1]):
            print(f"  0x{pid:02x}: {count} events ({100*count/len(extreme_events):.1f}%)")

        # Check time clustering
        print("\n--- Time distribution of extreme events ---")
        start_time = messages[0]['arrival_ts']
        for e in extreme_events[:10]:
            relative_time = e['arrival_time'].timestamp() - start_time
            print(f"  t={relative_time:.1f}s: {e['reorder_ms']:.0f}ms reorder")

        # Check if it's related to specific UUID sequences
        print("\n--- Context around first extreme event ---")
        first = extreme_events[0]
        idx = first['msg_idx']
        for i in range(max(0, idx-5), min(len(messages), idx+5)):
            msg = messages[i]
            marker = ">>>" if i == idx else "   "
            print(f"{marker} [{i}] UUID={msg['uuid'][-12:]}, "
                  f"device_t={msg['min_device_time']:.3f}s, "
                  f"pkt_id=0x{msg['packets'][0]['pkt_id']:02x}")


if __name__ == '__main__':
    main()
