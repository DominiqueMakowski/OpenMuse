"""
Correlate 0x88 packet data with OPTICS16 channels.

GOAL: Determine if 0x88 contains reprocessed/reordered OPTICS data.

APPROACH:
1. Extract both 0x88 and OPTICS16 (0x36) packets from raw data
2. Decode OPTICS16 normally
3. Try multiple decodings of 0x88 (int16, 20-bit packed, etc.)
4. Compute cross-correlation between 0x88 "channels" and known OPTICS channels
5. Look for patterns suggesting channel reordering or processing
"""

import struct
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from OpenMuse.decode import (
    SENSORS,
    PACKET_HEADER_SIZE,
    DEVICE_CLOCK_HZ,
    OPTICS_SCALE,
    _bytes_to_bits,
    _extract_packed_int,
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


def extract_packets_by_type(filepath: str, target_ids: list):
    """Extract packets of specific types from file."""
    messages = load_messages(filepath)
    packets = {pid: [] for pid in target_ids}

    for msg in messages:
        parts = msg.strip().split("\t", 2)
        if len(parts) < 3:
            continue
        ts, uuid, hexstring = parts
        payload = bytes.fromhex(hexstring.strip())

        offset = 0
        while offset < len(payload):
            if offset + PACKET_HEADER_SIZE > len(payload):
                break

            pkt_len = payload[offset]
            if offset + pkt_len > len(payload):
                break

            pkt_bytes = payload[offset : offset + pkt_len]
            pkt_id = pkt_bytes[9]

            if pkt_id in target_ids:
                pkt_index = pkt_bytes[1]
                pkt_time_raw = struct.unpack_from("<I", pkt_bytes, 2)[0]
                pkt_time = pkt_time_raw / DEVICE_CLOCK_HZ
                pkt_data = pkt_bytes[PACKET_HEADER_SIZE:]
                byte_13 = pkt_bytes[13]

                packets[pkt_id].append(
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

    return packets


def decode_optics16(data_bytes: bytes) -> np.ndarray:
    """Decode OPTICS16 data (20-bit packed, 16 channels, 1 sample)."""
    if len(data_bytes) < 40:
        return None

    bits = _bytes_to_bits(data_bytes, 40)
    data = np.zeros(16, dtype=np.float32)

    for ch in range(16):
        bit_start = ch * 20
        int_value = _extract_packed_int(bits, bit_start, 20)
        data[ch] = int_value * OPTICS_SCALE

    return data


def decode_0x88_as_int16(data_bytes: bytes, skip_header: int = 0) -> np.ndarray:
    """Decode 0x88 data as int16 values."""
    data = data_bytes[skip_header:]
    n_values = len(data) // 2
    if n_values == 0:
        return None
    return np.frombuffer(data[: n_values * 2], dtype="<i2").astype(np.float32)


def decode_0x88_as_20bit(data_bytes: bytes, skip_header: int = 0) -> np.ndarray:
    """Decode 0x88 data as 20-bit packed values (like OPTICS)."""
    data = data_bytes[skip_header:]
    n_values = (len(data) * 8) // 20
    if n_values == 0:
        return None

    bits = _bytes_to_bits(data, len(data))
    values = np.zeros(n_values, dtype=np.float32)

    for i in range(n_values):
        bit_start = i * 20
        if bit_start + 20 <= len(bits):
            int_value = _extract_packed_int(bits, bit_start, 20)
            values[i] = int_value * OPTICS_SCALE

    return values


def build_time_series(packets, decode_func, n_channels=None, filter_len=None):
    """Build time series from packets."""
    times = []
    data_list = []

    for pkt in packets:
        # Filter by data length if specified
        if filter_len is not None and pkt["data_len"] != filter_len:
            continue

        decoded = decode_func(pkt["data"])
        if decoded is not None:
            times.append(pkt["pkt_time"])
            if n_channels is not None:
                decoded = decoded[:n_channels]
            data_list.append(decoded)

    if not data_list:
        return None, None

    # Ensure all arrays have same shape
    min_len = min(len(d) for d in data_list)
    data_list = [d[:min_len] for d in data_list]

    times = np.array(times)
    data = np.vstack(data_list)
    return times, data


def resample_to_common_time(ts1, data1, ts2, data2):
    """Resample two time series to common timestamps."""
    t_start = max(ts1.min(), ts2.min())
    t_end = min(ts1.max(), ts2.max())

    # Use the denser time grid
    rate = max(len(ts1) / (ts1.max() - ts1.min()), len(ts2) / (ts2.max() - ts2.min()))
    n_samples = int((t_end - t_start) * rate)
    common_ts = np.linspace(t_start, t_end, n_samples)

    # Interpolate both
    resampled1 = np.zeros((n_samples, data1.shape[1]))
    resampled2 = np.zeros((n_samples, data2.shape[1]))

    for ch in range(data1.shape[1]):
        interp = interp1d(
            ts1, data1[:, ch], kind="linear", bounds_error=False, fill_value=np.nan
        )
        resampled1[:, ch] = interp(common_ts)

    for ch in range(data2.shape[1]):
        interp = interp1d(
            ts2, data2[:, ch], kind="linear", bounds_error=False, fill_value=np.nan
        )
        resampled2[:, ch] = interp(common_ts)

    return common_ts, resampled1, resampled2


def compute_cross_correlation(data1, data2):
    """Compute cross-correlation matrix between two datasets."""
    n_ch1 = data1.shape[1]
    n_ch2 = data2.shape[1]
    corr_matrix = np.zeros((n_ch1, n_ch2))

    for i in range(n_ch1):
        for j in range(n_ch2):
            valid = ~np.isnan(data1[:, i]) & ~np.isnan(data2[:, j])
            if np.sum(valid) > 10:
                corr_matrix[i, j] = np.corrcoef(data1[valid, i], data2[valid, j])[0, 1]
            else:
                corr_matrix[i, j] = np.nan

    return corr_matrix


def analyze_file(filepath: str):
    """Analyze a single file for 0x88 vs OPTICS correlation."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {filepath}")
    print("=" * 80)

    # Extract both packet types
    packets = extract_packets_by_type(filepath, [0x36, 0x88])

    optics_packets = packets[0x36]
    x88_packets = packets[0x88]

    print(f"\nPacket counts:")
    print(f"  OPTICS16 (0x36): {len(optics_packets)}")
    print(f"  Unknown  (0x88): {len(x88_packets)}")

    if len(optics_packets) == 0:
        print("No OPTICS16 packets found!")
        return None

    if len(x88_packets) == 0:
        print("No 0x88 packets found - this is likely OLD firmware")
        return None

    # Analyze 0x88 data lengths
    x88_lens = [p["data_len"] for p in x88_packets]
    common_len = max(set(x88_lens), key=x88_lens.count)
    print(f"\n0x88 packet data length: {common_len} bytes (most common)")

    # Build OPTICS time series
    optics_ts, optics_data = build_time_series(
        optics_packets, decode_optics16, n_channels=16
    )
    print(
        f"\nOPTICS16 time series: {optics_data.shape[0]} samples x {optics_data.shape[1]} channels"
    )
    print(f"  Time range: {optics_ts[0]:.2f} - {optics_ts[-1]:.2f} s")

    # Try different 0x88 decodings
    results = {}

    # Strategy 1: int16, skip first 2 bytes (battery)
    print(f"\n--- Testing 0x88 as int16 (skip 2 bytes for battery) ---")
    x88_ts, x88_int16 = build_time_series(
        x88_packets, lambda d: decode_0x88_as_int16(d, skip_header=2)
    )
    if x88_int16 is not None:
        print(f"  Shape: {x88_int16.shape[0]} samples x {x88_int16.shape[1]} channels")

        # Resample and correlate
        common_ts, optics_resampled, x88_resampled = resample_to_common_time(
            optics_ts, optics_data, x88_ts, x88_int16
        )

        corr = compute_cross_correlation(optics_resampled, x88_resampled)
        results["int16_skip2"] = {
            "correlation": corr,
            "n_channels": x88_int16.shape[1],
            "data": x88_int16,
            "ts": x88_ts,
        }

        # Find best matches
        print(f"\n  Best correlations (OPTICS -> 0x88 int16):")
        for optics_ch in range(16):
            best_x88_ch = np.nanargmax(np.abs(corr[optics_ch, :]))
            best_corr = corr[optics_ch, best_x88_ch]
            if abs(best_corr) > 0.3:  # Only show significant correlations
                print(
                    f"    OPTICS ch{optics_ch:2d} -> 0x88 ch{best_x88_ch:3d} (r={best_corr:+.3f})"
                )

    # Strategy 2: int16, skip first 20 bytes (header)
    print(f"\n--- Testing 0x88 as int16 (skip 20 bytes header) ---")
    x88_ts, x88_int16_20 = build_time_series(
        x88_packets, lambda d: decode_0x88_as_int16(d, skip_header=20)
    )
    if x88_int16_20 is not None:
        print(
            f"  Shape: {x88_int16_20.shape[0]} samples x {x88_int16_20.shape[1]} channels"
        )

        common_ts, optics_resampled, x88_resampled = resample_to_common_time(
            optics_ts, optics_data, x88_ts, x88_int16_20
        )

        corr = compute_cross_correlation(optics_resampled, x88_resampled)
        results["int16_skip20"] = {
            "correlation": corr,
            "n_channels": x88_int16_20.shape[1],
        }

        print(f"\n  Best correlations (OPTICS -> 0x88 int16 skip20):")
        for optics_ch in range(16):
            best_x88_ch = np.nanargmax(np.abs(corr[optics_ch, :]))
            best_corr = corr[optics_ch, best_x88_ch]
            if abs(best_corr) > 0.3:
                print(
                    f"    OPTICS ch{optics_ch:2d} -> 0x88 ch{best_x88_ch:3d} (r={best_corr:+.3f})"
                )

    # Strategy 3: 20-bit packed (skip first 2 bytes)
    print(f"\n--- Testing 0x88 as 20-bit packed (skip 2 bytes) ---")
    x88_ts, x88_20bit = build_time_series(
        x88_packets, lambda d: decode_0x88_as_20bit(d, skip_header=2)
    )
    if x88_20bit is not None:
        print(f"  Shape: {x88_20bit.shape[0]} samples x {x88_20bit.shape[1]} channels")

        common_ts, optics_resampled, x88_resampled = resample_to_common_time(
            optics_ts, optics_data, x88_ts, x88_20bit
        )

        corr = compute_cross_correlation(optics_resampled, x88_resampled)
        results["20bit_skip2"] = {
            "correlation": corr,
            "n_channels": x88_20bit.shape[1],
        }

        print(f"\n  Best correlations (OPTICS -> 0x88 20-bit):")
        for optics_ch in range(16):
            best_x88_ch = np.nanargmax(np.abs(corr[optics_ch, :]))
            best_corr = corr[optics_ch, best_x88_ch]
            if abs(best_corr) > 0.3:
                print(
                    f"    OPTICS ch{optics_ch:2d} -> 0x88 ch{best_x88_ch:3d} (r={best_corr:+.3f})"
                )

    # Plot the correlation heatmaps
    if results:
        fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 8))
        if len(results) == 1:
            axes = [axes]

        for idx, (name, data) in enumerate(results.items()):
            ax = axes[idx]
            corr = data["correlation"]

            # Only show first 50 0x88 channels for readability
            n_show = min(50, corr.shape[1])
            im = ax.imshow(
                corr[:, :n_show], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto"
            )
            ax.set_xlabel(f"0x88 channel (first {n_show})")
            ax.set_ylabel("OPTICS16 channel")
            ax.set_title(f"{name}\n(max |r|={np.nanmax(np.abs(corr)):.3f})")
            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        output_path = filepath.replace(".txt", "_0x88_correlation.png")
        plt.savefig(output_path, dpi=150)
        print(f"\nSaved correlation heatmap to: {output_path}")
        plt.close()

    # Summary: Find the best overall correlations
    print(f"\n{'='*80}")
    print("SUMMARY: Best correlations found across all strategies")
    print("=" * 80)

    best_overall = []
    for strategy_name, data in results.items():
        corr = data["correlation"]
        for optics_ch in range(16):
            for x88_ch in range(corr.shape[1]):
                r = corr[optics_ch, x88_ch]
                if not np.isnan(r) and abs(r) > 0.5:  # Strong correlation
                    best_overall.append(
                        {
                            "strategy": strategy_name,
                            "optics_ch": optics_ch,
                            "x88_ch": x88_ch,
                            "correlation": r,
                        }
                    )

    if best_overall:
        best_overall.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        print(f"\nStrong correlations (|r| > 0.5):")
        for item in best_overall[:20]:  # Top 20
            print(
                f"  {item['strategy']}: OPTICS ch{item['optics_ch']:2d} <-> 0x88 ch{item['x88_ch']:3d} (r={item['correlation']:+.3f})"
            )
    else:
        print("\nNo strong correlations found (|r| > 0.5)")
        print(
            "This suggests 0x88 data may be processed/derived rather than raw sensor data"
        )

    return results


if __name__ == "__main__":
    # Analyze new firmware files
    files = [
        "device2.txt",
        "device3.txt",
    ]

    for filename in files:
        filepath = Path(__file__).parent / filename
        if filepath.exists():
            analyze_file(str(filepath))
