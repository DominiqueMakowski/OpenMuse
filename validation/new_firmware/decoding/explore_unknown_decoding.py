"""
Explore different decoding strategies for the 0x88 unknown packets.

This script tries various interpretations of the unknown data to identify patterns
that might reveal what sensor data it contains.

Strategies:
1. Int16 big/little endian channels
2. Packed 14-bit values (like EEG)
3. Float16 values
4. Differential/delta encoding
5. Spectral analysis (FFT to find periodic signals)
6. Cross-correlation with known streams (EEG, PPG, ACC)
"""

import sys
import struct
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from scipy import signal
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from OpenMuse.decode import parse_message, SENSORS


def extract_0x88_packets(filepath: str, max_messages: int = None):
    """Extract raw 0x88 packet data from a device file."""
    from datetime import datetime

    content = Path(filepath).read_text()
    lines = content.strip().split("\n")
    if max_messages:
        lines = lines[:max_messages]

    print(f"  Processing {len(lines)} messages...")

    packets = []
    for line in lines:
        try:
            parts = line.strip().split("\t", 2)
            if len(parts) < 3:
                continue
            ts, uuid, hexstring = parts
            payload = bytes.fromhex(hexstring.strip())
        except (ValueError, AttributeError) as e:
            continue

        # Parse packets to find 0x88 tags
        offset = 0
        while offset < len(payload):
            if offset + 14 > len(payload):
                break

            pkt_len = payload[offset]
            if pkt_len < 14 or offset + pkt_len > len(payload):
                break

            pkt_bytes = payload[offset : offset + pkt_len]
            pkt_id = pkt_bytes[9] if len(pkt_bytes) > 9 else 0
            pkt_time_raw = (
                struct.unpack_from("<I", pkt_bytes, 2)[0] if len(pkt_bytes) > 5 else 0
            )
            pkt_time = pkt_time_raw / 256000.0  # Device clock Hz

            if pkt_id == 0x88 and len(pkt_bytes) > 14:
                packets.append(
                    {
                        "time": pkt_time,
                        "data": pkt_bytes[14:],  # Data section after 14-byte header
                        "tag": pkt_id,
                        "full_packet": pkt_bytes,
                    }
                )

            offset += pkt_len

    return packets


def decode_as_int16_le(data: bytes) -> np.ndarray:
    """Decode as little-endian int16 channels."""
    n_values = len(data) // 2
    return np.array(struct.unpack(f"<{n_values}h", data[: n_values * 2]))


def decode_as_int16_be(data: bytes) -> np.ndarray:
    """Decode as big-endian int16 channels."""
    n_values = len(data) // 2
    return np.array(struct.unpack(f">{n_values}h", data[: n_values * 2]))


def decode_as_uint16_le(data: bytes) -> np.ndarray:
    """Decode as little-endian uint16 channels."""
    n_values = len(data) // 2
    return np.array(struct.unpack(f"<{n_values}H", data[: n_values * 2]))


def decode_as_14bit_packed(data: bytes) -> np.ndarray:
    """Decode as packed 14-bit values (like Muse EEG)."""
    # Each 7 bytes contains 4 packed 14-bit values
    values = []
    for i in range(0, len(data) - 6, 7):
        chunk = data[i : i + 7]
        if len(chunk) < 7:
            break
        # Unpack 4 x 14-bit values from 7 bytes
        v0 = ((chunk[0] << 6) | (chunk[1] >> 2)) & 0x3FFF
        v1 = ((chunk[1] << 12) | (chunk[2] << 4) | (chunk[3] >> 4)) & 0x3FFF
        v2 = ((chunk[3] << 10) | (chunk[4] << 2) | (chunk[5] >> 6)) & 0x3FFF
        v3 = ((chunk[5] << 8) | chunk[6]) & 0x3FFF
        values.extend([v0, v1, v2, v3])
    return np.array(values, dtype=np.int16)


def decode_as_float16(data: bytes) -> np.ndarray:
    """Decode as IEEE float16 values."""
    n_values = len(data) // 2
    return np.frombuffer(data[: n_values * 2], dtype=np.float16)


def analyze_channel_statistics(packets, decoder, name):
    """Analyze statistics for each decoded channel."""
    if not packets:
        return None

    # Decode all packets
    decoded = [decoder(p["data"]) for p in packets]

    # Find minimum length (data sections may vary slightly)
    min_len = min(len(d) for d in decoded)
    if min_len == 0:
        return None

    # Stack into matrix (time x channels)
    matrix = np.array([d[:min_len] for d in decoded])

    results = {
        "name": name,
        "n_packets": len(packets),
        "n_channels": min_len,
        "mean": np.mean(matrix, axis=0),
        "std": np.std(matrix, axis=0),
        "min": np.min(matrix, axis=0),
        "max": np.max(matrix, axis=0),
        "matrix": matrix,
    }

    # Find channels with high variance (likely real signal)
    std_threshold = np.median(results["std"]) * 0.5
    results["varying_channels"] = np.where(results["std"] > std_threshold)[0]
    results["constant_channels"] = np.where(results["std"] <= std_threshold)[0]

    return results


def spectral_analysis(matrix, sample_rate=30):
    """Perform spectral analysis on decoded channels."""
    n_packets, n_channels = matrix.shape
    if n_packets < 10:
        return None

    # Find dominant frequencies for each varying channel
    freqs = []
    for ch in range(n_channels):
        f, psd = signal.welch(
            matrix[:, ch], fs=sample_rate, nperseg=min(64, n_packets // 2)
        )
        peak_idx = np.argmax(psd[1:]) + 1  # Skip DC
        freqs.append(
            {"channel": ch, "peak_freq": f[peak_idx], "peak_power": psd[peak_idx]}
        )

    return sorted(freqs, key=lambda x: x["peak_power"], reverse=True)


def cross_correlate_with_known(matrix, known_data, sample_rate=30):
    """Cross-correlate decoded channels with known streams (EEG, PPG)."""
    correlations = []

    # Resample known data to match matrix sample rate if needed
    for ch in range(matrix.shape[1]):
        if len(matrix[:, ch]) < 10 or len(known_data) < 10:
            continue

        # Use overlapping portion
        min_len = min(len(matrix[:, ch]), len(known_data))
        try:
            r, p = pearsonr(matrix[:min_len, ch], known_data[:min_len])
            correlations.append({"channel": ch, "correlation": r, "p_value": p})
        except:
            pass

    return sorted(correlations, key=lambda x: abs(x["correlation"]), reverse=True)


def plot_decoded_channels(results, max_channels=20):
    """Plot the decoded channels over time."""
    if results is None or results["matrix"].shape[0] < 2:
        return

    matrix = results["matrix"]
    varying = results["varying_channels"][:max_channels]

    if len(varying) == 0:
        print("  No varying channels found")
        return

    fig, axes = plt.subplots(
        len(varying), 1, figsize=(12, max(4, len(varying) * 0.8)), sharex=True
    )
    if len(varying) == 1:
        axes = [axes]

    for i, ch in enumerate(varying):
        axes[i].plot(matrix[:, ch], linewidth=0.5)
        axes[i].set_ylabel(f"Ch {ch}")
        axes[i].set_title(f'Std: {results["std"][ch]:.1f}', fontsize=8)

    axes[-1].set_xlabel("Packet Index")
    plt.suptitle(f'{results["name"]}: Varying Channels')
    plt.tight_layout()
    plt.savefig(f'decoded_{results["name"].replace(" ", "_")}.png', dpi=100)
    plt.close()


def main():
    # Find device files with 0x88 packets
    files = [
        "device2.txt",
        "device3.txt",
        "device3b.txt",
        "device3c.txt",
        "device3d.txt",
    ]

    # Use device3.txt as primary (longest recording with 0x88)
    primary_file = "device3.txt"
    if not Path(primary_file).exists():
        print(f"File {primary_file} not found")
        return

    print(f"Extracting 0x88 packets from {primary_file}...")
    packets = extract_0x88_packets(
        primary_file, max_messages=None
    )  # Process ALL messages

    if not packets:
        print("No 0x88 packets found")
        return

    # Filter to only 0x88 tag (should be all of them for Unknown type)
    packets_0x88 = [p for p in packets if p.get("tag") == 0x88 or "data" in p]
    print(f"Found {len(packets_0x88)} 0x88 packets")

    if not packets_0x88:
        return

    # Show sample packet structure
    sample = packets_0x88[0]["data"]
    print(f"\nSample packet length: {len(sample)} bytes")
    print(f"First 40 bytes (hex): {sample[:40].hex()}")

    # Skip first 2 bytes (battery) for analysis
    packets_data_only = [
        {"time": p["time"], "data": p["data"][2:]} for p in packets_0x88
    ]

    print("\n" + "=" * 80)
    print("TESTING DIFFERENT DECODING STRATEGIES")
    print("=" * 80)

    # Test various decoders
    decoders = [
        (decode_as_int16_le, "Int16 LE (skip 2)"),
        (decode_as_int16_be, "Int16 BE (skip 2)"),
        (decode_as_uint16_le, "UInt16 LE (skip 2)"),
        (decode_as_14bit_packed, "14-bit Packed (skip 2)"),
    ]

    for decoder, name in decoders:
        print(f"\n--- {name} ---")
        results = analyze_channel_statistics(packets_data_only, decoder, name)

        if results is None:
            print("  Failed to decode")
            continue

        print(f"  Channels: {results['n_channels']}")
        print(f"  Varying channels: {len(results['varying_channels'])}")
        print(f"  Constant channels: {len(results['constant_channels'])}")

        # Show top varying channels
        if len(results["varying_channels"]) > 0:
            top_varying = results["varying_channels"][:10]
            print(f"  Top varying channels (by std):")
            for ch in top_varying:
                print(
                    f"    Ch {ch}: mean={results['mean'][ch]:.1f}, std={results['std'][ch]:.1f}, "
                    f"range=[{results['min'][ch]:.0f}, {results['max'][ch]:.0f}]"
                )

        # Plot if we have varying channels
        if len(results["varying_channels"]) >= 5:
            plot_decoded_channels(results, max_channels=15)
            print(f"  Saved plot: decoded_{name.replace(' ', '_')}.png")

    # Special analysis: Look for patterns in the "constant" portion
    print("\n" + "=" * 80)
    print("ANALYZING CONSTANT/HEADER BYTES")
    print("=" * 80)

    if len(packets_0x88) > 10:
        # Analyze first 20 bytes for patterns
        first_20 = np.array([list(p["data"][:20]) for p in packets_0x88[:100]])

        print("\nByte-level analysis (first 20 bytes):")
        for i in range(min(20, first_20.shape[1])):
            unique_vals = np.unique(first_20[:, i])
            if len(unique_vals) <= 5:
                print(f"  Byte {i}: CONSTANT-ish: {[hex(v) for v in unique_vals]}")
            else:
                print(
                    f"  Byte {i}: VARIABLE, range [{first_20[:, i].min()}, {first_20[:, i].max()}], "
                    f"std={first_20[:, i].std():.1f}"
                )

    # Look for repeating patterns
    print("\n" + "=" * 80)
    print("LOOKING FOR REPEATING PATTERNS")
    print("=" * 80)

    # Check if there are repeating byte patterns (could indicate packed samples)
    sample_data = packets_0x88[0]["data"][2:]  # Skip battery

    # Try to find period
    for period in [2, 3, 4, 6, 7, 8, 12, 14, 16]:
        if len(sample_data) >= period * 3:
            chunks = [
                sample_data[i : i + period]
                for i in range(0, len(sample_data) - period, period)
            ]
            # Check variance between chunks
            if len(chunks) >= 3:
                chunk_array = np.array(
                    [list(c) for c in chunks[:20] if len(c) == period]
                )
                if len(chunk_array) > 2:
                    std_per_byte = np.std(chunk_array, axis=0)
                    if np.all(std_per_byte < 5):
                        print(
                            f"  Period {period}: Low variance - might be repeating structure!"
                        )

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        """
Based on analysis:
- First 2 bytes: Battery percentage (confirmed)
- Bytes 2-17: Likely header/config (several constant values)
- Remaining bytes: Unknown payload

Most likely interpretations:
1. Extended sensor data (higher-rate PPG, additional sensors)
2. Signal quality metrics per channel
3. Device diagnostics/telemetry
4. Raw ADC values from sensors

Next steps:
- Record simultaneously with external sensors for cross-correlation
- Analyze during specific activities (eye blinks, jaw clench, walking)
- Compare patterns between devices/firmware versions
"""
    )


if __name__ == "__main__":
    main()
