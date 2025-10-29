"""
Muse BLE Message Parser
============================================

Implementation that follows the actual message structure:
MESSAGE → PACKET → DATA SUBPACKETS

Message Structure:
------------------
Each BLE MESSAGE contains one or more PACKETS. Each PACKET has a 14-byte header
followed by a data section containing multiple DATA SUBPACKETS:

MESSAGE (BLE transmission with timestamp)
  └─ PACKET (14-byte header + data section)
       ├─ First Subpacket: Raw sensor data (no TAG, no header)
       └─ Additional Subpackets: [TAG (1 byte)][Header (4 bytes)][Data (variable)]
            ├─ TAG: Sensor type identifier (e.g., 0x47=ACCGYRO, 0x12=EEG8)
            ├─ subpkt_index: Per-sensor-type sequence counter (0-255, wraps)
            ├─ Unknown bytes: 3 metadata bytes (purpose unknown)
            └─ Sensor data: Variable length depending on sensor type

Timestamp Calculation & Device Timing:
---------------------------------------
Device timestamps (pkt_time) are derived from a 256 kHz hardware clock with 3.906 µs resolution.
Multiple packets often share identical pkt_time values (~11-30% are duplicates).

Timestamp generation per message:
  1. Sort packets by (pkt_time, pkt_index, subpkt_index)
     - pkt_index: Packet sequence counter (0-255), ensures correct ordering of duplicates
     - Analysis: 100% sequential in duplicate groups (1871/1871 tested)
  2. Use first packet's pkt_time as anchor
  3. Generate uniform timestamps: anchor + (sample_index / sampling_rate)

Hardware Timing Artifacts:
  - ~4% of pkt_time values have timing inversions (timestamps go backwards)
  - pkt_index remains sequential (100% accurate for packet order)
  - Inversions likely due to async sensor buffering and clock jitter
  - Final monotonicity ensured by stream.py buffering/sorting before LSL output

"""

import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from datetime import datetime
import numpy as np
import pandas as pd


# Protocol constants
PACKET_HEADER_SIZE = 14  # Size of packet header in bytes
SUBPACKET_HEADER_SIZE = 5  # Size of subpacket header (TAG + 4 bytes)
DEVICE_CLOCK_HZ = 256000.0  # Device clock frequency: 256 kHz

# Sensor configuration mapping
# Maps TAG byte to sensor type, channels, samples, sampling rate, and data length
SENSORS = {
    0x11: {
        "type": "EEG",
        "n_channels": 4,
        "n_samples": 4,
        "rate": 256.0,
        "data_len": 28,
    },
    0x12: {
        "type": "EEG",
        "n_channels": 8,
        "n_samples": 2,
        "rate": 256.0,
        "data_len": 28,
    },
    0x34: {
        "type": "OPTICS",
        "n_channels": 4,
        "n_samples": 3,
        "rate": 64.0,
        "data_len": 30,
    },
    0x35: {
        "type": "OPTICS",
        "n_channels": 8,
        "n_samples": 2,
        "rate": 64.0,
        "data_len": 40,
    },
    0x36: {
        "type": "OPTICS",
        "n_channels": 16,
        "n_samples": 1,
        "rate": 64.0,
        "data_len": 40,
    },
    0x47: {
        "type": "ACCGYRO",
        "n_channels": 6,
        "n_samples": 3,
        "rate": 52.0,
        "data_len": 36,
    },
    0x53: {
        "type": "Unknown",
        "n_channels": 0,
        "n_samples": 0,
        "rate": 0.0,
        "data_len": 24,
    },
    0x98: {
        "type": "BATTERY",
        "n_channels": 1,
        "n_samples": 1,
        "rate": 1.0,
        "data_len": 20,
    },
}

# Scaling constants
ACC_SCALE = 0.0000610352
GYRO_SCALE = -0.0074768
OPTICS_SCALE = 1.0 / 32768.0
EEG_SCALE = 1450.0 / 16383.0

# Standard channel label definitions shared across modules
EEG_CHANNELS = (
    "EEG_TP9",
    "EEG_AF7",
    "EEG_AF8",
    "EEG_TP10",
    "AUX_1",
    "AUX_2",
    "AUX_3",
    "AUX_4",
)

ACCGYRO_CHANNELS = (
    "ACC_X",
    "ACC_Y",
    "ACC_Z",
    "GYRO_X",
    "GYRO_Y",
    "GYRO_Z",
)

OPTICS_CHANNELS = (
    "OPTICS_LO_NIR",
    "OPTICS_RO_NIR",
    "OPTICS_LO_IR",
    "OPTICS_RO_IR",
    "OPTICS_LI_NIR",
    "OPTICS_RI_NIR",
    "OPTICS_LI_IR",
    "OPTICS_RI_IR",
    "OPTICS_LO_RED",
    "OPTICS_RO_RED",
    "OPTICS_LO_AMB",
    "OPTICS_RO_AMB",
    "OPTICS_LI_RED",
    "OPTICS_RI_RED",
    "OPTICS_LI_AMB",
    "OPTICS_RI_AMB",
)

BATTERY_CHANNELS = ("BATTERY_PERCENT",)


def _select_eeg_channels(count: int) -> List[str]:
    if count <= len(EEG_CHANNELS):
        return list(EEG_CHANNELS[:count])
    return [f"EEG_{i+1:02d}" for i in range(count)]


_OPTICS_INDEXES = {
    4: (4, 5, 6, 7),
    8: tuple(range(8)),
    16: tuple(range(16)),
}


def _select_optics_channels(count: int) -> List[str]:
    indices = _OPTICS_INDEXES.get(count)
    if indices is not None:
        return [OPTICS_CHANNELS[i] for i in indices]
    return [f"OPTICS_{i+1:02d}" for i in range(count)]


def parse_message(message: str) -> Dict[str, List[Dict]]:
    """
    Parse a BLE message into structured data.

    This parser follows the actual message structure:
    MESSAGE → PACKET (14-byte header) → DATA SUBPACKETS (TAG + 4-byte header + data)

    Parameters:
    -----------
    message : str
        Tab-separated string: timestamp, UUID, hex-encoded payload

    Returns:
    --------
    dict : Dictionary with sensor types as keys, lists of subpacket dicts as values
        Keys: "EEG", "ACCGYRO", "OPTICS", "BATTERY", "Unknown"
        Each list contains subpacket dictionaries with decoded sensor data
    """

    # Parse message line
    try:
        ts, uuid, hexstring = message.strip().split("\t", 2)
        message_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        payload = bytes.fromhex(hexstring.strip())
    except (ValueError, AttributeError):
        # Malformed message
        return _empty_result()

    # Initialize result dictionary for parsed subpackets
    parsed_data = _empty_result()

    # Step 1: Parse all packets in the message
    packets = _parse_packets(payload, message_time, uuid)

    # Step 2: Parse data subpackets within each packet
    for pkt in packets:
        subpackets = _parse_data_subpackets(pkt)

        # Step 3: Decode data for each subpacket
        for subpkt in subpackets:
            sensor_type = subpkt["sensor_type"]
            decoded = _decode_subpacket_data(subpkt)

            if decoded is not None:
                # Add packet-level metadata to decoded subpacket
                decoded.update(
                    {
                        "pkt_index": pkt["pkt_index"],
                        "pkt_time": pkt["pkt_time"],
                        "pkt_time_raw": pkt["pkt_time_raw"],
                        "message_time": pkt["message_time"],
                        "message_uuid": pkt["message_uuid"],
                    }
                )

                parsed_data[sensor_type].append(decoded)

    # Return raw subpackets for global timestamping
    return parsed_data


def _empty_result() -> Dict[str, List]:
    """Create empty result dictionary with all sensor types."""
    return {
        "EEG": [],
        "ACCGYRO": [],
        "OPTICS": [],
        "BATTERY": [],
        "Unknown": [],
    }


def _empty_result_arrays() -> Dict[str, np.ndarray]:
    """Create empty result dictionary with numpy arrays for all sensor types."""
    return {
        "EEG": np.empty((0, 0)),
        "ACCGYRO": np.empty((0, 0)),
        "OPTICS": np.empty((0, 0)),
        "BATTERY": np.empty((0, 0)),
        "Unknown": np.empty((0, 0)),
    }


def _parse_packets(payload: bytes, message_time: datetime, uuid: str) -> List[Dict]:
    """
    Parse all packets from payload based on declared length.

    Returns list of packet dicts with header information and data section.
    """
    packets = []
    offset = 0

    while offset < len(payload):
        # Need at least PACKET_HEADER_SIZE bytes for packet header
        if offset + PACKET_HEADER_SIZE > len(payload):
            break

        # Read declared packet length
        pkt_len = payload[offset]

        # Validate we have enough bytes
        if offset + pkt_len > len(payload):
            break

        # Extract full packet
        pkt_bytes = payload[offset : offset + pkt_len]

        # Parse packet header (PACKET_HEADER_SIZE bytes)
        pkt_index = pkt_bytes[1]
        pkt_time_raw = struct.unpack_from("<I", pkt_bytes, 2)[0]
        pkt_time = (
            pkt_time_raw / DEVICE_CLOCK_HZ
        )  # Convert device clock ticks to seconds
        pkt_unknown1 = pkt_bytes[6:9]
        pkt_id = pkt_bytes[9]
        pkt_unknown2 = pkt_bytes[10:13]
        byte_13 = pkt_bytes[13]

        # Decode pkt_id to get sensor type
        pkt_config = SENSORS.get(pkt_id)
        pkt_type = pkt_config["type"] if pkt_config else None

        # Validate packet
        pkt_valid = (
            pkt_type is not None
            and byte_13 == 0
            and pkt_len == len(pkt_bytes)
            and pkt_len >= PACKET_HEADER_SIZE
        )

        # Extract data section (everything after header)
        pkt_data = (
            pkt_bytes[PACKET_HEADER_SIZE:]
            if len(pkt_bytes) > PACKET_HEADER_SIZE
            else b""
        )

        packets.append(
            {
                "pkt_len": pkt_len,
                "pkt_index": pkt_index,
                "pkt_time": pkt_time,
                "pkt_time_raw": pkt_time_raw,
                "pkt_unknown1": pkt_unknown1,
                "pkt_id": pkt_id,
                "pkt_type": pkt_type,
                "pkt_unknown2": pkt_unknown2,
                "pkt_valid": pkt_valid,
                "pkt_data": pkt_data,
                "message_time": message_time,
                "message_uuid": uuid,
            }
        )

        offset += pkt_len

    return packets


def _parse_data_subpackets(pkt: Dict) -> List[Dict]:
    """
    Parse all data subpackets within a packet's data section.

    The packet data section contains:
    1. Primary data (sensor type matching pkt_type, NO TAG, NO 4-byte header, just raw data)
    2. Additional subpackets (each with [TAG][4-byte header][data])

    Returns list of subpacket dicts with raw bytes and metadata.
    """
    subpackets = []
    pkt_data = pkt["pkt_data"]
    offset = 0

    # Step 1: Parse first data subpacket (no TAG, no header, just raw sensor data)
    # This is always the sensor type indicated by the packet header
    if pkt["pkt_valid"] and pkt["pkt_type"] is not None:
        first_type = pkt["pkt_type"]
        # Get data length from SENSORS config using pkt_id as TAG
        first_data_len = SENSORS.get(pkt["pkt_id"], {}).get("data_len", 0)

        if first_data_len > 0 and offset + first_data_len <= len(pkt_data):
            # Extract first subpacket data bytes (no TAG, no header)
            first_data_bytes = pkt_data[offset : offset + first_data_len]

            # Add first subpacket (no index or unknown bytes available)
            subpackets.append(
                {
                    "sensor_type": first_type,
                    "tag_byte": pkt["pkt_id"],  # Use packet ID as tag for config lookup
                    "subpkt_index": None,  # First subpacket has no index in data section
                    "subpkt_unknown": None,  # First subpacket has no unknown bytes
                    "data_bytes": first_data_bytes,
                }
            )

            # Move offset past first data
            offset += first_data_len

    # Step 2: Parse all additional subpackets (each with TAG + 4-byte header + data)
    while offset < len(pkt_data):
        # Check if we have enough bytes for TAG + header
        if offset + SUBPACKET_HEADER_SIZE > len(pkt_data):
            break

        tag_byte = pkt_data[offset]

        # Validate TAG (must be a known sensor TAG)
        if tag_byte not in SENSORS:
            break

        sensor_type = SENSORS[tag_byte]["type"]

        # Parse 4-byte subpacket header
        subpkt_index = pkt_data[offset + 1]
        subpkt_unknown = pkt_data[offset + 2 : offset + SUBPACKET_HEADER_SIZE]

        # Get data length from SENSORS config
        data_len = SENSORS[tag_byte].get("data_len", 0)

        if data_len == 0:
            # Unknown data length, skip this TAG
            break

        # Check if we have enough bytes for full subpacket
        if offset + SUBPACKET_HEADER_SIZE + data_len > len(pkt_data):
            break

        # Extract data bytes
        data_bytes = pkt_data[
            offset + SUBPACKET_HEADER_SIZE : offset + SUBPACKET_HEADER_SIZE + data_len
        ]

        subpackets.append(
            {
                "sensor_type": sensor_type,
                "tag_byte": tag_byte,
                "subpkt_index": subpkt_index,
                "subpkt_unknown": subpkt_unknown,
                "data_bytes": data_bytes,
            }
        )

        # Move to next subpacket
        offset += SUBPACKET_HEADER_SIZE + data_len

    return subpackets


def _decode_subpacket_data(subpkt: Dict) -> Optional[Dict]:
    """
    Decode data bytes based on TAG byte configuration.

    Returns dict with:
        - data: np.ndarray with decoded values (no timestamps)
        - subpkt_index: index value (None for first subpacket)
        - subpkt_unknown: metadata bytes (None for first subpacket)
        - sensor_type: type string
        - n_channels: number of channels
        - n_samples: number of samples

    Returns None if sensor type is not decodable (e.g., "Unknown").
    """
    sensor_type = subpkt["sensor_type"]
    data_bytes = subpkt["data_bytes"]
    tag_byte = subpkt["tag_byte"]

    # Skip Unknown type (no decoder yet)
    if sensor_type == "Unknown":
        return None

    # Get configuration from TAG (cache lookup to avoid redundant dictionary access)
    config = SENSORS.get(tag_byte)
    if config is None:
        return None

    n_channels = config["n_channels"]
    n_samples = config["n_samples"]

    # Decode based on sensor type
    if sensor_type == "ACCGYRO":
        data = _decode_accgyro_data(data_bytes)
    elif sensor_type == "BATTERY":
        data = _decode_battery_data(data_bytes)
    elif sensor_type == "EEG":
        data = _decode_eeg_data(data_bytes, n_channels)
    elif sensor_type == "OPTICS":
        data = _decode_optics_data(data_bytes, n_channels)
    else:
        return None

    if data is None:
        return None

    return {
        "sensor_type": sensor_type,
        "tag_byte": tag_byte,  # Include tag_byte for timestamp calculation
        "subpkt_index": subpkt["subpkt_index"],
        "subpkt_unknown": subpkt["subpkt_unknown"],
        "data": data,
        "n_channels": n_channels,
        "n_samples": n_samples,
    }


def _decode_accgyro_data(data_bytes: bytes) -> Optional[np.ndarray]:
    """
    Decode ACCGYRO data (36 bytes → 3 samples × 6 channels).

    Returns: np.ndarray shape (3, 6) with columns [ACC_X, ACC_Y, ACC_Z, GYRO_X, GYRO_Y, GYRO_Z]
    """
    if len(data_bytes) < 36:
        return None

    # Parse 18 int16 values, reshape to 3 samples × 6 channels
    data = np.frombuffer(data_bytes[:36], dtype="<i2", count=18).reshape(3, 6)
    data = data.astype(np.float32)

    # Apply scaling
    data[:, 0:3] *= ACC_SCALE  # Accelerometer
    data[:, 3:6] *= GYRO_SCALE  # Gyroscope

    return data


def _decode_battery_data(data_bytes: bytes) -> Optional[np.ndarray]:
    """
    Decode Battery data (first 2 bytes = SOC).

    Returns: np.ndarray shape (1,) with battery percentage
    """
    if len(data_bytes) < 2:
        return None

    raw_soc = struct.unpack("<H", data_bytes[0:2])[0]
    battery_percent = raw_soc / 256.0

    return np.array([battery_percent], dtype=np.float32)


def _bytes_to_bits(data_bytes: bytes, n_bytes: int) -> List[int]:
    """
    Convert bytes to bit array (LSB first).

    Parameters:
    -----------
    data_bytes : bytes
        Input bytes to convert
    n_bytes : int
        Number of bytes to process

    Returns:
    --------
    list[int] : List of bits (0 or 1), LSB first
    """
    bits = []
    for byte in data_bytes[:n_bytes]:
        for bit_pos in range(8):
            bits.append((byte >> bit_pos) & 1)
    return bits


def _extract_packed_int(bits: List[int], bit_start: int, bit_width: int) -> int:
    """
    Extract an integer from a bit array.

    Parameters:
    -----------
    bits : list[int]
        Bit array (0 or 1 values)
    bit_start : int
        Starting bit position
    bit_width : int
        Number of bits to extract

    Returns:
    --------
    int : Extracted integer value
    """
    int_value = 0
    for bit_idx in range(bit_width):
        if bits[bit_start + bit_idx]:
            int_value |= 1 << bit_idx
    return int_value


def _decode_eeg_data(data_bytes: bytes, n_channels: int) -> Optional[np.ndarray]:
    """
    Decode EEG data (14-bit packed values).

    EEG4: 4 samples × 4 channels = 28 bytes
    EEG8: 2 samples × 8 channels = 28 bytes

    Returns: np.ndarray shape (n_samples, n_channels)
    """
    if len(data_bytes) < 28:
        return None

    # Determine samples per packet
    n_samples = 4 if n_channels == 4 else 2

    # Convert bytes to bit array (LSB first)
    bits = _bytes_to_bits(data_bytes, 28)

    # Parse 14-bit values
    data = np.zeros((n_samples, n_channels), dtype=np.float32)

    for sample_idx in range(n_samples):
        for channel_idx in range(n_channels):
            bit_start = (sample_idx * n_channels + channel_idx) * 14

            # Extract 14 bits and scale to microvolts
            int_value = _extract_packed_int(bits, bit_start, 14)
            data[sample_idx, channel_idx] = int_value * EEG_SCALE

    return data


def _decode_optics_data(data_bytes: bytes, n_channels: int) -> Optional[np.ndarray]:
    """
    Decode Optics data (20-bit packed values).

    Sensor Positioning:
    -------------------
    Sensors are positioned as: Left Outer (LO) → Left Inner (LI) → Right Inner (RI) → Right Outer (RO)

    Signal Types:
    -------------
    - NIR: 730nm wavelength (near-infrared)
    - IR: 850nm wavelength (infrared)
    - RED: Red wavelength
    - AMB: Ambient light

    Channel Configurations:
    -----------------------
    Optics4:  3 samples × 4 channels = 30 bytes
              Channels: LI_NIR, RI_NIR, LI_IR, RI_IR (inner sensors only)

    Optics8:  2 samples × 8 channels = 40 bytes
              Channels: LO_NIR, RO_NIR, LO_IR, RO_IR, LI_NIR, RI_NIR, LI_IR, RI_IR

    Optics16: 1 sample × 16 channels = 40 bytes
              Channels: All 4 sensors × 4 wavelengths (NIR, IR, RED, AMB)

    Returns: np.ndarray shape (n_samples, n_channels)
    """
    # Determine samples per packet and required bytes
    if n_channels == 4:
        n_samples = 3
        bytes_needed = 30
    elif n_channels == 8:
        n_samples = 2
        bytes_needed = 40
    elif n_channels == 16:
        n_samples = 1
        bytes_needed = 40
    else:
        return None

    if len(data_bytes) < bytes_needed:
        return None

    # Convert bytes to bit array (LSB first)
    bits = _bytes_to_bits(data_bytes, bytes_needed)

    # Parse 20-bit values
    data = np.zeros((n_samples, n_channels), dtype=np.float32)

    for sample_idx in range(n_samples):
        for channel_idx in range(n_channels):
            bit_start = (sample_idx * n_channels + channel_idx) * 20

            # Extract 20 bits and scale
            int_value = _extract_packed_int(bits, bit_start, 20)
            data[sample_idx, channel_idx] = int_value * OPTICS_SCALE

    return data


# ============================================================================
# Timestamp Resampling Functions
# ============================================================================
# Convenience Functions
# ============================================================================


def make_timestamps(
    subpackets: List[Dict],
    base_time: Optional[float] = None,
    wrap_offset: int = 0,
    last_abs_tick: int = 0,
    sample_counter: int = 0,
) -> tuple[np.ndarray, float, int, int, int]:
    """
    Add timestamps to a list of subpackets for a single sensor type and return as numpy array.

    This implements global timestamping: all subpackets are sorted by device time before
    applying wraparound detection and timestamp generation. This prevents artificial gaps
    that occur when per-message timestamping triggers false wraparound detection due to
    timing inversions between messages.

    Global Timestamping Logic:
    --------------------------
    1. **Global Sorting**: Sort ALL subpackets by (pkt_time_raw, pkt_index, subpkt_index)
       - Ensures correct temporal order across message boundaries
       - Handles BLE packet reordering and timing jitter

    2. **Unified Wraparound Detection**: Apply 32-bit clock wraparound logic once across
       all subpackets, preventing false detections from cross-message inversions

    3. **Stream-Relative Timestamp Generation**: Generate timestamps relative to stream start
       (base_time = 0.0), ensuring uniform spacing at declared sampling rates while
       maintaining device timing precision through tick-based calculations

    4. **State Management**: For streaming, maintain state across calls to handle ongoing
       data streams with correct wraparound detection

    Parameters:
    -----------
    subpackets : List[Dict]
        List of subpacket dicts for a single sensor type
    base_time : float, optional
        Base time for timestamp calculation. If None, defaults to 0.0 for streaming
        applications (relative to stream start) instead of device boot time.
    wrap_offset : int
        Current wraparound offset for 32-bit clock
    last_abs_tick : int
        Last absolute tick value for wraparound detection
    sample_counter : int
        Current sample counter for monotonic timestamping

    Returns:
    --------
    tuple : (array, base_time, wrap_offset, last_abs_tick, sample_counter)
        array : np.ndarray with shape (total_samples, n_channels + 1)
            First column is timestamp, remaining columns are sensor data
        base_time, wrap_offset, last_abs_tick, sample_counter : Updated state values
    """
    if len(subpackets) == 0:
        if base_time is None:
            base_time = 0.0
        return np.empty((0, 0)), base_time, wrap_offset, last_abs_tick, sample_counter

    # Get sensor configuration from first subpacket
    first_tag = subpackets[0].get("tag_byte")
    config = SENSORS.get(first_tag, {}) if first_tag else {}
    sampling_rate = config.get("rate", 1.0)
    n_samples_per_subpkt = config.get("n_samples", 1)

    # Sort all subpackets globally by device timestamp and sequence indices
    sorted_subpackets = sorted(
        subpackets,
        key=lambda s: (
            s.get("pkt_time_raw", 0),
            s["pkt_index"],
            s["subpkt_index"] if s["subpkt_index"] is not None else -1,
        ),
    )

    # Efficiently collect all data arrays and assign sequential timestamps
    if len(sorted_subpackets) == 0:
        if base_time is None:
            base_time = 0.0
        return np.empty((0, 0)), base_time, wrap_offset, last_abs_tick, sample_counter
    else:
        # Pre-calculate total samples to pre-allocate arrays
        total_samples = sum(
            subpkt.get("n_samples", n_samples_per_subpkt)
            for subpkt in sorted_subpackets
        )

        # Get number of channels from first subpacket
        # Handle both 1D and 2D arrays (Battery data is 1D)
        first_data = sorted_subpackets[0]["data"]
        if first_data.ndim == 1:
            n_channels = 1
            # Ensure 1D arrays are reshaped for consistency
            for subpkt in sorted_subpackets:
                if subpkt["data"].ndim == 1:
                    subpkt["data"] = subpkt["data"].reshape(-1, 1)
        else:
            n_channels = first_data.shape[1]

        # Pre-allocate arrays for efficiency
        data_array = np.empty((total_samples, n_channels), dtype=np.float32)
        times_array = np.empty(total_samples, dtype=np.float64)

        clock_mod = 1 << 32
        first_raw_tick = sorted_subpackets[0]["pkt_time_raw"]

        # Initialize rolling state for this sensor stream if not provided
        if base_time is None:
            # For streaming applications, use 0 as base_time to make timestamps relative
            # to when streaming started, not when device booted. This eliminates the
            # need for re-anchoring in LSL streaming while preserving device timing precision.
            base_time = 0.0
            wrap_offset = 0
            last_abs_tick = first_raw_tick
            sample_counter = 0

        # Track base tick for precision (avoid accumulating floating point error)
        base_tick = first_raw_tick

        row_idx = 0

        for subpkt in sorted_subpackets:
            data = subpkt["data"]
            n_samples = subpkt.get("n_samples", n_samples_per_subpkt)
            raw_tick = subpkt.get("pkt_time_raw", first_raw_tick)

            # Expand 32-bit tick into monotonically increasing absolute tick
            tick_diff = raw_tick - last_abs_tick
            abs_tick = raw_tick + wrap_offset

            # previous raw tick in [0, clock_mod)
            prev_raw_tick = last_abs_tick % clock_mod

            # If raw_tick < prev_raw_tick there are two possibilities:
            #  - a true wraparound (prev_raw_tick - raw_tick is large, near clock_mod)
            #  - a small inversion / reordering (prev_raw_tick - raw_tick is small)
            if raw_tick < prev_raw_tick:
                # check how large the backward jump is
                if (prev_raw_tick - raw_tick) > (clock_mod // 2):
                    # treat as true wraparound - increment offset once
                    wrap_offset += clock_mod
                    abs_tick = raw_tick + wrap_offset
                else:
                    # small inversion / jitter - force monotonicity
                    abs_tick = last_abs_tick
            else:
                # normal forward progress
                abs_tick = raw_tick + wrap_offset
                # defensive check: if abs_tick somehow still < last_abs_tick, force monotonicity
                if abs_tick < last_abs_tick:
                    abs_tick = last_abs_tick

            last_abs_tick = abs_tick

            # Use high-precision arithmetic to avoid floating point accumulation errors
            # Calculate time difference from stream start using ticks for precision
            tick_diff = abs_tick - base_tick
            time_diff = tick_diff / DEVICE_CLOCK_HZ

            # Convert to sample index with high precision
            # Use floating point division but minimize operations
            relative_samples_float = time_diff * sampling_rate
            relative_samples = int(round(relative_samples_float))

            # Maintain monotonic sample indices (original behavior)
            # This handles hardware timestamp limitations where multiple subpackets
            # may share identical timestamps
            if relative_samples < sample_counter:
                relative_samples = sample_counter

            # Generate uniform timestamps anchored to stream start (original behavior)
            # Hardware timestamps are used for ordering only, not for actual sample timing
            times = (
                base_time + (relative_samples + np.arange(n_samples)) / sampling_rate
            )

            # Copy data and timestamps into pre-allocated arrays
            data_array[row_idx : row_idx + n_samples] = data
            times_array[row_idx : row_idx + n_samples] = times

            # Update sample counter
            sample_counter = relative_samples + n_samples
            row_idx += n_samples

    # Add timestamps as first column
    assert base_time is not None  # Should always be set by now
    if data_array.size > 0:
        times_col = times_array.reshape(-1, 1).astype(np.float64)
        return (
            np.hstack((times_col, data_array)),
            base_time,
            wrap_offset,
            last_abs_tick,
            sample_counter,
        )
    else:
        return np.empty((0, 0)), base_time, wrap_offset, last_abs_tick, sample_counter


def decode_rawdata(messages: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Parse multiple messages and return organized data as Pandas DataFrames.

    This is the high-level convenience function that:
    1. Parses all messages (with timestamps)
    2. Combines data from all messages
    3. Converts to Pandas DataFrames

    Parameters:
    -----------
    messages : List[str]
        List of BLE message strings (one per line)

    Returns:
    --------
    Dict[str, pd.DataFrame] : Dictionary mapping sensor type to DataFrame
        Keys: "EEG", "ACCGYRO", "OPTICS", "BATTERY", "Unknown"
        Each DataFrame has columns: [time, channel_1, channel_2, ...]

    Example:
    --------
    >>> messages = load_messages("data.txt")
    >>> data = decode_rawdata(messages)
    >>> print(data["EEG"].shape)
    (1440, 5)  # 1440 samples x (time + 4 channels)
    >>> print(data["EEG"].columns)
    ['time', 'EEG_TP9', 'EEG_AF7', 'EEG_AF8', 'EEG_TP10']
    >>> print(data["ACCGYRO"].columns)
    ['time', 'ACC_X', 'ACC_Y', 'ACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z']
    >>> print(data["OPTICS"].columns)  # For Optics4
    ['time', 'OPTICS_LI_NIR', 'OPTICS_RI_NIR', 'OPTICS_LI_IR', 'OPTICS_RI_IR']
    """
    import pandas as pd

    # Collect all subpackets from all messages
    all_subpackets = {
        key: [] for key in ["EEG", "ACCGYRO", "OPTICS", "BATTERY", "Unknown"]
    }

    for message in messages:
        parsed = parse_message(message)
        for sensor_type, subpacket_list in parsed.items():
            all_subpackets[sensor_type].extend(subpacket_list)

    # Add global timestamps
    all_arrays = {}
    for sensor_type, subpackets in all_subpackets.items():
        array, _, _, _, _ = make_timestamps(subpackets)
        all_arrays[sensor_type] = array

    # Convert to DataFrames
    result = {}

    for sensor_type, data_array in all_arrays.items():
        if data_array.size == 0:
            # Empty DataFrame for this sensor type
            result[sensor_type] = pd.DataFrame()
            continue

        # Create column names based on sensor type
        n_cols = data_array.shape[1]

        if sensor_type == "EEG":
            n_channels = n_cols - 1
            columns = ["time", *_select_eeg_channels(n_channels)]
        elif sensor_type == "ACCGYRO":
            columns = ["time", *ACCGYRO_CHANNELS]
        elif sensor_type == "OPTICS":
            n_channels = n_cols - 1
            columns = ["time", *_select_optics_channels(n_channels)]
        elif sensor_type == "BATTERY":
            columns = ["time", "battery_percent"]
        else:
            # Generic column names
            columns = ["time"] + [f"ch_{i+1}" for i in range(n_cols - 1)]

        result[sensor_type] = pd.DataFrame(data_array, columns=columns)

    return result
