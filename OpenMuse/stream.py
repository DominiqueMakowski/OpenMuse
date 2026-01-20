"""
Muse BLE to LSL Streaming
==========================

This module streams decoded Muse sensor data over Lab Streaming Layer (LSL) in real-time.
It handles BLE data reception, decoding, timestamp conversion, packet reordering, and
LSL transmission.

Streaming Architecture:
-----------------------
1. BLE packets arrive asynchronously via Bleak callbacks (_on_data)
2. Packets are decoded using parse_message() from decode.py
3. Device timestamps are converted to LSL time using a Stable Clock model
4. Samples are buffered to allow packet reordering
5. Buffer is periodically flushed: samples sorted by timestamp and pushed to LSL
6. LSL outlets broadcast data to any connected LSL clients (e.g., LabRecorder)

Timestamp Handling - Stable Clock Synchronization:
--------------------------------------------------
This version implements a "Stable Clock" synchronization engine designed to prevent
linear drift caused by Bluetooth buffer bloat (latency spikes).

1. **device_time** (from make_timestamps):
   - A t=0 relative timestamp based on the device's 256kHz crystal oscillator.
   - This clock is physically stable and accurate over short/medium durations.

2. **lsl_now** (from local_clock()):
   - The computer's LSL clock (arrival time). This is subject to network jitter
     and buffer bloat (asymmetric latency).

3. **Correction Model (Physics-Constrained RLS)**:
   - We fit a linear model: `lsl_time = offset + (slope * device_time)`
   - **Crucial Difference:** Unlike standard regression, we **constrain the slope**
     (clock speed) to remain near 1.0.
   - **Why?** Pure regression misinterprets buffer bloat (late packets) as the
     device clock "slowing down," causing runaway linear drift.
   - **Result:** The filter effectively tracks the *offset* (intercept) while
     ignoring temporary latency spikes, ensuring the LSL stream remains synchronized
     with the "fastest" packets (minimum latency envelope).

Packet Reordering Buffer - Critical Design Component:
------------------------------------------------------
**WHY BUFFERING IS NECESSARY:**

BLE transmission can REORDER entire messages (not just individual packets). Analysis shows:
- Some messages arrive out of order
- Device's timestamps are CORRECT (device clock is monotonic and accurate)
- But messages processed in arrival order → non-monotonic timestamps

**EXAMPLE:**
  Device captures:  Msg 17 (t=13711.801s) → Msg 16 (t=13711.811s)
  BLE transmits:    Msg 16 arrives first, then Msg 17 (OUT OF ORDER!)
  Without buffer:   Push [t=811, t=801, ...] → NON-MONOTONIC to LSL ✗
  With buffer:      Sort [t=801, t=811, ...] → MONOTONIC to LSL ✓

**BUFFER OPERATION:**

1. Samples held in buffer for FLUSH_INTERVAL seconds (default: 200ms)
2. When buffer time limit reached, all buffered samples are:
   - Concatenated across packets/messages
   - **Sorted by device timestamp** (preserves device timing, corrects arrival order)
   - **Timestamps already in LSL time domain** (mapped via ConstrainedRLSClock)
   - Pushed as a single chunk to LSL
3. LSL receives samples in correct temporal order with device timing preserved

**BUFFER FLUSH TRIGGERS:**
- Time threshold: FLUSH_INTERVAL seconds elapsed since last flush
- Size threshold: MAX_BUFFER_PACKETS accumulated (safety limit)
- End of stream: Final flush when disconnecting

**BUFFER SIZE RATIONALE:**
- Original: 80ms (insufficient for ~90ms delays observed in data)
- Previous: 250ms (captures nearly all out-of-order messages)
- Current: 200ms (balances low latency with high temporal ordering accuracy)
- Trade-off: Latency (200ms delay) vs. timestamp quality (near-perfect monotonic output)
- For real-time applications: can reduce further, accept some non-monotonic timestamps
- For recording quality: 200ms provides excellent temporal ordering

Timestamp Quality & Device Timing Preservation:
------------------------------------------------
**MONOTONICITY:**

The decode.py output may show some non-monotonic timestamps, which might reflect
BLE message arrival order, NOT device timing errors. The timestamp VALUES are
correct and preserve the device's accurate 256 kHz clock timing.

**PIPELINE FLOW:**
  decode.py:  Processes messages in arrival order → some might be non-monotonic
              ↓ (but timestamp values preserve device timing)
  stream.py:  Sorts by device timestamp → 0% non-monotonic ✓
              ↓ (restores correct temporal order)
  LSL/XDF:    Monotonic timestamps with device timing preserved ✓

**DEVICE TIMING ACCURACY:**
- Device uses 256 kHz internal clock (accurate, monotonic)
- All subpackets within a message share same pkt_time (verified empirically)
- decode.py uses base_time + sequential offsets (preserves device timing)
- Intervals between samples match device's actual sampling rate (256 Hz, 52 Hz, etc.)
- This pipeline preserves device timing perfectly while handling BLE reordering

**VERIFICATION:**

When loading XDF files with pyxdf:
- Use dejitter_timestamps=False for actual timestamp quality

LSL Stream Configuration:
-------------------------
Four LSL streams are created:
- Muse_EEG: 8 channels at 256 Hz (EEG + AUX)
- Muse_ACCGYRO: 6 channels at 52 Hz (accelerometer + gyroscope)
- Muse_OPTICS: 16 channels at 64 Hz (PPG sensors)
- Muse_BATTERY: 1 channel at 0.2 Hz (battery percentage, new firmware)

Each stream includes:
- Channel labels (from decode.py)
- Nominal sampling rate (declared device rate)
- Data type (float32)
- Manufacturer metadata

Optional Raw Data Logging:
----------------------
If the 'record' parameter is provided, all raw BLE packets are logged to a text file
in the same format as the 'record' command:
- ISO8601 UTC timestamp
- Characteristic UUID
- Hex payload
- This is useful for verification and offline analysis/re-parsing.

"""

import asyncio
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, TextIO, Tuple, Union

import bleak
import numpy as np
from bleak.exc import BleakError
from mne_lsl.lsl import StreamInfo, StreamOutlet, local_clock

from .clocks import (
    AdaptiveOffsetClock,
    ConstrainedRLSClock,
    RobustOffsetClock,
    StandardRLSClock,
    WindowedRegressionClock,
)
from .decode import (
    ACCGYRO_CHANNELS,
    BATTERY_CHANNELS,
    make_timestamps,
    parse_message,
    select_eeg_channels,
    select_optics_channels,
)
from .muse import MuseS
from .utils import configure_lsl_api_cfg, get_utc_timestamp

MAX_BUFFER_PACKETS = 52  # ~200ms capacity for 256Hz
FLUSH_INTERVAL = 0.2  # 200ms jitter buffer

CLOCKS = {
    "adaptive": AdaptiveOffsetClock,
    "constrained": ConstrainedRLSClock,
    "robust": RobustOffsetClock,
    "standard": StandardRLSClock,
    "windowed": WindowedRegressionClock,
}


@dataclass
class SensorStream:
    """Holds the LSL outlet and a buffer for a single sensor stream."""

    outlet: StreamOutlet
    clock: Union[
        AdaptiveOffsetClock,
        ConstrainedRLSClock,
        RobustOffsetClock,
        StandardRLSClock,
        WindowedRegressionClock,
    ]
    buffer: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)

    # Track state for make_timestamps
    base_time: Optional[float] = None
    wrap_offset: int = 0
    last_abs_tick: int = 0
    sample_counter: int = 0
    last_update_device_time: float = -1.0


def create_stream_outlet(
    sensor_type: str,
    n_channels: int,
    device_name: str,
    device_id: str,
    clock_model: str,
) -> SensorStream:
    """Create an LSL outlet for a specific sensor stream."""
    if sensor_type == "EEG":
        ch_names = select_eeg_channels(n_channels)
        sfreq = 256.0
        stype = "EEG"
        source_id = f"{device_id}_eeg"
    elif sensor_type == "ACCGYRO":
        ch_names = list(ACCGYRO_CHANNELS)
        sfreq = 52.0
        stype = "ACCGYRO"
        source_id = f"{device_id}_accgyro"
    elif sensor_type == "OPTICS":
        ch_names = select_optics_channels(n_channels)
        sfreq = 64.0
        stype = "PPG"
        source_id = f"{device_id}_optics"
    elif sensor_type == "BATTERY":
        ch_names = list(BATTERY_CHANNELS)
        sfreq = 0.2  # New firmware sends battery every 5 seconds
        stype = "Battery"
        source_id = f"{device_id}_battery"
    else:
        raise ValueError(f"Unknown sensor type: {sensor_type}")

    info = StreamInfo(
        name=f"Muse-{sensor_type} ({device_id})",
        stype=stype,
        n_channels=len(ch_names),
        sfreq=sfreq,
        dtype="float32",
        source_id=source_id,
    )
    desc = info.desc
    desc.append_child_value("manufacturer", "Muse")
    desc.append_child_value("model", "MuseS")
    desc.append_child_value("device", device_name)
    channels = desc.append_child("channels")
    for ch_name in ch_names:
        channels.append_child("channel").append_child_value("label", ch_name)

    clock_class = CLOCKS.get(clock_model, AdaptiveOffsetClock)
    return SensorStream(outlet=StreamOutlet(info), clock=clock_class())


def _decode_with_channel_padding(
    sensor_type: str,
    pkt_list: List[dict],
    stream: SensorStream,
    verbose: bool = True,
) -> Tuple[np.ndarray, SensorStream]:
    """
    Decode packets and handle channel count mismatches by padding with NaN.

    This handles rare cases where Muse devices send packets with different TAG bytes
    than expected (e.g., 0x11 EEG4 vs 0x12 EEG8, or 0x34/0x35/0x36 for OPTICS).
    Instead of dropping mismatched packets, we preserve the available channels and
    pad the missing channels with NaN.

    Parameters
    ----------
    sensor_type : str
        Sensor type identifier (EEG, ACCGYRO, OPTICS, BATTERY)
    pkt_list : List[dict]
        List of decoded packet dictionaries from parse_message()
    stream : SensorStream
        The sensor stream object containing outlet and timestamp state
    verbose : bool
        Whether to print warnings for mismatched packets

    Returns
    -------
    Tuple[np.ndarray, SensorStream]
        Decoded data array (timestamps + channels) and updated stream object
    """
    expected_channels = stream.outlet.n_channels

    # Separate packets by channel count
    matching_pkts = []
    mismatched_pkts = []  # (pkt, actual_channels)

    for pkt in pkt_list:
        pkt_channels = pkt.get("n_channels")
        if pkt_channels == expected_channels:
            matching_pkts.append(pkt)
        else:
            mismatched_pkts.append((pkt, pkt_channels))

    result_array = np.empty((0, 1 + expected_channels))

    # Process matching packets normally
    if matching_pkts:
        current_state = (
            stream.base_time,
            stream.wrap_offset,
            stream.last_abs_tick,
            stream.sample_counter,
        )
        array, base_time, wrap_offset, last_abs_tick, sample_counter = make_timestamps(
            matching_pkts, *current_state
        )
        result_array = array
        stream.base_time = base_time
        stream.wrap_offset = wrap_offset
        stream.last_abs_tick = last_abs_tick
        stream.sample_counter = sample_counter

    # Process mismatched packets: pad with NaN to preserve available data
    for pkt, actual_channels in mismatched_pkts:
        if verbose:
            tag_hex = hex(pkt.get("tag_byte", 0))
            print(
                f"[{sensor_type}] Padding packet with {actual_channels} channels "
                f"(tag={tag_hex}) to {expected_channels} channels (filling with NaN)"
            )

        current_state = (
            stream.base_time,
            stream.wrap_offset,
            stream.last_abs_tick,
            stream.sample_counter,
        )
        mismatch_array, base_time, wrap_offset, last_abs_tick, sample_counter = (
            make_timestamps([pkt], *current_state)
        )

        if mismatch_array.size > 0:
            n_samples = mismatch_array.shape[0]
            # Create padded array: timestamp + expected_channels
            padded = np.full(
                (n_samples, 1 + expected_channels), np.nan, dtype=np.float64
            )
            padded[:, 0] = mismatch_array[:, 0]  # Copy timestamp
            padded[:, 1 : actual_channels + 1] = mismatch_array[:, 1:]  # Copy channels

            # Merge with existing data
            if result_array.size > 0:
                result_array = np.vstack([result_array, padded])
            else:
                result_array = padded

        stream.base_time = base_time
        stream.wrap_offset = wrap_offset
        stream.last_abs_tick = last_abs_tick
        stream.sample_counter = sample_counter

    return result_array, stream


async def _stream_async(
    address: str,
    preset: str,
    duration: Optional[float] = None,
    raw_data_file: Optional[TextIO] = None,
    verbose: bool = True,
    clock_model: str = "windowed",
    sensors: Optional[List[str]] = None,
):
    """Asynchronous context for BLE connection and LSL streaming.

    Parameters
    ----------
    sensors : Optional[List[str]]
        List of sensor types to stream. If None, streams all sensors.
        Valid values: "EEG", "ACCGYRO", "OPTICS", "BATTERY"
    """

    # Default to all sensors if not specified
    if sensors is None:
        sensors = ["EEG", "ACCGYRO", "OPTICS", "BATTERY"]
    else:
        # Normalize to uppercase
        sensors = [s.upper() for s in sensors]

    # --- Stream State ---
    streams: Dict[str, SensorStream] = {}
    last_flush_time = 0.0
    samples_sent = {"EEG": 0, "ACCGYRO": 0, "OPTICS": 0, "BATTERY": 0}
    start_time = 0.0

    def _queue_samples(sensor_type: str, data_array: np.ndarray, lsl_now: float):
        """
        Map timestamps and buffer samples.
        """
        if data_array.size == 0 or data_array.ndim != 2 or data_array.shape[1] < 2:
            return

        stream = streams.get(sensor_type)
        if stream is None:
            return

        # Extract device timestamps
        device_times = data_array[:, 0]
        samples = data_array[:, 1:]

        # --- Validate Channel Count (Safety Check) ---
        # Primary filtering happens in _on_data before make_timestamps.
        # This check catches any edge cases that slip through.
        expected_channels = stream.outlet.n_channels
        if samples.shape[1] != expected_channels:
            if verbose:
                print(
                    f"[{sensor_type}] Safety check: skipping packet with mismatched channel count: "
                    f"got {samples.shape[1]}, expected {expected_channels}"
                )
            return

        # --- Update Clock Model ---
        # We update the clock using the *latest* packet in this chunk
        last_device_time = device_times[-1]

        # Only update if time moved forward (avoids issues with out-of-order arrival for model update)
        if last_device_time > stream.last_update_device_time:
            stream.clock.update(last_device_time, lsl_now)
            stream.last_update_device_time = last_device_time
        else:
            # Log when clock updates are skipped (helps debug out-of-order packet timing issues)
            if verbose:
                print(
                    f"[{sensor_type}] Skipping clock update for non-monotonic device time: "
                    f"{last_device_time} <= {stream.last_update_device_time}"
                )

        # --- Map Timestamps ---
        # Transform the entire chunk using the current stable model
        lsl_timestamps = stream.clock.map_time(device_times)

        # Add to buffer
        stream.buffer.append((lsl_timestamps, samples))

    def _flush_buffer():
        """Sort and push all buffered samples to LSL."""
        nonlocal last_flush_time, samples_sent  # noqa: F824
        last_flush_time = time.monotonic()

        for sensor_type, stream in streams.items():
            if not stream.buffer:
                continue

            # Concatenate all buffered samples
            all_timestamps = np.concatenate([ts for ts, _ in stream.buffer])
            all_samples = np.concatenate([s for _, s in stream.buffer])
            stream.buffer.clear()

            # Sort by LSL timestamp to correct BLE packet reordering
            sort_indices = np.argsort(all_timestamps)
            sorted_timestamps = all_timestamps[sort_indices]
            sorted_data = all_samples[sort_indices, :]

            # Push chunk
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message=".*A single sample is pushed.*"
                    )
                    stream.outlet.push_chunk(
                        x=sorted_data.astype(np.float32, copy=False),
                        timestamp=sorted_timestamps.astype(np.float64, copy=False),
                        pushThrough=True,
                    )
                samples_sent[sensor_type] += len(sorted_data)
            except Exception as e:
                if verbose:
                    print(f"Error pushing LSL chunk for {sensor_type}: {e}")

    def _on_data(sender, data: bytearray):
        """Main data callback from Bleak."""
        ts = get_utc_timestamp()
        uuid_str = str(sender.uuid) if hasattr(sender, "uuid") else str(sender)
        message = f"{ts}\t{uuid_str}\t{data.hex()}"

        if raw_data_file:
            try:
                raw_data_file.write(message + "\n")
            except Exception:
                pass

        subpackets = parse_message(message)
        decoded: Dict[str, np.ndarray] = {}

        # Ensure streams exist (only for requested sensor types)
        for sensor_type, pkt_list in subpackets.items():
            if pkt_list and sensor_type not in streams and sensor_type in sensors:
                n_channels = pkt_list[0].get("n_channels")
                if n_channels:
                    streams[sensor_type] = create_stream_outlet(
                        sensor_type, n_channels, client.name, address, clock_model
                    )

        # Decode & Make Timestamps (Relative Device Time)
        for sensor_type, pkt_list in subpackets.items():
            stream = streams.get(sensor_type)
            if stream:
                array, stream = _decode_with_channel_padding(
                    sensor_type, pkt_list, stream, verbose
                )
                if array.size > 0:
                    decoded[sensor_type] = array

        # Get 'now' for clock sync
        lsl_now = local_clock()

        # Queue samples
        for sensor_type in ["EEG", "ACCGYRO", "OPTICS", "BATTERY"]:
            sensor_data = decoded.get(sensor_type, np.empty((0, 0)))
            if sensor_data.size > 0:
                _queue_samples(sensor_type, sensor_data, lsl_now)

        # Flush trigger
        should_flush = (time.monotonic() - last_flush_time > FLUSH_INTERVAL) or any(
            len(s.buffer) > MAX_BUFFER_PACKETS for s in streams.values()
        )
        if should_flush:
            _flush_buffer()

    # --- Connection ---
    if verbose:
        print(f"Connecting to {address}...")

    async with bleak.BleakClient(address, timeout=15.0) as client:
        if verbose:
            print(f"Connected. Device: {client.name}")
            print(f"Streaming sensors: {', '.join(sensors)}")

        start_time = time.monotonic()
        data_callbacks = {uuid: _on_data for uuid in MuseS.DATA_CHARACTERISTICS}
        await MuseS.connect_and_initialize(
            client, preset, data_callbacks, verbose=verbose
        )

        if verbose:
            print("Streaming data... (Press Ctrl+C to stop)")

        while True:
            await asyncio.sleep(0.5)
            if duration and (time.monotonic() - start_time) > duration:
                break
            if not client.is_connected:
                break

        _flush_buffer()
        if verbose:
            print("Stream stopped.")


def stream(
    address: Union[str, List[str]],
    preset: str = "p1041",
    duration: Optional[float] = None,
    record: Union[bool, str] = False,
    verbose: bool = True,
    clock: str = "windowed",
    sensors: Optional[List[str]] = None,
) -> None:
    """
    Stream decoded EEG and accelerometer/gyroscope data over LSL.
    Supports streaming from a single device (str) or multiple devices (List[str]) in parallel.

    When streaming multiple devices, if `record` is a string (filename), the device address
    will be appended to the filename to avoid collisions.

    Parameters
    ----------
    sensors : Optional[List[str]]
        List of sensor types to stream. If None, streams all sensors (EEG, ACCGYRO, OPTICS, BATTERY).
        Valid values: \"EEG\", \"ACCGYRO\", \"OPTICS\", \"BATTERY\".
    """
    configure_lsl_api_cfg()

    addresses = [address] if isinstance(address, str) else address
    # Remove duplicates if any
    addresses = list(set(addresses))

    tasks = []
    file_handles = []

    async def run_multistream():
        if verbose:
            print(f"Starting stream for {len(addresses)} device(s)...")

        for addr in addresses:
            raw_data_file = None
            if record:
                # Generate unique filename for this device
                # Sanitize address for filename
                sanitized_addr = addr.replace(":", "").replace("-", "")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                if isinstance(record, str):
                    if len(addresses) > 1:
                        # Insert address into provided filename
                        if "." in record:
                            parts = record.rsplit(".", 1)
                            filename = f"{parts[0]}_{sanitized_addr}.{parts[1]}"
                        else:
                            filename = f"{record}_{sanitized_addr}"
                    else:
                        filename = record
                else:
                    # Default filename format
                    filename = f"rawdata_stream_{sanitized_addr}_{timestamp}.txt"

                try:
                    f = open(filename, "w", encoding="utf-8")
                    file_handles.append(f)
                    raw_data_file = f
                    if verbose:
                        print(f"Recording raw data for {addr} to {filename}")
                except IOError as e:
                    print(f"Warning: Could not open file for recording {addr}: {e}")

            # Create task for this device
            tasks.append(
                _stream_async(
                    addr, preset, duration, raw_data_file, verbose, clock, sensors
                )
            )

        # Run all streams concurrently
        await asyncio.gather(*tasks)

    try:
        asyncio.run(run_multistream())
    except KeyboardInterrupt:
        if verbose:
            print("Streaming stopped by user.")
    except BleakError as e:
        print(f"BLEAK Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        for f in file_handles:
            try:
                f.close()
            except Exception:
                pass
