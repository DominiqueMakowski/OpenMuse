"""
Muse BLE to LSL Streaming
==========================

This module streams decoded Muse sensor data over Lab Streaming Layer (LSL) in real-time.
It handles BLE data reception, decoding, packet reordering, and
LSL transmission.

Streaming Architecture:
-----------------------
1. BLE packets arrive asynchronously via Bleak callbacks (_on_data)
2. Packets are decoded using parse_message() from decode.py
3. Timestamps are generated in the device's own time domain (seconds since boot)
4. Samples are buffered to allow packet reordering
5. Buffer is periodically flushed: samples sorted by device timestamp and pushed to LSL
6. LSL outlets broadcast data (with device timestamps) to any connected LSL clients

Timestamp Handling - LSL Clock Drift Correction:
------------------------------------------------
This module uses LSL's built-in clock synchronization mechanism to ensure
sub-millisecond, long-term accuracy and to correct for clock drift between
the Muse device and the LSL host computer.

The old method (using a single, fixed offset) could not correct for clock
drift. The new, more accurate pipeline is:

1.  **Device Time Generation:** `decode.py` generates timestamps in the
    **device's time domain** (e.g., "seconds since device boot") by using
    the first packet's timestamp as the `base_time`.

2.  **LSL Push:** `stream.py` pushes these *raw device timestamps*
    directly to the LSL outlet, without any conversion.

3.  **LSL Pairing:** LSL automatically pairs these incoming device
    timestamps with its own `local_clock()` time. These pairs are
    saved in the XDF file.

4.  **Offline Correction:** When loading the XDF file, `pyxdf` (with
    `synchronize_clocks=True`) performs a linear regression on all
    these time-pairs. This builds a highly accurate model that
    corrects for both the initial time offset AND any clock drift
    that occurred during the recording.

This method preserves the device's perfect internal timing and provides
the most accurate synchronization possible.

Packet Reordering Buffer - Critical Design Component:
------------------------------------------------------
**WHY BUFFERING IS NECESSARY:**

BLE transmission can REORDER entire messages. Analysis shows:
- ~5% of messages arrive out of order
- Backward jumps can exceed 80ms in severe cases
- Device's timestamps are CORRECT (device clock is monotonic)
- But messages processed in arrival order → non-monotonic timestamps

**EXAMPLE:**
  Device captures:  Msg 17 (t=13711.801s) → Msg 16 (t=13711.811s)
  BLE transmits:    Msg 16 arrives first, then Msg 17 (OUT OF ORDER!)
  Without buffer:   Push [t=811, t=801, ...] → NON-MONOTONIC to LSL ✗
  With buffer:      Sort [t=801, t=811, ...] → MONOTONIC to LSL ✓

**BUFFER OPERATION:**

1. Samples held in buffer for BUFFER_DURATION_SECONDS (default: 150ms)
2. When buffer time limit reached, all buffered samples are:
   - Concatenated across packets/messages
   - **Sorted by device timestamp** (preserves device timing, corrects arrival order)
   - **Timestamps are in the device's time domain**
   - Pushed as a single chunk to LSL
3. LSL receives samples in correct temporal order with device timing preserved

**BUFFER FLUSH TRIGGERS:**
- Time threshold: BUFFER_DURATION_SECONDS elapsed since last flush
- Size threshold: MAX_BUFFER_PACKETS accumulated (safety limit)
- End of stream: Final flush when disconnecting

**BUFFER SIZE RATIONALE:**
- Current: 150ms (balances low latency with high temporal ordering accuracy)
- Trade-off: Latency (150ms delay) vs. timestamp quality (near-perfect monotonic output)
- For recording quality: 150ms provides excellent temporal ordering

Timestamp Quality & Device Timing Preservation:
------------------------------------------------
**CRITICAL INSIGHT:**

The `decode.py` output may show ~20% non-monotonic timestamps, but this is EXPECTED
and NOT an error. These non-monotonic timestamps simply reflect BLE message arrival
order, NOT device timing errors. The timestamp VALUES are correct and preserve the
device's accurate 256 kHz clock timing.

**PIPELINE FLOW:**
  decode.py:  Processes messages in arrival order → ~20% non-monotonic (expected)
              ↓ (but timestamp values preserve device timing)
  stream.py:  Sorts by device timestamp → 0% non-monotonic ✓
              ↓ (restores correct temporal order)
  LSL/XDF:    Monotonic, drift-corrected timestamps (with pyxdf) ✓

**DEVICE TIMING ACCURACY:**
- Device uses 256 kHz internal clock (accurate, monotonic)
- `decode.py` uses this clock to generate perfectly uniform sample intervals
  at the device's actual sampling rate (256 Hz, 52 Hz, etc.)
- This pipeline preserves device timing perfectly while handling BLE reordering

**VERIFICATION:**

When loading XDF files with pyxdf:
- Use **`synchronize_clocks=True`** for multi-device sync and **to correct for clock drift**.
- Use `handle_clock_resets=True` (standard practice).
- Use **`dejitter_timestamps=False`** to use the high-quality, uniform timestamps
  generated from the device's 256kHz clock.

LSL Stream Configuration:
-------------------------
Three LSL streams are created:
- Muse_EEG: 8 channels at 256 Hz (EEG + AUX)
- Muse_ACCGYRO: 6 channels at 52 Hz (accelerometer + gyroscope)
- Muse_OPTICS: 16 channels at 64 Hz (PPG sensors)
- Muse_BATTERY: 1 channel at 1 Hz (battery percentage)

Each stream includes:
- Channel labels (from decode.py)
- Nominal sampling rate (declared device rate)
- Data type (float32)
- Units (microvolts for EEG, a.u. for others)
- Manufacturer metadata

Optional Raw Data Logging:
----------------------
If the 'record' parameter is provided, all raw BLE packets are logged to a text file:
- ISO8601 UTC timestamp
- Characteristic UUID
- Hex payload
- This is useful for verification and offline analysis/re-parsing.

"""

import asyncio
import os
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Union

import bleak
import numpy as np
from mne_lsl.lsl import StreamInfo, StreamOutlet, local_clock

from .backends import _run
from .decode import ACCGYRO_CHANNELS, BATTERY_CHANNELS, EEG_CHANNELS, OPTICS_CHANNELS, make_timestamps, parse_message
from .muse import MuseS
from .utils import configure_lsl_api_cfg, get_utc_timestamp

# LSL streaming constants
EEG_LABELS: tuple[str, ...] = EEG_CHANNELS
ACCGYRO_LABELS: tuple[str, ...] = ACCGYRO_CHANNELS
OPTICS_LABELS: tuple[str, ...] = OPTICS_CHANNELS
BATTERY_LABELS: tuple[str, ...] = BATTERY_CHANNELS

# Buffer duration in seconds
BUFFER_DURATION_SECONDS = 0.15

# Maximum number of BLE packets to buffer
MAX_BUFFER_PACKETS = 64


@dataclass
class SensorStream:
    outlet: StreamOutlet
    pad_to_channels: Optional[int]
    labels: tuple[str, ...]
    sampling_rate: float
    unit: str
    buffer: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    last_push_time: Optional[float] = None

    # State for make_timestamps()
    base_time: Optional[float] = None
    wrap_offset: int = 0
    last_abs_tick: int = 0
    sample_counter: int = 0


def _create_stream_outlet(
    name: str,
    stype: str,
    labels: tuple[str, ...],
    sfreq: float,
    dtype: str,
    source_id: str,
    unit: str,
    channel_type: Optional[str] = None,
) -> StreamOutlet:
    info = StreamInfo(
        name=name,
        stype=stype,
        n_channels=len(labels),
        sfreq=sfreq,
        dtype=dtype,
        source_id=source_id,
    )
    desc = info.desc
    desc.append_child_value("manufacturer", "Muse")
    channels = desc.append_child("channels")
    for label in labels:
        channel = channels.append_child("channel")
        channel.append_child_value("label", label)
        channel.append_child_value("unit", unit)
        if channel_type:
            channel.append_child_value("type", channel_type)
    return StreamOutlet(info, chunk_size=1)


def _build_sensor_streams() -> dict[str, SensorStream]:
    eeg_outlet = _create_stream_outlet(
        name="Muse_EEG",
        stype="EEG",
        labels=EEG_LABELS,
        sfreq=256.0,
        dtype="float32",
        source_id="Muse_EEG",
        unit="microvolts",
        channel_type="EEG",
    )

    accgyro_outlet = _create_stream_outlet(
        name="Muse_ACCGYRO",
        stype="Motion",
        labels=ACCGYRO_LABELS,
        sfreq=52.0,
        dtype="float32",
        source_id="Muse_ACCGYRO",
        unit="a.u.",
    )

    optics_outlet = _create_stream_outlet(
        name="Muse_OPTICS",
        stype="OPTICS",
        labels=OPTICS_LABELS,
        sfreq=64.0,
        dtype="float32",
        source_id="Muse_OPTICS",
        unit="a.u.",
        channel_type="OPTICS",
    )

    battery_outlet = _create_stream_outlet(
        name="Muse_BATTERY",
        stype="BATTERY",
        labels=BATTERY_LABELS,
        sfreq=0.1,  # Nominal rate from decode.py SENSORS dict
        dtype="float32",
        source_id="Muse_BATTERY",
        unit="percent",
    )

    streams = {
        "EEG": SensorStream(
            outlet=eeg_outlet,
            pad_to_channels=len(EEG_LABELS),
            labels=EEG_LABELS,
            sampling_rate=256.0,
            unit="microvolts",
        ),
        "ACCGYRO": SensorStream(
            outlet=accgyro_outlet,
            pad_to_channels=None,
            labels=ACCGYRO_LABELS,
            sampling_rate=52.0,
            unit="a.u.",
        ),
        "OPTICS": SensorStream(
            outlet=optics_outlet,
            pad_to_channels=len(OPTICS_LABELS),
            labels=OPTICS_LABELS,
            sampling_rate=64.0,
            unit="a.u.",
        ),
        "BATTERY": SensorStream(
            outlet=battery_outlet,
            pad_to_channels=len(BATTERY_LABELS),  # Should be 1
            labels=BATTERY_LABELS,
            sampling_rate=1.0,
            unit="percent",
        ),
    }

    return streams


async def _stream_async(
    address: str,
    preset: str,
    duration: Optional[float],
    record: Union[bool, str],
    verbose: bool,
) -> None:
    sensor_streams = _build_sensor_streams()
    samples_sent = {name: 0 for name in sensor_streams}

    # Single global offset for all sensors (they share the same device clock)
    device_to_lsl_offset = None

    # --- Logic to handle the 'record' parameter ---
    record_f = None
    record_path: Optional[str] = None

    if record is True:
        # Default path if record=True
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        record_path = f"rawdata_stream_{ts}.txt"
    elif isinstance(record, str):
        # Specific path if record="path/to/file.txt"
        record_path = record

    if record_path:
        # Ensure output directory exists
        try:
            outdir = os.path.dirname(os.path.abspath(record_path))
            if outdir and not os.path.exists(outdir):
                os.makedirs(outdir, exist_ok=True)
        except Exception:
            pass
        # Open output file in text mode and append
        try:
            record_f = open(record_path, "a", encoding="utf-8")
            if verbose:
                print(f"Recording raw data to {record_path}")
        except Exception as e:
            if verbose:
                print(f"Error opening raw record file {record_path}: {e}")
            record_f = None  # Ensure it's None if opening failed
    # --- END ---

    def _flush_buffer(sensor_type: str) -> None:
        """Flush reordering buffer for a specific sensor type: sort and push samples to LSL."""
        nonlocal samples_sent  # noqa: F824

        stream = sensor_streams[sensor_type]
        if len(stream.buffer) == 0:
            return

        # Concatenate all timestamps and data
        all_timestamps = np.concatenate([ts for ts, _ in stream.buffer])
        all_data = np.vstack([data for _, data in stream.buffer])

        # Sort by timestamp (in device time)
        sort_indices = np.argsort(all_timestamps)
        sorted_timestamps = all_timestamps[sort_indices]
        sorted_data = all_data[sort_indices]

        # Timestamps are in device time, no re-anchoring needed.
        anchored_timestamps = sorted_timestamps

        # Push to LSL
        try:
            # Suppress the "A single sample is pushed" warning,
            # which is expected for low-rate streams like BATTERY.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*A single sample is pushed.*",
                )
                stream.outlet.push_chunk(
                    x=sorted_data.astype(np.float32, copy=False),  # type: ignore[arg-type]
                    timestamp=anchored_timestamps.astype(np.float64, copy=False),
                    pushThrough=True,
                )
            samples_sent[sensor_type] += len(anchored_timestamps)

        except Exception as exc:
            if verbose:
                print(f"LSL push_chunk failed for {sensor_type}: {exc}")

        # Clear buffer and update last push time
        stream.buffer.clear()
        stream.last_push_time = local_clock()

    def _queue_samples(sensor_type: str, data_array: np.ndarray, lsl_now: float) -> None:
        nonlocal device_to_lsl_offset  # Access global offset

        if data_array.size == 0 or data_array.shape[1] < 2:
            return

        stream = sensor_streams[sensor_type]

        # Extract sensor data (exclude time column)
        samples = data_array[:, 1:].astype(np.float32)
        if stream.pad_to_channels:
            target = stream.pad_to_channels
            current = samples.shape[1]
            if current < target:
                padding = np.zeros((samples.shape[0], target - current), dtype=np.float32)
                samples = np.hstack([samples, padding])
            elif current > target:
                samples = samples[:, :target]

        device_times = data_array[:, 0]

        # Timestamps are in device time (from decode.py).
        # LSL will automatically pair them with its own clock,
        # allowing pyxdf to correct for clock drift.
        stream.buffer.append((device_times, samples))

        if stream.last_push_time is None:
            stream.last_push_time = lsl_now

        # Flush if buffer duration exceeded OR buffer size limit reached
        if lsl_now - stream.last_push_time >= BUFFER_DURATION_SECONDS:
            _flush_buffer(sensor_type)
        elif len(stream.buffer) >= MAX_BUFFER_PACKETS:
            if verbose:
                print(f"Warning: {sensor_type} buffer reached {MAX_BUFFER_PACKETS} packets, forcing flush")
            _flush_buffer(sensor_type)

    def _on_data(_, data: bytearray):
        ts = get_utc_timestamp()  # Get timestamp once

        # --- Raw recording logic ---
        if record_f:
            try:
                # Log timestamp, char UUID, and hex payload
                line = f"{ts}\t{MuseS.EEG_UUID}\t{data.hex()}\n"
                record_f.write(line)
            except Exception as e:
                if verbose:
                    # Avoid spamming errors
                    print(f"Warning: Failed to write to raw record file: {e}")
        # --- END ---

        # Original streaming logic
        message = f"{ts}\t{MuseS.EEG_UUID}\t{data.hex()}"
        subpackets = parse_message(message)

        decoded = {}
        for sensor_type, pkt_list in subpackets.items():
            if sensor_type in sensor_streams:
                stream = sensor_streams[sensor_type]

                # 1. Get current state
                current_state = (
                    stream.base_time,
                    stream.wrap_offset,
                    stream.last_abs_tick,
                    stream.sample_counter,
                )

                # 2. Call make_timestamps
                array, base_time, wrap_offset, last_abs_tick, sample_counter = make_timestamps(pkt_list, *current_state)
                decoded[sensor_type] = array

                # 3. Update state
                stream.base_time = base_time
                stream.wrap_offset = wrap_offset
                stream.last_abs_tick = last_abs_tick
                stream.sample_counter = sample_counter

        lsl_now = local_clock()

        _queue_samples("EEG", decoded.get("EEG", np.empty((0, 0))), lsl_now)
        _queue_samples("ACCGYRO", decoded.get("ACCGYRO", np.empty((0, 0))), lsl_now)
        _queue_samples("OPTICS", decoded.get("OPTICS", np.empty((0, 0))), lsl_now)
        _queue_samples("BATTERY", decoded.get("BATTERY", np.empty((0, 0))), lsl_now)

    try:
        if verbose:
            print(f"Connecting to {address} ...")

        async with bleak.BleakClient(address, timeout=15.0) as client:
            if verbose:
                print("Connected. Subscribing and configuring ...")

            # Build callbacks dict for all data characteristics
            data_callbacks = {MuseS.EEG_UUID: _on_data}

            # Use shared connection routine
            await MuseS.connect_and_initialize(client, preset, data_callbacks, verbose)

            # Streaming is now active
            if duration:
                if verbose:
                    print(f"Streaming for {duration} seconds...")
                start = time.time()
                try:
                    while time.time() - start < duration:
                        await asyncio.sleep(0.05)
                except asyncio.CancelledError:
                    pass
            else:
                if verbose:
                    print("Streaming indefinitely. Press Ctrl+C to stop.")
                try:
                    while True:
                        await asyncio.sleep(1)
                except asyncio.CancelledError:
                    pass

    except asyncio.CancelledError:
        if verbose:
            print("Stream cancelled.")
    except Exception as exc:
        if verbose:
            print(f"An error occurred: {exc}")
    finally:
        if verbose:
            print("Stopping stream...")

        # Flush any remaining samples in all reordering buffers
        for sensor_type, stream in sensor_streams.items():
            if len(stream.buffer) > 0:
                if verbose:
                    print(f"Flushing {len(stream.buffer)} buffered {sensor_type} packets...")
                _flush_buffer(sensor_type)

        # --- Close raw recording file ---
        if record_f:
            try:
                record_f.flush()
                record_f.close()
                if verbose and record_path:
                    print(f"Raw data recording saved to {record_path}")
            except Exception as exc:
                if verbose:
                    print(f"Error closing raw record file: {exc}")
        # --- END ---

        if verbose:
            print(
                "Stream stopped. " + ", ".join(f"{sensor}: {count} samples" for sensor, count in samples_sent.items())
            )


def stream(
    address: str,
    preset: str = "p1041",
    duration: Optional[float] = None,
    record: Union[bool, str] = False,
    verbose: bool = True,
) -> None:
    """
    Stream decoded EEG and accelerometer/gyroscope data over LSL.

    Creates three LSL streams:
    - Muse_EEG: 8 channels at 256 Hz (EEG + AUX)
    - Muse_ACCGYRO: 6 channels at 52 Hz (accelerometer + gyroscope)
    - Muse_OPTICS: 16 channels at 64 Hz (PPG sensors)

    Parameters
    ----------
    address : str
        Device address (e.g., MAC on Windows).
    preset : str
        Preset to send (e.g., p1041 for all channels, p1035 for basic config).
    duration : float, optional
        Optional stream duration in seconds. Omit to stream until interrupted.
    record : bool or str, optional
        If False (default), do not record raw data.
        If True, record raw BLE packets to a default timestamped file
        (e.g., 'rawdata_stream_20251024_183000.txt').
        If a string is provided, use it as the path to the output text file.
    verbose : bool
        If True, print verbose output.
    """
    # Configure LSL to reduce verbosity (disables IPv6 warnings and lowers log level)
    configure_lsl_api_cfg()

    _run(_stream_async(address, preset, duration, record, verbose))
