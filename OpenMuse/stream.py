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
3. Device timestamps are converted to LSL time
4. Samples are buffered to allow packet reordering
5. Buffer is periodically flushed: samples sorted by timestamp and pushed to LSL
6. LSL outlets broadcast data to any connected LSL clients (e.g., LabRecorder)

Timestamp Handling - Stream-Relative Timing:
-----------------------------------------------
Timestamps are now generated relative to when streaming starts (base_time = 0.0),
not when the Muse device was powered on. This eliminates the need for complex
re-anchoring while maintaining precise device timing.

The parse_message() function returns decoded data with THREE timestamp types:

1. **message_time** (datetime)
   - When the BLE message was received on the computer
   - Format: UTC datetime from get_utc_timestamp()
   - Source: Computer system clock
   - Used for: Debugging, logging
   - NOT used for: LSL timestamps

2. **pkt_time** (float, seconds)
   - When samples were captured on the Muse device
   - Format: Seconds since device boot (from 256 kHz device clock)
   - Source: Extracted from packet header (4-byte timestamp)
   - Used for: Calculating relative timing between samples
   - Device timing precision preserved through tick differences

3. **timestamps** (array column in decoded data)
   - Per-sample timestamps relative to stream start
   - Format: Seconds, uniformly spaced at nominal sampling rate
   - Source: 0.0 + (sample_index / sampling_rate) based on device timing
   - Used for: Final LSL timestamps
   - Naturally synchronized with other LSL streams

Simplified Timestamp Conversion Flow:
-------------------------------------
    Device Time Domain          →       LSL Time Domain
    ==================                  ===============

    Stream-relative timestamps          LSL timestamps
    (base_time = 0.0)          →→→→→   (anchored to LSL time)

    Where:
    - Timestamps start from 0 when streaming begins
    - Device timing precision preserved through 256kHz clock tick differences
    - Natural LSL synchronization without artificial re-anchoring

The conversion happens in two stages:

1. _queue_samples(): LSL time mapping
    - Extract stream-relative timestamps from decoded data
    - Add device_to_lsl_offset (maps stream start to LSL time)
    - Store in buffer with preserved device timing

2. _flush_buffer(): Conditional re-anchoring (rare)
    - Check if timestamps are already near current LSL time
    - Only apply re-anchoring for edge cases (timestamps >30s in past)
    - Preserves device timing precision while ensuring LSL compatibility
The parse_message() function returns decoded data with THREE timestamp types:

Timestamp Handling - Online Drift Correction:
---------------------------------------------
This version implements an online drift correction to compensate for clock skew
between the Muse device and the computer.

1. **device_time** (from make_timestamps):
   - A t=0 relative timestamp based on the device's 256kHz clock.
   - This clock has high precision but *skews* relative to the computer clock.
   - This value is saved as the first "muse_time" channel for debugging.

2. **lsl_now** (from local_clock()):
   - The computer's LSL clock. This is our "ground truth" time.

3. **Correction Model**:
   - We continuously fit a linear model: `lsl_time = a + (b * device_time)`
   - `a` (intercept) and `b` (slope/skew) are updated with every new packet.
   - The final `lsl_timestamps` pushed to LSL are the corrected values.

Packet Reordering Buffer - Critical Design Component:
------------------------------------------------------
**WHY BUFFERING IS NECESSARY:**

BLE transmission can REORDER entire messages (not just individual packets). Analysis shows:
- ~5% of messages arrive out of order
- Backward jumps can exceed 80ms in severe cases
- Device's timestamps are CORRECT (device clock is monotonic and accurate)
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
   - **Timestamps already in LSL time domain** (no conversion needed)
   - Pushed as a single chunk to LSL
3. LSL receives samples in correct temporal order with device timing preserved

**BUFFER FLUSH TRIGGERS:**
- Time threshold: BUFFER_DURATION_SECONDS elapsed since last flush
- Size threshold: MAX_BUFFER_PACKETS accumulated (safety limit)
- End of stream: Final flush when disconnecting

**BUFFER SIZE RATIONALE:**
- Original: 80ms (insufficient for ~90ms delays observed in data)
- Previous: 250ms (captures nearly all out-of-order messages)
- Current: 150ms (balances low latency with high temporal ordering accuracy)
- Trade-off: Latency (150ms delay) vs. timestamp quality (near-perfect monotonic output)
- For real-time applications: can reduce further, accept some non-monotonic timestamps
- For recording quality: 150ms provides excellent temporal ordering

Timestamp Quality & Device Timing Preservation:
------------------------------------------------
**CRITICAL INSIGHT:**

The decode.py output may show ~20% non-monotonic timestamps, but this is EXPECTED
and NOT an error. These non-monotonic timestamps simply reflect BLE message arrival
order, NOT device timing errors. The timestamp VALUES are correct and preserve the
device's accurate 256 kHz clock timing.

**PIPELINE FLOW:**
  decode.py:  Processes messages in arrival order → ~20% non-monotonic (expected)
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
Three LSL streams are created:
- Muse_EEG: 8 channels at 256 Hz (EEG + AUX)
- Muse_ACCGYRO: 6 channels at 52 Hz (accelerometer + gyroscope)
- Muse_OPTICS: 16 channels at 64 Hz (PPG sensors)
- Muse_BATTERY: 1 channel at 1 Hz (battery percentage)

Each stream includes:
- Channel labels (from decode.py: EEG_CHANNELS, ACCGYRO_CHANNELS, OPTICS_CHANNELS)
- Nominal sampling rate (declared device rate)
- Data type (float32)
- Units (microvolts for EEG, a.u. for others)
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
import os
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import bleak
import numpy as np
from mne_lsl.lsl import StreamInfo, StreamOutlet, local_clock

from .decode import (ACCGYRO_CHANNELS, BATTERY_CHANNELS, EEG_CHANNELS,
                     OPTICS_CHANNELS, make_timestamps, parse_message)
from .muse import MuseS
from .utils import configure_lsl_api_cfg, get_utc_timestamp


@dataclass
class SensorStream:
    """Holds the LSL outlet and a buffer for a single sensor stream."""

    outlet: StreamOutlet
    buffer: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    # Track state for make_timestamps (wraparound, sample counter, etc.)
    base_time: Optional[float] = None
    wrap_offset: int = 0
    last_abs_tick: int = 0
    sample_counter: int = 0


def _create_lsl_outlets(device_name: str, device_id: str) -> Dict[str, SensorStream]:
    """Create all LSL outlets for the available sensor streams."""
    streams = {}

    # --- EEG Stream ---
    info_eeg = StreamInfo(
        name=f"Muse_EEG",
        stype="EEG",
        n_channels=len(EEG_CHANNELS),
        sfreq=256.0,
        dtype="float32",
        source_id=f"{device_id}_eeg",
    )
    desc_eeg = info_eeg.desc  # <-- Access as attribute (no parentheses)
    desc_eeg.append_child_value("manufacturer", "Muse")
    desc_eeg.append_child_value("model", "MuseS")
    desc_eeg.append_child_value("device", device_name)
    channels = desc_eeg.append_child("channels")
    for ch_name in EEG_CHANNELS:
        channels.append_child("channel").append_child_value("label", ch_name)
    streams["EEG"] = SensorStream(outlet=StreamOutlet(info_eeg))

    # --- ACCGYRO Stream ---
    info_accgyro = StreamInfo(
        name=f"Muse_ACCGYRO",
        stype="ACC_GYRO",
        n_channels=len(ACCGYRO_CHANNELS),
        sfreq=52.0,
        dtype="float32",
        source_id=f"{device_id}_accgyro",
    )
    desc_acc = info_accgyro.desc
    desc_acc.append_child_value("manufacturer", "Muse")
    desc_acc.append_child_value("model", "MuseS")
    desc_acc.append_child_value("device", device_name)
    channels_acc = desc_acc.append_child("channels")
    for ch_name in ACCGYRO_CHANNELS:
        channels_acc.append_child("channel").append_child_value("label", ch_name)
    streams["ACCGYRO"] = SensorStream(outlet=StreamOutlet(info_accgyro))

    # --- OPTICS Stream ---
    info_optics = StreamInfo(
        name=f"Muse_OPTICS",
        stype="PPG",
        n_channels=len(OPTICS_CHANNELS),
        sfreq=64.0,
        dtype="float32",
        source_id=f"{device_id}_optics",
    )
    desc_opt = info_optics.desc
    desc_opt.append_child_value("manufacturer", "Muse")
    desc_opt.append_child_value("model", "MuseS")
    desc_opt.append_child_value("device", device_name)
    channels_opt = desc_opt.append_child("channels")
    for ch_name in OPTICS_CHANNELS:
        channels_opt.append_child("channel").append_child_value("label", ch_name)
    streams["OPTICS"] = SensorStream(outlet=StreamOutlet(info_optics))

    # --- Battery Stream ---
    info_battery = StreamInfo(
        name=f"Muse_BATTERY",
        stype="Battery",
        n_channels=len(BATTERY_CHANNELS),
        sfreq=1.0 / 60.0,  # ~1 per minute
        dtype="float32",
        source_id=f"{device_id}_battery",
    )
    desc_batt = info_battery.desc
    desc_batt.append_child_value("manufacturer", "Muse")
    desc_batt.append_child_value("model", "MuseS")
    desc_batt.append_child_value("device", device_name)
    channels_batt = desc_batt.append_child("channels")
    for ch_name in BATTERY_CHANNELS:
        channels_batt.append_child("channel").append_child_value("label", ch_name)
    streams["BATTERY"] = SensorStream(outlet=StreamOutlet(info_battery))

    return streams


async def _stream_async(
    address: str,
    preset: str,
    duration: Optional[float] = None,
    raw_data_file: Optional[str] = None,
    verbose: bool = True,
):
    """Asynchronous context for BLE connection and LSL streaming."""

    # --- State for Online Drift Correction ---
    drift_pairs: List[Tuple[float, float]] = []
    drift_a: Optional[float] = None  # Intercept (offset)
    drift_b: float = 1.0  # Slope (skew)
    DRIFT_WINDOW_SIZE = 200  # Use last 200 packets
    DRIFT_MIN_SAMPLES = 20  # Need at least 20 samples to start regression

    # --- Other Stream State ---
    streams: Dict[str, SensorStream] = {}
    last_flush_time = 0.0
    FLUSH_INTERVAL = 0.2  # 200ms
    samples_sent = {"EEG": 0, "ACCGYRO": 0, "OPTICS": 0, "BATTERY": 0}

    def _queue_samples(sensor_type: str, data_array: np.ndarray, lsl_now: float):
        """
        Apply drift correction and buffer samples.

        Parameters
        ----------
        sensor_type : str
            The name of the sensor (e.g., "EEG").
        data_array : np.ndarray
            The array from make_timestamps, shape (n_samples, 1 + n_channels).
            Column 0 is device_time, remaining are sensor values.
        lsl_now : float
            The computer's LSL clock time when the BLE message was received.
        """
        nonlocal drift_pairs, drift_a, drift_b

        if data_array.size == 0 or data_array.shape[1] < 2:
            return  # No data in this packet

        stream = streams.get(sensor_type)
        if stream is None:
            return  # No LSL outlet for this type

        # Extract device timestamps (relative to t=0 from make_timestamps)
        device_times = data_array[:, 0]
        samples = data_array[:, 1:]

        # --- Drift Correction ---

        # Use the first sample of this packet to update the regression
        drift_pairs.append((device_times[0], lsl_now))
        if len(drift_pairs) > DRIFT_WINDOW_SIZE:
            drift_pairs = drift_pairs[-DRIFT_WINDOW_SIZE:]

        if len(drift_pairs) >= DRIFT_MIN_SAMPLES:
            # We have enough data, perform linear regression
            # Model: lsl_time = b * device_time + a
            dev = np.array([p for p, _ in drift_pairs])
            lsl = np.array([_ for _, p in drift_pairs])

            # Demean for numerical stability
            xm = dev.mean()
            ym = lsl.mean()
            dev_dm = dev - xm
            lsl_dm = lsl - ym

            # Calculate slope (b) and intercept (a)
            b_num = np.dot(dev_dm, lsl_dm)
            b_den = np.dot(dev_dm, dev_dm)

            if b_den > 1e-9:  # Avoid division by zero
                drift_b = b_num / b_den
                drift_a = ym - (drift_b * xm)

                # Safety check for wildly incorrect fits
                if not (0.9 < drift_b < 1.1):
                    # Something is wrong, reset to offset-only
                    if verbose:
                        print(
                            f"Warning: Unstable drift fit (b={drift_b:.4f}). Resetting."
                        )
                    drift_a = None  # This will trigger re-calibration below
                    drift_b = 1.0
                    drift_pairs = []  # Clear bad data
            else:
                # All device times were identical, can't fit. Keep old values.
                pass

        if drift_a is None:
            # We don't have enough samples for regression OR fit failed
            # Set the initial offset (a) and use b=1.0
            drift_a = lsl_now - device_times[0]
            drift_b = 1.0

        # Apply the correction: lsl_timestamps = a + (b * device_times)
        lsl_timestamps = drift_a + (drift_b * device_times)

        # Add to this sensor's buffer
        stream.buffer.append((lsl_timestamps, samples))

    def _flush_buffer():
        """Sort and push all buffered samples to LSL."""
        nonlocal last_flush_time, samples_sent  # noqa: F824
        last_flush_time = time.monotonic()

        for sensor_type, stream in streams.items():
            if not stream.buffer:
                continue

            # Unzip buffer into lists of (timestamps, samples)
            try:
                all_timestamps = np.concatenate([ts for ts, _ in stream.buffer])
                all_samples = np.concatenate([s for _, s in stream.buffer])
            except ValueError:
                stream.buffer.clear()
                continue  # Skip if buffer was emptied by another flush

            stream.buffer.clear()

            # Sort by LSL timestamp to ensure correct order
            sort_indices = np.argsort(all_timestamps)
            sorted_timestamps = all_timestamps[sort_indices]
            sorted_data = all_samples[sort_indices, :]

            # Get the LSL timestamps for push_chunk
            # We must use the 'timestamp' argument here
            anchored_timestamps = sorted_timestamps.astype(np.float64, copy=False)

            # Push the chunk to LSL
            try:
                stream.outlet.push_chunk(
                    x=sorted_data.astype(np.float32, copy=False),
                    timestamp=anchored_timestamps,
                    pushThrough=True,
                )
                samples_sent[sensor_type] += len(sorted_data)
            except Exception as e:
                if verbose:
                    print(f"Error pushing LSL chunk for {sensor_type}: {e}")

    def _on_data(_, data: bytearray):
        """Main data callback from Bleak."""
        ts = get_utc_timestamp()  # Get system timestamp once
        message = f"{ts}\t{MuseS.EEG_UUID}\t{data.hex()}"

        # --- Optional: Write raw data to file ---
        if raw_data_file:
            try:
                raw_data_file.write(message + "\n")
            except Exception as e:
                if verbose:
                    print(f"Error writing to raw data file: {e}")

        # --- Decode all subpackets in the message ---
        subpackets = parse_message(message)
        decoded = {}
        for sensor_type, pkt_list in subpackets.items():
            stream = streams.get(sensor_type)
            if stream:
                # 1. Get current state for this sensor
                current_state = (
                    stream.base_time,
                    stream.wrap_offset,
                    stream.last_abs_tick,
                    stream.sample_counter,
                )

                # 2. Call make_timestamps (This creates the t=0 relative device_time)
                array, base_time, wrap_offset, last_abs_tick, sample_counter = (
                    make_timestamps(pkt_list, *current_state)
                )
                decoded[sensor_type] = array

                # 3. Update state
                stream.base_time = base_time
                stream.wrap_offset = wrap_offset
                stream.last_abs_tick = last_abs_tick
                stream.sample_counter = sample_counter

        # --- Queue Samples with Drift Correction ---
        # Get LSL clock time *once* for this entire BLE message
        lsl_now = local_clock()

        # Queue all decoded sensor data
        _queue_samples("EEG", decoded.get("EEG", np.empty((0, 0))), lsl_now)
        _queue_samples("ACCGYRO", decoded.get("ACCGYRO", np.empty((0, 0))), lsl_now)
        _queue_samples("OPTICS", decoded.get("OPTICS", np.empty((0, 0))), lsl_now)
        _queue_samples("BATTERY", decoded.get("BATTERY", np.empty((0, 0))), lsl_now)

        # --- Flush buffer if needed ---
        if time.monotonic() - last_flush_time > FLUSH_INTERVAL:
            _flush_buffer()

    # --- Main connection logic ---
    if verbose:
        print(f"Connecting to {address}...")

    async with bleak.BleakClient(address, timeout=15.0) as client:
        if verbose:
            print(f"Connected. Device: {client.name}")

        # Create LSL outlets
        streams = _create_lsl_outlets(client.name, address)
        if verbose:
            print("LSL outlets created:")
            for s in streams.values():
                print(
                    f"  - {s.outlet.get_info().name()} "
                    f"({s.outlet.get_info().channel_count()} channels)"
                )

        # Subscribe to data and configure device
        data_callbacks = {MuseS.EEG_UUID: _on_data}
        await MuseS.connect_and_initialize(
            client, preset, data_callbacks, verbose=verbose
        )

        if verbose:
            print("Streaming data... (Press Ctrl+C to stop)")

        # --- Main streaming loop ---
        start_time = time.monotonic()
        while True:
            await asyncio.sleep(0.5)  # Main loop sleep
            # Check duration
            if duration and (time.monotonic() - start_time) > duration:
                if verbose:
                    print(f"Streaming duration ({duration}s) elapsed.")
                break
            # Flush buffer one last time if connection is lost
            if not client.is_connected:
                if verbose:
                    print("Client disconnected.")
                break

        # --- Shutdown ---
        _flush_buffer()  # Final flush
        if verbose:
            print(
                "Stream stopped. "
                + ", ".join(
                    f"{sensor}: {count} samples"
                    for sensor, count in samples_sent.items()
                )
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
        If a string is provided, use it as the filename.
    verbose : bool
        If True (default), print connection and status messages.
    """
    # Configure MNE-LSL
    configure_lsl_api_cfg()

    # Handle 'record' argument
    raw_data_file = None
    file_handle = None
    if record:
        if isinstance(record, str):
            filename = record
        else:
            filename = f"rawdata_stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        try:
            file_handle = open(filename, "w", encoding="utf-8")
            raw_data_file = file_handle
            if verbose:
                print(f"Recording raw data to: {filename}")
        except IOError as e:
            print(f"Warning: Could not open file for recording: {e}")

    # --- Run the main asynchronous streaming loop ---
    try:
        asyncio.run(_stream_async(address, preset, duration, raw_data_file, verbose))
    except KeyboardInterrupt:
        if verbose:
            print("Streaming stopped by user.")
    except bleak.BleakError as e:
        print(f"BLEAK Error: {e}")
        print(
            "This may be a connection issue. Ensure the device is charged and nearby."
        )
        print("If on Linux, you may need to run with 'sudo' or set permissions.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if file_handle:
            file_handle.close()
            if verbose:
                print("Raw data file closed.")
