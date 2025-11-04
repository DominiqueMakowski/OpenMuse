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
   - `a` (intercept) and `b` (slope/skew) are updated with every new packet
     using an efficient Recursive Least Squares (RLS) adaptive filter.
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

from .decode import (
    ACCGYRO_CHANNELS,
    BATTERY_CHANNELS,
    EEG_CHANNELS,
    OPTICS_CHANNELS,
    make_timestamps,
    parse_message,
)
from .muse import MuseS
from .utils import configure_lsl_api_cfg, get_utc_timestamp

MAX_BUFFER_PACKETS = 52  # 52 packets per sensor


class _RLSFilter:
    """
    Implements a Recursive Least Squares (RLS) filter for online clock drift.

    Models the linear relationship: y = X * theta
    Where:
      y = lsl_now
      X = [device_time, 1.0]
      theta = [b, a] (slope, intercept)
    """

    def __init__(self, dim: int, lam: float = 0.999, P_init: float = 0.1):
        self.dim = dim
        self.lam = lam  # Forgetting factor
        self.P_init = P_init  # Initial covariance
        # Initialize parameters [b, a] = [1.0, 0.0]
        self.theta = np.array([1.0, 0.0])
        # Initialize covariance matrix
        self.P = np.eye(self.dim) * self.P_init

    def reset(self, lam: Optional[float] = None, P_init: Optional[float] = None):
        """Reset the filter state."""
        if lam:
            self.lam = lam
        if P_init:
            self.P_init = P_init
        self.theta = np.array([1.0, 0.0])
        self.P = np.eye(self.dim) * self.P_init

    def update(self, y: float, x: np.ndarray):
        """
        Numerically-stable RLS update using Joseph form.
        y : scalar output (lsl_now)
        x : input vector shape (2,) corresponding to [device_time, 1.0]
        """
        x = x.reshape(-1, 1)  # column
        P_x = self.P @ x
        den = float(self.lam + (x.T @ P_x))  # scalar

        # gain
        k = P_x / den  # shape (dim,1)

        # prediction error
        y_pred = float(x.T @ self.theta)
        e = y - y_pred

        # update theta
        self.theta = self.theta + (k * e).flatten()

        # Joseph form for P update to preserve symmetry
        I = np.eye(self.dim)
        KX = k @ x.T
        self.P = (I - KX) @ self.P @ (I - KX).T + (k @ k.T) * 1e-12
        # apply forgetting factor
        self.P /= self.lam


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
    # --- ADDED: Per-stream state for online drift correction ---
    drift_filter: _RLSFilter = field(
        default_factory=lambda: _RLSFilter(dim=2, lam=0.9999, P_init=1e6)
    )
    drift_initialized: bool = False
    last_update_device_time: float = 0.0


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
        sfreq=1.0,
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

    # --- Other Stream State ---
    streams: Dict[str, SensorStream] = {}
    last_flush_time = 0.0
    FLUSH_INTERVAL = 0.2  # 200ms
    samples_sent = {"EEG": 0, "ACCGYRO": 0, "OPTICS": 0, "BATTERY": 0}

    def _queue_samples(sensor_type: str, data_array: np.ndarray, lsl_now: float):
        """
        Apply drift correction and buffer samples using a per-stream filter.

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
        if data_array.size == 0 or data_array.shape[1] < 2:
            return  # No data in this packet

        stream = streams.get(sensor_type)
        if stream is None:
            return  # No LSL outlet for this type

        # --- Get PER-STREAM filter state ---
        drift_filter = stream.drift_filter
        drift_initialized = stream.drift_initialized
        last_update_device_time = stream.last_update_device_time

        # Extract device timestamps (relative to t=0 from make_timestamps)
        device_times = data_array[:, 0]
        samples = data_array[:, 1:]

        # --- Drift Correction ---
        first_device_time = device_times[0]

        old_last_update_device_time = last_update_device_time

        if not drift_initialized:
            # Initialize this sensor's filter
            initial_a = lsl_now - first_device_time
            drift_filter.theta = np.array([1.0, initial_a])
            stream.drift_initialized = True  # Set state on the stream object
            stream.last_update_device_time = first_device_time
        else:
            # Only update filter if this packet is 'newer'
            # This prevents out-of-order packets from corrupting the model
            if first_device_time > last_update_device_time:
                # Update the filter with the new (device_time, lsl_now) pair
                # y = lsl_now, x = [dev_time, 1.0]
                drift_filter.update(y=lsl_now, x=np.array([first_device_time, 1.0]))
                stream.last_update_device_time = first_device_time
            # (Else: This is an out-of-order packet, ignore it for filter updates)

        # Get current model parameters [b, a]
        drift_b, drift_a = drift_filter.theta

        # Safety check: If filter diverges, reset it
        if not (0.5 < drift_b < 1.5):
            if (
                verbose and (lsl_now - start_time) > 5.0
            ):  # Suppress early warnings due to warmup
                # Calculate the *correct* time diff using the old value
                time_diff = first_device_time - old_last_update_device_time

                # Print a single, dense line with all key info
                print(
                    f"Warning: Unstable drift fit for {sensor_type}. Resetting filter. "
                    f"[Slope(b)={drift_b:.4f}, TimeDiff={time_diff:.3f}s, "
                    f"NewDevTime={first_device_time:.3f}, LastDevTime={old_last_update_device_time:.3f}]"
                )
            # Reset and re-initialize on the next packet
            drift_filter.reset()
            stream.drift_initialized = False  # Set state on the stream object
            # Use a simple offset for this packet
            drift_a = lsl_now - first_device_time
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

            # Push the chunk to LSL
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=".*A single sample is pushed.*",
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

        # --- Flush buffer if needed (by time OR size) ---
        # Check time interval
        time_flush = time.monotonic() - last_flush_time > FLUSH_INTERVAL

        # Check size
        size_flush = False
        for stream in streams.values():
            if len(stream.buffer) > MAX_BUFFER_PACKETS:
                size_flush = True
                break

        if time_flush or size_flush:
            _flush_buffer()

    # --- Main connection logic ---
    if verbose:
        print(f"Connecting to {address}...")

    async with bleak.BleakClient(address, timeout=15.0) as client:
        if verbose:
            print(f"Connected. Device: {client.name}")

        # Create LSL outlets
        streams = _create_lsl_outlets(client.name, address)
        start_time = time.monotonic()

        # Subscribe to data and configure device
        data_callbacks = {MuseS.EEG_UUID: _on_data}
        await MuseS.connect_and_initialize(
            client, preset, data_callbacks, verbose=verbose
        )

        if verbose:
            print("Streaming data... (Press Ctrl+C to stop)")

        # --- Main streaming loop ---
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
