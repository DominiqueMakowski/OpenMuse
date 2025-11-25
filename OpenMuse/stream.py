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
   - **Timestamps already in LSL time domain** (mapped via StableClock)
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
- Muse_BATTERY: 1 channel at 1 Hz (battery percentage)

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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, TextIO
from collections import deque
import bleak
from bleak.exc import BleakError
import numpy as np
from mne_lsl.lsl import StreamInfo, StreamOutlet, local_clock

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


class StableClock:
    """
    A physics-constrained RLS filter for clock synchronization.

    Unlike a standard linear regression, this filter has a strong prior belief
    that the clock drift (slope) is 1.0. This prevents 'buffer bloat' or
    latency spikes from being misinterpreted as clock skew.

    Model:
        lsl_time = intercept + (slope * device_time)

    State:
        theta = [slope, intercept]
        P     = Covariance matrix
    """

    def __init__(self, forgetting_factor: float = 0.9995):
        self.lam = forgetting_factor
        self.dim = 2

        # Initialize State: Slope=1.0, Intercept=0.0
        self.theta = np.array([1.0, 0.0])

        # Initialize Covariance:
        # P[0,0] (Slope Variance): Set low (0.01).
        #   -> We are quite sure the Muse clock is running at real-time speed.
        # P[1,1] (Intercept Variance): Set high (1000).
        #   -> We have no idea what the initial offset is.
        self.P = np.zeros((2, 2))
        self.P[0, 0] = 0.01
        self.P[1, 1] = 1000.0

        self.initialized = False
        self.first_device_time = 0.0

    def update(self, device_time: float, lsl_now: float):
        """
        Update the clock model with a new measurement pair.
        """
        # On first packet, just snap the intercept to the current offset
        if not self.initialized:
            self.first_device_time = device_time
            self.theta[1] = lsl_now - device_time
            self.initialized = True
            return

        # RLS Update (Joseph form for stability)
        # Input vector x = [device_time, 1.0]
        x = np.array([device_time, 1.0]).reshape(-1, 1)

        # Prediction
        y_pred = float(x.T @ self.theta)
        error = lsl_now - y_pred

        # Gain Calculation
        Px = self.P @ x
        den = float(self.lam + (x.T @ Px))
        k = Px / den

        # State Update
        self.theta = self.theta + (k * error).flatten()

        # Covariance Update
        I = np.eye(self.dim)
        KX = k @ x.T
        self.P = (I - KX) @ self.P @ (I - KX).T + (k @ k.T) * 1e-12
        self.P /= self.lam

        # Safety Clamp: Ensure slope stays physically realistic (0.9 < slope < 1.1)
        self.theta[0] = np.clip(self.theta[0], 0.9, 1.1)

    def map_time(self, device_times: np.ndarray) -> np.ndarray:
        """Transform device timestamps to LSL time using current model."""
        if not self.initialized:
            # Fallback if called before update
            return device_times

        slope, intercept = self.theta
        return intercept + (slope * device_times)


class WindowedClock:
    """
    A robust clock synchronizer using a sliding window.

    It fits a linear regression (Time_LSL = slope * Time_Device + intercept)
    over a history window (e.g., 30 seconds).
    """

    def __init__(self, window_len_sec: float = 30.0):
        self.window_len = window_len_sec
        self.history = deque()  # Stores (device_time, lsl_time)

        # Current model state [slope, intercept]
        # Default to slope=1.0, intercept=0.0 until initialized
        self.slope = 1.0
        self.intercept = 0.0
        self.initialized = False

        # Optimization: Don't re-fit on every single packet
        self.last_fit_time = 0.0
        self.fit_interval = 1.0  # Re-calculate fit once per second

    def update(self, device_time: float, lsl_now: float):
        """Add a new time measurement and update the model."""

        # 1. Add new point
        self.history.append((device_time, lsl_now))

        # 2. Prune old history (keep only window_len seconds)
        # We assume monotonic time; prune from left
        limit = device_time - self.window_len
        while self.history and self.history[0][0] < limit:
            self.history.popleft()

        # 3. Fit model (only periodically to save CPU)
        if not self.initialized or (lsl_now - self.last_fit_time) > self.fit_interval:
            self._fit()
            self.last_fit_time = lsl_now
            self.initialized = True

    def _fit(self):
        """Perform linear regression on the history buffer."""
        n = len(self.history)
        if n < 10:
            return  # Not enough data yet

        # Convert to numpy for fast vectorized math
        data = np.array(self.history)
        x = data[:, 0]  # Device Time
        y = data[:, 1]  # LSL Time

        # Robustness: We ideally want to fit to the 'fastest' packets (lowest latency)
        # But a simple linear fit on the mean is usually sufficient for Muse.
        # We center x to improve numerical stability of polyfit
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        # Fit line: y = mx + c
        # Slope m = sum((x - x_mean)(y - y_mean)) / sum((x - x_mean)^2)
        x_centered = x - x_mean
        y_centered = y - y_mean

        denom = np.sum(x_centered**2)
        if denom < 1e-9:
            return  # Avoid division by zero if all timestamps are identical

        self.slope = np.sum(x_centered * y_centered) / denom
        self.intercept = y_mean - (self.slope * x_mean)

    def map_time(self, device_times: np.ndarray) -> np.ndarray:
        """Transform device timestamps to LSL time using current model."""
        if not self.initialized:
            # Fallback for first few packets: just offset by current diff
            if len(self.history) > 0:
                dt, lt = self.history[-1]
                return device_times + (lt - dt)
            return device_times

        return self.intercept + (self.slope * device_times)


@dataclass
class SensorStream:
    """Holds the LSL outlet and a buffer for a single sensor stream."""

    outlet: StreamOutlet
    buffer: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)

    # Track state for make_timestamps
    base_time: Optional[float] = None
    wrap_offset: int = 0
    last_abs_tick: int = 0
    sample_counter: int = 0

    # --- Stable Clock Sync ---
    clock: WindowedClock = field(default_factory=WindowedClock)
    # clock: StableClock = field(default_factory=StableClock)
    last_update_device_time: float = -1.0


def create_stream_outlet(
    sensor_type: str, n_channels: int, device_name: str, device_id: str
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
        stype = "ACC_GYRO"
        source_id = f"{device_id}_accgyro"
    elif sensor_type == "OPTICS":
        ch_names = select_optics_channels(n_channels)
        sfreq = 64.0
        stype = "PPG"
        source_id = f"{device_id}_optics"
    elif sensor_type == "BATTERY":
        ch_names = list(BATTERY_CHANNELS)
        sfreq = 1.0
        stype = "Battery"
        source_id = f"{device_id}_battery"
    else:
        raise ValueError(f"Unknown sensor type: {sensor_type}")

    info = StreamInfo(
        name=f"Muse_{sensor_type}",
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

    return SensorStream(outlet=StreamOutlet(info))


async def _stream_async(
    address: str,
    preset: str,
    duration: Optional[float] = None,
    raw_data_file: Optional[TextIO] = None,
    verbose: bool = True,
):
    """Asynchronous context for BLE connection and LSL streaming."""

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

        # --- Update Clock Model ---
        # We update the clock using the *latest* packet in this chunk
        last_device_time = device_times[-1]

        # Only update if time moved forward (avoids issues with out-of-order arrival for model update)
        if last_device_time > stream.last_update_device_time:
            stream.clock.update(last_device_time, lsl_now)
            stream.last_update_device_time = last_device_time

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

        # Ensure streams exist
        for sensor_type, pkt_list in subpackets.items():
            if pkt_list and sensor_type not in streams:
                n_channels = pkt_list[0].get("n_channels")
                if n_channels:
                    streams[sensor_type] = create_stream_outlet(
                        sensor_type, n_channels, client.name, address
                    )

        # Decode & Make Timestamps (Relative Device Time)
        for sensor_type, pkt_list in subpackets.items():
            stream = streams.get(sensor_type)
            if stream:
                current_state = (
                    stream.base_time,
                    stream.wrap_offset,
                    stream.last_abs_tick,
                    stream.sample_counter,
                )
                array, base_time, wrap_offset, last_abs_tick, sample_counter = (
                    make_timestamps(pkt_list, *current_state)
                )
                decoded[sensor_type] = array

                # Update state
                stream.base_time = base_time
                stream.wrap_offset = wrap_offset
                stream.last_abs_tick = last_abs_tick
                stream.sample_counter = sample_counter

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
    address: str,
    preset: str = "p1041",
    duration: Optional[float] = None,
    record: Union[bool, str] = False,
    verbose: bool = True,
) -> None:
    """
    Stream decoded EEG and accelerometer/gyroscope data over LSL.
    """
    configure_lsl_api_cfg()

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
        except IOError as e:
            print(f"Warning: Could not open file for recording: {e}")

    try:
        asyncio.run(_stream_async(address, preset, duration, raw_data_file, verbose))
    except KeyboardInterrupt:
        if verbose:
            print("Streaming stopped by user.")
    except BleakError as e:
        print(f"BLEAK Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if file_handle:
            file_handle.close()
