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

MAX_BUFFER_PACKETS = 52  # safety limit for buffered packets


class _RLSFilter:
    """
    Recursive Least Squares (RLS) filter for online clock drift.

    Model: lsl_time = a + b * device_time  (theta = [b, a])
    """

    def __init__(self, dim: int = 2, lam: float = 0.9999, P_init: float = 1e6):
        self.dim = dim
        self.lam = lam  # forgetting factor
        self.P_init = P_init
        self.theta = np.array([1.0, 0.0], dtype=float)  # [b, a]
        self.P = np.eye(self.dim, dtype=float) * self.P_init

    def reset(self, lam: Optional[float] = None, P_init: Optional[float] = None):
        """Reset filter to default or provided values."""
        if lam is not None:
            self.lam = lam
        if P_init is not None:
            self.P_init = P_init
        self.theta = np.array([1.0, 0.0], dtype=float)
        self.P = np.eye(self.dim, dtype=float) * self.P_init

    def update(self, y: float, x: np.ndarray):
        """
        Numerically-stable RLS update using Joseph form to preserve symmetry.

        Parameters
        ----------
        y : float
            Scalar observation (lsl_now).
        x : np.ndarray
            1D array [device_time, 1.0]
        """
        x = x.reshape(-1, 1).astype(float)  # column vector
        P_x = self.P @ x
        denom = float(self.lam + (x.T @ P_x))
        k = P_x / denom  # gain vector

        y_pred = float((x.T @ self.theta).item())
        e = float(y - y_pred)

        # update theta
        self.theta = (self.theta.reshape(-1, 1) + k * e).flatten()

        # Joseph form for P update (numeric stability)
        I = np.eye(self.dim, dtype=float)
        KX = k @ x.T
        self.P = (I - KX) @ self.P @ (I - KX).T + (k @ k.T) * 1e-12
        # apply forgetting factor
        self.P /= self.lam

        # keep P symmetric numerically
        self.P = (self.P + self.P.T) / 2.0


@dataclass
class SensorStream:
    """Container for per-sensor LSL outlet and buffering/state."""

    outlet: StreamOutlet
    buffer: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    base_time: Optional[float] = None
    wrap_offset: int = 0
    last_abs_tick: int = 0
    sample_counter: int = 0
    drift_filter: _RLSFilter = field(
        default_factory=lambda: _RLSFilter(dim=2, lam=0.9999, P_init=1e6)
    )
    drift_initialized: bool = False
    last_update_device_time: float = 0.0


def _create_lsl_outlets(device_name: str, device_id: str) -> Dict[str, SensorStream]:
    """Create LSL outlets for EEG, ACC+GYRO, OPTICS and BATTERY."""
    streams: Dict[str, SensorStream] = {}

    # EEG
    info_eeg = StreamInfo(
        name="Muse_EEG",
        stype="EEG",
        n_channels=len(EEG_CHANNELS),
        sfreq=256.0,
        dtype="float32",
        source_id=f"{device_id}_eeg",
    )
    desc_eeg = info_eeg.desc
    desc_eeg.append_child_value("manufacturer", "Muse")
    desc_eeg.append_child_value("model", "MuseS")
    desc_eeg.append_child_value("device", device_name)
    channels = desc_eeg.append_child("channels")
    for ch in EEG_CHANNELS:
        channels.append_child("channel").append_child_value("label", ch)
    streams["EEG"] = SensorStream(outlet=StreamOutlet(info_eeg))

    # ACCGYRO
    info_accgyro = StreamInfo(
        name="Muse_ACCGYRO",
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
    ch_acc = desc_acc.append_child("channels")
    for ch in ACCGYRO_CHANNELS:
        ch_acc.append_child("channel").append_child_value("label", ch)
    streams["ACCGYRO"] = SensorStream(outlet=StreamOutlet(info_accgyro))

    # OPTICS
    info_optics = StreamInfo(
        name="Muse_OPTICS",
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
    ch_opt = desc_opt.append_child("channels")
    for ch in OPTICS_CHANNELS:
        ch_opt.append_child("channel").append_child_value("label", ch)
    streams["OPTICS"] = SensorStream(outlet=StreamOutlet(info_optics))

    # BATTERY
    info_batt = StreamInfo(
        name="Muse_BATTERY",
        stype="Battery",
        n_channels=len(BATTERY_CHANNELS),
        sfreq=1.0,
        dtype="float32",
        source_id=f"{device_id}_battery",
    )
    desc_batt = info_batt.desc
    desc_batt.append_child_value("manufacturer", "Muse")
    desc_batt.append_child_value("model", "MuseS")
    desc_batt.append_child_value("device", device_name)
    ch_batt = desc_batt.append_child("channels")
    for ch in BATTERY_CHANNELS:
        ch_batt.append_child("channel").append_child_value("label", ch)
    streams["BATTERY"] = SensorStream(outlet=StreamOutlet(info_batt))

    return streams


async def _stream_async(
    address: str,
    preset: str,
    duration: Optional[float] = None,
    raw_data_file: Optional[str] = None,
    verbose: bool = True,
):
    """Main async BLE <-> LSL streaming logic."""

    streams: Dict[str, SensorStream] = {}
    last_flush_time = time.monotonic()
    FLUSH_INTERVAL = 0.2  # seconds
    samples_sent = {"EEG": 0, "ACCGYRO": 0, "OPTICS": 0, "BATTERY": 0}

    def _queue_samples(sensor_type: str, data_array: np.ndarray, lsl_now: float):
        """
        Apply drift correction and append samples to per-sensor buffer.

        data_array shape: (n_samples, 1 + n_channels)
        column 0: device_time ; remaining columns: sensor channels
        """
        if data_array is None:
            return
        # Ensure array shape is expected
        if data_array.size == 0 or data_array.ndim != 2 or data_array.shape[1] < 2:
            return

        stream = streams.get(sensor_type)
        if stream is None:
            return

        # Extract device timestamps and samples
        device_times = data_array[:, 0].astype(float)
        samples = data_array[:, 1:]

        # Representative device time for this packet - centre is robust to late arrivals
        device_time_center = float(device_times.mean())

        prev_last_update = stream.last_update_device_time
        drift_filter = stream.drift_filter

        if not stream.drift_initialized:
            # initialise RLS with a sensible offset
            initial_a = float(lsl_now - device_time_center)
            drift_filter.theta = np.array([1.0, initial_a], dtype=float)
            stream.drift_initialized = True
            stream.last_update_device_time = device_time_center
        else:
            # Minimum device-time advance to avoid duplicates / near-duplicates
            MIN_DT = 0.005  # seconds
            dt = device_time_center - prev_last_update
            if dt > MIN_DT:
                x_vec = np.array([device_time_center, 1.0], dtype=float)
                y_val = float(lsl_now)

                # residual gating to reject host jitter / outliers
                y_pred = float(np.dot(x_vec, drift_filter.theta))
                residual = abs(y_val - y_pred)
                MAX_RESIDUAL = 0.2  # seconds

                if residual < MAX_RESIDUAL:
                    try:
                        drift_filter.update(y=y_val, x=x_vec)
                        stream.last_update_device_time = device_time_center
                    except Exception:
                        # numerical error - reset filter and reinitialise on next packet
                        drift_filter.reset()
                        stream.drift_initialized = False
                else:
                    # reject the update but advance the last_update to avoid repeated rejects
                    stream.last_update_device_time = device_time_center

        # Obtain b,a
        drift_b, drift_a = drift_filter.theta

        # Safety bounds on slope - if violated, reset robustly
        if not (0.5 < drift_b < 1.5):
            if verbose:
                time_diff = device_time_center - prev_last_update
                print(f"--- DEBUG: UNSTABLE FIT DETECTED ({sensor_type}) ---")
                print(f"    New Slope (b): {drift_b:.6f}")
                print(f"    New Intercept (a): {drift_a:.6f}")
                print(f"    Input LSL Time (y): {lsl_now:.6f}")
                print(f"    Input Device Time (x): {device_time_center:.6f}")
                print(f"    Last Device Time: {prev_last_update:.6f}")
                print(f"    Time Diff (New - Last): {time_diff:.3f}s")
                print(
                    f"Warning: Unstable drift fit for {sensor_type} (b={drift_b:.4f}). Resetting filter."
                )
            # reset filter and use simple offset for this packet
            drift_filter.reset()
            stream.drift_initialized = False
            # use device_time_center (safe, defined) instead of undefined name
            drift_b = 1.0
            drift_a = float(lsl_now - device_time_center)

        # Convert device times -> lsl times
        lsl_timestamps = drift_a + (drift_b * device_times)

        # Append chunk to buffer (timestamp array, sample array)
        stream.buffer.append((lsl_timestamps, samples))

    def _flush_buffer():
        """Concatenate, sort by timestamp and push each sensor's buffered samples."""
        nonlocal last_flush_time, samples_sent
        last_flush_time = time.monotonic()

        for sensor_type, stream in streams.items():
            if not stream.buffer:
                continue

            # Build concatenated arrays efficiently
            try:
                ts_list = [ts for ts, _ in stream.buffer]
                data_list = [d for _, d in stream.buffer]
                all_timestamps = (
                    np.concatenate(ts_list) if len(ts_list) > 1 else ts_list[0]
                )
                all_samples = (
                    np.concatenate(data_list) if len(data_list) > 1 else data_list[0]
                )
            except Exception:
                stream.buffer.clear()
                continue

            stream.buffer.clear()

            # Sort by timestamps and push chunk
            sort_idx = np.argsort(all_timestamps)
            sorted_ts = all_timestamps[sort_idx]
            sorted_data = all_samples[sort_idx, :]

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message=".*A single sample is pushed.*"
                    )
                    stream.outlet.push_chunk(
                        x=sorted_data.astype(np.float32, copy=False),
                        timestamp=sorted_ts.astype(np.float64, copy=False),
                        pushThrough=True,
                    )
                samples_sent[sensor_type] += sorted_data.shape[0]
            except Exception as exc:
                if verbose:
                    print(f"Error pushing LSL chunk for {sensor_type}: {exc}")

    def _on_data(_, data: bytearray):
        """Callback for BLE incoming data provided to Bleak; decodes and queues samples."""
        ts = get_utc_timestamp()
        # Use the sender characteristic if needed; the callback is registered per-UUID.
        message = f"{ts}\t{MuseS.EEG_UUID}\t{data.hex()}"

        if raw_data_file:
            try:
                raw_data_file.write(message + "\n")
            except Exception as exc:
                if verbose:
                    print(f"Error writing raw data: {exc}")

        subpackets = parse_message(message)
        decoded: Dict[str, np.ndarray] = {}

        for sensor_type, pkt_list in subpackets.items():
            stream = streams.get(sensor_type)
            if not stream:
                continue

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
            stream.base_time = base_time
            stream.wrap_offset = wrap_offset
            stream.last_abs_tick = last_abs_tick
            stream.sample_counter = sample_counter

        # Compute a single LSL clock sample for this BLE message
        lsl_now = local_clock()

        # Queue decoded arrays for all streams (safely handle absent keys)
        _queue_samples("EEG", decoded.get("EEG", np.empty((0, 0))), lsl_now)
        _queue_samples("ACCGYRO", decoded.get("ACCGYRO", np.empty((0, 0))), lsl_now)
        _queue_samples("OPTICS", decoded.get("OPTICS", np.empty((0, 0))), lsl_now)
        _queue_samples("BATTERY", decoded.get("BATTERY", np.empty((0, 0))), lsl_now)

        # Flush buffer if interval exceeded or buffer too large
        time_flush = (time.monotonic() - last_flush_time) > FLUSH_INTERVAL
        size_flush = any(len(s.buffer) > MAX_BUFFER_PACKETS for s in streams.values())
        if time_flush or size_flush:
            _flush_buffer()

    # Connect and stream
    if verbose:
        print(f"Connecting to {address}...")

    async with bleak.BleakClient(address, timeout=15.0) as client:
        if verbose:
            print(f"Connected. Device: {client.name}")

        streams = _create_lsl_outlets(client.name, address)

        # Register callback per characteristic
        data_callbacks = {MuseS.EEG_UUID: _on_data}
        await MuseS.connect_and_initialize(
            client, preset, data_callbacks, verbose=verbose
        )

        if verbose:
            print("Streaming data... (Press Ctrl+C to stop)")

        start_time = time.monotonic()
        try:
            while True:
                await asyncio.sleep(0.5)
                if duration and (time.monotonic() - start_time) > duration:
                    if verbose:
                        print(f"Streaming duration ({duration}s) elapsed.")
                    break
                if not client.is_connected:
                    if verbose:
                        print("Client disconnected.")
                    break
        finally:
            _flush_buffer()
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
    """Public synchronous wrapper to start streaming (runs asyncio loop)."""
    configure_lsl_api_cfg()

    file_handle = None
    raw_data_file = None
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
        except IOError as exc:
            print(f"Warning: Could not open file for recording: {exc}")

    try:
        asyncio.run(_stream_async(address, preset, duration, raw_data_file, verbose))
    except KeyboardInterrupt:
        if verbose:
            print("Streaming stopped by user.")
    except bleak.BleakError as exc:
        print(f"BLEAK Error: {exc}")
        print(
            "This may be a connection issue. Ensure the device is charged and nearby."
        )
        print("If on Linux, you may need to run with 'sudo' or set permissions.")
    except Exception as exc:
        print(f"An unexpected error occurred: {exc}")
    finally:
        if file_handle:
            file_handle.close()
            if verbose:
                print("Raw data file closed.")
