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
   - This value is used internally for drift correction.

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
   - **Timestamps already in LSL time domain** (no conversion needed)
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
- Channel labels (from decode.py: EEG_CHANNELS, ACCGYRO_CHANNELS, OPTICS_CHANNELS)
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

"""
Muse BLE to LSL Streaming
==========================

( ... Omitted docstring for brevity ... )
"""

"""
Muse BLE to LSL Streaming
==========================

LSL Stream Configuration:
-------------------------
Four LSL streams are created:
- Muse_EEG
- Muse_ACCGYRO
- Muse_OPTICS
- Muse_BATTERY
The channel count for EEG and OPTICS is now determined dynamically by the first
incoming data packet from the Muse device.
"""

"""
Muse BLE to LSL Streaming
==========================

( ... Omitted docstring for brevity ... )
"""

"""
Muse BLE to LSL Streaming
==========================

LSL Stream Configuration:
-------------------------
Four LSL streams are created:
- Muse_EEG
- Muse_ACCGYRO
- Muse_OPTICS
- Muse_BATTERY
The channel count for EEG and OPTICS is now determined dynamically by the first
incoming data packet from the Muse device.
"""

import asyncio
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
    make_timestamps,
    parse_message,
)
from .muse import MuseS
from .utils import configure_lsl_api_cfg, get_utc_timestamp

# --- Local Channel Definitions ---
# These lists are used for naming channels once the count is known.
_FULL_EEG_CHANNELS = (
    "EEG_TP9",
    "EEG_AF7",
    "EEG_AF8",
    "EEG_TP10",
    "AUX_1",
    "AUX_2",
    "AUX_3",
    "AUX_4",
)

_FULL_OPTICS_CHANNELS = (
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

# Map specific counts to the correct indices from the full list
_OPTICS_INDEXES = {
    4: (4, 5, 6, 7),
    8: tuple(range(8)),
    16: tuple(range(16)),
}


def _select_eeg_channels(count: int) -> List[str]:
    """Select the correct EEG channel labels based on the actual count."""
    if count in (4, 8) and count <= len(_FULL_EEG_CHANNELS):
        return list(_FULL_EEG_CHANNELS[:count])
    # Generic fallback for any other count
    return [f"EEG_{i+1:02d}" for i in range(count)]


def _select_optics_channels(count: int) -> List[str]:
    """Select the correct OPTICS channel labels based on the actual count."""
    indices = _OPTICS_INDEXES.get(count)
    if indices is not None:
        return [_FULL_OPTICS_CHANNELS[i] for i in indices]

    # Generic fallback for any other count
    return [f"OPTICS_{i+1:02d}" for i in range(count)]


MAX_BUFFER_PACKETS = 52  # 52 packets per sensor
FLUSH_INTERVAL = 0.2  # 200ms


class RLSFilter:
    """
    Implements a Recursive Least Squares (RLS) filter for online clock drift.

    ( ... Omitted RLSFilter class for brevity ... )
    """

    def __init__(self, dim: int, lam: float, P_init: float):
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

    # Outlet is now optional and will be created later
    outlet: Optional[StreamOutlet] = None
    buffer: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    # Track state for make_timestamps (wraparound, sample counter, etc.)
    base_time: Optional[float] = None
    wrap_offset: int = 0
    last_abs_tick: int = 0
    sample_counter: int = 0
    # --- Per-stream state for online drift correction ---
    drift_filter: RLSFilter = field(
        default_factory=lambda: RLSFilter(dim=2, lam=0.9999, P_init=1e6)
    )
    drift_initialized: bool = False
    last_update_device_time: float = 0.0


def _create_lsl_outlets_initial(
    device_name: str,
    device_id: str,
) -> Dict[str, SensorStream]:
    """
    Initialize sensor stream objects, creating fixed LSL outlets immediately
    and placeholders for dynamic ones.
    """
    streams = {}

    # Initialize placeholders for streams where channel count is unknown (EEG, OPTICS)
    streams["EEG"] = SensorStream()
    streams["OPTICS"] = SensorStream()

    # --- ACCGYRO Stream (Fixed 6 channels, 52 Hz) ---
    sensor_type = "ACCGYRO"
    n_channels = len(ACCGYRO_CHANNELS)
    sfreq = 52.0
    info_accgyro = StreamInfo(
        name=f"Muse_{sensor_type}",
        stype="ACC_GYRO",
        n_channels=n_channels,
        sfreq=sfreq,
        dtype="float32",
        source_id=f"{device_id}_accgyro",
    )
    desc_accgyro = info_accgyro.desc
    desc_accgyro.append_child_value("manufacturer", "Muse")
    desc_accgyro.append_child_value("model", "MuseS")
    desc_accgyro.append_child_value("device", device_name)
    channels_accgyro = desc_accgyro.append_child("channels")
    for ch_name in ACCGYRO_CHANNELS:
        channels_accgyro.append_child("channel").append_child_value("label", ch_name)

    outlet_accgyro = StreamOutlet(info_accgyro)
    streams[sensor_type] = SensorStream(outlet=outlet_accgyro)
    # Removed: print(f"✅ LSL Outlet created for {sensor_type}: {n_channels} channels at {sfreq} Hz.")


    # --- Battery Stream (Fixed 1 channel, 1 Hz) ---
    sensor_type = "BATTERY"
    n_channels = len(BATTERY_CHANNELS)
    sfreq = 1.0
    info_battery = StreamInfo(
        name=f"Muse_{sensor_type}",
        stype="Battery",
        n_channels=n_channels,
        sfreq=sfreq,
        dtype="float32",
        source_id=f"{device_id}_battery",
    )
    desc_battery = info_battery.desc
    desc_battery.append_child_value("manufacturer", "Muse")
    desc_battery.append_child_value("model", "MuseS")
    desc_battery.append_child_value("device", device_name)
    channels_battery = desc_battery.append_child("channels")
    for ch_name in BATTERY_CHANNELS:
        channels_battery.append_child("channel").append_child_value("label", ch_name)

    outlet_battery = StreamOutlet(info_battery)
    streams[sensor_type] = SensorStream(outlet=outlet_battery)
    # Removed: print(f"✅ LSL Outlet created for {sensor_type}: {n_channels} channel at {sfreq} Hz.")

    return streams


def _create_dynamic_outlet(stream: SensorStream, sensor_type: str, device_name: str, device_id: str, n_channels: int, verbose: bool):
    """Creates the LSL StreamOutlet for a sensor type whose channel count is now known."""

    if sensor_type == "EEG":
        labels = _select_eeg_channels(n_channels)
        stype = "EEG"
        sfreq = 256.0
    elif sensor_type == "OPTICS":
        labels = _select_optics_channels(n_channels)
        stype = "PPG"
        sfreq = 64.0
    else:
        raise ValueError(f"Cannot dynamically create outlet for unknown type: {sensor_type}")

    info = StreamInfo(
        name=f"Muse_{sensor_type}",
        stype=stype,
        n_channels=n_channels,
        sfreq=sfreq,
        dtype="float32",
        source_id=f"{device_id}_{sensor_type.lower()}",
    )

    desc = info.desc
    desc.append_child_value("manufacturer", "Muse")
    desc.append_child_value("model", "MuseS")
    desc.append_child_value("device", device_name)
    channels = desc.append_child("channels")
    for ch_name in labels:
        channels.append_child("channel").append_child_value("label", ch_name)

    stream.outlet = StreamOutlet(info)
    # Removed: if verbose: print(f"✅ LSL Outlet created for {sensor_type}: {n_channels} channels at {sfreq} Hz.")


async def _stream_async(
    address: str,
    preset: str,
    duration: Optional[float] = None,
    raw_data_file: Optional[str] = None,
    verbose: bool = True,
):
    """Asynchronous context for BLE connection and LSL streaming."""

    # --- Stream State ---
    streams: Dict[str, SensorStream] = {}
    last_flush_time = 0.0
    samples_sent = {"EEG": 0, "ACCGYRO": 0, "OPTICS": 0, "BATTERY": 0}
    start_time = 0.0  # Will be set after connection
    device_name = ""  # Will be set after connection
    device_id = address.replace(":", "") # Used for source_id

    def _queue_samples(sensor_type: str, data_array: np.ndarray, lsl_now: float):
        """
        Apply drift correction, dynamically create outlet if needed, and buffer samples.
        """
        if data_array.size == 0:
            return

        stream = streams.get(sensor_type)
        if stream is None:
            return

        device_times = data_array[:, 0]
        samples = data_array[:, 1:]
        actual_channels = samples.shape[1]

        # --- Dynamic Outlet Creation (Only for EEG and OPTICS) ---
        if stream.outlet is None:
            # Check if this is one of the streams that needs dynamic creation
            if sensor_type in ["EEG", "OPTICS"]:
                try:
                    # Note: verbose is passed here, but the print statement inside
                    # _create_dynamic_outlet has been removed.
                    _create_dynamic_outlet(
                        stream, sensor_type, device_name, device_id, actual_channels, verbose
                    )
                except Exception as e:
                    if verbose:
                        # Retain non-emoji failure message
                        print(f"❌ Failed to create LSL Outlet for {sensor_type}: {e}")
                    return
            else:
                return

        # --- Validation ---
        expected_channels = stream.outlet.get_sinfo().n_channels

        # Allow EEG (4 or 8) and OPTICS (4, 8, or 16) to adapt dynamically
        if actual_channels != expected_channels:
            if sensor_type in ["EEG", "OPTICS"]:
                if verbose:
                    print(
                        f"⚙️ Detected change in {sensor_type} channel count: "
                        f"{expected_channels} → {actual_channels}. Recreating outlet..."
                    )
                # Recreate the outlet dynamically with the new count
                _create_dynamic_outlet(
                    stream, sensor_type, device_name, device_id, actual_channels, verbose
                )
                expected_channels = actual_channels
            else:
                if verbose:
                    print(
                        f"Warning: Channel mismatch for {sensor_type}! "
                        f"Expected {expected_channels}, got {actual_channels}. Dropping packet."
                    )
                return


        # --- Drift Correction ---
        drift_filter = stream.drift_filter
        drift_initialized = stream.drift_initialized
        last_update_device_time = stream.last_update_device_time

        last_device_time = device_times[-1]

        if not drift_initialized:
            initial_a = lsl_now - last_device_time
            drift_filter.theta = np.array([1.0, initial_a])
            stream.drift_initialized = True
            stream.last_update_device_time = last_device_time
            drift_b, drift_a = 1.0, initial_a
        else:
            prev_device_time = last_update_device_time
            if last_device_time > last_update_device_time:
                drift_filter.update(y=lsl_now, x=np.array([last_device_time, 1.0]))
                stream.last_update_device_time = last_device_time

            drift_b, drift_a = drift_filter.theta

            if not (0.5 < drift_b < 1.5) and (lsl_now - start_time) > 5.0:
                if verbose:
                    print(f"Warning: Unstable drift fit for {sensor_type}. Resetting filter.")
                drift_filter.reset()
                stream.drift_initialized = False
                drift_a = lsl_now - last_device_time
                drift_b = 1.0

        # Apply the correction: lsl_timestamps = a + (b * device_times)
        lsl_timestamps = drift_a + (drift_b * device_times)

        # Add to this sensor's buffer
        stream.buffer.append((lsl_timestamps, samples))

    def _flush_buffer():
        """Sort and push all buffered samples to LSL."""
        nonlocal last_flush_time, samples_sent
        last_flush_time = time.monotonic()

        for sensor_type, stream in streams.items():
            if not stream.buffer or stream.outlet is None:
                continue

            # Concatenate all buffered samples
            all_timestamps = np.concatenate([ts for ts, _ in stream.buffer])
            all_samples = np.concatenate([s for _, s in stream.buffer])
            stream.buffer.clear()

            # Sort by LSL timestamp to ensure correct order
            sort_indices = np.argsort(all_timestamps)
            sorted_timestamps = all_timestamps[sort_indices]
            sorted_data = all_samples[sort_indices, :]

            # Push the chunk to LSL
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*A single sample is pushed.*")
                    stream.outlet.push_chunk(
                        x=sorted_data.astype(np.float32, copy=False),
                        timestamp=sorted_timestamps.astype(np.float64, copy=False),
                        pushThrough=True,
                    )
                samples_sent[sensor_type] += len(sorted_data)
            except Exception as e:
                if verbose:
                    print(f"Error pushing LSL chunk for {sensor_type}: {e}")
                    # Use get_sinfo() here, as it's already an outlet object
                    print(f"    Data shape: {sorted_data.shape}, Outlet channels: {stream.outlet.get_sinfo().n_channels}")


    def _on_data(_, data: bytearray):
        """Main data callback from Bleak."""
        ts = get_utc_timestamp()
        message = f"{ts}\t{MuseS.EEG_UUID}\t{data.hex()}"

        if raw_data_file:
            try:
                raw_data_file.write(message + "\n")
            except Exception as e:
                if verbose:
                    print(f"Error writing to raw data file: {e}")

        subpackets = parse_message(message)
        decoded = {}
        for sensor_type, pkt_list in subpackets.items():
            stream = streams.get(sensor_type)
            if stream:
                current_state = (
                    stream.base_time, stream.wrap_offset, stream.last_abs_tick, stream.sample_counter,
                )
                array, base_time, wrap_offset, last_abs_tick, sample_counter = (
                    make_timestamps(pkt_list, *current_state)
                )
                decoded[sensor_type] = array

                stream.base_time = base_time
                stream.wrap_offset = wrap_offset
                stream.last_abs_tick = last_abs_tick
                stream.sample_counter = sample_counter

        lsl_now = local_clock()

        # Queue Samples - this is where the dynamic LSL outlet creation happens
        _queue_samples("EEG", decoded.get("EEG", np.empty((0, 0))), lsl_now)
        _queue_samples("ACCGYRO", decoded.get("ACCGYRO", np.empty((0, 0))), lsl_now)
        _queue_samples("OPTICS", decoded.get("OPTICS", np.empty((0, 0))), lsl_now)
        _queue_samples("BATTERY", decoded.get("BATTERY", np.empty((0, 0))), lsl_now)

        should_flush = (time.monotonic() - last_flush_time > FLUSH_INTERVAL) or any(
            len(s.buffer) > MAX_BUFFER_PACKETS for s in streams.values()
        )

        if should_flush:
            _flush_buffer()

    # --- Main connection logic ---
    if verbose:
        print(f"Connecting to {address}...")

    async with bleak.BleakClient(address, timeout=15.0) as client:
        if verbose:
            device_name = client.name
            print(f"Connected. Device: {device_name}")

        # Initialize stream objects (outlets are None for EEG/OPTICS)
        streams = _create_lsl_outlets_initial(client.name, device_id)
        start_time = time.monotonic()

        # Subscribe to data and configure device
        data_callbacks = {MuseS.EEG_UUID: _on_data}
        await MuseS.connect_and_initialize(
            client, preset, data_callbacks, verbose=verbose
        )

        # Removed: print("Streaming data... (LSL Outlets for EEG/OPTICS will be created on first data packet.)")

        # --- Main streaming loop ---
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

        # --- Shutdown ---
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
    """
    Stream decoded Muse S data over LSL with **automatic channel detection**.

    The number of EEG and OPTICS channels is automatically determined
    by the device's preset on the first incoming data packet.

    Parameters
    ----------
    address : str
        Device address (e.g., MAC on Windows).
    preset : str
        Preset to send (e.g., p1041 for all channels, p21 for 4 EEG channels).
    duration : float, optional
        Optional stream duration in seconds. Omit to stream until interrupted.
    record : bool or str, optional
        If False (default), do not record raw data. If True, records to a
        timestamped file. If a string is provided, use it as the filename.
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
        asyncio.run(
            _stream_async(
                address,
                preset,
                duration,
                raw_data_file,
                verbose,
            )
        )
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