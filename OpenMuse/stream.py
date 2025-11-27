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
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, TextIO, Tuple, Union

import bleak
import numpy as np
from bleak.exc import BleakError
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
    A physics-constrained clock synchronizer for BLE devices with accurate internal clocks.

    The Muse device uses a 256 kHz crystal oscillator which is highly stable. The main
    challenge is Bluetooth buffer bloat (asymmetric latency spikes) which can cause
    packets to arrive late. Standard regression misinterprets this as clock drift.

    This implementation uses a slope-constrained approach:
    - The slope (clock speed ratio) is heavily constrained near 1.0
    - The intercept (offset) adapts freely to track the minimum latency envelope
    - Late packets (buffer bloat) are effectively filtered out

    Model:
        lsl_time = intercept + (slope * device_time)

    Design Rationale:
        - Slope is constrained because the device crystal is physically stable
        - Intercept tracks the offset, adapting to clock offset changes
        - We use minimum latency tracking to avoid buffer bloat corruption
    """

    def __init__(
        self,
        forgetting_factor: float = 0.998,
        slope_variance: float = 1e-87,
        intercept_variance: float = 1.0,
    ):
        """
        Initialize the StableClock.

        Parameters
        ----------
        forgetting_factor : float
            RLS forgetting factor. Values close to 1.0 give slower adaptation.
            Default 0.998 corresponds to ~500 sample effective window.
        slope_variance : float
            Initial variance for slope estimate. Very low to strongly
            constrain slope near 1.0. The slope represents clock speed ratio
            which should be extremely stable for crystal oscillators.
        intercept_variance : float
            Initial variance for intercept estimate. Moderate value (1.0) allows
            reasonably fast adaptation without overshooting.
        """
        self.lam = forgetting_factor

        # State vector: [slope, intercept]
        # Initialize slope=1.0 (clocks run at same rate), intercept=0.0 (unknown offset)
        self.theta = np.array([1.0, 0.0])

        # Covariance matrix - diagonal initialization
        # Low slope variance = strong prior that slope ≈ 1.0
        # Moderate intercept variance = balanced adaptation speed
        self.P = np.diag([slope_variance, intercept_variance])

        self.initialized = False

        # For minimum latency envelope tracking (use percentile instead of min)
        self._offset_history: deque = deque(maxlen=200)
        self._percentile = 5.0  # Use 5th percentile instead of absolute minimum

    def update(self, device_time: float, lsl_now: float):
        """
        Update the clock model with a new measurement pair.

        Parameters
        ----------
        device_time : float
            Timestamp from the device's internal clock.
        lsl_now : float
            LSL local_clock() time when the packet arrived.
        """
        # Calculate current offset (positive = packet arrived "late" relative to model)
        current_offset = lsl_now - device_time

        if not self.initialized:
            # First packet: initialize intercept to current offset
            self.theta[1] = current_offset
            self.initialized = True
            self._offset_history.append(current_offset)
            return

        # Track offset history for robust envelope estimation
        self._offset_history.append(current_offset)

        # Use percentile instead of minimum (more robust to single outliers)
        if len(self._offset_history) >= 10:
            baseline_offset = np.percentile(
                list(self._offset_history), self._percentile
            )
        else:
            baseline_offset = min(self._offset_history)

        # Only update if this packet is near the low-latency envelope
        # Wider threshold (50ms) early on, tighter (20ms) once we have history
        latency_threshold = 0.050 if len(self._offset_history) < 50 else 0.025
        if current_offset > baseline_offset + latency_threshold:
            # Skip this packet for model update (likely buffer bloat)
            return

        # --- RLS Update ---
        x = np.array([device_time, 1.0]).reshape(-1, 1)

        # Prediction error
        y_pred = float(x.T @ self.theta)
        error = lsl_now - y_pred

        # Kalman-style gain
        Px = self.P @ x
        denominator = float(self.lam + x.T @ Px)
        if denominator < 1e-10:
            return  # Numerical safety
        k = Px / denominator

        # Update state
        self.theta = self.theta + (k * error).flatten()

        # Update covariance (standard RLS form)
        I = np.eye(2)
        self.P = (I - k @ x.T) @ self.P / self.lam

        # Enforce symmetry (numerical stability)
        self.P = (self.P + self.P.T) / 2

        # Clamp slope to physically realistic bounds (±10% of nominal)
        # Crystal oscillators are far more stable than this, but we allow margin
        self.theta[0] = np.clip(self.theta[0], 0.9, 1.1)

        # Prevent covariance collapse (maintain minimum adaptation ability)
        self.P[0, 0] = max(self.P[0, 0], 1e-8)
        self.P[1, 1] = max(self.P[1, 1], 1e-4)

    def map_time(self, device_times: np.ndarray) -> np.ndarray:
        """Transform device timestamps to LSL time using current model."""
        if not self.initialized:
            return device_times

        slope, intercept = self.theta
        return intercept + (slope * device_times)


class WindowedClock:
    """
    It fits a linear regression (Time_LSL = slope * Time_Device + intercept)
    over a history window (e.g., 30 seconds).
    """

    def __init__(self, window_len_sec: float = 30.0):
        self.window_len = window_len_sec
        self.history = deque()  # Stores (device_time, lsl_time)

        # Current model state [slope, intercept]
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
        limit = device_time - self.window_len
        while self.history and self.history[0][0] < limit:
            self.history.popleft()

        # 3. Fit model (only periodically to save CPU)
        # We also force a check if we aren't initialized yet but have enough data
        if (lsl_now - self.last_fit_time) > self.fit_interval or (
            not self.initialized and len(self.history) >= 10
        ):
            self._fit()
            self.last_fit_time = lsl_now

            # Only mark as initialized if we actually have enough data
            if len(self.history) >= 10:
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

        x_mean = np.mean(x)
        y_mean = np.mean(y)

        # Fit line: y = mx + c
        x_centered = x - x_mean
        y_centered = y - y_mean

        denom = np.sum(x_centered**2)
        if denom < 1e-9:
            return

        self.slope = np.sum(x_centered * y_centered) / denom
        self.intercept = y_mean - (self.slope * x_mean)

    def map_time(self, device_times: np.ndarray) -> np.ndarray:
        """Transform device timestamps to LSL time using current model."""
        if not self.initialized:
            # Fallback for first few packets: just offset by current diff
            # This ensures we don't send 0.0 to LSL while waiting for history to fill
            if len(self.history) > 0:
                dt, lt = self.history[-1]
                return device_times + (lt - dt)

            # Emergency fallback if history is empty (rare, but prevents 0.0 crash)
            from mne_lsl.lsl import local_clock

            return device_times + (
                local_clock() - device_times[-1] if device_times.size > 0 else 0
            )

        return self.intercept + (self.slope * device_times)


class RLSClock:
    """
    A clock synchronizer that uses a standard Recursive Least Squares (RLS) filter to fit:
        lsl_time = intercept + (slope * device_time)

    Unlike StableClock (which constrains the slope near 1.0), this model allows
    the slope to drift more freely but resets if it diverges significantly.
    """

    def __init__(
        self,
        forgetting_factor: float = 0.9999,
        initial_covariance: float = 1e6,
    ):
        self.lam = forgetting_factor
        self.P_init = initial_covariance

        # State: [slope, intercept]
        self.theta = np.array([1.0, 0.0])
        self.P = np.eye(2) * self.P_init

        self.initialized = False
        self.last_update_device_time = -1.0

    def reset(self, current_offset: float = 0.0):
        """Reset filter state to defaults."""
        self.theta = np.array([1.0, current_offset])
        self.P = np.eye(2) * self.P_init
        self.initialized = True

    def update(self, device_time: float, lsl_now: float):
        """
        Update the model using standard RLS as found in stream_old.py.
        """
        # Initialize on first packet
        if not self.initialized:
            self.reset(current_offset=lsl_now - device_time)
            self.last_update_device_time = device_time
            return

        # Prepare RLS input vector x = [device_time, 1.0]
        x = np.array([device_time, 1.0]).reshape(-1, 1)

        # --- RLS Update (Joseph form for numerical stability) ---
        # 1. Calculate Gain
        Px = self.P @ x
        denominator = float(self.lam + x.T @ Px)
        k = Px / denominator

        # 2. Prediction Error
        y_pred = float(x.T @ self.theta)
        error = lsl_now - y_pred

        # 3. Update Parameters (theta)
        self.theta = self.theta + (k * error).flatten()

        # 4. Update Covariance (P)
        I = np.eye(2)
        KX = k @ x.T
        self.P = (I - KX) @ self.P @ (I - KX).T + (k @ k.T) * 1e-12
        self.P /= self.lam

        # --- Stability Check (from stream_old.py) ---
        # If slope (theta[0]) diverges too far from 1.0, reset the filter.
        slope = self.theta[0]
        if not (0.5 < slope < 1.5):
            # Calculate simple offset for the reset
            current_offset = lsl_now - device_time
            self.reset(current_offset)

        self.last_update_device_time = device_time

    def map_time(self, device_times: np.ndarray) -> np.ndarray:
        """Transform device timestamps to LSL time."""
        if not self.initialized:
            return device_times

        slope, intercept = self.theta
        return intercept + (slope * device_times)


class RobustClock:
    """
    Key design principles:
    1. **Fixed slope = 1.0**: Crystal oscillators are extremely stable. Any apparent
        drift is due to buffer bloat, not actual clock skew. We ONLY track offset.

    2. **Robust offset estimation**: Use a weighted median approach on recent samples
       to reject outliers (buffer-bloated packets) without the instability of
       minimum-envelope tracking.

    3. **Windowed history**: Maintain a sliding window of offset measurements,
       giving more weight to recent samples while being robust to outliers.
    """

    def __init__(
        self,
        window_seconds: float = 10.0,
        percentile: float = 10.0,
        ema_alpha: float = 0.02,
    ):
        """
        Initialize the RobustClock.

        Parameters
        ----------
        window_seconds : float
            Time window for offset history (in seconds). Longer = more stable,
            shorter = faster adaptation. Default 10s balances both.
        percentile : float
            Percentile of offset distribution to use (0-50). Lower values track
            the minimum latency envelope more aggressively. Default 10 provides
            robustness while still favoring low-latency packets.
        ema_alpha : float
            Exponential moving average smoothing factor for the final offset.
            Lower = smoother but slower to adapt. Default 0.02 gives ~50 sample
            smoothing at 256 Hz.
        """
        self.window_seconds = window_seconds
        self.percentile = percentile
        self.ema_alpha = ema_alpha

        # Offset history: list of (device_time, offset) tuples
        self._history: deque = deque()

        # Current smoothed offset estimate
        self._offset = 0.0
        self._offset_initialized = False

        self.initialized = False

    def update(self, device_time: float, lsl_now: float):
        """
        Update the clock model with a new measurement pair.

        Parameters
        ----------
        device_time : float
            Timestamp from the device's internal clock.
        lsl_now : float
            LSL local_clock() time when the packet arrived.
        """
        # Calculate instantaneous offset
        current_offset = lsl_now - device_time

        if not self.initialized:
            # First packet: initialize with current offset
            self._offset = current_offset
            self._offset_initialized = True
            self.initialized = True
            self._history.append((device_time, current_offset))
            return

        # Add to history
        self._history.append((device_time, current_offset))

        # Prune old history (keep only window_seconds worth)
        cutoff_time = device_time - self.window_seconds
        while self._history and self._history[0][0] < cutoff_time:
            self._history.popleft()

        # Compute robust offset estimate using low percentile
        # This naturally rejects buffer-bloated packets (high offset = late arrival)
        if len(self._history) >= 5:
            offsets = np.array([o for _, o in self._history])
            # Use low percentile to track minimum latency envelope
            # But not minimum (which is too noisy) - percentile is more stable
            robust_offset = np.percentile(offsets, self.percentile)

            # Smooth with EMA to avoid sudden jumps
            self._offset = (
                self.ema_alpha * robust_offset + (1 - self.ema_alpha) * self._offset
            )
        else:
            # Not enough history yet - use simple EMA on raw offset
            self._offset = (
                self.ema_alpha * current_offset + (1 - self.ema_alpha) * self._offset
            )

    def map_time(self, device_times: np.ndarray) -> np.ndarray:
        """Transform device timestamps to LSL time using current offset."""
        if not self.initialized:
            return device_times

        # Simple offset model: lsl_time = device_time + offset
        return device_times + self._offset

class AdaptiveClock:
    """
    Improved clock synchronizer that adapts quickly to latency drops (e.g. connection
    interval changes) but resists jitter spikes.
    """

    def __init__(
        self,
        window_seconds: float = 10.0,
        percentile: float = 5.0, # Lower percentile (5th) to track min-latency better
        final_alpha: float = 0.02,
        initial_alpha: float = 1.0, # Start by trusting the measurement 100%
        warmup_packets: int = 500, # Approx 2 seconds at 256Hz
    ):
        self.window_seconds = window_seconds
        self.percentile = percentile
        self.final_alpha = final_alpha
        self.current_alpha = initial_alpha
        self.warmup_packets = warmup_packets

        self._history: deque = deque()
        self._offset = 0.0
        self.initialized = False
        self.packets_processed = 0

    def update(self, device_time: float, lsl_now: float):
        current_offset = lsl_now - device_time
        self.packets_processed += 1

        if not self.initialized:
            self._offset = current_offset
            self.initialized = True
            self._history.append((device_time, current_offset))
            return

        # 1. Update History
        self._history.append((device_time, current_offset))
        cutoff_time = device_time - self.window_seconds
        while self._history and self._history[0][0] < cutoff_time:
            self._history.popleft()

        # 2. Decay Alpha (Warmup Phase)
        # Linearly decay alpha from 1.0 down to 0.02 over the warmup period
        if self.packets_processed < self.warmup_packets:
            progress = self.packets_processed / self.warmup_packets
            self.current_alpha = self.current_alpha * (1 - progress) + self.final_alpha * progress
        else:
            self.current_alpha = self.final_alpha

        # 3. Calculate Target Offset (Low Percentile)
        if len(self._history) >= 5:
            offsets = np.array([o for _, o in self._history])
            # Use 5th percentile to track the "fastest" packets (minimum latency)
            target_offset = np.percentile(offsets, self.percentile)

            # 4. Asymmetric Update
            # If target is LOWER (less latency), adapt faster (trust the improvement).
            # If target is HIGHER (more lag), adapt slower (suspect buffer bloat).
            effective_alpha = self.current_alpha
            if target_offset < self._offset:
                # We found a better (lower latency) path. Adapt 5x faster.
                effective_alpha = min(1.0, self.current_alpha * 5.0)

            self._offset = (effective_alpha * target_offset) + ((1 - effective_alpha) * self._offset)
        else:
            # Fallback for very first few samples
            self._offset = (self.current_alpha * current_offset) + ((1 - self.current_alpha) * self._offset)

    def map_time(self, device_times: np.ndarray) -> np.ndarray:
        if not self.initialized:
            return device_times
        return device_times + self._offset

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

    # --- Robust Clock Sync ---
    clock: AdaptiveClock = field(default_factory=AdaptiveClock)
    # clock: RobustClock = field(default_factory=RobustClock)
    # clock: StableClock = field(default_factory=StableClock)
    # clock: WindowedClock = field(default_factory=WindowedClock)
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
