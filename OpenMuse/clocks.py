"""
Clock Synchronization Models for BLE Devices
=============================================

This module provides clock synchronization algorithms for mapping device timestamps
to LSL (Lab Streaming Layer) time. The main challenge is handling Bluetooth buffer
bloat - variable latency spikes caused by BLE connection intervals and packet queuing.

SYNCHRONIZATION VALIDATION SUMMARY (January 2026)
-------------------------------------------------

Extensive testing was performed using a 3-device photosensor experiment:
- Three Muse S Athena headbands simultaneously streaming:
  - Muse 2 (grey): 00:55:DA:BB:CD:CD
  - Muse 3 (black): 00:55:DA:BA:05:F9
  - Muse 1 old firmware (black): 00:55:DA:B9:FA:20
- External light sensor (BITalino) as ground truth reference
- Screen flashing at ~5Hz for 30+ minutes
- Compared event onset detection across different clock models

**KEY FINDINGS:**

    ┌─────────────────┬─────────────┬─────────────┬─────────────┬───────────┐
    │ Clock Type      │ Muse1_OldFW │ Muse2_Grey  │ Muse3_Black │ Match Rate│
    ├─────────────────┼─────────────┼─────────────┼─────────────┼───────────┤
    │ WindowedClock   │   43.3 ms   │   45.0 ms   │   40.7 ms   │   100%    │
    │ RobustClock     │  191.0 ms   │  189.8 ms   │  108.2 ms   │   69-93%  │
    │ ConstrainedRLS  │  141.7 ms   │  142.0 ms   │  145.7 ms   │   79-82%  │
    │ StandardRLS     │  139.8 ms   │  118.8 ms   │  102.4 ms   │   16-20%  │
    │ AdaptiveClock   │  170.1 ms   │  151.5 ms   │  123.1 ms   │   50-55%  │
    └─────────────────┴─────────────┴─────────────┴─────────────┴───────────┘

    Average OPTICS jitter (all 3 devices combined):
    1. windowed:     43.0 ms  ★ RECOMMENDED (100% event match rate)
    2. robust:      163.0 ms
    3. constrained: 143.1 ms
    4. standard:    120.3 ms  (but only 16-20% match rate!)
    5. adaptive:    148.2 ms

**RECOMMENDATIONS:**

1. **Default choice: WindowedRegressionClock ("windowed")**
   - Best overall stability across different devices
   - 100% event match rate for both old and new firmware devices
   - Handles variable BLE transmission delays well

2. **Alternative: RobustOffsetClock ("robust")**
   - Good for older/more stable devices
   - Uses percentile-based offset tracking
   - May struggle with high-jitter devices

3. **Avoid for timing-critical applications:**
   - AdaptiveClock: Can have 5x higher jitter on some devices
   - ConstrainedRLS: Strong slope constraint can cause offset drift

**WHY WINDOWED WORKS BEST:**

The WindowedRegressionClock fits a local linear regression over a sliding window
(default 30 seconds). This approach:
- Adapts to gradual clock drift without overcorrecting
- Doesn't assume constant offset (unlike offset-only models)
- Doesn't over-constrain slope (unlike ConstrainedRLS)
- Re-fits periodically (every 1s) rather than on every packet

**PHYSICAL BASIS:**

Crystal oscillators in Muse devices are highly stable (<<1% drift). The main
source of timestamp variability is NOT clock drift but Bluetooth buffer bloat:
- Packets queue at BLE connection interval boundaries
- Variable RF conditions cause transmission delays
- The "offset" between device and LSL time is actually the sum of:
  - True clock offset (stable)
  - Variable queuing delay (0-50+ ms jitter)

All clock models try to separate these effects, but windowed regression does
it most robustly by fitting a local linear model that tracks the minimum
latency envelope without being fooled by buffer-bloated packets.

USAGE
-----

In stream.py, set the `clock` parameter:

    python -m OpenMuse stream --clock windowed    # ★ RECOMMENDED (30s window)
    python -m OpenMuse stream --clock windowed10  # 10s window (faster adaptation)
    python -m OpenMuse stream --clock windowed15  # 15s window
    python -m OpenMuse stream --clock windowed30  # Same as "windowed"
    python -m OpenMuse stream --clock windowed45  # 45s window
    python -m OpenMuse stream --clock windowed60  # 60s window (more stable)
    python -m OpenMuse stream --clock robust
    python -m OpenMuse stream --clock adaptive
    python -m OpenMuse stream --clock constrained
    python -m OpenMuse stream --clock standard

Or use None/"none" to disable clock synchronization entirely and use raw
device timestamps (not recommended for multi-stream experiments).

"""

from collections import deque

import numpy as np
from mne_lsl.lsl import local_clock


class RobustOffsetClock:
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
        Initialize the RobustOffsetClock.

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


class AdaptiveOffsetClock:
    """
    Improved clock synchronizer that adapts quickly to latency drops (e.g. connection
    interval changes) but resists jitter spikes.
    """

    def __init__(
        self,
        window_seconds: float = 10.0,
        percentile: float = 5.0,  # Lower percentile (5th) to track min-latency better
        final_alpha: float = 0.02,
        initial_alpha: float = 1.0,  # Start by trusting the measurement 100%
        warmup_packets: int = 500,  # Approx 2 seconds at 256Hz
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
            self.current_alpha = (
                self.current_alpha * (1 - progress) + self.final_alpha * progress
            )
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

            self._offset = (effective_alpha * target_offset) + (
                (1 - effective_alpha) * self._offset
            )
        else:
            # Fallback for very first few samples
            self._offset = (self.current_alpha * current_offset) + (
                (1 - self.current_alpha) * self._offset
            )

    def map_time(self, device_times: np.ndarray) -> np.ndarray:
        if not self.initialized:
            return device_times
        return device_times + self._offset


class WindowedRegressionClock:
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
            return device_times + (
                local_clock() - device_times[-1] if device_times.size > 0 else 0
            )

        return self.intercept + (self.slope * device_times)


class StandardRLSClock:
    """
    A clock synchronizer that uses a standard Recursive Least Squares (RLS) filter to fit:
        lsl_time = intercept + (slope * device_time)

    Unlike ConstrainedRLSClock (which constrains the slope near 1.0), this model allows
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


class ConstrainedRLSClock:
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
        Initialize the ConstrainedRLSClock.

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
