"""
Test Phase Delay Hypothesis for INNER AMB Inversion
====================================================

Question: Could the negative INNER-OUTER AMB correlation be caused by a
temporal delay rather than signal inversion?

If the black/white flashing creates periodic signals, a phase shift of
half a period would cause anti-correlation. We can test this by:

1. Computing cross-correlation at multiple lags
2. If it's a delay: peak correlation should be at some non-zero lag
3. If it's true inversion: negative correlation at all lags

"""

import sys
from pathlib import Path
import numpy as np

try:
    import pyxdf
    from scipy import signal as scipy_signal
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)


def get_optics_stream(streams, device_pattern):
    """Find OPTICS stream matching device pattern."""
    for s in streams:
        name = s["info"]["name"][0]
        if "OPTICS" in name and device_pattern in name:
            return s
    return None


def get_channel_data(stream, channel_name):
    """Extract channel data and timestamps."""
    channels = stream["info"]["desc"][0]["channels"][0]["channel"]
    ch_labels = [ch["label"][0] for ch in channels]

    if channel_name not in ch_labels:
        return None, None

    idx = ch_labels.index(channel_name)
    return stream["time_series"][:, idx], stream["time_stamps"]


def compute_cross_correlation(sig1, sig2, max_lag_samples=500):
    """
    Compute normalized cross-correlation at multiple lags.

    Returns:
        lags: array of lag values (in samples)
        corr: correlation at each lag
    """
    # Normalize signals
    sig1_norm = (sig1 - np.mean(sig1)) / (np.std(sig1) + 1e-10)
    sig2_norm = (sig2 - np.mean(sig2)) / (np.std(sig2) + 1e-10)

    # Compute full cross-correlation
    full_corr = np.correlate(sig1_norm, sig2_norm, mode="full")
    full_corr /= len(sig1)  # Normalize

    # Extract centered portion
    mid = len(full_corr) // 2
    start = max(0, mid - max_lag_samples)
    end = min(len(full_corr), mid + max_lag_samples + 1)

    corr = full_corr[start:end]
    lags = np.arange(-(mid - start), end - mid)

    return lags, corr


def analyze_phase_delay(filepath: str):
    """Analyze whether INNER-OUTER correlation is due to phase delay."""
    print(f"\n{'='*80}")
    print("PHASE DELAY HYPOTHESIS TEST")
    print(f"File: {filepath}")
    print("=" * 80)

    streams, _ = pyxdf.load_xdf(
        filepath,
        synchronize_clocks=True,
        handle_clock_resets=True,
        dejitter_timestamps=False,
    )

    # Find both devices
    old_device = "B9:FA:20"  # OLD firmware (has battery stream)
    new_device = "BB:CD:CD"  # NEW firmware (no battery stream)

    for device_pattern, device_name in [(old_device, "OLD"), (new_device, "NEW")]:
        stream = get_optics_stream(streams, device_pattern)
        if stream is None:
            print(f"\n{device_name} device not found in this file")
            continue

        print(f"\n--- {device_name} DEVICE ({device_pattern}) ---")

        # Get INNER and OUTER AMB channels
        li_amb, ts = get_channel_data(stream, "OPTICS_LI_AMB")
        lo_amb, _ = get_channel_data(stream, "OPTICS_LO_AMB")

        if li_amb is None or lo_amb is None:
            print("  AMB channels not found")
            continue

        # Compute sample rate
        dt = np.median(np.diff(ts))
        sample_rate = 1.0 / dt
        print(f"  Sample rate: {sample_rate:.1f} Hz")

        # Compute cross-correlation
        max_lag_ms = 500  # Check up to 500ms delay
        max_lag_samples = int(max_lag_ms / 1000 * sample_rate)

        lags, corr = compute_cross_correlation(li_amb, lo_amb, max_lag_samples)
        lags_ms = lags * 1000 / sample_rate

        # Find peak correlation
        peak_idx = np.argmax(np.abs(corr))
        peak_lag_ms = lags_ms[peak_idx]
        peak_corr = corr[peak_idx]

        # Correlation at zero lag
        zero_idx = np.argmin(np.abs(lags))
        zero_lag_corr = corr[zero_idx]

        print(f"\n  Cross-correlation analysis (LI_AMB vs LO_AMB):")
        print(f"    Correlation at lag=0: {zero_lag_corr:+.3f}")
        print(f"    Peak correlation: {peak_corr:+.3f} at lag={peak_lag_ms:.1f}ms")

        # Interpretation
        print(f"\n  INTERPRETATION:")
        if abs(peak_corr) > 0.5:
            if abs(peak_lag_ms) < 20:  # Within 20ms of zero
                if peak_corr > 0:
                    print(f"    ✓ POSITIVE correlation at ~zero lag")
                    print(f"      → Signals are IN PHASE (no inversion)")
                else:
                    print(f"    ✗ NEGATIVE correlation at ~zero lag")
                    print(f"      → TRUE SIGNAL INVERSION (not a delay)")
            else:
                if peak_corr > 0:
                    print(f"    ? POSITIVE correlation at {peak_lag_ms:.0f}ms lag")
                    print(f"      → Possible PHASE DELAY of {abs(peak_lag_ms):.0f}ms")
                else:
                    print(f"    ? NEGATIVE correlation even at best lag")
                    print(f"      → Signals are INVERTED, not just delayed")
        else:
            print(f"    ? Low correlation ({peak_corr:.2f}) - signals may be unrelated")

        # Additional check: is there ANY positive correlation at any lag?
        max_positive = np.max(corr)
        max_positive_lag = lags_ms[np.argmax(corr)]
        min_negative = np.min(corr)
        min_negative_lag = lags_ms[np.argmin(corr)]

        print(f"\n  Full correlation range:")
        print(f"    Max positive: {max_positive:+.3f} at {max_positive_lag:.1f}ms")
        print(f"    Max negative: {min_negative:+.3f} at {min_negative_lag:.1f}ms")

        if max_positive > 0.5 and abs(max_positive) > abs(min_negative):
            print(f"    → Best match is POSITIVE at {max_positive_lag:.1f}ms delay")
            print(f"      This suggests a PHASE DELAY, not inversion")
        elif abs(min_negative) > 0.5 and abs(min_negative) > max_positive:
            print(
                f"    → Best match is NEGATIVE (inverted) at {min_negative_lag:.1f}ms"
            )
            if abs(min_negative_lag) < 50:
                print(f"      This confirms TRUE SIGNAL INVERSION")
            else:
                print(f"      This could be ~half-period phase shift")


def main():
    """Analyze all XDF files for phase delay."""

    # Find XDF files
    base_dir = Path(__file__).parent
    channels_dir = base_dir / "channels"
    sync_dir = base_dir.parent / "synchronization"

    xdf_files = []
    if channels_dir.exists():
        xdf_files.extend(channels_dir.glob("*.xdf"))
    if sync_dir.exists():
        xdf_files.extend(sync_dir.glob("*.xdf"))

    if not xdf_files:
        print("No XDF files found")
        return

    print(f"Found {len(xdf_files)} XDF files")
    print("\nTesting whether negative correlation is due to phase delay...")
    print("If it's a delay: peak correlation should shift to non-zero lag")
    print("If it's inversion: negative correlation persists at all lags")

    for xdf_file in xdf_files:
        analyze_phase_delay(str(xdf_file))

    print(f"\n{'='*80}")
    print("CONCLUSION")
    print("=" * 80)
    print(
        """
To rule out phase delay as the cause of negative correlation:
- If peak correlation is NEGATIVE and near lag=0 → TRUE INVERSION
- If peak correlation is POSITIVE at some non-zero lag → PHASE DELAY
- If negative correlation persists across all lags → TRUE INVERSION

The flashing period matters: if flashing at 1Hz (1000ms period),
a 500ms delay would cause anti-correlation (half-period shift).
"""
    )


if __name__ == "__main__":
    main()
