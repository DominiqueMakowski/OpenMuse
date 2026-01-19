"""
XDF Validation Script
=====================

Validates sampling rates and data quality from actual LSL recordings (XDF files).
This confirms that the streaming pipeline correctly handles both old and new firmware.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import pyxdf
except ImportError:
    print("pyxdf not installed. Run: pip install pyxdf")
    sys.exit(1)


def analyze_xdf_sampling_rates(filepath: str):
    """Analyze effective sampling rates for all streams in an XDF file."""
    print(f"\n{'='*80}")
    print(f"XDF SAMPLING RATE ANALYSIS")
    print(f"File: {filepath}")
    print(f"{'='*80}")

    # Load XDF with minimal processing
    streams, header = pyxdf.load_xdf(
        filepath,
        synchronize_clocks=False,  # Keep original timestamps
        handle_clock_resets=False,
        dejitter_timestamps=False,
    )

    results = {}

    print(
        f"\n{'Stream Name':<45} {'Samples':>10} {'Duration':>10} {'Nominal':>10} {'Effective':>10} {'Error':>10}"
    )
    print("-" * 100)

    for stream in streams:
        name = stream["info"]["name"][0]
        n_samples = len(stream["time_stamps"])

        if n_samples < 2:
            print(
                f"{name:<45} {n_samples:>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}"
            )
            continue

        ts = stream["time_stamps"]
        duration = ts[-1] - ts[0]

        if duration <= 0:
            print(
                f"{name:<45} {n_samples:>10} {duration:>10.2f}s {'N/A':>10} {'N/A':>10} {'N/A':>10}"
            )
            continue

        nominal_rate = float(stream["info"]["nominal_srate"][0])
        effective_rate = (n_samples - 1) / duration

        if nominal_rate > 0:
            error_pct = 100 * (effective_rate - nominal_rate) / nominal_rate
        else:
            error_pct = 0

        # Check monotonicity
        diffs = np.diff(ts)
        n_non_mono = np.sum(diffs < 0)

        results[name] = {
            "samples": n_samples,
            "duration": duration,
            "nominal_rate": nominal_rate,
            "effective_rate": effective_rate,
            "error_pct": error_pct,
            "non_monotonic": n_non_mono,
        }

        mono_flag = " ⚠" if n_non_mono > 0 else ""
        print(
            f"{name:<45} {n_samples:>10} {duration:>10.2f}s {nominal_rate:>10.1f} {effective_rate:>10.2f} {error_pct:>+10.2f}%{mono_flag}"
        )

    return streams, results


def check_amb_correlation_xdf(streams):
    """Check AMB channel correlations from XDF streams."""
    print(f"\n{'='*80}")
    print("AMB CHANNEL CORRELATION CHECK FROM XDF")
    print("=" * 80)

    for stream in streams:
        name = stream["info"]["name"][0]
        if "OPTICS" not in name:
            continue

        print(f"\n--- {name} ---")

        # Get channel labels
        try:
            channels = stream["info"]["desc"][0]["channels"][0]["channel"]
            ch_labels = [ch["label"][0] for ch in channels]
        except (KeyError, IndexError, TypeError):
            print("  Could not extract channel labels")
            continue

        # Find AMB channels
        amb_indices = [i for i, ch in enumerate(ch_labels) if "AMB" in ch]
        if len(amb_indices) < 4:
            print(f"  Only {len(amb_indices)} AMB channels found")
            continue

        data = stream["time_series"]
        amb_labels = [ch_labels[i] for i in amb_indices]

        print(f"  AMB channels: {amb_labels}")

        # Compute correlation matrix
        amb_data = data[:, amb_indices]
        corr = np.corrcoef(amb_data.T)

        print("  Correlation matrix:")
        for i, label_i in enumerate(amb_labels):
            row = "    "
            for j, label_j in enumerate(amb_labels):
                row += f"{corr[i,j]:+.3f}  "
            print(row)

        # Specifically check INNER vs OUTER
        li_idx = next((i for i, l in enumerate(amb_labels) if "LI_AMB" in l), None)
        ri_idx = next((i for i, l in enumerate(amb_labels) if "RI_AMB" in l), None)
        lo_idx = next((i for i, l in enumerate(amb_labels) if "LO_AMB" in l), None)
        ro_idx = next((i for i, l in enumerate(amb_labels) if "RO_AMB" in l), None)

        if all(x is not None for x in [li_idx, ri_idx, lo_idx, ro_idx]):
            print(f"\n  Key correlations:")
            print(f"    INNER-INNER (LI vs RI): {corr[li_idx, ri_idx]:.3f}")
            print(f"    OUTER-OUTER (LO vs RO): {corr[lo_idx, ro_idx]:.3f}")
            print(f"    INNER-OUTER (LI vs LO): {corr[li_idx, lo_idx]:.3f}")
            print(f"    INNER-OUTER (RI vs RO): {corr[ri_idx, ro_idx]:.3f}")


def main():
    """Analyze XDF files in validation directories."""

    # Find XDF files in multiple locations
    base_dir = Path(__file__).parent
    channels_dir = base_dir / "channels"
    sync_dir = base_dir.parent / "synchronization"  # validation/synchronization

    xdf_files = []
    if channels_dir.exists():
        xdf_files.extend(channels_dir.glob("*.xdf"))
    if sync_dir.exists():
        xdf_files.extend(sync_dir.glob("*.xdf"))

    if not xdf_files:
        print("No XDF files found")
        return

    print(f"Found {len(xdf_files)} XDF files")

    for xdf_file in xdf_files:
        streams, results = analyze_xdf_sampling_rates(str(xdf_file))
        check_amb_correlation_xdf(streams)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY: SAMPLING RATE VERIFICATION")
    print("=" * 80)
    print(
        """
Key findings:
1. EEG should be ~256 Hz (error < ±1%)
2. OPTICS should be ~64 Hz (error < ±1%)
3. ACCGYRO should be ~52 Hz (error < ±1%)
4. BATTERY: Old firmware ~1 Hz, New firmware ~0.2 Hz

If all rates are within tolerance, the decoding is correct and no data is lost.
"""
    )


if __name__ == "__main__":
    main()
