"""
Comprehensive AMB Inversion Analysis Across All XDF Files
=========================================================

This script analyzes all available XDF files to determine:
1. How often the INNER-OUTER AMB inversion occurs
2. Whether it's device-specific or recording-specific
"""

import os
from pathlib import Path

import numpy as np
import pyxdf

# Device identifiers
OLD_FIRMWARE_MAC = "00:55:DA:B9:FA:20"
NEW_FIRMWARE_MAC = "00:55:DA:BB:CD:CD"


def get_amb_correlations(stream):
    """Extract INNER-OUTER AMB correlations from a stream."""
    name = stream["info"]["name"][0]

    # Get channel labels
    try:
        channels = stream["info"]["desc"][0]["channels"][0]["channel"]
        ch_labels = [ch["label"][0] for ch in channels]
    except (KeyError, IndexError, TypeError):
        return None

    # Find AMB channel indices
    li_idx = next((i for i, l in enumerate(ch_labels) if "LI_AMB" in l), None)
    ri_idx = next((i for i, l in enumerate(ch_labels) if "RI_AMB" in l), None)
    lo_idx = next((i for i, l in enumerate(ch_labels) if "LO_AMB" in l), None)
    ro_idx = next((i for i, l in enumerate(ch_labels) if "RO_AMB" in l), None)

    if None in [li_idx, ri_idx, lo_idx, ro_idx]:
        return None

    data = stream["time_series"]
    if len(data) < 100:
        return None

    # Compute correlations
    left_io = np.corrcoef(data[:, lo_idx], data[:, li_idx])[0, 1]
    right_io = np.corrcoef(data[:, ro_idx], data[:, ri_idx])[0, 1]

    # Determine device from stream name
    if OLD_FIRMWARE_MAC in name:
        device = "OLD"
    elif NEW_FIRMWARE_MAC in name:
        device = "NEW"
    else:
        device = "UNKNOWN"

    return {
        "stream_name": name,
        "device": device,
        "left_inner_outer": left_io,
        "right_inner_outer": right_io,
        "mean_inner_outer": (left_io + right_io) / 2,
    }


def analyze_xdf_file(filepath):
    """Analyze a single XDF file for AMB correlations."""
    try:
        streams, _ = pyxdf.load_xdf(
            str(filepath),
            synchronize_clocks=True,
            handle_clock_resets=True,
            dejitter_timestamps=False,
        )
    except Exception as e:
        print(f"  ERROR loading {filepath}: {e}")
        return []

    results = []
    for stream in streams:
        name = stream["info"]["name"][0]
        if "OPTICS" in name:
            corr = get_amb_correlations(stream)
            if corr:
                corr["file"] = filepath.name
                results.append(corr)

    return results


def main():
    """Analyze all XDF files in the validation directories."""

    print("=" * 80)
    print("AMB INVERSION ANALYSIS ACROSS ALL XDF FILES")
    print("=" * 80)

    # Find all XDF files
    base_dir = Path(__file__).parent
    xdf_dirs = [
        base_dir,  # channels2
        base_dir.parent / "channels",  # channels
        base_dir.parent.parent / "synchronization",  # synchronization
        base_dir.parent.parent / "synchronization_old",  # synchronization_old
    ]

    xdf_files = []
    for d in xdf_dirs:
        if d.exists():
            xdf_files.extend(d.glob("*.xdf"))

    print(f"\nFound {len(xdf_files)} XDF files:")
    for f in xdf_files:
        print(f"  - {f.parent.name}/{f.name}")

    # Analyze each file
    all_results = []
    for xdf_file in xdf_files:
        print(f"\nAnalyzing: {xdf_file.name}")
        results = analyze_xdf_file(xdf_file)
        all_results.extend(results)
        for r in results:
            sign = "+" if r["mean_inner_outer"] > 0 else "-"
            print(f"  {r['device']}: IO={r['mean_inner_outer']:+.4f} ({sign})")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    old_results = [r for r in all_results if r["device"] == "OLD"]
    new_results = [r for r in all_results if r["device"] == "NEW"]

    print(f"\nOLD Firmware Recordings: {len(old_results)}")
    if old_results:
        old_io_values = [r["mean_inner_outer"] for r in old_results]
        old_positive = sum(1 for v in old_io_values if v > 0)
        old_negative = len(old_io_values) - old_positive
        print(
            f"  POSITIVE INNER-OUTER correlation: {old_positive}/{len(old_io_values)} ({100*old_positive/len(old_io_values):.0f}%)"
        )
        print(
            f"  NEGATIVE INNER-OUTER correlation: {old_negative}/{len(old_io_values)} ({100*old_negative/len(old_io_values):.0f}%)"
        )
        print(f"  Mean correlation: {np.mean(old_io_values):+.4f}")
        print(f"  Range: [{min(old_io_values):+.4f}, {max(old_io_values):+.4f}]")

    print(f"\nNEW Firmware Recordings: {len(new_results)}")
    if new_results:
        new_io_values = [r["mean_inner_outer"] for r in new_results]
        new_positive = sum(1 for v in new_io_values if v > 0)
        new_negative = len(new_io_values) - new_positive
        print(
            f"  POSITIVE INNER-OUTER correlation: {new_positive}/{len(new_io_values)} ({100*new_positive/len(new_io_values):.0f}%)"
        )
        print(
            f"  NEGATIVE INNER-OUTER correlation: {new_negative}/{len(new_io_values)} ({100*new_negative/len(new_io_values):.0f}%)"
        )
        print(f"  Mean correlation: {np.mean(new_io_values):+.4f}")
        print(f"  Range: [{min(new_io_values):+.4f}, {max(new_io_values):+.4f}]")

    # Detailed breakdown
    print("\n" + "=" * 80)
    print("DETAILED RESULTS BY RECORDING")
    print("=" * 80)
    print(f"\n{'File':<35} {'Device':<8} {'Left IO':>10} {'Right IO':>10} {'Mean':>10} {'Sign':>6}")
    print("-" * 85)

    for r in sorted(all_results, key=lambda x: (x["device"], x["file"])):
        sign = "+" if r["mean_inner_outer"] > 0 else "-"
        print(
            f"{r['file']:<35} {r['device']:<8} {r['left_inner_outer']:>+10.4f} {r['right_inner_outer']:>+10.4f} {r['mean_inner_outer']:>+10.4f} {sign:>6}"
        )

    # Conclusion
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    total = len(all_results)
    all_positive = sum(1 for r in all_results if r["mean_inner_outer"] > 0)
    all_negative = total - all_positive

    print(f"\nTotal OPTICS recordings analyzed: {total}")
    print(f"  POSITIVE INNER-OUTER: {all_positive} ({100*all_positive/total:.0f}%)")
    print(f"  NEGATIVE INNER-OUTER: {all_negative} ({100*all_negative/total:.0f}%)")

    if all_negative > 0:
        neg_recordings = [r for r in all_results if r["mean_inner_outer"] < 0]
        print(f"\nRecordings with NEGATIVE correlation:")
        for r in neg_recordings:
            print(f"  - {r['file']} ({r['device']}): {r['mean_inner_outer']:+.4f}")
    else:
        print("\n✓ NO INNER-OUTER AMB INVERSIONS DETECTED in any recording!")


if __name__ == "__main__":
    main()
