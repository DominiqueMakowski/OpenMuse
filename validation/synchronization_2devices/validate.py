"""
Compare clock synchronization across different clock types.

FOCUS: Stability of timing delays (standard deviation), not mean offset.
The mean offset depends on event detection cutoff and can be corrected post-hoc,
but stability (jitter) cannot be fixed.

This script automatically detects all XDF files in the current directory,
analyzes them, and produces comparison plots.

NOTES:
- For photosensor experiments (screen flashing), the OPTICS AMB channel may show
  inverted correlation. This is EXPECTED optical behavior, not a firmware bug.
  See summary.txt in validation/new_firmware/ for details.
- The "inverted" AMB detection tries 1.0 - normalized_signal as a fallback.
- Different clock types (adaptive, constrained, robust, windowed, standard) handle
  timestamp jitter differently. WINDOWED typically performs best for devices with
  variable BLE transmission delays.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyxdf
from scipy import ndimage
from scipy import signal as sp_signal

# --- Configuration ---
DEJITTER_TIMESTAMPS = ["OpenSignals"]

# Muse device MACs
OLD_FIRMWARE_MAC = "00:55:DA:B9:FA:20"  # OLD firmware
NEW_FIRMWARE_MAC = "00:55:DA:BB:CD:CD"  # NEW firmware

# OPTICS channels to try (will pick the most stable one)
OPTICS_CHANNELS = ["OPTICS_RI_IR", "OPTICS_RI_RED", "OPTICS_RI_AMB"]


def summarize_streams(streams):
    """Get range of timestamps for each stream."""
    data = []
    for stream in streams:
        name = stream["info"].get("name", ["Unnamed"])[0]
        n_samples = len(stream["time_stamps"])

        if n_samples == 0:
            data.append(
                {
                    "Stream": name,
                    "Samples": 0,
                    "Duration": np.nan,
                    "Start": np.nan,
                    "End": np.nan,
                    "Nominal SR": float(stream["info"]["nominal_srate"][0]),
                    "Effective SR": np.nan,
                }
            )
            continue

        ts_min = stream["time_stamps"].min()
        ts_max = stream["time_stamps"].max()
        duration = ts_max - ts_min
        nominal_srate = float(stream["info"]["nominal_srate"][0])
        effective_srate = n_samples / duration if duration > 0 else np.nan

        data.append(
            {
                "Stream": name,
                "Samples": n_samples,
                "Duration": duration,
                "Start": ts_min,
                "End": ts_max,
                "Nominal SR": nominal_srate,
                "Effective SR": effective_srate,
            }
        )

    return pd.DataFrame(data)


def rolling_normalize(signal_data, window_sec=5.0, fs=64.0):
    """Apply rolling min-max normalization to correct for drifts."""
    window_samples = max(int(window_sec * fs), 1)
    rolling_min = ndimage.minimum_filter1d(signal_data, size=window_samples, mode="nearest")
    rolling_max = ndimage.maximum_filter1d(signal_data, size=window_samples, mode="nearest")
    range_vals = rolling_max - rolling_min
    range_vals[range_vals < 1e-10] = 1e-10
    return (signal_data - rolling_min) / range_vals


def median_filter(signal_data, kernel_size=3):
    """Apply median filter to reduce noise."""
    return ndimage.median_filter(signal_data, size=kernel_size)


def find_channel(streams, stream_name, channel_name, normalize="rolling", invert=False):
    """Find a channel in streams and return signal and timestamps."""
    for s in streams:
        name = s["info"].get("name", ["Unnamed"])[0]
        if name == stream_name:
            try:
                channels = [d["label"][0] for d in s["info"]["desc"][0]["channels"][0]["channel"]]
            except (KeyError, IndexError, TypeError):
                return None, None
            matching = [ch for ch in channels if channel_name in ch]
            if not matching:
                return None, None
            ch_name = matching[0]
            signal_data = np.array(s["time_series"])[:, channels.index(ch_name)].astype(float)
            ts = s["time_stamps"]

            if len(ts) < 2:
                return None, None

            fs = len(ts) / (ts[-1] - ts[0])

            # Apply median filter first to reduce noise
            signal_data = median_filter(signal_data, kernel_size=3)

            # Invert if requested
            if invert:
                signal_data = -signal_data

            if normalize == "global":
                signal_data = (signal_data - np.min(signal_data)) / (np.max(signal_data) - np.min(signal_data) + 1e-10)
            elif normalize == "rolling":
                signal_data = rolling_normalize(signal_data, window_sec=5.0, fs=fs)

            return signal_data, ts
    return None, None


def find_events_midpoint(signal_data, timestamps, hysteresis=0.15):
    """
    Find events by detecting midpoint crossings with hysteresis.

    Uses Schmitt trigger logic: signal must go above 0.5+hysteresis to trigger HIGH,
    then below 0.5-hysteresis to trigger LOW. Event is at the midpoint (0.5) crossing.

    This is more robust than simple threshold crossing.
    """
    if len(signal_data) < 10:
        return np.array([])

    fs = len(timestamps) / (timestamps[-1] - timestamps[0])
    min_gap_samples = int(0.15 * fs)  # Minimum 150ms between events

    high_thresh = 0.5 + hysteresis
    low_thresh = 0.5 - hysteresis

    events = []
    state = None  # None = unknown, 'high' or 'low'

    # Initialize state
    if signal_data[0] > high_thresh:
        state = "high"
    elif signal_data[0] < low_thresh:
        state = "low"

    for i in range(1, len(signal_data)):
        if state == "low" and signal_data[i] > high_thresh:
            # Transition to high - find midpoint crossing
            # Search backwards from i to find where it crossed 0.5
            for j in range(i, max(0, i - 20), -1):
                if signal_data[j] <= 0.5:
                    events.append(j)
                    break
            state = "high"
        elif state == "high" and signal_data[i] < low_thresh:
            # Transition to low - find midpoint crossing
            for j in range(i, max(0, i - 20), -1):
                if signal_data[j] >= 0.5:
                    events.append(j)
                    break
            state = "low"
        elif state is None:
            # Initialize state based on current value
            if signal_data[i] > high_thresh:
                state = "high"
            elif signal_data[i] < low_thresh:
                state = "low"

    # Filter events that are too close
    filtered = []
    for e in sorted(events):
        if len(filtered) == 0 or e - filtered[-1] > min_gap_samples:
            filtered.append(e)

    return np.array(filtered)


def find_marker_onsets(streams, marker_value=1):
    """Find timestamps when marker equals specified value."""
    for s in streams:
        name = s["info"].get("name", ["Unnamed"])[0]
        if "jsPsych" in name.lower() or "marker" in name.lower():
            markers = np.array(s["time_series"]).flatten().astype(float)
            ts = s["time_stamps"]
            return ts[markers == marker_value]
    return np.array([])


def get_experiment_end_time(streams):
    """Get the timestamp of the last marker (experiment end)."""
    for s in streams:
        name = s["info"].get("name", ["Unnamed"])[0]
        if "jsPsych" in name.lower() or "marker" in name.lower():
            ts = s["time_stamps"]
            if len(ts) > 0:
                return ts[-1]
    return None


def match_events(ref_onsets, target_onsets, max_diff=0.25):
    """Match events from target to reference within max_diff seconds."""
    if len(target_onsets) == 0:
        return np.full(len(ref_onsets), np.nan)

    matched = []
    for ref_t in ref_onsets:
        diffs = np.abs(target_onsets - ref_t)
        min_diff = np.min(diffs)
        if min_diff < max_diff:
            matched.append(target_onsets[np.argmin(diffs)])
        else:
            matched.append(np.nan)
    return np.array(matched)


def load_xdf_with_dejitter(filename, dejitter_streams):
    """Load XDF with selective dejittering."""
    streams, header = pyxdf.load_xdf(
        filename,
        synchronize_clocks=True,
        handle_clock_resets=True,
        dejitter_timestamps=False,
    )

    streams_to_dejitter = []
    for i, s in enumerate(streams):
        name = s["info"].get("name", ["Unnamed"])[0]
        if name in dejitter_streams:
            streams_to_dejitter.append(i)

    if streams_to_dejitter:
        streams2, _ = pyxdf.load_xdf(
            filename,
            synchronize_clocks=True,
            handle_clock_resets=True,
            dejitter_timestamps=True,
        )
        for i in streams_to_dejitter:
            streams[i] = streams2[i]

    return streams, header


def evaluate_channel(lux_onsets, optics_signal, optics_ts, experiment_end):
    """
    Evaluate how good a given OPTICS channel is for event detection.
    Returns (diffs_array, std_dev, match_rate) where lower std_dev = more stable.
    """
    if optics_signal is None or len(optics_signal) < 10:
        return None, np.inf, 0

    # Only use data during experiment
    if experiment_end is not None:
        mask = optics_ts <= experiment_end + 10  # 10s buffer
        optics_signal = optics_signal[mask]
        optics_ts = optics_ts[mask]

    if len(optics_signal) < 10:
        return None, np.inf, 0

    optics_events = find_events_midpoint(optics_signal, optics_ts)
    if len(optics_events) < 10:
        return None, np.inf, 0

    optics_onsets = optics_ts[optics_events]

    # Only match lux events during experiment
    lux_mask = np.ones(len(lux_onsets), dtype=bool)
    if experiment_end is not None:
        lux_mask = lux_onsets <= experiment_end + 10

    valid_lux_onsets = lux_onsets[lux_mask]
    matched = match_events(valid_lux_onsets, optics_onsets)
    diffs = valid_lux_onsets - matched

    # Filter valid matches
    valid = ~np.isnan(diffs) & (np.abs(diffs) < 0.25)
    if np.sum(valid) < 10:
        return None, np.inf, 0

    match_rate = np.sum(valid) / len(valid_lux_onsets)
    valid_diffs = diffs[valid]

    return valid_diffs, np.std(valid_diffs), match_rate


def analyze_synchrony(filename, streams):
    """Analyze synchronization for a single file."""
    result = {
        "filename": filename,
        "streams_summary": summarize_streams(streams),
    }

    # Get experiment end time (last marker)
    experiment_end = get_experiment_end_time(streams)
    result["experiment_end"] = experiment_end

    # Get LUX (photosensor) as reference
    lux, lux_ts = find_channel(streams, "OpenSignals", "LUX", normalize="rolling")
    if lux is None:
        print("  WARNING: LUX channel not found")
        return result

    # Trim LUX to experiment duration
    if experiment_end is not None:
        mask = lux_ts <= experiment_end + 10
        lux = lux[mask]
        lux_ts = lux_ts[mask]
        print(f"  Experiment ends at {experiment_end:.1f}s, trimmed to {len(lux)} samples")

    # Find LUX events (reference)
    lux_events = find_events_midpoint(lux, lux_ts)
    if len(lux_events) == 0:
        print("  WARNING: No LUX events found")
        return result

    lux_onsets = lux_ts[lux_events]
    result["lux_onsets"] = lux_onsets
    result["lux"] = lux
    result["lux_ts"] = lux_ts

    # Get markers
    marker_onsets = find_marker_onsets(streams, marker_value=1)
    if len(marker_onsets) > 0:
        # Filter markers within experiment
        if experiment_end is not None:
            marker_onsets = marker_onsets[marker_onsets <= experiment_end + 10]
        matched_markers = match_events(lux_onsets, marker_onsets)
        result["diff_lux_markers"] = lux_onsets - matched_markers

    print(f"  LUX events: {len(lux_onsets)}, Markers: {len(marker_onsets)}")

    # For each device, try all OPTICS channels (both normal and inverted) and pick the best
    for device_label, device_mac in [("OLD", OLD_FIRMWARE_MAC), ("NEW", NEW_FIRMWARE_MAC)]:
        stream_name = f"Muse-OPTICS ({device_mac})"

        best_channel = None
        best_std = np.inf
        best_match_rate = 0
        best_signal = None
        best_ts = None
        best_inverted = False

        for ch_name in OPTICS_CHANNELS:
            for invert in [False, True]:
                signal, ts = find_channel(streams, stream_name, ch_name, normalize="rolling", invert=invert)

                if signal is None:
                    continue

                # Trim to experiment
                if experiment_end is not None:
                    mask = ts <= experiment_end + 10
                    signal_trimmed = signal[mask]
                    ts_trimmed = ts[mask]
                else:
                    signal_trimmed = signal
                    ts_trimmed = ts

                diffs, std, match_rate = evaluate_channel(lux_onsets, signal_trimmed, ts_trimmed, experiment_end)

                # Prefer channels with high match rate AND low std
                # Score: prioritize match rate > 80%, then minimize std
                if match_rate > 0.8 and std < best_std:
                    best_std = std
                    best_channel = ch_name
                    best_match_rate = match_rate
                    best_signal = signal_trimmed
                    best_ts = ts_trimmed
                    best_inverted = invert
                elif best_match_rate < 0.8 and match_rate > best_match_rate:
                    # If we don't have a good match yet, take higher match rate
                    best_std = std
                    best_channel = ch_name
                    best_match_rate = match_rate
                    best_signal = signal_trimmed
                    best_ts = ts_trimmed
                    best_inverted = invert

        if best_channel:
            result[f"optics_{device_label.lower()}"] = best_signal
            result[f"optics_{device_label.lower()}_ts"] = best_ts
            result[f"optics_{device_label.lower()}_channel"] = best_channel
            result[f"optics_{device_label.lower()}_inverted"] = best_inverted

            # Compute full diff array
            optics_events = find_events_midpoint(best_signal, best_ts)
            if len(optics_events) > 0:
                optics_onsets = best_ts[optics_events]
                matched = match_events(lux_onsets, optics_onsets)
                result[f"diff_lux_optics_{device_label.lower()}"] = lux_onsets - matched
            else:
                result[f"diff_lux_optics_{device_label.lower()}"] = np.full(len(lux_onsets), np.nan)

            inv_str = " (inverted)" if best_inverted else ""
            print(
                f"  {device_label}: {best_channel}{inv_str}, Std={best_std*1000:.1f}ms, Match={best_match_rate*100:.0f}%"
            )
        else:
            print(f"  {device_label}: No suitable OPTICS channel found")

    return result


def plot_comparison(results, save_prefix="clock_comparison"):
    """Create side-by-side comparison plots focusing on STABILITY."""
    n_clocks = len(results)
    clock_names = list(results.keys())

    # ===== Figure 1: Signal Alignment (per-file time ranges) =====
    fig1, axes1 = plt.subplots(1, n_clocks, figsize=(5 * n_clocks, 5), squeeze=False)
    fig1.suptitle("Signal Alignment (5-second window from each recording)", fontsize=14)

    for i, clock_name in enumerate(clock_names):
        result = results[clock_name]
        ax = axes1[0, i]
        ax.set_title(f"{clock_name}")

        # Use THIS file's time range (not common across files)
        lux_ts = result.get("lux_ts")
        if lux_ts is None or len(lux_ts) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        # Start 6 seconds into THIS recording
        xmin = lux_ts[0] + 6
        xmax = xmin + 5

        # Plot LUX
        lux = result.get("lux")
        if lux is not None:
            mask = (lux_ts >= xmin) & (lux_ts <= xmax)
            if np.any(mask):
                ax.plot(lux_ts[mask] - xmin, lux[mask], "b-", lw=1.5, label="LUX", alpha=0.9)

        # Plot OPTICS OLD
        optics = result.get("optics_old")
        optics_ts = result.get("optics_old_ts")
        ch_name = result.get("optics_old_channel", "?")
        inv = " (inv)" if result.get("optics_old_inverted", False) else ""
        if optics is not None and optics_ts is not None:
            mask = (optics_ts >= xmin) & (optics_ts <= xmax)
            if np.any(mask):
                ax.plot(
                    optics_ts[mask] - xmin,
                    optics[mask],
                    "r-",
                    lw=1,
                    label=f"OLD {ch_name.split('_')[-1]}{inv}",
                    alpha=0.7,
                )

        # Plot OPTICS NEW
        optics = result.get("optics_new")
        optics_ts = result.get("optics_new_ts")
        ch_name = result.get("optics_new_channel", "?")
        inv = " (inv)" if result.get("optics_new_inverted", False) else ""
        if optics is not None and optics_ts is not None:
            mask = (optics_ts >= xmin) & (optics_ts <= xmax)
            if np.any(mask):
                ax.plot(
                    optics_ts[mask] - xmin,
                    optics[mask],
                    "g-",
                    lw=1,
                    label=f"NEW {ch_name.split('_')[-1]}{inv}",
                    alpha=0.7,
                )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Normalized amplitude")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 5)
        ax.set_ylim(-0.1, 1.1)

    fig1.tight_layout()

    # ===== Figure 2: Delay Scatter Plots (full time series) =====
    fig2, axes2 = plt.subplots(2, n_clocks, figsize=(5 * n_clocks, 8), squeeze=False)
    fig2.suptitle("Event Onset Delays Over Time (stability visualization)", fontsize=14)

    for i, clock_name in enumerate(clock_names):
        result = results[clock_name]

        lux_onsets = result.get("lux_onsets")
        if lux_onsets is None or len(lux_onsets) == 0:
            continue

        time_minutes = (lux_onsets - lux_onsets[0]) / 60

        # Top row: LUX vs OPTICS
        ax_top = axes2[0, i]
        ax_top.set_title(f"{clock_name}")
        ax_top.axhline(0, color="black", ls="--", alpha=0.5)

        for label, key, color in [("OLD", "diff_lux_optics_old", "red"), ("NEW", "diff_lux_optics_new", "green")]:
            diff = result.get(key)
            if diff is not None:
                valid = ~np.isnan(diff) & (np.abs(diff) < 0.25)
                if np.any(valid):
                    ax_top.scatter(
                        time_minutes[valid], diff[valid] * 1000, c=color, s=15, alpha=0.5, label=f"OPTICS_{label}"
                    )

        ax_top.set_xlabel("Time (minutes)")
        ax_top.set_ylabel("LUX - OPTICS delay (ms)")
        ax_top.set_ylim(-150, 150)
        ax_top.legend(loc="upper right", fontsize=8)
        ax_top.grid(True, alpha=0.3)

        # Bottom row: LUX vs Markers
        ax_bot = axes2[1, i]
        ax_bot.axhline(0, color="black", ls="--", alpha=0.5)

        diff_markers = result.get("diff_lux_markers")
        if diff_markers is not None:
            valid = ~np.isnan(diff_markers) & (np.abs(diff_markers) < 0.25)
            if np.any(valid):
                ax_bot.scatter(
                    time_minutes[valid], diff_markers[valid] * 1000, c="purple", s=15, alpha=0.5, label="Markers"
                )

        ax_bot.set_xlabel("Time (minutes)")
        ax_bot.set_ylabel("LUX - Markers delay (ms)")
        ax_bot.set_ylim(-200, 200)
        ax_bot.legend(loc="upper right", fontsize=8)
        ax_bot.grid(True, alpha=0.3)

    fig2.tight_layout()

    # ===== Figure 2b: Signal Moments at Different Times =====
    # Show signal alignment at multiple time points throughout the recording
    time_moments = [1, 9, 18, 27]  # minutes into recording
    n_moments = len(time_moments)
    fig2b, axes2b = plt.subplots(n_moments, n_clocks, figsize=(4 * n_clocks, 3 * n_moments), squeeze=False)
    fig2b.suptitle("Signal Alignment at Different Time Points (5-second windows)", fontsize=14)

    for i, clock_name in enumerate(clock_names):
        result = results[clock_name]

        lux = result.get("lux")
        lux_ts = result.get("lux_ts")

        if lux is None or lux_ts is None or len(lux_ts) == 0:
            for row in range(n_moments):
                axes2b[row, i].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes2b[row, i].transAxes)
            continue

        recording_duration_min = (lux_ts[-1] - lux_ts[0]) / 60

        for row, t_min in enumerate(time_moments):
            ax = axes2b[row, i]

            # Calculate absolute time for this moment
            xmin = lux_ts[0] + t_min * 60
            xmax = xmin + 5  # 5-second window

            # Check if this time moment exists in the recording
            if t_min > recording_duration_min:
                ax.text(0.5, 0.5, f"Recording < {t_min}min", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{clock_name} @ {t_min}min")
                continue

            # Plot LUX
            mask = (lux_ts >= xmin) & (lux_ts <= xmax)
            if np.any(mask):
                ax.plot(lux_ts[mask] - xmin, lux[mask], "b-", lw=1.5, label="LUX", alpha=0.9)

            # Plot OPTICS OLD
            optics = result.get("optics_old")
            optics_ts = result.get("optics_old_ts")
            if optics is not None and optics_ts is not None:
                mask = (optics_ts >= xmin) & (optics_ts <= xmax)
                if np.any(mask):
                    ax.plot(optics_ts[mask] - xmin, optics[mask], "r-", lw=1, label="OLD", alpha=0.7)

            # Plot OPTICS NEW
            optics = result.get("optics_new")
            optics_ts = result.get("optics_new_ts")
            if optics is not None and optics_ts is not None:
                mask = (optics_ts >= xmin) & (optics_ts <= xmax)
                if np.any(mask):
                    ax.plot(optics_ts[mask] - xmin, optics[mask], "g-", lw=1, label="NEW", alpha=0.7)

            ax.set_xlim(0, 5)
            ax.set_ylim(-0.1, 1.1)
            ax.grid(True, alpha=0.3)

            if row == 0:
                ax.set_title(f"{clock_name}")
            if i == 0:
                ax.set_ylabel(f"@ {t_min}min")
            if row == n_moments - 1:
                ax.set_xlabel("Time (s)")
            if row == 0 and i == n_clocks - 1:
                ax.legend(loc="upper right", fontsize=7)

    fig2b.tight_layout()
    fig2b.savefig(f"{save_prefix}_moments.png", dpi=150, bbox_inches="tight")

    # ===== Collect Statistics =====
    stats_data = []
    for clock_name, result in results.items():
        for source, key in [
            ("OPTICS_OLD", "diff_lux_optics_old"),
            ("OPTICS_NEW", "diff_lux_optics_new"),
            ("Markers", "diff_lux_markers"),
        ]:
            diff = result.get(key)
            if diff is not None:
                valid = ~np.isnan(diff) & (np.abs(diff) < 0.25)
                if np.sum(valid) >= 10:
                    vals_ms = diff[valid] * 1000
                    stats_data.append(
                        {
                            "Clock": clock_name,
                            "Source": source,
                            "Mean (ms)": np.mean(vals_ms),
                            "Std (ms)": np.std(vals_ms),
                            "IQR (ms)": np.percentile(vals_ms, 75) - np.percentile(vals_ms, 25),
                            "N events": np.sum(valid),
                            "Match %": 100 * np.sum(valid) / len(diff),
                        }
                    )

    df_stats = pd.DataFrame(stats_data) if stats_data else None

    # ===== Figure 3: Stability Summary =====
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6))
    fig3.suptitle("SYNCHRONIZATION STABILITY (lower = better)", fontsize=14, fontweight="bold")

    if df_stats is not None and len(df_stats) > 0:
        clocks = df_stats["Clock"].unique()
        sources = ["OPTICS_OLD", "OPTICS_NEW", "Markers"]
        colors = {"OPTICS_OLD": "red", "OPTICS_NEW": "green", "Markers": "purple"}

        x = np.arange(len(clocks))
        width = 0.25

        for j, src in enumerate(sources):
            subset = df_stats[df_stats["Source"] == src]
            stds = []
            for c in clocks:
                row = subset[subset["Clock"] == c]
                stds.append(row["Std (ms)"].values[0] if len(row) > 0 else 0)

            ax3a.bar(x + j * width - width, stds, width, label=src, color=colors.get(src, "gray"), alpha=0.8)

        ax3a.set_xlabel("Clock Type", fontsize=12)
        ax3a.set_ylabel("Delay Std Dev (ms) - LOWER = MORE STABLE", fontsize=12)
        ax3a.set_title("Timing Stability by Clock Type")
        ax3a.set_xticks(x)
        ax3a.set_xticklabels(clocks)
        ax3a.legend()
        ax3a.grid(True, alpha=0.3, axis="y")

        # Right: Stability ranking
        optics_stats = df_stats[df_stats["Source"].str.contains("OPTICS")]
        if len(optics_stats) > 0:
            clock_scores = optics_stats.groupby("Clock")["Std (ms)"].mean().sort_values()

            bars = ax3b.barh(range(len(clock_scores)), clock_scores.values, color="steelblue", alpha=0.8)
            ax3b.set_yticks(range(len(clock_scores)))
            ax3b.set_yticklabels(clock_scores.index)
            ax3b.set_xlabel("Average OPTICS Std Dev (ms)", fontsize=12)
            ax3b.set_title("BEST CLOCK RANKING\n(lower = more stable = better)")
            ax3b.grid(True, alpha=0.3, axis="x")

            if len(bars) > 0:
                bars[0].set_color("green")
                bars[0].set_alpha(1.0)

            for idx, (clock, score) in enumerate(clock_scores.items()):
                label = f"{score:.1f} ms"
                if idx == 0:
                    label += " * BEST"
                ax3b.text(score + 1, idx, label, va="center", fontsize=10)

    fig3.tight_layout()

    # ===== Save Figures =====
    fig1.savefig(f"{save_prefix}_signals.png", dpi=150, bbox_inches="tight")
    fig2.savefig(f"{save_prefix}_delays.png", dpi=150, bbox_inches="tight")
    fig3.savefig(f"{save_prefix}_stability.png", dpi=150, bbox_inches="tight")

    print(
        f"\nSaved: {save_prefix}_signals.png, {save_prefix}_delays.png, {save_prefix}_moments.png, {save_prefix}_stability.png"
    )

    plt.show()

    return df_stats


def main():
    """Main function - analyzes all XDF files in current directory."""
    xdf_files = sorted([f for f in os.listdir(".") if f.endswith(".xdf")])

    if not xdf_files:
        print("No XDF files found in current directory!")
        print("Place XDF files here and re-run.")
        return

    print(f"Found {len(xdf_files)} XDF files: {xdf_files}")
    print("=" * 70)

    results = {}

    for filename in xdf_files:
        clock_name = Path(filename).stem.replace("test1_", "")

        print(f"\n{'='*70}")
        print(f"CLOCK TYPE: {clock_name.upper()}")
        print(f"File: {filename}")
        print("=" * 70)

        try:
            streams, _ = load_xdf_with_dejitter(filename, DEJITTER_TIMESTAMPS)
            result = analyze_synchrony(filename, streams)
            results[clock_name] = result
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()
            continue

    if not results:
        print("\nNo valid results to analyze!")
        return

    print("\n" + "=" * 70)
    print("CREATING COMPARISON PLOTS")
    print("=" * 70)

    stats_df = plot_comparison(results)

    if stats_df is not None:
        print("\n" + "=" * 70)
        print("SYNCHRONIZATION STATISTICS (sorted by stability)")
        print("=" * 70)
        print(stats_df.sort_values(["Source", "Std (ms)"]).to_string(index=False))

        print("\n" + "=" * 70)
        print("STABILITY RANKING (lower Std = more stable = BETTER)")
        print("=" * 70)

        optics_stats = stats_df[stats_df["Source"].str.contains("OPTICS")]
        if len(optics_stats) > 0:
            clock_scores = optics_stats.groupby("Clock")["Std (ms)"].mean().sort_values()

            print("\nBest clocks for stable OPTICS synchronization:")
            for rank, (clock, score) in enumerate(clock_scores.items(), 1):
                marker = " * RECOMMENDED" if rank == 1 else ""
                print(f"  {rank}. {clock}: {score:.1f} ms average jitter{marker}")

            print(f"\n→ Use '{clock_scores.index[0].upper()}' clock for most stable timing!")


if __name__ == "__main__":
    main()
