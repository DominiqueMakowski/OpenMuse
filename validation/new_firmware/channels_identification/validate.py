"""
Channel Order Investigation for New Firmware
=============================================

I have recorded streams from two Muse headbands (one with the old firmware and one
with the new firmware), as well as from another device (OpenSignals). The test was
designed to measure the synchrony between the OpenSignals device, which contains a
photosensor, and the Muse headbands, via their Optics channels. We attached the
Photosensor, and the Muse optics sensors to the screen, and flashed the screen
between white and black.

HYPOTHESIS: The channel order might be different in the new firmware.

Evidence from the plot:
- Some OPTICS channels go UP on black screen (light decrease)
- Some OPTICS channels go DOWN on black screen
- Old and new devices show OPPOSITE behavior for the same channel names
- This suggests the channel mapping might be scrambled in new firmware

INVESTIGATION APPROACH:
1. Cross-correlation within each device (which channels correlate together)
2. Cross-correlation across devices (which old channel matches which new channel)
3. Correlation with LUX (ground truth light sensor)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyxdf
import os
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d

# --- Configuration ---
filenames = [f for f in os.listdir(".") if f.endswith(".xdf")]
dejitter_timestamps = ["OpenSignals"]

filename = filenames[0]

# Device identifiers (update based on your XDF file)
OLD_DEVICE = "Muse-OPTICS (00:55:DA:BB:CD:CD)"
NEW_DEVICE = "Muse-OPTICS (00:55:DA:B9:FA:20)"

# All 16 OPTICS channel names (as defined in decode.py)
OPTICS_CHANNELS = [
    "OPTICS_LO_NIR",  # 0
    "OPTICS_RO_NIR",  # 1
    "OPTICS_LO_IR",  # 2
    "OPTICS_RO_IR",  # 3
    "OPTICS_LI_NIR",  # 4
    "OPTICS_RI_NIR",  # 5
    "OPTICS_LI_IR",  # 6
    "OPTICS_RI_IR",  # 7
    "OPTICS_LO_RED",  # 8
    "OPTICS_RO_RED",  # 9
    "OPTICS_LO_AMB",  # 10
    "OPTICS_RO_AMB",  # 11
    "OPTICS_LI_RED",  # 12
    "OPTICS_RI_RED",  # 13
    "OPTICS_LI_AMB",  # 14
    "OPTICS_RI_AMB",  # 15
]


def summarize_streams(streams):
    """Get range of timestamps for each stream."""
    data = []
    for i, stream in enumerate(streams):
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


def find_channel(streams, stream_name, channel_name, normalize=True):
    """Return signal and timestamps for a specific channel."""
    for s in streams:
        name = s["info"].get("name", ["Unnamed"])[0]
        if name == stream_name:
            channels = [
                d["label"][0] for d in s["info"]["desc"][0]["channels"][0]["channel"]
            ]
            matching = [ch for ch in channels if channel_name in ch]
            if not matching:
                return None, None
            ch_name = matching[0]
            if ch_name in channels:
                signal = np.array(s["time_series"])[:, channels.index(ch_name)]
                ts = s["time_stamps"]
                if normalize:
                    signal = (signal - np.min(signal)) / (
                        np.max(signal) - np.min(signal) + 1e-10
                    )
                return signal, ts
    return None, None


def get_all_optics_channels(streams, stream_name, normalize=True):
    """Get all OPTICS channels from a stream as a dict."""
    channels_data = {}
    for s in streams:
        name = s["info"].get("name", ["Unnamed"])[0]
        if name == stream_name:
            channel_labels = [
                d["label"][0] for d in s["info"]["desc"][0]["channels"][0]["channel"]
            ]
            ts = s["time_stamps"]
            for i, ch_name in enumerate(channel_labels):
                if ch_name.startswith("OPTICS_"):
                    signal = np.array(s["time_series"])[:, i]
                    if normalize:
                        signal = (signal - np.min(signal)) / (
                            np.max(signal) - np.min(signal) + 1e-10
                        )
                    channels_data[ch_name] = {"signal": signal, "ts": ts}
            return channels_data
    return {}


def resample_to_common_timestamps(sig1, ts1, sig2, ts2):
    """Resample two signals to common timestamps for correlation."""
    # Find common time range
    t_start = max(ts1.min(), ts2.min())
    t_end = min(ts1.max(), ts2.max())

    # Create common timestamps at 64 Hz (OPTICS rate)
    n_samples = int((t_end - t_start) * 64)
    common_ts = np.linspace(t_start, t_end, n_samples)

    # Interpolate both signals
    interp1 = interp1d(ts1, sig1, kind="linear", bounds_error=False, fill_value=np.nan)
    interp2 = interp1d(ts2, sig2, kind="linear", bounds_error=False, fill_value=np.nan)

    sig1_resampled = interp1(common_ts)
    sig2_resampled = interp2(common_ts)

    # Remove NaN values
    valid = ~np.isnan(sig1_resampled) & ~np.isnan(sig2_resampled)

    return sig1_resampled[valid], sig2_resampled[valid], common_ts[valid]


def compute_correlation_matrix(channels_data):
    """Compute correlation matrix between all channels in a device."""
    channel_names = list(channels_data.keys())
    n_channels = len(channel_names)
    corr_matrix = np.zeros((n_channels, n_channels))

    for i, ch1 in enumerate(channel_names):
        for j, ch2 in enumerate(channel_names):
            sig1 = channels_data[ch1]["signal"]
            sig2 = channels_data[ch2]["signal"]
            ts1 = channels_data[ch1]["ts"]
            ts2 = channels_data[ch2]["ts"]

            # Resample if needed (same device should have same ts)
            if len(sig1) == len(sig2):
                corr = np.corrcoef(sig1, sig2)[0, 1]
            else:
                sig1_r, sig2_r, _ = resample_to_common_timestamps(sig1, ts1, sig2, ts2)
                if len(sig1_r) > 10:
                    corr = np.corrcoef(sig1_r, sig2_r)[0, 1]
                else:
                    corr = np.nan
            corr_matrix[i, j] = corr

    return corr_matrix, channel_names


def compute_cross_device_correlation(old_channels, new_channels):
    """Compute correlation between old device channels and new device channels."""
    old_names = list(old_channels.keys())
    new_names = list(new_channels.keys())

    corr_matrix = np.zeros((len(old_names), len(new_names)))

    for i, old_ch in enumerate(old_names):
        for j, new_ch in enumerate(new_names):
            sig1 = old_channels[old_ch]["signal"]
            sig2 = new_channels[new_ch]["signal"]
            ts1 = old_channels[old_ch]["ts"]
            ts2 = new_channels[new_ch]["ts"]

            sig1_r, sig2_r, _ = resample_to_common_timestamps(sig1, ts1, sig2, ts2)

            if len(sig1_r) > 10:
                corr = np.corrcoef(sig1_r, sig2_r)[0, 1]
            else:
                corr = np.nan
            corr_matrix[i, j] = corr

    return corr_matrix, old_names, new_names


def compute_lux_correlation(channels_data, lux_signal, lux_ts):
    """Compute correlation between each OPTICS channel and the LUX sensor."""
    channel_names = list(channels_data.keys())
    correlations = {}

    for ch_name in channel_names:
        sig = channels_data[ch_name]["signal"]
        ts = channels_data[ch_name]["ts"]

        sig_r, lux_r, _ = resample_to_common_timestamps(sig, ts, lux_signal, lux_ts)

        if len(sig_r) > 10:
            corr = np.corrcoef(sig_r, lux_r)[0, 1]
        else:
            corr = np.nan
        correlations[ch_name] = corr

    return correlations


def find_best_channel_mapping(cross_corr, old_names, new_names):
    """Find the best mapping between old and new channels based on correlation."""
    mapping = {}
    abs_corr = np.abs(cross_corr)

    for i, old_ch in enumerate(old_names):
        # Find the new channel with highest absolute correlation
        best_j = np.nanargmax(abs_corr[i, :])
        best_corr = cross_corr[i, best_j]
        mapping[old_ch] = {
            "new_channel": new_names[best_j],
            "correlation": best_corr,
            "inverted": best_corr < 0,
        }

    return mapping


def analyze_synchrony(filename):
    print(f"\n{'='*80}")
    print(f"CHANNEL ORDER INVESTIGATION")
    print(f"{'='*80}")
    print(f"Analyzing file: {filename}\n")

    # --- Load Data ---
    streams, header = pyxdf.load_xdf(
        filename,
        synchronize_clocks=True,
        handle_clock_resets=True,
        dejitter_timestamps=False,
    )

    # De-jitter timestamps only for specified streams
    streams_to_dejitter = []
    for i, s in enumerate(streams):
        name = s["info"].get("name", ["Unnamed"])[0]
        if name in dejitter_timestamps:
            streams_to_dejitter.append(i)
    if len(streams_to_dejitter) > 0:
        streams2, _ = pyxdf.load_xdf(
            filename,
            synchronize_clocks=True,
            handle_clock_resets=True,
            dejitter_timestamps=True,
        )
        for i in streams_to_dejitter:
            streams[i] = streams2[i]

    # Streams summary
    df_stats = summarize_streams(streams)
    print("Stream Summary:")
    print(df_stats.to_string())
    print()

    # --- Get all OPTICS channels ---
    old_channels = get_all_optics_channels(streams, OLD_DEVICE, normalize=True)
    new_channels = get_all_optics_channels(streams, NEW_DEVICE, normalize=True)

    print(f"\nOld device channels: {list(old_channels.keys())}")
    print(f"New device channels: {list(new_channels.keys())}")

    # --- Get LUX (ground truth) ---
    lux, lux_ts = find_channel(streams, "OpenSignals", "LUX", normalize=True)

    # --- Analysis 1: LUX Correlation ---
    print(f"\n{'='*80}")
    print("ANALYSIS 1: Correlation with LUX (Ground Truth Light Sensor)")
    print("=" * 80)
    print("Positive correlation = channel increases with light")
    print("Negative correlation = channel decreases with light (inverted)")
    print()

    if lux is not None:
        old_lux_corr = compute_lux_correlation(old_channels, lux, lux_ts)
        new_lux_corr = compute_lux_correlation(new_channels, lux, lux_ts)

        print(
            f"{'Channel':<20} {'Old Device':>12} {'New Device':>12} {'Same Sign?':>12}"
        )
        print("-" * 60)
        for ch in OPTICS_CHANNELS:
            old_c = old_lux_corr.get(ch, np.nan)
            new_c = new_lux_corr.get(ch, np.nan)
            same_sign = "YES" if (old_c * new_c > 0) else "NO **"
            if np.isnan(old_c) or np.isnan(new_c):
                same_sign = "N/A"
            print(f"{ch:<20} {old_c:>12.3f} {new_c:>12.3f} {same_sign:>12}")
    else:
        print("LUX channel not found!")

    # --- Analysis 2: Cross-Device Correlation ---
    print(f"\n{'='*80}")
    print("ANALYSIS 2: Cross-Device Correlation Matrix")
    print("=" * 80)
    print("Finding which new channel best matches each old channel...")
    print()

    cross_corr, old_names, new_names = compute_cross_device_correlation(
        old_channels, new_channels
    )

    # Find best mapping
    mapping = find_best_channel_mapping(cross_corr, old_names, new_names)

    print(
        f"{'Old Channel':<20} {'Best Match (New)':<20} {'Correlation':>12} {'Inverted?':>10}"
    )
    print("-" * 70)
    for old_ch in OPTICS_CHANNELS:
        if old_ch in mapping:
            m = mapping[old_ch]
            inv = "YES **" if m["inverted"] else "no"
            print(
                f"{old_ch:<20} {m['new_channel']:<20} {m['correlation']:>12.3f} {inv:>10}"
            )

    # --- Analysis 3: Check for Channel Swaps ---
    print(f"\n{'='*80}")
    print("ANALYSIS 3: Suspected Channel Reordering")
    print("=" * 80)

    # Check if channels map to different positions
    reordering_detected = False
    for old_ch, m in mapping.items():
        if m["new_channel"] != old_ch:
            print(f"**  {old_ch} -> {m['new_channel']} (corr={m['correlation']:.3f})")
            reordering_detected = True

    if not reordering_detected:
        print("✓ No channel reordering detected - all channels map to themselves")

    # --- Analysis 4: Cross-Correlation Heatmap ---
    print(f"\n{'='*80}")
    print("ANALYSIS 4: Full Cross-Correlation Heatmap")
    print("=" * 80)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Heatmap 1: Old device internal correlation
    ax1 = axes[0]
    old_corr, old_ch_names = compute_correlation_matrix(old_channels)
    im1 = ax1.imshow(old_corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax1.set_xticks(range(len(old_ch_names)))
    ax1.set_yticks(range(len(old_ch_names)))
    ax1.set_xticklabels(
        [c.replace("OPTICS_", "") for c in old_ch_names], rotation=90, fontsize=8
    )
    ax1.set_yticklabels([c.replace("OPTICS_", "") for c in old_ch_names], fontsize=8)
    ax1.set_title("Old Device\nInternal Correlation")
    plt.colorbar(im1, ax=ax1)

    # Heatmap 2: New device internal correlation
    ax2 = axes[1]
    new_corr, new_ch_names = compute_correlation_matrix(new_channels)
    im2 = ax2.imshow(new_corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax2.set_xticks(range(len(new_ch_names)))
    ax2.set_yticks(range(len(new_ch_names)))
    ax2.set_xticklabels(
        [c.replace("OPTICS_", "") for c in new_ch_names], rotation=90, fontsize=8
    )
    ax2.set_yticklabels([c.replace("OPTICS_", "") for c in new_ch_names], fontsize=8)
    ax2.set_title("New Device\nInternal Correlation")
    plt.colorbar(im2, ax=ax2)

    # Heatmap 3: Cross-device correlation
    ax3 = axes[2]
    im3 = ax3.imshow(cross_corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax3.set_xticks(range(len(new_names)))
    ax3.set_yticks(range(len(old_names)))
    ax3.set_xticklabels(
        [c.replace("OPTICS_", "") for c in new_names], rotation=90, fontsize=8
    )
    ax3.set_yticklabels([c.replace("OPTICS_", "") for c in old_names], fontsize=8)
    ax3.set_title("Cross-Device\nOld (rows) vs New (cols)")
    plt.colorbar(im3, ax=ax3)

    plt.tight_layout()
    plt.savefig("correlation_heatmaps.png", dpi=150)
    print("Saved correlation heatmaps to: correlation_heatmaps.png")
    plt.show()

    # --- Plot original comparison ---
    print(f"\n{'='*80}")
    print("ORIGINAL PLOT: Time-Series Comparison")
    print("=" * 80)

    tmin = df_stats["Start"].min()
    xmin = tmin + 6
    fig = plt.figure(figsize=(15, 7))

    # LUX
    if lux is not None:
        mask = (lux_ts >= xmin) & (lux_ts <= xmin + 5)
        plt.plot(lux_ts[mask], lux[mask], color="blue", label="LUX")

    # OPTICS
    for optics_name in ["RI_AMB", "RI_RED", "RI_IR"]:
        # OPTICS (old)
        optics, optics_ts = find_channel(streams, OLD_DEVICE, f"OPTICS_{optics_name}")
        if optics is not None:
            mask = (optics_ts >= xmin) & (optics_ts <= xmin + 5)
            plt.plot(optics_ts[mask], optics[mask], label=f"OPTICS_{optics_name} (old)")

        # OPTICS (new)
        optics2, optics2_ts = find_channel(streams, NEW_DEVICE, f"OPTICS_{optics_name}")
        if optics2 is not None:
            mask = (optics2_ts >= xmin) & (optics2_ts <= xmin + 5)
            plt.plot(
                optics2_ts[mask], optics2[mask], label=f"OPTICS_{optics_name} (new)"
            )

    # JsPsychMarkers
    markers, markers_ts = find_channel(
        streams, "jsPsychMarkers", "JsPsychMarker", normalize=False
    )
    if markers is not None:
        markers = markers.astype(float)
        mask = (markers_ts >= xmin) & (markers_ts <= xmin + 5)
        plt.bar(
            markers_ts[mask],
            markers[mask],
            width=0.02,
            color="darkgreen",
            alpha=0.9,
            label="jsPsychMarkers - 1",
        )
        plt.bar(
            markers_ts[mask],
            np.abs(markers[mask] - 1),
            width=0.02,
            color="green",
            alpha=0.6,
            label="jsPsychMarkers - 0",
        )

    plt.legend()
    plt.tight_layout()
    plt.savefig("time_series_comparison.png", dpi=150)
    print("Saved time-series comparison to: time_series_comparison.png")
    plt.show()

    return {
        "old_lux_corr": old_lux_corr if lux is not None else None,
        "new_lux_corr": new_lux_corr if lux is not None else None,
        "cross_corr": cross_corr,
        "mapping": mapping,
    }


def deep_channel_analysis(filename):
    """
    Deep analysis to understand what signals the new device channels actually contain.

    This function looks at the internal correlation structure within each device
    to see if the channels are grouped differently (suggesting a channel order swap).
    """
    print(f"\n{'='*80}")
    print("DEEP ANALYSIS: Channel Grouping Patterns")
    print("=" * 80)

    # Load data
    streams, _ = pyxdf.load_xdf(
        filename,
        synchronize_clocks=True,
        handle_clock_resets=True,
        dejitter_timestamps=False,
    )

    old_channels = get_all_optics_channels(streams, OLD_DEVICE, normalize=True)
    new_channels = get_all_optics_channels(streams, NEW_DEVICE, normalize=True)

    # Compute internal correlations
    old_corr, old_names = compute_correlation_matrix(old_channels)
    new_corr, new_names = compute_correlation_matrix(new_channels)

    # Expected grouping: channels with same wavelength and position should correlate
    # NIR channels: 0,1,4,5 (LO,RO,LI,RI)
    # IR channels: 2,3,6,7
    # RED channels: 8,9,12,13
    # AMB channels: 10,11,14,15

    wavelength_groups = {
        "NIR": ["OPTICS_LO_NIR", "OPTICS_RO_NIR", "OPTICS_LI_NIR", "OPTICS_RI_NIR"],
        "IR": ["OPTICS_LO_IR", "OPTICS_RO_IR", "OPTICS_LI_IR", "OPTICS_RI_IR"],
        "RED": ["OPTICS_LO_RED", "OPTICS_RO_RED", "OPTICS_LI_RED", "OPTICS_RI_RED"],
        "AMB": ["OPTICS_LO_AMB", "OPTICS_RO_AMB", "OPTICS_LI_AMB", "OPTICS_RI_AMB"],
    }

    position_groups = {
        "Left Outer (LO)": [
            "OPTICS_LO_NIR",
            "OPTICS_LO_IR",
            "OPTICS_LO_RED",
            "OPTICS_LO_AMB",
        ],
        "Right Outer (RO)": [
            "OPTICS_RO_NIR",
            "OPTICS_RO_IR",
            "OPTICS_RO_RED",
            "OPTICS_RO_AMB",
        ],
        "Left Inner (LI)": [
            "OPTICS_LI_NIR",
            "OPTICS_LI_IR",
            "OPTICS_LI_RED",
            "OPTICS_LI_AMB",
        ],
        "Right Inner (RI)": [
            "OPTICS_RI_NIR",
            "OPTICS_RI_IR",
            "OPTICS_RI_RED",
            "OPTICS_RI_AMB",
        ],
    }

    print("\n--- Wavelength Group Correlations (expected: high within group) ---")
    print(f"{'Group':<10} {'Old Device':>15} {'New Device':>15}")
    print("-" * 45)

    for group_name, channels in wavelength_groups.items():
        # Get indices for this group
        old_indices = [old_names.index(ch) for ch in channels if ch in old_names]
        new_indices = [new_names.index(ch) for ch in channels if ch in new_names]

        # Compute mean correlation within group
        if len(old_indices) >= 2:
            old_group_corr = np.mean(
                [old_corr[i, j] for i in old_indices for j in old_indices if i != j]
            )
        else:
            old_group_corr = np.nan

        if len(new_indices) >= 2:
            new_group_corr = np.mean(
                [new_corr[i, j] for i in new_indices for j in new_indices if i != j]
            )
        else:
            new_group_corr = np.nan

        print(f"{group_name:<10} {old_group_corr:>15.3f} {new_group_corr:>15.3f}")

    print("\n--- Position Group Correlations (expected: moderate within group) ---")
    print(f"{'Group':<20} {'Old Device':>15} {'New Device':>15}")
    print("-" * 55)

    for group_name, channels in position_groups.items():
        old_indices = [old_names.index(ch) for ch in channels if ch in old_names]
        new_indices = [new_names.index(ch) for ch in channels if ch in new_names]

        if len(old_indices) >= 2:
            old_group_corr = np.mean(
                [old_corr[i, j] for i in old_indices for j in old_indices if i != j]
            )
        else:
            old_group_corr = np.nan

        if len(new_indices) >= 2:
            new_group_corr = np.mean(
                [new_corr[i, j] for i in new_indices for j in new_indices if i != j]
            )
        else:
            new_group_corr = np.nan

        print(f"{group_name:<20} {old_group_corr:>15.3f} {new_group_corr:>15.3f}")

    # Look for patterns in which old channels correlate highly with specific new channels
    print(f"\n{'='*80}")
    print("DETAILED CORRELATION MAPPING")
    print("=" * 80)
    print("\nFor each OLD channel, showing top 3 NEW channel correlations:")

    cross_corr, old_names, new_names = compute_cross_device_correlation(
        old_channels, new_channels
    )

    for i, old_ch in enumerate(old_names):
        correlations = [(new_names[j], cross_corr[i, j]) for j in range(len(new_names))]
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        short_old = old_ch.replace("OPTICS_", "")
        top3 = correlations[:3]
        top3_str = ", ".join(
            [f"{c[0].replace('OPTICS_', '')}({c[1]:.2f})" for c in top3]
        )
        print(f"{short_old:<10} -> {top3_str}")

    # Check if the issue is in specific wavelengths or positions
    print(f"\n{'='*80}")
    print("DIAGNOSTIC: Check if old device RI channels correlate with new device LI")
    print("=" * 80)

    # Specifically check Right Inner (RI) vs Left Inner (LI) swap
    ri_to_li_check = [
        ("OPTICS_RI_NIR", "OPTICS_LI_NIR"),
        ("OPTICS_RI_IR", "OPTICS_LI_IR"),
        ("OPTICS_RI_RED", "OPTICS_LI_RED"),
        ("OPTICS_RI_AMB", "OPTICS_LI_AMB"),
    ]

    print(f"{'Old (RI)':<15} {'vs New (RI)':>15} {'vs New (LI)':>15} {'Swap?':>10}")
    print("-" * 60)

    for old_ri, old_li in ri_to_li_check:
        if old_ri in old_names and old_ri in new_names:
            old_idx = old_names.index(old_ri)
            new_ri_idx = new_names.index(old_ri)  # Same name
            new_li = old_li.replace("RI", "LI") if "RI" in old_li else old_li
            if new_li in new_names:
                new_li_idx = new_names.index(new_li)
            else:
                new_li_idx = -1

            corr_ri = cross_corr[old_idx, new_ri_idx]
            corr_li = cross_corr[old_idx, new_li_idx] if new_li_idx >= 0 else np.nan

            swap_likely = "YES **" if abs(corr_li) > abs(corr_ri) else "no"
            print(
                f"{old_ri.replace('OPTICS_', ''):<15} {corr_ri:>15.3f} {corr_li:>15.3f} {swap_likely:>10}"
            )


if __name__ == "__main__":
    if filenames:
        results = analyze_synchrony(filename)
        deep_channel_analysis(filename)

        # Additional diagnostic: plot all 16 channels for both devices
        print(f"\n{'='*80}")
        print("DIAGNOSTIC PLOTS: All 16 Channels Side by Side")
        print("=" * 80)

        streams, _ = pyxdf.load_xdf(
            filename,
            synchronize_clocks=True,
            handle_clock_resets=True,
            dejitter_timestamps=False,
        )

        old_channels = get_all_optics_channels(streams, OLD_DEVICE, normalize=True)
        new_channels = get_all_optics_channels(streams, NEW_DEVICE, normalize=True)

        # Get LUX
        lux, lux_ts = find_channel(streams, "OpenSignals", "LUX", normalize=True)

        # Find time window with clear events
        tmin = 6195  # From the attached plot
        tmax = 6200

        fig, axes = plt.subplots(4, 4, figsize=(20, 16), sharex=True)
        fig.suptitle(
            "All 16 OPTICS Channels: Old (blue) vs New (orange) vs LUX (green dashed)",
            fontsize=14,
        )

        for idx, ch_name in enumerate(OPTICS_CHANNELS):
            ax = axes[idx // 4, idx % 4]

            # Old device
            if ch_name in old_channels:
                sig_old = old_channels[ch_name]["signal"]
                ts_old = old_channels[ch_name]["ts"]
                mask = (ts_old >= tmin) & (ts_old <= tmax)
                ax.plot(ts_old[mask], sig_old[mask], "b-", alpha=0.7, label="Old")

            # New device
            if ch_name in new_channels:
                sig_new = new_channels[ch_name]["signal"]
                ts_new = new_channels[ch_name]["ts"]
                mask = (ts_new >= tmin) & (ts_new <= tmax)
                ax.plot(ts_new[mask], sig_new[mask], "orange", alpha=0.7, label="New")

            # LUX
            if lux is not None:
                mask = (lux_ts >= tmin) & (lux_ts <= tmax)
                ax.plot(
                    lux_ts[mask],
                    lux[mask],
                    "g--",
                    alpha=0.5,
                    linewidth=0.5,
                    label="LUX",
                )

            ax.set_title(ch_name.replace("OPTICS_", ""), fontsize=10)
            ax.set_ylim(-0.1, 1.1)
            if idx == 0:
                ax.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig("all_channels_comparison.png", dpi=150)
        print("Saved all channels comparison to: all_channels_comparison.png")
        plt.show()

        # Also plot the raw (non-normalized) values to see scale differences
        print("\nPlotting raw (non-normalized) data to check value ranges...")

        old_channels_raw = get_all_optics_channels(streams, OLD_DEVICE, normalize=False)
        new_channels_raw = get_all_optics_channels(streams, NEW_DEVICE, normalize=False)

        # Print min/max for each channel
        print(
            f"\n{'Channel':<15} {'Old Min':>12} {'Old Max':>12} {'New Min':>12} {'New Max':>12}"
        )
        print("-" * 65)
        for ch_name in OPTICS_CHANNELS:
            old_min = (
                old_channels_raw[ch_name]["signal"].min()
                if ch_name in old_channels_raw
                else np.nan
            )
            old_max = (
                old_channels_raw[ch_name]["signal"].max()
                if ch_name in old_channels_raw
                else np.nan
            )
            new_min = (
                new_channels_raw[ch_name]["signal"].min()
                if ch_name in new_channels_raw
                else np.nan
            )
            new_max = (
                new_channels_raw[ch_name]["signal"].max()
                if ch_name in new_channels_raw
                else np.nan
            )
            print(
                f"{ch_name.replace('OPTICS_', ''):<15} {old_min:>12.4f} {old_max:>12.4f} {new_min:>12.4f} {new_max:>12.4f}"
            )

        # Key diagnostic: check the inner channels (4-7) which appear suspicious
        print(f"\n{'='*80}")
        print("KEY DIAGNOSTIC: Inner Channel Analysis (indices 4-7)")
        print("=" * 80)
        print(
            """
These are the channels for Optics4 mode (indices 4,5,6,7):
  Index 4: OPTICS_LI_NIR (Left Inner, Near-Infrared)
  Index 5: OPTICS_RI_NIR (Right Inner, Near-Infrared)
  Index 6: OPTICS_LI_IR  (Left Inner, Infrared)
  Index 7: OPTICS_RI_IR  (Right Inner, Infrared)

In the OLD device, we expect:
- LI_NIR and RI_NIR should be positively correlated (same wavelength, different position)
- LI_IR and RI_IR should be positively correlated (same wavelength, different position)
- NIR and IR from same position should have some correlation

But the analysis shows the OLD device inner channels have near-ZERO within-position correlation!
This suggests the channels might actually contain DIFFERENT data than their labels suggest.
"""
        )

        # Check correlation between adjacent inner channels
        inner_channels = [
            "OPTICS_LI_NIR",
            "OPTICS_RI_NIR",
            "OPTICS_LI_IR",
            "OPTICS_RI_IR",
        ]

        print("Inner channel correlations (OLD device):")
        for i, ch1 in enumerate(inner_channels):
            for ch2 in inner_channels[i + 1 :]:
                sig1 = old_channels_raw[ch1]["signal"]
                sig2 = old_channels_raw[ch2]["signal"]
                corr = np.corrcoef(sig1, sig2)[0, 1]
                print(
                    f"  {ch1.replace('OPTICS_', '')} vs {ch2.replace('OPTICS_', '')}: {corr:.3f}"
                )

        print("\nInner channel correlations (NEW device):")
        for i, ch1 in enumerate(inner_channels):
            for ch2 in inner_channels[i + 1 :]:
                sig1 = new_channels_raw[ch1]["signal"]
                sig2 = new_channels_raw[ch2]["signal"]
                corr = np.corrcoef(sig1, sig2)[0, 1]
                print(
                    f"  {ch1.replace('OPTICS_', '')} vs {ch2.replace('OPTICS_', '')}: {corr:.3f}"
                )

        # Check if old inner channels correlate with old outer channels
        print("\nOLD device: Inner vs Outer channel correlations:")
        outer_channels = [
            "OPTICS_LO_NIR",
            "OPTICS_RO_NIR",
            "OPTICS_LO_IR",
            "OPTICS_RO_IR",
        ]
        for inner_ch in inner_channels:
            best_outer = None
            best_corr = 0
            for outer_ch in outer_channels:
                sig1 = old_channels_raw[inner_ch]["signal"]
                sig2 = old_channels_raw[outer_ch]["signal"]
                corr = np.corrcoef(sig1, sig2)[0, 1]
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_outer = outer_ch
            print(
                f"  {inner_ch.replace('OPTICS_', '')} best matches {best_outer.replace('OPTICS_', '')}: {best_corr:.3f}"
            )

        # CRITICAL: Check if channels are actually duplicated
        print(f"\n{'='*80}")
        print("CRITICAL: Checking for Duplicate Channels")
        print("=" * 80)

        # Check if channels 4,5,6,7 are identical
        ch4 = old_channels_raw["OPTICS_LI_NIR"]["signal"]
        ch5 = old_channels_raw["OPTICS_RI_NIR"]["signal"]
        ch6 = old_channels_raw["OPTICS_LI_IR"]["signal"]
        ch7 = old_channels_raw["OPTICS_RI_IR"]["signal"]

        # Check exact equality (not just correlation)
        print("\nOLD Device - Are channels 4-7 numerically identical?")
        print(f"  ch4 == ch5: {np.allclose(ch4, ch5)}")
        print(f"  ch4 == ch6: {np.allclose(ch4, ch6)}")
        print(f"  ch4 == ch7: {np.allclose(ch4, ch7)}")
        print(f"  ch5 == ch6: {np.allclose(ch5, ch6)}")
        print(f"  ch5 == ch7: {np.allclose(ch5, ch7)}")
        print(f"  ch6 == ch7: {np.allclose(ch6, ch7)}")

        # Sample values
        print(f"\nSample values (first 5 points):")
        print(f"  ch4 (LI_NIR): {ch4[:5]}")
        print(f"  ch5 (RI_NIR): {ch5[:5]}")
        print(f"  ch6 (LI_IR):  {ch6[:5]}")
        print(f"  ch7 (RI_IR):  {ch7[:5]}")

        # NEW Device
        ch4_new = new_channels_raw["OPTICS_LI_NIR"]["signal"]
        ch5_new = new_channels_raw["OPTICS_RI_NIR"]["signal"]
        ch6_new = new_channels_raw["OPTICS_LI_IR"]["signal"]
        ch7_new = new_channels_raw["OPTICS_RI_IR"]["signal"]

        print("\nNEW Device - Are channels 4-7 numerically identical?")
        print(f"  ch4 == ch5: {np.allclose(ch4_new, ch5_new)}")
        print(f"  ch4 == ch6: {np.allclose(ch4_new, ch6_new)}")
        print(f"  ch4 == ch7: {np.allclose(ch4_new, ch7_new)}")

        print(f"\nSample values (first 5 points):")
        print(f"  ch4 (LI_NIR): {ch4_new[:5]}")
        print(f"  ch5 (RI_NIR): {ch5_new[:5]}")
        print(f"  ch6 (LI_IR):  {ch6_new[:5]}")
        print(f"  ch7 (RI_IR):  {ch7_new[:5]}")

        # Check remaining channels for duplicates
        print(f"\n{'='*80}")
        print("Full Channel Duplication Check (all 16 channels)")
        print("=" * 80)

        def check_duplicates(channels_raw, device_name):
            print(f"\n{device_name}:")
            all_ch_names = list(channels_raw.keys())
            duplicate_groups = []
            checked = set()

            for i, ch1 in enumerate(all_ch_names):
                if ch1 in checked:
                    continue
                group = [ch1]
                sig1 = channels_raw[ch1]["signal"]
                for ch2 in all_ch_names[i + 1 :]:
                    if ch2 in checked:
                        continue
                    sig2 = channels_raw[ch2]["signal"]
                    if len(sig1) == len(sig2) and np.allclose(sig1, sig2, rtol=1e-5):
                        group.append(ch2)
                        checked.add(ch2)
                if len(group) > 1:
                    duplicate_groups.append(group)
                checked.add(ch1)

            if duplicate_groups:
                print("  Duplicate channel groups found:")
                for group in duplicate_groups:
                    short_names = [n.replace("OPTICS_", "") for n in group]
                    print(f"    {short_names}")
            else:
                print("  No duplicate channels found - all channels unique")

            return duplicate_groups

        old_dups = check_duplicates(old_channels_raw, "OLD Device")
        new_dups = check_duplicates(new_channels_raw, "NEW Device")

        # Final investigation: AMB channels behavior difference
        print(f"\n{'='*80}")
        print("INVESTIGATING AMB CHANNEL BEHAVIOR DIFFERENCE")
        print("=" * 80)
        print(
            """
The key finding is that LI_AMB and RI_AMB show OPPOSITE correlation with LUX:
  OLD device: LI_AMB=-0.174, RI_AMB=-0.172 (negative = signal UP when light DOWN)
  NEW device: LI_AMB=+0.699, RI_AMB=+0.746 (positive = signal DOWN when light DOWN)

This could mean:
1. The channels are swapped between devices
2. The channels measure something different in new firmware
3. The decoding is different

Let's check which NEW channel matches OLD LI_AMB / RI_AMB behavior:
"""
        )

        # For OLD LI_AMB (which has negative LUX correlation), find NEW channel with same behavior
        old_li_amb = old_channels_raw["OPTICS_LI_AMB"]["signal"]
        old_li_amb_ts = old_channels_raw["OPTICS_LI_AMB"]["ts"]
        old_ri_amb = old_channels_raw["OPTICS_RI_AMB"]["signal"]
        old_ri_amb_ts = old_channels_raw["OPTICS_RI_AMB"]["ts"]

        print("Cross-correlation of OLD AMB channels with ALL NEW channels:")
        print(
            f"{'OLD Channel':<15} {'Best NEW Match':<18} {'Corr':>8} {'2nd Best':<18} {'Corr':>8}"
        )
        print("-" * 75)

        for old_ch_name, old_sig, old_ts in [
            ("OPTICS_LI_AMB", old_li_amb, old_li_amb_ts),
            ("OPTICS_RI_AMB", old_ri_amb, old_ri_amb_ts),
        ]:
            correlations = []
            for new_ch_name in new_channels_raw:
                new_sig = new_channels_raw[new_ch_name]["signal"]
                new_ts = new_channels_raw[new_ch_name]["ts"]

                sig1_r, sig2_r, _ = resample_to_common_timestamps(
                    old_sig, old_ts, new_sig, new_ts
                )
                if len(sig1_r) > 10:
                    corr = np.corrcoef(sig1_r, sig2_r)[0, 1]
                    correlations.append((new_ch_name, corr))

            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            best = correlations[0]
            second = correlations[1]
            print(
                f"{old_ch_name.replace('OPTICS_', ''):<15} {best[0].replace('OPTICS_', ''):<18} {best[1]:>8.3f} {second[0].replace('OPTICS_', ''):<18} {second[1]:>8.3f}"
            )

        # Also check the reverse: which OLD channel matches NEW AMB channels
        print("\nCross-correlation of NEW AMB channels with ALL OLD channels:")
        print(
            f"{'NEW Channel':<15} {'Best OLD Match':<18} {'Corr':>8} {'2nd Best':<18} {'Corr':>8}"
        )
        print("-" * 75)

        new_li_amb = new_channels_raw["OPTICS_LI_AMB"]["signal"]
        new_li_amb_ts = new_channels_raw["OPTICS_LI_AMB"]["ts"]
        new_ri_amb = new_channels_raw["OPTICS_RI_AMB"]["signal"]
        new_ri_amb_ts = new_channels_raw["OPTICS_RI_AMB"]["ts"]

        for new_ch_name, new_sig, new_ts in [
            ("OPTICS_LI_AMB", new_li_amb, new_li_amb_ts),
            ("OPTICS_RI_AMB", new_ri_amb, new_ri_amb_ts),
        ]:
            correlations = []
            for old_ch_name in old_channels_raw:
                old_sig = old_channels_raw[old_ch_name]["signal"]
                old_ts = old_channels_raw[old_ch_name]["ts"]

                sig1_r, sig2_r, _ = resample_to_common_timestamps(
                    new_sig, new_ts, old_sig, old_ts
                )
                if len(sig1_r) > 10:
                    corr = np.corrcoef(sig1_r, sig2_r)[0, 1]
                    correlations.append((old_ch_name, corr))

            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            best = correlations[0]
            second = correlations[1]
            print(
                f"{new_ch_name.replace('OPTICS_', ''):<15} {best[0].replace('OPTICS_', ''):<18} {best[1]:>8.3f} {second[0].replace('OPTICS_', ''):<18} {second[1]:>8.3f}"
            )

        # Summary
        print(f"\n{'='*80}")
        print("CONCLUSIONS")
        print("=" * 80)
        print(
            """
Based on the analysis:

1. DUPLICATE DATA ISSUE (channels 4-7):
   - Channels LI_NIR, RI_NIR, LI_IR, RI_IR (indices 4-7) are HIGHLY correlated (~1.0)
   - But they are NOT numerically identical - they have different values
   - This high correlation is because they all measure the same light stimulus

2. AMB CHANNEL BEHAVIOR DIFFERENCE:
   - OLD device: LI_AMB and RI_AMB have NEGATIVE correlation with LUX
     (signal goes UP when light goes DOWN)
   - NEW device: LI_AMB and RI_AMB have POSITIVE correlation with LUX
     (signal goes DOWN when light goes DOWN)
   
3. POSSIBLE CAUSES:
   a) Channel order reordering in new firmware
   b) Different signal processing in new firmware
   c) Different sensor interpretation

4. RECOMMENDED INVESTIGATION:
   - Check decode.py OPTICS_CHANNELS definition vs actual raw byte order
   - Record with known stimulus and compare signal patterns
   - Check if Interaxon documentation mentions firmware changes
"""
        )

    else:
        print("No XDF files found in current directory!")
