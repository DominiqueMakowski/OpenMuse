"""
I have recorded streams from the Muse headband, as well as from another device (OpenSignals). I recorded them using the latest version of LabRecorder, and try to open them using pyxdf (https://github.com/xdf-modules/pyxdf/blob/main/src/pyxdf/pyxdf.py).
The test was designed to measure the synchrony between the OpenSignals device, which contains a photosensor, and the Muse headband, via its Optics channels. We attached the Photosensor, and the Muse optics sensor to the screen, and flashed the screen between white and black. The experiment was also sending a digital trigger, JsPsychMarker, on screen change. The goal is to see whether the event (white -> black screen) onsets are aligned between the two streams.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyxdf
import scipy.signal

# --- Configuration ---
filename = "./validate_synchronization6_WindowedClock.xdf"
# filename = "./validate_synchronization7_WindowedClock.xdf"
# filename = "./validate_synchronization8_RobustClock.xdf"
# filename = "./validate_synchronization9_StableClockNew.xdf"
# filename = "./validate_synchronization10_AdaptiveClock.xdf"
dejitter_timestamps = ["OpenSignals"]


# --- Load Data ---
streams, header = pyxdf.load_xdf(
    filename,
    synchronize_clocks=True,
    handle_clock_resets=True,
    dejitter_timestamps=False,
    # select_streams=select_streams,
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
        # select_streams=select_streams,
    )
    for i in streams_to_dejitter:
        streams[i] = streams2[i]


# Get range of timestamps for each stream
tmin = np.nan
tmax = np.nan
for i, stream in enumerate(streams):
    name = stream["info"].get("name", ["Unnamed"])[0]
    if len(stream["time_stamps"]) == 0:
        print(f"{i} - Stream {name}: empty")
        continue
    ts_min = stream["time_stamps"].min()
    ts_max = stream["time_stamps"].max()
    tmin = np.nanmin([tmin, ts_min])
    tmax = np.nanmax([tmax, ts_max])
    duration = ts_max - ts_min
    n_samples = len(stream["time_stamps"])
    nominal_srate = float(stream["info"]["nominal_srate"][0])
    effective_srate = n_samples / duration
    print(
        f"{i} - Stream {name}: {n_samples} samples, duration {duration:.2f} s (from {ts_min:.2f} to {ts_max:.2f}), nominal srate {nominal_srate:.2f} Hz, effective srate {effective_srate:.2f} Hz"
    )


# --- Plot streams ---
xmin = tmin + 6
fig = plt.figure(figsize=(15, 7))
for i, s in enumerate(streams):
    name = s["info"].get("name", ["Unnamed"])[0]
    channels = [d["label"][0] for d in s["info"]["desc"][0]["channels"][0]["channel"]]
    if name in ["OpenSignals"]:
        lux_name = [ch for ch in channels if "LUX" in ch][0]
        lux = s["time_series"][:, channels.index(lux_name)]
        lux = (lux - np.min(lux)) / (np.max(lux) - np.min(lux))
        lux_ts = s["time_stamps"]
        mask = (lux_ts >= xmin) & (lux_ts <= xmin + 5)
        plt.plot(lux_ts[mask], lux[mask], color="blue", label="LUX")
    if name in ["Muse_OPTICS"]:
        optics = s["time_series"][:, channels.index("OPTICS_RI_AMB")]
        optics = (optics - np.min(optics)) / (np.max(optics) - np.min(optics))
        optics_ts = s["time_stamps"]
        mask = (optics_ts >= xmin) & (optics_ts <= xmin + 5)
        plt.plot(optics_ts[mask], optics[mask], color="red", label="OPTICS")
    if name in ["jsPsychMarkers"]:
        markers = np.array(s["time_series"]).astype("int").flatten()
        markers_ts = s["time_stamps"]
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
plt.show()


events_lux = nk.events_find(lux, threshold=0.8, threshold_keep="below", duration_min=5)
events_optics = nk.events_find(
    optics, threshold=0.5, threshold_keep="above", duration_min=5
)
print(
    f"N events LUX: {len(events_lux['onset'])}, N events OPTICS: {len(events_optics['onset'])}"
)
onsets_lux = lux_ts[events_lux["onset"]]
onsets_optics = optics_ts[events_optics["onset"]]
onsets_optics = nk.find_closest(onsets_lux, onsets_optics)
onsets_markers = markers_ts[markers == 1]
onsets_markers = nk.find_closest(onsets_lux, onsets_markers)

# Make differences
diff_luxoptics = onsets_lux - onsets_optics
mask_luxoptics = np.abs(diff_luxoptics) < 0.8
np.median(diff_luxoptics)

diff_luxmarkers = onsets_lux - onsets_markers
mask_luxmarkers = np.abs(diff_luxmarkers) < 0.8
np.median(diff_luxmarkers)

diff_opticsmarkers = onsets_optics - onsets_markers
mask_opticsmarkers = np.abs(diff_opticsmarkers) < 2
np.median(diff_opticsmarkers)


# plt.plot((onsets_lux - min(onsets_lux)) / 60, diff_luxoptics)
plt.axhline(0, color="black", linestyle="--")
plt.plot(
    (onsets_lux[mask_luxoptics] - min(onsets_lux)) / 60,
    diff_luxoptics[mask_luxoptics],
    label="LUX - OPTICS",
)
plt.plot(
    (onsets_lux[mask_luxmarkers] - min(onsets_lux)) / 60,
    diff_luxmarkers[mask_luxmarkers],
    label="LUX - MARKERS",
)
plt.plot(
    (onsets_optics[mask_opticsmarkers] - min(onsets_optics)) / 60,
    diff_opticsmarkers[mask_opticsmarkers],
    label="OPTICS - MARKERS",
)
plt.title(f"Device event onsets differences (WindowedClock)")
plt.xlabel("Time")
plt.ylabel("Difference (s)")
plt.ylim(-0.2, 0.2)
plt.legend()


# _ = plt.hist(diff, alpha=0.5, bins=200)


# Plot scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(
    diff_luxoptics[mask_luxmarkers],
    diff_luxmarkers[mask_luxmarkers],
    label="Correlation",
    alpha=0.5,
)
