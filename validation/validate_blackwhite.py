import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyxdf

# --- Configuration ---
filename = "./test-15-dev.xdf"
filename = "./test-15-clock.xdf"
dejitter_timestamps = ["OpenSignals"]

# --- Load Data ---
streams, header = pyxdf.load_xdf(
    filename,
    synchronize_clocks=True,
    handle_clock_resets=True,
    dejitter_timestamps=False,
)

# De-jitter timestamps for specified streams
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
fig = plt.figure(figsize=(15, 7))
for i, s in enumerate(streams):
    name = s["info"].get("name", ["Unnamed"])[0]
    channels = [d["label"][0] for d in s["info"]["desc"][0]["channels"][0]["channel"]]
    if name in ["OpenSignals"]:
        lux = s["time_series"][:, channels.index("LUX2")]
        lux = (lux - np.min(lux)) / (np.max(lux) - np.min(lux))
        lux_ts = s["time_stamps"]
        plt.plot(lux_ts, lux, color="blue")
    if name in ["Muse_OPTICS"]:
        optics = s["time_series"][:, channels.index("OPTICS_RI_AMB")]
        optics = (optics - np.min(optics)) / (np.max(optics) - np.min(optics))
        optics_ts = s["time_stamps"]
        plt.plot(optics_ts, optics, color="red")
    # ax = fig.add_subplot(len(streams), 1, i + 1)
    # ax.plot(s["time_stamps"], s["time_series"])
    # ax.set_xlim(tmin, tmax)
    # ax.set_title(s["info"]["name"][0])
plt.tight_layout()
plt.show()


events_lux = nk.events_find(lux, threshold_keep="below")
events_optics = nk.events_find(optics, threshold_keep="below")
onsets_lux = lux_ts[events_lux["onset"]]
onsets_optics = optics_ts[events_optics["onset"]]
onsets_optics = nk.find_closest(onsets_lux, onsets_optics)
diff = onsets_lux - onsets_optics

_ = plt.hist(diff, alpha=0.5)
plt.plot(onsets_lux, diff)
