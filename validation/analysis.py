import io
import urllib
import warnings

import neurokit2 as nk
import numpy as np
import pandas as pd
import pyxdf
import matplotlib.pyplot as plt

# Contains .xdf files recorded with LabRecorder, containing Muse data streams (recorded using the stream function, preset 1041) and a Bitalino stream with ECG and Photosensor data.
# Was recorded during a face presentation experiment.
filename = "./test-11.xdf"
upsample = 2.0
fillmissing = None

streams, header = pyxdf.load_xdf(
    filename,
    synchronize_clocks=True,
    handle_clock_resets=True,
    dejitter_timestamps=False,
)


# Get smaller time stamp to later use as offset (zero point)
min_ts = min([min(s["time_stamps"]) for s in streams if len(s["time_stamps"]) > 0])

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
fig = plt.figure(figsize=(15, 15))
for i, s in enumerate(streams):
    ax = fig.add_subplot(len(streams), 1, i + 1)
    if np.array(s["time_series"]).dtype == "<U1":
        continue
    ax.plot(s["time_stamps"], s["time_series"])
    ax.set_xlim(tmin, tmax)
    ax.set_title(s["info"]["name"][0])
plt.tight_layout()
plt.show()


# ========================================================================================
# Investigate issues with time synchronization
# ========================================================================================
plt.plot(streams[0]["time_stamps"], label="Timestamps")
np.diff(streams[2]["time_stamps"]).max()

# ========================================================================================
# Test the proximity of eletronic markers vs. Photosensor ones
# ========================================================================================
stream = streams[3]
jspsych_events = np.array([float(val[0]) for val in stream["time_series"]])
jspsych_onsets = jspsych_events == 1
jspsych_ts = np.array(stream["time_stamps"])
jspsych_ts_onset = jspsych_ts[jspsych_onsets]
jspsych_duration = jspsych_ts[jspsych_events == 0] - jspsych_ts_onset


# LUX is 4th column
stream = streams[4]
# stream["info"]["desc"][0]["channels"][0]["channel"]
lux_signal = stream["time_series"][:, 3]
# nk.signal_plot(lux_signal[0:50000])
lux_events = nk.events_find(
    lux_signal, threshold_keep="below", duration_min=100, duration_max=5000
)
lux_ts = stream["time_stamps"]
lux_ts_onset = lux_ts[lux_events["onset"].astype(int)]

print(
    f"len jspsych onsets: {len(jspsych_ts_onset)}, len lux onsets: {len(lux_ts_onset)}"
)

# lot a vertical lines from 0 to 0.5 and from 0.5 to 1 for each jspsych and lux onset
plt.figure()
plt.plot(
    lux_ts[0:50000] - min_ts, lux_signal[0:50000] / max(lux_signal), label="Photosensor"
)
for i in range(10):
    plt.vlines(
        (jspsych_ts_onset[i] - min_ts), 0.5, 1, color="red", label="JSpsych onsets"
    )
    plt.vlines(
        (lux_ts_onset[i] - min_ts), 0, 0.5, color="orange", label="Photosensor onsets"
    )

delays = jspsych_ts_onset - lux_ts_onset
_ = plt.hist(delays, bins=50)

# # ========================================================================================
# # Test GYRO sync
# # ========================================================================================
# # LUX is 4th column
# # streams[4]["info"]["desc"][0]["channels"][0]["channel"]
# lux_signal = streams[4]["time_series"][:, 3]
# # nk.signal_plot(lux_signal)
# lux_events = nk.events_find(lux_signal, threshold_keep="below", duration_min=1500, duration_max=2500)
# lux_ts = streams[4]["time_stamps"]
# lux_ts_onset = lux_ts[lux_events["onset"].astype(int)]

# # GYRO
# accgyro = pd.DataFrame(
#     streams[1]["time_series"],
#     columns=[ch["label"][0] for ch in streams[1]["info"]["desc"][0]["channels"][0]["channel"]],
# )
# accgyro.index = streams[1]["time_stamps"]

# accgyro.iloc[:3000].plot(y=["GYRO_X"], subplots=True)
# for ts in lux_ts_onset[0:11]:
#     plt.axvline(x=ts, color="red", linestyle="--")
