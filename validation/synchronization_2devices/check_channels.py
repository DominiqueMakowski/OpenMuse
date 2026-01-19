"""
Quick check: which channel is being used for each device in each file?
"""

import numpy as np
import pyxdf

NEW_FIRMWARE_MAC = "00:55:DA:BB:CD:CD"
OLD_FIRMWARE_MAC = "00:55:DA:B9:FA:20"


def find_channel(streams, stream_name, channel_name):
    for s in streams:
        name = s["info"].get("name", ["Unnamed"])[0]
        if name == stream_name:
            try:
                channels = [d["label"][0] for d in s["info"]["desc"][0]["channels"][0]["channel"]]
            except:
                return None, None
            matching = [ch for ch in channels if channel_name in ch]
            if not matching:
                return None, None
            ch_name = matching[0]
            signal_data = np.array(s["time_series"])[:, channels.index(ch_name)].astype(float)
            ts = s["time_stamps"]
            return signal_data, ts
    return None, None


files = ["test1_robust.xdf", "test1_windowed.xdf", "test1_constrained.xdf", "test1_adaptive.xdf"]

for filename in files:
    try:
        streams, _ = pyxdf.load_xdf(filename, synchronize_clocks=False)

        print(f"\n{filename}:")

        # Check what channels exist
        for device, mac in [("NEW", NEW_FIRMWARE_MAC), ("OLD", OLD_FIRMWARE_MAC)]:
            stream_name = f"Muse-OPTICS ({mac})"
            for s in streams:
                name = s["info"].get("name", ["Unnamed"])[0]
                if name == stream_name:
                    try:
                        channels = [d["label"][0] for d in s["info"]["desc"][0]["channels"][0]["channel"]]
                        print(f"  {device}: {channels}")
                    except:
                        pass
                    break
    except FileNotFoundError:
        print(f"\n{filename}: FILE NOT FOUND")
