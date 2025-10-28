import pyxdf
import numpy as np

filename = "./test-12.xdf"

# Step 1: Load ALL streams with synchronization OFF
streams, header = pyxdf.load_xdf(
    filename,
    synchronize_clocks=False,
    handle_clock_resets=True,
    dejitter_timestamps=False,
)

print("--- Applying Selective Manual Synchronization ---")
corrected_streams = []
for stream in streams:
    name = stream["info"]["name"][0]

    # Check if LSL saved clock synchronization data
    if (
        len(stream.get("clock_times", [])) > 0
        and len(stream.get("clock_values", [])) > 0
    ):

        y_lsl_times = stream["clock_times"]
        x_device_times = stream["clock_values"]

        # --- THIS IS THE NEW LOGIC ---
        # Check if correction is needed.
        # If mean device time and mean LSL time are far apart,
        # we need to apply the mapping.
        # (Using 1000s as a safe threshold)
        if np.abs(np.mean(x_device_times) - np.mean(y_lsl_times)) > 1000:
            # This is a Muse stream (e.g., 182 vs 873421).
            # We must correct it.

            # Find the linear mapping (m, b) for: y = mx + b
            m, b = np.polyfit(x_device_times, y_lsl_times, 1)

            # Apply correction to all samples in this stream
            raw_timestamps = stream["time_stamps"]
            corrected_timestamps = (m * raw_timestamps) + b

            # Replace the broken timestamps
            stream["time_stamps"] = corrected_timestamps
            print(f"Corrected stream: {name}")

        else:
            # This stream is already on the LSL clock.
            # (e.g., OpenSignals, jsPsychMarkers)
            # DO NOTHING. Its time_stamps are already correct.
            print(f"Skipped (already correct): {name}")

    else:
        # Stream had no clock data, must be correct already
        print(f"Skipped (no clock data): {name}")

    corrected_streams.append(stream)


# Step 2: Verify the results
print("\n--- Final Synchronized Timestamps ---")
for stream in corrected_streams:
    name = stream["info"]["name"][0]
    if len(stream["time_stamps"]) > 0:
        ts_min = stream["time_stamps"].min()
        ts_max = stream["time_stamps"].max()
        print(f"Stream {name}: (from {ts_min:.2f} to {ts_max:.2f})")
