"""
BITalino LSL Streaming
======================

Note: This code contains functionality to connect and stream data from a BITalino
(PLUX Biosignals) device. It is included in the OpenMuse package for convenience
as it shares logic and functionalities, but it is not directly related to Muse
devices.

This module connects to a BITalino device via Bluetooth/Serial and streams
data over LSL (Lab Streaming Layer) with high-precision timestamping.

It utilizes the StableClock RLS filter (identical to the Muse implementation)
to map device sample counts to LSL time, correcting for clock drift.
"""

import asyncio
import struct
from typing import List, Optional
from bleak import BleakClient, BleakScanner

# from mne_lsl.lsl import StreamInfo, StreamOutlet, local_clock
# from .stream import StableClock
from .backends import BleakBackend


# ===================================
# Find MAC addresses of BITalino devices
# ===================================
def find_bitalino(timeout=10, verbose=True):
    """Scan for BITalino devices via Bluetooth Low Energy (BLE).

    This uses the same BleakBackend as the Muse scanner, ensuring
    cross-platform compatibility (Windows/MacOS/Linux).
    """
    # Use the same backend class imported in your file
    backend = BleakBackend()

    if verbose:
        print(f"Searching for BITalinos (max. {timeout} seconds)...")

    # Use the identical scan method
    devices = backend.scan(timeout=timeout)
    bitalinos = []

    for d in devices:
        name = d.get("name")
        print("*Debug (remove me once we found pattern):Device found:", name)
        # Filter specifically for BITalino devices
        try:
            if isinstance(name, str) and "bitalino" in name.lower():
                bitalinos.append(d)
        except Exception:
            continue

    if verbose:
        if bitalinos:
            for b in bitalinos:
                # Matches the print format of find_muse
                print(f'Found device {b["name"]}, MAC Address {b["address"]}')
        else:
            print(
                "No BITalinos found. Ensure the device is on and Bluetooth is enabled."
            )

    return bitalinos


# ============================================================================
# BITALINO DRIVER
# ============================================================================


class BITalino:
    def __init__(self, address: str):
        self.address = address
        self.client: Optional[BleakClient] = None
        self._queue = asyncio.Queue()
        self._running = False
        self._analog_channels = []
        self._frame_size = 0

        # BITalino (BT121/BLE) Service & Characteristic UUIDs (based on https://github.com/BITalinoWorld/firmware-BT121/blob/master/GATT.xml)
        self._UART_SERVICE = "c566488a-0882-4e1b-a6d0-0b717e652234"
        self._CMD_CHAR = "4051eb11-bf0a-4c74-8730-a48f4193fcea"  # Write
        self._DATA_CHAR = "40fdba6b-672e-47c4-808a-e529adff3633"  # Notify

    async def connect(self, timeout: float = 10.0):
        """Connects to the device and subscribes to the data stream."""
        device = await BleakScanner.find_device_by_address(
            self.address, timeout=timeout
        )
        if not device:
            raise Exception(f"Device {self.address} not found.")

        self.client = BleakClient(device)
        await self.client.connect()
        await self.client.start_notify(self._DATA_CHAR, self._on_data_received)

    async def disconnect(self):
        """Stops streaming and disconnects."""
        if self._running:
            await self.stop()
        if self.client:
            await self.client.disconnect()

    async def start(
        self, sampling_rate: int = 1000, channels: List[int] = [0, 1, 2, 3, 4, 5]
    ):
        """
        Configures sampling rate and starts acquisition.
        Supported rates: 1, 10, 100, 1000 Hz.
        """
        if self._running:
            return

        # 1. Validate inputs
        rates = {1: 0, 10: 1, 100: 2, 1000: 3}
        if sampling_rate not in rates:
            raise ValueError(f"Invalid rate. Choose: {list(rates.keys())}")

        self._analog_channels = sorted(list(set(channels)))
        n_ch = len(self._analog_channels)

        # 2. Calculate frame size (Protocol definition)
        if n_ch <= 4:
            self._frame_size = int((12.0 + 10.0 * n_ch) / 8.0 + 0.99)  # ceil
        else:
            self._frame_size = int((52.0 + 6.0 * (n_ch - 4)) / 8.0 + 0.99)

        # 3. Send Sampling Rate Command: <Fs> 0 0 0 0 1 1
        cmd_rate = (rates[sampling_rate] << 6) | 0x03
        await self.client.write_gatt_char(self._CMD_CHAR, bytes([cmd_rate]))

        # 4. Send Start Command: A6 A5 A4 A3 A2 A1 0 1
        cmd_start = 1
        for ch in self._analog_channels:
            cmd_start |= 1 << (2 + ch)

        # Clear old data from queue
        while not self._queue.empty():
            self._queue.get_nowait()

        await self.client.write_gatt_char(self._CMD_CHAR, bytes([cmd_start]))
        self._running = True

    async def stop(self):
        """Stops acquisition."""
        if not self._running:
            return
        await self.client.write_gatt_char(self._CMD_CHAR, bytes([0]))
        self._running = False

    async def read(self, n_samples: int = 100) -> List[List[int]]:
        """
        Reads 'n' parsed frames from the buffer.
        Returns list of lists: [[Seq, D0, D1, D2, D3, A1, A2...], ...]
        """
        data = []
        for _ in range(n_samples):
            # This awaits until data is available in the queue
            data.append(await self._queue.get())
        return data

    def _on_data_received(self, sender, data: bytearray):
        """Internal callback: Decodes raw bytes into samples."""
        if not self._running or len(data) != self._frame_size:
            return

        # CRC Check (4-bit)
        crc = data[-1] & 0x0F
        check_byte = data[-1] & 0xF0
        x = 0
        # Iterate over all bytes (simulating the hardware CRC calculation)
        temp_data = list(data[:-1]) + [check_byte]
        for byte in temp_data:
            for bit in range(7, -1, -1):
                x <<= 1
                if x & 0x10:
                    x ^= 0x03
                x ^= (byte >> bit) & 0x01

        if crc != (x & 0x0F):
            return  # Drop corrupted frame

        # Decode Protocol
        seq = data[-1] >> 4
        digital = [
            (data[-2] >> 7) & 0x01,
            (data[-2] >> 6) & 0x01,
            (data[-2] >> 5) & 0x01,
            (data[-2] >> 4) & 0x01,
        ]

        sample = [seq] + digital

        # Decode Analog Channels (Dynamic packing based on channel count)
        n_ch = len(self._analog_channels)
        if n_ch > 0:
            sample.append(((data[-2] & 0x0F) << 6) | (data[-3] >> 2))
        if n_ch > 1:
            sample.append(((data[-3] & 0x03) << 8) | data[-4])
        if n_ch > 2:
            sample.append((data[-5] << 2) | (data[-6] >> 6))
        if n_ch > 3:
            sample.append(((data[-6] & 0x3F) << 4) | (data[-7] >> 4))
        if n_ch > 4:
            sample.append(((data[-7] & 0x0F) << 2) | (data[-8] >> 6))
        if n_ch > 5:
            sample.append(data[-8] & 0x3F)

        self._queue.put_nowait(sample)


# ============================================================================
# Stream
# ============================================================================


# def stream_bitalino(
#     address: str,
#     sampling_rate: int = 1000,
#     analog_channels: list = [0, 1, 2, 3, 4, 5],
#     chunk_size: int = 100,
# ):
#     """
#     Stream data from BITalino to LSL.

#     This function blocks until interrupted (Ctrl+C).
#     """

#     print(f"Connecting to BITalino at {address}...")
#     try:
#         device = BITalino(address)
#         v = device.version()
#         print(f"Connected. Device Version: {v}")
#     except Exception as e:
#         print(f"Failed to connect: {e}")
#         raise ValueError(f"Failed to connect: {e}")

#     # 1. Setup LSL Stream Info
#     n_analog = len(analog_channels)
#     # Total channels = Analog + 4 Digital (optional, included here for completeness)
#     # Let's stream only Analog for cleaner output, or mix.
#     # Standard: Stream everything.
#     channel_names = ["SEQ", "D1", "D2", "D3", "D4"] + [
#         f"A{x+1}" for x in analog_channels
#     ]
#     n_channels = len(channel_names)

#     info = StreamInfo(
#         name="BITalino",
#         stype="BioSignals",
#         n_channels=n_channels,
#         sfreq=float(sampling_rate),
#         dtype="float32",
#         source_id=f"bitalino_{device.address}",
#     )

#     # Add metadata
#     desc = info.desc
#     channels = desc.append_child("channels")
#     for name in channel_names:
#         channels.append_child("channel").append_child_value("label", name)

#     outlet = StreamOutlet(info)
#     print(f"LSL Stream '{info.name}' created. Sample Rate: {sampling_rate}Hz")

#     # 2. Initialize Logic
#     clock = StableClock()
#     total_samples_read = 0

#     try:
#         device.start(sampling_rate, analog_channels)
#         print("Acquisition started. Press Ctrl+C to stop.")

#         while True:
#             # Blocking read from device
#             # Note: This determines the loop cadence.
#             # 100 samples @ 1000Hz = 100ms latency chunks.
#             data = device.read(chunk_size)

#             # --- Timestamping Logic ---
#             # 1. Get current LSL time (arrival time of the chunk)
#             now = local_clock()

#             # 2. Determine "Device Time" of the *last* sample in this chunk
#             #    Device time is purely based on sample count / rate
#             current_chunk_len = data.shape[0]
#             total_samples_read += current_chunk_len

#             device_time_end = total_samples_read / sampling_rate

#             # 3. Update the Clock Model (Drift Correction)
#             clock.update(device_time_end, now)

#             # 4. Generate timestamps for the whole chunk retrospectively
#             #    t[i] = device_time_start + i * dt
#             chunk_device_times = (
#                 (np.arange(current_chunk_len) - current_chunk_len + 1) / sampling_rate
#             ) + device_time_end

#             # Map to LSL time using the corrected clock
#             lsl_timestamps = clock.map_time(chunk_device_times)

#             # --- Push to LSL ---
#             outlet.push_chunk(data.astype(np.float32), timestamp=lsl_timestamps)

#     except KeyboardInterrupt:
#         print("\nStreaming stopped by user.")
#     except Exception as e:
#         print(f"Error during streaming: {e}")
#     finally:
#         device.stop()
#         device.close()
#         print("Device disconnected.")
