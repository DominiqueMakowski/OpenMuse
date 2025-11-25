"""
BITalino LSL Streaming
======================

Note: This code contains functionality to connect and stream data from a BITalino
(PLUX Biosignals) device. It is included in the OpenMuse package for convenience
as it shares logic and functionalities, but it is not directly related to Muse
devices.

This module connects to a BITalino device via Bluetooth/Serial and streams
data over LSL (Lab Streaming Layer) with high-precision timestamping.
The code is inspired by https://github.com/BITalinoWorld/revolution-python-api

It utilizes the StableClock RLS filter (identical to the Muse implementation)
to map device sample counts to LSL time, correcting for clock drift.
"""

import asyncio
import traceback
from typing import Callable, List, Optional

import numpy as np
from bleak import BleakClient, BleakScanner
from mne_lsl.lsl import StreamInfo, StreamOutlet, local_clock
from mne_lsl.stream import StreamLSL

from .backends import BleakBackend
from .stream import StableClock
from .utils import configure_lsl_api_cfg
from .view import FastViewer


# ============================================================================
# Find BITalino devices
# ============================================================================
def find_bitalino(timeout=10, verbose=True):
    """Scan for BITalino devices via Bluetooth Low Energy (BLE)."""
    backend = BleakBackend()

    if verbose:
        print(f"Searching for BITalinos (max. {timeout} seconds)...")

    devices = backend.scan(timeout=timeout)
    bitalinos = []

    for d in devices:
        name = d.get("name")
        print(f"DEBUG: Found device: {name}, address: {d.get('address')}")
        try:
            if isinstance(name, str) and "bitalino" in name.lower():
                bitalinos.append(d)
        except Exception:
            continue

    if verbose:
        if bitalinos:
            for b in bitalinos:
                print(f'Found device {b["name"]}, MAC Address {b["address"]}')
        else:
            print("No BITalinos found. Ensure the device is on and Bluetooth is enabled.")

    return bitalinos


# ============================================================================
# BITalino Transfer Functions
# - https://github.com/pluxbiosignals/biosignalsnotebooks
# - https://github.com/pluxbiosignals/opensignals-samples
# ============================================================================
BITALINO_SENSORS = {
    "ECG": {
        "unit": "mV",
        # V = ((ADC/1024 - 0.5) * 3.3) / Gain * 1000 (to mV)
        "func": lambda x: (((x / 1024.0) - 0.5) * 3.3 / 1100) * 1000.0,
        "range": 1.5,
    },
    "EMG": {
        "unit": "mV",
        "func": lambda x: (((x / 1024.0) - 0.5) * 3.3 / 1009) * 1000.0,
        "range": 1.65,
    },
    "EEG": {
        "unit": "uV",
        "func": lambda x: (((x / 1024.0) - 0.5) * 3.3 / 40000) * 1e6,  # Gain 40,000
        "range": 40.0,
    },
    "EDA": {
        "unit": "uS",
        # EDA (uS) = (ADC / 2^n) * VCC / 0.132
        "func": lambda x: (x / 1024.0) * 3.3 / 0.132,
        "range": 25.0,
    },
    "ACC": {
        "unit": "g",
        # Accelerometer usually centered at 512 for 0g. Range -3g to 3g approx.
        # Simplified normalization: (ADC - min) / (max - min) * 2 - 1
        "func": lambda x: ((x - 0) / (1023 - 0) * 2 - 1) * 3.0,
        "range": 3.0,
    },
    "LUX": {"unit": "%", "func": lambda x: (x / 1024.0) * 100.0, "range": 100.0},
    # Default for unknown or None
    "RAW": {"unit": "raw", "func": lambda x: x, "range": 1024.0},
}


# ============================================================================
# BITALINO DRIVER
# ============================================================================
def generate_crc4_table() -> List[int]:
    """
    Generates a 4096-entry lookup table for the BITalino 4-bit CRC.
    Index = (Current_CRC << 8) | New_Byte
    Value = New_CRC
    """
    table = [0] * 4096
    for current_crc in range(16):
        for byte_val in range(256):
            x = current_crc
            for bit in range(7, -1, -1):
                x <<= 1
                if x & 0x10:
                    x ^= 0x03
                x ^= (byte_val >> bit) & 0x01
            index = (current_crc << 8) | byte_val
            table[index] = x & 0x0F
    return table


class BITalino:
    """
    Async driver for BITalino (BLE).
    """

    # Generate table at class level (shared by all instances)
    _CRC_TABLE = generate_crc4_table()

    def __init__(self, address: str):
        self.address = address
        self.client: Optional[BleakClient] = None
        self._running = False
        self._analog_channels = []
        self._frame_size = 0

        # Callback for decoded samples (seq, digital..., analog...)
        self._data_callback: Optional[Callable[[List[int]], None]] = None

        # BITalino (BT121/BLE) Service & Characteristic UUIDs
        self._CMD_CHAR = "4051eb11-bf0a-4c74-8730-a48f4193fcea"  # Write
        self._DATA_CHAR = "40fdba6b-672e-47c4-808a-e529adff3633"  # Notify

    def set_callback(self, callback: Callable[[List[int]], None]):
        """Set a callback function to receive decoded samples immediately."""
        self._data_callback = callback

    async def connect(self, timeout: float = 10.0):
        """Connects to the device and subscribes to the data stream."""
        device = await BleakScanner.find_device_by_address(self.address, timeout=timeout)
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

    async def start(self, sampling_rate: int = 1000, channels: List[int] = [0, 1, 2, 3, 4, 5]):
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
            self._frame_size = int((12.0 + 10.0 * n_ch) / 8.0 + 0.99)
        else:
            self._frame_size = int((52.0 + 6.0 * (n_ch - 4)) / 8.0 + 0.99)

        # 3. Send Sampling Rate Command: <Fs> 0 0 0 0 1 1
        cmd_rate = (rates[sampling_rate] << 6) | 0x03
        await self.client.write_gatt_char(self._CMD_CHAR, bytes([cmd_rate]))

        # 4. Send Start Command: A6 A5 A4 A3 A2 A1 0 1
        cmd_start = 1
        for ch in self._analog_channels:
            cmd_start |= 1 << (2 + ch)

        await self.client.write_gatt_char(self._CMD_CHAR, bytes([cmd_start]))
        self._running = True

    async def stop(self):
        """Stops acquisition."""
        if not self._running:
            return
        await self.client.write_gatt_char(self._CMD_CHAR, bytes([0]))
        self._running = False

    def _on_data_received(self, sender, data: bytearray):
        """Internal callback: Decodes raw bytes into samples."""
        if not self._running or len(data) != self._frame_size:
            return

        # --- Fast CRC Check (Lookup Table) ---
        crc = data[-1] & 0x0F
        check_byte = data[-1] & 0xF0

        # Start with CRC 0
        x = 0

        # 1. Process the main data bytes
        for byte in data[:-1]:
            # Lookup: (Current CRC << 8) | New Byte
            index = (x << 8) | byte
            x = self._CRC_TABLE[index]

        # 2. Process the final check byte (masked)
        index = (x << 8) | check_byte
        x = self._CRC_TABLE[index]

        if crc != (x & 0x0F):
            return  # Drop corrupted frame

        # --- Decode Protocol ---
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

        # Send to callback if registered
        if self._data_callback:
            self._data_callback(sample)


# ============================================================================
# Streaming Logic
# ============================================================================


async def stream_bitalino(
    address: str,
    channels: List[Optional[str]] = None,
    sampling_rate: int = 1000,
    buffer_size: int = 32,
):
    """
    Stream data from BITalino to LSL asynchronously.

    Parameters:
    - buffer_size: Number of samples to accumulate before pushing to LSL.
    - channels: List of length 6. e.g. ['ECG', None, 'EDA', None, None, None]
    """

    # 1. Validate Sensor Types
    if channels:
        if len(channels) != 6:
            raise ValueError("Length of 'channels' must be 6. Include 'None' for unused.")
    else:
        # Fill with "RAW" if not provided to activate all
        channels = ["RAW"] * 6

    # --- Identify Active Channels ---
    # Convert ['ECG', None, ...] to indices [0, ...]
    active_channels = [i for i, x in enumerate(channels) if x is not None]

    # 2. Setup LSL Stream Info with Correct Names and Units
    # ---------------------------------------------------
    # Start with Digital Channels
    channel_names = ["SEQ", "D1", "D2", "D3", "D4"]
    channel_units = ["raw"] * 5

    # Append Analog Channels (We always create 6 slots in LSL A1-A6)
    for ch_idx, s_type in enumerate(channels):
        if s_type and s_type.upper() in BITALINO_SENSORS:
            # e.g. "A1_ECG"
            name = f"A{ch_idx+1}_{s_type.upper()}"
            unit = BITALINO_SENSORS[s_type.upper()]["unit"]
        else:
            # e.g. "A1"
            name = f"A{ch_idx+1}"
            unit = "raw"

        channel_names.append(name)
        channel_units.append(unit)

    # 1 SEQ + 4 DIG + 6 ANALOG = 11 Channels
    n_channels_lsl = len(channel_names)

    info = StreamInfo(
        name="BITalino",
        stype="BioSignals",
        n_channels=n_channels_lsl,
        sfreq=float(sampling_rate),
        dtype="float32",
        source_id=f"bitalino_{address}",
    )

    desc = info.desc
    desc.append_child_value("manufacturer", "PLUX")
    channels_desc = desc.append_child("channels")
    for name, unit in zip(channel_names, channel_units):
        ch = channels_desc.append_child("channel")
        ch.append_child_value("label", name)
        ch.append_child_value("unit", unit)
        ch.append_child_value("type", "EEG" if "EEG" in name else "Biosignals")

    outlet = StreamOutlet(info)
    print(f"LSL Stream '{info.name}' created. Channels: {channel_names}. " f"Active Device indices: {active_channels}")

    # 2. State Management
    device = BITalino(address)
    clock = StableClock()

    # Buffering state
    sample_buffer = []
    total_samples = 0

    def _process_sample(sample: List[int]):
        """
        Callback triggered by the driver for every single sample.
        Accumulates samples and pushes chunks to LSL.
        """
        nonlocal total_samples

        sample_buffer.append(sample)
        total_samples += 1

        # Flush buffer when full
        if len(sample_buffer) >= buffer_size:
            lsl_now = local_clock()

            # The raw data only contains [SEQ, D1-D4, + Active Analog Channels]
            raw_chunk = np.array(sample_buffer, dtype=np.float32)
            n_chunk = len(raw_chunk)

            # We need to map this to the full LSL width (11 columns)
            full_chunk = np.zeros((n_chunk, n_channels_lsl), dtype=np.float32)

            # 1. Copy SEQ and Digital (First 5 columns are always present)
            full_chunk[:, 0:5] = raw_chunk[:, 0:5]

            # 2. Map Active Analog Channels
            # raw_chunk starts analog data at index 5
            # LSL starts analog data at index 5
            current_raw_col = 5

            for ch_idx in active_channels:
                s_type = channels[ch_idx]

                # Get the column from the compressed driver packet
                data_col = raw_chunk[:, current_raw_col]

                # Apply Transfer Function
                if s_type and s_type.upper() in BITALINO_SENSORS:
                    tf = BITALINO_SENSORS[s_type.upper()]["func"]
                    data_col = tf(data_col)

                # Place into the correct LSL column (5 + actual index A1..A6)
                lsl_col_idx = 5 + ch_idx
                full_chunk[:, lsl_col_idx] = data_col

                current_raw_col += 1

            # --- Timestamping (StableClock) ---
            # 1. Device time is purely sample count / Rate
            device_time_end = total_samples / sampling_rate

            # 2. Update Clock Model
            clock.update(device_time_end, lsl_now)

            # 3. Retroactively calculate timestamps for the whole chunk
            chunk_device_times = device_time_end - (np.arange(n_chunk)[::-1] / sampling_rate)

            # 4. Map to LSL time
            lsl_timestamps = clock.map_time(chunk_device_times)

            # Push
            outlet.push_chunk(full_chunk, timestamp=lsl_timestamps)
            sample_buffer.clear()

    # 3. Connect and Start
    try:
        print(f"Connecting to BITalino at {address}...")
        device.set_callback(_process_sample)
        await device.connect()

        print("Starting acquisition...")
        # Start only the active channels
        await device.start(sampling_rate, active_channels)

        print("Streaming... Press Ctrl+C to stop.")
        # Keep the loop alive
        while True:
            await asyncio.sleep(1)

    except asyncio.CancelledError:
        print("Streaming cancelled.")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        print("Stopping device...")
        await device.stop()
        await device.disconnect()
        print("Disconnected.")


# ============================================================================
# BITALINO VIEWER
# ============================================================================
class BitalinoViewer(FastViewer):
    """
    Subclass of FastViewer adapted for BITalino.
    Overrides channel setup to filter for Analog (A1-A6) channels
    and sets appropriate 10-bit ranges.
    """

    def _setup_channels(self):
        self.ch_configs = []

        # Distinct high-contrast colors for up to 6 analog channels
        colors = [
            (1.0, 0.3, 0.3),  # Red
            (0.2, 1.0, 0.2),  # Green
            (0.3, 0.5, 1.0),  # Blue
            (1.0, 1.0, 0.0),  # Yellow
            (0.0, 1.0, 1.0),  # Cyan
            (1.0, 0.0, 1.0),  # Magenta
        ]

        for s_idx, stream in enumerate(self.streams):
            # Inspect all channels in the stream
            for ch_i, name in enumerate(stream.info["ch_names"]):

                # Filter: BITalino sends [SEQ, D1-D4, A1-An].
                # We only want to visualize "A" (Analog) channels.
                if not name.startswith("A"):
                    continue

                # Determine color index from name "A1" or "A1_ECG"
                try:
                    # Remove 'A', split by '_', take first part "1", subtract 1 -> 0
                    c_idx = int(name[1:].split("_")[0]) - 1
                except (ValueError, IndexError):
                    c_idx = ch_i

                col = colors[c_idx % len(colors)]

                self.ch_configs.append(
                    {
                        "stream_idx": s_idx,
                        "ch_idx": ch_i,
                        "name": name,
                        "color": col,
                        # BITalino is 10-bit (0-1023).
                        # Range 1024 covers the full raw signal swing.
                        "base_range": 1024.0,
                        "scale": 1.0,
                        # Center the plot at the mid-rail (512)
                        "mean": 512.0,
                        "type": "BIO",
                    }
                )


def view_bitalino(stream_name="BITalino", window_duration=10.0):
    """
    Connects to a BITalino LSL stream and opens the viewer.
    """
    configure_lsl_api_cfg()

    print(f"Looking for LSL stream: '{stream_name}'...")
    try:
        # bufsize defines the internal buffer of the StreamLSL object
        s = StreamLSL(bufsize=window_duration, name=stream_name)
        s.connect(timeout=5.0)
    except Exception as e:
        print(f"Error: Could not connect to stream '{stream_name}'.")
        print("Ensure bitalino.py is running and streaming.")
        return

    print(f"Connected to {s.info['n_channels']} channels.")

    # Instantiate the specialized viewer
    v = BitalinoViewer([s], window_duration=window_duration)
    v.show()
