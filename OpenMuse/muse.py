"""Utilities for interacting with the Muse S Athena headset."""

import asyncio
import re
from typing import Callable, ClassVar, Iterable, Optional
from .backends import BleakBackend


# ===================================
# Find MAC addresses of Muse devices
# ===================================
def find_muse(timeout=10, verbose=True):
    """Scan for Muse devices via Bluetooth Low Energy (BLE).

    This is the canonical scanned used throughout the package and CLI.
    """
    backend = BleakBackend()
    if verbose:
        print(f"Searching for Muses (max. {timeout} seconds)...")
    devices = backend.scan(timeout=timeout)
    muses = []
    for d in devices:
        name = d.get("name")
        try:
            if isinstance(name, str) and "muse" in name.lower():
                muses.append(d)
        except Exception:
            continue

    if verbose:
        if muses:
            for m in muses:
                print(f'Found device {m["name"]}, MAC Address {m["address"]}')
        else:
            print("No Muses found. Ensure the device is on and Bluetooth is enabled.")

    return muses


# ===================================
# Muse S device
# ===================================
class MuseS:
    """Constants and helpers shared across Muse S interactions."""

    CONTROL_UUID: ClassVar[str] = "273e0001-4c4d-454d-96be-f03bac821358"
    EEG_UUID: ClassVar[str] = "273e0013-4c4d-454d-96be-f03bac821358"
    OTHER_UUID: ClassVar[str] = "273e0014-4c4d-454d-96be-f03bac821358"

    DATA_CHARACTERISTICS: ClassVar[tuple[str, str]] = (
        EEG_UUID,
        OTHER_UUID,
    )

    @staticmethod
    def encode_command(token: str) -> bytes:
        """Encode a Muse command token as length-prefixed ASCII."""
        if not isinstance(token, str) or not token:
            raise ValueError("command token must be a non-empty string")
        try:
            payload = (token + "\n").encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError("command token must be ASCII") from exc
        if len(payload) > 255:
            raise ValueError("command too long (max 254 chars plus newline)")
        return bytes([len(payload)]) + payload

    @staticmethod
    async def send_command(client, token: str, verbose: bool = False) -> None:
        """Send a command to the Muse device via the control characteristic."""
        data = MuseS.encode_command(token)
        await client.write_gatt_char(MuseS.CONTROL_UUID, data, response=False)

    @staticmethod
    async def initialize_device(client, preset: str, verbose: bool = False) -> None:
        """
        Initialize Muse device with standard handshake and preset configuration.

        Performs:
        1. Version query (v6)
        2. Status query (s)
        3. Halt/reset (h)
        4. Apply preset
        5. Start streaming (dc001)
        6. Enable low-latency mode (L1)
        """
        # Version/status handshake (best-effort)
        try:
            await MuseS.send_command(client, "v6", verbose)
            await asyncio.sleep(0.2)
            await MuseS.send_command(client, "s", verbose)
            await asyncio.sleep(0.2)
        except Exception as exc:
            if verbose:
                print(f"Warning: version/status failed: {exc}")

        # Halt/reset
        try:
            await MuseS.send_command(client, "h", verbose)
            await asyncio.sleep(0.2)
        except Exception as exc:
            if verbose:
                print(f"Warning: halt failed: {exc}")

        # Preset selection
        if verbose:
            print(f"Sending preset {preset!r} and start commands ...")
        try:
            await MuseS.send_command(client, preset, verbose)
        except Exception as exc:
            if verbose:
                print(f"Warning: preset {preset!r} failed: {exc}")
        await asyncio.sleep(0.2)

        # Query status again after preset change
        try:
            await MuseS.send_command(client, "s", verbose)
            await asyncio.sleep(0.2)
        except Exception:
            pass

        # Start streaming (SEND TWICE for reliability), then L1
        await MuseS.send_command(client, "dc001", verbose)
        await asyncio.sleep(0.05)
        await MuseS.send_command(client, "dc001", verbose)
        await asyncio.sleep(0.1)
        try:
            await MuseS.send_command(client, "L1", verbose)
            await asyncio.sleep(0.3)
        except Exception:
            pass

        # Final status query
        try:
            await MuseS.send_command(client, "s", verbose)
            await asyncio.sleep(0.2)
        except Exception:
            pass

    @staticmethod
    async def connect_and_initialize(
        client,
        preset: str,
        data_callbacks: dict[str, Callable],
        verbose: bool = False,
    ) -> None:
        """
        Complete connection routine: subscribe to notifications, initialize device, and log info.

        Parameters:
        - client: BleakClient instance (already connected)
        - preset: Preset string to apply (e.g., "p1035")
        - data_callbacks: Dict mapping characteristic UUIDs to notification callbacks
        - verbose: Print progress messages

        This function:
        1. Subscribes to control notifications (for device info)
        2. Subscribes to all requested data characteristics
        3. Initializes device with preset
        4. Waits for and logs device info (firmware, battery)
        """
        control_buffer = ""
        device_info_logged = False

        def _on_control(_, data: bytearray):
            nonlocal control_buffer, device_info_logged
            try:
                text = data.decode("utf-8", errors="ignore")
            except Exception:
                return

            control_buffer = (control_buffer + text)[-4096:]
            if device_info_logged:
                return

            # Try to extract device info
            firmware = re.search(r'"fw"\s*:\s*"([^"]+)"', control_buffer)
            battery = re.search(r'"bp"\s*:\s*([\d.]+)', control_buffer)

            if firmware and battery:
                print(
                    f"Connected to device (firmware {firmware.group(1)}): battery {battery.group(1)}%"
                )
                device_info_logged = True

        # Subscribe to control notifications first
        try:
            await client.start_notify(MuseS.CONTROL_UUID, _on_control)
            if verbose:
                print(f"Subscribed to control notifications on {MuseS.CONTROL_UUID}")
        except Exception as e:
            if verbose:
                print(
                    f"Warning: could not subscribe to control {MuseS.CONTROL_UUID}: {e}"
                )

        # Subscribe to data characteristics
        for uuid, callback in data_callbacks.items():
            try:
                await client.start_notify(uuid, callback)
                if verbose:
                    print(f"Subscribed to notifications on {uuid}")
            except Exception as e:
                if verbose:
                    print(f"Warning: could not subscribe to {uuid}: {e}")

        # Initialize device (sends commands and triggers control responses)
        await MuseS.initialize_device(client, preset, verbose)

        # Wait for device info to be logged (up to 2 seconds)
        if verbose:
            print("Waiting for device info...")
        for _ in range(20):
            if device_info_logged:
                break
            await asyncio.sleep(0.1)

        if not device_info_logged and verbose:
            print("Warning: Did not receive device info (firmware/battery)")

    @staticmethod
    async def stop_streaming(client, verbose: bool = False) -> None:
        """Send halt command to stop streaming."""
        try:
            await MuseS.send_command(client, "h", verbose)
        except Exception:
            pass
