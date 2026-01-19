import asyncio
import os
import time
from datetime import datetime
from typing import Iterable, List, Optional, Union

import bleak

from .backends import _run
from .muse import MuseS
from .utils import get_utc_timestamp


async def _record_async(
    address: str,
    duration: float,
    outfile: str,
    preset: str = "p1041",
    subscribe_chars: Optional[Iterable[str]] = None,
    verbose: bool = True,
) -> None:
    if subscribe_chars is None:
        subscribe_chars = list(MuseS.DATA_CHARACTERISTICS)

    notified = 0
    stream_started = asyncio.Event()

    # Ensure output directory exists
    try:
        outdir = os.path.dirname(os.path.abspath(outfile))
        if outdir and not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
    except Exception:
        pass

    # Open output file in text mode and append
    f = open(outfile, "a", encoding="utf-8")

    # Raw recording only
    def _callback(uuid: str):
        def inner(_, data: bytearray):
            nonlocal notified
            notified += 1
            # Log timestamp, char UUID, and hex payload
            ts = get_utc_timestamp()
            line = f"{ts}\t{uuid}\t{data.hex()}\n"
            f.write(line)
            # Raw recording only; no decoding or viewing
            if not stream_started.is_set():
                stream_started.set()

        return inner

    if verbose:
        print(f"Connecting to {address} ...")

    try:
        async with bleak.BleakClient(address, timeout=15.0) as client:
            if verbose:
                print("Connected. Subscribing and configuring ...")

            # Build callbacks dict for all data characteristics
            data_callbacks = {uuid: _callback(uuid) for uuid in subscribe_chars}

            # Use shared connection routine
            await MuseS.connect_and_initialize(client, preset, data_callbacks, verbose)

            # Streaming is now active (callbacks are registered and device is configured)
            # Data will start flowing asynchronously

            if verbose:
                print(f"Recording for {duration} seconds to {outfile} ...")

            start = time.time()
            try:
                while time.time() - start < duration:
                    await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                pass
            finally:
                # Try to stop streaming gracefully
                await MuseS.stop_streaming(client, verbose)

            # Unsubscribe
            for cuuid in subscribe_chars:
                try:
                    await client.stop_notify(cuuid)
                except Exception:
                    pass
            # Also try to stop control notify
            try:
                await client.stop_notify(MuseS.CONTROL_UUID)
            except Exception:
                pass

    finally:
        try:
            f.flush()
            f.close()
        except Exception:
            pass
    if verbose:
        print(f"Done. Wrote {notified} notifications to {outfile}.")


def record(
    address: Union[str, List[str]],
    duration: float = 30.0,
    outfile: str = "muse_record.txt",
    preset: str = "p1041",
    verbose: bool = True,
) -> None:
    """
    Connect to one or more Muse devices, stream notifications, and append raw packets to text file(s).
    Supports recording from multiple devices simultaneously (e.g., for hyperscanning studies).

    Each line written: ISO8601 UTC timestamp, characteristic UUID, hex payload.

    Parameters
    - address: Device MAC address (Windows) or identifier (platform-dependent).
               Can be a single address (str) or a list of addresses (List[str]) for multi-device recording.
    - duration: Recording duration in seconds.
    - outfile: Path to output text file. For multiple devices, device address will be appended to filename.
    - preset: Preset string to send (e.g., "p1041" or "p21").
    - verbose: Print progress messages.
    """
    # Convert single address to list for uniform handling
    addresses = [address] if isinstance(address, str) else address

    # Remove duplicates if any
    addresses = list(set(addresses))

    if not addresses:
        raise ValueError("address must be a non-empty string or list of strings")
    if duration <= 0:
        raise ValueError("duration must be positive")
    if not isinstance(outfile, str) or not outfile:
        raise ValueError("outfile must be a non-empty path string")

    chars = list(MuseS.DATA_CHARACTERISTICS)

    async def run_multirecord():
        if verbose and len(addresses) > 1:
            print(f"Starting recording for {len(addresses)} device(s)...")

        tasks = []
        for addr in addresses:
            # Generate unique filename for each device
            if len(addresses) > 1:
                # Sanitize address for filename
                sanitized_addr = addr.replace(":", "").replace("-", "")

                # Insert address into provided filename
                if "." in outfile:
                    parts = outfile.rsplit(".", 1)
                    filename = f"{parts[0]}_{sanitized_addr}.{parts[1]}"
                else:
                    filename = f"{outfile}_{sanitized_addr}"
            else:
                filename = outfile

            # Create task for this device
            tasks.append(
                _record_async(
                    address=addr,
                    duration=duration,
                    outfile=filename,
                    preset=preset,
                    subscribe_chars=chars,
                    verbose=verbose,
                )
            )

        # Run all recordings concurrently
        await asyncio.gather(*tasks)

    return _run(run_multirecord())
