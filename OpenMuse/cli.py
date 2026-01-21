import argparse
import sys

from .muse import find_muse
from .record import record


def _add_find_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=10,
        help="Scan timeout in seconds (default: 10)",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="MuseLSL3", description="MuseLSL3 utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # find subcommand
    p_find = subparsers.add_parser("find", help="Scan for Muse devices")
    _add_find_args(p_find)

    def handle_find(ns):
        find_muse(timeout=ns.timeout, verbose=True)
        return 0

    p_find.set_defaults(func=handle_find)

    # record subcommand
    p_rec = subparsers.add_parser(
        "record", help="Connect and record raw packets to a text file"
    )
    p_rec.add_argument(
        "--address",
        required=True,
        nargs="+",
        help="Device address(es) (e.g., MAC on Windows). Pass multiple space-separated addresses for multi-device recording.",
    )
    p_rec.add_argument(
        "--duration",
        "-d",
        type=float,
        default=30.0,
        help="Recording duration in seconds (default: 30)",
    )
    p_rec.add_argument(
        "--outfile",
        "-o",
        default="muse_record.txt",
        help="Output text file path (for multi-device, address will be appended)",
    )
    p_rec.add_argument(
        "--preset", default="p1041", help="Preset to send (by default, p1041)"
    )

    def handle_record(ns):
        if ns.duration <= 0:
            parser.error("--duration must be positive")
        record(
            address=ns.address,
            duration=ns.duration,
            outfile=ns.outfile,
            preset=ns.preset,
            verbose=True,
        )
        return 0

    p_rec.set_defaults(func=handle_record)

    # stream subcommand
    p_stream = subparsers.add_parser(
        "stream",
        help="Stream decoded EEG and accelerometer/gyroscope data over LSL",
    )
    p_stream.add_argument(
        "--address",
        required=True,
        nargs="+",
        help="Device address(es) (e.g., MAC on Windows). Pass multiple space-separated addresses for multi-device streaming.",
    )
    p_stream.add_argument(
        "--preset",
        default="p1041",
        help="Preset to send (default: p1041 for all channels including EEG)",
    )
    p_stream.add_argument(
        "--duration",
        "-d",
        type=float,
        default=None,
        help="Optional stream duration in seconds. Omit to stream until interrupted.",
    )

    # This combination of nargs, const, and default achieves the desired API:
    # - Not present:         ns.record = False (default)
    # - --record:            ns.record = True (const)
    # - --record "file.txt": ns.record = "file.txt" (nargs='?')
    p_stream.add_argument(
        "--record",
        nargs="?",
        const=True,
        default=False,
        help="Record raw BLE packets. If given without a path, saves to 'rawdata_stream_TIMESTAMP.txt'. "
        "If given with a path (e.g., --record 'myfile.txt'), saves to that file.",
    )
    p_stream.add_argument(
        "--clock",
        default="windowed",
        choices=["adaptive", "constrained", "robust", "standard", "windowed", "windowed10", "windowed30", "windowed60"],
        help="Clock synchronization model (default: windowed). windowed/windowed30 use 30s window, windowed10 uses 10s, windowed60 uses 60s.",
    )
    p_stream.add_argument(
        "--sensors",
        nargs="+",
        default=None,
        choices=["EEG", "ACCGYRO", "OPTICS", "BATTERY"],
        help="Sensor types to stream (default: all). Example: --sensors EEG OPTICS",
    )

    def handle_stream(ns):
        from .stream import stream

        if ns.duration is not None and ns.duration <= 0:
            parser.error("--duration must be positive when provided")
        stream(
            address=ns.address,
            preset=ns.preset,
            duration=ns.duration,
            record=ns.record,  # 'outfile' parameter removed
            verbose=True,
            clock=ns.clock,
            sensors=ns.sensors,
        )
        return 0

    p_stream.set_defaults(func=handle_stream)

    # view subcommand
    p_view = subparsers.add_parser(
        "view",
        help="Visualize EEG and ACC/GYRO data from LSL streams in real-time",
    )
    p_view.add_argument(
        "--address",
        default=None,
        help="MAC address to filter streams by (e.g., 00:55:DA:B9:73:D5). "
        "Only streams from this device will be shown. Useful with multiple devices.",
    )
    p_view.add_argument(
        "--stream-name",
        default=None,
        help="Name (or substring) of specific LSL stream to visualize (default: None = auto-detect Muse streams)",
    )
    p_view.add_argument(
        "--window",
        "-w",
        type=float,
        default=10.0,
        help="Time window to display in seconds (default: 10.0)",
    )

    def handle_view(ns):
        from .view import view

        if ns.window <= 0:
            parser.error("--window must be positive")

        view(
            stream_name=ns.stream_name,
            address=ns.address,
            window_duration=ns.window,
            verbose=True,
        )
        return 0

    p_view.set_defaults(func=handle_view)

    # ===============================================
    # BITalino subcommands
    # ===============================================
    p_find_bitalino = subparsers.add_parser(
        "find_bitalino", help="Scan for BITalino devices"
    )
    _add_find_args(p_find_bitalino)

    def handle_find_bitalino(ns):
        from .bitalino import find_bitalino

        find_bitalino(timeout=ns.timeout, verbose=True)
        return 0

    p_find_bitalino.set_defaults(func=handle_find_bitalino)

    # stream_bitalino subcommand
    p_stream_bitalino = subparsers.add_parser(
        "stream_bitalino", help="Stream data from BITalino to LSL"
    )
    p_stream_bitalino.add_argument(
        "--address", required=True, help="Device address (e.g., MAC on Windows)"
    )
    p_stream_bitalino.add_argument(
        "--channels",
        nargs=6,
        metavar="CH",
        help="Sensor types for the 6 analog channels (e.g., ECG EMG None ...). "
        "Use 'None' or '0' for unused channels. "
        "Available: ECG, EMG, EEG, EDA, ACC, LUX, RAW.",
    )

    def handle_stream_bitalino(ns):
        import asyncio

        from .bitalino import stream_bitalino

        # Prepare channels list: Convert CLI strings "None"/"0" to Python None
        # If --channels is not provided, we pass None (driver defaults to all RAW)
        channels_arg = None
        if ns.channels:
            channels_arg = [
                None if c.lower() in ("none", "0", "null") else c for c in ns.channels
            ]

        asyncio.run(
            stream_bitalino(
                address=ns.address,
                channels=channels_arg,
                sampling_rate=1000,
                buffer_size=32,
            )
        )
        return 0

    p_stream_bitalino.set_defaults(func=handle_stream_bitalino)

    # view_bitalino subcommand
    p_view_bitalino = subparsers.add_parser(
        "view_bitalino",
        help="Visualize BITalino data from LSL streams in real-time",
    )
    p_view_bitalino.add_argument(
        "--stream-name",
        default="BITalino",
        help="Name (or substring) of the LSL stream to visualize (default: BITalino)",
    )
    p_view_bitalino.add_argument(
        "--window",
        "-w",
        type=float,
        default=10.0,
        help="Time window to display in seconds (default: 10.0)",
    )

    def handle_view_bitalino(ns):
        from .bitalino import view_bitalino

        if ns.window <= 0:
            parser.error("--window must be positive")
        view_bitalino(
            stream_name=ns.stream_name,
            window_duration=ns.window,
        )
        return 0

    p_view_bitalino.set_defaults(func=handle_view_bitalino)
    # ===============================================

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
