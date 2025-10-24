import argparse
import sys

from .find import find_devices
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
        find_devices(timeout=ns.timeout, verbose=True)
        return 0

    p_find.set_defaults(func=handle_find)

    # record subcommand
    p_rec = subparsers.add_parser(
        "record", help="Connect and record raw packets to a text file"
    )
    p_rec.add_argument(
        "--address", required=True, help="Device address (e.g., MAC on Windows)"
    )
    p_rec.add_argument(
        "--duration",
        "-d",
        type=float,
        default=30.0,
        help="Recording duration in seconds (default: 30)",
    )
    p_rec.add_argument(
        "--outfile", "-o", default="muse_record.txt", help="Output text file path"
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
        help="Device address (e.g., MAC on Windows)",
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
        )
        return 0

    p_stream.set_defaults(func=handle_stream)

    # view subcommand
    p_view = subparsers.add_parser(
        "view",
        help="Visualize EEG and ACC/GYRO data from LSL streams in real-time",
    )
    p_view.add_argument(
        "--stream-name",
        default=None,
        help="Name of specific LSL stream to visualize (default: None = show all available streams: Muse_EEG + Muse_ACCGYRO)",
    )
    p_view.add_argument(
        "--window",
        "-w",
        type=float,
        default=10.0,
        help="Time window to display in seconds (default: 10.0)",
    )
    p_view.add_argument(
        "--duration",
        "-d",
        type=float,
        default=None,
        help="Optional viewing duration in seconds. Omit to view until window closed.",
    )

    def handle_view(ns):
        from .view import view

        if ns.window <= 0:
            parser.error("--window must be positive")
        if ns.duration is not None and ns.duration <= 0:
            parser.error("--duration must be positive when provided")

        view(
            stream_name=ns.stream_name,
            duration=ns.duration,
            window_size=ns.window,
            verbose=True,
        )
        return 0

    p_view.set_defaults(func=handle_view)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
