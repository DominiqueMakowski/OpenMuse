"""Display unmodified LSL samples in Rerun."""

import math
import time
from importlib.resources import files

import numpy as np

from .utils import configure_lsl_api_cfg


def _gravity_rotation(acceleration: np.ndarray) -> np.ndarray | None:
    """Map Muse gravity (forward, right, down) to the Z-up, -Y-facing head pose."""
    forward, right, down = acceleration
    # The model's +X is subject-left, +Y is back, and +Z is up.
    # Convert Muse gravity to head-local *up*: negate gravity after changing basis.
    ax, ay, az = right, forward, down
    length = math.hypot(ax, ay, az)
    if not math.isfinite(length) or length < 1e-6:
        return None
    roll = math.atan2(ay, az)
    pitch = math.atan2(-ax, math.hypot(ay, az))
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    return np.array([
        [cp, sp * sr, sp * cr],
        [0.0, cr, -sr],
        [-sp, cp * sr, cp * cr],
    ])


def view_rerun(
    stream_name: str | None = None,
    address: str | None = None,
    channels: list[str] | None = None,
    save: str | None = None,
    duration: float | None = None,
    spawn: bool = True,
    layout: str = "all",
) -> None:
    """Subscribe to existing Muse LSL outlets; never connect to BLE directly."""
    if layout not in ("all", "eeg", "accgyro", "optics"):
        raise ValueError("layout must be one of: all, eeg, accgyro, optics")
    orientation = layout in ("all", "accgyro")
    import rerun as rr
    import rerun.blueprint as rrb

    configure_lsl_api_cfg()
    from mne_lsl.lsl import StreamInlet, resolve_streams

    print("Scanning for LSL streams...")
    infos = [info for info in resolve_streams(timeout=2.0)
             if (stream_name in info.name if stream_name else info.name.startswith("Muse-"))
             and (address is None or address in info.name)
             and (layout == "all" or info.name.startswith(f"Muse-{layout.upper()} ("))]
    if not infos:
        raise RuntimeError("No matching LSL streams found. Start `OpenMuse stream` first.")

    recording = None
    inlets: list[StreamInlet] = []
    try:
        sources: list[tuple[StreamInlet, list[tuple[int, str]]]] = []
        views = []
        orientation_sources: dict[StreamInlet, tuple[str, list[int]]] = {}
        orientation_views = []
        found_channels: set[str] = set()
        for info in sorted(infos, key=lambda info: (info.name, info.source_id, info.uid)):
            inlet = StreamInlet(info, max_buffered=10, processing_flags=None)
            inlets.append(inlet)
            inlet.open_stream(timeout=2.0)
            metadata = inlet.get_sinfo(timeout=2.0)
            labels = metadata.get_channel_names() or []
            root = "streams/" + rr.escape_entity_path_part(info.name + " [" + info.uid + "]")
            if orientation and info.name.startswith("Muse-ACCGYRO ("):
                required = ("ACC_X", "ACC_Y", "ACC_Z")
                if not all(label in labels for label in required):
                    raise ValueError(f"{info.name} is missing ACC_X/ACC_Y/ACC_Z labels")
                frame = root + "/orientation"
                orientation_sources[inlet] = (frame, [labels.index(label) for label in required])
                orientation_views.append(rrb.Spatial3DView(
                    origin=frame, name=f"{info.name}: gravity tilt (yaw = 0)",
                    eye_controls=rrb.EyeControls3D(
                        position=[2.5, -4.0, 2.0], look_target=[0.0, 0.0, 0.0],
                        eye_up=[0.0, 0.0, 1.0],
                    ),
                ))
            selected: list[tuple[int, str]] = []
            for index in range(info.n_channels):
                label = labels[index] if index < len(labels) and labels[index] else f"channel_{index}"
                if channels and label not in channels:
                    continue
                found_channels.add(label)
                path = root + "/" + rr.escape_entity_path_part(f"{index}: {label}")
                selected.append((index, path))
                views.append(rrb.TimeSeriesView(origin=path, name=f"{info.name}: {label}"))
            sources.append((inlet, selected))
            print(f"Connected: {info.name} ({len(selected)} channels)")

        missing = set(channels or ()) - found_channels
        if missing:
            raise ValueError("Unknown channel(s): " + ", ".join(sorted(missing)))
        if not views:
            raise RuntimeError("No channels selected.")

        grid_columns = {"all": None, "eeg": 4, "accgyro": 3, "optics": 4}[layout]
        if layout == "all":
            content = rrb.Grid(*orientation_views, *views)
        else:
            content = rrb.Grid(*views, grid_columns=grid_columns)
            if orientation_views:
                content = rrb.Horizontal(
                    rrb.Vertical(*orientation_views), content, column_shares=[1, 3],
                )
        blueprint = rrb.Blueprint(
            content,
            rrb.BlueprintPanel(state="collapsed"),
            rrb.SelectionPanel(state="collapsed"),
        )
        recording = rr.RecordingStream("OpenMuse")
        sinks = []
        if spawn:
            rr.spawn(connect=False, recording=recording)
            sinks.append(rr.GrpcSink())
        if save:
            sinks.append(rr.FileSink(save))
        rr.set_sinks(*sinks, recording=recording)
        rr.send_blueprint(blueprint, make_active=True, recording=recording)
        head_asset = None
        head_visible: set[StreamInlet] = set()
        if orientation_sources:
            head_asset = rr.Asset3D(
                contents=files("OpenMuse").joinpath("assets/head.glb").read_bytes(),
                media_type="model/gltf-binary",
            )
        for frame, _ in orientation_sources.values():
            rr.log(frame, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True, recording=recording)
            rr.log(frame + "/reference", rr.TransformAxes3D(0.7),
                   static=True, recording=recording)

        print("Receiving LSL samples. Press Ctrl+C to stop.")
        started = time.monotonic()
        while duration is None or time.monotonic() - started < duration:
            received = False
            for inlet, selected in sources:
                values, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=1024)
                if len(timestamps) == 0:
                    continue
                received = True
                values = np.asarray(values)
                index = rr.TimeColumn("lsl_time", duration=timestamps)
                for channel_index, path in selected:
                    rr.send_columns(path, indexes=[index],
                                    columns=rr.Scalars.columns(scalars=values[:, channel_index]),
                                    recording=recording, strict=True)
                if inlet in orientation_sources:
                    frame, acceleration_indices = orientation_sources[inlet]
                    for timestamp, sample in zip(timestamps, values, strict=True):
                        rotation = _gravity_rotation(sample[acceleration_indices])
                        rr.set_time("lsl_time", duration=float(timestamp), recording=recording)
                        if rotation is None:
                            rr.log(frame + "/sensor", rr.Clear(recursive=True),
                                   recording=recording)
                            head_visible.discard(inlet)
                        else:
                            rr.log(frame + "/sensor", rr.Transform3D(mat3x3=rotation),
                                   rr.TransformAxes3D(1.0), recording=recording, strict=True)
                            if inlet not in head_visible:
                                assert head_asset is not None
                                rr.log(frame + "/sensor/head", head_asset,
                                       recording=recording, strict=True)
                                head_visible.add(inlet)
            if not received:
                time.sleep(0.01)
    finally:
        for inlet in inlets:
            inlet.close_stream()
        if recording is not None:
            recording.flush()
            rr.disconnect(recording=recording)
