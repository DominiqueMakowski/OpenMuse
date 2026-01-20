"""
Robust OpenMuse Viewer (Visual Enhanced)
========================================
Architecture: GPU Ring Buffer with Master Clock Synchronization.
Features:
- Synchronization: Upsamples aux streams to match EEG.
- Stability: EMA for DC offset removal.
- Performance: Static Ring Buffer (Low CPU).
- Visuals: Battery Text (Top Right), Y-Axis Ticks, Detailed Grid, Signal Quality.
"""

import time
import numpy as np
from vispy import app, gloo
from vispy.util.transforms import ortho
from vispy.visuals import TextVisual
from .utils import configure_lsl_api_cfg

# --- SIGNAL SHADERS ---
VERT_SHADER = """
#version 120
attribute float a_position;
attribute vec3 a_index;
attribute vec3 a_color;
attribute float a_y_scale;

uniform float u_offset;
uniform vec2 u_scale;
uniform vec2 u_size;
uniform float u_n_samples;
uniform mat4 u_projection;

varying vec4 v_color;

void main() {
    float channel_idx = a_index.x;
    float sample_idx = a_index.y;

    // Ring Buffer Logic
    float current_x = mod(sample_idx - u_offset + u_n_samples, u_n_samples);

    // Margins
    float margin_left = 0.12;
    float margin_right = 0.05;
    float plot_width = 1.0 - margin_left - margin_right;

    float x = margin_left + plot_width * (current_x / u_n_samples);

    // Y position (Stacking)
    float margin_bottom = 0.05;
    float margin_top = 0.03;
    float plot_height = 1.0 - margin_bottom - margin_top;

    float slot_height = plot_height / u_size.x;
    float slot_bottom = margin_bottom + (channel_idx * slot_height);
    float slot_center = slot_bottom + (slot_height * 0.5);

    // Scale: 0.30
    float y = slot_center + (a_position * slot_height * 0.30 * a_y_scale);

    gl_Position = u_projection * vec4(x * u_scale.x, y, 0.0, 1.0);
    v_color = vec4(a_color, 1.0);
}
"""

FRAG_SHADER = """
#version 120
varying vec4 v_color;
void main() { gl_FragColor = v_color; }
"""


class RealtimeViewer:
    def __init__(
        self, streams, window_duration=10.0, update_interval=0.02, verbose=True
    ):
        self.streams = streams
        self.window_duration = window_duration
        self.verbose = verbose
        self.start_time = time.time()

        # --- 1. Channel Configuration ---
        self.channel_info = []
        self.total_channels = 0
        self.battery_stream_idx = None
        self.battery_level = None  # 0-100

        # Colors
        colors_eeg = [(0, 1, 1), (0, 0.5, 1), (0, 0, 1), (0.5, 0, 1)]  # Cyans/Blues
        color_acc = (0.6, 0.8, 0.2)  # Greenish
        color_gyro = (0.8, 0.6, 0.2)  # Orangeish
        color_opt = (1, 0, 0)  # Red

        # Identify streams
        max_sfreq = 0
        self.master_stream_idx = 0

        for s_idx, stream in enumerate(streams):
            s_name = stream.name

            # Handle Battery Stream Separately
            if "BATTERY" in s_name.upper():
                self.battery_stream_idx = s_idx
                continue

            sfreq = stream.info["sfreq"]
            if sfreq > max_sfreq:
                max_sfreq = sfreq
                self.master_stream_idx = s_idx

            is_eeg = "EEG" in s_name.upper()

            for ch_i, ch_name in enumerate(stream.info.ch_names):
                if is_eeg:
                    col = colors_eeg[ch_i % 4]
                    rng = 800.0  # uV
                elif "ACC" in ch_name:
                    col = color_acc
                    rng = 2.0  # G
                elif "GYRO" in ch_name:
                    col = color_gyro
                    rng = 300.0  # deg/s
                else:
                    col = color_opt
                    rng = 1000.0

                # Determine sort order: EEG=0, ACCGYRO=1, OPTICS=2
                if is_eeg:
                    sort_order = 0
                elif "ACC" in ch_name or "GYRO" in ch_name:
                    sort_order = 1
                else:
                    sort_order = 2

                self.channel_info.append(
                    {
                        "stream_idx": s_idx,
                        "ch_idx": ch_i,
                        "name": ch_name,
                        "color": col,
                        "range": rng,  # Full span (Top - Bottom)
                        "scale": 1.0,
                        "is_eeg": is_eeg,
                        "sort_order": sort_order,
                        "dc_offset": 0.0,
                        "quality_buf": [],
                    }
                )

        # Sort channels for consistent display order: EEG at top, then ACCGYRO, then OPTICS at bottom
        # Lower sort_order appears at TOP of display (higher data_idx)
        self.channel_info.sort(key=lambda c: (c["sort_order"], c["name"]), reverse=True)

        # Assign data_idx after sorting
        for idx, ch in enumerate(self.channel_info):
            ch["data_idx"] = idx
            self.total_channels = idx + 1

        self.master_sfreq = max_sfreq
        self.n_samples = int(self.window_duration * self.master_sfreq)

        if verbose:
            print(f"Viewer Configured: {self.total_channels} signal channels.")
            if self.battery_stream_idx is not None:
                print("  + Battery Stream detected.")

        # --- 2. GPU Memory (Ring Buffer) ---
        self.data_buffer = np.zeros(
            (self.n_samples, self.total_channels), dtype=np.float32
        )
        self.write_ptr = 0
        self.last_timestamps = {i: 0.0 for i in range(len(streams))}

        # Track when we last received data (for "no data" warning)
        self.last_data_time = time.time()
        self.no_data_warning = False
        self.NO_DATA_TIMEOUT = 3.0  # seconds without data to trigger warning

        # --- 3. VisPy Setup ---
        self.canvas = app.Canvas(
            title="OpenMuse Realtime", keys="interactive", size=(1400, 900)
        )

        # Signal Program
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        ch_indices = np.repeat(np.arange(self.total_channels), self.n_samples)
        sa_indices = np.tile(np.arange(self.n_samples), self.total_channels)

        self.program["a_index"] = np.c_[
            ch_indices, sa_indices, np.zeros_like(ch_indices)
        ].astype(np.float32)

        colors_flat = np.array(
            [c["color"] for c in self.channel_info], dtype=np.float32
        )
        self.program["a_color"] = np.repeat(colors_flat, self.n_samples, axis=0)
        self.program["a_y_scale"] = np.ones(
            self.total_channels * self.n_samples, dtype=np.float32
        )
        self.vbo_pos = gloo.VertexBuffer(self.data_buffer.T.ravel().astype(np.float32))
        self.program["a_position"] = self.vbo_pos
        self.program["u_projection"] = ortho(0, 1, 0, 1, -1, 1)
        self.program["u_size"] = (float(self.total_channels), 1.0)
        self.program["u_n_samples"] = float(self.n_samples)
        self.program["u_offset"] = 0.0
        self.program["u_scale"] = (1.0, 1.0)

        # we generate indices that define separate lines for every point within a channel, but do NOT bridge the gap between channels.
        indices = []
        for i in range(self.total_channels):
            start = i * self.n_samples
            # We create segments (0,1), (1,2) ... (N-2, N-1)
            # We intentionally stop at N-1 and do not connect to N
            base = np.arange(start, start + self.n_samples - 1, dtype=np.uint32)

            # Interleave to create line pairs: [0, 1, 1, 2, 2, 3...]
            rows = np.vstack((base, base + 1)).T.flatten()
            indices.append(rows)

        self.index_buffer = gloo.IndexBuffer(np.concatenate(indices))

        # Grid Program
        self.prog_grid = gloo.Program(
            "attribute vec2 pos; uniform mat4 proj; void main() { gl_Position = proj * vec4(pos, 0.0, 1.0); }",
            "uniform vec4 color; void main() { gl_FragColor = color; }",
        )
        self.prog_grid["proj"] = ortho(0, 1, 0, 1, -1, 1)

        # --- 4. Visual Elements ---
        self._init_labels()
        self._init_grid_lines()

        # Events
        self.canvas.events.draw.connect(self.on_draw)
        self.canvas.events.resize.connect(self.on_resize)
        self.canvas.events.mouse_wheel.connect(self.on_mouse_wheel)

        # Timer
        self.timer = app.Timer(update_interval, connect=self.on_timer, start=True)
        # Force initial resize
        self.on_resize(type("Event", (object,), {"size": self.canvas.size}))

    def _init_labels(self):
        self.lbl_names = []
        self.lbl_qual = []
        self.lbl_ticks = []  # List of tuples (top, zero, bottom)

        for ch in self.channel_info:
            # Channel Name (Left)
            t = TextVisual(
                ch["name"], color="white", font_size=8, bold=True, anchor_x="right"
            )
            self.lbl_names.append(t)

            # Quality (Right)
            q = TextVisual("", color="green", font_size=7, bold=True, anchor_x="left")
            self.lbl_qual.append(q)

            # Ticks
            ticks = []
            for _ in range(3):
                tick = TextVisual(
                    "-", color="gray", font_size=6, anchor_x="right", anchor_y="center"
                )
                ticks.append(tick)
            self.lbl_ticks.append(ticks)

        # Time Axis
        self.lbl_time = []
        for i in range(6):
            t = TextVisual(
                f"-{self.window_duration * (1 - i/5):.0f}s",
                color="gray",
                font_size=7,
                anchor_y="top",
            )
            self.lbl_time.append(t)

        # Battery Label - Anchor top right
        self.lbl_bat = TextVisual(
            "--%",
            color="yellow",
            font_size=10,
            bold=True,
            anchor_x="right",
            anchor_y="top",
        )

        # No-data warning label - Center of screen
        self.lbl_no_data = TextVisual(
            "⚠ NO DATA RECEIVED",
            color="red",
            font_size=16,
            bold=True,
            anchor_x="center",
            anchor_y="center",
        )

        # Device MAC address – Anchor top center
        self.lbl_device = TextVisual(
            f"Device: {getattr(self, 'device_id', '')}",
            color="gray",
            font_size=8,
            bold=True,
            anchor_x="center",
            anchor_y="top",
        )

    def _update_time_labels(self):
        w, h = self.canvas.size

        # --- Plot margins (must match shader) ---
        margin_left = 0.12
        margin_right = 0.05
        margin_bottom = 0.05

        n_ticks = len(self.lbl_time)
        usable_width = 1.0 - margin_left - margin_right

        # Place labels slightly below plot
        y_norm = 1.0 - margin_bottom + 0.025

        # Font scaling
        BASE_FONT_TIME = 7
        scale_factor = min(w / 1400, h / 900)

        for i, t in enumerate(self.lbl_time):
            x_norm = margin_left + (i / (n_ticks - 1)) * usable_width

            # Convert normalized → pixel
            t.pos = (x_norm * w, y_norm * h)

            # Font scaling
            t.font_size = max(4, int(BASE_FONT_TIME * scale_factor))

            # Label text
            time_val = self.window_duration * (1 - i / (n_ticks - 1))
            t.text = (
                f"-{time_val:.1f}s".replace(".0s", "s")
                if time_val < 10
                else f"-{int(time_val)}s"
            )

    def _init_grid_lines(self):
        limit_pts = []
        zero_pts = []

        margin_bottom = 0.05
        margin_top = 0.03
        h_plot = 1.0 - margin_bottom - margin_top
        margin_left = 0.12
        margin_right = 0.05

        for i in range(self.total_channels):
            slot_h = h_plot / self.total_channels
            y_base = margin_bottom + (i * slot_h)
            y_center = y_base + (slot_h * 0.5)

            y_top = y_center + (slot_h * 0.30)
            y_bot = y_center - (slot_h * 0.30)

            x1, x2 = margin_left, 1.0 - margin_right

            zero_pts.extend([[x1, y_center], [x2, y_center]])
            limit_pts.extend([[x1, y_top], [x2, y_top]])
            limit_pts.extend([[x1, y_bot], [x2, y_bot]])

        self.grid_limit_vbo = gloo.VertexBuffer(np.array(limit_pts, dtype=np.float32))
        self.grid_zero_vbo = gloo.VertexBuffer(np.array(zero_pts, dtype=np.float32))

    def on_timer(self, event):
        # 1. Battery Update
        if self.battery_stream_idx is not None:
            try:
                # Poll battery rarely (every 1s roughly)
                if self.write_ptr % 50 == 0:
                    bat_data, _ = self.streams[self.battery_stream_idx].get_data(
                        winsize=1.0
                    )
                    if bat_data.size > 0:
                        self.battery_level = float(bat_data[0, -1])
            except:
                pass

        # 2. Signal Update (Master Clock Logic)
        master_stream = self.streams[self.master_stream_idx]
        try:
            chunk_m, ts_m = master_stream.get_data(winsize=0.5)
        except:
            return

        if len(ts_m) == 0:
            # Check for no-data timeout
            if time.time() - self.last_data_time > self.NO_DATA_TIMEOUT:
                self.no_data_warning = True
                self.canvas.update()  # Force redraw to show warning
            return

        # Data received - reset no-data warning
        self.last_data_time = time.time()
        self.no_data_warning = False

        last_t = self.last_timestamps[self.master_stream_idx]
        new_mask = ts_m > last_t
        if not np.any(new_mask):
            return

        data_new_m = chunk_m[:, new_mask]
        ts_new_m = ts_m[new_mask]
        self.last_timestamps[self.master_stream_idx] = ts_new_m[-1]

        n_slots_to_fill = data_new_m.shape[1]
        batch_update = np.zeros(
            (n_slots_to_fill, self.total_channels), dtype=np.float32
        )

        for s_idx, stream in enumerate(self.streams):
            if s_idx == self.battery_stream_idx:
                continue

            if s_idx == self.master_stream_idx:
                raw_data = data_new_m.T
            else:
                try:
                    chunk_s, ts_s = stream.get_data(winsize=0.5)
                except:
                    continue

                if len(ts_s) == 0:
                    raw_data = np.zeros((n_slots_to_fill, chunk_s.shape[0]))
                else:
                    last_t_s = self.last_timestamps[s_idx]
                    mask_s = ts_s > last_t_s
                    if np.any(mask_s):
                        d_s = chunk_s[:, mask_s]
                        self.last_timestamps[s_idx] = ts_s[mask_s][-1]
                        # Interpolate
                        x_old = np.linspace(0, 1, d_s.shape[1])
                        x_new = np.linspace(0, 1, n_slots_to_fill)
                        raw_data = np.zeros((n_slots_to_fill, d_s.shape[0]))
                        for i in range(d_s.shape[0]):
                            raw_data[:, i] = np.interp(x_new, x_old, d_s[i, :])
                    else:
                        raw_data = np.zeros((n_slots_to_fill, chunk_s.shape[0]))

            # Process Channels
            relevant_chs = [c for c in self.channel_info if c["stream_idx"] == s_idx]
            for ch_info in relevant_chs:
                signal = raw_data[:, ch_info["ch_idx"]]

                # DC Offset Removal (EMA)
                alpha = 0.005
                for val in signal:
                    ch_info["dc_offset"] = (1.0 - alpha) * ch_info[
                        "dc_offset"
                    ] + alpha * val

                centered = signal - ch_info["dc_offset"]

                # Quality
                if ch_info["is_eeg"]:
                    ch_info["quality_buf"].extend(centered)
                    if len(ch_info["quality_buf"]) > 256:
                        ch_info["quality_buf"] = ch_info["quality_buf"][-256:]

                # Normalize to range
                normalized = 2.0 * (centered / ch_info["range"])
                batch_update[:, ch_info["data_idx"]] = normalized

        # 3. Write to Buffer
        idx_start = self.write_ptr
        idx_end = self.write_ptr + n_slots_to_fill

        if idx_end <= self.n_samples:
            self.data_buffer[idx_start:idx_end, :] = batch_update
        else:
            part1 = self.n_samples - idx_start
            part2 = n_slots_to_fill - part1
            self.data_buffer[idx_start:, :] = batch_update[:part1]
            self.data_buffer[:part2, :] = batch_update[part1:]

        self.write_ptr = (self.write_ptr + n_slots_to_fill) % self.n_samples
        self.vbo_pos.set_data(self.data_buffer.T.ravel().astype(np.float32))
        self.program["u_offset"] = float(self.write_ptr)

        self._update_ui_labels()
        self.canvas.update()

    def _update_ui_labels(self):
        if self.write_ptr % 10 != 0:
            return

        w, h = self.canvas.size
        margin_bottom = 0.05
        margin_top = 0.03
        h_plot = 1.0 - margin_bottom - margin_top

        # Base font sizes
        BASE_FONT_NAME = 8
        BASE_FONT_QUAL = 7
        BASE_FONT_TICK = 4
        BASE_FONT_BAT = 10

        # Scale factor relative to canvas size
        scale_factor = min(w / 1400, h / 900)

        # Update Battery Label Text
        # Only show battery if we have a battery stream AND have received data
        if self.battery_level is not None:
            self.lbl_bat.text = f"{self.battery_level:.0f}%"
            if self.battery_level > 80:
                self.lbl_bat.color = "green"
            elif self.battery_level > 60:
                self.lbl_bat.color = "yellowgreen"
            elif self.battery_level > 40:
                self.lbl_bat.color = "yellow"
            elif self.battery_level > 20:
                self.lbl_bat.color = "orangered"
            else:
                self.lbl_bat.color = "red"
        elif self.battery_stream_idx is None:
            # No battery stream available - hide the label
            self.lbl_bat.text = ""

        self.lbl_bat.font_size = max(4, int(BASE_FONT_BAT * scale_factor))
        margin_norm_x = 0.96
        margin_norm_y = 0.035
        self.lbl_bat.pos = ((1.0 - margin_norm_x) * w, margin_norm_y * h)

        # Shader margins for left/right
        shader_margin_left = 0.12
        label_offset = 0.03
        tick_offset = 0.005

        for ch in self.channel_info:
            # Channel slot geometry in pixels
            y_rel_bot = margin_bottom + (ch["data_idx"] / self.total_channels) * h_plot
            slot_h_rel = h_plot / self.total_channels
            y_rel_center = y_rel_bot + slot_h_rel * 0.5

            # Convert to Vispy coordinates (origin top-left)
            y_px_center = h * (1.0 - y_rel_center)
            padding_factor = 0.30
            y_px_top = h * (1.0 - (y_rel_center + slot_h_rel * padding_factor))
            y_px_bot = h * (1.0 - (y_rel_center - slot_h_rel * padding_factor))

            # Name
            label_x = w * (shader_margin_left - label_offset)
            label_x = np.clip(label_x, 2, w - 2)
            lbl_name = self.lbl_names[ch["data_idx"]]
            lbl_name.pos = (label_x, y_px_center)
            lbl_name.font_size = max(4, int(BASE_FONT_NAME * scale_factor))

            # Quality
            lbl_q = self.lbl_qual[ch["data_idx"]]
            lbl_q.pos = (w * 0.96, y_px_center)
            lbl_q.font_size = max(4, int(BASE_FONT_QUAL * scale_factor))
            if ch["is_eeg"] and len(ch["quality_buf"]) > 50:
                imp = np.std(ch["quality_buf"])
                lbl_q.text = f"σ:{imp:.0f}"
                lbl_q.color = (
                    (0, 1, 0, 1)
                    if imp < 50
                    else (1, 1, 0, 1) if imp < 100 else (1, 0, 0, 1)
                )
            else:
                lbl_q.text = ""

            # Ticks
            tick_x = w * (shader_margin_left - tick_offset)
            tick_x = np.clip(tick_x, 2, w - 2)
            val_top = ch["range"] / 2.0
            val_bot = -ch["range"] / 2.0
            ticks = self.lbl_ticks[ch["data_idx"]]
            ticks[0].text = f"{val_top:.0f}" if abs(val_top) >= 10 else f"{val_top:.1f}"
            ticks[0].pos = (tick_x, y_px_top)
            ticks[0].font_size = max(4, int(BASE_FONT_TICK * scale_factor))
            ticks[1].text = "0"
            ticks[1].pos = (tick_x, y_px_center)
            ticks[1].font_size = max(4, int(BASE_FONT_TICK * scale_factor))
            ticks[2].text = f"{val_bot:.0f}" if abs(val_bot) >= 10 else f"{val_bot:.1f}"
            ticks[2].pos = (tick_x, y_px_bot)
            ticks[2].font_size = max(4, int(BASE_FONT_TICK * scale_factor))

    def on_draw(self, event):
        gloo.clear(color=(0.1, 0.1, 0.1, 1.0))

        # 1. Grid
        self.prog_grid["color"] = (0.25, 0.25, 0.25, 1.0)
        self.prog_grid["pos"] = self.grid_limit_vbo
        self.prog_grid.draw("lines")

        self.prog_grid["color"] = (0.35, 0.35, 0.35, 1.0)
        self.prog_grid["pos"] = self.grid_zero_vbo
        self.prog_grid.draw("lines")

        # 2. Signals
        # We use GL_LINES with our custom IndexBuffer to prevent connections between channels
        self.program.draw("lines", indices=self.index_buffer)

        # 3. Text Labels
        for t in (
            self.lbl_names
            + self.lbl_qual
            + self.lbl_time
            + [self.lbl_bat, self.lbl_device]
        ):
            t.draw()
        for group in self.lbl_ticks:
            for t in group:
                t.draw()

        # 4. No-data warning overlay (if active)
        if self.no_data_warning:
            self.lbl_no_data.draw()

    def on_resize(self, event):
        w, h = event.size
        gloo.set_viewport(0, 0, w, h)

        # Device label position
        top_margin = 0.0225
        self.lbl_device.pos = (w * 0.5, top_margin * h)

        # Font scaling relative to canvas
        BASE_FONT_DEVICE = 8
        scale_factor = min(w / 1400, h / 900)
        self.lbl_device.font_size = int(BASE_FONT_DEVICE * scale_factor)

        # Position no-data warning in center
        self.lbl_no_data.pos = (w * 0.5, h * 0.5)
        self.lbl_no_data.font_size = max(12, int(20 * scale_factor))

        # Update text transforms
        all_labels = (
            self.lbl_names
            + self.lbl_qual
            + self.lbl_time
            + [self.lbl_bat, self.lbl_device, self.lbl_no_data]
        )
        for group in self.lbl_ticks:
            all_labels.extend(group)

        for t in all_labels:
            t.transforms.configure(canvas=self.canvas, viewport=(0, 0, w, h))

        self._update_time_labels()

    def on_mouse_wheel(self, event):
        delta = event.delta[1] if hasattr(event.delta, "__getitem__") else event.delta
        scale = 1.1 if delta > 0 else 0.9

        y_mouse = event.pos[1]
        h = self.canvas.size[1]
        y_norm = 1.0 - (y_mouse / h)

        margin = 0.05
        h_usable = 1.0 - margin
        ch_idx = int(((y_norm - margin) / h_usable) * self.total_channels)

        if 0 <= ch_idx < self.total_channels:
            target = self.channel_info[ch_idx]
            grp = (
                "EEG"
                if target["is_eeg"]
                else (
                    "ACC"
                    if "ACC" in target["name"]
                    else "GYRO" if "GYRO" in target["name"] else "OPT"
                )
            )

            for ch in self.channel_info:
                curr_grp = (
                    "EEG"
                    if ch["is_eeg"]
                    else (
                        "ACC"
                        if "ACC" in ch["name"]
                        else "GYRO" if "GYRO" in ch["name"] else "OPT"
                    )
                )

                if curr_grp == grp:
                    ch["range"] /= scale

    def show(self):
        """Show the canvas and start the event loop."""
        self.canvas.show()

        @self.canvas.connect
        def on_close(event):
            self.timer.stop()
            for stream in self.streams:
                stream.disconnect()
            if self.verbose:
                print("Viewer closed.")


def view(stream_name=None, address=None, window_duration=10.0, **kwargs):
    """View LSL streams in real-time.

    Args:
        stream_name: Name (or substring) of specific LSL stream to visualize.
        address: MAC address to filter streams by. If provided, only streams
                 containing this address in their name will be shown.
                 Useful when multiple Muse devices are streaming.
        window_duration: Time window to display in seconds.
    """
    configure_lsl_api_cfg()
    from mne_lsl.stream import StreamLSL
    from mne_lsl.lsl import resolve_streams

    print("Connecting to Streams...")
    streams = []

    # Resolve all available streams
    print("Scanning for available LSL streams...")
    infos = resolve_streams()
    found_names = []

    if stream_name:
        # If a specific name is requested, try to find it exactly or as a substring
        for info in infos:
            if stream_name == info.name:
                found_names.append(info.name)

        # If not found exactly, try substring match
        if not found_names:
            for info in infos:
                if stream_name in info.name:
                    found_names.append(info.name)

        # If still not found, maybe the user provided the full name but resolve_streams missed it (rare)
        if not found_names:
            # Fallback: try to connect to it directly (maybe it wasn't resolved yet)
            found_names = [stream_name]
    else:
        # Auto-detect Muse streams (format: Muse-TYPE (DEVICE_ID))
        # Collect streams by device address
        device_streams = {}

        for info in infos:
            n = info.name
            if "Muse" in n:
                if any(t in n for t in ["EEG", "ACCGYRO", "OPTICS", "BATTERY"]):
                    # Extract device ID from stream name (e.g., "Muse-EEG (00:55:DA:B9:FA:20)")
                    if "(" in n and ")" in n:
                        device_id = n.split("(")[-1].split(")")[0].strip()
                    else:
                        device_id = "unknown"

                    if device_id not in device_streams:
                        device_streams[device_id] = []
                    device_streams[device_id].append(n)

        # Check if multiple devices detected
        detected_devices = list(device_streams.keys())

        if len(detected_devices) > 1 and address is None:
            print("\n⚠ WARNING: Multiple Muse devices detected!")
            print("The following devices are streaming:")
            for i, device_id in enumerate(detected_devices, 1):
                print(f"  {i}. {device_id}")
            print(
                "To view a specific device, open separate terminals and use the --address argument:"
            )
            print(f"  Example: OpenMuse view --address {detected_devices[0]}")
            print("\nProceeding to display streams from the first device only...")
            print(f"Selected device: {detected_devices[0]}\n")

            # Use only the first device's streams
            found_names = device_streams[detected_devices[0]]
            selected_device_id = detected_devices[0]
        elif detected_devices:
            # Single device or address filter specified
            if address:
                # Filter by address
                matching_devices = [d for d in detected_devices if address in d]
                if matching_devices:
                    found_names = device_streams[matching_devices[0]]
                    selected_device_id = matching_devices[0]
                    if len(matching_devices) > 1:
                        print(
                            f"Note: Address '{address}' matches multiple devices. Using first match: {matching_devices[0]}"
                        )
                else:
                    print(f"No devices matching address '{address}' found.")
                    print(f"Available devices: {', '.join(detected_devices)}")
                    return
            else:
                # Single device, use it
                found_names = device_streams[detected_devices[0]]
                selected_device_id = detected_devices[0]

    if not found_names:
        print("No Muse streams found. Is 'OpenMuse stream' running?")
        return

    for name in found_names:
        try:
            bufsize = window_duration if "BATTERY" not in name else 5.0
            s = StreamLSL(bufsize=bufsize, name=name)
            s.connect(timeout=1.5)
            streams.append(s)
            print(f"  + Connected: {name}")
        except:
            pass

    if not streams:
        print("Error: No streams found. Is 'OpenMuse stream' running?")
        return

    # Instantiate viewer
    v = RealtimeViewer(streams, window_duration=window_duration, **kwargs)
    v.device_id = selected_device_id
    v.lbl_device.text = f"Device: {selected_device_id}"
    v.show()

    try:
        app.run()
    except KeyboardInterrupt:
        pass
