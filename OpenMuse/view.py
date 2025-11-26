"""
Robust OpenMuse Viewer (Visual Enhanced)
========================================
Architecture: GPU Ring Buffer with Master Clock Synchronization.
Features:
- Synchronization: Upsamples aux streams to match EEG.
- Stability: EMA for DC offset removal.
- Performance: Static Ring Buffer (Low CPU).
- Visuals: Battery Indicator, Y-Axis Ticks, Detailed Grid, Signal Quality.
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
    float margin_left = 0.12;   // Increased for Y-ticks
    float margin_right = 0.05;
    float plot_width = 1.0 - margin_left - margin_right;

    float x = margin_left + plot_width * (current_x / u_n_samples);

    // Y position (Stacking)
    float margin_bottom = 0.05;
    float plot_height = 1.0 - margin_bottom;

    float slot_height = plot_height / u_size.x;
    float slot_bottom = margin_bottom + (channel_idx * slot_height);
    float slot_center = slot_bottom + (slot_height * 0.5);

    // Scale: 0.45 leaves 10% padding between channels
    float y = slot_center + (a_position * slot_height * 0.45 * a_y_scale);

    gl_Position = u_projection * vec4(x * u_scale.x, y, 0.0, 1.0);
    v_color = vec4(a_color, 1.0);
}
"""

FRAG_SHADER = """
#version 120
varying vec4 v_color;
void main() { gl_FragColor = v_color; }
"""

# --- BATTERY SHADERS ---
BAT_VERT = """
attribute vec2 a_position;
uniform mat4 u_projection;
void main() {
    gl_Position = u_projection * vec4(a_position, 0.0, 1.0);
}
"""
BAT_FRAG = """
uniform vec4 u_color;
void main() {
    gl_FragColor = u_color;
}
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

                self.channel_info.append(
                    {
                        "stream_idx": s_idx,
                        "ch_idx": ch_i,
                        "name": ch_name,
                        "color": col,
                        "range": rng,  # Full span (Top - Bottom)
                        "scale": 1.0,
                        "is_eeg": is_eeg,
                        "data_idx": self.total_channels,
                        "dc_offset": 0.0,
                        "quality_buf": [],
                    }
                )
                self.total_channels += 1

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

        # Battery Programs
        self.bat_prog_bg = gloo.Program(BAT_VERT, BAT_FRAG)
        self.bat_prog_fill = gloo.Program(BAT_VERT, BAT_FRAG)
        self._bat_bg_vbo = gloo.VertexBuffer(np.zeros((4, 2), dtype=np.float32))
        self._bat_fill_vbo = gloo.VertexBuffer(np.zeros((4, 2), dtype=np.float32))
        self.bat_prog_bg["a_position"] = self._bat_bg_vbo
        self.bat_prog_fill["a_position"] = self._bat_fill_vbo

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
        # Force initial resize to set battery bar positions
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

            # Ticks (Right-aligned, left of signal)
            # Create 3 ticks: Top (+), Zero, Bottom (-)
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

        # Battery Label
        self.lbl_bat = TextVisual(
            "BATT: --%", color="yellow", font_size=8, bold=True, anchor_x="right"
        )

    def _init_grid_lines(self):
        # We need two sets of lines:
        # 1. Separators & Limits (Darker)
        # 2. Zero Lines (Lighter/Thicker)

        limit_pts = []
        zero_pts = []

        margin_bottom = 0.05
        h_plot = 1.0 - margin_bottom
        margin_left = 0.12
        margin_right = 0.05

        for i in range(self.total_channels):
            # Calculate slot geometry
            slot_h = h_plot / self.total_channels
            y_base = margin_bottom + (i * slot_h)
            y_center = y_base + (slot_h * 0.5)

            # The signal shader scales data by 0.45.
            # So the visual "limit" of the data is center +/- (slot_h * 0.45)
            y_top = y_center + (slot_h * 0.45)
            y_bot = y_center - (slot_h * 0.45)

            # Horizontal Range
            x1, x2 = margin_left, 1.0 - margin_right

            # Zero Line
            zero_pts.extend([[x1, y_center], [x2, y_center]])

            # Limit Lines
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
            return

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
        # Update text visuals every ~10 frames to save CPU
        if self.write_ptr % 10 != 0:
            return

        w, h = self.canvas.size
        margin_bottom = 0.05
        h_plot = 1.0 - margin_bottom

        # Update Battery Label Text
        if self.battery_level is not None:
            self.lbl_bat.text = f"BATT: {self.battery_level:.0f}%"
            if self.battery_level > 50:
                self.lbl_bat.color = "lime"
            elif self.battery_level > 20:
                self.lbl_bat.color = "yellow"
            else:
                self.lbl_bat.color = "red"

        for ch in self.channel_info:
            # Channel slot geometry in pixels
            y_rel_bot = margin_bottom + (ch["data_idx"] / self.total_channels) * h_plot
            slot_h_rel = h_plot / self.total_channels
            y_rel_center = y_rel_bot + slot_h_rel * 0.5
            y_rel_top = y_rel_bot + slot_h_rel  # Top of slot

            # Convert to Vispy coordinates (origin top-left)
            y_px_center = h * (1.0 - y_rel_center)

            # Name
            self.lbl_names[ch["data_idx"]].pos = (w * 0.11, y_px_center)

            # Quality
            lbl_q = self.lbl_qual[ch["data_idx"]]
            lbl_q.pos = (w * 0.96, y_px_center)
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

            # Ticks (Top, Zero, Bottom)
            # Ticks are relative to the signal display scaling (0.45)
            # The signal range (ch['range']) spans from -1.0 to 1.0 in shader space,
            # which maps to center +/- 0.45 * slot_height visually.

            # Values
            val_top = ch["range"] / 2.0
            val_bot = -ch["range"] / 2.0

            # Positions
            y_px_top = h * (1.0 - (y_rel_center + (slot_h_rel * 0.45)))
            y_px_bot = h * (1.0 - (y_rel_center - (slot_h_rel * 0.45)))

            # Set Text & Pos
            ticks = self.lbl_ticks[ch["data_idx"]]

            # Top Tick
            ticks[0].text = f"{val_top:.0f}" if abs(val_top) >= 10 else f"{val_top:.1f}"
            ticks[0].pos = (w * 0.115, y_px_top)

            # Zero Tick
            ticks[1].text = "0"
            ticks[1].pos = (w * 0.115, y_px_center)

            # Bottom Tick
            ticks[2].text = f"{val_bot:.0f}" if abs(val_bot) >= 10 else f"{val_bot:.1f}"
            ticks[2].pos = (w * 0.115, y_px_bot)

    def on_draw(self, event):
        gloo.clear(color=(0.1, 0.1, 0.1, 1.0))

        # 1. Grid
        # Limit lines (Dark Gray)
        self.prog_grid["color"] = (0.25, 0.25, 0.25, 1.0)
        self.prog_grid["pos"] = self.grid_limit_vbo
        self.prog_grid.draw("lines")

        # Zero lines (Lighter Gray)
        self.prog_grid["color"] = (0.35, 0.35, 0.35, 1.0)
        self.prog_grid["pos"] = self.grid_zero_vbo
        self.prog_grid.draw("lines")

        # 2. Signals
        self.program.draw("line_strip")

        # 3. Text Labels
        for t in self.lbl_names + self.lbl_qual + self.lbl_time + [self.lbl_bat]:
            t.draw()
        for group in self.lbl_ticks:
            for t in group:
                t.draw()

        # 4. Battery Bar
        if self.battery_level is not None:
            # Draw Background
            self.bat_prog_bg["u_color"] = (0.3, 0.3, 0.3, 1.0)
            self.bat_prog_bg.draw("triangle_strip")

            # Draw Fill
            col = (0.0, 0.8, 0.0, 1.0)
            if self.battery_level < 50:
                col = (0.9, 0.9, 0.2, 1.0)
            if self.battery_level < 20:
                col = (0.9, 0.2, 0.2, 1.0)
            self.bat_prog_fill["u_color"] = col
            self.bat_prog_fill.draw("triangle_strip")

    def on_resize(self, event):
        w, h = event.size
        gloo.set_viewport(0, 0, w, h)

        # Update text transforms
        all_labels = self.lbl_names + self.lbl_qual + self.lbl_time + [self.lbl_bat]
        for group in self.lbl_ticks:
            all_labels.extend(group)

        for t in all_labels:
            t.transforms.configure(canvas=self.canvas, viewport=(0, 0, w, h))

        # Update Battery Bar VBOs (Top Right)
        bar_w = 40
        bar_h = 20
        pad_x = 20
        pad_y = 20

        x_start = w - bar_w - pad_x
        y_start = pad_y

        # Vertices (x, y) - Vispy ortho 0,0 is Bottom-Left, but we usually think Top-Left
        # Ortho is set to (0, w, 0, h). So y=h is top.
        y_gl = h - y_start

        bg_verts = np.array(
            [
                [x_start, y_gl],
                [x_start + bar_w, y_gl],
                [x_start, y_gl - bar_h],
                [x_start + bar_w, y_gl - bar_h],
            ],
            dtype=np.float32,
        )

        self._bat_bg_vbo.set_data(bg_verts)
        self.bat_prog_bg["u_projection"] = ortho(0, w, 0, h, -1, 1)

        # Fill calculation
        fill_pct = max(0.0, min(1.0, (self.battery_level or 0) / 100.0))
        fill_w = max(0.0, (bar_w - 4) * fill_pct)  # 2px border

        fill_verts = np.array(
            [
                [x_start + 2, y_gl - 2],
                [x_start + 2 + fill_w, y_gl - 2],
                [x_start + 2, y_gl - bar_h + 2],
                [x_start + 2 + fill_w, y_gl - bar_h + 2],
            ],
            dtype=np.float32,
        )

        self._bat_fill_vbo.set_data(fill_verts)
        self.bat_prog_fill["u_projection"] = ortho(0, w, 0, h, -1, 1)

        # Position Battery Text
        self.lbl_bat.pos = (x_start - 10, h - y_start - (bar_h / 2))

        # Position Time Axis
        for i, t in enumerate(self.lbl_time):
            x = w * (0.12 + (i / 5) * (0.83))
            t.pos = (x, h - 5)

    def on_mouse_wheel(self, event):
        # Zoom logic (Amplitude)
        delta = event.delta[1] if hasattr(event.delta, "__getitem__") else event.delta
        scale = 1.1 if delta > 0 else 0.9

        # Find channel under mouse
        y_mouse = event.pos[1]
        h = self.canvas.size[1]
        y_norm = 1.0 - (y_mouse / h)

        # Approx hit test
        margin = 0.05
        h_usable = 1.0 - margin
        ch_idx = int(((y_norm - margin) / h_usable) * self.total_channels)

        if 0 <= ch_idx < self.total_channels:
            target = self.channel_info[ch_idx]
            # Identify Group
            grp = (
                "EEG"
                if target["is_eeg"]
                else (
                    "ACC"
                    if "ACC" in target["name"]
                    else "GYRO" if "GYRO" in target["name"] else "OPT"
                )
            )

            # Apply zoom to all in group
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

    # --- ADDED THIS METHOD TO MATCH OLD STRUCTURE ---
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

        # In view_old.py, app.run() is called in the view() function,
        # so we just show the canvas here.


def view(stream_name=None, window_duration=10.0, **kwargs):
    configure_lsl_api_cfg()
    from mne_lsl.stream import StreamLSL

    print("Connecting to Streams...")
    streams = []

    # Auto-detect including battery
    targets = ["Muse_EEG", "Muse_ACCGYRO", "Muse_OPTICS", "Muse_BATTERY"]
    if stream_name:
        targets = [stream_name]

    for name in targets:
        try:
            bufsize = window_duration if "BATTERY" not in name else 5.0
            s = StreamLSL(bufsize=bufsize, name=name)
            s.connect(timeout=1.5)
            streams.append(s)
            print(f"  + Connected: {name}")
        except:
            pass

    if not streams:
        print("Error: No streams found. Is 'openmuse-stream' running?")
        return

    # Instantiate viewer
    v = RealtimeViewer(streams, window_duration=window_duration, **kwargs)
    v.show()  # Explicit show() before run()

    try:
        app.run()
    except KeyboardInterrupt:
        pass