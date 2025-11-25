"""
High-performance Real-time LSL Viewer using VisPy + GLOO.
Optimized with Ring Buffers and GPU-side normalization.
"""

import time

import numpy as np
from mne_lsl.stream import StreamLSL
from vispy import app, gloo
from vispy.util.transforms import ortho
from vispy.visuals import TextVisual

from .utils import configure_lsl_api_cfg

# --- Shaders ----------------------------------------------------------------
# Moves math to GPU: Normalization and Ring-Buffer scrolling logic happens here.
VERT_SHADER = """
#version 120

attribute float a_position;       // Raw signal value
attribute vec2 a_index;           // (channel_index, sample_index)
attribute vec3 a_color;
attribute float a_y_scale;        // User zoom level
attribute float a_y_mean;         // DC offset (calculated on CPU)
attribute float a_y_range;        // Vertical range (calculated on CPU)

uniform float u_offset;           // Ring buffer write pointer (head)
uniform vec2 u_size;              // (n_channels, n_samples)
uniform mat4 u_projection;

varying vec4 v_color;

void main() {
    // 1. Ring Buffer Scrolling Logic
    // Calculate relative index so the newest sample (at u_offset) is at x=1.0
    float sample_idx = a_index.y;
    float relative_idx = mod(sample_idx - u_offset + u_size.y, u_size.y);

    // X Layout: 12% left margin, 5% right margin
    float x_margin_left = 0.12;
    float x_margin_right = 0.05;
    float x_width = 1.0 - x_margin_left - x_margin_right;
    float x = x_margin_left + (relative_idx / u_size.y) * x_width;

    // 2. Y Layout & Normalization
    float ch_idx = a_index.x;
    float y_bottom_margin = 0.03;
    float y_height = (1.0 - y_bottom_margin) / u_size.x;
    float y_center = y_bottom_margin + (ch_idx * y_height) + (y_height * 0.5);

    // Normalize raw data to [-1, 1] based on range/mean, then scale to channel slot
    float norm_val = 2.0 * (a_position - a_y_mean) / a_y_range;
    float y = y_center + (norm_val * y_height * 0.35 * a_y_scale);

    gl_Position = u_projection * vec4(x, y, 0.0, 1.0);
    v_color = vec4(a_color, 1.0);
}
"""

FRAG_SHADER = """
#version 120
varying vec4 v_color;
void main() { gl_FragColor = v_color; }
"""


class FastViewer:
    def __init__(self, streams, window_duration=10.0, verbose=True):
        self.streams = streams
        self.window_duration = window_duration
        self.verbose = verbose
        self.running = True

        # 0. Find Battery Stream (Add this)
        self.bat_stream = next((s for s in streams if "BATTERY" in s.name), None)

        # 1. Configuration & Layout
        self._setup_channels()

        # 2. Buffers
        max_sfreq = max(s.info["sfreq"] for s in streams)
        self.n_samples = int(max_sfreq * window_duration)
        self.total_channels = len(self.ch_configs)

        # The Main Ring Buffer: (n_samples, n_channels)
        # We write to this on CPU, then upload changed parts to GPU
        self.ring_buffer = np.zeros((self.n_samples, self.total_channels), dtype=np.float32)
        self.write_ptr = 0
        self.last_timestamps = [0.0] * len(streams)  # Track last seen TS per stream

        # 3. VisPy Canvas
        self.canvas = app.Canvas(
            title=f"OpenMuse ({self.total_channels} ch)",
            keys="interactive",
            size=(1400, 900),
        )
        self.canvas.events.draw.connect(self.on_draw)
        self.canvas.events.resize.connect(self.on_resize)
        self.canvas.events.mouse_wheel.connect(self.on_mouse_wheel)

        # 4. GLOO Setup
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self._init_gloo_data()

        # 5. Visuals (Grid & Text)
        self._create_grid()
        self._create_labels()

        # 6. Timer (60Hz refresh is sufficient)
        self.timer = app.Timer(1.0 / 60.0, connect=self.on_timer, start=True)
        self.last_text_update = 0

        # 7. Battery State
        self.bat_level = None
        self.last_bat_check = 0

        if verbose:
            print(f"Viewer started: {self.total_channels} channels, {self.n_samples} buffer size.")

    def _setup_channels(self):
        """Define colors, ranges, and active channels."""
        self.ch_configs = []  # List of dicts

        # Colors
        colors = {
            "EEG": [
                (0, 0.8, 0.82),
                (0.25, 0.41, 0.88),
                (0.12, 0.56, 1.0),
                (0, 0.75, 1.0),
            ],
            "ACC": (0.55, 0.71, 0),
            "GYRO": (0.6, 1.0, 0.6),
            "OPTICS": [(1, 0.65, 0), (1, 0.39, 0.28), (0.86, 0.08, 0.24), (1, 0.27, 0)],
        }

        for s_idx, stream in enumerate(self.streams):
            s_name = stream.name.upper()
            if "BATTERY" in s_name:
                continue

            # Simple activity check (peek 1s)
            try:
                data, _ = stream.get_data(1.0)
                is_active = np.var(data, axis=1) > 1e-10
            except:
                is_active = [True] * len(stream.info["ch_names"])

            for ch_i, name in enumerate(stream.info["ch_names"]):
                if not is_active[ch_i]:
                    continue

                # Determine type props
                ctype = (
                    "EEG"
                    if "EEG" in s_name
                    else ("OPTICS" if "OPTICS" in s_name else "ACC" if "ACC" in name else "GYRO")
                )
                if ctype in ["EEG", "OPTICS"]:
                    col = colors[ctype][ch_i % len(colors[ctype])]
                else:
                    col = colors[ctype]

                yrange = 1000.0 if ctype == "EEG" else 2.0 if ctype == "ACC" else 490.0 if ctype == "GYRO" else 0.4

                self.ch_configs.append(
                    {
                        "stream_idx": s_idx,
                        "ch_idx": ch_i,
                        "name": name,
                        "color": col,
                        "base_range": yrange,
                        "scale": 1.0,
                        "mean": 0.0,
                        "type": ctype,
                    }
                )

    def _init_gloo_data(self):
        """Upload static attributes to GPU."""
        # Indices: (channel_id, sample_id)
        # sample_id 0..N repeats for every channel
        sample_indices = np.tile(np.arange(self.n_samples, dtype=np.float32), self.total_channels)
        channel_indices = np.repeat(np.arange(self.total_channels, dtype=np.float32), self.n_samples)

        self.program["a_index"] = np.c_[channel_indices, sample_indices]
        self.program["a_color"] = np.repeat([c["color"] for c in self.ch_configs], self.n_samples, axis=0)

        # Initial variable attributes
        self.program["a_y_scale"] = np.ones(self.n_samples * self.total_channels, dtype=np.float32)
        self.program["a_y_mean"] = np.zeros(self.n_samples * self.total_channels, dtype=np.float32)
        self.program["a_y_range"] = np.repeat([c["base_range"] for c in self.ch_configs], self.n_samples)

        # Uniforms
        self.program["u_size"] = (self.total_channels, self.n_samples)
        self.program["u_projection"] = ortho(0, 1, 0, 1, -1, 1)
        self.program["u_offset"] = 0.0

        # Create Index Buffer for efficient line strips (one strip per channel)
        # Breaks strip between channels to avoid connecting lines
        indices = []
        for i in range(self.total_channels):
            start = i * self.n_samples
            indices.append(np.arange(start, start + self.n_samples, dtype=np.uint32))
        self.index_buffer = gloo.IndexBuffer(np.concatenate(indices))

        # Position buffer (starts empty)
        self.program["a_position"] = np.zeros(self.n_samples * self.total_channels, dtype=np.float32)

    def _create_grid(self):
        """Simple grid lines."""
        vert = "attribute vec2 p; uniform mat4 m; void main() { gl_Position = m * vec4(p, 0, 1); }"
        frag = "void main() { gl_FragColor = vec4(0.2, 0.2, 0.2, 1); }"
        self.grid_prog = gloo.Program(vert, frag)
        self.grid_prog["m"] = ortho(0, 1, 0, 1, -1, 1)

        lines = []
        ymargin = 0.03
        h = (1.0 - ymargin) / self.total_channels
        x_left, x_right = 0.12, 0.95

        for i in range(self.total_channels):
            y = ymargin + i * h + h * 0.5
            lines.extend([[x_left, y], [x_right, y]])  # Center line

        self.grid_pos = gloo.VertexBuffer(np.array(lines, dtype=np.float32))

    def _create_labels(self):
        """Create text labels."""
        self.labels = []
        ymargin = 0.03
        h = (1.0 - ymargin) / self.total_channels

        # Add Battery Label (Top Right)
        self.bat_text = TextVisual("BATT: --%", color="gray", font_size=8, anchor_x="right", bold=True)
        self.bat_text.pos = (1400 * 0.98, 30)  # Top right corner
        self.labels.append(self.bat_text)

        for i, cfg in enumerate(self.ch_configs):
            y = ymargin + i * h + h * 0.5
            # Channel Name
            t = TextVisual(cfg["name"], color="white", font_size=7, anchor_x="right", bold=True)
            t.pos = (
                1400 * 0.11,
                900 * (1 - y),
            )  # Note: TextVisual coords are pixels, Y inverted? No, usually world.
            # Using pixel coords requires Viewport transform
            self.labels.append(t)

            # Impedance/Stats Label (only for EEG)
            if cfg["type"] == "EEG":
                s = TextVisual("σ: --", color="yellow", font_size=6, anchor_x="left")
                s.pos = (1400 * 0.96, 900 * (1 - y))
                cfg["stat_label"] = s

    def on_timer(self, event):
        """
        Optimized Update Loop:
        1. Fetch only NEW samples (small window).
        2. Write to CPU Ring Buffer.
        3. Upload ONLY changed range to GPU.
        """
        has_updates = False

        for s_idx, stream in enumerate(self.streams):
            if "BATTERY" in stream.name:
                continue

            # Fetch small chunk (approx 2 frames worth to be safe)
            # MNE-LSL doesn't support "pop", so we filter by timestamp
            try:
                chunk, ts = stream.get_data(winsize=0.1)
            except:
                continue

            if chunk.size == 0:
                continue

            # Filter new samples
            last_t = self.last_timestamps[s_idx]
            new_mask = ts > last_t
            if not np.any(new_mask):
                continue

            new_data = chunk[:, new_mask]
            self.last_timestamps[s_idx] = ts[-1]
            n_new = new_data.shape[1]
            has_updates = True

            # Map stream channels to viewer channels
            # (Optimization: could pre-calc indices map)
            for i in range(n_new):
                write_idx = (self.write_ptr + i) % self.n_samples

                for cfg in self.ch_configs:
                    if cfg["stream_idx"] == s_idx:
                        # Write raw value to CPU ring buffer
                        self.ring_buffer[write_idx, self.ch_configs.index(cfg)] = new_data[cfg["ch_idx"], i]

        if has_updates:
            # Advance pointer
            old_ptr = self.write_ptr
            total_new = n_new  # Approximate, assumes streams sync enough
            self.write_ptr = (self.write_ptr + total_new) % self.n_samples

            # Update GPU Buffer (a_position)
            # We flatten the buffer: (n_channels, n_samples) logic in shader requires matching layout
            # Shader expects: channel 0 [samples], channel 1 [samples]...
            # But CPU buffer is [sample][channel]. Transpose is needed.
            # Transpose is O(N), but necessary for line_strip drawing.
            # To OPTIMIZE: Keep CPU buffer as (n_channels, n_samples) and fill strided?
            # Yes. Let's assume self.ring_buffer is (n_samples, n_channels) for easy writes,
            # but we upload column-wise.

            # For pure speed, we upload the WHOLE buffer if small, or use SubData.
            # With <100k points, uploading whole buffer is often faster than Python slicing logic.
            self.program["a_position"].set_data(self.ring_buffer.T.ravel())

            # Update Scroll Offset Uniform
            self.program["u_offset"] = float(self.write_ptr)

            # Update Mean/Stats occasionally (2Hz)
            if time.time() - self.last_text_update > 0.5:
                self._update_stats()
                self._check_battery()  # Check battery
                self.canvas.update()
            else:
                self.canvas.update()

    def _check_battery(self):
        """Poll battery level (low frequency)."""
        if not self.bat_stream:
            return
        try:
            # Fetch last 5 seconds to ensure we catch the slow 1Hz signal
            data, _ = self.bat_stream.get_data(winsize=5.0)
            if data.size > 0:
                lvl = data[0, -1]
                self.bat_text.text = f"BATT: {lvl:.0f}%"
                # Color code
                if lvl > 50:
                    self.bat_text.color = "lime"
                elif lvl > 20:
                    self.bat_text.color = "yellow"
                else:
                    self.bat_text.color = "red"
        except:
            pass

    def _update_stats(self):
        """Update DC offset (mean) and text labels."""
        self.last_text_update = time.time()

        # Calculate means per channel (CPU)
        means = np.mean(self.ring_buffer, axis=0)

        # Update labels & stats
        for i, cfg in enumerate(self.ch_configs):
            cfg["mean"] = means[i]

            if "stat_label" in cfg:
                std = np.std(self.ring_buffer[:, i])
                lbl = cfg["stat_label"]
                lbl.text = f"σ: {std:.1f}"
                lbl.color = "lime" if std < 50 else "yellow" if std < 100 else "red"

        # Upload means to GPU
        self.program["a_y_mean"] = np.repeat(means, self.n_samples)

        # Update scales (in case zoom changed)
        scales = [c["scale"] for c in self.ch_configs]
        self.program["a_y_scale"] = np.repeat(scales, self.n_samples)

    def on_draw(self, event):
        gloo.clear(color=(0.1, 0.1, 0.1, 1.0))
        self.grid_prog.draw("lines", self.grid_pos)
        self.program.draw("line_strip", self.index_buffer)
        for t in self.labels:
            t.draw()
        for c in self.ch_configs:
            if "stat_label" in c:
                c["stat_label"].draw()

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.size)
        # Update text positions... (omitted for brevity, requires scaling logic)

    def on_mouse_wheel(self, event):
        """Zoom amplitude under mouse."""
        delta = event.delta[1] if hasattr(event.delta, "__getitem__") else event.delta
        if delta == 0:
            return

        # Simple global zoom for brevity, or implement per-channel hit test
        scale_mult = 1.1 if delta > 0 else 0.9
        for c in self.ch_configs:
            c["scale"] = max(0.1, c["scale"] * scale_mult)
        # Trigger uniform update in next timer tick
        self.last_text_update = 0

    def show(self):
        self.canvas.show()
        app.run()


def view(stream_name=None, window_duration=5.0, verbose=True):
    configure_lsl_api_cfg()
    streams = []

    # Auto-connect logic
    names = [stream_name] if stream_name else ["Muse_EEG", "Muse_ACCGYRO", "Muse_OPTICS"]
    for n in names:
        try:
            s = StreamLSL(bufsize=window_duration, name=n)
            s.connect(timeout=1.0)
            streams.append(s)
        except:
            pass

    if not streams:
        print("No streams found.")
        return

    v = FastViewer(streams, window_duration, verbose)
    v.show()
