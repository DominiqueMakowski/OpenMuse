"""Real-time visualization using GLOO (OpenGL) for maximum performance with many channels."""

import time
from typing import Optional, Sequence

import numpy as np
from vispy import app, gloo
from vispy.util.transforms import ortho
from vispy.visuals import TextVisual

from .utils import configure_lsl_api_cfg

# Visualization constants
CHANNEL_DETECTION_DURATION = 2.0  # Seconds to collect data for channel detection
CHANNEL_STATS_BUFFER_DURATION = (
    1.5  # Seconds of data for statistics (EEG std, running mean)
)
CHANNEL_VARIANCE_THRESHOLD = 1e-10  # Variance threshold to consider channel active
CHANNEL_MEAN_THRESHOLD = 1e-6  # Mean threshold to consider channel active

# Vertex shader - processes each vertex position
VERT_SHADER = """
#version 120

// Vertex attributes
attribute float a_position;      // Y value of the data point
attribute vec3 a_index;           // (channel_index, sample_index, unused)
attribute vec3 a_color;           // RGB color for this channel
attribute float a_y_scale;        // Per-channel amplitude scale

// Uniforms (constants for all vertices)
uniform vec2 u_scale;             // (x_scale, unused) for x-axis zooming
uniform vec2 u_size;              // (n_channels, 1)
uniform float u_n_samples;        // Number of samples per channel
uniform mat4 u_projection;        // Projection matrix

// Output to fragment shader
varying vec4 v_color;

void main() {
    // Calculate normalized coordinates
    float channel_idx = a_index.x;
    float sample_idx = a_index.y;
    
    // X position: Leave space on left for channel names and ticks
    float x_margin = 0.15;  // 15% of width reserved for labels and ticks
    float x = x_margin + (1.0 - x_margin) * (sample_idx / u_n_samples);
    
    // Y position: stack channels vertically with bottom margin for x-axis labels
    float y_bottom_margin = 0.03;  // 3% bottom margin for x-axis time labels
    float y_usable_height = 1.0 - y_bottom_margin;
    
    // Each channel gets 1/n_channels of usable vertical space
    float y_offset = y_bottom_margin + (channel_idx / u_size.x) * y_usable_height;
    float y_scale = y_usable_height / u_size.x;  // Height allocated to each channel
    
    // Center the signal within its allocated space
    // a_position is normalized to [-1, 1], scale by 0.35 to use more space
    // a_y_scale controls per-channel amplitude scaling (mouse wheel zoom)
    float y = y_offset + y_scale * 0.5 + (a_position * y_scale * 0.35 * a_y_scale);
    
    // Apply projection (only apply x scale to x coordinate)
    gl_Position = u_projection * vec4(x * u_scale.x, y, 0.0, 1.0);
    
    // Pass color to fragment shader
    v_color = vec4(a_color, 1.0);
}
"""

# Fragment shader - determines pixel colors
FRAG_SHADER = """
#version 120

varying vec4 v_color;

void main() {
    gl_FragColor = v_color;
}
"""


class RealtimeViewer:
    """High-performance real-time signal viewer using GLOO."""

    def _detect_active_channels(self, streams, verbose):
        """Detect which channels have actual data (non-zero variance)."""
        import time as time_module

        if verbose:
            print("Detecting active channels (collecting 2 seconds of data)...")

        self.active_channels = []

        for stream in streams:
            # Get number of channels from channel names list
            ch_names = stream.info.ch_names
            n_channels = len(ch_names)

            # Collect data for CHANNEL_DETECTION_DURATION seconds to detect active channels
            time_module.sleep(CHANNEL_DETECTION_DURATION)

            try:
                data, timestamps = stream.get_data(winsize=CHANNEL_DETECTION_DURATION)
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not get data from {stream.name}: {e}")
                # Mark all channels as active if we can't detect
                self.active_channels.append([True] * n_channels)
                continue

            if data.shape[1] == 0:
                # No data yet, mark all as active
                self.active_channels.append([True] * n_channels)
                continue

            # Check variance for each channel
            channel_active = []
            for ch_idx in range(n_channels):
                ch_data = data[ch_idx, :]
                variance = np.var(ch_data)
                # Consider active if variance > threshold or mean != 0
                is_active = (
                    variance > CHANNEL_VARIANCE_THRESHOLD
                    or np.abs(np.mean(ch_data)) > CHANNEL_MEAN_THRESHOLD
                )
                channel_active.append(is_active)

            self.active_channels.append(channel_active)

            # Report findings
            if verbose:
                active_names = [
                    ch_names[i] for i, active in enumerate(channel_active) if active
                ]
                n_active = len(active_names)
                print(
                    f"  {stream.name}: {n_active} active channel{'s' if n_active != 1 else ''}"
                )

        if verbose:
            print()

    def __init__(
        self,
        streams: Sequence,
        window_size: float = 10.0,
        update_interval: float = 0.005,
        duration: Optional[float] = None,
        verbose: bool = True,
    ):
        self.streams = streams
        self.window_size = window_size
        self.update_interval = update_interval
        self.duration = duration
        self.verbose = verbose
        self.start_time = time.time()

        # Collect channel info from all streams
        # First, detect active channels (non-zero variance)
        self._detect_active_channels(streams, verbose)

        self.channel_info = []  # List of (stream_idx, ch_idx, ch_name, color, y_range)
        self.total_channels = 0

        # Color schemes
        eeg_colors = [
            (0 / 255, 206 / 255, 209 / 255),  # Cyan
            (65 / 255, 105 / 255, 225 / 255),  # Royal Blue
            (30 / 255, 144 / 255, 255 / 255),  # Dodger Blue
            (0 / 255, 191 / 255, 255 / 255),  # Deep Sky Blue
        ]
        acc_color = (141 / 255, 182 / 255, 0 / 255)  # Apple green
        gyro_color = (152 / 255, 255 / 255, 152 / 255)  # Mint green
        optics_colors = [
            (255 / 255, 165 / 255, 0 / 255),  # Orange
            (255 / 255, 99 / 255, 71 / 255),  # Tomato
            (220 / 255, 20 / 255, 60 / 255),  # Crimson
            (255 / 255, 69 / 255, 0 / 255),  # Red-Orange
        ]

        for stream_idx, stream in enumerate(streams):
            stream_name = stream.name
            is_eeg = "EEG" in stream_name.upper()
            is_optics = "OPTICS" in stream_name.upper()

            for ch_idx, ch_name in enumerate(stream.info.ch_names):
                # Skip inactive channels (zero-padded)
                if not self.active_channels[stream_idx][ch_idx]:
                    continue
                # Determine color and display range (for proper scaling)
                # Range is the total vertical span; signals will be centered around their mean
                if is_eeg:
                    color = eeg_colors[ch_idx % len(eeg_colors)]
                    y_range = 1000.0  # Default range: max - min of original limits
                    y_ticks = [-500, 0, 500]  # Relative to center
                elif is_optics:
                    color = optics_colors[ch_idx % len(optics_colors)]
                    y_range = 0.4
                    y_ticks = [-0.5, 0, 0.5]  # Relative to center
                elif "ACC" in ch_name.upper():
                    color = acc_color
                    y_range = 2.0  # 1 - (-1) = 2.0
                    y_ticks = [-1, 0, 1]  # Relative to center
                else:  # GYRO
                    color = gyro_color
                    y_range = 490.0  # 245 - (-245) = 490
                    y_ticks = [-245, 0, 245]  # Relative to center

                self.channel_info.append(
                    {
                        "stream_idx": stream_idx,
                        "ch_idx": ch_idx,
                        "name": ch_name,
                        "color": color,
                        "y_range": y_range,  # Total vertical span
                        "y_ticks": y_ticks,  # Tick values relative to center
                        "y_scale": 1.0,  # Individual channel group scale
                        "y_mean": 0.0,  # Running mean (updated each frame)
                    }
                )
                self.total_channels += 1

        # Determine number of samples to display
        # Use the highest sampling rate to ensure smooth display
        max_sfreq = max(stream.info["sfreq"] for stream in streams)
        self.n_samples = int(max_sfreq * window_size)

        # Initialize data buffer (n_samples x n_channels)
        self.data = np.zeros((self.n_samples, self.total_channels), dtype=np.float32)

        # Create window title
        if len(streams) == 1:
            window_title = f"OpenMuse - {streams[0].name}"
        else:
            window_title = f"OpenMuse - {self.total_channels} channels"

        # Create canvas
        self.canvas = app.Canvas(
            title=window_title,
            keys="interactive",
            size=(1400, 900),
            position=(100, 100),
        )

        # Create GLOO program for signals
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)

        # Build vertex data (needs self.program to exist)
        self._create_vertex_data()

        # Create grid lines for better readability
        self._create_grid_lines()

        # Set uniforms
        self.program["u_scale"] = (1.0, 1.0)
        self.program["u_size"] = (self.total_channels, 1.0)
        self.program["u_n_samples"] = float(self.n_samples)
        self.program["u_projection"] = ortho(0, 1, 0, 1, -1, 1)

        # Create text labels for channel names (y-axis, left side, right-aligned)
        self.channel_labels = []
        for ch_idx, ch_info in enumerate(self.channel_info):
            text = TextVisual(
                ch_info["name"],
                pos=(10, 0),  # Will be positioned in on_draw
                color="white",
                font_size=10,
                anchor_x="right",  # Right-aligned at left edge
                anchor_y="center",
                bold=True,
            )
            text.transforms.configure(
                canvas=self.canvas, viewport=(0, 0, *self.canvas.size)
            )
            self.channel_labels.append(text)

        # Create EEG standard deviation labels (impedance indicator)
        self.eeg_std_labels = []
        for ch_idx, ch_info in enumerate(self.channel_info):
            # Only create std labels for EEG channels
            stream_name = streams[ch_info["stream_idx"]].name
            if "EEG" in stream_name.upper():
                text = TextVisual(
                    "σ: ---",
                    pos=(0, 0),  # Will be positioned in on_draw
                    color="yellow",
                    font_size=8,
                    anchor_x="right",
                    anchor_y="center",
                )
                text.transforms.configure(
                    canvas=self.canvas, viewport=(0, 0, *self.canvas.size)
                )
                self.eeg_std_labels.append((ch_idx, text))

        # Combined buffer for calculating statistics (last CHANNEL_STATS_BUFFER_DURATION seconds of data)
        # Used for both EEG std (impedance) and running mean (centering all channels)
        self.channel_stats_buffer = {}
        for ch_idx, ch_info in enumerate(self.channel_info):
            sfreq = streams[ch_info["stream_idx"]].info["sfreq"]
            buffer_size = int(sfreq * CHANNEL_STATS_BUFFER_DURATION)
            self.channel_stats_buffer[ch_idx] = np.zeros(buffer_size)

        # Create y-axis tick labels for each channel (right-aligned, close to signal edge)
        # Store base tick values and create/update text labels dynamically
        self.y_tick_labels = []
        for ch_idx, ch_info in enumerate(self.channel_info):
            self.y_tick_labels.append(
                []
            )  # Will be populated in _update_y_tick_labels()

        # Create text labels for time axis (x-axis)
        self.time_labels = []
        n_time_ticks = int(self.window_size) + 1  # One per second
        for i in range(n_time_ticks):
            time_val = -self.window_size + i  # From -window_size to 0
            text = TextVisual(
                f"{time_val:.0f}s",
                pos=(0, 0),  # Will be positioned in on_draw
                color="white",
                font_size=9,
                anchor_x="center",
                anchor_y="top",
            )
            text.transforms.configure(
                canvas=self.canvas, viewport=(0, 0, *self.canvas.size)
            )
            self.time_labels.append((time_val, text))

        # Initialize y-tick labels
        self._update_y_tick_labels(create_new=True)

        # Connect events
        self.canvas.events.draw.connect(self.on_draw)  # type: ignore[attr-defined]
        self.canvas.events.resize.connect(self.on_resize)  # type: ignore[attr-defined]
        self.canvas.events.key_press.connect(self.on_key_press)  # type: ignore[attr-defined]
        self.canvas.events.mouse_wheel.connect(self.on_mouse_wheel)  # type: ignore[attr-defined]

        # Setup timer for updates
        self.timer = app.Timer(self.update_interval, connect=self.on_timer, start=True)

        # Stats tracking
        self.stream_stats = [
            {"first_timestamp": None, "last_timestamp": None, "total_samples": 0}
            for _ in streams
        ]

        if verbose:
            print(f"\nGLOO-based visualization initialized:")
            print(f"  Total channels: {self.total_channels}")
            print(f"  Samples per channel: {self.n_samples}")
            print(f"  Window size: {window_size}s")
            print(f"  Update rate: {1/update_interval:.0f} Hz")
            print("\nPress '+' or '-' to zoom time axis")
            print("Scroll mouse wheel to zoom amplitude")
            print("Close window to stop.\n")

    def _create_vertex_data(self):
        """Create vertex arrays for all channels and samples."""
        # Create index arrays
        # a_index: (channel_idx, sample_idx, 0) for each vertex
        channel_indices = np.repeat(np.arange(self.total_channels), self.n_samples)
        sample_indices = np.tile(np.arange(self.n_samples), self.total_channels)

        a_index = np.c_[
            channel_indices, sample_indices, np.zeros(len(channel_indices))
        ].astype(np.float32)

        # Create color array (each channel has its own color, repeated for all samples)
        colors = np.array([ch["color"] for ch in self.channel_info], dtype=np.float32)
        a_color = np.repeat(colors, self.n_samples, axis=0)

        # Create y_scale array (each channel's scale, repeated for all samples)
        y_scales = np.array(
            [ch["y_scale"] for ch in self.channel_info], dtype=np.float32
        )
        a_y_scale = np.repeat(y_scales, self.n_samples)

        # Set attributes
        self.program["a_index"] = a_index
        self.program["a_color"] = a_color
        self.program["a_y_scale"] = a_y_scale
        # a_position will be updated each frame with actual data
        self.program["a_position"] = self.data.T.ravel().astype(np.float32)

        # Create index buffers for each channel to draw them separately
        self.index_buffers = []
        for ch_idx in range(self.total_channels):
            start = ch_idx * self.n_samples
            end = start + self.n_samples
            indices = np.arange(start, end, dtype=np.uint32)
            index_buffer = gloo.IndexBuffer(indices)
            self.index_buffers.append(index_buffer)

    def _create_grid_lines(self):
        """Create grid lines for horizontal separators and y-axis zero lines."""
        # Simple shader for drawing lines
        grid_vert = """
        attribute vec2 a_position;
        uniform mat4 u_projection;
        void main() {
            gl_Position = u_projection * vec4(a_position, 0.0, 1.0);
        }
        """
        grid_frag = """
        uniform vec4 u_color;
        void main() {
            gl_FragColor = u_color;
        }
        """

        self.grid_program = gloo.Program(grid_vert, grid_frag)
        self.grid_program["u_projection"] = ortho(0, 1, 0, 1, -1, 1)

        # Create horizontal grid lines at y-limits for each channel
        # These correspond to where the min/max ticks are shown
        # Signal uses 35% (0.35) of channel height on each side of center
        # Account for bottom margin (3% reserved for x-axis labels)
        y_bottom_margin = 0.03
        y_usable_height = 1.0 - y_bottom_margin

        y_limit_lines = []
        for ch_idx in range(self.total_channels):
            y_offset = (
                y_bottom_margin + (ch_idx / self.total_channels) * y_usable_height
            )
            channel_height = y_usable_height / self.total_channels
            y_center = y_offset + 0.5 * channel_height

            # Lines at ±35% from center (matching the 0.35 scale in shader)
            y_top = y_center + 0.35 * channel_height  # Upper y-limit
            y_bottom = y_center - 0.35 * channel_height  # Lower y-limit

            y_limit_lines.extend([[0.15, y_top], [1.0, y_top]])
            y_limit_lines.extend([[0.15, y_bottom], [1.0, y_bottom]])

        # Add zero lines for each channel (drawn separately with thicker line)
        self.zero_lines = []
        for ch_idx in range(self.total_channels):
            y_offset = (
                y_bottom_margin + (ch_idx / self.total_channels) * y_usable_height
            )
            y_center = y_offset + 0.5 * (y_usable_height / self.total_channels)
            self.zero_lines.extend([[0.15, y_center], [1.0, y_center]])

        self.y_limit_lines = np.array(y_limit_lines, dtype=np.float32)
        self.zero_lines = np.array(self.zero_lines, dtype=np.float32)

    def on_draw(self, event):
        """Render the scene."""
        gloo.clear(color=(0.1, 0.1, 0.1, 1.0))  # type: ignore[arg-type]

        # Draw y-limit lines (horizontal lines at min/max of each channel)
        if len(self.y_limit_lines) > 0:
            gloo.set_line_width(1.0)
            self.grid_program["a_position"] = self.y_limit_lines
            self.grid_program["u_color"] = (0.25, 0.25, 0.25, 1.0)  # Dark gray
            self.grid_program.draw("lines")

        # Draw zero lines (center of each channel) - thicker and lighter
        gloo.set_line_width(2.0)
        self.grid_program["a_position"] = self.zero_lines
        self.grid_program["u_color"] = (0.35, 0.35, 0.35, 1.0)  # Lighter gray, thicker
        self.grid_program.draw("lines")

        # Reset line width for signal drawing
        gloo.set_line_width(1.0)

        # Draw each channel separately to avoid connecting them
        for index_buffer in self.index_buffers:
            self.program.draw("line_strip", indices=index_buffer)

        # Draw channel labels and y-ticks
        width, height = self.canvas.size
        # Account for bottom margin (3% reserved for x-axis labels)
        y_bottom_margin = 0.03
        y_usable_height = 1.0 - y_bottom_margin

        for ch_idx, (text_visual, ch_info) in enumerate(
            zip(self.channel_labels, self.channel_info)
        ):
            # Calculate channel's vertical bounds with bottom margin
            y_offset = (
                y_bottom_margin + (ch_idx / self.total_channels) * y_usable_height
            )
            channel_height = (y_usable_height / self.total_channels) * height
            y_bottom = height - (y_offset * height)
            y_top = y_bottom - channel_height
            y_center = (y_bottom + y_top) / 2

            # Draw channel name at the left edge (right-aligned)
            # Increased space to accommodate longer names like "OPTICS_LO_NIR"
            text_visual.pos = (120, y_center)
            text_visual.draw()

            # Draw y-tick labels for this channel (right-aligned, close to signal edge)
            # Position ticks to match shader positioning
            # Place at width * 0.15 - 5 pixels (just before signal starts)
            tick_x = width * 0.15 - 5
            y_range = ch_info["y_range"]
            y_scale_zoom = ch_info["y_scale"]  # User's zoom level (from mouse wheel)

            for idx, (tick_val, tick_text) in enumerate(self.y_tick_labels[ch_idx]):
                # tick_val is relative to center (e.g., -500, 0, 500 for EEG)
                # Normalize: data is normalized as 2.0 * (value - mean) / y_range
                # For tick_val relative to center: normalized = 2.0 * tick_val / y_range
                normalized_tick = 2.0 * tick_val / y_range

                # Shader applies: y_center + (normalized * channel_height * 0.35 * y_scale_zoom)
                # So tick position in pixels:
                tick_y_offset = normalized_tick * channel_height * 0.35 * y_scale_zoom
                tick_y = (
                    y_center - tick_y_offset
                )  # Subtract because screen y is flipped

                tick_text.pos = (tick_x, tick_y)
                tick_text.draw()

            # Draw EEG standard deviation (impedance indicator) on the right side
            for eeg_ch_idx, std_text in self.eeg_std_labels:
                if eeg_ch_idx == ch_idx:
                    std_text.pos = (width - 10, y_center)
                    std_text.draw()
                    break

        # Draw time labels (x-axis)
        x_margin = 0.15  # Same as in shader (15% for labels and ticks)
        signal_width = width * (1.0 - x_margin)
        x_start = width * x_margin

        for time_val, text_visual in self.time_labels:
            # Calculate x position (time_val is negative, from -window_size to 0)
            # Map to signal area only (after the left margin)
            x_fraction = (time_val + self.window_size) / self.window_size
            x_pos = x_start + (x_fraction * signal_width)
            # Increased bottom margin to 25px to prevent overlap with bottom channel
            text_visual.pos = (x_pos, height - 25)
            text_visual.draw()

    def on_resize(self, event):
        """Handle window resize."""
        gloo.set_viewport(0, 0, *event.size)

        # Update text transforms
        for text in self.channel_labels:
            text.transforms.configure(canvas=self.canvas, viewport=(0, 0, *event.size))
        for _, text in self.eeg_std_labels:
            text.transforms.configure(canvas=self.canvas, viewport=(0, 0, *event.size))
        for ch_ticks in self.y_tick_labels:
            for _, text in ch_ticks:
                text.transforms.configure(
                    canvas=self.canvas, viewport=(0, 0, *event.size)
                )
        for _, text in self.time_labels:
            text.transforms.configure(canvas=self.canvas, viewport=(0, 0, *event.size))

    def _update_time_labels(self):
        """Update time labels based on current window_size."""
        # Remove old labels
        self.time_labels = []

        # Create new labels
        n_time_ticks = int(self.window_size) + 1  # One per second
        for i in range(n_time_ticks):
            time_val = -self.window_size + i  # From -window_size to 0
            text = TextVisual(
                f"{time_val:.0f}s",
                pos=(0, 0),  # Will be positioned in on_draw
                color="white",
                font_size=9,
                anchor_x="center",
                anchor_y="top",
            )
            text.transforms.configure(
                canvas=self.canvas, viewport=(0, 0, *self.canvas.size)
            )
            self.time_labels.append((time_val, text))

    def _update_y_tick_labels(self, create_new=False):
        """Update y-axis tick labels based on current channel scales.

        Args:
            create_new: If True, create new TextVisual objects. If False, just update text.
        """
        if create_new or not self.y_tick_labels:
            # Create new TextVisual objects (first time or forced)
            self.y_tick_labels = []
            for ch_idx, ch_info in enumerate(self.channel_info):
                tick_texts = []
                y_scale = ch_info["y_scale"]
                y_mean = ch_info["y_mean"]
                base_ticks = ch_info["y_ticks"]

                # Scale tick values based on current zoom and center around mean
                for base_tick in base_ticks:
                    # Tick shows value relative to current mean, scaled by zoom
                    actual_tick = y_mean + (base_tick / y_scale)

                    # Format based on magnitude
                    if abs(actual_tick) >= 10:
                        tick_str = str(int(actual_tick))
                    elif abs(actual_tick) >= 1:
                        tick_str = f"{actual_tick:.1f}"
                    else:
                        tick_str = f"{actual_tick:.2f}"

                    text = TextVisual(
                        tick_str,
                        pos=(0, 0),  # Will be positioned in on_draw
                        color="gray",
                        font_size=7,
                        anchor_x="right",
                        anchor_y="center",
                    )
                    text.transforms.configure(
                        canvas=self.canvas, viewport=(0, 0, *self.canvas.size)
                    )
                    tick_texts.append(
                        (base_tick, text)
                    )  # Store base value for positioning
                self.y_tick_labels.append(tick_texts)
        else:
            # Just update the text content (much faster!)
            for ch_idx, ch_info in enumerate(self.channel_info):
                y_scale = ch_info["y_scale"]
                y_mean = ch_info["y_mean"]
                base_ticks = ch_info["y_ticks"]

                for tick_idx, (base_tick, text) in enumerate(
                    self.y_tick_labels[ch_idx]
                ):
                    # Tick shows value relative to current mean, scaled by zoom
                    # base_tick is relative to center (e.g., -1, 0, 1 for ACC)
                    # Add mean and scale by zoom
                    actual_tick = y_mean + (base_tick / y_scale)

                    # Format based on magnitude
                    if abs(actual_tick) >= 10:
                        tick_str = str(int(actual_tick))
                    elif abs(actual_tick) >= 1:
                        tick_str = f"{actual_tick:.1f}"
                    else:
                        tick_str = f"{actual_tick:.2f}"

                    # Update text content only (no new object creation)
                    text.text = tick_str

    def _get_channel_at_y(self, y_pixel):
        """Get the channel index at a given y pixel coordinate."""
        if self.total_channels == 0:
            return None

        height = self.canvas.size[1]
        # Account for 3% bottom margin
        bottom_margin = height * 0.03
        usable_height = height - bottom_margin

        # Each channel gets equal vertical space
        channel_height = usable_height / self.total_channels

        # Flip y coordinate (canvas is bottom-up, but mouse is top-down)
        y_from_bottom = height - y_pixel

        # Check if in valid range
        if y_from_bottom < bottom_margin or y_from_bottom >= height:
            return None

        # Calculate channel index
        ch_idx = int((y_from_bottom - bottom_margin) / channel_height)
        if 0 <= ch_idx < self.total_channels:
            return ch_idx
        return None

    def on_key_press(self, event):
        """Handle keyboard input for time window zooming."""
        if event.key.name in ["+", "-", "="]:
            # Time window zoom - change window_size (not u_scale which shifts signals)
            if event.key.name in ["+", "="]:
                # Zoom in - show less time (smaller window)
                self.window_size = max(1.0, self.window_size * 0.8)
            else:
                # Zoom out - show more time (larger window)
                self.window_size = min(30.0, self.window_size * 1.25)

            # Regenerate time labels with new window size
            self._update_time_labels()
            self.canvas.update()

    def on_mouse_wheel(self, event):
        """Handle mouse wheel for amplitude zoom (per channel group under mouse)."""
        # event.delta can be a tuple or have different structures depending on platform
        try:
            if hasattr(event.delta, "__getitem__"):
                delta = event.delta[1] if len(event.delta) > 1 else event.delta[0]
            else:
                delta = event.delta
        except (TypeError, IndexError):
            delta = 0

        if delta != 0:
            # Get channel under mouse cursor
            mouse_y = event.pos[1] if hasattr(event, "pos") else None
            target_ch = self._get_channel_at_y(mouse_y) if mouse_y is not None else None

            if target_ch is not None:
                # Find the channel group (channels from same stream with similar type)
                target_info = self.channel_info[target_ch]
                target_name = target_info["name"]

                # Determine group type
                if "EEG" in target_name.upper():
                    group_type = "EEG"
                elif "OPTICS" in target_name.upper():
                    group_type = "OPTICS"
                elif "ACC" in target_name.upper():
                    group_type = "ACC"
                elif "GYRO" in target_name.upper():
                    group_type = "GYRO"
                else:
                    group_type = None

                # Scale all channels in the same group
                dy = np.sign(delta) * 0.1
                for ch_info in self.channel_info:
                    ch_name = ch_info["name"]
                    is_same_group = False

                    if group_type == "EEG" and "EEG" in ch_name.upper():
                        is_same_group = True
                    elif group_type == "OPTICS" and "OPTICS" in ch_name.upper():
                        is_same_group = True
                    elif group_type == "ACC" and "ACC" in ch_name.upper():
                        is_same_group = True
                    elif group_type == "GYRO" and "GYRO" in ch_name.upper():
                        is_same_group = True

                    if is_same_group:
                        # Update this channel's scale
                        current_scale = ch_info["y_scale"]
                        new_scale = max(0.5, min(5.0, current_scale * np.exp(dy)))
                        ch_info["y_scale"] = new_scale

                # Regenerate y-tick labels with new scales
                self._update_y_tick_labels()

                # Update shader attribute with new scales
                self._update_shader_scales()

                self.canvas.update()

    def _update_shader_scales(self):
        """Update the a_y_scale shader attribute with current channel scales."""
        y_scales = np.array(
            [ch["y_scale"] for ch in self.channel_info], dtype=np.float32
        )
        a_y_scale = np.repeat(y_scales, self.n_samples)
        self.program["a_y_scale"] = a_y_scale

    def on_timer(self, event):
        """Update data from streams."""
        # Update each channel
        for ch_info in self.channel_info:
            stream_idx = ch_info["stream_idx"]
            ch_idx = ch_info["ch_idx"]
            stream = self.streams[stream_idx]
            stats = self.stream_stats[stream_idx]

            # Get new data from stream
            try:
                data, timestamps = stream.get_data(winsize=self.window_size)
            except Exception:
                continue

            if data.shape[1] == 0:
                continue

            # Update stats
            if len(timestamps) > 0:
                if stats["first_timestamp"] is None:
                    stats["first_timestamp"] = timestamps[0]
                stats["last_timestamp"] = timestamps[-1]

                time_span = stats["last_timestamp"] - stats["first_timestamp"]
                sfreq = stream.info["sfreq"]
                if sfreq > 0:
                    stats["total_samples"] = int(time_span * sfreq) + 1
                else:
                    stats["total_samples"] = len(timestamps)

            # Get this channel's data
            channel_data = data[ch_idx, :]

            # Update channel statistics buffer (used for both mean and EEG std)
            channel_global_idx = self.channel_info.index(ch_info)
            buffer_size = len(self.channel_stats_buffer[channel_global_idx])
            if len(channel_data) > 0:
                # Take last samples to fill buffer
                samples_to_use = min(buffer_size, len(channel_data))
                self.channel_stats_buffer[channel_global_idx] = np.roll(
                    self.channel_stats_buffer[channel_global_idx], -samples_to_use
                )
                self.channel_stats_buffer[channel_global_idx][-samples_to_use:] = (
                    channel_data[-samples_to_use:]
                )

                # Calculate running mean (for all channels)
                y_mean = np.mean(self.channel_stats_buffer[channel_global_idx])
                ch_info["y_mean"] = y_mean

                # Calculate std for EEG channels (impedance indicator)
                stream_name = self.streams[ch_info["stream_idx"]].name
                if "EEG" in stream_name.upper():
                    std_val = np.std(self.channel_stats_buffer[channel_global_idx])
                    for eeg_ch_idx, std_text in self.eeg_std_labels:
                        if eeg_ch_idx == channel_global_idx:
                            std_text.text = f"σ: {std_val:.1f}"
                            # Color-code based on impedance quality
                            if std_val < 50:
                                std_text.color = "green"  # Good impedance
                            elif std_val > 100:
                                std_text.color = "red"  # Poor impedance
                            else:
                                std_text.color = "yellow"  # Acceptable impedance
                            break

            # Normalize to [-1, 1] range for consistent display
            # Center the signal around its mean, with y_range defining the vertical span
            # Don't clip - let values go beyond limits so user knows when signal exceeds range
            y_range = ch_info["y_range"]
            y_mean = ch_info["y_mean"]

            # Map [y_mean - y_range/2, y_mean + y_range/2] to [-1, 1]
            normalized_data = 2.0 * (channel_data - y_mean) / y_range

            # Resample to fit display buffer if needed
            if len(normalized_data) != self.n_samples:
                # Simple interpolation
                x_old = np.linspace(0, 1, len(normalized_data))
                x_new = np.linspace(0, 1, self.n_samples)
                normalized_data = np.interp(x_new, x_old, normalized_data)

            # Update data buffer for this channel
            channel_global_idx = self.channel_info.index(ch_info)
            self.data[:, channel_global_idx] = normalized_data

        # Update vertex positions
        self.program["a_position"].set_data(self.data.T.ravel().astype(np.float32))

        # Update y-tick labels to reflect new means (only update text, don't recreate)
        self._update_y_tick_labels(create_new=False)

        self.canvas.update()

        # Check duration limit
        if (
            self.duration is not None
            and (time.time() - self.start_time) >= self.duration
        ):
            self.canvas.close()

    def show(self):
        """Show the canvas and start the event loop."""
        self.canvas.show()

        @self.canvas.connect
        def on_close(event):
            self.timer.stop()
            for stream in self.streams:
                stream.disconnect()

            if self.verbose:
                elapsed = time.time() - self.start_time
                print(f"\nVisualization stopped.")
                print(f"  Duration: {elapsed:.1f} seconds")
                for stream_idx, stream in enumerate(self.streams):
                    stats = self.stream_stats[stream_idx]
                    print(f"  {stream.name}: {stats['total_samples']} samples")

                    if (
                        stats["first_timestamp"] is not None
                        and stats["last_timestamp"] is not None
                    ):
                        time_span = stats["last_timestamp"] - stats["first_timestamp"]
                        if time_span > 0 and stats["total_samples"] > 0:
                            avg_rate = (stats["total_samples"] - 1) / time_span
                            print(
                                f"    Mean rate: {avg_rate:.1f} Hz (expected: {stream.info['sfreq']} Hz)"
                            )

        try:
            app.run()
        except KeyboardInterrupt:
            pass


def view(
    stream_name: Optional[str] = None,
    duration: Optional[float] = None,
    window_size: float = 10.0,
    update_interval: float = 0.005,
    verbose: bool = True,
) -> None:
    """
    Visualize EEG and/or ACC/GYRO data from LSL streams in real-time using GLOO (OpenGL).

    This uses direct OpenGL rendering via GLOO for maximum performance with many channels.
    Optimized for 20+ channels (EEG + Motion + Optics).

    By default, displays ALL available Muse streams (EEG + ACCGYRO + Optics if available).

    Supported streams:
    - Muse_EEG: 4 EEG channels (TP9, AF7, AF8, TP10) at 256 Hz
    - Muse_ACCGYRO: 6 motion channels (ACC_X/Y/Z, GYRO_X/Y/Z) at 52 Hz
    - Muse_OPTICS: Up to 16 optical channels at 64 Hz

    Parameters:
    - stream_name: Name of specific LSL stream to connect to (default: None = show all available)
    - duration: Optional viewing duration in seconds. Omit to view until window closed.
    - window_size: Time window to display in seconds (default: 10.0)
    - update_interval: Update interval in seconds (default: 0.005 = 200 Hz)
    - verbose: Print progress messages
    """
    # Configure LSL to reduce verbosity (disables IPv6 warnings and lowers log level)
    configure_lsl_api_cfg()

    from mne_lsl.stream import StreamLSL

    # If no stream specified, try to connect to all available Muse streams
    streams = []
    if stream_name is None:
        # Try all possible Muse streams
        for name in ["Muse_EEG", "Muse_ACCGYRO", "Muse_OPTICS"]:
            if verbose:
                print(f"Looking for LSL stream '{name}'...")
            try:
                stream = StreamLSL(bufsize=window_size, name=name)
                stream.connect(timeout=2.0)
                streams.append(stream)
                if verbose:
                    print(f"  ✓ Connected to {name}")
            except Exception:
                if verbose:
                    print(f"  ✗ {name} not found")

        if len(streams) == 0:
            print("\nError: No Muse streams found!")
            print("Make sure streaming is running in another terminal:")
            print("  OpenMuse stream --address <MAC>")
            return
    else:
        # Connect to specific stream
        if verbose:
            print(f"Looking for LSL stream '{stream_name}'...")
        try:
            stream = StreamLSL(bufsize=window_size, name=stream_name)
            stream.connect(timeout=5.0)
            streams.append(stream)
            if verbose:
                print(f"  ✓ Connected to {stream_name}")
        except Exception as exc:
            print(f"Error: Could not connect to LSL stream '{stream_name}': {exc}")
            print("\nMake sure the stream is running in another terminal:")
            print("  OpenMuse stream --address <MAC> --preset <PRESET>")
            return

    # Create and show viewer
    viewer = RealtimeViewer(
        streams=streams,
        window_size=window_size,
        update_interval=update_interval,
        duration=duration,
        verbose=verbose,
    )
    viewer.show()
