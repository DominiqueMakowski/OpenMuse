"""OpenMuse: Minimal utilities for Muse EEG devices."""

from .decode import decode_rawdata, parse_message
from .muse import find_muse
from .muse import MuseS
from .record import record
from .stream import stream
from .view import view
from .drift_correction import correct_timestamps, DriftInfo

__version__ = "0.1.9"
