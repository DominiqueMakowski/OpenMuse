"""OpenMuse: Minimal utilities for Muse EEG devices."""

from .decode import decode_rawdata, parse_message
from .find import find_devices
from .muse import MuseS
from .record import record
from .stream import stream
from .view import view

__version__ = "0.1.5"
