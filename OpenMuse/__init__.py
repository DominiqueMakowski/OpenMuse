"""OpenMuse: Minimal utilities for Muse EEG devices."""

from .find import find_devices
from .record import record
from .muse import MuseS
from .decode import parse_message, decode_rawdata
from .stream import stream
from .view import view

__version__ = "0.1.4"
