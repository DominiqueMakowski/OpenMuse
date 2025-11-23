"""Compatibility wrapper module for finding Muse devices.

This module re-exports `find_muse` from `OpenMuse.muse` while providing
the `find_devices` name for backwards compatibility.

New code should import `find_muse` from `OpenMuse.muse` or from the
package root (i.e., `from OpenMuse import find_muse`).
"""

from .muse import find_muse

# Backwards-compatible names
find_devices = find_muse
