# Changelog

## [0.1.8]

### Changed
- **Stream Naming Convention**: Updated LSL stream names to include device identifiers for better multi-device support.
  - Muse: `Muse-{sensor_type} ({device_id})` (e.g., `Muse-EEG (0055DA)`)
  - BITalino: `BITalino ({address})` (e.g., `BITalino (20:17:09:18:49:99)`)
- **Clock Synchronization**: Refactored clock synchronization logic into a dedicated `clocks.py` module.
- **CLI**: Added `--clock` argument to `stream` command to select synchronization model.
- **Viewer**: Updated `view` and `view_bitalino` to support fuzzy matching for stream names.
- **Dependencies**: Replaced `pylsl` with `mne_lsl` for stream resolution.

### Fixed
- **Streaming**: Fixed `ValueError` and LSL push errors on newer Muse devices that interleave packets with different channel counts (e.g., 4ch vs 16ch OPTICS). The streamer now dynamically creates separate LSL streams for each channel configuration (e.g., `Muse-OPTICS-16` and `Muse-OPTICS-4`).



## [0.1.2]

### Added

### Changed
- **Stream-Relative Timestamps**: Timestamps are now generated relative to stream start (base_time = 0.0) instead of device boot time, eliminating the need for complex re-anchoring in LSL streaming while maintaining device timing precision.
- **Conditional Re-Anchoring**: LSL streaming now only applies timestamp re-anchoring for edge cases (timestamps >30s in past), providing better synchronization with other LSL streams.
- **Global Timestamping**: Replaced per-message timestamping with global subpacket sorting and timestamping in `decode_rawdata()` to account for cross-message timing inversions.
- **Output Change**: The `parse_message()` function now always returns raw subpackets (Dict[str, List[Dict]]) for flexible processing. Users should call `make_timestamps()` explicitly on the subpackets to get numpy arrays.

### Deprecated

### Removed

### Fixed

### Security

---

## [0.1.0] - 2025-10-16

Initial release.
