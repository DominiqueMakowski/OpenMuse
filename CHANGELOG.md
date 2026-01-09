# Changelog



## [0.1.8]

### Added
- **Hyperscanning Support**: The `stream` command now accepts multiple MAC addresses (space-separated) to stream from multiple Muse devices simultaneously in a single process.
- **Multi-device Recording Support**: The `record` command now accepts multiple MAC addresses (space-separated) to record from multiple Muse devices simultaneously in a single process, similar to the `stream` command. Device addresses are automatically appended to filenames to avoid collisions.


### Changed
- **Default Clock Model**: Changed default clock synchronization from `adaptive` to `windowed`. Validation testing showed `windowed` provides the most stable timing across different devices (based on internal controlled testing). See `clocks.py` docstring for full validation results.
- **Improved Multi-Device Viewer Behavior**: When multiple Muse devices are streaming, the `view` command now automatically displays only the first device and shows a warning message listing all detected devices. Users can specify which device to view using the `--address` argument. This prevents the confusing behavior where all channels from all devices were mixed in a single viewer window.
- **0x88 Packet Support**: Added handling for new 0x88 packet type found in newer firmware. Contains embedded battery info and ~200 bytes of unknown data (possibly processed signals).
- **Viewer Improvements**:
  - Battery display now hidden when battery stream is unavailable (instead of showing "--%%")
  - Channel order now consistent: EEG at top, then ACC/GYRO, then OPTICS at bottom
  - 5-level battery display colour indicator added (as opposed to previous 3-level colour indicator)
  - Movement of battery display to top-left (from top-right)
  - Tick values, time axis, channel labels, and battery indicator now appropriately scale with window size
  - Position of channel labels, tick values, and grid lines altered for readability
- **Stream Naming Convention**: Updated LSL stream names to include device identifiers for better multi-device support.
  - Muse: `Muse-{sensor_type} ({device_id})` (e.g., `Muse-EEG (0055DA)`)
  - BITalino: `BITalino ({address})` (e.g., `BITalino (20:17:09:18:49:99)`)
- **Clock Synchronization**: Refactored clock synchronization logic into a dedicated `clocks.py` module.
- **CLI**: Added `--clock` argument to `stream` command to select synchronization model.
- **View Address Filter**: Added `--address` argument to `view` command to filter streams by MAC address, enabling multi-device setups.
- **Dependencies**: Replaced `pylsl` with `mne_lsl` for stream resolution.



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
