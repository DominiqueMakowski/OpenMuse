# Changelog


## [0.1.9]

### Fixed
- **Zero-Data-Loss Channel Mismatch Handling**: Fixed rare issue where Muse devices occasionally send packets with different TAG bytes for the same sensor type (e.g., 0x11 EEG4 instead of 0x12 EEG8, or 0x34 OPTICS4 instead of 0x36 OPTICS16). Previously, these packets were skipped entirely, causing minor data loss (~0.001% of packets). The new implementation pads mismatched packets with NaN values for missing channels, preserving all available sensor data while clearly marking unavailable channels. This is a firmware-level glitch that occurs very rarely (observed: 2 anomalous packets in 237,000 during 30-minute recording).

  **Technical investigation findings:**
  - TAG 0x11 (EEG4) contains 4 channels × 4 samples in 28 bytes
  - TAG 0x12 (EEG8) contains 8 channels × 2 samples in 28 bytes
  - Both use the same 28-byte packed format (16 × 14-bit values), but with different channel/sample arrangements
  - The EEG4 format appears to contain only the 4 main EEG electrodes (TP9, AF7, AF8, TP10), without AUX channels
  - Similar pattern exists for OPTICS: 0x34 (4ch), 0x35 (8ch), 0x36 (16ch)
  - NaN padding is the correct approach as it preserves available data without misinterpreting bit layouts
  - TAG byte distribution across 30-minute multi-device recordings:
    - EEG8 (0x12): ~61% of packets
    - OPTICS16 (0x36): ~30% of packets
    - ACCGYRO (0x47): ~8% of packets
    - BATTERY (0x88/0x98): <0.1% of packets
    - EEG4 (0x11): <0.01% of packets (rare firmware quirk)

### Added
- **New Firmware Test Data**: Added test data files from 2026 firmware recordings (`data_new_firmware.txt`, `data_new_firmware_anomalous.txt`) to ensure decoder compatibility with latest hardware.
- **New Firmware Test Suite**: Added `TestNewFirmware` test class covering:
  - 0x88 battery packet decoding
  - 8-channel EEG (0x12) and 16-channel OPTICS (0x36) formats
  - Anomalous 0x11 (EEG4) packet detection and validation
- **Viewer "No Data" Warning**: The viewer now displays a prominent "⚠ NO DATA RECEIVED" warning when no data has been received for 3+ seconds. This makes it immediately obvious when streaming has stopped (previously signals would just go flat).
- **Stream Data Timeout Warning**: The stream command now monitors for data gaps and prints a warning if no data is received for 5+ seconds. This helps diagnose stalled BLE connections without causing false alarms during normal operation.

### Changed
- **Improved Warning Messages**: Channel mismatch warnings now include the TAG byte for easier debugging (e.g., "Padding packet with 4 channels (tag=0x11) to 8 channels (filling with NaN)").
- **Refactored Channel Padding Logic**: Extracted channel mismatch handling into dedicated `_decode_with_channel_padding()` function for better code organization and readability.
- **Simplified stream.py**: Removed verbose exception logging infrastructure to keep the code focused on core streaming logic. Simple print statements are used for the few warnings that may occur during normal operation.


## [0.1.8]

### Added
- **Hyperscanning Support**: The `stream` command now accepts multiple MAC addresses (space-separated) to stream from multiple Muse devices simultaneously in a single process.
- **Multi-device Recording Support**: The `record` command now accepts multiple MAC addresses (space-separated) to record from multiple Muse devices simultaneously in a single process, similar to the `stream` command. Device addresses are automatically appended to filenames to avoid collisions.
- **Stream selection**: Added the --sensors flag to select what sensors to stream. `OpenMuse stream <device> --sensors EEG OPTICS` (should just stream EEG and OPTICS).

### Changed
- **Battery Sampling Rate**: Updated battery sampling rate from 1.0 Hz to 0.2 Hz to reflect actual new firmware behavior (0x88 packets arrive ~every 5 seconds).
- **Documentation**: Updated `decode.py` docstring to accurately describe `pkt_index` behavior - provides 100% correct temporal ordering but indices may have gaps (not strictly sequential).
- **Streaming Stability**: Added channel count validation in `_queue_samples()` to prevent `ValueError: dimension mismatch` crashes when transitioning between different sensor configurations (e.g., 8-channel vs 4-channel EEG modes).
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
  - New header added above plot area containing idiosyncratic device MAC address
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
