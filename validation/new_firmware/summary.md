================================================================================
MUSE S ATHENA - FIRMWARE COMPATIBILITY SUMMARY
================================================================================
Last Updated: January 6, 2026

DEVICES TESTED:
  00:55:DA:B9:FA:20 = OLD firmware (sends 0x98 battery @ 1 Hz)
  00:55:DA:BB:CD:CD = NEW firmware (sends 0x88 battery @ 0.2 Hz)

================================================================================
ISSUES IDENTIFIED AND RESOLVED
================================================================================

1. BYTE_13 VALIDATION (RESOLVED ✅)
   - OLD firmware: byte_13 always = 0
   - NEW firmware: byte_13 varies 0-40+
   - FIX: Removed byte_13 == 0 validation check in decode.py

2. BATTERY PACKET CHANGE (RESOLVED ✅)
   - OLD firmware: Battery via 0x98 packets @ 1 Hz
   - NEW firmware: Battery via 0x88 packets @ 0.2 Hz (no 0x98 sent)
   - FIX: decode.py extracts battery from both 0x98 and 0x88 packets

3. 0x88 PACKET TYPE (DOCUMENTED)
   - New firmware only, 188-230 bytes, arrives ~every 5-6 seconds
   - First 2 bytes: battery percentage (same encoding as 0x98)
   - Remaining ~200 bytes: unknown (possibly processed sensor data)
   - Does NOT cause data loss from other channels

4. CHANNEL MAPPING VERIFIED ✅ (January 2026)
   - Cross-correlation analysis confirms ALL channels correctly mapped
   - ACCGYRO: All 6 channels have diagonal maximum (correct mapping)
   - EEG: All 4 channels have diagonal maximum (correct mapping)
   - No sign inversions detected in any sensor type

================================================================================
INNER-OUTER AMB CORRELATION - CONTEXT DEPENDENT (NOT A BUG)
================================================================================

UPDATED FINDING: The INNER-OUTER AMB correlation depends on RECORDING CONTEXT,
not firmware version. Previous analysis was misleading.

COMPREHENSIVE XDF ANALYSIS (19 files, 28 OPTICS streams):

  Recording Type              OLD Firmware         NEW Firmware
  ----------------            ----------------     ----------------
  On-head (test1_eeg.xdf)     +0.99 (POSITIVE)     +0.92 (POSITIVE)
  Photosensor/screen          -0.72 to +0.76       -0.97 (NEGATIVE)

STATISTICS BY DEVICE:
  OLD Firmware (9 recordings):
    - POSITIVE correlation: 56% (5/9)
    - NEGATIVE correlation: 44% (4/9)
    - Mean: +0.07, Range: [-0.89, +0.99]

  NEW Firmware (9 recordings):
    - POSITIVE correlation: 11% (1/9)  ← Only on-head recording!
    - NEGATIVE correlation: 89% (8/9)  ← All photosensor recordings
    - Mean: -0.76, Range: [-0.98, +0.92]

KEY INSIGHT:
  - The negative correlation appears in PHOTOSENSOR experiments (screen flashing)
  - BOTH devices show POSITIVE correlation during ON-HEAD recordings
  - This is likely due to how OPTICS sensors respond to direct vs reflected light
  - The "inversion" is NOT a firmware bug or channel mapping error

CONCLUSION:
  ✓ NO FIRMWARE BUG - channel mapping is correct
  ✓ Context-dependent behavior is EXPECTED for optical sensors
  ✓ On-head recordings work correctly for BOTH firmware versions

RECOMMENDATION:
  - For on-head EEG/PPG recordings: No special handling needed
  - For photosensor experiments: Be aware of this optical behavior

================================================================================
DATA INTEGRITY VERIFICATION ✅
================================================================================

All sampling rates verified correct (< ±1% error for EEG/OPTICS):

  Sensor      Expected    Measured (raw .txt)    Measured (XDF)
  --------    --------    -------------------    --------------
  EEG         256 Hz      255.7-256.0 Hz         256.1-256.5 Hz
  OPTICS      64 Hz       64.0 Hz                62.8-64.2 Hz
  ACCGYRO     52 Hz       51.9-52.0 Hz           50.3-52.0 Hz
  BATTERY     1/0.2 Hz    1.0/0.2 Hz             1.0/0.2 Hz

NO DATA LOSS - the 0x88 packet's large size does NOT affect other channels.

================================================================================
PACKET STRUCTURE REFERENCE
================================================================================

14-BYTE PACKET HEADER:
  Byte 0:       pkt_len (total packet length)
  Byte 1:       pkt_index (sequence counter 0-255)
  Bytes 2-5:    pkt_time_raw (32-bit timestamp, 256 kHz clock)
  Bytes 6-8:    unknown1
  Byte 9:       pkt_id (sensor type tag)
  Bytes 10-12:  unknown2
  Byte 13:      unknown (was 0 in old firmware, varies in new)

SENSOR TAGS:
  Tag     Type        Data Len    Rate      Description
  -----   ---------   --------    ------    ---------------------------
  0x11    EEG         28 bytes    256 Hz    4-channel EEG
  0x12    EEG         28 bytes    256 Hz    8-channel EEG
  0x34    OPTICS      30 bytes    64 Hz     4-channel PPG
  0x35    OPTICS      40 bytes    64 Hz     8-channel PPG
  0x36    OPTICS      40 bytes    64 Hz     16-channel PPG
  0x47    ACCGYRO     36 bytes    52 Hz     Accelerometer + Gyroscope
  0x53    Unknown     24 bytes    ?         Old firmware only
  0x88    BATTERY     188-230     0.2 Hz    New firmware battery/status
  0x98    BATTERY     20 bytes    1 Hz      Old firmware battery

================================================================================
TEST DATA FILES
================================================================================

RAW BLE CAPTURES (validation/new_firmware/decoding/):
  device1.txt    Old firmware    61s       Reference
  device2.txt    New firmware    5+ min    Short capture
  device3a.txt   New firmware    35 min    Long recording
  device3b-e.txt New firmware    Various   Additional captures

XDF RECORDINGS (validation/*/):
  test1_adaptive.xdf      Both devices, photosensor experiment
  test1_constrained.xdf   Both devices, photosensor experiment
  test1_robust.xdf        Both devices, photosensor experiment

================================================================================
ANALYSIS SCRIPTS
================================================================================

verification/new_firmware/:
  verify_decoding.py    - Sampling rate verification from raw .txt
  verify_xdf.py         - Sampling rate verification from XDF files

verification/new_firmware/decoding/:
  deep_analysis.py              - Decoder testing, sampling rates
  analyze_0x88_channels.py      - 0x88 packet structure analysis
  explore_unknown_decoding.py   - 0x88 decoding strategies
  correlate_0x88_with_optics.py - 0x88 vs OPTICS correlation
  analyze_firmware.py           - Firmware structure comparison
  visualize_data.py             - Data visualization

verification/new_firmware/channels/:
  validate.py           - Cross-device channel correlation analysis

================================================================================
REMAINING UNKNOWNS
================================================================================

1. 0x88 PACKET CONTENT
   - First 2 bytes = battery (confirmed)
   - Remaining ~200 bytes = unknown (possibly processed OPTICS data)

2. BYTE_13 PURPOSE
   - Varies 0-40+ in new firmware
   - May encode metadata, quality flags, or sequence info

3. 0x53 PACKETS
   - Old firmware only, 24 bytes, purpose unknown

================================================================================
DUAL-DEVICE VALIDATION (January 6, 2026)
================================================================================

Recording: test1_eeg.xdf (channels2/)
Setup: Both devices on same head, different forehead positions
Protocol: 1min rest → 1min blinks → 1min head movements

ACCGYRO CROSS-CORRELATION (validates channel mapping):
  Channel     Correlation     Result
  --------    -----------     ------
  ACC_X       0.83            ✓ Diagonal maximum (correct)
  ACC_Y       0.65            ✓ Diagonal maximum (correct)
  ACC_Z       0.75            ✓ Diagonal maximum (correct)
  GYRO_X      0.40            ✓ Diagonal maximum (correct)
  GYRO_Y      0.54            ✓ Diagonal maximum (correct)
  GYRO_Z      0.33            ✓ Diagonal maximum (correct)

  → All channels show highest correlation on diagonal
  → NO channel swaps or sign inversions detected

EEG VALIDATION:
  - Effective sampling rates: OLD 256.37 Hz, NEW 256.98 Hz (both <0.4% error)
  - Blink detection: Both devices capture blinks (OLD: 10, NEW: 8-9)
  - Cross-correlation of filtered blink signal: 0.70 at 23ms lag
  - Power spectra comparable (differences due to electrode position)

OPTICS VALIDATION:
  - Within-device INNER-OUTER AMB: POSITIVE for BOTH devices (+0.92 to +0.99)
  - Cross-device AMB correlation: +0.59 (ambient light similar at both positions)
  - PPG channels (NIR, IR, RED) don't correlate cross-device (expected - position dependent)

CONCLUSION:
  ✓ Both firmware versions produce equivalent, correctly-mapped data
  ✓ New firmware fully compatible with existing decode.py
  ✓ No special handling required for normal recordings
