================================================================================
DUAL DEVICE RECORDING ANALYSIS SUMMARY
================================================================================
Date: January 6, 2026
Recording: test1_eeg.xdf
Duration: ~190 seconds

SETUP:
  - Old Muse (00:55:DA:B9:FA:20): Bottom of forehead (closer to eyebrows)
  - New Muse (00:55:DA:BB:CD:CD): Top of forehead

PROTOCOL:
  - 0-60s: Rest
  - 60-120s: Blinks (10 both eyes, 10 left, 10 right)
  - 120-180s: Head movements in various directions

================================================================================
1. ACCGYRO CORRELATION ANALYSIS
================================================================================

PURPOSE: Validate channel mapping for new firmware by checking if accelerometer
and gyroscope readings correlate between devices (both on same head).

RESULTS:
  Channel       Correlation     Interpretation
  --------      -----------     --------------
  ACC_X         0.8307          ✓ Strong correlation
  ACC_Y         0.6522          ✓ Moderate correlation
  ACC_Z         0.7506          ✓ Good correlation
  GYRO_X        0.3993          ~ Modest correlation
  GYRO_Y        0.5355          ✓ Moderate correlation
  GYRO_Z        0.3307          ~ Modest correlation

  Mean ACC correlation: 0.7445
  Mean GYRO correlation: 0.4218

CONCLUSION:
  ✓ All channels show POSITIVE correlation (no sign inversions)
  ✓ Channel mapping is CORRECT for new firmware
  ✓ Accelerometer shows stronger correlation than gyroscope (expected -
    accelerometer measures orientation, gyroscope measures angular velocity
    which varies more with sensor position)

================================================================================
2. EEG CORRELATION & SPECTRUM ANALYSIS
================================================================================

SAMPLING RATE VERIFICATION:
  Device              Effective SR    Nominal    Error
  ----------------    ------------    -------    -----
  Old firmware        256.37 Hz       256 Hz     +0.15%
  New firmware        256.98 Hz       256 Hz     +0.38%

  → Both devices within spec (<1% error)

EEG CHANNEL CORRELATION (different head positions expected to reduce correlation):
  Channel       Correlation     Note
  --------      -----------     ----
  EEG_TP9       0.3131          Temporal channels, furthest apart
  EEG_AF7       0.4223          Frontal left
  EEG_AF8       0.5434          Frontal right
  EEG_TP10      0.2521          Temporal channels, furthest apart

  → Modest positive correlations expected given different positions

POWER SPECTRUM COMPARISON:
  Channel     Band          Old        New         Ratio
  -------     ----          ---        ---         -----
  EEG_TP9     Delta (1-4)   6.99e+03   2.57e+04    3.67x
              Alpha (8-12)  3.46e+02   1.11e+03    3.21x

  EEG_AF7     Delta (1-4)   3.51e+03   2.37e+03    0.68x
              Alpha (8-12)  7.57e+01   6.76e+01    0.89x

  EEG_AF8     Delta (1-4)   6.15e+03   2.36e+03    0.38x
              Alpha (8-12)  1.43e+02   6.13e+01    0.43x

  EEG_TP10    Delta (1-4)   7.80e+03   2.50e+04    3.21x
              Alpha (8-12)  1.13e+02   9.55e+02    8.43x

  → Temporal channels (TP9, TP10) show HIGHER power in new device
  → Frontal channels (AF7, AF8) show LOWER power in new device
  → This is consistent with different electrode positions on forehead

BLINK DETECTION ANALYSIS:
  Metric                  Old        New
  ------                  ---        ---
  Blinks detected (AF7)   10         8
  Blinks detected (AF8)   10         9

  Cross-correlation of blink signals: 0.70
  Optimal lag: 23.4 ms (new device slightly behind old)

  → Both devices clearly capture blink artifacts
  → Different positions explain slightly different detection counts

================================================================================
3. OPTICS CHANNEL ANALYSIS
================================================================================

CROSS-DEVICE CORRELATION BY WAVELENGTH:
  Wavelength    Mean Correlation    Interpretation
  ----------    ----------------    --------------
  NIR           -0.15               Near zero (expected - different positions)
  IR            -0.06               Near zero (expected)
  RED           -0.25               Weak negative
  AMB           +0.59               ✓ POSITIVE (ambient light correlated)

  → AMB channels show positive correlation because ambient light is similar
    at both forehead positions (same room lighting)
  → PPG-related channels (NIR, IR, RED) don't correlate because they measure
    blood flow which differs by position

WITHIN-DEVICE INNER-OUTER AMB CORRELATION:
  Device              Left IO       Right IO
  ------              -------       --------
  Old firmware        +0.9892       +0.9954
  New firmware        +0.9389       +0.9021

  → BOTH devices show POSITIVE INNER-OUTER correlation
  → This DIFFERS from the summary.txt finding of -0.97 for new firmware
  → Possible explanations:
    1. Different recording conditions (this was on-head, previous may have
       been a photosensor test against screen)
    2. The inversion may be data-dependent or context-specific

WITHIN-DEVICE CORRELATION PATTERNS:
  NIR channels: Both devices show expected patterns
    - OUTER-OUTER (LO-RO): ~0.89-0.90 (high - same position type)
    - INNER-INNER (LI-RI): ~0.97-0.99 (very high)
    - OUTER-INNER: ~0.1-0.4 (lower, different positions)

  IR channels: Interesting NEGATIVE OUTER-INNER correlation
    - Old: LO-LI = -0.35, LO-RI = -0.38
    - New: LO-LI = -0.66, LO-RI = -0.72
    → This is normal PPG behavior (arterial vs venous blood flow)
    → New device shows STRONGER negative correlation

  RED channels: All positive correlations (0.78-0.97)

  AMB channels: High positive correlations (0.72-0.99)
    → Old firmware: very uniform (0.97-0.99)
    → New firmware: more variable (0.72-0.94)

================================================================================
KEY FINDINGS
================================================================================

1. CHANNEL MAPPING VERIFIED ✓
   - ACCGYRO channels are correctly mapped (all positive correlations)
   - EEG channels are correctly mapped (blinks detected, positive correlations)
   - No sign inversions detected in this recording

2. SAMPLING RATES VERIFIED ✓
   - EEG: Both devices within 0.4% of nominal 256 Hz
   - ACCGYRO: Old 51.97 Hz, New 50.47 Hz (both near 52 Hz nominal)
   - OPTICS: Old 64.19 Hz, New 62.66 Hz (both near 64 Hz nominal)

3. OPTICS AMB INVERSION - INCONCLUSIVE
   - This recording shows POSITIVE INNER-OUTER correlation for BOTH devices
   - Previous findings (summary.txt) showed -0.97 for new firmware
   - The inversion may be context-dependent or require specific test conditions

4. POWER SPECTRUM DIFFERENCES
   - Different electrode positions cause different power profiles
   - Not indicative of firmware differences, just physical placement

================================================================================
FILES GENERATED
================================================================================

analyze_dual_device.py outputs:
  - eeg_power_spectrum_comparison.png
  - optics_amb_comparison.png
  - accgyro_comparison.png
  - eeg_timeseries_comparison.png

analyze_optics_detailed.py outputs:
  - optics_all_channels_comparison.png
  - optics_timevarying_correlation.png
  - optics_scatter_plots.png

analyze_blinks.py outputs:
  - eeg_blink_analysis.png
  - eeg_blink_examples.png
  - eeg_crosscorrelation.png

================================================================================
RECOMMENDATIONS
================================================================================

1. For OPTICS AMB inversion investigation:
   - Need to repeat the photosensor test (screen flashing) that originally
     showed the -0.97 correlation
   - The effect may only appear under specific lighting conditions

2. For further validation:
   - The current recording validates basic functionality
   - Both devices successfully record EEG, ACCGYRO, and OPTICS
   - Channel mappings are consistent between firmware versions

3. For on-head recordings:
   - No special handling needed for new firmware
   - Both devices produce comparable data quality
