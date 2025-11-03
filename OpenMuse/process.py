"""
The Muse S Athena has the following sensors (see decode.py):

ACCGYRO
----------------------------------------------------------
Movement sensors (52 Hz sampling rate):
- ACC_X, ACC_Y, ACC_Z: Accelerometer in g (gravity units), range +-2g
- GYRO_X, GYRO_Y, GYRO_Z: Gyroscope in °/s (degrees per second), range +-250°/s

OPTICS
----------------------------------------------------------
Optics sensors (64 Hz sampling rate):
Placement is on the forehead (roughly centred above the nasion, in-between AF7 and AF8).
The source-detector separation appears to be ~3.0 cm.

- OPTICS_LO_NIR
  - 730nm (NIR), Outer Left
  - Setups: 8-ch, 16-ch
- OPTICS_RO_NIR
  - 730nm (NIR), Outer Right
  - Setups: 8-ch, 16-ch
- OPTICS_LO_IR
  - 850nm (IR), Outer Left
  - Setups: 8-ch, 16-ch
- OPTICS_RO_IR
  - 850nm (IR), Outer Right
  - Setups: 8-ch, 16-ch
- OPTICS_LI_NIR
  - 730nm (NIR), Inner Left
  - Setups: 4-ch, 8-ch, 16-ch
- OPTICS_RI_NIR
  - 730nm (NIR), Inner Right
  - Setups: 4-ch, 8-ch, 16-ch
- OPTICS_LI_IR
  - 850nm (IR), Inner Left
  - Setups: 4-ch, 8-ch, 16-ch
- OPTICS_RI_IR
  - 850nm (IR), Inner Right
  - Setups: 4-ch, 8-ch, 16-ch
- OPTICS_LO_RED
  - 660 nm (RED), Outer Left
  - Setups: 16-ch
- OPTICS_RO_RED
  - 660 nm (RED), Outer Right
  - Setups: 16-ch
- OPTICS_LO_AMB
  - Ambient, Outer Left
  - Setups: 16-ch
- OPTICS_RO_AMB
  - Ambient, Outer Right
  - Setups: 16-ch
- OPTICS_LI_RED
  - 660 nm (RED), Inner Left
  - Setups: 16-ch
- OPTICS_RI_RED
  - 660 nm (RED), Inner Right
  - Setups: 16-ch
- OPTICS_LI_AMB
  - Ambient, Inner Left
  - Setups: 16-ch
- OPTICS_RI_AMB
  - Ambient, Inner Right
  - Setups: 16-ch
"""

import numpy as np
from scipy import signal
import pandas as pd
import warnings


# ===============================================================
# Utilities
# ===============================================================
def apply_butter_filter(
    data: np.ndarray,
    cutoff: float | tuple,
    sampling_rate: float,
    btype: str = "low",
    order: int = 4,
) -> np.ndarray:
    """
    Apply a Butterworth filter using second-order sections (SOS)
    for numerical stability. Supports 1D or 2D arrays.
    """
    data = np.asarray(data)
    nyq = 0.5 * sampling_rate

    if btype == "band":
        if not isinstance(cutoff, (list, tuple)) or len(cutoff) != 2:
            raise ValueError("For 'band' filter, cutoff must be a (low, high) tuple.")
        normal_cutoff = [c / nyq for c in cutoff]
    else:
        normal_cutoff = cutoff / nyq

    sos = signal.butter(order, normal_cutoff, btype=btype, output="sos")

    # Handle 1D and 2D signals
    if data.ndim == 1:
        return signal.sosfiltfilt(sos, data)
    elif data.ndim == 2:
        return np.vstack([signal.sosfiltfilt(sos, d) for d in data])
    else:
        raise ValueError("Data must be 1D or 2D.")


# ===============================================================
# ACCGYRO
# ===============================================================
def preprocess_accgyro(
    df: pd.DataFrame,
    sampling_rate: int = 52,
    lp_cutoff: float = 5.0,
) -> pd.DataFrame:
    """
    Preprocess ACCGYRO data to compute derived features.
    """
    required = ["ACC_X", "ACC_Y", "ACC_Z", "GYRO_X", "GYRO_Y", "GYRO_Z"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing ACCGYRO columns: {missing}")

    # Filter accelerometer and gyroscope signals (vectorised)
    acc = df[["ACC_X", "ACC_Y", "ACC_Z"]].values.T
    gyro = df[["GYRO_X", "GYRO_Y", "GYRO_Z"]].values.T

    acc_f = apply_butter_filter(acc, lp_cutoff, sampling_rate, btype="low")
    gyro_f = apply_butter_filter(gyro, lp_cutoff, sampling_rate, btype="low")

    # Compute features
    acc_mag = np.linalg.norm(acc_f, axis=0)
    gyro_mag = np.linalg.norm(gyro_f, axis=0)
    pitch = np.arctan2(-acc_f[0], np.sqrt(acc_f[1] ** 2 + acc_f[2] ** 2)) * 180 / np.pi
    roll = np.arctan2(acc_f[1], acc_f[2]) * 180 / np.pi

    return pd.DataFrame(
        {
            "ACC_MAG": acc_mag,
            "ACC_PITCH": pitch,
            "ACC_ROLL": roll,
            "GYRO_MAG": gyro_mag,
        }
    )


# ===============================================================
# PPG
# ===============================================================
def _compute_ppg_sqi(
    sig: np.ndarray,
    sampling_rate: int,
    pulse_band: tuple = (0.6, 4.0),
    total_band: tuple = (0.5, 8.0),
) -> float:
    """
    Compute SQI as ratio of pulse-band power to total-band power.
    """
    nperseg = min(len(sig), 4 * sampling_rate)
    noverlap = nperseg // 2

    try:
        f, Pxx = signal.welch(sig, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)
    except ValueError:
        return 0.0

    pulse_mask = (f >= pulse_band[0]) & (f <= pulse_band[1])
    total_mask = (f >= total_band[0]) & (f <= total_band[1])

    power_pulse = np.sum(Pxx[pulse_mask])
    power_total = np.sum(Pxx[total_mask])
    if power_total < 1e-10:
        return 0.0
    return power_pulse / power_total


def preprocess_ppg(
    df: pd.DataFrame,
    sampling_rate: int = 64,
    hp_cutoff: float = 0.5,
    lp_cutoff: float = 8.0,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Preprocess PPG / reflectance channels to yield a fused BVP signal.
    Returns:
      - bvp (np.ndarray)
      - info (dict with SQIs and weights)
    """

    # Mapping
    channel_map = {
        "li_nir": "OPTICS_LI_NIR",
        "ri_nir": "OPTICS_RI_NIR",
        "li_ir": "OPTICS_LI_IR",
        "ri_ir": "OPTICS_RI_IR",
        "lo_nir": "OPTICS_LO_NIR",
        "ro_nir": "OPTICS_RO_NIR",
        "lo_ir": "OPTICS_LO_IR",
        "ro_ir": "OPTICS_RO_IR",
        "li_red": "OPTICS_LI_RED",
        "ri_red": "OPTICS_RI_RED",
        "lo_red": "OPTICS_LO_RED",
        "ro_red": "OPTICS_RO_RED",
        "li_amb": "OPTICS_LI_AMB",
        "ri_amb": "OPTICS_RI_AMB",
        "lo_amb": "OPTICS_LO_AMB",
        "ro_amb": "OPTICS_RO_AMB",
    }

    # Extract channels
    channels = {}
    for internal, col in channel_map.items():
        if col in df.columns:
            channels[internal] = df[col].to_numpy()
        else:
            channels[internal] = None
            if verbose:
                warnings.warn(f"Missing channel: {col}")

    required = ["li_nir", "ri_nir", "li_ir", "ri_ir"]
    n = len(df)
    if any(channels[c] is None for c in required):
        raise ValueError(f"Missing required PPG channels: {required}")

    # Ambient subtraction
    def amb_sub(sig, amb):
        return sig - amb if (sig is not None and amb is not None) else sig

    cleaned = {}
    for key in channel_map:
        if "_amb" in key:
            continue
        loc = key.split("_")[0]
        amb_key = f"{loc}_amb"
        cleaned[f"{key}_c"] = amb_sub(channels[key], channels[amb_key])

    # Band-pass filtering (single step)
    def _filter(sig):
        if sig is None:
            return None
        return apply_butter_filter(
            sig, (hp_cutoff, lp_cutoff), sampling_rate, btype="band"
        )

    filtered = {k.replace("_c", "_f"): _filter(v) for k, v in cleaned.items()}

    # SQI and weighted fusion
    fuse_keys = [
        "li_nir_f",
        "ri_nir_f",
        "li_ir_f",
        "ri_ir_f",
        "lo_nir_f",
        "ro_nir_f",
        "lo_ir_f",
        "ro_ir_f",
    ]
    sqi_scores, weights = {}, {}
    total_signal = np.zeros(n)
    total_weight = 0.0

    for key in fuse_keys:
        sig = filtered.get(key)
        if sig is None:
            sqi_scores[key] = 0.0
            continue

        sqi = _compute_ppg_sqi(
            sig, sampling_rate, pulse_band=(0.6, 4.0), total_band=(hp_cutoff, lp_cutoff)
        )
        sqi = np.clip(sqi, 0, 1)
        sqi_scores[key] = sqi
        weights[key] = sqi**2
        total_signal += sig * weights[key]
        total_weight += weights[key]

    if total_weight < 1e-10:
        bvp = np.full(n, np.nan)
        mean_sqi = 0.0
    else:
        bvp = total_signal / total_weight
        mean_sqi = np.nanmean(list(sqi_scores.values()))

    # Note on polarity: This function returns a BVP signal from reflectance
    # PPG. In reflectance PPG, the systolic peak (max blood volume)
    # corresponds to maximum light absorption, resulting in a *trough*
    # (a minimum value, or "negative peak") in the measured signal.
    # For heart rate estimation, we typically want systolic peaks as maxima.
    bvp = -bvp

    info = {
        "sqi_scores": sqi_scores,
        "weights": weights,
        "mean_sqi": mean_sqi,
        "selected_channels": [k for k, v in weights.items() if v > 0],
    }

    return bvp, info


# ===============================================================
# fNIRS
# ===============================================================


def preprocess_fnirs(
    df: pd.DataFrame,
    sampling_rate: int = 64,
    band_cutoff: tuple = (0.01, 0.1),  # Hemodynamic band
    separation: float = 3.0,  # cm
    dpf: float = 7.0,  # Adult prefrontal
    sqi_window_sec: float = 10.0,  # Window for SQI calculation
    verbose: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    This function converts raw intensity signals from the Muse S Athena's
    forehead optics into continuous time series of hemodynamic changes.

    Args:
        df: Input DataFrame containing raw optics columns (e.g.,
            "OPTICS_LI_NIR", "OPTICS_LI_RED", "OPTICS_LI_AMB").
        sampling_rate: The sampling rate of the optics data (default: 64 Hz).
        band_cutoff: Tuple (low, high) for the hemodynamic bandpass
            filter in Hz. Default (0.01, 0.1) is standard for isolating
            task-related hemodynamic responses.
        separation: The source-detector separation in cm.
            This is critical for the mBLL calculation.
        dpf: The Differential Pathlength Factor, an age- and tissue-dependent
            scalar (default: 7.0, common for adult prefrontal cortex).
        sqi_window_sec: The length of the sliding window (in seconds)
            used to calculate the HbO/HbR anticorrelation SQI. (default: 10.0).
        verbose: If True, prints warnings for missing channels.

    Returns:
        tuple[pd.DataFrame, dict]:
        -   **hb_df (pd.DataFrame):** The primary output. A DataFrame indexed
            identically to the input, containing continuous time series
            of hemodynamic changes. Columns include:
            -   `{pos}_HbO`: Oxygenated hemoglobin (μM).
            -   `{pos}_HbR`: Deoxygenated hemoglobin (μM).
            -   `{pos}_HbDiff`: The Hemoglobin Differential (`HbO - HbR`),
                a robust activation indicator (μM).
            -   `{pos}_SQI`: A Signal Quality Index (-1 to 1) from a
                sliding-window anticorrelation of HbO and HbR.
                High positive values (-> 1) mean good, canonical signal
                (anticorrelated). Negative values (-> -1) indicate
                artifacts (correlated).
        -   **info (dict):** A dictionary containing metadata about the
            processing, including:
            -   "selected_positions" (list): Positions (e.g., "LI", "RO")
                where processing was successful (i.e., had >= 2 wavelengths).
            -   "wavelengths_per_position" (dict): The specific wavelengths
                (e.g., ["730", "850"]) used at each position.
            -   "dpf" (float): The DPF value used.
            -   "separation_cm" (float): The separation value used.

    ---

    ### 🧠 Practical Usage in Cognitive Neuroscience

    The `hb_df` DataFrame is the starting point for statistical analysis.
    It contains `_HbO`, `_HbR`, `_HbDiff`, and `_SQI` signals for each
    processed position.

    **Signal Choice:**
    -   **`_HbO`:** Primary activation indicator (should increase).
    -   **`_HbR`:** Confirmatory indicator (should decrease).
    -   **`_HbDiff`:** Combines `HbO` and `HbR` (`HbO - HbR`), improving signal-to-noise ratio
        and rejecting artifacts where HbO and HbR move together.
    -   **`_SQI`:** Use this to validate your data. You can exclude trials
        or time segments where the `_SQI` value is low or negative,
        as this indicates non-physiological signal contamination.

    **Channel Locations:**
    -   Inner (LI, RI): Medial frontopolar cortex (mPFC). Medial part of BA 10.
    -   Outer (LO, RO): Lateral frontopolar cortex (LPFC). Lateral part of BA 10.

    **1. Block Designs (e.g., 30s Task vs. 30s Rest):**
    * **Epoching:** Slice `hb_df` (e.g., `hb_df['LI_HbDiff']`) into
        "Task" and "Rest" blocks based on your event markers.
    * **Baseline Correction:** For each "Task" epoch, subtract the mean
        of the preceding "Rest" block.
    * **Averaging:** Average all baseline-corrected "Task" epochs to
        get a grand average response.
    * **Statistical Analysis:** Compare the mean activation (e.g.,
        average μM value of `LI_HbDiff` from 5s to 25s post-onset)
        between "Task" and "Rest" conditions.

    **2. Event-Related Designs:**
    * **Epoching:** Slice `hb_df` into short epochs around each
        stimulus (e.g., -2s to +15s).
    * **Baseline Correction:** Subtract the mean of the pre-stimulus
        period (e.g., -2s to 0s) from each epoch.
    * **GLM Analysis (Recommended):** Use the General Linear Model
        (GLM) by convolving your event onsets with a canonical HRF and
        fitting it to your signal (e.g., `LI_HbDiff`). The resulting
        beta-weights represent activation amplitude.

    **Important Considerations:**
    * **Motion Artifacts:** This function does *not* perform motion
        correction. Use `preprocess_accgyro` to identify motion.
        Use the `_SQI` columns to find segments where the signal
        is non-canonical (often due to motion or scalp-blood-flow).
    * **Global Mean Baseline:** Using the *global mean* for $I_0$ is a
        simplification. A dedicated rest-period baseline is more robust.
    """
    # Extinction coefficients (cm⁻¹ M⁻¹)
    eps = {
        "660": {"HbO": 319.6, "HbR": 3226.56},
        "730": {"HbO": 390, "HbR": 1102.2},
        "850": {"HbO": 1058, "HbR": 691.32},
    }

    # Channel mapping (group by position, include red)
    positions = {
        "LO": {
            "660": "OPTICS_LO_RED",
            "730": "OPTICS_LO_NIR",
            "850": "OPTICS_LO_IR",
            "amb": "OPTICS_LO_AMB",
        },
        "RO": {
            "660": "OPTICS_RO_RED",
            "730": "OPTICS_RO_NIR",
            "850": "OPTICS_RO_IR",
            "amb": "OPTICS_RO_AMB",
        },
        "LI": {
            "660": "OPTICS_LI_RED",
            "730": "OPTICS_LI_NIR",
            "850": "OPTICS_LI_IR",
            "amb": "OPTICS_LI_AMB",
        },
        "RI": {
            "660": "OPTICS_RI_RED",
            "730": "OPTICS_RI_NIR",
            "850": "OPTICS_RI_IR",
            "amb": "OPTICS_RI_AMB",
        },
    }

    # Extract and ambient-subtract
    cleaned = {}
    missing = []
    for pos, chans in positions.items():
        amb_col = chans.get("amb")
        amb = df[amb_col].to_numpy() if amb_col in df.columns else None
        for wl, col in {k: v for k, v in chans.items() if k != "amb"}.items():
            if col not in df.columns:
                missing.append(col)
                continue
            sig = df[col].to_numpy()
            cleaned[f"{pos}_{wl}"] = sig - amb if amb is not None else sig

    # Compute ΔOD = -log(I(t) / I0) where I0 = mean(I); filter
    delta_od = {}
    used_wls_per_pos = {}
    for pos in positions:
        wl_list = []
        dod_dict = {}
        for wl in ["660", "730", "850"]:
            sig = cleaned.get(f"{pos}_{wl}")
            if sig is None:
                continue
            # Baseline (full-signal mean; replace with rest-period if available)
            i0 = np.mean(sig)
            # Avoid log(0) or negative
            sig = np.clip(sig, 1e-10, None)
            dod = -np.log(sig / i0)
            # Bandpass filter for hemodynamics
            dod_f = apply_butter_filter(dod, band_cutoff, sampling_rate, btype="band")
            dod_dict[wl] = dod_f
            wl_list.append(wl)

        if len(wl_list) < 2:
            continue

        delta_od[pos] = dod_dict
        used_wls_per_pos[pos] = wl_list

    # Compute ΔHbO and ΔHbR per position using mBLL (least-squares for >=2 wl)
    hb_data = {}
    for pos, dod in delta_od.items():
        wl_list = used_wls_per_pos[pos]
        dod_vec = np.vstack([dod[wl] for wl in wl_list])  # (n_wl, n_time)

        # Epsilon matrix (n_wl x 2: cols=[HbO, HbR])
        eps_matrix = np.array([[eps[wl]["HbO"], eps[wl]["HbR"]] for wl in wl_list])

        # Solve for Δc = lstsq(eps, (dod_vec / (d * DPF))); vectorized
        path = separation * dpf
        if path == 0:
            raise ValueError("Separation and DPF must be positive.")
        scaled_dod = dod_vec / path  # (n_wl, n_time)
        delta_c = np.linalg.lstsq(eps_matrix, scaled_dod, rcond=None)[
            0
        ]  # (2, n_time): [ΔHbO, ΔHbR]

        # Convert to μM (10^6 * mol/L)
        hb_data[f"{pos}_HbO"] = delta_c[0] * 1e6
        hb_data[f"{pos}_HbR"] = delta_c[1] * 1e6

    hb_df = pd.DataFrame(hb_data)

    # --- Post-processing: Calculate HbDiff and SQI ---
    sqi_window_samples = int(sqi_window_sec * sampling_rate)
    # Ensure window is at least 2 samples
    min_periods = max(2, sqi_window_samples // 4)

    # Iterate over 'delta_od.keys()' which only contains positions
    # that were successfully processed (i.e., had >= 2 wavelengths)
    for pos in delta_od.keys():
        hbo_col = f"{pos}_HbO"
        hbr_col = f"{pos}_HbR"

        # 1. Calculate Hemoglobin Differential (HbDiff)
        # (Check columns exist, though they should if pos is in delta_od)
        if hbo_col in hb_df.columns and hbr_col in hb_df.columns:
            hb_df[f"{pos}_HbDiff"] = hb_df[hbo_col] - hb_df[hbr_col]

        # 2. Calculate Sliding-Window Anticorrelation SQI
        if hbo_col in hb_df.columns and hbr_col in hb_df.columns:
            # We calculate -corr(HbO, HbR).
            # High SQI (-> 1) = good (anticorrelated)
            # Low SQI (-> -1) = bad (correlated, artifact)
            sqi = (
                -hb_df[hbo_col]
                .rolling(window=sqi_window_samples, min_periods=min_periods)
                .corr(hb_df[hbr_col])
            )

            hb_df[f"{pos}_SQI"] = sqi

    info = {
        "selected_positions": list(delta_od.keys()),
        "wavelengths_per_position": used_wls_per_pos,
        "dpf": dpf,
        "separation_cm": separation,
    }

    return hb_df, info
