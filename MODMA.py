"""
EEG Klein-fusion + PLV + Logistic Regression pipeline -- MODMA adaptation
==========================================================================
Adapted from the kinnarhalder/eeg-dataset (19ch EDF) pipeline to run on the
MODMA 128-channel resting-state dataset
(EEG_128channels_resting_lanzhou_2015, .mat files), restricted to the 25
EGI HydroCel channels:

    E68, E69, E70, E72, E65, E95, E61, E76, E86, E111, E66, E81, E116, E103,
    E97, E82, E91, E83, E30, E77, E59, E52, E112, E127, E105

ONLY sections 1-2 (data indexing + loading) were rewritten. Sections 3-11
(feature extraction, Klein hyperbolic fusion, pipeline, nested CV, metrics,
persistence, plots, main) are unchanged from the EDF version other than
parameterizing on N_CHANNELS=25 and swapping the loader call.

>>> BEFORE YOUR FIRST FULL RUN, VERIFY THESE THREE ASSUMPTIONS <<<
--------------------------------------------------------------------
These could not be confirmed without the actual files, so the code
auto-detects where it can and falls back to a documented default
otherwise. Run `inspect_modma_mat_file()` on one sample .mat file, and
open the subjects-information .xlsx once, before trusting a full run:

1. .mat variable + channel order: `_load_mat_eeg_array()` picks the first
   2D array with 128 or 129 rows/cols as the EEG data, and assumes
   channels follow the standard GSN-HydroCel-129 net order E1..E128
   (+ Cz as channel 129 if present). If this dataset was exported with a
   different variable name or channel order, fix `_egi_channel_order()`.
2. Sampling rate: auto-detected from a field named srate/fs/sfreq/
   sampling_rate/samplerate in the .mat file if present, else falls back
   to `CFG.SFREQ_ORIG` (default 250 Hz, the rate commonly reported for
   this Lanzhou 128-channel resting recording -- confirm before relying
   on it).
3. Diagnosis labels: `_load_subject_labels()` scans the columns of
   `subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx` for a
   mostly-numeric subject-ID column and a column whose values match
   `CFG.MDD_LABEL_VALUES` / `CFG.HC_LABEL_VALUES`. It prints which columns
   it picked -- check that output. If auto-detection fails or mismatches,
   fill `CFG.MANUAL_LABEL_MAP = {"02010005": 0, "02030007": 1, ...}` by
   hand instead.
"""
import os
import re
import sys
import glob
import json
import time
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import mne
import pywt
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.stats import skew, kurtosis
from scipy.signal import hilbert
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                              recall_score, roc_auc_score, roc_curve, classification_report,
                              ConfusionMatrixDisplay, fowlkes_mallows_score)

try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
@dataclass
class CONFIG:
    DATA_ROOT: str = "/kaggle/input/datasets/kinnarhalder/modmaa/EEG_128channels_resting_lanzhou_2015"
    SUBJECT_INFO_XLSX: str = ""   # blank -> auto-find the single .xlsx under DATA_ROOT
    SFREQ_TARGET: int = 128
    SFREQ_ORIG: float = 250.0     # fallback ONLY if no srate/fs field is found in the .mat -- VERIFY
    BANDS: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 45.0),
    })
    BANDS_TO_RUN: Tuple[str, ...] = field(default_factory=lambda: (
        "delta",
    ))
    SCALES_SEC: Tuple[int, int, int] = (1, 2, 3)
    # 25 EGI HydroCel channels requested for the MODMA run.
    SELECTED_CHANNELS: Tuple[str, ...] = field(default_factory=lambda: (
        "E68", "E69", "E70", "E72", "E65", "E95", "E61", "E76", "E86", "E111",
        "E66", "E81", "E116", "E103", "E97", "E82", "E91", "E83", "E30", "E77",
        "E59", "E52", "E112", "E127", "E105",
    ))
    N_CHANNELS: int = 25
    WAVELET: str = "db4"
    WAVELET_LEVEL: int = 3
    INCLUDE_PLV: bool = True
    KLEIN_CURVATURE_GRID: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    KLEIN_SCALE_FACTOR_GRID: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    SATURATION_MARGIN: float = 0.95
    SELECT_K_GRID: Tuple = (30, 60, 120, "all")
    N_OUTER_FOLDS: int = 5
    N_INNER_FOLDS: int = 3
    N_REPEATS: int = 3
    N_BAGGING_ESTIMATORS: int = 31
    BAGGING_MAX_SAMPLES: float = 0.9
    SEED: int = 42
    SAVE_DIR: str = "results_modma"
    # values (case-insensitive) treated as "MDD" / "Healthy control" when
    # scanning the subject-info spreadsheet -- extend if the file uses
    # different wording (e.g. "patient" / "depressed" / "case").
    MDD_LABEL_VALUES: Tuple[str, ...] = ("mdd", "depression", "patient", "case", "1")
    HC_LABEL_VALUES: Tuple[str, ...] = ("hc", "h", "healthy", "control", "normal", "0")
    MANUAL_LABEL_MAP: Dict[str, int] = field(default_factory=dict)  # {"02010005": 0, ...}


CFG = CONFIG()
np.random.seed(CFG.SEED)


# --------------------------------------------------------------------------- #
# 0. Diagnostics -- run these by hand once before a full pipeline run
# --------------------------------------------------------------------------- #
def inspect_modma_mat_file(path: str) -> None:
    """Prints every variable name / shape / dtype in one .mat file so you can
    confirm the EEG-data variable and sample-rate field before trusting the
    auto-detection in `_load_mat_eeg_array()`. Not called automatically."""
    print(f"Inspecting {path}")
    try:
        mat = sio.loadmat(path)
        print("  format: MATLAB <= v7.2 (loaded via scipy.io.loadmat)")
        for k, v in mat.items():
            if k.startswith("__"):
                continue
            v = np.asarray(v)
            print(f"    {k!r:24s} shape={v.shape} dtype={v.dtype}")
    except NotImplementedError:
        if not _HAS_H5PY:
            print("  format: MATLAB v7.3 (HDF5) -- install h5py to inspect it (`pip install h5py`).")
            return
        with h5py.File(path, "r") as f:
            print("  format: MATLAB v7.3 (loaded via h5py)")
            def _walk(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"    {name!r:24s} shape={obj.shape} dtype={obj.dtype}")
            f.visititems(_walk)


def inspect_modma_subject_info(cfg: CONFIG) -> None:
    """Prints the columns + first few rows of the subject-info spreadsheet so
    you can confirm/override CFG.MDD_LABEL_VALUES / HC_LABEL_VALUES or supply
    CFG.MANUAL_LABEL_MAP. Not called automatically."""
    xlsx_path = cfg.SUBJECT_INFO_XLSX or next(iter(glob.glob(os.path.join(cfg.DATA_ROOT, "*.xlsx"))), None)
    if not xlsx_path:
        print(f"No .xlsx found under {cfg.DATA_ROOT}")
        return
    df = pd.read_excel(xlsx_path)
    print(f"Subject-info file: {xlsx_path}")
    print(f"Columns: {list(df.columns)}")
    print(df.head(10).to_string())


# --------------------------------------------------------------------------- #
# 1. Data indexing / labeling (MODMA: .mat files + subject-info spreadsheet)
# --------------------------------------------------------------------------- #
def _normalize_id(x) -> str:
    return re.sub(r"\D", "", str(x))


def _load_subject_labels(xlsx_path: str, cfg: CONFIG) -> Dict[str, int]:
    """Auto-detects the subject-ID column and the diagnosis column in the
    subjects-information spreadsheet, returns {normalized_subject_id: label}
    with label 1=MDD, 0=Healthy. Prints its column choices -- verify these
    against inspect_modma_subject_info() output."""
    df = pd.read_excel(xlsx_path)
    print(f"[subject-info] columns in {os.path.basename(xlsx_path)}: {list(df.columns)}")

    id_col, label_col = None, None
    mdd_vals = {v.lower() for v in cfg.MDD_LABEL_VALUES}
    hc_vals = {v.lower() for v in cfg.HC_LABEL_VALUES}
    for col in df.columns:
        vals = df[col].dropna().astype(str)
        if len(vals) == 0:
            continue
        if id_col is None and vals.str.match(r"^\d{4,8}$").mean() > 0.7:
            id_col = col
        lower_vals = vals.str.strip().str.lower()
        hits = lower_vals.isin(mdd_vals | hc_vals)
        if label_col is None and hits.mean() > 0.7:
            label_col = col

    if id_col is None or label_col is None:
        raise ValueError(
            f"Could not auto-detect subject-id / diagnosis columns in {xlsx_path} "
            f"(id_col={id_col}, label_col={label_col}). Run inspect_modma_subject_info(CFG), "
            f"then either widen CFG.MDD_LABEL_VALUES/HC_LABEL_VALUES, or skip auto-detection "
            f"entirely by filling CFG.MANUAL_LABEL_MAP by hand."
        )
    print(f"[subject-info] using id_col={id_col!r}, label_col={label_col!r}")

    label_map = {}
    for _, row in df.iterrows():
        if pd.isna(row[id_col]) or pd.isna(row[label_col]):
            continue
        sid = _normalize_id(row[id_col])
        lab = str(row[label_col]).strip().lower()
        if not sid:
            continue
        if lab in mdd_vals:
            label_map[sid] = 1
        elif lab in hc_vals:
            label_map[sid] = 0
    return label_map


def index_dataset_modma(cfg: CONFIG) -> List[Dict]:
    """Scan DATA_ROOT for .mat recordings and match each to an MDD/Healthy
    label via the subject-info spreadsheet (or CFG.MANUAL_LABEL_MAP)."""
    label_map = {}
    if not cfg.MANUAL_LABEL_MAP or True:  # still build the spreadsheet map as a fallback source
        xlsx_path = cfg.SUBJECT_INFO_XLSX
        if not xlsx_path:
            candidates = glob.glob(os.path.join(cfg.DATA_ROOT, "*.xlsx"))
            if candidates:
                xlsx_path = candidates[0]
        if xlsx_path:
            try:
                label_map = _load_subject_labels(xlsx_path, cfg)
            except ValueError as e:
                if not cfg.MANUAL_LABEL_MAP:
                    raise
                print(f"[WARN] {e}\nFalling back entirely to CFG.MANUAL_LABEL_MAP.")

    records, unmatched = [], []
    for path in sorted(glob.glob(os.path.join(cfg.DATA_ROOT, "*.mat"))):
        fname = os.path.basename(path)
        m = re.match(r"^(\d{6,8})", fname)
        if not m:
            continue
        file_id = m.group(1)
        norm_id = _normalize_id(file_id)

        if file_id in cfg.MANUAL_LABEL_MAP:
            label = cfg.MANUAL_LABEL_MAP[file_id]
        elif norm_id in label_map:
            label = label_map[norm_id]
        else:
            # fallback: suffix match, in case the spreadsheet ids are a
            # different length/padding than the filename ids
            match = next((v for k, v in label_map.items()
                          if k and (k.endswith(norm_id) or norm_id.endswith(k))), None)
            if match is None:
                unmatched.append(fname)
                continue
            label = match

        records.append({"path": path, "subject_id": file_id, "label": label})

    if unmatched:
        preview = unmatched[:10]
        print(f"[WARN] {len(unmatched)} .mat file(s) had no matching label and were skipped: "
              f"{preview}{' ...' if len(unmatched) > 10 else ''}")
    if not records:
        raise RuntimeError(
            "No MODMA .mat files could be matched to a diagnosis label. Run "
            "inspect_modma_subject_info(CFG) and check CFG.SUBJECT_INFO_XLSX / "
            "MDD_LABEL_VALUES / HC_LABEL_VALUES, or fill CFG.MANUAL_LABEL_MAP."
        )
    n_mdd = sum(r["label"] == 1 for r in records)
    print(f"Indexed {len(records)} MODMA recordings -> {n_mdd} MDD / {len(records) - n_mdd} Healthy")
    return records


# --------------------------------------------------------------------------- #
# 2. Signal loading + multi-scale windowing (MODMA: .mat, 25 selected channels)
# --------------------------------------------------------------------------- #
CHANNEL_ORDER_128 = [f"E{i}" for i in range(1, 129)]


def _egi_channel_order(n_channels_in_file: int) -> List[str]:
    """Standard GSN-HydroCel-129 net channel numbering. ASSUMPTION -- verify
    with inspect_modma_mat_file() that the .mat channel axis actually follows
    E1..E128 (+Cz) order; if not, fix this function."""
    if n_channels_in_file == 128:
        return CHANNEL_ORDER_128
    elif n_channels_in_file == 129:
        return CHANNEL_ORDER_128 + ["Cz"]
    raise ValueError(
        f"Unexpected channel count {n_channels_in_file} -- expected 128 (E1..E128) or "
        f"129 (E1..E128 + Cz). Run inspect_modma_mat_file() on this file and adjust "
        f"_egi_channel_order() if this dataset uses a non-standard layout."
    )


def _load_mat_eeg_array(path: str, cfg: CONFIG) -> Tuple[np.ndarray, float]:
    """Load (n_channels_total, n_samples) + sampling rate from one MODMA .mat
    file. Auto-detects the data variable as the first 2D array with a 128 or
    129 dimension, and the sample rate from a srate/fs/sfreq/sampling_rate/
    samplerate field if present (else cfg.SFREQ_ORIG)."""
    is_h5 = False
    try:
        mat = sio.loadmat(path)
    except NotImplementedError:
        if not _HAS_H5PY:
            raise RuntimeError(f"{path} is MATLAB v7.3 (HDF5) format -- `pip install h5py` to read it.")
        mat = h5py.File(path, "r")
        is_h5 = True

    def _to_np(v):
        return np.array(v) if is_h5 else np.asarray(v)

    keys = [k for k in mat.keys() if not k.startswith("__")]
    data_arr = None
    for k in keys:
        arr = _to_np(mat[k])
        if arr.ndim == 2 and (128 in arr.shape or 129 in arr.shape):
            data_arr = arr
            break
    if data_arr is None:
        shapes = {k: _to_np(mat[k]).shape for k in keys}
        if is_h5:
            mat.close()
        raise ValueError(
            f"No 128/129-channel 2D array found in {path}. Keys/shapes: {shapes}. "
            f"Run inspect_modma_mat_file({path!r}) and adjust _load_mat_eeg_array()."
        )
    if data_arr.shape[0] not in (128, 129):
        data_arr = data_arr.T  # ensure channels-first

    sfreq = None
    for k in keys:
        if k.lower() in ("srate", "fs", "sfreq", "sampling_rate", "samplerate"):
            sfreq = float(np.asarray(_to_np(mat[k])).squeeze())
            break
    if sfreq is None:
        sfreq = cfg.SFREQ_ORIG  # fallback -- verify against inspect_modma_mat_file()

    if is_h5:
        mat.close()
    return data_arr.astype(np.float64), sfreq


def load_mat_band(path: str, cfg: CONFIG, band: Tuple[float, float]) -> np.ndarray:
    data_full, sfreq_orig = _load_mat_eeg_array(path, cfg)
    ch_order = _egi_channel_order(data_full.shape[0])
    try:
        sel_idx = [ch_order.index(ch) for ch in cfg.SELECTED_CHANNELS]
    except ValueError as e:
        raise ValueError(f"selected channel not found in detected {len(ch_order)}-ch EGI layout: {e}")
    data_sel = data_full[sel_idx, :]

    info = mne.create_info(ch_names=list(cfg.SELECTED_CHANNELS), sfreq=sfreq_orig, ch_types="eeg")
    raw = mne.io.RawArray(data_sel, info, verbose="ERROR")
    raw.filter(band[0], band[1], verbose="ERROR")
    if raw.info["sfreq"] != cfg.SFREQ_TARGET:
        raw.resample(cfg.SFREQ_TARGET, verbose="ERROR")
    return raw.get_data()


def partition_signal(signal: np.ndarray, sfreq: int, window_sec: int) -> List[np.ndarray]:
    win_len = int(window_sec * sfreq)
    n_windows = signal.shape[1] // win_len
    return [signal[:, i * win_len:(i + 1) * win_len] for i in range(n_windows)]


# --------------------------------------------------------------------------- #
# 3. Feature extraction: wavelet sub-band statistics + Hjorth parameters
#    (unchanged from the EDF pipeline)
# --------------------------------------------------------------------------- #
def hjorth_parameters(x: np.ndarray) -> Tuple[float, float, float]:
    eps = 1e-12
    activity = np.var(x) + eps
    d1 = np.diff(x)
    d2 = np.diff(d1)
    var_d1 = np.var(d1) + eps
    var_d2 = np.var(d2) + eps
    mobility = np.sqrt(var_d1 / activity)
    mobility_d1 = np.sqrt(var_d2 / var_d1)
    complexity = mobility_d1 / (mobility + eps)
    return float(activity), float(mobility), float(complexity)


def wavelet_subband_stats(coeff: np.ndarray) -> Tuple[float, float, float, float]:
    eps = 1e-12
    energy = np.sum(coeff ** 2) / max(len(coeff), 1)
    log_energy = np.log1p(energy)
    std = np.std(coeff)
    sk = skew(coeff) if len(coeff) > 2 and std > eps else 0.0
    ku = kurtosis(coeff) if len(coeff) > 2 and std > eps else 0.0
    return float(log_energy), float(std), float(sk), float(ku)


def wavelet_window_features(window: np.ndarray, wavelet: str, level: int) -> np.ndarray:
    feats = []
    for ch in range(window.shape[0]):
        sig = window[ch]
        feats.extend(hjorth_parameters(sig))
        coeffs = pywt.wavedec(sig, wavelet=wavelet, level=level)
        for c in coeffs:
            feats.extend(wavelet_subband_stats(c))
    feats = np.array(feats, dtype=np.float64)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------- #
# 3b. PLV (Phase Locking Value) connectivity features (unchanged)
# --------------------------------------------------------------------------- #
def instantaneous_phase(window: np.ndarray) -> np.ndarray:
    analytic = hilbert(window, axis=-1)
    return np.angle(analytic)


def plv_matrix(window: np.ndarray) -> np.ndarray:
    phase = instantaneous_phase(window)
    phase_diff = phase[:, None, :] - phase[None, :, :]
    plv = np.abs(np.mean(np.exp(1j * phase_diff), axis=-1))
    return plv


def plv_window_features(window: np.ndarray) -> np.ndarray:
    n_ch = window.shape[0]
    plv = plv_matrix(window)
    iu = np.triu_indices(n_ch, k=1)
    feats = plv[iu].astype(np.float64)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def full_window_features(window: np.ndarray, wavelet: str, level: int, include_plv: bool) -> np.ndarray:
    feats = wavelet_window_features(window, wavelet, level)
    if include_plv:
        feats = np.concatenate([feats, plv_window_features(window)])
    return feats


def scale_feature_vector(signal: np.ndarray, sfreq: int, window_sec: int, cfg: CONFIG) -> np.ndarray:
    windows = partition_signal(signal, sfreq, window_sec)
    if not windows:
        raise ValueError(f"recording too short for a {window_sec}s window")
    feats = np.stack([
        full_window_features(w, cfg.WAVELET, cfg.WAVELET_LEVEL, cfg.INCLUDE_PLV)
        for w in windows
    ])
    return feats.mean(axis=0)


def build_dataset_for_band(records: List[Dict], cfg: CONFIG, band: Tuple[float, float]
                            ) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray]:
    X_by_scale = {s: [] for s in cfg.SCALES_SEC}
    y, groups = [], []
    iterator = tqdm(records, desc="Extracting features", unit="subj") if _HAS_TQDM else records
    for rec in iterator:
        try:
            signal = load_mat_band(rec["path"], cfg, band=band)
            per_scale = {s: scale_feature_vector(signal, cfg.SFREQ_TARGET, s, cfg)
                         for s in cfg.SCALES_SEC}
        except Exception as exc:
            print(f"[WARN] skipping {rec['path']}: {exc}")
            continue
        for s in cfg.SCALES_SEC:
            X_by_scale[s].append(per_scale[s])
        y.append(rec["label"])
        groups.append(rec["subject_id"])

    X_by_scale = {s: np.stack(v) for s, v in X_by_scale.items()}
    return X_by_scale, np.array(y), np.array(groups)


# --------------------------------------------------------------------------- #
# 4. Klein (Beltrami-Klein) hyperbolic model operations (unchanged)
# --------------------------------------------------------------------------- #
class KleinOps:
    """Beltrami-Klein disk operations for curvature -c (c > 0)."""

    def __init__(self, c: float, eps: float = 1e-3):
        self.c = c
        self.eps = eps

    def _max_norm(self) -> float:
        return (1.0 / np.sqrt(self.c)) * (1 - self.eps)

    def _clamp_norm(self, x: np.ndarray) -> np.ndarray:
        max_norm = self._max_norm()
        norm = np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-15, None)
        scale = np.where(norm > max_norm, max_norm / norm, 1.0)
        return x * scale

    def exp_map0(self, u: np.ndarray) -> np.ndarray:
        sqrt_c = np.sqrt(self.c)
        norm = np.clip(np.linalg.norm(u, axis=-1, keepdims=True), 1e-15, None)
        gain = np.tanh(sqrt_c * norm) / (sqrt_c * norm)
        return self._clamp_norm(gain * u)

    def log_map0(self, x: np.ndarray) -> np.ndarray:
        sqrt_c = np.sqrt(self.c)
        max_norm = self._max_norm()
        norm = np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-15, max_norm)
        margin = max(self.eps, 1e-3)
        arg = np.clip(sqrt_c * norm, None, 1 - margin)
        scale = np.arctanh(arg) / (sqrt_c * norm)
        return scale * x

    def lorentz_factor(self, x: np.ndarray) -> np.ndarray:
        sq_norm = np.sum(x * x, axis=-1, keepdims=True)
        return 1.0 / np.sqrt(np.clip(1 - self.c * sq_norm, self.eps, None))

    def einstein_midpoint(self, points: List[np.ndarray], weights: List[float] = None) -> np.ndarray:
        if weights is None:
            weights = [1.0] * len(points)
        num = np.zeros_like(points[0])
        den = np.zeros((points[0].shape[0], 1))
        for w, x in zip(weights, points):
            gamma = self.lorentz_factor(x)
            num = num + w * gamma * x
            den = den + w * gamma
        midpoint = num / np.clip(den, 1e-15, None)
        return self._clamp_norm(midpoint)

    def saturation_rate(self, x: np.ndarray, margin: float = 0.95) -> float:
        norm = np.linalg.norm(x, axis=-1)
        return float(np.mean(norm > margin * self._max_norm()))


class KleinMultiScaleFusion(BaseEstimator, TransformerMixin):
    def __init__(self, n_scales: int = 3, curvature: float = 1.0,
                 scale_factor: float = 1.0, saturation_margin: float = 0.95):
        self.n_scales = n_scales
        self.curvature = curvature
        self.scale_factor = scale_factor
        self.saturation_margin = saturation_margin

    def fit(self, X: np.ndarray, y=None):
        n_feat_total = X.shape[1]
        assert n_feat_total % self.n_scales == 0
        self.block_size_ = n_feat_total // self.n_scales
        self.scalers_ = []
        for i in range(self.n_scales):
            block = X[:, i * self.block_size_:(i + 1) * self.block_size_]
            self.scalers_.append(StandardScaler().fit(block))
        self.klein_ = KleinOps(c=self.curvature)
        self.pre_scale_ = self.scale_factor / np.sqrt(self.block_size_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        klein_points = []
        for i in range(self.n_scales):
            block = X[:, i * self.block_size_:(i + 1) * self.block_size_]
            block = self.scalers_[i].transform(block)
            block = self.pre_scale_ * block
            klein_points.append(self.klein_.exp_map0(block))
        fused = self.klein_.einstein_midpoint(klein_points)
        self.saturation_rate_ = self.klein_.saturation_rate(fused, self.saturation_margin)
        return self.klein_.log_map0(fused)


# --------------------------------------------------------------------------- #
# 5. Pipeline construction (Logistic Regression only, unchanged)
# --------------------------------------------------------------------------- #
def param_grid(cfg: CONFIG) -> Dict:
    return {
        "klein_fusion__curvature": list(cfg.KLEIN_CURVATURE_GRID),
        "klein_fusion__scale_factor": list(cfg.KLEIN_SCALE_FACTOR_GRID),
        "select__k": list(cfg.SELECT_K_GRID),
        "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"],
    }


def make_pipeline(n_scales: int, cfg: CONFIG) -> Pipeline:
    return Pipeline([
        ("klein_fusion", KleinMultiScaleFusion(n_scales=n_scales, saturation_margin=cfg.SATURATION_MARGIN)),
        ("select", SelectKBest(score_func=mutual_info_classif)),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=3000)),
    ])


# --------------------------------------------------------------------------- #
# 6. Metrics (unchanged)
# --------------------------------------------------------------------------- #
METRIC_KEYS = ["accuracy", "precision", "recall_sensitivity", "specificity", "roc_auc", "fowlkes_mallows"]

METRIC_DISPLAY = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall_sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "roc_auc": "ROC-AUC",
    "fowlkes_mallows": "Fowlkes-Mallows",
}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray = None) -> Dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "fowlkes_mallows": fowlkes_mallows_score(y_true, y_pred),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    youden_j = tpr - fpr
    best_idx = int(np.argmax(youden_j))
    return float(np.clip(thresholds[best_idx], 0.0, 1.0))


def make_bagging_classifier(base_pipe: Pipeline, cfg: CONFIG, seed: int) -> BaggingClassifier:
    kwargs = dict(n_estimators=cfg.N_BAGGING_ESTIMATORS, max_samples=cfg.BAGGING_MAX_SAMPLES,
                   bootstrap=True, random_state=seed, n_jobs=-1)
    try:
        return BaggingClassifier(estimator=base_pipe, **kwargs)
    except TypeError:
        return BaggingClassifier(base_estimator=base_pipe, **kwargs)


# --------------------------------------------------------------------------- #
# 7. Nested (outer + inner) cross-validation for one band (unchanged)
# --------------------------------------------------------------------------- #
def run_one_outer_fold(band_name: str, X_dev: np.ndarray, y_dev: np.ndarray, groups_dev: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        cfg: CONFIG, seed: int, fold_label: str):
    inner_cv = StratifiedGroupKFold(n_splits=cfg.N_INNER_FOLDS, shuffle=True, random_state=seed)
    grid = param_grid(cfg)

    pipe = make_pipeline(n_scales=len(cfg.SCALES_SEC), cfg=cfg)
    search = GridSearchCV(pipe, grid, cv=inner_cv, scoring="roc_auc", n_jobs=-1, refit=False)
    search.fit(X_dev, y_dev, groups=groups_dev)

    fixed_pipe = make_pipeline(n_scales=len(cfg.SCALES_SEC), cfg=cfg)
    fixed_pipe.set_params(**search.best_params_)

    diag_pipe = clone(fixed_pipe)
    diag_pipe.fit(X_dev, y_dev)
    saturation_rate = diag_pipe.named_steps["klein_fusion"].saturation_rate_

    bagged = make_bagging_classifier(fixed_pipe, cfg, seed)

    oof_proba = cross_val_predict(bagged, X_dev, y_dev, groups=groups_dev,
                                   cv=inner_cv, method="predict_proba", n_jobs=-1)[:, 1]
    best_threshold = find_best_threshold(y_dev, oof_proba)

    bagged.fit(X_dev, y_dev)
    y_score = bagged.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= best_threshold).astype(int)

    fold_metrics = compute_metrics(y_test, y_pred, y_score)
    fold_metrics["decision_threshold"] = best_threshold
    fold_metrics["saturation_rate"] = saturation_rate
    print(f"[{band_name}] {fold_label} | best_params={search.best_params_} | "
          f"threshold={best_threshold:.3f} | acc={fold_metrics['accuracy']:.3f} | "
          f"sens={fold_metrics['recall_sensitivity']:.3f} | spec={fold_metrics['specificity']:.3f} | "
          f"auc={fold_metrics['roc_auc']:.3f} | fm={fold_metrics['fowlkes_mallows']:.3f} | "
          f"klein_saturation={saturation_rate:.3f}")
    return fold_metrics, y_test, y_pred, y_score


def run_nested_cv_for_band(band_name: str, X_full: np.ndarray, y: np.ndarray, groups: np.ndarray,
                            cfg: CONFIG):
    all_fold_metrics = []
    pooled_y_true, pooled_y_pred, pooled_y_score = [], [], []

    total_folds = cfg.N_REPEATS * cfg.N_OUTER_FOLDS
    pbar = tqdm(total=total_folds, desc=f"{band_name} nested CV", unit="fold") if _HAS_TQDM else None
    t0 = time.time()

    for repeat in range(cfg.N_REPEATS):
        seed = cfg.SEED + repeat
        outer_cv = StratifiedGroupKFold(n_splits=cfg.N_OUTER_FOLDS, shuffle=True, random_state=seed)

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_full, y, groups), start=1):
            X_dev, X_test = X_full[train_idx], X_full[test_idx]
            y_dev, y_test = y[train_idx], y[test_idx]
            groups_dev = groups[train_idx]

            fold_label = f"Repeat {repeat + 1}/{cfg.N_REPEATS} | Fold {fold_idx}/{cfg.N_OUTER_FOLDS}"
            fold_metrics, y_t, y_p, y_s = run_one_outer_fold(
                band_name, X_dev, y_dev, groups_dev, X_test, y_test, cfg, seed, fold_label
            )
            all_fold_metrics.append(fold_metrics)
            pooled_y_true.extend(y_t.tolist())
            pooled_y_pred.extend(y_p.tolist())
            pooled_y_score.extend(y_s.tolist())

            if pbar is not None:
                elapsed = time.time() - t0
                done = len(all_fold_metrics)
                eta = elapsed / done * (total_folds - done) if done else 0.0
                pbar.set_postfix(elapsed=f"{elapsed/60:.1f}m", eta=f"{eta/60:.1f}m")
                pbar.update(1)
            else:
                elapsed = time.time() - t0
                done = len(all_fold_metrics)
                eta = elapsed / done * (total_folds - done) if done else 0.0
                pct = 100.0 * done / total_folds
                print(f"[{band_name}] completion: {done}/{total_folds} ({pct:.1f}%) | "
                      f"elapsed={elapsed/60:.1f}m | eta={eta/60:.1f}m")

    if pbar is not None:
        pbar.close()

    pooled_y_true = np.array(pooled_y_true)
    pooled_y_pred = np.array(pooled_y_pred)
    pooled_y_score = np.array(pooled_y_score)
    pooled = compute_metrics(pooled_y_true, pooled_y_pred, pooled_y_score)

    summary = {}
    for key in METRIC_KEYS:
        vals = [m[key] for m in all_fold_metrics]
        summary[f"{key}_mean"] = float(np.nanmean(vals))
        summary[f"{key}_std"] = float(np.nanstd(vals))
    sat_vals = [m["saturation_rate"] for m in all_fold_metrics]
    summary["saturation_rate_mean"] = float(np.mean(sat_vals))
    summary["saturation_rate_std"] = float(np.std(sat_vals))

    return pooled, summary, all_fold_metrics, (pooled_y_true, pooled_y_pred, pooled_y_score)


# --------------------------------------------------------------------------- #
# 8. Persistence (unchanged)
# --------------------------------------------------------------------------- #
def _band_result_dir(cfg: CONFIG, band_name: str) -> str:
    return os.path.join(cfg.SAVE_DIR, band_name.lower())


def save_band_result(cfg: CONFIG, band_name: str, summary: Dict,
                      pooled_arrays: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    band_dir = _band_result_dir(cfg, band_name)
    os.makedirs(band_dir, exist_ok=True)
    with open(os.path.join(band_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    y_true, y_pred, y_score = pooled_arrays
    np.savez(os.path.join(band_dir, "pooled.npz"), y_true=y_true, y_pred=y_pred, y_score=y_score)
    print(f"[SAVED] {band_name} results -> {band_dir}/ (summary.json, pooled.npz)")


def load_all_saved_band_results(cfg: CONFIG) -> Tuple[Dict[str, Dict], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    summaries, pooled_scores = {}, {}
    if not os.path.isdir(cfg.SAVE_DIR):
        return summaries, pooled_scores
    for band_key in cfg.BANDS:
        band_name = band_key.capitalize()
        band_dir = _band_result_dir(cfg, band_name)
        summary_path = os.path.join(band_dir, "summary.json")
        pooled_path = os.path.join(band_dir, "pooled.npz")
        if os.path.isfile(summary_path) and os.path.isfile(pooled_path):
            with open(summary_path) as f:
                summaries[band_name] = json.load(f)
            npz = np.load(pooled_path)
            pooled_scores[band_name] = (npz["y_true"], npz["y_score"])
    return summaries, pooled_scores


# --------------------------------------------------------------------------- #
# 9. Cross-band comparison table (unchanged)
# --------------------------------------------------------------------------- #
def summary_to_row(band_name: str, summary: Dict) -> Dict:
    row = {"Band": band_name}
    for key in METRIC_KEYS:
        mean, std = summary[f"{key}_mean"], summary[f"{key}_std"]
        row[METRIC_DISPLAY[key]] = f"{mean:.3f} \u00b1 {std:.3f}"
    row["Klein Saturation"] = f"{summary['saturation_rate_mean']:.3f} \u00b1 {summary['saturation_rate_std']:.3f}"
    return row


def build_comparison_table(all_summaries: Dict[str, Dict]) -> pd.DataFrame:
    rows = [summary_to_row(name, summary) for name, summary in all_summaries.items()]
    cols = ["Band"] + [METRIC_DISPLAY[k] for k in METRIC_KEYS] + ["Klein Saturation"]
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- #
# 10. Inline-only plots (unchanged)
# --------------------------------------------------------------------------- #
def show_confusion_matrix(band_name: str, y_true: np.ndarray, y_pred: np.ndarray):
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Healthy", "MDD"], cmap="Blues", ax=ax
    )
    ax.set_title(f"{band_name} — pooled outer-fold predictions (Logistic Regression)")
    plt.tight_layout()
    plt.show()


def show_roc_overlay(pooled_scores_by_band: Dict[str, Tuple[np.ndarray, np.ndarray]]):
    fig, ax = plt.subplots(figsize=(5, 5))
    for band_name, (y_true, y_score) in pooled_scores_by_band.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, label=f"{band_name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Pooled outer-fold ROC curves — band comparison (Logistic Regression)")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
# 11. Main
# --------------------------------------------------------------------------- #
def _resolve_bands_to_run(cfg: CONFIG) -> List[str]:
    all_band_keys = list(cfg.BANDS.keys())

    raw_arg = sys.argv[1] if len(sys.argv) > 1 else None
    cli_band = raw_arg.lower() if raw_arg is not None else None
    if cli_band is not None and cli_band in cfg.BANDS:
        return [cli_band]
    elif cli_band is not None and not cli_band.startswith("-"):
        raise ValueError(f"Unknown band '{raw_arg}'. Choose one of {all_band_keys}.")

    bands_to_run = cfg.BANDS_TO_RUN
    if isinstance(bands_to_run, str):
        bands_to_run = (bands_to_run,)

    if not bands_to_run:
        raise ValueError(f"CFG.BANDS_TO_RUN is empty -- list at least one of {all_band_keys}.")
    unknown = [b for b in bands_to_run if b not in cfg.BANDS]
    if unknown:
        raise ValueError(f"Unknown band(s) {unknown} in CFG.BANDS_TO_RUN. Choose from {all_band_keys}.")
    return list(bands_to_run)


def main():
    print("=" * 100)
    print("MODMA run -- verify assumptions before trusting results:")
    print("  1. .mat channel order (E1..E128[+Cz])  -> inspect_modma_mat_file(sample_path)")
    print("  2. sampling rate (falls back to CFG.SFREQ_ORIG if not found in the .mat)")
    print("  3. subject-info label columns           -> inspect_modma_subject_info(CFG)")
    print("=" * 100 + "\n")

    records = index_dataset_modma(CFG)

    bands_to_run = _resolve_bands_to_run(CFG)
    print(f"\nBands to run THIS execution: {bands_to_run}")
    print(f"Channels ({CFG.N_CHANNELS}): {list(CFG.SELECTED_CHANNELS)}")
    print(f"Classifier: Logistic Regression only")
    print(f"Running {CFG.N_REPEATS}x nested CV per band: "
          f"{CFG.N_OUTER_FOLDS} outer folds x {CFG.N_INNER_FOLDS} inner folds "
          f"= {CFG.N_REPEATS * CFG.N_OUTER_FOLDS} outer evaluations per band\n")

    for band_key in bands_to_run:
        band_range = CFG.BANDS[band_key]
        band_name = band_key.capitalize()

        print(f"\n{'=' * 20} {band_name} ({band_range[0]}-{band_range[1]} Hz) {'=' * 20}")
        X_by_scale, y, groups = build_dataset_for_band(records, CFG, band=band_range)
        print(f"Usable recordings: {len(y)} "
              f"(feature dim per scale: {X_by_scale[CFG.SCALES_SEC[0]].shape[1]}, "
              f"PLV included: {CFG.INCLUDE_PLV})")
        X_full = np.concatenate([X_by_scale[s] for s in CFG.SCALES_SEC], axis=1)

        pooled, summary, fold_results, pooled_arrays = run_nested_cv_for_band(
            band_name, X_full, y, groups, CFG
        )
        pooled_y_true, pooled_y_pred, pooled_y_score = pooled_arrays

        print(f"\n--- Mean +/- std across {len(fold_results)} outer-fold evaluations ({band_name}) ---")
        for key in METRIC_KEYS:
            print(f"{key}: {summary[f'{key}_mean']:.3f} +/- {summary[f'{key}_std']:.3f}")
        print(f"klein_saturation_rate: {summary['saturation_rate_mean']:.3f} "
              f"+/- {summary['saturation_rate_std']:.3f}")

        print(f"\n--- Pooled classification report ({band_name}) ---")
        print(classification_report(pooled_y_true, pooled_y_pred, target_names=["Healthy", "MDD"]))

        save_band_result(CFG, band_name, summary, pooled_arrays)
        show_confusion_matrix(band_name, pooled_y_true, pooled_y_pred)

    all_summaries, pooled_scores_by_band = load_all_saved_band_results(CFG)
    if not all_summaries:
        print("\nNo band results found on disk yet.")
        return None

    if len(pooled_scores_by_band) >= 1:
        show_roc_overlay(pooled_scores_by_band)

    comparison_df = build_comparison_table(all_summaries)
    print("\n" + "=" * 100)
    print(f"BAND COMPARISON (Logistic Regression) — {len(all_summaries)}/5 bands completed so far")
    print("=" * 100)
    try:
        from tabulate import tabulate
        print(tabulate(comparison_df, headers="keys", tablefmt="github", showindex=False))
    except ImportError:
        print(comparison_df.to_string(index=False))

    os.makedirs(CFG.SAVE_DIR, exist_ok=True)
    csv_path = os.path.join(CFG.SAVE_DIR, "band_comparison_logreg.csv")
    comparison_df.to_csv(csv_path, index=False)
    print(f"\nSaved comparison table to {os.path.abspath(csv_path)}")

    if len(all_summaries) < len(CFG.BANDS):
        remaining = [b.capitalize() for b in CFG.BANDS if b.capitalize() not in all_summaries]
        print(f"\nStill missing: {remaining}. Run again with e.g. "
              f"`python {os.path.basename(__file__)} {remaining[0].lower()}` to fill in the rest.")
    else:
        ranked = sorted(all_summaries.items(), key=lambda kv: kv[1]["roc_auc_mean"], reverse=True)
        print("\nAll 5 bands complete. Ranking by mean ROC-AUC:")
        for name, summary in ranked:
            print(f"  {name:<10} AUC={summary['roc_auc_mean']:.3f} +/- {summary['roc_auc_std']:.3f} | "
                  f"FM={summary['fowlkes_mallows_mean']:.3f} +/- {summary['fowlkes_mallows_std']:.3f} | "
                  f"KleinSat={summary['saturation_rate_mean']:.3f}")

    return comparison_df, all_summaries


if __name__ == "__main__":
    main()
