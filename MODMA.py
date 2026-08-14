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
from scipy.io import loadmat
from scipy.stats import skew, kurtosis
from scipy.signal import hilbert
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover -- fallback if tqdm isn't installed
    def tqdm(iterable, **kwargs):
        total = kwargs.get("total", None)
        desc = kwargs.get("desc", "")
        if desc:
            print(desc)
        return iterable
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                              recall_score, roc_auc_score, roc_curve, classification_report,
                              ConfusionMatrixDisplay, fowlkes_mallows_score)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# WHAT CHANGED IN THIS FILE vs. the previous version (leakage fix)
# =============================================================================
# The old select_channels_fast() ran the ANOVA F-test channel ranking ONCE,
# globally, on ALL 53 subjects, BEFORE any outer-CV split existed. Every
# outer-test fold's held-out subjects therefore contributed their own labels
# to the channel ranking that then got used to compute that same fold's
# "held-out" performance -- textbook feature-selection leakage (the docstring
# even flagged this as a caveat, but it wasn't actually fixed).
#
# Fix: channel selection is now refit per OUTER fold, using only that fold's
# dev/train subjects' bandpower + labels (see select_channels_for_fold()).
# The outer-test subjects never influence which channels are kept for the
# fold that scores them. This means:
#   - Channel selection moved from a one-time pre-processing step (section 1b)
#     to inside run_nested_cv_for_band()'s outer-fold loop (section 7).
#   - Wavelet/PLV feature extraction (which depends on which channels were
#     selected) also moved inside the outer-fold loop, since the channel set
#     now varies per fold instead of being fixed once for the whole band.
#   - To avoid re-running MNE filtering/resampling per fold (the actually
#     expensive I/O step), the band-filtered 128-channel raw signal for every
#     subject is loaded ONCE per band and cached in memory
#     (load_band_signal_cache()); each fold just slices + windows the cached
#     signal for its own selected channels. This is also strictly cheaper
#     than the old code, which reloaded raw signals separately in
#     select_channels_fast() and again in build_dataset_for_band().
#   - Selected channels are printed per fold before that fold's pipeline
#     (Klein fusion -> SelectKBest -> Logistic Regression, bagged) is fit,
#     per your request.
#
# Everything else -- KleinOps / KleinMultiScaleFusion, the sklearn Pipeline,
# the nested inner-CV GridSearchCV + BaggingClassifier + Youden's-J threshold
# logic inside run_one_outer_fold(), metrics, persistence, plotting -- is
# UNCHANGED from the previous version.
#
# Residual (much smaller) leakage note: channel selection is refit per OUTER
# fold, not per INNER fold. So within a given outer fold, the channel ranking
# is informed by all dev subjects, including the ones that land in inner-CV
# validation splits during hyperparameter search. That only affects which
# hyperparameters GridSearchCV picks, not the outer-test metrics that are
# actually reported/compared across bands. Fixing that too would mean
# refitting channel selection inside every inner split as well -- a bigger
# restructure than a channel-selection fix implies, so it's left as-is here.
# =============================================================================

USE_BIDS_EDF = False   # flip to True if your MODMA copy is the BIDS .edf release


@dataclass
class CONFIG:
    DATA_ROOT: str = "/kaggle/input/datasets/kinnarhalder/modmaa/EEG_128channels_resting_lanzhou_2015"
    SFREQ_TARGET: int = 128
    MODMA_NATIVE_SFREQ: int = 250          # MODMA HCGSN acquisition rate
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
    N_CHANNELS: int = 128                                  # raw MODMA channel count (before selection)
    N_CHANNELS_SELECT: int = 19                             # channels kept after fast selection, see select_channels_for_fold()
    WAVELET: str = "db4"
    WAVELET_LEVEL: int = 3
    INCLUDE_PLV: bool = True                               # see runtime note above -- set False to cut ~8k feats/scale
    KLEIN_CURVATURE_GRID: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    KLEIN_SCALE_FACTOR_GRID: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    SATURATION_MARGIN: float = 0.95
    SELECT_K_GRID: Tuple = (30, 60, 120, 240)              # dropped "all" -- selecting "all" of ~10k feats is expensive and rarely wins
    N_OUTER_FOLDS: int = 3                                 # revisit if a class has <~10 subjects (see note above)
    N_INNER_FOLDS: int = 3
    N_REPEATS: int = 3
    N_BAGGING_ESTIMATORS: int = 31
    BAGGING_MAX_SAMPLES: float = 0.9
    SEED: int = 42
    SAVE_DIR: str = "results_modma128"


CFG = CONFIG()
np.random.seed(CFG.SEED)


def _fmt_secs(seconds: float) -> str:
    """Human-readable duration, e.g. '2m 14s' or '48.3s'."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s" if h else f"{m}m {s}s"


# --------------------------------------------------------------------------- #
# 1. Data indexing / labeling -- UNCHANGED
# --------------------------------------------------------------------------- #
def _load_label_map_from_xlsx(root: str) -> Optional[Dict[str, int]]:
    candidates = sorted(glob.glob(os.path.join(root, "*resting*.xlsx"))) or \
                 sorted(glob.glob(os.path.join(root, "*.xlsx")))
    if not candidates:
        print("[WARN] no subjects_information .xlsx found next to the data -- "
              "falling back to filename-prefix labeling (0201=MDD, else=NC).")
        return None
    xlsx_path = candidates[0]
    try:
        df = pd.read_excel(xlsx_path)
    except Exception as exc:
        print(f"[WARN] could not read {xlsx_path} ({exc}) -- "
              f"falling back to filename-prefix labeling (0201=MDD, else=NC).")
        return None

    df.columns = [str(c).strip().lower() for c in df.columns]
    id_col = next((c for c in df.columns if any(k in c for k in ("subject", "id", "code"))), None)
    type_col = next((c for c in df.columns if any(k in c for k in ("type", "group", "label", "diagn"))), None)
    if id_col is None or type_col is None:
        print(f"[WARN] {xlsx_path} columns {list(df.columns)} didn't match an "
              f"expected subject-id / group column -- inspect the sheet manually "
              f"and adjust id_col/type_col matching above. Falling back to "
              f"filename-prefix labeling (0201=MDD, else=NC).")
        return None

    label_map = {}
    for _, row in df.iterrows():
        digits = re.sub(r"\D", "", str(row[id_col]))
        if not digits:
            continue
        type_val = str(row[type_col]).strip().upper()
        label = 1 if any(k in type_val for k in ("MDD", "DEPRES", "PATIENT")) else 0
        label_map[digits] = label
    if not label_map:
        print(f"[WARN] parsed 0 usable rows from {xlsx_path} -- "
              f"falling back to filename-prefix labeling (0201=MDD, else=NC).")
        return None
    print(f"Loaded {len(label_map)} subject labels from {os.path.basename(xlsx_path)} "
          f"(id_col='{id_col}', type_col='{type_col}')")
    return label_map


def index_dataset(root: str) -> List[Dict]:
    """Scan DATA_ROOT and return [{path, subject_id, label}, ...]."""
    pattern = "*.edf" if USE_BIDS_EDF else "*.mat"
    label_map = None if USE_BIDS_EDF else _load_label_map_from_xlsx(root)

    records = []
    n_fallback = 0
    for path in sorted(glob.glob(os.path.join(root, pattern))):
        fname = os.path.basename(path)
        m = re.match(r"(\d{8})", fname)
        if not m:
            continue
        subject_id = m.group(1)
        prefix = subject_id[:4]

        if label_map is not None and subject_id in label_map:
            label = label_map[subject_id]
        else:
            n_fallback += 1
            label = 1 if prefix == "0201" else 0   # 1 = MDD, 0 = NC/healthy

        records.append({"path": path, "subject_id": subject_id, "label": label})

    if not records:
        raise RuntimeError(
            f"No {pattern} files matched under {root} (looked for an 8-digit "
            f"subject code at the start of each filename)."
        )
    if n_fallback:
        print(f"[WARN] {n_fallback}/{len(records)} subjects labeled via filename-prefix "
              f"fallback rather than the xlsx metadata -- double-check these.")
    n_mdd = sum(r["label"] for r in records)
    prefixes_seen = sorted(set(r["subject_id"][:4] for r in records))
    print(f"Indexed {len(records)} MODMA recordings: {n_mdd} MDD, {len(records) - n_mdd} NC "
          f"(prefixes seen: {prefixes_seen})")
    return records


# --------------------------------------------------------------------------- #
# 1b. Fast, LEAKAGE-FREE channel selection (ANOVA F-test on per-channel
#     log-bandpower), refit per outer CV fold.
# --------------------------------------------------------------------------- #
# Two-step split, matching the original method's cost profile:
#   (a) load_band_signal_cache() loads + band-filters the 128-channel signal
#       for every subject ONCE per band (this is the expensive MNE step) and
#       compute_bandpower_matrix() turns that into one cheap (n_subj, 128)
#       log-variance bandpower matrix, shared across all outer folds/repeats.
#   (b) select_channels_for_fold() runs the actual ANOVA F-test / ranking,
#       but ONLY on the bandpower rows belonging to that fold's dev (train)
#       subjects. Outer-test subjects' bandpower/labels never enter the
#       ranking that selects the channels used to score them -- this is the
#       fix for the leakage in the original global select_channels_fast().
# --------------------------------------------------------------------------- #
def load_band_signal_cache(records: List[Dict], cfg: CONFIG, band: Tuple[float, float]) -> Dict[str, np.ndarray]:
    """Load + band-filter the raw (128, n_samples) signal for every record
    ONCE for this band, keyed by file path. Reused for both the bandpower
    matrix (channel ranking) and the wavelet/PLV feature extraction, so each
    subject's MNE filtering/resampling only runs once per band instead of
    twice (as it did in the previous global-selection version)."""
    cache = {}
    t0 = time.time()
    pbar = tqdm(records, desc="Loading+filtering raw signals", unit="subj")
    for rec in pbar:
        try:
            cache[rec["path"]] = load_raw_band(rec["path"], cfg, band=band)
        except Exception as exc:
            print(f"[WARN] could not load {rec['path']}: {exc}")
        pbar.set_postfix_str(rec["subject_id"])
    print(f"[TIMING] signal cache load: {_fmt_secs(time.time() - t0)} "
          f"for {len(cache)}/{len(records)} subjects")
    return cache


def compute_bandpower_matrix(records: List[Dict], signal_cache: Dict[str, np.ndarray], cfg: CONFIG
                              ) -> Tuple[List[Dict], np.ndarray, np.ndarray, np.ndarray]:
    """(n_subjects, N_CHANNELS) log-variance bandpower proxy per channel, plus
    the matching (kept_records, y, groups) -- subjects whose signal failed to
    load are dropped here and everywhere downstream for this band."""
    kept_records, X_bp = [], []
    for rec in records:
        sig = signal_cache.get(rec["path"])
        if sig is None:
            continue
        X_bp.append(np.log1p(np.var(sig, axis=1)))
        kept_records.append(rec)
    X_bp = np.stack(X_bp)
    y = np.array([r["label"] for r in kept_records])
    groups = np.array([r["subject_id"] for r in kept_records])
    return kept_records, X_bp, y, groups


def select_channels_for_fold(X_bp_train: np.ndarray, y_train: np.ndarray, cfg: CONFIG,
                              band: Tuple[float, float], n_select: int, fold_label: str = "") -> np.ndarray:
    """ANOVA F-test channel ranking, fit ONLY on this outer fold's dev/train
    subjects (see the module-level note above for why)."""
    ch_names = [f"E{i + 1}" for i in range(cfg.N_CHANNELS)]
    f_scores, p_values = f_classif(X_bp_train, y_train)
    f_scores = np.nan_to_num(f_scores, nan=0.0)
    ranked = np.argsort(f_scores)[::-1]
    selected = np.sort(ranked[:n_select])   # ascending order for consistent downstream indexing

    print(f"\nChannel selection {fold_label}(ANOVA F-test on log-bandpower, "
          f"band={band[0]}-{band[1]} Hz, n_select={n_select}, "
          f"fit on {len(y_train)} dev-fold subjects only):")
    for rank, idx in enumerate(ranked[:n_select], start=1):
        print(f"  {rank:2d}. {ch_names[idx]:>5s}  F={f_scores[idx]:8.3f}  p={p_values[idx]:.2e}")
    print(f"Selected channels (ascending): {[ch_names[i] for i in selected]}\n")
    return selected


# --------------------------------------------------------------------------- #
# 2. Signal loading + multi-scale windowing (MODMA .mat -> MNE RawArray)
#    -- UNCHANGED
# --------------------------------------------------------------------------- #
def _load_modma_mat_matrix(path: str, cfg: CONFIG) -> np.ndarray:
    """Load one MODMA resting-state .mat file and return a (128, n_samples)
    float64 array in volts, dropping the trailing Cz reference channel.

    MODMA's own documentation says EEGLAB loads a MATLAB struct where
    `EEG.data` is (129, n_samples): rows E1..E128 then Cz. Some re-uploads
    (including third-party Kaggle mirrors) instead store the raw matrix
    directly under a plain variable name -- sometimes the subject code
    itself -- rather than nested in an "EEG" struct. This checks the
    documented "EEG.data" path first, then falls back to scanning every
    top-level variable in the .mat for one shaped like a 128/129-channel
    recording, and raises with the actual keys/shapes found if nothing
    matches so you can tell me what's really in there.
    """
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    data = None

    if "EEG" in mat:
        try:
            data = np.asarray(mat["EEG"].data)
        except AttributeError:
            data = None

    if data is None:
        best = None
        for key, val in mat.items():
            if key.startswith("__"):
                continue
            arr = np.asarray(val)
            if arr.ndim == 2 and (128 in arr.shape or 129 in arr.shape):
                best = arr
                break
        if best is None:
            shapes = {k: np.asarray(v).shape for k, v in mat.items() if not k.startswith("__")}
            raise ValueError(
                f"couldn't find a 128/129-channel matrix in {path}. "
                f"Top-level variables and shapes: {shapes}"
            )
        data = best

    if data.ndim != 2:
        raise ValueError(f"unexpected data shape {data.shape} in {path}")
    if data.shape[0] not in (128, 129) and data.shape[1] in (128, 129):
        data = data.T   # MNE/EEG convention is (channels, samples); transpose if flipped
    if data.shape[0] == 129:
        data = data[:128, :]
    elif data.shape[0] != 128:
        raise ValueError(f"expected 128 or 129 channels, got shape {data.shape} in {path}")
    # MODMA amplitudes are typically in microvolts on export; MNE expects volts.
    return data.astype(np.float64) * 1e-6


def load_raw_band(path: str, cfg: CONFIG, band: Tuple[float, float]) -> np.ndarray:
    if USE_BIDS_EDF:
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        raw.pick_types(eeg=True)
    else:
        data = _load_modma_mat_matrix(path, cfg)
        ch_names = [f"E{i+1}" for i in range(cfg.N_CHANNELS)]
        info = mne.create_info(ch_names=ch_names, sfreq=cfg.MODMA_NATIVE_SFREQ, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose="ERROR")

    if len(raw.ch_names) < cfg.N_CHANNELS:
        raise ValueError(f"only {len(raw.ch_names)} EEG channels, need {cfg.N_CHANNELS}")
    if len(raw.ch_names) > cfg.N_CHANNELS:
        raw.pick(raw.ch_names[: cfg.N_CHANNELS])
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
#    -- UNCHANGED
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
# 3b. PLV (Phase Locking Value) connectivity features -- UNCHANGED
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


def build_dataset_from_cache(records_subset: List[Dict], signal_cache: Dict[str, np.ndarray],
                              cfg: CONFIG, selected_channels: np.ndarray, desc: str = "Feature extraction",
                              show_bar: bool = False
                              ) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """Wavelet/Hjorth + PLV feature extraction for a subset of records
    (e.g. one outer fold's dev or test subjects), using the already-loaded
    cached band-filtered signal, sliced to `selected_channels`. Replaces the
    disk-loading half of the old build_dataset_for_band(): the raw signal is
    sliced/windowed here rather than reloaded from disk."""
    X_by_scale = {s: [] for s in cfg.SCALES_SEC}
    y, groups = [], []
    iterator = tqdm(records_subset, desc=desc, unit="subj", leave=False) if show_bar else records_subset
    for rec in iterator:
        sig = signal_cache.get(rec["path"])
        if sig is None:
            continue
        sig = sig[selected_channels, :]
        try:
            per_scale = {s: scale_feature_vector(sig, cfg.SFREQ_TARGET, s, cfg)
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
# 4. Klein (Beltrami-Klein) hyperbolic model operations -- UNCHANGED
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
# 5. Pipeline construction (Logistic Regression only) -- UNCHANGED
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
# 6. Metrics -- UNCHANGED
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
# 7. Nested (outer + inner) cross-validation for one band
#    -- outer loop now also does per-fold channel selection (see section 1b)
# --------------------------------------------------------------------------- #
def run_one_outer_fold(band_name: str, X_dev: np.ndarray, y_dev: np.ndarray, groups_dev: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        cfg: CONFIG, seed: int, fold_label: str):
    t_fold0 = time.time()
    inner_cv = StratifiedGroupKFold(n_splits=cfg.N_INNER_FOLDS, shuffle=True, random_state=seed)
    grid = param_grid(cfg)

    t0 = time.time()
    pipe = make_pipeline(n_scales=len(cfg.SCALES_SEC), cfg=cfg)
    search = GridSearchCV(pipe, grid, cv=inner_cv, scoring="roc_auc", n_jobs=-1, refit=False)
    search.fit(X_dev, y_dev, groups=groups_dev)
    t_grid = time.time() - t0

    fixed_pipe = make_pipeline(n_scales=len(cfg.SCALES_SEC), cfg=cfg)
    fixed_pipe.set_params(**search.best_params_)

    diag_pipe = clone(fixed_pipe)
    diag_pipe.fit(X_dev, y_dev)
    saturation_rate = diag_pipe.named_steps["klein_fusion"].saturation_rate_

    bagged = make_bagging_classifier(fixed_pipe, cfg, seed)

    t0 = time.time()
    oof_proba = cross_val_predict(bagged, X_dev, y_dev, groups=groups_dev,
                                   cv=inner_cv, method="predict_proba", n_jobs=-1)[:, 1]
    best_threshold = find_best_threshold(y_dev, oof_proba)
    t_bag_cv = time.time() - t0

    t0 = time.time()
    bagged.fit(X_dev, y_dev)
    y_score = bagged.predict_proba(X_test)[:, 1]
    t_bag_fit = time.time() - t0
    y_pred = (y_score >= best_threshold).astype(int)

    fold_metrics = compute_metrics(y_test, y_pred, y_score)
    fold_metrics["decision_threshold"] = best_threshold
    fold_metrics["saturation_rate"] = saturation_rate
    t_fold_total = time.time() - t_fold0
    fold_metrics["_fold_seconds"] = t_fold_total
    print(f"[{band_name}] {fold_label} | best_params={search.best_params_} | "
          f"threshold={best_threshold:.3f} | acc={fold_metrics['accuracy']:.3f} | "
          f"sens={fold_metrics['recall_sensitivity']:.3f} | spec={fold_metrics['specificity']:.3f} | "
          f"auc={fold_metrics['roc_auc']:.3f} | fm={fold_metrics['fowlkes_mallows']:.3f} | "
          f"klein_saturation={saturation_rate:.3f}")
    print(f"    [TIMING] fold total={_fmt_secs(t_fold_total)}  "
          f"(grid_search={_fmt_secs(t_grid)}, bagged_oof_cv={_fmt_secs(t_bag_cv)}, "
          f"bagged_final_fit={_fmt_secs(t_bag_fit)})")
    return fold_metrics, y_test, y_pred, y_score


def run_nested_cv_for_band(band_name: str, band_range: Tuple[float, float],
                            records: List[Dict], signal_cache: Dict[str, np.ndarray],
                            X_bp: np.ndarray, y_all: np.ndarray, groups_all: np.ndarray,
                            cfg: CONFIG):
    """records/X_bp/y_all/groups_all must all be aligned (same order, same
    length) -- i.e. the `kept_records` output of compute_bandpower_matrix()
    and its accompanying y/groups, not the raw index_dataset() output."""
    all_fold_metrics = []
    pooled_y_true, pooled_y_pred, pooled_y_score = [], [], []
    channels_per_fold = []

    total_folds = cfg.N_REPEATS * cfg.N_OUTER_FOLDS
    fold_bar = tqdm(total=total_folds, desc=f"[{band_name}] outer folds", unit="fold")
    band_t0 = time.time()

    for repeat in range(cfg.N_REPEATS):
        seed = cfg.SEED + repeat
        outer_cv = StratifiedGroupKFold(n_splits=cfg.N_OUTER_FOLDS, shuffle=True, random_state=seed)

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_bp, y_all, groups_all), start=1):
            fold_num = (repeat * cfg.N_OUTER_FOLDS) + fold_idx
            fold_label = f"Repeat {repeat + 1}/{cfg.N_REPEATS} | Fold {fold_idx}/{cfg.N_OUTER_FOLDS}"

            # --- leakage-free channel selection: fit ONLY on this fold's dev subjects ---
            t0 = time.time()
            selected_channels = select_channels_for_fold(
                X_bp[train_idx], y_all[train_idx], cfg, band_range, cfg.N_CHANNELS_SELECT,
                fold_label=f"[{band_name}] {fold_label} "
            )
            t_select = time.time() - t0
            channels_per_fold.append(selected_channels)

            dev_records = [records[i] for i in train_idx]
            test_records = [records[i] for i in test_idx]

            # --- feature extraction for dev/test using only the fold's selected channels ---
            t0 = time.time()
            X_dev_by_scale, y_dev, groups_dev = build_dataset_from_cache(
                dev_records, signal_cache, cfg, selected_channels,
                desc=f"  fold {fold_num}/{total_folds} dev feats", show_bar=True)
            X_test_by_scale, y_test, _ = build_dataset_from_cache(
                test_records, signal_cache, cfg, selected_channels,
                desc=f"  fold {fold_num}/{total_folds} test feats", show_bar=True)
            t_feat = time.time() - t0

            X_dev = np.concatenate([X_dev_by_scale[s] for s in cfg.SCALES_SEC], axis=1)
            X_test = np.concatenate([X_test_by_scale[s] for s in cfg.SCALES_SEC], axis=1)
            print(f"    [TIMING] fold {fold_num}/{total_folds} pre-model: "
                  f"channel_select={_fmt_secs(t_select)}, feature_extraction={_fmt_secs(t_feat)}")

            # --- Klein fusion -> SelectKBest -> Logistic Regression (bagged), UNCHANGED ---
            fold_metrics, y_t, y_p, y_s = run_one_outer_fold(
                band_name, X_dev, y_dev, groups_dev, X_test, y_test, cfg, seed, fold_label
            )
            all_fold_metrics.append(fold_metrics)
            pooled_y_true.extend(y_t.tolist())
            pooled_y_pred.extend(y_p.tolist())
            pooled_y_score.extend(y_s.tolist())

            elapsed = time.time() - band_t0
            avg_per_fold = elapsed / fold_num
            remaining = avg_per_fold * (total_folds - fold_num)
            fold_bar.set_postfix_str(
                f"elapsed={_fmt_secs(elapsed)} | avg/fold={_fmt_secs(avg_per_fold)} | "
                f"ETA={_fmt_secs(remaining)}"
            )
            fold_bar.update(1)

    fold_bar.close()
    print(f"[TIMING] {band_name} band total ({total_folds} outer folds): {_fmt_secs(time.time() - band_t0)}")

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

    ch_names = [f"E{i + 1}" for i in range(cfg.N_CHANNELS)]
    channel_counts = {}
    for sel in channels_per_fold:
        for idx in sel:
            channel_counts[ch_names[idx]] = channel_counts.get(ch_names[idx], 0) + 1
    summary["channel_selection_frequency"] = dict(
        sorted(channel_counts.items(), key=lambda kv: kv[1], reverse=True)
    )

    return pooled, summary, all_fold_metrics, (pooled_y_true, pooled_y_pred, pooled_y_score)


# --------------------------------------------------------------------------- #
# 8. Persistence -- UNCHANGED
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
# 9. Cross-band comparison table -- UNCHANGED
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
# 10. Inline-only plots -- UNCHANGED
# --------------------------------------------------------------------------- #
def show_confusion_matrix(band_name: str, y_true: np.ndarray, y_pred: np.ndarray):
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Healthy", "MDD"], cmap="Blues", ax=ax
    )
    ax.set_title(f"{band_name} — pooled outer-fold predictions (Logistic Regression, MODMA-128)")
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
    ax.set_title("Pooled outer-fold ROC curves — band comparison (MODMA-128, Logistic Regression)")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
# 11. Main -- band loop now: load+cache raw signal once, compute bandpower
#     matrix once, then hand both to run_nested_cv_for_band() which does
#     per-outer-fold channel selection + feature extraction internally.
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
    records = index_dataset(CFG.DATA_ROOT)
    print(f"Indexed {len(records)} recordings across "
          f"{len(set(r['subject_id'] for r in records))} subjects")

    bands_to_run = _resolve_bands_to_run(CFG)
    print(f"\nBands to run THIS execution: {bands_to_run}")
    print(f"Classifier: Logistic Regression only")
    print(f"Channels: {CFG.N_CHANNELS} raw -> {CFG.N_CHANNELS_SELECT} selected per outer fold "
          f"(fast ANOVA F-test on bandpower, refit per fold to avoid leakage)")
    print(f"Running {CFG.N_REPEATS}x nested CV per band: "
          f"{CFG.N_OUTER_FOLDS} outer folds x {CFG.N_INNER_FOLDS} inner folds "
          f"= {CFG.N_REPEATS * CFG.N_OUTER_FOLDS} outer evaluations per band\n")

    run_t0 = time.time()
    band_bar = tqdm(bands_to_run, desc="Bands", unit="band")
    for band_key in band_bar:
        band_range = CFG.BANDS[band_key]
        band_name = band_key.capitalize()
        band_bar.set_postfix_str(band_name)

        print(f"\n{'=' * 20} {band_name} ({band_range[0]}-{band_range[1]} Hz) {'=' * 20}")

        signal_cache = load_band_signal_cache(records, CFG, band=band_range)
        kept_records, X_bp, y, groups = compute_bandpower_matrix(records, signal_cache, CFG)
        print(f"Usable recordings: {len(y)} "
              f"({y.sum()} MDD, {len(y) - y.sum()} NC) | "
              f"PLV included: {CFG.INCLUDE_PLV} | "
              f"channels selected per outer fold: {CFG.N_CHANNELS_SELECT}/{CFG.N_CHANNELS}")

        pooled, summary, fold_results, pooled_arrays = run_nested_cv_for_band(
            band_name, band_range, kept_records, signal_cache, X_bp, y, groups, CFG
        )
        pooled_y_true, pooled_y_pred, pooled_y_score = pooled_arrays

        print(f"\n--- Mean +/- std across {len(fold_results)} outer-fold evaluations ({band_name}) ---")
        for key in METRIC_KEYS:
            print(f"{key}: {summary[f'{key}_mean']:.3f} +/- {summary[f'{key}_std']:.3f}")
        print(f"klein_saturation_rate: {summary['saturation_rate_mean']:.3f} "
              f"+/- {summary['saturation_rate_std']:.3f}")
        print(f"\nChannel-selection frequency across all {len(fold_results)} outer folds "
              f"({band_name}) -- channels chosen most often across dev-only ANOVA reruns:")
        for ch, count in summary["channel_selection_frequency"].items():
            print(f"  {ch:>5s}: selected in {count}/{len(fold_results)} folds")

        print(f"\n--- Pooled classification report ({band_name}) ---")
        print(classification_report(pooled_y_true, pooled_y_pred, target_names=["Healthy", "MDD"]))

        save_band_result(CFG, band_name, summary, pooled_arrays)
        show_confusion_matrix(band_name, pooled_y_true, pooled_y_pred)

        # free the per-band signal cache before moving to the next band
        del signal_cache

    band_bar.close()
    print(f"\n[TIMING] full run ({len(bands_to_run)} band(s)) total: {_fmt_secs(time.time() - run_t0)}")

    all_summaries, pooled_scores_by_band = load_all_saved_band_results(CFG)
    if not all_summaries:
        print("\nNo band results found on disk yet.")
        return None

    if len(pooled_scores_by_band) >= 1:
        show_roc_overlay(pooled_scores_by_band)

    comparison_df = build_comparison_table(all_summaries)
    print("\n" + "=" * 100)
    print(f"BAND COMPARISON (Logistic Regression, MODMA-128) — {len(all_summaries)}/5 bands completed so far")
    print("=" * 100)
    try:
        from tabulate import tabulate
        print(tabulate(comparison_df, headers="keys", tablefmt="github", showindex=False))
    except ImportError:
        print(comparison_df.to_string(index=False))

    os.makedirs(CFG.SAVE_DIR, exist_ok=True)
    csv_path = os.path.join(CFG.SAVE_DIR, "band_comparison_logreg_modma128.csv")
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
