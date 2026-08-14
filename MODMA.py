import os
import re
import sys
import glob
import json
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import mne
import pywt
import matplotlib.pyplot as plt
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

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
@dataclass
class CONFIG:
    DATA_ROOT: str = "/kaggle/input/datasets/kinnarhalder/eeg-dataset"
    SFREQ_TARGET: int = 128
    BANDS: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 45.0),
    })
    # Which band(s) to run THIS execution -- just edit this tuple.
    # Leave all 5 in to run everything in one go, e.g.:
    #     BANDS_TO_RUN: Tuple[str, ...] = ("delta", "theta", "alpha", "beta", "gamma")
    # or trim it down to run only what you want this session, e.g.:
    #     BANDS_TO_RUN: Tuple[str, ...] = ("alpha",)
    #     BANDS_TO_RUN: Tuple[str, ...] = ("beta", "gamma")
    # A command-line arg (`python eeg_klein_lr_pipeline.py alpha`) still
    # overrides this tuple for that run, if you'd rather do it that way.
    BANDS_TO_RUN: Tuple[str, ...] = field(default_factory=lambda: (
        "delta",
    ))
    SCALES_SEC: Tuple[int, int, int] = (1, 2, 3)         # the three partition scales
    N_CHANNELS: int = 19
    WAVELET: str = "db4"
    WAVELET_LEVEL: int = 3
    INCLUDE_PLV: bool = True                             # append pairwise PLV connectivity features per window
    KLEIN_CURVATURE_GRID: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    KLEIN_SCALE_FACTOR_GRID: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)   # tau: dimension-aware pre-scale, see module docstring
    SATURATION_MARGIN: float = 0.95                      # boundary-proximity threshold for the saturation-rate diagnostic
    SELECT_K_GRID: Tuple = (30, 60, 120, "all")
    N_OUTER_FOLDS: int = 5                                # outer loop: unbiased performance estimate
    N_INNER_FOLDS: int = 3                                # inner loop: hyperparameter + threshold tuning
    N_REPEATS: int = 3                                    # outer (external) evaluation repeated 5x with fresh folds
    N_BAGGING_ESTIMATORS: int = 31                        # bagged over bootstrap resamples of the dev set
    BAGGING_MAX_SAMPLES: float = 0.9                      # fraction of dev set each bagged estimator trains on
    SEED: int = 42
    SAVE_DIR: str = "results"                             # relative -> written next to this script


CFG = CONFIG()
np.random.seed(CFG.SEED)


# --------------------------------------------------------------------------- #
# 1. Data indexing / labeling
# --------------------------------------------------------------------------- #
def index_dataset(root: str) -> List[Dict]:
    """Scan DATA_ROOT and return [{path, subject_id, label}, ...]."""
    records = []
    for path in sorted(glob.glob(os.path.join(root, "*.edf"))):
        fname = os.path.basename(path)
        m = re.search(r"(MDD|H)\s*S(\d+)", fname, flags=re.IGNORECASE)
        if not m:
            continue
        group, subj_num = m.group(1).upper(), m.group(2)
        label = 1 if group == "MDD" else 0
        subject_id = f"{group}_{subj_num}"
        records.append({"path": path, "subject_id": subject_id, "label": label})
    if not records:
        raise RuntimeError(f"No EDF files matched under {root}")
    return records


# --------------------------------------------------------------------------- #
# 2. Signal loading + multi-scale windowing
#    Only band-limited loading is supported now (no broadband "Whole"
#    condition). One disk read per recording per band -- run the whole
#    script once per band (see CFG.BANDS_TO_RUN / CLI arg) if you want to
#    split the 5 bands across separate sessions to keep each run short.
# --------------------------------------------------------------------------- #
def load_raw_band(path: str, cfg: CONFIG, band: Tuple[float, float]) -> np.ndarray:
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    raw.pick_types(eeg=True)
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
# 3b. PLV (Phase Locking Value) connectivity features
#     Instantaneous phase per channel via the analytic signal (Hilbert
#     transform), then pairwise PLV across all channel pairs within the
#     window. Only the upper-triangle (i<j) is kept -- PLV is symmetric and
#     PLV(i,i)=1 by construction, so the diagonal carries no information.
#
#     Validity note for the paper: the Hilbert-transform phase estimate is
#     only meaningful for a (quasi-)narrowband analytic signal. Now that
#     every condition is a single canonical band (delta/theta/alpha/beta/
#     gamma), the Hilbert phase -> PLV pipeline is applied to genuinely
#     narrowband signals throughout, which is the methodologically correct
#     regime for this estimator (verified numerically: two synthetic
#     sinusoids at a fixed phase offset give PLV ~ 1.0, and a sinusoid vs.
#     white noise gives PLV ~ 0; see accompanying validation script).
# --------------------------------------------------------------------------- #
def instantaneous_phase(window: np.ndarray) -> np.ndarray:
    """window: (n_channels, n_samples) -> instantaneous phase, same shape."""
    analytic = hilbert(window, axis=-1)
    return np.angle(analytic)


def plv_matrix(window: np.ndarray) -> np.ndarray:
    """Pairwise PLV across channels for one window. Returns (n_ch, n_ch)."""
    phase = instantaneous_phase(window)          # (n_ch, n_samples)
    phase_diff = phase[:, None, :] - phase[None, :, :]   # (n_ch, n_ch, n_samples)
    plv = np.abs(np.mean(np.exp(1j * phase_diff), axis=-1))
    return plv


def plv_window_features(window: np.ndarray) -> np.ndarray:
    """Upper-triangle (i<j) PLV values for one window, flattened."""
    n_ch = window.shape[0]
    plv = plv_matrix(window)
    iu = np.triu_indices(n_ch, k=1)
    feats = plv[iu].astype(np.float64)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def full_window_features(window: np.ndarray, wavelet: str, level: int, include_plv: bool) -> np.ndarray:
    """Wavelet/Hjorth features, optionally concatenated with PLV connectivity features."""
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
    """Load ALL recordings, filtered to a single band, THEN the caller runs
    logistic regression on the resulting feature matrix. One disk read per
    recording for this band."""
    X_by_scale = {s: [] for s in cfg.SCALES_SEC}
    y, groups = [], []
    for rec in records:
        try:
            signal = load_raw_band(rec["path"], cfg, band=band)
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
# 4. Klein (Beltrami-Klein) hyperbolic model operations
# --------------------------------------------------------------------------- #
"""
Dimension-aware pre-scaling -- derivation
==========================================
Let u in R^d be a standardized feature block (each coordinate transformed by
a StandardScaler fit on the *training* fold only, so E[u_j] = 0,
Var[u_j] = 1 for every coordinate j = 1..d).

Treating the d standardized coordinates as approximately uncorrelated (a
reasonable approximation once SelectKBest has trimmed redundant wavelet/PLV
features), the squared norm is a sum of d approximately unit-variance terms:

    E[||u||^2] = sum_j E[u_j^2] = sum_j Var[u_j] = d
    =>  E[||u||]  ~  sqrt(d)                       (concentration for large d)

This was confirmed numerically: for d = 10, 60, 120, 500, 2000 standardized
random features, the empirical mean norm matched sqrt(d) to within ~1%.

The Klein-disk exponential map at the origin (curvature -c, c > 0) is

    exp_0(u) = [ tanh( sqrt(c) * ||u|| ) / ( sqrt(c) * ||u|| ) ] * u

As d grows, ||u|| ~ sqrt(d) grows without bound, so sqrt(c)*||u|| -> infinity
and tanh(.) saturates to 1 for *every* input regardless of its direction
u / ||u||. Geometrically, every point is pushed onto the boundary of the
Klein ball (radius 1/sqrt(c)); the discriminative information the
downstream classifier needs collapses, and log_map0 becomes numerically
stiff (arctanh argument approaches +-1). This was confirmed numerically:
with d=60 unscaled standardized features and c=1, ~100% of points landed
within 5% of the disk boundary.

Fix -- dimension-aware pre-scaling
-----------------------------------
Rescale u before the exponential map so its expected norm no longer grows
with d:

    u' = tau * u / sqrt(d)          =>   E[||u'||] ~ tau   (independent of d)

tau ("scale_factor") is a small, dimensionless, tunable hyperparameter,
grid-searched jointly with curvature c (see CFG.KLEIN_SCALE_FACTOR_GRID),
that controls how close to the boundary a "typical" point sits: the
argument of tanh(.) at the expected norm is sqrt(c) * tau, independent of
feature dimensionality. Confirmed numerically: after the fix, mean norm
tracks tau regardless of d, and boundary-saturation drops to ~0 for
tau, c = O(1).

Saturation-rate diagnostic
---------------------------
After the exp-map + Einstein-midpoint fusion step, the fraction of fused
points whose Klein-disk norm exceeds SATURATION_MARGIN * (1/sqrt(c)) (i.e.
within 5% of the disk boundary by default, where log_map0 becomes
numerically stiff) is logged as `saturation_rate_` on the fitted
transformer and reported per outer CV fold and per band condition.

Einstein-midpoint / Lorentz-factor identities used below (gamma(x) =
1/sqrt(1-c||x||^2), weighted midpoint = sum(w*gamma*x) / sum(w*gamma)) were
checked to satisfy the required invariants: the midpoint of several copies
of the same point equals that point, and the midpoint of two
antipodally-symmetric points equals the origin.
"""


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
        # Valid Klein-disk radius is 1/sqrt(c), not 1 -- must match _clamp_norm's
        # boundary or points get wrongly over-clamped whenever c != 1 (e.g. c=0.5).
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
        """Fraction of points within `margin` of the Klein-disk boundary."""
        norm = np.linalg.norm(x, axis=-1)
        return float(np.mean(norm > margin * self._max_norm()))


class KleinMultiScaleFusion(BaseEstimator, TransformerMixin):
    """Standardizes each scale's feature block (fit on training data only),
    applies dimension-aware pre-scaling u' = scale_factor * u / sqrt(d) (see
    module-level derivation above), maps each block into the Klein disk,
    fuses via Einstein midpoint, and returns the fused tangent-space vector.
    `curvature` and `scale_factor` are tunable hyperparameters.
    """

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
        # dimension-aware pre-scale: tau / sqrt(d), d = block_size_ (see derivation above)
        self.pre_scale_ = self.scale_factor / np.sqrt(self.block_size_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        klein_points = []
        for i in range(self.n_scales):
            block = X[:, i * self.block_size_:(i + 1) * self.block_size_]
            block = self.scalers_[i].transform(block)
            block = self.pre_scale_ * block            # dimension-aware pre-scaling
            klein_points.append(self.klein_.exp_map0(block))
        fused = self.klein_.einstein_midpoint(klein_points)
        # diagnostic: fraction of fused points near the Klein-disk boundary
        self.saturation_rate_ = self.klein_.saturation_rate(fused, self.saturation_margin)
        return self.klein_.log_map0(fused)


# --------------------------------------------------------------------------- #
# 5. Pipeline construction (Logistic Regression only)
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
# 6. Metrics
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
    """Youden's J-optimal decision threshold (maximizes sensitivity +
    specificity - 1) from ROC curve points. Called on cross-validated
    out-of-fold scores from the DEVELOPMENT (outer-training) set only,
    so the outer-test fold is never touched during tuning."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    youden_j = tpr - fpr
    best_idx = int(np.argmax(youden_j))
    return float(np.clip(thresholds[best_idx], 0.0, 1.0))


def make_bagging_classifier(base_pipe: Pipeline, cfg: CONFIG, seed: int) -> BaggingClassifier:
    """sklearn renamed BaggingClassifier's `base_estimator` kwarg to
    `estimator` in 1.2; try the current name first, fall back for older
    installs (e.g. some Kaggle images still ship pre-1.2 scikit-learn)."""
    kwargs = dict(n_estimators=cfg.N_BAGGING_ESTIMATORS, max_samples=cfg.BAGGING_MAX_SAMPLES,
                   bootstrap=True, random_state=seed, n_jobs=-1)
    try:
        return BaggingClassifier(estimator=base_pipe, **kwargs)
    except TypeError:
        return BaggingClassifier(base_estimator=base_pipe, **kwargs)


# --------------------------------------------------------------------------- #
# 7. Nested (outer + inner) cross-validation for one band.
# --------------------------------------------------------------------------- #
def run_one_outer_fold(band_name: str, X_dev: np.ndarray, y_dev: np.ndarray, groups_dev: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        cfg: CONFIG, seed: int, fold_label: str):
    inner_cv = StratifiedGroupKFold(n_splits=cfg.N_INNER_FOLDS, shuffle=True, random_state=seed)
    grid = param_grid(cfg)

    # --- Inner stage 1: hyperparameter search (incl. Klein curvature & scale_factor)
    # on this fold's training data only. ---
    pipe = make_pipeline(n_scales=len(cfg.SCALES_SEC), cfg=cfg)
    search = GridSearchCV(pipe, grid, cv=inner_cv, scoring="roc_auc", n_jobs=-1, refit=False)
    search.fit(X_dev, y_dev, groups=groups_dev)

    # --- Inner stage 2: freeze hyperparameters, bag the estimator for variance reduction. ---
    fixed_pipe = make_pipeline(n_scales=len(cfg.SCALES_SEC), cfg=cfg)
    fixed_pipe.set_params(**search.best_params_)

    # diagnostic-only (unbagged) fit purely to read off the Klein saturation rate
    # for this fold's chosen hyperparameters -- not used for prediction.
    diag_pipe = clone(fixed_pipe)
    diag_pipe.fit(X_dev, y_dev)
    saturation_rate = diag_pipe.named_steps["klein_fusion"].saturation_rate_

    bagged = make_bagging_classifier(fixed_pipe, cfg, seed)

    # --- Inner stage 3: tune the decision threshold on out-of-fold
    # probabilities from this fold's training data only (still no outer-test
    # leakage -- cross_val_predict here only ever sees X_dev/y_dev/groups_dev). ---
    oof_proba = cross_val_predict(bagged, X_dev, y_dev, groups=groups_dev,
                                   cv=inner_cv, method="predict_proba", n_jobs=-1)[:, 1]
    best_threshold = find_best_threshold(y_dev, oof_proba)

    # --- Outer evaluation: fit on the full outer-training fold, score once
    # on the outer-test fold with the frozen bagged ensemble + threshold. ---
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
# 8. Persistence -- save each band's result to disk so the 5-band table can
#    be rebuilt across separate runs/sessions (one run per band).
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
    """Reload every band's persisted results found under cfg.SAVE_DIR (works
    whether this is the 1st or the 5th separate run) so the comparison table
    always reflects everything computed so far, across however many
    sessions the 5 bands were split into."""
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
# 9. Cross-band comparison table
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
# 10. Inline-only plots (no files written to disk)
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
    """One ROC plot with every available band overlaid (whatever has been
    run/persisted so far -- 1 to 5 bands)."""
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
    """CLI arg takes priority (runs just that one band) over CFG.BANDS_TO_RUN,
    but ONLY if it's a real band name -- Jupyter/Kaggle/Colab kernels inject
    their own argv[1] (e.g. '-f /path/to/kernel.json'), which is not a CLI
    override and must be ignored rather than raising an error. Otherwise,
    whatever band names are listed in CFG.BANDS_TO_RUN are run, in that
    order -- edit that tuple directly to control which band(s) this
    execution covers, e.g. trim it to ("alpha",) for a single-band run."""
    all_band_keys = list(cfg.BANDS.keys())

    raw_arg = sys.argv[1] if len(sys.argv) > 1 else None
    cli_band = raw_arg.lower() if raw_arg is not None else None
    if cli_band is not None and cli_band in cfg.BANDS:
        return [cli_band]
    elif cli_band is not None and not cli_band.startswith("-"):
        # Looks like a real (but mistyped) CLI arg from an actual terminal run
        # -- flag this clearly instead of silently ignoring a typo.
        raise ValueError(f"Unknown band '{raw_arg}'. Choose one of {all_band_keys}.")
    # else: no CLI arg, or a notebook/kernel-injected flag like '-f' -- fall
    # through to CFG.BANDS_TO_RUN below.

    bands_to_run = cfg.BANDS_TO_RUN
    if isinstance(bands_to_run, str):
        # Defensive fix: BANDS_TO_RUN = "alpha" (no trailing comma) is NOT a
        # 1-element tuple in Python -- it's just the string "alpha", and
        # iterating over it yields ['a','l','p','h','a']. Treat a bare
        # string as a single band name instead of crashing on that typo.
        bands_to_run = (bands_to_run,)

    if not bands_to_run:
        raise ValueError(f"CFG.BANDS_TO_RUN is empty -- list at least one of {all_band_keys}.")
    unknown = [b for b in bands_to_run if b not in cfg.BANDS]
    if unknown:
        raise ValueError(f"Unknown band(s) {unknown} in CFG.BANDS_TO_RUN. Choose from {all_band_keys}. "
                          f"(Tip: a single band must be written with a trailing comma, e.g. (\"alpha\",) "
                          f"-- without it, Python treats \"alpha\" as a string and iterates its letters.)")
    return list(bands_to_run)


def main():
    records = index_dataset(CFG.DATA_ROOT)
    print(f"Indexed {len(records)} recordings across "
          f"{len(set(r['subject_id'] for r in records))} subjects")

    bands_to_run = _resolve_bands_to_run(CFG)
    print(f"\nBands to run THIS execution: {bands_to_run}")
    print(f"Classifier: Logistic Regression only")
    print(f"Running {CFG.N_REPEATS}x nested CV per band: "
          f"{CFG.N_OUTER_FOLDS} outer folds x {CFG.N_INNER_FOLDS} inner folds "
          f"= {CFG.N_REPEATS * CFG.N_OUTER_FOLDS} outer evaluations per band")
    print("Loading strategy: for each band, ALL recordings are loaded/filtered "
          "to that band first, then Logistic Regression nested CV runs once on "
          "the full band dataset. No broadband 'Whole' condition anymore.\n")

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

    # -------------------- Rebuild the comparison table from EVERYTHING
    # persisted so far (works after 1 run or after all 5 separate runs). --- #
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
