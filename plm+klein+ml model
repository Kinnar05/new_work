"""
EEG MDD vs Healthy classification — multi-scale Klein/PLM feature fusion,
benchmarked across 7 classifiers under identical nested cross-validation.

This extends the original single-SVM script: everything upstream of the
classifier (loading, windowing, wavelet+Hjorth+PLM features, the Klein
multi-scale fusion transformer, nested CV scaffolding, bagging, metrics)
is unchanged. What changed is that the classifier is now a swappable
component, and the whole nested-CV loop runs once per model on the SAME
outer/inner fold splits, so the final comparison table is apples-to-apples.

Models benchmarked: SVM (RBF/linear), Logistic Regression, KNN, Random
Forest, AdaBoost, XGBoost, LightGBM. Each is wrapped in the same bagging
ensemble the original script used for SVM, so bagging is preserved as
the final variance-reduction step for every model, not just SVM.
"""

import os
import re
import glob
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable

import numpy as np
import pandas as pd
import mne
import pywt
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.signal import butter, filtfilt, hilbert
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier,
                               AdaBoostClassifier)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                              recall_score, roc_auc_score, roc_curve, classification_report,
                              ConfusionMatrixDisplay, RocCurveDisplay)

warnings.filterwarnings("ignore", category=UserWarning)

# Optional boosted-tree libraries — degrade gracefully if not installed so
# the rest of the benchmark still runs.
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
@dataclass
class CONFIG:
    DATA_ROOT: str = "/kaggle/input/datasets/kinnarhalder/eeg-dataset"
    SFREQ_TARGET: int = 128
    BANDPASS: Tuple[float, float] = (1.0, 45.0)
    SCALES_SEC: Tuple[int, int, int] = (1, 2, 3)
    N_CHANNELS: int = 19
    WAVELET: str = "db4"
    WAVELET_LEVEL: int = 3
    USE_PLM_FEATURES: bool = True
    PLM_BAND: Tuple[float, float] = (8.0, 13.0)
    PLM_INTEGRATION_BAND_HZ: float = 1.0
    PLM_VC_EPS: float = 0.05
    KLEIN_CURVATURE_GRID: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    SELECT_K_GRID: Tuple = (30, 60, 120, "all")
    N_OUTER_FOLDS: int = 5
    N_INNER_FOLDS: int = 5
    N_REPEATS: int = 1
    N_BAGGING_ESTIMATORS: int = 31
    BAGGING_MAX_SAMPLES: float = 0.9
    SEED: int = 42
    SAVE_MODEL_DIR: str = "final_models"          # relative -> Kaggle's /kaggle/working/
    RESULTS_CSV_PATH: str = "model_comparison.csv"
    # Which models to run. Trim this list to control runtime — nested CV is
    # N_REPEATS * N_OUTER_FOLDS grid searches PER MODEL, each searching
    # N_INNER_FOLDS x (grid size) fits, so cost multiplies fast across models.
    MODELS_TO_RUN: Tuple[str, ...] = (
        "SVM", "LogisticRegression", "KNN", "RandomForest",
        "AdaBoost", "XGBoost", "LightGBM",
    )


CFG = CONFIG()
np.random.seed(CFG.SEED)


# --------------------------------------------------------------------------- #
# 1. Data indexing / labeling  (unchanged)
# --------------------------------------------------------------------------- #
def index_dataset(root: str) -> List[Dict]:
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
# 2. Signal loading + multi-scale windowing  (unchanged)
# --------------------------------------------------------------------------- #
def load_raw(path: str, cfg: CONFIG) -> np.ndarray:
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    raw.pick_types(eeg=True)
    if len(raw.ch_names) < cfg.N_CHANNELS:
        raise ValueError(f"only {len(raw.ch_names)} EEG channels, need {cfg.N_CHANNELS}")
    if len(raw.ch_names) > cfg.N_CHANNELS:
        raw.pick(raw.ch_names[: cfg.N_CHANNELS])
    raw.filter(cfg.BANDPASS[0], cfg.BANDPASS[1], verbose="ERROR")
    if raw.info["sfreq"] != cfg.SFREQ_TARGET:
        raw.resample(cfg.SFREQ_TARGET, verbose="ERROR")
    return raw.get_data()


def partition_signal(signal: np.ndarray, sfreq: int, window_sec: int) -> List[np.ndarray]:
    win_len = int(window_sec * sfreq)
    n_windows = signal.shape[1] // win_len
    return [signal[:, i * win_len:(i + 1) * win_len] for i in range(n_windows)]


# --------------------------------------------------------------------------- #
# 3. Feature extraction: wavelet sub-band statistics + Hjorth parameters
#    (unchanged)
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
# 3b. Phase Linearity Measurement (unchanged — see chat note on the DC-bin
#     volume-conduction correction being single-bin rather than a small band;
#     left as-is here since it matches the original script's behavior).
# --------------------------------------------------------------------------- #
_FILTER_CACHE: Dict[Tuple, Tuple[np.ndarray, np.ndarray]] = {}


def _get_bandpass_coeffs(sfreq: float, band: Tuple[float, float], order: int = 4):
    key = (sfreq, band, order)
    if key not in _FILTER_CACHE:
        nyq = 0.5 * sfreq
        b, a = butter(order, [band[0] / nyq, band[1] / nyq], btype="band")
        _FILTER_CACHE[key] = (b, a)
    return _FILTER_CACHE[key]


def plm_node_strength(window: np.ndarray, sfreq: float, cfg: CONFIG) -> np.ndarray:
    n_ch, n_samp = window.shape
    b, a = _get_bandpass_coeffs(sfreq, cfg.PLM_BAND)

    analytic = np.zeros((n_ch, n_samp), dtype=complex)
    for ch in range(n_ch):
        try:
            filtered = filtfilt(b, a, window[ch])
        except ValueError:
            filtered = window[ch]
        analytic[ch] = hilbert(filtered)

    freqs = np.fft.fftfreq(n_samp, d=1.0 / sfreq)
    zero_idx = int(np.argmin(np.abs(freqs)))
    in_band = np.abs(freqs) <= cfg.PLM_INTEGRATION_BAND_HZ

    strength_sum = np.zeros(n_ch)
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            z = analytic[i] * np.conj(analytic[j])
            amp = np.clip(np.abs(z), 1e-12, None)
            zN = z / amp
            ZN = np.fft.fft(zN)
            if abs(np.angle(ZN[zero_idx])) < cfg.PLM_VC_EPS:
                ZN[zero_idx] = 0
            Sz = np.abs(ZN) ** 2
            denom = Sz.sum()
            plm = float(Sz[in_band].sum() / denom) if denom > 1e-15 else 0.0
            strength_sum[i] += plm
            strength_sum[j] += plm

    strength = strength_sum / max(n_ch - 1, 1)
    return np.nan_to_num(strength, nan=0.0, posinf=0.0, neginf=0.0)


def window_features(window: np.ndarray, sfreq: float, cfg: CONFIG) -> np.ndarray:
    feats = wavelet_window_features(window, cfg.WAVELET, cfg.WAVELET_LEVEL)
    if cfg.USE_PLM_FEATURES:
        feats = np.concatenate([feats, plm_node_strength(window, sfreq, cfg)])
    return feats


def scale_feature_vector(signal: np.ndarray, sfreq: int, window_sec: int, cfg: CONFIG) -> np.ndarray:
    windows = partition_signal(signal, sfreq, window_sec)
    if not windows:
        raise ValueError(f"recording too short for a {window_sec}s window")
    feats = np.stack([window_features(w, sfreq, cfg) for w in windows])
    return feats.mean(axis=0)


def build_dataset(records: List[Dict], cfg: CONFIG) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray]:
    X_by_scale = {s: [] for s in cfg.SCALES_SEC}
    y, groups = [], []
    for rec in records:
        try:
            signal = load_raw(rec["path"], cfg)
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
class KleinOps:
    def __init__(self, c: float, eps: float = 1e-3):
        self.c = c
        self.eps = eps

    def _clamp_norm(self, x: np.ndarray) -> np.ndarray:
        max_norm = (1.0 / np.sqrt(self.c)) * (1 - self.eps)
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
        # FIX: the valid Klein ball has radius 1/sqrt(c), not 1. The norm
        # clip below must be scaled by 1/sqrt_c to match _clamp_norm's
        # max_norm — previously this was hardcoded to (1 - self.eps),
        # which silently truncated legitimate points whenever c < 1 (ball
        # radius > 1), corrupting the tangent-space output for those
        # curvature values. For c >= 1 this bound was never binding, so
        # behavior there is unchanged.
        max_norm = (1.0 / sqrt_c) * (1 - self.eps)
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


class KleinMultiScaleFusion(BaseEstimator, TransformerMixin):
    def __init__(self, n_scales: int = 3, curvature: float = 1.0):
        self.n_scales = n_scales
        self.curvature = curvature

    def fit(self, X: np.ndarray, y=None):
        n_feat_total = X.shape[1]
        assert n_feat_total % self.n_scales == 0
        self.block_size_ = n_feat_total // self.n_scales
        self.scalers_ = []
        for i in range(self.n_scales):
            block = X[:, i * self.block_size_:(i + 1) * self.block_size_]
            self.scalers_.append(StandardScaler().fit(block))
        self.klein_ = KleinOps(c=self.curvature)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        klein_points = []
        for i in range(self.n_scales):
            block = X[:, i * self.block_size_:(i + 1) * self.block_size_]
            block = self.scalers_[i].transform(block)
            klein_points.append(self.klein_.exp_map0(block))
        fused = self.klein_.einstein_midpoint(klein_points)
        return self.klein_.log_map0(fused)


# --------------------------------------------------------------------------- #
# 5. Model registry — one classifier + its hyperparameter grid per entry.
#    Each factory takes `pos_weight` (neg/pos ratio of the CURRENT outer
#    training fold) so tree/boosting models get correct class-imbalance
#    handling without griding over it. Grids are kept modest on purpose:
#    nested CV multiplies grid size x N_INNER_FOLDS x N_OUTER_FOLDS x
#    N_REPEATS x N_MODELS, so a large grid here makes a 7-model benchmark
#    impractical. Widen individual grids once you know which models matter.
# --------------------------------------------------------------------------- #
ModelFactory = Callable[[float], Tuple[BaseEstimator, Dict]]


def _svm_factory(pos_weight: float) -> Tuple[BaseEstimator, Dict]:
    est = SVC(class_weight="balanced", probability=True, random_state=CFG.SEED)
    grid = {
        "clf__C": [0.1, 1.0, 10.0, 100.0],
        "clf__kernel": ["rbf", "linear"],
        "clf__gamma": ["scale", "auto"],
    }
    return est, grid


def _logreg_factory(pos_weight: float) -> Tuple[BaseEstimator, Dict]:
    est = LogisticRegression(class_weight="balanced", max_iter=5000, random_state=CFG.SEED)
    grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"],
    }
    return est, grid


def _knn_factory(pos_weight: float) -> Tuple[BaseEstimator, Dict]:
    # KNN has no native class_weight; imbalance is instead handled by the
    # shared decision-threshold tuning step (Youden's J) applied to every
    # model downstream, same as the others.
    est = KNeighborsClassifier()
    grid = {
        "clf__n_neighbors": [5, 9, 15, 25],
        "clf__weights": ["uniform", "distance"],
        "clf__p": [1, 2],
    }
    return est, grid


def _rf_factory(pos_weight: float) -> Tuple[BaseEstimator, Dict]:
    est = RandomForestClassifier(class_weight="balanced", random_state=CFG.SEED, n_jobs=-1)
    grid = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 6, 12],
        "clf__min_samples_leaf": [1, 3, 5],
    }
    return est, grid


def _adaboost_factory(pos_weight: float) -> Tuple[BaseEstimator, Dict]:
    est = AdaBoostClassifier(random_state=CFG.SEED)
    grid = {
        "clf__n_estimators": [100, 200, 300],
        "clf__learning_rate": [0.05, 0.1, 0.5, 1.0],
    }
    return est, grid


def _xgb_factory(pos_weight: float) -> Tuple[BaseEstimator, Dict]:
    est = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        scale_pos_weight=pos_weight, random_state=CFG.SEED,
        n_jobs=-1, tree_method="hist",
    )
    grid = {
        "clf__n_estimators": [150, 300],
        "clf__max_depth": [3, 5, 7],
        "clf__learning_rate": [0.03, 0.1, 0.2],
    }
    return est, grid


def _lgbm_factory(pos_weight: float) -> Tuple[BaseEstimator, Dict]:
    est = LGBMClassifier(
        objective="binary", class_weight="balanced",
        random_state=CFG.SEED, n_jobs=-1, verbosity=-1,
    )
    grid = {
        "clf__n_estimators": [150, 300],
        "clf__num_leaves": [15, 31, 63],
        "clf__learning_rate": [0.03, 0.1, 0.2],
    }
    return est, grid


MODEL_REGISTRY: Dict[str, ModelFactory] = {
    "SVM": _svm_factory,
    "LogisticRegression": _logreg_factory,
    "KNN": _knn_factory,
    "RandomForest": _rf_factory,
    "AdaBoost": _adaboost_factory,
}
if _HAS_XGB:
    MODEL_REGISTRY["XGBoost"] = _xgb_factory
else:
    print("[WARN] xgboost not installed — skipping XGBoost (pip install xgboost)")
if _HAS_LGBM:
    MODEL_REGISTRY["LightGBM"] = _lgbm_factory
else:
    print("[WARN] lightgbm not installed — skipping LightGBM (pip install lightgbm)")


# --------------------------------------------------------------------------- #
# 6. Metrics  (unchanged)
# --------------------------------------------------------------------------- #
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray = None) -> Dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    return metrics


def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    youden_j = tpr - fpr
    best_idx = int(np.argmax(youden_j))
    return float(np.clip(thresholds[best_idx], 0.0, 1.0))


def make_bagging_classifier(base_pipe: Pipeline, cfg: CONFIG, seed: int) -> BaggingClassifier:
    """Bagging is kept as the final variance-reduction wrapper for EVERY
    model in the registry (not just SVM) — this matches how the original
    script only ever deployed a bagged SVM, generalized here so every
    model benefits from the same ensembling at evaluation and deployment
    time."""
    kwargs = dict(n_estimators=cfg.N_BAGGING_ESTIMATORS, max_samples=cfg.BAGGING_MAX_SAMPLES,
                   bootstrap=True, random_state=seed, n_jobs=-1)
    try:
        return BaggingClassifier(estimator=base_pipe, **kwargs)
    except TypeError:
        return BaggingClassifier(base_estimator=base_pipe, **kwargs)


# --------------------------------------------------------------------------- #
# 7. Generic pipeline builder (Klein fusion + feature selection + any clf)
# --------------------------------------------------------------------------- #
def make_pipeline(n_scales: int, estimator: BaseEstimator) -> Pipeline:
    return Pipeline([
        ("klein_fusion", KleinMultiScaleFusion(n_scales=n_scales)),
        ("select", SelectKBest(score_func=mutual_info_classif)),
        ("clf", estimator),
    ])


def make_param_grid(cfg: CONFIG, clf_grid: Dict) -> Dict:
    grid = {
        "klein_fusion__curvature": list(cfg.KLEIN_CURVATURE_GRID),
        "select__k": list(cfg.SELECT_K_GRID),
    }
    grid.update(clf_grid)
    return grid


# --------------------------------------------------------------------------- #
# 8. Nested (outer + inner) cross-validation — now parameterized by model
# --------------------------------------------------------------------------- #
def run_one_outer_fold(model_name: str, X_dev: np.ndarray, y_dev: np.ndarray, groups_dev: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        cfg: CONFIG, seed: int, fold_label: str):
    inner_cv = StratifiedGroupKFold(n_splits=cfg.N_INNER_FOLDS, shuffle=True, random_state=seed)

    n_pos = int(y_dev.sum())
    n_neg = int(len(y_dev) - n_pos)
    pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    estimator, clf_grid = MODEL_REGISTRY[model_name](pos_weight)

    # --- Inner stage 1: hyperparameter search on this fold's training data. ---
    pipe = make_pipeline(n_scales=len(cfg.SCALES_SEC), estimator=clone(estimator))
    search = GridSearchCV(pipe, make_param_grid(cfg, clf_grid), cv=inner_cv,
                           scoring="accuracy", n_jobs=-1, refit=False)
    search.fit(X_dev, y_dev, groups=groups_dev)

    # --- Inner stage 2: freeze hyperparameters, bag over bootstrap resamples. ---
    fixed_pipe = make_pipeline(n_scales=len(cfg.SCALES_SEC), estimator=clone(estimator))
    fixed_pipe.set_params(**search.best_params_)
    bagged = make_bagging_classifier(fixed_pipe, cfg, seed)

    # --- Inner stage 3: tune decision threshold on out-of-fold probabilities
    # from this fold's training data only. ---
    oof_proba = cross_val_predict(bagged, X_dev, y_dev, groups=groups_dev,
                                   cv=inner_cv, method="predict_proba", n_jobs=-1)[:, 1]
    best_threshold = find_best_threshold(y_dev, oof_proba)

    # --- Outer evaluation. ---
    bagged.fit(X_dev, y_dev)
    y_score = bagged.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= best_threshold).astype(int)

    fold_metrics = compute_metrics(y_test, y_pred, y_score)
    fold_metrics["decision_threshold"] = best_threshold
    print(f"[{model_name}] {fold_label} | best_params={search.best_params_} | "
          f"threshold={best_threshold:.3f} | acc={fold_metrics['accuracy']:.3f} | "
          f"sens={fold_metrics['recall_sensitivity']:.3f} | spec={fold_metrics['specificity']:.3f}")
    return fold_metrics, y_test, y_pred, y_score


def get_outer_splits(y: np.ndarray, groups: np.ndarray, cfg: CONFIG):
    """Precompute the outer-fold splits ONCE and reuse them across every
    model, so the model-comparison table isn't confounded by different
    models seeing different train/test partitions."""
    splits = []
    for repeat in range(cfg.N_REPEATS):
        seed = cfg.SEED + repeat
        outer_cv = StratifiedGroupKFold(n_splits=cfg.N_OUTER_FOLDS, shuffle=True, random_state=seed)
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(np.zeros(len(y)), y, groups), start=1):
            splits.append((repeat, fold_idx, seed, train_idx, test_idx))
    return splits


def run_nested_cv_for_model(model_name: str, X_full: np.ndarray, y: np.ndarray, groups: np.ndarray,
                             outer_splits, cfg: CONFIG):
    all_fold_metrics = []
    pooled_y_true, pooled_y_pred, pooled_y_score = [], [], []

    for repeat, fold_idx, seed, train_idx, test_idx in outer_splits:
        X_dev, X_test = X_full[train_idx], X_full[test_idx]
        y_dev, y_test = y[train_idx], y[test_idx]
        groups_dev = groups[train_idx]

        fold_label = f"Repeat {repeat + 1}/{cfg.N_REPEATS} | Fold {fold_idx}/{cfg.N_OUTER_FOLDS}"
        fold_metrics, y_t, y_p, y_s = run_one_outer_fold(
            model_name, X_dev, y_dev, groups_dev, X_test, y_test, cfg, seed, fold_label
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
    for key in ["accuracy", "precision", "recall_sensitivity", "specificity", "roc_auc"]:
        vals = [m[key] for m in all_fold_metrics if key in m]
        if vals:
            summary[f"{key}_mean"] = float(np.mean(vals))
            summary[f"{key}_std"] = float(np.std(vals))

    return pooled, summary, all_fold_metrics, (pooled_y_true, pooled_y_pred, pooled_y_score)


# --------------------------------------------------------------------------- #
# 9. Comparison table across all models
# --------------------------------------------------------------------------- #
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall_sensitivity": "Sensitivity (Recall)",
    "specificity": "Specificity",
    "roc_auc": "ROC-AUC",
}


def build_comparison_table(all_summaries: Dict[str, Dict]) -> pd.DataFrame:
    rows = []
    for model_name, summary in all_summaries.items():
        row = {"Model": model_name}
        for key, label in METRIC_LABELS.items():
            mean_key, std_key = f"{key}_mean", f"{key}_std"
            if mean_key in summary:
                row[label] = f"{summary[mean_key]:.3f} \u00b1 {summary[std_key]:.3f}"
            else:
                row[label] = "n/a"
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Model")
    return df


# --------------------------------------------------------------------------- #
# 10. Inline-only plots  (unchanged, now takes a title so multiple models
#     don't overwrite each other's figures)
# --------------------------------------------------------------------------- #
def show_plots(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray, model_name: str):
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Healthy", "MDD"], cmap="Blues", ax=ax
    )
    ax.set_title(f"{model_name} — pooled outer-fold confusion matrix")
    plt.tight_layout()
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(4, 4))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax2)
    ax2.set_title(f"{model_name} — pooled outer-fold ROC curve")
    plt.tight_layout()
    plt.show()


def plot_metric_comparison(all_summaries: Dict[str, Dict]):
    """One grouped bar chart, all models x all metrics, mean with std as
    error bars — the quickest way to eyeball which model wins where."""
    models = list(all_summaries.keys())
    metrics = list(METRIC_LABELS.keys())
    x = np.arange(len(models))
    width = 0.8 / len(metrics)

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.4), 5))
    for i, key in enumerate(metrics):
        means = [all_summaries[m].get(f"{key}_mean", np.nan) for m in models]
        stds = [all_summaries[m].get(f"{key}_std", 0.0) for m in models]
        ax.bar(x + i * width, means, width, yerr=stds, capsize=3, label=METRIC_LABELS[key])

    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison — mean \u00b1 std across outer folds")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
# 11. Final deployable model per model type (fit on full data)
# --------------------------------------------------------------------------- #
def fit_final_model(model_name: str, X_full: np.ndarray, y: np.ndarray, groups: np.ndarray, cfg: CONFIG):
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
    estimator, clf_grid = MODEL_REGISTRY[model_name](pos_weight)

    inner_cv = StratifiedGroupKFold(n_splits=cfg.N_INNER_FOLDS, shuffle=True, random_state=cfg.SEED)
    pipe = make_pipeline(n_scales=len(cfg.SCALES_SEC), estimator=clone(estimator))
    search = GridSearchCV(pipe, make_param_grid(cfg, clf_grid), cv=inner_cv,
                           scoring="accuracy", n_jobs=-1, refit=False)
    search.fit(X_full, y, groups=groups)

    fixed_pipe = make_pipeline(n_scales=len(cfg.SCALES_SEC), estimator=clone(estimator))
    fixed_pipe.set_params(**search.best_params_)
    final_model = make_bagging_classifier(fixed_pipe, cfg, cfg.SEED)
    final_model.fit(X_full, y)
    print(f"[{model_name}] final params: {search.best_params_} | inner CV acc={search.best_score_:.3f}")
    return final_model


# --------------------------------------------------------------------------- #
# 12. Main
# --------------------------------------------------------------------------- #
def main():
    records = index_dataset(CFG.DATA_ROOT)
    print(f"Indexed {len(records)} recordings across "
          f"{len(set(r['subject_id'] for r in records))} subjects")

    X_by_scale, y, groups = build_dataset(records, CFG)
    print(f"Usable recordings after feature extraction: {len(y)} "
          f"(feature dim per scale: {X_by_scale[CFG.SCALES_SEC[0]].shape[1]})")

    X_full = np.concatenate([X_by_scale[s] for s in CFG.SCALES_SEC], axis=1)

    models_to_run = [m for m in CFG.MODELS_TO_RUN if m in MODEL_REGISTRY]
    skipped = [m for m in CFG.MODELS_TO_RUN if m not in MODEL_REGISTRY]
    if skipped:
        print(f"[WARN] skipping unavailable models: {skipped}")

    outer_splits = get_outer_splits(y, groups, CFG)
    print(f"\nBenchmarking {len(models_to_run)} models over "
          f"{len(outer_splits)} outer evaluations each "
          f"({CFG.N_OUTER_FOLDS} outer x {CFG.N_INNER_FOLDS} inner, {CFG.N_REPEATS} repeat(s))\n")

    all_summaries: Dict[str, Dict] = {}
    all_pooled_arrays: Dict[str, Tuple] = {}

    for model_name in models_to_run:
        print(f"\n===== {model_name} =====")
        pooled, summary, fold_results, pooled_arrays = run_nested_cv_for_model(
            model_name, X_full, y, groups, outer_splits, CFG
        )
        all_summaries[model_name] = summary
        all_pooled_arrays[model_name] = pooled_arrays

        print(f"-- {model_name}: mean \u00b1 std across {len(fold_results)} outer folds --")
        for key, label in METRIC_LABELS.items():
            if f"{key}_mean" in summary:
                print(f"  {label}: {summary[f'{key}_mean']:.3f} \u00b1 {summary[f'{key}_std']:.3f}")

        pooled_y_true, pooled_y_pred, pooled_y_score = pooled_arrays
        show_plots(pooled_y_true, pooled_y_pred, pooled_y_score, model_name)

    # --- Final comparison table across all models ---
    comparison_df = build_comparison_table(all_summaries)
    print("\n================ MODEL COMPARISON (mean \u00b1 std, outer folds) ================")
    print(comparison_df.to_string())

    if CFG.RESULTS_CSV_PATH:
        comparison_df.to_csv(CFG.RESULTS_CSV_PATH)
        print(f"\nSaved comparison table to {os.path.abspath(CFG.RESULTS_CSV_PATH)}")

    plot_metric_comparison(all_summaries)

    # --- Fit + save a final deployable (bagged) model per model type ---
    print("\n=== Fitting final deployable models on the full dataset ===")
    if CFG.SAVE_MODEL_DIR:
        os.makedirs(CFG.SAVE_MODEL_DIR, exist_ok=True)
        import joblib
        for model_name in models_to_run:
            final_model = fit_final_model(model_name, X_full, y, groups, CFG)
            out_path = os.path.join(CFG.SAVE_MODEL_DIR, f"{model_name}_final.joblib")
            joblib.dump(final_model, out_path)
            print(f"Saved {model_name} -> {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
