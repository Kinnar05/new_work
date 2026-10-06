import subprocess, sys, os
QUICK = os.environ.get('NDDM_QUICK') == '1'
if not QUICK:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'mne', 'PyWavelets', 'tqdm', 'xgboost', 'h5py', 'openpyxl'], check=True)

import re, glob, math, random, time, json, hashlib, inspect, copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import mne
import pywt
import scipy
import sklearn
import xgboost
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, r2_score
from xgboost import XGBClassifier
from scipy.signal import hilbert
from scipy.fft import next_fast_len
from scipy.stats import mannwhitneyu
from tqdm.auto import tqdm

PIPELINE_VERSION = 'nddm-modma-rest-v1'            # keys the DATASET cache
MODEL_VERSION = 'modma-rest-v1-3x5cv-m3'           # keys the per-fold RESULT cache
mne.set_log_level('ERROR')
SEED = 42
FEATURE_SEED = SEED + 7


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


seed_everything(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)

WORK_DIR = os.environ.get('NDDM_WORK_DIR', '/tmp/nddm_modma_quick' if QUICK else '/kaggle/working')
CKPT_DIR = os.path.join(WORK_DIR, 'ckpt')
OUT_DIR = os.path.join(WORK_DIR, 'results_modma_rest')
os.makedirs(CKPT_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)

START_TIME = time.time()
def log(stage):
    print(f'[{(time.time() - START_TIME) / 60:6.1f} min] {stage}')

# EGI HydroCel-128 face / neck / outer-edge channels that are commonly excluded (VERIFY against your montage).
PERIPHERY = [1, 8, 14, 17, 21, 25, 32, 38, 43, 44, 48, 49, 56, 63, 64, 68, 69, 73, 74, 81, 82, 88, 89, 94, 95, 99,
             107, 113, 114, 119, 120, 121, 125, 126, 127, 128]

CONFIG = dict(
    data_dir=os.environ.get('NDDM_DATA_DIR', '/kaggle/input/datasets/kinnarhalder/modmaa/EEG_128channels_resting_lanzhou_2015'),
    conditions=('REST',),
    mat_sfreq=250.0,                              # MODMA 128-ch recordings are 250 Hz (not stored in the .mat)
    n_eeg_channels=128,                           # E1..E128 kept, row 129 (Cz reference) dropped
    channel_mode='no_periphery',                  # 'no_periphery' | 'all128'   (decide BEFORE seeing results)
    average_reference=True,
    resample_sfreq=128.0,
    bandpass=(0.5, 45.0),
    notch=50.0,
    edge_trim_sec=4.0,
    epoch_len_sec=2.0,
    epoch_overlap=0.5,
    amp_reject_uv=150.0,
    reject_sensitivity_uv=(100.0, 150.0, 200.0),
    max_epochs_per_file=60,
    bands=dict(delta=(0.5, 4.0), theta=(4.0, 8.0), alpha=(8.0, 13.0), beta=(13.0, 30.0), gamma=(30.0, 45.0)),
    freqs=tuple(float(f) for f in np.geomspace(2, 40, 12)),
    wavelet='morl',
    n_time_bins=32,
    # diffusion
    diffusion_timesteps=200,
    batch_size=32,
    unet_base_channels=32,
    diffusion_draws_per_subject=60,
    dev_eval_timesteps=(10, 50, 100, 150),
    dev_repeats=2,
    cv_normative_epochs=40,
    cross_fit_k=3,
    latent_align_per_subject=40,
    latent_align_alphas=(1.0, 10.0, 100.0, 1000.0),
    # CV
    cv_n_splits=5,
    cv_repeats=3,
    cv_inner_splits=3,
    classifiers=['ensemble', 'subj_lda', 'subj_logreg', 'transformer', 'lstm', 'bagged_logreg', 'logreg', 'xgboost', 'svm_rbf', 'random_forest'],
    conn_vst=True,                                # variance-stabilising transforms on the connectivity/Hjorth block (label-free)
    ensemble_members=('subj_lda', 'subj_logreg', 'bagged_logreg'),
    subj_logreg_C=0.05,
    feature_set='fused',                          # fused | diffusion_combined | diffusion_deviation | diffusion_latent | connectivity_only
    fusion_diffusion_source='diffusion_combined',
    klein_dim=32,
    klein_curvatures=(0.5, 1.0, 2.0),
    klein_tau=0.5,
    # LSTM
    lstm_hidden=32, lstm_dropout=0.3, lstm_lr=2e-3, lstm_weight_decay=1e-2, lstm_epochs=60, lstm_batch=8,
    # epoch-set transformer v2 (tiny: ~6k params + linear skip)
    tf_d_model=16, tf_heads=2, tf_layers=1, tf_ff=32, tf_dropout=0.3, tf_in_dropout=0.2,
    tf_lr=5e-4, tf_weight_decay=1e-1, tf_epochs=40, tf_batch=8, tf_max_tokens=40,
    tf_noise=0.2, tf_mix_alpha=0.4, tf_mix_prob=0.5, tf_logit_l2=1e-2, tf_ema=0.98, tf_n_seeds=3, tf_tta=8,
    n_boot=2000,
    resume=os.environ.get('NDDM_RESUME', '1') == '1',
)

if QUICK:
    CONFIG.update(diffusion_timesteps=20, dev_eval_timesteps=(2, 5, 10, 15), unet_base_channels=8,
                  batch_size=64, cv_normative_epochs=2, diffusion_draws_per_subject=8, max_epochs_per_file=8,
                  klein_dim=8, n_boot=100, lstm_epochs=3, latent_align_per_subject=6, cv_repeats=1,
                  tf_epochs=3, tf_n_seeds=1, tf_tta=2)
assert max(CONFIG['dev_eval_timesteps']) < CONFIG['diffusion_timesteps']
assert CONFIG['feature_set'] in ('fused', 'diffusion_combined', 'diffusion_deviation', 'diffusion_latent', 'connectivity_only')
assert CONFIG['channel_mode'] in ('no_periphery', 'all128')

PRE_KEYS = ['conditions', 'mat_sfreq', 'n_eeg_channels', 'channel_mode', 'average_reference', 'resample_sfreq',
            'bandpass', 'notch', 'edge_trim_sec', 'epoch_len_sec', 'epoch_overlap', 'amp_reject_uv',
            'reject_sensitivity_uv', 'max_epochs_per_file', 'bands', 'freqs', 'wavelet', 'n_time_bins']

CLF_LABEL = {'ensemble': 'Soft-vote Ensemble', 'subj_lda': 'Subject-level Shrinkage LDA',
             'subj_logreg': 'Subject-level Ridge LogReg', 'transformer': 'Epoch-set Transformer', 'lstm': 'LSTM', 'bagged_logreg': 'Bagged Logistic Regression',
             'logreg': 'Logistic Regression', 'xgboost': 'XGBoost', 'svm_rbf': 'SVM (RBF)',
             'random_forest': 'Random Forest'}

ALL_CH = [f'E{i}' for i in range(1, CONFIG['n_eeg_channels'] + 1)]
if CONFIG['channel_mode'] == 'no_periphery':
    KEEP_CH = [c for i, c in enumerate(ALL_CH, start=1) if i not in set(PERIPHERY)]
else:
    KEEP_CH = list(ALL_CH)
print(f'channel mode = {CONFIG["channel_mode"]}: {len(KEEP_CH)} channels kept')


def check_edge_trim(cfg):
    sf, worst = cfg['resample_sfreq'], 0.0
    for lo, hi in [tuple(cfg['bandpass'])] + [tuple(v) for v in cfg['bands'].values()]:
        h = mne.filter.create_filter(None, sf, lo, hi, fir_design='firwin', verbose='ERROR')
        worst = max(worst, len(h) / 2 / sf)
    if cfg['edge_trim_sec'] < worst:
        raise ValueError(f"edge_trim_sec={cfg['edge_trim_sec']} s is shorter than the longest FIR half-length ({worst:.2f} s).")
    print(f'edge trim check OK: {cfg["edge_trim_sec"]} s >= longest FIR half-length {worst:.2f} s')


check_edge_trim(CONFIG)

# ---------------- 1. file discovery + labels [M2] ----------------
def parse_filename(path):
    base = os.path.basename(path)
    m = re.match(r'^(\d{8})', base)
    if not m:
        return None
    return dict(subject=m.group(1), condition='REST', path=path, file=base)


def prefix_label(sid):
    """MODMA 128-ch convention: 0201xxxx = MDD outpatients, 0202xxxx / 0203xxxx = healthy controls."""
    return 1 if sid.startswith('0201') else (0 if sid[:4] in ('0202', '0203') else None)


def labels_from_xlsx(data_dir):
    xs = glob.glob(os.path.join(data_dir, '**', '*.xlsx'), recursive=True)
    if not xs:
        print('no xlsx found; using ID-prefix labels'); return {}
    try:
        df = pd.read_excel(xs[0], header=None)
    except Exception as e:
        print(f'could not read {xs[0]} ({e}); using ID-prefix labels'); return {}
    out = {}
    for _, r in df.iterrows():
        sid, lab = None, None
        for v in r.values:
            if pd.isna(v):
                continue
            s = re.sub(r'\.0$', '', str(v).strip()); u = s.upper()
            if sid is None and re.fullmatch(r'\d{7,8}', s):
                sid = s.zfill(8)
            if lab is None:
                if 'MDD' in u or u in ('PATIENT', 'PATIENTS'):
                    lab = 1
                elif u == 'HC' or u.startswith('HC') or 'CONTROL' in u or 'NORMAL' in u:
                    lab = 0
        if sid and lab is not None:
            out[sid] = lab
    print(f'xlsx labels parsed for {len(out)} subjects from {os.path.basename(xs[0])}')
    return out


def build_manifest(data_dir):
    rows, unparsed = [], []
    for f in glob.glob(os.path.join(data_dir, '**', '*.mat'), recursive=True):
        p = parse_filename(f)
        (rows if p else unparsed).append(p or f)
    if unparsed:
        print(f'WARNING: {len(unparsed)} .mat files without an 8-digit ID prefix ignored, e.g. {unparsed[:3]}')
    if not rows:
        raise FileNotFoundError(f'No parseable .mat files under {data_dir}')
    m = pd.DataFrame(rows).sort_values(['subject', 'path']).reset_index(drop=True)
    xl = labels_from_xlsx(data_dir)
    labs, mism = [], []
    for s in m['subject']:
        a, b = xl.get(s), prefix_label(s)
        if a is not None and b is not None and a != b:
            mism.append((s, a, b))
        labs.append(a if a is not None else b)
    if mism:
        print(f'WARNING: xlsx vs ID-prefix label disagreement (xlsx used): {sorted(set(mism))}')
    m['label'] = labs
    if m['label'].isna().any():
        print('WARNING: subjects without a label are dropped:', sorted(set(m.loc[m.label.isna(), 'subject'])))
        m = m[m.label.notna()].reset_index(drop=True)
    m['label'] = m['label'].astype(int)
    dup = m.duplicated(subset=['subject'], keep=False)
    if dup.any():
        print('NOTE: several files share one subject; ALL kept and grouped under the same subject:')
        print(m[dup][['subject', 'file']].to_string(index=False))
    assert m.groupby('subject')['label'].nunique().max() == 1, 'a subject maps to more than one label'
    return m


def config_hash(cfg, manifest_sig):
    payload = dict(version=PIPELINE_VERSION, code=code_fingerprint(), sig=manifest_sig, keep=KEEP_CH,
                   **{k: cfg[k] for k in PRE_KEYS})
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:10]

# ---------------- 2/3. loader [M1], preprocessing, epoching, features ----------------
def read_mat_array(path):
    """Largest 2-D numeric array in the .mat, oriented channels x time (channels = the smaller axis)."""
    try:
        d = loadmat(path)
        cand = {k: v for k, v in d.items() if not k.startswith('__') and isinstance(v, np.ndarray)
                and v.ndim == 2 and np.issubdtype(v.dtype, np.number)}
    except NotImplementedError:                       # MATLAB v7.3 (HDF5)
        import h5py
        cand = {}
        with h5py.File(path, 'r') as f:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset) and obj.ndim == 2 and np.issubdtype(obj.dtype, np.number):
                    cand[name] = np.array(obj).T
            f.visititems(visit)
    if not cand:
        raise ValueError('no 2-D numeric array in .mat')
    k = max(cand, key=lambda n: cand[n].size)
    x = cand[k]
    if x.shape[0] > x.shape[1]:
        x = x.T
    return x, k


def analytic_phase(x):
    n = x.shape[1]
    return np.angle(hilbert(x, N=next_fast_len(n), axis=1)[:, :n])


def hjorth_all(seg_uv):
    d1 = np.diff(seg_uv, axis=1); d2 = np.diff(d1, axis=1)
    v0, v1, v2 = seg_uv.var(axis=1), d1.var(axis=1), d2.var(axis=1)
    mob = np.sqrt(v1 / (v0 + 1e-12))
    comp = np.sqrt(v2 / (v1 + 1e-12)) / (mob + 1e-12)
    return np.concatenate([np.log10(v0 + 1e-12), mob, comp])


def plv_from_phase(ph):
    z = np.exp(1j * ph)
    m = np.abs(z @ z.conj().T) / ph.shape[1]
    return m[np.triu_indices(ph.shape[0], k=1)]


def epoch_to_scalogram(epoch, cfg):
    sf = cfg['resample_sfreq']
    scales = pywt.central_frequency(cfg['wavelet']) * sf / np.asarray(cfg['freqs'])
    out = []
    for ch in epoch:
        coeffs, _ = pywt.cwt(ch, scales, cfg['wavelet'], sampling_period=1.0 / sf)
        mag = np.abs(coeffs)
        out.append(mag[:, np.linspace(0, mag.shape[1] - 1, cfg['n_time_bins']).astype(int)])
    out = np.stack(out).astype(np.float32)
    return (out - out.mean()) / (out.std() + 1e-8)


def load_and_epoch(path, cfg, keep_channels):
    x, _ = read_mat_array(path)
    if x.shape[0] < cfg['n_eeg_channels']:
        raise ValueError(f'only {x.shape[0]} channel rows (<{cfg["n_eeg_channels"]})')
    x = x[:cfg['n_eeg_channels']].astype(np.float64)               # E1..E128; row 129 (reference) dropped
    scale = 1e-6 if np.median(x.std(axis=1)) > 1e-2 else 1.0        # file in uV -> volts; already volts -> unchanged
    info = mne.create_info(ALL_CH, cfg['mat_sfreq'], ch_types='eeg')
    raw = mne.io.RawArray(x * scale, info, verbose='ERROR')
    raw.pick(keep_channels); raw.reorder_channels(keep_channels)
    if cfg['average_reference']:
        raw.set_eeg_reference('average', projection=False, verbose='ERROR')
    if raw.info['sfreq'] != cfg['resample_sfreq']:
        raw.resample(cfg['resample_sfreq'])
    raw.filter(cfg['bandpass'][0], cfg['bandpass'][1], fir_design='firwin', verbose='ERROR')
    try:
        raw.notch_filter(cfg['notch'], verbose='ERROR')
    except Exception:
        pass
    data = raw.get_data()
    sf = raw.info['sfreq']
    band_phase = {}
    for name, (lo, hi) in cfg['bands'].items():
        band = mne.filter.filter_data(data.copy(), sf, lo, hi, fir_design='firwin', verbose='ERROR')
        band_phase[name] = analytic_phase(band)

    win = int(cfg['epoch_len_sec'] * sf)
    step = int(win * (1 - cfg['epoch_overlap']))
    trim = int(cfg['edge_trim_sec'] * sf)
    starts = list(range(trim, data.shape[1] - trim - win + 1, step))
    n_considered = len(starts)
    peak_uv = np.array([np.max(np.abs(data[:, s:s + win])) * 1e6 for s in starts])
    clean = [s for s, p in zip(starts, peak_uv) if p <= cfg['amp_reject_uv']]
    n_rejected = n_considered - len(clean)
    rej_counts = {int(t): int((peak_uv > t).sum()) for t in cfg['reject_sensitivity_uv']}
    cap = cfg['max_epochs_per_file']
    if len(clean) > cap:
        clean = [clean[i] for i in np.linspace(0, len(clean) - 1, cap).round().astype(int)]
    n_ch = len(keep_channels)
    n_pairs = n_ch * (n_ch - 1) // 2
    n_conn = 3 * n_ch + len(cfg['bands']) * n_pairs
    if not clean:
        return (np.empty((0, n_ch, len(cfg['freqs']), cfg['n_time_bins']), np.float32),
                np.empty((0, n_conn), np.float32), n_considered, n_rejected, rej_counts)
    scalos, conns = [], []
    for s in clean:
        seg = data[:, s:s + win] * 1e6
        z = (seg - seg.mean(axis=1, keepdims=True)) / (seg.std(axis=1, keepdims=True) + 1e-8)
        scalos.append(epoch_to_scalogram(z, cfg))
        hj = hjorth_all(seg)
        conns.append(np.concatenate([hj] + [plv_from_phase(band_phase[b][:, s:s + win]) for b in cfg['bands']]).astype(np.float32))
    return np.stack(scalos), np.stack(conns), n_considered, n_rejected, rej_counts


def code_fingerprint():
    try:
        fns = [parse_filename, prefix_label, read_mat_array, analytic_phase, hjorth_all, plv_from_phase,
               epoch_to_scalogram, load_and_epoch]
        return hashlib.sha1(''.join(inspect.getsource(f) for f in fns).encode()).hexdigest()[:8]
    except (OSError, TypeError):
        print('WARNING: source unavailable for hashing; bump PIPELINE_VERSION by hand when preprocessing code changes.')
        return 'nosrc'


def load_real_dataset(cfg):
    manifest = build_manifest(cfg['data_dir'])
    sig = [(os.path.relpath(p, cfg['data_dir']), os.path.getsize(p)) for p in sorted(manifest['path'])]
    h = config_hash(cfg, sig)
    log(f'manifest: {len(manifest)} files, {manifest["subject"].nunique()} subjects, hash={h}')
    sl = manifest.groupby('subject')['label'].first()
    print(f'subjects: MDD={int(sl.sum())}, HC={int((1 - sl).sum())}  (MODMA 128-ch expectation: MDD=24, HC=29)')
    try:
        _x, _k = read_mat_array(manifest['path'].iloc[0])
        print(f'first file: variable "{_k}" oriented as {_x.shape} (channels x samples); '
              f'median channel std = {np.median(_x.std(axis=1)):.3g} -> '
              f'{"treated as uV" if np.median(_x.std(axis=1)) > 1e-2 else "treated as V"}')
    except Exception as e:
        print('WARNING: first-file inspection failed:', e)
    ckpt = os.path.join(CKPT_DIR, f'dataset_{h}.npz')
    rej_path = os.path.join(CKPT_DIR, f'rejected_epochs_{h}.csv')
    skip_path = os.path.join(CKPT_DIR, f'skipped_files_{h}.csv')
    if os.path.exists(ckpt) and os.path.exists(rej_path):
        d = np.load(ckpt)
        if os.path.exists(skip_path):
            sk = pd.read_csv(skip_path)
            print(f'WARNING (cached run): {len(sk)} files were skipped -- report this in the paper.')
            sk.to_csv(os.path.join(OUT_DIR, 'skipped_files.csv'), index=False)
        return d['X'], d['conn'], d['y'], d['groups'], pd.read_csv(rej_path), h
    Xs, Cs, ys, gs, rej, skipped = [], [], [], [], [], []
    for _, row in tqdm(list(manifest.iterrows()), desc='loading+epoching'):
        try:
            sc, cn, n_cons, n_rej, rc = load_and_epoch(row['path'], cfg, KEEP_CH)
        except Exception as e:
            skipped.append((row['path'], str(e))); print(f"skip {row['path']}: {e}"); continue
        rej.append(dict(subject=row['subject'], file=row['file'], label=row['label'], n_considered=n_cons,
                        n_rejected=n_rej, n_kept=len(sc), **{f'rej_{t}': v for t, v in rc.items()}))
        if len(sc) == 0:
            continue
        Xs.append(sc); Cs.append(cn)
        ys += [row['label']] * len(sc); gs += [row['subject']] * len(sc)
    if not Xs:
        raise RuntimeError('No epochs survived preprocessing.')
    if skipped:
        print(f'WARNING: {len(skipped)} files skipped (sample size reduced) -- report this in the paper.')
        sk = pd.DataFrame(skipped, columns=['path', 'error'])
        sk.to_csv(os.path.join(OUT_DIR, 'skipped_files.csv'), index=False); sk.to_csv(skip_path, index=False)
    X, conn = np.concatenate(Xs), np.concatenate(Cs)
    y, groups = np.array(ys), np.array(gs)
    rej = pd.DataFrame(rej)
    np.savez_compressed(ckpt, X=X, conn=conn, y=y, groups=groups)
    rej.to_csv(rej_path, index=False)
    return X, conn, y, groups, rej, h


if not os.path.isdir(CONFIG['data_dir']):
    cand = glob.glob('/kaggle/input/**/*rest*.mat', recursive=True)
    if not cand:
        raise FileNotFoundError('No MODMA .mat files under /kaggle/input -- attach the dataset.')
    CONFIG['data_dir'] = os.path.dirname(cand[0]); print('Auto-detected data_dir:', CONFIG['data_dir'])
X, conn, y, groups, rejection_log, PRE_HASH = load_real_dataset(CONFIG)
COMMON_CHANNELS = list(KEEP_CH)
N_CHANNELS = X.shape[1]
if CONFIG['conn_vst']:
    # Label-free, per-feature transforms (no fitting -> no leakage). Layout: [log var | mobility | complexity | PLV x 5 bands]
    _o = 3 * N_CHANNELS
    np.log(conn[:, N_CHANNELS:_o] + 1e-12, out=conn[:, N_CHANNELS:_o])        # positive ratios -> log scale
    np.clip(conn[:, _o:], 0.0, 0.995, out=conn[:, _o:])
    np.arctanh(conn[:, _o:], out=conn[:, _o:])                                # Fisher z of PLV in [0,1)
    log('connectivity block variance-stabilised (log Hjorth ratios, Fisher-z PLV)')
log(f'dataset: {len(X)} epochs | {N_CHANNELS} ch | conn dim={conn.shape[1]}')

# ---- rejection accounting + group-bias tests ----
zero = rejection_log[rejection_log['n_kept'] == 0]
if len(zero):
    print(f'WARNING: {len(zero)} recordings yielded ZERO clean epochs and are NOT in the analysis: {list(zero["subject"])}')
rejection_log['reject_pct'] = 100 * rejection_log['n_rejected'] / rejection_log['n_considered'].clip(lower=1)
rej_tab = rejection_log.groupby('label').agg(files=('subject', 'count'), considered=('n_considered', 'sum'),
                                             rejected=('n_rejected', 'sum'), mean_file_reject_pct=('reject_pct', 'mean'))
rej_tab['pooled_reject_pct'] = 100 * rej_tab['rejected'] / rej_tab['considered'].clip(lower=1)
print(f'\nRejected windows by class (0=HC, 1=MDD), threshold {CONFIG["amp_reject_uv"]:.0f} uV:'); print(rej_tab.round(2).to_string())
acc_tab = rejection_log.groupby('label')['n_kept'].agg(['count', 'mean', 'min', 'max'])
print('\nAccepted epochs per recording by class:'); print(acc_tab.round(1).to_string())
if (rejection_log['n_kept'] < 10).any():
    print('WARNING: recordings with <10 accepted epochs:', list(rejection_log.loc[rejection_log['n_kept'] < 10, 'subject']))
a, b = rejection_log.loc[rejection_log.label == 1, 'reject_pct'], rejection_log.loc[rejection_log.label == 0, 'reject_pct']
if len(a) > 1 and len(b) > 1:
    print(f'Mann-Whitney U on per-file reject % (MDD vs HC): p={mannwhitneyu(a, b).pvalue:.3f}')
rej_tab.to_csv(os.path.join(OUT_DIR, 'rejection_summary.csv')); acc_tab.to_csv(os.path.join(OUT_DIR, 'accepted_epochs_by_class.csv'))
rejection_log.to_csv(os.path.join(OUT_DIR, 'rejected_epochs.csv'), index=False)

# ---------------- 4. diffusion model ----------------
def group_rows(g):
    d = {}
    for i, s in enumerate(g):
        d.setdefault(s, []).append(i)
    return {s: np.asarray(v) for s, v in d.items()}


def subject_labels(y_, g_):
    df = pd.DataFrame({'y': np.asarray(y_), 'g': np.asarray(g_)})
    assert (df.groupby('g')['y'].nunique() == 1).all(), 'a subject has more than one label'
    return df.groupby('g')['y'].first()


class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.norm1 = nn.GroupNorm(8, out_ch); self.norm2 = nn.GroupNorm(8, out_ch)
    def forward(self, x, emb):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.emb_proj(emb)[:, :, None, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class SmallUNet(nn.Module):
    def __init__(self, n_channels, base=32, emb_dim=128):
        super().__init__()
        self.time_emb = nn.Sequential(SinusoidalTimeEmb(emb_dim), nn.Linear(emb_dim, emb_dim), nn.SiLU())
        self.in_conv = nn.Conv2d(n_channels, base, 3, padding=1)
        self.down1 = ResBlock(base, base * 2, emb_dim); self.pool1 = nn.AvgPool2d(2)
        self.down2 = ResBlock(base * 2, base * 4, emb_dim); self.pool2 = nn.AvgPool2d(2)
        self.mid = ResBlock(base * 4, base * 4, emb_dim)
        self.up2 = ResBlock(base * 8, base * 2, emb_dim)
        self.up1 = ResBlock(base * 4, base, emb_dim)
        self.out_conv = nn.Conv2d(base, n_channels, 3, padding=1)
    def forward(self, x, t):
        emb = self.time_emb(t)
        h1 = self.down1(self.in_conv(x), emb)
        h2 = self.down2(self.pool1(h1), emb)
        m = self.mid(self.pool2(h2), emb)
        u2 = self.up2(torch.cat([F.interpolate(m, size=h2.shape[-2:], mode='nearest'), h2], dim=1), emb)
        u1 = self.up1(torch.cat([F.interpolate(u2, size=h1.shape[-2:], mode='nearest'), h1], dim=1), emb)
        return self.out_conv(u1)


class DDPMScheduler:
    def __init__(self, timesteps, beta_start=1e-4, beta_end=2e-2, device=DEVICE):
        self.T = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
    def add_noise(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bars[t][:, None, None, None]
        return torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise, noise


scheduler = DDPMScheduler(CONFIG['diffusion_timesteps'])


def train_diffusion(model, X_np, g_np, epochs, batch_size, seed, draws_per_subject=None, lr=2e-4):
    """SUBJECT-BASIS training: every subject contributes exactly `draws_per_subject` windows per pass."""
    model.to(DEVICE)
    draws = draws_per_subject or CONFIG['diffusion_draws_per_subject']
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    rows = group_rows(g_np)
    subs = list(rows)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    hist = []
    for ep in range(epochs):
        model.train(); total = 0.0
        idx = np.concatenate([rng.choice(rows[s], size=draws, replace=len(rows[s]) < draws) for s in subs])
        rng.shuffle(idx)
        for b in range(0, len(idx), batch_size):
            xb = torch.from_numpy(np.ascontiguousarray(X_np[idx[b:b + batch_size]])).to(DEVICE)
            t = torch.randint(0, scheduler.T, (xb.size(0),), device=DEVICE)
            x_noisy, noise = scheduler.add_noise(xb, t)
            loss = F.mse_loss(model(x_noisy, t), noise)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * xb.size(0)
        hist.append(total / len(idx))
    return model, hist


@torch.no_grad()
def diffusion_features(model, X_in, eval_timesteps, n_repeats, batch_size=64, want_latent=True, seed=FEATURE_SEED,
                       latent_only=False):
    """(deviation, latent). deviation = normative denoising error per eval timestep; latent = U-Net bottleneck
    (spatially averaged) at the last timestep. Dedicated seeded generator -> deterministic, global RNG untouched."""
    model.eval().to(DEVICE)
    ts_run = tuple(eval_timesteps)[-1:] if latent_only else tuple(eval_timesteps)
    want_latent = want_latent or latent_only
    gen = torch.Generator(device=DEVICE); gen.manual_seed(seed)
    dev = np.zeros((len(X_in), len(ts_run)), dtype=np.float32)
    lats, store = [], {}
    handle = model.mid.register_forward_hook(lambda m, i, o: store.__setitem__('z', o.detach())) if want_latent else None
    try:
        for s in range(0, len(X_in), batch_size):
            xb = torch.from_numpy(np.ascontiguousarray(X_in[s:s + batch_size])).to(DEVICE)
            lat_acc = None
            for j, tv in enumerate(ts_run):
                t = torch.full((xb.size(0),), tv, device=DEVICE, dtype=torch.long)
                errs = []
                for _ in range(n_repeats):
                    noise = torch.randn(xb.shape, device=DEVICE, generator=gen)
                    x_noisy, _ = scheduler.add_noise(xb, t, noise)
                    errs.append(F.mse_loss(model(x_noisy, t), noise, reduction='none').mean(dim=[1, 2, 3]).cpu().numpy())
                    if want_latent and j == len(ts_run) - 1:
                        z = store['z'].mean(dim=[2, 3]).cpu().numpy()
                        lat_acc = z if lat_acc is None else lat_acc + z
                dev[s:s + xb.size(0), j] = np.mean(errs, axis=0)
            if want_latent:
                lats.append(lat_acc / n_repeats)
    finally:
        if handle is not None:
            handle.remove()
    lat = np.concatenate(lats, axis=0).astype(np.float32) if want_latent else None
    return (None if latent_only else dev), lat

# ---------------- 4.5 normative bank with cross-fitting ----------------
class NormativeBank:
    def __init__(self, full, members, held_out):
        self.full, self.members, self.held_out = full, members, held_out
        self.aligners = []


def build_bank(full_model, healthy_idx, D, cfg, seed):
    k = cfg['cross_fit_k']
    g_h = D['g'][healthy_idx]
    subs = np.array(sorted(set(g_h)))
    if k < 2 or len(subs) < 2 * k:
        if k >= 2:
            print(f'WARNING: only {len(subs)} healthy training subjects; cross-fitting disabled for this bank.')
        return NormativeBank(full_model, [], {}), []
    parts = np.array_split(np.random.default_rng(seed).permutation(subs), k)
    members, held, hists = [], {}, []
    for j, part in enumerate(parts):
        keep = healthy_idx[~np.isin(g_h, part)]
        m = SmallUNet(N_CHANNELS, base=cfg['unet_base_channels'])
        m, h = train_diffusion(m, D['X'][keep], D['g'][keep], cfg['cv_normative_epochs'], cfg['batch_size'], seed + 100 * (j + 1))
        members.append(m); hists.append(h)
        for s in part:
            held[s] = j
    return NormativeBank(full_model, members, held), hists


def fit_latent_aligners(bank, healthy_idx, D, cfg, seed):
    """Map each member's latent into the FULL model's frame by ridge regression on healthy-train anchor windows
    (only subjects that member trained on). Returns held-out-anchor-subject R^2 diagnostics."""
    if not bank.members:
        return []
    rng = np.random.default_rng(seed + 555)
    rows = group_rows(D['g'][healthy_idx])
    n_per = cfg['latent_align_per_subject']
    pos = np.sort(np.concatenate([rng.choice(rows[s], size=min(n_per, len(rows[s])), replace=False) for s in rows]))
    idx = healthy_idx[pos]
    Xa, ga = D['X'][idx], D['g'][idx]
    ts, nr, alphas = cfg['dev_eval_timesteps'], cfg['dev_repeats'], cfg['latent_align_alphas']
    z0 = diffusion_features(bank.full, Xa, ts, nr, latent_only=True)[1]
    out, bank.aligners = [], []
    for j, m in enumerate(bank.members):
        zj = diffusion_features(m, Xa, ts, nr, latent_only=True)[1]
        seen = np.array([bank.held_out[s] != j for s in ga])
        sc = StandardScaler().fit(zj[seen])
        rg = RidgeCV(alphas=alphas).fit(sc.transform(zj[seen]), z0[seen])
        bank.aligners.append((sc, rg))
        subs = np.array(sorted(set(ga[seen])))
        a_rows = seen & np.isin(ga, subs[::2]); b_rows = seen & ~np.isin(ga, subs[::2])
        r2 = np.nan
        if a_rows.sum() > 10 and b_rows.sum() > 10:
            sc2 = StandardScaler().fit(zj[a_rows])
            rg2 = RidgeCV(alphas=alphas).fit(sc2.transform(zj[a_rows]), z0[a_rows])
            r2 = float(r2_score(z0[b_rows], rg2.predict(sc2.transform(zj[b_rows])), multioutput='variance_weighted'))
        out.append(dict(member=j, n_anchor_fit=int(seen.sum()), alpha=float(rg.alpha_), r2_heldout_anchor_subjects=r2))
    return out


def bank_features(bank, X_in, g_in):
    """Cross-fitted deviation AND aligned latent. Healthy training subjects -> the member that never saw them;
    everyone else -> mean over all members."""
    ts, nr = CONFIG['dev_eval_timesteps'], CONFIG['dev_repeats']
    dev = np.zeros((len(X_in), len(ts)), np.float32); lat = None
    held = np.array([bank.held_out.get(s, -1) for s in g_in])

    def run(j, idx):
        d, z = diffusion_features(bank.members[j], X_in[idx], ts, nr, want_latent=True)
        sc, rg = bank.aligners[j]
        return d, rg.predict(sc.transform(z)).astype(np.float32)
    for j in range(len(bank.members)):
        idx = np.where(held == j)[0]
        if len(idx):
            d, z = run(j, idx)
            if lat is None: lat = np.zeros((len(X_in), z.shape[1]), np.float32)
            dev[idx], lat[idx] = d, z
    idx = np.where(held == -1)[0]
    if len(idx):
        outs = [run(j, idx) for j in range(len(bank.members))]
        if lat is None: lat = np.zeros((len(X_in), outs[0][1].shape[1]), np.float32)
        dev[idx] = np.mean([o[0] for o in outs], axis=0); lat[idx] = np.mean([o[1] for o in outs], axis=0)
    return dev, lat


def stream(bank, X_in, g_in):
    dev_full, lat_full = diffusion_features(bank.full, X_in, CONFIG['dev_eval_timesteps'], CONFIG['dev_repeats'])
    if not bank.members:
        return dict(dev=dev_full, lat=lat_full, dev_full=dev_full, lat_full=lat_full)
    dev, lat = bank_features(bank, X_in, g_in)
    return dict(dev=dev, lat=lat, dev_full=dev_full, lat_full=lat_full)


def feature_dict(s, conn_in):
    return dict(connectivity_only=conn_in, diffusion_deviation=s['dev'], diffusion_latent=s['lat'],
                diffusion_combined=np.hstack([s['dev'], s['lat']]))


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def _latent_gap(lat_tr, g_tr, y_tr, lat_te, g_te, y_te):
    hc = y_tr == 0
    mu, sd = lat_tr[hc].mean(0), lat_tr[hc].std(0) + 1e-6
    def hc_subject_means(lat, g, yy):
        t = pd.DataFrame({'v': np.linalg.norm((lat - mu) / sd, axis=1), 'g': g, 'y': yy}).groupby('g').agg(v=('v', 'mean'), y=('y', 'first'))
        return t.v[t.y == 0]
    return cohen_d(hc_subject_means(lat_te, g_te, y_te), hc_subject_means(lat_tr, g_tr, y_tr))


def gap_row(fold, s_tr, s_te, y_tr, g_tr, y_te, g_te):
    """Held-out HC vs in-sample training HC (Cohen d). d_used / d_lat_used should be ~0 once cross-fitting is active."""
    def sm(v, g, yy):
        return pd.DataFrame({'v': v.mean(axis=1), 'g': g, 'y': yy}).groupby('g').agg(v=('v', 'mean'), y=('y', 'first'))
    tr_full, tr_used = sm(s_tr['dev_full'], g_tr, y_tr), sm(s_tr['dev'], g_tr, y_tr)
    te_full, te_used = sm(s_te['dev_full'], g_te, y_te), sm(s_te['dev'], g_te, y_te)
    hc = lambda t: t.v[t.y == 0]
    return dict(fold=fold, hc_train_full_model=hc(tr_full).mean(), hc_test_full_model=hc(te_full).mean(),
                d_full_model=cohen_d(hc(te_full), hc(tr_full)),
                hc_train_used=hc(tr_used).mean(), hc_test_used=hc(te_used).mean(),
                d_used=cohen_d(hc(te_used), hc(tr_used)),
                d_lat_full_model=_latent_gap(s_tr['lat_full'], g_tr, y_tr, s_te['lat_full'], g_te, y_te),
                d_lat_used=_latent_gap(s_tr['lat'], g_tr, y_tr, s_te['lat'], g_te, y_te))

# ---------------- 5. Klein-model fusion (fitted on the fit set only) ----------------
def _nrm(x): return np.linalg.norm(x, axis=1, keepdims=True)


def poincare_exp0(v, c, tau):
    vs = v * tau; sc = np.sqrt(c); r = np.maximum(_nrm(vs), 1e-8)
    return np.tanh(sc * r) * vs / (sc * r)


def poincare_log0(p, c, tau):
    sc = np.sqrt(c); r = np.maximum(_nrm(p), 1e-8)
    v = np.arctanh(np.minimum(sc * r, 1 - 1e-6)) * (p / r) / sc
    return v / tau


def poincare_to_klein(p, c): return 2 * p / (1 + c * np.sum(p * p, axis=1, keepdims=True))
def klein_to_poincare(k, c): return k / (1 + np.sqrt(np.maximum(1 - c * np.sum(k * k, axis=1, keepdims=True), 1e-12)))


def einstein_midpoint(ks, c):
    gam = [1 / np.sqrt(np.maximum(1 - c * np.sum(k * k, axis=1, keepdims=True), 1e-12)) for k in ks]
    return sum(g * k for g, k in zip(gam, ks)) / sum(gam)


class KleinFuser:
    def __init__(self, k, curvatures, tau):
        self.k, self.curv, self.tau, self.diag = k, curvatures, tau, {}
    def fit(self, A, B):
        self.sa, self.sb = StandardScaler().fit(A), StandardScaler().fit(B)
        self.k = int(min(self.k, A.shape[1], B.shape[1], A.shape[0] - 1))
        # randomized SVD: connectivity block is ~20k-dim with 128-ch data; 'full' would be needlessly slow
        self.pa = PCA(self.k, svd_solver='randomized', random_state=0).fit(self.sa.transform(A))
        self.pb = PCA(self.k, svd_solver='randomized', random_state=0).fit(self.sb.transform(B))
        za, zb = self._proj(A, B, scale=False)
        self.ra = float(np.sqrt(np.mean(np.sum(za ** 2, axis=1))))
        self.rb = float(np.sqrt(np.mean(np.sum(zb ** 2, axis=1))))
        self.transform(A, B, record=True)
        return self
    def _proj(self, A, B, scale=True):
        za, zb = self.pa.transform(self.sa.transform(A)), self.pb.transform(self.sb.transform(B))
        return (za / self.ra, zb / self.rb) if scale else (za, zb)
    def transform(self, A, B, record=False):
        za, zb = (v.astype(np.float64) for v in self._proj(A, B))
        hyp, diag = [], {}
        for c in self.curv:
            pa, pb = poincare_exp0(za, c, self.tau), poincare_exp0(zb, c, self.tau)
            mid = klein_to_poincare(einstein_midpoint([poincare_to_klein(pa, c), poincare_to_klein(pb, c)], c), c)
            hyp.append(poincare_log0(mid, c, self.tau))
            diag[f'radius_a_c{c}'] = float(np.mean(np.sqrt(c) * _nrm(pa)))
            diag[f'radius_b_c{c}'] = float(np.mean(np.sqrt(c) * _nrm(pb)))
        if record:
            self.diag = diag
        return np.hstack(hyp + [za, zb]).astype(np.float32)


def take(f, idx):
    return {k: v[idx] for k, v in f.items()}


class FeatureBuilder:
    """Classifier input from RAW modality blocks; for 'fused' the scaler + PCA + Klein fusion is fitted on the fit set ONLY
    (once for the outer training fold, once per inner training split)."""
    def __init__(self, cfg):
        self.cfg, self.fuser = cfg, None
    def fit(self, f):
        if self.cfg['feature_set'] == 'fused':
            self.fuser = KleinFuser(self.cfg['klein_dim'], self.cfg['klein_curvatures'], self.cfg['klein_tau']).fit(
                f[self.cfg['fusion_diffusion_source']], f['connectivity_only'])
        return self
    def transform(self, f):
        if self.fuser is None:
            return f[self.cfg['feature_set']]
        return self.fuser.transform(f[self.cfg['fusion_diffusion_source']], f['connectivity_only'])

# ---------------- 6. classifiers, subject-level scoring, metrics ----------------
class SeqLSTM(nn.Module):
    def __init__(self, d_in, hidden, dropout):
        super().__init__()
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_in, hidden), nn.ReLU())
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))
    def forward(self, x, lengths):
        h = self.proj(x)
        packed = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        mask = (torch.arange(x.size(1))[None, :] < lengths[:, None]).float().unsqueeze(-1)
        return self.head((out * mask).sum(1) / mask.sum(1)).squeeze(-1)


def _pad(batch):
    lens = torch.tensor([len(s) for s in batch], dtype=torch.long)
    x = torch.zeros(len(batch), int(lens.max()), batch[0].shape[1])
    for i, s in enumerate(batch):
        x[i, :len(s)] = torch.from_numpy(s)
    return x, lens


def fit_lstm(seqs, ys, seed, cfg):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = SeqLSTM(seqs[0].shape[1], cfg['lstm_hidden'], cfg['lstm_dropout'])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lstm_lr'], weight_decay=cfg['lstm_weight_decay'])
    n_pos = max(1, int(ys.sum())); n_neg = max(1, len(ys) - n_pos)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / n_pos, dtype=torch.float32))
    yt = torch.tensor(ys, dtype=torch.float32)
    model.train()
    for _ in range(cfg['lstm_epochs']):
        order = rng.permutation(len(seqs))
        for b in range(0, len(order), cfg['lstm_batch']):
            ids = order[b:b + cfg['lstm_batch']]
            batch = []
            for i in ids:
                s = seqs[i]; L = len(s)
                k = int(rng.integers(max(1, L // 2), L + 1)); st = int(rng.integers(0, L - k + 1))
                batch.append(s[st:st + k])
            x, lens = _pad(batch)
            loss = lossf(model(x, lens), yt[ids])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model.eval()


@torch.no_grad()
def predict_lstm(model, seqs):
    out = []
    for b in range(0, len(seqs), 16):
        x, lens = _pad(seqs[b:b + 16])
        out.append(torch.sigmoid(model(x, lens)).numpy())
    return np.concatenate(out)


class EpochTransformer(nn.Module):
    """Linear skip + zero-initialised transformer residual over a subject's epoch set (no CLS token).
        logit(bag) = w.mean_t(x_t) + b  +  head( masked_mean_t( Enc(W x_t) ) ),   head initialised at 0.
    Training starts as a strongly-regularised linear model on the bag mean and the attention branch only
    learns what the linear model cannot. The token projection has NO LayerNorm, so token norm (e.g. the
    hyperbolic radius of the Klein-fused features) stays available to the network."""
    def __init__(self, d_in, d_model, n_heads, n_layers, ff, dropout, in_dropout):
        super().__init__()
        self.in_drop = nn.Dropout(in_dropout)
        self.lin = nn.Linear(d_in, 1); nn.init.normal_(self.lin.weight, std=0.01); nn.init.zeros_(self.lin.bias)
        self.proj = nn.Linear(d_in, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ff, dropout, activation='gelu',
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1); nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x, lengths):
        B, L, _ = x.shape
        valid = (torch.arange(L)[None, :] < lengths[:, None]).float().unsqueeze(-1)     # B,L,1 (1 = real epoch)
        n = lengths[:, None].float()
        xd = self.in_drop(x)
        lin = self.lin((xd * valid).sum(1) / n).squeeze(-1)
        h = self.enc(self.proj(xd), src_key_padding_mask=(valid.squeeze(-1) == 0))
        pooled = (self.norm(h) * valid).sum(1) / n
        return lin + self.head(self.drop(pooled)).squeeze(-1)


def _bag(seq, rng, cap, random_size=True):
    L = len(seq)
    hi = min(L, cap)
    k = int(rng.integers(min(max(1, L // 2), hi), hi + 1)) if random_size else hi
    return seq[np.sort(rng.choice(L, size=k, replace=False))]


def _mixed_bag(seqs, ys, i, rng, cap, alpha, prob):
    """Token-level SET mixup (vicinal risk): lam ~ Beta(alpha, alpha); take ~lam*k epochs from subject i and the rest
    from a random partner j. The soft target is the exact fraction of tokens that came from a positive subject."""
    if alpha <= 0 or rng.random() >= prob:
        return _bag(seqs[i], rng, cap), float(ys[i])
    j = int(rng.integers(len(seqs)))
    k = int(min(cap, max(2, max(len(seqs[i]), len(seqs[j])) // 2)))
    ka = int(round(float(rng.beta(alpha, alpha)) * k)); kb = k - ka
    a = seqs[i][rng.choice(len(seqs[i]), size=min(ka, len(seqs[i])), replace=False)]
    b = seqs[j][rng.choice(len(seqs[j]), size=min(kb, len(seqs[j])), replace=False)]
    bag = np.concatenate([a, b], axis=0)
    return bag, float((ys[i] * len(a) + ys[j] * len(b)) / len(bag))


def fit_transformer(seqs, ys, seed, cfg):
    """Returns a list of EMA-averaged models (seed ensemble)."""
    n = len(ys); n_pos = max(1, int(ys.sum())); n_neg = max(1, n - n_pos)
    w_pos, w_neg = n / (2 * n_pos), n / (2 * n_neg)                  # classes contribute equally to the loss
    steps = cfg['tf_epochs'] * math.ceil(len(seqs) / cfg['tf_batch'])
    warm = max(1, int(0.1 * steps))
    models = []
    for m in range(cfg['tf_n_seeds']):
        s = seed + 1000 * (m + 1)
        torch.manual_seed(s); rng = np.random.default_rng(s)
        model = EpochTransformer(seqs[0].shape[1], cfg['tf_d_model'], cfg['tf_heads'], cfg['tf_layers'],
                                 cfg['tf_ff'], cfg['tf_dropout'], cfg['tf_in_dropout'])
        ema = copy.deepcopy(model)
        for p in ema.parameters():
            p.requires_grad_(False)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg['tf_lr'], weight_decay=cfg['tf_weight_decay'])
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda i: min(1.0, (i + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, i / steps)))))
        model.train(); t = 0
        for _ in range(cfg['tf_epochs']):
            order = rng.permutation(len(seqs))
            for b in range(0, len(order), cfg['tf_batch']):
                ids = order[b:b + cfg['tf_batch']]
                pairs = [_mixed_bag(seqs, ys, i, rng, cfg['tf_max_tokens'], cfg['tf_mix_alpha'], cfg['tf_mix_prob']) for i in ids]
                x, lens = _pad([p[0] for p in pairs])
                tgt = torch.tensor([p[1] for p in pairs], dtype=torch.float32)
                x = x + cfg['tf_noise'] * torch.randn_like(x) * (torch.arange(x.size(1))[None, :, None] < lens[:, None, None])
                logit = model(x, lens)
                wts = tgt * w_pos + (1 - tgt) * w_neg
                loss = (wts * F.binary_cross_entropy_with_logits(logit, tgt, reduction='none')).mean() \
                       + cfg['tf_logit_l2'] * (logit ** 2).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
                d = min(cfg['tf_ema'], (1 + t) / (10 + t)); t += 1                 # EMA of weights (Polyak averaging)
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.mul_(d).add_(pm.detach(), alpha=1 - d)
        models.append(ema.eval())
    return models


@torch.no_grad()
def predict_transformer(models, seqs, cfg):
    """Mean probability over the seed ensemble and `tf_tta` random epoch bags per subject (deterministic)."""
    rng = np.random.default_rng(0)
    out = np.zeros(len(seqs))
    for _ in range(cfg['tf_tta']):
        bags = [_bag(s, rng, cfg['tf_max_tokens'], random_size=False) for s in seqs]
        for model in models:
            for b in range(0, len(bags), 16):
                x, lens = _pad(bags[b:b + 16])
                out[b:b + 16] += torch.sigmoid(model(x, lens)).numpy()
    return out / (cfg['tf_tta'] * len(models))


def make_classifier(name, seed, y_fit):
    if name == 'bagged_logreg':
        return BaggingClassifier(estimator=LogisticRegression(max_iter=2000, class_weight='balanced'),
                                 n_estimators=25, random_state=seed)
    if name == 'logreg':
        return LogisticRegression(max_iter=2000, class_weight='balanced')
    if name == 'random_forest':
        return RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=seed, n_jobs=-1)
    if name == 'svm_rbf':
        return SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=seed)
    if name == 'xgboost':
        n_pos = max(1, int((y_fit == 1).sum())); n_neg = max(1, int((y_fit == 0).sum()))
        return XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5,
                             min_child_weight=2, reg_lambda=1.0, scale_pos_weight=n_neg / n_pos, eval_metric='logloss',
                             tree_method='hist', n_jobs=4, random_state=seed, verbosity=0)
    raise ValueError(name)


def subject_summary(A, rows, subs):
    """Subject-level descriptor: [mean, std] over the subject's epochs (std = within-subject variability).
    Averaging n epochs cuts epoch-level noise by ~sqrt(n) before the classifier ever sees it."""
    return np.stack([np.concatenate([A[rows[s]].mean(0), A[rows[s]].std(0)]) for s in subs]).astype(np.float32)


def _score_one(clf_name, A, ytr, gtr, B, gte, seed):
    if clf_name in ('lstm', 'transformer'):
        tr_rows, te_rows = group_rows(gtr), group_rows(gte)
        subj_tr, subj_te = list(tr_rows), list(te_rows)
        seqs = [A[tr_rows[s]] for s in subj_tr]
        ys = np.array([ytr[tr_rows[s][0]] for s in subj_tr])
        te_seqs = [B[te_rows[s]] for s in subj_te]
        if clf_name == 'lstm':
            return np.array(subj_te), predict_lstm(fit_lstm(seqs, ys, seed, CONFIG), te_seqs)
        return np.array(subj_te), predict_transformer(fit_transformer(seqs, ys, seed, CONFIG), te_seqs, CONFIG)
    if clf_name in ('subj_lda', 'subj_logreg'):
        tr_rows, te_rows = group_rows(gtr), group_rows(gte)
        subj_tr, subj_te = list(tr_rows), list(te_rows)
        S_tr, S_te = subject_summary(A, tr_rows, subj_tr), subject_summary(B, te_rows, subj_te)
        ys = np.array([ytr[tr_rows[s][0]] for s in subj_tr])
        s2 = StandardScaler().fit(S_tr)
        S_tr, S_te = s2.transform(S_tr), s2.transform(S_te)
        if clf_name == 'subj_lda':
            # Ledoit-Wolf shrinkage covariance: Sigma = (1-d) S + d (tr S / p) I, d chosen analytically (p >> n regime); equal priors
            clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=np.array([0.5, 0.5]))
        else:
            clf = LogisticRegression(C=CONFIG['subj_logreg_C'], class_weight='balanced', max_iter=2000)
        clf.fit(S_tr, ys)
        return np.array(subj_te), clf.predict_proba(S_te)[:, 1]
    clf = make_classifier(clf_name, seed, ytr).fit(A, ytr)
    p = pd.Series(clf.predict_proba(B)[:, 1]).groupby(np.asarray(gte), sort=False).mean()
    return p.index.values, p.values


def score_subjects(clf_name, Ftr, ytr, gtr, Fte, gte, seed):
    sc = StandardScaler().fit(Ftr)
    A, B = sc.transform(Ftr).astype(np.float32), sc.transform(Fte).astype(np.float32)
    if clf_name == 'ensemble':
        # soft vote: mean of member probabilities (variance reduction; bounded so an over-confident member cannot dominate)
        ps = []
        for m in CONFIG['ensemble_members']:
            subj, p = _score_one(m, A, ytr, gtr, B, gte, seed)
            ps.append(pd.Series(np.asarray(p, float), index=subj))
        P = pd.concat(ps, axis=1).mean(axis=1)
        return P.index.values, P.values
    return _score_one(clf_name, A, ytr, gtr, B, gte, seed)


def tune_threshold(inner_sets, ytr, gtr, clf_name, seed):
    """Youden's-J threshold on SUBJECT-level inner out-of-fold probabilities, training fold only."""
    ps, ls = [], []
    for itr, iva, Fi_tr, Fi_va in inner_sets:
        subj, p = score_subjects(clf_name, Fi_tr, ytr[itr], gtr[itr], Fi_va, gtr[iva], seed)
        ps.append(p); ls.append(subject_labels(ytr[iva], gtr[iva]).loc[subj].values)
    p, l = np.concatenate(ps), np.concatenate(ls)
    if len(set(l)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(l, p)
    return float(thr[np.argmax(tpr - fpr)])


def classification_metrics(y_, p, pred):
    y_, pred = np.asarray(y_), np.asarray(pred)
    tp = int(((pred == 1) & (y_ == 1)).sum()); tn = int(((pred == 0) & (y_ == 0)).sum())
    fp = int(((pred == 1) & (y_ == 0)).sum()); fn = int(((pred == 0) & (y_ == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return dict(auc=roc_auc_score(y_, p) if len(np.unique(y_)) > 1 else np.nan,
                accuracy=(tp + tn) / max(1, len(y_)),
                precision=tp / (tp + fp) if tp + fp else 0.0,
                recall=sens, sensitivity=sens, specificity=spec,
                f1=2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
                mcc=(tp * tn - fp * fn) / den if den else 0.0,
                balanced_accuracy=float(np.nanmean([sens, spec])), tp=tp, tn=tn, fp=fp, fn=fn)

# ---------------- 7. one fold ----------------
def convergence_stat(h, k=5):
    if len(h) < 2 * k:
        return np.nan
    a, b = np.mean(h[-k:]), np.mean(h[-2 * k:-k])
    return (b - a) / b


def fold_stage(rep, fold, tr_subj, te_subj, D, cfg):
    """Healthy-only normative model + cross-fit bank + aligned latents -> cross-fitted streams -> embedding -> classifiers."""
    fid = rep * cfg['cv_n_splits'] + fold
    seed = SEED + fid + 1
    seed_everything(seed)
    Xd, y_, g_ = D['X'], D['y'], D['g']
    tr_idx, te_idx = np.where(np.isin(g_, tr_subj))[0], np.where(np.isin(g_, te_subj))[0]
    y_tr, y_te, g_tr, g_te = y_[tr_idx], y_[te_idx], g_[tr_idx], g_[te_idx]
    t0, tag = time.time(), f'[REST] rep {rep + 1} fold {fold + 1}'

    healthy = tr_idx[y_tr == 0]
    norm_model = SmallUNet(N_CHANNELS, base=cfg['unet_base_channels'])
    norm_model, hist = train_diffusion(norm_model, Xd[healthy], g_[healthy], cfg['cv_normative_epochs'], cfg['batch_size'], seed)
    loss_rows = [dict(analysis='REST', repeat=rep, fold=fold, model='normative', epoch=i + 1, loss=v) for i, v in enumerate(hist)]
    log(f'{tag}: normative model on {len(set(g_[healthy]))} healthy subjects; loss {hist[0]:.4f} -> {hist[-1]:.4f}; '
        f'last-5 vs prev-5 improvement = {100 * convergence_stat(hist):.2f}%')

    bank, member_hists = build_bank(norm_model, healthy, D, cfg, seed)
    for j, h in enumerate(member_hists):
        loss_rows += [dict(analysis='REST', repeat=rep, fold=fold, model=f'normative_member{j}', epoch=i + 1, loss=v) for i, v in enumerate(h)]
    align_rows = fit_latent_aligners(bank, healthy, D, cfg, seed)
    for r in align_rows:
        r.update(analysis='REST', repeat=rep, fold=fold)
    if align_rows:
        log(f'{tag}: latent alignment R^2 (held-out anchor subjects) = '
            f'{np.nanmean([r["r2_heldout_anchor_subjects"] for r in align_rows]):.2f}')

    s_tr, s_te = stream(bank, Xd[tr_idx], g_tr), stream(bank, Xd[te_idx], g_te)
    gr = gap_row(fold, s_tr, s_te, y_tr, g_tr, y_te, g_te); gr['analysis'] = 'REST'; gr['repeat'] = rep
    log(f'{tag}: HC in-sample-vs-heldout gap d: deviation {gr["d_full_model"]:+.2f} -> {gr["d_used"]:+.2f}, '
        f'latent {gr["d_lat_full_model"]:+.2f} -> {gr["d_lat_used"]:+.2f}')
    tr_f, te_f = feature_dict(s_tr, D['conn'][tr_idx]), feature_dict(s_te, D['conn'][te_idx])

    fb = FeatureBuilder(cfg).fit(tr_f)
    Ftr, Fte = fb.transform(tr_f), fb.transform(te_f)
    fus_diag = dict(analysis='REST', repeat=rep, fold=fold, common_dim=fb.fuser.k, **fb.fuser.diag) if fb.fuser is not None else None
    inner_sets = []
    for itr, iva in StratifiedGroupKFold(n_splits=cfg['cv_inner_splits'], shuffle=True, random_state=seed).split(Ftr, y_tr, g_tr):
        fbi = FeatureBuilder(cfg).fit(take(tr_f, itr))
        inner_sets.append((itr, iva, fbi.transform(take(tr_f, itr)), fbi.transform(take(tr_f, iva))))

    fold_rows, subj_preds = [], []
    for cn in cfg['classifiers']:
        subj, sp = score_subjects(cn, Ftr, y_tr, g_tr, Fte, g_te, seed)
        lab = subject_labels(y_te, g_te).loc[subj].values.astype(int)
        thr = tune_threshold(inner_sets, y_tr, g_tr, cn, seed)
        pred = (sp >= thr).astype(int)
        m = classification_metrics(lab, sp, pred)
        fold_rows.append(dict(analysis='REST', repeat=rep, fold=fold, classifier=cn, n_test_subjects=len(lab), threshold=thr, **m))
        subj_preds.append(pd.DataFrame(dict(analysis='REST', repeat=rep, fold=fold, classifier=cn, subject=subj, y=lab, p=sp, thr=thr, pred=pred)))
        log(f'{tag}: {CLF_LABEL[cn]:<27s} subject AUC={m["auc"]:.3f}  acc={m["accuracy"]:.3f}')
    del bank
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dict(fold_rows=fold_rows, subj_preds=pd.concat(subj_preds, ignore_index=True), loss_rows=loss_rows,
                gap_row=gr, fusion_diag=fus_diag, align_rows=align_rows, minutes=(time.time() - t0) / 60)

# ---------------- 8. repeated subject-wise CV ----------------
def model_fingerprint():
    try:
        fns = [group_rows, subject_labels, SmallUNet.forward, ResBlock.forward, train_diffusion, diffusion_features,
               build_bank, fit_latent_aligners, bank_features, stream, feature_dict, _latent_gap, gap_row, take,
               FeatureBuilder.fit, FeatureBuilder.transform, KleinFuser.fit, KleinFuser.transform,
               poincare_exp0, poincare_log0, SeqLSTM.forward, fit_lstm, EpochTransformer.forward, _bag, _mixed_bag,
               fit_transformer, predict_transformer, make_classifier, subject_summary, _score_one, score_subjects,
               tune_threshold, classification_metrics, fold_stage]
        src = ''.join(inspect.getsource(f) for f in fns)
    except (OSError, TypeError):
        print('WARNING: source unavailable for hashing; bump MODEL_VERSION by hand when modelling code changes.')
        src = 'nosrc'
    cfg = json.dumps({k: v for k, v in CONFIG.items() if k not in ('data_dir', 'resume', 'n_boot')}, sort_keys=True, default=str)
    return hashlib.sha1((PIPELINE_VERSION + MODEL_VERSION + PRE_HASH + src + cfg).encode()).hexdigest()[:10]


RUN_HASH = model_fingerprint()
log(f'run hash {RUN_HASH} (per-fold cache in {CKPT_DIR}; resume={CONFIG["resume"]})')

ysub_all = subject_labels(y, groups)
subj_arr = np.array(sorted(ysub_all.index))
ysub = ysub_all.loc[subj_arr]
N_SUBJ, N_MDD = len(subj_arr), int(ysub.sum())
assert min(N_MDD, N_SUBJ - N_MDD) >= CONFIG['cv_n_splits'], 'too few subjects per class for the outer CV'
R, K = CONFIG['cv_repeats'], CONFIG['cv_n_splits']
print('\n' + '#' * 78)
print(f'ANALYSIS: MODMA REST, {N_SUBJ} subjects (MDD={N_MDD}, HC={N_SUBJ - N_MDD}), {N_CHANNELS} channels | '
      f'{R} x {K}-fold StratifiedGroupKFold (groups = subject ID) | feature set = {CONFIG["feature_set"]}')
print('#' * 78)
D = dict(X=X, conn=conn, y=y, g=groups)

ST = dict(fold_rows=[], subj=[], loss=[], gap=[], fus=[], align=[], meta=[])
for rep in range(R):
    outer = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=SEED + rep)
    seen_test = []
    for fold, (a_, b_) in enumerate(outer.split(np.zeros(N_SUBJ), ysub.values, groups=subj_arr)):
        tr_subj, te_subj = subj_arr[a_], subj_arr[b_]
        yt_ = ysub.loc[te_subj]
        assert yt_.nunique() == 2, f'repeat {rep} fold {fold}: test set has a single class'
        assert not (set(tr_subj) & set(te_subj)), 'subject leakage across train/test'
        seen_test += list(te_subj)
        ST['meta'].append(dict(repeat=rep, fold=fold, train_subjects=len(tr_subj), test_subjects=len(te_subj),
                               test_mdd=int(yt_.sum()), test_hc=int((1 - yt_).sum())))
        log(f'REPEAT {rep + 1}/{R} FOLD {fold + 1}/{K}: train={len(tr_subj)}, test={len(te_subj)} '
            f'(MDD={ST["meta"][-1]["test_mdd"]}, HC={ST["meta"][-1]["test_hc"]})')
        cache = os.path.join(CKPT_DIR, f'stage_r{rep}_f{fold}_{RUN_HASH}.joblib')
        if CONFIG['resume'] and os.path.exists(cache):
            r = joblib.load(cache); log(f'rep {rep + 1} fold {fold + 1}: loaded from cache')
        else:
            r = fold_stage(rep, fold, tr_subj, te_subj, D, CONFIG)
            joblib.dump(r, cache)
            log(f'rep {rep + 1} fold {fold + 1} done in {r["minutes"]:.1f} min')
        ST['fold_rows'] += r['fold_rows']; ST['subj'].append(r['subj_preds']); ST['loss'] += r['loss_rows']
        ST['gap'].append(r['gap_row']); ST['align'] += r['align_rows']
        if r['fusion_diag'] is not None:
            ST['fus'].append(r['fusion_diag'])
    assert sorted(seen_test) == sorted(subj_arr.tolist()), 'every subject must be tested exactly once per repeat'

fold_df = pd.DataFrame(ST['fold_rows']); subj_df = pd.concat(ST['subj'], ignore_index=True)
loss_df = pd.DataFrame(ST['loss']); gap_df = pd.DataFrame(ST['gap'])
fold_df.to_csv(os.path.join(OUT_DIR, 'per_fold_metrics.csv'), index=False)
subj_df.to_csv(os.path.join(OUT_DIR, 'oof_subject_predictions.csv'), index=False)
loss_df.to_csv(os.path.join(OUT_DIR, 'loss_curves.csv'), index=False)
gap_df.to_csv(os.path.join(OUT_DIR, 'normative_gap_diagnostics.csv'), index=False)
pd.DataFrame(ST['meta']).to_csv(os.path.join(OUT_DIR, 'fold_composition.csv'), index=False)
if ST['fus']:
    pd.DataFrame(ST['fus']).to_csv(os.path.join(OUT_DIR, 'klein_fusion_diagnostics.csv'), index=False)
align_df = pd.DataFrame(ST['align'])
if len(align_df):
    align_df.to_csv(os.path.join(OUT_DIR, 'latent_alignment_diagnostics.csv'), index=False)

# ---------------- 9. out-of-fold results: per repeat, averaged over repeats ----------------
METRICS = ['auc', 'accuracy', 'precision', 'recall', 'sensitivity', 'specificity', 'f1', 'mcc', 'balanced_accuracy']
COUNTS = ['tp', 'tn', 'fp', 'fn']
subs_all = sorted(subj_df.subject.unique())
assert len(subs_all) == N_SUBJ


def oof_by_repeat(cn):
    dd = subj_df[subj_df.classifier == cn]
    out = []
    for _, d in dd.groupby('repeat'):
        assert d['subject'].is_unique and len(d) == N_SUBJ, 'each subject must be scored exactly once per repeat'
        d = d.set_index('subject').loc[subs_all]
        out.append((d['y'].values.astype(int), d['p'].values, d['pred'].values.astype(int)))
    assert len(out) == R
    return out


def bootstrap_ci(reps, n_boot, seed=SEED):
    """Cluster bootstrap: SUBJECTS resampled (each keeps all its repeats); metrics per repeat, averaged over repeats."""
    rng = np.random.default_rng(seed); n = len(reps[0][0]); rows = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(np.unique(reps[0][0][i])) < 2:
            continue
        ms = [classification_metrics(y_[i], p[i], pr[i]) for y_, p, pr in reps]
        rows.append({k: float(np.mean([m[k] for m in ms])) for k in METRICS})
    b = pd.DataFrame(rows)
    return {k: (b[k].quantile(.025), b[k].quantile(.975)) for k in METRICS}


pooled_rows, byrep_rows = [], []
for cn in CONFIG['classifiers']:
    reps = oof_by_repeat(cn)
    per_rep = pd.DataFrame([classification_metrics(*r_) for r_ in reps])
    for rp, mrow in per_rep.iterrows():
        byrep_rows.append(dict(classifier=cn, repeat=rp, **mrow.to_dict()))
    ci = bootstrap_ci(reps, CONFIG['n_boot'])
    row = dict(classifier=cn, feature_set=CONFIG['feature_set'], n_subjects=N_SUBJ, n_repeats=R, **per_rep[METRICS + COUNTS].mean().to_dict())
    for k in METRICS:
        row[f'{k}_repeat_sd'] = float(per_rep[k].std(ddof=1)) if R > 1 else np.nan
        row[f'{k}_ci_lo'], row[f'{k}_ci_hi'] = ci[k]
    pooled_rows.append(row)
pooled = pd.DataFrame(pooled_rows)
pooled.to_csv(os.path.join(OUT_DIR, 'pooled_subject_level_metrics.csv'), index=False)
pd.DataFrame(byrep_rows).to_csv(os.path.join(OUT_DIR, 'subject_level_metrics_by_repeat.csv'), index=False)


def to_md(df):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    lines += ['| ' + ' | '.join(str(v) for v in r_.values) + ' |' for _, r_ in df.iterrows()]
    return '\n'.join(lines)


rows = []
for cn in CONFIG['classifiers']:
    r = pooled[pooled.classifier == cn].iloc[0]
    rows.append({'Classifier': CLF_LABEL[cn],
                 'AUC [95% CI]': f'{r["auc"]:.3f} [{r["auc_ci_lo"]:.3f}, {r["auc_ci_hi"]:.3f}]',
                 'AUC SD (repeats)': f'{r["auc_repeat_sd"]:.3f}',
                 'Accuracy': f'{r["accuracy"]:.3f}', 'Precision': f'{r["precision"]:.3f}',
                 'Recall': f'{r["recall"]:.3f}', 'Specificity': f'{r["specificity"]:.3f}',
                 'F1': f'{r["f1"]:.3f}', 'MCC': f'{r["mcc"]:.3f}', 'Bal. Acc.': f'{r["balanced_accuracy"]:.3f}'})
final_tab = pd.DataFrame(rows)
CV_DESC = f'{R} x {K}-fold StratifiedGroupKFold (groups = subject ID, stratified by MDD/HC), new shuffle per repeat'
print('\n' + '=' * 110)
print(f'FINAL RESULTS | MODMA REST | {N_SUBJ} subjects (MDD={N_MDD}, HC={N_SUBJ - N_MDD}) | {N_CHANNELS} ch | feature set: '
      f'{CONFIG["feature_set"]} | {CV_DESC}\nSUBJECT-level out-of-fold metrics, MEAN over {R} repeats, 95% cluster-bootstrap CI (positive = MDD)')
print('=' * 110)
print(final_tab.to_string(index=False))
final_tab.to_csv(os.path.join(OUT_DIR, 'final_classification_table.csv'), index=False)
with open(os.path.join(OUT_DIR, 'final_classification_table.md'), 'w') as f:
    f.write(f'MODMA resting state, {N_CHANNELS} channels, feature set {CONFIG["feature_set"]}; {CV_DESC}; {N_SUBJ} subjects; '
            f'subject-level out-of-fold metrics averaged over {R} repeats (CI = cluster bootstrap over subjects).\n\n' + to_md(final_tab) + '\n')

# ---------------- 10. figures + diagnostics ----------------
FPR_GRID = np.linspace(0, 1, 201)
fig, ax = plt.subplots(figsize=(6, 5.5))
for cn in CONFIG['classifiers']:
    tprs, aucs = [], []
    for y_, p, _ in oof_by_repeat(cn):
        fp_, tp_, _t = roc_curve(y_, p)
        tprs.append(np.interp(FPR_GRID, fp_, tp_)); aucs.append(roc_auc_score(y_, p))
    t = np.mean(tprs, axis=0); t[0] = 0.0
    ax.plot(FPR_GRID, t, lw=1.6, label=f'{CLF_LABEL[cn]} ({np.mean(aucs):.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
ax.set(xlabel='1 - specificity', ylabel='sensitivity', title=f'MODMA REST: mean OOF subject-level ROC ({R} repeats)')
ax.legend(loc='lower right', fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'roc_rest.png'), dpi=300); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
for _, d in loss_df[loss_df.model == 'normative'].groupby(['repeat', 'fold']):
    ax.plot(d['epoch'], d['loss'], lw=0.9, alpha=0.6)
ax.set(xlabel='subject-balanced epoch', ylabel='noise-prediction MSE', title=f'normative diffusion loss ({R * K} fits)')
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'loss_curves.png'), dpi=200); plt.close(fig)

if ST['fus']:
    print(f'\nKlein fusion radii, mean over {R * K} folds (~0 = Euclidean, ->1 = boundary saturation):')
    print(pd.DataFrame(ST['fus']).drop(columns=['repeat', 'fold', 'analysis']).mean(numeric_only=True).round(3).to_string())
print(f'\nNormative in-sample vs held-out healthy gap (Cohen d), mean over {R * K} folds (d_*_used should be ~0):')
print(gap_df[['d_full_model', 'd_used', 'd_lat_full_model', 'd_lat_used']].mean().round(3).to_string())
if len(align_df):
    r2m = float(align_df['r2_heldout_anchor_subjects'].mean())
    print(f'\nLatent alignment R^2 on held-out anchor subjects (mean): {r2m:.3f}')
    if r2m < 0.5:
        print("WARNING: weak latent alignment -- consider fusion_diffusion_source='diffusion_deviation' (no latent stream).")

# ---------------- 11. run manifest ----------------
with open(os.path.join(OUT_DIR, 'run_manifest.json'), 'w') as f:
    json.dump(dict(pipeline=PIPELINE_VERSION, model_version=MODEL_VERSION, pre_hash=PRE_HASH, run_hash=RUN_HASH,
                   code_fingerprint=code_fingerprint(), seed=SEED, config=CONFIG, channels=COMMON_CHANNELS,
                   n_subjects=N_SUBJ, subjects=subj_arr.tolist(), n_epochs=int(len(X)), folds=ST['meta'],
                   versions=dict(torch=torch.__version__, mne=mne.__version__, numpy=np.__version__,
                                 pandas=pd.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__,
                                 xgboost=xgboost.__version__, pywt=pywt.__version__, python=sys.version.split()[0])),
              f, indent=2, default=str)
log('DONE. Results in ' + OUT_DIR)
