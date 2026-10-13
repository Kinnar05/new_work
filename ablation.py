# ================================================================
# ABLATION STUDY of SelectBest_inner  --  STANDALONE (no other script needed)
# Frozen multi-layer CBraMod -> cross-fitted diffusion streams (normative deviation + discriminative
# diffusion-consistency adapter) -> Klein fusion + microstate stream -> 35 candidate experts (5 streams x 7 classifiers)
# MODMA 128-ch resting EEG, MDD vs HC, repeated subject-wise CV (no leakage).
#
# SelectBest_inner := among the candidate experts, take the one with the highest inner-OOF AUC, calibrate it
# (balanced Platt) and threshold the calibrated probability at 0.5.   ONLY this method is evaluated here
# (no PoE fusions, no ensembles).
#
# Efficiency / exactness
#   * the expensive part (latent -> diffusion banks -> 35 expert probability tables) is built ONCE per outer fold
#     and cached; selector ablations are post-hoc re-selections on the cached inner-OOF / outer-test tables
#     (paired by construction, zero extra training).
#   * upstream ablations (Klein w/o delta, adapter losses, leave-one-layer-out) retrain only what they touch and swap
#     the affected expert columns into the pool.
#   * resumable (joblib per fold / variant).
#
# Ablation families
#   Pool       : leave-one-stream-out | single stream | leave-one-classifier-out | subject- vs epoch-level classifiers
#   Component  : Klein w/o delta input | adapter w/o (disc+cons) loss | adapter w/o consistency loss | leave-one-layer-out
#   Selection  : w/o activity gate | stability-penalised criterion | random active expert | oracle (post-hoc, optimistic)
#   Operating  : calibrated@0.5 (default) | raw prob + inner Youden | raw prob @0.5
#
# Env switches: CBR_QUICK=1 (tiny smoke test) | CBR_SYNTH=1 (synthetic data, no files/ckpt) | CBR_LAYERS="4,8,12"
#               CBR_RESUME=0/1 | ABL_DELTA=0 (skip adapter-loss ablations) | ABL_LAYERS=none (skip leave-one-layer-out)
# Outputs: <WORK_DIR>/results_selectbest_ablation/
# ================================================================
import os, re, sys, glob, math, json, copy, time, random, hashlib, warnings, subprocess
from dataclasses import dataclass
from typing import Optional, Tuple
warnings.filterwarnings('ignore')


def ensure_import(import_name, pip_name=None):
    try:
        return __import__(import_name)
    except ImportError:
        print(f'Installing {pip_name or import_name}')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pip_name or import_name])
        return __import__(import_name)


np = ensure_import('numpy')
pd = ensure_import('pandas')
mne = ensure_import('mne')
scipy = ensure_import('scipy')
h5py = ensure_import('h5py')
ensure_import('openpyxl')
sklearn = ensure_import('sklearn', 'scikit-learn')
joblib = ensure_import('joblib')
xgboost = ensure_import('xgboost')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from scipy.special import expit, logit
from scipy.stats import mannwhitneyu, rankdata
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier

QUICK = os.environ.get('CBR_QUICK') == '1'
SYNTH = os.environ.get('CBR_SYNTH') == '1'
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mne.set_log_level('ERROR')
MODEL_VERSION = 'cbramod-ml-dcadapter-ms-selpoe-v3-ablation-selectbest'
ABL_DELTA = os.environ.get('ABL_DELTA', '1') == '1'
ABL_LAYERS = os.environ.get('ABL_LAYERS', 'loo')


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)
START = time.time()
def log(msg):
    print(f'[{(time.time() - START) / 60:6.1f} min] {msg}', flush=True)

# ================================================================
# 0. CONFIG
# ================================================================
CBRAMOD_CKPT = '/kaggle/input/datasets/kinnarhalder/cbramod/pretrained_weights.pth'
DATA_DIR = '/kaggle/input/datasets/kinnarhalder/modmaa/EEG_128channels_resting_lanzhou_2015'
WORK_DIR = '/tmp/cbr_quick' if (QUICK or SYNTH) else '/kaggle/working'
OUT_DIR = os.path.join(WORK_DIR, 'results_selectbest_ablation')
CKPT_DIR = os.path.join(WORK_DIR, 'ckpt_cbramod_poe_v3')          # same folder as v3 -> re-uses its latent cache
CK_DIR = os.path.join(WORK_DIR, 'ckpt_selectbest_ablation')
for d_ in (OUT_DIR, CKPT_DIR, CK_DIR):
    os.makedirs(d_, exist_ok=True)
EXPECTED_SHA256 = '0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178'

PRE = dict(
    mat_sfreq=250.0, sfreq=200.0, bandpass=(0.5, 45.0), average_reference=True,
    window_sec=2.0, overlap=0.5, edge_trim_sec=4.0,
    window_ch_peak_uv=300.0, max_bad_ch_frac=0.25,
    cont_bad_scale_hi=6.0, cont_bad_scale_lo=0.15, cont_bad_abs_uv=500.0, cont_bad_abs_frac=0.01,
    max_cont_bad_interp=5, max_windows_per_subject=8 if QUICK else 60, cbramod_scale_uv=100.0,
)
EMBED_BATCH = 64
Q = QUICK or SYNTH
N_CBR_LAYERS = 12
LAYERS = tuple(sorted(int(v) for v in os.environ.get('CBR_LAYERS', '4,8,12').split(',')))
assert all(1 <= l <= N_CBR_LAYERS for l in LAYERS)

CFG = dict(
    layers=LAYERS, layer_pca_dim=16 if Q else 32, n_blocks=len(LAYERS),
    T=200, eval_timesteps=(10, 50, 100, 150), eval_repeats=2, cross_fit_k=3,
    NORM=dict(hidden=256, n_res=3, epochs=3 if Q else 80, batch=128, draws=60, lr=3e-4, wd=1e-4, ema=0.995),
    DELTA=dict(hidden=128, n_res=2, rank=16, adapter_dropout=0.1, epochs=3 if Q else 40, batch=256, draws=24,
               lr=3e-4, wd=1e-2, tmax=160, kappa0=10.0, w_den=1.0, w_base=0.25, w_disc=1.0, w_cons=0.5),
    member_scoring='single',
    klein_dim=16, klein_curvatures=(0.5, 1.0, 2.0), klein_tau=0.5, klein_source='both',
    subj_logreg_C=0.05,
    MS=dict(k=4, n_init=2 if Q else 8, iters=50, per_subj=250, min_seg_ms=30.0, tol=1e-5),
    logit_clip=1e-6, platt_C=1.0,
    sel_boot=50 if Q else 300, sel_lambda=1.0,
    cv_n_splits=5, cv_repeats=1 if Q else 3, cv_inner_splits=3 if Q else 4,
    n_boot=50 if Q else 1000,
    resume=os.environ.get('CBR_RESUME', '1') == '1',
)
STREAMS = ('cbramod', 'cbr_delta', 'ndm', 'klein', 'microstate')
CLFS = ('subj_lda', 'subj_logreg', 'bagged_logreg', 'logreg', 'svm_rbf', 'random_forest', 'xgboost')
SUBJ_LEVEL = {'subj_lda', 'subj_logreg'}
EPOCH_LEVEL = {'bagged_logreg', 'logreg', 'svm_rbf', 'random_forest', 'xgboost'}
STREAM_LABEL = {'cbramod': 'CBraMod latent', 'cbr_delta': 'adapter stream (latent+delta)', 'ndm': 'normative deviation',
                'klein': 'Klein fusion', 'microstate': 'microstate'}
METRICS = ['auc', 'accuracy', 'balanced_accuracy', 'precision', 'recall', 'specificity', 'f1', 'mcc']
assert max(CFG['eval_timesteps']) < CFG['T'] and max(CFG['eval_timesteps']) <= CFG['DELTA']['tmax'] < CFG['T']

CBRAMOD_CHANNELS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
EGI_128_TO_STD = {'Fp1': 22, 'Fp2': 9, 'F7': 33, 'F3': 24, 'Fz': 11, 'F4': 124, 'F8': 122, 'T3': 45, 'C3': 36, 'C4': 104,
                  'T4': 108, 'T5': 58, 'P3': 52, 'Pz': 62, 'P4': 92, 'T6': 96, 'O1': 70, 'O2': 83}
assert set(CBRAMOD_CHANNELS) == set(EGI_128_TO_STD)

if not SYNTH:
    if not os.path.exists(CBRAMOD_CKPT):
        hits = glob.glob('/kaggle/input/**/pretrained_weights.pth', recursive=True)
        if not hits:
            raise FileNotFoundError(f'CBraMod checkpoint not found: {CBRAMOD_CKPT}')
        CBRAMOD_CKPT = hits[0]; print('Auto-detected checkpoint:', CBRAMOD_CKPT)
    if not os.path.isdir(DATA_DIR):
        cand = glob.glob('/kaggle/input/**/*rest*.mat', recursive=True)
        if not cand:
            raise FileNotFoundError(f'MODMA directory not found: {DATA_DIR}')
        DATA_DIR = os.path.dirname(cand[0]); print('Auto-detected data dir:', DATA_DIR)
print('Device:', DEVICE, '| torch', torch.__version__, '| layers', LAYERS)

# ================================================================
# 1. CBraMod backbone (matches released checkpoint) + multi-layer read-out
# ================================================================
class CrissCrossEncoderLayer(nn.Module):
    def __init__(self, d_model=200, nhead=8, dim_feedforward=800, dropout=0.1):
        super().__init__()
        self.self_attn_s = nn.MultiheadAttention(d_model // 2, nhead // 2, dropout=dropout, batch_first=True)
        self.self_attn_t = nn.MultiheadAttention(d_model // 2, nhead // 2, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout); self.dropout2 = nn.Dropout(dropout)

    def _attention(self, x):
        b, c, s, d = x.shape
        h = d // 2
        xs, xt = x[..., :h], x[..., h:]
        xs = xs.transpose(1, 2).contiguous().view(b * s, c, h)
        xs = self.self_attn_s(xs, xs, xs, need_weights=False)[0]
        xs = xs.view(b, s, c, h).transpose(1, 2).contiguous()
        xt = xt.contiguous().view(b * c, s, h)
        xt = self.self_attn_t(xt, xt, xt, need_weights=False)[0]
        xt = xt.view(b, c, s, h)
        return self.dropout1(torch.cat([xs, xt], dim=-1))

    def forward(self, src):
        x = src + self._attention(self.norm1(src))
        ff = self.linear2(self.dropout(F.gelu(self.linear1(self.norm2(x)))))
        return x + self.dropout2(ff)


class CrissCrossEncoder(nn.Module):
    def __init__(self, layer, n_layers=12):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])
    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, in_dim=200, d_model=200):
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(19, 7), stride=1, padding=(9, 3), groups=d_model))
        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)
        self.proj_in = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)), nn.GroupNorm(5, 25), nn.GELU(),
            nn.Conv2d(25, 25, kernel_size=(1, 3), padding=(0, 1)), nn.GroupNorm(5, 25), nn.GELU(),
            nn.Conv2d(25, 25, kernel_size=(1, 3), padding=(0, 1)), nn.GroupNorm(5, 25), nn.GELU())
        self.spectral_proj = nn.Sequential(nn.Linear(101, d_model), nn.Dropout(0.1))

    def forward(self, x):
        b, c, s, p = x.shape
        if p != 200:
            raise ValueError(f'CBraMod expects 200 samples per 1-s patch, got {p}')
        flat = x.contiguous().view(b, 1, c * s, p)
        emb = self.proj_in(flat).permute(0, 2, 1, 3).contiguous().view(b, c, s, self.d_model)
        spec = torch.abs(torch.fft.rfft(flat.contiguous().view(b * c * s, p), dim=-1, norm='forward')).view(b, c, s, 101)
        emb = emb + self.spectral_proj(spec)
        pos = self.positional_encoding(emb.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return emb + pos


class CBraMod(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = PatchEmbedding(200, 200)
        self.encoder = CrissCrossEncoder(CrissCrossEncoderLayer(200, 8, 800, 0.1), N_CBR_LAYERS)
        self.proj_out = nn.Sequential(nn.Linear(200, 200))

    def forward(self, x):
        return self.proj_out(self.encoder(self.patch_embedding(x)))

    def forward_layers(self, x, layers):
        """Hidden states [b, c, s, d] after each requested 1-based encoder layer (last layer goes through proj_out)."""
        h = self.patch_embedding(x)
        want, outs = set(layers), []
        for i, l in enumerate(self.encoder.layers, 1):
            h = l(h)
            if i in want:
                outs.append(self.proj_out(h) if i == N_CBR_LAYERS else h)
            if i >= max(layers):
                break
        return outs


def sha256_file(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(block), b''):
            h.update(b)
    return h.hexdigest()


def load_cbramod():
    digest = sha256_file(CBRAMOD_CKPT)
    print('CBraMod checkpoint SHA256:', digest)
    if digest != EXPECTED_SHA256:
        print('WARNING: checkpoint hash differs from the official file this script was verified against.')
    model = CBraMod().to(DEVICE)
    try:
        sd = torch.load(CBRAMOD_CKPT, map_location=DEVICE, weights_only=True)
    except TypeError:
        sd = torch.load(CBRAMOD_CKPT, map_location=DEVICE)
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    print('Strict checkpoint load:', model.load_state_dict(sd, strict=True))
    model.proj_out = nn.Identity()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    with torch.no_grad():                                           # sanity: layer read-out consistent with full forward
        z = torch.zeros(2, 18, int(PRE['window_sec']), 200, device=DEVICE)
        o = model(z); hs = model.forward_layers(z, LAYERS)
        assert tuple(o.shape) == (2, 18, int(PRE['window_sec']), 200) and len(hs) == len(LAYERS)
        assert (LAYERS[-1] != N_CBR_LAYERS) or torch.allclose(hs[-1], o)
    return model

# ================================================================
# 2. MODMA discovery, labels, preprocessing, multi-layer latent extraction
# ================================================================
def prefix_label(sid):
    """MODMA 128-ch convention: 0201xxxx = MDD, 0202xxxx / 0203xxxx = HC."""
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
    print(f'xlsx labels parsed for {len(out)} subjects')
    return out


def build_manifest(data_dir):
    rows = []
    for f in glob.glob(os.path.join(data_dir, '**', '*.mat'), recursive=True):
        m = re.match(r'^(\d{8})', os.path.basename(f))
        if m:
            rows.append(dict(subject=m.group(1), path=f, file=os.path.basename(f)))
        else:
            print('ignored (no 8-digit ID):', os.path.basename(f))
    if not rows:
        raise FileNotFoundError(f'No parseable .mat files under {data_dir}')
    man = pd.DataFrame(rows).sort_values(['subject', 'path']).reset_index(drop=True)
    xl = labels_from_xlsx(data_dir)
    labs, mism = [], []
    for s in man['subject']:
        a, b = xl.get(s), prefix_label(s)
        if a is not None and b is not None and a != b:
            mism.append((s, a, b))
        labs.append(a if a is not None else b)
    if mism:
        print('WARNING: xlsx vs ID-prefix label disagreement (xlsx used):', sorted(set(mism)))
    man['label'] = labs
    if man['label'].isna().any():
        print('WARNING: unlabeled subjects dropped:', sorted(set(man.loc[man.label.isna(), 'subject'])))
        man = man[man.label.notna()].reset_index(drop=True)
    man['label'] = man['label'].astype(int)
    assert man.groupby('subject')['label'].nunique().max() == 1, 'a subject maps to >1 label'
    return man


def read_mat_array(path):
    try:
        d = loadmat(path)
        cand = {k: v for k, v in d.items() if not k.startswith('__') and isinstance(v, np.ndarray)
                and v.ndim == 2 and np.issubdtype(v.dtype, np.number)}
    except NotImplementedError:
        cand = {}
        with h5py.File(path, 'r') as f:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset) and obj.ndim == 2 and np.issubdtype(obj.dtype, np.number):
                    cand[name] = np.asarray(obj)
            f.visititems(visit)
    if not cand:
        raise ValueError('no 2-D numeric array in .mat')
    k = max(cand, key=lambda n: cand[n].size)
    x = np.asarray(cand[k])
    if x.shape[0] > x.shape[1]:
        x = x.T
    return x, k


def _mad_scale(x, axis=-1):
    med = np.median(x, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(x - med), axis=axis)


_pos = mne.channels.make_standard_montage('standard_1020').get_positions()['ch_pos']
_CH_POS = np.stack([np.asarray(_pos[c], float) for c in CBRAMOD_CHANNELS])
_CH_DIST = np.linalg.norm(_CH_POS[:, None] - _CH_POS[None], axis=2)


def _repair_epoch_channels(seg, bad_mask):
    seg = np.asarray(seg, np.float64).copy()
    bad, good = np.where(bad_mask)[0], np.where(~bad_mask)[0]
    if len(bad) == 0:
        return seg
    if len(good) < 3:
        return None
    for b in bad:
        d = _CH_DIST[b, good]; o = np.argsort(d)[:min(4, len(good))]
        w = 1.0 / np.maximum(d[o], 1e-6) ** 2; w /= w.sum()
        seg[b] = np.sum(seg[good[o]] * w[:, None], axis=0)
    return seg


def preprocess_one_mat(path):
    P = PRE
    x, var_name = read_mat_array(path)
    if x.shape[0] < 128:
        raise ValueError(f'only {x.shape[0]} channel rows (<128)')
    eeg = x[:128].astype(np.float64, copy=False)
    med_std = float(np.median(np.std(eeg, axis=1)))
    scale = 1e-6 if med_std > 1e-2 else 1.0
    sel = eeg[[EGI_128_TO_STD[c] - 1 for c in CBRAMOD_CHANNELS]] * scale
    raw = mne.io.RawArray(sel, mne.create_info(CBRAMOD_CHANNELS, P['mat_sfreq'], 'eeg'), verbose='ERROR')
    raw.set_montage(mne.channels.make_standard_montage('standard_1020'), match_case=True, on_missing='raise', verbose='ERROR')
    raw.filter(P['bandpass'][0], P['bandpass'][1], fir_design='firwin', verbose='ERROR')
    if abs(raw.info['sfreq'] - P['sfreq']) > 1e-6:
        raw.resample(P['sfreq'])

    pre = raw.get_data() * 1e6
    rs = _mad_scale(pre, axis=1); ratio = rs / (float(np.median(rs)) + 1e-8)
    abs_frac = np.mean(np.abs(pre) > P['cont_bad_abs_uv'], axis=1)
    bad = np.where((ratio > P['cont_bad_scale_hi']) | (ratio < P['cont_bad_scale_lo']) | (abs_frac > P['cont_bad_abs_frac']))[0]
    if len(bad) > P['max_cont_bad_interp']:
        sev = np.maximum(ratio, 1.0 / np.maximum(ratio, 1e-6)) + 5.0 * abs_frac
        bad = bad[np.argsort(sev[bad])[::-1][:P['max_cont_bad_interp']]]
    bad_names = [CBRAMOD_CHANNELS[i] for i in bad]
    if bad_names:
        raw.info['bads'] = bad_names
        raw.interpolate_bads(reset_bads=True, mode='accurate', verbose='ERROR')
    if P['average_reference']:
        raw.set_eeg_reference('average', projection=False, verbose='ERROR')

    data = raw.get_data() * 1e6
    win = int(round(P['window_sec'] * P['sfreq'])); step = int(round(win * (1 - P['overlap'])))
    trim = int(round(P['edge_trim_sec'] * P['sfreq']))
    patches = int(round(P['window_sec']))
    if win != patches * 200:
        raise RuntimeError('window must be an integer number of 1-s / 200-sample patches')
    starts = np.arange(trim, data.shape[1] - trim - win + 1, step, dtype=int)
    qc = dict(considered=len(starts), rejected=0, repaired=0, kept=0, persistent_bad=';'.join(bad_names),
              persistent_bad_count=len(bad_names), med_std=med_std, var_name=var_name)
    empty = np.empty((0, len(CBRAMOD_CHANNELS), patches, 200), np.float32)
    if len(starts) == 0:
        return empty, qc
    max_bad = int(math.floor(P['max_bad_ch_frac'] * len(CBRAMOD_CHANNELS)))
    kept = []
    for s0 in starts:
        seg = data[:, s0:s0 + win]
        badc = np.max(np.abs(seg), axis=1) > P['window_ch_peak_uv']
        nb = int(badc.sum())
        if nb > max_bad:
            qc['rejected'] += 1; continue
        if nb:
            seg = _repair_epoch_channels(seg, badc)
            if seg is None:
                qc['rejected'] += 1; continue
            qc['repaired'] += 1
        kept.append(seg.astype(np.float32))
    cap = P['max_windows_per_subject']
    if len(kept) > cap:
        kept = [kept[i] for i in np.linspace(0, len(kept) - 1, cap).round().astype(int)]
    qc['kept'] = len(kept)
    if not kept:
        return empty, qc
    out = np.stack([(k / P['cbramod_scale_uv']).reshape(len(CBRAMOD_CHANNELS), patches, 200) for k in kept])
    return out.astype(np.float32), qc


@torch.no_grad()
def extract_latent(model, windows):
    """-> [n, len(LAYERS), 18*200]: patch-mean-pooled hidden state of every requested layer."""
    feats = []
    for i in range(0, len(windows), EMBED_BATCH):
        x = torch.from_numpy(np.ascontiguousarray(windows[i:i + EMBED_BATCH])).to(DEVICE)
        hs = model.forward_layers(x, LAYERS)
        feats.append(torch.stack([h.mean(dim=2).flatten(1) for h in hs], dim=1).float().cpu().numpy())
    return np.concatenate(feats)


def build_latent_dataset(manifest):
    """Same cache key as the v3 script -> an existing v3 latent cache in CKPT_DIR is re-used (backbone only loaded on a miss)."""
    sig = json.dumps(dict(pre=PRE, ckpt=sha256_file(CBRAMOD_CKPT)[:12], layers=LAYERS, files=sorted(
        (os.path.basename(p), os.path.getsize(p)) for p in manifest['path']), v=3), sort_keys=True, default=str)
    key = hashlib.sha1(sig.encode()).hexdigest()[:10]
    cache = os.path.join(CKPT_DIR, f'cbramod_ml_latent_{key}.npz')
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        log(f'loaded latent cache {cache}')
        return d['X'].astype(np.float32), d['y'].astype(int), d['g'].astype(str), d['W'].astype(np.float32), key
    model = load_cbramod()
    XX, WW, yy, gg, audit = [], [], [], [], []
    for k, row in manifest.iterrows():
        try:
            w, qc = preprocess_one_mat(row.path)
            rec = dict(subject=row.subject, label=row.label, file=row.file, error='', **qc)
            if len(w) == 0:
                rec['error'] = 'zero usable windows'; audit.append(rec); print('ZERO WINDOWS:', row.file); continue
            z = extract_latent(model, w)
            XX.append(z); WW.append(w.reshape(len(w), len(CBRAMOD_CHANNELS), -1))
            yy += [row.label] * len(z); gg += [row.subject] * len(z); audit.append(rec)
            print(f'[{k + 1:02d}/{len(manifest)}] {row.subject} {"MDD" if row.label else "HC "} windows={len(z):2d} '
                  f'rejected={qc["rejected"]}/{qc["considered"]} repaired={qc["repaired"]} bad_ch={qc["persistent_bad_count"]}')
        except Exception as e:
            audit.append(dict(subject=row.subject, label=row.label, file=row.file, error=str(e)))
            print('SKIP', row.file, '->', e)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    audit_df = pd.DataFrame(audit); audit_df.to_csv(os.path.join(OUT_DIR, 'preprocessing_audit.csv'), index=False)
    if not XX:
        raise RuntimeError('No usable windows.')
    X = np.concatenate(XX).astype(np.float32); W = np.concatenate(WW).astype(np.float32)
    y = np.asarray(yy, int); g = np.asarray(gg, str)
    a = audit_df[audit_df.error == '']
    ra = (a.rejected / a.considered.clip(lower=1))
    if (a.label == 1).sum() > 1 and (a.label == 0).sum() > 1:
        print(f'reject-rate MDD vs HC Mann-Whitney p={mannwhitneyu(ra[a.label == 1], ra[a.label == 0]).pvalue:.3f}')
    np.savez_compressed(cache, X=X, y=y, g=g, W=W)
    return X, y, g, W, key


def synthetic_latent(n_mdd=24, n_hc=29, win=12, dim=800, seed=0):
    """Pipeline smoke test: layered low-rank latent + subject random effect + weak class shift; synthetic EEG windows
    from 4 switching topographies (class-dependent state probabilities). No real data involved."""
    rng = np.random.default_rng(seed)
    n_layers = len(LAYERS)
    Ws = [rng.normal(size=(30, dim)).astype(np.float32) / np.sqrt(30) for _ in range(n_layers)]
    tmpl = rng.normal(size=(4, len(CBRAMOD_CHANNELS)))
    T = int(round(PRE['window_sec'] * PRE['sfreq']))
    XX, WW, yy, gg = [], [], [], []
    for i in range(n_mdd + n_hc):
        lab = int(i < n_mdd)
        mu = rng.normal(size=30) * 1.0
        mu[:3] += 0.5 * (2 * lab - 1)
        z = mu[None] + rng.normal(size=(win, 30)) * 0.8
        XX.append(np.stack([(z @ Wl + 0.5 * rng.normal(size=(win, dim))) for Wl in Ws], axis=1).astype(np.float32))
        pst = np.array([0.35, 0.25, 0.2, 0.2]) + (0.12 * (2 * lab - 1)) * np.array([1, -0.5, -0.25, -0.25])
        for _ in range(win):
            sig = np.zeros((len(CBRAMOD_CHANNELS), T)); t0 = 0
            while t0 < T:
                L = int(rng.integers(8, 30)); st = rng.choice(4, p=pst / pst.sum())
                tt = np.arange(min(L, T - t0)); sig[:, t0:t0 + len(tt)] = tmpl[st][:, None] * np.sin(np.pi * tt / L)[None]
                t0 += L
            sig += 0.2 * rng.normal(size=sig.shape)
            WW.append((sig - sig.mean(0, keepdims=True)).astype(np.float32))
        yy += [lab] * win; gg += [f'S{i:03d}'] * win
    return np.concatenate(XX), np.asarray(yy), np.asarray(gg), np.stack(WW)

# ================================================================
# 3. Per-layer latent + diffusion banks (normative deviation + discriminative consistency adapter)
# ================================================================
def group_rows(g):
    d = {}
    for i, s in enumerate(g):
        d.setdefault(str(s), []).append(i)
    return {s: np.asarray(v, dtype=int) for s, v in d.items()}


def subject_labels(y, g):
    df = pd.DataFrame({'y': np.asarray(y), 'g': np.asarray(g)})
    assert (df.groupby('g')['y'].nunique() == 1).all(), 'a subject has more than one label'
    return df.groupby('g')['y'].first()


class LayerLatent:
    """Train-only, per-layer StandardScaler + whitened PCA; blocks concatenated -> [n, n_layers * k]."""
    def fit(self, X, seed):
        n, L, D = X.shape
        self.k = int(min(CFG['layer_pca_dim'], n - 1, D))
        self.sc, self.pca = [], []
        for l in range(L):
            s = StandardScaler().fit(X[:, l])
            self.sc.append(s)
            self.pca.append(PCA(self.k, whiten=True, svd_solver='randomized', random_state=seed + l).fit(s.transform(X[:, l])))
        return self
    def transform(self, X):
        return np.hstack([p.transform(s.transform(X[:, l])) for l, (s, p) in enumerate(zip(self.sc, self.pca))]).astype(np.float32)


class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        f = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
        a = t.float()[:, None] * f[None]
        return torch.cat([torch.sin(a), torch.cos(a)], dim=1)


class ResMLPBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.norm = nn.LayerNorm(h); self.fc1 = nn.Linear(h, 2 * h); self.fc2 = nn.Linear(2 * h, h); self.drop = nn.Dropout(0.05)
    def forward(self, x):
        return x + self.fc2(self.drop(F.silu(self.fc1(self.norm(x)))))


class LatentDenoiser(nn.Module):
    """Unconditional eps-predictor on the whitened multi-layer latent (healthy-only normative model)."""
    def __init__(self, dim, hidden, n_res):
        super().__init__()
        self.inp = nn.Linear(dim, hidden)
        self.time = nn.Sequential(SinusoidalTimeEmb(64), nn.Linear(64, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.blocks = nn.Sequential(*[ResMLPBlock(hidden) for _ in range(n_res)])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, dim))
    def forward(self, zt, t):
        return self.out(self.blocks(self.inp(zt) + self.time(t)))


class DCAdapter(nn.Module):
    """Shared denoising trunk + two zero-initialised low-rank class adapters; returns (eps_trunk, eps_HC, eps_MDD)."""
    def __init__(self, dim, hidden, n_res, rank, drop, kappa0):
        super().__init__()
        self.inp = nn.Linear(dim, hidden)
        self.time = nn.Sequential(SinusoidalTimeEmb(64), nn.Linear(64, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.blocks = nn.Sequential(*[ResMLPBlock(hidden) for _ in range(n_res)])
        self.norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, dim)
        self.adapters = nn.ModuleList()
        for _ in range(2):
            up = nn.Linear(rank, dim); nn.init.zeros_(up.weight); nn.init.zeros_(up.bias)
            self.adapters.append(nn.Sequential(nn.Dropout(drop), nn.Linear(hidden, rank), nn.SiLU(), up))
        self.log_kappa = nn.Parameter(torch.tensor(math.log(kappa0)))
    def kappa(self):
        return self.log_kappa.exp().clamp(max=200.0)
    def forward(self, zt, t):
        h = self.norm(self.blocks(self.inp(zt) + self.time(t)))
        base = self.out(h)
        return base, base + self.adapters[0](h), base + self.adapters[1](h)


class Schedule:
    def __init__(self, T):
        beta = torch.linspace(1e-4, 2e-2, T, device=DEVICE)
        self.T = T
        self.alpha_bar = torch.cumprod(1.0 - beta, dim=0)
    def noise(self, z, t, eps):
        ab = self.alpha_bar[t][:, None]
        return torch.sqrt(ab) * z + torch.sqrt(1.0 - ab) * eps

SCHED = Schedule(CFG['T'])


def _cosine_sched(opt, total):
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, i / max(1, total)))))


def train_latent_diffusion(Z, g, seed, P):
    """Subject-balanced unconditional DDPM (each subject contributes P['draws'] windows per epoch)."""
    seed_everything(seed); rng = np.random.default_rng(seed)
    model = LatentDenoiser(Z.shape[1], P['hidden'], P['n_res']).to(DEVICE)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    rows = group_rows(g); subs = list(rows)
    draws, bs = P['draws'], P['batch']
    total = P['epochs'] * math.ceil(len(subs) * draws / bs)
    opt = torch.optim.AdamW(model.parameters(), lr=P['lr'], weight_decay=P['wd'])
    sched = _cosine_sched(opt, total)
    Zt = torch.from_numpy(np.ascontiguousarray(Z)).float().to(DEVICE)
    hist, step = [], 0
    for ep in range(P['epochs']):
        model.train()
        idx = np.concatenate([rng.choice(rows[s], size=draws, replace=len(rows[s]) < draws) for s in subs])
        rng.shuffle(idx)
        tot = 0.0
        for b in range(0, len(idx), bs):
            zb = Zt[torch.from_numpy(idx[b:b + bs]).to(DEVICE)]
            t = torch.randint(0, SCHED.T, (len(zb),), device=DEVICE)
            eps = torch.randn_like(zb)
            loss = F.mse_loss(model(SCHED.noise(zb, t, eps), t), eps)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            d = min(P['ema'], (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(d).add_(pm.detach(), alpha=1 - d)
            step += 1
            tot += float(loss.item()) * len(zb)
        hist.append(tot / len(idx))
    return ema.eval(), hist


def train_dc_adapter(Z, y, g, seed, P):
    """Discriminative diffusion-consistency adapter.
       L = w_den*denoise(true-class head) + w_base*denoise(trunk) + w_disc*BCE(kappa*[log e_HC - log e_MDD], y)
           + w_cons*(logit(t_a,eps_a) - logit(t_b,eps_b))^2     (subject-balanced sampling, class-balanced weights)"""
    seed_everything(seed); rng = np.random.default_rng(seed)
    model = DCAdapter(Z.shape[1], P['hidden'], P['n_res'], P['rank'], P['adapter_dropout'], P['kappa0']).to(DEVICE)
    rows = group_rows(g); subs = list(rows)
    sub_y = np.array([y[rows[s][0]] for s in subs])
    n_pos = max(1, int(sub_y.sum())); n_neg = max(1, len(subs) - n_pos)
    w_pos, w_neg = len(subs) / (2 * n_pos), len(subs) / (2 * n_neg)
    draws, bs, tmax = P['draws'], P['batch'], int(P['tmax'])
    total = P['epochs'] * math.ceil(len(subs) * draws / bs)
    opt = torch.optim.AdamW(model.parameters(), lr=P['lr'], weight_decay=P['wd'])
    sched = _cosine_sched(opt, total)
    Zt = torch.from_numpy(np.ascontiguousarray(Z)).float().to(DEVICE)
    Yt = torch.from_numpy(np.asarray(y)).float().to(DEVICE)

    def draw(zb):
        t = torch.randint(1, tmax + 1, (len(zb),), device=DEVICE)
        eps = torch.randn_like(zb)
        base, p0, p1 = model(SCHED.noise(zb, t, eps), t)
        e0, e1, eb = ((eps - p0) ** 2).mean(1), ((eps - p1) ** 2).mean(1), ((eps - base) ** 2).mean(1)
        return torch.log(e0 + 1e-8) - torch.log(e1 + 1e-8), e0, e1, eb

    hist = []
    for ep in range(P['epochs']):
        model.train()
        idx = np.concatenate([rng.choice(rows[s], size=draws, replace=len(rows[s]) < draws) for s in subs])
        rng.shuffle(idx)
        tot = 0.0
        for b in range(0, len(idx), bs):
            ib = torch.from_numpy(idx[b:b + bs]).to(DEVICE)
            zb, yb = Zt[ib], Yt[ib]
            wts = yb * w_pos + (1 - yb) * w_neg
            da, e0a, e1a, eba = draw(zb); db, e0b, e1b, ebb = draw(zb)
            kap = model.kappa()
            la, lb = (kap * da).clamp(-20, 20), (kap * db).clamp(-20, 20)
            l_den = 0.5 * (torch.where(yb > 0.5, e1a, e0a).mean() + torch.where(yb > 0.5, e1b, e0b).mean())
            l_base = 0.5 * (eba.mean() + ebb.mean())
            l_disc = 0.5 * sum((wts * F.binary_cross_entropy_with_logits(l, yb, reduction='none')).mean() for l in (la, lb))
            l_cons = F.mse_loss(la, lb)
            loss = P['w_den'] * l_den + P['w_base'] * l_base + P['w_disc'] * l_disc + P['w_cons'] * l_cons
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tot += float(loss.item()) * len(zb)
        hist.append(tot / len(idx))
    return model.eval(), hist


@torch.no_grad()
def deviation_features(model, Z, seed, batch=512):
    """Normative deviation: log denoising error per noise level - global + n_blocks (= layers) blocks."""
    model.eval()
    ts = CFG['eval_timesteps']; K = len(ts)
    blocks = np.array_split(np.arange(Z.shape[1]), CFG['n_blocks'])
    out = np.zeros((len(Z), K * (1 + len(blocks))), np.float32)
    gen = torch.Generator(device=DEVICE); gen.manual_seed(int(seed))
    for b0 in range(0, len(Z), batch):
        zb = torch.from_numpy(np.ascontiguousarray(Z[b0:b0 + batch])).float().to(DEVICE)
        nb = len(zb)
        err = torch.zeros(nb, K, zb.shape[1], device=DEVICE)
        for k, tv in enumerate(ts):
            t = torch.full((nb,), int(tv), dtype=torch.long, device=DEVICE)
            for _ in range(CFG['eval_repeats']):
                eps = torch.randn(zb.shape, generator=gen, device=DEVICE)
                err[:, k] += (eps - model(SCHED.noise(zb, t, eps), t)) ** 2
        err /= CFG['eval_repeats']
        cols = [torch.log(err.mean(2) + 1e-8)] + [torch.log(err[:, :, torch.from_numpy(bi).to(DEVICE)].mean(2) + 1e-8) for bi in blocks]
        out[b0:b0 + nb] = torch.cat(cols, dim=1).cpu().numpy()
    return out


@torch.no_grad()
def adapter_features(model, Z, seed, batch=512):
    """Per-timestep diffusion-classifier log-ratio log e_HC - log e_MDD (K cols) + kappa-weighted mean -> K+1 columns."""
    model.eval()
    ts = CFG['eval_timesteps']; K = len(ts)
    out = np.zeros((len(Z), K + 1), np.float32)
    gen = torch.Generator(device=DEVICE); gen.manual_seed(int(seed))
    kap = float(model.kappa())
    for b0 in range(0, len(Z), batch):
        zb = torch.from_numpy(np.ascontiguousarray(Z[b0:b0 + batch])).float().to(DEVICE)
        nb = len(zb); acc = torch.zeros((nb, K), device=DEVICE)
        for _ in range(CFG['eval_repeats']):
            for k, tv in enumerate(ts):
                t = torch.full((nb,), int(tv), dtype=torch.long, device=DEVICE)
                eps = torch.randn(zb.shape, generator=gen, device=DEVICE)
                _, p0, p1 = model(SCHED.noise(zb, t, eps), t)
                acc[:, k] += torch.log(((eps - p0) ** 2).mean(1) + 1e-8) - torch.log(((eps - p1) ** 2).mean(1) + 1e-8)
        acc /= CFG['eval_repeats']
        out[b0:b0 + nb, :K] = acc.cpu().numpy()
        out[b0:b0 + nb, K] = (kap * acc.mean(1)).cpu().numpy()
    return out


class Bank:
    def __init__(self, members, held, full=None):
        self.members, self.held, self.full = members, held, full


def build_norm_bank(Ztr, ytr, gtr, seed):
    hc = np.where(ytr == 0)[0]
    full, hist = train_latent_diffusion(Ztr[hc], gtr[hc], seed, CFG['NORM'])
    subs = np.array(sorted(set(gtr[hc])))
    k = min(CFG['cross_fit_k'], len(subs) // 2)
    if k < 2:
        raise RuntimeError('too few healthy training subjects for cross-fitting')
    parts = np.array_split(np.random.default_rng(seed).permutation(subs), k)
    members, held = [], {}
    for j, part in enumerate(parts):
        keep = hc[~np.isin(gtr[hc], part)]
        m, _ = train_latent_diffusion(Ztr[keep], gtr[keep], seed + 100 * (j + 1), CFG['NORM'])
        members.append(m)
        for s in part:
            held[str(s)] = j
    return Bank(members, held, full)


def build_delta_bank(Ztr, ytr, gtr, seed):
    sl = subject_labels(ytr, gtr)
    subs, lab = sl.index.values.astype(str), sl.values.astype(int)
    k = int(min(CFG['cross_fit_k'], lab.sum(), (1 - lab).sum()))
    if k < 2:
        raise RuntimeError('too few subjects per class for adapter cross-fitting')
    members, held = [], {}
    for j, (fi, hi) in enumerate(StratifiedKFold(k, shuffle=True, random_state=seed + 17).split(subs, lab)):
        rows = np.isin(gtr, subs[fi])
        m, _ = train_dc_adapter(Ztr[rows], ytr[rows], gtr[rows], seed + 100 * (j + 1) + 50, CFG['DELTA'])
        members.append(m)
        for s in subs[hi]:
            held[str(s)] = j
    return Bank(members, held)


def crossfit_apply(bank, Z, g, seed, fn):
    """Training subjects a member never saw -> scored by that member. Everyone else (MDD-train for the normative bank,
    all test subjects) -> ONE member each ('single', distribution-matched) or the member mean ('mean')."""
    g = np.asarray(g).astype(str); k = len(bank.members)
    held = np.array([bank.held.get(s, -1) for s in g])
    out = [None]
    def put(idx, val):
        if out[0] is None:
            out[0] = np.zeros((len(Z), val.shape[1]), np.float32)
        out[0][idx] = val
    for j in range(k):
        idx = np.where(held == j)[0]
        if len(idx):
            put(idx, fn(bank.members[j], Z[idx], seed))
    idx = np.where(held == -1)[0]
    if len(idx):
        if CFG['member_scoring'] == 'mean':
            put(idx, np.mean([fn(m, Z[idx], seed) for m in bank.members], axis=0))
        else:
            subs = sorted(set(g[idx])); order = np.random.default_rng(seed + 31).permutation(len(subs))
            assign = {subs[i]: int(r % k) for r, i in enumerate(order)}
            mem = np.array([assign[s] for s in g[idx]])
            for j in range(k):
                sel = idx[mem == j]
                if len(sel):
                    put(sel, fn(bank.members[j], Z[sel], seed))
    return out[0]

# ================================================================
# 4. Microstate dynamics stream (templates fitted on TRAIN windows only)
# ================================================================
def _smooth_labels(lab, ac, min_len):
    lab = lab.copy()
    for _ in range(6):
        change = np.flatnonzero(np.diff(lab)) + 1
        starts = np.r_[0, change]; ends = np.r_[change, len(lab)]
        short = np.flatnonzero((ends - starts) < min_len)
        if len(short) == 0 or len(starts) == 1:
            break
        for si in short:
            s, e = starts[si], ends[si]
            cands = []
            if si > 0:
                cands.append(int(lab[starts[si - 1]]))
            if si < len(starts) - 1:
                cands.append(int(lab[starts[si + 1]]))
            lab[s:e] = max(cands, key=lambda k: ac[k, s:e].sum())
    return lab


class MicrostateModel:
    """Polarity-invariant modified K-means on GFP-peak topographies + back-fitting + 33 dynamics features."""
    def __init__(self):
        self.P = CFG['MS']; self.k = self.P['k']
        self.min_len = max(1, int(round(self.P['min_seg_ms'] * PRE['sfreq'] / 1000.0)))
        self.n_feat = 4 * self.k + self.k * (self.k - 1) + 5

    def fit(self, W, g, seed):
        rng = np.random.default_rng(seed + 7)
        topo, wt = [], []
        for s, rows in group_rows(np.asarray(g).astype(str)).items():
            Wb = W[rows].astype(np.float32); Wb = Wb - Wb.mean(1, keepdims=True)
            G = Wb.std(1)
            wi, ti = np.nonzero((G[:, 1:-1] > G[:, :-2]) & (G[:, 1:-1] >= G[:, 2:])); ti = ti + 1
            if len(wi) == 0:
                continue
            if len(wi) > self.P['per_subj']:
                sel = rng.choice(len(wi), self.P['per_subj'], replace=False); wi, ti = wi[sel], ti[sel]
            x = Wb[wi, :, ti]
            topo.append(x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)); wt.append(G[wi, ti] ** 2)
        X = np.concatenate(topo).astype(np.float64); w = np.concatenate(wt).astype(np.float64); w /= w.sum()
        best, best_gev = None, -1.0
        for _ in range(self.P['n_init']):
            T = X[rng.choice(len(X), self.k, replace=False)].copy()
            for _it in range(self.P['iters']):
                lab = np.argmax(np.abs(X @ T.T), axis=1)
                Tn = T.copy()
                for c in range(self.k):
                    Xc = X[lab == c]
                    if len(Xc) < 2:
                        Tn[c] = X[rng.integers(len(X))]; continue
                    Tn[c] = np.linalg.eigh(Xc.T @ Xc)[1][:, -1]
                done = np.max(1 - np.abs(np.sum(Tn * T, axis=1))) < self.P['tol']
                T = Tn
                if done:
                    break
            gev = float(np.sum(w * np.max(np.abs(X @ T.T), axis=1) ** 2))
            if gev > best_gev:
                best, best_gev = T, gev
        lab = np.argmax(np.abs(X @ best.T), axis=1)
        self.T = best[np.argsort(-np.bincount(lab, minlength=self.k), kind='stable')]
        self.train_gev = best_gev
        return self

    def features(self, W, chunk=128):
        out = np.zeros((len(W), self.n_feat), np.float32)
        for b0 in range(0, len(W), chunk):
            Wb = W[b0:b0 + chunk].astype(np.float32); Wb = Wb - Wb.mean(1, keepdims=True)
            gfp = Wb.std(1); nrm = np.linalg.norm(Wb, axis=1) + 1e-8
            ac = np.abs(np.einsum('kc,nct->nkt', self.T.astype(np.float32), Wb)) / nrm[:, None, :]
            lab_all = ac.argmax(1)
            for i in range(len(Wb)):
                out[b0 + i] = self._one(lab_all[i], ac[i], gfp[i])
        return out

    def _one(self, lab, ac, gfp):
        K, T, fs = self.k, len(lab), PRE['sfreq']
        lab = _smooth_labels(lab, ac, self.min_len)
        change = np.flatnonzero(np.diff(lab)) + 1
        starts = np.r_[0, change]; ends = np.r_[change, T]
        lens = (ends - starts).astype(float); segl = lab[starts]
        best = ac[lab, np.arange(T)]
        g2 = gfp.astype(np.float64) ** 2; den = g2.sum() + 1e-12
        cov, dur, occ, gev = np.zeros(K), np.zeros(K), np.zeros(K), np.zeros(K)
        for k in range(K):
            m = segl == k
            if m.any():
                cov[k] = lens[m].sum() / T; dur[k] = lens[m].mean() * 1000.0 / fs; occ[k] = m.sum() / (T / fs)
            gev[k] = (g2 * best ** 2)[lab == k].sum() / den
        trans = np.zeros((K, K))
        if len(segl) > 1:
            np.add.at(trans, (segl[:-1], segl[1:]), 1.0); trans /= trans.sum()
        off = trans[~np.eye(K, dtype=bool)]
        pc, po = cov[cov > 0], off[off > 0]
        glob_ = [gev.sum(), math.log1p(lens.mean() * 1000.0 / fs), len(lens) / (T / fs),
                 float(-(pc * np.log(pc)).sum()), float(-(po * np.log(po)).sum())]
        return np.concatenate([cov, np.log1p(dur), occ, gev, off, glob_]).astype(np.float32)

# ================================================================
# 5. Klein-model hyperbolic fusion (fit on the fit set only)
# ================================================================
def _nrm(x): return np.linalg.norm(x, axis=1, keepdims=True)
def poincare_exp0(v, c, tau):
    vs = v * tau; sc = np.sqrt(c); r = np.maximum(_nrm(vs), 1e-8)
    return np.tanh(sc * r) * vs / (sc * r)
def poincare_log0(p, c, tau):
    sc = np.sqrt(c); r = np.maximum(_nrm(p), 1e-8)
    return np.arctanh(np.minimum(sc * r, 1 - 1e-6)) * (p / r) / sc / tau
def poincare_to_klein(p, c): return 2 * p / (1 + c * np.sum(p * p, axis=1, keepdims=True))
def klein_to_poincare(k, c): return k / (1 + np.sqrt(np.maximum(1 - c * np.sum(k * k, axis=1, keepdims=True), 1e-12)))
def einstein_midpoint(ks, c):
    gam = [1 / np.sqrt(np.maximum(1 - c * np.sum(k * k, axis=1, keepdims=True), 1e-12)) for k in ks]
    return sum(g * k for g, k in zip(gam, ks)) / sum(gam)


class KleinFuser:
    def __init__(self, k, curvatures, tau):
        self.k, self.curv, self.tau = k, curvatures, tau
    def fit(self, A, B):
        self.sa, self.sb = StandardScaler().fit(A), StandardScaler().fit(B)
        self.k = int(min(self.k, A.shape[1], B.shape[1], A.shape[0] - 1))
        self.pa = PCA(self.k, svd_solver='full').fit(self.sa.transform(A))
        self.pb = PCA(self.k, svd_solver='full').fit(self.sb.transform(B))
        za, zb = self._proj(A, B, scale=False)
        self.ra = float(np.sqrt(np.mean(np.sum(za ** 2, axis=1)))) + 1e-8
        self.rb = float(np.sqrt(np.mean(np.sum(zb ** 2, axis=1)))) + 1e-8
        return self
    def _proj(self, A, B, scale=True):
        za, zb = self.pa.transform(self.sa.transform(A)), self.pb.transform(self.sb.transform(B))
        return (za / self.ra, zb / self.rb) if scale else (za, zb)
    def transform(self, A, B):
        za, zb = (v.astype(np.float64) for v in self._proj(A, B))
        hyp = []
        for c in self.curv:
            pa, pb = poincare_exp0(za, c, self.tau), poincare_exp0(zb, c, self.tau)
            mid = klein_to_poincare(einstein_midpoint([poincare_to_klein(pa, c), poincare_to_klein(pb, c)], c), c)
            hyp.append(poincare_log0(mid, c, self.tau))
        return np.hstack(hyp + [za, zb]).astype(np.float32)

# ================================================================
# 6. Streams and base classifiers
# ================================================================
def take(d, idx):
    return {k: v[idx] for k, v in d.items()}


def _klein_A(d):
    return np.hstack([d['dev'], d['dlt']]) if CFG['klein_source'] == 'both' else d['dev']


def make_streams(fit, others):
    """Fit the Klein fusion on `fit` only; return stream dicts for fit and each of `others`."""
    kf = KleinFuser(CFG['klein_dim'], CFG['klein_curvatures'], CFG['klein_tau']).fit(_klein_A(fit), fit['Z'])
    def tf(d):
        return {'cbramod': d['Z'], 'cbr_delta': np.hstack([d['Z'], d['dlt']]), 'ndm': d['dev'],
                'klein': kf.transform(_klein_A(d), d['Z']), 'microstate': d['ms']}
    return [tf(fit)] + [tf(o) for o in others]


def subject_weights(g):
    vals, counts = np.unique(g, return_counts=True)
    d = dict(zip(vals, counts))
    w = np.asarray([1.0 / d[s] for s in g], np.float64)
    return w / w.mean()


def subject_summary(A, rows, subs):
    return np.stack([np.concatenate([A[rows[s]].mean(0), A[rows[s]].std(0)]) for s in subs]).astype(np.float32)


def score_subjects(clf_name, Ftr, ytr, gtr, Fte, gte, seed):
    """-> (subject ids, P(MDD)) for one base classifier."""
    sc = StandardScaler().fit(Ftr)
    A, B = sc.transform(Ftr).astype(np.float32), sc.transform(Fte).astype(np.float32)
    gtr, gte = np.asarray(gtr).astype(str), np.asarray(gte).astype(str)
    if clf_name in ('subj_lda', 'subj_logreg'):
        tr_rows, te_rows = group_rows(gtr), group_rows(gte)
        s_tr, s_te = list(tr_rows), list(te_rows)
        S_tr, S_te = subject_summary(A, tr_rows, s_tr), subject_summary(B, te_rows, s_te)
        ys = np.array([ytr[tr_rows[s][0]] for s in s_tr])
        s2 = StandardScaler().fit(S_tr)
        S_tr, S_te = s2.transform(S_tr), s2.transform(S_te)
        if clf_name == 'subj_lda':
            clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=np.array([0.5, 0.5]))
        else:
            clf = LogisticRegression(C=CFG['subj_logreg_C'], class_weight='balanced', max_iter=2000)
        clf.fit(S_tr, ys)
        return np.array(s_te), clf.predict_proba(S_te)[:, 1]
    w = subject_weights(gtr)
    if clf_name == 'bagged_logreg':
        clf = BaggingClassifier(estimator=LogisticRegression(max_iter=2000, class_weight='balanced'),
                                n_estimators=25, random_state=seed).fit(A, ytr)
    elif clf_name == 'logreg':
        clf = LogisticRegression(max_iter=2000, class_weight='balanced').fit(A, ytr, sample_weight=w)
    elif clf_name == 'svm_rbf':
        clf = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=seed).fit(A, ytr, sample_weight=w)
    elif clf_name == 'random_forest':
        clf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=seed, n_jobs=-1).fit(A, ytr, sample_weight=w)
    elif clf_name == 'xgboost':
        n_pos = max(1, int((ytr == 1).sum())); n_neg = max(1, int((ytr == 0).sum()))
        clf = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5,
                            min_child_weight=2, reg_lambda=1.0, scale_pos_weight=n_neg / n_pos, eval_metric='logloss',
                            tree_method='hist', n_jobs=4, random_state=seed, verbosity=0).fit(A, ytr, sample_weight=w)
    else:
        raise ValueError(clf_name)
    p = pd.Series(clf.predict_proba(B)[:, 1]).groupby(gte, sort=False).mean()
    return p.index.values.astype(str), p.values

# ================================================================
# 7. Calibration / metrics helpers
# ================================================================
def to_logit(p):
    c = CFG['logit_clip']
    return logit(np.clip(np.asarray(p, float), c, 1 - c))


def platt(x, y, C):
    """Balanced 1-D logistic calibration: logit P(MDD) = a*x + b."""
    lr = LogisticRegression(C=C, class_weight='balanced', max_iter=1000).fit(np.asarray(x, float)[:, None], y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def youden_threshold(y, p):
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.isfinite(thr)
    return float(thr[ok][np.argmax((tpr - fpr)[ok])]) if ok.any() else 0.5


def classification_metrics(y_, p, pred):
    y_, pred = np.asarray(y_), np.asarray(pred)
    tp = int(((pred == 1) & (y_ == 1)).sum()); tn = int(((pred == 0) & (y_ == 0)).sum())
    fp = int(((pred == 1) & (y_ == 0)).sum()); fn = int(((pred == 0) & (y_ == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return dict(auc=roc_auc_score(y_, p) if len(np.unique(y_)) > 1 else np.nan,
                accuracy=(tp + tn) / max(1, len(y_)), precision=tp / (tp + fp) if tp + fp else 0.0,
                recall=sens, sensitivity=sens, specificity=spec,
                f1=2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
                mcc=(tp * tn - fp * fn) / den if den else 0.0,
                balanced_accuracy=float(np.nanmean([sens, spec])), tp=tp, tn=tn, fp=fp, fn=fn)

# ================================================================
# 8. Per-fold feature computation and expert tables
# ================================================================
def compute_delta(Ztr, Zte, ytr, gtr, gte, seed, over=None):
    """Cross-fitted discriminative diffusion-consistency adapter features; `over` temporarily overrides CFG['DELTA']."""
    old = dict(CFG['DELTA'])
    try:
        if over:
            CFG['DELTA'].update(over)
        bank = build_delta_bank(Ztr, ytr, gtr, seed)
    finally:
        CFG['DELTA'].clear(); CFG['DELTA'].update(old)
    return (crossfit_apply(bank, Ztr, gtr, seed + 2, adapter_features),
            crossfit_apply(bank, Zte, gte, seed + 2, adapter_features))


def compute_features(Xtr, Xte, ytr, gtr, gte, seed, n_blocks):
    """Per-layer scaler+PCA latent, healthy-only normative deviation stream, adapter delta stream."""
    old_nb = CFG['n_blocks']; CFG['n_blocks'] = n_blocks
    try:
        LL = LayerLatent().fit(Xtr, seed)
        Ztr, Zte = LL.transform(Xtr), LL.transform(Xte)
        nbank = build_norm_bank(Ztr, ytr, gtr, seed)
        dev_tr = crossfit_apply(nbank, Ztr, gtr, seed + 1, deviation_features)
        dev_te = crossfit_apply(nbank, Zte, gte, seed + 1, deviation_features)
        del nbank
    finally:
        CFG['n_blocks'] = old_nb
    dlt_tr, dlt_te = compute_delta(Ztr, Zte, ytr, gtr, gte, seed)
    return dict(Z=Ztr, dev=dev_tr, dlt=dlt_tr), dict(Z=Zte, dev=dev_te, dlt=dlt_te)


def score_pool(F_tr, F_te, ytr, gtr, gte, streams, seed, klein_source=None):
    """Inner-OOF (train subjects) and outer-test subject probabilities for every stream|classifier in `streams`."""
    old = CFG['klein_source']
    if klein_source:
        CFG['klein_source'] = klein_source
    try:
        inner = {f'{s}|{c}': [] for s in streams for c in CLFS}
        cv_in = StratifiedGroupKFold(n_splits=CFG['cv_inner_splits'], shuffle=True, random_state=seed)
        for ii, (itr, iva) in enumerate(cv_in.split(F_tr['Z'], ytr, gtr)):
            S_fit, S_va = make_streams(take(F_tr, itr), [take(F_tr, iva)])
            for s in streams:
                for c in CLFS:
                    subj, p = score_subjects(c, S_fit[s], ytr[itr], gtr[itr], S_va[s], gtr[iva], seed + ii)
                    inner[f'{s}|{c}'].append(pd.Series(p, index=subj))
        P_in = pd.DataFrame({e: pd.concat(v) for e, v in inner.items()}).sort_index()
        S_tr, S_te = make_streams(F_tr, [F_te])
        te = {}
        for s in streams:
            for c in CLFS:
                subj, p = score_subjects(c, S_tr[s], ytr, gtr, S_te[s], gte, seed)
                te[f'{s}|{c}'] = pd.Series(p, index=subj)
        return P_in, pd.DataFrame(te).sort_index()
    finally:
        CFG['klein_source'] = old


def variant_list():
    v = [dict(name='klein_dev', kind='klein_source')]
    if ABL_DELTA:
        v += [dict(name='delta_nodisc', kind='delta', over=dict(w_disc=0.0, w_cons=0.0)),
              dict(name='delta_nocons', kind='delta', over=dict(w_cons=0.0))]
    if ABL_LAYERS == 'loo' and len(LAYERS) > 1:
        for i, l in enumerate(LAYERS):
            v.append(dict(name=f'no_layer{l}', kind='layers', keep=[j for j in range(len(LAYERS)) if j != i]))
    return v


def build_fold_cache(rep, fold, tr_subj, te_subj, D, variants, tag_hash):
    seed = SEED + rep * CFG['cv_n_splits'] + fold + 1
    seed_everything(seed)
    X, y, g, W = D['X'], D['y'], D['g'], D['W']
    tr, te = np.where(np.isin(g, tr_subj))[0], np.where(np.isin(g, te_subj))[0]
    ytr, yte, gtr, gte = y[tr], y[te], g[tr], g[te]
    tag = f'rep {rep + 1} fold {fold + 1}'
    base_path = os.path.join(CK_DIR, f'base_r{rep}_f{fold}_{tag_hash}.joblib')
    tr_labels = subject_labels(ytr, gtr); tr_labels.index = tr_labels.index.astype(str)
    te_labels = subject_labels(yte, gte); te_labels.index = te_labels.index.astype(str)

    if CFG['resume'] and os.path.exists(base_path):
        base = joblib.load(base_path); log(f'{tag}: base cache loaded')
    else:
        t0 = time.time()
        F_tr, F_te = compute_features(X[tr], X[te], ytr, gtr, gte, seed, len(LAYERS))
        msm = MicrostateModel().fit(W[tr], gtr, seed)
        F_tr['ms'], F_te['ms'] = msm.features(W[tr]), msm.features(W[te])
        P_in, P_te = score_pool(F_tr, F_te, ytr, gtr, gte, STREAMS, seed)
        assert P_in.index.is_unique and set(P_in.index) == set(tr_labels.index), 'inner OOF accounting failed'
        base = dict(F_tr=F_tr, F_te=F_te, P_in=P_in, P_te=P_te)
        joblib.dump(base, base_path)
        log(f'{tag}: base expert tables built ({(time.time() - t0) / 60:.1f} min)')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    P_in, P_te = base['P_in'], base['P_te']
    cache = dict(base=(P_in, P_te), variants={}, y_in=tr_labels.loc[P_in.index].values.astype(int),
                 y_te=te_labels.loc[P_te.index].values.astype(int), seed=seed)

    for var in variants:
        vp = os.path.join(CK_DIR, f'var_{var["name"]}_r{rep}_f{fold}_{tag_hash}.joblib')
        if CFG['resume'] and os.path.exists(vp):
            cache['variants'][var['name']] = joblib.load(vp); continue
        t0 = time.time()
        F_tr, F_te = dict(base['F_tr']), dict(base['F_te'])
        if var['kind'] == 'klein_source':                       # Klein fusion fed by deviation features only
            res = score_pool(F_tr, F_te, ytr, gtr, gte, ('klein',), seed, klein_source='dev')
        elif var['kind'] == 'delta':                            # re-train adapter with modified loss
            F_tr['dlt'], F_te['dlt'] = compute_delta(F_tr['Z'], F_te['Z'], ytr, gtr, gte, seed, var['over'])
            res = score_pool(F_tr, F_te, ytr, gtr, gte, ('cbr_delta', 'klein'), seed)
        elif var['kind'] == 'layers':                           # drop one backbone layer, full re-run (microstates reused)
            keep = var['keep']
            f_tr, f_te = compute_features(X[tr][:, keep], X[te][:, keep], ytr, gtr, gte, seed, len(keep))
            f_tr['ms'], f_te['ms'] = F_tr['ms'], F_te['ms']
            res = score_pool(f_tr, f_te, ytr, gtr, gte, [s for s in STREAMS if s != 'microstate'], seed)
        else:
            raise ValueError(var)
        assert res[0].index.equals(P_in.index) and res[1].index.equals(P_te.index)
        joblib.dump(res, vp); cache['variants'][var['name']] = res
        log(f'{tag}: variant {var["name"]} done ({(time.time() - t0) / 60:.1f} min)')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return cache

# ================================================================
# 9. Ablation specs and the SelectBest_inner selector
# ================================================================
@dataclass(frozen=True)
class Spec:
    name: str
    family: str
    streams: Optional[Tuple[str, ...]] = None      # allowed streams (None = all)
    clfs: Optional[Tuple[str, ...]] = None         # allowed classifiers (None = all)
    override: Optional[str] = None                 # upstream variant whose columns replace the base pool's
    gate: bool = True                              # require calibration slope > 0 ("active"), as in v3
    crit: str = 'auc'                              # auc | stable | random | oracle
    thr: str = 'calib'                             # calib@0.5 | raw_youden | raw_half


FULL = 'Full SelectBest_inner'


def make_specs(variants):
    names = {v['name'] for v in variants}
    sp = [Spec(FULL, 'Reference')]
    sp += [Spec(f'w/o {STREAM_LABEL[s]}', 'Pool: leave-one-stream-out', streams=tuple(x for x in STREAMS if x != s)) for s in STREAMS]
    sp += [Spec(f'only {STREAM_LABEL[s]}', 'Pool: single stream', streams=(s,)) for s in STREAMS]
    sp += [Spec(f'w/o {c}', 'Pool: leave-one-classifier-out', clfs=tuple(x for x in CLFS if x != c)) for c in CLFS]
    sp += [Spec('subject-level classifiers only', 'Pool: classifier family', clfs=tuple(c for c in CLFS if c in SUBJ_LEVEL)),
           Spec('epoch-level classifiers only', 'Pool: classifier family', clfs=tuple(c for c in CLFS if c in EPOCH_LEVEL))]
    if 'klein_dev' in names:
        sp.append(Spec('Klein w/o delta input', 'Component', override='klein_dev'))
    if 'delta_nodisc' in names:
        sp += [Spec('adapter w/o disc.+cons. loss', 'Component', override='delta_nodisc'),
               Spec('adapter w/o consistency loss', 'Component', override='delta_nocons')]
    sp += [Spec(f'w/o layer {v["name"][8:]}', 'Component', override=v['name']) for v in variants if v['kind'] == 'layers']
    sp += [Spec('w/o activity gate', 'Selection', gate=False),
           Spec('stability-penalised criterion', 'Selection', crit='stable'),
           Spec('random active expert (1 draw/fold)', 'Selection', crit='random'),
           Spec('oracle (post-hoc, optimistic)', 'Selection', crit='oracle'),
           Spec('thr: raw prob + inner Youden', 'Operating point', thr='raw_youden'),
           Spec('thr: raw prob @0.5', 'Operating point', thr='raw_half')]
    assert len({s.name for s in sp}) == len(sp)
    return sp


def pool_for(cache, spec):
    P_in, P_te = (d.copy() for d in cache['base'])
    if spec.override:
        v_in, v_te = cache['variants'][spec.override]
        P_in[v_in.columns] = v_in; P_te[v_te.columns] = v_te
    cols = [c for c in P_in.columns
            if (spec.streams is None or c.split('|')[0] in spec.streams) and (spec.clfs is None or c.split('|')[1] in spec.clfs)]
    assert cols, spec
    return P_in, P_te, cols


def stability_scores(L, y, seed):
    """mean - lambda*sd of subject-bootstrap AUC (the v3 stability-penalised selection score)."""
    rng = np.random.default_rng(seed + 977); n = len(y); a = []
    for _ in range(CFG['sel_boot']):
        ii = rng.integers(0, n, n); yb = y[ii]; n1 = int(yb.sum()); n0 = n - n1
        if n1 == 0 or n0 == 0:
            continue
        rk = rankdata(L[ii], axis=0)
        a.append((rk[yb == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0))
    a = np.asarray(a)
    return a.mean(0) - CFG['sel_lambda'] * a.std(0)


def select_and_predict(P_in, P_te, y_in, y_te, cols, spec, seed):
    L_in, L_te = to_logit(P_in[cols].values), to_logit(P_te[cols].values)
    E = len(cols)
    ab = np.array([platt(L_in[:, j], y_in, CFG['platt_C']) for j in range(E)])           # balanced Platt calibration
    act = ab[:, 0] > 0                                                                   # slope <= 0 => expert off
    inner_auc = np.array([roc_auc_score(y_in, L_in[:, j]) for j in range(E)])
    cand = [j for j in range(E) if act[j]] if spec.gate else list(range(E))
    if not cand:
        cand = list(range(E))
    if spec.crit == 'auc':
        j = max(cand, key=lambda k: inner_auc[k])
    elif spec.crit == 'stable':
        j = cand[int(np.argmax(stability_scores(L_in[:, cand], y_in, seed)))]
    elif spec.crit == 'random':
        j = cand[int(np.random.default_rng(seed + 4242).integers(len(cand)))]
    elif spec.crit == 'oracle':
        j = cand[int(np.argmax([roc_auc_score(y_te, L_te[:, k]) for k in cand]))]
    else:
        raise ValueError(spec.crit)
    a, b = (ab[j, 0], ab[j, 1]) if (act[j] or not spec.gate) else (0.0, 0.0)
    col = cols[j]
    if spec.thr == 'calib':
        p_te, thr = expit(a * L_te[:, j] + b), 0.5
    elif spec.thr == 'raw_youden':
        p_te, thr = P_te[col].values, youden_threshold(y_in, P_in[col].values)
    else:
        p_te, thr = P_te[col].values, 0.5
    return dict(chosen=col, p_te=p_te, thr=thr, inner_auc=float(inner_auc[j]),
                outer_auc=float(roc_auc_score(y_te, L_te[:, j])), n_pool=E, n_active=int(act.sum()))


def evaluate_fold(cache, specs, rep, fold):
    rows, preds = [], []
    y_in, y_te = cache['y_in'], cache['y_te']
    subj = cache['base'][1].index.values
    for sp in specs:
        P_in, P_te, cols = pool_for(cache, sp)
        r = select_and_predict(P_in, P_te, y_in, y_te, cols, sp, cache['seed'])
        pred = (r['p_te'] >= r['thr']).astype(int)
        m = classification_metrics(y_te, r['p_te'], pred)
        rows.append(dict(repeat=rep, fold=fold, method=sp.name, family=sp.family, chosen=r['chosen'], inner_auc=r['inner_auc'],
                         outer_auc_chosen=r['outer_auc'], n_pool=r['n_pool'], n_active=r['n_active'], threshold=r['thr'], **m))
        preds.append(pd.DataFrame(dict(repeat=rep, fold=fold, method=sp.name, subject=subj, y=y_te, p=r['p_te'], pred=pred)))
    return rows, preds


def tex_escape(s):
    for a, b in [('\\', r'\textbackslash{}'), ('_', r'\_'), ('&', r'\&'), ('%', r'\%'), ('#', r'\#'), ('|', '/')]:
        s = s.replace(a, b)
    return s


def write_latex(df, path):
    lines = [r'\begin{table}[t]', r'\centering',
             r'\caption{Ablation of SelectBest\_inner on MODMA (REST). Subject-level OOF metrics, mean over repeats; '
             r'$\Delta$ is paired against the full method (95\% subject-bootstrap CI).}',
             r'\label{tab:ablation_selectbest}', r'\small', r'\begin{tabular}{llccc}', r'\toprule',
             r'Family & Variant & AUC & Acc. & $\Delta$AUC [95\% CI] \\', r'\midrule']
    last = None
    for _, r in df.iterrows():
        if last is not None and r['family'] != last:
            lines.append(r'\midrule')
        fam = tex_escape(r['family']) if r['family'] != last else ''
        d = '--' if r['method'] == FULL else f"{r['d_auc']:+.3f} [{r['d_auc_lo']:+.3f}, {r['d_auc_hi']:+.3f}]"
        lines.append(f"{fam} & {tex_escape(r['method'])} & {r['auc']:.3f} & {r['accuracy']:.3f} & {d} \\\\")
        last = r['family']
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    with open(path, 'w') as f:
        f.write('\n'.join(lines))

# ================================================================
# 10. MAIN
# ================================================================
if __name__ == '__main__':
    print('\nABLATION of SelectBest_inner | frozen multi-layer CBraMod + diffusion streams + Klein + microstates | MODMA REST')
    if SYNTH:
        X, y, g, Wraw = synthetic_latent(); EMB_KEY = 'synthetic'; log('SYNTHETIC smoke-test data (not MODMA)')
    else:
        manifest = build_manifest(DATA_DIR)
        sl = manifest.groupby('subject')['label'].first()
        log(f'manifest: {len(manifest)} files, {len(sl)} subjects (MDD={int(sl.sum())}, HC={int((1 - sl).sum())}; expected 24/29)')
        X, y, g, Wraw, EMB_KEY = build_latent_dataset(manifest)
    assert X.ndim == 3 and X.shape[1] == len(LAYERS) and len(Wraw) == len(X)

    ysub_all = subject_labels(y, g)
    subj_arr = np.array(sorted(ysub_all.index)); ysub = ysub_all.loc[subj_arr]
    N_SUBJ, N_MDD = len(subj_arr), int(ysub.sum())
    R, K = CFG['cv_repeats'], CFG['cv_n_splits']
    assert min(N_MDD, N_SUBJ - N_MDD) >= K
    variants = variant_list(); specs = make_specs(variants)
    tag_hash = hashlib.sha1((MODEL_VERSION + EMB_KEY + json.dumps(CFG, sort_keys=True, default=str)).encode()).hexdigest()[:10]
    log(f'{N_SUBJ} subjects (MDD={N_MDD}, HC={N_SUBJ - N_MDD}) | {len(CLFS) * len(STREAMS)} candidate experts | '
        f'upstream variants {[v["name"] for v in variants]} | {len(specs)} ablation specs | hash {tag_hash}')

    D = dict(X=X, y=y, g=g, W=Wraw)
    fold_rows, preds = [], []
    for rep in range(R):
        outer = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=SEED + rep)
        seen = []
        for fold, (a_, b_) in enumerate(outer.split(np.zeros(N_SUBJ), ysub.values, groups=subj_arr)):
            tr_subj, te_subj = subj_arr[a_], subj_arr[b_]
            assert not (set(tr_subj) & set(te_subj)) and ysub.loc[te_subj].nunique() == 2
            seen += list(te_subj)
            log(f'REPEAT {rep + 1}/{R} FOLD {fold + 1}/{K}: train={len(tr_subj)} test={len(te_subj)}')
            cache = build_fold_cache(rep, fold, tr_subj, te_subj, D, variants, tag_hash)
            r_, p_ = evaluate_fold(cache, specs, rep, fold)
            fold_rows += r_; preds += p_
            full = next(r for r in r_ if r['method'] == FULL)
            log(f'  Full: chose {full["chosen"]} | inner AUC {full["inner_auc"]:.3f} -> outer AUC {full["auc"]:.3f} acc {full["accuracy"]:.3f}')
        assert sorted(seen) == sorted(subj_arr.tolist()), 'each subject must be tested exactly once per repeat'

    fold_df, pred_df = pd.DataFrame(fold_rows), pd.concat(preds, ignore_index=True)
    fold_df.to_csv(os.path.join(OUT_DIR, 'per_fold_metrics.csv'), index=False)
    pred_df.to_csv(os.path.join(OUT_DIR, 'oof_subject_predictions.csv'), index=False)

    # ---- pooled subject-level metrics, bootstrap CIs, paired deltas vs Full (same bootstrap draws) ----
    subs_all = sorted(pred_df.subject.unique()); assert len(subs_all) == N_SUBJ
    def oof_by_repeat(method):
        out = []
        for _, d in pred_df[pred_df.method == method].groupby('repeat'):
            assert d.subject.is_unique and len(d) == N_SUBJ
            d = d.set_index('subject').loc[subs_all]
            out.append((d.y.values.astype(int), d.p.values.astype(float), d.pred.values.astype(int)))
        assert len(out) == R
        return out

    BOOT = np.random.default_rng(SEED + 5).integers(0, N_SUBJ, size=(CFG['n_boot'], N_SUBJ))
    def boot_metrics(reps):
        out = np.full((len(BOOT), len(METRICS)), np.nan)
        for bi, i in enumerate(BOOT):
            if len(np.unique(reps[0][0][i])) < 2:
                continue
            ms = [classification_metrics(a[i], b[i], c[i]) for a, b, c in reps]
            out[bi] = [np.mean([m[k] for m in ms]) for k in METRICS]
        return out

    reps_by = {sp.name: oof_by_repeat(sp.name) for sp in specs}
    per_by = {n: pd.DataFrame([classification_metrics(*r_) for r_ in rp]) for n, rp in reps_by.items()}
    boots = {n: boot_metrics(rp) for n, rp in reps_by.items()}
    rows = []
    for sp in specs:
        per = per_by[sp.name]
        row = dict(method=sp.name, family=sp.family)
        for mi, k in enumerate(METRICS):
            row[k] = float(per[k].mean()); row[f'{k}_sd'] = float(per[k].std(ddof=1)) if R > 1 else np.nan
            row[f'{k}_lo'], row[f'{k}_hi'] = (float(np.nanquantile(boots[sp.name][:, mi], q)) for q in (.025, .975))
        for k, short in (('auc', 'd_auc'), ('accuracy', 'd_acc'), ('balanced_accuracy', 'd_bacc')):
            mi = METRICS.index(k)
            d = boots[sp.name][:, mi] - boots[FULL][:, mi]; d = d[np.isfinite(d)]
            row[short] = float(per[k].mean() - per_by[FULL][k].mean())
            row[f'{short}_lo'], row[f'{short}_hi'] = float(np.quantile(d, .025)), float(np.quantile(d, .975))
            row[f'{short}_p'] = float(min(1.0, 2 * (min((d <= 0).sum(), (d >= 0).sum()) + 1) / (len(d) + 1)))
        g_ = fold_df[fold_df.method == sp.name]
        row['mean_inner_minus_outer_auc'] = float((g_.inner_auc - g_.auc).mean())
        row['mean_pool_size'] = float(g_.n_pool.mean())
        row['top_chosen'] = g_.chosen.value_counts().index[0]
        row['top_chosen_freq'] = float(g_.chosen.value_counts(normalize=True).iloc[0])
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, 'ablation_selectbest_inner.csv'), index=False)
    write_latex(res, os.path.join(OUT_DIR, 'ablation_table.tex'))

    # ---- console report ----
    def fmt(r, k):
        sd = r[f'{k}_sd']
        return f'{r[k]:.3f}±{sd:.3f}' if np.isfinite(sd) else f'{r[k]:.3f}'
    print('\n' + '=' * 132)
    print(f'ABLATION OF SelectBest_inner | MODMA REST | {N_SUBJ} subjects | {R}x{K} subject-wise CV | 95% subject-bootstrap CI, paired vs Full')
    print('=' * 132)
    for fam, d in res.groupby('family', sort=False):
        print(f'\n--- {fam} ---')
        tab = pd.DataFrame({
            'Variant': d.method,
            'AUC [95% CI]': [f'{fmt(r, "auc")} [{r.auc_lo:.3f},{r.auc_hi:.3f}]' for _, r in d.iterrows()],
            'dAUC [95% CI]': ['--' if r.method == FULL else f'{r.d_auc:+.3f} [{r.d_auc_lo:+.3f},{r.d_auc_hi:+.3f}] p={r.d_auc_p:.2f}' for _, r in d.iterrows()],
            'Acc': [fmt(r, 'accuracy') for _, r in d.iterrows()], 'dAcc': ['--' if r.method == FULL else f'{r.d_acc:+.3f}' for _, r in d.iterrows()],
            'BalAcc': [fmt(r, 'balanced_accuracy') for _, r in d.iterrows()], 'Sens': [fmt(r, 'recall') for _, r in d.iterrows()],
            'Spec': [fmt(r, 'specificity') for _, r in d.iterrows()], 'F1': [fmt(r, 'f1') for _, r in d.iterrows()],
            'MCC': [fmt(r, 'mcc') for _, r in d.iterrows()], 'gap(in-out AUC)': d.mean_inner_minus_outer_auc.round(3).values,
            'top choice (freq)': [f'{r.top_chosen} ({r.top_chosen_freq:.2f})' for _, r in d.iterrows()]})
        print(tab.to_string(index=False))

    ch = fold_df[fold_df.method == FULL]
    print('\nFull SelectBest_inner: expert chosen per fold (frequency over all outer folds):')
    print(ch.chosen.value_counts(normalize=True).round(2).to_string())
    print('By stream:'); print(ch.chosen.str.split('|').str[0].value_counts(normalize=True).round(2).to_string())
    orc = fold_df[fold_df.method == 'oracle (post-hoc, optimistic)']
    print(f'\nSelection regret (mean outer AUC of oracle choice - Full choice, per fold): '
          f'{(orc.outer_auc_chosen.values - ch.outer_auc_chosen.values).mean():.3f} | '
          f'mean inner->outer AUC drop of Full: {(ch.inner_auc - ch.auc).mean():.3f}')
    ch.to_csv(os.path.join(OUT_DIR, 'full_selection_choices.csv'), index=False)

    # ---- forest plot of paired dAUC ----
    pl = res[res.method != FULL].reset_index(drop=True)
    fams = list(dict.fromkeys(pl.family)); cmap = plt.get_cmap('tab10')
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(pl) + 1.6))
    for i, r in pl.iterrows():
        ax.errorbar(r.d_auc, i, xerr=[[r.d_auc - r.d_auc_lo], [r.d_auc_hi - r.d_auc]], fmt='o', ms=4,
                    color=cmap(fams.index(r.family) % 10), capsize=2, lw=1.2)
    ax.axvline(0, color='k', lw=0.8, ls='--'); ax.set_yticks(range(len(pl))); ax.set_yticklabels(pl.method, fontsize=7)
    ax.invert_yaxis(); ax.set_xlabel('paired $\\Delta$AUC vs full SelectBest_inner (95% subject-bootstrap CI)')
    ax.set_title(f'SelectBest_inner ablation | MODMA REST (full AUC {res.loc[res.method == FULL, "auc"].iloc[0]:.3f})', fontsize=9)
    ax.legend(handles=[plt.Line2D([], [], marker='o', ls='', color=cmap(i % 10), label=f) for i, f in enumerate(fams)],
              fontsize=6, loc='lower left')
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'ablation_delta_auc.png'), dpi=250); plt.close(fig)

    with open(os.path.join(OUT_DIR, 'run_manifest.json'), 'w') as f:
        json.dump(dict(model_version=MODEL_VERSION, hash=tag_hash, emb_key=EMB_KEY, seed=SEED, n_subjects=N_SUBJ,
                       variants=variants, specs=[s.__dict__ for s in specs], cfg=CFG, pre=PRE,
                       versions=dict(torch=torch.__version__, mne=mne.__version__, numpy=np.__version__, pandas=pd.__version__,
                                     sklearn=sklearn.__version__, scipy=scipy.__version__, xgboost=xgboost.__version__,
                                     python=sys.version.split()[0])), f, indent=2, default=str)
    log('DONE. Results in ' + OUT_DIR)
