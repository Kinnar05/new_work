# ================================================================
# Frozen MULTI-LAYER CBraMod latent -> cross-fitted diffusion streams (normative deviation + DISCRIMINATIVE
# diffusion-consistency adapter) -> Klein fusion + MICROSTATE dynamics stream
#   -> in-fold STABLE-EXPERT SELECTION (6-8 of 35 candidates) -> calibrated hierarchical Product-of-Experts
# MODMA 128-ch resting EEG, MDD vs HC, repeated subject-wise CV (no leakage).   v3
#
# What changed vs v2
#   1. MULTI-LAYER BACKBONE. CBraMod is still frozen, but hidden states of CFG['layers'] (default 4, 8, 12) are
#      patch-mean-pooled and cached as [n_windows, n_layers, 18*200]. Inside every outer fold each layer gets its own
#      train-only StandardScaler + whitened PCA (layer_pca_dim comps); the per-layer blocks are concatenated into
#      the diffusion latent. The deviation features' `n_blocks` blocks coincide with the layers, i.e. the normative
#      model now reports a deviation per layer.
#   2. DISCRIMINATIVE DIFFUSION-CONSISTENCY ADAPTER (replaces the class-conditional DDPM delta).
#      A shared denoising trunk + two zero-initialised low-rank class adapters (LoRA-style heads, eps_c = eps_trunk +
#      A_c(h)). Loss = denoising(true-class) + trunk denoising + DISCRIMINATIVE term (BCE on kappa*(log e_HC - log e_MDD),
#      i.e. the diffusion-classifier logit is trained directly) + CONSISTENCY term (the logit must agree between two
#      independent (t, eps) draws of the same latent). Features = per-timestep log-error ratio + kappa-weighted mean.
#      Still cross-fitted: every training subject is scored by a member that never saw it.
#   3. STABLE EXPERT SELECTION. Candidate pool = 5 streams x 7 classifiers = 35 (LSTM / epoch-transformer are code-gated by
#      CBR_SEQ=1 because they are the slowest, least stable members). Inside every outer fold, using inner-OOF data only:
#      per-expert calibration -> subject-bootstrap stability score (mean AUC - lambda*sd) -> greedy selection with
#      (a) best expert of each stream first, (b) AUC floor, (c) per-stream / per-classifier caps, (d) logit
#      decorrelation. 6-8 experts are fused. Selection frequencies across folds are reported ("consensus set").
#   4. MICROSTATE STREAM. K=4 polarity-invariant modified K-means maps are fitted on GFP-peak topographies of the
#      TRAINING windows only; every window is back-fitted, temporally smoothed (min segment 30 ms) and summarised by
#      coverage / duration / occurrence / GEV / transition probabilities / entropies (33 features). Independent of
#      CBraMod, so it is a genuinely different expert. An ablation `PoE_sel_hier_eq_noMS` quantifies its contribution.
#
#   Housekeeping fixes: removed the dead `* 0` terms (delta_features, paired-accuracy lambda).
#
#   Env switches: CBR_QUICK=1 (tiny), CBR_SYNTH=1 (synthetic, no data/ckpt), CBR_SEQ=1 (add LSTM + transformer),
#                 CBR_LAYERS="4,8,12", CBR_RESUME=0/1.
# ================================================================
import os, re, sys, glob, math, json, copy, time, random, hashlib, warnings, subprocess
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
from scipy.optimize import minimize
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

QUICK = os.environ.get('CBR_QUICK') == '1'      # tiny training, 1 repeat (smoke test)
SYNTH = os.environ.get('CBR_SYNTH') == '1'      # synthetic data, no files / checkpoint needed (pipeline smoke test)
USE_SEQ = os.environ.get('CBR_SEQ') == '1'      # also run the LSTM / epoch-transformer experts
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mne.set_log_level('ERROR')
MODEL_VERSION = 'cbramod-ml-dcadapter-ms-selpoe-v3'


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
OUT_DIR = os.path.join(WORK_DIR, 'results_cbramod_poe_v3')
CKPT_DIR = os.path.join(WORK_DIR, 'ckpt_cbramod_poe_v3')
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(CKPT_DIR, exist_ok=True)
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
    layers=LAYERS, layer_pca_dim=16 if Q else 32,          # latent_dim = len(layers) * layer_pca_dim
    n_blocks=len(LAYERS),                                   # deviation blocks == layers
    T=200, eval_timesteps=(10, 50, 100, 150), eval_repeats=2, cross_fit_k=3,
    # healthy-only normative diffusion (deviation stream)
    NORM=dict(hidden=256, n_res=3, epochs=3 if Q else 80, batch=128, draws=60, lr=3e-4, wd=1e-4, ema=0.995),
    # discriminative diffusion-consistency adapter (delta stream)
    DELTA=dict(hidden=128, n_res=2, rank=16, adapter_dropout=0.1, epochs=3 if Q else 40, batch=256, draws=24,
               lr=3e-4, wd=1e-2, tmax=160, kappa0=10.0,
               w_den=1.0, w_base=0.25, w_disc=1.0, w_cons=0.5),
    member_scoring='single',                    # 'single' (distribution-matched) | 'mean'
    klein_dim=16, klein_curvatures=(0.5, 1.0, 2.0), klein_tau=0.5, klein_source='both',
    subj_logreg_C=0.05,
    # microstates
    MS=dict(k=4, n_init=2 if Q else 8, iters=50, per_subj=250, min_seg_ms=30.0, tol=1e-5),
    # LSTM / epoch transformer (only used when CBR_SEQ=1)
    lstm_hidden=32, lstm_dropout=0.3, lstm_lr=2e-3, lstm_weight_decay=1e-2, lstm_epochs=3 if Q else 60, lstm_batch=8,
    tf_d_model=16, tf_heads=2, tf_layers=1, tf_ff=32, tf_dropout=0.3, tf_in_dropout=0.2,
    tf_lr=5e-4, tf_weight_decay=1e-1, tf_epochs=3 if Q else 40, tf_batch=8, tf_max_tokens=40,
    tf_noise=0.2, tf_mix_alpha=0.4, tf_mix_prob=0.5, tf_logit_l2=1e-2, tf_ema=0.98, tf_n_seeds=1 if Q else 2, tf_tta=8,
    # fusion + selection
    logit_clip=1e-6, platt_C=1.0, temp_C=10.0, lam_hier=0.3, lam_flat=3.0,
    sel_boot=50 if Q else 300, sel_lambda=1.0, sel_min_auc=0.55, sel_min=6, sel_max=8,
    sel_per_stream=3, sel_per_clf=3, sel_corr=0.95,
    fusion_thr='half',
    # CV
    cv_n_splits=5, cv_repeats=1 if Q else 3, cv_inner_splits=3 if Q else 4,
    n_boot=50 if Q else 1000,
    resume=os.environ.get('CBR_RESUME', '1') == '1',
)
STREAMS = ('cbramod', 'cbr_delta', 'ndm', 'klein', 'microstate')
BASE_CLFS = ('subj_lda', 'subj_logreg', 'bagged_logreg', 'logreg', 'svm_rbf', 'random_forest', 'xgboost') \
    + (('lstm', 'transformer') if USE_SEQ else ())
ENS_MEMBERS = ('subj_lda', 'subj_logreg', 'bagged_logreg')
BASE_EXPERTS = [f'{s}|{c}' for s in STREAMS for c in BASE_CLFS]            # candidate pool
ENS_EXPERTS = [f'{s}|ensemble' for s in STREAMS]                            # derived soft votes (reported only)
ALL_EXPERTS = BASE_EXPERTS + ENS_EXPERTS
FUSION_NAMES = (['SelectBest_inner', 'PoE_all_flat_eq', 'PoE_all_hier_eq', 'PoE_sel_flat_eq', 'PoE_sel_flat',
                 'PoE_sel_hier_eq', 'PoE_sel_hier', 'PoE_sel_hier_eq_noMS'] + [f'PoE_stream_{s}' for s in STREAMS])
METHODS = ALL_EXPERTS + FUSION_NAMES
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
print('Device:', DEVICE, '| torch', torch.__version__, '| mne', mne.__version__, '| sklearn', sklearn.__version__,
      '| layers', LAYERS, '| seq models', USE_SEQ)

# ================================================================
# 1. SELF-CONTAINED CBraMod BACKBONE (matches released checkpoint) + multi-layer read-out
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
    return model

# ================================================================
# 2. MODMA discovery, labels, preprocessing, multi-layer CBraMod latent extraction
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


def build_latent_dataset(model, manifest):
    sig = json.dumps(dict(pre=PRE, ckpt=sha256_file(CBRAMOD_CKPT)[:12], layers=LAYERS, files=sorted(
        (os.path.basename(p), os.path.getsize(p)) for p in manifest['path']), v=3), sort_keys=True, default=str)
    key = hashlib.sha1(sig.encode()).hexdigest()[:10]
    cache = os.path.join(CKPT_DIR, f'cbramod_ml_latent_{key}.npz')
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        log(f'loaded latent cache {cache}')
        return d['X'].astype(np.float32), d['y'].astype(int), d['g'].astype(str), d['W'].astype(np.float32), key
    XX, WW, yy, gg, audit = [], [], [], [], []
    for k, row in manifest.iterrows():
        try:
            w, qc = preprocess_one_mat(row.path)
            rec = dict(subject=row.subject, label=row.label, file=row.file, error='', **qc)
            if len(w) == 0:
                rec['error'] = 'zero usable windows'; audit.append(rec); print('ZERO WINDOWS:', row.file); continue
            z = extract_latent(model, w)
            XX.append(z); WW.append(w.reshape(len(w), len(CBRAMOD_CHANNELS), -1))     # raw windows feed the microstate stream
            yy += [row.label] * len(z); gg += [row.subject] * len(z); audit.append(rec)
            print(f'[{k + 1:02d}/{len(manifest)}] {row.subject} {"MDD" if row.label else "HC "} windows={len(z):2d} '
                  f'rejected={qc["rejected"]}/{qc["considered"]} repaired={qc["repaired"]} bad_ch={qc["persistent_bad_count"]}')
        except Exception as e:
            audit.append(dict(subject=row.subject, label=row.label, file=row.file, error=str(e)))
            print('SKIP', row.file, '->', e)
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


def synthetic_latent(n_mdd=24, n_hc=29, win=12, dim=800, n_layers=None, seed=0):
    """Pipeline smoke test: layered low-rank latent + subject random effect + weak class shift, and synthetic
    EEG windows built from 4 switching topographies (class-dependent state probabilities). No real data involved."""
    rng = np.random.default_rng(seed)
    n_layers = n_layers or len(LAYERS)
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
# 3. Per-layer latent space + diffusion banks (normative deviation + discriminative consistency adapter)
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
    """Train-only, per-layer StandardScaler + whitened PCA; blocks are concatenated -> [n, n_layers * k]."""
    def fit(self, X, seed):
        n, L, D = X.shape
        self.k = int(min(CFG['layer_pca_dim'], n - 1, D))
        self.sc, self.pca = [], []
        for l in range(L):
            s = StandardScaler().fit(X[:, l])
            self.sc.append(s)
            self.pca.append(PCA(self.k, whiten=True, svd_solver='randomized', random_state=seed + l).fit(s.transform(X[:, l])))
        self.explained = [float(p.explained_variance_ratio_.sum()) for p in self.pca]
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
    """Shared denoising trunk + two zero-initialised low-rank class adapters.
       eps_hat_c = out(h) + A_c(h),  h = trunk(z_t, t).   Returns (eps_trunk, eps_HC, eps_MDD)."""
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
       L = w_den * denoise(true-class head) + w_base * denoise(trunk)
         + w_disc * BCE(kappa * [log e_HC - log e_MDD], y)         (diffusion-classifier logit trained directly)
         + w_cons * (logit(t_a, eps_a) - logit(t_b, eps_b))^2      (consistency across independent noise draws)
       Subject-balanced sampling and class-balanced weights; t restricted to [1, tmax] (the evaluation range)."""
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
    """Normative deviation: log denoising error per noise level - global + n_blocks (= layers) blocks -> K*(1+n_blocks) columns."""
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
    """Per-timestep diffusion-classifier log-ratio d_t = log e_HC - log e_MDD on the SAME noisy latent (K columns),
    plus one consistency-averaged logit kappa * mean_t d_t  -> K+1 columns."""
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
    members, held, hists = [], {}, [hist]
    for j, part in enumerate(parts):
        keep = hc[~np.isin(gtr[hc], part)]
        m, h = train_latent_diffusion(Ztr[keep], gtr[keep], seed + 100 * (j + 1), CFG['NORM'])
        members.append(m); hists.append(h)
        for s in part:
            held[str(s)] = j
    return Bank(members, held, full), hists


def build_delta_bank(Ztr, ytr, gtr, seed):
    sl = subject_labels(ytr, gtr)
    subs, lab = sl.index.values.astype(str), sl.values.astype(int)
    k = int(min(CFG['cross_fit_k'], lab.sum(), (1 - lab).sum()))
    if k < 2:
        raise RuntimeError('too few subjects per class for adapter cross-fitting')
    members, held, hists = [], {}, []
    for j, (fi, hi) in enumerate(StratifiedKFold(k, shuffle=True, random_state=seed + 17).split(subs, lab)):
        rows = np.isin(gtr, subs[fi])
        m, h = train_dc_adapter(Ztr[rows], ytr[rows], gtr[rows], seed + 100 * (j + 1) + 50, CFG['DELTA'])
        members.append(m); hists.append(h)
        for s in subs[hi]:
            held[str(s)] = j
    return Bank(members, held), hists


def crossfit_apply(bank, Z, g, seed, fn):
    """Training subjects that a member never saw -> scored by that member. Everyone else (MDD-train for the normative
    bank, all test subjects) -> ONE member each ('single', distribution-matched) or the member mean ('mean')."""
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
            assign = {subs[i]: int(r % k) for r, i in enumerate(order)}          # label-free, balanced over members
            mem = np.array([assign[s] for s in g[idx]])
            for j in range(k):
                sel = idx[mem == j]
                if len(sel):
                    put(sel, fn(bank.members[j], Z[sel], seed))
    return out[0]


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def hc_gap(v_tr, y_tr, g_tr, v_te, y_te, g_te, cols):
    """Held-out HC vs training HC subject means of the mean over `cols` (Cohen d). ~0 => train/test features exchangeable."""
    def hm(v, y, g):
        t = pd.DataFrame({'v': v[:, cols].mean(1), 'g': g, 'y': y}).groupby('g').agg(v=('v', 'mean'), y=('y', 'first'))
        return t.v[t.y == 0]
    return cohen_d(hm(v_te, y_te, g_te), hm(v_tr, y_tr, g_tr))

# ================================================================
# 4. Microstate dynamics stream (templates fitted on TRAIN windows only)
# ================================================================
def _smooth_labels(lab, ac, min_len):
    """Merge segments shorter than min_len into the neighbouring state that explains them best."""
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
    """Polarity-invariant modified K-means on GFP-peak topographies + back-fitting + dynamics features."""
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
            x = Wb[wi, :, ti]                                                    # [m, C]
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
        self.T = best[np.argsort(-np.bincount(lab, minlength=self.k), kind='stable')]    # state 0 = most frequent in TRAIN
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
# 6. Streams and classifiers
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


def fit_lstm(seqs, ys, seed):
    c = CFG
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = SeqLSTM(seqs[0].shape[1], c['lstm_hidden'], c['lstm_dropout'])
    opt = torch.optim.AdamW(model.parameters(), lr=c['lstm_lr'], weight_decay=c['lstm_weight_decay'])
    n_pos = max(1, int(ys.sum())); n_neg = max(1, len(ys) - n_pos)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / n_pos, dtype=torch.float32))
    yt = torch.tensor(ys, dtype=torch.float32)
    model.train()
    for _ in range(c['lstm_epochs']):
        order = rng.permutation(len(seqs))
        for b in range(0, len(order), c['lstm_batch']):
            ids = order[b:b + c['lstm_batch']]
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
    """Linear skip + zero-initialised transformer residual over a subject's epoch set."""
    def __init__(self, d_in, d_model, n_heads, n_layers, ff, dropout, in_dropout):
        super().__init__()
        self.in_drop = nn.Dropout(in_dropout)
        self.lin = nn.Linear(d_in, 1); nn.init.normal_(self.lin.weight, std=0.01); nn.init.zeros_(self.lin.bias)
        self.proj = nn.Linear(d_in, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ff, dropout, activation='gelu', batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1); nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x, lengths):
        B, L, _ = x.shape
        valid = (torch.arange(L)[None, :] < lengths[:, None]).float().unsqueeze(-1)
        n = lengths[:, None].float()
        xd = self.in_drop(x)
        lin = self.lin((xd * valid).sum(1) / n).squeeze(-1)
        h = self.enc(self.proj(xd), src_key_padding_mask=(valid.squeeze(-1) == 0))
        pooled = (self.norm(h) * valid).sum(1) / n
        return lin + self.head(self.drop(pooled)).squeeze(-1)


def _bag(seq, rng, cap, random_size=True):
    L = len(seq); hi = min(L, cap)
    k = int(rng.integers(min(max(1, L // 2), hi), hi + 1)) if random_size else hi
    return seq[np.sort(rng.choice(L, size=k, replace=False))]


def _mixed_bag(seqs, ys, i, rng, cap, alpha, prob):
    if alpha <= 0 or rng.random() >= prob:
        return _bag(seqs[i], rng, cap), float(ys[i])
    j = int(rng.integers(len(seqs)))
    k = int(min(cap, max(2, max(len(seqs[i]), len(seqs[j])) // 2)))
    ka = int(round(float(rng.beta(alpha, alpha)) * k)); kb = k - ka
    a = seqs[i][rng.choice(len(seqs[i]), size=min(ka, len(seqs[i])), replace=False)]
    b = seqs[j][rng.choice(len(seqs[j]), size=min(kb, len(seqs[j])), replace=False)]
    bag = np.concatenate([a, b], axis=0)
    return bag, float((ys[i] * len(a) + ys[j] * len(b)) / len(bag))


def fit_transformer(seqs, ys, seed):
    c = CFG
    n = len(ys); n_pos = max(1, int(ys.sum())); n_neg = max(1, n - n_pos)
    w_pos, w_neg = n / (2 * n_pos), n / (2 * n_neg)
    steps = c['tf_epochs'] * math.ceil(len(seqs) / c['tf_batch'])
    warm = max(1, int(0.1 * steps))
    models = []
    for m in range(c['tf_n_seeds']):
        s = seed + 1000 * (m + 1)
        torch.manual_seed(s); rng = np.random.default_rng(s)
        model = EpochTransformer(seqs[0].shape[1], c['tf_d_model'], c['tf_heads'], c['tf_layers'], c['tf_ff'], c['tf_dropout'], c['tf_in_dropout'])
        ema = copy.deepcopy(model)
        for p in ema.parameters():
            p.requires_grad_(False)
        opt = torch.optim.AdamW(model.parameters(), lr=c['tf_lr'], weight_decay=c['tf_weight_decay'])
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda i: min(1.0, (i + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, i / steps)))))
        model.train(); t = 0
        for _ in range(c['tf_epochs']):
            order = rng.permutation(len(seqs))
            for b in range(0, len(order), c['tf_batch']):
                ids = order[b:b + c['tf_batch']]
                pairs = [_mixed_bag(seqs, ys, i, rng, c['tf_max_tokens'], c['tf_mix_alpha'], c['tf_mix_prob']) for i in ids]
                x, lens = _pad([p[0] for p in pairs])
                tgt = torch.tensor([p[1] for p in pairs], dtype=torch.float32)
                x = x + c['tf_noise'] * torch.randn_like(x) * (torch.arange(x.size(1))[None, :, None] < lens[:, None, None])
                lg = model(x, lens)
                wts = tgt * w_pos + (1 - tgt) * w_neg
                loss = (wts * F.binary_cross_entropy_with_logits(lg, tgt, reduction='none')).mean() + c['tf_logit_l2'] * (lg ** 2).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
                d = min(c['tf_ema'], (1 + t) / (10 + t)); t += 1
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.mul_(d).add_(pm.detach(), alpha=1 - d)
        models.append(ema.eval())
    return models


@torch.no_grad()
def predict_transformer(models, seqs):
    rng = np.random.default_rng(0)
    out = np.zeros(len(seqs))
    for _ in range(CFG['tf_tta']):
        bags = [_bag(s, rng, CFG['tf_max_tokens'], random_size=False) for s in seqs]
        for model in models:
            for b in range(0, len(bags), 16):
                x, lens = _pad(bags[b:b + 16])
                out[b:b + 16] += torch.sigmoid(model(x, lens)).numpy()
    return out / (CFG['tf_tta'] * len(models))


def subject_weights(g):
    vals, counts = np.unique(g, return_counts=True)
    d = dict(zip(vals, counts))
    w = np.asarray([1.0 / d[s] for s in g], np.float64)
    return w / w.mean()


def subject_summary(A, rows, subs):
    return np.stack([np.concatenate([A[rows[s]].mean(0), A[rows[s]].std(0)]) for s in subs]).astype(np.float32)


def score_subjects(clf_name, Ftr, ytr, gtr, Fte, gte, seed):
    """-> (subject ids, P(MDD)) for one base classifier. All classifiers share the same interface."""
    sc = StandardScaler().fit(Ftr)
    A, B = sc.transform(Ftr).astype(np.float32), sc.transform(Fte).astype(np.float32)
    gtr, gte = np.asarray(gtr).astype(str), np.asarray(gte).astype(str)
    if clf_name in ('lstm', 'transformer'):
        tr_rows, te_rows = group_rows(gtr), group_rows(gte)
        s_tr, s_te = list(tr_rows), list(te_rows)
        seqs = [A[tr_rows[s]] for s in s_tr]
        ys = np.array([ytr[tr_rows[s][0]] for s in s_tr])
        te_seqs = [B[te_rows[s]] for s in s_te]
        if clf_name == 'lstm':
            return np.array(s_te), predict_lstm(fit_lstm(seqs, ys, seed), te_seqs)
        return np.array(s_te), predict_transformer(fit_transformer(seqs, ys, seed), te_seqs)
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
    w = subject_weights(gtr)                     # equal total weight per subject for epoch-level classifiers
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


def add_ensembles(P):
    """Derived soft-vote of subj_lda / subj_logreg / bagged_logreg per stream (no extra fitting)."""
    for s in STREAMS:
        P[f'{s}|ensemble'] = P[[f'{s}|{c}' for c in ENS_MEMBERS]].mean(axis=1)
    return P

# ================================================================
# 7. Stable-expert selection + calibrated hierarchical Product-of-Experts
# ================================================================
def to_logit(p):
    c = CFG['logit_clip']
    return logit(np.clip(np.asarray(p, float), c, 1 - c))


def platt(x, y, C):
    """Balanced 1-D logistic calibration: logit P(MDD) = a*x + b  (balanced => 0.5 is the balanced-error operating point)."""
    lr = LogisticRegression(C=C, class_weight='balanced', max_iter=1000).fit(np.asarray(x, float)[:, None], y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def fit_poe(L, y, lam, w0):
    """Weighted PoE  logit P = b + sum_e w_e * l_e,  w_e >= 0, MAP under w ~ N(w0, 1/(2 lam)): shrinks to the equal-weight PoE."""
    n, E = L.shape
    y = np.asarray(y, float)
    npos, nneg = max(1, int(y.sum())), max(1, int(n - y.sum()))
    sw = np.where(y == 1, n / (2 * npos), n / (2 * nneg))
    def f(th):
        w, b = th[:E], th[E]
        z = L @ w + b
        nll = np.sum(sw * (np.logaddexp(0.0, z) - y * z)) / n + lam * np.sum((w - w0) ** 2)
        r = sw * (expit(z) - y)
        return nll, np.concatenate([L.T @ r / n + 2 * lam * (w - w0), [r.sum() / n]])
    res = minimize(f, np.concatenate([w0, [0.0]]), jac=True, method='L-BFGS-B', bounds=[(0, None)] * E + [(None, None)])
    return res.x[:E], float(res.x[E])


def youden_threshold(y, p):
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.isfinite(thr)
    return float(thr[ok][np.argmax((tpr - fpr)[ok])]) if ok.any() else 0.5


def select_experts(L_in, y_in, cols, cand, seed):
    """Stable-expert selection on inner-OOF logits only.
       score_e = mean_b AUC_b(e) - lambda * sd_b AUC_b(e)   over subject-bootstrap resamples b  (stability-penalised AUC).
       Greedy: (1) best expert of every stream (needs mean AUC >= floor), (2) fill by score up to sel_max subject to
       per-stream / per-classifier caps and |corr| < sel_corr with already chosen experts, (3) if fewer than sel_min,
       relax the AUC floor. Returns (chosen column indices, {col: score})."""
    c = CFG
    if not cand:
        return [], {}
    y_in = np.asarray(y_in).astype(int); n = len(y_in)
    rng = np.random.default_rng(seed + 977)
    aucs = []
    for _ in range(c['sel_boot']):
        ii = rng.integers(0, n, n); yb = y_in[ii]; n1 = int(yb.sum()); n0 = n - n1
        if n1 == 0 or n0 == 0:
            continue
        rk = rankdata(L_in[ii][:, cand], axis=0)
        aucs.append((rk[yb == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0))
    if not aucs:
        return [cand[0]], {cand[0]: np.nan}
    aucs = np.asarray(aucs); mu, sd = aucs.mean(0), aucs.std(0)
    score = mu - c['sel_lambda'] * sd
    order = list(np.argsort(-score))
    st = [cols[j].split('|')[0] for j in cand]; cl = [cols[j].split('|')[1] for j in cand]
    chosen = []
    def ok(i, floor=True):
        if floor and mu[i] < c['sel_min_auc']:
            return False
        if sum(st[k] == st[i] for k in chosen) >= c['sel_per_stream'] or sum(cl[k] == cl[i] for k in chosen) >= c['sel_per_clf']:
            return False
        for k in chosen:
            r = np.corrcoef(L_in[:, cand[i]], L_in[:, cand[k]])[0, 1]
            if np.isfinite(r) and abs(r) > c['sel_corr']:
                return False
        return True
    for s in STREAMS:                                                   # (1) one representative per stream
        for i in order:
            if st[i] == s and ok(i):
                chosen.append(i); break
    for i in order:                                                     # (2) fill
        if len(chosen) >= c['sel_max']:
            break
        if i not in chosen and ok(i):
            chosen.append(i)
    for i in order:                                                     # (3) reach the minimum size
        if len(chosen) >= c['sel_min']:
            break
        if i not in chosen and ok(i, floor=False):
            chosen.append(i)
    if not chosen:
        chosen = [order[0]]
    return [cand[i] for i in chosen], {cand[i]: float(score[i]) for i in range(len(cand))}


def fuse_all(L_in, L_te, y_in, cols, seed):
    """L_in: inner-OOF subject logits [n_in, E]; L_te: outer-test logits [n_te, E]; cols: 'stream|clf'.
    Everything (calibration, selection, PoE weights) is fitted on the inner-OOF data only.
    Returns ({method: (p_in, p_te)}, diagnostic rows)."""
    y_in = np.asarray(y_in).astype(int)
    E, n_in, n_te = len(cols), len(L_in), len(L_te)
    stream_of = [c.split('|')[0] for c in cols]

    ab = np.array([platt(L_in[:, j], y_in, CFG['platt_C']) for j in range(E)])      # 1) per-expert calibration
    act = ab[:, 0] > 0                                                              #    slope <= 0 => no skill => off
    a = np.where(act, ab[:, 0], 0.0); b = np.where(act, ab[:, 1], 0.0)
    Ci, Ct = L_in * a + b, L_te * a + b
    inner_auc = np.array([roc_auc_score(y_in, L_in[:, j]) if len(np.unique(y_in)) > 1 else 0.5 for j in range(E)])
    cand_all = [j for j in range(E) if act[j]]

    def half():
        return np.full(n_in, .5), np.full(n_te, .5)
    def gm(C, idx):
        return C[:, idx].mean(1) if idx else np.zeros(len(C))
    def temp(gi, gt):                                                               # 2) global temperature + bias
        s, b0 = platt(gi, y_in, CFG['temp_C']); s = max(s, 0.05)
        return expit(s * gi + b0), expit(s * gt + b0), s

    def block(idx, tag, learned):
        """flat / hierarchical equal-weight PoE (+ learned shrunk versions) over the experts `idx`."""
        res, wr = {}, []
        if not idx:
            for k in ('flat_eq', 'flat', 'hier_eq', 'hier'):
                res[k] = half()
            return res, wr
        pi, pt, s_flat = temp(gm(Ci, idx), gm(Ct, idx)); res['flat_eq'] = (pi, pt)
        s_act = [s for s in STREAMS if any(stream_of[j] == s for j in idx)]
        Si = np.column_stack([gm(Ci, [j for j in idx if stream_of[j] == s]) for s in s_act])
        St = np.column_stack([gm(Ct, [j for j in idx if stream_of[j] == s]) for s in s_act])
        pi, pt, s_eq = temp(Si.mean(1), St.mean(1)); res['hier_eq'] = (pi, pt)
        if learned:
            Xi, Xt = Ci[:, idx], Ct[:, idx]
            w, b0 = fit_poe(Xi, y_in, CFG['lam_flat'], np.full(len(idx), s_flat / len(idx)))
            res['flat'] = (expit(Xi @ w + b0), expit(Xt @ w + b0))
            wr += [dict(fusion=f'PoE_{tag}_flat', expert=cols[j], weight=float(wi)) for j, wi in zip(idx, w)]
            w, b0 = fit_poe(Si, y_in, CFG['lam_hier'], np.full(len(s_act), s_eq / len(s_act)))
            res['hier'] = (expit(Si @ w + b0), expit(St @ w + b0))
            wr += [dict(fusion=f'PoE_{tag}_hier', expert=s, weight=float(wi)) for s, wi in zip(s_act, w)]
        return res, wr

    chosen, sc = select_experts(L_in, y_in, cols, cand_all, seed)                   # 3) stable-expert selection
    chosen_noms, _ = select_experts(L_in, y_in, cols, [j for j in cand_all if stream_of[j] != 'microstate'], seed)

    out, wrows = {}, []
    r, _ = block(cand_all, 'all', False)
    out['PoE_all_flat_eq'], out['PoE_all_hier_eq'] = r['flat_eq'], r['hier_eq']
    r, wr = block(chosen, 'sel', True); wrows += wr
    out['PoE_sel_flat_eq'], out['PoE_sel_flat'], out['PoE_sel_hier_eq'], out['PoE_sel_hier'] = r['flat_eq'], r['flat'], r['hier_eq'], r['hier']
    r, _ = block(chosen_noms, 'selnoMS', False)
    out['PoE_sel_hier_eq_noMS'] = r['hier_eq']
    j_best = max(cand_all, key=lambda j: inner_auc[j]) if cand_all else int(np.argmax(inner_auc))
    out['SelectBest_inner'] = (expit(Ci[:, j_best]), expit(Ct[:, j_best]))
    for s in STREAMS:
        idx = [j for j in cand_all if stream_of[j] == s]
        out[f'PoE_stream_{s}'] = temp(gm(Ci, idx), gm(Ct, idx))[:2] if idx else half()
    wrows += [dict(fusion='calibration', expert=cols[j], weight=float(a[j]), bias=float(b[j]), inner_auc=float(inner_auc[j]),
                   active=bool(act[j])) for j in range(E)]
    wrows += [dict(fusion='selection', expert=cols[j], selected=bool(j in chosen), score=float(sc.get(j, np.nan)),
                   active=bool(act[j])) for j in range(E)]
    return out, wrows


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
# 8. One outer fold
# ================================================================
def fold_stage(rep, fold, tr_subj, te_subj, D):
    seed = SEED + rep * CFG['cv_n_splits'] + fold + 1
    seed_everything(seed)
    t0, tag = time.time(), f'rep {rep + 1} fold {fold + 1}'
    X, y, g, Wraw = D['X'], D['y'], D['g'], D['W']
    tr, te = np.where(np.isin(g, tr_subj))[0], np.where(np.isin(g, te_subj))[0]
    ytr, yte, gtr, gte = y[tr], y[te], g[tr], g[te]
    K = len(CFG['eval_timesteps'])

    # (a) train-only per-layer scaler + whitened PCA of the frozen multi-layer CBraMod latent
    LL = LayerLatent().fit(X[tr], seed)
    Ztr, Zte = LL.transform(X[tr]), LL.transform(X[te])

    # (b) healthy-only normative bank -> deviation stream (cross-fitted)
    nbank, nh = build_norm_bank(Ztr, ytr, gtr, seed)
    devf_tr, devf_te = deviation_features(nbank.full, Ztr, seed + 1), deviation_features(nbank.full, Zte, seed + 1)
    dev_tr = crossfit_apply(nbank, Ztr, gtr, seed + 1, deviation_features)
    dev_te = crossfit_apply(nbank, Zte, gte, seed + 1, deviation_features)
    gr = dict(repeat=rep, fold=fold, d_norm_full_model=hc_gap(devf_tr, ytr, gtr, devf_te, yte, gte, list(range(K))),
              d_norm_used=hc_gap(dev_tr, ytr, gtr, dev_te, yte, gte, list(range(K))),
              pca_explained=float(np.mean(LL.explained)))
    loss_rows = [dict(repeat=rep, fold=fold, model=('norm_full' if i == 0 else f'norm_member{i - 1}'), epoch=e + 1, loss=v)
                 for i, h in enumerate(nh) for e, v in enumerate(h)]
    del nbank

    # (c) discriminative diffusion-consistency adapter bank -> delta stream (cross-fitted)
    dbank, dh = build_delta_bank(Ztr, ytr, gtr, seed)
    dlt_tr = crossfit_apply(dbank, Ztr, gtr, seed + 2, adapter_features)
    dlt_te = crossfit_apply(dbank, Zte, gte, seed + 2, adapter_features)
    gr['d_delta_hc_gap'] = hc_gap(dlt_tr, ytr, gtr, dlt_te, yte, gte, list(range(K)))
    gr['adapter_kappa'] = float(np.mean([float(m.kappa()) for m in dbank.members]))
    loss_rows += [dict(repeat=rep, fold=fold, model=f'adapter_member{i}', epoch=e + 1, loss=v)
                  for i, h in enumerate(dh) for e, v in enumerate(h)]
    del dbank

    # (d) microstate dynamics stream (templates from TRAIN windows only)
    msm = MicrostateModel().fit(Wraw[tr], gtr, seed)
    ms_tr, ms_te = msm.features(Wraw[tr]), msm.features(Wraw[te])
    gr['ms_train_gev'] = float(msm.train_gev)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log(f'{tag}: HC gap d (should be ~0): norm in-sample {gr["d_norm_full_model"]:+.2f} -> cross-fit {gr["d_norm_used"]:+.2f} | '
        f'adapter {gr["d_delta_hc_gap"]:+.2f} (kappa {gr["adapter_kappa"]:.1f}) | MS peak-GEV {msm.train_gev:.2f}')

    F_tr, F_te = dict(Z=Ztr, dev=dev_tr, dlt=dlt_tr, ms=ms_tr), dict(Z=Zte, dev=dev_te, dlt=dlt_te, ms=ms_te)
    tr_labels = subject_labels(ytr, gtr)

    # (e) inner OOF subject probabilities of every (stream, classifier) candidate expert
    inner = {e: [] for e in BASE_EXPERTS}
    cv_in = StratifiedGroupKFold(n_splits=CFG['cv_inner_splits'], shuffle=True, random_state=seed)
    for ii, (itr, iva) in enumerate(cv_in.split(Ztr, ytr, gtr)):
        S_fit, S_va = make_streams(take(F_tr, itr), [take(F_tr, iva)])
        for s in STREAMS:
            for c in BASE_CLFS:
                subj, p = score_subjects(c, S_fit[s], ytr[itr], gtr[itr], S_va[s], gtr[iva], seed + ii)
                inner[f'{s}|{c}'].append(pd.Series(p, index=subj))
    P_in = add_ensembles(pd.DataFrame({e: pd.concat(v) for e, v in inner.items()}))
    assert P_in.index.is_unique and set(P_in.index) == set(tr_labels.index.astype(str)), 'inner OOF accounting failed'
    y_in = tr_labels.copy(); y_in.index = y_in.index.astype(str); y_in = y_in.loc[P_in.index].values.astype(int)

    # (f) outer-test subject probabilities
    S_tr, S_te = make_streams(F_tr, [F_te])
    te_cols = {}
    for s in STREAMS:
        for c in BASE_CLFS:
            subj, p = score_subjects(c, S_tr[s], ytr, gtr, S_te[s], gte, seed)
            te_cols[f'{s}|{c}'] = pd.Series(p, index=subj)
    P_te = add_ensembles(pd.DataFrame(te_cols))
    sl_te = subject_labels(yte, gte); sl_te.index = sl_te.index.astype(str)
    y_te_s = sl_te.loc[P_te.index].values.astype(int)

    # (g) operating points + selection + calibrated PoE fusion
    fold_rows, preds, w_rows = [], [], []
    def record(name, p_in, p_te, thr):
        pred = (p_te >= thr).astype(int)
        m = classification_metrics(y_te_s, p_te, pred)
        ia = roc_auc_score(y_in, p_in) if len(np.unique(y_in)) > 1 else np.nan
        fold_rows.append(dict(repeat=rep, fold=fold, method=name, n_test_subjects=len(y_te_s), threshold=thr, inner_auc=ia, **m))
        preds.append(pd.DataFrame(dict(repeat=rep, fold=fold, method=name, subject=P_te.index.values, y=y_te_s, p=p_te, thr=thr, pred=pred)))
    for e in ALL_EXPERTS:
        record(e, P_in[e].values, P_te[e].values, youden_threshold(y_in, P_in[e].values))
    L_in, L_te = to_logit(P_in[BASE_EXPERTS].values), to_logit(P_te[BASE_EXPERTS].values)
    fused, wr = fuse_all(L_in, L_te, y_in, BASE_EXPERTS, seed)
    assert set(fused) == set(FUSION_NAMES), set(fused) ^ set(FUSION_NAMES)
    for name in FUSION_NAMES:
        p_in, p_te = fused[name]
        thr = 0.5 if CFG['fusion_thr'] == 'half' else youden_threshold(y_in, p_in)
        record(name, p_in, p_te, thr)
    w_rows += [dict(repeat=rep, fold=fold, **r) for r in wr]
    fr = pd.DataFrame(fold_rows).set_index('method')
    best_single = fr.loc[ALL_EXPERTS, 'auc'].idxmax()
    chosen = [r['expert'] for r in wr if r['fusion'] == 'selection' and r['selected']]
    log(f'{tag}: selected {len(chosen)} experts: {", ".join(chosen)}')
    log(f'{tag}: PoE_sel_hier_eq AUC={fr.loc["PoE_sel_hier_eq", "auc"]:.3f} acc={fr.loc["PoE_sel_hier_eq", "accuracy"]:.3f} | '
        f'noMS AUC={fr.loc["PoE_sel_hier_eq_noMS", "auc"]:.3f} | PoE_all_hier_eq AUC={fr.loc["PoE_all_hier_eq", "auc"]:.3f} | '
        f'best single (post-hoc) {best_single} {fr.loc[best_single, "auc"]:.3f} ({(time.time() - t0) / 60:.1f} min)')
    return dict(fold_rows=fold_rows, preds=pd.concat(preds, ignore_index=True), w_rows=w_rows, gap=gr, loss_rows=loss_rows)

# ================================================================
# 9. MAIN
# ================================================================
if __name__ == '__main__':
    print('\nFrozen multi-layer CBraMod + cross-fitted diffusion (normative + consistency adapter) + Klein + microstates'
          ' + stable-expert calibrated hierarchical PoE | MODMA REST')
    if SYNTH:
        X, y, g, Wraw = synthetic_latent(); EMB_KEY = 'synthetic'
        log('SYNTHETIC smoke-test data (not MODMA)')
    else:
        manifest = build_manifest(DATA_DIR)
        sl = manifest.groupby('subject')['label'].first()
        log(f'manifest: {len(manifest)} files, {len(sl)} subjects (MDD={int(sl.sum())}, HC={int((1 - sl).sum())}; expected 24/29)')
        backbone = load_cbramod()
        with torch.no_grad():
            o = backbone(torch.zeros(2, 18, int(PRE['window_sec']), 200, device=DEVICE))
            assert tuple(o.shape) == (2, 18, int(PRE['window_sec']), 200), o.shape
            hs = backbone.forward_layers(torch.zeros(2, 18, int(PRE['window_sec']), 200, device=DEVICE), LAYERS)
            assert len(hs) == len(LAYERS) and torch.allclose(hs[-1], o) if LAYERS[-1] == N_CBR_LAYERS else True
        X, y, g, Wraw, EMB_KEY = build_latent_dataset(backbone, manifest)
        del backbone
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    assert X.ndim == 3 and X.shape[1] == len(LAYERS) and len(Wraw) == len(X)
    ysub_all = subject_labels(y, g)
    subj_arr = np.array(sorted(ysub_all.index)); ysub = ysub_all.loc[subj_arr]
    N_SUBJ, N_MDD = len(subj_arr), int(ysub.sum())
    log(f'dataset: {len(X)} windows | {N_SUBJ} subjects (MDD={N_MDD}, HC={N_SUBJ - N_MDD}) | '
        f'latent {len(LAYERS)} layers {LAYERS} x {X.shape[2]} -> {len(LAYERS) * CFG["layer_pca_dim"]} PCs | {len(BASE_EXPERTS)} candidate experts')
    R, K = CFG['cv_repeats'], CFG['cv_n_splits']
    assert min(N_MDD, N_SUBJ - N_MDD) >= K

    run_hash = hashlib.sha1((MODEL_VERSION + EMB_KEY + json.dumps(CFG, sort_keys=True, default=str)).encode()).hexdigest()[:10]
    log(f'run hash {run_hash} (resume={CFG["resume"]})')
    D = dict(X=X, y=y, g=g, W=Wraw)
    ST = dict(fold=[], preds=[], w=[], gap=[], loss=[], meta=[])
    for rep in range(R):
        outer = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=SEED + rep)
        seen = []
        for fold, (a_, b_) in enumerate(outer.split(np.zeros(N_SUBJ), ysub.values, groups=subj_arr)):
            tr_subj, te_subj = subj_arr[a_], subj_arr[b_]
            assert not (set(tr_subj) & set(te_subj)) and ysub.loc[te_subj].nunique() == 2
            seen += list(te_subj)
            ST['meta'].append(dict(repeat=rep, fold=fold, train=len(tr_subj), test=len(te_subj), test_mdd=int(ysub.loc[te_subj].sum())))
            log(f'REPEAT {rep + 1}/{R} FOLD {fold + 1}/{K}: train={len(tr_subj)} test={len(te_subj)}')
            cp = os.path.join(CKPT_DIR, f'fold_r{rep}_f{fold}_{run_hash}.joblib')
            if CFG['resume'] and os.path.exists(cp):
                r = joblib.load(cp); log('  loaded from cache')
            else:
                r = fold_stage(rep, fold, tr_subj, te_subj, D); joblib.dump(r, cp)
            ST['fold'] += r['fold_rows']; ST['preds'].append(r['preds']); ST['w'] += r['w_rows']
            ST['gap'].append(r['gap']); ST['loss'] += r['loss_rows']
        assert sorted(seen) == sorted(subj_arr.tolist()), 'each subject must be tested exactly once per repeat'

    fold_df, subj_df = pd.DataFrame(ST['fold']), pd.concat(ST['preds'], ignore_index=True)
    fold_df.to_csv(os.path.join(OUT_DIR, 'per_fold_metrics.csv'), index=False)
    subj_df.to_csv(os.path.join(OUT_DIR, 'oof_subject_predictions.csv'), index=False)
    wdf = pd.DataFrame(ST['w'])
    wdf.to_csv(os.path.join(OUT_DIR, 'poe_weights_calibration_selection.csv'), index=False)
    pd.DataFrame(ST['gap']).to_csv(os.path.join(OUT_DIR, 'crossfit_gap_diagnostics.csv'), index=False)
    pd.DataFrame(ST['loss']).to_csv(os.path.join(OUT_DIR, 'diffusion_losses.csv'), index=False)
    pd.DataFrame(ST['meta']).to_csv(os.path.join(OUT_DIR, 'fold_composition.csv'), index=False)

    METRICS = ['auc', 'accuracy', 'precision', 'recall', 'specificity', 'f1', 'mcc', 'balanced_accuracy']
    subs_all = sorted(subj_df.subject.unique()); assert len(subs_all) == N_SUBJ

    def oof_by_repeat(method):
        out = []
        for _, d in subj_df[subj_df.method == method].groupby('repeat'):
            assert d['subject'].is_unique and len(d) == N_SUBJ
            d = d.set_index('subject').loc[subs_all]
            out.append((d['y'].values.astype(int), d['p'].values.astype(float), d['pred'].values.astype(int)))
        assert len(out) == R
        return out

    BOOT = np.random.default_rng(SEED + 5).integers(0, N_SUBJ, size=(CFG['n_boot'], N_SUBJ))
    def boot_ci(reps):
        rows = []
        for i in BOOT:
            if len(np.unique(reps[0][0][i])) < 2:
                continue
            ms = [classification_metrics(a[i], b[i], c[i]) for a, b, c in reps]
            rows.append({k: float(np.mean([m[k] for m in ms])) for k in METRICS})
        b = pd.DataFrame(rows)
        return {k: (b[k].quantile(.025), b[k].quantile(.975)) for k in METRICS}

    pooled, table = [], []
    for m in METHODS:
        reps = oof_by_repeat(m)
        per = pd.DataFrame([classification_metrics(*r_) for r_ in reps])
        ci = boot_ci(reps)
        row = dict(method=m, n_subjects=N_SUBJ, n_repeats=R, **per[METRICS].mean().to_dict())
        for k in METRICS:
            row[f'{k}_sd'] = float(per[k].std(ddof=1)) if R > 1 else np.nan
            row[f'{k}_ci_lo'], row[f'{k}_ci_hi'] = ci[k]
        pooled.append(row)
        table.append({'Method': m, 'AUC [95% CI]': f'{row["auc"]:.3f} [{ci["auc"][0]:.3f}, {ci["auc"][1]:.3f}]',
                      'AUC SD': f'{row["auc_sd"]:.3f}', 'Acc': f'{row["accuracy"]:.3f}', 'Bal.Acc': f'{row["balanced_accuracy"]:.3f}',
                      'Sens': f'{row["recall"]:.3f}', 'Spec': f'{row["specificity"]:.3f}', 'F1': f'{row["f1"]:.3f}', 'MCC': f'{row["mcc"]:.3f}'})
    pooled_df = pd.DataFrame(pooled)
    pooled_df.to_csv(os.path.join(OUT_DIR, 'pooled_subject_level_metrics.csv'), index=False)
    final = pd.DataFrame(table)
    final.to_csv(os.path.join(OUT_DIR, 'final_classification_table.csv'), index=False)

    # paired delta-AUC / delta-accuracy (same subjects, same bootstrap draws)
    best_expert = pooled_df[pooled_df.method.isin(ALL_EXPERTS)].sort_values('auc', ascending=False).method.iloc[0]
    pairs = [('PoE_sel_hier_eq', 'PoE_all_hier_eq'), ('PoE_sel_hier_eq', 'PoE_all_flat_eq'), ('PoE_sel_hier_eq', best_expert),
             ('PoE_sel_hier_eq', 'PoE_sel_hier_eq_noMS'), ('PoE_sel_hier', 'PoE_sel_hier_eq'),
             ('PoE_sel_flat_eq', 'PoE_sel_hier_eq'), ('PoE_sel_hier_eq', 'SelectBest_inner')]
    delta_rows = []
    for a_name, b_name in pairs:
        ra, rb = oof_by_repeat(a_name), oof_by_repeat(b_name)
        for metric in ('auc', 'accuracy'):
            def mv(r_): return roc_auc_score(r_[0], r_[1]) if metric == 'auc' else float((r_[2] == r_[0]).mean())
            pt = float(np.mean([mv(x) - mv(z) for x, z in zip(ra, rb)]))
            bd = []
            for i in BOOT:
                if len(np.unique(ra[0][0][i])) < 2:
                    continue
                bd.append(float(np.mean([mv((x[0][i], x[1][i], x[2][i])) - mv((z[0][i], z[1][i], z[2][i])) for x, z in zip(ra, rb)])))
            bd = np.asarray(bd)
            delta_rows.append(dict(a=a_name, b=b_name, metric=metric, delta=pt, ci_lo=float(np.quantile(bd, .025)), ci_hi=float(np.quantile(bd, .975)),
                                   p_boot=min(1.0, 2 * min(float((bd <= 0).mean()), float((bd >= 0).mean())))))
    delta_df = pd.DataFrame(delta_rows); delta_df.to_csv(os.path.join(OUT_DIR, 'paired_deltas.csv'), index=False)

    print('\n' + '=' * 120)
    print(f'FINAL | MODMA REST | {N_SUBJ} subjects (MDD={N_MDD}, HC={N_SUBJ - N_MDD}) | {R} x {K}-fold StratifiedGroupKFold (groups=subject)')
    print('Subject-level OOF metrics, mean over repeats; 95% subject-cluster bootstrap CI. Singles: inner-OOF Youden threshold;'
          f' fusions: threshold mode = {CFG["fusion_thr"]}')
    print('=' * 120)
    print('\n--- FUSION methods ---'); print(final[final.Method.isin(FUSION_NAMES)].to_string(index=False))
    print(f'\n--- all {len(BASE_EXPERTS)} candidate experts + {len(ENS_EXPERTS)} ensembles, sorted by AUC ---')
    order = pooled_df[pooled_df.method.isin(ALL_EXPERTS)].sort_values('auc', ascending=False).method.tolist()
    print(final.set_index('Method').loc[order].reset_index().to_string(index=False))
    print('\nPaired deltas:'); print(delta_df.round(4).to_string(index=False))

    fd = fold_df.assign(gen_gap=lambda d: d.inner_auc - d.auc, kind=lambda d: np.where(d.method.isin(FUSION_NAMES), 'fusion', 'expert'))
    print('\nInner-OOF AUC minus outer-test AUC (mean over folds; large positive = optimistic / unstable fitting):')
    print(fd.groupby('kind').gen_gap.mean().round(3).to_string())
    gd = pd.DataFrame(ST['gap'])
    print('\nCross-fit HC gap (Cohen d, held-out HC vs train HC; target ~0):')
    print(gd[['d_norm_full_model', 'd_norm_used', 'd_delta_hc_gap']].mean().round(3).to_string())
    print(f'Mean adapter kappa {gd.adapter_kappa.mean():.1f} | mean microstate peak-GEV {gd.ms_train_gev.mean():.3f}')
    if len(wdf):
        h = wdf[wdf.fusion == 'PoE_sel_hier']
        if len(h):
            print('\nMean PoE_sel_hier stream weights:'); print(h.groupby('expert').weight.mean().round(3).to_string())
        sel = wdf[wdf.fusion == 'selection']
        if len(sel):
            per_fold = sel.groupby(['repeat', 'fold']).selected.sum()
            print(f'\nSelected experts per fold: mean {per_fold.mean():.1f} (min {int(per_fold.min())}, max {int(per_fold.max())})')
            freq = sel.groupby('expert').selected.mean().sort_values(ascending=False)
            print('Selection frequency across folds (consensus set = >= 0.5):'); print(freq.head(12).round(2).to_string())
            sel = sel.assign(stream=sel.expert.str.split('|').str[0])
            print('Mean experts selected per stream per fold:')
            print(sel.groupby(['repeat', 'fold', 'stream']).selected.sum().groupby('stream').mean().round(2).to_string())
            freq.to_csv(os.path.join(OUT_DIR, 'expert_selection_frequency.csv'))
        cal = wdf[wdf.fusion == 'calibration']
        print('\nFraction of folds in which each expert was ACTIVE (calibration slope > 0), top/bottom:')
        act = cal.groupby('expert').active.mean().sort_values(ascending=False)
        print(pd.concat([act.head(8), act.tail(4)]).round(2).to_string())
    with open(os.path.join(OUT_DIR, 'final_classification_table.md'), 'w') as f:
        cols = list(final.columns)
        f.write('| ' + ' | '.join(cols) + ' |\n|' + '|'.join(['---'] * len(cols)) + '|\n')
        for _, r_ in final.iterrows():
            f.write('| ' + ' | '.join(str(v) for v in r_.values) + ' |\n')

    fig, ax = plt.subplots(figsize=(7, 5.8)); grid = np.linspace(0, 1, 201)
    for m in ['cbramod|logreg', 'cbr_delta|logreg', 'microstate|logreg', best_expert, 'PoE_all_hier_eq', 'PoE_sel_flat_eq', 'PoE_sel_hier', 'PoE_sel_hier_eq']:
        tprs, aucs = [], []
        for y_, p, _ in oof_by_repeat(m):
            fp_, tp_, _t = roc_curve(y_, p); tprs.append(np.interp(grid, fp_, tp_)); aucs.append(roc_auc_score(y_, p))
        t = np.mean(tprs, axis=0); t[0] = 0.0
        ax.plot(grid, t, lw=1.9 if m == 'PoE_sel_hier_eq' else 1.1, label=f'{m} ({np.mean(aucs):.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax.set(xlabel='1 - specificity', ylabel='sensitivity', title='MODMA REST: mean OOF subject-level ROC')
    ax.legend(loc='lower right', fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'roc.png'), dpi=250); plt.close(fig)

    with open(os.path.join(OUT_DIR, 'run_manifest.json'), 'w') as f:
        json.dump(dict(model_version=MODEL_VERSION, run_hash=run_hash, emb_key=EMB_KEY, seed=SEED, cfg=CFG, pre=PRE,
                       n_subjects=N_SUBJ, n_windows=int(len(X)), subjects=subj_arr.tolist(),
                       base_experts=BASE_EXPERTS,
                       versions=dict(torch=torch.__version__, mne=mne.__version__, numpy=np.__version__, pandas=pd.__version__,
                                     sklearn=sklearn.__version__, scipy=scipy.__version__, xgboost=xgboost.__version__,
                                     python=sys.version.split()[0])), f, indent=2, default=str)
    log('DONE. Results in ' + OUT_DIR)
