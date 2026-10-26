#!/usr/bin/env python
"""
RubricMarkov-MRI (OASIS-2) -- ONE FILE: raw-MRI atrophy emissions (GPU) + continuous-time (semi-)Markov staging,
forecasting and network-coupled spread (autograd/GPU likelihood).  Works in Kaggle / Colab / Jupyter / `python file.py`.
NO argparse: edit the CONFIG block and run.

IMAGING PIPELINE v4 (what changed vs. v3; the Markov / staging / spread code is UNCHANGED)
  * REAL SKULL-STRIP MASK: SynthStrip (primary) -> HD-BET (backup) are auto-installed if missing (AUTO_INSTALL=True, needs Kaggle
    Internet) and REQUIRE_MASK=True by default, so the heuristic mask is NEVER used.  If the stripper dies mid-run the run stops
    (finished scans are cached) instead of silently skipping every scan.  Bad masks are never cached; masks < 10k voxels are rejected.
    A `mask_suspect` flag in mri_qc.csv marks masks that disagree strongly with the heuristic one (audit them in qc_png).
  * (v3) repeat alignment sign fix + self-test; RAS orientation; MRF tissue segmentation; deep-central-component ventricle proxy;
    CNR QC term; atlas lateral-ventricle sanity check; anatomical montages.
  If you change anything in the imaging block, DELETE OUT/mri (cached features / slices / masks / qwen_raw.csv) and rerun.
  !! If you previously ran with heuristic masks:  !rm -rf rm_out/mri   before the first run of this version.

RUN modes
  "all"      = "mri" -> "validate" -> "qwen" -> "stage".  The one-click paper pipeline (qwen / validate are skipped with a
               warning if unavailable).
  "export"   write the aligned repeat-average of every session as NIfTI (OUT/mri/mean/<MRI ID>.nii.gz) so a VALIDATED
             skull-stripper (SynthStrip / HD-BET) can be run outside this script.  Then set MASK_DIR (+ REQUIRE_MASK=True).
             (Not needed when SynthStrip / HD-BET are installable: SKULLSTRIP="auto" calls them from inside RUN="mri".)
  "mri"      raw OASIS-2 MPRAGE repeats -> RAS reorientation -> aligned average -> brain mask (external MASK_DIR mask, else
             automatic SynthStrip (HD-BET only if SynthStrip weights are unavailable); heuristic only if REQUIRE_MASK=False) -> bias-field correction (N4 if SimpleITK, else
             WM-anchored GPU estimate) -> MRF-regularised CSF/GM/WM -> emissions (higher = more atrophy) + repeat-consistency QC
             weight `qc_conf` (Dice of two half-average segmentations, NCC, volume plausibility, tissue contrast).
  "validate" agreement of the intensity-based tissue measures with OASIS nWBV/eTIV (rank) and, if REF_CSV is given,
             with FreeSurfer / FSL references (Spearman + cluster-bootstrap CI, Lin's CCC, Bland-Altman).
  "qwen"     Qwen2.5-VL reads the skull-stripped slice montages and scores 3 visual rubrics (ventricles, global cortical
             atrophy, medial-temporal atrophy) in a CONSTRAINED answer format ('Score: <digit>', assistant prefill).  Scores =
             expected value over the renormalised digit distribution; the FULL-vocabulary probability mass on the digits is
             logged and used as a validity diagnostic / mask.  Per-rubric confidence = agreement of two independent slice
             views x (1 - normalised entropy) x digit mass.  Rubrics are validated against INDEPENDENT anatomical measures
             and are only used as emissions if they PASS.  The prompt contains NO demographics, cognition or diagnosis.
  "stage"    3-class subject outcome (Nondemented / Converted / Demented), visit-level CDR staging, forecasts,
             baselines, bootstrap CIs, progression curves, per-subject forecast table.
             PROSPECTIVE tasks (cutoff != None): the Markov model, all preprocessing statistics and all baselines see ONLY
             visits <= cutoff in train AND test folds; future visits are used only to build outcome labels.
             RETROSPECTIVE task: Mode C (CDR input) is an ORACLE (Group is defined from CDR) and is skipped unless
             RETRO_ORACLE=True (then labelled "C-ORACLE").  Headline = Modes A and B.
  "ablate"   K, semi-Markov, discrete-time, supervision, image-emission and QC-weight ablations.
  "spread"   regional network-coupled model (needs REGIONAL_CSV; mode "mri" with REGIONS=True writes one: all 112
             Harvard-Oxford ROIs (96 cortical L/R + 16 subcortical, SyN-warped, no fallback) + lobar aggregates; needs torch + nilearn + antspyx).
  "synth"    write synthetic data (incl. regional files) for smoke-testing everything.

Compute: the Markov-model NLL is implemented in PyTorch (float64, GPU if present) with EXACT autograd gradients inside
scipy L-BFGS-B (the numpy/finite-difference path is kept as a fallback and as a self-test reference).
Restarts: CV folds use ONE noise-free start; anything above MAX_RESTARTS is capped; bootstrap replicates are warm-started.

Kaggle/Colab:  %%writefile rubricmarkov_mri.py   then in a NEW cell   %run rubricmarkov_mri.py
(enable Settings -> Accelerator -> GPU AND Settings -> Internet for the "mri" step: SynthStrip/HD-BET are pip-installed and
their weights downloaded on first use.  Offline: attach synthstrip.1.pt as a dataset and set SYNTHSTRIP_MODEL, pre-install the
packages, or use RUN="export" + MASK_DIR.)
If you change MASK_DIR / BIAS / the pipeline, delete OUT/mri (cached features / slices / qwen_raw.csv) and rerun.
"""
from __future__ import annotations
import functools, glob, itertools, json, math, os, re, sys, time, warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from types import SimpleNamespace
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import torch
    _HAS_TORCH = True
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:                                   # missing / broken torch -> numpy fallback for the Markov fits
    torch, _HAS_TORCH, DEV = None, False, None

warnings.filterwarnings("ignore")
print = functools.partial(print, flush=True)

# =============================================================================== CONFIG (edit here)
CONFIG = dict(
    RUN="all",                   # "all" | "export" | "mri" | "validate" | "qwen" | "stage" | "ablate" | "spread" | "synth"
    XLSX=None,                   # path to oasis_longitudinal_demographics.xlsx (None = auto-search)
    OUT="rm_out",
    QUICK=False,                 # True = smoke run (3 folds x 1 repeat, 1 restart, 60 iters, 5 bootstraps)
    CUTOFFS=[None, 1.0],         # None = retrospective staging (all visits); 1.0 = PROSPECTIVE (see first 1.0 y, predict later)
    K=3, PHASES=1,               # stages; Coxian sub-phases per stage (>1 = semi-Markov)
    SUP="full",                  # "full": train with MMSE+CDR as weak labels (train folds only), hide at test.  "matched": no weak labels
    DISCRETE=False,
    SPLITS=5, REPEATS=3, SEED=0, BOOT=40, GRU=False,
    RESTARTS=1,                  # CV starts per fold (r=0 is noise-free; exact gradients make 1 reliable)
    MAX_RESTARTS=2,              # hard cap for ANY fit (final descriptive fit uses 2 and prints the start NLLs)
    MAXITER=150, ALPHA=1.0,
    HORIZONS=[1, 2, 3, 5],       # years, for the per-subject forecast table
    VLM_CSV=None, EXTRA_COLS=None,   # per-scan CSV keyed by 'MRI ID'. None => auto-use OUT/mri/mri_features.csv if USE_MRI
    USE_MRI=True,                # auto-attach image emissions from OUT/mri/mri_features.csv (+ VALIDATED Qwen rubrics if present)
    USE_QWEN=True,               # auto-attach OUT/mri/features_all.csv (MRI geometry + Qwen-VL rubric scores) when it exists
    CONF_ON="image",             # "image": qc/VLM confidence tempers ONLY image-derived channels (nWBV, MRI, Qwen); "all": also MMSE/CDR (old behaviour)
    RETRO_ORACLE=False,          # True = also run retrospective Mode C, always labelled "C-ORACLE" (never a headline result)
    # ---- mri step
    RAW_ROOTS=["/kaggle/input/datasets/kinnarhalder/oasis-2-1/OAS2_RAW_PART1",
               "/kaggle/input/datasets/kinnarhalder/oasis-2-2/OAS2_RAW_PART2"],
    CANONICAL=True,              # reorient every repeat (and external mask) to RAS before anything else (anatomical montages, atlas frame)
    ALIGN="phase",               # repeat alignment: "phase" (GPU integer phase-correlation, sign-verified) | "rigid" (+ ANTs rigid refinement, slower)
    MASK_DIR=None,               # <MRI ID>.nii.gz (or _mask / _bet_mask / _brain_mask) brain masks from SynthStrip / HD-BET -> paper-grade
    SKULLSTRIP="auto",           # when no MASK_DIR mask exists: "auto" (SynthStrip, then HD-BET as backup) | "synthstrip" | "hdbet" | "heuristic"
    AUTO_INSTALL=True,           # pip-install nipreps-synthstrip / hd-bet when none is found (needs Kaggle Internet ON)
    SYNTHSTRIP_MODEL=None,       # offline: path to synthstrip.1.pt (attach it as a Kaggle dataset); None = download once to OUT/mri
    STRIP_TIMEOUT=900,           # seconds per scan for the external skull-stripper
    REQUIRE_MASK=True,           # paper mode: NEVER fall back to the heuristic mask (scans without a real mask are not processed)
    BIAS="auto",                 # "auto" (N4 if SimpleITK present, else WM-anchored GPU estimate) | "n4" | "wm" | "none"
    SEG_MRF=1.0,                 # strength of the Potts/MRF spatial prior in the tissue segmentation (0 = plain GMM, as before)
    SEG_EM=3,                    # EM rounds of the (MRF-regularised) 3-class GMM
    VENT_MIN_DEPTH_MM=8.0,       # ventricle proxy: CSF must lie >= this depth (mm) inside the brain mask ...
    VENT_MIN_ML=0.5,             # ... and belong to a connected component of at least this volume (mL), inside the central box
    REF_CSV=None,                # optional reference table: 'MRI ID', ref_<feature> (ref_vent_icv, ref_csf_icv, ref_gm_deficit) from FreeSurfer/FSL
    ERODE=6, MAX_SHIFT=12, MRI_LIMIT=None, QC_PNG=40,
    REGIONS=False,               # also write regional.csv: the 112-ROI Harvard-Oxford atlas (96 cortical L/R + 16 subcortical) SyN-warped into each scan, no fallback
    ICV_ML=(1000.0, 2100.0),
    ATLAS_DIR=None,              # offline: dir with HarvardOxford-cortl-maxprob-thr25-1mm.nii.gz (LATERALISED) + HarvardOxford-sub-maxprob-thr25-1mm.nii.gz (None = nilearn download)
    MNI_TEMPLATE=None,           # offline/exact: path to a skull-stripped MNI152 T1 1mm brain (FSL: MNI152_T1_1mm_brain.nii.gz); None = nilearn template
    ATLAS_TRANSFORM="SyN",       # ANTs transform type for atlas registration ("SyN" | "SyNCC" | "SyNRA" ...)
    ATLAS_MIN_DICE=0.80,         # Dice(SyN-warped MNI brain mask, subject brain mask) needed to use the atlas for that scan, else its regional values are missing
    ATLAS_MIN_VENT_CSF=0.35,     # mean CSF posterior inside the warped atlas lateral ventricles needed to accept the registration (ROI-level sanity check)
    VENT_BAND_MM=8.0,            # periventricular band (mm) around the atlas lateral ventricles used for vol_vent
    # ---- qwen step (vision-language rubric scoring of the skull-stripped MRI; no demographics/diagnosis in the prompt)
    QWEN_MODEL="Qwen/Qwen2.5-VL-3B-Instruct",   # HF id OR a local dir (Kaggle: attach a Qwen2.5-VL model, internet may be off)
    QWEN_QUANT="fp16",           # "fp16" (3B fits a T4) | "4bit" (7B on a T4; needs bitsandbytes)
    QWEN_MAX_PIXELS=640*28*28, QWEN_LIMIT=None,
    QWEN_PREFILL="Score: ",      # forced answer format (assistant prefill) so the next token is the digit
    QWEN_MIN_MASS=0.5,           # min FULL-vocab probability mass on digits 0-3, else the rubric is masked for that scan
    QWEN_MIN_RHO=0.20,           # min |Spearman| vs independent anatomical measures for a rubric to be used as an emission
    # ---- spread only
    REGIONAL_CSV=None, REGIONS_LIST="mtl,frontal,parietal,vent", W_CSV=None, CENTROIDS_CSV=None,
    USE_RUBRICS=False, SPREAD_STEPS=150, N_SHUFFLE=3, BETA_BOOT=0,
    # ---- synth only
    SYNTH_N=90,
)
MAX_RESTARTS = CONFIG["MAX_RESTARTS"]
CONF_ON = CONFIG["CONF_ON"]

CLASSES = ["Nondemented", "Converted", "Demented"]
COV = ["Age", "edu", "SES", "eTIV"]                # static covariates imputed with TRAIN-fold medians
# NOTE mode C feeds CDR to the model, and Group is DEFINED from CDR -> in the RETROSPECTIVE task mode C is an ORACLE
# (upper bound), not a result.  In the PROSPECTIVE task CDR at observed visits is legitimately known.
MODES = {"A": {"nwbv"}, "B": {"nwbv", "mmse"}, "C": {"nwbv", "mmse", "cdr"}}


# =============================================================================== data
def cdr_cat(c):
    c = np.asarray(c, float)
    return np.where(np.isnan(c), 0, np.where(c >= 1, 2, np.where(c >= 0.5, 1, 0))).astype(int)


def find_table(path):
    if path: return path
    pats = ["/kaggle/input/**/*.xlsx", "/content/**/*.xlsx", "./**/*.xlsx", "/mnt/user-data/uploads/*.xlsx"]
    c = [x for p in pats for x in glob.glob(p, recursive=True)]
    c = [x for x in c if "longitudinal" in x.lower() or "oasis" in x.lower()] or c
    assert c, "no .xlsx found: set CONFIG['XLSX'] to the oasis_longitudinal_demographics.xlsx path"
    return c[0]


def load_table(path):
    d = pd.read_excel(path).rename(columns={"Subject ID": "sid", "MRI ID": "mri", "MR Delay": "delay",
                                            "M/F": "sex", "EDUC": "edu"})
    need = {"sid", "mri", "Group", "Visit", "delay", "sex", "Age", "edu", "SES", "MMSE", "CDR", "eTIV", "nWBV"}
    assert need <= set(d.columns), f"missing columns: {need - set(d.columns)}"
    assert d["delay"].notna().all(), "MR Delay must be present for every visit"
    d["male"] = (d["sex"] == "M").astype(float)
    d["t"] = d["delay"] / 365.25
    d = d.sort_values(["sid", "Visit"]).reset_index(drop=True)
    assert d.groupby("sid")["Group"].nunique().max() == 1, "Group must be constant within a subject"
    assert set(d["Group"].unique()) <= set(CLASSES), f"unexpected Group values {d['Group'].unique()}"
    assert (d.groupby("sid")["t"].diff().dropna() > 0).all(), "visit times must strictly increase"
    d["Age_raw"] = d["Age"]
    d["Age"] = d.groupby("sid")["Age"].transform("first") + d["t"]     # continuous age (raw Age is integer-rounded)
    return d


def load_vlm(d, csv, cols=None):
    """Optional per-scan CSV keyed by 'MRI ID'. Numeric columns become extra emissions (higher = more atrophy);
    'qc_conf' in (0,1] tempers that visit's emission likelihood."""
    d = d.copy(); use = []
    if csv:
        v = pd.read_csv(csv).rename(columns={"MRI ID": "mri"})
        d = d.merge(v, on="mri", how="left")
        allc = [c for c in v.columns if c not in ("mri", "qc_conf") and not c.endswith(("_conf", "_mass"))
                and pd.api.types.is_numeric_dtype(v[c])]
        use = allc if cols is None else [c for c in cols if c in allc]
        miss = d[use[0]].isna().mean() if use else 0.0
        if use: print(f"image emissions {use}: {100*(1-miss):.0f}% of visits covered (missing visits are masked, not imputed)")
    if "qc_conf" not in d: d["qc_conf"] = 1.0
    d["qc_conf"] = d["qc_conf"].fillna(1.0).clip(0.05, 1.0)
    return d, use


def observed_window(d, cutoff):
    """Visits available at prediction time (each subject's first visit is always kept)."""
    if cutoff is None: return d
    first = d["Visit"] == d.groupby("sid")["Visit"].transform("min")
    return d[(d["t"] <= cutoff + 1e-9) | first]


def assert_no_future(d, cutoff, what=""):
    """Hard guard: after windowing, no visit later than `cutoff` (other than a subject's first visit) may remain."""
    if cutoff is None: return
    first = d["Visit"] == d.groupby("sid")["Visit"].transform("min")
    bad = d[(d["t"] > cutoff + 1e-9) & ~first]
    assert bad.empty, f"temporal leakage ({what}): {len(bad)} visits beyond cutoff={cutoff}"


def build_task(d, cutoff):
    """(data restricted to usable subjects, outcome frame indexed by sid: y, horizon, nsteps, t_conv, cdr_final).
    NOTE: the returned data still contains the future visits; they must only be used to derive labels (the outcome
    frame). run_cv / cmd_stage window the data with observed_window() before any fitting."""
    if cutoff is None:
        y = d.groupby("sid", sort=False)["Group"].first().map(CLASSES.index)
        return d, pd.DataFrame({"y": y.values, "horizon": 0.0, "nsteps": 0, "t_conv": np.nan, "cdr_final": np.nan},
                               index=y.index)
    assert cutoff >= 0
    rows = []
    for sid, g in d.groupby("sid", sort=False):
        obs = observed_window(g, cutoff); fut = g.drop(obs.index)
        last = obs["CDR"].dropna()
        if fut.empty or last.empty: continue
        pos = fut[fut["CDR"].fillna(0) > 0]
        if last.iloc[-1] > 0: y = 2                     # already impaired at the last observed visit
        elif len(pos): y = 1                            # unimpaired now, impaired later
        else: y = 0
        cf = fut["CDR"].dropna()
        rows.append(dict(sid=sid, y=y, horizon=max(fut["t"].iloc[-1] - obs["t"].iloc[-1], 1e-3), nsteps=len(fut),
                         t_conv=(pos["t"].iloc[0] - obs["t"].iloc[-1]) if (y == 1) else np.nan,
                         cdr_final=cdr_cat([cf.iloc[-1]])[0] if len(cf) else np.nan))
    out = pd.DataFrame(rows).set_index("sid")
    return d[d.sid.isin(out.index)].reset_index(drop=True), out


class Prep:
    """All statistics (mean/sd/median) are fitted on the training fold only (and, prospectively, on the windowed train data)."""
    def fit(self, d, extra=()):
        self.extra = list(extra)
        cols = COV + ["nWBV", "MMSE"] + self.extra
        self.mu = {c: d[c].mean() for c in cols}
        self.sd = {c: d[c].std() + 1e-9 for c in cols}
        self.med = {c: d[c].median() for c in COV}
        return self

    def z(self, d, c):
        x = d[c].astype(float)
        if c in COV: x = x.fillna(self.med[c])
        return (x - self.mu[c]) / self.sd[c]            # NaN stays NaN for observed variables -> masked

    def tensors(self, d):
        d = d.reset_index(drop=True)
        sids = d["sid"].unique(); B, T = len(sids), int(d.groupby("sid").size().max())
        D = {"sids": sids, "B": B, "T": T}
        for k in ("age", "nwbv", "mmse", "male", "etiv", "ses", "edu", "mask", "tt", "ok_nwbv", "ok_mmse", "ok_cdr"):
            D[k] = np.zeros((B, T))
        D["conf"] = np.ones((B, T)); D["cdr_cat"] = np.zeros((B, T), dtype=int)
        for e in self.extra:
            D[e] = np.zeros((B, T)); D["ok_" + e] = np.zeros((B, T)); D["conf_" + e] = np.ones((B, T))
        zs = {c: self.z(d, c).values for c in COV + ["nWBV", "MMSE"] + self.extra}
        obs_vars = [("nwbv", "nWBV"), ("mmse", "MMSE")] + [(e, e) for e in self.extra]
        for i, (sid, g) in enumerate(d.groupby("sid", sort=False)):
            idx = g.index.values; n = len(idx)
            D["mask"][i, :n] = 1
            D["age"][i, :n] = zs["Age"][idx]; D["male"][i, :n] = g["male"].values
            D["etiv"][i, :n] = zs["eTIV"][idx]; D["ses"][i, :n] = zs["SES"][idx]; D["edu"][i, :n] = zs["edu"][idx]
            D["tt"][i, :n] = g["t"].values
            if "qc_conf" in g: D["conf"][i, :n] = g["qc_conf"].values
            for e in self.extra:
                if e + "_conf" in g: D["conf_" + e][i, :n] = g[e + "_conf"].fillna(1.0).clip(0.05, 1.0).values
            for key, col in obs_vars:
                v = zs[col][idx]; ok = ~np.isnan(v)
                D[key][i, :n] = np.where(ok, v, 0.0); D["ok_" + key][i, :n] = ok
            c = g["CDR"].values.astype(float)
            D["cdr_cat"][i, :n] = cdr_cat(c); D["ok_cdr"][i, :n] = ~np.isnan(c)
        D["mask"] = D["mask"].astype(bool)
        D["dt"] = np.clip(np.diff(D["tt"], axis=1), 0, None) * D["mask"][:, 1:]
        D["Xe"] = np.stack([D["age"], D["male"], D["etiv"]], -1)
        D["cov"] = np.stack([D["male"][:, 0], D["edu"][:, 0], D["ses"][:, 0]], -1)
        D["age0"] = D["age"][:, 0]
        return D


# =============================================================================== CT (semi-)Markov model
def softplus(x): return np.logaddexp(0, x)
def _lsm(x, axis=-1): return x - logsumexp(x, axis=axis, keepdims=True)


def expm_batch(A):
    nrm = np.abs(A).sum(-2).max(-1).max()
    s = int(max(0, np.ceil(np.log2(max(nrm, 1e-12) / 0.5))))
    A = A / 2.0 ** s
    I = np.broadcast_to(np.eye(A.shape[-1]), A.shape)
    E, term = I.copy(), I.copy()
    for k in range(1, 11):
        term = term @ A / k; E = E + term
    for _ in range(s): E = E @ E
    return E


@dataclass
class Spec:
    name: str; key: str; direction: int; regress: bool; dist: str; imaging: bool = True


class RubricMarkov:
    """K disease stages, each expanded into `phases` Coxian sub-phases (phases=1 -> plain CTMC).
    Q(subject,visit) = exp(b*age + c.cov) * G0, G0 a fixed upper-triangular generator (no recovery).
    Emissions depend on the stage only. `discrete=True` replaces the elapsed time by one unit step per visit."""
    NU = 5.0

    def __init__(self, K=3, phases=1, l2=2.0, extra=(), high_stage=1, discrete=False):
        self.K, self.Ph, self.S = K, phases, K * phases
        self.l2, self.high, self.discrete = l2, high_stage, discrete
        self.stage_of = np.repeat(np.arange(K), phases)
        self.specs = [Spec("nwbv", "nwbv", -1, True, "n"), Spec("mmse", "mmse", -1, False, "t", False)]
        self.specs += [Spec(e, e, +1, False, "n") for e in extra]
        layout = [("qa", (K-1)*phases), ("qf", (K-1)*(phases-1)), ("qb", 1), ("qc", 3),
                  ("pi_w", K-1), ("pi_u", 5*(K-1)), ("cdr", K*2)]
        for s in self.specs:
            layout += [(s.name+"_mu1", 1), (s.name+"_inc", K-1), (s.name+"_ls", K)]
            if s.regress: layout += [(s.name+"_g", 3)]
        self.off, o = {}, 0
        for n, sz in layout: self.off[n] = (o, o+sz); o += sz
        self.n_par = o; self.theta = None; self.active = None; self.start_nll = []
        lo, hi = np.full(o, -8.0), np.full(o, 8.0)
        for s in self.specs:
            a, b = self.off[s.name+"_ls"]; lo[a:b], hi[a:b] = np.log(0.08), np.log(3.0)
        for n in ("qa", "qf"):
            a, b = self.off[n]; lo[a:b], hi[a:b] = np.log(1e-3), np.log(5.0)
        self.bounds = list(zip(lo, hi))

    def P(self, th): return {n: th[a:b] for n, (a, b) in self.off.items()}

    def _init(self, rng, r):
        K, Ph = self.K, self.Ph; th = np.zeros(self.n_par); p = self.off
        def put(n, v): th[p[n][0]:p[n][1]] = v
        put("qa", np.log(0.15 * Ph) + 0.3*r*rng.standard_normal((K-1)*Ph))
        put("qf", np.log(0.5) + 0.3*r*rng.standard_normal((K-1)*(Ph-1)))
        put("pi_w", -1.0 - np.arange(K-1)*0.5)
        # CDR readout init: stage k centred on CDR category 2k/(K-1) (0 -> CDR 0, middle -> 0.5, last -> >=1).
        c = 2.0 * np.arange(K) / max(K - 1, 1)
        lg = -1.5 * (np.arange(3)[None, :] - c[:, None]) ** 2
        lg = lg - lg[:, :1]
        put("cdr", (lg[:, 1:] + 0.2*rng.standard_normal((K, 2))).ravel())
        for s in self.specs:
            put(s.name+"_mu1", 0.7 if s.direction == -1 else -0.5)
            put(s.name+"_inc", 0.2*rng.standard_normal(K-1))
            put(s.name+"_ls", np.full(K, -0.2))
        return th

    def _G0(self, p):
        K, Ph, S = self.K, self.Ph, self.S
        ex = np.exp(p["qa"]).reshape(K-1, Ph)
        ad = np.exp(p["qf"]).reshape(K-1, Ph-1) if Ph > 1 else np.zeros((K-1, 0))
        G = np.zeros((S, S))
        for k in range(K-1):
            for j in range(Ph):
                s = k*Ph + j
                G[s, (k+1)*Ph] += ex[k, j]; G[s, s] -= ex[k, j]
                if j < Ph-1: G[s, s+1] += ad[k, j]; G[s, s] -= ad[k, j]
        return G

    def _mult(self, p, age, cov): return np.exp(p["qb"][0]*age + (cov @ p["qc"])[:, None])

    def _logP(self, p, D):
        B, S = D["B"], self.S
        if D["T"] < 2: return np.zeros((B, 0, S, S))
        m = self._mult(p, D["age"][:, :-1], D["cov"])
        dt = D["mask"][:, 1:].astype(float) if self.discrete else D["dt"]
        return np.log(np.clip(expm_batch((m*dt)[..., None, None] * self._G0(p)), 1e-300, None))

    def _logpi_stage(self, p, D):
        K = self.K; lg = np.zeros((D["B"], K))
        Z0 = np.stack([D["age0"], D["male"][:, 0], D["edu"][:, 0], D["ses"][:, 0], D["etiv"][:, 0]], 1)
        lg[:, 1:] = p["pi_w"][None] + Z0 @ p["pi_u"].reshape(5, K-1)
        return _lsm(lg)

    def _logpi(self, p, D):
        ls = self._logpi_stage(p, D)
        if self.Ph == 1: return ls
        out = np.full((D["B"], self.S), -np.inf); out[:, ::self.Ph] = ls
        return out

    def _mu(self, p, s):
        inc = softplus(p[s.name+"_inc"])
        return p[s.name+"_mu1"][0] + s.direction*np.concatenate([[0.0], np.cumsum(inc)])

    def cdr_probs(self):
        p = self.P(self.theta); lg = np.concatenate([np.zeros((self.K, 1)), p["cdr"].reshape(self.K, 2)], 1)
        return np.exp(_lsm(lg))                      # (K, 3) P(CDR category | stage)

    @staticmethod
    def _w(D, s):
        """tempering weight of one emission channel: scan-QC (image channels) x per-channel confidence (e.g. Qwen agreement)."""
        w = D["conf"] if (s.imaging or CONF_ON == "all") else np.ones_like(D["conf"])
        if "conf_" + s.name in D: w = w * D["conf_" + s.name]
        return w[..., None]

    def _emis(self, p, D, active):
        K = self.K; E = np.zeros((D["B"], D["T"], K))
        for s in self.specs:
            if s.name not in active: continue
            m = self._mu(p, s)[None, None, :]
            if s.regress: m = m + (D["Xe"] @ p[s.name+"_g"])[..., None]
            sig = np.exp(p[s.name+"_ls"])[None, None, :]
            z = (D[s.key][..., None] - m)/sig
            if s.dist == "t":
                nu = self.NU
                lp = (gammaln((nu+1)/2) - gammaln(nu/2) - 0.5*np.log(nu*np.pi) - np.log(sig)
                      - (nu+1)/2*np.log1p(z**2/nu))
            else:
                lp = -0.5*z**2 - 0.5*np.log(2*np.pi) - np.log(sig)
            E += np.where(D["ok_"+s.key][..., None] > 0, lp, 0.0) * self._w(D, s)
        if "cdr" in active:
            lg = np.concatenate([np.zeros((K, 1)), p["cdr"].reshape(K, 2)], 1)
            lc = _lsm(lg)[:, D["cdr_cat"]].transpose(1, 2, 0)
            E += np.where(D["ok_cdr"][..., None] > 0, lc, 0.0) * (D["conf"][..., None] if CONF_ON == "all" else 1.0)
        E = E * D["mask"][..., None]
        return np.repeat(E, self.Ph, axis=2)

    @staticmethod
    def _forward(logpi, logP, E, mask):
        B, T, S = E.shape; a = np.empty((B, T, S)); a[:, 0] = logpi + E[:, 0]
        for t in range(1, T):
            st = logsumexp(a[:, t-1, :, None] + logP[:, t-1], axis=1) + E[:, t]
            a[:, t] = np.where(mask[:, t, None], st, a[:, t-1])
        return a

    @staticmethod
    def _backward(logP, E, mask):
        B, T, S = E.shape; b = np.zeros((B, T, S))
        for t in range(T-2, -1, -1):
            nx = logsumexp(logP[:, t] + (E[:, t+1]+b[:, t+1])[:, None, :], axis=2)
            b[:, t] = np.where(mask[:, t+1, None], nx, 0.0)
        return b

    def _obj(self, th, D, active):
        """numpy reference NLL (used as fallback and as the self-test target for the torch NLL)."""
        p = self.P(th)
        a = self._forward(self._logpi(p, D), self._logP(p, D), self._emis(p, D, active), D["mask"])
        ll = logsumexp(a[:, -1], axis=1).sum()
        pen = 0.5*self.l2*(p["qb"]@p["qb"] + p["qc"]@p["qc"] + p["pi_u"]@p["pi_u"])
        for s in self.specs:
            if s.regress: g = p[s.name+"_g"]; pen += 0.5*self.l2*g@g
        return -ll + pen

    def fit_numpy(self, D, active, restarts=1, seed=0, maxiter=250, x0=None):
        self.active = set(active); rng = np.random.default_rng(seed); best = None; self.start_nll = []
        starts = ([np.asarray(x0, float)] if x0 is not None else []) + \
                 [self._init(rng, r) for r in range(min(restarts, MAX_RESTARTS) if x0 is None else 0)]
        for s0 in starts:
            res = minimize(self._obj, s0, args=(D, self.active), method="L-BFGS-B",
                           bounds=self.bounds, options=dict(maxiter=maxiter))
            self.start_nll.append(float(res.fun))
            if best is None or res.fun < best.fun: best = res
        self.theta, self.fun = best.x, best.fun
        return self

    def fit(self, D, active, restarts=1, seed=0, maxiter=250, x0=None):
        """torch/autograd (GPU if available) when possible, numpy finite-difference fallback otherwise."""
        if _HAS_TORCH: return fit_torch(self, D, active, restarts, seed, maxiter, x0)
        return self.fit_numpy(D, active, restarts, seed, maxiter, x0)

    def _stage_sum(self, x): return x.reshape(x.shape[:-1] + (self.K, self.Ph)).sum(-1)

    def infer(self, D, active, horizon=None):
        """horizon=None  -> joint posterior over (initial stage i, final stage k):
                              Nondemented=P(i<h,k<h)  Converted=P(i<h,k>=h)  Demented=P(i>=h)
           horizon=array -> filtered posterior at the last observed visit propagated forward `horizon`
                              (years; visits if discrete): Demented=P(last>=h)  Converted=P(last<h,future>=h)
                              Nondemented=P(last<h,future<h). Also returns stage_future and the expected time
                              to reach the impaired set from the last visit (t_hit)."""
        active = set(active)
        p = self.P(self.theta); K, Ph, S, h = self.K, self.Ph, self.S, self.high
        logpi, logP, E = self._logpi(p, D), self._logP(p, D), self._emis(p, D, active)
        a = self._forward(logpi, logP, E, D["mask"]); b = self._backward(logP, E, D["mask"])
        gam = self._stage_sum(np.exp(_lsm(a+b)))
        res = dict(gamma_stage=gam, stage_future=None, t_hit=None)
        if horizon is None:
            j = np.full((D["B"], K, S), -np.inf)
            for i in range(K): j[:, i, i*Ph] = logpi[:, i*Ph] + E[:, 0, i*Ph]
            for t in range(1, D["T"]):
                st = logsumexp(j[:, :, :, None] + logP[:, t-1][:, None], axis=2) + E[:, t][:, None, :]
                j = np.where(D["mask"][:, t, None, None], st, j)
            joint = np.exp(j - logsumexp(j, axis=(1, 2), keepdims=True))
            js = joint.reshape(D["B"], K, K, Ph).sum(-1)
            p_nd, p_conv, p_dem = js[:, :h, :h].sum((1, 2)), js[:, :h, h:].sum((1, 2)), js[:, h:, :].sum((1, 2))
        else:
            alpha = np.exp(_lsm(a[:, -1]))
            n = D["mask"].sum(1); age_last = D["age"][np.arange(D["B"]), n-1]
            m = np.exp(p["qb"][0]*age_last + D["cov"] @ p["qc"])
            hz = np.asarray(horizon, float)
            G0 = self._G0(p); Pf = expm_batch((m*hz)[:, None, None] * G0)
            low, high = np.where(self.stage_of < h)[0], np.where(self.stage_of >= h)[0]
            p_dem = alpha[:, high].sum(1)
            p_conv = (alpha[:, low, None] * Pf[:, low][:, :, high]).sum((1, 2))
            p_nd = (alpha[:, low, None] * Pf[:, low][:, :, low]).sum((1, 2))
            res["stage_future"] = self._stage_sum(np.einsum("bi,bij->bj", alpha, Pf))
            # E[time to impairment | impairment happens within the horizon], by CDF differences on a fine grid
            g = np.linspace(0.0, 1.0, 41)
            ts = hz[:, None] * g[None, :]                                             # (B, 41) years
            Pg = expm_batch((m[:, None] * ts)[..., None, None] * G0)                  # (B, 41, S, S)
            F = np.einsum("bi,bjik->bj", alpha[:, low], Pg[:, :, low][:, :, :, high])  # P(impaired by t_j | unimpaired now)
            dF = np.diff(F, axis=1); tmid = 0.5 * (ts[:, 1:] + ts[:, :-1])
            Fh = F[:, -1]
            res["t_hit"] = np.where(Fh > 1e-12, (dF * tmid).sum(1) / np.maximum(Fh, 1e-12), hz / 2.0)
        proba = np.stack([p_nd, p_conv, p_dem], 1)
        assert np.all(proba > -1e-9) and np.allclose(proba.sum(1), 1.0, atol=1e-4), "class probabilities invalid"
        proba = np.clip(proba, 0, None); proba /= proba.sum(1, keepdims=True)
        res["proba"] = proba
        return res

    def summary(self, prep):
        p = self.P(self.theta); K, Ph = self.K, self.Ph; G = self._G0(p)
        dwell = [float(np.linalg.solve(-G[k*Ph:(k+1)*Ph, k*Ph:(k+1)*Ph], np.ones(Ph))[0]) for k in range(K-1)]
        out = {"active_emissions": sorted(self.active), "phases_per_stage": Ph,
               "mean_dwell_years_ref": dwell, "rate_per_year_ref": [1/x for x in dwell],
               "HR_per_SD_age": float(np.exp(p["qb"][0])), "HR_male": float(np.exp(p["qc"][0])),
               "HR_per_SD_education": float(np.exp(p["qc"][1])), "HR_per_SD_SES": float(np.exp(p["qc"][2]))}
        out["nWBV_stage_means_raw"] = (prep.mu["nWBV"] + prep.sd["nWBV"]*self._mu(p, self.specs[0])).tolist()
        if "mmse" in self.active:
            out["MMSE_stage_means_raw"] = (prep.mu["MMSE"] + prep.sd["MMSE"]*self._mu(p, self.specs[1])).tolist()
        if "cdr" in self.active:
            out["P(CDR=0|0.5|>=1 | stage)"] = self.cdr_probs().round(3).tolist()
        for s in self.specs[2:]:
            out[f"{s.name}_stage_means_raw"] = (prep.mu[s.name] + prep.sd[s.name]*self._mu(p, s)).tolist()
        return out

    def progression_curves(self, years, male=0.5, age_z=0.0):
        """Reference-subject stage occupancy P(stage k at time t | healthy at t=0): the population 'how it progresses' curve."""
        p = self.P(self.theta); G0 = self._G0(p)
        m = float(np.exp(p["qb"][0]*age_z + p["qc"][0]*male))
        Pt = expm_batch((m*np.asarray(years, float))[:, None, None] * G0)[:, 0, :]
        return self._stage_sum(Pt)                                       # (len(years), K)


# =============================================================================== torch / autograd likelihood
_SELF_TESTED = False


def _to_t(D):
    T = {}
    for k, v in D.items():
        if isinstance(v, np.ndarray):
            if v.dtype == bool: T[k] = torch.tensor(v, device=DEV)
            elif k == "cdr_cat": T[k] = torch.tensor(v, dtype=torch.long, device=DEV)
            elif v.dtype.kind in "fiu": T[k] = torch.tensor(v, dtype=torch.float64, device=DEV)
    T["B"], T["T"] = D["B"], D["T"]
    return T


def _G0_t(m, p):
    K, Ph, S = m.K, m.Ph, m.S
    ex = torch.exp(p["qa"]).reshape(K - 1, Ph)
    ad = torch.exp(p["qf"]).reshape(K - 1, Ph - 1) if Ph > 1 else None
    rows, cols, vals = [], [], []
    for k in range(K - 1):
        for j in range(Ph):
            s = k * Ph + j
            rows += [s, s]; cols += [(k + 1) * Ph, s]; vals += [ex[k, j], -ex[k, j]]
            if j < Ph - 1:
                rows += [s, s]; cols += [s + 1, s]; vals += [ad[k, j], -ad[k, j]]
    return torch.zeros(S, S, dtype=torch.float64, device=DEV).index_put(
        (torch.tensor(rows, device=DEV), torch.tensor(cols, device=DEV)), torch.stack(vals), accumulate=True)


def torch_nll(m, th, T, active):
    """Exact torch mirror of RubricMarkov._obj (differentiable)."""
    f = torch.float64
    P = {n: th[a:b] for n, (a, b) in m.off.items()}
    B, Tn, K, Ph, S = T["B"], T["T"], m.K, m.Ph, m.S
    G0 = _G0_t(m, P)
    if Tn >= 2:
        mult = torch.exp(P["qb"][0] * T["age"][:, :-1] + (T["cov"] @ P["qc"])[:, None])
        dt = T["mask"][:, 1:].to(f) if m.discrete else T["dt"]
        logP = torch.log(torch.clamp(torch.linalg.matrix_exp((mult * dt)[..., None, None] * G0), min=1e-300))
    Z0 = torch.stack([T["age0"], T["male"][:, 0], T["edu"][:, 0], T["ses"][:, 0], T["etiv"][:, 0]], 1)
    lg = torch.cat([torch.zeros(B, 1, dtype=f, device=DEV), P["pi_w"][None] + Z0 @ P["pi_u"].reshape(5, K - 1)], 1)
    ls = torch.log_softmax(lg, 1)
    logpi = ls if Ph == 1 else torch.full((B, S), -math.inf, dtype=f, device=DEV).index_copy(
        1, torch.arange(0, S, Ph, device=DEV), ls)
    E = torch.zeros(B, Tn, K, dtype=f, device=DEV)
    for s in m.specs:
        if s.name not in active: continue
        inc = torch.nn.functional.softplus(P[s.name + "_inc"])
        mu = P[s.name + "_mu1"][0] + s.direction * torch.cat([torch.zeros(1, dtype=f, device=DEV), torch.cumsum(inc, 0)])
        mm = mu[None, None, :]
        if s.regress: mm = mm + (T["Xe"] @ P[s.name + "_g"])[..., None]
        sig = torch.exp(P[s.name + "_ls"])[None, None, :]
        z = (T[s.key][..., None] - mm) / sig
        if s.dist == "t":
            nu = m.NU
            lp = (math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi) - torch.log(sig)
                  - (nu + 1) / 2 * torch.log1p(z ** 2 / nu))
        else:
            lp = -0.5 * z ** 2 - 0.5 * math.log(2 * math.pi) - torch.log(sig)
        w = T["conf"] if (s.imaging or CONF_ON == "all") else torch.ones_like(T["conf"])
        if "conf_" + s.name in T: w = w * T["conf_" + s.name]
        E = E + torch.where(T["ok_" + s.key][..., None] > 0, lp, torch.zeros_like(lp)) * w[..., None]
    if "cdr" in active:
        lc = torch.log_softmax(torch.cat([torch.zeros(K, 1, dtype=f, device=DEV), P["cdr"].reshape(K, 2)], 1), 1)
        lc = lc[:, T["cdr_cat"]].permute(1, 2, 0)
        E = E + torch.where(T["ok_cdr"][..., None] > 0, lc, torch.zeros_like(lc)) * (T["conf"][..., None] if CONF_ON == "all" else 1.0)
    E = torch.repeat_interleave(E * T["mask"][..., None], Ph, dim=2)
    a = logpi + E[:, 0]
    for t in range(1, Tn):
        st = torch.logsumexp(a[:, :, None] + logP[:, t - 1], dim=1) + E[:, t]
        a = torch.where(T["mask"][:, t, None], st, a)
    ll = torch.logsumexp(a, 1).sum()
    pen = 0.5 * m.l2 * (P["qb"] @ P["qb"] + P["qc"] @ P["qc"] + P["pi_u"] @ P["pi_u"])
    for s in m.specs:
        if s.regress: pen = pen + 0.5 * m.l2 * (P[s.name + "_g"] @ P[s.name + "_g"])
    return -ll + pen


def fit_torch(m, D, active, restarts=1, seed=0, maxiter=250, x0=None):
    global _SELF_TESTED
    m.active = set(active); T = _to_t(D)
    if not _SELF_TESTED:
        th = m._init(np.random.default_rng(123), 1); ref = m._obj(th, D, m.active)
        t = torch.tensor(th, dtype=torch.float64, device=DEV, requires_grad=True); v = torch_nll(m, t, T, m.active); v.backward()
        print(f"[torch] device={DEV} | self-test NLL numpy={ref:.6f} torch={float(v):.6f} |diff|={abs(ref-float(v)):.2e}")
        assert abs(ref - float(v)) < 1e-6 * max(1.0, abs(ref)), "torch NLL disagrees with numpy NLL -- do not trust fits"
        _SELF_TESTED = True

    def fun(th):
        t = torch.tensor(th, dtype=torch.float64, device=DEV, requires_grad=True)
        loss = torch_nll(m, t, T, m.active); loss.backward()
        return float(loss), t.grad.cpu().numpy().astype(np.float64)

    rng = np.random.default_rng(seed); best = None; m.start_nll = []
    starts = ([np.asarray(x0, float)] if x0 is not None else []) + \
             [m._init(rng, r) for r in range(min(restarts, MAX_RESTARTS) if x0 is None else 0)]
    for s0 in starts:
        res = minimize(fun, s0, jac=True, method="L-BFGS-B", bounds=m.bounds, options=dict(maxiter=maxiter))
        m.start_nll.append(float(res.fun))
        if best is None or res.fun < best.fun: best = res
    m.theta, m.fun = best.x, best.fun
    return m


# =============================================================================== evaluation helpers
def decide(proba, prior, alpha=1.0):
    return np.argmax(np.log(proba+1e-12) - alpha*np.log(prior+1e-12), 1)


def ece(proba, y, bins=5):
    conf, pred = proba.max(1), proba.argmax(1)
    edges = np.linspace(1/proba.shape[1], 1, bins+1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean()*abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e


def metrics(y, pred, proba=None):
    r = {"acc": (y == pred).mean(), "bal_acc": balanced_accuracy_score(y, pred),
         "macro_f1": f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0)}
    cm = confusion_matrix(y, pred, labels=[0, 1, 2]); rec = cm.diagonal()/np.maximum(cm.sum(1), 1)
    for c, v in zip(CLASSES, rec): r["recall_"+c] = v
    if proba is not None:
        try: r["auc_ovr"] = roc_auc_score(y, proba, multi_class="ovr", labels=[0, 1, 2])
        except ValueError: r["auc_ovr"] = np.nan
        r["ece"] = ece(proba, y)
    return r, cm


def visit_cdr_metrics(mdl, out, D):
    pc = out["gamma_stage"] @ mdl.cdr_probs()
    ok = D["mask"] & (D["ok_cdr"] > 0)
    if not ok.any(): return {}
    y, pred = D["cdr_cat"][ok], pc[ok].argmax(1)
    return dict(cdr_visit_acc=(pred == y).mean(), cdr_visit_mae=np.abs(pred - y).mean(),
                cdr_visit_f1=f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0))


def forecast_cdr_metrics(mdl, out, oc):
    pf = out["stage_future"] @ mdl.cdr_probs()
    y = oc["cdr_final"].values; ok = ~np.isnan(y)
    if not ok.any(): return {}
    yi = y[ok].astype(int)
    return dict(fc_cdr_acc=(pf[ok].argmax(1) == yi).mean(), fc_cdr_ll=np.log(pf[ok][np.arange(ok.sum()), yi] + 1e-12).mean())


def ttc_mae(t_hit, oc):
    ok = (oc["y"].values == 1) & ~np.isnan(oc["t_conv"].values) & ~np.isnan(t_hit)
    return float(np.abs(t_hit[ok] - oc["t_conv"].values[ok]).mean()) if ok.any() else np.nan


def ttc_mae_naive(oc):
    """Reference: always predict half of the follow-up horizon."""
    ok = (oc["y"].values == 1) & ~np.isnan(oc["t_conv"].values)
    return float(np.abs(oc["horizon"].values[ok] / 2 - oc["t_conv"].values[ok]).mean()) if ok.any() else np.nan


def conv_auc(y, proba):
    """AUC of 'converter vs stable' among subjects not already impaired (the clinically hard question)."""
    k = np.isin(y, [0, 1])
    if k.sum() < 3 or len(np.unique(y[k])) < 2: return np.nan
    s = proba[k, 1] / np.maximum(proba[k, 0] + proba[k, 1], 1e-12)
    return float(roc_auc_score(y[k] == 1, s))


def _slope(t, y):
    t = np.asarray(t, float); y = np.asarray(y, float)
    ok = ~(np.isnan(t) | np.isnan(y)); t, y = t[ok], y[ok]
    if len(t) < 2 or t.max() - t.min() <= 0: return np.nan
    return np.polyfit(t, y, 1)[0]
def _first(s): s = s.dropna(); return s.iloc[0] if len(s) else np.nan
def _last(s): s = s.dropna(); return s.iloc[-1] if len(s) else np.nan


def subject_features(d, mode, extra=()):
    rows = []
    for sid, g in d.groupby("sid", sort=False):
        t = g["t"].values
        f = dict(sid=sid, age0=g["Age"].iloc[0], male=g["male"].iloc[0], edu=g["edu"].iloc[0],
                 ses=g["SES"].iloc[0], etiv=g["eTIV"].iloc[0], nwbv0=_first(g["nWBV"]), nwbv_last=_last(g["nWBV"]),
                 nwbv_slope=_slope(t, g["nWBV"].values), span=t.max() - t.min(), n=len(g))
        if mode in ("B", "C"):
            m = g["MMSE"].astype(float)
            f.update(mmse0=_first(m), mmse_last=_last(m), mmse_slope=_slope(t, m.values))
        if mode == "C":
            f.update(cdr0=_first(g["CDR"]), cdr_last=_last(g["CDR"]), cdr_max=g["CDR"].max())
        for e in extra:
            f[e+"_0"] = _first(g[e]); f[e+"_slope"] = _slope(t, g[e].values)
        rows.append(f)
    return pd.DataFrame(rows).set_index("sid")


def baselines(seed):
    imp = lambda: SimpleImputer(strategy="median")
    return {"LogReg": make_pipeline(imp(), StandardScaler(),
                                    LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000)),
            "RandForest": make_pipeline(imp(), RandomForestClassifier(300, min_samples_leaf=2,
                                        class_weight="balanced_subsample", random_state=seed, n_jobs=-1)),
            "HistGB": make_pipeline(imp(), HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                    max_iter=150, class_weight="balanced", random_state=seed))}


def gru_fit_predict(Dtr, ytr, Dte, mode, seed, epochs=80, hidden=16):
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    def feats(D):
        f = [D["age"], D["male"], D["etiv"], D["ses"], D["edu"], D["nwbv"], D["ok_nwbv"]]
        if mode in ("B", "C"): f += [D["mmse"], D["ok_mmse"]]
        if mode == "C": f += [D["cdr_cat"] / 2.0, D["ok_cdr"]]
        f.append(np.concatenate([np.zeros((D["B"], 1)), D["dt"]], 1))
        return (torch.tensor(np.stack(f, -1), dtype=torch.float32),
                torch.tensor(D["mask"].sum(1) - 1, dtype=torch.long))
    class Net(nn.Module):
        def __init__(s, i):
            super().__init__(); s.g = nn.GRU(i, hidden, batch_first=True); s.o = nn.Linear(hidden, 3)
        def forward(s, x, last):
            h, _ = s.g(x); return s.o(h[torch.arange(len(x)), last])
    Xtr, ltr = feats(Dtr); Xte, lte = feats(Dte); y = torch.tensor(ytr, dtype=torch.long)
    cnt = np.bincount(ytr, minlength=3).astype(float); w = torch.tensor(len(ytr) / (3 * np.maximum(cnt, 1)), dtype=torch.float32)
    net = Net(Xtr.shape[-1]); opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss(weight=w)
    for _ in range(epochs):
        opt.zero_grad(); lossf(net(Xtr, ltr), y).backward(); opt.step()
    with torch.no_grad(): return torch.softmax(net(Xte, lte), 1).numpy()


@dataclass
class Cfg:
    K: int = 3; phases: int = 1; discrete: bool = False; sup: str = "full"
    splits: int = 5; repeats: int = 3; restarts: int = 1; maxiter: int = 150
    alpha: float = 1.0; seed: int = 0; gru: bool = False; retro_oracle: bool = False


def run_cv(d, outcome, cutoff, cfg, extra, log=print):
    """Prospective (cutoff!=None): the Markov model, ALL preprocessing statistics and the baselines see only visits
    <= cutoff in BOTH train and test folds. Future visits are used only to build outcome labels (outcome frame)."""
    t0 = time.time()
    sids = outcome.index.values; y_all = outcome["y"].values
    cnt = np.bincount(y_all, minlength=3); n_min = cnt[cnt > 0].min()
    splits = int(min(cfg.splits, max(2, n_min)))
    if splits < cfg.splits: log(f"  (smallest class has {n_min} subjects -> using {splits} splits)")
    cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=cfg.repeats, random_state=cfg.seed)
    # retrospective Mode C is an oracle (Group is defined from CDR): skipped unless explicitly requested, always labelled
    modes = {m: s for m, s in MODES.items() if m != "C" or cutoff is not None or cfg.retro_oracle}
    recs, cms, oof = [], {}, []
    for fold, (tr, te) in enumerate(cv.split(sids, y_all)):
        rep = fold // splits
        dtr = observed_window(d[d.sid.isin(sids[tr])], cutoff).reset_index(drop=True)   # TRAIN ON THE WINDOW ONLY
        dte = observed_window(d[d.sid.isin(sids[te])], cutoff).reset_index(drop=True)
        assert_no_future(dtr, cutoff, "train"); assert_no_future(dte, cutoff, "test")
        prep = Prep().fit(dtr, extra)                                                   # stats from the restricted train data
        Dtr, Dte = prep.tensors(dtr), prep.tensors(dte)
        oc = outcome.loc[Dte["sids"]]
        ytr = outcome.loc[Dtr["sids"], "y"].values; yte = oc["y"].values                # future-derived labels: eval/prior only
        hor = None if cutoff is None else (oc["nsteps"].values if cfg.discrete else oc["horizon"].values)
        prior = np.bincount(ytr, minlength=3) / len(ytr)
        fitted = {}
        for m, act0 in modes.items():
            mlabel = m + "-ORACLE" if (cutoff is None and m == "C") else m
            act = act0 | set(extra)
            fit_act = act if cfg.sup == "matched" else (MODES["C"] | set(extra))
            key = frozenset(fit_act)
            if key not in fitted:
                fitted[key] = RubricMarkov(cfg.K, cfg.phases, extra=extra, discrete=cfg.discrete).fit(
                    Dtr, fit_act, restarts=cfg.restarts, seed=cfg.seed + fold, maxiter=cfg.maxiter)
            mdl = fitted[key]
            out = mdl.infer(Dte, act, hor); pr = out["proba"]
            em = {}
            if "cdr" in fit_act and "cdr" not in act:
                em.update(visit_cdr_metrics(mdl, out, Dte))
                if cutoff is not None: em.update(forecast_cdr_metrics(mdl, out, oc))
            if cutoff is not None and not cfg.discrete:
                em["ttc_mae_years"] = ttc_mae(out["t_hit"], oc); em["ttc_mae_naive_halfH"] = ttc_mae_naive(oc)
            em["auc_conv_vs_stable"] = conv_auc(yte, pr)
            for tag, pred in (("RubricMarkov (MAP)", pr.argmax(1)), ("RubricMarkov (balanced)", decide(pr, prior, cfg.alpha))):
                r, cm = metrics(yte, pred, pr); r.update(em); r.update(mode=mlabel, method=tag, fold=fold, rep=rep); recs.append(r)
                cms.setdefault((m, tag), np.zeros((3, 3))); cms[(m, tag)] += cm
            if rep == 0:
                oof.append(pd.DataFrame(dict(sid=Dte["sids"], mode=mlabel, y=yte, p_nondemented=pr[:, 0], p_converted=pr[:, 1],
                                             p_demented=pr[:, 2], pred_balanced=decide(pr, prior, cfg.alpha))))
            Xtr = subject_features(dtr, m, extra).loc[Dtr["sids"]]
            Xte = subject_features(dte, m, extra).loc[Dte["sids"]]
            for name, clf in baselines(cfg.seed + fold).items():
                clf.fit(Xtr, ytr); pb = np.zeros((len(yte), 3)); pb[:, clf.classes_] = clf.predict_proba(Xte)
                r, cm = metrics(yte, pb.argmax(1), pb); r["auc_conv_vs_stable"] = conv_auc(yte, pb)
                r.update(mode=mlabel, method=name, fold=fold, rep=rep); recs.append(r)
                cms.setdefault((m, name), np.zeros((3, 3))); cms[(m, name)] += cm
            if cfg.gru:
                pb = gru_fit_predict(Dtr, ytr, Dte, m, cfg.seed + fold)
                r, cm = metrics(yte, pb.argmax(1), pb); r.update(mode=mlabel, method="GRU", fold=fold, rep=rep); recs.append(r)
                cms.setdefault((m, "GRU"), np.zeros((3, 3))); cms[(m, "GRU")] += cm
        log(f"  fold {fold+1}/{splits*cfg.repeats} done ({time.time()-t0:.0f}s elapsed)")
    return pd.DataFrame(recs), cms, (pd.concat(oof) if oof else pd.DataFrame()), splits


def mean_sd_table(res):
    base = ["acc", "bal_acc", "macro_f1", "auc_ovr", "auc_conv_vs_stable", "ece"] + ["recall_"+c for c in CLASSES]
    opt = [c for c in ("cdr_visit_acc", "cdr_visit_mae", "cdr_visit_f1", "fc_cdr_acc", "fc_cdr_ll", "ttc_mae_years", "ttc_mae_naive_halfH")
           if c in res and res[c].notna().any()]
    cols = base + opt
    g = res.groupby(["mode", "method"])[cols]; m, s = g.mean(), g.std(); out = m.copy().astype(object)
    for c in cols: out[c] = [("" if np.isnan(a) else f"{a:.3f} ± {0 if np.isnan(b) else b:.3f}") for a, b in zip(m[c], s[c])]
    return out


def atrophy_tempo(d):
    from scipy.stats import kruskal, mannwhitneyu
    rows = []
    for sid, g in d.groupby("sid", sort=False):
        base = _first(g["nWBV"]); sl = _slope(g["t"].values, g["nWBV"].values)
        if not np.isnan(sl) and not np.isnan(base):
            rows.append(dict(sid=sid, Group=g["Group"].iloc[0], slope_pct=100*sl/base))
    s = pd.DataFrame(rows)
    tab = s.groupby("Group")["slope_pct"].agg(["count", "mean", "std", "median"]).reindex(CLASSES)
    p_kw = kruskal(*[s[s.Group == c].slope_pct for c in CLASSES]).pvalue
    pw = {f"{a} vs {b}": mannwhitneyu(s[s.Group == a].slope_pct, s[s.Group == b].slope_pct).pvalue
          for a, b in (("Nondemented", "Converted"), ("Nondemented", "Demented"), ("Converted", "Demented"))}
    return tab, p_kw, pw, s


def bootstrap_hr(d, cfg, extra, n_boot, seed, active, log=print):
    """Subject bootstrap of dwell times / hazard ratios. One full-data fit (2 starts, NLLs printed), then every
    replicate is WARM-STARTED from it (1 start, <=100 iters). Warm starts are used ONLY here, never in CV."""
    rng = np.random.default_rng(seed); sids = d.sid.unique(); out = []
    prep0 = Prep().fit(d, extra)
    base = RubricMarkov(cfg.K, cfg.phases, extra=extra, discrete=cfg.discrete).fit(
        prep0.tensors(d), active, restarts=2, seed=seed, maxiter=cfg.maxiter)
    if len(base.start_nll) > 1: log(f"  full-data start NLLs {np.round(base.start_nll, 3)} (equal => stable optimum)")
    x0 = base.theta.copy()
    for b in range(n_boot):
        pick = rng.choice(sids, len(sids), replace=True)
        db = pd.concat([d[d.sid == s].assign(sid=f"{s}_{i}") for i, s in enumerate(pick)]).reset_index(drop=True)
        prep = Prep().fit(db, extra); Db = prep.tensors(db)
        s = RubricMarkov(cfg.K, cfg.phases, extra=extra, discrete=cfg.discrete).fit(
            Db, active, restarts=1, seed=seed+b, maxiter=min(cfg.maxiter, 100), x0=x0).summary(prep)
        out.append([*s["mean_dwell_years_ref"], s["HR_per_SD_age"], s["HR_male"], s["HR_per_SD_education"], s["HR_per_SD_SES"]])
        if (b+1) % 10 == 0: log(f"  bootstrap {b+1}/{n_boot}")
    names = [f"dwell_stage{k+1}_yrs" for k in range(cfg.K-1)] + ["HR_age_per_SD", "HR_male", "HR_edu_per_SD", "HR_SES_per_SD"]
    return pd.DataFrame(out, columns=names)


def forecast_table(mdl, prep, d, act, horizons):
    """Per-subject forecast from ALL observed visits (descriptive/in-sample; the prospective CV is the validation)."""
    D = prep.tensors(d); n = D["mask"].sum(1)
    gam = mdl.infer(D, act, np.full(D["B"], 1e-3))["gamma_stage"]
    cur = gam[np.arange(D["B"]), n-1]
    tab = pd.DataFrame({"sid": D["sids"], "n_visits": n})
    for k in range(mdl.K): tab[f"P_stage{k}_now"] = cur[:, k]
    tab["P_impaired_now"] = cur[:, mdl.high:].sum(1)
    for hz in horizons:
        o = mdl.infer(D, act, np.full(D["B"], float(hz)))
        sf = o["stage_future"]
        tab[f"P_impaired_{hz}y"] = sf[:, mdl.high:].sum(1)
        tab[f"P_stage{mdl.K-1}_{hz}y"] = sf[:, -1]
    o10 = mdl.infer(D, act, np.full(D["B"], 10.0))
    tab["exp_years_to_impairment_given_within_10y"] = np.where(tab["P_impaired_now"] > 0.5, 0.0, o10["t_hit"])
    return tab.round(4)


def prepare(a):
    d = load_table(find_table(a.XLSX))
    cols = a.EXTRA_COLS.split(",") if isinstance(a.EXTRA_COLS, str) else a.EXTRA_COLS
    d, extra = load_vlm(d, a.VLM_CSV, cols)
    return d, extra


def make_cfg(a):
    c = Cfg(K=a.K, phases=a.PHASES, discrete=a.DISCRETE, sup=a.SUP, splits=a.SPLITS, repeats=a.REPEATS,
            restarts=a.RESTARTS, maxiter=a.MAXITER, alpha=a.ALPHA, seed=a.SEED, gru=a.GRU,
            retro_oracle=a.RETRO_ORACLE)
    if a.QUICK: c = replace(c, splits=3, repeats=1, restarts=1, maxiter=60)
    return c


# =============================================================================== stage command
def _plot_all(path, cms, cfg, splits, curves, years, slopes, title):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 9)); ax = ax.ravel()
    for axi, key in zip(ax[:2], [("A", "RubricMarkov (balanced)"), ("B", "RubricMarkov (balanced)")]):
        cm = cms[key]/np.maximum(cms[key].sum(1, keepdims=True), 1); axi.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3): axi.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center", color="w" if cm[i, j] > .5 else "k")
        axi.set_xticks(range(3)); axi.set_xticklabels(CLASSES, rotation=20); axi.set_yticks(range(3)); axi.set_yticklabels(CLASSES)
        axi.set_title(f"{title} | mode {key[0]} confusion (row-normalised)"); axi.set_xlabel("predicted"); axi.set_ylabel("true")
    names = ["stage 0 (CDR 0)", "stage 1 (CDR 0.5)", "stage 2 (CDR>=1)"] if curves.shape[1] == 3 else [f"stage {k}" for k in range(curves.shape[1])]
    for k in range(curves.shape[1]): ax[2].plot(years, curves[:, k], label=names[k])
    ax[2].set_xlabel("years from a cognitively normal baseline"); ax[2].set_ylabel("probability")
    ax[2].set_title("Model-implied disease progression (reference subject)"); ax[2].legend()
    for c in CLASSES: ax[3].hist(slopes[slopes.Group == c].slope_pct, bins=15, alpha=.5, label=c)
    ax[3].set_xlabel("nWBV change (% of baseline / yr)"); ax[3].legend(); ax[3].set_title("Atrophy tempo by group")
    plt.tight_layout(); plt.savefig(path, dpi=150)
    try:
        from IPython import get_ipython
        if get_ipython() is not None: plt.show()
    except Exception: pass
    plt.close(fig)


def cmd_stage(a, cutoff=None, out_dir=None):
    cfg = make_cfg(a); boot = min(a.BOOT, 5) if a.QUICK else a.BOOT
    out_dir = out_dir or a.OUT; os.makedirs(out_dir, exist_ok=True)
    d, extra = prepare(a); d, outcome = build_task(d, cutoff)
    print(f"\n{'='*100}\ntask: {'RETROSPECTIVE staging (all visits)' if cutoff is None else f'PROSPECTIVE prediction, cutoff={cutoff}y'} | "
          f"K={cfg.K} phases={cfg.phases} discrete={cfg.discrete} sup={cfg.sup} | image emissions={extra}")
    print(f"{d.sid.nunique()} subjects / {len(d)} visits;",
          {CLASSES[k]: int(v) for k, v in outcome['y'].value_counts().sort_index().items()})
    res, cms, oof, splits = run_cv(d, outcome, cutoff, cfg, extra)
    res.to_csv(f"{out_dir}/cv_folds.csv", index=False); oof.to_csv(f"{out_dir}/oof_predictions.csv", index=False)
    tab = mean_sd_table(res); tab.to_csv(f"{out_dir}/cv_mean_sd.csv")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(tab.to_string())
    if cutoff is None:
        print("\nNOTE (retrospective): Group is DEFINED from CDR. Mode C (if run via RETRO_ORACLE) is an ORACLE labelled 'C-ORACLE' "
              "and must not be reported as a result. Headline = modes A (nWBV+image) and B (+MMSE). SUP='full' uses CDR only as "
              "a weak label in TRAIN folds; report SUP='matched' (ablation) alongside it.")
    else:
        print("\nNOTE (prospective): train AND test folds are restricted to visits <= cutoff (asserted); future visits only define "
              "the outcome labels. CDR/MMSE at the observed visits are legitimately known at prediction time, so mode C is valid here. "
              "'Demented' (already impaired at last observed visit) is easy in C; the hard, clinically relevant column is "
              "'Converted' -> look at auc_conv_vs_stable and recall_Converted. Few converters (~15) => wide intervals.")
    print("\nPooled confusion matrices, RubricMarkov (balanced) (rows = truth, averaged over repeats)")
    for key in [("A", "RubricMarkov (balanced)"), ("B", "RubricMarkov (balanced)")]:
        print(key[0], "\n", pd.DataFrame((cms[key]/cfg.repeats if not a.QUICK else cms[key]).round().astype(int), index=CLASSES, columns=CLASSES))
    d_fit = observed_window(d, cutoff).reset_index(drop=True); assert_no_future(d_fit, cutoff, "descriptive fit")
    prep = Prep().fit(d_fit, extra); D = prep.tensors(d_fit)
    fit_act = MODES["C"] | set(extra)                         # descriptive full-data fit, NOT a performance estimate
    mdl = RubricMarkov(cfg.K, cfg.phases, extra=extra, discrete=cfg.discrete).fit(
        D, fit_act, restarts=2, seed=cfg.seed, maxiter=cfg.maxiter)
    if len(mdl.start_nll) > 1: print(f"full-data fit: NLL of the {len(mdl.start_nll)} starts = {np.round(mdl.start_nll, 3)}")
    summ = mdl.summary(prep); print("\nDescriptive full-data fit:\n", json.dumps(summ, indent=1))
    years = np.linspace(0, 10, 41); curves = mdl.progression_curves(years)
    pd.DataFrame(curves, columns=[f"P_stage{k}" for k in range(cfg.K)]).assign(years=years).to_csv(f"{out_dir}/progression_curves.csv", index=False)
    ft = forecast_table(mdl, prep, d_fit, MODES["C"] | set(extra), list(a.HORIZONS))
    ft.to_csv(f"{out_dir}/subject_forecasts.csv", index=False)
    print("\nper-subject forecasts saved (first rows):\n", ft.head(8).to_string(index=False))
    bt = bootstrap_hr(d_fit, cfg, extra, boot, cfg.seed, fit_act)
    ci = bt.quantile([0.025, 0.5, 0.975]).T.round(3); ci.columns = ["2.5%", "median", "97.5%"]
    print("\nBootstrap intervals (subject bootstrap):\n", ci.to_string()); ci.to_csv(f"{out_dir}/bootstrap_ci.csv")
    tab2, p_kw, pw, slopes = atrophy_tempo(d)                 # descriptive only (not a prediction) -> full data
    print("\nnWBV slope (% of baseline / yr):\n", tab2.round(3).to_string())
    print(f"Kruskal-Wallis p = {p_kw:.2e};", {k: f"{v:.2e}" for k, v in pw.items()})
    json.dump(dict(model=summ, kruskal_p=p_kw, pairwise_p=pw), open(f"{out_dir}/progression.json", "w"), indent=1)
    _plot_all(f"{out_dir}/results.png", cms, cfg, splits, curves, years, slopes,
              "retro" if cutoff is None else f"prospective {cutoff}y")
    print(f"\nDONE. Outputs in {out_dir}/")
    return tab


# =============================================================================== ablations
def cmd_ablate(a, cutoff=None):
    cfg0 = make_cfg(a); d0, extra0 = prepare(a); os.makedirs(a.OUT, exist_ok=True)
    variants = {"base": {}, "K=2": {"K": 2}, "K=4": {"K": 4}, "semi-Markov (2 phases)": {"phases": 2},
                "discrete-time": {"discrete": True}, "no weak CDR/MMSE supervision": {"sup": "matched"}}
    if extra0:
        variants["no image emissions"] = {"_vlm": False}
        variants["no Dice-QC weights"] = {"_qc": False}
        if any(e.startswith("vlm_") for e in extra0): variants["no Qwen rubrics"] = {"_qwen": False}
        if any(not e.startswith("vlm_") for e in extra0): variants["Qwen rubrics only"] = {"_geom": False}
    rows = []
    for name, v in variants.items():
        print(f"\n=== ablation: {name}")
        d, extra = d0.copy(), list(extra0)
        if v.get("_vlm") is False: extra = []
        if v.get("_qwen") is False: extra = [e for e in extra if not e.startswith("vlm_")]
        if v.get("_geom") is False: extra = [e for e in extra if e.startswith("vlm_")]
        if v.get("_qc") is False: d["qc_conf"] = 1.0
        cfg = replace(cfg0, **{k: x for k, x in v.items() if not k.startswith("_")})
        dd, oc = build_task(d, cutoff)
        res, _, _, _ = run_cv(dd, oc, cutoff, cfg, extra, log=lambda *_: None)
        s = res[res.method == "RubricMarkov (balanced)"].groupby("mode")[["bal_acc", "macro_f1", "auc_ovr", "ece", "recall_Converted"]].mean()
        s.insert(0, "variant", name); rows.append(s.reset_index()); print(s.round(3).to_string())
    tab = pd.concat(rows); tab.to_csv(f"{a.OUT}/ablations.csv", index=False)
    print("\nDONE. Saved", f"{a.OUT}/ablations.csv")


# =============================================================================== spread (needs torch + regional CSV)
def _torch():
    if not _HAS_TORCH: raise SystemExit("`spread` needs PyTorch (preinstalled on Kaggle/Colab).")
    return torch


def normalize_W(W):
    W = np.array(W, float); np.fill_diagonal(W, 0.0)
    return W / np.maximum(W.sum(1, keepdims=True), 1e-12)


def distance_kernel(cent):
    X = cent.values.astype(float); Dm = np.linalg.norm(X[:, None] - X[None], axis=-1)
    ell = np.median(Dm[Dm > 0]); return normalize_W(np.exp(-Dm / ell))


class Coupled:
    """Joint chain over R regions, K_r=3 states each (3^R joint states). Region r advances with rate
       exp(a_r + beta * sum_s W_rs h(x_s)) * exp(b*age + c.cov),  h(x)=x/(K-1).  W=None => beta fixed at 0."""
    def __init__(self, R, W, K=3, use_rub=False):
        torch = _torch(); self.t = torch; self.R, self.K, self.use_rub = R, K, use_rub
        self.states = np.array(list(itertools.product(range(K), repeat=R))); self.S = len(self.states)
        self.W = np.zeros((R, R)) if W is None else normalize_W(W); self.beta_free = W is not None
        src, dst, reg, cf = [], [], [], []
        h = self.states / (K - 1); radix = K ** (R - 1 - np.arange(R))
        for i, s in enumerate(self.states):
            for r in range(R):
                if s[r] < K - 1:
                    src.append(i); dst.append(i + radix[r]); reg.append(r)
                    cf.append(sum(self.W[r, q] * h[i, q] for q in range(R) if q != r))
        f64 = torch.float64
        self.src, self.dst, self.reg = (torch.tensor(x, dtype=torch.long, device=DEV) for x in (src, dst, reg))
        self.cf = torch.tensor(cf, dtype=f64, device=DEV)
        self.st = [torch.tensor(self.states[:, r], dtype=torch.long, device=DEV) for r in range(R)]

    def init_params(self, seed=0):
        t, R, K = self.t, self.R, self.K; f = t.float64; kw = dict(dtype=f, device=DEV)
        P = dict(a=t.full((R,), float(np.log(0.15)), **kw), beta=t.zeros((), **kw), qb=t.zeros((), **kw),
                 qc=t.zeros(3, **kw), pi_w=(-1.0 - 0.5*t.arange(K-1, **kw)).repeat(R, 1),
                 pi_u=t.zeros(R, K-1, **kw),
                 v_mu1=t.full((R,), 0.7, **kw), v_inc=t.zeros(R, K-1, **kw), v_ls=t.full((R, K), -0.2, **kw))
        if self.use_rub:
            P.update(r_mu1=t.full((R,), -0.7, **kw), r_inc=t.zeros(R, K-1, **kw), r_ls=t.full((R, K), -0.2, **kw))
        for k, v in P.items():
            if not (k == "beta" and not self.beta_free): v.requires_grad_(True)
        return P

    def _mu(self, mu1, inc, direction):
        t = self.t
        return mu1[:, None] + direction * t.cat([t.zeros(self.R, 1, dtype=mu1.dtype, device=DEV), t.cumsum(t.nn.functional.softplus(inc), 1)], 1)

    def loglik(self, P, D, vcols, rcols=()):
        t = self.t; f = t.float64; R, K, S = self.R, self.K, self.S
        T_ = lambda x: t.as_tensor(np.asarray(x), dtype=f, device=DEV)
        age, cov, dt, mask, age0 = T_(D["age"]), T_(D["cov"]), T_(D["dt"]), t.as_tensor(D["mask"], device=DEV), T_(D["age0"])
        B, T = age.shape
        L = t.zeros(B, T, R, K, dtype=f, device=DEV)
        mods = [("v", vcols, -1.0)] + ([("r", rcols, +1.0)] if self.use_rub else [])
        for pre, cols, dr in mods:
            X = t.stack([T_(D[c]) for c in cols], -1); ok = t.stack([T_(D["ok_"+c]) for c in cols], -1)
            mu = self._mu(P[pre+"_mu1"], P[pre+"_inc"], dr); sd = t.exp(P[pre+"_ls"].clamp(np.log(0.08), np.log(3.0)))
            z = (X[..., None] - mu) / sd
            L = L + (-0.5*z**2 - t.log(sd) - 0.9189385332046727) * ok[..., None]
        E = sum(L[:, :, r, :].index_select(2, self.st[r]) for r in range(R))
        lg = t.cat([t.zeros(B, R, 1, dtype=f, device=DEV), P["pi_w"][None] + P["pi_u"][None] * age0[:, None, None]], 2)
        lpi = t.log_softmax(lg, 2)
        logpi = sum(lpi[:, r, :].index_select(1, self.st[r]) for r in range(R))
        beta = P["beta"] if self.beta_free else P["beta"].detach()
        rates = t.exp(P["a"][self.reg] + beta * self.cf)
        G = t.zeros(S, S, dtype=f, device=DEV).index_put((self.src, self.dst), rates)
        G = G - t.diag(t.zeros(S, dtype=f, device=DEV).index_add(0, self.src, rates))
        m = t.exp(P["qb"] * age[:, :-1] + (cov @ P["qc"])[:, None])
        Pm = t.linalg.matrix_exp((m * dt)[..., None, None] * G).clamp_min(0.0) if T > 1 else None
        la = logpi + E[:, 0]; c = t.logsumexp(la, 1); alpha = t.exp(la - c[:, None]); ll = c
        for k in range(1, T):
            pred = t.bmm(alpha[:, None, :], Pm[:, k-1])[:, 0].clamp_min(1e-300)
            la = t.log(pred) + E[:, k]; c = t.logsumexp(la, 1); new = t.exp(la - c[:, None])
            mk = mask[:, k]; ll = ll + t.where(mk, c, t.zeros_like(c)); alpha = t.where(mk[:, None], new, alpha)
        return ll

    def fit(self, D, vcols, rcols=(), steps=150, lr=0.05, tau=1.0, l2=2.0, seed=0):
        t = self.t; P = self.init_params(seed)
        opt = t.optim.Adam([v for v in P.values() if v.requires_grad], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            pen = 0.5*l2*(P["qb"]**2 + (P["qc"]**2).sum() + (P["pi_u"]**2).sum())
            if self.beta_free: pen = pen + 0.5 * P["beta"]**2 / tau**2
            loss = -self.loglik(P, D, vcols, rcols).sum() + pen
            loss.backward(); opt.step()
        return P


def cmd_spread(a):
    torch = _torch(); torch.set_num_threads(max(1, os.cpu_count() or 1))
    assert a.REGIONAL_CSV, "spread needs CONFIG['REGIONAL_CSV'] (per-scan 'MRI ID', vol_<region> [mm^3], optional rub_<region>); " \
                           "RUN='mri' with REGIONS=True writes one to OUT/mri/regional.csv"
    regions = a.REGIONS_LIST.split(","); R = len(regions)
    if R > 6: raise SystemExit("R<=6 supported (3^R joint states); R=4 recommended on CPU")
    d = load_table(find_table(a.XLSX)); reg = pd.read_csv(a.REGIONAL_CSV).rename(columns={"MRI ID": "mri"})
    d = d.merge(reg, on="mri", how="left"); vcols = []; rcols = []
    for r in regions:
        d[f"rv_{r}"] = d[f"vol_{r}"] / d["eTIV"]; vcols.append(f"rv_{r}")
        if a.USE_RUBRICS and f"rub_{r}" in d: rcols.append(f"rub_{r}")
    use_rub = len(rcols) == R
    Wc = pd.read_csv(a.W_CSV, index_col=0).loc[regions, regions].values if a.W_CSV else None
    Wd = distance_kernel(pd.read_csv(a.CENTROIDS_CSV, index_col=0).loc[regions]) if a.CENTROIDS_CSV else None
    rng = np.random.default_rng(a.SEED); variants = {"none": [None]}
    if Wc is not None:
        variants["connectome"] = [Wc]
        variants["shuffled"] = [Wc[np.ix_(p, p)] for p in (rng.permutation(R) for _ in range(a.N_SHUFFLE))]
    if Wd is not None: variants["distance"] = [Wd]
    sub = d.groupby("sid", sort=False)["Group"].first(); sids = sub.index.values; y = sub.map(CLASSES.index).values
    cv = StratifiedKFold(n_splits=a.SPLITS, shuffle=True, random_state=a.SEED)
    ll_subj = {v: pd.Series(np.nan, index=sids) for v in variants}; rows = []; t0 = time.time()
    for fold, (tr, te) in enumerate(cv.split(sids, y)):
        dtr, dte = (d[d.sid.isin(sids[i])].reset_index(drop=True) for i in (tr, te))
        prep = Prep().fit(dtr, vcols + rcols); Dtr, Dte = prep.tensors(dtr), prep.tensors(dte)
        for v, Ws in variants.items():
            lls = []
            for Wm in Ws:
                m = Coupled(R, Wm, use_rub=use_rub)
                P = m.fit(Dtr, vcols, rcols, steps=a.SPREAD_STEPS, seed=a.SEED + fold)
                with torch.no_grad(): lls.append(m.loglik(P, Dte, vcols, rcols).cpu().numpy())
            ll = np.mean(lls, 0); ll_subj[v].loc[Dte["sids"]] = ll
            rows.append(dict(variant=v, fold=fold, test_ll_per_subject=float(ll.mean())))
        print(f"  fold {fold+1}/{a.SPLITS} done ({time.time()-t0:.0f}s elapsed)")
    res = pd.DataFrame(rows); os.makedirs(a.OUT, exist_ok=True); res.to_csv(f"{a.OUT}/spread_folds.csv", index=False)
    base = ll_subj["none"].values; out = []
    from scipy.stats import wilcoxon
    for v in variants:
        if v == "none": continue
        dl = ll_subj[v].values - base; bs = [rng.choice(dl, len(dl)).mean() for _ in range(2000)]
        pf = res[res.variant == v].sort_values("fold").test_ll_per_subject.values - res[res.variant == "none"].sort_values("fold").test_ll_per_subject.values
        try: pw = wilcoxon(pf).pvalue
        except Exception: pw = np.nan
        out.append(dict(variant=v, mean_dLL_vs_none=dl.mean(), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                        folds_better=int((pf > 0).sum()), n_folds=len(pf), wilcoxon_p=pw))
    summ = pd.DataFrame(out); print("\nHeld-out log-likelihood per subject, relative to beta=0 (higher = better):")
    print(summ.round(4).to_string(index=False)); summ.to_csv(f"{a.OUT}/spread_summary.csv", index=False)
    if "connectome" in variants and a.BETA_BOOT > 0:
        bs = []
        for b in range(a.BETA_BOOT):
            pick = rng.choice(sids, len(sids), replace=True)
            db = pd.concat([d[d.sid == s].assign(sid=f"{s}_{i}") for i, s in enumerate(pick)]).reset_index(drop=True)
            Db = Prep().fit(db, vcols + rcols).tensors(db)
            m = Coupled(R, Wc, use_rub=use_rub); P = m.fit(Db, vcols, rcols, steps=a.SPREAD_STEPS, seed=b)
            bs.append(float(P["beta"])); print(f"  beta boot {b+1}/{a.BETA_BOOT}: {bs[-1]:.3f}")
        print("beta (connectome) bootstrap 2.5/50/97.5%:", np.percentile(bs, [2.5, 50, 97.5]).round(3))
    print("\nNote: better fit with coupling is CONSISTENT WITH spread; it does not prove transneuronal propagation.")


# =============================================================================== synthetic data
def cmd_synth(a):
    rng = np.random.default_rng(a.SEED + 1); rows, regs = [], []; names = ["mtl", "frontal", "parietal", "vent"]
    for i in range(a.SYNTH_N):
        g = ["Nondemented", "Converted", "Demented"][i % 3]; age = rng.normal(75, 7); n = int(rng.integers(2, 5))
        etiv = rng.normal(1500, 100); gap = int(rng.integers(300, 700))      # one inter-visit interval per subject => strictly increasing
        for v in range(n):
            st = 1 + (v > 0) if g == "Demented" else (0 if (g == "Nondemented" or v < n-1) else 1)
            t = v * gap; mid = f"OAS2_{i:04d}_MR{v+1}"
            rows.append({"Subject ID": f"OAS2_{i:04d}", "MRI ID": mid, "Group": g, "Visit": v+1, "MR Delay": t,
                         "M/F": rng.choice(["M", "F"]), "Age": int(age + t/365.25), "EDUC": int(rng.integers(8, 20)),
                         "SES": rng.choice([1, 2, 3, 4, np.nan]), "CDR": [0, 0.5, 1][st], "eTIV": etiv,
                         "MMSE": np.nan if rng.random() < .1 else 30 - 4*st - rng.normal(0, 1),
                         "nWBV": np.nan if rng.random() < .05 else 0.78 - 0.03*st - 0.004*v + rng.normal(0, .01)})
            r = {"MRI ID": mid}
            for j, nm in enumerate(names):
                sr = min(2, max(0, st + int(rng.integers(-1, 2)) * (j > 0)))
                r[f"vol_{nm}"] = etiv * (0.006 - 0.0007*sr + rng.normal(0, 3e-4)); r[f"rub_{nm}"] = sr*0.9 + rng.normal(0, .3)
            regs.append(r)
    os.makedirs(a.OUT, exist_ok=True)
    df = pd.DataFrame(rows); df.to_excel(f"{a.OUT}/syn.xlsx", index=False)
    pd.DataFrame(regs).to_csv(f"{a.OUT}/syn_regional.csv", index=False)
    # image-emission CSV in the same format the "mri" step writes (higher = more atrophy)
    vf = pd.DataFrame({"MRI ID": df["MRI ID"], "qc_conf": np.clip(rng.beta(8, 2, len(df)), 0.05, 1),
                       "vent_icv": 0.02 + 0.012 * df["CDR"].map({0: 0, 0.5: 1, 1: 2}) + rng.normal(0, .004, len(df))})
    vf.to_csv(f"{a.OUT}/syn_mri_features.csv", index=False)
    Wm = rng.uniform(0.1, 1, (4, 4)); Wm = (Wm + Wm.T) / 2
    pd.DataFrame(Wm, index=names, columns=names).to_csv(f"{a.OUT}/syn_W.csv")
    pd.DataFrame(rng.normal(0, 30, (4, 3)), index=names, columns=list("xyz")).to_csv(f"{a.OUT}/syn_centroids.csv")
    print("wrote synthetic data to", a.OUT)


# =============================================================================== MRI step (GPU): raw repeats -> emissions + QC
# Per OAS2_xxxx_MRn session: RAS reorientation -> align repeats (FFT phase correlation, sign-verified; optional ANTs rigid) -> average
# -> brain mask (external SynthStrip / HD-BET mask from MASK_DIR, else automatic SynthStrip -> HD-BET (backup); the Otsu/erode/CC/
# dilate/close heuristic only when REQUIRE_MASK=False) -> bias-field correction (N4 or WM-anchored) -> MRF-regularised soft
# CSF/GM/WM GMM -> vent_icv / csf_icv / gm_deficit / bpf (HIGHER csf/vent/deficit = MORE ATROPHY) and qc_conf from
# {Dice of two half-average segmentations, leave-one-out NCC, brain-volume plausibility, tissue contrast}.
from scipy import ndimage as ndi
if _HAS_TORCH:
    import torch.nn.functional as F
    _ng = torch.no_grad()
else:
    F = None
    _ng = lambda f: f


def _nib():
    try:
        import nibabel as nib
        return nib
    except ImportError:
        raise SystemExit("`mri` needs nibabel (preinstalled on Kaggle; else: pip install nibabel)")


def _need_torch():
    if not _HAS_TORCH: raise SystemExit("this step needs a working PyTorch install")


def _try_ants():
    try:
        import ants
        return ants
    except ImportError:
        return None

# ------------------------------------------------------------------ io
def find_scans(roots):
    roots = [r for r in roots if os.path.isdir(r)] or glob.glob("/kaggle/input/**/OAS2_RAW_PART*", recursive=True)
    assert roots, "no OAS2_RAW_PART* directory found; set CONFIG['RAW_ROOTS']"
    scans = {}
    for r in roots:
        for d in sorted(glob.glob(f"{r}/OAS2_*_MR*")):
            imgs = sorted(glob.glob(f"{d}/RAW/mpr-*.nifti.img"))
            if imgs: scans[os.path.basename(d)] = imgs
    return scans


def _zooms_from_affine(aff):
    return tuple(float(z) for z in np.sqrt((np.asarray(aff, float)[:3, :3] ** 2).sum(0)))


def load_scan(paths, canonical=True):
    """-> (list of 3-D float32 repeats on a common grid, voxel sizes (mm) of that grid, affine of that grid).
    canonical=True reorients every repeat to the closest RAS orientation (voxel sizes / affine follow)."""
    nib = _nib(); vols, aff, hz = [], None, None
    for p in paths:
        try:
            im = nib.load(p); v = np.asarray(im.dataobj, dtype=np.float32).squeeze()
            if v.ndim != 3: continue
            A, hdr_z = im.affine, tuple(float(z) for z in im.header.get_zooms()[:3])
            if canonical:
                c = nib.as_closest_canonical(nib.Nifti1Image(v, A)); v = np.asarray(c.dataobj, dtype=np.float32); A = c.affine
            if vols and v.shape != vols[0].shape: continue
            vols.append(np.ascontiguousarray(v, dtype=np.float32))     # as_closest_canonical returns flipped views (negative strides)
            if aff is None: aff, hz = A, hdr_z
        except Exception as e:
            print(f"   [warn] cannot read {p}: {e}")
    if not vols: return [], None, None
    zooms = _zooms_from_affine(aff)
    if not all(np.isfinite(z) and z > 0.1 for z in zooms):          # degenerate affine -> fall back to header voxel sizes
        zooms = hz
    return vols, zooms, aff


def find_mask(mask_dir, name):
    """External brain mask lookup (SynthStrip / HD-BET naming conventions)."""
    if not mask_dir: return None
    for suf in (".nii.gz", "_mask.nii.gz", "_bet_mask.nii.gz", "_brain_mask.nii.gz"):
        p = f"{mask_dir}/{name}{suf}"
        if os.path.exists(p): return p
    return None


def load_mask(path, canonical=True):
    """binary brain mask as a numpy bool array, in the same (RAS) grid as load_scan(canonical=...)."""
    nib = _nib(); im = nib.load(path); v = np.asarray(im.dataobj, dtype=np.float32).squeeze()
    if canonical:
        v = np.asarray(nib.as_closest_canonical(nib.Nifti1Image(v, im.affine)).dataobj, dtype=np.float32)
    return np.ascontiguousarray(v > 0)


# ------------------------------------------------------------------ automatic skull-stripping: SynthStrip (primary) -> HD-BET (backup)
_STRIP = {"tool": None, "queue": [], "fail": 0, "weights": None}
SYNTHSTRIP_SHA_PREFIX = "37417f80"       # sha256 prefix of the official synthstrip.1.pt (with-CSF model)
SYNTHSTRIP_URLS = [                      # official FreeSurfer distribution (synthstrip.1.pt, ~29 MB). NOTE: nipreps-synthstrip ships NO weights.
    "https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/synthstrip/models/synthstrip.1.pt",
]


def _detect_strippers(method):
    """installed skull-strippers: SynthStrip first, HD-BET second."""
    import shutil
    found = []
    if method in ("auto", "synthstrip"):
        for exe in ("mri_synthstrip", "nipreps-synthstrip"):
            p = shutil.which(exe)
            if p: found.append(("synthstrip", p)); break
    if method in ("auto", "hdbet"):
        p = shutil.which("hd-bet")
        if p: found.append(("hdbet", p))
    return found


def _synthstrip_weights(cfg):
    """Path to a valid synthstrip.1.pt: SYNTHSTRIP_MODEL -> $FREESURFER_HOME/models -> cache -> download (browser UA, atomic, sha256 logged).
    Returns None (with the real reason printed) if unavailable.  The with-CSF model is required (CSF measures depend on it)."""
    import hashlib, shutil, urllib.request, urllib.error
    ok = lambda p: bool(p) and os.path.exists(p) and os.path.getsize(p) > 20e6
    p = cfg.get("SYNTHSTRIP_MODEL")
    if p:
        if ok(p): return p
        print(f"   [warn] SYNTHSTRIP_MODEL={p!r} not found or too small")
    fs = os.environ.get("FREESURFER_HOME")
    if fs and ok(f"{fs}/models/synthstrip.1.pt"): return f"{fs}/models/synthstrip.1.pt"
    cp = f"{cfg['OUT']}/mri/synthstrip.1.pt"
    if ok(cp): return cp
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    for url in SYNTHSTRIP_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"})
            with urllib.request.urlopen(req, timeout=180) as r, open(cp + ".part", "wb") as f: shutil.copyfileobj(r, f)
            if not ok(cp + ".part"): raise RuntimeError(f"downloaded file too small ({os.path.getsize(cp + '.part')} bytes)")
            h = hashlib.sha256(open(cp + ".part", "rb").read()).hexdigest()
            print(f"[strip] SynthStrip weights downloaded ({os.path.getsize(cp + '.part') / 1e6:.1f} MB, sha256 {h[:12]}...) " +
                  ("= official" if h.startswith(SYNTHSTRIP_SHA_PREFIX) else "!= expected prefix " + SYNTHSTRIP_SHA_PREFIX + " (check the file)"))
            os.replace(cp + ".part", cp); return cp
        except urllib.error.HTTPError as e: print(f"   [warn] weights download failed: HTTP {e.code} {e.reason} ({url})")
        except Exception as e: print(f"   [warn] weights download failed: {type(e).__name__}: {e} ({url})")
    return None


def _setup_stripper(a):
    """Decide ONCE, before any scan, which skull-stripper is used for the WHOLE dataset (one tool => consistent masks).
    SynthStrip needs its weights; if they cannot be obtained it is not used (no failing scans) and HD-BET is tried instead."""
    import subprocess, importlib
    _STRIP.update(tool=None, queue=[], fail=0, weights=None)
    if a.SKULLSTRIP == "heuristic": return
    cfg = vars(a)

    def pip(pkg):
        print(f"[strip] installing {pkg} ...")
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], capture_output=True, text=True)
        if r.returncode != 0: print(f"   [warn] pip install {pkg} failed: {r.stderr[-300:]}")
        importlib.invalidate_caches()

    queue = []
    if a.SKULLSTRIP in ("auto", "synthstrip"):
        if not _detect_strippers("synthstrip") and a.AUTO_INSTALL: pip("nipreps-synthstrip")
        ss = _detect_strippers("synthstrip")
        if ss:
            w = _synthstrip_weights(cfg)
            if w: _STRIP["weights"] = w; queue.append(ss[0]); print(f"[strip] SynthStrip ready: {ss[0][1]} | weights {w}")
            elif os.path.basename(ss[0][1]) == "mri_synthstrip" and os.environ.get("FREESURFER_HOME"): queue.append(ss[0])
            else: print("   [warn] SynthStrip installed but its weights are unavailable -> not used. Attach synthstrip.1.pt as a Kaggle dataset and set SYNTHSTRIP_MODEL.")
    if a.SKULLSTRIP == "hdbet" or (a.SKULLSTRIP == "auto" and not queue):
        if not _detect_strippers("hdbet") and a.AUTO_INSTALL: pip("hd-bet")
        queue += _detect_strippers("hdbet")
        if queue: print("[strip] using HD-BET for the whole dataset (SynthStrip unavailable)")
    _STRIP.update(queue=queue, tool=queue[0] if queue else None)


def auto_mask(name, mean, aff, cfg):
    """Run the active skull-stripper on the aligned mean -> (torch bool mask on the same grid, tool kind) or (None, None).
    Masks are cached in OUT/mri/masks/<MRI ID>.nii.gz.  3 consecutive failures disable the tool and STOP the run
    (no mid-run switch to another tool: that would mix mask styles within one dataset)."""
    tool = _STRIP.get("tool")
    if tool is None: return None, None
    kind, exe = tool; nib = _nib()
    mdir = f"{cfg['OUT']}/mri/masks"; os.makedirs(mdir, exist_ok=True); mp = f"{mdir}/{name}.nii.gz"
    try:
        if not os.path.exists(mp):
            import subprocess, tempfile, shutil
            with tempfile.TemporaryDirectory() as td:
                ip = f"{td}/{name}.nii.gz"
                nib.save(nib.Nifti1Image(mean.cpu().numpy().astype(np.float32), aff), ip)
                if kind == "synthstrip":
                    base = [exe, "-i", ip, "-o", f"{td}/{name}_ss.nii.gz", "-m", mp]   # default (with-CSF) model keeps a CSF border: what csf/vent/bpf need
                    if _STRIP.get("weights"): base += ["--model", _STRIP["weights"]]    # REQUIRED by nipreps-synthstrip
                    runs = ([base + ["-g"]] if DEV.type == "cuda" else []) + [base]
                else:
                    op = f"{td}/{name}_bet.nii.gz"; dv = "cuda" if DEV.type == "cuda" else "cpu"
                    runs = [[exe, "-i", ip, "-o", op, "-device", dv, "--disable_tta"],
                            [exe, "-i", ip, "-o", op, "-device", dv, "-tta", "0"], [exe, "-i", ip, "-o", op]]
                ok, r = False, None
                for cmd in runs:
                    if os.path.exists(mp): os.remove(mp)
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg["STRIP_TIMEOUT"])
                    if kind == "synthstrip": ok = (r.returncode == 0 and os.path.exists(mp))
                    else:
                        cands = sorted(glob.glob(f"{td}/{name}_bet*mask*.nii.gz")); ok = (r.returncode == 0 and bool(cands))
                        if ok: shutil.copy(cands[0], mp)
                    if ok: break
                if not ok: raise RuntimeError(((r.stderr or r.stdout) if r is not None else "")[-400:])
        m = load_mask(mp, False)
        if m.sum() < 10000: raise RuntimeError(f"mask too small ({int(m.sum())} voxels)")
        _STRIP["fail"] = 0
        return torch.from_numpy(m), kind
    except Exception as e:
        _STRIP["fail"] += 1
        if os.path.exists(mp): os.remove(mp)                                   # never cache a bad mask
        print(f"   [warn] {kind} failed on {name}: {type(e).__name__}: {str(e)[-400:]}")
        if _STRIP["fail"] >= 3:
            print(f"   [warn] {kind} failed 3 times in a row -> stopping (no mid-run tool switch; fix it and rerun, finished scans are cached)")
            _STRIP["tool"] = None
        return None, None


# ------------------------------------------------------------------ gpu primitives
def gauss(x, sigma):
    r = max(1, int(np.ceil(3 * sigma))); t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2); k = k / k.sum(); y = x[None, None]
    for ax in range(3):
        shp = [1, 1, 1, 1, 1]; shp[2 + ax] = -1; pad = [0, 0, 0, 0, 0, 0]; pad[2 * (2 - ax) : 2 * (2 - ax) + 2] = [r, r]
        y = F.conv3d(F.pad(y, pad, mode="replicate"), k.view(shp))
    return y[0, 0]


def otsu(x, bins=256):
    lo, hi = float(x.min()), float(x.max())
    p = torch.histc(x, bins=bins, min=lo, max=hi); p = p / p.sum()
    w = torch.cumsum(p, 0); mu = torch.cumsum(p * torch.arange(bins, device=x.device), 0)
    sb = (mu[-1] * w - mu) ** 2 / (w * (1 - w) + 1e-12)
    return lo + (int(torch.argmax(sb)) + 0.5) * (hi - lo) / bins


def phase_shift(ref, mov, max_shift):
    """Integer shift s (per axis) such that torch.roll(mov, shifts=s) best matches ref.
    (R = F(ref) * conj(F(mov)); the peak of irfft(R) sits at +d when ref = roll(mov, d).)"""
    R = torch.fft.rfftn(ref) * torch.fft.rfftn(mov).conj(); R = R / (R.abs() + 1e-6)
    r = torch.fft.irfftn(R, s=ref.shape); idx = int(torch.argmax(r)); sh = []
    for n in reversed(ref.shape): sh.append(idx % n); idx //= n
    sh = [s - n if s > n // 2 else s for s, n in zip(sh[::-1], ref.shape)]
    return sh if max(abs(s) for s in sh) <= max_shift else [0, 0, 0]


_ALIGN_TESTED = False


def _selftest_align():
    """Roll a smooth random volume by a known shift and check that phase_shift + torch.roll(+s) recovers it exactly."""
    global _ALIGN_TESTED
    if _ALIGN_TESTED: return
    g = torch.Generator().manual_seed(0)
    v = gauss(torch.randn(32, 32, 32, generator=g, dtype=torch.float32), 1.5)
    mov = torch.roll(v, shifts=(-3, 2, -1), dims=(0, 1, 2))
    s = phase_shift(v, mov, 8)
    back = torch.roll(mov, shifts=tuple(s), dims=(0, 1, 2))
    assert torch.allclose(back, v, atol=1e-4), f"alignment sign self-test FAILED (recovered shift {s}, expected [3,-2,1])"
    _ALIGN_TESTED = True
    print(f"[align] phase-correlation sign self-test passed (recovered shift {s})")


def ncc(a, b, m):
    a, b = a[m], b[m]; a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-8))


def erode(m, r): return -F.max_pool3d(-m.float()[None, None], 2 * r + 1, 1, r)[0, 0] > 0.5
def dilate(m, r): return F.max_pool3d(m.float()[None, None], 2 * r + 1, 1, r)[0, 0] > 0.5


def _clean_mask(m):
    """largest connected component + hole filling (applied to every mask, external or heuristic)."""
    a = m.cpu().numpy().astype(bool)
    lab, n = ndi.label(a)
    if n == 0: return m
    sz = np.bincount(lab.ravel()); sz[0] = 0
    return torch.from_numpy(ndi.binary_fill_holes(lab == sz.argmax())).to(m.device)


def brain_mask(mean, r, dev):
    """HEURISTIC fallback mask (Otsu + erosion + largest component + closing). Not a validated skull-stripper:
    with REQUIRE_MASK=True it is used ONLY as a reference to audit the real mask (mask_dice_vs_heuristic / mask_suspect)."""
    sm = gauss(mean, 1.0); t = otsu(sm.flatten()); head = sm > t
    lab, n = ndi.label(erode(head, r).cpu().numpy())
    if n == 0: return None
    sz = np.bincount(lab.ravel()); sz[0] = 0
    m = dilate(torch.from_numpy(lab == sz.argmax()).to(dev), r + 1)
    m = erode(dilate(m, 3), 3)                                  # closing: re-includes sulcal CSF at the rim
    return torch.from_numpy(ndi.binary_fill_holes(m.cpu().numpy())).to(dev)


def gmm1d(x, mu0, iters=30):
    mu = mu0.clone(); var = torch.full_like(mu, float(x.var()) / 9); pi = torch.full_like(mu, 1 / len(mu))
    for _ in range(iters):
        ll = -0.5 * (x[:, None] - mu) ** 2 / var - 0.5 * torch.log(var) + torch.log(pi)
        r = torch.softmax(ll, 1); nk = r.sum(0) + 1e-6
        mu = (r * x[:, None]).sum(0) / nk; var = (r * (x[:, None] - mu) ** 2).sum(0) / nk + 1e-5; pi = nk / nk.sum()
    o = torch.argsort(mu); return mu[o], var[o], pi[o]


def segment(img, brain, seed=0, mrf=0.0, em=1, mf_iters=3, params=False):
    """soft CSF/GM/WM posteriors for brain voxels (N,3) ordered like img[brain].
    mrf=0, em=1 -> plain 3-class GMM (used for the fast QC half-average segmentations).
    mrf>0 -> EM with a mean-field Potts prior over the 26-neighbourhood: logit_k = loglik_k + mrf * (mean posterior_k of the
    in-mask neighbours), then the Gaussians are re-estimated from the regularised posteriors (em rounds).
    params=True also returns (mu, var) of the last E-step (used for the tissue-contrast QC)."""
    v = gauss(img, 0.7)[brain]; g = torch.Generator(device=v.device).manual_seed(seed)
    sub = v[torch.randperm(len(v), device=v.device, generator=g)[:400000]]
    mu, var, pi = gmm1d(sub, torch.quantile(sub, torch.tensor([0.10, 0.45, 0.85], device=v.device)))
    if mrf > 0:
        bf = brain.float()
        cnt_b = ((F.avg_pool3d(bf[None, None], 3, 1, 1)[0, 0] * 27.0 - bf)[brain]).clamp_min(1.0)
    for it in range(max(1, em)):
        ll = -0.5 * (v[:, None] - mu) ** 2 / var - 0.5 * torch.log(var) + torch.log(pi)
        post = torch.softmax(ll, 1)
        if mrf > 0:
            for _ in range(mf_iters):
                dense = torch.zeros((3,) + tuple(brain.shape), dtype=v.dtype, device=v.device); dense[:, brain] = post.T
                nb = ((F.avg_pool3d(dense[None], 3, 1, 1)[0] * 27.0 - dense)[:, brain]) / cnt_b
                post = torch.softmax(ll + mrf * nb.T, 1)
        if it < max(1, em) - 1:
            nk = post.sum(0) + 1e-6
            mu = (post * v[:, None]).sum(0) / nk; var = (post * (v[:, None] - mu) ** 2).sum(0) / nk + 1e-5; pi = nk / nk.sum()
    return (post, (mu, var)) if params else post


def dice_macro(la, lb):
    d = []
    for k in range(3):
        a, b = la == k, lb == k; s = a.sum() + b.sum()
        d.append(float(2 * (a & b).sum() / s) if s > 0 else np.nan)
    return float(np.nanmean(d))


def dice_bin(a, b):
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s > 0 else float("nan")


def _rigid_refine(ants, ref, mov, aff):
    """ANTs rigid refinement of an (already integer-aligned) repeat; accepted only if it increases NCC to the reference."""
    try:
        fa, ma = _ants_img(ants, ref.cpu().numpy(), aff), _ants_img(ants, mov.cpu().numpy(), aff)
        r = ants.registration(fixed=fa, moving=ma, type_of_transform="Rigid", verbose=False)
        new = torch.from_numpy(np.ascontiguousarray(r["warpedmovout"].numpy())).to(ref.device).float()
        if new.shape != ref.shape: return mov
        m = ref > ref.mean()
        return new if ncc(ref, new, m) > ncc(ref, mov, m) else mov
    except Exception as e:
        print(f"   [warn] rigid refinement failed ({type(e).__name__}); keeping the phase-aligned repeat")
        return mov


@_ng
def align_mean(vols, max_shift, dev=None, mode="phase", aff=None):
    """FFT phase-correlation alignment of repeats to repeat 0 (shift sign verified by a self-test), optional ANTs rigid refinement
    -> (aligned list, mean, shifts). Mean lives on repeat-0's grid."""
    dev = dev or DEV; _selftest_align(); X = []
    for v in vols:
        x = torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)).to(dev); X.append(x / torch.quantile(x.flatten()[::7], 0.99).clamp_min(1e-6))
    ants = _try_ants() if (mode == "rigid" and aff is not None and len(X) > 1) else None
    if mode == "rigid" and ants is None and len(X) > 1: print("   [warn] ALIGN='rigid' needs antspyx and the scan affine; using phase correlation only")
    shifts = [[0, 0, 0]]
    for i in range(1, len(X)):
        s = phase_shift(X[0], X[i], max_shift); shifts.append(s)
        X[i] = torch.roll(X[i], shifts=tuple(s), dims=(0, 1, 2))          # +s: moves repeat i onto repeat 0
        if ants is not None: X[i] = _rigid_refine(ants, X[0], X[i], aff)
    return X, torch.stack(X).mean(0), shifts


@_ng
def _field_wm(img, brain, sigma_vox, iters=3):
    """WM-anchored multiplicative bias estimate (GPU fallback): iteratively fit a smooth field to log(WM-like voxels / WM mean)
    with normalised convolution, so ventricles/GM do not drive the field."""
    field = torch.ones_like(img)
    for _ in range(iters):
        cur = img / field; v = cur[brain]; sub = v[:: max(1, len(v) // 200000)]
        mu, var, pi = gmm1d(sub, torch.quantile(sub, torch.tensor([0.10, 0.45, 0.85], device=img.device)))
        wm = brain & (cur > 0.5 * (mu[1] + mu[2]))
        if int(wm.sum()) < 5000: break
        r = torch.log(cur.clamp_min(1e-6)) - torch.log(mu[2])
        num, den = gauss(r * wm, sigma_vox), gauss(wm.float(), sigma_vox)
        field = field * torch.exp(torch.where(den > 1e-3, num / den.clamp_min(1e-3), torch.zeros_like(num)))
        field = field.clamp(0.5, 2.0)
        field = field / torch.exp(torch.log(field[brain]).mean())
    return field


@_ng
def bias_correct(mean, brain, zooms, mode="auto"):
    """-> (corrected image, multiplicative field). Field is reused for the half-average QC segmentations."""
    if mode == "none": return mean, torch.ones_like(mean)
    if mode in ("auto", "n4"):
        try:
            import SimpleITK as sitk
            im = sitk.GetImageFromArray(mean.cpu().numpy().astype(np.float32)); im.SetSpacing(tuple(float(z) for z in zooms[::-1]))
            mk = sitk.GetImageFromArray(brain.cpu().numpy().astype(np.uint8)); mk.CopyInformation(im)
            c = sitk.N4BiasFieldCorrectionImageFilter(); c.SetMaximumNumberOfIterations([40, 30, 20])
            c.Execute(sitk.Shrink(im, [4, 4, 4]), sitk.Shrink(mk, [4, 4, 4]))
            fld = torch.from_numpy(sitk.GetArrayFromImage(sitk.Exp(c.GetLogBiasFieldAsImage(im)))).to(mean.device).float()
            fld = fld.clamp(0.25, 4.0); fld = fld / torch.exp(torch.log(fld[brain]).mean())
            return mean / fld, fld
        except Exception as e:
            if mode == "n4": print(f"   [warn] N4 failed ({type(e).__name__}); using WM-anchored estimate")
    fld = _field_wm(mean, brain, 15.0 / float(np.mean(zooms)))
    return mean / fld, fld


def _vent_components(brain_np, csf_np, zooms, cfg):
    """Ventricle proxy v2 (no atlas): hard-CSF voxels that (i) lie >= VENT_MIN_DEPTH_MM inside the brain mask (excludes sulcal /
    cisternal CSF connected to the outside), (ii) fall in the central box (|u-0.5|<0.3 of the brain bbox) and (iii) belong to a
    connected component >= VENT_MIN_ML. -> (bool mask, number of kept components)."""
    dist = ndi.distance_transform_edt(brain_np, sampling=zooms)
    idx = np.argwhere(brain_np); lo, hi = idx.min(0), idx.max(0); c = (lo + hi) / 2.0; half = 0.3 * (hi - lo)
    box = np.zeros(brain_np.shape, bool)
    box[tuple(slice(int(max(0, np.floor(c[i] - half[i]))), int(np.ceil(c[i] + half[i])) + 1) for i in range(3))] = True
    cand = csf_np & box & (dist >= cfg["VENT_MIN_DEPTH_MM"])
    lab, n = ndi.label(cand, structure=ndi.generate_binary_structure(3, 2))
    if n == 0: return cand, 0
    ml = np.bincount(lab.ravel()).astype(float) * float(np.prod(zooms)) / 1000.0; ml[0] = 0
    keep = np.where(ml >= cfg["VENT_MIN_ML"])[0]
    return np.isin(lab, keep), int(len(keep))


# ------------------------------------------------------------------ Harvard-Oxford atlas: the 112-ROI parcellation, NO fallback
# 112 ROIs = 96 cortical (48 Harvard-Oxford cortical regions x Left/Right; the lateralised 'cortl' maxprob thr25 1 mm atlas)
#          + 16 subcortical (lateral ventricle, thalamus, caudate, putamen, pallidum, hippocampus, amygdala, accumbens x L/R;
#            the sub atlas' cerebral white matter / cerebral cortex / brainstem labels are NOT used).
# The skull-stripped, bias-corrected session mean (FIXED) is registered with ANTs SyN (affine + deformable) to an MNI152 T1 brain
# (MOVING); the same transforms carry the atlas (generic-label interpolation) into the subject grid.  Nothing else is attempted:
# a missing dependency / missing NIfTI affine / unreadable or mismatching atlas STOPS the run.  A scan is used only if
# Dice(warped MNI brain mask, subject brain mask) >= ATLAS_MIN_DICE AND the warped lateral-ventricle label is mostly CSF
# (>= ATLAS_MIN_VENT_CSF), otherwise its regional values stay missing (logged in mri_qc.csv).
# Per-ROI value vol_<ROI> [mm^3]: cortical ROIs = grey-matter volume; subcortical ROIs = non-CSF (GM+WM) volume;
# lateral ventricles = CSF volume (this one GROWS with atrophy).  Lobar aggregates (for RUN='spread') are sums of these ROI values.
_ATLAS = {}
HO_CORT_BASE = [    # Harvard-Oxford cortical labels 1..48 (the lateralised atlas stores them as 2i-1 = Left, 2i = Right)
    "Frontal Pole", "Insular Cortex", "Superior Frontal Gyrus", "Middle Frontal Gyrus", "Inferior Frontal Gyrus, pars triangularis",
    "Inferior Frontal Gyrus, pars opercularis", "Precentral Gyrus", "Temporal Pole", "Superior Temporal Gyrus, anterior division",
    "Superior Temporal Gyrus, posterior division", "Middle Temporal Gyrus, anterior division", "Middle Temporal Gyrus, posterior division",
    "Middle Temporal Gyrus, temporooccipital part", "Inferior Temporal Gyrus, anterior division", "Inferior Temporal Gyrus, posterior division",
    "Inferior Temporal Gyrus, temporooccipital part", "Postcentral Gyrus", "Superior Parietal Lobule", "Supramarginal Gyrus, anterior division",
    "Supramarginal Gyrus, posterior division", "Angular Gyrus", "Lateral Occipital Cortex, superior division",
    "Lateral Occipital Cortex, inferior division", "Intracalcarine Cortex", "Frontal Medial Cortex",
    "Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)", "Subcallosal Cortex", "Paracingulate Gyrus",
    "Cingulate Gyrus, anterior division", "Cingulate Gyrus, posterior division", "Precuneous Cortex", "Cuneal Cortex",
    "Frontal Orbital Cortex", "Parahippocampal Gyrus, anterior division", "Parahippocampal Gyrus, posterior division", "Lingual Gyrus",
    "Temporal Fusiform Cortex, anterior division", "Temporal Fusiform Cortex, posterior division", "Temporal Occipital Fusiform Cortex",
    "Occipital Fusiform Gyrus", "Frontal Opercular Cortex", "Central Opercular Cortex", "Parietal Opercular Cortex", "Planum Polare",
    "Heschl's Gyrus (includes H1 and H2)", "Planum Temporale", "Supracalcarine Cortex", "Occipital Pole"]
HO_SUB_BASE = [     # (name, value in LEFT, value in RIGHT, kind) in the Harvard-Oxford subcortical maxprob atlas
    ("Lateral_Ventricle", 3, 14, "vent"), ("Thalamus", 4, 15, "sub"), ("Caudate", 5, 16, "sub"), ("Putamen", 6, 17, "sub"),
    ("Pallidum", 7, 18, "sub"), ("Hippocampus", 9, 19, "sub"), ("Amygdala", 10, 20, "sub"), ("Accumbens", 11, 21, "sub")]


def _san(s): return re.sub(r"[^0-9A-Za-z]+", "_", s).strip("_")
def _alnum(s): return re.sub(r"[^0-9a-z]", "", str(s).lower())


# (column name, kind, atlas value, base key): kind in {"ctx" (value in cortl), "sub", "vent" (value in sub atlas)}
HO_ROIS = [(f"{'L' if v % 2 else 'R'}_{_san(HO_CORT_BASE[(v + 1) // 2 - 1])}", "ctx", v, (v + 1) // 2) for v in range(1, 97)]
for _n, _l, _r, _k in HO_SUB_BASE:
    HO_ROIS += [(f"L_{_n}", _k, _l, _n), (f"R_{_n}", _k, _r, _n)]
assert len(HO_ROIS) == 112 and len({r[0] for r in HO_ROIS}) == 112
# lobar aggregates (sums of ROI values): region -> (cortical base indices, subcortical base names)
HO_AGG = {
    "mtl":       ([34, 35], ["Hippocampus", "Amygdala"]),                          # parahippocampal ant/post + hippocampus + amygdala (L+R)
    "frontal":   ([1, 3, 4, 5, 6, 7, 25, 26, 28, 33, 41], []),
    "parietal":  ([17, 18, 19, 20, 21, 31, 43], []),
    "temporal":  ([8, 9, 10, 11, 12, 13, 14, 15, 16, 37, 38, 39, 44, 45, 46], []),
    "occipital": ([22, 23, 24, 32, 36, 40, 47, 48], []),
    "cingulate": ([29, 30], []),
}
HO_VENTRICLE_SUB = [3, 14]                                                         # Left / Right Lateral Ventricle (sub atlas values)


def _geom(aff):
    """NIfTI voxel->RAS affine -> (spacing, origin, direction) in ITK/ANTs LPS convention."""
    L = np.diag([-1.0, -1.0, 1.0, 1.0]) @ np.asarray(aff, float)
    sp = np.linalg.norm(L[:3, :3], axis=0)
    return sp, L[:3, 3], L[:3, :3] / sp


def _ants_img(ants, arr, aff):
    sp, org, dr = _geom(aff)
    return ants.from_numpy(np.ascontiguousarray(arr, dtype=np.float32), origin=[float(x) for x in org],
                           spacing=[float(x) for x in sp], direction=dr)


def _require_ants():
    try:
        import ants
        return ants
    except ImportError:
        raise SystemExit("atlas regions need ANTsPy for SyN registration: pip install antspyx (no fallback registration is used)")


def _check_atlas(ic, is_, lab_c, lab_s):
    """Hard checks that the files really are the lateralised cortical + subcortical Harvard-Oxford atlases with the assumed values."""
    dc = np.rint(ic.get_fdata()).astype(np.int32); ds = np.rint(is_.get_fdata()).astype(np.int32)
    if set(np.unique(dc)) != set(range(0, 97)):
        raise SystemExit("cortical atlas must be the LATERALISED maxprob atlas (HarvardOxford-cortl-maxprob-thr25-1mm: values 0..96)")
    if not set(range(0, 22)) >= set(np.unique(ds)): raise SystemExit("subcortical atlas must have values 0..21 (HarvardOxford-sub-maxprob-thr25-1mm)")
    def side_x(img, data, vals):
        cm = ndi.center_of_mass(np.ones(data.shape, np.float32), data, vals)
        return {v: float((img.affine @ np.array([*c, 1.0]))[0]) for v, c in zip(vals, cm)}
    xc = side_x(ic, dc, list(range(1, 97)))
    for v, x in xc.items():
        if (v % 2 == 1 and x > 3) or (v % 2 == 0 and x < -3):
            raise SystemExit(f"cortl value {v}: centroid x={x:.1f} mm contradicts the Left(odd)/Right(even) convention")
    xs = side_x(is_, ds, [b for _, l, r, _ in HO_SUB_BASE for b in (l, r)])
    for _, l, r, _ in HO_SUB_BASE:
        if xs[l] > 3 or xs[r] < -3: raise SystemExit(f"subcortical values {l}/{r}: centroid sides contradict Left/Right")
    if lab_c is not None:                                                         # name check when nilearn labels are available
        for v in range(1, 97):
            n = _alnum(lab_c[v]); want = "left" if v % 2 else "right"
            if not n.startswith(want) or _alnum(HO_CORT_BASE[(v + 1) // 2 - 1])[:12] not in n:
                raise SystemExit(f"cortl label {v} is {lab_c[v]!r}: does not match HO_CORT_BASE / lateralisation")
    if lab_s is not None:
        for nm, l, r, _ in HO_SUB_BASE:
            for v, w in ((l, "left"), (r, "right")):
                n = _alnum(lab_s[v])
                if not n.startswith(w) or _alnum(nm) not in n: raise SystemExit(f"sub label {v} is {lab_s[v]!r}, expected {w} {nm}")


def _load_atlas(cfg):
    if _ATLAS: return _ATLAS
    nib = _nib(); ad = cfg.get("ATLAS_DIR"); lab_c = lab_s = None
    asimg = lambda x: nib.load(str(x)) if isinstance(x, (str, os.PathLike)) else x
    if ad:
        ic = nib.load(f"{ad}/HarvardOxford-cortl-maxprob-thr25-1mm.nii.gz"); is_ = nib.load(f"{ad}/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz")
    else:
        try: from nilearn import datasets as nd
        except ImportError: raise SystemExit("atlas regions need nilearn (pip install nilearn) or CONFIG['ATLAS_DIR'] with the FSL HarvardOxford-cortl-maxprob-thr25-1mm + HarvardOxford-sub-maxprob-thr25-1mm files")
        ac, asub = nd.fetch_atlas_harvard_oxford("cortl-maxprob-thr25-1mm"), nd.fetch_atlas_harvard_oxford("sub-maxprob-thr25-1mm")
        ic, is_ = asimg(ac["maps"]), asimg(asub["maps"]); lab_c, lab_s = list(ac["labels"]), list(asub["labels"])
    if not (ic.shape[:3] == is_.shape[:3] and np.allclose(ic.affine, is_.affine, atol=1e-3)): raise SystemExit("cortical / subcortical atlas grids differ")
    _check_atlas(ic, is_, lab_c, lab_s)
    code = (np.rint(ic.get_fdata()).astype(np.int32) + 1000 * np.rint(is_.get_fdata()).astype(np.int32)).astype(np.int32)   # cortl + 1000*sub
    tp = cfg.get("MNI_TEMPLATE")
    if tp:
        t = nib.load(tp); tarr = np.asarray(t.get_fdata(), np.float32); taff = t.affine; tmask = tarr > 0
    else:
        try: from nilearn import datasets as nd
        except ImportError: raise SystemExit("needs nilearn for the MNI template, or set CONFIG['MNI_TEMPLATE'] to a skull-stripped MNI152 T1 1mm brain")
        try: t, m = nd.load_mni152_template(resolution=1), nd.load_mni152_brain_mask(resolution=1)
        except TypeError: t, m = nd.load_mni152_template(), nd.load_mni152_brain_mask()
        tmask = np.asarray(m.get_fdata()) > 0; tarr = np.asarray(t.get_fdata(), np.float32); taff = t.affine
    _ATLAS.update(code=(code, ic.affine), tmpl=((tarr * tmask).astype(np.float32), taff), tmask=(tmask.astype(np.uint8), taff))
    return _ATLAS


def atlas_labels(img, brain, aff, cfg):
    """SyN-warped Harvard-Oxford labels in this scan's voxel grid -> dict(ok, dice, cort, sub); cort (0..96) / sub (0..21) are full-grid torch long tensors."""
    if aff is None: raise SystemExit("atlas regions need the NIfTI affine of the raw scans (header orientation); none was found")
    ants = _require_ants(); A = _load_atlas(cfg)
    arr = (img * brain).cpu().numpy().astype(np.float32); bm = brain.cpu().numpy().astype(bool)
    fa, ma = _ants_img(ants, arr, aff), _ants_img(ants, *A["tmpl"])
    kw = dict(fixed=fa, moving=ma, type_of_transform=cfg.get("ATLAS_TRANSFORM", "SyN"), verbose=False)
    try: r = ants.registration(random_seed=0, **kw)
    except TypeError: r = ants.registration(**kw)
    tl = r["fwdtransforms"]
    tm = ants.apply_transforms(fixed=fa, moving=_ants_img(ants, *A["tmask"]), transformlist=tl, interpolator="nearestNeighbor").numpy() > 0.5
    code = ants.apply_transforms(fixed=fa, moving=_ants_img(ants, *A["code"]), transformlist=tl, interpolator="genericLabel").numpy()
    dice = dice_bin(tm, bm); code = torch.from_numpy(np.rint(code).astype(np.int64)).to(DEV)
    return dict(ok=bool(dice >= cfg["ATLAS_MIN_DICE"]), dice=float(dice), cort=code % 1000, sub=code // 1000)


# ------------------------------------------------------------------ per-scan pipeline
@_ng
def process(name, vols, zooms, cfg, ext_mask=None, aff=None):
    t0 = time.time(); dev = DEV
    X, mean, shifts = align_mean(vols, cfg["MAX_SHIFT"], dev, cfg["ALIGN"], aff)
    brain, src, heur = None, None, None
    if ext_mask is not None:                                        # validated external mask (SynthStrip / HD-BET run outside)
        brain = ext_mask.to(dev).bool(); src = "external"
    elif cfg["SKULLSTRIP"] != "heuristic":                          # automatic SynthStrip -> HD-BET
        am, kind = auto_mask(name, mean, aff, cfg)
        if am is not None: brain, src = am.to(dev).bool(), kind
    if brain is not None and tuple(brain.shape) != tuple(mean.shape):
        print(f"   [skip] {name}: mask grid {tuple(brain.shape)} != image grid {tuple(mean.shape)}"); return None, None, None
    if brain is None:
        if cfg["REQUIRE_MASK"]: return None, None, None            # paper mode: no heuristic fallback
        brain, src = brain_mask(mean, cfg["ERODE"], dev), "heuristic"
        if brain is None: return None, None, None
        mask_dice = float("nan")
    else:
        heur = brain_mask(mean, cfg["ERODE"], dev)                  # reference only: audits the real mask
        mask_dice = dice_bin(brain, heur) if heur is not None else float("nan")
    brain = _clean_mask(brain)
    if int(brain.sum()) < 10000: return None, None, None
    vox = float(np.prod(zooms)); icv_ml = int(brain.sum()) * vox / 1000.0
    n = len(X)
    mean_c, field = bias_correct(mean, brain, zooms, cfg["BIAS"])
    # ---- repeat-consistency QC (bias-corrected with the same field)
    ncc_min = np.nan
    if n > 1:
        ncc_min = min(ncc(X[i], torch.stack([X[j] for j in range(n) if j != i]).mean(0), brain) for i in range(n))
    dice = np.nan
    if n > 1:
        h = n // 2; A = torch.stack(X[:h]).mean(0) / field; B = torch.stack(X[h:]).mean(0) / field
        dice = dice_macro(segment(A, brain, cfg["SEED"]).argmax(1), segment(B, brain, cfg["SEED"]).argmax(1))
    lo, hi = cfg["ICV_ML"]
    plaus = 1.0 if lo <= icv_ml <= hi else float(np.exp(-(((lo - icv_ml) if icv_ml < lo else (icv_ml - hi)) / 200.0) ** 2))
    # ---- tissue volumes (soft, MRF-regularised, on the bias-corrected mean)
    post, (gmu, gvar) = segment(mean_c, brain, cfg["SEED"], mrf=cfg["SEG_MRF"], em=cfg["SEG_EM"], params=True)
    sd_gw = float(torch.sqrt(0.5 * (gvar[1] + gvar[2]))); sd_cg = float(torch.sqrt(0.5 * (gvar[0] + gvar[1])))
    cnr_gw = float((gmu[2] - gmu[1]) / max(sd_gw, 1e-8)); cnr_cg = float((gmu[1] - gmu[0]) / max(sd_cg, 1e-8))
    cnr_min = min(cnr_gw, cnr_cg)
    q_d = 1.0 if np.isnan(dice) else float(np.clip((dice - 0.70) / 0.20, 0, 1))
    q_n = 1.0 if np.isnan(ncc_min) else float(np.clip((ncc_min - 0.85) / 0.12, 0, 1))
    q_c = float(np.clip((cnr_min - 0.8) / 1.2, 0, 1))              # tissue contrast: GM/WM and CSF/GM must be separable
    qc = float(np.clip(np.sqrt(q_d * q_n) * np.sqrt(q_c) * plaus, 0.05, 1.0))
    csf, gm, wm = (post[:, k].sum() * vox for k in range(3)); tot = csf + gm + wm
    # ---- ventricle proxy: deep, central, large CSF components (distance-to-surface in mm)
    brain_np = brain.cpu().numpy().astype(bool); lab = post.argmax(1)
    csf_np = np.zeros(brain_np.shape, bool); csf_np[brain_np] = (lab == 0).cpu().numpy()
    vm, n_comp = _vent_components(brain_np, csf_np, zooms, cfg)
    vm_b = torch.from_numpy(vm[brain_np]).to(dev)
    vent = post[vm_b, 0].sum() * vox if int(vm_b.sum()) > 0 else torch.zeros((), device=dev)
    feats = dict(**{"MRI ID": name}, qc_conf=qc, vent_icv=float(vent / tot), csf_icv=float(csf / tot),
                 gm_deficit=float(1 - gm / tot), bpf=float((gm + wm) / tot), vent_atlas_icv=float("nan"))
    diag = {"MRI ID": name, "n_repeats": n, "dice": dice, "ncc_min": ncc_min, "icv_ml": icv_ml, "plaus": plaus,
            "seg_cnr_gw": cnr_gw, "seg_cnr_cg": cnr_cg, "seg_cnr_min": cnr_min, "vent_n_comp": n_comp,
            "vent_ml": float(vent) / 1000.0,
            "mask_src": src, "mask_dice_vs_heuristic": mask_dice,
            "mask_suspect": bool((not np.isnan(mask_dice)) and mask_dice < 0.6),     # real mask disagrees strongly with the heuristic -> inspect qc_png
            "bias": cfg["BIAS"], "align": cfg["ALIGN"],
            "max_shift_vox": max(max(abs(a) for a in s) for s in shifts), "sec": time.time() - t0}
    reg = None
    if cfg["REGIONS"]:
        at = atlas_labels(mean_c, brain, aff, cfg)                                           # SyN-warped Harvard-Oxford labels in this scan's grid
        lv = (at["sub"] == HO_VENTRICLE_SUB[0]) | (at["sub"] == HO_VENTRICLE_SUB[1])         # atlas lateral ventricles (L+R)
        lv_b = lv[brain]
        vc = float(post[lv_b, 0].mean()) if int(lv_b.sum()) > 0 else float("nan")           # ROI-level sanity: ventricle label should be CSF
        at["ok"] = bool(at["ok"] and (not np.isnan(vc)) and vc >= cfg["ATLAS_MIN_VENT_CSF"])
        diag.update(atlas_reg="syn", atlas_dice=at["dice"], atlas_vent_csf=vc, atlas_ok=at["ok"])
        if not at["ok"]:
            print(f"   [warn] {name}: atlas registration rejected (Dice={at['dice']:.3f} < {cfg['ATLAS_MIN_DICE']} "
                  f"or ventricle-CSF={vc:.2f} < {cfg['ATLAS_MIN_VENT_CSF']}) -> no regional values")
        else:
            cl, sl = at["cort"][brain], at["sub"][brain]                                      # voxel order == post rows
            csf_p = post[:, 0]
            v_ctx = torch.bincount(cl, weights=post[:, 1], minlength=97) * vox               # cortical: grey-matter volume
            v_sub = torch.bincount(sl, weights=1 - csf_p, minlength=22) * vox                # subcortical: non-CSF volume
            v_csf = torch.bincount(sl, weights=csf_p, minlength=22) * vox                    # lateral ventricles: CSF volume
            reg = {"MRI ID": name}
            for nm, kind, v, _b in HO_ROIS:                                                   # all 112 ROIs
                reg[f"vol_{nm}"] = float((v_ctx if kind == "ctx" else v_csf if kind == "vent" else v_sub)[v])
            for k, (cb, sb) in HO_AGG.items():                                                # lobar aggregates for RUN='spread'
                reg[f"vol_{k}"] = float(sum(reg[f"vol_{nm}"] for nm, kind, v, b in HO_ROIS
                                            if (kind == "ctx" and b in cb) or (kind == "sub" and b in sb)))
            lv_np = lv.cpu().numpy().astype(bool)                                             # periventricular band: true mm distance (EDT)
            band = torch.from_numpy(ndi.distance_transform_edt(~lv_np, sampling=zooms) <= cfg["VENT_BAND_MM"]).to(dev) & brain
            band_b = band[brain]
            reg["vol_vent"] = float(band_b.sum() * vox - post[band_b, 0].sum() * vox)         # non-CSF in the band (falls as ventricles grow)
            feats["vent_atlas_icv"] = float(post[lv_b, 0].sum() * vox / tot)                  # atlas lateral-ventricle CSF / brain volume
    return feats, diag, (reg, mean_c.cpu().numpy(), brain.cpu().numpy(), post.argmax(1).cpu().numpy())


def save_png(path, mean, brain, labels, title):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    seg = np.zeros(mean.shape, np.int8); seg[brain] = labels + 1
    c = [s // 2 for s in mean.shape]; fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    for a in range(3):
        sl = [slice(None)] * 3; sl[a] = c[a]; sl = tuple(sl)
        ax[0, a].imshow(mean[sl].T, cmap="gray", origin="lower"); ax[0, a].contour(brain[sl].T, [0.5], colors="y", linewidths=.7)
        ax[1, a].imshow(seg[sl].T, cmap="viridis", origin="lower", vmin=0, vmax=3)
        ax[0, a].set_title(f"axis {a} slice {c[a]}"); ax[0, a].axis("off"); ax[1, a].axis("off")
    fig.suptitle(title); plt.tight_layout(); plt.savefig(path, dpi=80); plt.close(fig)


def render_montage(mean, brain, zooms, offs, size=224):
    """3x3 grayscale montage (rows = sagittal / coronal / axial for RAS data, cols = 3 slice offsets as fractions of the brain bbox),
    skull-stripped, physically scaled for anisotropic voxels, drawn with superior up.  This is what Qwen-VL sees."""
    from PIL import Image
    idx = np.argwhere(brain); lo, hi = idx.min(0), idx.max(0); c = (lo + hi) / 2; ext = hi - lo
    vmax = max(np.percentile(mean[brain], 99.5), 1e-6); img = np.clip(np.where(brain, mean, 0.0) / vmax, 0, 1)
    rows = []
    for ax in range(3):
        o = [x for x in range(3) if x != ax]; row = []                 # o[0] = horizontal axis, o[1] = vertical axis (RAS: S/A are last)
        for f in offs:
            k = int(np.clip(round(c[ax] + f * ext[ax]), 0, mean.shape[ax] - 1))
            sl = np.take(img, k, axis=ax)[lo[o[0]]:hi[o[0]] + 1, lo[o[1]]:hi[o[1]] + 1]
            pw, ph = max(1, int(sl.shape[0] * zooms[o[0]])), max(1, int(sl.shape[1] * zooms[o[1]]))
            im = Image.fromarray((sl.T[::-1] * 255).astype(np.uint8)).resize((pw, ph), Image.BILINEAR)
            m = max(pw, ph); canvas = Image.new("L", (m, m)); canvas.paste(im, ((m - pw) // 2, (m - ph) // 2))
            row.append(np.asarray(canvas.resize((size, size), Image.BILINEAR)))
        rows.append(np.concatenate(row, 1))
    return Image.fromarray(np.concatenate(rows, 0)).convert("RGB")


VIEWS = {"A": (-0.15, 0.0, 0.15), "B": (-0.25, -0.07, 0.10)}     # two independent slice sets -> agreement = Qwen QC


def cmd_export(a):
    """Write the aligned repeat-average of each session as NIfTI (RAS grid if CANONICAL), for SynthStrip / HD-BET.
       Then e.g.   mri_synthstrip -i rm_out/mri/mean/<ID>.nii.gz -m masks/<ID>.nii.gz     (FreeSurfer>=7.3 / SynthStrip container)
       or          hd-bet on the same folder (<ID>_bet_mask.nii.gz is also recognised).
       Finally set MASK_DIR='masks', REQUIRE_MASK=True and rerun RUN='mri'.
       (Unnecessary if SynthStrip / HD-BET are installable: SKULLSTRIP='auto' runs them inside RUN='mri'.)"""
    _need_torch(); nib = _nib(); out = f"{a.OUT}/mri/mean"; os.makedirs(out, exist_ok=True)
    scans = find_scans(a.RAW_ROOTS); k = 0
    for name in list(scans)[: a.MRI_LIMIT]:
        p = f"{out}/{name}.nii.gz"
        if os.path.exists(p): continue
        vols, zooms, aff = load_scan(scans[name], a.CANONICAL)
        if not vols: continue
        _, mean, _ = align_mean(vols, a.MAX_SHIFT, mode=a.ALIGN, aff=aff)
        nib.save(nib.Nifti1Image(mean.cpu().numpy().astype(np.float32), aff), p); k += 1
    print(f"exported {k} aligned-average volumes to {out}")


def cmd_mri(a):
    _need_torch(); nib = _nib(); cfg = vars(a); out = f"{a.OUT}/mri"
    os.makedirs(f"{out}/qc_png", exist_ok=True); os.makedirs(f"{out}/slices", exist_ok=True)
    print(f"device: {DEV}" + (f" ({torch.cuda.get_device_name(0)})" if DEV.type == "cuda" else "  [no GPU: enable it in Kaggle settings]"))
    _setup_stripper(a)
    print("skull-stripping:", f"{a.MASK_DIR} (external masks) -> " if a.MASK_DIR else "",
          " -> ".join(f"{k} ({p})" for k, p in _STRIP["queue"]) if _STRIP["queue"] else "no SynthStrip / HD-BET available",
          "" if (_STRIP["queue"] or a.MASK_DIR) else " -> HEURISTIC mask (not paper-grade)")
    if a.REQUIRE_MASK and not a.MASK_DIR and _STRIP["tool"] is None:
        raise SystemExit("REQUIRE_MASK=True but no usable skull-stripper. SynthStrip needs synthstrip.1.pt (Internet ON to download it, or attach it "
                         "as a Kaggle dataset and set SYNTHSTRIP_MODEL); or pip install hd-bet; or RUN='export' + MASK_DIR.")
    if a.REGIONS: _require_ants(); _load_atlas(cfg)                  # fail BEFORE processing anything if the atlas pipeline is unavailable
    scans = find_scans(a.RAW_ROOTS); names = list(scans)[: a.MRI_LIMIT]
    fp, dp, rp = f"{out}/mri_features.csv", f"{out}/mri_qc.csv", f"{out}/regional.csv"
    feats = pd.read_csv(fp).to_dict("records") if os.path.exists(fp) else []
    diags = pd.read_csv(dp).to_dict("records") if os.path.exists(dp) else []
    regs = pd.read_csv(rp).to_dict("records") if a.REGIONS and os.path.exists(rp) else []
    done = {r["MRI ID"] for r in feats}; todo = [n for n in names if n not in done]
    print(f"{len(scans)} sessions found, {len(done)} done, {len(todo)} to process")

    def flush():
        pd.DataFrame(feats).to_csv(fp, index=False); pd.DataFrame(diags).to_csv(dp, index=False)
        if a.REGIONS: pd.DataFrame(regs).to_csv(rp, index=False)

    n_png = len(glob.glob(f"{out}/qc_png/*.png")); t0 = time.time(); k = 0
    with ThreadPoolExecutor(2) as ex:                       # disk IO overlaps GPU compute
        q = deque(); it = iter(todo)
        for n in it:
            q.append((n, ex.submit(load_scan, scans[n], a.CANONICAL)))
            if len(q) >= 3: break
        while q:
            name, fut = q.popleft(); nxt = next(it, None)
            if nxt is not None: q.append((nxt, ex.submit(load_scan, scans[nxt], a.CANONICAL)))
            vols, zooms, aff = fut.result(); k += 1
            if not vols: print(f"  [skip] {name}: unreadable"); continue
            em = None; mp = find_mask(a.MASK_DIR, name)
            if mp: em = torch.from_numpy(load_mask(mp, a.CANONICAL))
            f, d, extra = process(name, vols, zooms, cfg, em, aff)
            if f is None:
                if a.REQUIRE_MASK and em is None and not a.MASK_DIR and _STRIP["tool"] is None:
                    flush(); raise SystemExit(f"skull-stripper unavailable/disabled at {name}; fix it and rerun (finished scans are cached).")
                print(f"  [skip] {name}: " + ("no external/automatic mask (REQUIRE_MASK)" if (em is None and a.REQUIRE_MASK) else "mask failed")); continue
            feats.append(f); diags.append(d)
            reg, mean, brain, labels = extra
            if reg: regs.append(reg)
            for v, offs in VIEWS.items(): render_montage(mean, brain, zooms, offs).save(f"{out}/slices/{name}_{v}.png")
            if n_png < a.QC_PNG and (k <= 4 or f["qc_conf"] < 0.5 or d.get("mask_suspect")):
                save_png(f"{out}/qc_png/{name}.png", mean, brain, labels,
                         f"{name} qc={f['qc_conf']:.2f} dice={d['dice']:.3f} cnr={d['seg_cnr_min']:.2f} icv={d['icv_ml']:.0f} mL [{d['mask_src']}]"); n_png += 1
            if k % 25 == 0: flush(); print(f"  {k}/{len(todo)}  ({(time.time()-t0)/k:.1f}s/scan)")
    flush()
    if not feats: return
    df, dg = pd.DataFrame(feats), pd.DataFrame(diags)
    print(f"\nMRI step done: {len(df)} sessions -> {fp}")
    cols = [c for c in ("dice", "ncc_min", "icv_ml", "max_shift_vox", "seg_cnr_min", "vent_ml", "mask_dice_vs_heuristic", "atlas_dice", "atlas_vent_csf") if c in dg]
    print(dg[cols].describe().round(3).to_string())
    if "mask_src" in dg:
        print("mask source:", dg["mask_src"].value_counts().to_dict())
        if (dg["mask_src"] == "heuristic").any():
            print("  [warn] some sessions use the HEURISTIC mask -> not paper-grade. Install SynthStrip / HD-BET (SKULLSTRIP='auto'), "
                  "or run RUN='export', segment with SynthStrip/HD-BET, set MASK_DIR, and use REQUIRE_MASK=True.")
    if "mask_suspect" in dg:
        ns = int(dg["mask_suspect"].fillna(False).astype(bool).sum())
        print(f"mask_suspect (Dice vs heuristic < 0.6): {ns}/{len(dg)} sessions" + ("  -> inspect those in qc_png" if ns else ""))
    if "atlas_ok" in dg: print(f"atlas registration accepted: {int(dg['atlas_ok'].fillna(False).astype(bool).sum())}/{len(dg)} sessions (Dice >= {a.ATLAS_MIN_DICE}, ventricle-CSF >= {a.ATLAS_MIN_VENT_CSF})")
    print(f"qc_conf<0.5: {(df.qc_conf < 0.5).sum()} sessions  (inspect {out}/qc_png before trusting the numbers)")


# =============================================================================== tissue-measure validation
def _assoc_ci(d, x, y, n_boot=500, seed=0):
    """Spearman rho with SUBJECT-cluster bootstrap CI (visits of a subject are not independent). -> (rho, lo, hi, n)"""
    from scipy.stats import spearmanr
    z = d[["sid", x, y]].dropna()
    if len(z) < 20 or z[x].nunique() < 3: return np.nan, np.nan, np.nan, len(z)
    rho = spearmanr(z[x], z[y])[0]; rng = np.random.default_rng(seed)
    groups = [g[[x, y]].values for _, g in z.groupby("sid")]; bs = []
    for _ in range(n_boot):
        arr = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(arr[:, 0])) > 2: bs.append(spearmanr(arr[:, 0], arr[:, 1])[0])
    lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan)
    return float(rho), float(lo), float(hi), len(z)


def _ccc(x, y):
    """Lin's concordance correlation coefficient."""
    return float(2 * np.mean((x - x.mean()) * (y - y.mean())) / (x.var() + y.var() + (x.mean() - y.mean()) ** 2 + 1e-12))


def cmd_validate(a):
    """Agreement of the intensity-based tissue measures with (i) OASIS's own nWBV/eTIV (rank only: different denominators) and
    (ii) optional REF_CSV (FreeSurfer / FSL-FAST) columns `ref_<feature>` -> Spearman (+CI), Lin's CCC, Bland-Altman bias & 95% LoA.
    NOTE: vent_icv is a deep-central-CSF-component fraction (ventricle PROXY), so expect rank agreement with true ventricular volume, not equality."""
    out = f"{a.OUT}/mri"; feats = pd.read_csv(f"{out}/mri_features.csv"); qc = pd.read_csv(f"{out}/mri_qc.csv")[["MRI ID", "icv_ml"]]
    d = load_table(find_table(a.XLSX)).merge(feats, left_on="mri", right_on="MRI ID").merge(qc, on="MRI ID")
    d["brain_frac"] = 1 - d["csf_icv"]; rows = []
    for lab, x, y in (("1-CSF fraction vs OASIS nWBV", "brain_frac", "nWBV"), ("mask volume vs OASIS eTIV", "icv_ml", "eTIV")):
        rho, lo, hi, n = _assoc_ci(d, x, y); rows.append(dict(comparison=lab, n=n, spearman=rho, ci_lo=lo, ci_hi=hi))
    if a.REF_CSV:
        ref = pd.read_csv(a.REF_CSV)
        d = d.merge(ref, on="MRI ID", how="inner")
        for c in [c for c in ref.columns if c.startswith("ref_") and c[4:] in d]:
            f = c[4:]; z = d[["sid", f, c]].dropna(); rho, lo, hi, n = _assoc_ci(d, f, c)
            dif = (z[f] - z[c]).values; sd = dif.std(ddof=1)
            rows.append(dict(comparison=f"{f} vs {c}", n=n, spearman=rho, ci_lo=lo, ci_hi=hi,
                             ccc=_ccc(z[f].values, z[c].values), ba_bias=dif.mean(),
                             ba_loa_lo=dif.mean() - 1.96 * sd, ba_loa_hi=dif.mean() + 1.96 * sd))
    else:
        print("  (no REF_CSV: only the built-in OASIS comparisons; add FreeSurfer/FSL references for a proper validation)")
    res = pd.DataFrame(rows); res.to_csv(f"{out}/validation_tissue.csv", index=False)
    print("\nTissue-measure validation:\n", res.round(3).to_string(index=False))


# =============================================================================== Qwen2.5-VL rubric scoring
RUBRICS = {"vent": "enlargement of the lateral ventricles",
           "gca": "global cortical atrophy (widening of the cortical sulci and thinning of the gyri)",
           "mta": "medial temporal lobe atrophy (shrinkage of the hippocampus with widening of the temporal horn and choroid fissure)"}
PROMPT = ("You are assisting with a research study. The image is a montage of nine skull-stripped T1-weighted brain MRI "
          "slices (three orthogonal planes, three slices each; orientation may be arbitrary). Rate the {what} on an ordinal "
          "scale: 0 = none, 1 = mild, 2 = moderate, 3 = severe. Answer in the exact format 'Score: <digit>'.")


def load_qwen(a):
    _need_torch()
    from transformers import AutoProcessor
    kw = dict(torch_dtype=torch.float16, device_map="auto")
    if a.QWEN_QUANT == "4bit":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                                       bnb_4bit_quant_type="nf4")
    if "Qwen2-VL" in a.QWEN_MODEL:
        from transformers import Qwen2VLForConditionalGeneration as M
    else:
        from transformers import Qwen2_5_VLForConditionalGeneration as M
    model = M.from_pretrained(a.QWEN_MODEL, **kw).eval()
    proc = AutoProcessor.from_pretrained(a.QWEN_MODEL, max_pixels=a.QWEN_MAX_PIXELS)
    ids = [proc.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(4)]
    return model, proc, ids


@_ng
def qwen_score(model, proc, ids, img, what, prefill):
    """Constrained format (assistant prefill 'Score: ') so the next token IS the digit. Returns
       E[s] over the digits (renormalised), its normalised entropy, and the FULL-vocabulary probability mass on {0..3}.
       The mass is the validity diagnostic: low mass => the renormalised distribution is not trustworthy."""
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT.format(what=what)}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + prefill
    inp = proc(text=[text], images=[img], return_tensors="pt").to(model.device)
    full = torch.softmax(model(**inp).logits[0, -1].float(), 0)                  # FULL vocabulary
    pdig = full[ids]; mass = float(pdig.sum()); p = (pdig / pdig.sum().clamp_min(1e-12)).cpu().numpy()
    return float((p * np.arange(4)).sum()), float(-(p * np.log(p + 1e-12)).sum() / np.log(4)), mass


def cmd_qwen(a):
    from PIL import Image
    out = f"{a.OUT}/mri"; sdir = f"{out}/slices"; mf = f"{out}/mri_features.csv"
    assert os.path.exists(mf), "run RUN='mri' first (it writes mri_features.csv and the slice montages)"
    ids_all = sorted({os.path.basename(p).rsplit("_", 1)[0] for p in glob.glob(f"{sdir}/*_A.png")})[: a.QWEN_LIMIT]
    rp = f"{out}/qwen_raw.csv"
    raw = pd.read_csv(rp).to_dict("records") if os.path.exists(rp) else []
    if raw and "vent_A_M" not in raw[0]:
        print("  old qwen_raw.csv (no digit-mass column / unconstrained prompt) -> discarded and rescored"); raw = []
    done = {r["MRI ID"] for r in raw}; todo = [i for i in ids_all if i not in done]
    print(f"Qwen: {len(ids_all)} montage sets, {len(done)} done, {len(todo)} to score with {a.QWEN_MODEL} ({a.QWEN_QUANT})")
    if todo:
        model, proc, ids = load_qwen(a); t0 = time.time()
        for n, name in enumerate(todo, 1):
            row = {"MRI ID": name}
            for v in VIEWS:
                img = Image.open(f"{sdir}/{name}_{v}.png")
                for r, what in RUBRICS.items():
                    e, h, m = qwen_score(model, proc, ids, img, what, a.QWEN_PREFILL)
                    row[f"{r}_{v}"] = e; row[f"{r}_{v}_H"] = h; row[f"{r}_{v}_M"] = m
            raw.append(row)
            if n % 20 == 0:
                pd.DataFrame(raw).to_csv(rp, index=False); print(f"  {n}/{len(todo)} ({(time.time()-t0)/n:.1f}s/scan)")
        pd.DataFrame(raw).to_csv(rp, index=False)
        del model; torch.cuda.empty_cache()
    R = pd.DataFrame(raw); feats = pd.read_csv(mf)
    out_df = pd.DataFrame({"MRI ID": R["MRI ID"]}); rel = {}
    for r in RUBRICS:
        ea, eb = R[f"{r}_A"], R[f"{r}_B"]; hbar = 0.5 * (R[f"{r}_A_H"] + R[f"{r}_B_H"]); mbar = np.minimum(R[f"{r}_A_M"], R[f"{r}_B_M"])
        score = 0.5 * (ea + eb)
        out_df[f"vlm_{r}"] = score.where(mbar >= a.QWEN_MIN_MASS)                          # low digit-mass => masked, not imputed
        out_df[f"vlm_{r}_conf"] = np.clip(np.exp(-(ea - eb).abs() / 0.75) * (1 - hbar) * mbar, 0.05, 1.0)
        out_df[f"vlm_{r}_mass"] = mbar
        rel[r] = dict(view_rho=float(ea.corr(eb, method="spearman")), mean_mass=float(mbar.mean()),
                      frac_valid=float((mbar >= a.QWEN_MIN_MASS).mean()))
        print(f"  vlm_{r}: view A-vs-B Spearman = {rel[r]['view_rho']:.3f} | mean digit mass = {rel[r]['mean_mass']:.3f} "
              f"| valid scans = {100*rel[r]['frac_valid']:.0f}% | mean conf = {out_df[f'vlm_{r}_conf'].mean():.3f}")
    merged = feats.merge(out_df, on="MRI ID", how="left"); merged.to_csv(f"{out}/features_all.csv", index=False)
    print(f"-> {out}/features_all.csv  (MRI geometry + Qwen rubrics + per-channel confidences / digit mass)")
    report_qwen_validity(a, merged, rel)


# expected direction of each rubric vs INDEPENDENT anatomical measures (higher rubric = more atrophy)
EXPECT = {"vent": [("vent_icv", +1), ("nWBV", -1)], "gca": [("csf_icv", +1), ("nWBV", -1)], "mta": [("vent_icv", +1), ("nWBV", -1)]}


def report_qwen_validity(a, merged, rel):
    """Labels are NEVER shown to Qwen and NOT used here for gating. A rubric passes only if (i) the two slice views agree,
    (ii) enough full-vocab mass sits on the digits, (iii) every expected association with an independent measure has the right
    sign, a subject-cluster-bootstrap CI excluding 0 and |rho| >= QWEN_MIN_RHO. Failing rubrics are NOT used as emissions."""
    d = load_table(find_table(a.XLSX)).merge(merged, left_on="mri", right_on="MRI ID", how="inner"); rows = []
    print("\nQwen rubric validity (independent anatomical measures; subject-cluster bootstrap CIs):")
    for r in RUBRICS:
        c = f"vlm_{r}"; good = []; start = len(rows)
        for ref, sg in EXPECT[r]:
            rho, lo, hi, n = _assoc_ci(d, c, ref)
            ok = (not np.isnan(rho)) and np.sign(rho) == sg and ((lo > 0) if sg > 0 else (hi < 0)) and abs(rho) >= a.QWEN_MIN_RHO
            good.append(bool(ok)); rows.append(dict(rubric=r, reference=ref, expected_sign=sg, rho=rho, ci_lo=lo, ci_hi=hi, n=n, pair_ok=bool(ok)))
            print(f"  {c} vs {ref}: rho={rho:+.3f} [{lo:+.3f},{hi:+.3f}] n={n} expected sign {sg:+d} -> {'ok' if ok else 'FAIL'}")
        passed = bool(all(good) and rel[r]["view_rho"] >= 0.3 and rel[r]["mean_mass"] >= a.QWEN_MIN_MASS)
        for x in rows[start:]:
            x.update(view_rho=rel[r]["view_rho"], mean_mass=rel[r]["mean_mass"], rubric_pass=passed)
        okd = d[[c, "CDR"]].dropna()
        if (okd["CDR"] > 0).nunique() > 1 and len(okd) > 20:
            print(f"  {c}: [info only] AUC for CDR>0 = {roc_auc_score(okd['CDR'] > 0, okd[c]):.3f} | rubric {'PASSES' if passed else 'FAILS'} validation")
        else:
            print(f"  {c}: rubric {'PASSES' if passed else 'FAILS'} validation")
    pd.DataFrame(rows).to_csv(f"{a.OUT}/mri/qwen_validity.csv", index=False)
    print("  -> only rubrics marked PASS are attached as emissions; report failures as a negative result or use a larger / LoRA-tuned VLM.")


# =============================================================================== entry point
def _attach_features(a):
    """auto-use image emissions from the mri / qwen steps unless VLM_CSV is set; Qwen rubrics are attached ONLY if they passed validation."""
    if a.VLM_CSV is not None or not a.USE_MRI: return
    allf, geo = f"{a.OUT}/mri/features_all.csv", f"{a.OUT}/mri/mri_features.csv"
    valf = f"{a.OUT}/mri/qwen_validity.csv"
    if a.USE_QWEN and os.path.exists(allf):
        a.VLM_CSV = allf
        if a.EXTRA_COLS is None:
            cols = ["vent_icv"]
            if os.path.exists(valf):
                v = pd.read_csv(valf); ok = v.groupby("rubric")["rubric_pass"].first()
                passed = [f"vlm_{r}" for r in ("mta", "vent", "gca") if bool(ok.get(r, False))]
                cols += passed[:1]                                         # one rubric channel (mta preferred) to limit collinearity
                if not passed: print("  [Qwen] no rubric passed validation -> Qwen channels NOT used")
            else:
                print("  [Qwen] no qwen_validity.csv -> Qwen channels NOT used")
            a.EXTRA_COLS = cols
    elif os.path.exists(geo):
        a.VLM_CSV = geo
        if a.EXTRA_COLS is None: a.EXTRA_COLS = ["vent_icv"]
    if a.VLM_CSV: print(f"image emissions: {a.VLM_CSV} -> {a.EXTRA_COLS}")


def main(overrides=None):
    global CONF_ON, MAX_RESTARTS
    a = SimpleNamespace(**{**CONFIG, **(overrides or {})})
    CONF_ON, MAX_RESTARTS = a.CONF_ON, a.MAX_RESTARTS
    run = a.RUN; t0 = time.time()
    if run == "export": cmd_export(a)
    if run in ("all", "mri"): cmd_mri(a)
    if run in ("all", "validate"):
        try: cmd_validate(a)
        except Exception as e:
            if run == "validate": raise
            print(f"[tissue validation skipped: {type(e).__name__}: {e}]")
    if run in ("all", "qwen"):
        if run == "qwen" or a.USE_QWEN:
            try: cmd_qwen(a)
            except Exception as e:
                if run == "qwen": raise
                print(f"[qwen step skipped: {type(e).__name__}: {e}]\n  (no internet/model? attach a Qwen2.5-VL model and set QWEN_MODEL to its path; continuing without)")
    if run in ("all", "stage"):
        _attach_features(a)
        for cutoff in a.CUTOFFS:
            sub = f"{a.OUT}/{'retrospective' if cutoff is None else f'prospective_{cutoff}y'}"
            cmd_stage(a, cutoff, sub)
    elif run == "ablate":
        _attach_features(a); cmd_ablate(a, a.CUTOFFS[0])
    elif run == "spread":
        if a.REGIONAL_CSV is None and os.path.exists(f"{a.OUT}/mri/regional.csv"): a.REGIONAL_CSV = f"{a.OUT}/mri/regional.csv"
        cmd_spread(a)
    elif run == "synth":
        cmd_synth(a)
    elif run not in ("all", "mri", "qwen", "export", "validate"):
        raise ValueError(f"unknown RUN={run!r}")
    print(f"\nTotal wall time {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
