"""
scan_and_save_signals.py
-------------------------
Scans the EEG dataset (181 EDF files), preprocesses each recording
(band-pass filter + resample + channel selection, same as the original
pipeline), and saves the preprocessed signal arrays + a metadata index
to disk. No ML/classification is done here -- this is purely the
scan -> preprocess -> save step so the cached signals can be reused
later for PLM/PLV connectivity pipelines or ML feature extraction
without re-reading/re-filtering every EDF file from scratch.

Outputs (under CFG.SAVE_DIR):
    signals/<subject_id>__<condition>__<idx>.npy   -> preprocessed signal, shape (n_channels, n_samples)
    metadata.csv                                    -> one row per saved recording
    channel_names.npy                               -> channel name list (assumed consistent across files)
    scan_log.txt                                     -> any files skipped + reasons
"""

import os
import re
import glob
import json
import warnings
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import mne

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
@dataclass
class CONFIG:
    DATA_ROOT: str = "/kaggle/input/datasets/kinnarhalder/eeg-dataset"
    SFREQ_TARGET: int = 128
    BANDPASS: Tuple[float, float] = (1.0, 45.0)
    N_CHANNELS: int = 19
    SAVE_DIR: str = "processed_signals"   # relative -> written next to this script


CFG = CONFIG()


# --------------------------------------------------------------------------- #
# 1. Data indexing / labeling
# --------------------------------------------------------------------------- #
def index_dataset(root: str) -> List[Dict]:
    """Scan DATA_ROOT and return [{path, subject_id, label, condition}, ...].

    Extracts:
      - group/subject id from patterns like 'MDD S12' / 'H S7'
      - condition (EO / EC / TASK) from the filename if present, else 'UNK'
    """
    records = []
    for path in sorted(glob.glob(os.path.join(root, "*.edf"))):
        fname = os.path.basename(path)
        m = re.search(r"(MDD|H)\s*S(\d+)", fname, flags=re.IGNORECASE)
        if not m:
            continue
        group, subj_num = m.group(1).upper(), m.group(2)
        label = 1 if group == "MDD" else 0
        subject_id = f"{group}_{subj_num}"

        cond_m = re.search(r"\b(EO|EC|TASK)\b", fname, flags=re.IGNORECASE)
        condition = cond_m.group(1).upper() if cond_m else "UNK"

        records.append({
            "path": path,
            "filename": fname,
            "subject_id": subject_id,
            "label": label,
            "condition": condition,
        })
    if not records:
        raise RuntimeError(f"No EDF files matched under {root}")
    return records


# --------------------------------------------------------------------------- #
# 2. Signal loading + preprocessing (filter, resample, channel selection)
# --------------------------------------------------------------------------- #
def load_raw(path: str, cfg: CONFIG) -> Tuple[np.ndarray, List[str], float]:
    """Returns (data, channel_names, sfreq) after band-pass filter,
    resampling to SFREQ_TARGET, and trimming/validating channel count."""
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    raw.pick_types(eeg=True)
    if len(raw.ch_names) < cfg.N_CHANNELS:
        raise ValueError(f"only {len(raw.ch_names)} EEG channels, need {cfg.N_CHANNELS}")
    if len(raw.ch_names) > cfg.N_CHANNELS:
        raw.pick(raw.ch_names[: cfg.N_CHANNELS])
    raw.filter(cfg.BANDPASS[0], cfg.BANDPASS[1], verbose="ERROR")
    if raw.info["sfreq"] != cfg.SFREQ_TARGET:
        raw.resample(cfg.SFREQ_TARGET, verbose="ERROR")
    return raw.get_data(), list(raw.ch_names), float(raw.info["sfreq"])


# --------------------------------------------------------------------------- #
# 3. Scan all files, preprocess, and save to disk
# --------------------------------------------------------------------------- #
def scan_and_save(records: List[Dict], cfg: CONFIG):
    signals_dir = os.path.join(cfg.SAVE_DIR, "signals")
    os.makedirs(signals_dir, exist_ok=True)

    meta_rows = []
    skipped = []
    reference_channels: Optional[List[str]] = None

    for idx, rec in enumerate(records):
        try:
            data, ch_names, sfreq = load_raw(rec["path"], cfg)
        except Exception as exc:
            skipped.append(f"{rec['filename']}: {exc}")
            print(f"[WARN] skipping {rec['filename']}: {exc}")
            continue

        if reference_channels is None:
            reference_channels = ch_names
        elif ch_names != reference_channels:
            print(f"[WARN] channel order/name mismatch in {rec['filename']} "
                  f"(will still save, but check channel_names.npy vs this file's channels)")

        # Unique filename per saved array: subject__condition__index
        save_name = f"{rec['subject_id']}__{rec['condition']}__{idx:03d}.npy"
        save_path = os.path.join(signals_dir, save_name)
        np.save(save_path, data.astype(np.float32))

        n_samples = data.shape[1]
        meta_rows.append({
            "subject_id": rec["subject_id"],
            "label": rec["label"],              # 1 = MDD, 0 = Healthy
            "condition": rec["condition"],       # EO / EC / TASK / UNK
            "original_filename": rec["filename"],
            "original_path": rec["path"],
            "npy_path": os.path.abspath(save_path),
            "n_channels": data.shape[0],
            "n_samples": n_samples,
            "sfreq": sfreq,
            "duration_sec": n_samples / sfreq,
        })

        print(f"[{idx + 1}/{len(records)}] saved {save_name} "
              f"(shape={data.shape}, sfreq={sfreq})")

    meta_df = pd.DataFrame(meta_rows)
    meta_csv_path = os.path.join(cfg.SAVE_DIR, "metadata.csv")
    meta_df.to_csv(meta_csv_path, index=False)

    if reference_channels is not None:
        np.save(os.path.join(cfg.SAVE_DIR, "channel_names.npy"), np.array(reference_channels))

    with open(os.path.join(cfg.SAVE_DIR, "scan_log.txt"), "w") as f:
        f.write(f"Total files found: {len(records)}\n")
        f.write(f"Successfully saved: {len(meta_rows)}\n")
        f.write(f"Skipped: {len(skipped)}\n\n")
        if skipped:
            f.write("Skipped files:\n")
            for line in skipped:
                f.write(f"  - {line}\n")

    with open(os.path.join(cfg.SAVE_DIR, "config_used.json"), "w") as f:
        json.dump({
            "DATA_ROOT": cfg.DATA_ROOT,
            "SFREQ_TARGET": cfg.SFREQ_TARGET,
            "BANDPASS": cfg.BANDPASS,
            "N_CHANNELS": cfg.N_CHANNELS,
        }, f, indent=2)

    return meta_df, skipped


# --------------------------------------------------------------------------- #
# 4. Main
# --------------------------------------------------------------------------- #
def main():
    records = index_dataset(CFG.DATA_ROOT)
    print(f"Indexed {len(records)} recordings across "
          f"{len(set(r['subject_id'] for r in records))} subjects")

    meta_df, skipped = scan_and_save(records, CFG)

    print("\n" + "=" * 70)
    print(f"Done. Saved {len(meta_df)} / {len(records)} recordings.")
    print(f"Skipped {len(skipped)} file(s).")
    print(f"Outputs written to: {os.path.abspath(CFG.SAVE_DIR)}")
    print("  - signals/*.npy      -> preprocessed (filtered+resampled) signal arrays")
    print("  - metadata.csv       -> subject_id, label, condition, path, sfreq, duration, etc.")
    print("  - channel_names.npy  -> channel name list")
    print("  - scan_log.txt       -> summary + skipped-file reasons")
    print("  - config_used.json   -> preprocessing config used for this run")
    print("=" * 70)

    print("\nCondition/label breakdown:")
    print(meta_df.groupby(["condition", "label"]).size())

    return meta_df


if __name__ == "__main__":
    main()
