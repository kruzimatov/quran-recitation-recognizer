"""Unify all audio into 5s WAV clips (16 kHz mono) + extract MFCC features.

Input roots  (all optional, script skips missing ones):
  data/base/Dataset/Dataset/<Reciter>/*.wav   ← Kaggle base 12 qaris (already 1s WAV)
  data/raw_extra/<Reciter>/*.mp3              ← everyayah.com Arab reciters
  data/raw_uzbek/<Reciter>/*.mp3              ← YouTube Uzbek qaris

Output:
  data/clips/<Reciter>/<uid>.wav              ← standard 5s mono 16kHz WAV clips
  data/features.parquet                       ← MFCC + delta stats per clip
  data/labels.csv                             ← clip → reciter mapping
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

ROOT       = Path(__file__).resolve().parent.parent
BASE_DIR   = ROOT / "data" / "base"
EXTRA_DIR  = ROOT / "data" / "raw_extra"
UZBEK_DIR  = ROOT / "data" / "raw_uzbek"
CLIPS_DIR  = ROOT / "data" / "clips"
FEAT_PATH  = ROOT / "data" / "features.parquet"
LABEL_PATH = ROOT / "data" / "labels.csv"

SR          = 16_000
CLIP_SEC    = 5
CLIP_LEN    = SR * CLIP_SEC
MAX_CLIPS_PER_RECITER = 400  # cap to keep dataset balanced-ish


def iter_reciter_files():
    # base: Kaggle dataset extracts to Dataset/Dataset/<name>/*.wav
    for candidate in [BASE_DIR / "Dataset" / "Dataset", BASE_DIR / "Dataset", BASE_DIR]:
        if candidate.exists():
            for reciter in sorted(p for p in candidate.iterdir() if p.is_dir()):
                yield reciter.name, sorted(reciter.rglob("*.wav")) + sorted(reciter.rglob("*.mp3"))
            break

    for root in (EXTRA_DIR, UZBEK_DIR):
        if not root.exists():
            continue
        for reciter in sorted(p for p in root.iterdir() if p.is_dir()):
            files = sorted(reciter.rglob("*.mp3")) + sorted(reciter.rglob("*.wav"))
            if files:
                yield reciter.name, files


def segment_file(audio_path: Path, reciter_out: Path, start_idx: int) -> int:
    try:
        y, _ = librosa.load(audio_path, sr=SR, mono=True)
    except Exception as exc:  # noqa: BLE001
        print(f"skip {audio_path}: {exc}")
        return start_idx

    y, _ = librosa.effects.trim(y, top_db=30)
    if len(y) < CLIP_LEN // 2:
        return start_idx

    if len(y) < CLIP_LEN:
        y = np.pad(y, (0, CLIP_LEN - len(y)))

    n_clips = len(y) // CLIP_LEN
    for i in range(n_clips):
        clip = y[i * CLIP_LEN:(i + 1) * CLIP_LEN]
        clip = clip / (np.max(np.abs(clip)) + 1e-9)  # peak normalise
        uid = hashlib.md5(f"{audio_path.name}-{i}".encode()).hexdigest()[:12]
        out = reciter_out / f"{start_idx + i:05d}_{uid}.wav"
        sf.write(out, clip, SR, subtype="PCM_16")
    return start_idx + n_clips


def mfcc_stats(y: np.ndarray) -> dict:
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=SR)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=SR)[0]

    feats: dict = {}
    for name, m in [("mfcc", mfcc), ("delta", delta), ("delta2", delta2)]:
        feats.update({f"{name}_{i}_mean": float(m[i].mean()) for i in range(m.shape[0])})
        feats.update({f"{name}_{i}_std":  float(m[i].std())  for i in range(m.shape[0])})
    feats["zcr_mean"] = float(zcr.mean())
    feats["zcr_std"]  = float(zcr.std())
    feats["centroid_mean"] = float(centroid.mean())
    feats["centroid_std"]  = float(centroid.std())
    feats["rolloff_mean"]  = float(rolloff.mean())
    feats["rolloff_std"]   = float(rolloff.std())
    return feats


def main() -> None:
    if CLIPS_DIR.exists():
        shutil.rmtree(CLIPS_DIR)
    CLIPS_DIR.mkdir(parents=True)

    # ------- 1. segment ---------
    reciter_files = list(iter_reciter_files())
    print(f"Found {len(reciter_files)} reciter folders")

    for reciter, files in reciter_files:
        reciter_out = CLIPS_DIR / reciter
        reciter_out.mkdir(exist_ok=True)
        idx = 0
        for f in tqdm(files, desc=f"seg {reciter[:24]:24s}", leave=False):
            if idx >= MAX_CLIPS_PER_RECITER:
                break
            # for base dataset the files are already 1s WAV — concatenate 5 into one clip
            if f.suffix.lower() == ".wav" and f.stat().st_size < 100_000:
                # accumulate until we have 5s
                buf = []
                for wav in files[files.index(f):]:
                    y, _ = librosa.load(wav, sr=SR, mono=True)
                    buf.append(y)
                    if sum(len(x) for x in buf) >= CLIP_LEN:
                        break
                y_join = np.concatenate(buf)[:CLIP_LEN]
                if len(y_join) < CLIP_LEN:
                    y_join = np.pad(y_join, (0, CLIP_LEN - len(y_join)))
                y_join = y_join / (np.max(np.abs(y_join)) + 1e-9)
                uid = hashlib.md5(f.name.encode()).hexdigest()[:12]
                sf.write(reciter_out / f"{idx:05d}_{uid}.wav", y_join, SR, subtype="PCM_16")
                idx += 1
            else:
                idx = segment_file(f, reciter_out, idx)

    # ------- 2. features ---------
    rows = []
    labels = []
    for reciter_dir in sorted(CLIPS_DIR.iterdir()):
        if not reciter_dir.is_dir():
            continue
        clips = sorted(reciter_dir.glob("*.wav"))
        print(f"{reciter_dir.name:35s} {len(clips)} clips")
        for wav in tqdm(clips, desc=f"feat {reciter_dir.name[:20]:20s}", leave=False):
            y, _ = librosa.load(wav, sr=SR, mono=True)
            feats = mfcc_stats(y)
            feats["clip"] = wav.name
            rows.append(feats)
            labels.append({"clip": wav.name, "reciter": reciter_dir.name})

    feat_df = pd.DataFrame(rows)
    label_df = pd.DataFrame(labels)
    feat_df.to_parquet(FEAT_PATH, index=False)
    label_df.to_csv(LABEL_PATH, index=False)
    print(f"\nSaved features: {FEAT_PATH} ({feat_df.shape})")
    print(f"Saved labels:   {LABEL_PATH} ({label_df.shape})")


if __name__ == "__main__":
    main()
