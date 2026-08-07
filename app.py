"""Streamlit app — Quran Qari Classifier (CNN version).

Shazam-style: user uploads audio, PyTorch CNN on mel-spectrograms predicts
top-3 qaris with confidence, plus waveform + spectrogram of the input.

Local run:
    pip install -r requirements.txt
    streamlit run app.py

Requires artifacts produced by `quran_qari_cnn.ipynb`:
    qari_cnn.pt          -- model weights (state dict)
    qari_cnn_meta.joblib -- label encoder + sr/n_mels/img dims + eval scores
"""

from __future__ import annotations

import io
from pathlib import Path

import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from skimage.transform import resize

MODEL_PATH = Path(__file__).with_name("qari_cnn.pt")
META_PATH  = Path(__file__).with_name("qari_cnn_meta.joblib")


class QariCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


@st.cache_resource
def load_artifacts():
    meta   = joblib.load(META_PATH)
    device = (
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    model = QariCNN(num_classes=len(meta["classes"]))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    return meta, model, device


def wav_to_melspec(y: np.ndarray, sr: int, n_mels: int, img_h: int, img_w: int) -> np.ndarray:
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=n_mels)
    spec_db = librosa.power_to_db(spec, ref=np.max)
    spec_resized = resize(spec_db, (img_h, img_w), anti_aliasing=True)
    m, s = spec_resized.mean(), spec_resized.std() + 1e-6
    return ((spec_resized - m) / s).astype(np.float32)


def segment(y: np.ndarray, sr: int, clip_sec: int) -> list[np.ndarray]:
    clip_len = sr * clip_sec
    y_trim, _ = librosa.effects.trim(y, top_db=30)
    if len(y_trim) < clip_len // 2:
        return []
    if len(y_trim) < clip_len:
        y_trim = np.pad(y_trim, (0, clip_len - len(y_trim)))
    n = len(y_trim) // clip_len
    return [y_trim[i * clip_len:(i + 1) * clip_len] for i in range(n)]


def predict(y: np.ndarray, meta: dict, model: nn.Module, device: str) -> tuple[np.ndarray, int]:
    sr, n_mels, img_h, img_w = meta["sr"], meta["n_mels"], meta["img_h"], meta["img_w"]
    clips = segment(y, sr, clip_sec=5)
    if not clips:
        return np.zeros(len(meta["classes"])), 0

    specs = [wav_to_melspec(c / (np.max(np.abs(c)) + 1e-9), sr, n_mels, img_h, img_w) for c in clips]
    batch = torch.tensor(np.stack(specs)).unsqueeze(1).to(device)  # (N, 1, H, W)
    with torch.no_grad():
        logits = model(batch)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
    return probs.mean(axis=0), len(clips)


def main() -> None:
    st.set_page_config(page_title="Quran Qari Classifier", page_icon="📖")
    meta, model, device = load_artifacts()
    classes = meta["classes"]

    st.title("Quran Qari Classifier")
    st.caption(
        f"CNN on mel-spectrograms · {len(classes)} qaris · "
        f"test accuracy {meta['test_acc']*100:.1f}% · top-3 {meta['test_top3']*100:.1f}% · "
        f"device: {device}"
    )

    uploaded = st.file_uploader(
        "Upload a Quran recitation clip",
        type=["mp3", "wav", "m4a", "ogg", "flac"],
    )
    if uploaded is None:
        st.info("Upload an audio clip (≥ 5 seconds recommended).")
        st.markdown(
            "**Tip.** Clean recitation only. Lectures, background music, or multi-voice audio "
            "confuse the model."
        )
        with st.expander("Supported qaris"):
            st.write(classes)
        return

    with st.spinner("Analyzing audio…"):
        raw = uploaded.read()
        try:
            y, sr = librosa.load(io.BytesIO(raw), sr=meta["sr"], mono=True)
        except Exception:
            # write to temp file so audioread/ffmpeg backend can decode odd containers
            import tempfile
            suffix = "." + (uploaded.name.rsplit(".", 1)[-1].lower() if "." in uploaded.name else "bin")
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(raw)
                tmp_path = tmp.name
            y, sr = librosa.load(tmp_path, sr=meta["sr"], mono=True)
        st.audio(raw, format=uploaded.type or "audio/mpeg")
        probs, n_clips = predict(y, meta, model, device)

    if n_clips == 0:
        st.error("Audio too short after trimming silence. Try a longer clip.")
        return

    ranked = sorted(zip(classes, probs), key=lambda x: -x[1])
    st.subheader(f"Top 3 predictions ({n_clips} clip{'s' if n_clips > 1 else ''} averaged)")
    for i, (name, p) in enumerate(ranked[:3], 1):
        st.write(f"**{i}. {name}** — {p*100:.1f}%")
        st.progress(float(p))

    with st.expander("All predictions"):
        st.dataframe(
            pd.DataFrame(ranked, columns=["qari", "probability"]).assign(
                probability=lambda d: d["probability"].map(lambda x: f"{x*100:.2f}%")
            ),
            use_container_width=True,
        )

    st.divider()
    st.subheader("Waveform + mel spectrogram")
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))

    librosa.display.waveshow(y, sr=sr, ax=axes[0], color="steelblue")
    axes[0].set(title=f"Waveform ({n_clips} × 5s clips analyzed)", xlabel="time (s)")

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=axes[1], cmap="magma")
    axes[1].set(title="Mel spectrogram")
    plt.tight_layout()
    st.pyplot(fig)


if __name__ == "__main__":
    main()
