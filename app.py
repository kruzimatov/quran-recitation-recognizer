"""Streamlit app — Quran Qari Classifier (CNN version).

Shazam-style: upload audio → CNN predicts top-3 qaris with confidence,
plus per-segment probability heatmap, waveform, and mel spectrogram.

Local run:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import io
import tempfile
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

MODEL_PATH   = Path(__file__).with_name("qari_cnn.pt")
META_PATH    = Path(__file__).with_name("qari_cnn_meta.joblib")
SAMPLES_DIR  = Path(__file__).with_name("samples")


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


def predict_per_segment(y: np.ndarray, meta: dict, model: nn.Module, device: str):
    sr = meta["sr"]
    clips = segment(y, sr, clip_sec=5)
    if not clips:
        return np.zeros((0, len(meta["classes"]))), []
    specs = [wav_to_melspec(c / (np.max(np.abs(c)) + 1e-9), sr, meta["n_mels"], meta["img_h"], meta["img_w"]) for c in clips]
    batch = torch.tensor(np.stack(specs)).unsqueeze(1).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(batch), dim=1).cpu().numpy()
    return probs, clips


def load_audio_bytes(raw: bytes, filename: str, sr_target: int) -> np.ndarray:
    try:
        y, _ = librosa.load(io.BytesIO(raw), sr=sr_target, mono=True)
        return y
    except Exception:
        suffix = "." + (filename.rsplit(".", 1)[-1].lower() if "." in filename else "bin")
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw)
            path = tmp.name
        y, _ = librosa.load(path, sr=sr_target, mono=True)
        return y


def render_results(y: np.ndarray, name: str, mime: str, raw: bytes, meta: dict, model: nn.Module, device: str) -> None:
    st.audio(raw, format=mime or "audio/mpeg")
    with st.spinner("Analyzing…"):
        seg_probs, clips = predict_per_segment(y, meta, model, device)

    if seg_probs.shape[0] == 0:
        st.error("Audio too short after trimming silence. Try a longer clip (≥ 5 s).")
        return

    classes = np.array(meta["classes"])
    mean_probs = seg_probs.mean(axis=0)
    order = mean_probs.argsort()[::-1]
    top_idx = order[:3]
    top_names = classes[top_idx]
    top_scores = mean_probs[top_idx]
    certainty = float(top_scores[0] - top_scores[1]) if len(top_scores) > 1 else float(top_scores[0])

    st.subheader(f"Top 3 predictions — {seg_probs.shape[0]} × 5 s segment{'s' if seg_probs.shape[0] > 1 else ''} averaged")

    c1, c2, c3 = st.columns(3)
    for col, (n, s) in zip((c1, c2, c3), zip(top_names, top_scores)):
        col.metric(n, f"{s*100:.1f}%")

    st.caption(f"Certainty margin (top-1 – top-2): {certainty*100:.1f} pp")
    st.progress(min(max(float(top_scores[0]), 0.0), 1.0))

    # top-5 bar chart
    top5_idx = order[:5]
    fig1, ax1 = plt.subplots(figsize=(9, 3.5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(top5_idx)))
    bars = ax1.barh(classes[top5_idx][::-1], mean_probs[top5_idx][::-1] * 100, color=colors[::-1])
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("probability (%)")
    ax1.set_title("Top-5 qari confidence")
    for bar, val in zip(bars, mean_probs[top5_idx][::-1] * 100):
        ax1.text(val + 1, bar.get_y() + bar.get_height() / 2, f"{val:.1f}", va="center", fontsize=9)
    ax1.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig1)

    # per-segment heatmap over top-5 qaris
    if seg_probs.shape[0] > 1:
        st.subheader("Per-segment probability across time")
        heat = seg_probs[:, top5_idx].T
        fig2, ax2 = plt.subplots(figsize=(9, max(2.5, 0.4 * len(top5_idx))))
        im = ax2.imshow(heat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax2.set_yticks(range(len(top5_idx)))
        ax2.set_yticklabels(classes[top5_idx])
        ax2.set_xticks(range(seg_probs.shape[0]))
        ax2.set_xticklabels([f"{i*5}-{(i+1)*5}s" for i in range(seg_probs.shape[0])], rotation=45, ha="right")
        ax2.set_title("Probability per 5-second segment")
        fig2.colorbar(im, ax=ax2, label="probability")
        st.pyplot(fig2)

    # waveform + mel spectrogram
    st.subheader("Audio inspection")
    sr = meta["sr"]
    fig3, axes = plt.subplots(2, 1, figsize=(10, 5))
    librosa.display.waveshow(y, sr=sr, ax=axes[0], color="#2E8B57")
    axes[0].set(title=f"Waveform — {name}", xlabel="time (s)")
    axes[0].spines[["top", "right"]].set_visible(False)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=axes[1], cmap="magma")
    axes[1].set(title="Mel spectrogram")
    plt.tight_layout()
    st.pyplot(fig3)

    with st.expander("All 34 qari probabilities"):
        st.dataframe(
            pd.DataFrame({"qari": classes[order], "probability": mean_probs[order]})
              .assign(probability=lambda d: d["probability"].map(lambda x: f"{x*100:.2f}%")),
            use_container_width=True,
            height=420,
        )


def main() -> None:
    st.set_page_config(page_title="Quran Qari Classifier", page_icon="📖", layout="wide")

    meta, model, device = load_artifacts()

    # ---- sidebar ---------------------------------------------------------
    with st.sidebar:
        st.header("Model")
        st.write(f"**Architecture:** 4-block CNN on mel-spectrograms")
        st.write(f"**Trainable params:** ~100 k")
        st.write(f"**Compute:** `{device}`")
        st.write(f"**Sample rate:** {meta['sr']} Hz")
        st.write(f"**Spectrogram:** {meta['n_mels']} mel × {meta['img_w']} frames")
        st.divider()

        st.header("Performance")
        st.metric("Test accuracy", f"{meta['test_acc']*100:.2f} %")
        st.metric("Top-3 accuracy", f"{meta['test_top3']*100:.2f} %")
        st.metric("Best val accuracy", f"{meta['best_val_acc']*100:.2f} %")
        st.divider()

        st.header("Supported qaris")
        st.caption(f"{len(meta['classes'])} reciters — Arab + Uzbek")
        for name in meta["classes"]:
            st.text(f"• {name.replace('_', ' ')}")

    # ---- header ----------------------------------------------------------
    st.title("📖 Quran Qari Classifier")
    st.caption(
        "Upload a Quran recitation clip. A convolutional neural network on mel-spectrograms "
        "predicts the reciter."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Qaris", len(meta["classes"]))
    m2.metric("Test accuracy", f"{meta['test_acc']*100:.1f}%")
    m3.metric("Top-3 accuracy", f"{meta['test_top3']*100:.1f}%")
    m4.metric("Model size", "0.4 MB")

    st.divider()

    # ---- input tabs -----------------------------------------------------
    tab_upload, tab_samples = st.tabs(["📤 Upload", "🎧 Try a sample"])

    with tab_upload:
        uploaded = st.file_uploader(
            "Choose an audio file",
            type=["mp3", "wav", "m4a", "ogg", "flac"],
            help="Any length ≥ 5 seconds. Clean recitation only — no lectures, no background music.",
        )
        if uploaded is not None:
            raw = uploaded.read()
            y = load_audio_bytes(raw, uploaded.name, meta["sr"])
            render_results(y, uploaded.name, uploaded.type, raw, meta, model, device)
        else:
            st.info("Upload an audio clip to predict.")

    with tab_samples:
        if not SAMPLES_DIR.exists():
            st.warning("No bundled samples found.")
        else:
            samples = sorted(SAMPLES_DIR.glob("*.wav"))
            if not samples:
                st.warning("No .wav files in samples/.")
            else:
                st.write("Pick a bundled test clip to see the model in action:")
                cols = st.columns(min(3, len(samples)))
                chosen = st.session_state.get("chosen_sample")
                for i, wav in enumerate(samples):
                    label = wav.stem.replace("_", " ")
                    if cols[i % len(cols)].button(label, key=f"btn_{wav.name}", use_container_width=True):
                        st.session_state["chosen_sample"] = str(wav)
                        chosen = str(wav)
                if chosen:
                    raw = Path(chosen).read_bytes()
                    y   = load_audio_bytes(raw, Path(chosen).name, meta["sr"])
                    st.markdown(f"**Sample:** `{Path(chosen).stem.replace('_', ' ')}`")
                    render_results(y, Path(chosen).name, "audio/wav", raw, meta, model, device)


if __name__ == "__main__":
    main()
