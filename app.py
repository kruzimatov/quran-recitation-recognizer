"""Streamlit app — Quran Qari Classifier (CNN, Shazam-style)."""

from __future__ import annotations

import io
import tempfile
from datetime import datetime
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

MODEL_PATH  = Path(__file__).with_name("qari_cnn.pt")
META_PATH   = Path(__file__).with_name("qari_cnn_meta.joblib")
SAMPLES_DIR = Path(__file__).with_name("samples")


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
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


@st.cache_resource
def load_artifacts():
    meta = joblib.load(META_PATH)
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
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


# ---------- Shazam-style circular gauge ---------------------------------
def confidence_gauge(pct: float, label: str) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 3.5), subplot_kw={"aspect": "equal"})
    ax.pie(
        [pct, 100 - pct],
        colors=["#2E8B57", "#EBE7D8"],
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.22, "edgecolor": "white", "linewidth": 2},
    )
    ax.text(0, 0.05, f"{pct:.0f}%", ha="center", va="center", fontsize=28, fontweight="bold", color="#1B2A20")
    ax.text(0, -0.25, label, ha="center", va="center", fontsize=9, color="#5c6579")
    ax.set(xticks=[], yticks=[])
    for s in ax.spines.values():
        s.set_visible(False)
    st.pyplot(fig, use_container_width=False)


def shazam_card(name: str, pct: float) -> None:
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #2E8B57 0%, #1B4A32 100%);
            padding: 32px 24px; border-radius: 18px; text-align: center;
            box-shadow: 0 8px 24px rgba(46,139,87,0.25); color: #F7F5EE;
            margin-bottom: 8px;">
            <div style="font-size: 12px; letter-spacing: 3px; text-transform: uppercase; opacity: 0.7;">
                Detected qari
            </div>
            <div style="font-size: 34px; font-weight: 700; margin-top: 8px; line-height: 1.1;">
                {name}
            </div>
            <div style="font-size: 14px; margin-top: 12px; opacity: 0.85;">
                {pct:.1f}% confidence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------- render results (Shazam + classic) ---------------------------
def render_results(y: np.ndarray, name: str, mime: str, raw: bytes, meta: dict, model: nn.Module, device: str) -> None:
    st.audio(raw, format=mime or "audio/mpeg")
    with st.spinner("Analyzing…"):
        seg_probs, _ = predict_per_segment(y, meta, model, device)

    if seg_probs.shape[0] == 0:
        st.error("Audio too short after trimming silence. Try a longer clip (≥ 5 s).")
        return

    classes = np.array(meta["classes"])
    mean_probs = seg_probs.mean(axis=0)
    order = mean_probs.argsort()[::-1]
    top_idx = order[:3]
    top_names = classes[top_idx]
    top_scores = mean_probs[top_idx]

    # log recent prediction
    st.session_state.setdefault("history", [])
    st.session_state["history"].insert(0, {
        "when": datetime.now().strftime("%H:%M:%S"),
        "file": name,
        "top": top_names[0],
        "conf": float(top_scores[0]),
    })
    st.session_state["history"] = st.session_state["history"][:5]

    # Shazam card + circular gauge
    left, right = st.columns([2, 1])
    with left:
        shazam_card(top_names[0].replace("_", " "), top_scores[0] * 100)
    with right:
        confidence_gauge(top_scores[0] * 100, "confidence")

    with st.expander("Not this one? — see runner-ups"):
        c1, c2 = st.columns(2)
        c1.metric(top_names[1].replace("_", " "), f"{top_scores[1]*100:.1f}%")
        c2.metric(top_names[2].replace("_", " "), f"{top_scores[2]*100:.1f}%")

    st.divider()

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

    # per-segment heatmap
    if seg_probs.shape[0] > 1:
        st.subheader("Per-segment probability across time")
        heat = seg_probs[:, top5_idx].T
        fig2, ax2 = plt.subplots(figsize=(9, max(2.5, 0.4 * len(top5_idx))))
        im = ax2.imshow(heat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax2.set_yticks(range(len(top5_idx))); ax2.set_yticklabels(classes[top5_idx])
        ax2.set_xticks(range(seg_probs.shape[0]))
        ax2.set_xticklabels([f"{i*5}-{(i+1)*5}s" for i in range(seg_probs.shape[0])], rotation=45, ha="right")
        ax2.set_title("Probability per 5-second segment")
        fig2.colorbar(im, ax=ax2, label="probability")
        st.pyplot(fig2)

    # audio inspection
    st.subheader("Audio inspection")
    sr = meta["sr"]
    fig3, axes = plt.subplots(2, 1, figsize=(10, 5))
    librosa.display.waveshow(y, sr=sr, ax=axes[0], color="#2E8B57")
    axes[0].set(title=f"Waveform — {name}", xlabel="time (s)")
    axes[0].spines[["top", "right"]].set_visible(False)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), sr=sr, x_axis="time", y_axis="mel", ax=axes[1], cmap="magma")
    axes[1].set(title="Mel spectrogram")
    plt.tight_layout()
    st.pyplot(fig3)

    with st.expander("All 34 qari probabilities"):
        st.dataframe(
            pd.DataFrame({"qari": classes[order], "probability": mean_probs[order]})
              .assign(probability=lambda d: d["probability"].map(lambda x: f"{x*100:.2f}%")),
            use_container_width=True, height=420,
        )


# ---------- pages -------------------------------------------------------
def page_home(meta, model, device) -> None:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Qaris", len(meta["classes"]))
    m2.metric("Test accuracy", f"{meta['test_acc']*100:.1f}%")
    m3.metric("Top-3 accuracy", f"{meta['test_top3']*100:.1f}%")
    m4.metric("Model size", "0.4 MB")
    st.divider()

    tab_upload, tab_samples = st.tabs(["📤 Upload", "🎧 Try a sample"])

    with tab_upload:
        uploaded = st.file_uploader(
            "Choose an audio file",
            type=["mp3", "wav", "m4a", "ogg", "flac"],
            help="Any length ≥ 5 seconds. Clean recitation only.",
        )
        if uploaded is not None:
            raw = uploaded.read()
            y = load_audio_bytes(raw, uploaded.name, meta["sr"])
            render_results(y, uploaded.name, uploaded.type, raw, meta, model, device)
        else:
            st.info("Upload an audio clip to predict.")

    with tab_samples:
        if not SAMPLES_DIR.exists() or not list(SAMPLES_DIR.glob("*.wav")):
            st.warning("No bundled samples found.")
        else:
            samples = sorted(SAMPLES_DIR.glob("*.wav"))
            st.write("Pick a bundled test clip:")
            cols = st.columns(min(3, len(samples)))
            for i, wav in enumerate(samples):
                if cols[i % len(cols)].button(wav.stem.replace("_", " "), key=f"btn_{wav.name}", use_container_width=True):
                    st.session_state["chosen_sample"] = str(wav)
            chosen = st.session_state.get("chosen_sample")
            if chosen:
                raw = Path(chosen).read_bytes()
                y = load_audio_bytes(raw, Path(chosen).name, meta["sr"])
                st.markdown(f"**Sample:** `{Path(chosen).stem.replace('_', ' ')}`")
                render_results(y, Path(chosen).name, "audio/wav", raw, meta, model, device)


def page_record(meta, model, device) -> None:
    st.subheader("🎤 Record recitation")
    st.caption(
        "Tap the mic → play the qari's recitation from your phone / speaker next to the laptop mic "
        "(or recite yourself) for at least 5 seconds → tap to stop. Model predicts the qari."
    )
    st.info(
        "**Tip.** Model was trained with mic-path augmentation (noise + reverb + EQ + low-pass), "
        "so it handles laptop-mic recordings reasonably well. For the cleanest possible test, "
        "still prefer the **📤 Upload** tab on the Home page with a downloaded mp3."
    )

    # big pulsing pill on ONLY the record/stop trigger (first button in the widget);
    # never touches playback toolbar buttons that appear after recording.
    st.markdown(
        """
        <style>
        [data-testid="stAudioInput"] button:first-of-type {
            background: #2E8B57 !important;
            color: white !important;
            border-radius: 999px !important;
            padding: 14px 22px !important;
            font-weight: 600 !important;
            box-shadow: 0 0 0 rgba(46,139,87,0.6);
            animation: pulse 1.8s infinite;
        }
        [data-testid="stAudioInput"] button:first-of-type svg {
            color: white !important; fill: white !important;
            width: 22px !important; height: 22px !important;
        }
        [data-testid="stAudioInput"] button:first-of-type:hover {
            background: #226b43 !important;
            transform: scale(1.03);
            transition: transform 120ms ease;
        }
        @keyframes pulse {
            0%   { box-shadow: 0 0 0 0 rgba(46,139,87,0.55); }
            70%  { box-shadow: 0 0 0 18px rgba(46,139,87,0); }
            100% { box-shadow: 0 0 0 0 rgba(46,139,87,0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if hasattr(st, "audio_input"):
        audio_file = st.audio_input("Tap to record")
        if audio_file is None:
            st.info(
                "Waiting for recording. Grant mic permission when the browser asks, "
                "then tap the mic and **play the qari's recitation from your phone / speaker "
                "next to the laptop mic** for at least 5 seconds, then tap stop. "
                "You can also recite yourself — but the model was trained on famous qaris, "
                "so playback of their recordings works best."
            )
            return
        raw = audio_file.getvalue()
    else:
        # fallback for older Streamlit versions
        try:
            from streamlit_mic_recorder import mic_recorder
        except ImportError:
            st.error("Recording requires Streamlit ≥ 1.36 or `streamlit-mic-recorder`.")
            return
        rec = mic_recorder(start_prompt="🔴 Start recording", stop_prompt="⏹ Stop",
                           just_once=False, use_container_width=True, format="wav", key="micrec")
        if not rec or not rec.get("bytes"):
            st.info("Waiting for recording.")
            return
        raw = rec["bytes"]

    if not raw or len(raw) < 1000:
        st.warning("Recording seems empty. Try again — speak louder or record longer.")
        return

    with st.status("🎧 Listening… extracting mel spectrogram…", expanded=False) as status:
        y = load_audio_bytes(raw, "recording.wav", meta["sr"])
        status.update(label="🧠 CNN inferring…", state="running")
    render_results(y, "recording.wav", "audio/wav", raw, meta, model, device)


def page_gallery(meta) -> None:
    st.subheader("📚 Qari gallery")
    st.caption(f"All {len(meta['classes'])} reciters the model can recognise.")
    classes = meta["classes"]
    per_row = 4
    for row_start in range(0, len(classes), per_row):
        cols = st.columns(per_row)
        for j, name in enumerate(classes[row_start:row_start + per_row]):
            with cols[j]:
                st.markdown(
                    f"""
                    <div style="
                        background: #EBE7D8; border-radius: 12px; padding: 14px;
                        text-align: center; margin-bottom: 10px; min-height: 90px;
                        display: flex; align-items: center; justify-content: center;
                        border: 1px solid #d5cfbc;">
                        <div style="font-weight: 600; font-size: 14px; color: #1B2A20;">
                            {name.replace('_', ' ')}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                sample_path = SAMPLES_DIR / f"{name}.wav"
                if sample_path.exists():
                    st.audio(str(sample_path), format="audio/wav")


def page_about(meta) -> None:
    st.subheader("ℹ️ About this project")
    st.markdown(
        f"""
        **Task.** Predict which qari (Quran reciter) is reading from a short audio clip.

        **Dataset.** {len(meta['classes'])} reciters (Arab + Uzbek), 13,561 five-second clips at
        {meta['sr']} Hz mono, unified from Kaggle base + everyayah.com scrape + manual Uzbek collection.

        **Model.** A compact 4-block convolutional neural network in PyTorch, trained on
        {meta['n_mels']}×{meta['img_w']} mel-spectrograms, ~100 k parameters, 0.4 MB weight file.

        **Metrics on held-out test set.**
        - Accuracy: **{meta['test_acc']*100:.2f}%**
        - Top-3 accuracy: **{meta['test_top3']*100:.2f}%**
        - Best validation accuracy: **{meta['best_val_acc']*100:.2f}%**

        **Compared with classical ML.** Same 34-class task, hand-crafted 126-dim MFCC statistics fed
        into Logistic Regression reached 74.5% accuracy. End-to-end deep learning gains +23 pp.

        **Deployment.** Streamlit app loads `qari_cnn.pt` + `qari_cnn_meta.joblib`. Runs on CPU
        in Streamlit Cloud, ~1-3 s per prediction.

        **Repo.** [github.com/kruzimatov/quran-recitation-recognizer](https://github.com/kruzimatov/quran-recitation-recognizer)
        """
    )


# ---------- main --------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Quran Qari Classifier", page_icon="📖", layout="wide")

    st.markdown(
        """
        <style>
        [data-testid="stExpandSidebarButton"], [data-testid="collapsedControl"] {
            background: #2E8B57 !important; color: #FFFFFF !important;
            border-radius: 0 8px 8px 0 !important; padding: 8px 10px !important;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.2) !important; top: 0.75rem !important;
        }
        [data-testid="stExpandSidebarButton"] svg, [data-testid="collapsedControl"] svg {
            color: #FFFFFF !important; width: 22px !important; height: 22px !important;
        }
        [data-testid="stExpandSidebarButton"]:hover, [data-testid="collapsedControl"]:hover {
            background: #226b43 !important; transform: scale(1.05);
            transition: transform 120ms ease;
        }
        [data-testid="stExpandSidebarButton"]::after, [data-testid="collapsedControl"]::after {
            content: " menu"; font-size: 13px; font-weight: 600; margin-left: 4px; vertical-align: middle;
        }
        div.stButton > button {
            border-radius: 10px; font-weight: 500;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    meta, model, device = load_artifacts()

    # ---- sidebar navigation (nav-pill buttons) ----
    st.markdown(
        """
        <style>
        .nav-pill { width: 100%; text-align: left !important; margin-bottom: 6px !important;
                    padding: 10px 14px !important; border-radius: 10px !important;
                    font-size: 15px !important; font-weight: 500 !important;
                    background: transparent !important; color: #1B2A20 !important;
                    border: 1px solid transparent !important; }
        .nav-pill:hover { background: #DED8C4 !important; }
        .nav-pill-active button { background: #2E8B57 !important; color: #FFFFFF !important;
                                  box-shadow: 0 2px 8px rgba(46,139,87,0.25) !important; }
        section[data-testid="stSidebar"] div.stButton > button {
            width: 100%; text-align: left; margin-bottom: 6px;
            padding: 10px 14px; border-radius: 10px;
            font-size: 15px; font-weight: 500;
            background: transparent; color: #1B2A20; border: 1px solid transparent;
        }
        section[data-testid="stSidebar"] div.stButton > button:hover {
            background: #DED8C4; border-color: #DED8C4;
        }
        section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
            background: #2E8B57 !important; color: white !important;
            border-color: #2E8B57 !important;
            box-shadow: 0 2px 8px rgba(46,139,87,0.25);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    NAV = [("home", "🏠  Home"), ("record", "🎤  Record"),
           ("gallery", "📚  Gallery"), ("about", "ℹ️  About")]
    if "page" not in st.session_state:
        st.session_state["page"] = "home"

    with st.sidebar:
        st.markdown(
            "<div style='font-family: serif; font-size: 22px; font-weight: 700;"
            " margin: 4px 0 14px; color:#1B2A20;'>📖 Quran Qari</div>",
            unsafe_allow_html=True,
        )
        for key, label in NAV:
            if st.button(label, key=f"nav_{key}",
                         type=("primary" if st.session_state["page"] == key else "secondary")):
                st.session_state["page"] = key
                st.rerun()
        st.divider()

        st.header("Model")
        st.write("**Architecture:** 4-block CNN on mel-spectrograms")
        st.write("**Trainable params:** ~100 k")
        st.write(f"**Compute:** `{device}`")

        st.header("Performance")
        st.metric("Test accuracy", f"{meta['test_acc']*100:.2f} %")
        st.metric("Top-3 accuracy", f"{meta['test_top3']*100:.2f} %")
        st.divider()

        if st.session_state.get("history"):
            st.header("Recent")
            for h in st.session_state["history"]:
                st.text(f"{h['when']}  {h['top'][:18]}\n  {h['conf']*100:.0f}% · {h['file'][:18]}")

        st.divider()
        with st.expander(f"Supported qaris ({len(meta['classes'])})", expanded=False):
            for name in meta["classes"]:
                st.text(f"• {name.replace('_', ' ')}")

    page = st.session_state["page"]

    # ---- header ----
    st.title("📖 Quran Qari Classifier")
    st.caption(
        "Convolutional neural network on mel-spectrograms · "
        f"{len(meta['classes'])} qaris · test accuracy {meta['test_acc']*100:.1f}%"
    )

    if page == "home":
        page_home(meta, model, device)
    elif page == "record":
        page_record(meta, model, device)
    elif page == "gallery":
        page_gallery(meta)
    else:
        page_about(meta)


if __name__ == "__main__":
    main()
