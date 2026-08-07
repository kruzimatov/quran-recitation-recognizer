# Quran Qari (Reciter) Classifier

Shazam-style: upload a Quran recitation clip → returns the top-3 predicted qaris with confidence.

**34 qaris supported.** PyTorch CNN on mel-spectrograms — **test accuracy 97.35%, top-3 99.80%.**

---

## Approaches compared

| approach | features | model | test accuracy | top-3 |
|---|---|---|---|---|
| Classical ML | 126 MFCC statistics | Logistic Regression (tuned) | 74.5% | 85.1% |
| **Deep learning** | **128×256 mel-spectrogram** | **PyTorch CNN (100k params)** | **97.35%** | **99.80%** |

Both approaches are in the repo. CNN ships.

---

## Dataset

Assembled from three sources into a unified format (5-second clips, mono, 16 kHz, peak-normalized):

- **12 Arab qaris** — Kaggle `mohammedalrajeh/quran-recitations-for-audio-classification` (pre-segmented WAV)
- **13 more Arab qaris** — scraped from [everyayah.com](https://everyayah.com), Juz 30 sample per qari
- **9 Uzbek qaris** — Alijon Qori, Hasanxon Yahyo, Husaynxon Yahyo, Ilyos Qori Ibn Oʻgʻli, Muhammadjon Temirov, Muhammadloiq Qori, Rahmatulloh Qori (Ustoz Obidov), Uzbek Qori Ubaydulloh, Yoʻldoshbek Ibrohim — manually collected (islom.uz + user's own audio)

Final: **13,561 clips across 34 qaris.**

---

## Layout

```
final project/
├── data/                              # excluded from git; regenerate via scripts
│   ├── base/                          # Kaggle 12 qaris (auto-downloaded)
│   ├── raw_extra/                     # everyayah.com scrape
│   ├── raw_uzbek/<Qari>/*.mp3         # your manual drops
│   ├── clips/<Qari>/*.wav             # unified 5s WAV clips
│   ├── features.parquet               # MFCC feature matrix (for sklearn)
│   └── labels.csv                     # clip → reciter mapping
├── scripts/
│   ├── scrape_everyayah.py            # Arab reciter downloader
│   ├── scrape_uzbek_qaris.py          # yt-dlp Uzbek scraper (needs cookies)
│   └── segment_and_features.py        # unify audio + extract MFCC
├── quran_qari_classifier.ipynb        # classical ML pipeline (sklearn + XGBoost)
├── quran_qari_cnn.ipynb               # deep learning pipeline (PyTorch CNN)
├── *_executed.ipynb                   # notebooks with outputs baked in
├── app.py                             # Streamlit app (uses the CNN)
├── requirements.txt
├── qari_cnn.pt                        # trained CNN weights (0.42 MB)
├── qari_cnn_meta.joblib               # label encoder + config for CNN
└── qari_artifacts.joblib              # sklearn LogReg model + scaler (fallback)
```

---

## Pipeline

1. Download base dataset via Kaggle CLI.
2. Scrape extra Arab reciters: `python scripts/scrape_everyayah.py`.
3. Drop Uzbek qari mp3s into `data/raw_uzbek/<QariName>/`.
4. Segment + extract features: `python scripts/segment_and_features.py`.
5. Train:
   - Classical: run `quran_qari_classifier.ipynb`.
   - Deep learning: run `quran_qari_cnn.ipynb` (uses Apple MPS / CUDA / CPU auto).
6. Deploy: `streamlit run app.py` or push to Streamlit Cloud.

---

## CNN architecture

Compact 4-block CNN:

```
Conv2d(1→16, 3×3, pad 1) → BatchNorm → ReLU → MaxPool 2×2
Conv2d(16→32, 3×3, pad 1) → BatchNorm → ReLU → MaxPool 2×2
Conv2d(32→64, 3×3, pad 1) → BatchNorm → ReLU → MaxPool 2×2
Conv2d(64→128, 3×3, pad 1) → BatchNorm → ReLU → AdaptiveAvgPool 1×1
Flatten → Dropout 0.3 → Linear(128 → 34)
```

Trained 24 epochs with early stopping, Adam (lr=1e-3), batch 32, cross-entropy loss. Total time: ~8 min on Apple M-series MPS.

---

## Streamlit app

- User uploads audio (mp3/wav/m4a/ogg/flac).
- Audio → 5s segments → mel-spectrograms → CNN → probability per class.
- Per-clip probabilities averaged for the final ranking.
- Displays top-3 qaris + waveform + mel-spectrogram of the input.

**Local:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

**Cloud:**
1. Push to a public GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → pick this repo → Deploy.

---

## Notes

- Uzbek qari audio was gathered manually; the scraper script exists for reference but YouTube blocks yt-dlp without cookies.
- Classical vs CNN gap: 74.5% → 97.35% — proof that end-to-end representation learning beats hand-crafted MFCCs on this task.
- Adding more qaris is a matter of dropping folders into `data/raw_uzbek/` (or wherever), rerunning the segmenter, and retraining the CNN.
