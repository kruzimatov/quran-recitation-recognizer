# Quran Qari (Reciter) Classifier

End-to-end audio classification: given a short recitation clip, predict which qari is reading. Shazam-style deployment.

## Dataset

- **Base:** 12 Arab qaris from Kaggle `mohammedalrajeh/quran-recitations-for-audio-classification` (pre-segmented WAV)
- **Extra:** 15 more Arab qaris scraped from [everyayah.com](https://everyayah.com) (Juz 30 sample per qari)
- **Uzbek:** 3–5 qaris added manually (YouTube blocks yt-dlp without cookies)

All audio unified: 5-second clips, mono, 16 kHz, peak-normalized.

## Layout

```
final project/
├── data/
│   ├── base/          # Kaggle base 12 qaris
│   ├── raw_extra/     # everyayah.com scrape
│   ├── raw_uzbek/     # manually added Uzbek qaris
│   ├── clips/         # unified 5s WAV clips
│   ├── features.parquet   # MFCC + delta + spectral features
│   └── labels.csv         # clip → reciter mapping
├── scripts/
│   ├── scrape_everyayah.py       # Arab reciters download
│   ├── scrape_uzbek_qaris.py     # YouTube (needs cookies)
│   └── segment_and_features.py   # unify audio + extract features
├── quran_qari_classifier.ipynb   # main notebook
├── app.py                        # Streamlit app (upload → predict)
├── requirements.txt
└── qari_artifacts.joblib         # model + scaler + label encoder (created by notebook)
```

## Pipeline

1. Download base dataset (Kaggle CLI).
2. Scrape extra Arab reciters (`python scripts/scrape_everyayah.py`).
3. Optionally add Uzbek qari mp3s into `data/raw_uzbek/<QariName>/`.
4. Segment + extract features (`python scripts/segment_and_features.py`).
5. Run the notebook end-to-end.
6. Deploy: push to GitHub → share.streamlit.io.

## Features

Per 5-second clip:
- 20 MFCC + 20 delta + 20 delta-delta → mean & std → 120 dims
- ZCR, spectral centroid, rolloff → mean & std → 6 dims
- **Total: 126 features per clip**

## Models compared

Logistic Regression, KNN, Random Forest, XGBoost. Metric: macro-F1 (imbalanced multi-class).

## Deployment

Streamlit app supports mp3/wav/m4a/ogg/flac upload. Returns top-3 qaris + confidence bars + waveform + mel spectrogram.
