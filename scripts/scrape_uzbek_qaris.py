"""Download Uzbek qari recitations from YouTube using yt-dlp.

Uses search queries. Filters by duration to skip long lectures.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "raw_uzbek"
OUT.mkdir(parents=True, exist_ok=True)

# key = qari folder, value = (yt search query, how many results to fetch)
QARIS = {
    "Muhammadyusuf_Muhammad_Sodiq": ("Muhammadyusuf Muhammad Sodiq qori Quran", 8),
    "Abror_Mukhtor_Aliy":           ("Abror Mukhtor Aliy Quran tilovat", 8),
    "Rahmatullo_Qori":              ("Rahmatulloh qori Quran tilovat", 8),
    "Uzbek_Qori_Nuriddin":          ("Nuriddin qori tilovat Quran uzbek", 6),
    "Uzbek_Qori_Sardor":            ("Sardor qori Quran tilovat", 6),
}

# clips 60–600 seconds — long enough for signal, short enough to skip dars/lecture
YTDLP_ARGS = [
    sys.executable, "-m", "yt_dlp",
    "--extract-audio",
    "--audio-format", "mp3",
    "--audio-quality", "64K",
    "--match-filter", "duration >= 60 & duration <= 600",
    "--no-playlist",
    "--ignore-errors",
    "--no-warnings",
    "--restrict-filenames",
    "-o", "%(title).80s.%(ext)s",
]


def download_qari(folder: str, query: str, n: int) -> None:
    out_dir = OUT / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = YTDLP_ARGS + [
        "--paths", str(out_dir),
        f"ytsearch{n}:{query}",
    ]
    print(f"\n[{folder}] search: {query!r}  wanted={n}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-400:] if result.stdout else "")
    if result.returncode != 0:
        print("stderr tail:", result.stderr[-400:], file=sys.stderr)
    got = len(list(out_dir.glob("*.mp3")))
    print(f"[{folder}] files on disk: {got}")


def main() -> None:
    for folder, (query, n) in QARIS.items():
        download_qari(folder, query, n)
    print("\nDone. Total files per qari:")
    for folder in QARIS:
        count = len(list((OUT / folder).glob("*.mp3")))
        print(f"  {folder:40s} {count} files")


if __name__ == "__main__":
    main()
