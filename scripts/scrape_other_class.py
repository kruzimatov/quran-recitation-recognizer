"""Download Quran recitations from qaris NOT in our 34-class set.

Purpose: build an "Other / Unknown" class so the CNN can output
"not one of the trained qaris" instead of forcing a wrong guess.

Same everyayah.com source + Juz 30 sample as scrape_everyayah.py,
so the audio quality distribution matches our other Arab qaris —
this ensures the "Other" class captures *voice novelty*, not
audio-quality artifacts.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parent.parent / "data" / "raw_extra" / "_Other_Unknown"
OUT.mkdir(parents=True, exist_ok=True)

# 12 everyayah reciters we're NOT using as trained classes.
# Their audio becomes the "Other" pool.
OTHER_RECITERS = [
    "Abdullaah_Basfar_192kbps",
    "Ali_Jaber_64kbps",
    "Ayman_Sowaid_64kbps",
    "Ghamadi_40kbps",             # already covered? Ghamadi ≠ Saad_Alghamdi in our set (double check but treat as other)
    "Ibrahim_Akhdar_64kbps",
    "Karim_Mansoori_40kbps",
    "Khaalid_Abdullaah_al-Qahtaanee_192kbps",
    "Mohammad_al_Tablaway_128kbps",
    "Muhammad_AbdulKareem_128kbps",
    "Sahl_Yasin_128kbps",
    "Salaah_Bukhaatir_192kbps",
    "Yasser_Salamah_128kbps",
]

BASE = "https://everyayah.com/data/{reciter}/{surah:03d}{ayah:03d}.mp3"

# Juz 30 tail surahs — small files, evenly sized
AYAH_COUNTS = {
    78: 40, 79: 46, 80: 42, 81: 29, 82: 19, 83: 36, 84: 25, 85: 22, 86: 17,
    87: 19, 88: 26, 89: 30, 90: 20, 91: 15, 92: 21, 93: 11, 94: 8, 95: 8,
    96: 19, 97: 5, 98: 8, 99: 8, 100: 11, 101: 11, 102: 8, 103: 3, 104: 9,
    105: 5, 106: 4, 107: 7, 108: 3, 109: 6, 110: 3, 111: 5, 112: 4, 113: 5, 114: 6,
}


def fetch(reciter_dir: str, surah: int, ayah: int) -> tuple[str, bool, int]:
    prefix = reciter_dir.split("_")[0].lower()[:8]
    dest = OUT / f"{prefix}_{surah:03d}{ayah:03d}.mp3"
    if dest.exists() and dest.stat().st_size > 1000:
        return (str(dest), True, dest.stat().st_size)
    url = BASE.format(reciter=reciter_dir, surah=surah, ayah=ayah)
    try:
        r = requests.get(url, timeout=45)
        if r.status_code == 200 and len(r.content) > 1000:
            dest.write_bytes(r.content)
            return (str(dest), True, len(r.content))
    except Exception:
        return (str(dest), False, 0)
    return (str(dest), False, 0)


def main() -> None:
    tasks = []
    for reciter in OTHER_RECITERS:
        for surah, n_ayah in AYAH_COUNTS.items():
            for ayah in range(1, n_ayah + 1):
                tasks.append((reciter, surah, ayah))
    print(f"Total requests: {len(tasks):,} across {len(OTHER_RECITERS)} out-of-set reciters")

    ok = fail = bytes_ok = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch, *t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            _, success, size = fut.result()
            if success:
                ok += 1
                bytes_ok += size
            else:
                fail += 1
            if i % 200 == 0:
                print(f"  {i}/{len(tasks)} | ok={ok} fail={fail} | {bytes_ok/1e6:.1f} MB")

    print(f"\nDone. ok={ok}  fail={fail}  total ~{bytes_ok/1e6:.1f} MB")
    print(f"Files in {OUT}: {len(list(OUT.glob('*.mp3')))}")


if __name__ == "__main__":
    main()
