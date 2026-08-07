"""Download sample surahs from everyayah.com for extra reciters.

Pulls one Juz (~a fraction of the Quran) per reciter to save disk.
Adjust SURAH_RANGE if more data is wanted.
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parent.parent / "data" / "raw_extra"
OUT.mkdir(parents=True, exist_ok=True)

# key = folder name we use, value = everyayah reciter dir
RECITERS = {
    "Mishary_Alafasy":      "Alafasy_128kbps",
    "Abdul_Basit":          "Abdul_Basit_Murattal_192kbps",
    "Husary":               "Husary_128kbps",
    "Minshawy":             "Minshawy_Murattal_128kbps",
    "Muhammad_Jibreel":     "Muhammad_Jibreel_64kbps",
    "Abu_Bakr_Shatri":      "Abu_Bakr_Ash-Shaatree_64kbps",
    "Ahmed_al_Ajamy":       "Ahmed_ibn_Ali_al-Ajamy_128kbps",
    "Fares_Abbad":          "Fares_Abbad_64kbps",
    "Hani_Rifai":           "Hani_Rifai_192kbps",
    "Salah_Bukhatir":       "Salaah_AbdulRahman_Bukhatir_128kbps",
    "Salah_Al_Budair":      "Salah_Al_Budair_128kbps",
    "Abdullah_Al_Juhany":   "Abdullaah_3awwaad_Al-Juhaynee_128kbps",
    "Muhsin_Al_Qasim":      "Muhsin_Al_Qasim_192kbps",
    "Khalid_Al_Qahtani":    "Khaalid_Abdullaah_al-Qahtaanee_192kbps",
    "Ibrahim_Akhdar":       "Ibrahim_Akhdar_64kbps",
}

BASE = "https://everyayah.com/data/{reciter}/{surah:03d}{ayah:03d}.mp3"
AYAH_COUNTS = {  # a few well-known surahs, cover ~5-8 minutes of audio each
    78: 40,  # An-Naba
    79: 46,  # An-Nazi'at
    80: 42,  # Abasa
    81: 29,  # At-Takwir
    82: 19,
    83: 36,
    84: 25,
    85: 22,
    86: 17,
    87: 19,
    88: 26,
    89: 30,
    90: 20,
    91: 15,
    92: 21,
    93: 11,
    94: 8,
    95: 8,
    96: 19,
    97: 5,
    98: 8,
    99: 8,
    100: 11,
    101: 11,
    102: 8,
    103: 3,
    104: 9,
    105: 5,
    106: 4,
    107: 7,
    108: 3,
    109: 6,
    110: 3,
    111: 5,
    112: 4,
    113: 5,
    114: 6,
}  # Juz 30 (Amma) — small, evenly sized, per-reciter ~200 MB total or less


def fetch(reciter_folder: str, reciter_dir: str, surah: int, ayah: int) -> tuple[str, bool, int]:
    dest = OUT / reciter_folder / f"{surah:03d}{ayah:03d}.mp3"
    if dest.exists() and dest.stat().st_size > 1000:
        return (str(dest), True, dest.stat().st_size)
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = BASE.format(reciter=reciter_dir, surah=surah, ayah=ayah)
    try:
        r = requests.get(url, timeout=45)
        if r.status_code == 200 and len(r.content) > 1000:
            dest.write_bytes(r.content)
            return (str(dest), True, len(r.content))
    except Exception as exc:  # noqa: BLE001
        return (str(dest), False, 0)
    return (str(dest), False, 0)


def main() -> None:
    tasks = []
    for folder, remote in RECITERS.items():
        for surah, n_ayah in AYAH_COUNTS.items():
            for ayah in range(1, n_ayah + 1):
                tasks.append((folder, remote, surah, ayah))
    print(f"Total requests: {len(tasks)} across {len(RECITERS)} reciters")

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
            if i % 100 == 0:
                print(f"  {i}/{len(tasks)} | ok={ok} fail={fail} | {bytes_ok/1e6:.1f} MB")
    print(f"\nDone. ok={ok} fail={fail}  total ~{bytes_ok/1e6:.1f} MB")


if __name__ == "__main__":
    main()
