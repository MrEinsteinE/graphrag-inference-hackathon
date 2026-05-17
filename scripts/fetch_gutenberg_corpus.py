#!/usr/bin/env python3
"""
Download public-domain plain text from Project Gutenberg into data/corpus/gutenberg/.

Usage (from repo root):
  python scripts/fetch_gutenberg_corpus.py

See https://www.gutenberg.org/policy/license.html — use polite rate limits.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import httpx

# (gutenberg_id, filename_stem) — IDs verified against Gutenberg epub cache URLs.
BOOKS: list[tuple[int, str]] = [
    (1342, "pride_and_prejudice_austen"),
    (2701, "moby_dick_melville"),
    (84, "frankenstein_shelley"),
    (11, "alice_in_wonderland_carroll"),
    (345, "dracula_stoker"),
    (1661, "sherlock_adventures_doyle"),
    (16328, "enchanted_april_von_arnim"),
    (16389, "room_with_a_view_forster"),
    (6130, "great_expectations_dickens"),
    (1513, "romeo_and_juliet_shakespeare"),
    (1260, "jane_eyre_bronte"),
    (768, "wuthering_heights_bronte"),
    (74, "treasure_island_stevenson"),
    (1952, "yellow_wallpaper_gilman"),
    (1232, "jekyll_hyde_stevenson"),
    (5200, "metamorphosis_kafka_bell_translation"),
    (244, "study_in_scarlet_doyle"),
    (1080, "modest_proposal_swift"),
    (2591, "awakening_chopin"),
    (15432, "poetics_aristotle_butcher"),
    (2610, "hunchback_of_notre_dame_hugo"),
    (158, "emma_austen"),
    (730, "oliver_twist_dickens"),
    (64317, "great_gatsby_fitzgerald"),
    (36034, "crime_and_punishment_dostoevsky"),
    (4300, "ulysses_joyce"),
    (4078, "picture_of_dorian_gray_wilde"),
    (883, "memoirs_of_sherlock_holmes_doyle"),
]

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "corpus" / "gutenberg"
LICENSE_NOTE = ROOT / "data" / "corpus" / "GUTENBERG_ATTRIBUTION.txt"

HEADERS = {"User-Agent": "GraphRAG-Hackathon-CorpusFetcher/1.0 (educational)"}


def _urls_for_id(gid: int) -> list[str]:
    return [
        f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}.txt",
    ]


def _strip_gutenberg_boilerplate(text: str) -> str:
    start_markers = (
        r"\*\*\*\s*START OF (?:THIS|THE) PROJECT GUTENBERG EBOOK",
        r"\*END\*THE SMALL PRINT",
    )
    end_markers = (
        r"\*\*\*\s*END OF (?:THIS|THE) PROJECT GUTENBERG EBOOK",
        r"End of (?:the )?Project Gutenberg",
    )
    for pat in start_markers:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            text = text[m.end() :]
            break
    for pat in end_markers:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            text = text[: m.start()]
            break
    return text.strip()


def download_one(client: httpx.Client, gid: int, stem: str) -> tuple[Path, int]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / f"{stem}.txt"
    if dest.exists() and dest.stat().st_size > 10_000:
        return dest, dest.stat().st_size

    last_err = None
    for url in _urls_for_id(gid):
        try:
            r = client.get(url, timeout=120.0, follow_redirects=True)
            if r.status_code != 200:
                last_err = f"{url} -> {r.status_code}"
                continue
            text = r.text
            if len(text) < 5000:
                last_err = f"{url} too short ({len(text)} chars)"
                continue
            text = _strip_gutenberg_boilerplate(text)
            dest.write_text(text, encoding="utf-8")
            return dest, len(text.encode("utf-8"))
        except Exception as e:
            last_err = str(e)
    raise RuntimeError(f"Gutenberg id {gid} ({stem}): {last_err}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LICENSE_NOTE.write_text(
        "Texts in gutenberg/ are from Project Gutenberg (public domain in the USA).\n"
        "https://www.gutenberg.org/policy/license.html\n",
        encoding="utf-8",
    )

    total_bytes = 0
    files: list[Path] = []
    with httpx.Client(headers=HEADERS) as client:
        for gid, stem in BOOKS:
            try:
                path, nbytes = download_one(client, gid, stem)
                files.append(path)
                total_bytes += nbytes
                print(f"OK {gid} -> {path.name} ({nbytes // 1024} KB)")
            except Exception as e:
                print(f"SKIP {gid} {stem}: {e}")
            time.sleep(2.0)

    if not files:
        print("No files downloaded.")
        return

    approx_tokens = sum(len(p.read_text(encoding="utf-8", errors="ignore")) for p in files) // 4
    print(f"\nDownloaded {len(files)} files, ~{total_bytes // 1_048_576} MiB")
    print(f"Rough token estimate (chars/4): {approx_tokens:,}")
    if approx_tokens < 2_000_000:
        print(
            "\nTarget ~2M+ tokens for hackathon Round 1: add more IDs to BOOKS in this script and re-run."
        )


if __name__ == "__main__":
    main()
