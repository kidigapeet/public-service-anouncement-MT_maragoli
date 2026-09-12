#!/usr/bin/env python3
"""
build_trilingual_corpus.py
==========================
Builds an English-Ekegusii-Swahili parallel corpus.

Supersedes ``scrape_ekegusii_corpus.py`` (which produced English-Ekegusii pairs
against the archaic KJV). Changes that matter for downstream MT training:

  * THREE-WAY verse alignment (English / Ekegusii / Swahili) on shared USFM
    verse keys, so every row is a genuine triple.
  * MODERN English source. Default is the Berean Standard Bible (public domain,
    contemporary register), falling back to the World English Bible. The KJV is
    only used as a last resort because its archaic register actively harms
    transfer to a plain-English target domain such as PSAs.
  * MODERN Swahili source: "Neno: Bibilia Takatifu" (Biblica open licence),
    falling back to the 1850 union version.
  * Proper USFM cleaning: footnotes and cross-references are removed as whole
    spans instead of having their tags stripped and their text merged into the
    verse.
  * Verse-range handling: \\v 1-2 spans are only kept when all three
    translations use the identical span, which prevents silent misalignment.
  * Length-ratio filtering to catch residual alignment errors.

Outputs (written to --output-dir, default "data/"):
  bible_en_guz_swh.csv      3-way verse triples          <- main training file
  storybooks_en_guz_swh.csv page-aligned storybook triples (needs human review)
  corpus_manifest.json      source IDs, counts, filter statistics

Usage
-----
    pip install requests pandas
    python build_trilingual_corpus.py
    python build_trilingual_corpus.py --english-id engwebp --no-storybooks

Runs identically on a local machine and in Google Colab.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import zipfile
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

EBIBLE_ZIP = "https://ebible.org/Scriptures/{tid}_usfm.zip"

# Ordered by preference. The first ID that downloads and parses wins.
SOURCES: "OrderedDict[str, List[Tuple[str, str]]]" = OrderedDict(
    english=[
        ("engbsb", "Berean Standard Bible (public domain, modern)"),
        ("engwebp", "World English Bible, Protestant (public domain, modern)"),
        ("eng-web", "World English Bible (public domain, modern)"),
        ("engwebpb", "World English Bible, British (public domain, modern)"),
        ("eng-kjv2006", "King James Version 2006 (ARCHAIC - last resort)"),
    ],
    ekegusii=[
        ("guz", "Ekegusii Revised Bible 2021, Bible Society of Kenya"),
    ],
    swahili=[
        ("swhonen", "Neno: Bibilia Takatifu (Biblica open licence, modern)"),
        ("swh1850", "Biblia Takatifu 1850 union version (ARCHAIC - fallback)"),
    ],
)

ARCHAIC_IDS = {"eng-kjv2006", "swh1850"}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; AcademicResearchBot/1.0; "
        "+https://github.com/SamAbr/PSA-MT)"
    )
}

STORYBOOK_ZIP = "https://codeload.github.com/global-asp/asp-source/zip/refs/heads/master"

# Filter thresholds. Bantu languages are agglutinative and run longer than
# English, so the upper bounds are deliberately generous.
MIN_WORDS_EN = 3
MIN_WORDS_TARGET = 2
RATIO_BOUNDS = {
    ("ekegusii", "english"): (0.45, 3.20),
    ("swahili", "english"): (0.45, 3.20),
    ("ekegusii", "swahili"): (0.40, 2.60),
}

# Cross-reference / footnote residue that survives as whole "verses"
NOISE_RE = re.compile(
    r"^-\s+"                # footnote dash
    r"|^\d+\s+[A-Z][a-z]"   # "12 Matayo" style cross-reference
    r"|^\d+:\d+"            # bare chapter:verse
)


# ---------------------------------------------------------------------------
# DOWNLOAD
# ---------------------------------------------------------------------------

def fetch_zip(url: str, timeout: int = 120) -> Optional[zipfile.ZipFile]:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        return zipfile.ZipFile(io.BytesIO(resp.content))
    except Exception as exc:
        print(f"      failed: {exc}")
        return None


def fetch_translation(candidates: List[Tuple[str, str]],
                      label: str,
                      forced_id: Optional[str] = None) -> Tuple[Optional[zipfile.ZipFile], Optional[str]]:
    """Try each candidate translation ID until one downloads."""
    if forced_id:
        candidates = [(forced_id, "user-specified")] + [c for c in candidates if c[0] != forced_id]

    print(f"\n  [{label}]")
    for tid, description in candidates:
        url = EBIBLE_ZIP.format(tid=tid)
        print(f"    trying {tid:<14} {description}")
        zf = fetch_zip(url)
        if zf is not None:
            print(f"      OK ({len(zf.namelist())} files)")
            if tid in ARCHAIC_IDS:
                print(f"      WARNING: '{tid}' is an archaic register. Downstream "
                      f"models will learn that register. Prefer a modern edition.")
            return zf, tid
    print(f"    ERROR: no usable source found for {label}")
    return None, None


# ---------------------------------------------------------------------------
# USFM PARSING
# ---------------------------------------------------------------------------

# Whole spans that must be deleted with their contents, not just untagged.
SPAN_MARKERS = ["f", "fe", "x", "ef", "ex", "fig", "rq"]

CHAPTER_RE = re.compile(r"\\c\s+(\d+)\b")
VERSE_RE = re.compile(r"\\v\s+(\d+)(?:\s*[-\u2013]\s*(\d+))?[a-z]?\s*(.*)")
BOOK_ID_RE = re.compile(r"\\id\s+([A-Z0-9]{3})")


def clean_usfm(text: str) -> str:
    """Strip USFM markup, returning only the verse's own words."""
    if not text:
        return ""

    # 1. Delete footnotes / cross-references / figures entirely (contents too).
    for marker in SPAN_MARKERS:
        text = re.sub(rf"\\\+?{marker}\b.*?\\\+?{marker}\*", " ", text, flags=re.DOTALL)
    # Unterminated span at end of verse
    for marker in SPAN_MARKERS:
        text = re.sub(rf"\\\+?{marker}\b.*$", " ", text, flags=re.DOTALL)

    # 2. \w word|lemma="x" \w*  ->  word
    text = re.sub(r"\\\+?w\s+([^|\\]*?)(?:\|[^\\]*?)?\\\+?w\*", r"\1", text)

    # 3. Drop attribute residue (Strong's numbers, alignment attributes).
    text = re.sub(r'\|[^\s\\|]*="[^"]*"', " ", text)

    # 4. Remove all remaining markers, keeping their inline content.
    text = re.sub(r"\\\+?[a-zA-Z][a-zA-Z0-9\-]*\*?", " ", text)

    # 5. Tidy up.
    text = text.replace("\u00b6", " ").replace("|", " ")
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_usfm_zip(zf: zipfile.ZipFile, label: str) -> Tuple[Dict[str, str], Dict[str, Tuple[int, int]]]:
    """
    Parse an eBible USFM zip.

    Returns (verses, spans) where verses maps 'GEN_1_1' -> text and spans maps
    the same key -> (first_verse, last_verse) so that \\v 1-2 ranges can be
    compared across translations.
    """
    verses: Dict[str, str] = {}
    spans: Dict[str, Tuple[int, int]] = {}

    files = sorted(
        n for n in zf.namelist()
        if not n.startswith("__MACOSX")
        and n.rsplit(".", 1)[-1].lower() in ("usfm", "sfm", "txt")
    )
    if not files:
        print(f"    WARNING: no USFM files inside {label} zip")
        return verses, spans

    for fname in files:
        try:
            raw = zf.read(fname).decode("utf-8", errors="ignore")
        except Exception:
            continue

        match = BOOK_ID_RE.search(raw)
        if not match:
            continue
        book = match.group(1).upper()

        chapter = ""
        key = ""
        buf: List[str] = []

        def flush():
            if key and buf:
                cleaned = clean_usfm(" ".join(buf))
                if cleaned:
                    verses[key] = cleaned

        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue

            chap = CHAPTER_RE.match(line)
            if chap:
                flush()
                key, buf = "", []
                chapter = chap.group(1)
                continue

            vm = VERSE_RE.match(line)
            if vm and chapter:
                flush()
                start = int(vm.group(1))
                end = int(vm.group(2)) if vm.group(2) else start
                key = f"{book}_{chapter}_{start}"
                spans[key] = (start, end)
                buf = [vm.group(3)]
                continue

            # Skip structural lines that carry no verse text.
            if line.startswith("\\") and not key:
                continue

            if key:
                buf.append(line)

        flush()

    print(f"    parsed {len(verses):,} verses from {label}")
    return verses, spans


# ---------------------------------------------------------------------------
# ALIGNMENT
# ---------------------------------------------------------------------------

def align_three_way(parsed: Dict[str, Tuple[Dict[str, str], Dict[str, Tuple[int, int]]]],
                    stats: dict) -> pd.DataFrame:
    """Intersect verse keys across English, Ekegusii and Swahili."""
    en_v, en_s = parsed["english"]
    guz_v, guz_s = parsed["ekegusii"]
    swh_v, swh_s = parsed["swahili"]

    common = set(en_v) & set(guz_v) & set(swh_v)
    stats["verses_per_language"] = {
        "english": len(en_v), "ekegusii": len(guz_v), "swahili": len(swh_v),
    }
    stats["keys_in_all_three"] = len(common)
    print(f"\n  verse keys present in all three: {len(common):,}")

    rows = []
    span_mismatch = 0
    for key in sorted(common):
        # Only keep a verse when all three editions cover the identical span,
        # otherwise a merged "1-2" on one side is aligned to a bare "1".
        if not (en_s.get(key) == guz_s.get(key) == swh_s.get(key)):
            span_mismatch += 1
            continue
        book, chapter, verse = key.split("_")
        rows.append({
            "ref": f"{book} {chapter}:{verse}",
            "book": book,
            "chapter": int(chapter),
            "verse": int(verse),
            "english": en_v[key],
            "ekegusii": guz_v[key],
            "swahili": swh_v[key],
            "source": "bible_ebible",
        })

    stats["dropped_span_mismatch"] = span_mismatch
    print(f"  dropped for verse-span mismatch: {span_mismatch:,}")
    return pd.DataFrame(rows)


def apply_filters(df: pd.DataFrame, stats: dict, columns: List[str]) -> pd.DataFrame:
    """Empty / length / ratio / noise / duplicate filtering."""
    if df.empty:
        return df

    start = len(df)
    log = {}

    for col in columns:
        df[col] = df[col].fillna("").astype(str).str.strip()

    before = len(df)
    df = df[(df[columns] != "").all(axis=1)]
    log["empty_cell"] = before - len(df)

    for col in columns:
        df[f"{col[:3]}_words"] = df[col].str.split().str.len()

    before = len(df)
    minimum = {c: (MIN_WORDS_EN if c == "english" else MIN_WORDS_TARGET) for c in columns}
    mask = pd.Series(True, index=df.index)
    for col in columns:
        mask &= df[f"{col[:3]}_words"] >= minimum[col]
    df = df[mask]
    log["too_short"] = before - len(df)

    if "ekegusii" in columns:
        before = len(df)
        df = df[~df["ekegusii"].str.match(NOISE_RE, na=False)]
        log["noise_lines"] = before - len(df)

    before = len(df)
    ratio_mask = pd.Series(True, index=df.index)
    for (a, b), (low, high) in RATIO_BOUNDS.items():
        if a in columns and b in columns:
            ratio = df[a].str.len() / df[b].str.len().clip(lower=1)
            ratio_mask &= ratio.between(low, high)
    df = df[ratio_mask]
    log["length_ratio"] = before - len(df)

    before = len(df)
    df = df.drop_duplicates(subset=columns).reset_index(drop=True)
    log["duplicate_rows"] = before - len(df)

    stats["filters"] = log
    stats["rows_before_filters"] = start
    stats["rows_after_filters"] = len(df)

    print(f"\n  filtering: {start:,} -> {len(df):,}")
    for reason, count in log.items():
        if count:
            print(f"    -{count:>7,}  {reason}")
    return df


# ---------------------------------------------------------------------------
# SUPPLEMENTARY SOURCE: African Storybook Project
# ---------------------------------------------------------------------------

def harvest_storybooks() -> pd.DataFrame:
    """
    Page-align African Storybook titles that exist in Ekegusii, English and
    Swahili. Stories share a numeric ID prefix across language directories and
    are split into pages by '##' headings.

    This is the only non-biblical Ekegusii text available at any scale, so it is
    worth having - but the page alignment is structural, not verified. Output is
    written to its own file and flagged for human review.
    """
    print("\n=== Supplementary: African Storybook Project ===")
    zf = fetch_zip(STORYBOOK_ZIP, timeout=180)
    if zf is None:
        print("  could not download - skipping")
        return pd.DataFrame()

    def index(lang: str) -> Dict[str, str]:
        out = {}
        prefix = f"asp-source-master/{lang}/"
        for name in zf.namelist():
            if name.startswith(prefix) and name.endswith(".md"):
                base = os.path.basename(name)
                if base.upper().startswith("README"):
                    continue
                story_id = base.split("_")[0]
                if story_id.isdigit():
                    out[story_id] = name
        return out

    en_idx, guz_idx, sw_idx = index("en"), index("guz"), index("sw")
    shared = sorted(set(en_idx) & set(guz_idx) & set(sw_idx))
    print(f"  stories: en={len(en_idx)} guz={len(guz_idx)} sw={len(sw_idx)} "
          f"| in all three: {len(shared)}")

    # Every African Storybook file ends with a licence/credits block rendered as
    # a page ("* License: [CC-BY] * Text: ... * Illustration: ..."). It is
    # identical boilerplate in all three languages, so it aligns perfectly and
    # slips through structural checks - but it is metadata, not translation, and
    # would teach the model to emit credit lines.
    CREDITS_RE = re.compile(
        r"License:|Illustration:|Translation:|^\s*\*\s*Text:|Language:\s*\w+\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    def pages(path: str) -> List[str]:
        text = zf.read(path).decode("utf-8", errors="ignore")
        blocks = re.split(r"^##\s*$|^##\s+.*$", text, flags=re.MULTILINE)
        cleaned = []
        for block in blocks[1:]:  # blocks[0] is the title block
            block = re.sub(r"!\[.*?\]\(.*?\)", " ", block)
            block = re.sub(r"\s+", " ", block).strip()
            if not block or CREDITS_RE.search(block):
                continue
            cleaned.append(block)
        return cleaned

    rows, skipped = [], 0
    for story_id in shared:
        en_p, guz_p, sw_p = pages(en_idx[story_id]), pages(guz_idx[story_id]), pages(sw_idx[story_id])
        if not (len(en_p) == len(guz_p) == len(sw_p)) or not en_p:
            skipped += 1
            continue
        for i, (e, g, s) in enumerate(zip(en_p, guz_p, sw_p)):
            rows.append({
                "ref": f"asb_{story_id}_p{i:02d}",
                "story_id": story_id,
                "page": i,
                "english": e,
                "ekegusii": g,
                "swahili": s,
                "source": "african_storybook",
                "needs_review": True,
            })

    print(f"  stories with matching page counts: {len(shared) - skipped} "
          f"(skipped {skipped} for page-count mismatch)")
    print(f"  page-aligned triples: {len(rows)}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--english-id", default=None,
                        help="Force an eBible English translation ID (default: engbsb)")
    parser.add_argument("--swahili-id", default=None,
                        help="Force an eBible Swahili translation ID (default: swhonen)")
    parser.add_argument("--ekegusii-id", default=None,
                        help="Force an eBible Ekegusii translation ID (default: guz)")
    parser.add_argument("--no-storybooks", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    stats: dict = {"sources": {}}

    print("=" * 68)
    print("  English-Ekegusii-Swahili Parallel Corpus Builder")
    print("=" * 68)
    print("\n=== Downloading Bible sources from eBible.org ===")

    forced = {"english": args.english_id, "swahili": args.swahili_id,
              "ekegusii": args.ekegusii_id}
    parsed = {}
    for lang, candidates in SOURCES.items():
        zf, tid = fetch_translation(candidates, lang, forced.get(lang))
        if zf is None:
            print(f"\nFATAL: could not obtain a {lang} Bible. Aborting.")
            return 1
        stats["sources"][lang] = tid
        parsed[lang] = parse_usfm_zip(zf, f"{lang} ({tid})")

    print("\n=== Aligning ===")
    bible = align_three_way(parsed, stats)
    bible = apply_filters(bible, stats, ["english", "ekegusii", "swahili"])

    triples_path = os.path.join(args.output_dir, "bible_en_guz_swh.csv")
    bible.to_csv(triples_path, index=False, encoding="utf-8")
    print(f"\n  wrote {len(bible):,} triples -> {triples_path}")

    if not args.no_storybooks:
        books = harvest_storybooks()
        if not books.empty:
            books_path = os.path.join(args.output_dir, "storybooks_en_guz_swh.csv")
            books.to_csv(books_path, index=False, encoding="utf-8")
            print(f"  wrote {len(books):,} storybook triples -> {books_path}")
            stats["storybook_triples"] = len(books)

    stats["bible_triples"] = len(bible)
    manifest_path = os.path.join(args.output_dir, "corpus_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    print(f"  wrote manifest -> {manifest_path}")

    if not bible.empty:
        print("\n=== Sample triples ===")
        for _, row in bible.sample(min(3, len(bible)), random_state=0).iterrows():
            print(f"\n  [{row['ref']}]")
            print(f"    EN : {row['english'][:100]}")
            print(f"    GUZ: {row['ekegusii'][:100]}")
            print(f"    SWH: {row['swahili'][:100]}")

        print("\n=== Coverage by book (top 10) ===")
        for book, count in bible["book"].value_counts().head(10).items():
            print(f"    {book:<6} {count:>6,}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
