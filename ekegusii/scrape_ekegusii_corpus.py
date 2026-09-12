import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

"""
scrape_ekegusii_corpus.py
=========================
Scrapes all available English-Ekegusii parallel text from the internet
and exports a clean, deduplicated CSV to data/scraped_english_ekegusii.csv.

Sources
-------
1. eBible.org -- Ekegusii Revised Bible 2021 (USFM ZIP)  aligned verse-by-verse
                 with the KJV English Bible USFM ZIP.  ~31,000 verse pairs.
                 Free for research (c) 2021 Bible Society of Kenya.
2. 4laws.com  -- "The Four Spiritual Laws" booklet in English & Ekegusii.
                 ~80-120 sentence pairs.

Usage
-----
    pip install requests beautifulsoup4 pandas
    python scrape_ekegusii_corpus.py

Works identically on local machines and Google Colab.
"""

import io as _io
import os
import re
import zipfile

import pandas as pd
import requests
from bs4 import BeautifulSoup

OUTPUT_DIR  = "output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "scraped_english_ekegusii.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; AcademicResearchBot/1.0; "
        "+https://github.com/SamAbr/PSA-MT)"
    )
}


# ---------------------------------------------------------------------------
# DOWNLOAD HELPER
# ---------------------------------------------------------------------------

def fetch_zip(urls: list, label: str):
    """Try each URL until one returns a valid ZIP; return ZipFile or None."""
    for url in urls:
        try:
            print(f"  Downloading {label}:\n    {url}")
            r = requests.get(url, headers=HEADERS, timeout=90)
            r.raise_for_status()
            zf = zipfile.ZipFile(_io.BytesIO(r.content))
            print(f"  OK  {label}  ({len(r.content):,} bytes, "
                  f"{len(zf.namelist())} files inside)")
            return zf
        except Exception as exc:
            print(f"  FAIL  ({exc})")
    return None


# ---------------------------------------------------------------------------
# SOURCE 1 -- eBible USFM Bible verse alignment
# ---------------------------------------------------------------------------

# Confirmed URLs from ebible.org/find/details.php
EBIBLE_GUZ_URLS = [
    "https://ebible.org/Scriptures/guz_usfm.zip",     # USFM (preferred)
    "https://ebible.org/Scriptures/guz_readaloud.zip", # plain-text fallback
    "https://ebible.org/Scriptures/guz_html.zip",      # HTML fallback
]

EBIBLE_EN_URLS = [
    "https://ebible.org/Scriptures/eng-kjv2006_usfm.zip",
    "https://ebible.org/Scriptures/eng-kjv2006_readaloud.zip",
    "https://ebible.org/Scriptures/eng-kjv2006_html.zip",
]


def _strip_usfm_tags(text: str) -> str:
    """Remove USFM markers, Strong's numbers and extra whitespace from verse text."""
    # Remove Strong's concordance markers: |strong="H1234" or |strong="G5678"
    text = re.sub(r'\|strong="[HG]\d+"', "", text)
    # Remove USFM backslash tags
    text = re.sub(r"\\[a-zA-Z0-9*+\-]+\s*", " ", text)
    # Remove pilcrow / paragraph sign often added by KJV
    text = text.replace("¶", "").replace("  ", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()



def parse_usfm_zip(zf: zipfile.ZipFile, label: str) -> dict:
    """
    Parse a USFM-format eBible ZIP.
    Returns {  'GEN_1_1': 'verse text', ... }
    Falls back to readaloud plain-text if no USFM tags found.
    """
    verses = {}
    txt_files = sorted(
        n for n in zf.namelist()
        if not n.startswith("__MACOSX")
        and n.split(".")[-1].lower() in ("usfm", "sfm", "txt")
    )
    if not txt_files:
        print(f"  WARN  No text files in {label} zip. Contents: {zf.namelist()[:10]}")
        return verses

    print(f"  Parsing {len(txt_files)} files in {label} zip...")

    for fname in txt_files:
        try:
            raw = zf.read(fname).decode("utf-8", errors="ignore")
        except Exception:
            continue

        # Extract book code from filename or \id tag
        book_tag = ""
        id_match = re.search(r"\\id\s+([A-Z0-9]{2,3})", raw)
        if id_match:
            book_tag = id_match.group(1).upper()
        else:
            # Guess from filename e.g. "01GEN.usfm" or "GEN.txt"
            bm = re.search(r"([A-Z]{2,3})(?:\.|_)", os.path.basename(fname).upper())
            if bm:
                book_tag = bm.group(1)

        if not book_tag:
            continue

        chapter = ""
        pending_verse = ""
        pending_key   = ""

        for line in raw.splitlines():
            line = line.strip()

            # Chapter
            ch = re.match(r"\\c\s+(\d+)\b", line)
            if ch:
                # Flush any pending verse
                if pending_key and pending_verse:
                    verses[pending_key] = _strip_usfm_tags(pending_verse)
                    pending_key = pending_verse = ""
                chapter = ch.group(1)
                continue

            # Verse start
            vm = re.match(r"\\v\s+(\d+)\s*(.*)", line)
            if vm and chapter:
                # Flush previous
                if pending_key and pending_verse:
                    verses[pending_key] = _strip_usfm_tags(pending_verse)
                pending_key   = f"{book_tag}_{chapter}_{vm.group(1)}"
                pending_verse = vm.group(2)
                continue

            # Continuation of current verse
            if pending_key:
                pending_verse += " " + line

        # Flush final verse
        if pending_key and pending_verse:
            verses[pending_key] = _strip_usfm_tags(pending_verse)

    print(f"  Extracted {len(verses):,} verse references from {label}")
    return verses


def parse_readaloud_zip(zf: zipfile.ZipFile, label: str) -> dict:
    """
    Parse eBible 'readaloud' plain-text ZIPs (chapter files).
    Each file is named like  001GEN.txt or GEN_001.txt.
    Each non-empty line is one verse in order.
    Returns {  'GEN_1_1': 'verse text', ... }
    """
    verses = {}
    txt_files = sorted(
        n for n in zf.namelist()
        if not n.startswith("__MACOSX") and n.endswith(".txt")
    )
    print(f"  Parsing {len(txt_files)} readaloud files in {label} zip...")

    for fname in txt_files:
        basename = os.path.basename(fname).upper()

        # Extract book + chapter from filename
        # Patterns: 001GEN001.txt, GEN_001.txt, GEN001.txt
        bm = re.search(r"([A-Z1-9]{2,3})[_\-]?0*(\d+)\.TXT", basename)
        if not bm:
            bm = re.search(r"([A-Z]{2,3})\.TXT", basename)
            if not bm:
                continue
            book_tag = bm.group(1)
            chapter  = "1"
        else:
            book_tag = bm.group(1).lstrip("0") or bm.group(1)
            chapter  = str(int(bm.group(2)))

        try:
            raw = zf.read(fname).decode("utf-8", errors="ignore")
        except Exception:
            continue

        vnum = 0
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            vnum += 1
            key = f"{book_tag}_{chapter}_{vnum}"
            verses[key] = line

    print(f"  Extracted {len(verses):,} verse references from {label}")
    return verses


def smart_parse(zf: zipfile.ZipFile, label: str) -> dict:
    """Detect zip format and parse accordingly."""
    names = zf.namelist()
    has_usfm = any(n.lower().endswith((".usfm", ".sfm")) for n in names)
    has_txt  = any(n.lower().endswith(".txt") for n in names)

    # Peek at first text file to see if it has USFM markers
    if has_txt:
        sample_file = next(
            (n for n in names if n.lower().endswith(".txt") and "__MACOSX" not in n),
            None
        )
        if sample_file:
            sample = zf.read(sample_file).decode("utf-8", errors="ignore")[:500]
            if "\\v " in sample or "\\c " in sample:
                has_usfm = True

    if has_usfm:
        result = parse_usfm_zip(zf, label)
        if result:
            return result

    # Fallback to readaloud plain-text parser
    return parse_readaloud_zip(zf, label)


def scrape_ebible() -> pd.DataFrame:
    print("\n=== Source 1: eBible.org Bible verse pairs ===")

    guz_zf = fetch_zip(EBIBLE_GUZ_URLS, "Ekegusii Bible (guz)")
    if guz_zf is None:
        print("  Could not download Ekegusii Bible -- skipping eBible source.")
        return pd.DataFrame(columns=["english", "ekegusii", "source"])

    en_zf = fetch_zip(EBIBLE_EN_URLS, "English KJV")
    if en_zf is None:
        print("  Could not download English Bible -- skipping eBible source.")
        return pd.DataFrame(columns=["english", "ekegusii", "source"])

    guz_verses = smart_parse(guz_zf, "Ekegusii")
    en_verses  = smart_parse(en_zf,  "English KJV")

    if not guz_verses or not en_verses:
        print("  Parsing yielded 0 verses -- check zip contents above.")
        return pd.DataFrame(columns=["english", "ekegusii", "source"])

    common = sorted(set(en_verses) & set(guz_verses))
    print(f"  Aligned {len(common):,} verse pairs "
          f"(EN total: {len(en_verses):,} | GUZ total: {len(guz_verses):,})")

    rows = [
        {
            "english":  en_verses[k],
            "ekegusii": guz_verses[k],
            "source":   "bible_ebible",
            "ref":      k,
        }
        for k in common
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SOURCE 2 -- 4laws.com
# ---------------------------------------------------------------------------

LAWS_URLS = {
    "english":  [
        "https://4laws.com/laws/englishkgp/default.htm",
        "https://4laws.com/laws/english/default.htm",
    ],
    "ekegusii": [
        "https://4laws.com/laws/ekegusii/default.htm",
    ],
}


def _extract_sentences(html: str) -> list:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    sentences = []
    for elem in soup.find_all(["p", "li", "h2", "h3", "td"]):
        text = elem.get_text(" ", strip=True)
        for sent in re.split(r"(?<=[.!?])\s+", text):
            sent = sent.strip()
            if len(sent) > 15 and not sent.startswith("http"):
                sentences.append(sent)

    return list(dict.fromkeys(sentences))


def scrape_4laws() -> pd.DataFrame:
    print("\n=== Source 2: 4laws.com -- Four Spiritual Laws ===")
    results = {}
    for lang, urls in LAWS_URLS.items():
        for url in urls:
            try:
                r = requests.get(url, headers=HEADERS, timeout=30)
                r.raise_for_status()
                sents = _extract_sentences(r.text)
                results[lang] = sents
                print(f"  OK  {lang}: {len(sents)} sentences from {url}")
                break
            except Exception as exc:
                print(f"  FAIL  {lang} ({url}): {exc}")

    if not results.get("english") or not results.get("ekegusii"):
        return pd.DataFrame(columns=["english", "ekegusii", "source"])

    en  = results["english"]
    guz = results["ekegusii"]
    n   = min(len(en), len(guz))
    if n == 0:
        return pd.DataFrame(columns=["english", "ekegusii", "source"])

    df = pd.DataFrame({
        "english":  en[:n],
        "ekegusii": guz[:n],
        "source":   "4laws",
        "ref":      [f"4laws_{i}" for i in range(n)],
    })
    print(f"  Aligned pairs: {len(df)}")
    return df


# ---------------------------------------------------------------------------
# CLEAN & EXPORT
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["english", "ekegusii"])
    df["english"]  = df["english"].str.strip()
    df["ekegusii"] = df["ekegusii"].str.strip()
    df = df[df["english"]  != ""]
    df = df[df["ekegusii"] != ""]

    # Minimum word count
    df = df[df["english"].str.split().str.len()  >= 3]
    df = df[df["ekegusii"].str.split().str.len() >= 2]

    # Filter Ekegusii cross-reference / footnote lines
    # Pattern: starts with "-" or begins with digits followed by Bible book codes
    noise_pattern = (
        r"^-\s+"               # starts with dash (footnote)
        r"|^\d+\s+[A-Z][a-z]"  # starts with number + book abbreviation (cross-ref)
        r"|^\d+:\d+"           # bare chapter:verse reference
    )
    df = df[~df["ekegusii"].str.match(noise_pattern, na=False)]

    # Drop exact duplicates
    df = df.drop_duplicates(subset=["english"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["ekegusii"]).reset_index(drop=True)

    print(f"\n  Cleaning: {before:,} -> {len(df):,} rows ({before - len(df):,} removed)")
    return df



def main():
    print("=" * 60)
    print("  English-Ekegusii Parallel Corpus Scraper")
    print("=" * 60)

    frames = []

    df_bible = scrape_ebible()
    if len(df_bible) > 0:
        frames.append(df_bible)
        print(f"  -> Bible pairs: {len(df_bible):,}")

    df_laws = scrape_4laws()
    if len(df_laws) > 0:
        frames.append(df_laws)
        print(f"  -> 4laws pairs: {len(df_laws):,}")

    if not frames:
        print("\nNo data collected from any source. Exiting.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined  = clean(combined)

    print("\n  Pairs by source:")
    for src, cnt in combined["source"].value_counts().items():
        print(f"    {src:<22} {cnt:>6,}")

    # Save with and without ref column
    combined.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"\nSaved {len(combined):,} pairs to:")
    print(f"  {OUTPUT_FILE}")

    print("\nSample rows:")
    sample = combined[["english", "ekegusii", "source"]].sample(min(5, len(combined)))
    for _, row in sample.iterrows():
        print(f"  [{row['source']}]")
        print(f"    EN:  {row['english'][:80]}")
        print(f"    GUZ: {row['ekegusii'][:80]}")
        print()


if __name__ == "__main__":
    main()
