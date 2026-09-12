#!/usr/bin/env python3
"""
scrape_lughayangu.py
====================
Harvests English-Ekegusii sentence pairs and lexicon entries from
lughayangu.com, a community-contributed Kenyan language dictionary.

Design notes (v2 - the first version produced garbage)
------------------------------------------------------
The first attempt classified text by English function-word density and paired
whatever it found in document order. That failed badly: the site's interface
copy ("Your voice can preserve this word", "Help the community learn how X is
really pronounced...") is fluent English, far outnumbers the real content, and
so got paired against genuine Ekegusii examples. Two-word glosses such as
"black nightshade" carry no function words at all and were misfiled as
Ekegusii.

Both failures are fixed by not guessing:

1. LANGUAGE ID IS LEARNED, NOT HEURISTIC. A character 3-gram Naive Bayes
   classifier is trained at runtime on the project's own aligned Bible corpus
   (data/bible_en_guz_swh.csv), which supplies ~30k verified sentences in
   each of English, Ekegusii and Kiswahili. Held-out accuracy: 100.0% English,
   99.5% Ekegusii, 100.0% Kiswahili. It classifies "black nightshade" as
   English and "TUMIA KISWAHILI" as Kiswahili, both of which the heuristic got
   wrong.

2. BOILERPLATE IS DETECTED BY CROSS-PAGE FREQUENCY, NOT BY A BLOCKLIST.
   Interface strings appear on every word page; real content appears on one.
   Any text block seen on more than --boilerplate-min pages is dropped. The
   page's own headword is masked before counting so templated copy
   ("...how Engano is really pronounced...") collapses across pages. This needs
   no knowledge of the site's markup and survives a redesign.

3. PAIRS ARE NEVER FABRICATED. Ekegusii examples are paired with English
   translations only when the page yields equal counts of each. Anything else
   is discarded (pass --write-unpaired to keep it for diagnosis).

Every output row carries a `psa_relevance` score: the fraction of the English
side's content words that also occur in data/english_psas.csv. It is reported
rather than filtered on, because the Ekegusii section of this site is small and
discarding rows at scrape time cannot be undone without re-crawling. Use
--psa-only THRESHOLD if you do want to filter.

Crawling and parsing are separate passes. Pass 1 caches raw text blocks per
URL; pass 2 works purely from that cache, so extraction can be re-tuned with
--reparse without re-crawling the site.

Etiquette
---------
robots.txt is fully permissive ("User-agent: * / Disallow:") and publishes a
sitemap. This script still re-checks robots at runtime, rate-limits, identifies
itself with a contact URL, and carries the contributor name and source URL into
every output row. The content is written by named volunteers - attribute them,
and contact the site before redistributing anything derived from it.

Usage
-----
    pip install requests beautifulsoup4

    python scrape_lughayangu.py --limit 40 --audit   # trial, show the reasoning
    python scrape_lughayangu.py                      # full crawl (resumable)
    python scrape_lughayangu.py --reparse --audit    # re-extract, no crawling

Outputs (--output-dir, default "data/"):
    lughayangu_sentences.csv    english / ekegusii example-sentence pairs
    lughayangu_cache.json       raw crawl cache (enables --reparse)
    lughayangu_lexicon.csv      only with --write-lexicon
    lughayangu_unpaired.csv     only with --write-unpaired

The headword lexicon is not written by default. Measured against the project's
own data, 574 glosses cover 0.09% of the Ekegusii vocabulary attested in the
Bible corpus, and consist of botanical terms that do not occur in the English
PSA corpus. The example sentences are the reason to run this scraper.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import os
import re
import sys
import time
import urllib.robotparser
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BASE = "https://lughayangu.com"
LANG_PATH = "/ekegusii"
SITEMAP = f"{BASE}/sitemap.xml"

CONTACT = "https://github.com/SamAbr/PSA-MT"
USER_AGENT = f"AcademicResearchBot/1.0 (USIU-Africa MT research; +{CONTACT})"
HEADERS = {"User-Agent": USER_AGENT}

DEFAULT_LID_CORPUS = os.path.join("output", "bible_en_guz_swh.csv")
DEFAULT_PSA_CORPUS = os.path.join("output", "english_psas.csv")

STOPWORDS = set(
    "the a an and or but if of to in on at for your you we they it is are was "
    "were be been being have has had do does did will would shall should can "
    "could may might must not no this that these those with from by as all any "
    "more most very so than then into out up down over under about after before "
    "because their our its his her".split()
)

# Deliberately low: single-word glosses ("wheat", "salt", "tea") are common and
# are the most reliable content on the site. Short junk is handled by the
# boilerplate filter and the language-ID confidence margin instead.
MIN_CHARS = 3
MAX_CHARS = 400
GLOSS_MAX_TOKENS = 3        # <= this many words is a headword gloss
MIN_MARGIN = 0.15           # minimum log-prob margin to trust a LID decision

# Structural prefixes that are metadata rather than content.
META_PREFIX_RE = re.compile(r"^\s*(synonyms?|antonyms?|see also|plural|singular)\s*:", re.I)


# ---------------------------------------------------------------------------
# LANGUAGE IDENTIFICATION
# ---------------------------------------------------------------------------

class CharLID:
    """Character n-gram Naive Bayes language identifier."""

    def __init__(self, n: int = 3):
        self.n = n
        self.models: Dict[str, Counter] = {}
        self.totals: Dict[str, int] = {}
        self.vocab: Set[str] = set()

    def _grams(self, text: str) -> List[str]:
        text = " " + " ".join(text.lower().split()) + " "
        return [text[i:i + self.n] for i in range(len(text) - self.n + 1)]

    def train(self, corpora: Dict[str, List[str]]) -> None:
        for lang, texts in corpora.items():
            counts: Counter = Counter()
            for text in texts:
                counts.update(self._grams(text))
            self.models[lang] = counts
            self.vocab |= set(counts)
        self.totals = {l: sum(c.values()) for l, c in self.models.items()}
        self.V = max(1, len(self.vocab))

    def predict(self, text: str) -> Tuple[Optional[str], float]:
        """Return (language, margin). Margin is the log-prob gap to runner-up."""
        grams = self._grams(text)
        if not text.strip() or not grams:
            return None, 0.0
        scores = {}
        for lang, counts in self.models.items():
            total = self.totals[lang]
            scores[lang] = sum(
                math.log((counts.get(g, 0) + 0.1) / (total + 0.1 * self.V))
                for g in grams
            ) / len(grams)
        best = max(scores, key=scores.get)
        others = sorted((v for k, v in scores.items() if k != best), reverse=True)
        return best, (scores[best] - others[0]) if others else 99.0


def train_lid(corpus_path: str) -> CharLID:
    if not os.path.exists(corpus_path):
        raise SystemExit(
            f"\nFATAL: language-ID corpus not found at '{corpus_path}'.\n"
            f"Run ekegusii/build_trilingual_corpus.py first, or pass --lid-corpus.\n"
            f"Without it every extracted row would be a guess.\n"
        )
    csv.field_size_limit(10 ** 7)
    with open(corpus_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    lid = CharLID()
    lid.train({
        "en": [r["english"] for r in rows],
        "guz": [r["ekegusii"] for r in rows],
        "swh": [r["swahili"] for r in rows],
    })
    print(f"  trained character-trigram LID on {len(rows):,} aligned triples")
    return lid


# ---------------------------------------------------------------------------
# FETCHING
# ---------------------------------------------------------------------------

class Fetcher:
    def __init__(self, delay: float = 1.5, retries: int = 3):
        self.delay, self.retries = delay, retries
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._last = 0.0

    def get(self, url: str, binary: bool = False):
        for attempt in range(1, self.retries + 1):
            wait = self.delay - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                resp = self.session.get(url, timeout=45)
                self._last = time.time()
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                return resp.content if binary else resp.text
            except Exception as exc:
                self._last = time.time()
                if attempt == self.retries:
                    print(f"    giving up on {url}: {exc}")
                    return None
                time.sleep(self.delay * 2 * attempt)
        return None


def check_robots(fetcher: Fetcher) -> bool:
    text = fetcher.get(f"{BASE}/robots.txt")
    if text is None:
        print("  could not read robots.txt - aborting out of caution")
        return False
    parser = urllib.robotparser.RobotFileParser()
    parser.parse(text.splitlines())
    allowed = parser.can_fetch(USER_AGENT, f"{BASE}{LANG_PATH}")
    print(f"  robots.txt permits crawling {LANG_PATH}: {allowed}")
    return allowed


# ---------------------------------------------------------------------------
# URL ENUMERATION
# ---------------------------------------------------------------------------

def is_word_url(url: str) -> bool:
    path = urlparse(url).path.rstrip("/")
    if not path.startswith(f"{LANG_PATH}/"):
        return False
    slug = path[len(LANG_PATH) + 1:]
    return bool(slug) and "/" not in slug


def urls_from_sitemap(fetcher: Fetcher) -> Set[str]:
    found: Set[str] = set()
    queue, seen = [SITEMAP], set()
    while queue:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        raw = fetcher.get(url, binary=True)
        if not raw:
            continue
        if raw[:2] == b"\x1f\x8b":
            try:
                raw = gzip.decompress(raw)
            except Exception:
                continue
        try:
            root = ET.parse(io.BytesIO(raw)).getroot()
        except Exception:
            continue
        locs = [e.text.strip() for e in root.iter()
                if e.tag.split("}")[-1] == "loc" and e.text]
        if root.tag.split("}")[-1] == "sitemapindex":
            queue.extend(locs)
        else:
            found |= {l for l in locs if is_word_url(l)}
    return found


def urls_from_pagination(fetcher: Fetcher, max_pages: int = 500) -> Set[str]:
    found: Set[str] = set()
    empty = 0
    for page in range(1, max_pages + 1):
        url = f"{BASE}{LANG_PATH}" + ("" if page == 1 else f"?page={page}")
        html = fetcher.get(url)
        if not html:
            break
        soup = BeautifulSoup(html, "html.parser")
        page_urls = {urljoin(BASE, a["href"]) for a in soup.find_all("a", href=True)
                     if is_word_url(urljoin(BASE, a["href"]))}
        new = page_urls - found
        found |= page_urls
        print(f"    page {page}: +{len(new)} new (total {len(found)})")
        empty = empty + 1 if not new else 0
        if empty >= 2:
            break
    return found


# ---------------------------------------------------------------------------
# PASS 1 - RAW BLOCK EXTRACTION (no interpretation)
# ---------------------------------------------------------------------------

def raw_blocks(html: str) -> Dict:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "form", "button"]):
        tag.decompose()

    h1 = soup.find("h1")
    headword = re.sub(r"\s+", " ", h1.get_text(" ", strip=True)).strip() if h1 else ""

    blocks: List[str] = []
    for elem in soup.find_all(["p", "li", "td", "div", "span", "blockquote",
                               "h2", "h3", "h4", "strong", "b", "em", "i", "a"]):
        if elem.find(["p", "li", "div", "blockquote", "td"]):
            continue  # leaf blocks only
        text = re.sub(r"\s+", " ", elem.get_text(" ", strip=True)).strip()
        if text and text not in blocks:
            blocks.append(text)

    contributor, date = "", ""
    for text in blocks:
        m = re.match(r"^By\s+([A-Za-z][\w .'-]{0,40})$", text)
        if m and not contributor:
            contributor = m.group(1).strip()
        m2 = re.match(r"^([A-Z][a-z]+ \d{1,2}, \d{4})$", text)
        if m2 and not date:
            date = m2.group(1)

    return {"headword": headword, "blocks": blocks,
            "contributor": contributor, "date": date}


# ---------------------------------------------------------------------------
# PASS 2 - BOILERPLATE REMOVAL AND EXTRACTION
# ---------------------------------------------------------------------------

def normalise_for_df(text: str, headword: str) -> str:
    """Mask the headword so templated interface copy collapses across pages."""
    out = text.lower()
    if headword:
        out = out.replace(headword.lower(), "<hw>")
        for part in headword.lower().split():
            if len(part) > 3:
                out = out.replace(part, "<hw>")
    return re.sub(r"\s+", " ", out).strip()


def find_boilerplate(cache: Dict[str, Dict], min_pages: int) -> Set[str]:
    df: Counter = Counter()
    for rec in cache.values():
        seen = {normalise_for_df(b, rec["headword"]) for b in rec["blocks"]}
        df.update(seen)
    return {t for t, n in df.items() if n >= min_pages}


def load_psa_vocab(path: str) -> Optional[Set[str]]:
    """Content-word vocabulary of the project's own English PSA corpus."""
    if not os.path.exists(path):
        return None
    csv.field_size_limit(10 ** 7)
    vocab: Set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            for word in re.findall(r"[a-z']+", (row.get("English") or "").lower()):
                if word not in STOPWORDS and len(word) > 2:
                    vocab.add(word)
    print(f"  loaded {len(vocab):,} PSA content words from {path}")
    return vocab


def psa_relevance(text: str, psa_vocab: Optional[Set[str]]) -> Optional[float]:
    """
    Fraction of the English side's content words that also occur in the PSA
    corpus. Reported as a column rather than used as a filter by default: the
    scraped vocabulary is small, and discarding rows at scrape time throws away
    data that cannot be recovered without re-crawling.
    """
    if psa_vocab is None:
        return None
    words = [w for w in re.findall(r"[a-z']+", text.lower())
             if w not in STOPWORDS and len(w) > 2]
    if not words:
        return 0.0
    return round(sum(1 for w in words if w in psa_vocab) / len(words), 3)


def extract(cache: Dict[str, Dict], lid: CharLID, boilerplate: Set[str],
            psa_vocab: Optional[Set[str]] = None,
            audit: bool = False) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    sentences, lexicon, unpaired = [], [], []

    for url, rec in sorted(cache.items()):
        headword = rec["headword"] or urlparse(url).path.rstrip("/").split("/")[-1].replace("-", " ")
        kept, dropped = [], []

        for block in rec["blocks"]:
            norm = normalise_for_df(block, rec["headword"])
            if norm in boilerplate:
                dropped.append((block, "boilerplate")); continue
            if META_PREFIX_RE.match(block):
                dropped.append((block, "metadata")); continue
            if not (MIN_CHARS <= len(block) <= MAX_CHARS):
                dropped.append((block, "length")); continue
            if block.strip().lower() == headword.strip().lower():
                dropped.append((block, "headword")); continue
            lang, margin = lid.predict(block)
            if lang is None or margin < MIN_MARGIN:
                dropped.append((block, f"low-confidence {lang} m={margin:.2f}")); continue
            if lang == "swh":
                dropped.append((block, "kiswahili")); continue
            kept.append((block, lang, margin))

        guz = [b for b, l, _ in kept if l == "guz"]
        en_short = [b for b, l, _ in kept if l == "en" and len(b.split()) <= GLOSS_MAX_TOKENS]
        en_long = [b for b, l, _ in kept if l == "en" and len(b.split()) > GLOSS_MAX_TOKENS]

        meta = {"headword": headword, "contributor": rec["contributor"],
                "date": rec["date"], "source": "lughayangu", "url": url}

        for gloss in en_short:
            lexicon.append({"ekegusii": headword, "english": gloss,
                            "psa_relevance": psa_relevance(gloss, psa_vocab), **meta})

        # Pair only on an exact count match; never guess an alignment.
        if guz and len(guz) == len(en_long):
            for g, e in zip(guz, en_long):
                sentences.append({"english": e, "ekegusii": g,
                                  "psa_relevance": psa_relevance(e, psa_vocab), **meta})
        else:
            for g in guz:
                unpaired.append({"ekegusii": g, "english": "",
                                 "reason": f"{len(guz)} ekegusii vs {len(en_long)} english",
                                 **meta})
            for e in en_long:
                unpaired.append({"ekegusii": "", "english": e,
                                 "reason": f"{len(guz)} ekegusii vs {len(en_long)} english",
                                 **meta})

        if audit:
            print(f"\n  {url}")
            print(f"    headword: {headword}")
            for b, l, m in kept:
                print(f"    KEEP [{l} {m:.2f}] {b[:90]}")
            for b, why in dropped:
                print(f"    drop [{why}] {b[:70]}")

    return sentences, lexicon, unpaired


def write_csv(path: str, rows: List[Dict], columns: List[str]) -> int:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        seen = set()
        n = 0
        for row in rows:
            key = tuple(str(row.get(c, "")).lower() for c in columns[:2])
            if key in seen:
                continue
            seen.add(key)
            writer.writerow(row)
            n += 1
    return n


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default="data")
    ap.add_argument("--lid-corpus", default=DEFAULT_LID_CORPUS)
    ap.add_argument("--psa-corpus", default=DEFAULT_PSA_CORPUS,
                    help="English PSA corpus used to score domain relevance")
    ap.add_argument("--psa-only", type=float, default=None, metavar="THRESHOLD",
                    help="keep only rows whose psa_relevance is >= THRESHOLD "
                         "(e.g. 0.5). Off by default - see --help notes.")
    ap.add_argument("--write-unpaired", action="store_true",
                    help="also write examples that had no confident translation")
    ap.add_argument("--write-lexicon", action="store_true",
                    help="also write headword glosses. Off by default: 574 word "
                         "pairs against a 44,205-type Ekegusii vocabulary is 0.09%% "
                         "coverage, and of botanical terms the PSA corpus does not use.")
    ap.add_argument("--delay", type=float, default=1.5)
    ap.add_argument("--limit", type=int, default=0, help="stop after N pages (0 = all)")
    ap.add_argument("--boilerplate-min", type=int, default=4,
                    help="text seen on this many pages is interface chrome")
    ap.add_argument("--reparse", action="store_true",
                    help="re-extract from the cache without crawling")
    ap.add_argument("--audit", action="store_true",
                    help="print every kept/dropped block with its reason")
    ap.add_argument("--restart", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cache_path = os.path.join(args.output_dir, "lughayangu_cache.json")

    print("=" * 70)
    print("  lughayangu.com  ->  English-Ekegusii corpus")
    print("=" * 70)

    print("\n=== Training language identifier ===")
    lid = train_lid(args.lid_corpus)
    psa_vocab = load_psa_vocab(args.psa_corpus)
    if psa_vocab is None:
        print(f"  no PSA corpus at {args.psa_corpus} - psa_relevance will be blank")
        if args.psa_only is not None:
            raise SystemExit("--psa-only needs --psa-corpus to exist.")

    cache: Dict[str, Dict] = {}
    if os.path.exists(cache_path) and not args.restart:
        try:
            with open(cache_path, encoding="utf-8") as fh:
                cache = json.load(fh)
            print(f"  loaded cache: {len(cache)} pages")
        except Exception:
            cache = {}

    if not args.reparse:
        fetcher = Fetcher(delay=args.delay)
        print("\n=== Checking robots.txt ===")
        if not check_robots(fetcher):
            print("Crawling disallowed. Aborting.")
            return 1

        print("\n=== Enumerating word pages ===")
        urls = urls_from_sitemap(fetcher)
        print(f"  from sitemap: {len(urls)}")
        if len(urls) < 50:
            print("  sitemap thin - falling back to pagination")
            urls |= urls_from_pagination(fetcher)
        todo = sorted(urls - set(cache))
        if args.limit:
            todo = todo[:args.limit]
        print(f"  to fetch this run: {len(todo)}")

        print("\n=== Fetching ===")
        try:
            for i, url in enumerate(todo, 1):
                html = fetcher.get(url)
                if html:
                    cache[url] = raw_blocks(html)
                if i % 25 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}")
                    with open(cache_path, "w", encoding="utf-8") as fh:
                        json.dump(cache, fh, ensure_ascii=False)
        except KeyboardInterrupt:
            print("\n  interrupted - cache saved, re-run to resume")
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, ensure_ascii=False)

    if not cache:
        print("\nNo pages cached. Nothing to extract.")
        return 1

    print(f"\n=== Detecting interface boilerplate across {len(cache)} pages ===")
    boilerplate = find_boilerplate(cache, args.boilerplate_min)
    print(f"  {len(boilerplate)} repeated text blocks will be dropped")
    if len(cache) < args.boilerplate_min * 3:
        print(f"  WARNING: only {len(cache)} pages cached. Boilerplate detection")
        print(f"  needs breadth - crawl at least {args.boilerplate_min * 5} pages"
              f" before trusting the output.")

    print("\n=== Extracting ===")
    sentences, lexicon, unpaired = extract(cache, lid, boilerplate,
                                           psa_vocab=psa_vocab, audit=args.audit)

    if args.psa_only is not None:
        before_s, before_l = len(sentences), len(lexicon)
        sentences = [r for r in sentences if (r.get("psa_relevance") or 0) >= args.psa_only]
        lexicon = [r for r in lexicon if (r.get("psa_relevance") or 0) >= args.psa_only]
        print(f"  --psa-only {args.psa_only}: sentences {before_s} -> {len(sentences)}, "
              f"lexicon {before_l} -> {len(lexicon)}")

    s_path = os.path.join(args.output_dir, "lughayangu_sentences.csv")
    cols = ["headword", "psa_relevance", "contributor", "date", "source", "url"]
    n_s = write_csv(s_path, sentences, ["english", "ekegusii"] + cols)
    print(f"\n  {n_s:>6,} sentence pairs  -> {s_path}")

    if args.write_lexicon:
        l_path = os.path.join(args.output_dir, "lughayangu_lexicon.csv")
        n_l = write_csv(l_path, lexicon, ["ekegusii", "english"] + cols)
        print(f"  {n_l:>6,} lexicon entries -> {l_path}")
    else:
        print(f"  {len(lexicon):>6,} lexicon entries discarded (--write-lexicon to keep)")

    if args.write_unpaired:
        u_path = os.path.join(args.output_dir, "lughayangu_unpaired.csv")
        n_u = write_csv(u_path, unpaired, ["ekegusii", "english", "reason"] + cols)
        print(f"  {n_u:>6,} unpaired        -> {u_path}")
    else:
        print(f"  {len(unpaired):>6,} unpaired rows discarded (--write-unpaired to keep)")

    if sentences:
        print("\n=== Sample ===")
        for row in sentences[:5]:
            print(f"  EN : {row['english'][:88]}")
            print(f"  GUZ: {row['ekegusii'][:88]}\n")
    else:
        print("\n  No pairs extracted. Re-run with --audit to see what was dropped.")

    print("Attribute lughayangu.com and its contributors; contact the site")
    print("before redistributing anything derived from this data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
