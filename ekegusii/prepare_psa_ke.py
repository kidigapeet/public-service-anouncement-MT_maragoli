#!/usr/bin/env python3
"""
prepare_psa_ke.py
=================
Merges and cleans the two professor-supplied Kenyan PSA corpora:

  * `data/PSA_KE_Final.csv`  - English / Kiswahili / Ekegusii / Dholuo / Somali
  * `data/_PSA_EnGuz.csv`    - English / Ekegusii, and a superset of the above

Why this data matters more than anything else in the project
------------------------------------------------------------
Every other Ekegusii source we have is scripture or storybooks. This is Ekegusii
in **PSA register** - the exact gap the rest of the pipeline was working around.

How the two files relate
------------------------
2,897 of PSA_KE_Final's 2,903 English rows appear verbatim in `_PSA_EnGuz`, and
on that overlap the Ekegusii agrees 99.4% of the time. The 16 disagreements are
almost all cases where PSA_KE_Final is blank or left English untranslated and
`_PSA_EnGuz` supplies a real translation. So `_PSA_EnGuz` is treated as the
authority for Ekegusii, and PSA_KE_Final contributes the extra languages
(Kiswahili, Dholuo, Somali) for the rows it covers.

Filtering philosophy
--------------------
Two different kinds of problem, handled differently.

**Hard gates (row is dropped)** - the row is wrong, and training on it would
teach the model something false:
  * encoding damage that cannot be repaired
  * an Ekegusii column that is not Ekegusii, or is a verbatim copy of the English
  * ambiguous alignment: one Ekegusii string paired with several unrelated
    English sentences, so at most one of them can be correct
  * exact duplicates, empty cells, degenerate lengths

**Soft flags (row is KEPT and tagged)** - the translation is fine, it just is
not a public service announcement:
  * `document_extract` - scraped news, speeches and press releases
  * `ratio_outlier` - unusual length ratio between the two sides

Flagged rows stay in **training**, where they still teach the model Ekegusii, but
are excluded from the **test** set, because a test set of presidential speeches
would not measure PSA translation quality.

Encoding repair
---------------
Em dashes in both files were UTF-8 encoded, mis-decoded as cp1252 and
re-encoded, up to three times over, producing sequences like
`ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å"`. `ftfy` is run to a fixed point, then a targeted rule
handles residues it cannot resolve.

Usage
-----
    pip install ftfy
    python ekegusii/prepare_psa_ke.py
    python ekegusii/prepare_psa_ke.py --test-frac 0.2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(10 ** 7)

MOJIBAKE_RE = re.compile(r"Ã|Â|â€|�|Å|ƒ|'¢'¬")
DASH_RESIDUE = re.compile(
    r"(?:ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å[\"“”]?|âƒÂ¢Ã¢â€šÂ¬Ã¢â'¬Å[\"“”]?|'¢'¬â€œ|Ã¢â‚¬â€œ|â€“|â€”)")

RATIO_MIN, RATIO_MAX = 0.4, 2.5
MIN_WORDS, MAX_WORDS = 3, 150
PARAPHRASE_SIM = 0.5     # English Jaccard above this = genuine paraphrase, not misalignment


# ---------------------------------------------------------------------------
# TEXT REPAIR
# ---------------------------------------------------------------------------

def repair_text(value, max_passes: int = 6) -> str:
    try:
        import ftfy
    except ImportError:
        raise SystemExit("pip install ftfy")
    text = str(value or "").strip()
    for _ in range(max_passes):
        fixed = ftfy.fix_text(text)
        if fixed == text:
            break
        text = fixed
    text = DASH_RESIDUE.sub("–", text)
    return re.sub(r"\s+", " ", text).strip()


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 0.0


# ---------------------------------------------------------------------------
# LANGUAGE IDENTIFICATION (trained on the project's own aligned corpus)
# ---------------------------------------------------------------------------

class CharLID:
    def __init__(self, n: int = 3):
        self.n, self.models = n, {}

    def _grams(self, text: str):
        text = " " + " ".join(text.lower().split()) + " "
        return [text[i:i + self.n] for i in range(len(text) - self.n + 1)]

    def train(self, corpora: dict) -> None:
        vocab = set()
        for lang, texts in corpora.items():
            counts = Counter()
            for t in texts:
                counts.update(self._grams(t))
            self.models[lang] = counts
            vocab |= set(counts)
        self.totals = {l: sum(c.values()) for l, c in self.models.items()}
        self.V = max(1, len(vocab))

    def predict(self, text: str):
        grams = self._grams(text)
        if not grams:
            return None
        scores = {l: sum(math.log((c.get(g, 0) + 0.1) / (self.totals[l] + 0.1 * self.V))
                         for g in grams) / len(grams)
                  for l, c in self.models.items()}
        return max(scores, key=scores.get)


def build_lid(bible_csv: Path) -> CharLID:
    with open(bible_csv, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    lid = CharLID()
    lid.train({"en": [r["english"] for r in rows],
               "guz": [r["ekegusii"] for r in rows],
               "swh": [r["swahili"] for r in rows]})
    print(f"  language identifier trained on {len(rows):,} aligned triples")
    return lid


# ---------------------------------------------------------------------------
# MERGE
# ---------------------------------------------------------------------------

def merge_sources(ke_path: Path, enguz_path: Path) -> list:
    """`_PSA_EnGuz` is authoritative for Ekegusii; PSA_KE_Final adds languages."""
    merged: dict = {}

    if ke_path.exists():
        with open(ke_path, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                en = repair_text(r.get("English"))
                if not en:
                    continue
                merged[norm(en)] = {
                    "psa_id": str(r.get("PSA_Id") or "").strip(),
                    "domain": str(r.get("Domain") or "").strip(),
                    "english": en,
                    "ekegusii": repair_text(r.get("Ekegusii")),
                    "kiswahili": repair_text(r.get("Kiswahili")),
                    "dholuo": repair_text(r.get("Dholuo")),
                    "somali": repair_text(r.get("Somali")),
                    "sources": ["PSA_KE_Final"],
                }
        print(f"  PSA_KE_Final : {len(merged):,} rows")

    added = replaced = 0
    if enguz_path.exists():
        with open(enguz_path, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                en = repair_text(r.get("en"))
                guz = repair_text(r.get("guz"))
                if not en:
                    continue
                key = norm(en)
                if key in merged:
                    rec = merged[key]
                    rec["sources"].append("_PSA_EnGuz")
                    if guz and norm(guz) != norm(rec["ekegusii"]):
                        rec["ekegusii"] = guz          # newer file wins
                        replaced += 1
                else:
                    merged[key] = {
                        "psa_id": str(r.get("PSA_Id") or "").strip(),
                        "domain": str(r.get("Domain") or "").strip(),
                        "english": en, "ekegusii": guz,
                        "kiswahili": "", "dholuo": "", "somali": "",
                        "sources": ["_PSA_EnGuz"],
                    }
                    added += 1
        print(f"  _PSA_EnGuz   : +{added:,} new rows, {replaced} Ekegusii values updated")

    rows = list(merged.values())
    both = sum(1 for r in rows if len(r["sources"]) > 1)
    with_swh = sum(1 for r in rows if r["kiswahili"])
    print(f"  merged       : {len(rows):,} unique English "
          f"({both:,} in both files, {with_swh:,} have Kiswahili)")
    return rows


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parent.parent
    ap.add_argument("--psa-ke", default=str(root / "data" / "PSA_KE_Final.csv"))
    ap.add_argument("--en-guz", default=str(root / "data" / "_PSA_EnGuz.csv"))
    ap.add_argument("--bible", default=str(root / "data" / "bible_en_guz_swh.csv"))
    ap.add_argument("--output-dir", default=str(root / "data"))
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    print("=" * 72)
    print("  Kenyan PSA Ekegusii corpora - merge, clean, split")
    print("=" * 72)

    print("\n=== Merging sources ===")
    rows = merge_sources(Path(args.psa_ke), Path(args.en_guz))
    if not rows:
        print("no input rows found"); return 1

    print("\n=== Hard quality gates ===")
    lid = build_lid(Path(args.bible))
    dropped = Counter()
    kept = []
    for r in rows:
        if MOJIBAKE_RE.search(r["english"]) or MOJIBAKE_RE.search(r["ekegusii"]):
            dropped["mojibake_unrepairable"] += 1; continue
        if not r["english"] or not r["ekegusii"]:
            dropped["empty_cell"] += 1; continue
        if norm(r["ekegusii"]) == norm(r["english"]):
            dropped["untranslated_copy"] += 1; continue
        if lid.predict(r["ekegusii"]) != "guz":
            dropped["ekegusii_not_ekegusii"] += 1; continue
        n_en, n_gz = len(r["english"].split()), len(r["ekegusii"].split())
        if not (MIN_WORDS <= n_en <= MAX_WORDS and MIN_WORDS <= n_gz <= MAX_WORDS):
            dropped["degenerate_length"] += 1; continue
        kept.append(r)

    # Ambiguous alignment: one Ekegusii string against several unrelated English
    # sentences. Near-paraphrases are legitimate and kept; the rest cannot all be
    # right, so one representative survives.
    by_guz = defaultdict(list)
    for r in kept:
        by_guz[norm(r["ekegusii"])].append(r)
    survivors = []
    for _, group in by_guz.items():
        if len(group) == 1:
            survivors.append(group[0]); continue
        toks = [tokens(g["english"]) for g in group]
        paraphrases = all(jaccard(toks[0], t) >= PARAPHRASE_SIM for t in toks[1:])
        if paraphrases:
            survivors.extend(group)
        else:
            best = max(group, key=lambda g: len(g["english"].split()))
            dropped["ambiguous_alignment"] += len(group) - 1
            survivors.append(best)
    kept = survivors

    print(f"  {len(rows):,} -> {len(kept):,}")
    for reason, n in dropped.most_common():
        print(f"    -{n:>5}  {reason}")

    print("\n=== Soft flags (kept for training, excluded from the test set) ===")
    flag_counts = Counter()
    for r in kept:
        flags = []
        if r["english"][:1].islower() or re.search(r"\.\.\.|…", r["english"]):
            flags.append("document_extract")
        ratio = len(r["ekegusii"].split()) / max(1, len(r["english"].split()))
        if not (RATIO_MIN <= ratio <= RATIO_MAX):
            flags.append("ratio_outlier")
        r["quality_flags"] = "|".join(flags)
        r["is_psa_register"] = not flags
        for f in flags:
            flag_counts[f] += 1
    clean = [r for r in kept if r["is_psa_register"]]
    print(f"  document_extract : {flag_counts['document_extract']:>5}")
    print(f"  ratio_outlier    : {flag_counts['ratio_outlier']:>5}")
    print(f"  clean PSA rows   : {len(clean):>5} of {len(kept):,} "
          f"(eligible for the test set)")

    # ---- split: test drawn only from clean rows, stratified by domain -------
    rng = random.Random(args.seed)
    by_domain = defaultdict(list)
    for r in clean:
        by_domain[r["domain"]].append(r)

    test_keys = set()
    for _, group in sorted(by_domain.items()):
        rng.shuffle(group)
        k = max(1, int(round(len(group) * args.test_frac)))
        test_keys |= {norm(g["english"]) for g in group[:k]}

    test = [r for r in kept if norm(r["english"]) in test_keys]
    train = [r for r in kept if norm(r["english"]) not in test_keys]
    assert not ({norm(r["english"]) for r in train} & test_keys), "train/test leak"

    print("\n=== Split ===")
    print(f"  train {len(train):,}  ({sum(1 for r in train if not r['is_psa_register']):,} flagged)")
    print(f"  test  {len(test):,}  (all clean PSA register)")
    print("  test per domain:", dict(Counter(r["domain"] for r in test).most_common()))

    # NB: the column is "quality_flags", not "flags" - pandas DataFrames already
    # have a .flags property, so df.flags would silently return the wrong object.
    cols = ["psa_id", "domain", "english", "ekegusii", "kiswahili",
            "dholuo", "somali", "quality_flags", "is_psa_register"]
    def dump(path, data):
        with open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(data)
        print(f"  wrote {len(data):>5,} rows -> {path}")

    dump(out_dir / "psa_ke_train.csv", train)
    dump(out_dir / "psa_ke_test.csv", test)

    eval_path = out_dir / "psa_ke_test_en_guz.csv"
    with open(eval_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["english", "ekegusii", "domain", "psa_id"])
        for r in test:
            w.writerow([r["english"], r["ekegusii"], r["domain"], r["psa_id"]])
    print(f"  wrote {len(test):>5,} rows -> {eval_path}  (evaluation set)")

    with open(out_dir / "psa_ke_manifest.json", "w", encoding="utf-8") as fh:
        json.dump({"merged_rows": len(rows), "kept": len(kept),
                   "train": len(train), "test": len(test),
                   "clean_psa_rows": len(clean),
                   "dropped": dict(dropped), "soft_flags": dict(flag_counts),
                   "with_kiswahili": sum(1 for r in kept if r["kiswahili"]),
                   "test_frac": args.test_frac, "seed": args.seed,
                   "domains": dict(Counter(r["domain"] for r in kept))},
                  fh, indent=2, ensure_ascii=False)

    print("\n--- sample from the test set ---")
    for r in test[:3]:
        print(f"\n  [{r['domain']}]")
        print(f"    EN : {r['english'][:96]}")
        print(f"    GUZ: {r['ekegusii'][:96]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
