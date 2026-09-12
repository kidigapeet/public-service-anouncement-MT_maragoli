#!/usr/bin/env python3
"""
count_records.py - what exactly did the released model train on?

Replays notebook 02's split logic against the real CSVs, so the breakdown is
computed rather than recalled. Same constants, same seed, same order of
operations - if this disagrees with the notebook, the notebook is right and this
script is wrong, so it is written to mirror it line for line.
"""
import csv
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent   # PSA-MT/docs
DATA = HERE.parent / "data"                      # PSA-MT/data

SEED = 42
LUGHAYANGU_HELDOUT = 200
BIBLE_TEST = 1500
MIN_WORDS, MAX_WORDS = 2, 60
PSA_KE_UPSAMPLE = 4

ENG, SWH, GUZ = "eng_Latn", "swh_Latn", "guz_Latn"


def rec(src_lang, tgt_lang, src, tgt, corpus, domain=""):
    return {"src_lang": src_lang, "tgt_lang": tgt_lang,
            "src": str(src).strip(), "tgt": str(tgt).strip(),
            "corpus": corpus, "domain": domain}


def usable(r):
    ns, nt = len(r["src"].split()), len(r["tgt"].split())
    return (MIN_WORDS <= ns <= MAX_WORDS) and (MIN_WORDS <= nt <= MAX_WORDS)


def dedupe(rows):
    seen, out = set(), []
    for r in rows:
        key = (r["src_lang"], r["tgt_lang"], r["src"].lower(), r["tgt"].lower())
        if key not in seen:
            seen.add(key); out.append(r)
    return out


bible = pd.read_csv(DATA / "bible_en_guz_swh.csv")
lg = pd.read_csv(DATA / "lughayangu_sentences.csv")
ke = pd.read_csv(DATA / "psa_ke_train.csv").fillna("")
ket = pd.read_csv(DATA / "psa_ke_test.csv").fillna("")

print(f"bible_en_guz_swh.csv      {len(bible):>7,} rows   columns: {list(bible.columns)}")
if "source" in bible.columns:
    print("  by source:")
    for k, v in bible["source"].value_counts().items():
        print(f"    {k:<28} {v:>7,}")
print(f"lughayangu_sentences.csv  {len(lg):>7,} rows")
print(f"psa_ke_train.csv          {len(ke):>7,} rows")
print(f"psa_ke_test.csv           {len(ket):>7,} rows")
print()

bible_en_guz = [rec(ENG, GUZ, r.english, r.ekegusii, "bible") for r in bible.itertuples()]
bible_swh_guz = [rec(SWH, GUZ, r.swahili, r.ekegusii, "bible") for r in bible.itertuples()]
lugha = [rec(ENG, GUZ, r.english, r.ekegusii, "lughayangu") for r in lg.itertuples()]

ke_swh = ke[ke["kiswahili"].astype(str).str.strip() != ""]
ke_en_guz = [rec(ENG, GUZ, r.english, r.ekegusii, "psa_ke", r.domain) for r in ke.itertuples()]
ke_swh_guz = [rec(SWH, GUZ, r.kiswahili, r.ekegusii, "psa_ke", r.domain) for r in ke_swh.itertuples()]

rng = np.random.default_rng(SEED)
perm = rng.permutation(len(bible))
test_idx = set(perm[:BIBLE_TEST].tolist())
dev_idx = set(perm[BIBLE_TEST:BIBLE_TEST + BIBLE_TEST // 2].tolist())


def bible_split(rows):
    tr, dv, te = [], [], []
    for i, r in enumerate(rows):
        if usable(r):
            (te if i in test_idx else dv if i in dev_idx else tr).append(r)
    return tr, dv, te


b1_tr, b1_dv, b1_te = bible_split(bible_en_guz)
b2_tr, b2_dv, b2_te = bible_split(bible_swh_guz)

lg_use = [r for r in lugha if usable(r)]
rng.shuffle(lg_use)
lg_te, lg_tr = lg_use[:LUGHAYANGU_HELDOUT], lg_use[LUGHAYANGU_HELDOUT:]

stage1 = dedupe(b1_tr + b2_tr + lg_tr)
psa_pool = dedupe(ke_en_guz + ke_swh_guz)
psa_up = psa_pool * PSA_KE_UPSAMPLE
mixed = stage1 + psa_up

print("TRAINING PAIRS IN THE RELEASED MODEL")
print(f"  Bible/storybook  eng->guz         {len(b1_tr):>7,}")
print(f"  Bible/storybook  swh->guz         {len(b2_tr):>7,}")
print(f"  Lughayangu       eng->guz         {len(lg_tr):>7,}")
print(f"  (deduplication removed)           {len(b1_tr) + len(b2_tr) + len(lg_tr) - len(stage1):>7,}")
print(f"  {'general subtotal':<33} {len(stage1):>7,}")
print(f"  Kenyan PSA       eng->guz         {len(ke_en_guz):>7,}")
print(f"  Kenyan PSA       swh->guz         {len(ke_swh_guz):>7,}")
print(f"  {'PSA subtotal (unique)':<33} {len(psa_pool):>7,}")
print(f"  {'UNIQUE PAIRS':<33} {len(stage1) + len(psa_pool):>7,}")
print(f"  PSA upsampled x{PSA_KE_UPSAMPLE}                  {len(psa_up):>7,}")
print(f"  {'EXAMPLES PER EPOCH':<33} {len(mixed):>7,}")
print()
print("Cross-check against the numbers the notebooks printed:")
print(f"  stage1 expected 56,977  got {len(stage1):,}   "
      f"{'MATCH' if len(stage1) == 56977 else 'MISMATCH'}")
print(f"  mixed  expected 79,745  got {len(mixed):,}   "
      f"{'MATCH' if len(mixed) == 79745 else 'MISMATCH'}")
print(f"  psa_up expected 22,768  got {len(psa_up):,}   "
      f"{'MATCH' if len(psa_up) == 22768 else 'MISMATCH'}")
