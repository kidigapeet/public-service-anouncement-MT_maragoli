---
title: Ekegusii PSA Translator
emoji: 📣
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: cc-by-nc-4.0
short_description: Translate Kenyan public service announcements into Ekegusii
---

# Ekegusii PSA Translator

Translates Kenyan **public service announcements** into **Ekegusii** (Ekegusii /
Kisii, `guz_Latn`), a Kenyan Bantu language spoken by roughly 2.7 million people
that **NLLB-200 does not support**.

Choose a direction — **English → Ekegusii** or **Kiswahili → Ekegusii** — type an
announcement, and the model translates it.

## What is behind it

`facebook/nllb-200-distilled-600M`, fine-tuned after adding a `guz_Latn` token
whose embedding was seeded from Kikuyu (`kik_Latn`, the closest Kenyan Bantu
language NLLB does support) rather than from noise.

| Test set | chrF2++ | vs stock NLLB |
|---|---|---|
| Real Kenyan PSAs | **40.97** | +26.41 |
| Scripture (held out) | **49.81** | +34.93 |
| Everyday prose | **32.98** | +21.57 |

Stock NLLB-200 cannot produce Ekegusii at all; the comparison asks it for Kikuyu
instead, which is a floor rather than a fair baseline.

## Please read before you trust it

**This is a draft for a human to check, not a finished translation.** Never
publish an unedited machine translation of a health, legal or safety notice.

The confidence label reports the model's own certainty, not a probability that
the translation is correct — neural translation is routinely fluent, confident
and wrong. 53.6% of English content-word types in Kenyan PSAs never appeared in
the training data, so institution names (`HELB`, `KUCCPS`, `iTax`, `NTSA`) are
where it is weakest and where a confident answer deserves the most suspicion.

There has been **no human evaluation** of this model. If you speak Ekegusii, the
correction box under each translation is the single most useful thing on this
page.

## Reproducing

Code, data preparation, notebooks and evaluation:
<https://github.com/SamAbr/PSA-MT>

Fine-Tuning Neural Machine Translation Models for Kenyan Public Service
Announcements. United States International University–Africa, Department of
Computing, 2026.
