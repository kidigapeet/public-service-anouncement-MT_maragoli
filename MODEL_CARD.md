---
license: cc-by-nc-4.0
language:
  - guz
  - en
  - sw
base_model: facebook/nllb-200-distilled-600M
pipeline_tag: translation
tags:
  - translation
  - ekegusii
  - kisii
  - low-resource
  - kenya
  - nllb
---

# NLLB-200 600M fine-tuned for Ekegusii (guz_Latn)

Adds **Ekegusii** — a Kenyan Bantu language spoken by roughly 2.7 million people —
to NLLB-200, which does not support it, and adapts the model to the register of
Kenyan **public service announcements**.

Supported directions: `eng_Latn → guz_Latn`, `swh_Latn → guz_Latn`.

## Results (chrF2++)

Four systems, identical test sets. **Single pass is the released model.**

| Direction | Test set | n | Stock NLLB\* | Stage 1 | Two-stage | **Single pass** |
|---|---|---|---|---|---|---|
| eng→guz | Bible (held out) | 1,459 | 14.88 | 49.66 | 48.95 | **49.81** |
| eng→guz | Contemporary prose | 200 | 11.41 | 28.97 | 29.37 | **32.98** |
| eng→guz | **Real PSAs** | 570 | 14.56 | 23.96 | 34.93 | **40.97** |
| swh→guz | Bible (held out) | 1,463 | 14.65 | 49.11 | 48.80 | **49.30** |
| swh→guz | **Real PSAs** | 371 | 14.13 | 24.49 | 34.50 | **39.61** |

\* Stock NLLB-200 cannot produce Ekegusii at all; the floor asks it for
`kik_Latn` (Kikuyu), the nearest supported language. It is a floor, not a fair
comparison, and should not be presented as one.

On real PSAs the released model gains **+26.41 chrF2++ over the floor (+181%)**
and **+17.01 over stage 1 (+71%)**.

## The curriculum did not work, and that is the finding

The project hypothesis was that a **two-stage curriculum** — learn Ekegusii from
scripture first, then adapt to PSA register with replay — would beat training on
everything at once. A single-pass model was trained as the control.

**The control won on every test set**, by 6.04 chrF2++ on English→Ekegusii PSAs
and 5.11 on Kiswahili→Ekegusii. It also scored *higher than stage 1 on
scripture*, meaning it exhibits no catastrophic forgetting whatsoever — the
problem the replay mixture existed to solve does not arise when the data is not
split in the first place.

Two things rule out the obvious alternative explanations:

- **Not a data-volume artifact.** The single pass sees every unique example the
  two-stage run sees. The replay slice is duplicated scripture already present
  in stage 1, not additional material.
- **Not a training-budget artifact.** At a fixed epoch count the two-stage run
  takes roughly **9% more gradient updates**, because replay rows are trained on
  twice. It had more compute and still lost.

The two-stage model's dev loss bottomed at 1.357 and drifted back to 1.417,
consistent with overfitting a small PSA-heavy set, while the single pass saw PSA
examples interleaved with scripture throughout — which appears to regularise it.

Both checkpoints are published so the comparison can be checked. The two-stage
model is the **ablation that lost**; do not report its numbers as the result.

## How it was built

This is **transfer learning**: `guz_Latn` was added to the tokenizer and its
embedding initialised from `kik_Latn` (Kikuyu, the closest Kenyan Bantu language
NLLB supports) plus 1% noise, rather than randomly — so the new language starts
from what the model already knows about related Bantu languages. The released
model was then trained in a single pass over **62,669 unique parallel sentence
pairs**, presented as 79,745 training examples (PSA data upsampled 4×):

- **56,977 general pairs** — Ekegusii Bible verses aligned to the Berean
  Standard Bible and Neno Kiswahili, plus African Storybook pages and
  contemporary sentence pairs.
- **5,692 PSA pairs** — real Kenyan public service announcements, upsampled ×4
  to 22,768 examples so the domain is not swamped by scripture.

Full fine-tune, 8-bit AdamW, effective batch 48, lr 5e-5, 3 epochs, bf16,
gradient checkpointing, on a single A100-SXM4-80GB.

## Usage

```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

REPO = "samuelabrha/nllb-200-600M-ekegusii-mixed"
tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForSeq2SeqLM.from_pretrained(REPO)

text = "Report suspected cholera cases to the nearest health facility."
ids = ([tok.convert_tokens_to_ids("eng_Latn")]
       + tok(text, add_special_tokens=False)["input_ids"]
       + [tok.eos_token_id])

out = model.generate(input_ids=torch.tensor([ids]),
                     forced_bos_token_id=tok.convert_tokens_to_ids("guz_Latn"),
                     max_new_tokens=128, num_beams=4)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

Special tokens are built explicitly (`[lang_code] … [eos]`) rather than through
`tokenizer.src_lang`, because the tokenizer's language machinery does not know
about a language added after pretraining.

NLLB is a sentence-level model. Split paragraphs into sentences and translate
each one; handing it a whole paragraph makes it drop clauses.

## Limitations

- **The Ekegusii references were not verified by the authors.** The PSA corpus
  was supplied by the project supervisor; the translator and the
  quality-assurance process are unknown. No inter-annotator agreement exists.
- **Most training data is scripture.** ~57k of the 62,669 unique pairs are Bible verses,
  so the model may still lean formal on unfamiliar input.
- **Institutional vocabulary is thin.** 53.6% of English content-word types in
  the PSA corpus never appear in the aligned training data — portals, bursaries,
  agency names. Expect weak handling of `HELB`, `KUCCPS`, `iTax` and similar.
- **Little human evaluation.** Every number here is an automatic metric against
  a single reference. Ekegusii speakers on the project have looked at output
  informally, but no structured ratings have been collected at a size worth
  quoting. chrF2++ is reported in preference to BLEU because word
  n-grams are structurally unfair to an agglutinative Bantu language, but no
  automatic metric tells you whether the output reads as a public notice rather
  than as scripture.
- Not suitable for medical, legal or emergency communication without review by a
  fluent Ekegusii speaker.

## Reproducing

Every checkpoint, split and metric is reproducible from
[the repository](https://github.com/SamAbr/PSA-MT):
notebooks `03` → `07`, or `train_stages.py` for the training runs.
`tests/identify_hf_models.py` verifies by weight fingerprint that each published
repository holds the checkpoint its name claims.

## Data licensing

Derived from the Ekegusii Revised Bible (© Bible Society of Kenya, eBible.org),
the Berean Standard Bible (public domain), Neno: Bibilia Takatifu (Biblica open
licence), African Storybook (CC-BY), and PSA corpora supplied by USIU-Africa.
Verify redistribution terms before commercial use.

## Citation

Fine-Tuning Neural Machine Translation Models for Kenyan Public Service
Announcements. United States International University–Africa, Department of
Computing, 2026.
