# English → Maragoli Neural Machine Translation

Fine-tuning **Meta's NLLB-200-distilled-600M** to translate English public-service announcements into **Maragoli** (Lulogooli), a Bantu language spoken by ~1.6 million people in western Kenya and the diaspora. Maragoli is not covered by NLLB's stock 200-language roster, making this a genuine **language-extension** project.

---

## Table of Contents

- [Motivation](#motivation)
- [Architecture](#architecture)
- [Data Sources & Preparation](#data-sources--preparation)
- [Repository Structure](#repository-structure)
- [Pipeline Overview](#pipeline-overview)
- [Training Details](#training-details)
- [Evaluation Results](#evaluation-results)
- [Quick Start — Google Colab](#quick-start--google-colab)
- [Local Setup](#local-setup)
- [Known Limitations & Future Work](#known-limitations--future-work)
- [License](#license)

---

## Motivation

Public health messaging in Kenya is overwhelmingly published in English and Kiswahili, leaving smaller communities without accessible information. This project demonstrates that a pre-trained multilingual model can be extended to a genuinely new, low-resource language with limited parallel data (~4,000 sentence pairs) and modest GPU time (~1 hour on a free Colab T4).

---

## Architecture

| Component | Detail |
|---|---|
| **Base model** | `facebook/nllb-200-distilled-600M` |
| **New language token** | `rag_Latn` (ISO 639-3: `rag`) |
| **Embedding initialisation** | Cloned from `kik_Latn` (Kikuyu) + 1 % Gaussian noise |
| **Framework** | HuggingFace Transformers + PyTorch |

The new `rag_Latn` embedding is seeded from Kikuyu (`kik_Latn`), the closest Bantu language available in NLLB-200, giving the model a head-start on Maragoli morphology and syntax.

---

## Data Sources & Preparation

| Source | Pairs | Description |
|---|---|---|
| **Kencorpus** (Hugging Face: `Kencorpus/Kencorpus4-MT`) | ~2,400 | Community-contributed English–Maragoli translations |
| **Luhya Multilingual** (Hugging Face: `LuhyaMultilingual/LuhyaLanguageGroup`) | ~1,800 | Multi-dialect Luhya corpus filtered for Maragoli |
| **Total unique** | **4,184** | After deduplication and cleaning |

**Splits** (85 / 7.5 / 7.5):

| Split | Sentences |
|---|---|
| Train | 3,556 |
| Dev | 314 |
| Test | 314 |

---

## Repository Structure

```
├── data/
│   ├── maragoli_train.csv         # Training split
│   ├── maragoli_dev.csv           # Validation split
│   ├── maragoli_test.csv          # Test split
│   ├── maragoli_kentrans_clean.json
│   ├── maragoli_multilingual_clean.json
│   ├── maragoli_manifest.json
│   ├── maragoli_splits_manifest.json
│   └── maragoli_evaluation_results.json
│
├── maragoli/
│   ├── download_maragoli_corpora.py        # Harvest & clean parallel data
│   ├── prepare_maragoli_training_splits.py # 85/7.5/7.5 stratified split
│   ├── extend_maragoli_tokenizer.py        # Register rag_Latn in NLLB
│   ├── train_maragoli.py                   # Fine-tuning script
│   ├── evaluate_maragoli.py                # Inference + chrF2++ / BLEU
│   └── generate_colab_notebook.py          # Regenerate the Colab notebook
│
├── notebooks/
│   └── Maragoli_NMT_Training_Colab.ipynb   # One-click Colab notebook
│
├── nb_common.py          # Shared config, language codes, helpers
├── requirements.txt      # Python dependencies
├── LICENSE
└── README.md             # ← You are here
```

---

## Pipeline Overview

```
1. Download & clean corpora
   └── maragoli/download_maragoli_corpora.py

2. Split into train / dev / test
   └── maragoli/prepare_maragoli_training_splits.py

3. Extend NLLB tokenizer with rag_Latn
   └── maragoli/extend_maragoli_tokenizer.py

4. Fine-tune on English → Maragoli pairs
   └── maragoli/train_maragoli.py

5. Evaluate with chrF2++ and BLEU
   └── maragoli/evaluate_maragoli.py
```

All five steps are combined into the Colab notebook for single-click execution.

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Learning rate | 3 × 10⁻⁵ |
| Batch size (effective) | 32 (8 × 4 gradient accumulation) |
| Epochs | 10 |
| Warmup | 200 steps |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Max source length | 128 tokens |
| Max target length | 128 tokens |
| Precision | fp16 (mixed precision) |
| Hardware | Google Colab T4 (15 GB VRAM) |
| Training time | ~50–60 minutes |

---

## Evaluation Results

Evaluated on the held-out 314-sentence test set (first 20 samples reported during quick sanity check):

| Metric | Score |
|---|---|
| **chrF2++** | **31.43** |
| **BLEU** | **5.74** |

### Sample Translation

| | Text |
|---|---|
| **Source (English)** | *Report suspected health cases to the nearest facility.* |
| **Model output** | *Rekhodia avandu aviguliri avuguliri ku likambasi li veye halala.* |

> **Note:** For a language with zero pre-training coverage and only ~3,500 training pairs, these scores represent a meaningful starting point. Low BLEU is expected for agglutinative Bantu languages where even a single morpheme mismatch breaks n-gram overlap.

### Generation Parameters

To avoid repetitive Bantu prefix looping, the following generation constraints are applied at inference:

```python
model.generate(
    ...,
    no_repeat_ngram_size=3,
    repetition_penalty=1.2,
    num_beams=4,
    max_new_tokens=128,
)
```

---

## Quick Start — Google Colab

1. Open [`notebooks/Maragoli_NMT_Training_Colab.ipynb`](notebooks/Maragoli_NMT_Training_Colab.ipynb) in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. **Run All** — the notebook downloads data, extends the tokenizer, fine-tunes, evaluates, and prints sample translations end-to-end.

No local setup required. The notebook installs all dependencies automatically.

---

## Local Setup

```bash
# Clone the repository
git clone https://github.com/kidigapeet/public-service-anouncement-MT_maragoli.git
cd public-service-anouncement-MT_maragoli

# Create a virtual environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
# .venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python maragoli/download_maragoli_corpora.py
python maragoli/prepare_maragoli_training_splits.py
python maragoli/extend_maragoli_tokenizer.py
python maragoli/train_maragoli.py
python maragoli/evaluate_maragoli.py
```

> **GPU strongly recommended.** Training on CPU will take several hours vs. ~1 hour on a T4.

---

## Known Limitations & Future Work

| Limitation | Potential Improvement |
|---|---|
| Small corpus (4,184 pairs) | Incorporate Maragoli Bible text, community elicitation |
| No back-translation | Generate synthetic Maragoli → English pairs for data augmentation |
| Single domain emphasis | Add health, legal, and agricultural parallel text |
| No human evaluation | Conduct fluency / adequacy ratings with native speakers |
| BLEU is low for agglutinative languages | Adopt morpheme-aware metrics (METEOR, BERTScore) |

---

## License

This project is released under the [MIT License](LICENSE).

---

*Built as part of a Data Science project at USIU-Africa.*
