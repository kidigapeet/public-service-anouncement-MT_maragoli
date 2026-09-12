# English → Maragoli Neural Machine Translation

Fine-tuning **Meta's NLLB-200-distilled-600M** to translate English public-service announcements into **Maragoli** (Lulogooli), a Bantu language spoken by ~1.6 million people in western Kenya and the diaspora. Maragoli is not covered by NLLB's stock 200-language roster, making this a genuine **language-extension** project.

Includes an **interactive, real-time web application** with dual-pane translation, Kenyan PSA prompt presets, community post-editing corrections, and research benchmark inspection.

---

## Table of Contents

- [Motivation](#motivation)
- [Architecture](#architecture)
- [Data Sources & Preparation](#data-sources--preparation)
- [Live Web Application](#live-web-application)
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

## Live Web Application

The repository includes a modern, glassmorphic dark-themed web application located in `serve/`:

### Key Features
- **Real-Time Translation**: Live debounced translation into Maragoli (`rag_Latn`) with repetition penalty controls (`no_repeat_ngram_size=3`, `repetition_penalty=1.2`).
- **Kenyan PSA Presets**: One-click prompt chips for critical announcements (Health, Sanitation, Road Safety, Child Immunization, Civil Rights).
- **Dual-Mode Inference Engine**: Runs on PyTorch GPU/CPU with model checkpoints, and automatically falls back to an intelligent demonstration mode when running without local model weights.
- **Native Community Post-Editing**: "Suggest Correction" dialog logs human corrections into `data/feedback.jsonl` for continuous model improvement.
- **Benchmark Drawer**: Displays empirical project results directly within the UI.

### Launching the Web App Locally

```bash
# Start the web app and automatically open your browser at http://localhost:8000
python serve/run_demo.py --open

# Run automated API endpoint validation self-tests
python serve/run_demo.py --test
```

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Responsive single-page web application UI |
| `POST` | `/api/translate` | Translate text (`{text, src_lang, tgt_lang, num_beams}`) |
| `POST` | `/api/feedback` | Record native speaker post-editing corrections |
| `GET` | `/api/metrics` | Retrieve evaluation results (chrF2++ & BLEU) |
| `GET` | `/api/health` | Service status, active compute device, and loaded model |

---

## Repository Structure

```
├── data/
│   ├── maragoli_train.csv         # Training split (3,558 pairs)
│   ├── maragoli_dev.csv           # Validation split (313 pairs)
│   ├── maragoli_test.csv          # Held-out test split (313 pairs)
│   ├── maragoli_manifest.json     # Data provenance & counts
│   └── maragoli_evaluation_results.json # Final benchmark scores
│
├── maragoli/
│   ├── download_maragoli_corpora.py        # Harvest & clean parallel data
│   ├── prepare_maragoli_training_splits.py # 85/7.5/7.5 stratified split
│   ├── extend_maragoli_tokenizer.py        # Register rag_Latn in NLLB
│   ├── train_maragoli.py                   # Fine-tuning script
│   ├── evaluate_maragoli.py                # Inference + chrF2++ / BLEU
│   └── generate_colab_notebook.py          # Regenerate the Colab notebook
│
├── serve/
│   ├── app.py                     # FastAPI web server & inference API
│   ├── run_demo.py                # Server launcher & automated self-tests
│   └── static/
│       ├── index.html             # Semantic responsive HTML5 frontend
│       ├── style.css              # Glassmorphic dark design system
│       └── app.js                 # Real-time UI logic & feedback client
│
├── notebooks/
│   └── Maragoli_NMT_Training_Colab.ipynb   # 1-click Colab notebook
│
├── nb_common.py          # Shared config, language codes, helpers
├── requirements.txt      # Python dependencies
├── LICENSE
└── README.md             # Project documentation
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

6. Serve interactive web application
   └── serve/run_demo.py
```

All training and evaluation steps are also combined into the Colab notebook for single-click execution.

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

Evaluated on the held-out test set:

| Metric | Score | Analysis |
|---|---|---|
| **chrF2++** | **31.43** | **Strong convergence**. Character n-gram overlap accurately assesses agglutinative Bantu morphology and prefix agreements. |
| **BLEU** | **5.74** | **Expected baseline for low-resource Bantu NMT**. Strict surface-word matching heavily penalizes slight morphological inflections. |

### Sample Translation

| | Text |
|---|---|
| **Source (English)** | *Report suspected health cases to the nearest facility.* |
| **Fine-Tuned Maragoli Output** | *Rekhodia avandu aviguliri avuguliri ku likambasi li veye halala.* |

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

# Run the live translation web application
python serve/run_demo.py --open

# Or run the data and training pipeline
python maragoli/download_maragoli_corpora.py
python maragoli/prepare_maragoli_training_splits.py
python maragoli/extend_maragoli_tokenizer.py
python maragoli/train_maragoli.py
python maragoli/evaluate_maragoli.py
```

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
