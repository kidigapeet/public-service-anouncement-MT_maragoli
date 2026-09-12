"""
nb_common.py - shared configuration and helpers for the fine-tuning notebooks.

Imported by every notebook so that paths, plot styling and random seeds are
defined in exactly one place. Keep notebook cells about *what* is being done;
keep the plumbing here.
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

def find_project_root(start: Path | None = None) -> Path:
    """Walk upwards to the repository root: the folder holding data/ and notebooks/."""
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "data").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
    return here


# ---------------------------------------------------------------------------
# GITHUB
# ---------------------------------------------------------------------------
# The notebooks are designed to run on a bare GPU node. Any data file that is
# missing locally is fetched from the repository over HTTPS, so a node needs
# only this file and the notebook - no manual uploads, no shared filesystem.

GITHUB_USER = "SamAbr"
GITHUB_REPO = "PSA-MT"
GITHUB_BRANCH = "main"
RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}"

# filename -> path from the REPOSITORY root, which is one level above this
# project folder. The same string serves as the URL suffix and the local path,
# so there is one place to correct if a file moves.
REPO_PATHS = {
    "bible_en_guz_swh.csv": "data/bible_en_guz_swh.csv",
    "psa_ke_train.csv": "data/psa_ke_train.csv",
    "psa_ke_test.csv": "data/psa_ke_test.csv",
    "psa_ke_test_en_guz.csv": "data/psa_ke_test_en_guz.csv",
    "lughayangu_sentences.csv": "data/lughayangu_sentences.csv",
    "PSA_KE_Final.csv": "data/PSA_KE_Final.csv",
    "_PSA_EnGuz.csv": "data/_PSA_EnGuz.csv",
}

ROOT = find_project_root()
REPO_ROOT = ROOT                 # kept as a name; the project IS the repository
INPUTS = ROOT / "data"           # corpora this project trains and tests on
OUTPUT = INPUTS                  # backwards-compatible alias for older cells

# The synthetic English corpus and its 4-way translation were produced by a
# separate project that is NOT part of this repository. Only notebook 01 reads
# them, and only to compare register, so they are optional: point this at that
# project's data folder if you want those figures.
#
#     export CORPUS_GENERATION_DATA=/path/to/corpus_generation/data
#
GEN_INPUTS = Path(os.environ.get(
    "CORPUS_GENERATION_DATA", ROOT.parent / "corpus_generation" / "data"))
ARTIFACTS = ROOT / "artifacts"          # models, tokenizers, checkpoints
DATA = ARTIFACTS / "data"               # training splits
FIGURES = ARTIFACTS / "figures"
for _d in (ARTIFACTS, DATA, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

# Inputs produced by earlier stages of the project
BIBLE_CSV = INPUTS / "bible_en_guz_swh.csv"           # en / guz / swh triples
LUGHAYANGU_CSV = INPUTS / "lughayangu_sentences.csv"  # en / guz pairs
# Optional, and produced elsewhere - see GEN_INPUTS above. Notebook 01 checks
# .exists() rather than requiring these, so the pipeline runs without them.
PSA_PARALLEL_CSV = GEN_INPUTS / "psa_parallel_dataset.csv"
ENGLISH_PSA_CSV = GEN_INPUTS / "english_psas.csv"

# Professor-supplied Kenyan PSA corpora, merged by ekegusii/prepare_psa_ke.py
PSA_KE_TRAIN_CSV = INPUTS / "psa_ke_train.csv"
PSA_KE_TEST_CSV = INPUTS / "psa_ke_test.csv"

# Produced by the notebooks
EXTENDED_MODEL = ARTIFACTS / "nllb600m-guz-init"      # notebook 03

# train_stages.py trains three models on identical data for a three-way
# comparison. The notebook that used to do this was removed: it was never the
# path that produced the released weights.
STAGE1_MODEL = ARTIFACTS / "nllb600m-stage1-general"   # Bible + storybooks only
STAGE2_MODEL = ARTIFACTS / "nllb600m-stage2-psa"       # stage 1 -> PSA + replay
MIXED_MODEL = ARTIFACTS / "nllb600m-mixed-control"     # everything at once

# The headline model is the SINGLE-PASS one, not the curriculum.
#
# `mixed` was built as the control for the two-stage hypothesis and beat it on
# every test set: +6.04 chrF2++ on real PSAs (40.97 vs 34.93), and higher than
# stage 1 even on scripture, so it shows no forgetting at all. The two-stage run
# also took ~9% more gradient steps and still lost, which rules out training
# budget as the explanation. The ablation falsified the hypothesis it was
# designed to test; the honest response is to ship the control.
FINETUNED_MODEL = MIXED_MODEL

# ---------------------------------------------------------------------------
# LANGUAGE CODES
# ---------------------------------------------------------------------------

ENG = "eng_Latn"
SWH = "swh_Latn"
GUZ = "guz_Latn"        # NOT in stock NLLB-200 - added in notebook 03
GUZ_INIT_FROM = "kik_Latn"   # Kikuyu: Kenyan Bantu, nearest available neighbour
RAG = "rag_Latn"        # Maragoli / Lulogooli - added for Maragoli NMT
RAG_INIT_FROM = "kik_Latn"   # Kikuyu / Ganda: nearest available Bantu language in NLLB

BASE_MODEL = "facebook/nllb-200-distilled-600M"
TEACHER_MODEL = "facebook/nllb-200-1.3B"

# Kenyan acronyms that must survive translation intact
ACRONYMS = ["NTSA", "KEPHIS", "HELB", "KUCCPS", "KRA", "CBK", "EACC",
            "SHA", "KEMRI", "KEMSA", "NDMA", "KNEC", "TSC", "ODPC",
            "PPB", "DCI", "NPS", "Huduma", "iTax"]

SEED = 42

# ---------------------------------------------------------------------------
# PLOT STYLE
# ---------------------------------------------------------------------------
# Categorical hues in FIXED order - never cycled, never reordered per chart.
# Assign slot 1 to the first series, slot 2 to the second, and so on, so a
# language keeps its colour across every figure in the series.

PALETTE = ["#2a78d6",  # 1 blue
           "#eb6834",  # 2 orange
           "#1baf7a",  # 3 aqua
           "#eda100",  # 4 yellow
           "#e87ba4",  # 5 magenta
           "#008300",  # 6 green
           "#4a3aa7",  # 7 violet
           "#e34948"]  # 8 red

SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
              "#256abf", "#184f95", "#0d366b"]

# One colour per language, held constant across all notebooks.
LANG_COLOR = {"english": PALETTE[0], "ekegusii": PALETTE[1],
              "swahili": PALETTE[2], "kiswahili": PALETTE[2],
              "somali": PALETTE[3], "luo": PALETTE[4]}

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e5e4e0"


def use_house_style() -> None:
    """Recessive grid, thin marks, no chartjunk. Call once per notebook."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "figure.figsize": (9, 4.5),
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "600",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelcolor": INK_MUTED,
        "axes.edgecolor": GRID,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "grid.color": GRID,
        "grid.linewidth": 1.0,
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 8,
    })


def save_fig(fig, name: str) -> Path:
    """Save a figure into artifacts/figures/ and report where it went."""
    path = FIGURES / f"{name}.png"
    fig.savefig(path)
    print(f"saved figure -> {path}")
    return path


# ---------------------------------------------------------------------------
# MISC HELPERS
# ---------------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def gpu_report() -> None:
    """Print what hardware we actually have, and how much of it is free."""
    try:
        import torch
    except ImportError:
        print("torch not installed")
        return
    if not torch.cuda.is_available():
        print("NO GPU VISIBLE - training will be unusably slow")
        return
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total = props.total_memory / 1024 ** 3
        free = torch.cuda.mem_get_info(i)[0] / 1024 ** 3
        print(f"GPU {i}: {props.name}")
        print(f"  total VRAM : {total:6.1f} GB")
        print(f"  free  VRAM : {free:6.1f} GB   <- size batches against THIS,")
        print("                                   the node is shared")
    print(f"\ntorch {torch.__version__}  |  CUDA {torch.version.cuda}")


def download(filename: str, dest: Path | None = None, force: bool = False) -> Path:
    """Fetch one data file from the repository over HTTPS."""
    import urllib.error
    import urllib.request

    if filename not in REPO_PATHS:
        raise KeyError(f"{filename!r} is not in REPO_PATHS - add it there first")
    dest = Path(dest) if dest else ROOT / REPO_PATHS[filename]
    if dest.exists() and not force:
        return dest

    url = f"{RAW_BASE}/{REPO_PATHS[filename]}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {filename} ...", end="", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:
            total = 0
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
                total += len(chunk)
        tmp.replace(dest)
        print(f" {total / 1024 ** 2:.1f} MB")
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        raise FileNotFoundError(
            f"\n{url}\nreturned HTTP {exc.code}.\n"
            f"Either the file has not been pushed yet, or GITHUB_BRANCH "
            f"({GITHUB_BRANCH!r}) is wrong for this repository."
        ) from None
    return dest


def require_files(*paths, allow_download: bool = True) -> None:
    """
    Verify inputs exist, fetching them from GitHub when they do not.

    This is what lets a fresh GPU node run the notebooks with nothing staged on
    disk: anything listed in REPO_PATHS is pulled on demand.
    """
    for p in paths:
        path = Path(p)
        if not path.exists() and allow_download and path.name in REPO_PATHS:
            download(path.name, path)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required input: {path}\n"
                + ("It is not in REPO_PATHS, so it cannot be fetched - "
                   "run the notebook that produces it."
                   if path.name not in REPO_PATHS else
                   "Download failed; check network access from this node.")
            )
        size = path.stat().st_size / 1024 ** 2
        print(f"  ok  {path.name:<34} {size:8.1f} MB")


# ---------------------------------------------------------------------------
# NLLB TOKENIZER HELPERS
# ---------------------------------------------------------------------------
# `tokenizer.additional_special_tokens` used to be the way to enumerate NLLB's
# language codes, but newer transformers releases dropped it from NllbTokenizer
# ("NllbTokenizer has no attribute additional_special_tokens"), and
# `lang_code_to_id` is deprecated. These helpers work whichever version is
# installed, so the notebooks do not break on an environment upgrade.

LANG_CODE_RE = re.compile(r"^[a-z]{3}_[A-Z][a-z]{3}$")


def nllb_language_tokens(tok) -> list:
    """
    Every NLLB language code the tokenizer knows.

    Tries the documented attributes first and falls back to the vocabulary,
    which is the one thing whose shape never changes: an NLLB language code is
    always three lowercase letters, an underscore, then a title-case four-letter
    script - `eng_Latn`, `swh_Latn`, `zho_Hans`.
    """
    candidates = (
        lambda: getattr(tok, "additional_special_tokens", None),
        lambda: (getattr(tok, "special_tokens_map", None) or {}).get("additional_special_tokens"),
        lambda: list(getattr(tok, "lang_code_to_id", None) or {}),
    )
    for source in candidates:
        try:
            values = source()
        except Exception:
            values = None
        if values:
            hits = sorted(t for t in values if LANG_CODE_RE.match(str(t)))
            if hits:
                return hits
    return sorted(t for t in tok.get_vocab() if LANG_CODE_RE.match(t))


def add_language_token(tok, model, new_lang: str, init_from: str):
    """
    Add a new language to an NLLB tokenizer and model, seeding its embedding
    from an existing related language rather than from noise.

    Uses `add_tokens(..., special_tokens=True)` rather than
    `add_special_tokens({"additional_special_tokens": [...]})`: the latter needs
    the current list, which newer versions no longer expose, and in some
    versions it *replaces* that list instead of extending it.

    Returns (new_token_id, source_token_id).
    """
    import torch

    if tok.convert_tokens_to_ids(init_from) == tok.unk_token_id:
        raise ValueError(f"{init_from!r} is not in this model's vocabulary")

    tok.add_tokens([new_lang], special_tokens=True)
    model.resize_token_embeddings(len(tok))

    new_id = tok.convert_tokens_to_ids(new_lang)
    src_id = tok.convert_tokens_to_ids(init_from)
    if new_id == tok.unk_token_id:
        raise RuntimeError(f"{new_lang!r} was not added to the tokenizer")

    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        emb[new_id] = emb[src_id].clone()
        # a little noise so the two tokens can diverge during training
        emb[new_id] += torch.randn_like(emb[new_id]) * 0.01 * emb.std()
    return new_id, src_id


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
    print(f"wrote {path}")


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def banner(title: str) -> None:
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
