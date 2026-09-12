"""
nb_common.py - shared configuration and helpers for the Maragoli NMT project.

Imported by training, evaluation and notebook scripts so that paths, language
codes and random seeds are defined in exactly one place.
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


ROOT = find_project_root()
INPUTS = ROOT / "data"           # corpora this project trains and tests on
ARTIFACTS = ROOT / "artifacts"   # models, tokenizers, checkpoints
DATA = ARTIFACTS / "data"        # training splits
FIGURES = ARTIFACTS / "figures"
for _d in (ARTIFACTS, DATA, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# LANGUAGE CODES
# ---------------------------------------------------------------------------

ENG = "eng_Latn"
SWH = "swh_Latn"
RAG = "rag_Latn"             # Maragoli / Lulogooli
RAG_INIT_FROM = "kik_Latn"   # Kikuyu: nearest available Bantu language in NLLB

BASE_MODEL = "facebook/nllb-200-distilled-600M"

SEED = 42

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
        print(f"  free  VRAM : {free:6.1f} GB")
    print(f"\ntorch {torch.__version__}  |  CUDA {torch.version.cuda}")


# ---------------------------------------------------------------------------
# NLLB TOKENIZER HELPERS
# ---------------------------------------------------------------------------

LANG_CODE_RE = re.compile(r"^[a-z]{3}_[A-Z][a-z]{3}$")


def nllb_language_tokens(tok) -> list:
    """
    Every NLLB language code the tokenizer knows.
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
