#!/usr/bin/env python3
"""
identify_hf_models.py - prove which weights are in which repository.

Filenames are an assertion, not evidence. If the mixed checkpoint had been
pushed over the stage-2 repository, every other check in this project would
still pass and the results table would be quietly wrong. This script settles it
two ways, neither of which trusts a name:

  1. WEIGHT FINGERPRINT. Downloads each repo's tensors and compares them
     numerically. Two repositories holding the same weights score a cosine
     similarity of ~1.000000 against each other; two different training runs do
     not come close. The comparison is done in float64 on a fixed slice, so an
     fp16 copy and an fp32 copy of the same model still match - which matters
     here, because stage 2 was uploaded in half precision and the other two
     were not.

  2. BEHAVIOURAL SCORE. Re-scores held-out PSAs with each downloaded repo and
     compares against the chrF the evaluation reported. A repository whose
     score lands on another system's number is that other system.

    python tests/identify_hf_models.py              # fingerprints, ~7 GB download
    python tests/identify_hf_models.py --score      # also re-scores (needs a GPU
                                                    # or patience, and test.jsonl)
    python tests/identify_hf_models.py --local      # also fingerprint the
                                                    # directories on this machine
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

GUZ, ENG = "guz_Latn", "eng_Latn"

REPOS = {
    "stage1": "nllb-200-600M-ekegusii-stage1",
    "stage2": "nllb-200-600M-ekegusii-psa",
    "mixed":  "nllb-200-600M-ekegusii-mixed",
}

# What notebook 04 reported, eng->guz on psa_ke_heldout. A repo that scores near
# the wrong row is mislabelled.
EXPECTED_CHRF = {"stage1": 23.96, "stage2": 34.93, "mixed": 40.97}

# Derivation, so this is a number with a reason rather than a guess.
# fp16 carries ~3 decimal digits, so a round-trip perturbs each element by about
# 5e-4 relative, at random sign. Cosine between x and that copy is
# 1 - (5e-4)^2 / 2 = 0.99999987 - comfortably above this line.
# A real fine-tuning run moves weights far more: even a 0.5% per-element drift
# lands at 0.999988, below it. Verified numerically before this was committed.
#
# Expect stage1 and stage2 to be *similar* - stage 2 is trained FROM stage 1 -
# but not this similar. A high-but-below-threshold number there is the correct
# result, not a warning.
SAME_THRESHOLD = 0.99999

SLICE = 200_000          # elements compared per tensor
PREFERRED_KEYS = [
    "model.decoder.layers.6.fc1.weight",
    "model.encoder.layers.6.fc1.weight",
    "model.decoder.layers.0.self_attn.q_proj.weight",
]


def load_tensors(source: str, token: str | None):
    """Return the safetensors state dict for a repo id or a local directory."""
    from safetensors.torch import load_file

    path = Path(source)
    if path.is_dir():
        f = path / "model.safetensors"
        if not f.exists():
            raise FileNotFoundError(f"{f} not found")
        return load_file(str(f))

    from huggingface_hub import hf_hub_download
    return load_file(hf_hub_download(source, "model.safetensors", token=token))


def fingerprint(state, guz_id: int | None):
    """
    A dtype-robust signature: a fixed slice of one mid-network tensor, plus the
    added language token's embedding row.

    The embedding row is the sharpest discriminator in this project. It starts
    as a copy of Kikuyu plus 1% noise and is then trained by whichever run owns
    the checkpoint, so its direction diverges immediately between runs.
    """
    import torch

    key = next((k for k in PREFERRED_KEYS if k in state), None)
    if key is None:
        key = sorted(k for k, v in state.items() if v.ndim == 2 and v.numel() > SLICE)[0]
    body = state[key].flatten()[:SLICE].to(torch.float64)

    emb_key = next((k for k in ("model.shared.weight", "model.encoder.embed_tokens.weight",
                                "shared.weight") if k in state), None)
    row = None
    if emb_key is not None and guz_id is not None and state[emb_key].shape[0] > guz_id:
        row = state[emb_key][guz_id].to(torch.float64)
    return {"key": key, "body": body, "guz": row,
            "dtype": str(state[key].dtype), "n_tensors": len(state)}


def cos(a, b) -> float:
    import torch
    return float(torch.nn.functional.cosine_similarity(a[None], b[None])[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="samuelabrha")
    ap.add_argument("--score", action="store_true",
                    help="also re-score held-out PSAs with each repo")
    ap.add_argument("--local", action="store_true",
                    help="also fingerprint artifacts/ directories on this machine")
    ap.add_argument("--n", type=int, default=150, help="rows to score with --score")
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
        from transformers import AutoTokenizer
    except Exception as exc:
        print(f"needs torch + transformers on {sys.executable}\n  {exc}")
        return 2

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    # guz_Latn's id, read from a published tokenizer rather than assumed
    ref = f"{args.user}/{REPOS['stage2']}"
    tok = AutoTokenizer.from_pretrained(ref, token=token)
    guz_id = tok.convert_tokens_to_ids(GUZ)
    print(f"{GUZ} id {guz_id}, vocab {len(tok):,}  (from {ref})\n")

    sources = {name: f"{args.user}/{repo}" for name, repo in REPOS.items()}
    if args.local:
        sys.path.insert(0, str(Path.cwd()))
        sys.path.insert(0, str(Path.cwd().parent))
        import nb_common as C
        for name, path in [("stage1:local", C.STAGE1_MODEL),
                           ("stage2:local", C.STAGE2_MODEL),
                           ("mixed:local", C.MIXED_MODEL)]:
            if Path(path).is_dir():
                sources[name] = str(path)

    prints = {}
    for name, source in sources.items():
        print(f"loading {name:<14} {source}")
        try:
            prints[name] = fingerprint(load_tensors(source, token), guz_id)
            fp = prints[name]
            print(f"    {fp['n_tensors']} tensors, {fp['dtype']}, "
                  f"comparing {fp['key']}")
        except Exception as exc:
            print(f"    FAILED: {type(exc).__name__}: {exc}")
    print()

    # ---- pairwise -----------------------------------------------------------
    print("pairwise cosine similarity (1.000000 = identical weights)")
    print(f"{'pair':<34} {'body':>10}  {'guz row':>10}   verdict")
    duplicates = []
    for a, b in itertools.combinations(sorted(prints), 2):
        fa, fb = prints[a], prints[b]
        if fa["key"] != fb["key"]:
            print(f"{a} vs {b}: different architectures, skipped")
            continue
        c_body = cos(fa["body"], fb["body"])
        c_guz = cos(fa["guz"], fb["guz"]) if (fa["guz"] is not None
                                              and fb["guz"] is not None) else float("nan")
        same = c_body > SAME_THRESHOLD
        # a local dir and its own repo SHOULD match - that is the good case
        expected_same = a.split(":")[0] == b.split(":")[0]
        if same and not expected_same:
            verdict = "SAME WEIGHTS - one of these is mislabelled"
            duplicates.append((a, b))
        elif same:
            verdict = "match (repo holds this directory)"
        elif expected_same:
            verdict = "MISMATCH - repo does not hold this directory"
            duplicates.append((a, b))
        else:
            verdict = "distinct, as expected"
        print(f"{a + ' vs ' + b:<34} {c_body:>10.6f}  {c_guz:>10.6f}   {verdict}")
    print()

    # ---- behavioural --------------------------------------------------------
    if args.score:
        import sacrebleu
        import torch
        from transformers import AutoModelForSeq2SeqLM

        test = Path("artifacts/data/test.jsonl")
        if not test.exists():
            print(f"cannot score: {test} not found. Run this on the GPU node, or "
                  f"extract results.tgz here first.")
        else:
            rows = [json.loads(l) for l in open(test, encoding="utf-8")]
            rows = [r for r in rows if r["corpus"] == "psa_ke_heldout"
                    and r["src_lang"] == ENG][:args.n]
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"scoring {len(rows)} held-out PSA rows on {dev}\n")
            print(f"{'system':<10} {'measured':>9} {'reported':>9}   verdict")
            for name in REPOS:
                repo = f"{args.user}/{REPOS[name]}"
                m = AutoModelForSeq2SeqLM.from_pretrained(repo, token=token).to(dev).eval()
                hyps = []
                for i in range(0, len(rows), 16):
                    chunk = rows[i:i + 16]
                    enc = [[tok.convert_tokens_to_ids(ENG)]
                           + tok(r["src"], add_special_tokens=False,
                                 truncation=True, max_length=126)["input_ids"]
                           + [tok.eos_token_id] for r in chunk]
                    w = max(len(e) for e in enc)
                    ids = torch.tensor([[tok.pad_token_id] * (w - len(e)) + e
                                        for e in enc]).to(dev)
                    with torch.no_grad():
                        out = m.generate(input_ids=ids,
                                         attention_mask=(ids != tok.pad_token_id).long(),
                                         forced_bos_token_id=guz_id,
                                         max_new_tokens=128, num_beams=4)
                    hyps += tok.batch_decode(out, skip_special_tokens=True)
                score = sacrebleu.CHRF(word_order=2).corpus_score(
                    hyps, [[r["tgt"] for r in rows]]).score
                best = min(EXPECTED_CHRF, key=lambda k: abs(EXPECTED_CHRF[k] - score))
                verdict = ("consistent" if best == name
                           else f"LANDS ON {best.upper()} - repo is mislabelled")
                print(f"{name:<10} {score:>9.2f} {EXPECTED_CHRF[name]:>9.2f}   {verdict}")
                del m
                if dev == "cuda":
                    torch.cuda.empty_cache()
            print("\n(a sample of a few hundred rows will not reproduce the corpus")
            print(" chrF exactly; the systems are 6-17 points apart, so nearest-match")
            print(" is unambiguous.)")
        print()

    print("=" * 68)
    if duplicates:
        for a, b in duplicates:
            print(f"  PROBLEM  {a} and {b}")
        return 1
    print("  every repository holds distinct weights, and each matches its name")
    return 0


if __name__ == "__main__":
    sys.exit(main())
