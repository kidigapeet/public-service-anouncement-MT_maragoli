#!/usr/bin/env python3
"""
verify_hf_uploads.py - prove the trained checkpoints are actually usable from
the Hugging Face Hub.

"push_to_hub returned without an exception" is not the same as "a stranger can
download this and translate Ekegusii with it". The failure this catches is
specific and quiet: `guz_Latn` is an *added* token, so a repository that got the
weights but not the tokenizer looks complete, downloads fine, and cannot produce
Ekegusii at all.

    python tests/verify_hf_uploads.py                 # metadata + tokenizer
    python tests/verify_hf_uploads.py --full          # also translates, ~7 GB
    python tests/verify_hf_uploads.py --user samuelabrha

It first lists every model on the account, so a repository under an unexpected
name shows up as a name mismatch rather than as a missing model.

Authentication: uses the token cached by `hf auth login`, or $HF_TOKEN. Never
paste a token into a chat window or a shell whose history you do not control.
"""
from __future__ import annotations

import argparse
import os
import sys

GUZ = "guz_Latn"
ENG = "eng_Latn"
PROBE = "Report suspected cholera cases to the nearest health facility."

# name -> (expected repo suffix, what it is)
EXPECTED = {
    "stage1": ("nllb-200-600M-ekegusii-stage1",
               "baseline - general Ekegusii from Bible + storybooks"),
    "stage2": ("nllb-200-600M-ekegusii-psa",
               "headline - PSA-adapted"),
    "mixed":  ("nllb-200-600M-ekegusii-mixed",
               "control - one pass, no curriculum"),
}

# What a usable repo needs. Note what is NOT here: special_tokens_map.json.
# transformers v5 folds those entries into tokenizer_config.json and frequently
# does not write the file at all, so requiring it fails perfectly good uploads.
# The authoritative check is further down - load the tokenizer and look for the
# token. A file list can only ever be a cheap pre-filter.
WEIGHTS_ANY = {"model.safetensors", "pytorch_model.bin", "model.safetensors.index.json"}
TOKENIZER_ANY = {"tokenizer.json", "sentencepiece.bpe.model"}
REQUIRED = {"config.json", "tokenizer_config.json"}


def find_repo(models: dict, suffix: str, user: str):
    """Exact name first, then anything on the account that looks like it."""
    exact = f"{user}/{suffix}"
    if exact in models:
        return exact, None
    key = suffix.rsplit("-", 1)[-1].lower()           # "stage1" / "psa" / "mixed"
    near = [r for r in models if key in r.lower().rsplit("/", 1)[-1]]
    if len(near) == 1:
        return near[0], f"found under a different name: {near[0]}"
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="samuelabrha", help="Hugging Face username")
    ap.add_argument("--full", action="store_true",
                    help="download each model and actually translate (~7 GB)")
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        print(f"cannot import huggingface_hub: {type(exc).__name__}: {exc}\n")
        print(f"this script is running on:\n  {sys.executable}\n")
        print("install into THAT interpreter specifically:")
        print(f'  "{sys.executable}" -m pip install huggingface_hub transformers')
        return 2

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)

    try:
        me = api.whoami()
        print(f"authenticated as {me.get('name', '?')}\n")
    except Exception:
        print("NOT AUTHENTICATED - private repositories are invisible without a "
              "token.\n")
        print("Log in with the interpreter running this script:")
        print(f'  "{sys.executable}" -c "from huggingface_hub import login; login()"')
        return 2

    # ---- what is actually on the account -----------------------------------
    try:
        found = {m.id: m for m in api.list_models(author=args.user)}
    except Exception as exc:
        print(f"could not list models for {args.user}: {exc}")
        found = {}

    print(f"models on {args.user} ({len(found)}):")
    for rid in sorted(found):
        print(f"    {rid}")
    print()

    failures, notes = [], []
    for name, (suffix, role) in EXPECTED.items():
        repo, note = find_repo(found, suffix, args.user)
        print(f"--- {name}  ({role})")
        if repo is None:
            print(f"    expected {args.user}/{suffix}")
            print(f"    FAIL  no such repository, and nothing on the account "
                  f"resembles it")
            failures.append((name, f"{args.user}/{suffix} does not exist"))
            print()
            continue
        print(f"    {repo}")
        if note:
            print(f"    NOTE  {note}")
            notes.append((name, repo))

        try:
            info = api.model_info(repo, files_metadata=True)
        except Exception as exc:
            print(f"    FAIL  cannot read repo: {type(exc).__name__}: {exc}\n")
            failures.append((name, "repo not readable"))
            continue

        sizes = {s.rfilename: (s.size or 0) for s in info.siblings}
        files = set(sizes)
        total = sum(sizes.values()) / 1024 ** 3
        print(f"    private={info.private}  files={len(files)}  {total:.2f} GB")
        for fn in sorted(files):
            if sizes[fn] > 1024 ** 2:
                print(f"      {fn:<34} {sizes[fn] / 1024 ** 2:8.1f} MB")

        missing = sorted(REQUIRED - files)
        if not (WEIGHTS_ANY & files):
            missing.append("weights (model.safetensors / pytorch_model.bin)")
        if not (TOKENIZER_ANY & files):
            missing.append("tokenizer (tokenizer.json / sentencepiece.bpe.model)")
        if missing:
            print(f"    FAIL  missing: {', '.join(missing)}")
            failures.append((name, f"missing {missing}"))
            print()
            continue
        if not info.private:
            print("    WARN  PUBLIC. These weights derive from the Ekegusii Revised")
            print("          Bible (c) Bible Society of Kenya - confirm before release.")

        # ---- the decisive check --------------------------------------------
        try:
            from transformers import AutoTokenizer
        except Exception as exc:
            print(f"    SKIP  transformers not importable: {type(exc).__name__}")
            print("          file check passed; token check needs transformers\n")
            continue

        try:
            tok = AutoTokenizer.from_pretrained(repo, token=token)
            guz_id = tok.convert_tokens_to_ids(GUZ)
            if guz_id == tok.unk_token_id:
                print(f"    FAIL  tokenizer loads but has no {GUZ} - this repo "
                      f"cannot produce Ekegusii")
                failures.append((name, f"no {GUZ} in tokenizer"))
                print()
                continue
            print(f"    ok    {GUZ} id {guz_id}, vocab {len(tok):,}")
        except Exception as exc:
            print(f"    FAIL  tokenizer unusable: {type(exc).__name__}: {exc}")
            failures.append((name, "tokenizer unusable"))
            print()
            continue

        if args.full:
            try:
                import torch
                from transformers import AutoModelForSeq2SeqLM
                m = AutoModelForSeq2SeqLM.from_pretrained(repo, token=token).eval()
                ids = ([tok.convert_tokens_to_ids(ENG)]
                       + tok(PROBE, add_special_tokens=False)["input_ids"]
                       + [tok.eos_token_id])
                with torch.no_grad():
                    out = m.generate(input_ids=torch.tensor([ids]),
                                     forced_bos_token_id=guz_id,
                                     max_new_tokens=64, num_beams=4)
                text = tok.batch_decode(out, skip_special_tokens=True)[0].strip()
                if not text:
                    raise RuntimeError("generated an empty string")
                print(f"    ok    {text}")
                del m
            except Exception as exc:
                print(f"    FAIL  generation: {type(exc).__name__}: {exc}")
                failures.append((name, "generation failed"))
        print()

    print("=" * 68)
    for name, repo in notes:
        print(f"  NOTE  {name} lives at {repo} - update HF_MIXED in serve/.env "
              f"and REPOS in notebook 05 to match")
    if failures:
        for name, why in failures:
            print(f"  FAIL  {name}: {why}")
        print(f"\n{len(failures)} of {len(EXPECTED)} systems are not usable.")
        return 1
    print(f"  all {len(EXPECTED)} systems are present and usable")
    if not args.full:
        print("  (re-run with --full to also confirm each one generates Ekegusii)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
