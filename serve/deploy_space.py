#!/usr/bin/env python3
"""
deploy_space.py - create or update the public Hugging Face Space.

    python serve/deploy_space.py --space samuelabrha/ekegusii-psa-translator

Run it from anywhere; paths are resolved relative to this file.

What it uploads: app.py, static/, metrics/, requirements.txt, Dockerfile.space
(renamed to Dockerfile, which is what Spaces looks for) and space_README.md
(renamed to README.md, whose YAML front matter configures the Space).

What it does NOT upload: your token. The Space needs HF_TOKEN as a *secret* to
read the private model repository, and this script only tells you to set it - it
never asks you to type a token on a command line, where it would persist in
shell history.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# local path -> path inside the Space
FILES = {
    "app.py": "app.py",
    "requirements.txt": "requirements.txt",
    "static/index.html": "static/index.html",
    "metrics/evaluation_results.csv": "metrics/evaluation_results.csv",
    "Dockerfile.space": "Dockerfile",
    "space_README.md": "README.md",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True,
                    help="target Space, e.g. samuelabrha/ekegusii-psa-translator")
    ap.add_argument("--private", action="store_true",
                    help="create the Space private (default is public - the point "
                         "of this is a shareable link)")
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print(f'"{sys.executable}" -m pip install huggingface_hub')
        return 2

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    try:
        who = api.whoami()["name"]
    except Exception:
        print("Not logged in. Run:")
        print(f'  "{sys.executable}" -c "from huggingface_hub import login; login()"')
        return 2

    missing = [f for f in FILES if not (HERE / f).exists()]
    if missing:
        print(f"missing locally: {missing}")
        return 1

    api.create_repo(args.space, repo_type="space", space_sdk="docker",
                    private=args.private, exist_ok=True)
    print(f"space ready: https://huggingface.co/spaces/{args.space}")

    for local, remote in FILES.items():
        api.upload_file(path_or_fileobj=str(HERE / local), path_in_repo=remote,
                        repo_id=args.space, repo_type="space")
        print(f"  uploaded {local:<34} -> {remote}")

    print(f"\nAuthenticated as {who}. Two things left, both in the Space's "
          f"Settings tab:\n")
    print("  1. Add a SECRET named HF_TOKEN, holding a READ-scoped token.")
    print("     Without it the Space cannot read your private model repo and")
    print("     every translation will fail with a 401.")
    print("  2. Optionally add a secret FEEDBACK_REPO naming a dataset repo,")
    print("     e.g. 'yourname/ekegusii-feedback'. Without it, corrections are")
    print("     written to the container's disk and lost on every restart.\n")
    print(f"Then watch the build log at:")
    print(f"  https://huggingface.co/spaces/{args.space}?logs=build")
    print("First build takes several minutes - it compiles nothing, but it does")
    print("download the CPU torch wheel.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
