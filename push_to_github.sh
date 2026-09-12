#!/usr/bin/env bash
# Push the fine-tuning pipeline and its data to GitHub.
#
# The notebooks download data from raw.githubusercontent.com on the BRANCH named
# in nb_common.GITHUB_BRANCH, which is "master".
#
# This repo has both branches, and they have diverged: "master" carries the real
# work, while "main" is stale (an old flat layout from an earlier merge). We push
# to master, which is a fast-forward and needs no force. If you ever switch to
# main, change GITHUB_BRANCH to match or every data download will 404.
set -euo pipefail
cd "$(dirname "$0")"

echo "== staging pipeline code =="
git add .gitignore nb_common.py notebooks/ ekegusii/prepare_psa_ke.py \
        ekegusii/build_trilingual_corpus.py tests/test_aligner.py \
        README.md report.md

echo "== staging data the notebooks fetch at runtime =="
git add data/bible_en_guz_swh.csv \
        data/psa_ke_train.csv data/psa_ke_test.csv data/psa_ke_test_en_guz.csv \
        data/psa_ke_manifest.json \
        data/lughayangu_sentences.csv \
        data/PSA_KE_Final.csv data/_PSA_EnGuz.csv

git status --short

echo
read -r -p "Commit and push to origin/master? [y/N] " reply
[[ "$reply" == "y" || "$reply" == "Y" ]] || { echo "aborted"; exit 0; }

git diff --cached --quiet && { echo "nothing new to commit"; } || git commit -m "Add NLLB-200 Ekegusii fine-tuning pipeline

- Two-stage curriculum: general Ekegusii (Bible) then PSA adaptation,
  with 25% replay, plus a single-stage mixed control run
- Merge and clean the professor-supplied PSA corpora (PSA_KE_Final,
  _PSA_EnGuz) into 3,509 train / 573 PSA-register test rows
- Add guz_Latn to the NLLB tokenizer, initialised from kik_Latn
- Notebooks fetch their data from GitHub so a bare GPU node needs no uploads"

git push origin HEAD:master
echo
echo "pushed. Verify a raw URL resolves before running on the node:"
echo "  curl -sI https://raw.githubusercontent.com/SamAbr/PSA-MT/main/data/psa_ke_train.csv | head -1"
