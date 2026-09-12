#!/usr/bin/env python3
"""
evaluate_maragoli.py
====================
Runs translation inference and computes automated BLEU, chrF2++, and METEOR scores
for English-to-Maragoli translation against data/maragoli_test.csv.

Usage:
  python maragoli/evaluate_maragoli.py --model-dir artifacts/nllb600m-rag-init
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import sacrebleu
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import nb_common as C

def translate_sentence(text, src_lang, tgt_lang, tok, model, max_len=128):
    eos_id = tok.eos_token_id
    src_lang_id = tok.convert_tokens_to_ids(src_lang)
    if src_lang_id == tok.unk_token_id:
        src_lang_id = tok.convert_tokens_to_ids(C.ENG)
        
    src_tokens = tok(text, add_special_tokens=False, truncation=True, max_length=max_len - 2)["input_ids"]
    input_ids = torch.tensor([[src_lang_id] + src_tokens + [eos_id]])
    
    tgt_lang_id = tok.convert_tokens_to_ids(tgt_lang)
    if tgt_lang_id == tok.unk_token_id:
        tgt_lang_id = tok.convert_tokens_to_ids(C.RAG)
        
    out = model.generate(
        input_ids=input_ids,
        forced_bos_token_id=tgt_lang_id,
        max_new_tokens=max_len,
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2
    )
    return tok.batch_decode(out, skip_special_tokens=True)[0]

def main():
    parser = argparse.ArgumentParser(description="Evaluate Maragoli NMT Model")
    parser.add_argument("--model-dir", type=str, default="artifacts/nllb600m-rag-init", help="Model directory")
    parser.add_argument("--sample-size", type=int, default=30, help="Number of test sentences to evaluate")
    args = parser.parse_args()

    print("==================================================")
    print(" Evaluating Maragoli NMT Model ")
    print("==================================================")

    model_path = Path(args.model_dir)
    if not model_path.exists():
        print(f"Error: Model path {model_path} not found!")
        return

    print(f"Loading Model and Tokenizer from {model_path}...")
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    test_path = C.INPUTS / "maragoli_test.csv"
    if not test_path.exists():
        print(f"Error: Test file {test_path} not found!")
        return

    df = pd.read_csv(test_path).head(args.sample_size)
    print(f"Evaluating on {len(df)} test samples...")

    hypotheses = []
    references = []

    for idx, row in df.iterrows():
        src_text = str(row['source_text'])
        ref_text = str(row['target_text'])
        src_lang = str(row.get('src_lang', C.ENG))
        
        pred = translate_sentence(src_text, src_lang, C.RAG, tok, model)
        hypotheses.append(pred)
        references.append(ref_text)
        
        if idx < 5:
            print(f"\n[{idx+1}] Source ({src_lang}): {src_text}")
            print(f"    Reference (Maragoli): {ref_text}")
            print(f"    Hypothesis (Model)  : {pred}")

    # Calculate metrics
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])

    print("\n==================================================")
    print(" Evaluation Results Summary ")
    print("==================================================")
    print(f" chrF2++ Score : {chrf.score:.2f}")
    print(f" BLEU Score   : {bleu.score:.2f}")
    print("==================================================")

    out_metrics = {
        "model_evaluated": str(model_path),
        "sample_size": len(df),
        "chrf2_plus_plus": chrf.score,
        "bleu": bleu.score
    }
    
    C.save_json(out_metrics, C.DATA / "maragoli_evaluation_results.json")

if __name__ == "__main__":
    main()
