#!/usr/bin/env python3
"""
extend_maragoli_tokenizer.py
============================
Extends the NLLB-200 tokenizer with the Maragoli language token (rag_Latn),
initializes its embedding weights from Kikuyu (kik_Latn) with subtle noise,
and saves the initial checkpoint to artifacts/nllb600m-rag-init.

Usage:
  python maragoli/extend_maragoli_tokenizer.py
"""

import os
import sys
from pathlib import Path

# Insert project root into sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import nb_common as C
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def main():
    print("==================================================")
    print(" Extending NLLB-200 Tokenizer for Maragoli (rag_Latn) ")
    print("==================================================")
    
    C.set_seed()
    print(f"Loading Base Model: {C.BASE_MODEL}...")
    tok = AutoTokenizer.from_pretrained(C.BASE_MODEL, src_lang=C.ENG, tgt_lang=C.SWH)
    model = AutoModelForSeq2SeqLM.from_pretrained(C.BASE_MODEL)
    
    initial_vocab_size = len(tok)
    print(f"Initial Vocabulary Size: {initial_vocab_size:,}")
    print(f"Initial Embedding Matrix: {tuple(model.get_input_embeddings().weight.shape)}")
    
    # Extend tokenizer with rag_Latn token initialized from kik_Latn
    print(f"\nAdding language token '{C.RAG}' initialized from '{C.RAG_INIT_FROM}'...")
    rag_id, src_id = C.add_language_token(tok, model, C.RAG, C.RAG_INIT_FROM)
    
    emb = model.get_input_embeddings().weight
    sim = torch.nn.functional.cosine_similarity(emb[rag_id], emb[src_id], dim=0).item()
    
    print(f"New Vocab Size: {len(tok):,}")
    print(f"Token '{C.RAG}' ID: {rag_id}")
    print(f"Similarity cosine(rag_Latn, kik_Latn): {sim:.4f}")
    print(f"Updated Embedding Matrix: {tuple(emb.shape)}")
    
    # Output path
    out_dir = C.ARTIFACTS / "nllb600m-rag-init"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    
    info = {
        "target_language": "Maragoli / Lulogooli",
        "language_token": C.RAG,
        "init_from": C.RAG_INIT_FROM,
        "rag_token_id": int(rag_id),
        "source_token_id": int(src_id),
        "vocab_size": len(tok),
        "similarity_to_kikuyu": sim
    }
    
    info_path = C.DATA / "maragoli_tokenizer_extension.json"
    C.save_json(info, info_path)
    
    print("\n==================================================")
    print(f" Extended Tokenizer & Checkpoint Saved: {out_dir}")
    print(" Next Step: Prepare Maragoli parallel dataset splits for training.")
    print("==================================================")

if __name__ == "__main__":
    main()
