#!/usr/bin/env python3
"""
prepare_maragoli_training_splits.py
===================================
Converts acquired Maragoli corpora (KenTrans + Luhya Multilingual) into
train / dev / test CSV splits formatted for Hugging Face Seq2SeqTrainer fine-tuning.

Outputs:
  data/maragoli_train.csv
  data/maragoli_dev.csv
  data/maragoli_test.csv
  data/maragoli_splits_manifest.json
"""

import json
import os
import random
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import nb_common as C

def main():
    print("==================================================")
    print(" Preparing Maragoli Dataset Train/Dev/Test Splits ")
    print("==================================================")
    
    C.set_seed(42)
    data_dir = C.INPUTS
    
    kentrans_path = data_dir / "maragoli_kentrans_clean.json"
    multi_path = data_dir / "maragoli_multilingual_clean.json"
    
    pairs = []
    
    if kentrans_path.exists():
        with open(kentrans_path, 'r', encoding='utf-8') as f:
            kt_data = json.load(f)
            print(f"Loaded {len(kt_data)} KenTrans rows.")
            for row in kt_data:
                pairs.append({
                    'id': row['id'],
                    'src_lang': C.SWH if row['src_lang'] == 'swa' else C.ENG,
                    'tgt_lang': C.RAG,
                    'source_text': row['source'],
                    'target_text': row['target'],
                    'domain': row.get('domain', 'general')
                })
                
    if multi_path.exists():
        with open(multi_path, 'r', encoding='utf-8') as f:
            ml_data = json.load(f)
            print(f"Loaded {len(ml_data)} Multilingual rows.")
            for row in ml_data:
                if row.get('english') and row.get('maragoli_luhya'):
                    pairs.append({
                        'id': row['id'] + "_en",
                        'src_lang': C.ENG,
                        'tgt_lang': C.RAG,
                        'source_text': row['english'],
                        'target_text': row['maragoli_luhya'],
                        'domain': row.get('domain', 'general')
                    })
                if row.get('swahili') and row.get('maragoli_luhya'):
                    pairs.append({
                        'id': row['id'] + "_sw",
                        'src_lang': C.SWH,
                        'tgt_lang': C.RAG,
                        'source_text': row['swahili'],
                        'target_text': row['maragoli_luhya'],
                        'domain': row.get('domain', 'general')
                    })
                    
    print(f"\nTotal parallel sentence pairs assembled: {len(pairs):,}")
    
    if not pairs:
        print("Error: No data pairs found!")
        return

    # Shuffle deterministically
    random.seed(42)
    random.shuffle(pairs)
    
    df = pd.DataFrame(pairs)
    # Deduplicate on source and target
    df = df.drop_duplicates(subset=['source_text', 'target_text']).reset_index(drop=True)
    print(f"Unique sentence pairs after deduplication: {len(df):,}")
    
    # Splits: 85% train, 7.5% dev, 7.5% test
    n_total = len(df)
    n_dev = int(n_total * 0.075)
    n_test = int(n_total * 0.075)
    n_train = n_total - n_dev - n_test
    
    train_df = df.iloc[:n_train]
    dev_df = df.iloc[n_train:n_train+n_dev]
    test_df = df.iloc[n_train+n_dev:]
    
    train_path = data_dir / "maragoli_train.csv"
    dev_path = data_dir / "maragoli_dev.csv"
    test_path = data_dir / "maragoli_test.csv"
    
    train_df.to_csv(train_path, index=False, encoding='utf-8')
    dev_df.to_csv(dev_path, index=False, encoding='utf-8')
    test_df.to_csv(test_path, index=False, encoding='utf-8')
    
    manifest = {
        'total_pairs': len(df),
        'train_count': len(train_df),
        'dev_count': len(dev_df),
        'test_count': len(test_df),
        'files': {
            'train': str(train_path),
            'dev': str(dev_path),
            'test': str(test_path)
        }
    }
    
    C.save_json(manifest, data_dir / "maragoli_splits_manifest.json")
    
    print("\n==================================================")
    print(f" Train Split : {len(train_df):,} rows -> {train_path}")
    print(f" Dev Split   : {len(dev_df):,} rows -> {dev_path}")
    print(f" Test Split  : {len(test_df):,} rows -> {test_path}")
    print("==================================================")

if __name__ == "__main__":
    main()
