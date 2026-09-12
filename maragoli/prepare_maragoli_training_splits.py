#!/usr/bin/env python3
"""
prepare_maragoli_training_splits.py
===================================
Converts acquired Maragoli corpora (KenTrans + Luhya Multilingual) into
train / dev / test CSV splits formatted for Hugging Face Seq2SeqTrainer fine-tuning.

Generates:
  1. Kiswahili -> Maragoli splits (Primary Bantu-to-Bantu transfer benchmark):
     data/swahili_maragoli_train.csv
     data/swahili_maragoli_dev.csv
     data/swahili_maragoli_test.csv
  2. English -> Maragoli splits (Cross-family baseline):
     data/english_maragoli_train.csv
     data/english_maragoli_dev.csv
     data/english_maragoli_test.csv
  3. Default training splits (pointing to Swahili -> Maragoli for optimal Bantu transfer):
     data/maragoli_train.csv
     data/maragoli_dev.csv
     data/maragoli_test.csv
  4. data/maragoli_splits_manifest.json
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

def clean_pair(src, tgt):
    if not src or not tgt or not isinstance(src, str) or not isinstance(tgt, str):
        return None, None
    src = src.strip()
    tgt = tgt.strip()
    if len(src) < 2 or len(tgt) < 2:
        return None, None
    return src, tgt

def main():
    print("==================================================")
    print(" Preparing Maragoli Dataset Train/Dev/Test Splits ")
    print(" Kiswahili -> Maragoli (Bantu) & English -> Maragoli ")
    print("==================================================")
    
    C.set_seed(42)
    data_dir = C.INPUTS
    
    kentrans_path = data_dir / "maragoli_kentrans_clean.json"
    multi_path = data_dir / "maragoli_multilingual_clean.json"
    
    swahili_pairs = []
    english_pairs = []
    
    # 1. Process KenTrans (KenCorpus: llg <-> swa)
    if kentrans_path.exists():
        with open(kentrans_path, 'r', encoding='utf-8') as f:
            kt_data = json.load(f)
            print(f"Loaded {len(kt_data)} KenTrans rows.")
            for row in kt_data:
                # In KenTrans 'llg-swa':
                # 'source' is Lulogooli/Maragoli
                # 'target' is Kiswahili
                maragoli_text = row.get('source')
                swahili_text = row.get('target')
                
                s_clean, m_clean = clean_pair(swahili_text, maragoli_text)
                if s_clean and m_clean:
                    swahili_pairs.append({
                        'id': f"{row['id']}_swh",
                        'src_lang': C.SWH,
                        'tgt_lang': C.RAG,
                        'source_text': s_clean,
                        'target_text': m_clean,
                        'domain': row.get('domain', 'general_prose')
                    })
                    
    # 2. Process Luhya Multilingual (English, Swahili, Maragoli)
    if multi_path.exists():
        with open(multi_path, 'r', encoding='utf-8') as f:
            ml_data = json.load(f)
            print(f"Loaded {len(ml_data)} Multilingual rows.")
            for row in ml_data:
                maragoli_text = row.get('maragoli_luhya')
                swahili_text = row.get('swahili')
                english_text = row.get('english')
                
                # Swahili -> Maragoli
                s_clean, m_clean = clean_pair(swahili_text, maragoli_text)
                if s_clean and m_clean:
                    swahili_pairs.append({
                        'id': f"{row['id']}_multi_swh",
                        'src_lang': C.SWH,
                        'tgt_lang': C.RAG,
                        'source_text': s_clean,
                        'target_text': m_clean,
                        'domain': row.get('domain', 'general')
                    })
                
                # English -> Maragoli
                e_clean, m2_clean = clean_pair(english_text, maragoli_text)
                if e_clean and m2_clean:
                    english_pairs.append({
                        'id': f"{row['id']}_multi_eng",
                        'src_lang': C.ENG,
                        'tgt_lang': C.RAG,
                        'source_text': e_clean,
                        'target_text': m2_clean,
                        'domain': row.get('domain', 'general')
                    })

    # Deduplicate Swahili -> Maragoli
    df_swh = pd.DataFrame(swahili_pairs).drop_duplicates(subset=['source_text', 'target_text']).reset_index(drop=True)
    random.seed(42)
    df_swh = df_swh.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Deduplicate English -> Maragoli
    df_eng = pd.DataFrame(english_pairs).drop_duplicates(subset=['source_text', 'target_text']).reset_index(drop=True) if english_pairs else pd.DataFrame()
    if not df_eng.empty:
        df_eng = df_eng.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"\nUnique Kiswahili -> Maragoli pairs : {len(df_swh):,}")
    print(f"Unique English -> Maragoli pairs   : {len(df_eng):,}")

    def split_and_save(df: pd.DataFrame, prefix: str):
        if df.empty:
            return None, None, None
        n_total = len(df)
        n_dev = max(20, int(n_total * 0.075))
        n_test = max(20, int(n_total * 0.075))
        n_train = n_total - n_dev - n_test
        
        train = df.iloc[:n_train]
        dev = df.iloc[n_train:n_train+n_dev]
        test = df.iloc[n_train+n_dev:]
        
        p_train = data_dir / f"{prefix}_train.csv"
        p_dev = data_dir / f"{prefix}_dev.csv"
        p_test = data_dir / f"{prefix}_test.csv"
        
        train.to_csv(p_train, index=False, encoding='utf-8')
        dev.to_csv(p_dev, index=False, encoding='utf-8')
        test.to_csv(p_test, index=False, encoding='utf-8')
        return train, dev, test

    # Save Kiswahili -> Maragoli splits
    swh_tr, swh_dv, swh_ts = split_and_save(df_swh, "swahili_maragoli")
    
    # Save English -> Maragoli splits
    if not df_eng.empty:
        eng_tr, eng_dv, eng_ts = split_and_save(df_eng, "english_maragoli")
    
    # Save default training files as Kiswahili -> Maragoli (our primary superior model)
    df_swh_train = data_dir / "swahili_maragoli_train.csv"
    df_swh_dev = data_dir / "swahili_maragoli_dev.csv"
    df_swh_test = data_dir / "swahili_maragoli_test.csv"
    
    df_swh.iloc[:len(swh_tr)].to_csv(data_dir / "maragoli_train.csv", index=False, encoding='utf-8')
    df_swh.iloc[len(swh_tr):len(swh_tr)+len(swh_dv)].to_csv(data_dir / "maragoli_dev.csv", index=False, encoding='utf-8')
    df_swh.iloc[len(swh_tr)+len(swh_dv):].to_csv(data_dir / "maragoli_test.csv", index=False, encoding='utf-8')

    manifest = {
        'primary_system': 'Kiswahili -> Maragoli (swh_Latn -> rag_Latn)',
        'baseline_system': 'English -> Maragoli (eng_Latn -> rag_Latn)',
        'swahili_maragoli': {
            'total_pairs': len(df_swh),
            'train_count': len(swh_tr),
            'dev_count': len(swh_dv),
            'test_count': len(swh_ts),
        },
        'english_maragoli': {
            'total_pairs': len(df_eng),
            'train_count': len(df_eng) - (2 * max(20, int(len(df_eng) * 0.075))) if not df_eng.empty else 0,
        },
        'files': {
            'train': str(data_dir / "maragoli_train.csv"),
            'dev': str(data_dir / "maragoli_dev.csv"),
            'test': str(data_dir / "maragoli_test.csv"),
            'swahili_train': str(df_swh_train),
            'swahili_dev': str(df_swh_dev),
            'swahili_test': str(df_swh_test)
        }
    }
    
    C.save_json(manifest, data_dir / "maragoli_splits_manifest.json")
    
    print("\n==================================================")
    print(f" Kiswahili -> Maragoli Train: {len(swh_tr):,} rows")
    print(f" Kiswahili -> Maragoli Dev  : {len(swh_dv):,} rows")
    print(f" Kiswahili -> Maragoli Test : {len(swh_ts):,} rows")
    print(f" Saved default splits to data/maragoli_*.csv")
    print("==================================================")

if __name__ == "__main__":
    main()
