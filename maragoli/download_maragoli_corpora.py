#!/usr/bin/env python3
"""
download_maragoli_corpora.py
============================
Acquires and preprocesses parallel datasets for Maragoli (Lulogooli / rag_Latn):
  1. KenTrans Lulogooli parallel dataset (Kencorpus/KenTrans - config 'llg')
  2. Luhya Multilingual dataset (mamakobe/luhya-multilingual-dataset)
  3. Prepares cleaned train/test/dev datasets for Maragoli NMT fine-tuning.

Outputs:
  data/maragoli_kentrans_clean.json
  data/maragoli_multilingual_clean.json
  data/maragoli_manifest.json
"""

import json
import os
import re
import sys
import urllib.request

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_hf_json(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode('utf-8'))

def clean_text(text):
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def acquire_kentrans():
    print("\n--- 1. Acquiring KenTrans Lulogooli Dataset ---")
    rows_data = []
    offset = 0
    length = 100
    total = 3692
    
    print(f"Downloading {total} rows from HuggingFace Kencorpus/KenTrans (llg)...")
    while offset < total:
        url = f"https://datasets-server.huggingface.co/rows?dataset=Kencorpus/KenTrans&config=llg&split=train&offset={offset}&length={length}"
        try:
            res = fetch_hf_json(url)
            batch = res.get('rows', [])
            if not batch:
                break
            for r in batch:
                row = r.get('row', {})
                src = clean_text(row.get('source'))
                tgt = clean_text(row.get('target'))
                pair = row.get('pair')
                src_lang = row.get('src_lang')
                tgt_lang = row.get('tgt_lang')
                
                if src and tgt:
                    rows_data.append({
                        'id': f"kentrans_{len(rows_data)+1}",
                        'src_lang': src_lang,
                        'tgt_lang': tgt_lang,
                        'source': src,
                        'target': tgt,
                        'pair': pair,
                        'domain': 'general_prose'
                    })
            offset += len(batch)
            if offset % 500 == 0 or offset >= total:
                print(f"  Downloaded {offset}/{total} rows...")
        except Exception as e:
            print(f"  Batch fetch failed at offset {offset}: {e}")
            break
            
    print(f"Successfully downloaded {len(rows_data)} KenTrans rows.")
    return rows_data

def acquire_luhya_multilingual():
    print("\n--- 2. Acquiring Luhya Multilingual Dataset ---")
    rows_data = []
    offset = 0
    length = 100
    limit = 1000
    
    print("Downloading Luhya Multilingual dataset...")
    while offset < limit:
        url = f"https://datasets-server.huggingface.co/rows?dataset=mamakobe/luhya-multilingual-dataset&config=default&split=train&offset={offset}&length={length}"
        try:
            res = fetch_hf_json(url)
            batch = res.get('rows', [])
            if not batch:
                break
            for r in batch:
                row = r.get('row', {})
                eng = clean_text(row.get('english_text'))
                luh = clean_text(row.get('luhya_text'))
                swh = clean_text(row.get('swahili_text'))
                dialect = row.get('dialect_name', '')
                domain = row.get('domain', 'general')
                
                if eng and luh:
                    rows_data.append({
                        'id': f"luhya_multi_{len(rows_data)+1}",
                        'english': eng,
                        'maragoli_luhya': luh,
                        'swahili': swh,
                        'dialect': dialect,
                        'domain': domain
                    })
            offset += len(batch)
        except Exception as e:
            print(f"  Multi batch fetch failed at offset {offset}: {e}")
            break
            
    print(f"Successfully downloaded {len(rows_data)} Luhya Multilingual rows.")
    return rows_data

def main():
    print("==================================================")
    print(" Maragoli NMT Data Acquisition Pipeline ")
    print("==================================================")
    
    kentrans_rows = acquire_kentrans()
    luhya_multi_rows = acquire_luhya_multilingual()
    
    # Save raw/clean outputs
    kentrans_path = os.path.join(DATA_DIR, "maragoli_kentrans_clean.json")
    with open(kentrans_path, 'w', encoding='utf-8') as f:
        json.dump(kentrans_rows, f, ensure_ascii=False, indent=2)
        
    multi_path = os.path.join(DATA_DIR, "maragoli_multilingual_clean.json")
    with open(multi_path, 'w', encoding='utf-8') as f:
        json.dump(luhya_multi_rows, f, ensure_ascii=False, indent=2)
        
    manifest = {
        'language': 'Maragoli / Lulogooli',
        'iso_code': 'rag_Latn / llg',
        'kentrans_pairs_count': len(kentrans_rows),
        'luhya_multilingual_pairs_count': len(luhya_multi_rows),
        'total_acquired_pairs': len(kentrans_rows) + len(luhya_multi_rows),
        'output_files': [
            os.path.basename(kentrans_path),
            os.path.basename(multi_path)
        ]
    }
    
    manifest_path = os.path.join(DATA_DIR, "maragoli_manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        
    print("\n==================================================")
    print(f" Manifest Saved: {manifest_path}")
    print(f" Total acquired parallel sentence pairs: {manifest['total_acquired_pairs']:,}")
    print("==================================================")

if __name__ == "__main__":
    main()
