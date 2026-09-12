#!/usr/bin/env python3
"""
train_maragoli.py
=================
Fine-tunes NLLB-200 for English-to-Maragoli and Swahili-to-Maragoli translation.

Uses the extended tokenizer checkpoint (artifacts/nllb600m-rag-init) and
trains on data/maragoli_train.csv and data/maragoli_dev.csv.

Usage:
  python maragoli/train_maragoli.py --epochs 3 --batch-size 8 --lr 5e-5
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import nb_common as C

class MaragoliDataset(Dataset):
    def __init__(self, df, tok, max_len=128):
        self.rows = df.to_dict('records')
        self.tok = tok
        self.max_len = max_len
        self.eos_id = tok.eos_token_id

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        src_text = str(row['source_text'])
        tgt_text = str(row['target_text'])
        src_lang = str(row.get('src_lang', C.ENG))
        tgt_lang = str(row.get('tgt_lang', C.RAG))

        # Format input tokens: [src_lang] + tokens + [eos]
        src_lang_id = self.tok.convert_tokens_to_ids(src_lang)
        if src_lang_id == self.tok.unk_token_id:
            src_lang_id = self.tok.convert_tokens_to_ids(C.ENG)
            
        src_token_ids = self.tok(src_text, add_special_tokens=False, truncation=True, max_length=self.max_len - 2)["input_ids"]
        input_ids = [src_lang_id] + src_token_ids + [self.eos_id]

        # Format target labels: [tgt_lang] + tokens + [eos]
        tgt_lang_id = self.tok.convert_tokens_to_ids(tgt_lang)
        if tgt_lang_id == self.tok.unk_token_id:
            tgt_lang_id = self.tok.convert_tokens_to_ids(C.RAG)

        tgt_token_ids = self.tok(tgt_text, add_special_tokens=False, truncation=True, max_length=self.max_len - 2)["input_ids"]
        labels = [tgt_lang_id] + tgt_token_ids + [self.eos_id]

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels
        }

def main():
    parser = argparse.ArgumentParser(description="Fine-tune NLLB-200 for Maragoli NMT")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    args = parser.parse_args()

    print("==================================================")
    print(" Fine-Tuning NLLB-200 for Maragoli NMT ")
    print("==================================================")

    init_model_dir = C.ARTIFACTS / "nllb600m-rag-init"
    if not init_model_dir.exists():
        print(f"Error: Initial model checkpoint {init_model_dir} not found!")
        print("Please run `py maragoli/extend_maragoli_tokenizer.py` first.")
        return

    print(f"Loading Extended Tokenizer & Model from {init_model_dir}...")
    tok = AutoTokenizer.from_pretrained(init_model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(init_model_dir)

    train_path = C.INPUTS / "maragoli_train.csv"
    dev_path = C.INPUTS / "maragoli_dev.csv"

    if not train_path.exists():
        print(f"Error: Training split {train_path} not found!")
        return

    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)

    print(f"Train Dataset: {len(train_df):,} rows")
    print(f"Dev Dataset  : {len(dev_df):,} rows")

    train_dataset = MaragoliDataset(train_df, tok)
    dev_dataset = MaragoliDataset(dev_df, tok)

    output_dir = C.ARTIFACTS / "nllb600m-maragoli-fine-tuned"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    collator = DataCollatorForSeq2Seq(tok, model=model, pad_to_multiple_of=8)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=collator,
        tokenizer=tok
    )

    print("\nStarting model fine-tuning...")
    trainer.train()

    print(f"\nSaving fine-tuned model checkpoint to {output_dir}...")
    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)
    print("==================================================")
    print(" Fine-tuning complete! Model saved successfully. ")
    print("==================================================")

if __name__ == "__main__":
    main()
