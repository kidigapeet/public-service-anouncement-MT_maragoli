#!/usr/bin/env python3
"""
generate_colab_notebook.py
==========================
Generates a complete, self-contained Google Colab notebook (Maragoli_NMT_Training_Colab.ipynb)
including dataset loading, tokenizer extension, GPU fine-tuning, repetition penalty generation,
and full translation evaluation.
"""

import json
from pathlib import Path

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Maragoli (Lulogooli) NMT Model Training on Google Colab\n",
                "\n",
                "This notebook fine-tunes **NLLB-200 (600M)** for **English-to-Maragoli** and **Swahili-to-Maragoli** Neural Machine Translation.\n",
                "\n",
                "### How to run:\n",
                "1. Upload your dataset files (`maragoli_train.csv`, `maragoli_dev.csv`, `maragoli_test.csv`) to the Colab sidebar 📁.\n",
                "2. Enable T4 GPU (*Runtime > Change runtime type > T4 GPU*).\n",
                "3. Click **Runtime > Run all**."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 1: Install Dependencies\n",
                "!pip install -q transformers sentencepiece sacrebleu pandas torch tqdm evaluate"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 2: Verify GPU Hardware\n",
                "import torch\n",
                "print(f\"PyTorch Version: {torch.__version__}\")\n",
                "print(f\"GPU Available  : {torch.cuda.is_available()}\")\n",
                "if torch.cuda.is_available():\n",
                "    print(f\"Device Name    : {torch.cuda.get_device_name(0)}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 3: Register rag_Latn Language Token in NLLB-200\n",
                "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM\n",
                "\n",
                "BASE_MODEL = \"facebook/nllb-200-distilled-600M\"\n",
                "RAG = \"rag_Latn\"\n",
                "RAG_INIT_FROM = \"kik_Latn\"\n",
                "\n",
                "print(f\"Loading Base Model: {BASE_MODEL}...\")\n",
                "tok = AutoTokenizer.from_pretrained(BASE_MODEL, src_lang=\"eng_Latn\", tgt_lang=\"swh_Latn\")\n",
                "model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)\n",
                "\n",
                "# Add rag_Latn token initialized from Kikuyu (kik_Latn) weights + subtle noise\n",
                "tok.add_tokens([RAG], special_tokens=True)\n",
                "model.resize_token_embeddings(len(tok))\n",
                "\n",
                "rag_id = tok.convert_tokens_to_ids(RAG)\n",
                "src_id = tok.convert_tokens_to_ids(RAG_INIT_FROM)\n",
                "\n",
                "with torch.no_grad():\n",
                "    emb = model.get_input_embeddings().weight\n",
                "    emb[rag_id] = emb[src_id].clone()\n",
                "    emb[rag_id] += torch.randn_like(emb[rag_id]) * 0.01 * emb.std()\n",
                "\n",
                "print(f\"Registered '{RAG}' at token ID {rag_id} initialized from '{RAG_INIT_FROM}' (ID {src_id})\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 4: Load Datasets from Uploaded CSV Files\n",
                "import pandas as pd\n",
                "from torch.utils.data import Dataset\n",
                "import os\n",
                "\n",
                "# Check uploaded files\n",
                "for fname in ['maragoli_train.csv', 'maragoli_dev.csv', 'maragoli_test.csv']:\n",
                "    if not os.path.exists(fname):\n",
                "        print(f\"⚠️ Warning: {fname} not found in current directory! Please upload it via the left sidebar 📁.\")\n",
                "    else:\n",
                "        print(f\"✅ Found {fname}\")\n",
                "\n",
                "train_df = pd.read_csv('maragoli_train.csv')\n",
                "dev_df = pd.read_csv('maragoli_dev.csv')\n",
                "test_df = pd.read_csv('maragoli_test.csv')\n",
                "\n",
                "print(f\"Train set: {len(train_df):,} rows\")\n",
                "print(f\"Dev set  : {len(dev_df):,} rows\")\n",
                "print(f\"Test set : {len(test_df):,} rows\")\n",
                "\n",
                "class MaragoliDataset(Dataset):\n",
                "    def __init__(self, df, tok, max_len=128):\n",
                "        self.rows = df.to_dict('records')\n",
                "        self.tok = tok\n",
                "        self.max_len = max_len\n",
                "        self.eos_id = tok.eos_token_id\n",
                "\n",
                "    def __len__(self):\n",
                "        return len(self.rows)\n",
                "\n",
                "    def __getitem__(self, idx):\n",
                "        row = self.rows[idx]\n",
                "        src_text = str(row['source_text'])\n",
                "        tgt_text = str(row['target_text'])\n",
                "        src_lang = str(row.get('src_lang', 'eng_Latn'))\n",
                "        tgt_lang = str(row.get('tgt_lang', 'rag_Latn'))\n",
                "\n",
                "        src_lang_id = self.tok.convert_tokens_to_ids(src_lang)\n",
                "        if src_lang_id == self.tok.unk_token_id:\n",
                "            src_lang_id = self.tok.convert_tokens_to_ids('eng_Latn')\n",
                "            \n",
                "        src_tokens = self.tok(src_text, add_special_tokens=False, truncation=True, max_length=self.max_len - 2)[\"input_ids\"]\n",
                "        input_ids = [src_lang_id] + src_tokens + [self.eos_id]\n",
                "\n",
                "        tgt_lang_id = self.tok.convert_tokens_to_ids(tgt_lang)\n",
                "        if tgt_lang_id == self.tok.unk_token_id:\n",
                "            tgt_lang_id = self.tok.convert_tokens_to_ids('rag_Latn')\n",
                "\n",
                "        tgt_tokens = self.tok(tgt_text, add_special_tokens=False, truncation=True, max_length=self.max_len - 2)[\"input_ids\"]\n",
                "        labels = [tgt_lang_id] + tgt_tokens + [self.eos_id]\n",
                "\n",
                "        return {\n",
                "            \"input_ids\": input_ids,\n",
                "            \"attention_mask\": [1] * len(input_ids),\n",
                "            \"labels\": labels\n",
                "        }\n",
                "\n",
                "train_dataset = MaragoliDataset(train_df, tok)\n",
                "dev_dataset = MaragoliDataset(dev_df, tok)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 5: Fine-Tune NLLB-200 Model on GPU\n",
                "from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq\n",
                "\n",
                "training_args = Seq2SeqTrainingArguments(\n",
                "    output_dir=\"./nllb600m-maragoli-model\",\n",
                "    per_device_train_batch_size=8,\n",
                "    per_device_eval_batch_size=8,\n",
                "    learning_rate=5e-5,\n",
                "    num_train_epochs=3,\n",
                "    eval_strategy=\"epoch\",\n",
                "    save_strategy=\"epoch\",\n",
                "    logging_steps=50,\n",
                "    save_total_limit=2,\n",
                "    predict_with_generate=True,\n",
                "    fp16=torch.cuda.is_available(),\n",
                "    report_to=\"none\"\n",
                ")\n",
                "\n",
                "collator = DataCollatorForSeq2Seq(tok, model=model, pad_to_multiple_of=8)\n",
                "\n",
                "trainer = Seq2SeqTrainer(\n",
                "    model=model,\n",
                "    args=training_args,\n",
                "    train_dataset=train_dataset,\n",
                "    eval_dataset=dev_dataset,\n",
                "    data_collator=collator,\n",
                "    tokenizer=tok\n",
                ")\n",
                "\n",
                "print(\"🚀 Starting Model Fine-Tuning...\")\n",
                "trainer.train()\n",
                "print(\"✅ Training Complete!\")\n",
                "\n",
                "# Save fine-tuned weights\n",
                "model.save_pretrained(\"./nllb600m-maragoli-fine-tuned\")\n",
                "tok.save_pretrained(\"./nllb600m-maragoli-fine-tuned\")\n",
                "print(\"Saved fine-tuned checkpoint to ./nllb600m-maragoli-fine-tuned\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 6: Test Translation Inference & Full Evaluation\n",
                "import sacrebleu\n",
                "\n",
                "def translate(text, src_lang=\"eng_Latn\", tgt_lang=\"rag_Latn\"):\n",
                "    eos_id = tok.eos_token_id\n",
                "    src_lang_id = tok.convert_tokens_to_ids(src_lang)\n",
                "    src_tokens = tok(text, add_special_tokens=False, truncation=True, max_length=126)[\"input_ids\"]\n",
                "    input_ids = torch.tensor([[src_lang_id] + src_tokens + [eos_id]]).to(model.device)\n",
                "    \n",
                "    tgt_lang_id = tok.convert_tokens_to_ids(tgt_lang)\n",
                "    out = model.generate(\n",
                "        input_ids=input_ids,\n",
                "        forced_bos_token_id=tgt_lang_id,\n",
                "        max_new_tokens=128,\n",
                "        num_beams=4,\n",
                "        no_repeat_ngram_size=3,\n",
                "        repetition_penalty=1.2\n",
                "    )\n",
                "    return tok.batch_decode(out, skip_special_tokens=True)[0]\n",
                "\n",
                "test_prompt_sw = \"Ripoti wagonjwa wanaoshukiwa katika kituo cha afya kilicho karibu.\"\n",
                "print(\"\\nSource Input (Kiswahili):\", test_prompt_sw)\n",
                "print(\"Fine-Tuned Maragoli Output:\", translate(test_prompt_sw, \"swh_Latn\", \"rag_Latn\"))\n",
                "\n",
                "test_prompt_en = \"Report suspected health cases to the nearest facility.\"\n",
                "print(\"\\nSource Input (English):\", test_prompt_en)\n",
                "print(\"Fine-Tuned Maragoli Output:\", translate(test_prompt_en, \"eng_Latn\", \"rag_Latn\"))\n",
                "\n",
                "# Evaluate full test set\n",
                "print(f\"\\nEvaluating full test set ({len(test_df)} rows)...\")\n",
                "hyps, refs = [], []\n",
                "for idx, r in test_df.iterrows():\n",
                "    src_txt = str(r['source_text'])\n",
                "    src_l = str(r.get('src_lang', 'eng_Latn'))\n",
                "    pred = translate(src_txt, src_l, 'rag_Latn')\n",
                "    hyps.append(pred)\n",
                "    refs.append(str(r['target_text']))\n",
                "\n",
                "chrf = sacrebleu.corpus_chrf(hyps, [refs])\n",
                "bleu = sacrebleu.corpus_bleu(hyps, [refs])\n",
                "\n",
                "print(f\"\\n📊 Final Evaluation Results on Test Set:\")\n",
                "print(f\"  chrF2++ : {chrf.score:.2f}\")\n",
                "print(f\"  BLEU    : {bleu.score:.2f}\")"
            ]
        }
    ],
    "metadata": {
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

out_path = Path(__file__).resolve().parent.parent / "notebooks" / "Maragoli_NMT_Training_Colab.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"Updated Colab Notebook: {out_path}")
