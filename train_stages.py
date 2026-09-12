#!/usr/bin/env python3
"""
train_stages.py
===============
Runs the fine-tuning curriculum from the command line instead of a notebook.

Why a script
------------
* A Jupyter kernel dies when the browser disconnects. An 8-hour run should not
  depend on a websocket.
* The GPU on this node is SHARED and frequently has almost nothing free. This
  script WAITS for enough VRAM before it starts, rather than crashing, so you
  can launch it and walk away.

Usage
-----
    # wait for 12 GiB free, then run stage 1 and stage 2
    nohup python train_stages.py --stages 1 2 --min-free 12 > train.log 2>&1 &
    tail -f train.log

    # just look at what the GPU is doing
    python train_stages.py --watch

Resuming a killed run:
    python train_stages.py --stages 1 --resume
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nb_common as C  # noqa: E402

import torch  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,  # noqa: E402
                          DataCollatorForSeq2Seq, EarlyStoppingCallback,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)

VOCAB_FALLBACK = 256_206


# ---------------------------------------------------------------------------
# GPU AVAILABILITY
# ---------------------------------------------------------------------------

def free_gib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.mem_get_info()[0] / 2 ** 30


def watch(interval: int = 10) -> None:
    """Print free VRAM until interrupted - use this to decide when to launch."""
    print("free VRAM on device 0 (Ctrl-C to stop)")
    try:
        while True:
            print(f"  {time.strftime('%H:%M:%S')}  {free_gib():6.2f} GiB free")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("stopped")


def wait_for_vram(min_free: float, timeout_min: int, poll: int = 30) -> None:
    """Block until the card has room, or give up after timeout_min."""
    deadline = time.time() + timeout_min * 60
    first = True
    while free_gib() < min_free:
        if time.time() > deadline:
            raise SystemExit(
                f"\nGave up after {timeout_min} min: never saw {min_free} GiB free.\n"
                f"The card is held by other tenants. Check `nvidia-smi`, and if it\n"
                f"stays full, ask Kinesis for a less contended node.\n")
        if first:
            print(f"waiting for {min_free} GiB free (currently {free_gib():.1f} GiB) ...")
            first = False
        time.sleep(poll)
    print(f"{free_gib():.1f} GiB free - starting")


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


class TranslationDataset(Dataset):
    def __init__(self, rows, tok, max_len):
        self.rows, self.tok, self.max_len = rows, tok, max_len

    def _enc(self, text, lang):
        ids = self.tok(text, add_special_tokens=False, truncation=True,
                       max_length=self.max_len - 2)["input_ids"]
        return [self.tok.convert_tokens_to_ids(lang)] + ids + [self.tok.eos_token_id]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return {"input_ids": self._enc(r["src"], r["src_lang"]),
                "labels": self._enc(r["tgt"], r["tgt_lang"])}


class Seq2SeqCollator:
    """Pads, and guarantees decoder_input_ids exist (see train_stages.py)."""

    def __init__(self, tok, model):
        self.base = DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100,
                                           pad_to_multiple_of=8)
        self.pad_id = model.config.pad_token_id
        self.start_id = model.config.decoder_start_token_id

    def __call__(self, features):
        batch = self.base(features)
        if "decoder_input_ids" not in batch and "labels" in batch:
            labels = batch["labels"]
            shifted = labels.new_zeros(labels.shape)
            shifted[:, 1:] = labels[:, :-1].clone()
            shifted[:, 0] = self.start_id
            shifted.masked_fill_(shifted == -100, self.pad_id)
            batch["decoder_input_ids"] = shifted
        return batch


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

def pick_optimizer():
    try:
        import bitsandbytes  # noqa: F401  # probe only
        return "adamw_bnb_8bit", 5.7
    except ImportError:
        return "adafactor", 4.7


def make_training_args(**kwargs):
    """Drop arguments this transformers version no longer accepts."""
    import dataclasses
    import inspect
    supported = set(inspect.signature(Seq2SeqTrainingArguments.__init__).parameters)
    try:
        supported |= {f.name for f in dataclasses.fields(Seq2SeqTrainingArguments)}
    except Exception:
        pass
    unknown = sorted(k for k in kwargs if k not in supported)
    if unknown:
        import transformers
        print(f"  note: transformers {transformers.__version__} does not support "
              f"{unknown} - dropped")
    return Seq2SeqTrainingArguments(**{k: v for k, v in kwargs.items() if k in supported})


def plan_batch(args, vocab: int) -> tuple:
    """Choose a batch size from VRAM that is actually free right now."""
    _, floor = pick_optimizer()
    per_example = args.max_len * vocab * 4 * 2 / 2 ** 30      # fp32 logits + grad
    headroom = max(0.0, free_gib() - floor - 1.0)
    batch = max(1, min(args.max_batch, int(headroom / per_example)))
    accum = max(1, round(args.effective_batch / batch))
    return batch, accum, per_example


# Generation parameters must live on `model.generation_config` in transformers
# v5. If any are left on `model.config`, save_pretrained() raises - which happens
# at the FIRST checkpoint, i.e. after you have already burned the GPU time.
GENERATION_ONLY = (
    "max_length", "min_length", "num_beams", "early_stopping", "length_penalty",
    "no_repeat_ngram_size", "encoder_no_repeat_ngram_size", "num_return_sequences",
    "do_sample", "top_k", "top_p", "temperature", "repetition_penalty",
    "diversity_penalty", "num_beam_groups", "bad_words_ids", "forced_bos_token_id",
    "forced_eos_token_id", "output_scores", "return_dict_in_generate",
)
# NB: decoder_start_token_id and pad_token_id stay on config - the forward pass
# uses them to shift labels, they are not generation-only.


def tidy_generation_config(model, max_len):
    """Move generation params off model.config so checkpoints can be saved."""
    moved = []
    for key in GENERATION_ONLY:
        if key in model.config.__dict__:
            setattr(model.generation_config, key, model.config.__dict__[key])
            del model.config.__dict__[key]
            moved.append(key)
    model.generation_config.max_length = max_len
    if moved:
        print(f"  moved {moved} from config to generation_config")
    return model


def train_one(name, data_file, init_from, lr, epochs, out_dir, tok, args):
    print("=" * 70, flush=True)
    print(f"  {name}   init={Path(init_from).name}  lr={lr}  epochs={epochs}")
    print("=" * 70, flush=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(init_from)
    tidy_generation_config(model, args.max_len)
    vocab = model.get_input_embeddings().weight.shape[0]

    batch, accum, per_example = plan_batch(args, vocab)
    optim, floor = pick_optimizer()
    print(f"  free {free_gib():.1f} GiB | optim {optim} (~{floor} GiB) | "
          f"{per_example*1024:.0f} MB/example")
    print(f"  batch {batch} x accum {accum} = effective {batch*accum}", flush=True)

    train_ds = TranslationDataset(load_jsonl(C.DATA / data_file), tok, args.max_len)
    dev_ds = TranslationDataset(load_jsonl(C.DATA / "dev.jsonl"), tok, args.max_len)
    collator = Seq2SeqCollator(tok, model)

    total_steps = math.ceil(len(train_ds) / (batch * accum)) * epochs
    targs = make_training_args(
        output_dir=str(C.ARTIFACTS / "checkpoints" / name),
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        gradient_accumulation_steps=accum,
        learning_rate=lr,
        num_train_epochs=epochs,
        warmup_steps=max(1, int(0.03 * total_steps)),
        label_smoothing_factor=0.0,     # HF's smoother doubles logits memory
        weight_decay=0.01,
        lr_scheduler_type="linear",
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        optim=optim,
        auto_find_batch_size=True,
        logging_steps=50,
        eval_strategy="steps", eval_steps=args.eval_every,
        save_strategy="steps", save_steps=args.eval_every,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        predict_with_generate=False,
        report_to="none", seed=C.SEED,
    )

    trainer = Seq2SeqTrainer(model=model, args=targs, train_dataset=train_ds,
                             eval_dataset=dev_ds, data_collator=collator,
                             callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
    print(f"  {len(train_ds):,} examples, ~{total_steps:,} steps", flush=True)

    result = trainer.train(resume_from_checkpoint=args.resume or None)
    trainer.save_model(str(out_dir))
    tok.save_pretrained(out_dir)
    print(f"  saved -> {out_dir}", flush=True)

    hist_path = C.DATA / f"history_{name}.json"
    C.save_json([{k: v for k, v in h.items() if k in ("step", "loss", "eval_loss")}
                 for h in trainer.state.log_history], hist_path)

    del model, trainer
    torch.cuda.empty_cache()
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", default=["1", "2"],
                    choices=["1", "2", "mixed"], help="which runs to perform")
    ap.add_argument("--min-free", type=float, default=12.0,
                    help="GiB of free VRAM required before starting")
    ap.add_argument("--wait-mins", type=int, default=180,
                    help="how long to wait for that memory before giving up")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--effective-batch", type=int, default=48)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--watch", action="store_true", help="just report free VRAM")
    args = ap.parse_args()

    if args.watch:
        watch()
        return 0

    C.set_seed()
    print(f"project root : {C.ROOT}")
    if not C.EXTENDED_MODEL.exists():
        raise SystemExit(f"{C.EXTENDED_MODEL} missing - run notebook 03 first")
    for f in ("stage1.jsonl", "stage2.jsonl", "dev.jsonl"):
        if not (C.DATA / f).exists():
            raise SystemExit(f"{C.DATA / f} missing - run notebook 02 first")

    wait_for_vram(args.min_free, args.wait_mins)
    tok = AutoTokenizer.from_pretrained(C.EXTENDED_MODEL)
    print(f"tokenizer    : {type(tok).__name__}  vocab {len(tok):,}")

    plan = {
        "1": ("stage1", "stage1.jsonl", str(C.EXTENDED_MODEL), 5e-5, 3, C.STAGE1_MODEL),
        "2": ("stage2", "stage2.jsonl", str(C.STAGE1_MODEL), 1.5e-5, 3, C.STAGE2_MODEL),
        "mixed": ("mixed", "mixed.jsonl", str(C.EXTENDED_MODEL), 5e-5, 3, C.MIXED_MODEL),
    }
    for key in args.stages:
        name, data, init, lr, epochs, out = plan[key]
        if key == "2" and not C.STAGE1_MODEL.exists():
            raise SystemExit("stage 2 continues from stage 1, which has not been trained")
        train_one(name, data, init, lr, epochs, out, tok, args)

    print("\nall requested stages complete. Next: notebooks/04_evaluate.ipynb")
    return 0


if __name__ == "__main__":
    sys.exit(main())
