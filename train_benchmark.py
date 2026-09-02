"""
Step 1 — Reproduce the published benchmark.

Fine-tunes IndicBERTv2-MLM-only (ai4bharat/IndicBERTv2-MLM-only) on the
Odia ("or") split of IndicXNLI, matching the fine-tuning protocol from:

  Doddapaneni, Aralikatte, Ramesh, Goyal, Khapra, Kunchukuttan & Kumar
  (2023). "Towards Leaving No Indic Language Behind: Building
  Monolingual Corpora, Benchmark and Models for Indic Languages." ACL 2023.
  https://aclanthology.org/2023.acl-long.693/
  (Table 16: per-language IndicXNLI results; "IndicBERT" row = MLM-only.
  Official fine-tuning code: https://github.com/AI4Bharat/IndicBERT
  under fine-tuning/xnli/)

The dataset (IndicXNLI) itself was introduced by:
  Aggarwal, Gupta & Kunchukuttan (2022), EMNLP 2022.
  https://aclanthology.org/2022.emnlp-main.755/

Target to reproduce (Table 16, IndicBERTv2-MLM-only, Odia column): 72.6% accuracy.
Tolerance: within 1-2 percentage points (i.e. ~70.6-74.6%).

Usage:
    python train_benchmark.py --output_dir ./indicbertv2-odia-nli --seed 42

Run this on a GPU (Colab T4/A100, or any CUDA machine). It will NOT run
usefully on CPU (392K training examples).
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

LANG = "or"  # Odia
MODEL_NAME = "ai4bharat/IndicBERTv2-MLM-only"
# Group's dataset link (ai4bharat/IndicXNLI-Translated) appears to be a small
# eval-only mirror (no per-language train split for fine-tuning). We try it
# first and fall back to the full IndicXNLI release (train/validation/test,
# identical content, same paper) which definitely has all three splits.
DATASET_CANDIDATES = ["ai4bharat/IndicXNLI-Translated", "Divyanshu/indicxnli"]
NUM_LABELS = 3  # 0 = entailment, 1 = neutral, 2 = contradiction (see dataset card)
MAX_SEQ_LEN = 128
TARGET_ACCURACY = 72.6  # Table 16, IndicBERTv2-MLM-only, Odia ('or')


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def build_tokenize_fn(tokenizer):
    def tokenize_fn(batch):
        return tokenizer(
            batch["premise"],
            batch["hypothesis"],
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding=False,
        )
    return tokenize_fn


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


def load_indicxnli(lang):
    """Try each candidate dataset repo in order; use the first that has a
    usable train split for the given language config."""
    last_err = None
    for repo in DATASET_CANDIDATES:
        try:
            print(f"Trying to load {repo!r} (config={lang!r})...")
            raw = load_dataset(repo, lang)
            print(raw)
            if "train" in raw and len(raw["train"]) > 1000:
                print(f"Using {repo!r} — has a train split with {len(raw['train'])} examples.")
                return raw, repo
            else:
                print(f"{repo!r} has no usable train split for fine-tuning "
                      f"(likely an eval-only mirror) — trying next candidate.")
        except Exception as e:  # noqa: BLE001
            print(f"Could not load {repo!r} with config {lang!r}: {e}")
            last_err = e
    raise RuntimeError(
        f"None of {DATASET_CANDIDATES} yielded a usable train split for '{lang}'. "
        f"Last error: {last_err}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./indicbertv2-odia-nli")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=3e-5)  # Table 13, IndicBERT Best-IN, XNLI
    parser.add_argument("--train_batch_size", type=int, default=32)   # Appendix N default
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=0.01)   # Table 13, IndicBERT Best-IN, XNLI
    parser.add_argument("--warmup_ratio", type=float, default=0.10)   # Appendix N: "initial warmup of 10%"
    parser.add_argument("--num_train_epochs", type=int, default=6)    # paper's best epoch was 4; allow headroom
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--eval_steps_fraction", type=float, default=0.5,
                         help="Evaluate every 0.5 epochs.")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--subset_train", type=int, default=None,
                         help="Optional: cap training examples for a quick smoke test.")
    args = parser.parse_args()

    set_all_seeds(args.seed)

    raw, used_repo = load_indicxnli(LANG)
    # Expect (Divyanshu/indicxnli): train ~392702, validation ~2490, test ~5010

    if args.subset_train:
        raw["train"] = raw["train"].select(range(min(args.subset_train, len(raw["train"]))))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenize_fn = build_tokenize_fn(tokenizer)

    tokenized = raw.map(tokenize_fn, batched=True, remove_columns=["premise", "hypothesis"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )

    steps_per_epoch = max(1, len(tokenized["train"]) // args.train_batch_size)
    eval_steps = max(1, int(steps_per_epoch * args.eval_steps_fraction))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=50,
        fp16=args.fp16 and torch.cuda.is_available(),
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],  # model selection on validation only
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    print("Starting fine-tuning...")
    trainer.train()

    print("Evaluating on the held-out benchmark TEST set (once)...")
    test_metrics = trainer.evaluate(eval_dataset=tokenized["test"], metric_key_prefix="test")
    print(json.dumps(test_metrics, indent=2))

    preds = trainer.predict(tokenized["test"])
    y_pred = np.argmax(preds.predictions, axis=-1)
    y_true = preds.label_ids
    report = classification_report(y_true, y_pred, digits=4)
    print(report)

    benchmark_accuracy = test_metrics["test_accuracy"] * 100
    target = TARGET_ACCURACY
    diff = benchmark_accuracy - target
    print(f"\nBenchmark (Odia, IndicXNLI test set) accuracy: {benchmark_accuracy:.2f}%")
    print(f"Published target (IndicBERTv2-MLM-only, Odia):   {target:.2f}%")
    print(f"Difference:                                       {diff:+.2f} points")
    if abs(diff) <= 2.0:
        print("WITHIN the 1-2 point tolerance. Proceed to Step 2 (native Odia eval).")
    else:
        print("OUTSIDE tolerance. Stop and debug before touching native data "
              "(check seed, data split sizes, label mapping, hyperparameters).")

    # Save final model + tokenizer + a results file
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "benchmark_results.json"), "w") as f:
        json.dump(
            {
                "model": MODEL_NAME,
                "dataset_repo": used_repo,
                "language": LANG,
                "strategy": "In-language fine-tune (translate-train)",
                "seed": args.seed,
                "benchmark_test_accuracy_pct": benchmark_accuracy,
                "published_target_pct": target,
                "difference_pts": diff,
                "classification_report": report,
                "n_train": len(tokenized["train"]),
                "n_validation": len(tokenized["validation"]),
                "n_test": len(tokenized["test"]),
            },
            f,
            indent=2,
        )
    print(f"\nModel + tokenizer + benchmark_results.json saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
