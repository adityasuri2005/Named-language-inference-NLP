"""
Group 9 — Named-Language Inference (NLI) on Odia
Task    : Fine-tune ai4bharat/IndicBERTv2-MLM-only on IndicXNLI (Odia, in-language/
          translate-train), reproduce the published benchmark, then evaluate on the
          professor's native Odia test set.

Reference paper : Doddapaneni et al., ACL 2023, "Towards Leaving No Indic Language
                   Behind" (Table 16, IndicBERT/MLM-only row, 'or' column = 72.6%)
                   https://aclanthology.org/2023.acl-long.693/
Dataset origin   : Aggarwal, Gupta & Kunchukuttan, EMNLP 2022, IndicXNLI
                   https://aclanthology.org/2022.emnlp-main.755/

NOTE ON PROTOCOL: the 72.6% figure was produced by fine-tuning on English MultiNLI
and testing zero-shot on Odia. This script instead fine-tunes directly on the Odia
train split (translate-train / in-language), since that's the data provided. Table 13
(Appendix N) shows the same hyperparameters (lr 3e-5, wd 0.01, best epoch 4) apply to
both the Best-EN and Best-IN configurations for this task, so they're used here too.
Flag this protocol difference to your professor — matching 72.6% exactly is not
guaranteed under translate-train, only under zero-shot.

Usage:
    python train_benchmark.py \
        --train_file indicxnli-train.parquet \
        --val_file indicxnli-validation.parquet \
        --test_file indicxnli-test.parquet \
        --output_dir ./indicbertv2-odia-nli
"""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
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

MODEL_NAME = "ai4bharat/IndicBERTv2-MLM-only"
NUM_LABELS = 3  # 0 = entailment, 1 = neutral, 2 = contradiction
MAX_SEQ_LEN = 128
TARGET_ACCURACY = 72.6  # Table 16, IndicBERTv2-MLM-only, Odia ('or') — zero-shot protocol


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def load_local_splits(train_file, val_file, test_file):
    train_df = pd.read_parquet(train_file)
    val_df = pd.read_parquet(val_file)
    test_df = pd.read_parquet(test_file)
    for name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        missing = {"premise", "hypothesis", "label"} - set(df.columns)
        if missing:
            raise ValueError(f"{name} file is missing columns: {missing}")
    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, preserve_index=False),
            "validation": Dataset.from_pandas(val_df, preserve_index=False),
            "test": Dataset.from_pandas(test_df, preserve_index=False),
        }
    )


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--val_file", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--output_dir", default="./indicbertv2-odia-nli")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=3e-5)   # Table 13
    parser.add_argument("--train_batch_size", type=int, default=32)    # Appendix N default
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=0.01)    # Table 13
    parser.add_argument("--warmup_ratio", type=float, default=0.10)    # Appendix N
    parser.add_argument("--num_train_epochs", type=int, default=6)     # paper's best = 4; headroom + early stop
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--eval_steps_fraction", type=float, default=0.5)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--subset_train", type=int, default=None,
                         help="Optional: cap training examples for a quick smoke test.")
    args = parser.parse_args()

    set_all_seeds(args.seed)

    raw = load_local_splits(args.train_file, args.val_file, args.test_file)
    print(raw)
    n_val, n_test = len(raw["validation"]), len(raw["test"])

    if args.subset_train:
        raw["train"] = raw["train"].select(range(min(args.subset_train, len(raw["train"]))))
    n_train = len(raw["train"])  # reflects the subset actually trained on, if any

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenize_fn = build_tokenize_fn(tokenizer)
    tokenized = raw.map(tokenize_fn, batched=True, remove_columns=["premise", "hypothesis"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

    steps_per_epoch = max(1, len(tokenized["train"]) // args.train_batch_size)
    eval_steps = max(1, int(steps_per_epoch * args.eval_steps_fraction))
    # transformers v5.15+ removed TrainingArguments(warmup_ratio=...); compute the
    # equivalent absolute warmup_steps ourselves so the effective warmup span (10%
    # of total optimizer steps, per the paper's Appendix N) stays the same.
    total_steps = steps_per_epoch * args.num_train_epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio)) if args.warmup_ratio > 0 else 0

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
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
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,  # `tokenizer=` was removed in transformers v5.0+
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

    # Save per-example predictions
    os.makedirs(args.output_dir, exist_ok=True)
    test_df_out = pd.read_parquet(args.test_file)
    test_df_out["predicted_label"] = y_pred
    test_df_out.to_csv(os.path.join(args.output_dir, "benchmark_test_predictions.csv"), index=False)

    benchmark_accuracy = test_metrics["test_accuracy"] * 100
    diff = benchmark_accuracy - TARGET_ACCURACY

    print("\n===== SCORECARD FIELDS =====")
    print("Task: Named-language Inference (NLI) — IndicBERTv2-MLM-only fine-tuned on IndicXNLI (Odia, translate-train)")
    print(f"Target Result: {TARGET_ACCURACY:.2f}% accuracy (Table 16, ACL 2023 paper, zero-shot protocol)")
    print(f"Achieved Result: {benchmark_accuracy:.2f}% accuracy (benchmark test set, translate-train protocol)")
    print(f"Sample size — Train/Val/Test: {n_train} / {n_val} / {n_test}")
    print(f"Difference from target: {diff:+.2f} points")
    if abs(diff) <= 2.0:
        print("WITHIN the 1-2 point tolerance.")
    else:
        print("OUTSIDE tolerance — remember this used a different protocol (translate-train, "
              "not the zero-shot MultiNLI->Odia protocol that produced 72.6%).")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "benchmark_results.json"), "w") as f:
        json.dump(
            {
                "task": "Named-language Inference (NLI)",
                "model": MODEL_NAME,
                "language": "or",
                "strategy": "In-language fine-tune (translate-train)",
                "seed": args.seed,
                "hyperparameters": {
                    "learning_rate": args.learning_rate,
                    "train_batch_size": args.train_batch_size,
                    "weight_decay": args.weight_decay,
                    "warmup_ratio": args.warmup_ratio,
                    "max_seq_len": MAX_SEQ_LEN,
                },
                "target_accuracy_pct": TARGET_ACCURACY,
                "benchmark_test_accuracy_pct": benchmark_accuracy,
                "difference_pts": diff,
                "n_train": n_train,
                "n_validation": n_val,
                "n_test": n_test,
                "classification_report": report,
            },
            f,
            indent=2,
        )
    print(f"\nModel + tokenizer + benchmark_results.json + predictions saved to: {args.output_dir}")


if __name__ == "__main__":
    main()