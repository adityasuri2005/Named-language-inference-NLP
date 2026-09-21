"""
Step 2 — Evaluate the FROZEN Step-1 model on the native Odia annotated
test set (provided by the professor), with no retraining or tuning.

This produces the second of the two scorecard numbers, and lets you
compute the translationese gap:

    translationese_gap = benchmark_test_accuracy - native_odia_accuracy

Usage:
    python evaluate_native.py \
        --model_dir ./muril-odia-nli \
        --native_file /path/to/native_odia_test.csv \
        --premise_col premise --hypothesis_col hypothesis --label_col label \
        --benchmark_results ./muril-odia-nli/benchmark_results.json

Accepts .csv, .tsv, .json, .jsonl, .parquet, or .xlsx for --native_file.
Labels can be integers (0/1/2) or strings; pass --label_map to remap
strings to the model's label scheme if needed, e.g.:
    --label_map '{"entailment":0,"neutral":1,"contradiction":2}'

IMPORTANT: Do not change the model, its weights, or any hyperparameter
between Step 1 and Step 2. This script only loads the saved checkpoint
and runs inference.

This script is unaffected by which IndicXNLI dataset repo Step 1 used for
fine-tuning/benchmark-testing — it only reads the professor-provided
native_file. See train_benchmark.py's header for why that step sources
Odia data from Divyanshu/indicxnli rather than ai4bharat/IndicXNLI-Translated.
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
    n_train, n_val, n_test = len(raw["train"]), len(raw["validation"]), len(raw["test"])
 
    if args.subset_train:
        raw["train"] = raw["train"].select(range(min(args.subset_train, n_train)))
 
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenize_fn = build_tokenize_fn(tokenizer)
    tokenized = raw.map(tokenize_fn, batched=True, remove_columns=["premise", "hypothesis"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
 
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
 
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
        eval_dataset=tokenized["validation"],
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
