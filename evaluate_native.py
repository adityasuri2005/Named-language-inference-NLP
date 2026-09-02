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

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_SEQ_LEN = 128
DEFAULT_LABEL_NAMES = {0: "entailment", 1: "neutral", 2: "contradiction"}


def load_native_file(path, premise_col, hypothesis_col, label_col):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv",):
        df = pd.read_csv(path)
    elif ext in (".tsv",):
        df = pd.read_csv(path, sep="\t")
    elif ext in (".json",):
        df = pd.read_json(path)
    elif ext in (".jsonl",):
        df = pd.read_json(path, lines=True)
    elif ext in (".parquet",):
        df = pd.read_parquet(path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    missing = [c for c in (premise_col, hypothesis_col, label_col) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Column(s) {missing} not found in {path}. "
            f"Available columns: {list(df.columns)}. "
            f"Pass --premise_col / --hypothesis_col / --label_col to match your file."
        )
    return df[[premise_col, hypothesis_col, label_col]].rename(
        columns={premise_col: "premise", hypothesis_col: "hypothesis", label_col: "label"}
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Path to the Step-1 saved model")
    parser.add_argument("--native_file", required=True, help="Professor-provided native Odia test file")
    parser.add_argument("--premise_col", default="premise")
    parser.add_argument("--hypothesis_col", default="hypothesis")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--label_map", default=None,
                         help='JSON string mapping string labels to ints, e.g. '
                              '\'{"entailment":0,"neutral":1,"contradiction":2}\'')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--benchmark_results", default=None,
                         help="Path to Step 1's benchmark_results.json, to auto-compute the gap")
    parser.add_argument("--output_file", default=None,
                         help="Where to write native_results.json (defaults next to model_dir)")
    args = parser.parse_args()

    df = load_native_file(args.native_file, args.premise_col, args.hypothesis_col, args.label_col)
    print(f"Loaded {len(df)} native Odia examples from {args.native_file}")

    if args.label_map:
        mapping = json.loads(args.label_map)
        df["label"] = df["label"].map(mapping)
        if df["label"].isna().any():
            bad = df[df["label"].isna()]
            raise ValueError(f"Some labels didn't map cleanly, e.g.:\n{bad.head()}")
    df["label"] = df["label"].astype(int)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading frozen model from {args.model_dir} onto {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    all_preds = []
    premises = df["premise"].tolist()
    hypotheses = df["hypothesis"].tolist()

    with torch.no_grad():
        for i in range(0, len(df), args.batch_size):
            batch_p = premises[i:i + args.batch_size]
            batch_h = hypotheses[i:i + args.batch_size]
            enc = tokenizer(
                batch_p, batch_h,
                truncation=True, max_length=MAX_SEQ_LEN, padding=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())

    y_true = df["label"].to_numpy()
    y_pred = np.array(all_preds)

    native_accuracy = accuracy_score(y_true, y_pred) * 100
    report = classification_report(y_true, y_pred, digits=4)
    cm = confusion_matrix(y_true, y_pred).tolist()

    print(f"\nNative Odia test accuracy: {native_accuracy:.2f}%")
    print(report)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    result = {
        "native_file": args.native_file,
        "n_examples": len(df),
        "native_odia_accuracy_pct": native_accuracy,
        "classification_report": report,
        "confusion_matrix": cm,
    }

    if args.benchmark_results and os.path.exists(args.benchmark_results):
        with open(args.benchmark_results) as f:
            bench = json.load(f)
        bench_acc = bench["benchmark_test_accuracy_pct"]
        gap = bench_acc - native_accuracy
        result["benchmark_test_accuracy_pct"] = bench_acc
        result["translationese_gap_pts"] = gap
        print(f"\n--- SCORECARD ---")
        print(f"Benchmark (IndicXNLI Odia test) accuracy: {bench_acc:.2f}%")
        print(f"Native Odia accuracy:                     {native_accuracy:.2f}%")
        print(f"Translationese gap (benchmark - native):  {gap:+.2f} points")
    else:
        print("\n(Pass --benchmark_results path to also compute the translationese gap.)")

    output_file = args.output_file or os.path.join(
        os.path.dirname(args.model_dir.rstrip("/")) or ".", "native_results.json"
    )
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved results to: {output_file}")


if __name__ == "__main__":
    main()
