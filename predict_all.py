"""
Post-hoc evaluation of one saved checkpoint on BOTH test sets, writing every
file the submission needs. No training and no weight updates.

Outputs (in --out_dir):
  predictions_benchmark.csv        premise, hypothesis, label, predicted_label
  predictions_native.csv           same for the native Odia file
  benchmark_results.json           benchmark accuracy, report, confusion matrix
  native_results.json              native accuracy, report, confusion matrix, gap
  experiment_config.json           training arguments read from training_args.bin

Usage:
    python predict_all.py \
        --model_dir indicbertv2-odia-nli-failed/checkpoint-18405 \
        --test_file indicxnli-test.parquet \
        --native_file 03_nli_groups_3_9_updated.jsonl \
        --label_map '{"entailment":0,"neutral":1,"contradiction":2}' \
        --out_dir submission_outputs
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_SEQ_LEN = 128
TARGET_ACCURACY = 72.6
BASE_MODEL = "ai4bharat/IndicBERTv2-MLM-only"


def predict(model, tok, premises, hypotheses, device, batch_size):
    preds = []
    with torch.no_grad():
        for i in range(0, len(premises), batch_size):
            enc = tok(
                premises[i:i + batch_size],
                hypotheses[i:i + batch_size],
                truncation=True, max_length=MAX_SEQ_LEN, padding=True,
                return_tensors="pt",
            ).to(device)
            preds += model(**enc).logits.argmax(-1).cpu().tolist()
    return preds


def metrics(y_true, y_pred):
    return {
        "accuracy_pct": float(accuracy_score(y_true, y_pred) * 100),
        "classification_report": classification_report(y_true, y_pred, digits=4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--test_file", required=True, help="Benchmark test parquet (premise, hypothesis, label)")
    p.add_argument("--native_file", required=True, help="Native Odia .jsonl (premise, hypothesis, label)")
    p.add_argument("--label_map", default='{"entailment":0,"neutral":1,"contradiction":2}')
    p.add_argument("--out_dir", default="submission_outputs")
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    try:
        tok = AutoTokenizer.from_pretrained(args.model_dir)
    except Exception:  # checkpoint without tokenizer files
        tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device).eval()

    # ---- benchmark test set (all rows) ----
    bench = pd.read_parquet(args.test_file)
    bench_pred = predict(model, tok, bench["premise"].tolist(), bench["hypothesis"].tolist(),
                         device, args.batch_size)
    out_b = bench[["premise", "hypothesis", "label"]].copy()
    out_b["predicted_label"] = bench_pred
    out_b.to_csv(os.path.join(args.out_dir, "predictions_benchmark.csv"), index=False)
    bm = metrics(bench["label"].to_numpy(), bench_pred)
    bench_result = {
        "note": "Created post hoc by evaluating a saved checkpoint on the full benchmark test set.",
        "model": BASE_MODEL,
        "checkpoint": args.model_dir,
        "test_file": args.test_file,
        "n_test": len(bench),
        "target_accuracy_pct": TARGET_ACCURACY,
        "benchmark_test_accuracy_pct": bm["accuracy_pct"],
        "difference_pts": bm["accuracy_pct"] - TARGET_ACCURACY,
        "classification_report": bm["classification_report"],
        "confusion_matrix": bm["confusion_matrix"],
    }
    with open(os.path.join(args.out_dir, "benchmark_results.json"), "w") as f:
        json.dump(bench_result, f, indent=2)
    print(f"Benchmark accuracy ({len(bench)} examples): {bm['accuracy_pct']:.2f}%")

    # ---- native Odia test set (all rows) ----
    nat = pd.read_json(args.native_file, lines=True)
    nat["gold"] = nat["label"].map(json.loads(args.label_map))
    if nat["gold"].isna().any():
        raise ValueError("Some native labels did not map; check --label_map")
    nat_pred = predict(model, tok, nat["premise"].tolist(), nat["hypothesis"].tolist(),
                       device, args.batch_size)
    nat["predicted_label"] = nat_pred
    nat.to_csv(os.path.join(args.out_dir, "predictions_native.csv"), index=False)
    nm = metrics(nat["gold"].to_numpy().astype(int), nat_pred)
    native_result = {
        "native_file": args.native_file,
        "n_examples": len(nat),
        "native_odia_accuracy_pct": nm["accuracy_pct"],
        "benchmark_test_accuracy_pct": bm["accuracy_pct"],
        "translationese_gap_pts": bm["accuracy_pct"] - nm["accuracy_pct"],
        "classification_report": nm["classification_report"],
        "confusion_matrix": nm["confusion_matrix"],
    }
    with open(os.path.join(args.out_dir, "native_results.json"), "w") as f:
        json.dump(native_result, f, indent=2)
    print(f"Native accuracy ({len(nat)} examples): {nm['accuracy_pct']:.2f}%")
    print(f"Translationese gap (benchmark - native): {native_result['translationese_gap_pts']:+.2f} points")

    # ---- experiment configuration ----
    cfg_path = os.path.join(args.model_dir, "training_args.bin")
    if os.path.exists(cfg_path):
        try:
            a = torch.load(cfg_path, weights_only=False)
            with open(os.path.join(args.out_dir, "experiment_config.json"), "w") as f:
                json.dump(a.to_dict(), f, indent=2, default=str)
            print("Saved experiment_config.json")
        except Exception as e:  # noqa: BLE001
            print(f"Could not read {cfg_path}: {e}")

    print(f"\nAll outputs written to: {args.out_dir}")


if __name__ == "__main__":
    main()
