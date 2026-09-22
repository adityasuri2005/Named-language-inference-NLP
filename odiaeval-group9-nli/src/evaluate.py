"""Evaluate the saved fine-tuned model.

One code path serves both evaluations, which is the point: STEP 6 of the brief
requires the same saved model, the same preprocessing, the same label mapping
and the same metric on the native Odia set as on the benchmark set. Any
difference between the two scores is then attributable to the data alone.

    # STEP 5 - benchmark reproduction (held-out IndicXNLI Odia test, 5,010 pairs)
    python -m src.evaluate --split benchmark_test

    # STEP 6 - native Odia gold test set from the professor
    python -m src.evaluate --split native --native-file data/native/<file>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from .data import (
    LABEL_NAMES,
    as_hf_dataset,
    load_native_test,
    load_split,
    make_tokenize_fn,
    verify_odia,
)
from .utils import capture_environment, load_config, resolve, set_all_seeds, write_json


def score(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Accuracy is the headline metric. The rest is for the error analysis."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)) * 100, 2),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")) * 100, 2),
        "per_class": classification_report(
            y_true, y_pred, target_names=LABEL_NAMES,
            output_dict=True, zero_division=0,
        ),
        "confusion_matrix": {
            "labels": LABEL_NAMES,
            "rows_are_gold": True,
            "matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        },
        "n_examples": int(len(y_true)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned Odia NLI model.")
    parser.add_argument("--config", default="configs/indicbertv2_odia.yaml")
    parser.add_argument("--split", choices=["benchmark_test", "validation", "native"],
                        default="benchmark_test")
    parser.add_argument("--native-file", default=None,
                        help="Path to the professor's native Odia test set.")
    parser.add_argument("--model-dir", default=None,
                        help="Overrides output.model_dir from the config.")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    if args.split == "native" and not args.native_file:
        parser.error("--split native requires --native-file")

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

    cfg = load_config(args.config)
    set_all_seeds(cfg["experiment"]["seed"])

    model_dir = resolve(args.model_dir or cfg["output"]["model_dir"])
    results_dir = resolve(cfg["output"]["results_dir"])
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(
            f"No fine-tuned model at {model_dir}. Run `python -m src.train` first."
        )

    # ------------------------------------------------------------ load data
    if args.split == "native":
        df = load_native_test(args.native_file)
        source = str(args.native_file)
    elif args.split == "validation":
        df = load_split(cfg["data"]["validation_file"])
        source = cfg["data"]["validation_file"]
    else:
        df = load_split(cfg["data"]["test_file"])
        source = cfg["data"]["test_file"]
        expected = cfg["data"]["expected_sizes"]["test"]
        if len(df) != expected:
            raise ValueError(f"Test split has {len(df)} rows, expected {expected}.")

    print(f"Evaluating '{args.split}': {len(df):,} examples from {source}")
    script_check = verify_odia(df, args.split)
    print(f"  script check: mean Odia ratio {script_check['mean_odia_ratio']:.4f} "
          f"-> {script_check['verdict']}")

    # ----------------------------------------------------- load saved model
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    # Same tokenisation function, same max length, as training.
    tokenize = make_tokenize_fn(tokenizer, cfg["model"]["max_seq_length"])
    ds = as_hf_dataset(df).map(
        tokenize, batched=True, remove_columns=["premise", "hypothesis"],
        desc="tokenizing")

    # No .with_format("torch") here: sequences are ragged until the collator pads
    # them, and DataCollatorWithPadding accepts plain python lists.
    collator = DataCollatorWithPadding(tokenizer)
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collator, shuffle=False)

    # ------------------------------------------------------------- predict
    all_logits = []
    with torch.no_grad():
        for batch in loader:
            batch.pop("labels", None)
            batch = {k: v.to(device) for k, v in batch.items()}
            all_logits.append(model(**batch).logits.float().cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    preds = logits.argmax(axis=-1)
    gold = df["label"].to_numpy()

    metrics = score(gold, preds)
    print(f"\n  accuracy : {metrics['accuracy']:.2f}%")
    print(f"  macro F1 : {metrics['macro_f1']:.2f}%")

    if args.split == "benchmark_test":
        target = cfg["targets"]["published_accuracy"]
        tol = cfg["targets"]["tolerance_points"]
        delta = metrics["accuracy"] - target
        inside = abs(delta) <= tol
        metrics["published_target"] = target
        metrics["difference_from_target"] = round(delta, 2)
        metrics["within_tolerance"] = bool(inside)
        print(f"  published target: {target}%  |  difference: {delta:+.2f} pts  "
              f"-> {'WITHIN' if inside else 'OUTSIDE'} the {tol}-point tolerance")

    # ------------------------------------------------------------ persist
    results_dir.mkdir(parents=True, exist_ok=True)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)

    predictions = pd.DataFrame({
        "idx": range(len(df)),
        "premise": df["premise"],
        "hypothesis": df["hypothesis"],
        "gold_label_id": gold,
        "gold_label": [LABEL_NAMES[i] for i in gold],
        "pred_label_id": preds,
        "pred_label": [LABEL_NAMES[i] for i in preds],
        "correct": gold == preds,
        **{f"prob_{name}": probs[:, i] for i, name in enumerate(LABEL_NAMES)},
    })
    pred_path = results_dir / f"predictions_{args.split}.csv"
    predictions.to_csv(pred_path, index=False, encoding="utf-8")
    print(f"[write] {pred_path}")

    write_json(results_dir / f"metrics_{args.split}.json", {
        "split": args.split,
        "source_file": source,
        "model_dir": str(model_dir),
        "metrics": metrics,
        "script_verification": script_check,
        "label_distribution_gold": {
            LABEL_NAMES[i]: int((gold == i).sum()) for i in range(3)
        },
        "label_distribution_pred": {
            LABEL_NAMES[i]: int((preds == i).sum()) for i in range(3)
        },
        "environment": capture_environment(),
        "preprocessing": {
            "max_seq_length": cfg["model"]["max_seq_length"],
            "normalisation": "NFC + whitespace collapse",
            "encoding": "premise as segment A, hypothesis as segment B",
            "note": "Identical to the training and benchmark-test path.",
        },
    })


if __name__ == "__main__":
    main()
