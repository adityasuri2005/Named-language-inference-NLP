"""Fine-tune IndicBERTv2-MLM-only on the Odia split of IndicXNLI.

Protocol (STEP 3-4 of the project pipeline):
  * Train on the benchmark TRAIN split only (392,702 pairs).
  * Select the checkpoint on VALIDATION accuracy only (2,490 pairs).
  * The TEST split is never loaded here. It is touched once, by evaluate.py.

Hyperparameters come from Doddapaneni et al. (ACL 2023), Appendix N, Table 13,
row "IndicXNLI / IndicBERT", column "Best IN" (in-language validation set).

    python -m src.train --config configs/indicbertv2_odia.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .data import LABEL_NAMES, as_hf_dataset, build_dataset_stats, load_split, make_tokenize_fn
from .utils import (
    capture_environment,
    load_config,
    resolve,
    set_all_seeds,
    write_json,
)


def compute_metrics(eval_pred):
    """Accuracy, as used by XNLI and by both reference papers."""
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune IndicBERTv2 on IndicXNLI Odia.")
    parser.add_argument("--config", default="configs/indicbertv2_odia.yaml")
    parser.add_argument("--output-dir", default=None,
                        help="Overrides output.model_dir from the config.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the last checkpoint in the output dir.")
    parser.add_argument("--skip-checks", action="store_true",
                        help="Skip the data integrity gate. Not recommended.")
    args = parser.parse_args()

    import torch
    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    cfg = load_config(args.config)
    seed = cfg["experiment"]["seed"]
    set_all_seeds(seed)

    model_dir = resolve(args.output_dir or cfg["output"]["model_dir"])
    results_dir = resolve(cfg["output"]["results_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- checks
    if not args.skip_checks:
        print("=" * 70)
        print("STEP 3 - data integrity gate")
        print("=" * 70)
        stats = build_dataset_stats(cfg)
        write_json(results_dir / "dataset_stats.json", stats)
        for entry in stats["script_verification"]:
            print(f"  script  {entry['split']:<11} mean Odia ratio "
                  f"{entry['mean_odia_ratio']:.4f}  -> {entry['verdict']}")
        print(f"  leakage {stats['leakage']['verdict']}")
        print(f"  sizes   {stats['split_sizes']}")

    # ------------------------------------------------------------ load data
    train_df = load_split(cfg["data"]["train_file"])
    val_df = load_split(cfg["data"]["validation_file"])
    print(f"\nTrain {len(train_df):,} | Validation {len(val_df):,} "
          f"| Test withheld until evaluate.py")

    # ---------------------------------------------------- model + tokenizer
    model_name = cfg["model"]["name"]
    revision = cfg["model"].get("revision", "main")
    print(f"\nLoading {model_name} (revision={revision})")

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model_config = AutoConfig.from_pretrained(
        model_name,
        revision=revision,
        num_labels=cfg["model"]["num_labels"],
        id2label={i: name for i, name in enumerate(LABEL_NAMES)},
        label2id={name: i for i, name in enumerate(LABEL_NAMES)},
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, revision=revision, config=model_config
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params/1e6:.1f}M")

    tokenize = make_tokenize_fn(tokenizer, cfg["model"]["max_seq_length"])
    train_ds = as_hf_dataset(train_df).map(
        tokenize, batched=True, remove_columns=["premise", "hypothesis"],
        desc="tokenizing train")
    val_ds = as_hf_dataset(val_df).map(
        tokenize, batched=True, remove_columns=["premise", "hypothesis"],
        desc="tokenizing validation")

    # ------------------------------------------------------------- training
    t = cfg["training"]
    use_fp16 = bool(t["fp16"]) and torch.cuda.is_available()
    if t["fp16"] and not use_fp16:
        print("  note: fp16 requested but no CUDA device found; running fp32.")

    training_args = TrainingArguments(
        output_dir=str(model_dir / "checkpoints"),
        seed=seed,
        data_seed=seed,
        learning_rate=float(t["learning_rate"]),
        weight_decay=float(t["weight_decay"]),
        num_train_epochs=float(t["num_train_epochs"]),
        per_device_train_batch_size=int(t["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(t["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(t["gradient_accumulation_steps"]),
        warmup_ratio=float(t["warmup_ratio"]),
        optim=t["optimizer"],
        lr_scheduler_type=t["lr_scheduler_type"],
        max_grad_norm=float(t["max_grad_norm"]),
        fp16=use_fp16,
        eval_strategy=t["eval_strategy"],
        save_strategy=t["save_strategy"],
        metric_for_best_model=t["metric_for_best_model"],
        greater_is_better=bool(t["greater_is_better"]),
        load_best_model_at_end=bool(t["load_best_model_at_end"]),
        save_total_limit=int(t["save_total_limit"]),
        logging_steps=int(t["logging_steps"]),
        report_to=[],
        dataloader_num_workers=2,
    )

    # `tokenizer=` was renamed to `processing_class=` in transformers 4.46.
    # Pick whichever this install accepts so the script runs on either.
    import inspect

    tok_kwarg = (
        "processing_class"
        if "processing_class" in inspect.signature(Trainer.__init__).parameters
        else "tokenizer"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=int(t["early_stopping_patience"]))],
        **{tok_kwarg: tokenizer},
    )

    print("\n" + "=" * 70)
    print("STEP 4 - fine-tuning")
    print("=" * 70)
    started = time.time()
    train_result = trainer.train(resume_from_checkpoint=args.resume or None)
    minutes = (time.time() - started) / 60

    # Final validation pass with the best checkpoint restored.
    val_metrics = trainer.evaluate(eval_dataset=val_ds)
    print(f"\nBest-checkpoint validation accuracy: {val_metrics['eval_accuracy']*100:.2f}%")

    # ---------------------------------------------------------------- save
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    print(f"\nModel + tokenizer + config saved to {model_dir}")

    write_json(results_dir / "train_results.json", {
        "config": cfg,
        "environment": capture_environment(),
        "model": {
            "name": model_name,
            "revision": revision,
            "parameters": int(n_params),
            "resolved_commit": getattr(model_config, "_commit_hash", None),
        },
        "train_runtime_minutes": round(minutes, 2),
        "train_metrics": {k: float(v) for k, v in train_result.metrics.items()
                          if isinstance(v, (int, float))},
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_accuracy": float(val_metrics["eval_accuracy"]),
        "validation_metrics": {k: float(v) for k, v in val_metrics.items()
                               if isinstance(v, (int, float))},
        "log_history": trainer.state.log_history,
        "note": "Selection used validation only. Test labels were not loaded here.",
    })

    # Per-epoch view, so the grader can see the trajectory at a glance.
    stage_stats = [
        {
            "epoch": entry.get("epoch"),
            "step": entry.get("step"),
            "eval_accuracy": entry.get("eval_accuracy"),
            "eval_macro_f1": entry.get("eval_macro_f1"),
            "eval_loss": entry.get("eval_loss"),
        }
        for entry in trainer.state.log_history if "eval_accuracy" in entry
    ]
    write_json(results_dir / "stage_stats.json", {
        "per_epoch_validation": stage_stats,
        "selected_epoch": max(stage_stats, key=lambda e: e["eval_accuracy"])["epoch"]
        if stage_stats else None,
        "train_runtime_minutes": round(minutes, 2),
    })

    print(f"\nDone in {minutes:.1f} min. Next: python -m src.evaluate")


if __name__ == "__main__":
    main()
