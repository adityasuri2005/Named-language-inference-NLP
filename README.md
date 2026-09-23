# OdiaEval — Group 9 (Named-Language Inference)

| | |
|---|---|
| **Dataset** | IndicXNLI (Odia, `or`) — `Divyanshu/indicxnli` |
| **Model** | `ai4bharat/IndicBERTv2-MLM-only` |
| **Target** | **72.6% accuracy** |
| **Achieved (benchmark test)** | **77.29% accuracy** (+4.69 pts vs. target — see Protocol Note below) |
| **Reference paper** | Aggarwal, Gupta & Kunchukuttan (2022), IndicXNLI, EMNLP 2022 — https://aclanthology.org/2022.emnlp-main.755/ |

The 72.6% target itself is not actually reported *in* that EMNLP 2022
paper (which only benchmarks XLM-R / IndicBERT-v1 / mBERT / MuRIL). It
comes from **Table 16** of the follow-up paper that introduced
`IndicBERTv2-MLM-only` and evaluated it on IndicXNLI:

> Doddapaneni, Aralikatte, Ramesh, Goyal, Khapra, Kunchukuttan & Kumar
> (2023), *Towards Leaving No Indic Language Behind*, ACL 2023.
> https://aclanthology.org/2023.acl-long.693/

Table 16 there reports, per language, IndicXNLI accuracy for several
models; the "IndicBERT" row (MLM-only, no extra parallel-data objective)
gives **or (Odia) = 72.6**, exactly matching our assignment's target.

Official fine-tuning code for this exact setup exists at
https://github.com/AI4Bharat/IndicBERT under `fine-tuning/xnli/` — worth
skimming if you want to sanity-check hyperparameters against ours.

## ⚠️ Protocol note — read before interpreting the 77.29% result

The paper's 72.6% figure was produced by a **zero-shot cross-lingual**
protocol: fine-tune on **English MultiNLI**, then evaluate zero-shot on
each language's IndicXNLI test set — the model never sees Odia training
data in that setup.

We instead fine-tuned directly on the **Odia train split of IndicXNLI**
itself (392,702 translate-train examples) — a different, and arguably
easier, protocol ("in-language" / "translate-train" fine-tuning), since
that's the data our group was actually given. The paper doesn't report
a per-language translate-train number for IndicXNLI, so there's no
exact number from the paper to match under this protocol — 72.6% is a
reasonable reference point, not a guaranteed target.

Our result (77.29%) landing *above* 72.6% by more than the 1–2 point
tolerance isn't a red flag — training on ~400K in-language examples
commonly outperforms zero-shot transfer from English. This should be
noted explicitly in the report rather than treated as a failed
reproduction: we reproduced the *task* well, under a different (and
typically stronger) training condition than the one that produced 72.6%.

## Data note

Our group's originally-suggested dataset link
(`ai4bharat/IndicXNLI-Translated`) turned out to be a small **eval-only**
mirror (validation + test splits only, no train split) — not usable for
fine-tuning. We used **`Divyanshu/indicxnli`** instead, the original
IndicXNLI release (same paper, same content), which has full
train/validation/test splits: **392,702 / 2,490 / 5,010** for Odia.
Confirmed directly by loading the group-provided parquet files
(`indicxnli-train.parquet`, `indicxnli-validation.parquet`,
`indicxnli-test.parquet`).

## Setup

Run this on a GPU — Colab's free T4 is enough. **Tested and confirmed working
on Google Colab with `transformers==5.17.0`.**

```bash
pip install -U transformers accelerate datasets scikit-learn pandas pyarrow -q
```

> **Note on transformers v5:** if you're on transformers v5.15+, two
> `TrainingArguments`/`Trainer` kwargs used in older tutorials were
> removed and are already handled in `train_benchmark.py`:
> - `TrainingArguments(warmup_ratio=...)` no longer exists — the script
>   now computes an equivalent `warmup_steps` value itself (10% of total
>   optimizer steps, matching the paper's Appendix N).
> - `Trainer(tokenizer=...)` was removed in favor of
>   `Trainer(processing_class=...)` — already updated.
>
> If you hit a `TypeError: unexpected keyword argument` on some other
> kwarg, it likely means transformers moved again since this was
> written — paste the traceback and patch the corresponding line.

## Step 1 — Reproduce the benchmark ✅ Done

```bash
python train_benchmark.py \
    --train_file indicxnli-train.parquet \
    --val_file indicxnli-validation.parquet \
    --test_file indicxnli-test.parquet \
    --output_dir ./indicbertv2-odia-nli
```

This:
1. Loads IndicXNLI Odia (`or`) train/validation/test from the local parquet files.
2. Fine-tunes `IndicBERTv2-MLM-only` with the paper's Table 13 "Best-IN"
   hyperparameters: LR 3e-5, weight decay 0.01, batch size 32, 10% warmup,
   max sequence length 128, AdamW. Up to 6 epochs with early stopping
   (patience 3), checkpoint chosen by validation accuracy.
3. Evaluates **once** on the held-out IndicXNLI Odia test set (5,010 examples).
4. Prints the accuracy, compares it to the 72.6% target, and reports the
   difference (see Protocol Note above for why this differs from strict
   1-2 point tolerance).
5. Saves the model, tokenizer, `benchmark_results.json`, and
   `benchmark_test_predictions.csv`.

**Result: 77.29% accuracy** on the benchmark test set (392,702 / 2,490 /
5,010 train/val/test). Full run took a few hours on a free Colab T4.

### Housekeeping after training
`optimizer.pt` and `scheduler.pt` in the output directory are only needed
to *resume* training — safe to delete once you have your final numbers,
and they're large (optimizer state alone is ~2x model size). Same for any
intermediate `checkpoint-XXXX/` subfolders — the best checkpoint is
already copied to the top-level output directory.

## Step 2 — Evaluate on native Odia (frozen model) ⏳ Waiting on file from professor

```bash
python evaluate_native.py \
    --model_dir ./indicbertv2-odia-nli \
    --native_file /path/to/professors_native_odia_test.csv \
    --premise_col premise --hypothesis_col hypothesis --label_col label \
    --benchmark_results ./indicbertv2-odia-nli/benchmark_results.json
```

- Adjust `--premise_col` / `--hypothesis_col` / `--label_col` to match our
  professor's file once we have it. If labels are strings, pass e.g.
  `--label_map '{"entailment":0,"neutral":1,"contradiction":2}'`.
- Model, weights, and hyperparameters are unchanged from Step 1 — this
  script only loads the saved checkpoint and runs inference.
- Prints native accuracy, a classification report, a confusion matrix, and
  (given `--benchmark_results`) the translationese gap, and saves
  `native_results.json`.
- Expected finding: since the model was trained on 392K *translated*
  (translationese) Odia examples, some accuracy drop on natural,
  originally-written Odia would not be surprising — that's the effect
  this whole exercise is designed to surface.

## Reporting

Fill in `REPORT_TEMPLATE.md`:
1. Benchmark (IndicXNLI Odia test) accuracy — **77.29%** ✅
2. Native Odia accuracy — pending
3. Translationese gap = benchmark − native — pending
4. Note the protocol difference (translate-train vs. the paper's
   zero-shot MultiNLI→Odia setup) in the Notes/limitations section.

Never average, merge, or replace one score with the other — report both,
separately, alongside the gap.
