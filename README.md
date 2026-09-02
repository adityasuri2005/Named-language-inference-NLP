# OdiaEval — Group 9 (Named-Language Inference)

| | |
|---|---|
| **Dataset** | IndicXNLI |
| **Model** | `ai4bharat/IndicBERTv2-MLM-only` |
| **Target** | **72.6% accuracy** |
| **Reference paper** | Aggarwal, Gupta & Kunchukuttan (2022), IndicXNLI, EMNLP 2022 — https://aclanthology.org/2022.emnlp-main.755/ |

The 72.6% target itself is not actually reported *in* that EMNLP 2022
paper (which only benchmarks XLM-R / IndicBERT-v1 / mBERT / MuRIL). It
comes from **Table 16** of the follow-up paper that introduced
`IndicBERTv2-MLM-only` and evaluated it on IndicXNLI:

> Doddapaneni, Aralikatte, Ramesh, Goyal, Khapra, Kunchukuttan & Kumar
> (2023), *Towards Leaving No Indic Language Behind*, ACL 2023.
> https://aclanthology.org/2023.acl-long.693/ — this is also the paper
> Groups 1/8's model link points to, and it's where `IndicBERTv2-MLM-only`
> itself comes from.

Table 16 there reports, per language, IndicXNLI accuracy for several
models; the "IndicBERT" row (MLM-only, no extra parallel-data objective)
gives **or (Odia) = 72.6**, exactly matching your assignment's target —
so that's confirmed as the right number to reproduce.

Official fine-tuning code for this exact setup exists at
https://github.com/AI4Bharat/IndicBERT under `fine-tuning/xnli/` — worth
skimming if you want to sanity-check hyperparameters against ours.

## Data note

Our group's dataset link (`ai4bharat/IndicXNLI-Translated`) looks, from
its HF listing, like a small eval-only mirror (1K–10K rows total) used for
benchmarking instruction-tuned LLMs — it may not actually contain a
per-language **train** split we can fine-tune a classifier on. `train_benchmark.py`
tries it first and automatically falls back to `Divyanshu/indicxnli`
(the original IndicXNLI release — same paper, same content, has the full
train/validation/test splits: 392,702 / 2,490 / 5,010 per language) if no
usable train split is found. Whichever repo it actually used gets recorded
in `benchmark_results.json`, so we can cite the right one in our report.

## Setup

Run this on a GPU — Colab's free T4 is enough.

```bash
pip install -r requirements.txt
```

## Step 1 — Reproduce the benchmark

```bash
python train_benchmark.py --output_dir ./indicbertv2-odia-nli --seed 42
```

This:
1. Loads IndicXNLI Odia (`or`) train/validation/test.
2. Fine-tunes `IndicBERTv2-MLM-only` with the paper's Table 13 "Best-IN"
   (in-language) hyperparameters for the XNLI task: LR 3e-5, weight decay
   0.01, batch size 32, 10% warmup, max sequence length 128, AdamW. The
   paper's own best epoch was 4 — we allow up to 6 with early stopping
   (patience 3) and pick the checkpoint by validation accuracy.
3. Evaluates **once** on the held-out IndicXNLI Odia test set (5,010 examples).
4. Prints the accuracy, compares it to the 72.6% target, and tells you
   whether you're within the 1–2 point tolerance (~70.6–74.6%).
5. Saves the model, tokenizer, and `benchmark_results.json`.

If we land outside tolerance: stop and debug before touching native data.
Likely culprits, in order:
- Wrong label mapping — verify 0/1/2 against the dataset card.
- Batch-size / LR mismatch with what actually produced 72.6% (the paper
  ran a grid search — try nearby learning rates 1e-5 or 5e-5 if 3e-5 undershoots).
- Pure seed variance — try 2-3 seeds and pick the best by *validation*
  accuracy, never by peeking at test accuracy.

## Step 2 — Evaluate on native Odia (frozen model)

```bash
python evaluate_native.py \
    --model_dir ./indicbertv2-odia-nli \
    --native_file /path/to/professors_native_odia_test.csv \
    --premise_col premise --hypothesis_col hypothesis --label_col label \
    --benchmark_results ./indicbertv2-odia-nli/benchmark_results.json
```

- Adjust `--premise_col` / `--hypothesis_col` / `--label_col` to match our
  professor's file. If labels are strings, pass e.g.
  `--label_map '{"entailment":0,"neutral":1,"contradiction":2}'`.
- Model, weights, and hyperparameters are unchanged from Step 1 — this
  script only loads the saved checkpoint and runs inference.
- Prints native accuracy, a classification report, a confusion matrix, and
  (given `--benchmark_results`) the translationese gap, and saves
  `native_results.json`.

## Reporting

Fill in `REPORT_TEMPLATE.md`:
1. Benchmark (IndicXNLI Odia test) accuracy.
2. Native Odia accuracy.
3. Translationese gap = benchmark − native.

Never average, merge, or replace one score with the other — report both,
separately, alongside the gap.
