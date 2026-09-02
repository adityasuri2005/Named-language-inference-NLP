# OdiaEval — Group 9 Scorecard

**Group:** 9 (Group leader: Rudra) — twinned with Group 3
**Task:** Named-language Inference (NLI) — News / YouTube Comments domain
**Model:** `ai4bharat/IndicBERTv2-MLM-only`
**Dataset:** IndicXNLI (Odia, `or`)
**Reference paper:** Aggarwal, Gupta & Kunchukuttan (2022), IndicXNLI, EMNLP 2022
https://aclanthology.org/2022.emnlp-main.755/
(target score sourced from Table 16 of Doddapaneni et al. 2023, ACL —
https://aclanthology.org/2023.acl-long.693/)

## Shared Scorecard

| Evaluation Dataset | Metric | Published Target | Our Score | Difference from Target |
|---|---|---|---|---|
| Benchmark Test (IndicXNLI, Odia) | Accuracy | 72.6% | ____ % | ____ pts |
| Native Odia Test | Accuracy | N/A | ____ % | N/A |

**Translationese Gap (Benchmark − Native): ____ points**

## Before

- Dataset + model pinned: IndicXNLI Odia (`Divyanshu/indicxnli` or
  `ai4bharat/IndicXNLI-Translated` — record which one `train_benchmark.py`
  actually used), `ai4bharat/IndicBERTv2-MLM-only`
- Train / validation / test splits: 392,702 / 2,490 / 5,010 (if using the
  full IndicXNLI release)
- Label mapping confirmed: 0 = entailment, 1 = neutral, 2 = contradiction

## During

- Random seed: ____
- Hyperparameters: LR 3e-5, batch size 32, weight decay 0.01, warmup ratio
  10%, max seq len 128, AdamW, early stopping patience 3
- Checkpoint selection: validation accuracy only

## After

- Benchmark test set evaluated once: ____ %
- Native Odia test set evaluated once (frozen model): ____ %
- Translationese gap: ____ points

## Notes / limitations

_(Document anything that didn't go as planned — failed tolerance on first
attempt, which dataset repo was actually used, seed sensitivity, ambiguity
in the native test set's label scheme, class imbalance, etc.)_
