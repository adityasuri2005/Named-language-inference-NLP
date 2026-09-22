# OdiaEval — Group 9 — Natural Language Inference

Reproduction of the published IndicXNLI Odia baseline, followed by evaluation of the
same frozen model on a native Odia gold test set, to measure the **translationese gap**.

| | |
| --- | --- |
| **Group** | 9 |
| **Task** | M2 — Natural Language Inference (Wikipedia passages) |
| **Dataset** | [IndicXNLI](https://huggingface.co/datasets/ai4bharat/IndicXNLI-Translated), Odia (`or` / `ory`) |
| **Model** | [`ai4bharat/IndicBERTv2-MLM-only`](https://huggingface.co/ai4bharat/IndicBERTv2-MLM-only) (278M params) |
| **Metric** | Accuracy |
| **Published target** | **72.6%** |
| **Sample sizes** | 392,702 train / 2,490 validation / 5,010 test |

**Reference papers**
- Dataset — Aggarwal, Gupta & Kunchukuttan, *IndicXNLI: Evaluating Multilingual Inference for Indian Languages*, EMNLP 2022. <https://aclanthology.org/2022.emnlp-main.755/>
- Model and target — Doddapaneni et al., *Towards Leaving No Indic Language Behind*, ACL 2023. <https://aclanthology.org/2023.acl-long.693/>

> The 72.6% figure is Table 16 of the ACL 2023 paper: IndicXNLI per-language accuracy,
> row `IndicBERT`, column `or` (Odia).

---

## Results

Filled in after the run. Full version with per-class breakdowns: [`docs/RESULTS.md`](docs/RESULTS.md).

| Evaluation Dataset | Metric | Published Target | Our Score | Difference |
| --- | --- | --- | --- | --- |
| Benchmark Test (IndicXNLI Odia, 5,010) | accuracy | 72.6% | _TBD_ | _TBD_ |
| Native Odia Test | accuracy | N/A | _TBD_ | N/A |

| Comparison | Score |
| --- | --- |
| Benchmark Score | _TBD_ |
| Native Odia Score | _TBD_ |
| **Translationese Gap** | _TBD_ |

---

## Quick start

Easiest path is the Colab notebook — it handles the GPU, the data and the
checkpoint backup:

**[`notebooks/OdiaEval_Group9_Colab.ipynb`](notebooks/OdiaEval_Group9_Colab.ipynb)**

Or locally, on a machine with a CUDA GPU:

```bash
git clone https://github.com/<your-username>/odiaeval-group9-nli.git
cd odiaeval-group9-nli
pip install -r requirements.txt

# Put the three IndicXNLI Odia parquet files in data/benchmark/
# Put the professor's native test set in data/native/

./scripts/run_all.sh data/native/<native-test-file>
```

Or step by step:

```bash
python scripts/verify_data.py                 # STEP 3 - integrity gate
python -m src.train                           # STEP 4 - fine-tune on TRAIN only
python -m src.evaluate --split benchmark_test # STEP 5 - benchmark reproduction
python -m src.evaluate --split native --native-file data/native/<file>   # STEP 6
python -m src.report                          # STEP 7-8 - gap + report
```

Training takes roughly 3.5–5 h on a T4 and 1.5 h on an A100. Checkpoints are written
every epoch; `python -m src.train --resume` picks up from the last one.

---

## How the protocol is enforced

The brief requires several guarantees. Each one is a check in code rather than a
claim in prose, and each writes its evidence to `results/`.

**Sample sizes are fixed.** `src/data.py` compares the loaded split sizes against
`expected_sizes` in the config and raises if they differ. Training cannot run on a
subsample.

**The language is verified.** Every split is checked for Odia script (Unicode block
U+0B00–U+0B7F) before training. If the mean script ratio drops below 0.90 the run
aborts. Measured on the supplied data: **0.996 train, 0.997 validation, 0.997 test**.

**Leakage is measured, not assumed.** Exact `(premise, hypothesis)` overlap is
computed between every pair of splits after NFC normalisation. Measured on the
supplied data: **0 overlapping pairs** between train and test, train and validation,
or validation and test.

**Test labels never influence training.** `src/train.py` does not load the test
split at all — it only opens `train_file` and `validation_file`. Checkpoint selection
is on validation accuracy. The test set is read once, by `src/evaluate.py`, after
training has finished.

**Both evaluations share one code path.** `src/evaluate.py --split benchmark_test`
and `--split native` run the same tokenizer, the same max sequence length, the same
normalisation, the same label mapping and the same metric. Any difference between
the two scores comes from the data, which is what makes the gap interpretable.

**Runs are reproducible.** Seed 42 across Python, NumPy and PyTorch; cuDNN set to
deterministic. Every output JSON records the config, the resolved model revision,
library versions and the GPU used. SHA-256 hashes of all input files are recorded in
`results/dataset_stats.json`.

---

## Repository layout

```
configs/indicbertv2_odia.yaml   All hyperparameters, each one cited to its source table
src/data.py                     Loading, Odia script verification, leakage checks
src/train.py                    Fine-tuning (train + validation only)
src/evaluate.py                 Evaluation for benchmark test and native test alike
src/report.py                   Translationese gap and the report tables
src/utils.py                    Seeding, environment capture, file hashing
scripts/verify_data.py          Standalone integrity gate
scripts/run_all.sh              Full pipeline in one command
notebooks/                      Colab notebook (recommended path)
results/                        Committed evidence - metrics, predictions, stats
docs/RESULTS.md                 Generated report
```

### What lands in `results/`

These files are committed deliberately: they are the evidence behind the claimed
numbers, and a grader can check the claims without rerunning anything.

| File | Contents |
| --- | --- |
| `dataset_stats.json` | split sizes, label distributions, script verification, leakage report, SHA-256 hashes |
| `train_results.json` | config, environment, model revision, training metrics, full log history |
| `stage_stats.json` | per-epoch validation accuracy, selected epoch |
| `metrics_benchmark_test.json` | benchmark accuracy, macro-F1, per-class scores, confusion matrix |
| `metrics_native.json` | same structure, native Odia set |
| `predictions_benchmark_test.csv` | per-example gold, prediction, per-class probabilities |
| `predictions_native.csv` | same, native set |
| `final_report.json` | the summary numbers |

Model weights are too large for git and are excluded by `.gitignore`. Attach them as a
GitHub Release asset and record the link in `docs/ARTIFACTS.md`.

---

## Configuration and where it comes from

| Setting | Value | Source |
| --- | --- | --- |
| Learning rate | 3e-5 | Doddapaneni et al. 2023, Table 13, IndicXNLI / IndicBERT, "Best IN" |
| Weight decay | 0.01 | same row |
| Epochs (max) | 4 | same row (B\* = 4) |
| Batch size | 32 | same paper, Appendix N |
| Warmup ratio | 0.1 | same paper, Appendix N |
| Max sequence length | 128 | Aggarwal et al. 2022, Table 5 (MSL) |
| Optimizer | AdamW | both papers |
| Precision | fp16 | Doddapaneni et al. 2023, Appendix N |
| Seed | 42 | fixed for this project |

The **"Best IN"** column is the right one because we have an in-language (Odia)
validation set of 2,490 examples, which is exactly the condition that column describes.

---

## One documented deviation

The brief pins the sample sizes at 392,702 / 2,490 / 5,010. Those are the **Odia**
IndicXNLI splits, so this is the *translate-train* setting: fine-tune on
Odia-translated MultiNLI, evaluate on the Odia test set.

Doddapaneni et al. produced their 72.6 for Odia under *zero-shot* transfer —
fine-tuning on English MultiNLI, evaluating on the same Odia test set. The test set,
the metric and the model are identical under both protocols; only the training
language differs.

We follow the brief's prescribed sample sizes, and therefore translate-train. Both
reference papers observe that translate-train matches or slightly exceeds English
zero-shot transfer for Indic NLI, so a score at or a little above 72.6 is the
expected outcome. This is recorded here and in `docs/RESULTS.md` because the brief
asks for any choice that moves the score to appear in the report.

---

## Team

Group 9 — Rudra (lead), Tanay, Aditya, Dhananjay
DTSC422 Natural Language Processing · FLAME University
