# OdiaEval — Group 9: Natural Language Inference in Odia

Reproduces a published NLI benchmark result for Odia, then evaluates the same
frozen model on a native (non-translated) Odia test set and reports the gap.

| | |
|---|---|
| **Task** | Natural Language Inference (entailment / neutral / contradiction) |
| **Dataset** | IndicXNLI, Odia (`or`) |
| **Model** | `ai4bharat/IndicBERTv2-MLM-only`, fine-tuned for 3-way classification |
| **Published target** | 72.6% accuracy (Table 16, Doddapaneni et al., 2023, ACL; https://aclanthology.org/2023.acl-long.693/) |
| **Dataset paper** | Aggarwal, Gupta & Kunchukuttan (2022), EMNLP; https://aclanthology.org/2022.emnlp-main.755/ |

## Results

| Evaluation set | Metric | Published target | Our score | Difference |
|---|---|---|---|---|
| Benchmark test (IndicXNLI Odia, 5,010 examples) | Accuracy | 72.6% | **75.21%** | +2.61 pts |
| Native Odia test (264 examples) | Accuracy | N/A | **84.85%** | N/A |

**Translationese gap (benchmark − native): −9.64 points.** The native score is
*higher* than the benchmark score, the opposite of the usual expectation. See
Limitations before reading anything into this.

Per-class F1 (entailment / neutral / contradiction):
- Benchmark: 0.750 / 0.726 / 0.782
- Native: 0.775 / 0.862 / 0.902

Native confusion matrix (rows = true, columns = predicted, order
entailment / neutral / contradiction):

```
[[62, 21,  5],
 [ 2, 84,  2],
 [ 8,  2, 78]]
```
The main error pattern is entailment predicted as neutral (21 of 88).

The 264 native examples are 88 per class. With a set this small, the native
accuracy has a margin of roughly ±4 points.

## The model behind these numbers

The scores come from **the checkpoint at training step 18405** (about 1.5 of up
to 6 configured epochs), not from a completed training schedule.

- Settings (read from the checkpoint's `training_args.bin`, saved as
  `experiment_config.json`): seed 42, learning rate 3e-5, batch size 32,
  weight decay 0.01, fp16 on, up to 6 epochs.
- Validation accuracy of that run, evaluated every half epoch:

  | Step | Validation accuracy |
  |---|---|
  | 6135 | 72.61% |
  | 12270 | 73.37% |
  | **18405** | **74.70%** (the checkpoint used here) |
  | 24540 | 75.94% (weights lost) |
  | 30675 | 76.79% (weights lost; best by validation) |

- Why step 18405: it is the best checkpoint whose weights survived. The
  checkpoints at 24540 and 30675 have their logs but no `model.safetensors`
  (most likely a Drive write failure, not confirmed). Checkpoint choice was
  made from this validation history and file availability, never from test or
  native scores.
- The step counts are consistent with training on the full IndicXNLI Odia train
  split (392,702 examples; 6,135 steps = half an epoch at batch size 32).

## Repository contents

| File | Purpose |
|---|---|
| `train_benchmark.py` | Fine-tunes the model on the IndicXNLI Odia train split and evaluates once on the benchmark test set |
| `evaluate_native.py` | Loads a saved model and scores it on the native file (inference only) |
| `predict_all.py` | Scores one checkpoint on both test sets and writes all result files below |
| `submission_outputs/predictions_benchmark.csv` | Per-example benchmark predictions (premise, hypothesis, label, predicted_label) |
| `submission_outputs/predictions_native.csv` | Per-example native predictions |
| `submission_outputs/benchmark_results.json` | Benchmark accuracy, classification report, confusion matrix |
| `submission_outputs/native_results.json` | Native accuracy, report, confusion matrix, translationese gap |
| `submission_outputs/experiment_config.json` | Training arguments of the run that produced the checkpoint |
| `requirements.txt` | Dependencies (see Environment) |
| `REPORT_TEMPLATE.md` | Filled scorecard |

The trained model is **not** stored in this repository (the weights file is
about 1.1 GB). Download it from: **[TODO: add link to the checkpoint folder]**.

## Reproducing

Evaluate the checkpoint on both test sets (inference only, a few minutes on a
GPU):

```bash
python predict_all.py \
    --model_dir path/to/checkpoint-18405 \
    --test_file indicxnli-test.parquet \
    --native_file 03_nli_groups_3_9_updated.jsonl \
    --label_map '{"entailment":0,"neutral":1,"contradiction":2}' \
    --out_dir submission_outputs
```

Train from scratch:

```bash
python train_benchmark.py --output_dir ./indicbertv2-odia-nli --seed 42 --fp16
```

`--fp16` matches the settings of the run that produced the checkpoint. Note
that the current script defaults to fp16 **off**, because earlier attempts with
fp16 on collapsed to chance accuracy (about 36%) in one run. The cause of that
collapse was not determined, so fp16-on training may or may not converge. Full
training takes hours on a single T4. A fresh run will not necessarily
reproduce the exact checkpoint above (see Limitations).

## Environment

Colab with a T4 GPU. Library versions: **[TODO: paste the output of
`pip freeze | grep -E "transformers|datasets|torch|accelerate"`]**. The pins in
`requirements.txt` are older than what the patched scripts were run with, so
update it to match.

## Preprocessing

Premise and hypothesis are tokenized as a sentence pair with the model's
tokenizer, truncated to 128 tokens, dynamic padding, no other cleaning. Labels
are 0 = entailment, 1 = neutral, 2 = contradiction. The native file uses string
labels, mapped with the same scheme. The benchmark test labels were never used
in training or checkpoint selection.

## Limitations

- **Single seed (42).** There is no estimate of run-to-run variation. Two runs
  with the same settings and seed gave a collapsed model (35.93% test
  accuracy) and a working one; the reason is unexplained.
- **Incomplete record of the training run.** The run that produced the
  checkpoint did not save its own results file, and it ran with an earlier
  version of `train_benchmark.py` than the one in this repository (the current
  version has compatibility patches for newer `transformers`/`datasets`). The
  dataset repository that run loaded was not recorded. The benchmark result
  files here were created afterwards by evaluating the checkpoint.
- **Not a full training schedule.** The model is a 1.5-epoch checkpoint of a
  run configured for up to 6 epochs.
- **Benchmark score is above the published target.** 75.21% is 2.61 points
  above 72.6%, which is slightly outside a ±2 point tolerance on the high
  side.
- **Native test set is small** (264 examples), and how it was constructed and
  how its difficulty compares with IndicXNLI is not known to us.
- **The negative gap is unexplained.** Possible reasons (a noisier translated
  test set, easier native examples, overlap with training data) were not
  established. Overlap check: none of the 264 native examples
  matches an IndicXNLI Odia train, validation or test example, either by
  exact premise or by exact premise-hypothesis pair (0/264 in every split).
  This only rules out exact string matches. Near-duplicates, paraphrases, or
  independent translations of the same English source sentences would not be
  caught.
- The benchmark test set was scored twice with the same checkpoint (a
  1,000-example sample, then the full set). No changes were made in between.
- The native evaluation used the frozen model: no training or tuning on
  native data.
