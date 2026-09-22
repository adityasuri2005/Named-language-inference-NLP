# Native Odia gold test set

Provided by the professor. Not committed — it is not ours to redistribute.

Drop the file here and pass its path:

```bash
python -m src.evaluate --split native --native-file data/native/<filename>
```

## Accepted formats

`.csv`, `.tsv`, `.json`, `.jsonl`, `.parquet`, `.xlsx`.

## Accepted column names

The loader maps common spellings automatically, so the file usually needs no editing:

| Meaning | Accepted column names |
| --- | --- |
| Premise | `premise`, `sentence1`, `sent1`, `text_a`, `context`, `premise_odia` |
| Hypothesis | `hypothesis`, `sentence2`, `sent2`, `text_b`, `hypothesis_odia` |
| Label | `label`, `gold_label`, `gold`, `annotation`, `labels`, `class` |

## Accepted label values

Strings (`entailment` / `neutral` / `contradiction`, plus short forms like `e`/`n`/`c`)
or integers (`0` / `1` / `2`) using the XNLI convention.

Anything the loader cannot map raises an error rather than guessing. Nothing is silently
dropped or reinterpreted.

See `example_format.csv` for the shape. That file is three illustrative rows only — it is
not evaluation data.
