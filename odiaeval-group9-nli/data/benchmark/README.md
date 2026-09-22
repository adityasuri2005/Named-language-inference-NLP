# Benchmark data — IndicXNLI (Odia)

The parquet files are excluded from git (~73 MB). Put them here before running anything:

```
data/benchmark/indicxnli-train.parquet        392,702 rows
data/benchmark/indicxnli-validation.parquet     2,490 rows
data/benchmark/indicxnli-test.parquet           5,010 rows
```

Each file has three columns: `premise` (string), `hypothesis` (string), `label` (int64).

Labels follow the XNLI convention: `0 = entailment`, `1 = neutral`, `2 = contradiction`.

## Getting them

Either copy the files the professor supplied, or rebuild them from HuggingFace:

```python
from datasets import load_dataset
ds = load_dataset("Divyanshu/indicxnli", "or")      # 'or' = Odia
for split in ["train", "validation", "test"]:
    ds[split].to_pandas()[["premise","hypothesis","label"]] \
        .to_parquet(f"data/benchmark/indicxnli-{split}.parquet", index=False)
```

Run `python scripts/verify_data.py` afterwards. It confirms the sizes, checks the text
is Odia script, and reports split overlap. SHA-256 hashes of whatever you place here are
recorded in `results/dataset_stats.json`, so the exact bytes used are on record.

## Verified properties of the supplied files

| Split | Rows | entailment | neutral | contradiction | mean Odia script ratio |
| --- | --- | --- | --- | --- | --- |
| train | 392,702 | 130,899 | 130,900 | 130,903 | 0.996 |
| validation | 2,490 | 830 | 830 | 830 | 0.997 |
| test | 5,010 | 1,670 | 1,670 | 1,670 | 0.997 |

No nulls, no empty strings. Zero exact `(premise, hypothesis)` overlap between any two splits.
