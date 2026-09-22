"""Dataset loading plus the integrity checks the project brief asks us to record.

Three things get verified before any training happens:

1.  Split sizes match the published benchmark exactly (392,702 / 2,490 / 5,010).
    The brief is explicit that the sample size must not be reduced or augmented.
2.  The text is actually Odia script (U+0B00-U+0B7F). A sibling group had to redo
    their whole experiment after training on the wrong language, so this is a
    hard gate rather than a warning.
3.  No (premise, hypothesis) pair appears in both a training split and an
    evaluation split.

The same tokenisation path is used for the benchmark test set and the native
Odia test set. That is what makes the translationese gap meaningful: any
difference in the two scores comes from the data, not from the pipeline.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .utils import resolve, sha256_file

# XNLI label convention, shared by IndicXNLI and MultiNLI.
LABEL_NAMES: List[str] = ["entailment", "neutral", "contradiction"]
LABEL2ID: Dict[str, int] = {name: i for i, name in enumerate(LABEL_NAMES)}

# Common aliases seen in hand-annotated files, so a native test set that spells
# labels differently still loads without anyone editing it by hand.
_LABEL_ALIASES: Dict[str, int] = {
    **LABEL2ID,
    "entail": 0,
    "entailed": 0,
    "e": 0,
    "0": 0,
    "neutral": 1,
    "neither": 1,
    "n": 1,
    "1": 1,
    "contradiction": 2,
    "contradict": 2,
    "contradictory": 2,
    "c": 2,
    "2": 2,
}

ODIA_BLOCK = (0x0B00, 0x0B7F)


# --------------------------------------------------------------------------
# Script verification
# --------------------------------------------------------------------------
def odia_script_ratio(text: str) -> float:
    """Fraction of alphabetic characters that fall in the Odia Unicode block.

    Punctuation, digits and whitespace are ignored, so a sentence containing
    Latin numerals or ASCII punctuation is not penalised.
    """
    letters = [c for c in str(text) if c.isalpha()]
    if not letters:
        return 0.0
    lo, hi = ODIA_BLOCK
    return sum(1 for c in letters if lo <= ord(c) <= hi) / len(letters)


def verify_odia(
    df: pd.DataFrame,
    split_name: str,
    sample_size: int = 3000,
    min_mean_ratio: float = 0.90,
    seed: int = 42,
) -> Dict[str, Any]:
    """Confirm a split is genuinely Odia. Raises if it is not."""
    sample = df.sample(min(sample_size, len(df)), random_state=seed)
    premise_ratio = sample["premise"].map(odia_script_ratio)
    hypothesis_ratio = sample["hypothesis"].map(odia_script_ratio)
    combined = pd.concat([premise_ratio, hypothesis_ratio])

    stats = {
        "split": split_name,
        "sampled_rows": int(len(sample)),
        "mean_odia_ratio": round(float(combined.mean()), 4),
        "median_odia_ratio": round(float(combined.median()), 4),
        "pct_rows_above_0.9": round(float((combined > 0.9).mean() * 100), 2),
        "min_odia_ratio": round(float(combined.min()), 4),
    }

    if stats["mean_odia_ratio"] < min_mean_ratio:
        raise ValueError(
            f"Split '{split_name}' does not look like Odia: mean script ratio "
            f"{stats['mean_odia_ratio']:.3f} < {min_mean_ratio}. Check you loaded "
            f"the 'ory'/'or' subset and not another language."
        )
    stats["verdict"] = "PASS - text is Odia script"
    return stats


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _normalise(text: Any) -> str:
    """NFC-normalise and collapse whitespace.

    Odia is an abugida with combining marks, and the same visual string can have
    more than one Unicode encoding. Normalising means an identical sentence
    hashes identically, which the leakage check depends on. This is applied to
    the benchmark data and the native data alike.
    """
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(text))).strip()


def load_split(path: str | Path, normalise: bool = True) -> pd.DataFrame:
    """Load one benchmark split from parquet (or csv/tsv/jsonl)."""
    path = resolve(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Put the IndicXNLI Odia parquet files in "
            f"data/benchmark/ (see data/benchmark/README.md)."
        )
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in {".csv", ".tsv"}:
        df = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    elif suffix in {".jsonl", ".json"}:
        df = pd.read_json(path, lines=(suffix == ".jsonl"))
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    missing = {"premise", "hypothesis", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required column(s): {sorted(missing)}")

    df = df[["premise", "hypothesis", "label"]].copy()
    if normalise:
        df["premise"] = df["premise"].map(_normalise)
        df["hypothesis"] = df["hypothesis"].map(_normalise)
    df["label"] = df["label"].astype(int)

    bad = sorted(set(df["label"]) - {0, 1, 2})
    if bad:
        raise ValueError(f"{path.name} has out-of-range labels: {bad}")
    return df.reset_index(drop=True)


def load_native_test(path: str | Path) -> pd.DataFrame:
    """Load the professor's native Odia test set.

    Deliberately tolerant about column naming and label spelling, because the
    file is hand-built and we must not silently drop rows or reinterpret labels.
    Anything it cannot map raises rather than guesses.
    """
    path = resolve(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Native Odia test set not found at {path}.\n"
            f"Place the file the professor provided in data/native/ and pass its "
            f"path with --native-file."
        )

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in {".csv", ".tsv"}:
        df = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    elif suffix in {".jsonl", ".json"}:
        df = pd.read_json(path, lines=(suffix == ".jsonl"))
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported native test file type: {suffix}")

    # Map common column spellings onto premise / hypothesis / label.
    aliases = {
        "premise": "premise", "sentence1": "premise", "sent1": "premise",
        "text_a": "premise", "context": "premise", "premise_odia": "premise",
        "hypothesis": "hypothesis", "sentence2": "hypothesis", "sent2": "hypothesis",
        "text_b": "hypothesis", "hypothesis_odia": "hypothesis",
        "label": "label", "gold_label": "label", "gold": "label",
        "annotation": "label", "labels": "label", "class": "label",
    }
    renamed = {c: aliases[c.strip().lower()] for c in df.columns
               if c.strip().lower() in aliases}
    df = df.rename(columns=renamed)

    missing = {"premise", "hypothesis", "label"} - set(df.columns)
    if missing:
        raise ValueError(
            f"Could not find column(s) {sorted(missing)} in {path.name}. "
            f"Columns present: {list(df.columns)}. Rename them to "
            f"premise / hypothesis / label, or extend the alias map in src/data.py."
        )

    df = df[["premise", "hypothesis", "label"]].copy()
    df["premise"] = df["premise"].map(_normalise)
    df["hypothesis"] = df["hypothesis"].map(_normalise)

    def to_id(value: Any) -> int:
        if isinstance(value, (int,)) or (isinstance(value, float) and float(value).is_integer()):
            iv = int(value)
            if iv in (0, 1, 2):
                return iv
            raise ValueError(f"Unmappable numeric label: {value!r}")
        key = str(value).strip().lower()
        if key in _LABEL_ALIASES:
            return _LABEL_ALIASES[key]
        raise ValueError(
            f"Unmappable label {value!r}. Known labels: {sorted(set(_LABEL_ALIASES))}"
        )

    df["label"] = df["label"].map(to_id)

    empty = df[(df["premise"] == "") | (df["hypothesis"] == "")]
    if len(empty):
        raise ValueError(f"{len(empty)} row(s) in the native test set have empty text.")

    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# Leakage
# --------------------------------------------------------------------------
def _pair_keys(df: pd.DataFrame) -> set:
    return set(df["premise"] + "\u241f" + df["hypothesis"])


def leakage_report(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    native: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Exact (premise, hypothesis) overlap between every pair of splits.

    Training data overlapping an evaluation set would inflate the score, so this
    runs before training and the output is committed to results/.
    """
    keys = {"train": _pair_keys(train), "validation": _pair_keys(validation),
            "test": _pair_keys(test)}
    if native is not None:
        keys["native_test"] = _pair_keys(native)

    report: Dict[str, Any] = {
        "unique_pairs": {k: len(v) for k, v in keys.items()},
        "overlaps": {},
    }
    names = list(keys)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            report["overlaps"][f"{a}__vs__{b}"] = len(keys[a] & keys[b])

    contaminating = {
        k: v for k, v in report["overlaps"].items()
        if v > 0 and k.startswith("train__vs__")
    }
    report["train_eval_contamination"] = contaminating
    report["verdict"] = (
        "PASS - no training pair appears in any evaluation split"
        if not contaminating
        else f"FAIL - training data overlaps evaluation data: {contaminating}"
    )
    return report


# --------------------------------------------------------------------------
# One-shot integrity gate
# --------------------------------------------------------------------------
def build_dataset_stats(cfg: Dict[str, Any], native_file: Optional[str] = None) -> Dict[str, Any]:
    """Load everything, run every check, and return a committable stats blob."""
    paths = cfg["data"]
    train = load_split(paths["train_file"])
    validation = load_split(paths["validation_file"])
    test = load_split(paths["test_file"])
    native = load_native_test(native_file) if native_file else None

    expected = paths["expected_sizes"]
    actual = {"train": len(train), "validation": len(validation), "test": len(test)}
    mismatches = {k: {"expected": expected[k], "actual": actual[k]}
                  for k in expected if expected[k] != actual[k]}
    if mismatches:
        raise ValueError(
            f"Split sizes do not match the published benchmark: {mismatches}. "
            f"The brief forbids reducing or augmenting the sample size."
        )

    stats: Dict[str, Any] = {
        "split_sizes": actual,
        "split_sizes_match_published": True,
        "label_distribution": {
            name: {LABEL_NAMES[i]: int((frame["label"] == i).sum()) for i in range(3)}
            for name, frame in
            [("train", train), ("validation", validation), ("test", test)]
        },
        "script_verification": [
            verify_odia(train, "train"),
            verify_odia(validation, "validation"),
            verify_odia(test, "test"),
        ],
        "leakage": leakage_report(train, validation, test, native),
        "file_hashes": {
            k: sha256_file(resolve(paths[f"{k}_file"]))
            for k in ("train", "validation", "test")
        },
    }

    if native is not None:
        stats["native_test"] = {
            "rows": len(native),
            "label_distribution": {
                LABEL_NAMES[i]: int((native["label"] == i).sum()) for i in range(3)
            },
            "script_verification": verify_odia(native, "native_test"),
            "file_hash": sha256_file(resolve(native_file)),
        }

    return stats


def as_hf_dataset(df: pd.DataFrame):
    """pandas -> datasets.Dataset, dropping the pandas index column."""
    from datasets import Dataset

    return Dataset.from_pandas(df.reset_index(drop=True), preserve_index=False)


def make_tokenize_fn(tokenizer, max_length: int):
    """Tokenisation used identically for train, benchmark test and native test.

    Premise is the first segment and hypothesis the second, matching the standard
    XNLI sentence-pair encoding.
    """

    def tokenize(batch):
        return tokenizer(
            batch["premise"],
            batch["hypothesis"],
            truncation=True,
            max_length=max_length,
            padding=False,  # dynamic padding per batch via the collator
        )

    return tokenize
