#!/usr/bin/env python3
"""Standalone data integrity gate. Run this before training.

    python scripts/verify_data.py
    python scripts/verify_data.py --native-file data/native/<file>

Writes results/dataset_stats.json and exits non-zero if any check fails.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import build_dataset_stats  # noqa: E402
from src.utils import load_config, resolve, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the IndicXNLI Odia splits.")
    parser.add_argument("--config", default="configs/indicbertv2_odia.yaml")
    parser.add_argument("--native-file", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    try:
        stats = build_dataset_stats(cfg, native_file=args.native_file)
    except (ValueError, FileNotFoundError) as exc:
        print(f"FAIL: {exc}")
        return 1

    write_json(resolve(cfg["output"]["results_dir"]) / "dataset_stats.json", stats)

    print("\nSplit sizes")
    for split, size in stats["split_sizes"].items():
        print(f"  {split:<11} {size:>8,}  (expected {cfg['data']['expected_sizes'][split]:,})")

    print("\nLabel distribution")
    for split, dist in stats["label_distribution"].items():
        print(f"  {split:<11} {dist}")

    print("\nScript verification (Odia block U+0B00-U+0B7F)")
    for entry in stats["script_verification"]:
        print(f"  {entry['split']:<11} mean {entry['mean_odia_ratio']:.4f}  "
              f"{entry['pct_rows_above_0.9']:.2f}% of rows >0.9  -> {entry['verdict']}")

    print("\nLeakage (exact premise+hypothesis pairs)")
    for pair, count in stats["leakage"]["overlaps"].items():
        print(f"  {pair:<34} {count}")
    print(f"  {stats['leakage']['verdict']}")

    if "native_test" in stats:
        native = stats["native_test"]
        print(f"\nNative Odia test: {native['rows']} rows  {native['label_distribution']}")
        print(f"  {native['script_verification']['verdict']}")

    passed = stats["leakage"]["verdict"].startswith("PASS")
    print("\nALL CHECKS PASSED" if passed else "\nCHECKS FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
