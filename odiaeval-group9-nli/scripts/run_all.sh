#!/usr/bin/env bash
# End-to-end pipeline. Run from the repo root.
#   ./scripts/run_all.sh                       # benchmark half only
#   ./scripts/run_all.sh data/native/<file>    # both halves
set -euo pipefail

NATIVE="${1:-}"

echo "== STEP 3: data integrity gate =="
python scripts/verify_data.py ${NATIVE:+--native-file "$NATIVE"}

echo "== STEP 4: fine-tune on benchmark TRAIN only =="
python -m src.train

echo "== STEP 5: benchmark reproduction (held-out TEST, evaluated once) =="
python -m src.evaluate --split benchmark_test

if [[ -n "$NATIVE" ]]; then
  echo "== STEP 6: native Odia evaluation (same model, same preprocessing) =="
  python -m src.evaluate --split native --native-file "$NATIVE"
else
  echo "== STEP 6: skipped, no native test set supplied =="
fi

echo "== STEP 7-8: translationese gap + report =="
python -m src.report
