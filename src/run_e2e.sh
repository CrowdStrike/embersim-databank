#!/usr/bin/env bash

# Run end-to-end pipeline
# (EMBER + AVClass) -> (Tag-based augmentation) -> (tag ranking) -> (Relevance@K eval)

set -euo pipefail

HERE="$(dirname $0)"
DATA_DIR="$HERE/../data/processed"

python3 $HERE/e2e.py \
    --input-dataframe $DATA_DIR/ember_with_avclass_dataset.pkl \
    --sim-results $DATA_DIR/xgb-sim-results/test_vs_train_test.pkl \
    --rank-by FAM \
    --rank-top-only \
    --relevance-type exact \
    --topk 100 \
    --output-results-file $DATA_DIR/eval-results/most-prevalent-tag/test_vs_traintest_prec_at_100_top_FAM.pkl