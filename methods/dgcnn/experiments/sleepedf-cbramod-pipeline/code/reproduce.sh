#!/bin/bash
# The DGCNN runs (our implementation and braindecode's) of CBraMod's Sleep-EDF pipeline. The code is shared by every model and lives in
# methods/cbramod/experiments/finetune-sleepedf/code (README there: pipeline, data, split, run configuration).
#
#   code/reproduce.sh collect   recompute results/ from logs/runs into results_rerun/ and diff (numpy, scipy, scikit-learn)
#   code/reproduce.sh train     enqueue the 10 runs (seeds 0-4 per model) for tools/run_queue.py; needs the preprocessed data
#                               (shared README, "Reproducing") and the paths documented in add_jobs.py
set -e
EXP=$(cd "$(dirname "$0")/.." && pwd)
C=$EXP/../../../cbramod/experiments/finetune-sleepedf/code
PY=${PYTHON:-python}
MODELS="dgcnn bd-dgcnn"
case "$1" in
  collect)
    mkdir -p "$EXP/results_rerun"
    "$PY" "$C/collect.py" "$EXP/logs/runs" "$EXP/results_rerun" > "$EXP/results_rerun/results.txt"
    diff "$EXP/results_rerun/results.txt" "$EXP/results/results.txt" && echo "results.txt reproduced" ;;
  train)
    for m in $MODELS; do "$PY" "$C/add_jobs.py" $m 0 1 2 3 4; done ;;
  *) echo "usage: $0 collect|train"; exit 1 ;;
esac
