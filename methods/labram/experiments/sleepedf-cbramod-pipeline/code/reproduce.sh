#!/bin/bash
# The 'labram' runs of CBraMod's Sleep-EDF pipeline. The code is shared by every model and lives in
# methods/cbramod/experiments/finetune-sleepedf/code (README there: pipeline, data, split, run configuration).
#
#   code/reproduce.sh collect   recompute results/ from logs/runs into results_rerun/ and diff (numpy, scipy, scikit-learn)
#   code/reproduce.sh train     enqueue the 5 runs (seeds 0-4) for tools/run_queue.py; needs the preprocessed data
#                               (shared README, "Reproducing") and the paths documented in add_jobs.py
set -e
EXP=$(cd "$(dirname "$0")/.." && pwd)
C=$EXP/../../../cbramod/experiments/finetune-sleepedf/code
PY=${PYTHON:-python}
MODEL=labram
case "$1" in
  collect)
    mkdir -p "$EXP/results_rerun"
    "$PY" "$C/collect.py" "$EXP/logs/runs" "$EXP/results_rerun" > "$EXP/results_rerun/results.txt"
    diff "$EXP/results_rerun/results.txt" "$EXP/results/results.txt" && echo "results.txt reproduced" ;;
  train)
    "$PY" "$C/add_jobs.py" $MODEL 0 1 2 3 4 ;;
  *) echo "usage: $0 collect|train"; exit 1 ;;
esac
