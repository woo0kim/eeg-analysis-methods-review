#!/bin/bash
# The bd-deepsleepnet runs on STEW (7-fold GroupKFold across subjects, two tasks). The code is shared by every model and lives in
# methods/cbramod/experiments/finetune-stew/code (README there: data, windows, folds, run configuration).
#
#   code/reproduce.sh collect   recompute results/ from logs/runs into results_rerun/ and diff (numpy, scikit-learn)
#   code/reproduce.sh train     enqueue the runs for tools/run_queue.py; needs STEW's windows (shared README, "Reproducing")
set -e
EXP=$(cd "$(dirname "$0")/.." && pwd)
C=$EXP/../../../cbramod/experiments/finetune-stew/code
PY=${PYTHON:-python}
case "$1" in
  collect)
    mkdir -p "$EXP/results_rerun"
    "$PY" "$C/collect.py" "$EXP/logs/runs" "$EXP/results_rerun" > "$EXP/results_rerun/results.txt"
    diff "$EXP/results_rerun/results.txt" "$EXP/results/results.txt" && echo "results.txt reproduced" ;;
  train)
    for t in rest-task rating3; do for m in bd-deepsleepnet; do "$PY" "$C/add_jobs.py" $t $m 0 1 2 3 4; done; done ;;
  *) echo "usage: $0 collect|train"; exit 1 ;;
esac
