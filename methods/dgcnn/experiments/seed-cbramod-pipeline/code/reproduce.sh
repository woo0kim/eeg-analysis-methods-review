#!/bin/bash
# The dgcnn runs of CBraMod's SEED-V pipeline on SEED and SEED-V. The code is shared by every model and lives in
# methods/cbramod/experiments/finetune-seed/code (README there: data, splits, pipeline, run configuration).
#
#   code/reproduce.sh collect   recompute results/ from logs/runs into results_rerun/ and diff (numpy, scikit-learn)
#   code/reproduce.sh train     enqueue the runs for tools/run_queue.py; needs the LMDBs (shared README, "Reproducing")
set -e
EXP=$(cd "$(dirname "$0")/.." && pwd)
C=$EXP/../../../cbramod/experiments/finetune-seed/code
PY=${PYTHON:-python}
case "$1" in
  collect)
    mkdir -p "$EXP/results_rerun"
    "$PY" "$C/collect.py" "$EXP/logs/runs" "$EXP/results_rerun" "$C/../results/test_trials.json" > "$EXP/results_rerun/results.txt"
    diff "$EXP/results_rerun/results.txt" "$EXP/results/results.txt" && echo "results.txt reproduced" ;;
  train)
    for ds in SEED SEED-V; do
      for m in dgcnn; do "$PY" "$C/add_jobs.py" $ds trial555 50 $m 0 1 2 3 4; done
      for m in dgcnn; do "$PY" "$C/add_jobs.py" $ds subject 80 $m 0 1 2 3 4; done
    done ;;
  *) echo "usage: $0 collect|train"; exit 1 ;;
esac
