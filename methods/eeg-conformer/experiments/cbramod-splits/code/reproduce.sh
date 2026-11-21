#!/bin/bash
# The conformer bd-conformer runs on CBraMod's four public datasets (BCI-IV-2a, PhysioNet-MI, Mumtaz2016, MentalArithmetic), through
# CBraMod's released fine-tuning pipeline. The code is shared and lives in methods/eeg-conformer/experiments/cbramod-splits/code.
#
#   code/reproduce.sh collect   recompute results/ from logs/runs into results_rerun/ and diff (numpy, scipy)
#   code/reproduce.sh train     enqueue the runs for tools/run_queue.py; needs CBraMod's LMDBs (see add_jobs.py)
set -e
EXP=$(cd "$(dirname "$0")/.." && pwd)
C=$EXP/code
PY=${PYTHON:-python}
case "$1" in
  collect)
    mkdir -p "$EXP/results_rerun"
    "$PY" "$C/collect.py" "$EXP/logs/runs" "$EXP/results_rerun" conformer bd-conformer > "$EXP/results_rerun/results.txt"
    diff "$EXP/results_rerun/results.txt" "$EXP/results/results.txt" && echo "results.txt reproduced" ;;
  train)
    for m in conformer bd-conformer; do "$PY" "$C/add_jobs.py" $m 0 1 2 3 4; done ;;
  *) echo "usage: $0 collect|train"; exit 1 ;;
esac
