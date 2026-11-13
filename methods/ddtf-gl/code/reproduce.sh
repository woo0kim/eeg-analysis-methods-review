#!/bin/bash
# Reproduce a dDTF-GL experiment.   code/reproduce.sh <experiment> collect|train
#
#   collect  recompute results/ from the committed logs/runs into results_rerun/ and diff (numpy, scikit-learn; no data needed)
#   train    compute the features (extract.py) and every run (run_jobs.sh), then copy the runs into logs/runs.
#            Needs the data under $EEG_DATA_ROOT (datasets/README.md), the shared pipelines' preprocessed files (datasets.py),
#            upstream/ (setup_upstream.sh) and the ddtf-gl environment (../env/environment.yml); CPU only.
set -e
C=$(cd "$(dirname "$0")" && pwd)
NAME=$1
EXP=$C/../experiments/$NAME
PY=${PYTHON:-python}
WORK=${GL_WORK:-$HOME/repro-gl}
case "$NAME" in                                   # datasets of each experiment (no associative arrays: macOS ships bash 3.2)
  stew-cbramod-pipeline) DATA="STEW" ;;           stew-paper-protocol) DATA="STEW-6s" ;;
  cbramod-splits) DATA="MentalArithmetic Mumtaz2016 BCIC-IV-2a PhysioNet-MI" ;;
  sleepedf-cbramod-pipeline) DATA="SleepEDF" ;;   seed-cbramod-pipeline) DATA="SEED SEED-V" ;;
  *) echo "unknown experiment $NAME"; exit 1 ;;
esac
case "$2" in
  collect)
    mkdir -p "$EXP/results_rerun"
    "$PY" "$C/collect.py" "$NAME" "$EXP/logs/runs" "$EXP/results_rerun" > "$EXP/results_rerun/results.txt"
    diff "$EXP/results_rerun/results.txt" "$EXP/results/results.txt" && echo "results.txt reproduced"
    diff <("$PY" -c "import json,sys; json.dump(json.load(open(sys.argv[1])), sys.stdout, sort_keys=True)" "$EXP/results_rerun/results.json") \
         <("$PY" -c "import json,sys; json.dump(json.load(open(sys.argv[1])), sys.stdout, sort_keys=True)" "$EXP/results/results.json") \
      > /dev/null && echo "results.json reproduced" ;;
  train)
    for ds in $DATA; do "$PY" "$C/extract.py" "$ds"; done
    [ "$NAME" = stew-cbramod-pipeline ] && "$PY" "$C/extract.py" STEW --features dwt
    GL_PY=$PY "$C/run_jobs.sh" "$NAME"
    [ "$NAME" = stew-paper-protocol ] && GL_PY=$PY "$C/run_bundle.sh" 0 1 2 3 4
    mkdir -p "$EXP/logs/runs"
    cp -r "$WORK/runs/$NAME/." "$EXP/logs/runs/" ;;
  *) echo "usage: $0 <experiment> collect|train"; exit 1 ;;
esac
