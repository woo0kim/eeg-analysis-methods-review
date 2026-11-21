#!/bin/bash
# dDTF-GL+, the feature-tuning experiment. The code is shared and lives in methods/ddtf-gl/code (gl_plus.py, run_gl_plus.py, tune.py,
# tune_report.py, spectral.py).
#
#   code/reproduce.sh collect   results/results.txt and results.json from results/selection/ and the pipeline experiments' results
#                               (whose collect re-scores the final runs), and every candidate's validation score recomputed from logs/
#                               (numpy, scipy, scikit-learn; no data); diff against the committed results.txt
#   code/reproduce.sh train     the spectral features (extract.py --features spec | spect), every search and the final runs (CPU; needs
#                               the dDTF features of the other experiments, upstream/ and the ddtf-gl environment), then copy the runs:
#                               search runs into logs/<target>/, final runs into the pipeline experiments' logs/runs
set -e
EXP=$(cd "$(dirname "$0")/.." && pwd)
C=$EXP/../../code
PY=${PYTHON:-python}
case "$1" in
  collect)
    mkdir -p "$EXP/results_rerun" && cp -r "$EXP/results/selection" "$EXP/results_rerun/"
    "$PY" "$C/tune_report.py" "$EXP/results_rerun" > "$EXP/results_rerun/results.txt"
    diff "$EXP/results_rerun/results.txt" "$EXP/results/results.txt" && echo "results.txt reproduced" ;;
  train)
    for ds in STEW STEW-6s MentalArithmetic Mumtaz2016 BCIC-IV-2a SleepEDF PhysioNet-MI; do
      for k in spec spect; do "$PY" "$C/extract.py" $ds --features $k; done
    done
    for ds in SEED SEED-V; do "$PY" "$C/extract.py" $ds --features spec; done
    for t in STEW-rest STEW-rating3 MentalArithmetic Mumtaz2016 SleepEDF BCIC-IV-2a PhysioNet-MI SEED; do "$PY" "$C/tune.py" $t; done
    "$PY" "$C/tune.py" STEW-6s-SI --final-only
    # SEED-V: stages 1-2 only (every candidate at chance), then the final runs: run tune.py SEED-V, stop it after the 'projections'
    # stage is written to selection.json, then: "$PY" "$C/tune.py" SEED-V --final-only
    echo "copy \$GL_WORK/runs-plus/<target>/search into logs/<target>/ and .../final into the pipeline experiments' logs/runs" ;;
  *) echo "usage: $0 collect|train"; exit 1 ;;
esac
