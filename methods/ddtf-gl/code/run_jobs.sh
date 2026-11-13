#!/bin/bash
# Run every job of an experiment (jobs.py) one after another, skipping runs that already have metrics.json.
#   code/run_jobs.sh <experiment> [name filter regex]
# Paths: GL_WORK (default ~/repro-gl: features/, cache/, runs/<experiment>/), GL_PY (python of the ddtf-gl env), GL_THREADS (20).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
EXP=$1
FILTER=${2:-.}
WORK=${GL_WORK:-$HOME/repro-gl}
PY=${GL_PY:-$HOME/miniforge3/envs/ddtf-gl/bin/python}
mkdir -p "$WORK/runs/$EXP"
"$PY" "$HERE/jobs.py" "$EXP" | grep -E -- "$FILTER" | while read -r name args; do
  out="$WORK/runs/$EXP/$name"
  [ -f "$out/metrics.json" ] && continue
  echo "$(date '+%F %T') start $name"
  # shellcheck disable=SC2086
  "$PY" "$HERE/run_gl.py" $args --out "$out" > "$out.stdout" 2>&1 || { echo "$(date '+%F %T') FAILED $name"; continue; }
  rm -f "$out.stdout"                                   # run_gl.py also writes everything to $out/log.txt
  echo "$(date '+%F %T') done  $name: $(grep '^test' "$out/log.txt")"
done
