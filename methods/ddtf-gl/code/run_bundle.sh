#!/bin/bash
# The authors' bundle as released (upstream/mwl_gl_si7.py: DWT -> DFT -> LNT -> XGBoost, its own subject-independent GroupKFold(7)
# over the raw STEW recordings), one run per seed.
#   code/run_bundle.sh <seed> [<seed> ...]
# Its config file is copied with four keys changed: mwl_dir (the STEW files), save_json (the run directory), seed, and
# progress: false (tqdm bars only). Everything else, including XGBoost's n_jobs = -1 (every core), is as released.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
UP=${GL_UPSTREAM:-$HERE/../upstream}
WORK=${GL_WORK:-$HOME/repro-gl}
PY=${GL_PY:-$HOME/miniforge3/envs/ddtf-gl/bin/python}
RAW=${STEW_RAW:-$HOME/data/STEW/STEW Dataset}
for s in "$@"; do
  out=$WORK/runs/stew-paper-protocol/bundle__s$s
  [ -f "$out/mwl_gl_si7.json" ] && grep -q '"complete": true' "$out/mwl_gl_si7.json" && continue
  mkdir -p "$out"
  sed -e "s|^mwl_dir:.*|mwl_dir: $RAW|" -e "s|^save_json:.*|save_json: $out/mwl_gl_si7.json|" -e "s|^seed:.*|seed: $s|" \
      -e "s|^progress:.*|progress: false|" "$UP/config_mwl_gl_si7.yaml" > "$out/config.yaml"
  diff "$UP/config_mwl_gl_si7.yaml" "$out/config.yaml" > "$out/config.diff" || true
  echo "$(date '+%F %T') bundle seed $s"
  (cd "$UP" && "$PY" mwl_gl_si7.py --config "$out/config.yaml") > "$out/log.txt" 2>&1
  grep -A12 '"segment_acc"' "$out/mwl_gl_si7.json" | tail -14
done
