#!/bin/bash
# One unmodified run of the authors' EEG/eeg_DE.py: all 45 SEED sessions, one after the other.
# usage: run_authors.sh <gpu> <de_LDS|de_movingAve> <repeat> <work dir> <de root> <upstream EEG-TAC dir> <python>
# eeg_DE.py reads ../../../de_feature relative to its folder and rewrites summaries/ and checkpoints/ next to lib/,
# so every run gets its own copy of the sources. It ends in pdb.set_trace(), which exits on the closed stdin.
set -e
GPU=$1; KEY=$2; REP=$3; WORK=$4; DE=$5; SRC=$6; PY=$7
RUN=$WORK/runs/authors__${KEY}__r$REP
mkdir -p "$RUN/a/b"
ln -sfn "$DE/$KEY" "$RUN/a/de_feature"
rm -rf "$RUN/a/b/EEG-TAC"; cp -r "$SRC" "$RUN/a/b/EEG-TAC"
cd "$RUN/a/b/EEG-TAC/EEG"
export LD_LIBRARY_PATH=${NVML_FIX:-$HOME/repro-braindecode/nvml-580.173.02/lib}:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=$GPU "$PY" -u eeg_DE.py < /dev/null > "$RUN/log.txt" 2>&1 || true
cp accuracy.mat "$RUN/accuracy.mat"
rm -rf "$RUN/a/b/EEG-TAC" "$RUN/a/de_feature" "$RUN/a"
