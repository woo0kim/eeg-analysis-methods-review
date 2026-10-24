#!/bin/bash
# The braindecode-ATCNet runs use the conda env "bd-cbramod" of the CBraMod braindecode experiment, unchanged: Python
# 3.11.7, PyTorch 2.1.2 + CUDA 12.1, NumPy 1.26, braindecode 1.8.1 from the pinned submodule (methods/eegnet/upstream).
# NumPy < 2 is needed because the released preprocess.py calls int() on 1-element arrays, which NumPy 2 rejects (the
# "eegnet" env has NumPy 2.4). That script builds it.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
bash "$HERE/../../../../cbramod/experiments/braindecode-public-datasets/env/setup_env.sh"
WORK=${ATCNET_WORK:-~/repro-atcnet}
mkdir -p "$WORK"
~/miniforge3/envs/bd-cbramod/bin/pip freeze > "$WORK/pip_freeze_bd-cbramod.txt"   # compare with env/pip_freeze.txt
