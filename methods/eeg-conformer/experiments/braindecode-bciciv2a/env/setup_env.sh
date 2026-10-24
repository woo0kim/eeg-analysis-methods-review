#!/bin/bash
# The braindecode-EEGConformer runs use the conda env "eegnet" of the EEGNet experiment, unchanged: Python 3.11.17,
# torch 2.6.0+cu124 (the version of the released-code EEG Conformer runs), braindecode 1.8.1 from the pinned
# submodule (methods/eegnet/upstream), plus the EEG Conformer experiment's requirements.txt. That script builds it.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
bash "$HERE/../../../../eegnet/experiments/bciciv2a-epoch-selection/env/setup_env.sh"
WORK=${BDCONF_WORK:-~/repro-braindecode/conformer}
mkdir -p "$WORK"
~/miniforge3/envs/eegnet/bin/pip freeze > "$WORK/pip_freeze.txt"   # compare with env/pip_freeze.txt
