#!/bin/bash
# The braindecode-EEGNet runs use the conda env "bd-cbramod" of the braindecode-CBraMod experiment, unchanged: a
# clone of CBraMod's "cbramod" env (Python 3.11.7, PyTorch 2.1.2 + CUDA 12.1, lmdb 1.4.1) plus braindecode 1.8.1 from
# the pinned submodule (methods/eegnet/upstream). That script builds it.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
bash "$HERE/../../../../cbramod/experiments/braindecode-public-datasets/env/setup_env.sh"
