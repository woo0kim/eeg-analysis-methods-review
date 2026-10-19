#!/bin/bash
# Conda env "eegnet": braindecode at the pinned submodule commit (methods/eegnet/upstream, v1.8.1) plus the EEG
# Conformer experiment's requirements.txt, whose training loop run_eegnet.py imports. torch is the version of
# the EEG Conformer runs (2.6.0); its cu124 wheel still includes sm_50 kernels for the Quadro M6000 (sm_52).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
source ~/miniforge3/etc/profile.d/conda.sh
conda env remove -y -n eegnet 2>/dev/null || true
conda create -y -n eegnet python=3.11
conda activate eegnet
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# torch repeated so that braindecode's dependencies cannot pull in a different one
pip install torch==2.6.0 torchaudio==2.6.0 -r "$HERE/../../../../eeg-conformer/experiments/bciciv2a-epoch-selection/requirements.txt" \
    "$HERE/../../../upstream"
python -c "import torch, braindecode; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_arch_list(), braindecode.__version__)"
WORK=${EEGNET_WORK:-~/repro-eegnet}
mkdir -p "$WORK"
pip freeze > "$WORK/pip_freeze.txt"   # compare with env/pip_freeze.txt
