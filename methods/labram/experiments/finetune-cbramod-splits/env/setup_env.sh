#!/bin/bash
# Conda env "labram" for the released LaBraM code. LaBraM's README: Python 3.11, PyTorch 2.0.1 + CUDA 11.8,
# tensorboardX, requirements.txt. deepspeed==0.4.0 is left out: it is only imported with --enable_deepspeed.
set -e
source ~/miniforge3/etc/profile.d/conda.sh
conda env remove -y -n labram 2>/dev/null || true
conda create -y -n labram python=3.11
conda activate labram
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
# torch/torchvision repeated so that timm's torchvision dependency cannot pull in a newer torch
pip install torch==2.0.1 torchvision==0.15.2 "numpy<2" timm==0.4.12 einops tensorboardX h5py mne==1.4.2 pyhealth==1.1.4 scipy pandas scikit-learn
# reads CBraMod's LMDBs; lmdb 2.x refuses the loaders' repeated open of the same environment
pip install lmdb==1.4.1
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_arch_list())"
WORK=${LABRAM_WORK:-~/repro-labram}
mkdir -p "$WORK"
pip freeze > "$WORK/pip_freeze.txt"   # compare with env/pip_freeze.txt
