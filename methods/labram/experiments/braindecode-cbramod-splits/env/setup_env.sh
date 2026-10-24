#!/bin/bash
# Conda env "bd-labram": a clone of the "labram" env the released-code runs used (Python 3.11.16, PyTorch 2.0.1 +
# CUDA 11.8, timm 0.4.12, lmdb 1.4.1; ../../finetune-cbramod-splits/env/setup_env.sh) plus braindecode at the
# pinned submodule commit (methods/eegnet/upstream, v1.8.1) and huggingface_hub for braindecode's from_pretrained().
# torch is raised from 2.0.1 to 2.1.2 (same CUDA 11.8 build): braindecode 1.8.1 declares torch>=2.0 but imports
# torch.nn.utils.parametrizations.weight_norm (torch >= 2.1) when braindecode.models is imported. timm, numpy and
# the rest stay as in the released runs.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
source ~/miniforge3/etc/profile.d/conda.sh
conda env remove -y -n bd-labram 2>/dev/null || true
conda create -y -n bd-labram --clone labram
conda activate bd-labram
# torch, timm, numpy and lmdb repeated so that braindecode's dependencies cannot replace them
pip install torch==2.1.2+cu118 torchaudio==2.1.2+cu118 timm==0.4.12 "numpy<2" lmdb==1.4.1 "huggingface_hub[torch]>=0.20.0" \
    "$HERE/../../../../eegnet/upstream" --extra-index-url https://download.pytorch.org/whl/cu118
python -c "import torch, braindecode, numpy, timm; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), braindecode.__version__, numpy.__version__, timm.__version__)"
WORK=${BD_LABRAM_WORK:-~/repro-braindecode/labram}
mkdir -p "$WORK"
pip freeze > "$WORK/pip_freeze.txt"   # compare with env/pip_freeze.txt
echo SETUP_DONE
