#!/bin/bash
# Conda env "bd-cbramod": a clone of the "cbramod" env the released-code runs used (Python 3.11.7, PyTorch 2.1.2 +
# CUDA 12.1, lmdb 1.4.1; ../../finetune-public-datasets/env/setup_env.sh) plus braindecode at the pinned submodule
# commit (methods/eegnet/upstream, v1.8.1) and huggingface_hub for braindecode's from_pretrained(). Keeping the
# released runs' torch means the braindecode runs differ from them only in the model code.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
source ~/miniforge3/etc/profile.d/conda.sh
conda env remove -y -n bd-cbramod 2>/dev/null || true
conda create -y -n bd-cbramod --clone cbramod
conda activate bd-cbramod
# torch, numpy and lmdb repeated so that braindecode's dependencies cannot replace them
pip install torch==2.1.2 torchaudio==2.1.2 "numpy<2" lmdb==1.4.1 "huggingface_hub[torch]>=0.20.0" \
    "$HERE/../../../../eegnet/upstream" --extra-index-url https://download.pytorch.org/whl/cu121
python -c "import torch, braindecode, numpy; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), braindecode.__version__, numpy.__version__)"
WORK=${BD_CBRAMOD_WORK:-~/repro-braindecode/cbramod}
mkdir -p "$WORK"
pip freeze > "$WORK/pip_freeze.txt"   # compare with env/pip_freeze.txt
echo SETUP_DONE
