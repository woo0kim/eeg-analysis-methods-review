#!/bin/bash
set -e
cd ~
if [ ! -d ~/miniforge3 ]; then
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/Miniforge3.sh
  bash ~/Miniforge3.sh -b -p ~/miniforge3
  rm ~/Miniforge3.sh
fi
source ~/miniforge3/etc/profile.d/conda.sh
conda create -y -n cbramod python=3.11.7
conda activate cbramod
# Paper: Python 3.11.7, PyTorch 2.1.2 + CUDA 12.1
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2" einops h5py lmdb matplotlib mne pandas ptflops pyEDFlib scikit_learn scipy torchinfo tqdm umap_learn
# lmdb 2.x refuses the loaders' triple open of the same environment
pip install lmdb==1.4.1
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_arch_list()); x=torch.randn(64,64,device=\"cuda\"); print((x@x).sum().item())"
WORK=${CBRAMOD_WORK:-~/repro}
mkdir -p "$WORK"
pip freeze > "$WORK/pip_freeze.txt"   # compare with env/pip_freeze.txt
echo SETUP_DONE
