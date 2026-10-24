#!/bin/bash
# Conda env "atcnet-tf" for the released ATCNet code: the versions its README lists (Python 3.7, TensorFlow 2.7,
# NumPy 1.20, SciPy 1.7, scikit-learn 1.0, matplotlib 3.5), CUDA 11.2 / cuDNN 8.1 from conda-forge (TensorFlow 2.7's
# build), protobuf < 3.20 (TensorFlow 2.7 fails to import with protobuf 4), h5py 3.1 (TensorFlow 2.7's pin), and mne
# + python-dateutil because preprocess.py imports them at the top (they are used only for the CS2R dataset).
# TensorFlow 2.7 ships kernels for sm_50 and runs on the Quadro M6000 (sm_52).
set -e
source ~/miniforge3/etc/profile.d/conda.sh
conda env remove -y -n atcnet-tf 2>/dev/null || true
mamba create -y -n atcnet-tf -c conda-forge python=3.7 cudatoolkit=11.2 cudnn=8.1.0 pip
conda activate atcnet-tf
pip install tensorflow==2.7.0 keras==2.7.0 protobuf==3.19.6 numpy==1.20.3 scipy==1.7.3 scikit-learn==1.0.2 \
    matplotlib==3.5.3 h5py==3.1.0 mne==1.3.1 python-dateutil
LD_LIBRARY_PATH=$CONDA_PREFIX/lib python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
WORK=${ATCNET_WORK:-~/repro-atcnet}
mkdir -p "$WORK"
pip freeze > "$WORK/pip_freeze_atcnet-tf.txt"   # compare with env/pip_freeze.txt
echo SETUP_DONE
