"""Every model on STEW windows: (batch, 14, 4, 128) at 128 Hz; CBraMod and LaBraM read (batch, 14, 4, 200) at 200 Hz.

Each model as its own code builds it, with its own classifier; the training loop is ../../finetune-seed/code/trainer.py.

  cbramod, cbramod-scratch   released backbone and the released 'all_patch_reps' head for 14 x 4 x 200 features
  eegnet                     our PyTorch EEGNet-8,2, temporal kernel 64 (0.5 s at 128 Hz, the EEGNet paper's)
  conformer                  released model of conformer.py (10 heads, 3-layer head), input of the first head layer sized to 28 tokens
  dgcnn                      methods/dgcnn/code/dgcnn.py on 5-band DE of each of the 4 seconds: 20 values per channel
  bd-*                       braindecode 1.8.1 at n_times 512, sfreq 128 (bd-cbramod: 800 at 200 Hz); bd-eegnet kernel 64
"""
import os
import sys

import torch.nn as nn
from einops.layers.torch import Rearrange

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '../../finetune-seed/code')))
import emo_models as em  # noqa: E402  (paths, Flat, BDCBraMod, CBRAMOD_WEIGHTS)
from dgcnn import DGCNN, WithDE, chs_info  # noqa: E402

CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
N_CH, N_T, FS = 14, 512, 128
FOUNDATION = ('cbramod', 'cbramod-scratch', 'bd-cbramod', 'labram', 'bd-labram')


class CBraModSTEW(nn.Module):
    def __init__(self, k, pretrained):
        super().__init__()
        import torch
        sys.path.insert(0, em.CBRAMOD_SRC)
        from models.cbramod import CBraMod
        self.backbone = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8)
        if pretrained:
            self.backbone.load_state_dict(torch.load(em.CBRAMOD_WEIGHTS, map_location='cpu'))
        self.backbone.proj_out = nn.Identity()
        self.classifier = nn.Sequential(Rearrange('b c s d -> b (c s d)'), nn.Linear(N_CH * 4 * 200, 4 * 200), nn.ELU(), nn.Dropout(0.1),
                                        nn.Linear(4 * 200, 200), nn.ELU(), nn.Dropout(0.1), nn.Linear(200, k))

    def forward(self, x):
        return self.classifier(self.backbone(x))


class ConformerSTEW(nn.Module):
    """conformer.py's Conformer for 14 channels. Released ClassificationHead: Linear(2440, 256) + ELU + Dropout 0.5 + Linear(256, 32)
    + ELU + Dropout 0.3 + Linear(32, k); 2440 = 61 tokens x 40 for 1,000 samples, here 28 x 40 for 512."""

    def __init__(self, k):
        super().__init__()
        sys.path.insert(0, os.path.join(em.METHODS, 'cbramod/experiments/finetune-sleepedf/code'))
        from conformer_released import PatchEmbedding, TransformerEncoder
        self.patch, self.transformer = PatchEmbedding(40, n_chans=N_CH), TransformerEncoder(6, 40)
        self.fc = nn.Sequential(nn.Linear(28 * 40, 256), nn.ELU(), nn.Dropout(0.5), nn.Linear(256, 32), nn.ELU(), nn.Dropout(0.3), nn.Linear(32, k))

    def forward(self, x):
        return self.fc(self.transformer(self.patch(x.reshape(x.shape[0], 1, N_CH, -1))).flatten(1))


def build(name, k):
    if name in ('cbramod', 'cbramod-scratch'):
        return CBraModSTEW(k, name == 'cbramod')
    if name == 'eegnet':
        sys.path.insert(0, os.path.join(em.METHODS, 'eegnet/experiments/pytorch-cbramod-splits/code'))
        sys.path.insert(0, em.CBRAMOD_SRC)
        from run_eegnet import EEGNet
        return EEGNet(N_CH, N_T, k, kern=64)
    if name == 'conformer':
        return ConformerSTEW(k)
    if name == 'dgcnn':
        return WithDE(DGCNN(N_CH, 20, k), N_CH, FS, 4)
    if name in ('bd-cbramod', 'bd-cbramod-scratch'):
        em.N_CH, em.N_TIMES = N_CH, 800
        return em.BDCBraMod(k, name == 'bd-cbramod')
    from braindecode import models as bd
    F = em.Flat
    if name == 'bd-eegnet':
        return F(bd.EEGNet(n_chans=N_CH, n_outputs=k, n_times=N_T, F1=8, D=2, kernel_length=64, drop_prob=0.25, final_layer_with_constraint=True))
    if name == 'bd-conformer':
        return F(bd.EEGConformer(n_chans=N_CH, n_outputs=k, n_times=N_T, sfreq=FS))
    if name == 'bd-atcnet':
        return F(bd.ATCNet(n_chans=N_CH, n_outputs=k, n_times=N_T, sfreq=FS))
    if name == 'bd-deepsleepnet':
        return F(bd.DeepSleepNet(n_chans=N_CH, n_outputs=k, n_times=N_T, sfreq=FS))
    if name == 'bd-dgcnn':
        return F(bd.DGCNN(n_chans=N_CH, n_outputs=k, n_times=N_T, chs_info=chs_info(CHANNELS)))
    if name == 'bd-dgcnn-de':
        return WithDE(bd.DGCNN(n_chans=N_CH, n_outputs=k, n_times=20, chs_info=chs_info(CHANNELS)), N_CH, FS, 4)
    raise ValueError(name)
