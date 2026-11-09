"""Every model on SEED / SEED-V samples: 1-s windows of 62 channels at 200 Hz, shape (batch, 62, 1, 200) from CBraMod's
released loader, signal in units of 100 uV (loader: x / 100).

Each model is built as its own code builds it, with its own classifier; only the training loop is shared (trainer.py).

  cbramod, cbramod-scratch   released models/model_for_seedv.py (three-layer head), released weights / random init
  eegnet                     our PyTorch EEGNet-8,2 (eegnet/experiments/pytorch-cbramod-splits/code/run_eegnet.py)
  conformer                  released SEED model of conformer_seed_1s_5fold.py: 5 heads, head Linear(280, 32) + ELU +
                             Dropout(0.3) + Linear(32, classes)
  dgcnn                      methods/dgcnn/code/dgcnn.py on 5-band DE of the 1-s window, standardised per band
  bd-cbramod, bd-eegnet, bd-conformer, bd-atcnet, bd-deepsleepnet   braindecode 1.8.1, as in the earlier experiments
  bd-dgcnn                   braindecode's DGCNN as it ships: raw samples as node features (200 per node)
  bd-dgcnn-de                braindecode's DGCNN on the same DE input as dgcnn

ATCNet: on a 1-s input braindecode rescales its pools and drops to one sliding window. conv_block_pool_size_2=2 keeps
the five sliding windows (Tc = 12). DeepSleepNet: its 200 Hz first-layer sizes do not fit a 1-s window, so braindecode's
default sizes are used (kernels 0.25 s and 2 s at 200 Hz).
"""
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = os.path.normpath(os.path.join(HERE, '../../../..'))
sys.path.insert(0, os.environ.get('DGCNN_CODE', os.path.join(METHODS, 'dgcnn/code')))
from dgcnn import DGCNN, WithDE, chs_info  # noqa: E402
CBRAMOD_SRC = os.environ.get('CBRAMOD_SRC', os.path.join(METHODS, 'cbramod/upstream'))
CBRAMOD_WEIGHTS = os.environ.get('CBRAMOD_WEIGHTS', os.path.join(CBRAMOD_SRC, 'pretrained_weights/pretrained_weights.pth'))
BD_CBRAMOD = ('braindecode/cbramod-pretrained', '584cdc415913739a05d84bf0c1cb3db397764507')
N_CH, N_TIMES, FS = 62, 200, 200
# SEED's and SEED-V's channel order (the same 62 electrodes)
CHANNELS = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3',
            'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5',
            'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7',
            'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']


class Flat(nn.Module):
    """(batch, 62, 1, 200) -> (batch, 62, 200) for models that take channels x time."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x.reshape(x.shape[0], x.shape[1], -1))


class ConformerSEED(nn.Module):
    """The model of conformer_seed_1s_5fold.py (released). Attention, feed-forward and patch blocks are conformer_released.py,
    which match the SEED script line by line except for the number of heads (5 here)."""

    def __init__(self, n_classes):
        super().__init__()
        sys.path.insert(0, os.path.join(METHODS, 'cbramod/experiments/finetune-sleepedf/code'))
        from conformer_released import PatchEmbedding, TransformerEncoderBlock
        self.patch = PatchEmbedding(40, n_chans=N_CH)
        self.transformer = nn.Sequential(*[TransformerEncoderBlock(40, num_heads=5) for _ in range(6)])
        self.fc = nn.Sequential(nn.Linear(280, 32), nn.ELU(), nn.Dropout(0.3), nn.Linear(32, n_classes))

    def forward(self, x):
        return self.fc(self.transformer(self.patch(x.reshape(x.shape[0], 1, x.shape[1], -1))).flatten(1))


class BDCBraMod(nn.Module):
    """As cbramod/experiments/braindecode-public-datasets: encoder -> `backbone`, braindecode's head -> `head`."""

    def __init__(self, n_outputs, pretrained):
        super().__init__()
        from braindecode.models import CBraMod
        kw = dict(n_chans=N_CH, n_times=N_TIMES, n_outputs=n_outputs)
        m = CBraMod.from_pretrained(BD_CBRAMOD[0], revision=BD_CBRAMOD[1], **kw) if pretrained else CBraMod(**kw)
        self.head = m.final_layer
        m.final_layer = nn.Identity()
        self.backbone = m

    def forward(self, x):
        return self.head(self.backbone(x.reshape(x.shape[0], x.shape[1], -1)))


def build(name, params):
    k = params.num_of_classes
    if name in ('cbramod', 'cbramod-scratch'):
        sys.path.insert(0, CBRAMOD_SRC)
        from models import model_for_seedv
        params.use_pretrained_weights = name == 'cbramod'
        params.foundation_dir = CBRAMOD_WEIGHTS

        class SeedVModel(model_for_seedv.Model):
            # [MOD] the released forward flattens the features before a classifier whose first layer needs 4-D input,
            # which raises KeyError with every released head except avgpooling_patch_reps' sibling paths; this is the
            # forward of the other models (model_for_physio, model_for_faced, ...)
            def forward(self, x):
                return self.classifier(self.backbone(x))
        return SeedVModel(params)
    if name == 'eegnet':
        sys.path.insert(0, os.path.join(METHODS, 'eegnet/experiments/pytorch-cbramod-splits/code'))
        sys.path.insert(0, CBRAMOD_SRC)
        from run_eegnet import EEGNet
        return EEGNet(N_CH, N_TIMES, k)
    if name == 'conformer':
        return ConformerSEED(k)
    if name == 'dgcnn':
        return WithDE(DGCNN(N_CH, 5, k), N_CH, FS, 1)
    if name in ('bd-cbramod', 'bd-cbramod-scratch'):
        return BDCBraMod(k, name == 'bd-cbramod')
    from braindecode import models as bd
    if name == 'bd-eegnet':
        return Flat(bd.EEGNet(n_chans=N_CH, n_outputs=k, n_times=N_TIMES, F1=8, D=2, kernel_length=100, drop_prob=0.25,
                              final_layer_with_constraint=True))
    if name == 'bd-conformer':
        return Flat(bd.EEGConformer(n_chans=N_CH, n_outputs=k, n_times=N_TIMES, sfreq=FS))
    if name == 'bd-atcnet':
        return Flat(bd.ATCNet(n_chans=N_CH, n_outputs=k, n_times=N_TIMES, sfreq=FS, conv_block_pool_size_2=2))
    if name == 'bd-deepsleepnet':
        return Flat(bd.DeepSleepNet(n_chans=N_CH, n_outputs=k, n_times=N_TIMES, sfreq=FS))
    if name == 'bd-dgcnn':
        return Flat(bd.DGCNN(n_chans=N_CH, n_outputs=k, n_times=N_TIMES, chs_info=chs_info(CHANNELS)))
    if name == 'bd-dgcnn-de':
        return WithDE(bd.DGCNN(n_chans=N_CH, n_outputs=k, n_times=5, chs_info=chs_info(CHANNELS)), N_CH, FS, 1)
    raise ValueError(name)
