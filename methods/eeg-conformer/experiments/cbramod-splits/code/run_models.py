"""EEG Conformer, ATCNet and DeepSleepNet through CBraMod's released fine-tuning pipeline on its four public datasets: the same
LMDBs and subject splits, Trainer, loss, optimizer, schedule, model selection and Evaluator as every other model run on these
splits (methods/eegnet/experiments/pytorch-cbramod-splits, methods/dgcnn/experiments/cbramod-splits). Only the model differs.

  conformer        the released model of conformer.py (BCI-IV-2a script): patch embedding, 6 encoder blocks with 10 heads,
                   its 3-layer head, whose first layer is sized to the tokens of the input (61 x 40 = 2440 for 1,000 samples,
                   the released size; 47 x 40 for 800)
  bd-conformer     braindecode 1.8.1 EEGConformer(n_chans, n_outputs, n_times, sfreq=200)
  bd-atcnet        braindecode 1.8.1 ATCNet(n_chans, n_outputs, n_times, sfreq=200), its defaults (800 and 1,000 samples are
                   above its 616-sample minimum, so nothing is rescaled: 5 sliding windows)
  bd-deepsleepnet  braindecode 1.8.1 DeepSleepNet(n_chans, n_outputs, n_times, sfreq=200), its default kernel sizes, as on SEED
                   and STEW (off-label: a sleep-staging model)

Every model gets CBraMod's samples as they are: (batch, channels, patches, 200) at 200 Hz in units of 100 uV, reshaped to
(batch, channels, time). Binary tasks give (batch,) logits, as the released heads.

usage: python run_models.py --downstream_dataset X --datasets_dir D --model conformer --seed S --model_dir M
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = os.path.normpath(os.path.join(HERE, '../../../..'))
sys.path.insert(0, os.environ.get('CBRAMOD_SRC', os.path.join(METHODS, 'cbramod/upstream')))
from datasets import physio_dataset, bciciv2a_dataset, mumtaz_dataset, stress_dataset  # noqa: E402
from finetune_trainer import Trainer  # noqa: E402

# dataset -> (loader module, n_channels, n_times @200 Hz, n_outputs, task), as run_eegnet.py and run_dgcnn.py
CFG = {
    'PhysioNet-MI': (physio_dataset, 64, 800, 4, 'multi'),
    'BCIC-IV-2a': (bciciv2a_dataset, 22, 800, 4, 'multi'),
    'Mumtaz2016': (mumtaz_dataset, 19, 1000, 1, 'binary'),
    'MentalArithmetic': (stress_dataset, 20, 1000, 1, 'binary'),
}
MODELS = ('conformer', 'bd-conformer', 'bd-atcnet', 'bd-deepsleepnet')


class Conformer(nn.Module):
    """conformer.py's Conformer (released, BCI-IV-2a): the blocks are conformer_released.py, which matches the released file
    line by line except for the channel count, an argument here. Its ClassificationHead: Linear(tokens x 40, 256) + ELU +
    Dropout 0.5 + Linear(256, 32) + ELU + Dropout 0.3 + Linear(32, k)."""

    def __init__(self, n_ch, n_times, k):
        super().__init__()
        sys.path.insert(0, os.path.join(METHODS, 'cbramod/experiments/finetune-sleepedf/code'))
        from conformer_released import PatchEmbedding, TransformerEncoder
        self.patch, self.transformer = PatchEmbedding(40, n_chans=n_ch), TransformerEncoder(6, 40)
        with torch.no_grad():
            tokens = self.patch(torch.zeros(2, 1, n_ch, n_times)).shape[1]
        self.fc = nn.Sequential(nn.Linear(tokens * 40, 256), nn.ELU(), nn.Dropout(0.5), nn.Linear(256, 32), nn.ELU(),
                                nn.Dropout(0.3), nn.Linear(32, k))

    def forward(self, x):
        return self.fc(self.transformer(self.patch(x.unsqueeze(1))).flatten(1))


def build(name, n_ch, n_times, n_out):
    if name == 'conformer':
        return Conformer(n_ch, n_times, n_out)
    from braindecode import models as bd
    kw = dict(n_chans=n_ch, n_outputs=n_out, n_times=n_times, sfreq=200)
    if name == 'bd-conformer':
        return bd.EEGConformer(**kw)
    if name == 'bd-atcnet':
        return bd.ATCNet(**kw)
    if name == 'bd-deepsleepnet':
        return bd.DeepSleepNet(**kw)
    raise ValueError(name)


class Wrap(nn.Module):
    """(batch, channels, patches, 200) -> (batch, channels, time) -> the model; (batch, 1) -> (batch,) for binary tasks."""

    def __init__(self, m, n_ch):
        super().__init__()
        self.m, self.n_ch = m, n_ch

    def forward(self, x):
        out = self.m(x.reshape(x.shape[0], self.n_ch, -1))
        return out.squeeze(-1) if out.shape[-1] == 1 else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--downstream_dataset', required=True, choices=list(CFG))
    ap.add_argument('--datasets_dir', required=True)
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--model', required=True, choices=MODELS)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cuda', type=int, default=0)
    a = ap.parse_args()
    params = argparse.Namespace(   # finetune_main.py defaults, as run_eegnet.py and run_dgcnn.py
        seed=a.seed, cuda=a.cuda, epochs=50, batch_size=64, lr=1e-4, weight_decay=5e-2, optimizer='AdamW',
        clip_value=1, dropout=0.1, classifier='all_patch_reps', downstream_dataset=a.downstream_dataset,
        datasets_dir=a.datasets_dir, num_of_classes=0, model_dir=a.model_dir, num_workers=16,
        label_smoothing=0.1, multi_lr=True, frozen=False, use_pretrained_weights=False, foundation_dir='')
    print(params)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(a.cuda)
    mod, n_ch, n_times, n_out, task = CFG[a.downstream_dataset]
    params.num_of_classes = max(n_out, 2)
    loader = mod.LoadDataset(params).get_data_loader()
    model = Wrap(build(a.model, n_ch, n_times, n_out), n_ch)
    print(a.model, 'params:', sum(p.numel() for p in model.parameters()))
    t = Trainer(params, loader, model)          # no 'backbone' params -> whole model uses the head LR (5e-4)
    t.train_for_multiclass() if task == 'multi' else t.train_for_binaryclass()
    print('Done!!!!!')


if __name__ == '__main__':
    main()
