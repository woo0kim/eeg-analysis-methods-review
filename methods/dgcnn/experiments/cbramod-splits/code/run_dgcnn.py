"""DGCNN through CBraMod's released fine-tuning pipeline on its four public datasets (same LMDBs and subject splits, Trainer,
loss, optimizer, schedule, model selection and Evaluator as every other model run on these splits). Only the model
differs, as for EEGNet in methods/eegnet/experiments/pytorch-cbramod-splits.

  dgcnn        our DGCNN on 5-band DE of consecutive 1-s segments: node features = 5 x seconds
  bd-dgcnn     braindecode's DGCNN as it ships: raw samples as node features
  bd-dgcnn-de  braindecode's DGCNN on the same DE input as dgcnn

usage: python run_dgcnn.py --downstream_dataset X --datasets_dir D --model dgcnn --seed S --model_dir M
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get('DGCNN_CODE', os.path.join(HERE, '../../../code')))
from dgcnn import DGCNN, WithDE, chs_info  # noqa: E402
sys.path.insert(0, os.environ.get('CBRAMOD_SRC', os.path.join(HERE, '../../../../cbramod/upstream')))
from datasets import physio_dataset, bciciv2a_dataset, mumtaz_dataset, stress_dataset  # noqa: E402
from finetune_trainer import Trainer  # noqa: E402

# dataset -> (loader module, n_channels, n_times @200 Hz, n_outputs, task), as run_eegnet.py
CFG = {
    'PhysioNet-MI': (physio_dataset, 64, 800, 4, 'multi'),
    'BCIC-IV-2a': (bciciv2a_dataset, 22, 800, 4, 'multi'),
    'Mumtaz2016': (mumtaz_dataset, 19, 1000, 1, 'binary'),
    'MentalArithmetic': (stress_dataset, 20, 1000, 1, 'binary'),
}
# channel order of each LMDB, from methods/labram/experiments/finetune-cbramod-splits/code/run_labram.py; for
# MentalArithmetic the 20th channel is the ear derivation A2-A1, which LaBraM drops and DGCNN keeps as a node
CH = {
    'PhysioNet-MI': ['FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6',
                     'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'FP1', 'FPZ', 'FP2', 'AF7', 'AF3', 'AFZ',
                     'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FT8', 'T7', 'T8',
                     'T9', 'T10', 'TP7', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7',
                     'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2', 'IZ'],
    'BCIC-IV-2a': ['FZ', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'CP3',
                   'CP1', 'CPZ', 'CP2', 'CP4', 'P1', 'PZ', 'P2', 'POZ'],
    'Mumtaz2016': ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5',
                   'T6', 'FZ', 'CZ', 'PZ'],
    'MentalArithmetic': ['FP1', 'FP2', 'F3', 'F4', 'F7', 'F8', 'T3', 'T4', 'C3', 'C4', 'T5', 'T6', 'P3', 'P4',
                         'O1', 'O2', 'FZ', 'CZ', 'PZ', 'A2'],
}


class Wrap(nn.Module):
    """(batch, channels, patches, 200) -> the model; binary tasks give (batch,) logits, as the released heads."""

    def __init__(self, m, n_ch):
        super().__init__()
        self.m, self.n_ch = m, n_ch

    def forward(self, x):
        out = self.m(x.reshape(x.shape[0], self.n_ch, -1))
        return out.squeeze(-1) if out.shape[-1] == 1 else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--downstream_dataset', required=True)
    ap.add_argument('--datasets_dir', required=True)
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--model', required=True, choices=['dgcnn', 'bd-dgcnn', 'bd-dgcnn-de'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cuda', type=int, default=0)
    a = ap.parse_args()
    params = argparse.Namespace(   # finetune_main.py defaults, as run_eegnet.py
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
    sec = n_times // 200
    if a.model == 'dgcnn':
        m = WithDE(DGCNN(n_ch, 5 * sec, n_out), n_ch, 200, sec)
    else:
        from braindecode.models import DGCNN as BD
        info = chs_info(CH[a.downstream_dataset])
        m = (BD(n_chans=n_ch, n_outputs=n_out, n_times=n_times, chs_info=info) if a.model == 'bd-dgcnn'
             else WithDE(BD(n_chans=n_ch, n_outputs=n_out, n_times=5 * sec, chs_info=info), n_ch, 200, sec))
    if hasattr(m, 'fit_norm'):
        m.fit_norm(loader['train'], n_batches=100)
    model = Wrap(m, n_ch)
    print(a.model, 'params:', sum(p.numel() for p in model.parameters()))
    t = Trainer(params, loader, model)          # no 'backbone' params -> whole model uses the head LR (5e-4)
    t.train_for_multiclass() if task == 'multi' else t.train_for_binaryclass()
    print('Done!!!!!')


if __name__ == '__main__':
    main()
