"""EEGNet-8,2 (braindecode) trained through CBraMod's *released* fine-tuning pipeline.

Same dataset loaders and splits, Trainer (loss, AdamW, cosine schedule, clipping, label smoothing, checkpoint on the
best validation kappa / AUROC) and Evaluator as CBraMod's released-code runs and as our own PyTorch EEGNet-8,2 there
(../../../cbramod/experiments/finetune-public-datasets/scripts/run_eegnet.py); only the model is braindecode's EEGNet.

usage: python run_eegnet.py --downstream_dataset X --datasets_dir D --seed S --cuda 0 --model_dir M
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from braindecode.models import EEGNet

sys.path.insert(0, os.environ.get('CBRAMOD_SRC', os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../../cbramod/upstream')))
from datasets import physio_dataset, bciciv2a_dataset, mumtaz_dataset, stress_dataset  # noqa: E402
from finetune_trainer import Trainer  # noqa: E402

# dataset -> (loader module, n_channels, n_times @200 Hz, n_outputs, task), as in the CBraMod experiments
CFG = {
    'PhysioNet-MI': (physio_dataset, 64, 800, 4, 'multi'),
    'BCIC-IV-2a': (bciciv2a_dataset, 22, 800, 4, 'multi'),
    'Mumtaz2016': (mumtaz_dataset, 19, 1000, 1, 'binary'),
    'MentalArithmetic': (stress_dataset, 20, 1000, 1, 'binary'),
}
# EEGNet-8,2 with the settings of our PyTorch EEGNet in the CBraMod experiment: temporal kernel 100 samples (0.5 s at
# 200 Hz, the paper's 64 at 128 Hz) and dropout 0.25 (Lawhern et al.'s cross-subject value, braindecode's default).
# Classifier: dense layer with max norm 0.25 (EEGNet paper, Table 2), as in the within-subject braindecode runs.
# Everything else is braindecode's default (separable kernel 16, average pooling 4 and 8, depthwise max norm 1).
EEGNET = dict(F1=8, D=2, kernel_length=100, drop_prob=0.25, final_layer_with_constraint=True)


class EEGNetForTrainer(nn.Module):
    """The released loaders give (batch, channels, patches, 200); EEGNet takes (batch, channels, time)."""

    def __init__(self, n_chans, n_times, n_outputs):
        super().__init__()
        self.net = EEGNet(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times, **EEGNET)

    def forward(self, x):
        b, c, s, p = x.shape
        out = self.net(x.reshape(b, c, s * p))
        return out.squeeze(-1) if out.shape[-1] == 1 else out    # binary: (batch,) logits, as the released heads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--downstream_dataset', required=True)
    ap.add_argument('--datasets_dir', required=True)
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cuda', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=50)           # 50 = released default; fewer only for smoke tests
    a = ap.parse_args()
    # identical to finetune_main.py defaults (= paper Table 6)
    params = argparse.Namespace(
        seed=a.seed, cuda=a.cuda, epochs=a.epochs, batch_size=64, lr=1e-4, weight_decay=5e-2, optimizer='AdamW',
        clip_value=1, dropout=0.1, classifier='braindecode EEGNet', downstream_dataset=a.downstream_dataset,
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
    model = EEGNetForTrainer(n_ch, n_times, n_out)
    print('EEGNet', EEGNET, 'params:', sum(p.numel() for p in model.parameters()))
    t = Trainer(params, loader, model)          # no 'backbone' params -> whole model uses the head LR (5e-4)
    t.train_for_multiclass() if task == 'multi' else t.train_for_binaryclass()
    print('Done!!!!!')


if __name__ == '__main__':
    main()
