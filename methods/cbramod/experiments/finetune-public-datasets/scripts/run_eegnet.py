"""Supplementary sanity baseline: EEGNet-8,2 (Lawhern et al., 2018) trained through CBraMod's *released*
fine-tuning pipeline (same dataset loaders/splits, Trainer, loss, optimizer, schedule, model selection and
Evaluator) so that it is directly comparable with the CBraMod runs. Only the model is swapped.

usage: python run_eegnet.py --downstream_dataset X --datasets_dir D --seed S --cuda 0 --model_dir M
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.environ.get('CBRAMOD_SRC', os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../upstream')))
from datasets import physio_dataset, bciciv2a_dataset, mumtaz_dataset, stress_dataset  # noqa: E402
from finetune_trainer import Trainer  # noqa: E402

# dataset -> (loader module, n_channels, n_times @200 Hz, n_outputs, task)
CFG = {
    'PhysioNet-MI': (physio_dataset, 64, 800, 4, 'multi'),
    'BCIC-IV-2a': (bciciv2a_dataset, 22, 800, 4, 'multi'),
    'Mumtaz2016': (mumtaz_dataset, 19, 1000, 1, 'binary'),
    'MentalArithmetic': (stress_dataset, 20, 1000, 1, 'binary'),
}


class EEGNet(nn.Module):
    """EEGNet-8,2; temporal kernel = 0.5 s (100 samples at 200 Hz, as 64 at 128 Hz in the original)."""

    def __init__(self, n_ch, n_times, n_out, F1=8, D=2, F2=16, kern=100, drop=0.25):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern), padding=(0, kern // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),   # depthwise spatial filter
            nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(drop))
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),  # separable conv
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(drop))
        with torch.no_grad():
            n_feat = self.block2(self.block1(torch.zeros(1, 1, n_ch, n_times))).numel()
        self.classifier = nn.Linear(n_feat, n_out)

    def forward(self, x):                      # x: (batch, channels, patches, 200)
        b, c, s, p = x.shape
        x = self.block2(self.block1(x.reshape(b, 1, c, s * p))).flatten(1)
        out = self.classifier(x)
        return out.squeeze(-1) if out.shape[-1] == 1 else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--downstream_dataset', required=True)
    ap.add_argument('--datasets_dir', required=True)
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cuda', type=int, default=0)
    a = ap.parse_args()
    # identical to finetune_main.py defaults (= paper Table 6)
    params = argparse.Namespace(
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
    model = EEGNet(n_ch, n_times, n_out)
    print('EEGNet params:', sum(p.numel() for p in model.parameters()))
    t = Trainer(params, loader, model)          # no 'backbone' params -> whole model uses the head LR (5e-4)
    t.train_for_multiclass() if task == 'multi' else t.train_for_binaryclass()
    print('Done!!!!!')


if __name__ == '__main__':
    main()
