"""DGCNN trained by EEG Conformer's released training loop on BCI IV 2a, as methods/eegnet/experiments/bciciv2a-epoch-selection
does for EEGNet: run_validation.py's data, z-scoring, S&R augmentation, batch size, optimizer, epochs, seeds and per-epoch
test / validation scoring are unchanged; only the model name it builds is rebound. run_validation.py is GPL-3.0 (it copies
conformer.py), and so is this file.

Input: 5-band DE of the 4 consecutive 1-s segments of a 4-s trial at 250 Hz (20 values per channel), standardised per
band with the first training batch (that loop has no hook for a pass over the data).

usage: as run_validation.py: python code/run_dgcnn.py --subject 1 --seed 2023 --val-frac 0.0 --tag A --root data_proc/ --out results
"""
import os
import sys

import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get('DGCNN_CODE', os.path.join(HERE, '../../../code')))
from dgcnn import DGCNN, WithDE  # noqa: E402
sys.path.insert(0, os.environ.get('EEGCONF_CODE', os.path.join(
    HERE, '../../../../eeg-conformer/experiments/bciciv2a-epoch-selection/code')))
import run_validation  # noqa: E402


class DGCNNForExP(nn.Module):
    """ExP feeds (batch, 1, 22, 1000) and unpacks (features, logits)."""

    def __init__(self):
        super().__init__()
        self.net = WithDE(DGCNN(22, 20, 4), 22, 250, 4, lazy=True)
        print('DGCNN params:', sum(p.numel() for p in self.net.parameters()), flush=True)

    def forward(self, x):
        logits = self.net(x.squeeze(1))
        return logits, logits


run_validation.Conformer = DGCNNForExP

if __name__ == '__main__':
    run_validation.main()
