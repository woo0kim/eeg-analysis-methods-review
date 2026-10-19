"""EEGNet (braindecode) trained by EEG Conformer's released training loop on BCI IV 2a.

Imports the EEG Conformer experiment's code/run_validation.py (conformer.py's training loop, copied verbatim,
plus protocol B's validation split) and replaces only the model: ExP builds its network as `Conformer()`,
and that name is rebound here to braindecode's EEGNet. Data, z-scoring, S&R augmentation, batch size,
optimizer, epochs, seeding and the per-epoch test / validation scoring are run_validation.py's, unchanged.
run_validation.py is GPL-3.0 (it copies conformer.py), and so is this file.

usage: python code/run_eegnet.py --subject 1 --seed 2023 --val-frac 0.0 --tag A --root data_proc/ --out results
       (every argument is run_validation.py's; --root needs the trailing slash)
"""
import os, sys
import torch.nn as nn
from braindecode.models import EEGNet

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get('EEGCONF_CODE', os.path.join(
    HERE, '../../../../eeg-conformer/experiments/bciciv2a-epoch-selection/code')))
import run_validation  # noqa: E402

# EEGNet-8,2. braindecode defaults, except the two settings Lawhern et al. (2018) give for this case:
# dropout 0.5 for within-subject training (Sec. 2.2.1) and a dense classifier with max norm 0.25 (Table 2).
# kernel_length 64 at 250 Hz = 0.256 s, the paper's 32 samples at 128 Hz for the 4 Hz high-passed SMR data.
EEGNET = dict(n_chans=22, n_outputs=4, n_times=1000, F1=8, D=2, kernel_length=64, drop_prob=0.5,
              final_layer_with_constraint=True)


class EEGNetForExP(nn.Module):
    """ExP feeds (batch, 1, 22, 1000) and unpacks (features, logits); EEGNet takes (batch, 22, 1000)."""

    def __init__(self):
        super().__init__()
        self.net = EEGNet(**EEGNET)
        print('EEGNet', EEGNET, 'params:', sum(p.numel() for p in self.net.parameters()), flush=True)

    def forward(self, x):
        logits = self.net(x.squeeze(1))
        return logits, logits


run_validation.Conformer = EEGNetForExP

if __name__ == '__main__':
    run_validation.main()
