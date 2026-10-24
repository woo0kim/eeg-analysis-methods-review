"""EEG Conformer as implemented in braindecode, trained by the released training loop on BCI IV 2a.

Imports the bciciv2a-epoch-selection experiment's code/run_validation.py (conformer.py's training loop, copied
verbatim, plus protocol B's validation split) and replaces only the model: ExP builds its network as `Conformer()`,
and that name is rebound here to braindecode's EEGConformer. Data, z-scoring, S&R augmentation, batch size,
optimizer, epochs, seeding and the per-epoch test / validation scoring are run_validation.py's, unchanged.
run_validation.py is GPL-3.0 (it copies conformer.py), and so is this file.

usage: python code/run_braindecode.py --subject 1 --seed 2023 --val-frac 0.0 --tag A --root data_proc/ --out results
       (every argument is run_validation.py's; --root needs the trailing slash)
"""
import os, sys
import torch.nn as nn
from braindecode.models import EEGConformer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get('EEGCONF_CODE', os.path.join(HERE, '../../bciciv2a-epoch-selection/code')))
import run_validation  # noqa: E402

# braindecode's defaults are the released model's Dataset I settings: 40 temporal filters of 25 samples, average
# pooling 75 / stride 15, depth 6, 10 heads, dropout 0.5, FC head 2440 -> 256 -> 32 -> 4. Nothing is overridden.
CONFORMER = dict(n_chans=22, n_outputs=4, n_times=1000)


class EEGConformerForExP(nn.Module):
    """ExP feeds (batch, 1, 22, 1000) and unpacks (features, logits); EEGConformer takes (batch, 22, 1000)."""

    def __init__(self):
        super().__init__()
        self.net = EEGConformer(**CONFORMER)
        print('EEGConformer', CONFORMER, 'params:', sum(p.numel() for p in self.net.parameters()), flush=True)

    def forward(self, x):
        logits = self.net(x.squeeze(1))
        return logits, logits


run_validation.Conformer = EEGConformerForExP

if __name__ == '__main__':
    run_validation.main()
