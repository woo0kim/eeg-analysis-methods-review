"""CBraMod as implemented in braindecode, fine-tuned through CBraMod's *released* pipeline.

Same dataset loaders and splits, Trainer (loss, AdamW, multi_lr, cosine schedule, clipping, label smoothing,
checkpoint on the best validation kappa / AUROC) and Evaluator as the released-code runs in
../../finetune-public-datasets; only the model is swapped for braindecode's CBraMod, used the way braindecode
ships it:
  pretrained  CBraMod.from_pretrained('braindecode/cbramod-pretrained') at a pinned revision, with braindecode's
              own task head (the pre-training output projection `proj_out` is kept, then Flatten + one Linear)
  scratch     the same model from braindecode's random initialisation
  frozen      pretrained, with the backbone frozen by the released Trainer (--frozen)
The released Trainer gives parameters whose name contains 'backbone' the backbone learning rate (1e-4) and all
others the head learning rate (5e-4, multi_lr), and freezes the 'backbone' ones under --frozen. The wrapper below
puts braindecode's encoder under `backbone` and its final_layer under `head`, so that split is the released one.

usage: python run_braindecode.py --downstream_dataset X --datasets_dir D --setting S --seed N --cuda 0 --model_dir M
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from braindecode.models import CBraMod

sys.path.insert(0, os.environ.get('CBRAMOD_SRC', os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../upstream')))
from datasets import physio_dataset, bciciv2a_dataset, mumtaz_dataset, stress_dataset  # noqa: E402
from finetune_trainer import Trainer  # noqa: E402

HF_REPO = 'braindecode/cbramod-pretrained'
HF_REVISION = '584cdc415913739a05d84bf0c1cb3db397764507'   # 2026-01-12; same tensors as the released weights
# dataset -> (loader module, n_channels, n_times @200 Hz, n_outputs, task), as in ../../finetune-public-datasets
CFG = {
    'PhysioNet-MI': (physio_dataset, 64, 800, 4, 'multi'),
    'BCIC-IV-2a': (bciciv2a_dataset, 22, 800, 4, 'multi'),
    'Mumtaz2016': (mumtaz_dataset, 19, 1000, 1, 'binary'),
    'MentalArithmetic': (stress_dataset, 20, 1000, 1, 'binary'),
}


class BraindecodeCBraMod(nn.Module):
    """braindecode's CBraMod with the released Trainer's parameter split: encoder -> `backbone`, head -> `head`."""

    def __init__(self, n_chans, n_times, n_outputs, pretrained):
        super().__init__()
        kw = dict(n_chans=n_chans, n_times=n_times, n_outputs=n_outputs)
        m = CBraMod.from_pretrained(HF_REPO, revision=HF_REVISION, **kw) if pretrained else CBraMod(**kw)
        self.head = m.final_layer          # Flatten + Linear(n_chans * n_patches * 200, n_outputs), braindecode's init
        m.final_layer = nn.Identity()
        self.backbone = m

    def forward(self, x):                  # x: (batch, channels, patches, 200), from the released loaders
        b, c, s, p = x.shape
        out = self.head(self.backbone(x.reshape(b, c, s * p)))   # braindecode takes (batch, channels, time)
        return out.squeeze(-1) if out.shape[-1] == 1 else out    # binary: (batch,) logits, as the released heads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--downstream_dataset', required=True)
    ap.add_argument('--datasets_dir', required=True)
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--setting', choices=['pretrained', 'scratch', 'frozen'], required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cuda', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=50)           # 50 = released default; fewer only for smoke tests
    a = ap.parse_args()
    # finetune_main.py defaults (= paper Table 6), as in the released-code runs
    params = argparse.Namespace(
        seed=a.seed, cuda=a.cuda, epochs=a.epochs, batch_size=64, lr=1e-4, weight_decay=5e-2, optimizer='AdamW',
        clip_value=1, dropout=0.1, classifier='braindecode final_layer', downstream_dataset=a.downstream_dataset,
        datasets_dir=a.datasets_dir, num_of_classes=0, model_dir=a.model_dir, num_workers=16,
        label_smoothing=0.1, multi_lr=True, frozen=a.setting == 'frozen',
        use_pretrained_weights=a.setting != 'scratch', foundation_dir=f'{HF_REPO}@{HF_REVISION}')
    print(params)
    # finetune_main.setup_seed
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(a.cuda)

    mod, n_ch, n_times, n_out, task = CFG[a.downstream_dataset]
    params.num_of_classes = max(n_out, 2)
    loader = mod.LoadDataset(params).get_data_loader()
    model = BraindecodeCBraMod(n_ch, n_times, n_out, pretrained=params.use_pretrained_weights)
    n_bb = sum(p.numel() for p in model.backbone.parameters())
    print(f'braindecode CBraMod params: backbone {n_bb:,}, head {sum(p.numel() for p in model.head.parameters()):,}')
    t = Trainer(params, loader, model)
    t.train_for_multiclass() if task == 'multi' else t.train_for_binaryclass()
    print('Done!!!!!')


if __name__ == '__main__':
    main()
