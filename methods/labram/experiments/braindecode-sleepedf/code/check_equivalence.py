"""Is the braindecode LaBraM encoder of the Sleep-EDF runs the released LaBraM encoder?

Builds both encoders as the runs do (shared seq_models.py: LaBraMEncoder and BDLabramEncoder, each with its own
copy of the pre-trained weights), copies the released encoder's freshly initialised fc_norm into braindecode's
(the one layer neither checkpoint holds), and compares the features of random 30-s epochs in float64, eval mode.

usage: python code/check_equivalence.py [> results/equivalence.txt]
       (torch, timm 0.4.12, braindecode 1.8.1, huggingface_hub; LaBraM's released code and labram-base.pth as in
       seq_models.py)
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '../../../../cbramod/experiments/finetune-sleepedf/code')))
import seq_models  # noqa: E402

torch.manual_seed(0)
rel, bd = seq_models.LaBraMEncoder(), seq_models.BDLabramEncoder()
print('channels', seq_models.CH_LABRAM, '-> position rows: released', rel.input_chans,
      'braindecode', bd.input_chans.tolist())
bd.m.fc_norm.load_state_dict(rel.m.fc_norm.state_dict())
n_rel, n_bd = (sum(p.numel() for p in m.parameters()) for m in (rel, bd))
print(f'parameters: released {n_rel:,}, braindecode {n_bd:,}')

rel, bd = rel.double().eval(), bd.double().eval()
x = torch.randn(16, 2, seq_models.EPOCH, dtype=torch.float64)       # 16 epochs, 2 channels, 30 s at 200 Hz
with torch.no_grad():
    fr, fb = rel(x), bd(x)
print(f'features: {tuple(fr.shape)} and {tuple(fb.shape)}, max |difference| {(fr - fb).abs().max().item():.3g}')
