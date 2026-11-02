"""Do the braindecode and released CBraMod encoders of the Sleep-EDF runs differ only by proj_out?

Builds both encoders as the runs do (shared seq_models.py: CBraModEncoder, whose proj_out is Identity as in
model_for_isruc.py, and BDCBraModEncoder, whose return_features output comes after proj_out), each from its own copy
of the pre-trained weights, and compares braindecode's features with the released features passed through
braindecode's proj_out, for random 30-s epochs in float64, eval mode.

usage: python code/check_equivalence.py [> results/equivalence.txt]
       (torch, braindecode 1.8.1, huggingface_hub; CBraMod's released code and weights as in seq_models.py)
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '../../finetune-sleepedf/code')))
import seq_models  # noqa: E402

torch.manual_seed(0)
rel, bd = seq_models.CBraModEncoder(2, pretrained=True), seq_models.BDCBraModEncoder(2, pretrained=True)
w_rel = torch.load(seq_models.CBRAMOD_WEIGHTS, map_location='cpu')
same = all(torch.equal(w_rel['proj_out.' + k], v) for k, v in bd.m.proj_out.state_dict().items())
print('braindecode proj_out = released pretrained proj_out:', same)
n_rel, n_bd = (sum(p.numel() for p in m.parameters()) for m in (rel, bd))
print(f'parameters: released (proj_out removed) {n_rel:,}, braindecode {n_bd:,}, difference {n_bd - n_rel:,}')

rel, bd = rel.double().eval(), bd.double().eval()
x = torch.randn(8, 2, seq_models.EPOCH, dtype=torch.float64)        # 8 epochs, 2 channels, 30 s at 200 Hz
with torch.no_grad():
    fr, fb = rel(x), bd(x)
    fr_proj = bd.m.proj_out(fr.view(8, 2, 30, 200)).flatten(1)
print(f'features: {tuple(fr.shape)} and {tuple(fb.shape)}; max |difference| {(fr - fb).abs().max().item():.3g} '
      f'as is, {(fr_proj - fb).abs().max().item():.3g} with proj_out applied to the released features')
