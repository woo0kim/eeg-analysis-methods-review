"""Our PyTorch DGCNN against braindecode 1.8.1's DGCNN (methods/eegnet/upstream): same weights -> same outputs?

braindecode's model cites the same official archive. Weights are copied from a braindecode model into ours (the
Chebyshev projection is reordered: ours concatenates per feature, braindecode per polynomial) and both run in eval mode.
"""
import os
import sys

import torch

sys.path.insert(0, os.environ.get('DGCNN_CODE', os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../code')))
from dgcnn import DGCNN  # noqa: E402
from braindecode.models import DGCNN as BD  # noqa: E402
import mne  # noqa: E402

CH = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1',
      'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1',
      'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ',
      'PO4', 'PO6', 'PO8', 'I1', 'O1', 'OZ', 'O2', 'I2']       # CB1 / CB2 have no entry in MNE's montage: I1 / I2 stand in
torch.manual_seed(0)
info = mne.create_info([c.replace('FP', 'Fp').replace('Z', 'z') for c in CH], 200, 'eeg')
info.set_montage('standard_1005', match_case=False)
for K in (2, 3, 4):
    bd = BD(n_outputs=3, n_chans=62, n_times=5, chs_info=info['chs'], cheb_order=K).eval()
    m = DGCNN(62, 5, 3, K=K).eval()
    with torch.no_grad():
        m.A[:] = bd.learnable_adjacency.adjacency
        m.b_A[:] = bd.learnable_adjacency.bias
        w = bd.graph_conv.projection.weight.T                 # (5 * K, 64), polynomial first
        order = [k * 5 + f for f in range(5) for k in range(K)]   # ours: feature first
        m.G[:] = w[order]
        m.c[:] = bd.graph_conv_bias
        m.fc_w[:] = bd.classifier[0].weight.T
        m.fc_b[:] = bd.classifier[0].bias
        m.out_w[:] = bd.final_layer.weight.T
        m.out_b[:] = bd.final_layer.bias
        x = torch.randn(32, 62, 5)
        a, b = m(x), bd(x)
    print(f'K={K}: parameters ours {sum(p.numel() for p in m.parameters()):,}, braindecode '
          f'{sum(p.numel() for p in bd.parameters()):,}; max |output difference| {(a - b).abs().max().item():.2e}, '
          f'same predictions: {bool((a.argmax(1) == b.argmax(1)).all())}')
