"""Build every model of run_models.py for every dataset, count parameters and run one forward and backward pass on a batch of
64 on the GPU (peak memory). No data needed. usage: CUDA_VISIBLE_DEVICES=0 python smoke_test.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_models import CFG, MODELS, Wrap, build  # noqa: E402

for ds, (_, n_ch, n_times, n_out, _) in CFG.items():
    for name in MODELS:
        torch.cuda.reset_peak_memory_stats()
        m = Wrap(build(name, n_ch, n_times, n_out), n_ch).cuda()
        x = torch.randn(64, n_ch, n_times // 200, 200, device='cuda')
        out = m(x)
        out.float().sum().backward()
        print(f'{ds:17s} {name:16s} params {sum(p.numel() for p in m.parameters()):>10,d}  out {tuple(out.shape)}  '
              f'peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB')
        del m, x, out
