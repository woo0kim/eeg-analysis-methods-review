"""Build one model, report its parameter count, and find the largest micro-batch (in 20-epoch sequences) whose
training step fits on the GPU, with the time per step. Used to choose --accum per model (add_jobs.py).

usage: python smoke_test.py <model name>
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seq_models  # noqa: E402

name = sys.argv[1]
torch.manual_seed(0)
model = seq_models.build(name).cuda()
enc = getattr(model, model.enc_name)
n_enc = sum(p.numel() for p in enc.parameters())
n_all = sum(p.numel() for p in model.parameters())
print(f'{name}: encoder ({model.enc_name}) {n_enc:,} parameters, total {n_all:,}, feat_dim {enc.feat_dim}')
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
best = None
for mb in (1, 2, 4, 8, 16, 32, 64):       # upwards: a failed allocation can trip the server's NVML assert
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(mb, 20, 2, 6000, device='cuda')
    y = torch.randint(0, 5, (mb, 20), device='cuda')
    ts = []
    for it in range(4):
        torch.cuda.synchronize()
        t0 = time.time()
        loss = torch.nn.functional.cross_entropy(model(x).transpose(1, 2), y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        torch.cuda.synchronize()
        ts.append(time.time() - t0)
    del loss
    peak = torch.cuda.max_memory_allocated() / 2**30
    per_seq = min(ts[1:]) / mb
    print(f'  micro-batch {mb:2d}: peak {peak:.1f} GB, {min(ts[1:]):.3f} s/step, '
          f'~{per_seq * 7800 / 60:.1f} min per training epoch (7,800 sequences)', flush=True)
    if peak > 5.0:                          # the next doubling would exceed ~10 GB of the 12 GB card
        best = (mb, peak, per_seq)
        break
    best = (mb, peak, per_seq)
mb, peak, per_seq = best
print(f'RESULT {name} microbatch={mb} accum={64 // mb} peak_gb={peak:.1f} '
      f'train_min_per_epoch={per_seq * 7800 / 60:.1f} params={n_all}')
