"""Forward + backward of a batch of 64 per model on one GPU: parameter counts and peak memory, to set --accum."""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emo_models  # noqa: E402
sys.path.insert(0, emo_models.CBRAMOD_SRC)

NAMES = ['cbramod', 'cbramod-scratch', 'eegnet', 'conformer', 'dgcnn', 'bd-cbramod', 'bd-eegnet', 'bd-conformer', 'bd-atcnet',
         'bd-deepsleepnet', 'bd-dgcnn', 'bd-dgcnn-de']
for name in sys.argv[1:] or NAMES:
    try:
        p = argparse.Namespace(num_of_classes=3, cuda=0, classifier='all_patch_reps', dropout=0.1)
        m = emo_models.build(name, p).cuda().train()
        x = torch.randn(64, 62, 1, 200).cuda()
        torch.cuda.reset_peak_memory_stats()
        out = m(x)
        torch.nn.functional.cross_entropy(out, torch.zeros(64).long().cuda()).backward()
        print(f'{name:18s} out {tuple(out.shape)} params {sum(q.numel() for q in m.parameters()):>10,} '
              f'peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB', flush=True)
        del m, x, out
        torch.cuda.empty_cache()
    except Exception as e:
        print(f'{name:18s} FAILED {e!r}'[:300], flush=True)
