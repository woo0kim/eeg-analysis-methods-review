"""Is braindecode's EEGConformer the released model? Copy the released model's weights into it and compare outputs.

1. Build the released Conformer (verbatim in ../../bciciv2a-epoch-selection/code/run_validation.py) and braindecode's
   EEGConformer with the settings run_braindecode.py uses, and compare their parameters tensor by tensor.
2. Copy the released model's random-init weights into braindecode's (same tensors in the same order; the released
   head's unused `clshead` branch has no counterpart) and compare the logits of both models in eval mode, float64.
3. Replace braindecode's attention forward with the released formula (softmax(QK^T / sqrt(emb_size)), instead of
   scaled_dot_product_attention's sqrt(head_dim)) and compare again: if the logits now agree to rounding error, the
   attention scaling is the only difference between the two forward passes.
CPU only. usage: python code/check_equivalence.py > results/equivalence.txt
"""
import math, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
import braindecode
from braindecode.models import EEGConformer
from braindecode.modules import MultiHeadAttention

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get('EEGCONF_CODE', os.path.join(HERE, '../../bciciv2a-epoch-selection/code')))
import run_validation  # noqa: E402
ReleasedConformer = run_validation.Conformer          # run_braindecode rebinds run_validation.Conformer on import
from run_braindecode import CONFORMER  # noqa: E402

torch.manual_seed(0)
released = ReleasedConformer().double().eval()
bd = EEGConformer(**CONFORMER).double().eval()
n_rel = sum(p.numel() for p in released.parameters())
n_unused = sum(p.numel() for n, p in released.named_parameters() if '.clshead.' in n)
n_bd = sum(p.numel() for p in bd.parameters())
print(f'torch {torch.__version__}, braindecode {braindecode.__version__}')
print(f'parameters: released {n_rel:,} (of which {n_unused:,} in the unused clshead branch), '
      f'braindecode {n_bd:,}; released minus clshead = {n_rel - n_unused:,}')

src = [(k, v) for k, v in released.state_dict().items() if '.clshead.' not in k]
dst = list(bd.state_dict().items())
assert len(src) == len(dst), (len(src), len(dst))
print('\nreleased tensor                     braindecode tensor                          shape')
for (ks, vs), (kd, vd) in zip(src, dst):
    assert vs.shape == vd.shape, (ks, kd, vs.shape, vd.shape)
    print(f'{ks:35s} {kd:43s} {tuple(vs.shape)}')
bd.load_state_dict({kd: vs for (_, vs), (kd, _) in zip(src, dst)})

x = torch.randn(64, 1, 22, 1000, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
with torch.no_grad():
    ref = released(x)[1]
    out = bd(x.squeeze(1))


def report(label, a, b):
    print(f'{label}: max |logit difference| {(a - b).abs().max().item():.3g}, '
          f'same predicted class for {(a.argmax(1) == b.argmax(1)).sum().item()} / {len(a)} inputs')


print(f'\nSame weights, 64 standard-normal inputs (the training data are z-scored), eval mode, float64')
report('braindecode as released (scaled_dot_product_attention, scale 1/sqrt(head_dim) = 1/2)', ref, out)


def released_attention(self, x, mask=None):
    """MultiHeadAttention.forward of the released conformer.py (attention dropout is inactive in eval mode)."""
    q = rearrange(self.queries(x), 'b n (h d) -> b h n d', h=self.num_heads)
    k = rearrange(self.keys(x), 'b n (h d) -> b h n d', h=self.num_heads)
    v = rearrange(self.values(x), 'b n (h d) -> b h n d', h=self.num_heads)
    att = F.softmax(torch.einsum('bhqd, bhkd -> bhqk', q, k) / self.emb_size ** 0.5, dim=-1)
    return self.projection(rearrange(torch.einsum('bhal, bhlv -> bhav ', att, v), 'b h n d -> b n (h d)'))


MultiHeadAttention.forward = released_attention
with torch.no_grad():
    out2 = bd(x.squeeze(1))
report(f'braindecode with the released attention scale 1/sqrt(emb_size) = 1/sqrt(40) = {1 / math.sqrt(40):.4f}',
       ref, out2)
