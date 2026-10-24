"""Step 2 of the equivalence check (braindecode env): load the authors' trained weights into braindecode's ATCNet and
compare its logits with the Keras model's, subject by subject, on session E (288 trials each).

Reads $ATCNET_WORK/braindecode/equivalence/{released,paper}_s<n>.npz (code/export_keras.py): today's released
weights (run-1, main_TrainValTest.py) and the 2022 best-of-10 models behind the paper's 85.38%. braindecode's ATCNet
is built with
its defaults for BCI IV 2a (n_chans=22, n_outputs=4, n_times=1125); the two places where its forward pass can differ
from the Keras one are then switched to the Keras behaviour one at a time:
  1. BatchNorm eps: Keras 1e-3; braindecode 1e-4 in the conv block and 1e-5 (torch default) in the TCN,
  2. the max-norm (0.25) braindecode puts on the final linear layers (the Keras Dense layers have none); it rescales,
     per input feature, the column of weights over the 4 outputs whose L2 norm exceeds 0.25.
Prints the table of the README and writes results/equivalence.txt.
usage: python code/check_equivalence.py [RESULTS_DIR]
"""
import contextlib, io, os, sys
import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize
from braindecode.models import ATCNet

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.expanduser(os.environ.get('ATCNET_WORK', '~/repro-atcnet'))
EQ = os.path.join(WORK, 'braindecode', 'equivalence')
RESULTS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '../results')
torch.set_default_dtype(torch.float64)
out = io.StringIO()


def say(*a):
    print(*a)
    print(*a, file=out)


def build(sd, keras_eps=False, drop_final_max_norm=False):
    m = ATCNet(n_chans=22, n_outputs=4, n_times=1125)
    state = {k: torch.as_tensor(v) for k, v in sd.items()}
    for k in [k for k in state if k.startswith('final_layer.') and k.endswith('.weight')]:
        state[k.replace('.weight', '.parametrizations.weight.original')] = state.pop(k)
    for k, v in m.state_dict().items():           # BatchNorm counters are not in the Keras file
        if k.endswith('num_batches_tracked'):
            state[k] = v
    missing, unexpected = m.load_state_dict(state, strict=True)
    assert not missing and not unexpected
    if keras_eps:
        for mod in m.modules():
            if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                mod.eps = 1e-3
    if drop_final_max_norm:
        for lin in m.final_layer:
            parametrize.remove_parametrizations(lin, 'weight', leave_parametrized=False)
    return m.eval()


cols = [('braindecode as is', {}), ('+ Keras BatchNorm eps', {'keras_eps': True}),
        ('+ no max-norm on the final layer', {'keras_eps': True, 'drop_final_max_norm': True})]
TITLE = {'released': 'today\'s released weights (upstream results/saved models/run-1, main_TrainValTest.py)',
         'paper': 'the paper\'s best-of-10 models (upstream commit 3a318b2, results/best models.txt)'}


def compare(weights):
    say(f'\nbraindecode ATCNet (1.8.1) with {TITLE[weights]}, session E, float64\n')
    say('| Subject | Keras acc. | ' + ' | '.join(f'{c}: max abs logit diff / acc. / same predictions' for c, _ in cols) +
        ' | final-layer weight columns with norm > 0.25 |')
    say('|---|---|' + '---|' * (len(cols) + 1))
    tot = {c: [] for c, _ in cols}
    keras_acc, clipped_all = [], []
    for s in range(1, 10):
        d = np.load(os.path.join(EQ, f'{weights}_s{s}.npz'))
        sd = {k[3:]: d[k] for k in d.files if k.startswith('sd/')}
        X = torch.as_tensor(d['X_test'][:, 0].astype(np.float64))          # (288, 22, 1125)
        y, kl = d['labels'], d['logits'].astype(np.float64)
        keras_acc.append(float((kl.argmax(1) == y).mean()))
        norms = np.concatenate([np.linalg.norm(sd[f'final_layer.{w}.weight'], axis=0) for w in range(5)])
        clipped_all.append(int((norms > 0.25).sum()))
        cells = []
        for c, kw in cols:
            with torch.no_grad(), contextlib.redirect_stderr(io.StringIO()):
                bl = build(sd, **kw)(X).numpy()
            diff, acc = float(np.abs(bl - kl).max()), float((bl.argmax(1) == y).mean())
            agree = int((bl.argmax(1) == kl.argmax(1)).sum())
            tot[c].append((diff, acc, agree))
            cells.append(f'{diff:.2e} / {acc * 100:.2f} / {agree}/288')
        say(f'| {s} | {keras_acc[-1] * 100:.2f} | ' + ' | '.join(cells) + f' | {clipped_all[-1]} / 160 |')
    say(f'| **Avg** | **{np.mean(keras_acc) * 100:.2f}** | ' + ' | '.join(
        f'max {max(t[0] for t in tot[c]):.2e} / **{np.mean([t[1] for t in tot[c]]) * 100:.2f}** / '
        f'{sum(t[2] for t in tot[c])}/2592' for c, _ in cols) + f' | {sum(clipped_all)} / 1440 |')


for weights in ('released', 'paper'):
    compare(weights)

m = ATCNet(n_chans=22, n_outputs=4, n_times=1125)
n_bd = sum(p.numel() for p in m.parameters())
d = np.load(os.path.join(EQ, 'released_s1.npz'))
say(f'\nParameters: braindecode {n_bd:,} trainable; Keras {int(d["keras_trainable"]):,} trainable '
    f'({int(d["keras_params"]):,} incl. BatchNorm moving statistics). Same layer shapes: '
    f'{len([k for k in d.files if k.startswith("sd/")])} weight tensors mapped one to one.')
open(os.path.join(RESULTS, 'equivalence.txt'), 'w').write(out.getvalue())
print('wrote', os.path.join(RESULTS, 'equivalence.txt'))
