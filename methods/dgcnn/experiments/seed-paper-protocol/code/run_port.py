"""DGCNN on SEED in the paper's protocols, from SEED's distributed DE features (prepare_de.py), PyTorch port.

Training is eeg_DE.py's: SGD with momentum 0.9, lr 0.07 decayed by 0.95 every 3 epochs (staircase), batch 100, 20
epochs, L2 5e-4, per-band normalisation 2 (x - mean) / mean with training-set means, and the test set evaluated every 5
steps and at the end. That evaluation only records a curve here: nothing is selected during training.

protocols
  sd      subject-dependent, as eeg_DE.py: each of the 45 sessions alone, first 9 trials -> last 6
  sd-val  the same with a real validation set: trials 1-7 train, 8-9 validation, last 6 test
  loso    leave one subject out, one session index at a time (--sessions 0 1 2) or pooled (--sessions all): no script
          exists in the authors' archive, so the data used is a choice, and the training settings are eeg_DE.py's
          (eval every 5 steps, decay every 3 epochs). --norm zscore standardises each session on its own instead.

usage: python run_port.py --protocol sd --de_dir D/de_LDS --seed 0 --out R.npz
"""
import argparse
import collections
import os
import sys
import time

import numpy as np
import scipy.io as sio
import torch

sys.path.insert(0, os.environ.get('DGCNN_CODE', os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../code')))
from dgcnn import DGCNN  # noqa: E402

TRIAL_LEN = [235, 233, 206, 238, 185, 195, 237, 216, 265, 237, 235, 233, 235, 238, 206]   # s, SEED's 15 clips


def load(de_dir):
    y = sio.loadmat(os.path.join(de_dir, 'DE_labels.mat'))['de_labels'].ravel().astype(np.int64)
    X = [np.transpose(sio.loadmat(os.path.join(de_dir, f'DE_{i + 1}.mat'))['DE_feature'], (1, 0, 2)) for i in range(45)]
    return np.stack(X).astype(np.float32), y          # (45, 3394, 62, 5)


def authors_norm(tr, others):
    m = tr.mean(axis=(0, 1), keepdims=True)
    return [2 * (x - m) / m for x in [tr] + others]


def zscore(x):
    return (x - x.mean(axis=(0, 1), keepdims=True)) / x.std(axis=(0, 1), keepdims=True)


def fit(Xtr, ytr, evalsets, dev, seed, epochs=20, batch=100, lr0=0.07, decay_rate=0.95, every=5, reg=5e-4, n_classes=3):
    """Returns steps and, per evaluation set, the accuracy (%) at every evaluation."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DGCNN(Xtr.shape[1], Xtr.shape[2], n_classes).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=lr0, momentum=0.9)
    Xtr_t, ytr_t = torch.from_numpy(Xtr).to(dev), torch.from_numpy(ytr).to(dev)
    ev = {k: (torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)) for k, (x, y) in evalsets.items()}
    n = len(Xtr)
    num_steps = int(epochs * n / batch)
    decay_steps = 3 * n / batch
    idx = collections.deque()
    steps, curves = [], {k: [] for k in ev}
    for step in range(1, num_steps + 1):
        if len(idx) < batch:
            idx.extend(np.random.permutation(n).tolist())
        b = torch.tensor([idx.popleft() for _ in range(batch)], device=dev)
        for g in opt.param_groups:
            g['lr'] = lr0 * decay_rate ** ((step - 1) // decay_steps)
        model.train()
        loss = torch.nn.functional.cross_entropy(model(Xtr_t[b]), ytr_t[b]) + reg * model.l2()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % every == 0 or step == num_steps:
            model.eval()
            with torch.no_grad():
                for k, (x, y) in ev.items():
                    pred = torch.cat([model(x[i:i + 4096]).argmax(1) for i in range(0, len(x), 4096)])
                    curves[k].append(100 * (pred == y).float().mean().item())
            steps.append(step)
    return np.array(steps), {k: np.array(v, dtype=np.float32) for k, v in curves.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', required=True, choices=['sd', 'sd-val', 'loso'])
    ap.add_argument('--de_dir', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sessions', nargs='+', default=['0', '1', '2'])
    ap.add_argument('--norm', default='authors', choices=['authors', 'zscore'])
    ap.add_argument('--lr', type=float, default=0.07, help="eeg_DE.py's value; others are the learning-rate check")
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, y = load(a.de_dir)
    t0 = time.time()
    res = {}
    if a.protocol in ('sd', 'sd-val'):
        end7, end9 = sum(TRIAL_LEN[:7]), sum(TRIAL_LEN[:9])
        assert sum(TRIAL_LEN) == 3394 and end9 == 2010
        for s in range(45):
            if a.protocol == 'sd':
                tr, te = authors_norm(X[s, :end9], [X[s, end9:]])
                sets = {'test': (te, y[end9:])}
                steps, c = fit(tr, y[:end9], sets, dev, a.seed * 1000 + s, lr0=a.lr)
            else:
                tr, va, te = authors_norm(X[s, :end7], [X[s, end7:end9], X[s, end9:]])
                sets = {'val': (va, y[end7:end9]), 'test': (te, y[end9:])}
                steps, c = fit(tr, y[:end7], sets, dev, a.seed * 1000 + s, lr0=a.lr)
            res[f's{s}'] = np.stack([c[k] for k in sorted(c)])       # (sets, evals)
            print(s, f'last {c["test"][-1]:.2f} peak {c["test"].max():.2f}  {time.time() - t0:.0f}s', flush=True)
        res['sets'] = np.array(sorted(c))
        res['steps'] = steps
    else:
        groups = [[0, 1, 2]] if a.sessions == ['all'] else [[int(s)] for s in a.sessions]
        for g in groups:
            for sub in range(15):
                tr_s = [s * 3 + k for s in range(15) if s != sub for k in g]
                te_s = [sub * 3 + k for k in g]
                if a.norm == 'zscore':
                    Xtr = np.concatenate([zscore(X[[i]]) for i in tr_s])
                    Xte = np.concatenate([zscore(X[[i]]) for i in te_s])
                    tr, te = Xtr.reshape(-1, 62, 5), Xte.reshape(-1, 62, 5)
                else:
                    tr, te = authors_norm(X[tr_s].reshape(-1, 62, 5)[None], [X[te_s].reshape(-1, 62, 5)[None]])
                    tr, te = tr[0], te[0]
                ytr = np.tile(y, len(tr_s))
                steps, c = fit(tr, ytr, {'test': (te, np.tile(y, len(te_s)))}, dev, a.seed * 1000 + sub, lr0=a.lr)
                key = f'g{"".join(map(str, g))}_sub{sub}'
                res[key] = c['test'][None]
                res[f'steps_g{"".join(map(str, g))}'] = steps
                print(key, f'last {c["test"][-1]:.2f} peak {c["test"].max():.2f}  {time.time() - t0:.0f}s', flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **res)


if __name__ == '__main__':
    main()
