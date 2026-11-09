"""EEG Conformer on SEED: every number of this experiment from the per-epoch test curves.

usage: python analyze.py <runs/conformer dir> <results dir> > <results dir>/results.txt

kfold: the released protocol. Session 1 of 15 subjects, 5 folds inside each clip (adjacent 1-s windows of a clip land in
train and test), 600 epochs, test accuracy after every epoch. The released result is best accuracy over the epochs,
averaged over folds and subjects. trial96: the same model, session and training, with whole clips held out (first 9
train, last 6 test), no adjacent windows across the split.
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats

runs, out = sys.argv[1:3]
os.makedirs(os.path.join(out, 'curves'), exist_ok=True)
PAPER = 95.30


def load(split):
    r = {}
    for f in glob.glob(os.path.join(runs, f'{split}_s*', 'curves', f'S*_{split}.npy')):
        m = re.search(r'S(\d+)_f(\d+)_', os.path.basename(f))
        seed_dir = os.path.basename(os.path.dirname(os.path.dirname(f)))
        c = np.load(f)
        r[(int(m.group(1)), int(m.group(2)), seed_dir)] = c
        np.save(os.path.join(out, 'curves', f'{seed_dir}_{os.path.basename(f)}'), np.round(c, 4))
    return r


RULES = {'best epoch on test (released)': lambda c: c.max(), 'mean of all epochs (released second number)': lambda c: c.mean(),
         'mean of the last 100 epochs': lambda c: c[-100:].mean(), 'final epoch': lambda c: c[-1]}
res = {}
for split, label in (('kfold', 'released 5-fold inside clips'), ('trial96', 'whole clips held out')):
    r = load(split)
    if not r:
        continue
    subs = sorted({k[0] for k in r})
    print(f'## {label}: {len(r)} runs, {len(subs)} subjects\n')
    print('| Rule | mean accuracy % | SD over subjects |\n|---|---|---|')
    res[split] = {'n_runs': len(r), 'n_subjects': len(subs)}
    for name, fn in RULES.items():
        per = np.array([np.mean([fn(c) for k, c in r.items() if k[0] == s]) for s in subs]) * 100
        print(f'| {name} | {per.mean():.2f} | {per.std(ddof=1):.2f} |')
        res[split][name] = dict(mean=round(float(per.mean()), 2), sd=round(float(per.std(ddof=1)), 2), per_subject=[round(float(x), 2) for x in per])
    print()
if len(res) == 2:
    print('## 5-fold inside clips − whole clips held out, same subjects, Wilcoxon signed-rank\n')
    print('| Rule | difference (pp) | p |\n|---|---|---|')
    for name in RULES:
        a, b = np.array(res['kfold'][name]['per_subject']), np.array(res['trial96'][name]['per_subject'])
        if len(a) == len(b):
            p = stats.wilcoxon(a, b).pvalue
            print(f'| {name} | {a.mean() - b.mean():+.2f} | {p:.4f} |')
            res.setdefault('difference', {})[name] = dict(diff=round(float(a.mean() - b.mean()), 2), p=float(p))
res['paper'] = PAPER
json.dump(res, open(os.path.join(out, 'results.json'), 'w'), indent=1)
