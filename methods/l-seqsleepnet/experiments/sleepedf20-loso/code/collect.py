"""L-SeqSleepNet on Sleep-EDF-20 (LOSO, released code and data): metrics per repetition vs. the paper.

usage: python collect.py <preds dir> <out dir> > <out dir>/results.txt
  <preds dir>/repeat<r>_n<n>.npz   per-fold predictions written by export_preds.py
writes <out dir>/results.json. Metrics as the released evaluate.py computes them on the 20 folds' pooled predictions
of one repetition: accuracy, Cohen's kappa, per-class F1 and their mean (MF1), macro sensitivity and specificity.
The paper reports the mean +- SD over 5 repetitions; a 95% bootstrap interval over subjects is added per repetition.
"""
import glob
import json
import os
import re
import sys

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, recall_score

STAGES = ['W', 'N1', 'N2', 'N3', 'REM']
# L-SeqSleepNet paper (Phan et al., IEEE JBHI 2023), Table III, SleepEDF (= Sleep-EDF-20), random initialisation:
# mean +- SD over 5 repetitions of LOSO CV
PAPER = {'accuracy': (0.863, 0.002), 'kappa': (0.813, 0.003), 'macro_f1': (0.793, 0.004),
         'sensitivity': (0.788, 0.005), 'specificity': (0.963, 0.001),
         'f1': [(0.916, 0.004), (0.453, 0.014), (0.885, 0.003), (0.862, 0.007), (0.852, 0.002)]}


def specificity_macro(y, p):
    cm = confusion_matrix(y, p, labels=range(5))
    tn = cm.sum() - cm.sum(0) - cm.sum(1) + np.diag(cm)
    return float(np.mean(tn / (tn + cm.sum(0) - np.diag(cm))))   # = imblearn specificity_score(average='macro')


def metrics(y, p):
    f1 = f1_score(y, p, average=None, labels=range(5))
    return {'accuracy': accuracy_score(y, p), 'kappa': cohen_kappa_score(y, p, labels=range(5)),
            'macro_f1': float(np.mean(f1)), 'f1': f1.tolist(),
            'sensitivity': recall_score(y, p, average='macro', labels=range(5)), 'specificity': specificity_macro(y, p),
            'cm': confusion_matrix(y, p, labels=range(5)).tolist(), 'n_epochs': int(len(y))}


def main():
    pdir, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    folds = {}
    for f in glob.glob(os.path.join(pdir, 'repeat*_n*.npz')):
        r, n = map(int, re.findall(r'repeat(\d+)_n(\d+)', f)[0])
        d = np.load(f)
        folds.setdefault(r, {})[n] = (d['y_true'].astype(int), d['y_pred'].astype(int))
    res = {'paper': PAPER, 'repeats': {}}
    rng = np.random.RandomState(0)
    for r in sorted(folds):
        fr = folds[r]
        y = np.concatenate([fr[n][0] for n in sorted(fr)])
        p = np.concatenate([fr[n][1] for n in sorted(fr)])
        m = metrics(y, p)
        m['n_folds'] = len(fr)
        m['per_fold_accuracy'] = {n: accuracy_score(*fr[n]) for n in sorted(fr)}
        boot = {k: [] for k in ('accuracy', 'kappa', 'macro_f1')}
        for _ in range(10000):
            pick = rng.choice(sorted(fr), len(fr))
            yb = np.concatenate([fr[n][0] for n in pick])
            pb = np.concatenate([fr[n][1] for n in pick])
            boot['accuracy'].append(accuracy_score(yb, pb))
            boot['kappa'].append(cohen_kappa_score(yb, pb, labels=range(5)))
            boot['macro_f1'].append(np.mean(f1_score(yb, pb, average=None, labels=range(5))))
        m['bootstrap95'] = {k: np.percentile(v, [2.5, 97.5]).tolist() for k, v in boot.items()}
        res['repeats'][r] = m
    if res['repeats']:
        keys = ('accuracy', 'kappa', 'macro_f1', 'sensitivity', 'specificity')
        res['mean'] = {k: float(np.mean([m[k] for m in res['repeats'].values()])) for k in keys}
        res['sd'] = {k: float(np.std([m[k] for m in res['repeats'].values()], ddof=1)) if len(res['repeats']) > 1
                     else None for k in keys}
    json.dump(res, open(os.path.join(out, 'results.json'), 'w'), indent=1)

    print('L-SeqSleepNet, Sleep-EDF-20 Fpz-Cz, leave-one-subject-out (released code and data; each fold tested on')
    print('its own held-out subject), random initialisation\n')
    print(f'{"":12s} ' + ' '.join(f'{"repeat " + str(r):>18s}' for r in res['repeats']) + f' {"paper (mean ± SD, 5 rep.)":>26s}')
    for k in ('accuracy', 'kappa', 'macro_f1', 'sensitivity', 'specificity'):
        cells = ' '.join(f'{m[k]:18.4f}' for m in res['repeats'].values())
        print(f'{k:12s} {cells} {PAPER[k][0]:19.3f} ± {PAPER[k][1]:.3f}')
    for r, m in res['repeats'].items():
        b = m['bootstrap95']
        print(f'repeat {r}: {m["n_folds"]} folds, {m["n_epochs"]} test epochs; 95% CI over subjects: accuracy '
              f'[{b["accuracy"][0]:.4f}, {b["accuracy"][1]:.4f}], kappa [{b["kappa"][0]:.4f}, {b["kappa"][1]:.4f}], '
              f'MF1 [{b["macro_f1"][0]:.4f}, {b["macro_f1"][1]:.4f}]')
        print('  per-stage F1 ' + ' '.join(f'{s} {v:.4f} (paper {pv[0]:.3f})' for s, v, pv in zip(STAGES, m['f1'], PAPER['f1'])))


if __name__ == '__main__':
    main()
