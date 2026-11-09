"""Every number of a STEW pipeline experiment from its runs.

usage: python collect.py <runs dir> <results dir> > <results dir>/results.txt

A run directory is <task>__fold<k>__<model>__s<seed>, holding metrics.json and test_preds.npz (trainer.py), or LaBraM's preds.npz, from
which the same two files are written (first epoch with the best validation kappa). Each seed's 7 folds cover every window once,
so a seed's score is computed on its pooled test predictions; reported: mean ± SD over seeds, and the SD of the 7 folds within a seed.
"""
import glob
import json
import os
import sys
import zipfile

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score

RUNS, OUT = sys.argv[1:3]
os.makedirs(OUT, exist_ok=True)
EPOCHS = 50
NAMES = ['accuracy', 'balanced_acc', 'kappa', 'weighted_f1', 'macro_f1']


def from_labram(d):
    try:
        z = np.load(os.path.join(d, 'preds.npz'))
        yv, yt = z['y_val'].astype(int), z['y_test'].astype(int)
        pv, pt = z['val'].argmax(-1), z['test'].argmax(-1)
    except (zipfile.BadZipFile, EOFError, KeyError, OSError):
        return
    n = min(len(pv), len(pt))
    if n < EPOCHS:
        return
    hist = [{'epoch': e + 1, 'val': {'kappa': cohen_kappa_score(yv, pv[e])}, 'test': {'kappa': cohen_kappa_score(yt, pt[e])}} for e in range(n)]
    json.dump(hist, open(os.path.join(d, 'metrics.json'), 'w'))
    best = int(np.argmax([h['val']['kappa'] for h in hist]))
    np.savez_compressed(os.path.join(d, 'test_preds.npz'), y=yt.astype(np.int8), pred=pt[best].astype(np.int8), best_epoch=best + 1)


def score(y, p):
    return [accuracy_score(y, p), balanced_accuracy_score(y, p), cohen_kappa_score(y, p), f1_score(y, p, average='weighted'), f1_score(y, p, average='macro')]


data = {}
for d in sorted(glob.glob(os.path.join(RUNS, '*__fold*__*__s*'))):
    task, fold, model, seed = os.path.basename(d).split('__')
    if os.path.exists(os.path.join(d, 'preds.npz')) and not os.path.exists(os.path.join(d, 'test_preds.npz')):
        from_labram(d)
    m, t = os.path.join(d, 'metrics.json'), os.path.join(d, 'test_preds.npz')
    if not (os.path.exists(m) and os.path.exists(t)) or len(json.load(open(m))) < EPOCHS:
        continue
    z = np.load(t)
    hist = json.load(open(m))
    data.setdefault((task, model, int(seed[1:])), {})[int(fold[4:])] = (z['y'].astype(int), z['pred'].astype(int), int(np.argmax([h['val']['kappa'] for h in hist])) + 1,
                                                                         max(h['test']['kappa'] for h in hist))
rows = []
for (task, model, seed), folds in sorted(data.items()):
    if len(folds) < 7:
        print('incomplete', task, model, seed, sorted(folds))
        continue
    y, p = np.concatenate([folds[k][0] for k in range(7)]), np.concatenate([folds[k][1] for k in range(7)])
    per = [score(*folds[k][:2]) for k in range(7)]
    rows.append(dict(task=task, model=model, seed=seed, metric_names=NAMES, metrics=score(y, p), per_fold=per, best_epochs=[folds[k][2] for k in range(7)],
                     cm=np.bincount(int(y.max() + 1) * y + p, minlength=int(y.max() + 1) ** 2).reshape(int(y.max() + 1), -1).tolist()))
summary = []
for key in sorted({(r['task'], r['model']) for r in rows}):
    rs = [r for r in rows if (r['task'], r['model']) == key]
    a, f = np.array([r['metrics'] for r in rs]), np.array([r['per_fold'] for r in rs])
    summary.append(dict(task=key[0], model=key[1], n_seeds=len(rs), metric_names=NAMES, mean=a.mean(0).tolist(), std=(a.std(0, ddof=1) if len(rs) > 1 else np.zeros(5)).tolist(),
                        fold_mean=f.mean((0, 1)).tolist(), fold_sd=f.std(1, ddof=1).mean(0).tolist()))
    print(f'{key[0]:10s} {key[1]:17s} seeds={len(rs)} ' + ' '.join(f'{n} {m:.4f}±{s:.4f}' for n, m, s in zip(NAMES, a.mean(0), summary[-1]['std'])) + f'  fold SD(acc) {summary[-1]["fold_sd"][0]:.4f}')
json.dump(dict(runs=rows, summary=summary), open(os.path.join(OUT, 'results.json'), 'w'), indent=1)
