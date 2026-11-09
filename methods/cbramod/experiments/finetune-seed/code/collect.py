"""Every number of a SEED / SEED-V pipeline experiment from its runs.

usage: python collect.py <runs dir> <results dir> <test_trials.json> > <results dir>/results.txt

A run directory is <dataset>__<split>__<model>__s<seed>. Runs through trainer.py hold metrics.json (per-epoch validation and
test scores) and test_preds.npz (labels and predictions at the selected epoch, the first with the best validation
kappa). LaBraM's runs hold preds.npz (probabilities of every epoch), from which collect.py picks the same epoch and
writes the same two files (metrics.json, test_preds.npz): the probabilities are too large to commit.

Metrics: CBraMod's (balanced accuracy, kappa, weighted F1) plus accuracy and macro-F1, on the 1-s samples, and the
accuracy of a per-clip majority vote over the clip's 1-s predictions, which is the label level the data really has.
"""
import glob
import json
import os
import sys
import zipfile

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score

RUNS, OUT, TRIALS = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
trials = json.load(open(TRIALS))
EPOCHS = 50
NAMES = ['balanced_acc', 'kappa', 'weighted_f1', 'accuracy', 'macro_f1', 'clip_vote_acc']


def scores(y, p, rle, n_classes):
    start, votes = 0, []
    for _, _, n in rle:
        votes.append(np.bincount(p[start:start + n], minlength=n_classes).argmax() == y[start])
        assert len(set(y[start:start + n].tolist())) == 1
        start += n
    assert start == len(y), (start, len(y))
    return [balanced_accuracy_score(y, p), cohen_kappa_score(y, p), f1_score(y, p, average='weighted'),
            accuracy_score(y, p), f1_score(y, p, average='macro'), float(np.mean(votes))]


def from_labram(d):
    """preds.npz -> metrics.json + test_preds.npz, selecting as trainer.py does: first epoch with the highest val kappa."""
    try:
        z = np.load(os.path.join(d, 'preds.npz'))
        yv, yt = z['y_val'].astype(int), z['y_test'].astype(int)
        pv, pt = z['val'].argmax(-1), z['test'].argmax(-1)
    except (zipfile.BadZipFile, EOFError, KeyError, OSError):    # being rewritten by a run in progress
        return
    n = min(len(pv), len(pt))
    if n < EPOCHS:                                               # a run in progress: never score a partial run
        return
    hist = []
    for e in range(n):
        hist.append({'epoch': e + 1,
                     'val': {'balanced_acc': balanced_accuracy_score(yv, pv[e]), 'kappa': cohen_kappa_score(yv, pv[e]),
                             'weighted_f1': f1_score(yv, pv[e], average='weighted')},
                     'test': {'balanced_acc': balanced_accuracy_score(yt, pt[e]), 'kappa': cohen_kappa_score(yt, pt[e]),
                              'weighted_f1': f1_score(yt, pt[e], average='weighted')}})
    json.dump(hist, open(os.path.join(d, 'metrics.json'), 'w'))
    best = int(np.argmax([h['val']['kappa'] for h in hist]))
    np.savez_compressed(os.path.join(d, 'test_preds.npz'), y=yt.astype(np.int8), pred=pt[best].astype(np.int8), best_epoch=best + 1)


rows = []
for d in sorted(glob.glob(os.path.join(RUNS, '*__s*'))):
    name = os.path.basename(d)
    parts = name.split('__')
    if len(parts) != 4:
        continue
    ds, split, model, seed = parts
    if os.path.exists(os.path.join(d, 'preds.npz')) and not os.path.exists(os.path.join(d, 'test_preds.npz')):
        from_labram(d)
    if not os.path.exists(os.path.join(d, 'test_preds.npz')):
        continue
    z = np.load(os.path.join(d, 'test_preds.npz'))
    y, p = z['y'].astype(int), z['pred'].astype(int)
    hist = json.load(open(os.path.join(d, 'metrics.json')))
    if len(hist) < EPOCHS:
        continue
    val = [h['val']['kappa'] for h in hist]
    best = int(np.argmax(val))
    k = {'SEED': 3, 'SEED-V': 5}[ds]
    m = scores(y, p, trials[f'{ds}/{split}'], k)
    tk = [h['test']['kappa'] for h in hist]
    rows.append(dict(dataset=ds, split=split, model=model, seed=int(seed[1:]), metric_names=NAMES, metrics=m, best_epoch=best + 1,
                     epochs=len(hist), val_kappa=val[best], best_test_epoch_kappa=max(tk), last_epoch_kappa=tk[-1],
                     per_class_f1=f1_score(y, p, average=None, labels=list(range(k))).tolist(),
                     cm=np.bincount(k * y + p, minlength=k * k).reshape(k, k).tolist()))

summary = []
for key in sorted({(r['dataset'], r['split'], r['model']) for r in rows}):
    rs = sorted((r for r in rows if (r['dataset'], r['split'], r['model']) == key), key=lambda r: r['seed'])
    a = np.array([r['metrics'] for r in rs])
    summary.append(dict(dataset=key[0], split=key[1], model=key[2], n=len(rs), seeds=[r['seed'] for r in rs], metric_names=NAMES,
                        mean=a.mean(0).tolist(), std=(a.std(0, ddof=1) if len(rs) > 1 else np.zeros(a.shape[1])).tolist(),
                        best_epochs=[r['best_epoch'] for r in rs]))
    print(f"{key[0]:7s} {key[1]:9s} {key[2]:17s} n={len(rs)} " +
          ' '.join(f'{n} {m:.4f}±{s:.4f}' for n, m, s in zip(NAMES, summary[-1]['mean'], summary[-1]['std'])))
json.dump(dict(runs=rows, summary=summary), open(os.path.join(OUT, 'results.json'), 'w'), indent=1)
