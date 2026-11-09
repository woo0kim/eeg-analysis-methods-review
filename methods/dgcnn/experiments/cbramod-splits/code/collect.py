"""Parse the DGCNN logs on CBraMod's four public datasets (same parsing and results.json schema as
../../../../eegnet/experiments/pytorch-cbramod-splits/code/collect.py). DGCNN is not in the CBraMod paper.

usage: python collect.py <runs dir> <out dir> > <out dir>/results.txt
"""
import glob
import json
import os
import re
import sys

import numpy as np

RUNS, OUT = sys.argv[1:3]
os.makedirs(OUT, exist_ok=True)
MC, BC = ['balanced_acc', 'kappa', 'weighted_f1'], ['balanced_acc', 'pr_auc', 'roc_auc']
runs = []
for log in sorted(glob.glob(f'{RUNS}/*/log.txt')):
    ds, setting, seed = os.path.basename(os.path.dirname(log)).split('__')
    txt = open(log, errors='ignore').read()
    m = re.search(r'Test Evaluation: acc: ([\d.]+), (kappa|pr_auc): ([\d.]+), (f1|roc_auc): ([\d.]+)', txt)
    if not m:
        continue
    ep = re.search(r'model save in .*/epoch(\d+)_', txt)
    runs.append(dict(dataset=ds, setting=setting, seed=int(seed[1:]), metrics=[float(m.group(1)), float(m.group(3)), float(m.group(5))],
                     metric_names=MC if m.group(2) == 'kappa' else BC, best_epoch=int(ep.group(1)) if ep else None,
                     epochs_run=len(re.findall(r'^Epoch \d+ :', txt, re.M))))
summary = []
for key in sorted({(r['dataset'], r['setting']) for r in runs}):
    rs = sorted((r for r in runs if (r['dataset'], r['setting']) == key), key=lambda r: r['seed'])
    a = np.array([r['metrics'] for r in rs])
    row = dict(dataset=key[0], setting=key[1], n=len(rs), seeds=[r['seed'] for r in rs], metric_names=rs[0]['metric_names'], per_seed=a.tolist(),
               best_epochs=[r['best_epoch'] for r in rs], mean=a.mean(0).tolist(), std=a.std(0, ddof=1).tolist() if len(rs) > 1 else [0, 0, 0])
    summary.append(row)
    print(f"{key[0]:17s} {key[1]:12s} n={len(rs)} " + '  '.join(f'{n} {m:.4f} ± {s:.4f}' for n, m, s in zip(row['metric_names'], row['mean'], row['std'])))
json.dump(dict(runs=runs, summary=summary), open(os.path.join(OUT, 'results.json'), 'w'), indent=1)
