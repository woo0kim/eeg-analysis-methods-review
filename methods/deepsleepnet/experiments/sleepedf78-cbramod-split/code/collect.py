"""DeepSleepNet on the CBraMod-pipeline split: metrics per run on all test epochs and on the common test set.

usage: python collect.py <preds dir> <out dir> > <out dir>/results.txt
  <preds dir>/run<r>/subject<i>.npz   ../../sleepedf20-20fold/code/export_preds.py of run r's predict_split.py output
  <preds dir>/run<r>/subjects.json    index -> test subject and its night files (predict_split.py)
  <preds dir>/keys.json               absolute epoch indices per night (prepare_split_data.py)
writes <out dir>/results.json in the schema of ../../../../cbramod/experiments/finetune-sleepedf/code/collect.py.
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '../../../../cbramod/experiments/finetune-sleepedf/code'))
import split_metrics  # noqa: E402

pdir, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
keys = json.load(open(os.path.join(pdir, 'keys.json')))
per_run = {}
for rdir in sorted(glob.glob(os.path.join(pdir, 'run*'))):
    r = int(os.path.basename(rdir)[3:])
    subjects = json.load(open(os.path.join(rdir, 'subjects.json')))
    pred, lab = {}, {}
    for i, info in subjects.items():
        d = np.load(os.path.join(rdir, f'subject{i}.npz'))
        yt, yp, start = d['y_true'].astype(int), d['y_pred'].astype(int), 0
        for fname, n in zip(info['files'], d['night_lengths']):
            rec = fname[:6]
            assert n == len(keys[rec]) // 25 * 25, (rec, len(keys[rec]), n)   # predict.py drops each night's last n mod 25
            for j, k in enumerate(keys[rec][:n]):
                pred[(rec, k)], lab[(rec, k)] = yp[start + j], yt[start + j]
            start += n
    per_run[r] = split_metrics.score_run(pred, lab)

res = {'models': {}}
if per_run:
    names = list(next(iter(per_run.values()))['common'])
    for part in ('common', 'all'):
        res['models']['deepsleepnet' + ('' if part == 'common' else '@all')] = {
            'label': 'DeepSleepNet', 'implementation': 'released (TF 1.15)', 'epochs': part,
            'seeds': sorted(per_run), 'per_seed': {r: per_run[r][part] for r in sorted(per_run)},
            'mean': {k: float(np.mean([per_run[r][part][k] for r in per_run])) for k in names},
            'std': {k: float(np.std([per_run[r][part][k] for r in per_run], ddof=1)) if len(per_run) > 1 else None
                    for k in names}}
json.dump(res, open(os.path.join(out, 'results.json'), 'w'), indent=1, default=float)

print('DeepSleepNet (released code) on Sleep-EDF-78, CBraMod-pipeline split (62 / 8 / 8 subjects), Fpz-Cz\n')
for key, m in res['models'].items():
    print(f'{"common test set (the CBraMod-pipeline test epochs)" if m["epochs"] == "common" else "all kept test epochs"}: '
          f'{m["per_seed"][m["seeds"][0]]["n_epochs"]} epochs, runs {m["seeds"]}')
    for k in ('balanced_acc', 'kappa', 'weighted_f1', 'accuracy', 'macro_f1'):
        sd = f' ± {m["std"][k]:.4f}' if m['std'][k] is not None else ''
        print(f'  {k:13s} {m["mean"][k]:.4f}{sd}   per run: ' + ' '.join(f'{m["per_seed"][r][k]:.4f}' for r in m['seeds']))
    print('  per-stage F1  ' + ' '.join(f'{s} {m["mean"]["f1_" + s]:.4f}' for s in split_metrics.STAGES))
