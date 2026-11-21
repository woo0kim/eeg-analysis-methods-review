"""Parse the logs of run_models.py on CBraMod's four public datasets (same parsing and results.json schema as
../../../../eegnet/experiments/pytorch-cbramod-splits/code/collect.py) and compare EEG Conformer with the CBraMod paper's
EEGConformer rows (Welch's t-test from mean and SD, 5 seeds each). The paper has no ATCNet or DeepSleepNet rows.

usage: python collect.py <runs dir> <out dir> [<setting> ...] > <out dir>/results.txt
       only the given settings (e.g. bd-atcnet) are collected; all settings when none are given
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats

RUNS, OUT, KEEP = sys.argv[1], sys.argv[2], set(sys.argv[3:])
os.makedirs(OUT, exist_ok=True)
MC, BC = ['balanced_acc', 'kappa', 'weighted_f1'], ['balanced_acc', 'pr_auc', 'roc_auc']
# EEGConformer rows of the CBraMod paper (ICLR 2025): mean, std over 5 seeds; BIOT's code, settings not reported.
# Table 15's header says AUC-PR / AUROC, but its multi-class columns are κ and weighted F1 (as for EEGNet's row there).
CONFORMER = {
    'PhysioNet-MI': ('Table 3', [(0.6049, 0.0104), (0.4736, 0.0171), (0.6062, 0.0095)]),
    'BCIC-IV-2a': ('Table 15', [(0.4696, 0.0106), (0.2924, 0.0141), (0.4533, 0.0128)]),
    'Mumtaz2016': ('Table 10', [(0.9308, 0.0117), (0.9684, 0.0105), (0.9702, 0.0101)]),
    'MentalArithmetic': ('Table 12', [(0.6805, 0.0123), (0.5829, 0.0134), (0.7424, 0.0128)]),
}
PAPER = {(ds, s): v for ds, v in CONFORMER.items() for s in ('conformer', 'bd-conformer')}
runs = []
for log in sorted(glob.glob(f'{RUNS}/*/log.txt')):
    ds, setting, seed = os.path.basename(os.path.dirname(log)).split('__')
    if KEEP and setting not in KEEP:
        continue
    txt = open(log, errors='ignore').read()
    m = re.search(r'Test Evaluation: acc: ([\d.]+), (kappa|pr_auc): ([\d.]+), (f1|roc_auc): ([\d.]+)', txt)
    if not m:
        continue
    ep = re.search(r'model save in .*/epoch(\d+)_', txt)
    n_par = re.search(r'params: (\d+)', txt)
    runs.append(dict(dataset=ds, setting=setting, seed=int(seed[1:]), metrics=[float(m.group(1)), float(m.group(3)), float(m.group(5))],
                     metric_names=MC if m.group(2) == 'kappa' else BC, best_epoch=int(ep.group(1)) if ep else None,
                     epochs_run=len(re.findall(r'^Epoch \d+ :', txt, re.M)), params=int(n_par.group(1)) if n_par else None))
summary = []
for key in sorted({(r['dataset'], r['setting']) for r in runs}):
    rs = sorted((r for r in runs if (r['dataset'], r['setting']) == key), key=lambda r: r['seed'])
    a = np.array([r['metrics'] for r in rs])
    row = dict(dataset=key[0], setting=key[1], n=len(rs), seeds=[r['seed'] for r in rs], metric_names=rs[0]['metric_names'],
               params=rs[0]['params'], per_seed=a.tolist(), best_epochs=[r['best_epoch'] for r in rs], mean=a.mean(0).tolist(),
               std=a.std(0, ddof=1).tolist() if len(rs) > 1 else [0, 0, 0])
    if key in PAPER:
        tab, pv = PAPER[key]
        row.update(paper_table=tab, paper=pv, p_welch=[
            float(stats.ttest_ind_from_stats(mu, sd, len(rs), pm, ps, 5, equal_var=False).pvalue) if len(rs) > 1 else None
            for mu, sd, (pm, ps) in zip(row['mean'], row['std'], pv)])
    summary.append(row)
json.dump(dict(runs=runs, summary=summary), open(os.path.join(OUT, 'results.json'), 'w'), indent=1)
for row in summary:
    print(f"\n{row['dataset']} [{row['setting']}] n={row['n']} params={row['params']} best_epochs={row['best_epochs']}")
    for i, mn in enumerate(row['metric_names']):
        s = f"  {mn:13s} ours {row['mean'][i]:.4f} ± {row['std'][i]:.4f}"
        if 'paper' in row:
            pm, ps = row['paper'][i]
            p = row['p_welch'][i]
            s += f" | paper {pm:.4f} ± {ps:.4f} ({row['paper_table']}) | diff {row['mean'][i] - pm:+.4f}" + (f" | p={p:.3g}" if p is not None else '')
        print(s)
        print('      per-seed:', ' '.join(f'{v[i]:.4f}' for v in row['per_seed']))
