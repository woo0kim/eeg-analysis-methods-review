"""DGCNN vs. EEG Conformer and EEGNet on BCI IV 2a: same training loop, seeds and epoch-selection rules.

Reads this experiment's curves (results/curve_*.npz) and those committed by the EEG Conformer and EEGNet experiments.
Accuracy in %, per subject the mean over seeds 2023-2025; two-sided Wilcoxon signed-rank over the 9 subjects.
usage: python code/analyze.py [results]
"""
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = sys.argv[1] if len(sys.argv) > 1 else 'results'
OTHERS = {'EEG Conformer': os.path.join(HERE, '../../../../eeg-conformer/experiments/bciciv2a-epoch-selection/results'),
          'EEGNet': os.path.join(HERE, '../../../../eegnet/experiments/bciciv2a-epoch-selection/results')}
SUBJECTS, SEEDS = range(1, 10), (2023, 2024, 2025)
RULES = {'max test acc over 2000 epochs (released rule)': ('A', lambda d: d['test'].max()),
         'mean of the last 100 epochs': ('A', lambda d: d['test'][-100:].mean()),
         'final epoch': ('A', lambda d: d['test'][-1]),
         'validation-selected epoch (protocol B)': ('B', lambda d: d['test'][int(np.nanargmax(d['val']))])}


def load(path):
    runs = {}
    for f in glob.glob(os.path.join(path, 'curve_*.npz')):
        d = np.load(f)
        if str(d['tag']) in ('A', 'B'):
            runs[(str(d['tag']), int(d['subject']), int(d['seed']))] = {'test': d['test'], 'val': d['val'] if 'val' in d.files else None}
    return runs


models = {'DGCNN': load(RESULTS), **{k: load(v) for k, v in OTHERS.items()}}
print({k: len(v) for k, v in models.items()})
out = {}
print('\n| Rule | DGCNN | EEG Conformer | EEGNet | Conformer − DGCNN (p) | EEGNet − DGCNN (p) |\n|---|---|---|---|---|---|')
for rule, (tag, fn) in RULES.items():
    per = {m: np.array([np.mean([fn(models[m][(tag, s, sd)]) * 100 for sd in SEEDS if (tag, s, sd) in models[m]]) for s in SUBJECTS]) for m in models}
    p1, p2 = stats.wilcoxon(per['EEG Conformer'], per['DGCNN']).pvalue, stats.wilcoxon(per['EEGNet'], per['DGCNN']).pvalue
    print(f"| {rule} | {per['DGCNN'].mean():.2f} | {per['EEG Conformer'].mean():.2f} | {per['EEGNet'].mean():.2f} | "
          f"{per['EEG Conformer'].mean() - per['DGCNN'].mean():+.2f} ({p1:.3f}) | {per['EEGNet'].mean() - per['DGCNN'].mean():+.2f} ({p2:.3f}) |")
    out[rule] = {m: [round(float(x), 2) for x in v] for m, v in per.items()}
json.dump(out, open(os.path.join(RESULTS, 'results.json'), 'w'), indent=1)
