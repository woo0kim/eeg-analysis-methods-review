"""Write results/summary.csv from results/results.json in the repo-wide schema (docs/adding-an-experiment.md).
usage: python export_summary.py <results dir> <method name>      e.g. results "EEG Conformer", results ATCNet"""
import csv
import json
import os
import sys

RESULTS, METHOD = sys.argv[1], sys.argv[2]
DATASET = {'PhysioNet-MI': ('PhysioNet-MI', 'cross-subject 70/19/20'), 'BCIC-IV-2a': ('BCI-IV-2a', 'cross-subject 5/2/2'),
           'Mumtaz2016': ('Mumtaz2016', 'cross-subject 43/9/11'), 'MentalArithmetic': ('MentalArithmetic', 'cross-subject 28/4/4')}
SETTING = {'conformer': "released EEG Conformer model through CBraMod's fine-tuning pipeline",
           'bd-conformer': "braindecode EEGConformer through CBraMod's fine-tuning pipeline",
           'bd-atcnet': "braindecode ATCNet through CBraMod's fine-tuning pipeline",
           'bd-deepsleepnet': "braindecode DeepSleepNet through CBraMod's fine-tuning pipeline (off-label)"}
HEADLINE = {'conformer', 'bd-atcnet', 'bd-deepsleepnet'}
rows = []
for s in json.load(open(os.path.join(RESULTS, 'results.json')))['summary']:
    dataset, protocol = DATASET[s['dataset']]
    for i, metric in enumerate(s['metric_names']):
        pm, ps = s['paper'][i] if 'paper' in s else ('', '')
        rows.append(dict(method=METHOD, dataset=dataset, protocol=protocol, setting=SETTING[s['setting']], metric=metric,
                         paper_mean=pm, paper_std=ps, paper_source=f"CBraMod paper {s['paper_table']}" if 'paper' in s else '',
                         ours_mean=round(s['mean'][i], 4), ours_std=round(s['std'][i], 4), n_runs=s['n'],
                         headline=int(metric == 'balanced_acc' and s['setting'] in HEADLINE)))
with open(os.path.join(RESULTS, 'summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print('wrote', len(rows))
