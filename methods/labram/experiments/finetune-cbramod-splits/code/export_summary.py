"""Write results/summary.csv from results/results.json in the repo-wide schema (docs/adding-an-experiment.md).

usage: python code/export_summary.py [results]
"""
import csv, json, os, sys

RESULTS = sys.argv[1] if len(sys.argv) > 1 else 'results'
# same names and subject splits as methods/cbramod/experiments/finetune-public-datasets/scripts/export_summary.py
DATASET = {
    'PhysioNet-MI': ('PhysioNet-MI', 'cross-subject 70/19/20'),
    'BCIC-IV-2a': ('BCI-IV-2a', 'cross-subject 5/2/2'),
    'Mumtaz2016': ('Mumtaz2016', 'cross-subject 43/9/11'),
    'MentalArithmetic': ('MentalArithmetic', 'cross-subject 28/4/4'),
}
SETTING = {
    'labram': "fine-tuned from the released weights (LaBraM README recipe), CBraMod's splits and checkpoint rule",
    'labram@val-acc': "same runs, checkpoint by validation accuracy (LaBraM's own rule)",
}

rows = []
for s in json.load(open(os.path.join(RESULTS, 'results.json')))['summary']:
    dataset, protocol = DATASET[s['dataset']]
    for i, metric in enumerate(s['metric_names']):
        rows.append(dict(method='LaBraM', dataset=dataset, protocol=protocol, setting=SETTING[s['setting']],
                         metric=metric, paper_mean=s['paper'][i][0], paper_std=s['paper'][i][1],
                         paper_source=f"CBraMod paper {s['paper_table']}", ours_mean=round(s['mean'][i], 4),
                         ours_std=round(s['std'][i], 4) if s['n'] > 1 else '', n_runs=s['n'],
                         headline=int(metric == 'balanced_acc' and s['setting'] == 'labram')))

with open(os.path.join(RESULTS, 'summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f'wrote {len(rows)} rows to', os.path.join(RESULTS, 'summary.csv'))
