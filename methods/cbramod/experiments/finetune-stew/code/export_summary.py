"""Write <results>/summary.csv from <results>/results.json (collect.py) in the repo-wide schema (docs/adding-an-experiment.md).

No paper among these methods reports STEW. The dataset paper (Lim et al., 2018) reports 69% / kappa 0.46 for its own model on 3 rating
levels; it is quoted in the READMEs, not here, because it is not a number for any of these models.
usage: python export_summary.py <results dir>
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../finetune-seed/code')))
from export_summary import METHOD, HEADLINE  # noqa: E402

RESULTS = sys.argv[1]
PROTOCOL = {'rest-task': 'cross-subject GroupKFold(7), rest vs. task', 'rating3': 'cross-subject GroupKFold(7), 3 rating levels'}
rows = []
for s in json.load(open(os.path.join(RESULTS, 'results.json')))['summary']:
    method, impl = METHOD[s['model']]
    for i, metric in enumerate(s['metric_names']):
        if metric == 'macro_f1':
            continue
        rows.append(dict(method=method, dataset='STEW', protocol=PROTOCOL[s['task']], setting=f"{impl}, CBraMod's training loop, 4-s windows",
                         metric=metric, paper_mean='', paper_std='', paper_source='', ours_mean=round(s['mean'][i], 4),
                         ours_std=round(s['std'][i], 4) if s['n_seeds'] > 1 else '', n_runs=s['n_seeds'] * 7,
                         headline=int(metric == 'balanced_acc' and s['model'] in HEADLINE)))
with open(os.path.join(RESULTS, 'summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f'wrote {len(rows)} rows to', os.path.join(RESULTS, 'summary.csv'))
