"""Write results/summary.csv from results/results.json in the repo-wide schema (docs/adding-an-experiment.md).

usage: python scripts/export_summary.py [results]
"""
import csv, json, os, sys

RESULTS = sys.argv[1] if len(sys.argv) > 1 else 'results'
# CBraMod dataset name -> (repo-wide dataset name, subject split of the released loaders: train/val/test)
DATASET = {
    'PhysioNet-MI': ('PhysioNet-MI', 'cross-subject 70/19/20'),
    'BCIC-IV-2a': ('BCI-IV-2a', 'cross-subject 5/2/2'),
    'BCIC-IV-2a@v0616': ('BCI-IV-2a', 'cross-subject 5/2/2'),
    'Mumtaz2016': ('Mumtaz2016', 'cross-subject 43/9/11'),
    'MentalArithmetic': ('MentalArithmetic', 'cross-subject 28/4/4'),
}
SETTING = {
    'pretrained': 'fine-tuned from the released weights (paper defaults)',
    'scratch': 'trained from random init',
    'frozen': 'released weights, backbone frozen',
    'pt-lr5e-4-single': 'fine-tuned, lr 5e-4 without multi_lr (issue #6)',
    'pretrained@seed3407': 'fine-tuned, code-default seed 3407 only',
    'eegnet': "EEGNet-8,2 through CBraMod's fine-tuning pipeline",
}

rows = []
for s in json.load(open(os.path.join(RESULTS, 'results.json')))['summary']:
    dataset, protocol = DATASET[s['dataset']]
    setting = SETTING[s['setting']]
    if s['dataset'].endswith('@v0616'):
        setting += ', data from the pre-A04T-fix preprocessing (v0616)'
    for i, metric in enumerate(s['metric_names']):
        paper = s.get('paper', [('', '')] * 3)[i]
        rows.append(dict(method='EEGNet' if s['setting'] == 'eegnet' else 'CBraMod', dataset=dataset,
                         protocol=protocol, setting=setting, metric=metric,
                         paper_mean=paper[0], paper_std=paper[1], paper_source=s.get('paper_table', ''),
                         ours_mean=round(s['mean'][i], 4), ours_std=round(s['std'][i], 4) if s['n'] > 1 else '',
                         n_runs=s['n'],
                         headline=int(metric == 'balanced_acc' and s['setting'] in ('pretrained', 'eegnet')
                                      and '@' not in s['dataset'])))

with open(os.path.join(RESULTS, 'summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f'wrote {len(rows)} rows to', os.path.join(RESULTS, 'summary.csv'))
