"""Write results/summary.csv from results/results.json in the repo-wide schema (docs/adding-an-experiment.md).

usage: python code/export_summary.py [results]
"""
import csv, json, os, sys

RESULTS = sys.argv[1] if len(sys.argv) > 1 else 'results'
res = json.load(open(os.path.join(RESULTS, 'results.json')))
rows = [dict(method='DeepSleepNet', dataset='Sleep-EDF', protocol='Sleep-EDF-20 Fpz-Cz, 20-fold subject CV',
             setting='released code (pre-training + fine-tuning, final model), 20 folds pooled', metric=metric,
             paper_mean=res['paper'][metric], paper_std='', paper_source='Table IV',
             ours_mean=round(res['pooled'][metric], 4), ours_std='', n_runs=1, headline=int(metric == 'accuracy'))
        for metric in ('accuracy', 'macro_f1', 'kappa')]
with open(os.path.join(RESULTS, 'summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f'wrote {len(rows)} rows to', os.path.join(RESULTS, 'summary.csv'))
