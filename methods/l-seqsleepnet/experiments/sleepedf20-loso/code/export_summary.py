"""Write results/summary.csv from results/results.json in the repo-wide schema (docs/adding-an-experiment.md).

usage: python code/export_summary.py [results]
"""
import csv, json, os, sys

RESULTS = sys.argv[1] if len(sys.argv) > 1 else 'results'
res = json.load(open(os.path.join(RESULTS, 'results.json')))
n = len(res['repeats'])
rows = [dict(method='L-SeqSleepNet', dataset='Sleep-EDF', protocol='Sleep-EDF-20 Fpz-Cz, leave-one-subject-out',
             setting=f'released code and data, random initialisation; {n} of the 5 repetitions, 20 folds pooled',
             metric=metric, paper_mean=res['paper'][metric][0], paper_std=res['paper'][metric][1],
             paper_source='Table III (mean ± SD, 5 repetitions)', ours_mean=round(res['mean'][metric], 4),
             ours_std=round(res['sd'][metric], 4) if res['sd'][metric] is not None else '', n_runs=n,
             headline=int(metric == 'accuracy'))
        for metric in ('accuracy', 'macro_f1', 'kappa')]
with open(os.path.join(RESULTS, 'summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f'wrote {len(rows)} rows to', os.path.join(RESULTS, 'summary.csv'))
