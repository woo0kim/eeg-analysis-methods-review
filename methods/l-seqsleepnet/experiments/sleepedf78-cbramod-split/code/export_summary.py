"""Write results/summary.csv from results/results.json in the repo-wide schema (docs/adding-an-experiment.md).

The numbers are the common test set (the 20,976 test epochs every Sleep-EDF model scores); no paper reports this split.
usage: python code/export_summary.py [results]
"""
import csv, json, os, sys

RESULTS = sys.argv[1] if len(sys.argv) > 1 else 'results'
r = json.load(open(os.path.join(RESULTS, 'results.json')))['models']['l-seqsleepnet']
rows = [dict(method='L-SeqSleepNet', dataset='Sleep-EDF', protocol='cross-subject 62/8/8',
             setting="released code (TF 1.15) on the CBraMod-pipeline subject split, scored on the common test set",
             metric=metric, paper_mean='', paper_std='', paper_source='', ours_mean=round(r['mean'][metric], 4),
             ours_std=round(r['std'][metric], 4) if r['std'][metric] is not None else '', n_runs=len(r['seeds']),
             headline=int(metric == 'balanced_acc'))
        for metric in ('balanced_acc', 'kappa', 'weighted_f1', 'accuracy', 'macro_f1')]
with open(os.path.join(RESULTS, 'summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f'wrote {len(rows)} rows to', os.path.join(RESULTS, 'summary.csv'))
