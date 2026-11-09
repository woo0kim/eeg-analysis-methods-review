"""Write results/summary.csv from results/results.json in the repo-wide schema (docs/adding-an-experiment.md).
usage: python code/export_summary.py [results]"""
import csv, json, os, sys

RESULTS = sys.argv[1] if len(sys.argv) > 1 else 'results'
r = json.load(open(os.path.join(RESULTS, 'results.json')))
BEST, FINAL = 'best epoch on test (released)', 'final epoch'
rows = []
for split, proto in (('kfold', 'within-subject, session 1, 5-fold inside clips'), ('trial96', 'within-subject, session 1, first 9 clips -> last 6')):
    for rule, setting, head in ((BEST, 'released model and training, best test epoch of 600', 1), (FINAL, 'released model and training, final epoch', 1)):
        v = r[split][rule]
        paper = split == 'kfold' and rule == BEST
        rows.append(dict(method='EEG-Conformer', dataset='SEED', protocol=proto, setting=setting, metric='accuracy',
                         paper_mean=0.9530 if paper else '', paper_std='', paper_source='README, 5-fold' if paper else '',
                         ours_mean=round(v['mean'] / 100, 4), ours_std='', n_runs=r[split]['n_runs'], headline=head))
with open(os.path.join(RESULTS, 'summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print('wrote', len(rows))
