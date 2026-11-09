"""Write results/summary.csv from results/results.json in the repo-wide schema (docs/adding-an-experiment.md).
usage: python code/export_summary.py [results]"""
import csv, json, os, sys

RESULTS = sys.argv[1] if len(sys.argv) > 1 else 'results'
r = json.load(open(os.path.join(RESULTS, 'results.json')))
rows = []
for rule, values in r.items():
    v = values['DGCNN']
    rows.append(dict(method='DGCNN', dataset='BCI-IV-2a', protocol='within-subject, session T -> E', setting=f"our DGCNN on 5-band DE, EEG Conformer's training loop: {rule}",
                     metric='accuracy', paper_mean='', paper_std='', paper_source='', ours_mean=round(sum(v) / len(v) / 100, 4), ours_std='',
                     n_runs=27, headline=int(rule.startswith('validation'))))
with open(os.path.join(RESULTS, 'summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print('wrote', len(rows))
