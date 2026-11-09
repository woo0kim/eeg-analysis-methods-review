"""Write results/summary.csv from results/results.json in the repo-wide schema (docs/adding-an-experiment.md).
DGCNN's paper does not report these datasets, so the paper columns are empty. usage: python code/export_summary.py [results]"""
import csv, json, os, sys

RESULTS = sys.argv[1] if len(sys.argv) > 1 else 'results'
DATASET = {'PhysioNet-MI': ('PhysioNet-MI', 'cross-subject 70/19/20'), 'BCIC-IV-2a': ('BCI-IV-2a', 'cross-subject 5/2/2'),
           'Mumtaz2016': ('Mumtaz2016', 'cross-subject 43/9/11'), 'MentalArithmetic': ('MentalArithmetic', 'cross-subject 28/4/4')}
SETTING = {'dgcnn': "our DGCNN on 5-band DE through CBraMod's fine-tuning pipeline",
           'bd-dgcnn': "braindecode DGCNN on raw samples through CBraMod's fine-tuning pipeline",
           'bd-dgcnn-de': "braindecode DGCNN on 5-band DE through CBraMod's fine-tuning pipeline"}
rows = []
for s in json.load(open(os.path.join(RESULTS, 'results.json')))['summary']:
    dataset, protocol = DATASET[s['dataset']]
    for i, metric in enumerate(s['metric_names']):
        rows.append(dict(method='DGCNN', dataset=dataset, protocol=protocol, setting=SETTING[s['setting']], metric=metric, paper_mean='',
                         paper_std='', paper_source='', ours_mean=round(s['mean'][i], 4), ours_std=round(s['std'][i], 4), n_runs=s['n'],
                         headline=int(metric == 'balanced_acc' and s['setting'] == 'dgcnn')))
with open(os.path.join(RESULTS, 'summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print('wrote', len(rows))
