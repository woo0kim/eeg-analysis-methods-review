"""Write results/summary.csv from results/results.json in the repo-wide schema (docs/adding-an-experiment.md).

usage: python code/export_summary.py [RESULTS_DIR]
"""
import csv, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '../results')
r = json.load(open(os.path.join(RESULTS, 'results.json')))
WITHIN, LOSO = 'within-subject, session T -> E', 'cross-subject, leave-one-subject-out (9 folds)'
ROWS = [  # key in results.json headline, protocol, setting, paper (acc, kappa, source), headline flag
    ('paper2022_best', WITHIN,
     'code behind the paper (upstream 3a318b2, 2022): session E is the validation data for checkpoint and early '
     'stopping; best of 10 runs per subject (its test()), the rule behind 85.38%', (0.8538, 0.805, 'Tables IV-V'), 1),
    ('paper2022_mean', WITHIN,
     'code behind the paper (upstream 3a318b2, 2022): session E is the validation data; mean of 10 runs',
     (0.8538, 0.805, 'Tables IV-V'), 0),
    ('traintest_best', WITHIN,
     "today's main_TrainTest.py: session E is the validation data; best of 10 runs per subject (released test())",
     (0.8538, 0.805, 'Tables IV-V'), 0),
    ('traintest_mean', WITHIN,
     "today's main_TrainTest.py: session E is the validation data; mean of 10 runs", (0.8538, 0.805, 'Tables IV-V'), 0),
    ('trainvaltest_mean', WITHIN,
     "today's main_TrainValTest.py (20% of session T is the validation data, test set used once); mean of 10 seeds",
     (0.8538, 0.805, 'Tables IV-V'), 1),
    ('loso_best', LOSO,
     "today's main_TrainTest.py with LOSO=True (held-out subject is the validation data); best of {n} runs per fold",
     (0.7097, 0.613, 'Table VI'), 1),
    ('loso_mean', LOSO,
     "today's main_TrainTest.py with LOSO=True (held-out subject is the validation data); mean of {n} runs",
     (0.7097, 0.613, 'Table VI'), 0),
]
rows = []
for key, protocol, setting, (pa, pk, src), head in ROWS:
    if key not in r['headline']:
        continue
    h = r['headline'][key]
    n = r['n_runs'][h['protocol']]
    per_fold = n // 9
    setting = setting.format(n=per_fold)
    sd = h.get('sd_over_seed_means')
    for metric, paper, ours, std in (('accuracy', pa, h['mean_acc'] / 100, sd / 100 if sd is not None else ''),
                                     ('kappa', pk, h['kappa'], '')):
        rows.append(dict(method='ATCNet', dataset='BCI-IV-2a', protocol=protocol, setting=setting, metric=metric,
                         paper_mean=paper, paper_std='', paper_source=src, ours_mean=round(ours, 4),
                         ours_std=round(std, 4) if std != '' else '', n_runs=n,
                         headline=int(head and metric == 'accuracy')))
with open(os.path.join(RESULTS, 'summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f'wrote {len(rows)} rows to', os.path.join(RESULTS, 'summary.csv'))
