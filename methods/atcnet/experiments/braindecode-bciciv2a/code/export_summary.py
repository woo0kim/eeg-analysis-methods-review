"""Write results/summary.csv from results/results.json in the repo-wide schema (docs/adding-an-experiment.md).

usage: python code/export_summary.py [RESULTS_DIR]
"""
import csv, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '../results')
C = json.load(open(os.path.join(RESULTS, 'results.json')))['comparisons']
WITHIN, LOSO = 'within-subject, session T -> E', 'cross-subject, leave-one-subject-out (9 folds)'
VARIANT = {'default': 'braindecode ATCNet (defaults)',
           'faithful': 'braindecode ATCNet with its "source-faithful" options (conv max-norm 0.6, '
                       'source_optimizer_param_groups)',
           'keras': 'control: braindecode ATCNet switched to the current Keras training behaviour',
           'keras2022': 'control: braindecode ATCNet switched to the 2022 Keras training behaviour'}
PROTOCOL = {'paper2022': 'PyTorch port of the 2022 training loop behind the paper (session E as validation data)',
            'traintest': 'PyTorch port of main_TrainTest.py (session E as validation data)',
            'trainvaltest': 'PyTorch port of main_TrainValTest.py (20% of session T as validation data)',
            'loso': 'PyTorch port of main_TrainTest.py with LOSO=True'}
PAPER = {'loso': (0.7097, 0.613, 'Table VI')}
HEADLINE = {('default', 'paper2022', 'best'), ('default', 'trainvaltest', 'mean')}

rows = []
for key, c in C.items():
    v, p, rule = key.split('/')
    kind = 'best' if rule.startswith('best') else 'mean'
    pa, pk, src = PAPER.get(p, (0.8538, 0.805, 'Tables IV-V'))
    n = len(c['seeds']) * 9
    setting = f'{VARIANT[v]} in a {PROTOCOL[p]}; {rule}'
    sd = c.get('bd_sd_over_seed_means')
    for metric, paper, ours, std in (('accuracy', pa, c['bd_acc'] / 100, sd / 100 if sd is not None else ''),
                                     ('kappa', pk, c['bd_kappa'], '')):
        rows.append(dict(method='ATCNet', dataset='BCI-IV-2a', protocol=LOSO if p == 'loso' else WITHIN,
                         setting=setting, metric=metric, paper_mean=paper, paper_std='', paper_source=src,
                         ours_mean=round(ours, 4), ours_std=round(std, 4) if std != '' else '', n_runs=n,
                         headline=int((v, p, kind) in HEADLINE and metric == 'accuracy')))
with open(os.path.join(RESULTS, 'summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f'wrote {len(rows)} rows to', os.path.join(RESULTS, 'summary.csv'))
