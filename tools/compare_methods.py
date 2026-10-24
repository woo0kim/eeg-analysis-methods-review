"""Welch tests between the models run on CBraMod's splits (same samples, subject splits, seeds 0-4,
checkpoint rule and metric code), read from the committed results.json files. Feeds SUMMARY.md.

Three tables: the released-code models against each other; each braindecode implementation against the original
implementation of the same model; and the braindecode implementations against each other.
usage: python tools/compare_methods.py [metric]     # balanced_acc (default), kappa, weighted_f1, roc_auc, pr_auc
"""
import json, os, sys
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CBRAMOD = 'methods/cbramod/experiments/finetune-public-datasets'
LABRAM = 'methods/labram/experiments/finetune-cbramod-splits'
CBRAMOD_BD = 'methods/cbramod/experiments/braindecode-public-datasets'
LABRAM_BD = 'methods/labram/experiments/braindecode-cbramod-splits'
EEGNET_BD = 'methods/eegnet/experiments/braindecode-cbramod-splits'
MODELS = {'EEGNet': (CBRAMOD, 'eegnet'), 'CBraMod scratch': (CBRAMOD, 'scratch'),
          'CBraMod': (CBRAMOD, 'pretrained'), 'LaBraM-Base': (LABRAM, 'labram'),
          'EEGNet (bd)': (EEGNET_BD, 'eegnet-braindecode'), 'CBraMod scratch (bd)': (CBRAMOD_BD, 'scratch'),
          'CBraMod (bd)': (CBRAMOD_BD, 'pretrained'), 'LaBraM-Base (bd)': (LABRAM_BD, 'labram')}
TABLES = [
    ('released code (EEGNet: our PyTorch implementation)',
     [('CBraMod', 'CBraMod scratch'), ('CBraMod', 'LaBraM-Base'), ('CBraMod', 'EEGNet'), ('LaBraM-Base', 'EEGNet')]),
    ('braindecode implementation (bd) − original implementation, same pipeline',
     [('CBraMod (bd)', 'CBraMod'), ('CBraMod scratch (bd)', 'CBraMod scratch'), ('LaBraM-Base (bd)', 'LaBraM-Base'),
      ('EEGNet (bd)', 'EEGNet')]),
    ('braindecode implementations against each other',
     [('CBraMod (bd)', 'CBraMod scratch (bd)'), ('CBraMod (bd)', 'LaBraM-Base (bd)'), ('CBraMod (bd)', 'EEGNet (bd)'),
      ('LaBraM-Base (bd)', 'EEGNet (bd)')]),
]
DATASETS = {'PhysioNet-MI': 'PhysioNet-MI', 'BCIC-IV-2a': 'BCI-IV-2a',  # results.json key -> RESULTS.md name
            'Mumtaz2016': 'Mumtaz2016', 'MentalArithmetic': 'MentalArithmetic'}
metric = sys.argv[1] if len(sys.argv) > 1 else 'balanced_acc'

runs = {}
for exp in {e for e, _ in MODELS.values()}:
    path = os.path.join(ROOT, exp, 'results/results.json')
    runs[exp] = json.load(open(path))['runs'] if os.path.exists(path) else []


def values(model, ds):
    exp, setting = MODELS[model]
    return [r['metrics'][r['metric_names'].index(metric)] for r in runs[exp]
            if r['dataset'] == ds and r['setting'] == setting and r['seed'] in range(5) and metric in r['metric_names']]


for title, pairs in TABLES:
    print(f'{metric}, {title}: A − B, mean over seeds 0–4, Welch p\n')
    print('| Dataset | ' + ' | '.join(f'{a} − {b}' for a, b in pairs) + ' |')
    print('|---|' + '---|' * len(pairs))
    for ds in DATASETS:
        cells = []
        for a, b in pairs:
            va, vb = values(a, ds), values(b, ds)
            if len(va) != 5 or len(vb) != 5:
                cells.append('–')
                continue
            p = stats.ttest_ind(va, vb, equal_var=False).pvalue
            d = sum(va) / 5 - sum(vb) / 5
            cells.append(f'{d:+.4f} (p {"< 0.001" if p < 0.001 else f"= {p:.3f}"})')
        print(f'| {DATASETS[ds]} | ' + ' | '.join(cells) + ' |')
    print()
