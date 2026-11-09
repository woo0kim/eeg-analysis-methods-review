"""Write results/summary.csv from results/results.json in the repo-wide schema (docs/adding-an-experiment.md).

No paper in this repository reports Sleep-EDF, so the paper columns are empty.
usage: python code/export_summary.py [results]
"""
import csv, json, os, sys

RESULTS = sys.argv[1] if len(sys.argv) > 1 else 'results'
# model key -> (method name used in the other summary.csv files, setting)
SETTING = {
    'cbramod': ('CBraMod', "fine-tuned from the released weights, CBraMod's released sleep-staging (ISRUC) pipeline"),
    'cbramod-scratch': ('CBraMod', "trained from random init, CBraMod's released sleep-staging (ISRUC) pipeline"),
    'labram': ('LaBraM', "released LaBraM-Base weights as the epoch encoder (two 15-s halves) in CBraMod's sleep-staging pipeline"),
    'eegnet': ('EEGNet', "our PyTorch EEGNet-8,2 as the epoch encoder in CBraMod's sleep-staging pipeline"),
    'conformer': ('EEG-Conformer', "released EEG Conformer (conv + 6 attention blocks) as the epoch encoder in CBraMod's sleep-staging pipeline"),
    'bd-atcnet': ('ATCNet', "braindecode ATCNet (defaults) as the epoch encoder in CBraMod's sleep-staging pipeline"),
    'bd-cbramod': ('CBraMod', "braindecode CBraMod, fine-tuned from braindecode's pretrained weights, CBraMod's sleep-staging pipeline"),
    'bd-labram': ('LaBraM', "braindecode Labram with braindecode's pretrained weights as the epoch encoder, CBraMod's sleep-staging pipeline"),
    'bd-eegnet': ('EEGNet', "braindecode EEGNet-8,2 as the epoch encoder in CBraMod's sleep-staging pipeline"),
    'bd-conformer': ('EEG-Conformer', "braindecode EEGConformer as the epoch encoder in CBraMod's sleep-staging pipeline"),
    'dgcnn': ('DGCNN', "our DGCNN on 5-band DE of each 1-s segment as the epoch encoder in CBraMod's sleep-staging pipeline"),
    'bd-dgcnn': ('DGCNN', "braindecode DGCNN (raw input) as the epoch encoder in CBraMod's sleep-staging pipeline"),
    'bd-deepsleepnet': ('DeepSleepNet', "braindecode DeepSleepNet (single-epoch; first-layer kernels scaled to 200 Hz) as the epoch encoder in CBraMod's sleep-staging pipeline"),
}
METRICS = ['balanced_acc', 'kappa', 'weighted_f1', 'accuracy', 'macro_f1']
HEADLINE = {'cbramod', 'labram', 'eegnet', 'conformer', 'bd-atcnet', 'bd-deepsleepnet', 'dgcnn'}  # one row per method in the pipeline (ATCNet, DeepSleepNet: braindecode only)

rows = []
for key, r in json.load(open(os.path.join(RESULTS, 'results.json')))['models'].items():
    method, setting = SETTING[key]
    for metric in METRICS:
        rows.append(dict(method=method, dataset='Sleep-EDF', protocol='cross-subject 62/8/8', setting=setting,
                         metric=metric, paper_mean='', paper_std='', paper_source='',
                         ours_mean=round(r['mean'][metric], 4),
                         ours_std=round(r['std'][metric], 4) if len(r['seeds']) > 1 else '',
                         n_runs=len(r['seeds']),
                         headline=int(metric == 'balanced_acc' and key in HEADLINE)))

with open(os.path.join(RESULTS, 'summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f'wrote {len(rows)} rows to', os.path.join(RESULTS, 'summary.csv'))
