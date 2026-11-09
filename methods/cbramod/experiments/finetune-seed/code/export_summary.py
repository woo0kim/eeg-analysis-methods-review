"""Write <results>/summary.csv from <results>/results.json (collect.py) in the repo-wide schema (docs/adding-an-experiment.md).

Paper numbers: only the CBraMod paper reports SEED-V (Table 2: CBraMod, LaBraM-Base, EEGNet, EEG Conformer; trial split 5:5:5). No paper
here reports SEED in this pipeline; DGCNN's and EEG Conformer's SEED numbers come from other protocols (their own experiments).
usage: python export_summary.py <results dir>
"""
import csv
import json
import os
import sys

METHOD = {'cbramod': ('CBraMod', 'released code and weights'), 'cbramod-scratch': ('CBraMod', 'released code, no pre-training'),
          'labram': ('LaBraM', "released code and weights, LaBraM's fine-tuning recipe"), 'conformer': ('EEG-Conformer', 'released SEED model'),
          'eegnet': ('EEGNet', 'our PyTorch EEGNet-8,2'), 'dgcnn': ('DGCNN', 'our DGCNN on 5-band DE'),
          'bd-cbramod': ('CBraMod', 'braindecode, its pre-trained weights'), 'bd-labram': ('LaBraM', "braindecode, its pre-trained weights, LaBraM's recipe"),
          'bd-conformer': ('EEG-Conformer', 'braindecode'), 'bd-eegnet': ('EEGNet', 'braindecode'), 'bd-atcnet': ('ATCNet', 'braindecode'),
          'bd-deepsleepnet': ('DeepSleepNet', 'braindecode'), 'bd-dgcnn': ('DGCNN', 'braindecode, raw samples'),
          'bd-dgcnn-de': ('DGCNN', 'braindecode, 5-band DE')}
HEADLINE = {'cbramod', 'labram', 'conformer', 'eegnet', 'dgcnn', 'bd-atcnet', 'bd-deepsleepnet'}
PROTOCOL = {('SEED', 'trial555'): 'all subjects pooled, clips 1-5 / 6-10 / 11-15 of every session', ('SEED-V', 'trial555'): 'all subjects pooled, clips 1-5 / 6-10 / 11-15 of every session',
            ('SEED', 'subject'): 'cross-subject 9/3/3', ('SEED-V', 'subject'): 'cross-subject 10/3/3'}
PAPER = {'cbramod': (0.4091, 0.0097, 0.2569, 0.0143, 0.4101, 0.0108), 'labram': (0.3976, 0.0138, 0.2386, 0.0209, 0.3974, 0.0111),
         'eegnet': (0.2961, 0.0102, 0.1006, 0.0143, 0.2749, 0.0098), 'conformer': (0.3537, 0.0112, 0.1772, 0.0174, 0.3487, 0.0136)}


def main():
    RESULTS = sys.argv[1]
    rows = []
    for s in json.load(open(os.path.join(RESULTS, 'results.json')))['summary']:
        method, impl = METHOD[s['model']]
        for i, metric in enumerate(s['metric_names']):
            if metric == 'macro_f1':
                continue
            p = PAPER.get(s['model']) if (s['dataset'], s['split']) == ('SEED-V', 'trial555') and metric in ('balanced_acc', 'kappa', 'weighted_f1') else None
            j = ['balanced_acc', 'kappa', 'weighted_f1'].index(metric) if p else 0
            rows.append(dict(method=method, dataset=s['dataset'], protocol=PROTOCOL[(s['dataset'], s['split'])],
                             setting=f"{impl}, CBraMod's SEED-V pipeline, 1-s samples", metric=metric,
                             paper_mean=p[2 * j] if p else '', paper_std=p[2 * j + 1] if p else '', paper_source='CBraMod paper, Table 2' if p else '',
                             ours_mean=round(s['mean'][i], 4), ours_std=round(s['std'][i], 4) if s['n'] > 1 else '', n_runs=s['n'],
                             headline=int(metric == 'balanced_acc' and s['model'] in HEADLINE)))
    with open(os.path.join(RESULTS, 'summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {len(rows)} rows to', os.path.join(RESULTS, 'summary.csv'))


if __name__ == '__main__':
    main()
