"""Write results/summary.csv: the headline numbers in the repo-wide schema (docs/adding-an-experiment.md).

ours_mean = grand mean over 9 subjects x 3 seeds; ours_std = SD over the 3 seeds of the 9-subject mean.
usage: python code/export_summary.py [results]
"""
import collections, csv, glob, os, sys
import numpy as np

RESULTS = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('EEGCONF_RESULTS', 'results')
PAPER_T2 = [88.19, 61.46, 93.40, 78.13, 52.08, 65.28, 92.36, 88.19, 88.89]
PROTOCOL = 'within-subject, session T -> E'

runs = collections.defaultdict(list)   # (tag, seed) -> curves of the 9 subjects
for f in sorted(glob.glob(os.path.join(RESULTS, 'curve_*.npz'))):
    d = np.load(f)
    if str(d['tag']) in ('A', 'B'):
        runs[(str(d['tag']), int(d['seed']))].append(d)

RULES = [  # setting, protocol tag, reduction of one run's per-epoch test curve, headline
    ('released rule: max test acc over 2000 epochs', 'A', lambda d: d['test'].max(), 1),
    ('mean test acc of the last 100 epochs', 'A', lambda d: d['test'][-100:].mean(), 0),
    ('test acc at the final epoch', 'A', lambda d: d['test'][-1], 0),
    ('test acc at the validation-selected epoch (20% of session T held out)', 'B',
     lambda d: d['test'][int(np.nanargmax(d['val']))], 1),
]

rows = []
for setting, tag, fn, headline in RULES:
    per_seed = [np.mean([fn(d) for d in v]) for (t, _), v in sorted(runs.items()) if t == tag]
    if not per_seed:
        continue
    rows.append(dict(method='EEG-Conformer', dataset='BCI-IV-2a', protocol=PROTOCOL, setting=setting,
                     metric='accuracy', paper_mean=round(np.mean(PAPER_T2) / 100, 4), paper_std='',
                     paper_source='Table II', ours_mean=round(float(np.mean(per_seed)), 4),
                     ours_std=round(float(np.std(per_seed, ddof=1)), 4) if len(per_seed) > 1 else '',
                     n_runs=sum(len(v) for (t, _), v in runs.items() if t == tag), headline=headline))

with open(os.path.join(RESULTS, 'summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f'wrote {len(rows)} rows to', os.path.join(RESULTS, 'summary.csv'))
