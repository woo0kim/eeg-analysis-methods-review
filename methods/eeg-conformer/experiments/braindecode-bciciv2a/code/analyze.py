"""braindecode's EEGConformer vs. the released EEG Conformer on BCI IV 2a: same training loop, seeds and rules.

Reads the braindecode curves (results/curve_*.npz), the released-code curves committed in
../../bciciv2a-epoch-selection/results and the EEGNet curves of methods/eegnet/experiments/bciciv2a-epoch-selection,
prints every table in the README, and writes results/results.json. Accuracy
in %, per subject the mean over seeds 2023-2025; kappa from the grand-mean accuracy (4 classes). braindecode -
released is tested with a two-sided Wilcoxon signed-rank test over the 9 subjects (scipy defaults), the test the
EEG Conformer paper uses.
usage: python code/analyze.py [results] [released_results]
"""
import glob, json, os, sys, warnings
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = sys.argv[1] if len(sys.argv) > 1 else 'results'
RELEASED = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, '../../bciciv2a-epoch-selection/results')
EEGNET = os.path.join(HERE, '../../../../eegnet/experiments/bciciv2a-epoch-selection/results')
PAPER = [88.19, 61.46, 93.40, 78.13, 52.08, 65.28, 92.36, 88.19, 88.89]   # Table II, "Conformer"
SUBJECTS, SEEDS = range(1, 10), (2023, 2024, 2025)
RULES = {  # name -> (protocol tag, reduction of one run's per-epoch curves)
    'max test acc over 2000 epochs (released rule)': ('A', lambda d: d['test'].max()),
    'mean of the last 100 epochs': ('A', lambda d: d['test'][-100:].mean()),
    'final epoch': ('A', lambda d: d['test'][-1]),
    'mean over all 2000 epochs': ('A', lambda d: d['test'].mean()),
    'validation-selected epoch (protocol B)': ('B', lambda d: d['test'][int(np.nanargmax(d['val']))]),
}
NAMES = {'braindecode': 'braindecode EEGConformer', 'released': 'released conformer.py model'}


def load(path):
    runs = {}
    for f in glob.glob(os.path.join(path, 'curve_*.npz')):
        d = np.load(f)
        if str(d['tag']) in ('A', 'B'):
            runs[(str(d['tag']), int(d['subject']), int(d['seed']))] = d
    return runs


models = {'braindecode': load(RESULTS), 'released': load(RELEASED)}
for m, runs in models.items():
    print(f'{NAMES[m]}: {len(runs)} runs ({sum(k[0] == "A" for k in runs)} A, {sum(k[0] == "B" for k in runs)} B)')


def per_run(m, rule):
    """(9 subjects x 3 seeds) array of accuracy in %, NaN where a run is missing."""
    tag, fn = RULES[rule]
    return np.array([[fn(models[m][(tag, s, sd)]) * 100 if (tag, s, sd) in models[m] else np.nan
                      for sd in SEEDS] for s in SUBJECTS])


def kappa(acc):
    return (acc / 100 - 0.25) / 0.75


def wilcoxon(a, b):
    if np.allclose(a, b):
        return float('nan')
    with warnings.catch_warnings():   # 9 pairs with ties or zeros: scipy falls back to the normal approximation
        warnings.simplefilter('ignore')
        return float(stats.wilcoxon(a, b).pvalue)


out = {'paper': {'per_subject': PAPER, 'mean': round(float(np.mean(PAPER)), 2), 'kappa': round(kappa(np.mean(PAPER)), 4)},
       'rules': {}}

print('\nbraindecode EEGConformer per subject, mean ± SD over 3 seeds (accuracy, %)\n')
cols = list(RULES)
print('| Subject | Paper | ' + ' | '.join(cols) + ' |')
print('|---|---|' + '---|' * len(cols))
bd = {r: per_run('braindecode', r) for r in cols}
for i, s in enumerate(SUBJECTS):
    print(f'| {s} | {PAPER[i]:.2f} | ' +
          ' | '.join(f'{np.nanmean(bd[r][i]):.2f} ± {np.nanstd(bd[r][i], ddof=1):.2f}' for r in cols) + ' |')
print(f'| **Avg** | **{np.mean(PAPER):.2f}** | ' + ' | '.join(f'**{np.nanmean(bd[r]):.2f}**' for r in cols) + ' |')

print('\nbraindecode vs. released model, same training loop and rule (grand mean over 9 subjects x 3 seeds)\n')
print('| Rule | braindecode acc. (κ) | released acc. (κ) | braindecode − released | Subjects where braindecode is '
      'higher | Wilcoxon p (9 subjects) | braindecode − paper |')
print('|---|---|---|---|---|---|---|')
for r in cols:
    b, c = np.nanmean(per_run('braindecode', r), axis=1), np.nanmean(per_run('released', r), axis=1)
    rec = {'braindecode': {'per_subject': np.round(b, 2).tolist(), 'mean': round(float(b.mean()), 2),
                           'kappa': round(kappa(b.mean()), 4),
                           'seed_means': np.round(np.nanmean(per_run('braindecode', r), axis=0), 2).tolist()},
           'released': {'per_subject': np.round(c, 2).tolist(), 'mean': round(float(c.mean()), 2),
                        'kappa': round(kappa(c.mean()), 4),
                        'seed_means': np.round(np.nanmean(per_run('released', r), axis=0), 2).tolist()},
           'braindecode_minus_released': round(float((b - c).mean()), 2),
           'subjects_braindecode_higher': int((b > c).sum()), 'wilcoxon_p': round(wilcoxon(b, c), 4),
           'braindecode_minus_paper': round(float(b.mean() - np.mean(PAPER)), 2)}
    out['rules'][r] = rec
    print(f"| {r} | {rec['braindecode']['mean']:.2f} ({rec['braindecode']['kappa']:.4f}) | "
          f"{rec['released']['mean']:.2f} ({rec['released']['kappa']:.4f}) | {rec['braindecode_minus_released']:+.2f} | "
          f"{rec['subjects_braindecode_higher']} / 9 | {rec['wilcoxon_p']:.4f} | {rec['braindecode_minus_paper']:+.2f} |")

print('\nHow much the reported number depends on the rule, and how much the test curve moves, per run '
      '(pp, mean ± SD over 27 runs)\n')
print('| Gap | braindecode | released |')
print('|---|---|---|')
gaps = {'max − last-100 mean (protocol A)': lambda d: d['test'].max() - d['test'][-100:].mean(),
        'max − final epoch (protocol A)': lambda d: d['test'].max() - d['test'][-1],
        'max − validation-selected (protocol B, same run)':
            lambda d: d['test'].max() - d['test'][int(np.nanargmax(d['val']))],
        'SD of test acc over the last 100 epochs (protocol A)': lambda d: d['test'][-100:].std(),
        'training acc on the last batch of an epoch, mean of the last 100 epochs (protocol A)': lambda d: d['train'][-100:].mean()}
out['gaps_pp'] = {}
for g, fn in gaps.items():
    tag = 'B' if 'protocol B' in g else 'A'
    cells = []
    for m in models:
        v = np.array([fn(d) for k, d in models[m].items() if k[0] == tag]) * 100
        out['gaps_pp'].setdefault(g, {})[m] = [round(float(v.mean()), 2), round(float(v.std(ddof=1)), 2), len(v)]
        cells.append(f'{v.mean():.2f} ± {v.std(ddof=1):.2f}')
    print(f'| {g} | ' + ' | '.join(cells) + ' |')

print('\nWhen the curve gets there, and where the maximum lands (protocol A, epochs 0-based as conformer.py prints)\n')
print('| | braindecode | released |')
print('|---|---|---|')
rows = {}
for m in models:
    A = [d for k, d in models[m].items() if k[0] == 'A']
    am = np.array([int(np.argmax(d['test'])) for d in A])
    tail = [d['test'][-100:].mean() for d in A]
    first = np.array([int(np.argmax(d['test'] >= t - 0.01)) for d, t in zip(A, tail)])
    above = np.array([int(np.argmax(d['test'] > 0.30)) for d in A])
    rows[m] = {'median first epoch above 30% test acc': int(np.median(above)),
               'median first epoch within 1 pp of the last-100 mean': int(np.median(first)),
               'median epoch of the maximum': int(np.median(am)), 'range of the epoch of the maximum':
                   [int(am.min()), int(am.max())], 'share of runs whose maximum comes after epoch 1000':
                   round(float((am > 1000).mean()), 2),
               'seconds per epoch (all runs)': round(float(np.mean([float(d['sec_per_epoch'])
                                                                   for d in models[m].values()])), 3)}
out['timing'] = rows
for key in rows['braindecode']:
    print(f'| {key} | {rows["braindecode"][key]} | {rows["released"][key]} |')

print('\nbraindecode EEGConformer vs. EEGNet (../../../eegnet/experiments/bciciv2a-epoch-selection, braindecode EEGNet '
      'in the same loop): does Conformer\'s lead depend on the Conformer implementation?\n')
print('| Rule | braindecode Conformer | EEGNet | braindecode Conformer − EEGNet | Subjects where Conformer is higher | '
      'Wilcoxon p | released Conformer − EEGNet |')
print('|---|---|---|---|---|---|---|')
models['eegnet'] = load(EEGNET)
out['vs_eegnet'] = {}
for r in ('max test acc over 2000 epochs (released rule)', 'mean of the last 100 epochs', 'final epoch',
          'validation-selected epoch (protocol B)'):
    b, e, c = (np.nanmean(per_run(m, r), axis=1) for m in ('braindecode', 'eegnet', 'released'))
    rec = {'braindecode': round(float(b.mean()), 2), 'eegnet': round(float(e.mean()), 2),
           'braindecode_minus_eegnet': round(float((b - e).mean()), 2), 'subjects_braindecode_higher': int((b > e).sum()),
           'wilcoxon_p': round(wilcoxon(b, e), 4), 'released_minus_eegnet': round(float((c - e).mean()), 2)}
    out['vs_eegnet'][r] = rec
    print(f"| {r} | {rec['braindecode']:.2f} | {rec['eegnet']:.2f} | {rec['braindecode_minus_eegnet']:+.2f} | "
          f"{rec['subjects_braindecode_higher']} / 9 | {rec['wilcoxon_p']:.4f} | {rec['released_minus_eegnet']:+.2f} |")

with open(os.path.join(RESULTS, 'results.json'), 'w') as f:
    json.dump(out, f, indent=1)
