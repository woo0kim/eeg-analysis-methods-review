"""EEGNet vs. EEG Conformer on BCI IV 2a: same training loop, same seeds, same epoch-selection rules.

Reads the EEGNet curves (results/curve_*.npz) and the EEG Conformer curves committed in its experiment, prints
every table in the README, and writes results/results.json. Accuracy in %, per subject the mean over seeds
2023-2025; kappa from the grand-mean accuracy (4 classes). Conformer - EEGNet is tested with a two-sided
Wilcoxon signed-rank test over the 9 subjects (scipy defaults), the test the EEG Conformer paper uses.
usage: python code/analyze.py [results] [conformer_results]
"""
import glob, json, os, sys, warnings
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = sys.argv[1] if len(sys.argv) > 1 else 'results'
CONF = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    HERE, '../../../../eeg-conformer/experiments/bciciv2a-epoch-selection/results')
# EEG Conformer paper (Song et al., IEEE TNSRE 2023), Table II, Dataset I: rows "EEGNet [17]" and "Conformer"
PAPER = {'EEGNet': [85.76, 61.46, 88.54, 67.01, 55.90, 52.08, 89.58, 83.33, 86.81],
         'Conformer': [88.19, 61.46, 93.40, 78.13, 52.08, 65.28, 92.36, 88.19, 88.89]}
SUBJECTS, SEEDS = range(1, 10), (2023, 2024, 2025)
RULES = {  # name -> (protocol tag, reduction of one run's per-epoch curves)
    'max test acc over 2000 epochs (released rule)': ('A', lambda d: d['test'].max()),
    'mean of the last 100 epochs': ('A', lambda d: d['test'][-100:].mean()),
    'final epoch': ('A', lambda d: d['test'][-1]),
    'validation-selected epoch (protocol B)': ('B', lambda d: d['test'][int(np.nanargmax(d['val']))]),
}
VALSEL = 'validation-selected epoch (protocol B)'


def load(path):
    runs = {}
    for f in glob.glob(os.path.join(path, 'curve_*.npz')):
        d = np.load(f)
        if str(d['tag']) in ('A', 'B'):
            runs[(str(d['tag']), int(d['subject']), int(d['seed']))] = d
    return runs


models = {'EEGNet': load(RESULTS), 'Conformer': load(CONF)}
for m, runs in models.items():
    print(f'{m}: {len(runs)} runs ({sum(k[0] == "A" for k in runs)} A, {sum(k[0] == "B" for k in runs)} B)')


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


out = {'paper': {m: {'per_subject': v, 'mean': round(float(np.mean(v)), 2), 'kappa': round(kappa(np.mean(v)), 4)}
                 for m, v in PAPER.items()}, 'rules': {}}
p = np.array(PAPER['Conformer']) - np.array(PAPER['EEGNet'])
out['paper']['conformer_minus_eegnet'] = {'mean': round(float(p.mean()), 2),
                                          'wilcoxon_p': round(wilcoxon(PAPER['Conformer'], PAPER['EEGNet']), 4),
                                          'subjects_conformer_higher': int((p > 0).sum())}

print('\nEEGNet per subject, mean ± SD over 3 seeds (accuracy, %)\n')
cols = list(RULES)
print('| Subject | Paper | ' + ' | '.join(cols) + ' |')
print('|---|---|' + '---|' * len(cols))
eeg = {r: per_run('EEGNet', r) for r in cols}
for i, s in enumerate(SUBJECTS):
    print(f'| {s} | {PAPER["EEGNet"][i]:.2f} | ' +
          ' | '.join(f'{np.nanmean(eeg[r][i]):.2f} ± {np.nanstd(eeg[r][i], ddof=1):.2f}' for r in cols) + ' |')
print(f'| **Avg** | **{np.mean(PAPER["EEGNet"]):.2f}** | ' +
      ' | '.join(f'**{np.nanmean(eeg[r]):.2f}**' for r in cols) + ' |')

print('\nEEGNet vs. EEG Conformer, same training loop and rule (grand mean over 9 subjects x 3 seeds)\n')
print('| Rule | EEGNet acc. (κ) | Conformer acc. (κ) | Conformer − EEGNet | Subjects where Conformer is higher | '
      'Wilcoxon p (9 subjects) |')
print('|---|---|---|---|---|---|')
pp = out['paper']
print(f"| Paper, Table II | {pp['EEGNet']['mean']:.2f} ({pp['EEGNet']['kappa']:.4f}) | {pp['Conformer']['mean']:.2f} "
      f"({pp['Conformer']['kappa']:.4f}) | {pp['conformer_minus_eegnet']['mean']:+.2f} | "
      f"{pp['conformer_minus_eegnet']['subjects_conformer_higher']} / 9 | {pp['conformer_minus_eegnet']['wilcoxon_p']:.4f} |")
for r in cols:
    e, c = np.nanmean(per_run('EEGNet', r), axis=1), np.nanmean(per_run('Conformer', r), axis=1)
    rec = {'eegnet': {'per_subject': np.round(e, 2).tolist(), 'mean': round(float(e.mean()), 2),
                      'kappa': round(kappa(e.mean()), 4),
                      'seed_means': np.round(np.nanmean(per_run('EEGNet', r), axis=0), 2).tolist()},
           'conformer': {'per_subject': np.round(c, 2).tolist(), 'mean': round(float(c.mean()), 2),
                         'kappa': round(kappa(c.mean()), 4)},
           'conformer_minus_eegnet': round(float((c - e).mean()), 2), 'subjects_conformer_higher': int((c > e).sum()),
           'wilcoxon_p': round(wilcoxon(c, e), 4)}
    out['rules'][r] = rec
    print(f"| {r} | {rec['eegnet']['mean']:.2f} ({rec['eegnet']['kappa']:.4f}) | {rec['conformer']['mean']:.2f} "
          f"({rec['conformer']['kappa']:.4f}) | {rec['conformer_minus_eegnet']:+.2f} | "
          f"{rec['subjects_conformer_higher']} / 9 | {rec['wilcoxon_p']:.4f} |")

print('\nHow much the reported number depends on the rule, and how much the test curve moves, per run '
      '(pp, mean ± SD over 27 runs)\n')
print('| Gap | EEGNet | Conformer |')
print('|---|---|---|')
gaps = {'max − last-100 mean (protocol A)': lambda d: d['test'].max() - d['test'][-100:].mean(),
        'max − final epoch (protocol A)': lambda d: d['test'].max() - d['test'][-1],
        'max − validation-selected (protocol B, same run)':
            lambda d: d['test'].max() - d['test'][int(np.nanargmax(d['val']))],
        'SD of test acc over the last 100 epochs (protocol A)': lambda d: d['test'][-100:].std()}
out['gaps_pp'] = {}
for g, fn in gaps.items():
    tag = 'B' if 'protocol B' in g else 'A'
    cells = []
    for m in models:
        v = np.array([fn(d) for k, d in models[m].items() if k[0] == tag]) * 100
        out['gaps_pp'].setdefault(g, {})[m] = [round(float(v.mean()), 2), round(float(v.std(ddof=1)), 2), len(v)]
        cells.append(f'{v.mean():.2f} ± {v.std(ddof=1):.2f}')
    print(f'| {g} | ' + ' | '.join(cells) + ' |')

print('\nProtocol B runs (80% of session T for training, 58 validation trials): what validation selection adds\n')
print('| | EEGNet | Conformer |')
print('|---|---|---|')
rows = {}
for m in models:
    B = [d for k, d in models[m].items() if k[0] == 'B']
    vsel = np.array([int(np.nanargmax(d['val'])) for d in B])     # first epoch at the maximum, as conformer.py
    tail = np.mean([d['test'][-100:].mean() for d in B]) * 100
    sel = np.mean([d['test'][i] for d, i in zip(B, vsel)]) * 100
    rows[m] = {'test acc, mean of the last 100 epochs': round(float(tail), 2),
               'test acc at the validation-selected epoch': round(float(sel), 2),
               'validation-selected − last-100 mean (pp)': round(float(sel - tail), 2),
               'median validation-selected epoch (0-based)': int(np.median(vsel)),
               'epochs tied at the maximum validation acc (mean per run)':
                   round(float(np.mean([(d['val'] == np.nanmax(d['val'])).sum() for d in B])), 1)}
out['protocol_b'] = rows
for key in rows['EEGNet']:
    print(f'| {key} | {rows["EEGNet"][key]} | {rows["Conformer"][key]} |')

print('\nWhere the maximum lands (protocol A)\n')
print('| | EEGNet | Conformer |')
print('|---|---|---|')
rows = {}
for m in models:
    A = [d for k, d in models[m].items() if k[0] == 'A']
    am = np.array([int(np.argmax(d['test'])) for d in A])            # 0-based, as conformer.py prints epochs
    tail = [d['test'][-100:].mean() for d in A]
    first = np.array([int(np.argmax(d['test'] >= t - 0.01)) for d, t in zip(A, tail)])
    rows[m] = {'median_argmax_epoch': int(np.median(am)), 'argmax_range': [int(am.min()), int(am.max())],
               'share_argmax_after_1000': round(float((am > 1000).mean()), 2),
               'median_first_epoch_within_1pp_of_last100': int(np.median(first)),
               'sec_per_epoch (all runs)': round(float(np.mean([float(d['sec_per_epoch'])
                                                           for d in models[m].values()])), 3)}
out['argmax'] = rows
for key in rows['EEGNet']:
    print(f'| {key} | {rows["EEGNet"][key]} | {rows["Conformer"][key]} |')

with open(os.path.join(RESULTS, 'results.json'), 'w') as f:
    json.dump(out, f, indent=1)
