"""Sleep-EDF: every model on the common test set (results/test_epochs.json of cbramod/experiments/finetune-sleepedf:
the 20,976 test epochs of the 8 test subjects that every model scores), read from the committed results.json files.
Feeds SUMMARY.md and the Sleep-EDF report.

Same subject split (62 / 8 / 8) for all. Models in CBraMod's sleep pipeline (encoder + released 20-epoch sequence
head and trainer): seeds 0-4, epoch chosen by validation kappa. DeepSleepNet and L-SeqSleepNet with their own released
training code: 3 runs each (the released code sets no seed), checkpoint by their own rule.
Pairwise tests: Welch's t-test over runs.

usage: python tools/compare_sleepedf.py [--figure <png>]
"""
import json
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = 'methods/{}/experiments/{}'
# (name, model, implementation, training, experiment dir, results.json model key)
MODELS = [
    ('CBraMod', 'CBraMod', 'released', 'pipeline', M.format('cbramod', 'finetune-sleepedf'), 'cbramod'),
    ('CBraMod scratch', 'CBraMod, no pre-training', 'released', 'pipeline', M.format('cbramod', 'finetune-sleepedf'),
     'cbramod-scratch'),
    ('LaBraM-Base', 'LaBraM-Base', 'released', 'pipeline', M.format('labram', 'sleepedf-cbramod-pipeline'), 'labram'),
    ('EEG Conformer', 'EEG Conformer', 'released', 'pipeline', M.format('eeg-conformer', 'sleepedf-cbramod-pipeline'),
     'conformer'),
    ('EEGNet', 'EEGNet', 'ours (PyTorch)', 'pipeline', M.format('eegnet', 'sleepedf-cbramod-pipeline'), 'eegnet'),
    ('ATCNet (bd)', 'ATCNet', 'braindecode', 'pipeline', M.format('atcnet', 'braindecode-sleepedf'), 'bd-atcnet'),
    ('DeepSleepNet (bd)', 'DeepSleepNet', 'braindecode', 'pipeline', M.format('deepsleepnet', 'braindecode-sleepedf'),
     'bd-deepsleepnet'),
    ('DeepSleepNet', 'DeepSleepNet', 'released (TF 1.15)', 'own', M.format('deepsleepnet', 'sleepedf78-cbramod-split'),
     'deepsleepnet'),
    ('L-SeqSleepNet', 'L-SeqSleepNet', 'released (TF 1.15)', 'own',
     M.format('l-seqsleepnet', 'sleepedf78-cbramod-split'), 'l-seqsleepnet'),
    ('CBraMod (bd)', 'CBraMod', 'braindecode', 'pipeline', M.format('cbramod', 'braindecode-sleepedf'), 'bd-cbramod'),
    ('LaBraM-Base (bd)', 'LaBraM-Base', 'braindecode', 'pipeline', M.format('labram', 'braindecode-sleepedf'),
     'bd-labram'),
    ('EEG Conformer (bd)', 'EEG Conformer', 'braindecode', 'pipeline',
     M.format('eeg-conformer', 'braindecode-sleepedf'), 'bd-conformer'),
    ('EEGNet (bd)', 'EEGNet', 'braindecode', 'pipeline', M.format('eegnet', 'braindecode-sleepedf'), 'bd-eegnet'),
    ('DGCNN', 'DGCNN', 'ours (DE input)', 'pipeline', M.format('dgcnn', 'sleepedf-cbramod-pipeline'), 'dgcnn'),
    ('DGCNN (bd)', 'DGCNN', 'braindecode (raw input)', 'pipeline', M.format('dgcnn', 'sleepedf-cbramod-pipeline'), 'bd-dgcnn'),
    ('dDTF-GL', 'dDTF-GL', 'ours (dDTF maps, XGBoost)', 'features', M.format('ddtf-gl', 'sleepedf-cbramod-pipeline'), 'ddtf-gl'),
]
METRICS = [('balanced_acc', 'bal. acc'), ('kappa', 'κ'), ('macro_f1', 'macro F1'), ('accuracy', 'accuracy'),
           ('weighted_f1', 'weighted F1')]
STAGES = ['W', 'N1', 'N2', 'N3', 'REM']
PAIRS = [  # the questions of the Sleep-EDF report
    ('pre-training', [('CBraMod', 'CBraMod scratch'), ('LaBraM-Base', 'CBraMod scratch')]),
    ('foundation models', [('LaBraM-Base', 'CBraMod')]),
    ('sleep models vs. foundation models',
     [('L-SeqSleepNet', 'CBraMod'), ('L-SeqSleepNet', 'LaBraM-Base'), ('DeepSleepNet (bd)', 'CBraMod')]),
    ('sleep models', [('L-SeqSleepNet', 'DeepSleepNet'), ('DeepSleepNet (bd)', 'DeepSleepNet')]),
    ('task-agnostic supervised models vs. CBraMod',
     [('CBraMod', 'ATCNet (bd)'), ('CBraMod', 'EEGNet'), ('CBraMod', 'EEG Conformer'), ('CBraMod', 'DGCNN')]),
    ('dDTF-GL (each epoch alone, no sequence head) vs. the pipeline models', [('EEGNet', 'dDTF-GL'), ('DGCNN', 'dDTF-GL'), ('CBraMod', 'dDTF-GL')]),
    ('braindecode − original implementation, same pipeline',
     [('CBraMod (bd)', 'CBraMod'), ('LaBraM-Base (bd)', 'LaBraM-Base'), ('EEG Conformer (bd)', 'EEG Conformer'),
      ('EEGNet (bd)', 'EEGNet'), ('DGCNN (bd)', 'DGCNN')]),
]


def load():
    """name -> {'runs': {run: {metric: value}}, plus the MODELS fields}; models without results are left out."""
    out, cache = {}, {}
    for name, model, impl, training, exp, key in MODELS:
        path = os.path.join(ROOT, exp, 'results/results.json')
        if path not in cache:
            cache[path] = json.load(open(path))['models'] if os.path.exists(path) else {}
        if key not in cache[path]:
            continue
        r = cache[path][key]
        if training in ('pipeline', 'features'):     # collect.py: common_* keys next to the pipeline's own test epochs
            runs = {int(s): {m: v[f'common_{m}'] for m in [k for k, _ in METRICS] + ['f1_' + s_ for s_ in STAGES]}
                    | {'n_epochs': v['common_n_epochs']} for s, v in r['per_seed'].items()}
        else:                          # split_metrics.score_run(): the 'common' entry
            assert r['epochs'] == 'common'
            runs = {int(s): v for s, v in r['per_seed'].items()}
        out[name] = {'model': model, 'impl': impl, 'training': training, 'exp': exp, 'runs': runs}
    n = {v['n_epochs'] for m in out.values() for v in m['runs'].values()}
    assert len(n) == 1, f'models scored on different epoch sets: {n}'
    return out, n.pop()


def vals(res, name, metric):
    return [v[metric] for _, v in sorted(res[name]['runs'].items())]


def ms(v):
    return np.mean(v), (np.std(v, ddof=1) if len(v) > 1 else float('nan'))


def fmt_p(p):
    return 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'


def tables(res, n_epochs):
    order = sorted(res, key=lambda k: -ms(vals(res, k, 'kappa'))[0])
    print(f'Sleep-EDF-78 (Sleep Cassette), common test set: {n_epochs} epochs of the 8 test subjects; '
          'mean ± SD over runs, sorted by κ\n')
    print('| Model | Implementation | Training | Runs | ' + ' | '.join(l for _, l in METRICS) + ' |')
    print('|---|---|---|---|' + '---|' * len(METRICS))
    for k in order:
        r = res[k]
        cells = ' | '.join('{:.4f} ± {:.4f}'.format(*ms(vals(res, k, m))) for m, _ in METRICS)
        training = {'pipeline': 'CBraMod pipeline', 'own': 'own released code', 'features': 'each epoch alone (no sequence head)'}[r['training']]
        print(f'| {r["model"]} | {r["impl"]} | {training} | {len(r["runs"])} | {cells} |')
    print('\nPer-stage F1 (mean over runs)\n')
    print('| Model | Implementation | ' + ' | '.join(STAGES) + ' |')
    print('|---|---|' + '---|' * len(STAGES))
    for k in order:
        print(f'| {res[k]["model"]} | {res[k]["impl"]} | '
              + ' | '.join(f'{np.mean(vals(res, k, "f1_" + s)):.4f}' for s in STAGES) + ' |')
    tests = []
    for title, pairs in PAIRS:
        pairs = [(a, b) for a, b in pairs if a in res and b in res]
        if not pairs:
            continue
        print(f'\n{title}: A − B, mean over runs, Welch p\n')
        print('| A − B | ' + ' | '.join(l for m, l in METRICS if m in ('balanced_acc', 'kappa', 'macro_f1')) + ' |')
        print('|---|---|---|---|')
        for a, b in pairs:
            cells = []
            for m in ('balanced_acc', 'kappa', 'macro_f1'):
                va, vb = vals(res, a, m), vals(res, b, m)
                d, p = np.mean(va) - np.mean(vb), stats.ttest_ind(va, vb, equal_var=False).pvalue
                tests.append({'a': a, 'b': b, 'metric': m, 'diff': d, 'p': p})
                cells.append(f'{d:+.4f} ({fmt_p(p)})')
            print(f'| {a} − {b} | ' + ' | '.join(cells) + ' |')
    return order, tests


def figure(res, order, path):
    """A: kappa per run and mean ± SD per model; B: per-stage F1 (mean over runs). One row per configuration."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    # validated categorical slots 1-3 (light surface); marker shape is the secondary encoding
    GROUP = {('pipeline', False): ('#2a78d6', 'o', 'released / own code, CBraMod pipeline'),
             ('pipeline', True): ('#eb6834', 's', 'braindecode, CBraMod pipeline'),
             ('own', False): ('#1baf7a', 'D', "sleep model's released training code"),
             ('features', False): ('#eda100', '^', 'no network (dDTF-GL), each epoch alone')}   # slot 4; validated with slots 1-3
    rows = order[::-1]                 # best at the top
    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                         'ytick.color': INK, 'font.family': 'DejaVu Sans'})
    fig, (ax, axh) = plt.subplots(1, 2, figsize=(12.5, 0.5 * len(rows) + 1.9),
                                  gridspec_kw={'width_ratios': [1.3, 1]}, facecolor=SURFACE)
    for a in (ax, axh):
        a.set_facecolor(SURFACE)
    for y, k in enumerate(rows):
        r = res[k]
        color, marker, _ = GROUP[(r['training'], r['impl'] == 'braindecode')]
        v = vals(res, k, 'kappa')
        m, sd = ms(v)
        ax.scatter(v, np.full(len(v), y), s=22, color=color, marker=marker, alpha=0.45, linewidths=0, zorder=2)
        ax.errorbar(m, y, xerr=sd, fmt='none', ecolor=color, elinewidth=2, capsize=0, zorder=3)
        ax.scatter([m], [y], s=70, color=color, marker=marker, edgecolors=SURFACE, linewidths=2, zorder=4)
        ax.annotate(f'{m:.3f}', (max(max(v), m + sd), y), xytext=(7, 0), textcoords='offset points', va='center',
                    fontsize=8.5, color=INK)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xlabel("Cohen's κ on the common test set (20,976 epochs, 8 subjects)")
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis='y', length=0)
    lo = min(min(vals(res, k, 'kappa')) for k in rows)
    hi = max(max(vals(res, k, 'kappa')) for k in rows)
    ax.set_xlim(lo - 0.01, hi + 0.03)
    present = {(res[k]['training'], res[k]['impl'] == 'braindecode') for k in rows}
    handles = [plt.Line2D([], [], marker=g[1], ls='', color=g[0], markersize=8, label=g[2])
               for key, g in GROUP.items() if key in present]
    ax.legend(handles=handles, loc='lower right', frameon=False, fontsize=8.5)
    ax.set_title('A. Agreement with the scorers (dots: runs; bar: ± SD)', loc='left', fontsize=10.5, color=INK)
    F = np.array([[np.mean(vals(res, k, 'f1_' + s)) for s in STAGES] for k in rows])
    vmin, vmax = np.floor(F.min() * 10) / 10 - 0.05, 1.0
    im = axh.imshow(F, cmap='Blues', vmin=vmin, vmax=vmax, aspect='auto', origin='lower')
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            axh.text(j, i, f'{F[i, j]:.2f}', ha='center', va='center', fontsize=8.5,
                     color='white' if F[i, j] > vmin + 0.6 * (vmax - vmin) else INK)
    cb = fig.colorbar(im, ax=axh, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, length=0)
    axh.set_xticks(range(len(STAGES)), STAGES)
    axh.set_yticks(range(len(rows)), rows)
    axh.tick_params(length=0)
    for s in axh.spines.values():
        s.set_visible(False)
    axh.set_title('B. F1 per sleep stage (mean over runs)', loc='left', fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    print(f'\nwrote {path}')


if __name__ == '__main__':
    res, n_epochs = load()
    order, _ = tables(res, n_epochs)
    if '--figure' in sys.argv:
        figure(res, order, sys.argv[sys.argv.index('--figure') + 1])
