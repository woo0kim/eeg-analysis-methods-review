"""Figures of the SEED / SEED-V / STEW round, from the committed results.

  fig A: methods/dgcnn/experiments/seed-paper-protocol/results/fig_selection.png  (what the reported SEED numbers depend on)
  fig B: methods/cbramod/experiments/finetune-seed/results/fig_seed_stew.png      (every model, six dataset / split columns)
usage: python tools/fig_seed_stew.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import compare_seed as cs  # noqa: E402
import compare_stew as ct  # noqa: E402

SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, GRAY = '#2a78d6', '#eb6834', '#9a9993'
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK,
                     'axes.titlesize': 9, 'axes.titleweight': 'bold', 'axes.titlecolor': INK})


def bars(ax, labels, vals, colors, err=None, ref=None, xmax=1.0, fmt='{:.3f}', rows=None):
    rows = rows or len(labels)
    y = np.arange(rows)[::-1][:len(labels)]
    ax.barh(y, vals, height=0.55, color=colors, xerr=err, error_kw=dict(ecolor=INK2, lw=0.8, capsize=2), zorder=2)
    for yi, v, e in zip(y, vals, err if err is not None else [0] * len(vals)):
        if not np.isnan(v):
            ax.text(v + (e or 0) + xmax * 0.012, yi, fmt.format(v), va='center', fontsize=7, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, xmax)
    ax.grid(axis='x', color=GRID, lw=0.6, zorder=0)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(-0.6, rows - 0.4)
    if ref is not None:
        ax.axvline(ref, color=INK2, lw=1, ls=(0, (3, 2)), zorder=3)


def fig_a(path):
    r = json.load(open(os.path.join(ROOT, 'methods/dgcnn/experiments/seed-paper-protocol/results/results.json')))
    sd = {x['rule']: x for x in r if x.get('feature') == 'de_LDS' and 'rule' in x}
    pick = next(x['picked'] for x in r if 'picked' in x)
    cmp_ = next(x['dgcnn_vs_svm'] for x in r if 'dgcnn_vs_svm' in x)
    conf = json.load(open(os.path.join(ROOT, 'methods/eeg-conformer/experiments/seed-5fold/results/results.json')))
    auth = {x['rule']: x for x in json.load(open(os.path.join(ROOT, 'methods/dgcnn/experiments/seed-paper-protocol/results/authors.json')))
            if x['feature'] == 'de_LDS'}
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.1), facecolor=SURFACE, gridspec_kw=dict(width_ratios=[1.25, 1, 1.05], wspace=0.95))
    for ax in axs:
        ax.set_facecolor(SURFACE)
    a = [('Paper (reported)', 90.40, GRAY),
         ("Authors' code, their rule:\ntest peak, best 2 of 3 sessions", auth["peak on test, best 2 of 3 sessions (the authors' rule)"]['authors_mean'], BLUE),
         ("Our port, their rule", sd['peak on test']['best2_mean'], BLUE),
         ('Test peak, all sessions', sd['peak on test']['all3_mean'], BLUE), ('Last step, all sessions', sd['last step']['all3_mean'], BLUE),
         ('Last step, lr from validation', pick['test_last'], BLUE), ('Linear SVM, C from validation', cmp_['sd']['svm'], ORANGE)]
    bars(axs[0], [x[0] for x in a], [x[1] for x in a], [x[2] for x in a], xmax=100, fmt='{:.2f}', rows=7)
    axs[0].set_title('DGCNN, SEED, within subject (%)', loc='left')
    b = [('Paper (reported)', 79.95, GRAY), ('Our port, last step', cmp_['loso']['dgcnn'], BLUE), ('Linear SVM', cmp_['loso']['svm'], ORANGE)]
    bars(axs[1], [x[0] for x in b], [x[1] for x in b], [x[2] for x in b], xmax=100, fmt='{:.2f}', rows=7)
    axs[1].set_title('DGCNN, SEED, leave one subject out (%)', loc='left')
    k, t = conf['kfold'], conf['trial96']
    c = [('Paper (reported)', 95.30, GRAY), ('Released folds,\nbest test epoch', k['best epoch on test (released)']['mean'], BLUE),
         ('Released folds, final epoch', k['final epoch']['mean'], BLUE), ('Clips held out,\nbest test epoch', t['best epoch on test (released)']['mean'], BLUE),
         ('Clips held out, final epoch', t['final epoch']['mean'], BLUE)]
    bars(axs[2], [x[0] for x in c], [x[1] for x in c], [x[2] for x in c], xmax=100, fmt='{:.2f}', rows=7)
    axs[2].set_title('EEG Conformer, SEED, session 1 (%)', loc='left')
    h = [plt.Rectangle((0, 0), 1, 1, color=x) for x in (GRAY, BLUE, ORANGE)]
    fig.legend(h, ['paper', 'run here', 'linear SVM baseline'], loc='lower center', ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.1))
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor=SURFACE)


def fig_b(path):
    a, b = cs.load(), ct.load()
    models = [('LaBraM-Base', 'LaBraM-Base', BLUE), ('CBraMod', 'CBraMod', BLUE), ('CBraMod scratch', 'CBraMod, no pre-training', ORANGE),
              ('DGCNN', 'DGCNN', ORANGE), ('DeepSleepNet (bd)', 'DeepSleepNet (bd)', ORANGE), ('ATCNet (bd)', 'ATCNet (bd)', ORANGE),
              ('EEG Conformer', 'EEG Conformer', ORANGE), ('EEGNet', 'EEGNet', ORANGE)]
    cols = [('SEED, trial split', lambda m: cs.vals(a, 'SEED', 'trial555', m, 'balanced_acc'), 1 / 3),
            ('SEED-V, trial split', lambda m: cs.vals(a, 'SEED-V', 'trial555', m, 'balanced_acc'), 1 / 5),
            ('STEW, rest vs. task', lambda m: ct.vals(b, 'rest-task', m, 'balanced_acc'), 1 / 2),
            ('SEED, subject split', lambda m: cs.vals(a, 'SEED', 'subject', m, 'balanced_acc'), 1 / 3),
            ('SEED-V, subject split', lambda m: cs.vals(a, 'SEED-V', 'subject', m, 'balanced_acc'), 1 / 5),
            ('STEW, 3 rating levels', lambda m: ct.vals(b, 'rating3', m, 'balanced_acc'), 1 / 3)]
    fig, axs = plt.subplots(2, 3, figsize=(11, 6.2), facecolor=SURFACE, gridspec_kw=dict(wspace=0.75, hspace=0.45))
    for ax, (title, get, chance) in zip(axs.ravel(), cols):
        ax.set_facecolor(SURFACE)
        v = [get(m[0]) for m in models]
        keep = [i for i, x in enumerate(v) if x]
        bars(ax, [models[i][1] for i in keep], [np.mean(v[i]) for i in keep], [models[i][2] for i in keep],
             err=[np.std(v[i], ddof=1) for i in keep], ref=chance, xmax=1.0, rows=len(models))
        ax.set_title(title, loc='left')
        ax.set_xlabel('balanced accuracy (mean ± SD)')
    h = [plt.Rectangle((0, 0), 1, 1, color=x) for x in (BLUE, ORANGE)] + [plt.Line2D([0], [0], color=INK2, lw=1, ls=(0, (3, 2)))]
    fig.legend(h, ['pre-trained foundation model', 'trained from scratch', 'chance (balanced accuracy)'], loc='lower center', ncol=3,
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.03))
    fig.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.15, facecolor=SURFACE)


if __name__ == '__main__':
    fig_a(os.path.join(ROOT, 'methods/dgcnn/experiments/seed-paper-protocol/results/fig_selection.png'))
    fig_b(os.path.join(ROOT, 'methods/cbramod/experiments/finetune-seed/results/fig_seed_stew.png'))
    print('written')
