"""SEED and SEED-V: every model through one pipeline, read from the committed results.json files. Feeds SUMMARY.md and the report.

Protocols. trial555: CBraMod's released SEED-V split, one model over all subjects, clips 1-5 train, 6-10 validation,
11-15 test in every session (the same subjects appear in all three parts). subject: the same 1-s samples split by subject
(SEED 9 / 3 / 3, SEED-V 10 / 3 / 3), released and baseline implementations only. Seeds 0-4, epoch chosen by validation
kappa, CBraMod's metric code. L-SeqSleepNet classifies 30-s windows (3 runs, its own training): only its clip-level
accuracy compares with the other models.
Pairwise tests: Welch's t-test over runs, without multiple-comparison correction.

usage: python tools/compare_seed.py
"""
import json
import os

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = 'methods/{}/experiments/{}'
# (name, implementation, experiment dir, results.json model key)
MODELS = [
    ('CBraMod', 'released', M.format('cbramod', 'finetune-seed'), 'cbramod'),
    ('CBraMod scratch', 'released, no pre-training', M.format('cbramod', 'finetune-seed'), 'cbramod-scratch'),
    ('LaBraM-Base', 'released', M.format('labram', 'seed-cbramod-pipeline'), 'labram'),
    ('EEG Conformer', 'released', M.format('eeg-conformer', 'seed-cbramod-pipeline'), 'conformer'),
    ('EEGNet', 'ours (PyTorch)', M.format('eegnet', 'seed-cbramod-pipeline'), 'eegnet'),
    ('DGCNN', 'ours (PyTorch, DE input)', M.format('dgcnn', 'seed-cbramod-pipeline'), 'dgcnn'),
    ('ATCNet (bd)', 'braindecode', M.format('atcnet', 'braindecode-seed'), 'bd-atcnet'),
    ('DeepSleepNet (bd)', 'braindecode', M.format('deepsleepnet', 'braindecode-seed'), 'bd-deepsleepnet'),
    ('CBraMod (bd)', 'braindecode', M.format('cbramod', 'braindecode-seed'), 'bd-cbramod'),
    ('LaBraM-Base (bd)', 'braindecode', M.format('labram', 'braindecode-seed'), 'bd-labram'),
    ('EEG Conformer (bd)', 'braindecode', M.format('eeg-conformer', 'braindecode-seed'), 'bd-conformer'),
    ('EEGNet (bd)', 'braindecode', M.format('eegnet', 'braindecode-seed'), 'bd-eegnet'),
    ('DGCNN (bd, raw)', 'braindecode, raw input', M.format('dgcnn', 'braindecode-seed'), 'bd-dgcnn'),
    ('DGCNN (bd, DE)', 'braindecode, DE input', M.format('dgcnn', 'braindecode-seed'), 'bd-dgcnn-de'),
    ('L-SeqSleepNet', 'released, off-label', M.format('l-seqsleepnet', 'seed-offlabel'), 'l-seqsleepnet'),
    ('dDTF-GL', 'ours (dDTF maps, XGBoost; no network)', M.format('ddtf-gl', 'seed-cbramod-pipeline'), 'ddtf-gl'),
]
METRICS = [('balanced_acc', 'bal. acc'), ('kappa', 'κ'), ('weighted_f1', 'weighted F1'), ('accuracy', 'accuracy'),
           ('clip_vote_acc', 'clip vote')]
# CBraMod paper (ICLR 2025), Table 2, SEED-V, 5 seeds: balanced accuracy, kappa, weighted F1
PAPER = {'CBraMod': (0.4091, 0.0097, 0.2569, 0.0143, 0.4101, 0.0108), 'LaBraM-Base': (0.3976, 0.0138, 0.2386, 0.0209, 0.3974, 0.0111),
         'EEG Conformer': (0.3537, 0.0112, 0.1772, 0.0174, 0.3487, 0.0136), 'EEGNet': (0.2961, 0.0102, 0.1006, 0.0143, 0.2749, 0.0098)}
PAIRS = [
    ('pre-training', [('CBraMod', 'CBraMod scratch')]),
    ('foundation models', [('CBraMod', 'LaBraM-Base')]),
    ('foundation models vs. supervised models',
     [('CBraMod', 'EEG Conformer'), ('CBraMod', 'EEGNet'), ('CBraMod', 'DGCNN'), ('LaBraM-Base', 'DGCNN')]),
    ('DGCNN vs. the other supervised models', [('DGCNN', 'EEGNet'), ('DGCNN', 'EEG Conformer'), ('DGCNN', 'ATCNet (bd)')]),
    ('dDTF-GL against the other models', [('dDTF-GL', 'DGCNN'), ('dDTF-GL', 'EEGNet'), ('dDTF-GL', 'CBraMod'), ('dDTF-GL', 'LaBraM-Base')]),
    ('braindecode − original implementation, same pipeline',
     [('CBraMod (bd)', 'CBraMod'), ('LaBraM-Base (bd)', 'LaBraM-Base'), ('EEG Conformer (bd)', 'EEG Conformer'),
      ('EEGNet (bd)', 'EEGNet'), ('DGCNN (bd, DE)', 'DGCNN'), ('DGCNN (bd, raw)', 'DGCNN')]),
]
CHANCE = {'SEED': 1 / 3, 'SEED-V': 1 / 5}


def load():
    res = {}
    for name, impl, exp, key in MODELS:
        path = os.path.join(ROOT, exp, 'results/results.json')
        if not os.path.exists(path):
            continue
        for r in json.load(open(path))['runs']:
            if r['model'] == key:
                res.setdefault((r['dataset'], r['split'], name), []).append(r)
    return res


def vals(res, ds, split, name, metric):
    rs = res.get((ds, split, name), [])
    return [r['metrics'][r['metric_names'].index(metric)] for r in rs]


def ms(v):
    return f'{np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}' if len(v) > 1 else (f'{v[0]:.4f}' if v else '–')


def fmt_p(p):
    return 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'


def main():
    res = load()
    for ds in ('SEED', 'SEED-V'):
        for split, title in (('trial555', 'trial split, as CBraMod on SEED-V'), ('subject', 'subject-independent split')):
            names = [m[0] for m in MODELS if (ds, split, m[0]) in res]
            if not names:
                continue
            names.sort(key=lambda n: (n == 'L-SeqSleepNet', -np.mean(vals(res, ds, split, n, 'clip_vote_acc' if n == 'L-SeqSleepNet' else 'kappa'))))
            print(f'### {ds}, {title} (chance: {CHANCE[ds]:.3f})\n')
            print('| Model | Implementation | Runs | ' + ' | '.join(lab for _, lab in METRICS) + ' |')
            print('|---|---|---|' + '---|' * len(METRICS))
            impl = {m[0]: m[1] for m in MODELS}
            for n in names:
                offlabel = n == 'L-SeqSleepNet'
                cells = [ms(vals(res, ds, split, n, k)) if not offlabel or k == 'clip_vote_acc' else '(30-s windows)' for k, _ in METRICS]
                print(f'| {n} | {impl[n]} | {len(res[(ds, split, n)])} | ' + ' | '.join(cells) + ' |')
            if 'L-SeqSleepNet' in names:
                print('\nL-SeqSleepNet (off-label) classifies 30-s windows in sequences of 20; only its clip vote compares.' +
                      (' On this split its sequences span clips shown to every subject in the same order, so they can read a clip'
                       "'s identity from its position: the number is not valid." if split == 'subject' else ''))
            print()
            if ds == 'SEED-V' and split == 'trial555':
                print('Against the CBraMod paper (Table 2): ours − paper, Welch p from the two means, SDs and n = 5\n')
                print('| Model | bal. acc: paper / ours / Δ (p) | κ: paper / ours / Δ (p) | weighted F1: paper / ours / Δ (p) |')
                print('|---|---|---|---|')
                for n, p in PAPER.items():
                    cells = []
                    for i, k in enumerate(('balanced_acc', 'kappa', 'weighted_f1')):
                        v = vals(res, ds, split, n, k)
                        if len(v) < 2:
                            cells.append('–')
                            continue
                        pm, ps = p[2 * i], p[2 * i + 1]
                        t = stats.ttest_ind_from_stats(np.mean(v), np.std(v, ddof=1), len(v), pm, ps, 5, equal_var=False)
                        cells.append(f'{pm:.4f} / {np.mean(v):.4f} / {np.mean(v) - pm:+.4f} ({fmt_p(t.pvalue)})')
                    print(f'| {n} | ' + ' | '.join(cells) + ' |')
                print()
            for label, pairs in PAIRS:
                lines = []
                for a, b in pairs:
                    cells = []
                    for k in ('balanced_acc', 'kappa', 'clip_vote_acc'):
                        va, vb = vals(res, ds, split, a, k), vals(res, ds, split, b, k)
                        if len(va) < 2 or len(vb) < 2:
                            cells = None
                            break
                        p = stats.ttest_ind(va, vb, equal_var=False).pvalue
                        cells.append(f'{np.mean(va) - np.mean(vb):+.4f} ({fmt_p(p)})')
                    if cells:
                        lines.append(f'| {a} − {b} | ' + ' | '.join(cells) + ' |')
                if lines:
                    print(f'{label}: A − B, mean over runs, Welch p\n')
                    print('| A − B | bal. acc | κ | clip vote |\n|---|---|---|---|')
                    print('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
