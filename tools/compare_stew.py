"""STEW: every model on the same 7 subject-independent folds (GroupKFold, pinned in
methods/cbramod/experiments/finetune-stew/results/folds.json), read from the committed results.json files. Feeds SUMMARY.md and the report.

Two tasks: rest vs. task (2 classes, 48 subjects) and 3 rating levels (45 subjects). Per seed, the 7 test folds cover every window once; a
seed's score is computed on its pooled predictions. Seeds 0-4, checkpoint = best validation kappa (validation = the next fold).
Pairwise tests: Welch's t-test over seeds, without multiple-comparison correction.
usage: python tools/compare_stew.py
"""
import json
import os

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = 'methods/{}/experiments/{}'
MODELS = [
    ('CBraMod', 'released', M.format('cbramod', 'finetune-stew'), 'cbramod'),
    ('CBraMod scratch', 'released, no pre-training', M.format('cbramod', 'finetune-stew'), 'cbramod-scratch'),
    ('LaBraM-Base', 'released', M.format('labram', 'stew-cbramod-pipeline'), 'labram'),
    ('EEG Conformer', 'released model', M.format('eeg-conformer', 'stew-cbramod-pipeline'), 'conformer'),
    ('EEGNet', 'ours (PyTorch)', M.format('eegnet', 'stew-cbramod-pipeline'), 'eegnet'),
    ('DGCNN', 'ours (PyTorch, DE input)', M.format('dgcnn', 'stew-cbramod-pipeline'), 'dgcnn'),
    ('ATCNet (bd)', 'braindecode', M.format('atcnet', 'braindecode-stew'), 'bd-atcnet'),
    ('DeepSleepNet (bd)', 'braindecode', M.format('deepsleepnet', 'braindecode-stew'), 'bd-deepsleepnet'),
    ('CBraMod (bd)', 'braindecode', M.format('cbramod', 'braindecode-stew'), 'bd-cbramod'),
    ('LaBraM-Base (bd)', 'braindecode', M.format('labram', 'braindecode-stew'), 'bd-labram'),
    ('EEG Conformer (bd)', 'braindecode', M.format('eeg-conformer', 'braindecode-stew'), 'bd-conformer'),
    ('EEGNet (bd)', 'braindecode', M.format('eegnet', 'braindecode-stew'), 'bd-eegnet'),
    ('DGCNN (bd, raw)', 'braindecode, raw input', M.format('dgcnn', 'braindecode-stew'), 'bd-dgcnn'),
    ('DGCNN (bd, DE)', 'braindecode, DE input', M.format('dgcnn', 'braindecode-stew'), 'bd-dgcnn-de'),
    ('L-SeqSleepNet', 'released, off-label', M.format('l-seqsleepnet', 'stew-offlabel'), 'l-seqsleepnet'),
]
METRICS = [('accuracy', 'accuracy'), ('balanced_acc', 'bal. acc'), ('kappa', 'κ'), ('weighted_f1', 'weighted F1')]
PAIRS = [('pre-training', [('CBraMod', 'CBraMod scratch')]), ('foundation models', [('LaBraM-Base', 'CBraMod')]),
         ('foundation models vs. supervised models', [('CBraMod', 'DGCNN'), ('CBraMod', 'EEG Conformer'), ('CBraMod', 'EEGNet'), ('LaBraM-Base', 'DGCNN')]),
         ('DGCNN vs. the other supervised models', [('DGCNN', 'EEG Conformer'), ('DGCNN', 'EEGNet'), ('DGCNN', 'ATCNet (bd)')]),
         ('braindecode − original implementation, same pipeline',
          [('CBraMod (bd)', 'CBraMod'), ('LaBraM-Base (bd)', 'LaBraM-Base'), ('EEG Conformer (bd)', 'EEG Conformer'), ('EEGNet (bd)', 'EEGNet'),
           ('DGCNN (bd, DE)', 'DGCNN'), ('DGCNN (bd, raw)', 'DGCNN')])]
TASK = {'rest-task': ('rest vs. task, 48 subjects', 0.5), 'rating3': ('3 rating levels, 45 subjects', 1 / 3)}


def load():
    res = {}
    for name, impl, exp, key in MODELS:
        p = os.path.join(ROOT, exp, 'results/results.json')
        if os.path.exists(p):
            for r in json.load(open(p))['runs']:
                if r['model'] == key:
                    res.setdefault((r['task'], name), []).append(r)
    return res


def vals(res, task, name, metric):
    return [r['metrics'][r['metric_names'].index(metric)] for r in res.get((task, name), [])]


def ms(v):
    return f'{np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}' if len(v) > 1 else (f'{v[0]:.4f}' if v else '–')


def main():
    res = load()
    impl = {m[0]: m[1] for m in MODELS}
    for task, (title, chance) in TASK.items():
        names = sorted((m[0] for m in MODELS if (task, m[0]) in res), key=lambda n: (n == 'L-SeqSleepNet', -np.mean(vals(res, task, n, 'kappa'))))
        print(f'### STEW, {title} (chance accuracy {chance:.3f})\n')
        print('| Model | Implementation | Seeds | ' + ' | '.join(l for _, l in METRICS) + ' |\n|---|---|---|' + '---|' * len(METRICS))
        for n in names:
            print(f'| {n} | {impl[n]} | {len(res[(task, n)])} | ' + ' | '.join(ms(vals(res, task, n, k)) for k, _ in METRICS) + ' |')
        if 'L-SeqSleepNet' in names:
            print('\nL-SeqSleepNet (off-label, 3 runs) classifies 30-s windows, five per recording in one sequence, not the 4-s windows of the '
                  'other models; its recording vote is in l-seqsleepnet/experiments/stew-offlabel.')
        print()
        for label, pairs in PAIRS:
            lines = []
            for a, b in pairs:
                va, vb = vals(res, task, a, 'accuracy'), vals(res, task, b, 'accuracy')
                ka, kb = vals(res, task, a, 'kappa'), vals(res, task, b, 'kappa')
                if min(len(va), len(vb)) < 2:
                    continue
                pa, pk = stats.ttest_ind(va, vb, equal_var=False).pvalue, stats.ttest_ind(ka, kb, equal_var=False).pvalue
                f = lambda p: 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'
                lines.append(f'| {a} − {b} | {np.mean(va) - np.mean(vb):+.4f} ({f(pa)}) | {np.mean(ka) - np.mean(kb):+.4f} ({f(pk)}) |')
            if lines:
                print(f'{label}: A − B, mean over seeds, Welch p\n\n| A − B | accuracy | κ |\n|---|---|---|\n' + '\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
