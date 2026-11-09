"""Collect the Sleep-EDF runs: every number in the README, recomputed from the committed per-run files.

usage: python collect.py <runs dir> <out dir>  > <out dir>/results.txt
  <runs dir>/SleepEDF__<model>__s<seed>/metrics.json   per-epoch validation and test metrics (seq_trainer.py)
  <runs dir>/SleepEDF__<model>__s<seed>/test_preds.npz test labels and predictions at the selected epoch
writes <out dir>/results.json. Needs numpy, scipy, scikit-learn.

Reported number = the released rule: test score at the epoch with the highest validation kappa (first such epoch).
The same predictions are also scored on the common test set (results/test_epochs.json, epoch_keys.py): the test
epochs that every model in the Sleep-EDF comparison scores, including DeepSleepNet and L-SeqSleepNet (common_* keys).
Also reported, from the same runs: the test score at the best *test* epoch (selection on the test set, for comparison
with test-selected protocols) and at the last epoch; plain accuracy, macro-F1 and per-stage F1 at the selected epoch
(the metrics sleep-staging papers usually report). Tests: Welch's t-test over seeds between models.
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import epoch_keys  # noqa: E402
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score

MODELS = [  # (key, label, implementation)
    ('eegnet', 'EEGNet', 'ours (PyTorch)'), ('conformer', 'EEG Conformer', 'released'),
    ('bd-atcnet', 'ATCNet', 'braindecode'), ('labram', 'LaBraM-Base', 'released'),
    ('cbramod', 'CBraMod', 'released'), ('cbramod-scratch', 'CBraMod, no pre-training', 'released'),
    ('bd-eegnet', 'EEGNet', 'braindecode'), ('bd-conformer', 'EEG Conformer', 'braindecode'),
    ('bd-labram', 'LaBraM-Base', 'braindecode'), ('bd-cbramod', 'CBraMod', 'braindecode'),
    ('bd-deepsleepnet', 'DeepSleepNet', 'braindecode'),
    ('dgcnn', 'DGCNN', 'ours (PyTorch), DE input'), ('bd-dgcnn', 'DGCNN', 'braindecode, raw input'),
]
STAGES = ['W', 'N1', 'N2', 'N3', 'REM']
METRICS = ['balanced_acc', 'kappa', 'weighted_f1']


def load(runs):
    out = defaultdict(dict)
    for d in sorted(glob.glob(os.path.join(runs, 'SleepEDF__*__s*'))):
        _, model, s = os.path.basename(d).split('__')
        mfile, pfile = os.path.join(d, 'metrics.json'), os.path.join(d, 'test_preds.npz')
        if not (os.path.exists(mfile) and os.path.exists(pfile)):
            continue                                    # unfinished run
        hist = json.load(open(mfile))
        p = np.load(pfile)
        out[model][int(s[1:])] = (hist, p['y'], p['pred'], int(p['best_epoch']))
    return out


def run_numbers(hist, y, pred, best_epoch):
    val_k = [h['val']['kappa'] for h in hist]
    sel = int(np.argmax(val_k))                       # first maximum = the released `kappa > kappa_best` rule
    assert sel + 1 == best_epoch, (sel + 1, best_epoch)
    r = {'epochs': len(hist), 'selected_epoch': sel + 1}
    for m in METRICS:
        r[m] = hist[sel]['test'][m]
        r[m + '@best_test'] = max(h['test'][m] for h in hist)
        r[m + '@last'] = hist[-1]['test'][m]
    r['best_test_epoch_kappa'] = int(np.argmax([h['test']['kappa'] for h in hist])) + 1
    # the saved predictions reproduce the logged test metrics of the selected epoch
    assert abs(balanced_accuracy_score(y, pred) - r['balanced_acc']) < 1e-9
    assert abs(cohen_kappa_score(y, pred) - r['kappa']) < 1e-9
    r['accuracy'] = accuracy_score(y, pred)
    r['macro_f1'] = f1_score(y, pred, average='macro')
    for st, f in zip(STAGES, f1_score(y, pred, average=None, labels=range(5))):
        r['f1_' + st] = f
    r['minutes'] = sum(h['minutes'] for h in hist)
    keys, labels, mask = COMMON
    assert np.array_equal(y, labels), 'test_preds.npz order differs from epoch_keys.pipeline_test_keys()'
    yc, pc = y[mask], pred[mask]
    r.update({'common_' + k: v for k, v in (('balanced_acc', balanced_accuracy_score(yc, pc)),
                                            ('kappa', cohen_kappa_score(yc, pc)),
                                            ('weighted_f1', f1_score(yc, pc, average='weighted')),
                                            ('accuracy', accuracy_score(yc, pc)),
                                            ('macro_f1', f1_score(yc, pc, average='macro')))})
    for st, f in zip(STAGES, f1_score(yc, pc, average=None, labels=range(5))):
        r['common_f1_' + st] = f
    r['common_n_epochs'] = int(mask.sum())
    return r


def ms(v):
    v = np.asarray(v, float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else float('nan'))


def welch(a, b):
    return stats.ttest_ind(a, b, equal_var=False).pvalue if len(a) > 1 and len(b) > 1 else float('nan')


def fmt_p(p):
    return 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'


COMMON = None


def main():
    global COMMON
    COMMON = epoch_keys.load_test_epochs()
    runs, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    data = load(runs)
    res = {}
    for key, label, impl in MODELS:
        if key not in data:
            continue
        per = {s: run_numbers(*v) for s, v in sorted(data[key].items())}
        keys = per[next(iter(per))].keys()
        res[key] = {'label': label, 'implementation': impl, 'seeds': sorted(per), 'per_seed': per,
                    'mean': {k: ms([per[s][k] for s in per])[0] for k in keys},
                    'std': {k: ms([per[s][k] for s in per])[1] for k in keys}}

    print('Sleep-EDF Expanded, Sleep Cassette (78 subjects, 153 nights), CBraMod sleep-staging pipeline')
    print('test score at the epoch with the best validation kappa (released rule); mean +- SD over seeds\n')
    hdr = f'{"model":28s} {"impl.":15s} {"n":>2s}  {"bal. acc":>15s}  {"kappa":>15s}  {"weighted F1":>15s}  {"accuracy":>15s}  {"macro F1":>15s}'
    print(hdr)
    for key, r in res.items():
        cells = '  '.join(f'{r["mean"][m]:.4f} +- {r["std"][m]:.4f}' for m in METRICS + ['accuracy', 'macro_f1'])
        print(f'{r["label"]:28s} {r["implementation"]:15s} {len(r["seeds"]):2d}  {cells}')

    print('\nper-stage F1 at the selected epoch (mean over seeds)')
    print(f'{"model":28s} {"impl.":15s} ' + ' '.join(f'{s:>7s}' for s in STAGES))
    for key, r in res.items():
        print(f'{r["label"]:28s} {r["implementation"]:15s} ' + ' '.join(f'{r["mean"]["f1_" + s]:7.4f}' for s in STAGES))

    print('\nepoch-selection rule, kappa (mean over seeds): validation-selected | best test epoch | last epoch')
    for key, r in res.items():
        m = r['mean']
        print(f'{r["label"]:28s} {r["implementation"]:15s} {m["kappa"]:.4f} | {m["kappa@best_test"]:.4f} '
              f'(+{m["kappa@best_test"] - m["kappa"]:.4f}) | {m["kappa@last"]:.4f}   '
              f'selected epochs {[r["per_seed"][s]["selected_epoch"] for s in r["seeds"]]}')

    def pair_table(title, pairs, metric):
        pairs = [(a, b) for a, b in pairs if a in res and b in res]   # a runs dir holds one experiment's models
        if pairs:
            print(f'\n{title} ({metric}, A - B, Welch p over seeds)')
        rows = []
        for a, b in pairs:
            va = [res[a]['per_seed'][s][metric] for s in res[a]['seeds']]
            vb = [res[b]['per_seed'][s][metric] for s in res[b]['seeds']]
            d, p = np.mean(va) - np.mean(vb), welch(va, vb)
            rows.append({'a': a, 'b': b, 'metric': metric, 'diff': d, 'p': p})
            print(f'  {res[a]["label"]} ({res[a]["implementation"]}) - {res[b]["label"]} '
                  f'({res[b]["implementation"]}): {d:+.4f} ({fmt_p(p)})')
        return rows

    tests = []
    released = ['cbramod', 'labram', 'cbramod-scratch', 'conformer', 'bd-atcnet', 'eegnet']
    pairs = [(a, b) for i, a in enumerate(released) for b in released[i + 1:]]
    bd_pairs = [('bd-eegnet', 'eegnet'), ('bd-conformer', 'conformer'), ('bd-labram', 'labram'), ('bd-cbramod', 'cbramod')]
    for metric in ('balanced_acc', 'kappa', 'macro_f1'):
        tests += pair_table('between models', pairs, metric)
    for metric in ('balanced_acc', 'kappa'):
        tests += pair_table('braindecode - original implementation', bd_pairs, metric)
    json.dump({'models': res, 'tests': tests}, open(os.path.join(out, 'results.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
