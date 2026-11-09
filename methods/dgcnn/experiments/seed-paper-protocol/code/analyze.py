"""Every number of this experiment from the runs' per-evaluation test curves.

usage: python analyze.py <runs dir with sd__*/result.npz ...> <results dir>

Subject-dependent, the two choices in the authors' eeg_DE.py and their alternatives:
  checkpoint   peak = highest test accuracy over all evaluations (what eeg_DE.py keeps); last = the final step;
               val = the step with the best validation accuracy (protocol sd-val, 2 of the 9 training trials held out)
  aggregation  best 2 of 3 = each subject's two best sessions only (eeg_DE.py: sorted(...)[1], [2]); all 3 = all 45
Means and SDs: 'mean' is the mean over sessions (and subjects' kept sessions) of one run; 'sd' is np.std over the same
values, as eeg_DE.py prints it; each cell then averages over the runs.
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats

runs, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
FEATS = ('de_LDS', 'de_movingAve')
lines, rows = [], []


def p(s=''):
    print(s)
    lines.append(s)


def keep(acc):
    """acc: 45 values -> (all 3 sessions, best 2 of 3): the eeg_DE.py aggregation on the second."""
    best2 = np.sort(acc.reshape(15, 3), axis=1)[:, 1:]
    return acc, best2.ravel()


def sd_table(feat):
    out_rows = {}
    for proto in ('sd', 'sdval'):
        files = sorted(glob.glob(f'{runs}/{proto}__{feat}__s*/result.npz'))
        for f in files:
            z = np.load(f)
            sets = list(z['sets'])
            ti = sets.index('test')
            test = np.stack([z[f's{i}'][ti] for i in range(45)])                 # (45, evals)
            rules = {}
            if proto == 'sd':
                rules['peak on test'] = test.max(1)
                rules['last step'] = test[:, -1]
            else:
                va = np.stack([z[f's{i}'][sets.index('val')] for i in range(45)])
                # the latest of the steps tied at the best validation accuracy
                ix = va.shape[1] - 1 - va[:, ::-1].argmax(1)
                rules['best validation step'] = test[np.arange(45), ix]
                rules['last step (7 trials)'] = test[:, -1]
            for r, acc in rules.items():
                a3, b2 = keep(acc)
                out_rows.setdefault(r, []).append((b2.mean(), b2.std(), a3.mean(), a3.std()))
    return {k: np.array(v) for k, v in out_rows.items()}


for feat in FEATS:
    t = sd_table(feat)
    if not t:
        continue
    p(f'## Subject-dependent, {feat}: first 9 trials -> last 6, mean accuracy % (SD) over runs')
    p()
    p('| Checkpoint | n runs | best 2 of 3 sessions | all 3 sessions |')
    p('|---|---|---|---|')
    for r, v in t.items():
        m = v.mean(0)
        p(f'| {r} | {len(v)} | {m[0]:.2f} ({m[1]:.2f}) | {m[2]:.2f} ({m[3]:.2f}) |')
        rows.append({'feature': feat, 'rule': r, 'n_runs': len(v), 'best2_mean': round(float(m[0]), 2), 'best2_sd': round(float(m[1]), 2),
                     'all3_mean': round(float(m[2]), 2), 'all3_sd': round(float(m[3]), 2),
                     'best2_per_run': [round(float(x), 2) for x in v[:, 0]], 'all3_per_run': [round(float(x), 2) for x in v[:, 2]]})
    p()

# Learning-rate check (de_LDS): validation clips pick the rate; test accuracy read at the last step of the 9-trial runs.
lrs = sorted({re.match(r'sdlr([\d.e-]+)__', os.path.basename(os.path.dirname(f))).group(1)
              for f in glob.glob(f'{runs}/sdlr*__de_LDS__s*/result.npz')}, key=float, reverse=True)
if lrs:
    def curves(fmt, name='test'):
        out_ = []
        for f in sorted(glob.glob(fmt)):
            z = np.load(f)
            i = list(z['sets']).index(name)
            out_.append(np.stack([z[f's{s}'][i] for s in range(45)]))
        return np.array(out_)                                      # (runs, 45, evals)

    p('## Learning-rate check, de_LDS, mean % over runs: all 3 sessions (best 2 of 3 in parentheses)')
    p()
    p('| lr | validation clips, last step | test, last step | test, peak |')
    p('|---|---|---|---|')
    table, best = [], None
    for lr in ['0.07'] + lrs:
        a, b = (f'{runs}/sd__de_LDS__s*/result.npz', f'{runs}/sdval__de_LDS__s*/result.npz') if lr == '0.07' else \
            (f'{runs}/sdlr{lr}__de_LDS__s*/result.npz', f'{runs}/sdvallr{lr}__de_LDS__s*/result.npz')
        T, V = curves(a), curves(b, 'val')
        last, peak, val = T[:, :, -1], T.max(2), V[:, :, -1]
        b2 = lambda x: np.mean([np.sort(r.reshape(15, 3), axis=1)[:, 1:].mean() for r in x])
        p(f'| {lr} | {val.mean():.2f} | {last.mean():.2f} ({b2(last):.2f}) | {peak.mean():.2f} ({b2(peak):.2f}) |')
        table.append({'lr': float(lr), 'val_last': round(float(val.mean()), 2), 'test_last': round(float(last.mean()), 2),
                      'test_last_best2': round(float(b2(last)), 2), 'test_peak': round(float(peak.mean()), 2),
                      'test_peak_best2': round(float(b2(peak)), 2), 'sd_sessions_last': round(float(last.mean(0).std()), 2)})
    p()
    pick = max(table, key=lambda r: r['val_last'])
    p(f"Picked on the validation clips: lr {pick['lr']}, test {pick['test_last']} on all 3 sessions.")
    p()
    rows.append({'lr_check': table, 'picked': pick})

p('## Leave one subject out, mean accuracy % over the 15 test subjects (SD over subjects), mean over runs')
p()
p('| Data | Features | Normalisation | n runs | peak on test | last step |')
p('|---|---|---|---|---|---|')
for kind, label in (('loso-sessions', 'one session at a time'), ('loso-pooled', 'all 3 sessions pooled')):
    for feat in FEATS:
        for norm in ('authors', 'zscore', 'zscore-lr0.01'):
            files = sorted(glob.glob(f'{runs}/{kind}__{feat}__{norm}__s*/result.npz'))
            if not files:
                continue
            P, L = [], []
            for f in files:
                z = np.load(f)
                keys = [k for k in z.files if re.match(r'g\d+_sub\d+', k)]
                groups = sorted({k.split('_')[0] for k in keys})
                pk = np.array([[z[f'{g}_sub{s}'][0].max() for s in range(15)] for g in groups])     # (groups, 15)
                la = np.array([[z[f'{g}_sub{s}'][0][-1] for s in range(15)] for g in groups])
                P.append((pk.mean(), pk.mean(0).std()))
                L.append((la.mean(), la.mean(0).std()))
            P, L = np.array(P), np.array(L)
            p(f'| {label} | {feat} | {norm} | {len(files)} | {P[:, 0].mean():.2f} ({P[:, 1].mean():.2f}) | {L[:, 0].mean():.2f} ({L[:, 1].mean():.2f}) |')
            rows.append({'protocol': kind, 'feature': feat, 'norm': norm, 'n_runs': len(files),
                         'peak_mean': round(float(P[:, 0].mean()), 2), 'peak_sd': round(float(P[:, 1].mean()), 2),
                         'last_mean': round(float(L[:, 0].mean()), 2), 'last_sd': round(float(L[:, 1].mean()), 2)})
p()

# DGCNN against the linear SVM (svm_baseline.py, svm_loso.py), both with nothing chosen on the test set.
sd_json, loso_json = os.path.join(out, 'svm_de_LDS.json'), os.path.join(out, 'svm_loso.json')
if os.path.exists(sd_json) and lrs:
    svm = json.load(open(sd_json))
    key = next(k for k in svm if 'validation' in k)
    a = np.array(svm[key]['per_session']).reshape(15, 3).mean(1)
    lr = str(pick['lr'])
    d = curves(f'{runs}/sdlr{lr}__de_LDS__s*/result.npz')[:, :, -1].mean(0).reshape(15, 3).mean(1)
    w = stats.wilcoxon(a, d)
    p('## DGCNN against a linear SVM, de_LDS, nothing chosen on the test set (Wilcoxon over the 15 subjects)')
    p()
    p('| Protocol | DGCNN | linear SVM | SVM − DGCNN | p |')
    p('|---|---|---|---|---|')
    p(f'| subject-dependent, all 3 sessions | {d.mean():.2f} | {a.mean():.2f} | {a.mean() - d.mean():+.2f} | {w.pvalue:.3f} |')
    cmp_rows = {'sd': dict(dgcnn=round(float(d.mean()), 2), svm=round(float(a.mean()), 2), p=float(w.pvalue))}
    if os.path.exists(loso_json):
        L = np.array(json.load(open(loso_json))['per_session_subject']).mean(0)
        per = []
        for f in sorted(glob.glob(f'{runs}/loso-sessions__de_LDS__zscore-lr0.01__s*/result.npz')):
            z = np.load(f)
            per.append([np.mean([z[f'g{g}_sub{t}'][0][-1] for g in range(3)]) for t in range(15)])
        D = np.mean(per, axis=0)
        w = stats.wilcoxon(L, D)
        p(f'| leave one subject out, one session at a time | {D.mean():.2f} | {L.mean():.2f} | {L.mean() - D.mean():+.2f} | {w.pvalue:.4f} |')
        cmp_rows['loso'] = dict(dgcnn=round(float(D.mean()), 2), svm=round(float(L.mean()), 2), p=float(w.pvalue))
    p()
    rows.append({'dgcnn_vs_svm': cmp_rows})
open(os.path.join(out, 'results.txt'), 'w').write('\n'.join(lines) + '\n')
json.dump(rows, open(os.path.join(out, 'results.json'), 'w'), indent=1)
