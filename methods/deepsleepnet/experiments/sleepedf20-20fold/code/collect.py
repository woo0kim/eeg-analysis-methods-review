"""DeepSleepNet on Sleep-EDF-20 (20-fold CV, released code): pooled metrics vs. the paper, from the per-subject
predictions of the released predict.py.

usage: python collect.py <preds dir> <out dir> > <out dir>/results.txt
  <preds dir>/subject<i>.npz   y_true, y_pred of subject i (all nights), copied from predict.py's output_subject<i>.npz
writes <out dir>/results.json. As in the paper (Sec. IV-B/C) and the released summary.py, predictions of all folds are
pooled and scored once; a 95% bootstrap interval over subjects (10,000 resamples) shows the spread the paper omits.
"""
import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, precision_score, recall_score

STAGES = ['W', 'N1', 'N2', 'N3', 'REM']
# DeepSleepNet paper (Supratak et al., IEEE TNSRE 2017), Sleep-EDF Fpz-Cz, 20-fold CV: Table IV (overall, F1) and
# Table III (confusion matrix: rows true W..REM, columns predicted)
PAPER = {'accuracy': 0.820, 'macro_f1': 0.769, 'kappa': 0.76,
         'f1': [0.847, 0.466, 0.859, 0.848, 0.824], 'n_epochs': 41950,
         'cm': [[6614, 745, 181, 81, 306], [295, 1406, 631, 30, 442], [391, 618, 14542, 1473, 775],
                [29, 9, 291, 5370, 4], [360, 457, 419, 7, 6474]]}


def metrics(y, p):
    return {'accuracy': accuracy_score(y, p), 'macro_f1': f1_score(y, p, average='macro', labels=range(5)),
            'kappa': cohen_kappa_score(y, p), 'f1': f1_score(y, p, average=None, labels=range(5)).tolist(),
            'precision': precision_score(y, p, average=None, labels=range(5), zero_division=0).tolist(),
            'recall': recall_score(y, p, average=None, labels=range(5), zero_division=0).tolist(),
            'cm': confusion_matrix(y, p, labels=range(5)).tolist(), 'n_epochs': int(len(y))}


def main():
    pdir, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    subj = {}
    for f in sorted(glob.glob(os.path.join(pdir, 'subject*.npz')), key=lambda f: int(f.split('subject')[-1][:-4])):
        d = np.load(f)
        subj[int(f.split('subject')[-1][:-4])] = (d['y_true'].astype(int), d['y_pred'].astype(int))
    y = np.concatenate([v[0] for v in subj.values()])
    p = np.concatenate([v[1] for v in subj.values()])
    res = {'pooled': metrics(y, p), 'n_subjects': len(subj),
           'per_subject_accuracy': {s: accuracy_score(*v) for s, v in subj.items()}}
    rng = np.random.RandomState(0)
    ids = list(subj)
    boot = {k: [] for k in ('accuracy', 'macro_f1', 'kappa')}
    for _ in range(10000):
        pick = rng.choice(ids, len(ids))
        yb = np.concatenate([subj[s][0] for s in pick])
        pb = np.concatenate([subj[s][1] for s in pick])
        boot['accuracy'].append(accuracy_score(yb, pb))
        boot['macro_f1'].append(f1_score(yb, pb, average='macro', labels=range(5)))
        boot['kappa'].append(cohen_kappa_score(yb, pb))
    res['bootstrap95'] = {k: np.percentile(v, [2.5, 97.5]).tolist() for k, v in boot.items()}
    res['paper'] = PAPER
    json.dump(res, open(os.path.join(out, 'results.json'), 'w'), indent=1)

    m, b = res['pooled'], res['bootstrap95']
    print(f'DeepSleepNet, Sleep-EDF-20 Fpz-Cz, 20-fold CV (released code), {res["n_subjects"]} subjects, '
          f'{m["n_epochs"]} test epochs (paper: {PAPER["n_epochs"]})\n')
    print(f'{"metric":10s} {"ours":>7s}  {"95% CI over subjects":>22s}  {"paper":>7s}  {"diff":>7s}')
    for k in ('accuracy', 'macro_f1', 'kappa'):
        print(f'{k:10s} {m[k]:7.4f}  [{b[k][0]:.4f}, {b[k][1]:.4f}]  {PAPER[k]:7.3f}  {m[k] - PAPER[k]:+7.4f}')
    print('\nper-stage F1      ' + ' '.join(f'{s:>7s}' for s in STAGES))
    print('  ours            ' + ' '.join(f'{v:7.4f}' for v in m['f1']))
    print('  paper           ' + ' '.join(f'{v:7.3f}' for v in PAPER['f1']))
    print('\nconfusion matrix (rows true, columns predicted; paper in parentheses)')
    for i, s in enumerate(STAGES):
        print(f'  {s:4s} ' + ' '.join(f'{m["cm"][i][j]:6d} ({PAPER["cm"][i][j]:5d})' for j in range(5)))
    print('\nper-subject accuracy: ' + ' '.join(f'{s}:{a:.3f}' for s, a in res['per_subject_accuracy'].items()))


if __name__ == '__main__':
    main()
