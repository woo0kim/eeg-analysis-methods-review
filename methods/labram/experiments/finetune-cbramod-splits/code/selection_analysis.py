"""How much does the reported LaBraM number depend on which epoch is reported, and on the 0.5 threshold?

Reads the committed preds.npz / log.txt of every run (no GPU) and prints, per dataset, the test balanced
accuracy (mean ± std over seeds) at the epoch chosen by:
  val-kappa/AUROC  first epoch with the best validation kappa (multi-class) / AUROC (binary): the headline rule,
                   CBraMod's monitor and the monitor the LaBraM paper states (Sec. 3.2)
  val-acc          first epoch with the best validation accuracy: the rule in LaBraM's run_class_finetuning.py
  last             the final (50th) epoch
  test-max         the epoch with the best *test* balanced accuracy: uses the test labels, shown only as an
                   upper bound (it is the EEG Conformer rule, methods/eeg-conformer)
For the binary datasets it also scores the headline epoch with the decision threshold chosen on the
validation set (Youden's J) instead of 0.5, and reports the last-epoch train accuracy from LaBraM's log.
Metric formulas are those of collect.py (CBraMod's finetune_evaluator.py).
usage: python code/selection_analysis.py [RUNS_DIR] > results/selection_analysis.txt
"""
import glob, json, os, sys
import numpy as np
from sklearn.metrics import (auc, balanced_accuracy_score, cohen_kappa_score, f1_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '../logs/runs')
RULES = ['val-kappa/AUROC', 'val-acc', 'last', 'test-max']


# copied from collect.py (importing it would run the whole collection and rewrite results/)
def multiclass(prob, y):   # finetune_evaluator.get_metrics_for_multiclass
    pred = prob.argmax(-1)
    return [balanced_accuracy_score(y, pred), cohen_kappa_score(y, pred), f1_score(y, pred, average='weighted')]


def binary(prob, y):       # finetune_evaluator.get_metrics_for_binaryclass
    score = prob.reshape(-1)
    pred = (score > 0.5).astype(int)
    precision, recall, _ = precision_recall_curve(y, score, pos_label=1)
    return [balanced_accuracy_score(y, pred), auc(recall, precision), roc_auc_score(y, score)]


def youden_threshold(score, y):
    fpr, tpr, thr = roc_curve(y, score)
    return thr[int(np.argmax(tpr - fpr))]


rows = {}
for d in sorted(glob.glob(os.path.join(RUNS, '*__labram__s*'))):
    ds, _, seed = os.path.basename(d).split('__')
    p = np.load(os.path.join(d, 'preds.npz'))
    y_val, y_test = p['y_val'].astype(int), p['y_test'].astype(int)
    n_ep = min(len(p['val']), len(p['test']))
    is_bin = p['val'].shape[-1] == 1
    fn = binary if is_bin else multiclass
    val = np.array([fn(p['val'][e], y_val) for e in range(n_ep)])
    test = np.array([fn(p['test'][e], y_test) for e in range(n_ep)])
    log = [json.loads(line) for line in open(os.path.join(d, 'log.txt'))][:n_ep]
    ep = {'val-kappa/AUROC': int(np.argmax(val[:, 2 if is_bin else 1])),
          'val-acc': int(np.argmax([r['val_accuracy'] for r in log])),
          'last': n_ep - 1,
          'test-max': int(np.argmax(test[:, 0]))}
    r = {k: test[e, 0] for k, e in ep.items()}
    r['epochs'] = {k: e + 1 for k, e in ep.items()}
    e = ep['val-kappa/AUROC']
    r['train_acc_last'] = log[-1]['train_class_acc']
    r['test_acc_last'] = log[-1]['test_accuracy']
    if is_bin:
        thr = youden_threshold(p['val'][e].reshape(-1), y_val)
        pred = (p['test'][e].reshape(-1) > thr).astype(int)
        r['ba_val_threshold'] = balanced_accuracy_score(y_test, pred)
        r['threshold'] = float(thr)
    rows.setdefault(ds, []).append((int(seed[1:]), r))

f = lambda a: f'{np.mean(a):.4f} ± {np.std(a, ddof=1):.4f}'
for ds, rs in rows.items():
    rs = [r for _, r in sorted(rs, key=lambda t: t[0])]
    print(f'\n{ds} (n={len(rs)} seeds) test balanced accuracy by epoch-selection rule')
    for k in RULES:
        print(f"  {k:16s} {f([r[k] for r in rs])}   epochs {[r['epochs'][k] for r in rs]}")
    gap = [r['test-max'] - r['val-kappa/AUROC'] for r in rs]
    print(f'  test-max - headline: {np.mean(gap):+.4f} (per seed {" ".join(f"{g:+.4f}" for g in gap)})')
    if 'ba_val_threshold' in rs[0]:
        print(f"  headline epoch, threshold 0.5        {f([r['val-kappa/AUROC'] for r in rs])}")
        print(f"  headline epoch, threshold from val   {f([r['ba_val_threshold'] for r in rs])}   "
              f"thresholds {[round(r['threshold'], 3) for r in rs]}")
    print(f"  last epoch: train acc {f([r['train_acc_last'] for r in rs])} | test acc {f([r['test_acc_last'] for r in rs])}")
