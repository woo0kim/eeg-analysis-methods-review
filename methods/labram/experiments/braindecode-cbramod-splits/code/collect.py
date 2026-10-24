"""Score the braindecode-LaBraM runs as the released-code LaBraM runs are scored; compare the two and the paper.

Each logs/runs/<dataset>__labram__s<seed>/ holds preds.npz (validation and test probabilities for every epoch) and
log.txt (LaBraM's per-epoch JSON log), written by the same hooks as in ../../finetune-cbramod-splits. Two read-outs,
as there (code/collect.py):
  labram         epoch = first epoch with the highest validation Cohen's kappa (multi-class) or AUROC (binary),
                 CBraMod's monitor; the headline rule
  labram@val-acc epoch = first epoch with the highest validation accuracy, LaBraM's own rule
Test metrics use the formulas of CBraMod's finetune_evaluator.py. "Released" = the same read-out of the released-code
runs (../../finetune-cbramod-splits/results/results.json).
usage: python code/collect.py [RUNS_DIR] [OUT_DIR] > OUT_DIR/results.txt
"""
import glob, json, os, sys
import numpy as np
from scipy import stats
from sklearn.metrics import (auc, balanced_accuracy_score, cohen_kappa_score, f1_score, precision_recall_curve,
                             roc_auc_score)

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '../logs/runs')
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, '../results')
RELEASED = os.path.join(HERE, '../../finetune-cbramod-splits/results/results.json')
MC = ['balanced_acc', 'kappa', 'weighted_f1']
BC = ['balanced_acc', 'pr_auc', 'roc_auc']
# LaBraM-Base as reported in the CBraMod paper (ICLR 2025): mean, std over 5 seeds
PAPER = {
    'PhysioNet-MI': ('Table 3', [(0.6173, 0.0122), (0.4912, 0.0192), (0.6177, 0.0141)]),
    'BCIC-IV-2a': ('Table 15', [(0.4869, 0.0085), (0.3159, 0.0154), (0.4758, 0.0103)]),
    'Mumtaz2016': ('Table 10', [(0.9409, 0.0079), (0.9798, 0.0093), (0.9782, 0.0057)]),
    'MentalArithmetic': ('Table 12', [(0.6909, 0.0125), (0.5999, 0.0155), (0.7721, 0.0093)]),
}


def multiclass(prob, y):   # finetune_evaluator.get_metrics_for_multiclass
    pred = prob.argmax(-1)
    return [balanced_accuracy_score(y, pred), cohen_kappa_score(y, pred), f1_score(y, pred, average='weighted')]


def binary(prob, y):       # finetune_evaluator.get_metrics_for_binaryclass
    score = prob.reshape(-1)
    pred = (score > 0.5).astype(int)
    precision, recall, _ = precision_recall_curve(y, score, pos_label=1)
    return [balanced_accuracy_score(y, pred), auc(recall, precision), roc_auc_score(y, score)]


def welch(m1, s1, n1, m2, s2, n2):
    return float(stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False).pvalue)


runs = []
for d in sorted(glob.glob(os.path.join(RUNS, '*__labram__s*'))):
    if not os.path.exists(os.path.join(d, 'preds.npz')):
        continue
    ds, _, seed = os.path.basename(d).split('__')
    p = np.load(os.path.join(d, 'preds.npz'))
    y_val, y_test = p['y_val'].astype(int), p['y_test'].astype(int)
    n_ep = min(len(p['val']), len(p['test']))
    is_bin = p['val'].shape[-1] == 1
    fn, names = (binary, BC) if is_bin else (multiclass, MC)
    val = np.array([fn(p['val'][e], y_val) for e in range(n_ep)])
    test = np.array([fn(p['test'][e], y_test) for e in range(n_ep)])
    log = [json.loads(line) for line in open(os.path.join(d, 'log.txt'))]
    val_acc = np.array([r['val_accuracy'] for r in log[:n_ep]])
    for setting, ep in [('labram', int(np.argmax(val[:, 2 if is_bin else 1]))),   # val AUROC / kappa
                        ('labram@val-acc', int(np.argmax(val_acc)))]:
        runs.append(dict(dataset=ds, setting=setting, seed=int(seed[1:]), metric_names=names,
                         metrics=[round(float(v), 5) for v in test[ep]], best_epoch=ep + 1, epochs_run=n_ep,
                         test_balanced_acc_per_epoch=[round(float(v), 5) for v in test[:, 0]]))

released = {(s['dataset'], s['setting']): s for s in json.load(open(RELEASED))['summary']}
summary = []
for key in sorted({(r['dataset'], r['setting']) for r in runs}):
    rs = sorted([r for r in runs if (r['dataset'], r['setting']) == key], key=lambda r: r['seed'])
    arr = np.array([r['metrics'] for r in rs])
    n = len(rs)
    row = dict(dataset=key[0], setting=key[1], n=n, seeds=[r['seed'] for r in rs], metric_names=rs[0]['metric_names'],
               per_seed=arr.tolist(), best_epochs=[r['best_epoch'] for r in rs], mean=arr.mean(0).tolist(),
               std=(arr.std(0, ddof=1) if n > 1 else np.zeros(3)).tolist())
    row['paper_table'], row['paper'] = PAPER[key[0]]
    row['p_welch_vs_paper'] = [welch(m, s, n, pm, ps, 5) if n > 1 else None
                               for m, s, (pm, ps) in zip(row['mean'], row['std'], row['paper'])]
    rel = released.get(key)
    if rel:
        row['released'] = [list(x) for x in zip(rel['mean'], rel['std'])]
        row['released_best_epochs'] = rel['best_epochs']
        row['p_welch_vs_released'] = [welch(m, s, n, rm, rsd, rel['n']) if n > 1 else None
                                      for m, s, rm, rsd in zip(row['mean'], row['std'], rel['mean'], rel['std'])]
    summary.append(row)

os.makedirs(OUT, exist_ok=True)
json.dump(dict(runs=runs, summary=summary), open(os.path.join(OUT, 'results.json'), 'w'), indent=1)
for row in summary:
    print(f"\n{row['dataset']} [{row['setting']}] n={row['n']} seeds={row['seeds']} best_epochs={row['best_epochs']}"
          + (f" (released: {row['released_best_epochs']})" if 'released' in row else ''))
    for i, mn in enumerate(row['metric_names']):
        s = f"  {mn:13s} braindecode {row['mean'][i]:.4f} ± {row['std'][i]:.4f}"
        if 'released' in row:
            rm, rsd = row['released'][i]
            p = row['p_welch_vs_released'][i]
            s += f" | released code {rm:.4f} ± {rsd:.4f} | diff {row['mean'][i] - rm:+.4f}" + \
                 (f" | p={p:.3g}" if p is not None else '')
        pm, ps = row['paper'][i]
        p = row['p_welch_vs_paper'][i]
        s += f" || paper {pm:.4f} ± {ps:.4f} | diff {row['mean'][i] - pm:+.4f}" + (f" | p={p:.3g}" if p is not None else '')
        print(s)
        print('      per-seed:', ' '.join(f'{v[i]:.4f}' for v in row['per_seed']))
