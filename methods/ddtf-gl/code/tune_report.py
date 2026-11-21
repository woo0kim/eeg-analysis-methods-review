"""The feature-tuning experiment's results: per target, the configuration the validation search selected (results/selection/
<target>.json, written by tune.py), and dDTF-GL+ (that configuration, seeds 0-4) against dDTF-GL, both scored by the collector of
the pipeline experiment their runs belong to (each experiment's results/results.json). Welch's t-test over the 5 seeds.

  python tune_report.py <feature-tuning results dir> > <dir>/results.txt      (writes <dir>/results.json too)
"""
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.normpath(os.path.join(HERE, '../experiments'))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(EXP, 'feature-tuning/results')
# target -> (label, experiment, how to read per-seed values of (model) for (metric))
TARGETS = [
    ('STEW-rest', 'STEW, rest vs. task (shared folds)', 'stew-cbramod-pipeline', ('task', 'rest-task'), ['accuracy', 'kappa']),
    ('STEW-rating3', 'STEW, 3 rating levels (shared folds)', 'stew-cbramod-pipeline', ('task', 'rating3'), ['accuracy', 'kappa']),
    ('STEW-6s-SI', "STEW, the dDTF-GL paper's subject-independent setting", 'stew-paper-protocol', ('protocol', 'si'), ['accuracy']),
    ('BCIC-IV-2a', 'BCI-IV-2a, cross-subject', 'cbramod-splits', ('dataset', 'BCIC-IV-2a'), ['balanced_acc', 'kappa']),
    ('PhysioNet-MI', 'PhysioNet-MI, cross-subject', 'cbramod-splits', ('dataset', 'PhysioNet-MI'), ['balanced_acc', 'kappa']),
    ('Mumtaz2016', 'Mumtaz2016, cross-subject', 'cbramod-splits', ('dataset', 'Mumtaz2016'), ['balanced_acc', 'roc_auc']),
    ('MentalArithmetic', 'MentalArithmetic, cross-subject', 'cbramod-splits', ('dataset', 'MentalArithmetic'), ['balanced_acc', 'roc_auc']),
    ('SleepEDF', 'Sleep-EDF-78, common test set', 'sleepedf-cbramod-pipeline', None, ['kappa', 'balanced_acc']),
    ('SEED', 'SEED, subject split', 'seed-cbramod-pipeline', ('dataset', 'SEED'), ['balanced_acc', 'kappa']),
    ('SEED-V', 'SEED-V, subject split', 'seed-cbramod-pipeline', ('dataset', 'SEED-V'), ['balanced_acc', 'kappa']),
]


def per_seed(exp, key, model, metric):
    d = json.load(open(os.path.join(EXP, exp, 'results/results.json')))
    if exp == 'sleepedf-cbramod-pipeline':
        m = d['models'].get(model)
        return [m['per_seed'][s][metric] for s in sorted(m['per_seed'], key=int)] if m else []
    if exp == 'stew-paper-protocol':
        return [r['fold_mean'][metric] for r in sorted(d['runs'], key=lambda r: r['seed'])
                if r['protocol'] == key[1] and r['model'] == model]
    out = []
    for r in sorted(d['runs'], key=lambda r: r['seed']):
        if r.get(key[0]) != key[1] or (r.get('model', r.get('setting')) != model) or r.get('split', 'subject') != 'subject':
            continue
        if metric in r['metric_names']:
            out.append(r['metrics'][r['metric_names'].index(metric)])
    return out


def recheck(target, sel):
    """Recompute every candidate's validation score from the committed search runs (logs/<target>/), as tune.val_score does."""
    import hashlib
    from sklearn.metrics import cohen_kappa_score, roc_auc_score
    logs = os.path.join(os.path.dirname(OUT), 'logs', target)
    if not sel.get('stages'):
        return None
    worst = 0.0
    for st in sel.get('stages', []):
        for c in st['candidates']:
            cid = hashlib.md5(c['config'].encode()).hexdigest()[:10]
            runs = sorted(d for d in os.listdir(logs) if d.startswith(cid + '__'))
            if not runs:
                return None
            seeds = sorted({r.rsplit('__s', 1)[1] for r in runs})
            vals = []
            for s_ in seeds:
                z = [np.load(os.path.join(logs, r, 'test_preds.npz')) for r in runs if r.endswith('__s' + s_)]
                y = np.concatenate([q['y_val'].astype(int) for q in z])
                if st['metric'] == 'roc_auc':
                    vals.append(roc_auc_score(y, np.concatenate([q['proba_val'].astype(float)[:, 1] for q in z])))
                else:
                    vals.append(cohen_kappa_score(y, np.concatenate([q['pred_val'].astype(int) for q in z])))
            worst = max(worst, abs(float(np.mean(vals)) - c['val']))
    return worst


def main():
    rows = []
    for target, label, exp, key, metrics in TARGETS:
        sel_path = os.path.join(OUT, 'selection', f'{target}.json')
        if not os.path.exists(sel_path):
            continue
        sel = json.load(open(sel_path))
        row = dict(target=target, label=label, experiment=exp, selected=sel['selected'], config_from=sel.get('config_from'),
                   stages=[dict(stage=s['stage'], metric=s['metric'], selected=s['selected'],
                                candidates=s['candidates']) for s in sel.get('stages', [])], metrics={},
                   recheck_max_diff=recheck(target, sel))
        for metric in metrics:
            base, plus = per_seed(exp, key, 'ddtf-gl', metric), per_seed(exp, key, 'ddtf-gl-plus', metric)
            if len(plus) < 2:
                continue
            r = dict(base=base, plus=plus, base_mean=float(np.mean(base)), base_sd=float(np.std(base, ddof=1)),
                     plus_mean=float(np.mean(plus)), plus_sd=float(np.std(plus, ddof=1)),
                     diff=float(np.mean(plus) - np.mean(base)), p=float(stats.ttest_ind(plus, base, equal_var=False).pvalue))
            if exp == 'stew-paper-protocol':
                saf = per_seed(exp, key, 'safari-gl', metric)
                r.update(safari_mean=float(np.mean(saf)), safari_sd=float(np.std(saf, ddof=1)),
                         p_vs_safari=float(stats.ttest_ind(plus, saf, equal_var=False).pvalue))
            row['metrics'][metric] = r
        rows.append(row)
    json.dump(rows, open(os.path.join(OUT, 'results.json'), 'w'), indent=1)
    fp = lambda p: 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'   # noqa: E731
    print('dDTF-GL+ (configuration chosen on validation data, seeds 0-4) against dDTF-GL (the paper\'s chain), mean ± SD over seeds\n')
    for r in rows:
        print(f"{r['label']}\n  selected: {r['selected']}" + (f" (from {r['config_from']}, not tuned here)" if r['config_from'] else ''))
        if r['recheck_max_diff'] is not None:
            print(f"  validation scores recomputed from logs/: max difference {r['recheck_max_diff']:.2e}")
        for m, v in r['metrics'].items():
            extra = (f"; on Safari et al.'s maps dDTF-GL {v['safari_mean']:.4f} ± {v['safari_sd']:.4f}, dDTF-GL+ − it {v['plus_mean'] - v['safari_mean']:+.4f} "
                     f"({fp(v['p_vs_safari'])})") if 'safari_mean' in v else ''
            print(f"  {m:13s} dDTF-GL {v['base_mean']:.4f} ± {v['base_sd']:.4f}  dDTF-GL+ {v['plus_mean']:.4f} ± {v['plus_sd']:.4f}  "
                  f"{v['diff']:+.4f} ({fp(v['p'])}){extra}")
        for s in r['stages']:
            best = max(c['val'] for c in s['candidates'] if c['val'] == c['val'])
            print(f"  stage {s['stage']:12s} ({s['metric']} on validation): " + ', '.join(
                f"{c['val']:.4f}{'*' if c['config'] == s['selected'] and c['val'] == best else ''}" for c in s['candidates']))
        print()


if __name__ == '__main__':
    main()
