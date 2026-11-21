"""Staged search of dDTF-GL's features and settings on validation data only (experiments/feature-tuning), then the selected
configuration on every seed for the test scores.

  python tune.py <target> [--workers 2] [--threads 4] [--final-only]

Targets are the subject-independent protocols dDTF-GL already ran in (each with a validation part that no fit uses):

  STEW-rest, STEW-rating3   STEW's shared 7-fold GroupKFold (validation = the next fold); search on seed 0 over the 7 folds
  BCIC-IV-2a, PhysioNet-MI, Mumtaz2016, MentalArithmetic   CBraMod's subject splits; search on seeds 0-1 (PhysioNet-MI: 0)
  SleepEDF                  the shared 62 / 8 / 8 subject split; seeds 0-1
  STEW-6s-SI                the dDTF-GL paper's subject-independent setting (6-s windows): STEW-rest's configuration, final runs only
  SEED, SEED-V              the subject split; seed 0; the search runs train on every 4th training window (consecutive 1-s windows of
                            a 4-minute clip are near-duplicates), the final runs on all of them

Stages, each starting from the best configuration so far (gl_plus.py explains the options):
  1 features     ddtf (the paper's) | ddtf+graph | spec | spec+spect | ddtf+spec | ddtf+graph+spec+spect   (SEED: no spect)
  2 projections  lnt (the paper's) | lnt+lda | lnt+lda+svml | lnt+lda+svml+svmr | lda+svml+svmr
  3 counts       DFT keeps 0.1 | 0.25 | 0.5 | 1 of the features, each with and without AXB
  4 classifier   (depth, trees, learning rate) = (3, 500, 0.1) | (6, 500, 0.1) | (3, 1000, 0.05) | (6, 1000, 0.05)
A candidate replaces the current best only if its validation score is higher (ties keep the earlier, simpler one). The score is the
checkpoint metric every model in that pipeline is selected on: kappa (multi-class and STEW), AUROC (Mumtaz2016, MentalArithmetic),
computed on the validation predictions pooled over folds (STEW) and averaged over the search seeds.

The selected configuration then runs on seeds 0-4 (every fold) under the run names of the experiment it joins, with variant
'ddtf-gl-plus', so collect.py scores it exactly as the original runs. Everything is written under $GL_WORK/runs-plus/<target>/.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gl_plus  # noqa: E402

WORK = os.path.expanduser(os.environ.get('GL_WORK', '~/repro-gl'))
PY = sys.executable
CB = dict(protocol='cbramod-split', folds=[None])
TARGETS = {
    'STEW-rest': dict(dataset='STEW', protocol='cbramod', task='rest-task', folds=list(range(7)), seeds=[0], select='kappa',
                      run=lambda f, s: f'rest-task__fold{f}__ddtf-gl-plus__s{s}', exp='stew-cbramod-pipeline'),
    'STEW-rating3': dict(dataset='STEW', protocol='cbramod', task='rating3', folds=list(range(7)), seeds=[0], select='kappa',
                         run=lambda f, s: f'rating3__fold{f}__ddtf-gl-plus__s{s}', exp='stew-cbramod-pipeline'),
    'BCIC-IV-2a': dict(dataset='BCIC-IV-2a', **CB, seeds=[0, 1], select='kappa'),
    'PhysioNet-MI': dict(dataset='PhysioNet-MI', **CB, seeds=[0], select='kappa'),
    'Mumtaz2016': dict(dataset='Mumtaz2016', **CB, seeds=[0, 1], select='roc_auc'),
    'MentalArithmetic': dict(dataset='MentalArithmetic', **CB, seeds=[0, 1], select='roc_auc'),
    'SleepEDF': dict(dataset='SleepEDF', protocol='split', folds=[None], seeds=[0, 1], select='kappa',
                     run=lambda f, s: f'SleepEDF__ddtf-gl-plus__s{s}', exp='sleepedf-cbramod-pipeline'),
    'SEED': dict(dataset='SEED', protocol='subject', folds=[None], seeds=[0], select='kappa', sub=4,
                 run=lambda f, s: f'SEED__subject__ddtf-gl-plus__s{s}', exp='seed-cbramod-pipeline'),
    'SEED-V': dict(dataset='SEED-V', protocol='subject', folds=[None], seeds=[0], select='kappa', sub=4,
                   run=lambda f, s: f'SEED-V__subject__ddtf-gl-plus__s{s}', exp='seed-cbramod-pipeline'),
}
# the dDTF-GL paper's own subject-independent setting (6-s windows, 7 subject folds, no validation part): nothing is tuned here; the
# configuration selected for STEW-rest (4-s windows, the same subjects' folds) runs as it is (--final-only)
TARGETS['STEW-6s-SI'] = dict(dataset='STEW-6s', protocol='si', folds=list(range(7)), seeds=[0], select=None, config_from='STEW-rest',
                             run=lambda f, s: f'si__fold{f}__ddtf-gl-plus__s{s}', exp='stew-paper-protocol')
for _k in ('BCIC-IV-2a', 'PhysioNet-MI', 'Mumtaz2016', 'MentalArithmetic'):
    TARGETS[_k].update(run=lambda f, s, k=_k: f'{k}__ddtf-gl-plus__s{s}', exp='cbramod-splits')


def stages(t):
    one_s = t['dataset'] in ('SEED', 'SEED-V')
    feats = ['ddtf', 'ddtf+graph', 'spec'] + ([] if one_s else ['spec+spect']) + ['ddtf+spec'] + \
            (['ddtf+graph+spec'] if one_s else ['ddtf+graph+spec+spect'])
    return [('features', lambda b: [dict(b, feat=tuple(f.split('+'))) for f in feats]),
            ('projections', lambda b: [dict(b, proj=tuple(p.split('+'))) for p in
                                       ('lnt', 'lnt+lda', 'lnt+lda+svml', 'lnt+lda+svml+svmr', 'lda+svml+svmr')]),
            ('counts', lambda b: [dict(b, dft=d, axb=a) for d in (0.1, 0.25, 0.5, 1.0) for a in (True, False)]),
            ('classifier', lambda b: [dict(b, depth=d, trees=n, lr=r) for d, n, r in ((3, 500, 0.1), (6, 500, 0.1), (3, 1000, 0.05), (6, 1000, 0.05))])]


def cid(c):
    return hashlib.md5(gl_plus.name(c).encode()).hexdigest()[:10]


def run(target, t, c, fold, seed, threads, final=False):
    sub = 'final' if final else 'search'
    name = t['run'](fold, seed) if final else f'{cid(c)}__fold{fold}__s{seed}'
    out = os.path.join(WORK, 'runs-plus', target, sub, name)
    if os.path.exists(os.path.join(out, 'metrics.json')):
        return out
    args = [PY, os.path.join(HERE, 'run_gl_plus.py'), '--dataset', t['dataset'], '--protocol', t['protocol'], '--seed', str(seed),
            '--cfg', gl_plus.name(c), '--out', out, '--threads', str(threads)]
    if t.get('task'):
        args += ['--task', t['task']]
    if fold is not None:
        args += ['--fold', str(fold)]
    if not final and t.get('sub', 1) > 1:
        args += ['--train-sub', str(t['sub'])]
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, **{k: str(threads) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
    with open(os.path.join(out, 'stdout.txt'), 'w') as f:      # BLAS (LDA, SVMs) limited to the run's threads: the servers are shared
        rc = subprocess.call(args, stdout=f, stderr=subprocess.STDOUT, env=env)
    if rc != 0:
        print(f'FAILED {target} {gl_plus.name(c)} fold {fold} seed {seed}: see {out}/stdout.txt', flush=True)
    return out


def val_score(target, t, c):
    """Selection score: the pipeline's checkpoint metric on the validation predictions, pooled over folds, mean over seeds."""
    vals = []
    for seed in t['seeds']:
        ys, ps, pr = [], [], []
        for fold in t['folds']:
            z = np.load(os.path.join(WORK, 'runs-plus', target, 'search', f'{cid(c)}__fold{fold}__s{seed}', 'test_preds.npz'))
            ys.append(z['y_val'].astype(int)), ps.append(z['pred_val'].astype(int)), pr.append(z['proba_val'].astype(float))
        y, p, pr = np.concatenate(ys), np.concatenate(ps), np.concatenate(pr)
        vals.append(roc_auc_score(y, pr[:, 1]) if t['select'] == 'roc_auc' else cohen_kappa_score(y, p))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('target', choices=list(TARGETS))
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--final-only', action='store_true')
    a = ap.parse_args()
    t = TARGETS[a.target]
    log_path = os.path.join(WORK, 'runs-plus', a.target, 'selection.json')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    state = json.load(open(log_path)) if os.path.exists(log_path) else {'target': a.target, 'stages': []}
    best = dict(gl_plus.DEFAULT)
    best_score = None
    pool = ThreadPoolExecutor(a.workers)
    if not a.final_only:
        for stage, gen in stages(t):
            cands = gen(best)
            jobs = [pool.submit(run, a.target, t, c, f, s, a.threads) for c in cands for f in t['folds'] for s in t['seeds']]
            [j.result() for j in jobs]
            table = []
            for c in cands:
                try:
                    table.append((gl_plus.name(c), val_score(a.target, t, c), c))
                except FileNotFoundError:
                    table.append((gl_plus.name(c), float('nan'), c))
            for nm, sc, c in table:
                if np.isfinite(sc) and (best_score is None or sc > best_score + 1e-12) and (nm != gl_plus.name(best) or best_score is None):
                    best, best_score = c, sc
            print(f'[{time.strftime("%H:%M")}] {a.target} stage {stage}: best {gl_plus.name(best)} ({t["select"]} {best_score:.4f})', flush=True)
            for nm, sc, _ in table:
                print(f'    {sc:.4f}  {nm}', flush=True)
            state['stages'] = [s for s in state['stages'] if s['stage'] != stage] + [
                dict(stage=stage, metric=t['select'], candidates=[dict(config=nm, val=sc) for nm, sc, _ in table], selected=gl_plus.name(best))]
            state.update(selected=gl_plus.name(best), selected_val=best_score)
            json.dump(state, open(log_path, 'w'), indent=1)
    else:
        src = t.get('config_from')
        if src:
            state = json.load(open(os.path.join(WORK, 'runs-plus', src, 'selection.json')))
            json.dump(dict(target=a.target, config_from=src, selected=state['selected']), open(log_path, 'w'), indent=1)
        best = gl_plus.parse(state['selected'])
    jobs = [pool.submit(run, a.target, t, best, f, s, a.threads, True) for f in t['folds'] for s in range(5)]
    [j.result() for j in jobs]
    print(f'[{time.strftime("%H:%M")}] {a.target}: final runs of {gl_plus.name(best)} done', flush=True)


if __name__ == '__main__':
    main()
