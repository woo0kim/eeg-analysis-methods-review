"""One run of the feature-tuning experiment: run_gl.py with a gl_plus configuration instead of a variant.

  python run_gl_plus.py --dataset STEW --protocol cbramod --task rest-task --fold 0 --seed 0 --cfg 'feat=ddtf+spec;proj=lnt+lda' --out <dir>

Same outputs as run_gl.py: <dir>/metrics.json (validation and test scores, feature counts, timings, model size, the configuration)
and test_preds.npz. The validation scores choose the configuration (tune.py); the test scores are only reported.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gl_plus  # noqa: E402
import protocols  # noqa: E402
from run_gl import WORK, scores  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--task', default=None)
    ap.add_argument('--fold', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cfg', default='')
    ap.add_argument('--out', required=True)
    ap.add_argument('--threads', type=int, default=int(os.environ.get('GL_THREADS', 8)))
    ap.add_argument('--train-sub', type=int, default=1, help='keep every k-th training sample (search runs on SEED / SEED-V)')
    a = ap.parse_args()
    c = gl_plus.parse(a.cfg)
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, 'log.txt'), 'w')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    t0 = time.time()
    fdir = os.path.join(WORK, 'features', a.dataset, 'ddtf')
    meta = dict(np.load(os.path.join(fdir, 'meta.npz')))
    if os.path.exists(os.path.join(fdir, 'split_keys.json')):
        meta['_split_keys'] = json.load(open(os.path.join(fdir, 'split_keys.json')))
    X, ranges = gl_plus.load(WORK, a.dataset, c['feat'])
    sp = protocols.splits(a.dataset, meta, a.protocol, a.task, a.fold, a.seed)
    if a.train_sub > 1:
        sp['train'] = sp['train'][::a.train_sub]
    y = sp['y']
    groups = gl_plus.subjects(a.dataset, meta)
    log(f'{a.dataset} {a.protocol} task={a.task} fold={a.fold} seed={a.seed}: {gl_plus.name(c)}; features {X.shape} {ranges}; '
        f'train {len(sp["train"])} ({len(np.unique(groups[sp["train"]]))} subjects), val {len(sp["val"])}, test {len(sp["test"])}')
    (pv, pt), rec = gl_plus.fit_predict(X, y, sp['train'], [sp['val'], sp['test']], a.seed, c, groups, threads=a.threads,
                                        cache=os.path.join(WORK, 'cache', a.dataset, '+'.join(c['feat'])), log=log)
    import socket
    res = dict(dataset=a.dataset, protocol=a.protocol, task=a.task, fold=a.fold, seed=a.seed, variant='ddtf-gl-plus', host=socket.gethostname(),
               threads=a.threads, train_sub=a.train_sub, blocks={k: list(v) for k, v in ranges.items()}, val=scores(y[sp['val']], pv), test=scores(y[sp['test']], pt), **rec)
    res['seconds']['predict_ms_per_sample'] = round(1000 * rec['seconds']['predict'] / max(1, len(sp['val']) + len(sp['test'])), 6)
    res['seconds']['run'] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(a.out, 'metrics.json'), 'w'), indent=1)
    np.savez_compressed(os.path.join(a.out, 'test_preds.npz'), y=y[sp['test']].astype(np.int8), pred=pt.argmax(1).astype(np.int8),
                        proba=pt.astype(np.float16), y_val=y[sp['val']].astype(np.int8), pred_val=pv.argmax(1).astype(np.int8),
                        proba_val=pv.astype(np.float16))
    log('test  ' + ' '.join(f'{k} {v:.4f}' for k, v in res['test'].items() if k != 'cm'))
    if res['val']:
        log('val   ' + ' '.join(f'{k} {v:.4f}' for k, v in res['val'].items() if k != 'cm'))
    log(f'done in {res["seconds"]["run"]:.0f} s')


if __name__ == '__main__':
    main()
