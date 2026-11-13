"""One dDTF-GL run: a dataset, protocol (task, fold), seed and variant, on the cached features (extract.py).

  python run_gl.py --dataset STEW --protocol cbramod --task rest-task --fold 0 --seed 0 --variant ddtf-gl --out <run dir>

Writes <run dir>/metrics.json (validation and test scores, feature counts per stage, model size, timings) and test_preds.npz
(labels, predictions and class probabilities of the test samples in the protocol's test order, and of the validation samples).
The validation part, where a protocol has one, is scored and recorded but never used: there is no epoch or model to select.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gl  # noqa: E402
import protocols  # noqa: E402
from sklearn.metrics import (accuracy_score, auc, balanced_accuracy_score, cohen_kappa_score, f1_score,  # noqa: E402
                             precision_recall_curve, roc_auc_score)

WORK = os.path.expanduser(os.environ.get('GL_WORK', '~/repro-gl'))


def scores(y, proba):
    """CBraMod's metrics (balanced accuracy, kappa, weighted F1; binary: AUC-PR as auc(recall, precision), AUROC) plus accuracy and
    macro-F1. Predictions are the most probable class (binary: probability > 0.5, as CBraMod's sigmoid threshold)."""
    if len(y) == 0:
        return {}
    p = proba.argmax(1)
    r = {'balanced_acc': balanced_accuracy_score(y, p), 'kappa': cohen_kappa_score(y, p), 'weighted_f1': f1_score(y, p, average='weighted'),
         'accuracy': accuracy_score(y, p), 'macro_f1': f1_score(y, p, average='macro')}
    if proba.shape[1] == 2 and len(np.unique(y)) == 2:
        prec, rec, _ = precision_recall_curve(y, proba[:, 1], pos_label=1)
        r.update(pr_auc=auc(rec, prec), roc_auc=roc_auc_score(y, proba[:, 1]))
    k = proba.shape[1]
    r['cm'] = np.bincount(k * y + p, minlength=k * k).reshape(k, k).tolist()
    return {a: (float(b) if not isinstance(b, list) else b) for a, b in r.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--task', default=None)
    ap.add_argument('--fold', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--variant', default='ddtf-gl', choices=sorted(gl.VARIANTS))
    ap.add_argument('--out', required=True)
    ap.add_argument('--threads', type=int, default=int(os.environ.get('GL_THREADS', 20)))
    a = ap.parse_args()
    feats, axb = gl.VARIANTS[a.variant]
    fdir = os.path.join(WORK, 'features', a.dataset, feats)
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, 'log.txt'), 'w')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    t0 = time.time()
    info = json.load(open(os.path.join(fdir, 'info.json')))
    meta = dict(np.load(os.path.join(fdir, 'meta.npz')))
    if os.path.exists(os.path.join(fdir, 'split_keys.json')):    # SEED / SEED-V: both LMDBs' key lists, so no LMDB is needed here
        meta['_split_keys'] = json.load(open(os.path.join(fdir, 'split_keys.json')))
    X = np.load(os.path.join(fdir, 'X.npy'), mmap_mode='r')
    sp = protocols.splits(a.dataset, meta, a.protocol, a.task, a.fold, a.seed)
    y = sp['y']
    log(f'{a.dataset} {a.protocol} task={a.task} fold={a.fold} seed={a.seed} variant={a.variant}: features {X.shape} ({feats}'
        + (f', MVAR order {info["order"]}' if feats == 'ddtf' else '') + f'); train {len(sp["train"])}, val {len(sp["val"])}, '
        f'test {len(sp["test"])}; training classes {np.bincount(y[sp["train"]]).tolist()}')
    X = np.asarray(X)
    device = gl.DEVICE.get(a.variant, 'cpu')
    (pv, pt), rec = gl.fit_predict(X, y, sp['train'], [sp['val'], sp['test']], a.seed, axb=axb, threads=a.threads,
                                   cache=os.path.join(WORK, 'cache', a.dataset, feats), log=log, device=device)
    rec['device'] = device
    yv, yt = y[sp['val']], y[sp['test']]
    import socket
    res = dict(dataset=a.dataset, protocol=a.protocol, task=a.task, fold=a.fold, seed=a.seed, variant=a.variant,
               host=socket.gethostname(), threads=a.threads,
               features=dict(kind=feats, **{k: info[k] for k in ('n_features', 'fs', 'samples', 'order', 'dropped_last_channel') if k in info}),
               val=scores(yv, pv), test=scores(yt, pt), **rec)
    res['seconds']['predict_ms_per_sample'] = round(1000 * rec['seconds']['predict'] / max(1, len(sp['val']) + len(sp['test'])), 6)
    res['seconds']['run'] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(a.out, 'metrics.json'), 'w'), indent=1)
    np.savez_compressed(os.path.join(a.out, 'test_preds.npz'), y=yt.astype(np.int8), pred=pt.argmax(1).astype(np.int8),
                        proba=pt.astype(np.float16), y_val=yv.astype(np.int8), pred_val=pv.argmax(1).astype(np.int8),
                        proba_val=pv.astype(np.float16))
    log('test  ' + ' '.join(f'{k} {v:.4f}' for k, v in res['test'].items() if k != 'cm'))
    if res['val']:
        log('val   ' + ' '.join(f'{k} {v:.4f}' for k, v in res['val'].items() if k != 'cm'))
    log(f'done in {res["seconds"]["run"]:.0f} s')


if __name__ == '__main__':
    main()
