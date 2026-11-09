"""L-SeqSleepNet on SEED / SEED-V: per-window predictions and metrics per run.

  python collect.py export <lseq dir> <preds dir>       # runs/<split>/run<r>/test_ret.mat -> npz (needs hdf5storage)
  python collect.py <preds dir> <out dir> > <out dir>/results.txt

Aggregation: the released evaluate.py sums log10 softmax probabilities over the overlapping sequences that cover an
epoch. Written here without its np.squeeze, which fails on a file holding exactly one sequence (a 20-epoch part); the
extra ones evaluate.py pads with do not change the argmax, so this is the same rule.
Metrics in the schema of ../../../../cbramod/experiments/finetune-seed/code/collect.py, on 30-s windows, plus the clip vote.
"""
import glob
import json
import os
import sys

import numpy as np

SEQ = 20                 # sub_seq_len 5 x nsubseq 4
NAMES = ['balanced_acc', 'kappa', 'weighted_f1', 'accuracy', 'macro_f1', 'clip_vote_acc']


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def aggregate(score):    # (SEQ, n_windows, K) -> argmax per epoch
    nw, k = score.shape[1:]
    fused = np.zeros((nw + SEQ - 1, k))
    for i in range(SEQ):
        fused[i:i + nw] += np.log10(softmax(score[i]))
    return fused.argmax(1)


def export(root, dst):
    import hdf5storage
    os.makedirs(dst, exist_ok=True)
    for ret in sorted(glob.glob(os.path.join(root, '*', 'runs', '*', 'run*', 'test_ret.mat'))):
        ds, _, split, run = ret.split(os.sep)[-5:-1]
        score = np.transpose(hdf5storage.loadmat(file_name=ret)['score'], (1, 0, 2))
        files = [(l.split('\t')[0], int(l.split('\t')[1])) for l in open(os.path.join(root, ds, f'{split}_test_eeg.txt')) if l.strip()]
        yt, yp, clip, sess, start = [], [], [], [], 0
        for path, n in files:
            tag = os.path.basename(path)[:-len('_eeg.mat')]
            z = np.load(os.path.join(root, ds, tag + '.npz'))
            assert len(z['label']) == n
            yt.append(z['label'] - 1)
            yp.append(aggregate(score[:, start:start + n - (SEQ - 1)]))
            clip.append(z['clip'])
            sess.append(np.full(n, len(sess)))
            start += n - (SEQ - 1)
        assert start == score.shape[1], (start, score.shape)
        out = os.path.join(dst, f'{ds}__{split}__l-seqsleepnet__s{run[3:]}.npz')
        np.savez_compressed(out, y=np.concatenate(yt).astype(np.int8), pred=np.concatenate(yp).astype(np.int8),
                            clip=np.concatenate(clip).astype(np.int8), file_id=np.concatenate(sess).astype(np.int16))
        print(out, sum(n for _, n in files), 'windows')


def collect(pdir, out):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
    os.makedirs(out, exist_ok=True)
    rows = []
    for f in sorted(glob.glob(os.path.join(pdir, '*.npz'))):
        ds, split, model, seed = os.path.basename(f)[:-4].split('__')
        z = np.load(f)
        y, p = z['y'].astype(int), z['pred'].astype(int)
        keys = z['file_id'].astype(int) * 100 + z['clip'].astype(int)
        votes = [np.bincount(p[keys == k], minlength=5).argmax() == y[keys == k][0] for k in np.unique(keys)]
        rows.append(dict(dataset=ds, split=split, model=model, seed=int(seed[1:]), metric_names=NAMES,
                         metrics=[balanced_accuracy_score(y, p), cohen_kappa_score(y, p), f1_score(y, p, average='weighted'),
                                  accuracy_score(y, p), f1_score(y, p, average='macro'), float(np.mean(votes))]))
    summary = []
    for key in sorted({(r['dataset'], r['split'], r['model']) for r in rows}):
        rs = [r for r in rows if (r['dataset'], r['split'], r['model']) == key]
        a = np.array([r['metrics'] for r in rs])
        summary.append(dict(dataset=key[0], split=key[1], model=key[2], n=len(rs), metric_names=NAMES, mean=a.mean(0).tolist(),
                            std=(a.std(0, ddof=1) if len(rs) > 1 else np.zeros(6)).tolist()))
        print(f'{key[0]:7s} {key[1]:9s} n={len(rs)} ' + ' '.join(f'{n} {m:.4f}±{s:.4f}' for n, m, s in zip(NAMES, a.mean(0), summary[-1]['std'])))
    json.dump(dict(runs=rows, summary=summary), open(os.path.join(out, 'results.json'), 'w'), indent=1)


if __name__ == '__main__':
    export(sys.argv[2], sys.argv[3]) if sys.argv[1] == 'export' else collect(sys.argv[1], sys.argv[2])
