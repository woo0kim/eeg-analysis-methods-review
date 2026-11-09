"""L-SeqSleepNet on STEW: per-window predictions and metrics per run.

  python collect.py export <lseq STEW dir> <preds dir>     # runs/<task>/fold<k>/run<r>/test_ret.mat -> npz (needs hdf5storage)
  python collect.py <preds dir> <out dir> > <out dir>/results.txt

One sequence per recording: its five 30-s windows (sub_seq_len 5 x nsubseq 1), so each window gets exactly one score and the
released evaluate.py's sum of log10 softmax probabilities over overlapping sequences reduces to the argmax of that score.
Each run is scored on the pooled predictions of its 7 test folds, as every other STEW model
(../../../../cbramod/experiments/finetune-stew/code/collect.py), on 30-s windows; plus the recording vote (majority of a
recording's five windows, ties to the lower class).
"""
import glob
import json
import os
import sys

import numpy as np

SEQ = 5
NAMES = ['accuracy', 'balanced_acc', 'kappa', 'weighted_f1', 'macro_f1', 'recording_vote_acc']


def export(root, dst):
    import hdf5storage
    os.makedirs(dst, exist_ok=True)
    for ret in sorted(glob.glob(os.path.join(root, 'runs', '*', 'fold*', 'run*', 'test_ret.mat'))):
        task, fold, run = ret.split(os.sep)[-4:-1]
        score = np.transpose(hdf5storage.loadmat(file_name=ret)['score'], (1, 0, 2))      # (SEQ, sequences, classes)
        files = [(l.split('\t')[0], int(l.split('\t')[1])) for l in open(os.path.join(root, f'{task}_{fold}_test_eeg.txt')) if l.strip()]
        assert score.shape[:2] == (SEQ, len(files)) and all(n == SEQ for _, n in files), (score.shape, len(files))
        yt, yp, subj, rec = [], [], [], []
        for i, (path, _) in enumerate(files):
            lab = hdf5storage.loadmat(file_name=path)['label'].ravel().astype(int) - 1
            yt.append(lab)
            yp.append(score[:, i].argmax(1))
            tag = os.path.basename(path)[len(task) + 1:-len('_eeg.mat')]                     # e.g. sub03_lo
            subj.append(np.full(SEQ, int(tag[3:5])))
            rec.append(np.full(SEQ, i))
        out = os.path.join(dst, f'{task}__{fold}__l-seqsleepnet__s{run[3:]}.npz')
        np.savez_compressed(out, y=np.concatenate(yt).astype(np.int8), pred=np.concatenate(yp).astype(np.int8),
                            subject=np.concatenate(subj).astype(np.int8), recording=np.concatenate(rec).astype(np.int16))
        print(out, len(files), 'recordings')


def score(y, p, rec):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
    votes = [np.bincount(p[rec == r], minlength=5).argmax() == y[rec == r][0] for r in np.unique(rec)]
    return [accuracy_score(y, p), balanced_accuracy_score(y, p), cohen_kappa_score(y, p), f1_score(y, p, average='weighted'),
            f1_score(y, p, average='macro'), float(np.mean(votes))]


def collect(pdir, out):
    os.makedirs(out, exist_ok=True)
    runs = {}
    for f in sorted(glob.glob(os.path.join(pdir, '*.npz'))):
        task, fold, model, seed = os.path.basename(f)[:-4].split('__')
        runs.setdefault((task, model, int(seed[1:])), {})[int(fold[4:])] = np.load(f)
    rows = []
    for (task, model, seed), folds in sorted(runs.items()):
        if sorted(folds) != list(range(7)):
            print('incomplete', task, model, seed, sorted(folds))
            continue
        y = np.concatenate([folds[k]['y'] for k in range(7)]).astype(int)
        p = np.concatenate([folds[k]['pred'] for k in range(7)]).astype(int)
        rec = np.concatenate([folds[k]['recording'].astype(int) + 1000 * k for k in range(7)])
        per = [score(folds[k]['y'].astype(int), folds[k]['pred'].astype(int), folds[k]['recording'].astype(int)) for k in range(7)]
        rows.append(dict(task=task, model=model, seed=seed, metric_names=NAMES, metrics=score(y, p, rec), per_fold=per,
                         unit='30-s windows, 5 per recording'))
    summary = []
    for key in sorted({(r['task'], r['model']) for r in rows}):
        rs = [r for r in rows if (r['task'], r['model']) == key]
        a, f = np.array([r['metrics'] for r in rs]), np.array([r['per_fold'] for r in rs])
        summary.append(dict(task=key[0], model=key[1], n_seeds=len(rs), metric_names=NAMES, mean=a.mean(0).tolist(),
                            std=(a.std(0, ddof=1) if len(rs) > 1 else np.zeros(len(NAMES))).tolist(), fold_sd=f.std(1, ddof=1).mean(0).tolist()))
        print(f'{key[0]:10s} {key[1]:14s} runs={len(rs)} ' + ' '.join(f'{n} {m:.4f}±{s:.4f}' for n, m, s in zip(NAMES, a.mean(0), summary[-1]['std']))
              + f'  fold SD(acc) {summary[-1]["fold_sd"][0]:.4f}')
    json.dump(dict(runs=rows, summary=summary), open(os.path.join(out, 'results.json'), 'w'), indent=1)


if __name__ == '__main__':
    export(sys.argv[2], sys.argv[3]) if sys.argv[1] == 'export' else collect(sys.argv[1], sys.argv[2])
