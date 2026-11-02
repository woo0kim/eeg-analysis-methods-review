"""L-SeqSleepNet on the CBraMod-pipeline split: per-epoch predictions and metrics per run.

  python collect.py export <runs dir> <preds dir>   # run<r>/test_ret.mat -> <preds dir>/run<r>.npz (needs the mat files)
  python collect.py <preds dir> <out dir> > <out dir>/results.txt
Export uses the released evaluate.py aggregation (../../sleepedf20-loso/code/export_preds.py) over the nights of
test_list.txt, in its order; <preds dir>/keys.json (prepare_split_data.py) gives each night's absolute epoch indices.
Metrics on all test epochs and on the common test set, in the schema of
../../../../cbramod/experiments/finetune-sleepedf/code/collect.py.
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '../../sleepedf20-loso/code'))
sys.path.insert(0, os.path.join(HERE, '../../../../cbramod/experiments/finetune-sleepedf/code'))
DATA = os.path.expanduser(os.environ.get('LSEQ_DATA78', '~/repro-sleepmodels/data/lseq_sc78'))


def export(runs, dst):
    import hdf5storage
    from export_preds import SEQ_LEN, aggregate_mul
    os.makedirs(dst, exist_ok=True)
    files = [(l.split('\t')[0], int(l.split('\t')[1])) for l in open(os.path.join(DATA, 'test_list.txt')) if l.strip()]
    for ret in sorted(glob.glob(os.path.join(runs, 'run*', 'test_ret.mat'))):
        r = int(os.path.basename(os.path.dirname(ret))[3:])
        score = np.transpose(hdf5storage.loadmat(file_name=ret)['score'], (1, 0, 2))
        yt, yp, start = [], [], 0
        for path, n in files:
            lab = np.squeeze(np.transpose(np.array(hdf5storage.loadmat(file_name=path)['label']), (1, 0)))
            yp.append(aggregate_mul(score[:, start:start + n - (SEQ_LEN - 1)]) - 1)
            yt.append(lab.astype(int) - 1)
            start += n - (SEQ_LEN - 1)
        assert start == score.shape[1]
        np.savez_compressed(os.path.join(dst, f'run{r}.npz'), y_true=np.concatenate(yt).astype(np.int8),
                            y_pred=np.concatenate(yp).astype(np.int8),
                            nights=[os.path.basename(p)[:6] for p, _ in files], night_lengths=[n for _, n in files])
        print(f'run {r}: {start + len(files) * (SEQ_LEN - 1)} epochs in {len(files)} nights')


def collect(pdir, out):
    import split_metrics
    os.makedirs(out, exist_ok=True)
    keys = json.load(open(os.path.join(pdir, 'keys.json')))
    per_run = {}
    for f in sorted(glob.glob(os.path.join(pdir, 'run*.npz'))):
        r = int(os.path.basename(f)[3:-4])
        d = np.load(f)
        pred, lab, start = {}, {}, 0
        for rec, n in zip(d['nights'], d['night_lengths']):
            assert len(keys[rec]) == n, (rec, len(keys[rec]), n)
            for j, k in enumerate(keys[rec]):
                pred[(rec, k)], lab[(rec, k)] = int(d['y_pred'][start + j]), int(d['y_true'][start + j])
            start += n
        per_run[r] = split_metrics.score_run(pred, lab)
    res = {'models': {}}
    if per_run:
        names = list(next(iter(per_run.values()))['common'])
        for part in ('common', 'all'):
            res['models']['l-seqsleepnet' + ('' if part == 'common' else '@all')] = {
                'label': 'L-SeqSleepNet', 'implementation': 'released (TF 1.15)', 'epochs': part,
                'seeds': sorted(per_run), 'per_seed': {r: per_run[r][part] for r in sorted(per_run)},
                'mean': {k: float(np.mean([per_run[r][part][k] for r in per_run])) for k in names},
                'std': {k: float(np.std([per_run[r][part][k] for r in per_run], ddof=1)) if len(per_run) > 1 else None
                        for k in names}}
    json.dump(res, open(os.path.join(out, 'results.json'), 'w'), indent=1, default=float)
    print('L-SeqSleepNet (released code) on Sleep-EDF-78, CBraMod-pipeline split (62 / 8 / 8 subjects), Fpz-Cz\n')
    for m in res['models'].values():
        print(f'{"common test set" if m["epochs"] == "common" else "all kept test epochs"}: '
              f'{m["per_seed"][m["seeds"][0]]["n_epochs"]} epochs, runs {m["seeds"]}')
        for k in ('balanced_acc', 'kappa', 'weighted_f1', 'accuracy', 'macro_f1'):
            sd = f' ± {m["std"][k]:.4f}' if m['std'][k] is not None else ''
            print(f'  {k:13s} {m["mean"][k]:.4f}{sd}   per run: ' + ' '.join(f'{m["per_seed"][r][k]:.4f}' for r in m['seeds']))
        print('  per-stage F1  ' + ' '.join(f'{s} {m["mean"]["f1_" + s]:.4f}' for s in split_metrics.STAGES))


if __name__ == '__main__':
    export(sys.argv[2], sys.argv[3]) if sys.argv[1] == 'export' else collect(sys.argv[1], sys.argv[2])
