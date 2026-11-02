"""Per-epoch predictions of each fold, from test_lseqsleepnet.py's test_ret.mat, with the released evaluate.py's
aggregation: every epoch is classified in up to 200 overlapping 200-epoch windows, and evaluate.py fuses them by
summing log10 softmax probabilities ('multiplication', its default). aggregate_mul() and softmax() are copied from
sleepedf-20/network/lseqsleepnet/evaluate.py (L-SeqSleepNet, CC-BY-NC-4.0; changed lines marked [MOD]).
Labels are read from the released mat files of the fold's test list, as evaluate.py does.

usage: python export_preds.py <runs dir with repeat<r>/n<n>/test_ret.mat> <preds dir> [<repeat> ...]
writes <preds dir>/repeat<r>_n<n>.npz with y_true, y_pred (stage 0..4) and the night lengths.
"""
import os
import sys

import hdf5storage
import numpy as np

SRC = os.path.expanduser(os.environ.get('LSEQ_SRC', '~/repro-sleepmodels/lseqsleepnet-upstream'))
LISTS = os.path.join(SRC, 'sleepedf-20/file_list_20sub/eeg')
SEQ_LEN, NSTAGE = 200, 5          # subseqlen 10 x nsubseq 20, as evaluate_performance.sh


def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div


def aggregate_mul(score):
    fused_score = None
    for i in range(SEQ_LEN):  # [MOD] config.seq_len -> SEQ_LEN
        prob_i = np.log10(softmax(np.squeeze(score[i, :, :])))
        prob_i = np.concatenate((np.ones((SEQ_LEN - 1, NSTAGE)), prob_i), axis=0)  # [MOD] config.* -> constants
        prob_i = np.roll(prob_i, -(SEQ_LEN - i - 1), axis=0)  # [MOD]
        if fused_score is None:
            fused_score = prob_i
        else:
            fused_score += prob_i
    label = np.argmax(fused_score, axis=-1) + 1 # +1 as label stored in mat files start from 1
    return label


def read_list(path):
    """[(mat file, n epochs)] of a released file list; paths are relative to network/lseqsleepnet."""
    base = os.path.join(SRC, 'sleepedf-20/network/lseqsleepnet')
    return [(os.path.normpath(os.path.join(base, l.split()[0])), int(l.split()[1])) for l in open(path) if l.strip()]


def main():
    runs, dst = sys.argv[1], sys.argv[2]
    repeats = [int(r) for r in sys.argv[3:]] or [1]
    os.makedirs(dst, exist_ok=True)
    for r in repeats:
        for n in range(1, 21):
            ret = os.path.join(runs, f'repeat{r}', f'n{n}', 'test_ret.mat')
            if not os.path.exists(ret):
                print('missing', ret)
                continue
            score = np.transpose(hdf5storage.loadmat(file_name=ret)['score'], (1, 0, 2))   # as evaluate.py
            files = read_list(os.path.join(LISTS, f'test_list_n{n}.txt'))
            yt, yp, start = [], [], 0
            for path, N in files:
                lab = np.squeeze(np.transpose(np.array(hdf5storage.loadmat(file_name=path)['label']), (1, 0)))
                yp.append(aggregate_mul(score[:, start:start + N - (SEQ_LEN - 1)]) - 1)
                yt.append(lab.astype(int) - 1)
                start += N - (SEQ_LEN - 1)
            assert start == score.shape[1], (ret, start, score.shape)
            np.savez_compressed(os.path.join(dst, f'repeat{r}_n{n}.npz'), y_true=np.concatenate(yt).astype(np.int8),
                                y_pred=np.concatenate(yp).astype(np.int8), night_lengths=[len(y) for y in yt])
            print(f'repeat {r} fold {n}: {sum(len(y) for y in yt)} epochs, {len(files)} night(s)')


if __name__ == '__main__':
    main()
