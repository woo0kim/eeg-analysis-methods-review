"""Compact copies of predict.py's outputs: output_subject<i>.npz -> subject<i>.npz with y_true / y_pred only.

The released files also hold the LSTM memory cells (large); the labels and predictions are flattened as the released
summary.py does (its lines 52-66).
usage: python export_preds.py <predict.py output dir> <preds dir>
"""
import glob
import os
import sys

import numpy as np

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
for f in sorted(glob.glob(os.path.join(src, 'output_subject*.npz'))):
    with np.load(f, allow_pickle=True) as d:
        if len(d['y_true'].shape) == 1:                 # summary.py: one array per night, or one flat array
            yt = np.hstack(d['y_true']) if len(d['y_true']) < 10 else d['y_true']
            yp = np.hstack(d['y_pred']) if len(d['y_pred']) < 10 else d['y_pred']
        else:
            yt, yp = d['y_true'].flatten(), d['y_pred'].flatten()
        nights = [len(n) for n in d['y_true']] if len(d['y_true']) < 10 else [len(yt)]
    i = int(os.path.basename(f)[len('output_subject'):-4])
    np.savez_compressed(os.path.join(dst, f'subject{i}.npz'), y_true=yt.astype(np.int8), y_pred=yp.astype(np.int8),
                        night_lengths=np.array(nights))
    print(f'subject{i}: {len(yt)} epochs in {len(nights)} night(s)')
