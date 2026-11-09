"""STEW -> 4-s windows, and the folds of every task.

STEW (Lim, Sourina, Wang, IEEE TNSRE 2018): 48 subjects, Emotiv EPOC, 14 channels at 128 Hz; 150 s at rest (sub<NN>_lo.txt) and
150 s during the SIMKAP multitasking test (sub<NN>_hi.txt), 19,200 samples each; a 1-9 workload rating after each stage
(ratings.txt, missing for subjects 5, 24 and 42).

Windows, as specified for this experiment: sample_rate 128 Hz, window_sec 4, overlap_ratio 0.25, so 512 samples with a stride of
384: 49 windows per recording, 4,704 in all. Emotiv's raw output rides on an offset of about 4,300 uV, so each recording is
high-pass filtered at 0.3 Hz (the low cut of CBraMod's preprocessing) before windowing; nothing else is done to the signal.
CBraMod and LaBraM embed 200-sample (1-s) patches and their weights fix that, so the same windows are also written resampled to
200 Hz (800 samples, resample_poly 25:16); they read those, every other model the 128 Hz windows.

Tasks (keys of folds.json):
  rest-task    2 classes: the recording, rest or test. All 48 subjects.
  rating3      3 classes from the rating of the recording: 1-3 low, 4-6 moderate, 7-9 high (the paper's "3 workload levels"; the
               abstract does not give its cut-offs). 45 subjects.

Folds: sklearn GroupKFold(n_splits=7) with subject as the group. It takes no seed, and it is deterministic in one environment,
but every subject here has the same number of windows, and with such ties the assignment differs between numpy / scikit-learn
versions, so it is computed once and pinned in results/folds.json. The test set of fold k is GroupKFold's fold k; validation is
the next fold (k + 1 mod 7), from which the checkpoint is chosen; training is the other five.

usage: python preprocess_stew.py <dir with sub*_*.txt> <out dir> <folds.json>
"""
import json
import os
import sys

import mne
import numpy as np
from scipy import signal
from sklearn.model_selection import GroupKFold

src, out, folds_path = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
FS, WIN, OVERLAP = 128.0, 4.0, 0.25
L, STEP = int(FS * WIN), int(FS * WIN * (1 - OVERLAP))
assert (L, STEP) == (512, 384)
ratings = {int(r[0]): (int(r[1]), int(r[2])) for r in np.loadtxt(os.path.join(src, 'ratings.txt'), delimiter=',')}
level = lambda v: 0 if v <= 3 else 1 if v <= 6 else 2

X, sub, stage, rate = [], [], [], []
for s in range(1, 49):
    for st, name in ((0, 'lo'), (1, 'hi')):
        x = np.loadtxt(os.path.join(src, f'sub{s:02d}_{name}.txt')).T                  # (14, 19200), uV
        assert x.shape == (14, 19200), x.shape
        x = mne.filter.filter_data(x, FS, 0.3, None, verbose='ERROR')
        for i in range(0, x.shape[1] - L + 1, STEP):
            X.append(x[:, i:i + L])
            sub.append(s)
            stage.append(st)
            rate.append(ratings[s][st] if s in ratings else 0)
X, sub, stage, rate = np.array(X, dtype=np.float32), np.array(sub), np.array(stage), np.array(rate)
assert len(X) == 96 * 49, len(X)
X200 = signal.resample_poly(X.astype(np.float64), 25, 16, axis=2).astype(np.float32)
assert X200.shape[2] == 800, X200.shape
np.savez(os.path.join(out, 'windows.npz'), x128=X, x200=X200, subject=sub, stage=stage, rating=rate.astype(np.int8))

folds = {}
for task, keep, y in (('rest-task', np.ones(len(sub), bool), stage),
                      ('rating3', rate > 0, np.array([level(v) for v in rate]))):
    idx = np.where(keep)[0]
    groups = sub[idx]
    split = [sorted(set(int(s) for s in groups[te])) for _, te in GroupKFold(n_splits=7).split(idx, groups=groups)]
    folds[task] = {'test_subjects': split, 'n_subjects': len(set(groups.tolist())), 'n_windows': len(idx),
                   'class_windows': np.bincount(y[idx]).tolist()}
    print(task, 'subjects per fold', [len(t) for t in split], 'windows', len(idx), 'per class', np.bincount(y[idx]).tolist())
    assert sorted(s for t in split for s in t) == sorted(set(groups.tolist()))
json.dump(folds, open(folds_path, 'w'), indent=1)
print('SD of a channel, median over windows: %.1f uV' % np.median(X.std(2)))
