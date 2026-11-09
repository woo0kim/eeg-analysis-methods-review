"""Linear SVM on DE features, leave one subject out, for the comparison with DGCNN's LOSO runs.

One session index at a time, each session standardised on its own (the normalisation that gave DGCNN its best LOSO
numbers). C comes from an inner split that never sees the test subject: train on 13 subjects, score the next subject
(cyclically), then refit on all 14.

usage: python svm_loso.py <de dir> <out.json>
"""
import json
import multiprocessing as mp
import os
import sys

import numpy as np
import scipy.io as sio
from sklearn.svm import LinearSVC

de_dir, out = sys.argv[1:3]
y = sio.loadmat(os.path.join(de_dir, 'DE_labels.mat'))['de_labels'].ravel().astype(int)
Cs = [2.0 ** k for k in (-14, -12, -10, -8, -6, -4)]


def load(i):
    x = np.transpose(sio.loadmat(os.path.join(de_dir, f'DE_{i + 1}.mat'))['DE_feature'], (1, 0, 2)).astype(np.float32)
    return ((x - x.mean((0, 1), keepdims=True)) / x.std((0, 1), keepdims=True)).reshape(3394, -1)


def fold(a):
    ses, t = a
    X = {s: load(s * 3 + ses) for s in range(15)}
    v = (t + 1) % 15
    tr = [s for s in range(15) if s not in (t, v)]
    Xt, yt = np.concatenate([X[s] for s in tr]), np.tile(y, len(tr))
    val = [LinearSVC(C=c, max_iter=2000, dual=False).fit(Xt, yt).score(X[v], y) for c in Cs]
    c = Cs[int(np.argmax(val))]
    tr = [s for s in range(15) if s != t]
    m = LinearSVC(C=c, max_iter=2000, dual=False).fit(np.concatenate([X[s] for s in tr]), np.tile(y, len(tr)))
    acc = 100 * m.score(X[t], y)
    print(ses, t, c, round(acc, 2), flush=True)
    return acc


with mp.Pool(12) as p:
    A = np.array(p.map(fold, [(ses, t) for ses in range(3) for t in range(15)], chunksize=1)).reshape(3, 15)
print(f'LOSO SVM: {A.mean():.2f} (SD over subjects {A.mean(0).std():.2f})')
json.dump({'per_session_subject': np.round(A, 3).tolist(), 'mean': float(A.mean()), 'sd_subjects': float(A.mean(0).std())}, open(out, 'w'))
