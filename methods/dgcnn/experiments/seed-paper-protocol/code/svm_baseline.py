"""Linear SVM on the same DE features and the same 9 / 6 clip split: a check of the input data, not of DGCNN.

Zheng & Lu (2015) report 83.99 (9.72) for a linear SVM on all five bands of SEED's DE features, a number papers in this
field quote widely, so an SVM near it says these features match the ones the literature used. C is searched over
2^-10 .. 2^10 as they do. Three C rules are printed: one C per session chosen on the test set, one C for all sessions
chosen on the test set, and one C for all sessions chosen on validation clips (train on clips 1-7, score clips 8-9, then
refit on clips 1-9), the rule that compares fairly with DGCNN's validation-chosen learning rate.
Features: per-band standardisation with the training set.

usage: python svm_baseline.py <de dir> <out.json>
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
Cs = [2.0 ** k for k in range(-10, 11)]
TRIAL_LEN = [235, 233, 206, 238, 185, 195, 237, 216, 265, 237, 235, 233, 235, 238, 206]   # s, SEED's 15 clips
assert sum(TRIAL_LEN[:9]) == 2010


def one(s):
    X = np.transpose(sio.loadmat(os.path.join(de_dir, f'DE_{s + 1}.mat'))['DE_feature'], (1, 0, 2)).astype(np.float64)
    tr, te = X[:2010], X[2010:]
    m, sd = tr.mean((0, 1), keepdims=True), tr.std((0, 1), keepdims=True)
    tr, te = ((tr - m) / sd).reshape(2010, -1), ((te - m) / sd).reshape(1384, -1)
    test = [100 * LinearSVC(C=c, max_iter=3000, dual=False).fit(tr, y[:2010]).score(te, y[2010:]) for c in Cs]
    e7 = sum(TRIAL_LEN[:7])                                                  # end of clip 7
    X7, Xv = X[:e7], X[e7:2010]
    m, sd = X7.mean((0, 1), keepdims=True), X7.std((0, 1), keepdims=True)
    X7, Xv = ((X7 - m) / sd).reshape(e7, -1), ((Xv - m) / sd).reshape(2010 - e7, -1)
    val = [100 * LinearSVC(C=c, max_iter=3000, dual=False).fit(X7, y[:e7]).score(Xv, y[e7:2010]) for c in Cs]
    return test, val


with mp.Pool(10) as p:
    r = p.map(one, range(45))
A, V = np.array([t for t, _ in r]), np.array([v for _, v in r])
best2 = lambda v: np.sort(v.reshape(15, 3), axis=1)[:, 1:].ravel()
peak = A.max(1)
c = A.mean(0).argmax()
cv = V.mean(0).argmax()
rows = {'C per session, chosen on test': (peak, Cs), f'one C, chosen on test (2^{c - 10})': (A[:, c], None),
        f'one C, chosen on validation clips (2^{cv - 10})': (A[:, cv], None)}
res = {}
for k, (v, _) in rows.items():
    res[k] = dict(best2_mean=float(best2(v).mean()), best2_sd=float(best2(v).std()), all3_mean=float(v.mean()), all3_sd=float(v.std()),
                  per_session=[round(float(x), 3) for x in v])
    print(f'{k}: best 2 of 3 {best2(v).mean():.2f} ({best2(v).std():.2f}); all 3 {v.mean():.2f} ({v.std():.2f})')
json.dump(res, open(out, 'w'), indent=1)
