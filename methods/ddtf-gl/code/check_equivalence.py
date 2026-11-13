"""Check that our wrappers compute what the authors' bundle computes, on STEW's fold 0 (rest vs. task) training data.

1. DFT: gl.dft_scores (columns scored in parallel processes) equals one call of the bundle's feature_selection on all columns,
   loss by loss, and selects the same features.
2. LNT: mwl_gl_si7.run_groupwise_lnt applied to validation and test stacked (as gl.fit_predict calls it) gives the same rows as
   two separate calls (the LNT kernel is learnt on training data only and applied row by row).
3. The same run twice gives identical predictions (determinism of DFT, AXB, LNT and XGBoost for a seed).

usage: python check_equivalence.py > ../experiments/stew-cbramod-pipeline/results/equivalence.txt
"""
import contextlib
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gl  # noqa: E402
import protocols  # noqa: E402
from lib.feat_utils import feature_selection  # noqa: E402  (upstream, via gl's sys.path)

WORK = os.path.expanduser(os.environ.get('GL_WORK', '~/repro-gl'))
fdir = os.path.join(WORK, 'features', 'STEW', 'ddtf')
X = np.load(os.path.join(fdir, 'X.npy'))
meta = dict(np.load(os.path.join(fdir, 'meta.npz')))
sp = protocols.splits('STEW', meta, 'cbramod', 'rest-task', 0)
y = sp['y']
from sklearn.preprocessing import StandardScaler  # noqa: E402
sc = StandardScaler().fit(X[sp['train']])
Xtr = sc.transform(X[sp['train']]).astype(np.float32)
ytr = y[sp['train']]

par, _ = gl.dft_scores(Xtr, ytr, 32, workers=20, cache=None)
gl.BUNDLE.patch_dft_loss_scalar_outputs()
with contextlib.redirect_stderr(io.StringIO()):
    sel_ref, score_ref, _ = feature_selection(Xtr, ytr, FStype='DFT_entropy', thrs=0.5, B=32)
sel_par = np.argsort(par)[:int(0.5 * par.size)]
print(f'1. DFT, {Xtr.shape[1]} features x {Xtr.shape[0]} training windows: max |loss difference| = {np.abs(par - score_ref).max():.3g}; '
      f'same selected features in the same order: {np.array_equal(sel_par, sel_ref)}')

cfg = gl.config(0)
A, B = sc.transform(X[sp['val']]).astype(np.float32)[:, sel_ref], sc.transform(X[sp['test']]).astype(np.float32)[:, sel_ref]
_, both, _ = gl.quiet(gl.BUNDLE.run_groupwise_lnt, Xtr[:, sel_ref], ytr, np.concatenate([A, B]), cfg)
_, a1, _ = gl.quiet(gl.BUNDLE.run_groupwise_lnt, Xtr[:, sel_ref], ytr, A, cfg)
_, b1, _ = gl.quiet(gl.BUNDLE.run_groupwise_lnt, Xtr[:, sel_ref], ytr, B, cfg)
print(f'2. LNT, validation + test stacked vs. separately: max |difference| = {np.abs(both - np.concatenate([a1, b1])).max():.3g}')

r = [gl.fit_predict(X, y, sp['train'], [sp['test']], 0, axb=True, threads=20, log=lambda m: None) for _ in range(2)]
print(f'3. two identical runs: identical test probabilities: {np.array_equal(r[0][0][0], r[1][0][0])}; '
      f'record {json.dumps(r[0][1]["n_features"])}')
