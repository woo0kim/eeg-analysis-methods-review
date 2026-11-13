"""The paper's feature-learning chain on precomputed features: standardise -> DFT -> AXB -> LNT -> XGBoost.

Every stage the authors' bundle implements is called from the bundle itself (upstream/, unchanged):
  - standardisation: StandardScaler fitted on the training samples (mwl_gl_si7.standardize_train_test does the same);
  - DFT: lib.feat_utils.feature_selection(FStype='DFT_entropy', B=32), keeping the 50% of features with the lowest DFT loss
    (fs_thrs 0.5), with the bundle's patch that makes the loss functions return scalars (mwl_gl_si7.patch_dft_loss_scalar_outputs).
    The loss of each feature is computed independently, so the columns are scored in parallel processes and ranked once, with the
    same np.argsort as feature_selection: the selection is identical to one call on all columns;
  - LNT: mwl_gl_si7.run_groupwise_lnt (groups of 8 consecutive features, 2 components requested per group, concatenated with the
    selected features), which calls lnt_generate.lnt_kernel / select_comb;
  - classifier: mwl_gl_si7.build_xgb (500 trees, depth 3, learning rate 0.1, subsample 0.8, colsample 0.8, random_state = seed);
    only n_jobs is set (the bundle uses every core; the servers are shared).
AXB is not in the bundle ("No auxiliary branch ... in this version"), so it is ours: an XGBoost with build_xgb's settings, trained on
the DFT-selected features; the features with positive gain importance are kept, ranked by gain, at most half of the DFT set, and
passed on in DFT order (the order the bundle's LNT groups them in).
"""
import contextlib
import hashlib
import io
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.path.abspath(os.environ.get('GL_UPSTREAM', os.path.join(HERE, '../upstream')))
sys.path.insert(0, UPSTREAM)
import mwl_gl_si7 as BUNDLE  # noqa: E402  (the bundle's script; importing it runs nothing)
from lib.feat_utils import feature_selection  # noqa: E402

VARIANTS = {   # name -> (features, AXB)
    'ddtf-gl': ('ddtf', True),          # the paper's method, maps with the settings calibrated on Safari et al.'s
    'ddtf-gl-noaxb': ('ddtf', False),   # without the auxiliary XGBoost (the bundle's chain on dDTF maps)
    'ddtf-gl-v0': ('ddtf-v0', True),    # maps with our first settings (ablation)
    'ddtf-gl-noaxb-v0': ('ddtf-v0', False),
    'dwt-gl': ('dwt', False),           # the bundle's method: DWT statistics, no AXB
    'dwt-gl-axb': ('dwt', True),
    'dc-gl': ('dc', False),             # control: only each channel's mean, through the bundle's chain
    'safari-gl': ('safari', True),      # the paper's method on Safari et al.'s released SIFT maps (STEW-6s only; safari_maps.py)
    'safari-gl-gpu': ('safari', True),  # the same with XGBoost (AXB and classifier) on the GPU: the paper's "Our Method (GPU)"
}
DEVICE = {'safari-gl-gpu': 'cuda'}     # every other variant: XGBoost on the CPU
_X, _y, _B = None, None, None


def config(seed):
    return BUNDLE.Config(seed=seed, verbose=False, progress=False)


def _dft_block(cols):
    with contextlib.redirect_stderr(io.StringIO()):          # feature_selection's tqdm bar
        _, score, _ = feature_selection(_X[:, cols], _y, FStype='DFT_entropy', thrs=1.0, B=_B)
    return score


def dft_scores(X, y, B, workers, cache=None):
    """DFT loss of every column (lower = more discriminant), as feature_selection computes it; cached by the training data."""
    global _X, _y, _B
    if cache:
        h = hashlib.md5(np.ascontiguousarray(y).tobytes() + str(X.shape).encode() + X[:: max(1, len(X) // 64)].tobytes()).hexdigest()[:16]
        path = os.path.join(cache, f'dft_B{B}_{h}.npy')
        if os.path.exists(path):
            return np.load(path), True
    BUNDLE.patch_dft_loss_scalar_outputs()
    _X, _y, _B = X, y, B
    blocks = [b for b in np.array_split(np.arange(X.shape[1]), max(1, min(X.shape[1], workers * 8))) if len(b)]
    import multiprocessing as mp
    with mp.get_context('fork').Pool(workers) as p:
        score = np.concatenate(p.map(_dft_block, blocks, chunksize=1))
    _X = _y = None
    if cache:
        os.makedirs(cache, exist_ok=True)
        np.save(path, score)
    return score, False


def axb_select(Xd, y, cfg, threads, keep_frac=0.5, device='cpu'):
    """Positions (into Xd's columns) kept by the auxiliary XGBoost, in Xd's order, and every column's gain."""
    m = BUNDLE.build_xgb(cfg)
    m.set_params(n_jobs=threads, device=device)
    m.fit(Xd, y)
    gain = np.zeros(Xd.shape[1])
    for k, v in m.get_booster().get_score(importance_type='gain').items():
        gain[int(k[1:])] = v
    k = min(int((gain > 0).sum()), int(keep_frac * Xd.shape[1]))
    return np.sort(np.argsort(-gain, kind='stable')[:k]), gain


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*a, **kw)


def fit_predict(X, y, tr, evals, seed, axb=True, threads=8, cache=None, log=print, device='cpu'):
    """Fit on X[tr] / y[tr]; return class probabilities for every index array in `evals`, and a record of the run."""
    from sklearn.preprocessing import StandardScaler
    cfg = config(seed)
    rec, t = {'n_train': int(len(tr))}, time.time()
    scaler = StandardScaler().fit(X[tr])
    Xtr = scaler.transform(X[tr]).astype(np.float32)
    ytr = y[tr]
    rec['n_features'] = {'input': int(X.shape[1])}
    score, hit = dft_scores(Xtr, ytr, cfg.fs_B, threads, cache)
    sel = np.argsort(score)[:int(cfg.fs_thrs * score.size)]      # = feature_selection's selected_idx
    rec['n_features']['dft'] = int(len(sel))
    rec['seconds'] = {'scale_dft': round(time.time() - t, 1)}
    log(f'DFT: {len(sel)} of {X.shape[1]} features ({"cached" if hit else "computed"}, {time.time() - t:.0f} s)')
    if axb:
        t1 = time.time()
        keep, gain = axb_select(Xtr[:, sel], ytr, cfg, threads, device=device)
        rec['n_features']['axb_positive_gain'] = int((gain > 0).sum())
        sel = sel[keep]
        rec['n_features']['axb'] = int(len(sel))
        rec['seconds']['axb'] = round(time.time() - t1, 1)
        log(f'AXB: {len(sel)} kept ({rec["n_features"]["axb_positive_gain"]} with positive gain, {time.time() - t1:.0f} s)')
    t1 = time.time()
    Xev = np.concatenate([scaler.transform(X[e]).astype(np.float32)[:, sel] if len(e) else np.zeros((0, len(sel)), np.float32) for e in evals]
                         + [np.zeros((0, len(sel)), np.float32)])     # protocols without a validation part pass an empty index array
    ftr, fev, kernels = quiet(BUNDLE.run_groupwise_lnt, Xtr[:, sel], ytr, Xev, cfg)
    rec['n_features']['lnt'] = int(ftr.shape[1] - len(sel))
    rec['n_features']['final'] = int(ftr.shape[1])
    rec['seconds']['lnt'] = round(time.time() - t1, 1)
    log(f'LNT: {rec["n_features"]["lnt"]} new features from {len(kernels)} groups; classifier input {ftr.shape[1]}')
    t1 = time.time()
    clf = BUNDLE.build_xgb(cfg)
    clf.set_params(n_jobs=threads, device=device)                 # device: the paper's GPU variant; the trees are XGBoost's 'hist' on both
    clf.fit(ftr, ytr)
    rec['seconds']['xgboost'] = round(time.time() - t1, 1)
    dump = clf.get_booster().get_dump()
    rec['model'] = {'trees': len(dump), 'nodes': int(sum(d.count('\n') for d in dump)), 'classes': int(len(np.unique(ytr)))}
    t1 = time.time()
    proba = clf.predict_proba(fev) if len(fev) else np.zeros((0, len(np.unique(ytr))))
    rec['seconds']['predict'] = round(time.time() - t1, 5)
    rec['seconds']['fit_total'] = round(time.time() - t, 1)
    log(f'XGBoost: {rec["model"]["trees"]} trees, {rec["model"]["nodes"]} nodes; fit in {rec["seconds"]["fit_total"]:.0f} s')
    out, s = [], 0
    for e in evals:
        out.append(proba[s:s + len(e)])
        s += len(e)
    return out, rec
