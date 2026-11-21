"""dDTF-GL with configurable features, feature counts, projection features and classifier: the chain of gl.py, extended for the
feature-tuning experiment (experiments/feature-tuning). With the default configuration it calls gl.py's stages in gl.py's order and
gives gl.fit_predict's predictions exactly.

A configuration is a string, e.g. 'feat=ddtf+graph+spec;dft=0.25;axb=1;proj=lnt+lda+svml;depth=6;trees=1000;lr=0.05':

  feat   feature blocks, concatenated before standardisation
           ddtf   the band-mean dDTF maps (extract.py --features ddtf): 5 * C * C
           graph  node summaries of the same maps, per band and channel: out-strength (flow from the channel to the others), in-strength
                  (flow into it), net flow (out - in) and self-flow (the diagonal): 4 * 5 * C
           spec   log10 band power and relative band power over the window (extract.py --features spec): 10 * C
           spect  log10 band power of each 1-s segment (extract.py --features spect): 5 * C * seconds
  dft    fraction of features kept by the DFT ranking (1 = no DFT selection); the paper's chain keeps 0.5
  axb    1 = the auxiliary XGBoost keeps the features with positive gain, at most half (gl.axb_select); 0 = no AXB
  proj   projection features appended to the selected features before the classifier ('LNT features' and more):
           lnt    the bundle's groupwise LNT (groups of 8, 1-2 projections each), fitted on the training samples as the bundle does
           lda    shrinkage LDA's decision values (1 for 2 classes, k otherwise)
           svml   a linear SVM's decision values (LinearSVC, C = 0.01)
           svmr   an RBF SVM's decision values (SVC, C = 1, gamma 'scale'; above 8,000 training samples a Nystroem approximation with
                  1,000 components and a LinearSVC, C = 0.1)
         lda, svml and svmr are stacked: their training-sample values are out-of-fold predictions of a 5-fold GroupKFold over the
         training subjects (a model never scores samples of subjects it was fitted on), and their validation and test values come from
         a refit on all training samples. They see the selected features (after DFT and AXB).
  depth, trees, lr   the classifier's max_depth, n_estimators, learning_rate (the bundle's: 3, 500, 0.1); subsample and colsample
                     0.8 and random_state = seed as in the bundle
"""
import os
import time

import numpy as np

import gl

BUNDLE = gl.BUNDLE
DEFAULT = dict(feat=('ddtf',), dft=0.5, axb=True, proj=('lnt',), depth=3, trees=500, lr=0.1)
KINDS = ('ddtf', 'graph', 'spec', 'spect')
PROJ = ('lnt', 'lda', 'svml', 'svmr')


def parse(s):
    c = dict(DEFAULT)
    for part in filter(None, (s or '').split(';')):
        k, v = part.split('=')
        if k in ('feat', 'proj'):
            v = tuple(x for x in v.split('+') if x and x != 'none')
            assert all(x in (KINDS if k == 'feat' else PROJ) for x in v), (k, v)
        elif k == 'axb':
            v = bool(int(v))
        elif k in ('depth', 'trees'):
            v = int(v)
        else:
            v = float(v)
        c[k] = v
    return c


def name(c):
    return (f"feat={'+'.join(c['feat'])};dft={c['dft']:g};axb={int(c['axb'])};proj={'+'.join(c['proj']) or 'none'};"
            f"depth={c['depth']};trees={c['trees']};lr={c['lr']:g}")


def graph(Xd, C, chunk=8192):
    """dDTF maps (n, 5*C*C; entry [b, i, j] = flow j -> i) -> (n, 4*5*C): out-strength, in-strength, net flow, self-flow."""
    out = np.empty((len(Xd), 4 * 5 * C), np.float32)
    for s in range(0, len(Xd), chunk):
        A = np.asarray(Xd[s:s + chunk], np.float32).reshape(-1, 5, C, C)
        d = np.diagonal(A, axis1=2, axis2=3)                     # (n, 5, C)
        o, i = A.sum(2) - d, A.sum(3) - d                         # column sums = flow from j; row sums = flow into i
        out[s:s + chunk] = np.concatenate([o, i, o - i, d], 1).reshape(len(A), -1)
    return out


def load(work, dataset, feat):
    """The feature blocks of `feat`, concatenated (n, F) float32, and their column ranges."""
    blocks, ranges, n0 = [], {}, 0
    for k in feat:
        src = 'ddtf' if k == 'graph' else k
        X = np.load(os.path.join(work, 'features', dataset, src, 'X.npy'), mmap_mode='r')
        if k == 'graph':
            import json
            C = len(json.load(open(os.path.join(work, 'features', dataset, 'ddtf', 'info.json')))['channels'])
            X = graph(X, C)
        else:
            X = np.asarray(X, np.float32)
        blocks.append(X)
        ranges[k] = (n0, n0 + X.shape[1])
        n0 += X.shape[1]
    return (blocks[0] if len(blocks) == 1 else np.concatenate(blocks, 1)), ranges


def subjects(dataset, meta):
    """Subject of every sample (for the out-of-fold projections)."""
    if 'subject' in meta:
        return np.asarray(meta['subject']).astype(str)
    keys = [str(k) for k in meta['keys']]
    rule = {'BCIC-IV-2a': lambda k: k[:3], 'PhysioNet-MI': lambda k: k.split('R')[0], 'MentalArithmetic': lambda k: k.split('_')[0],
            'Mumtaz2016': lambda k: k.split('_')[1].rsplit(' ', 1)[0], 'SEED': lambda k: k.split('_')[0],
            'SEED-V': lambda k: k.split('_')[0]}[dataset]
    return np.array([rule(k) for k in keys])


def _estimator(kind, n, F, var, seed):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.svm import SVC, LinearSVC
    if kind == 'lda':
        return LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    if kind == 'svml':
        return LinearSVC(C=0.01, dual='auto', max_iter=5000, random_state=seed)
    if n <= 8000:
        return SVC(C=1.0, kernel='rbf', gamma='scale', decision_function_shape='ovr')
    from sklearn.kernel_approximation import Nystroem
    from sklearn.pipeline import make_pipeline
    return make_pipeline(Nystroem(gamma=1.0 / (F * max(var, 1e-12)), n_components=1000, random_state=seed),
                         LinearSVC(C=0.1, dual='auto', max_iter=5000, random_state=seed))


def _decision(m, X):
    d = m.decision_function(X)
    return (d[:, None] if d.ndim == 1 else d).astype(np.float32)


def stacked(kind, Xtr, ytr, gtr, Xev, seed):
    """Out-of-fold decision values on the training samples (GroupKFold by subject) and a full refit's values on Xev."""
    from sklearn.model_selection import GroupKFold
    n, F = Xtr.shape
    var = float(Xtr.var())
    k = min(5, len(np.unique(gtr)))
    oof = None
    for a, b in GroupKFold(k).split(Xtr, ytr, gtr):
        m = _estimator(kind, len(a), F, var, seed).fit(Xtr[a], ytr[a])
        d = _decision(m, Xtr[b])
        oof = np.zeros((n, d.shape[1]), np.float32) if oof is None else oof
        oof[b] = d
    m = _estimator(kind, n, F, var, seed).fit(Xtr, ytr)
    return oof, _decision(m, Xev) if len(Xev) else np.zeros((0, oof.shape[1]), np.float32)


def fit_predict(X, y, tr, evals, seed, c, groups, threads=8, cache=None, log=print, device='cpu'):
    """gl.fit_predict with configuration c (parse()); returns class probabilities for each index array in evals and a record."""
    from sklearn.preprocessing import StandardScaler
    cfg = gl.config(seed)
    rec, t = {'n_train': int(len(tr)), 'config': name(c)}, time.time()
    scaler = StandardScaler().fit(X[tr])
    Xtr = scaler.transform(X[tr]).astype(np.float32)
    ytr = y[tr]
    rec['n_features'] = {'input': int(X.shape[1])}
    if c['dft'] < 1:
        score, hit = gl.dft_scores(Xtr, ytr, cfg.fs_B, threads, cache)
        sel = np.argsort(score)[:max(1, int(c['dft'] * score.size))]
        log(f'DFT: {len(sel)} of {X.shape[1]} features ({"cached" if hit else "computed"}, {time.time() - t:.0f} s)')
    else:
        sel = np.arange(X.shape[1])
    rec['n_features']['dft'] = int(len(sel))
    rec['seconds'] = {'scale_dft': round(time.time() - t, 1)}
    if c['axb']:
        t1 = time.time()
        keep, gain = gl.axb_select(Xtr[:, sel], ytr, cfg, threads, device=device)
        rec['n_features']['axb_positive_gain'] = int((gain > 0).sum())
        sel = sel[keep]
        rec['n_features']['axb'] = int(len(sel))
        rec['seconds']['axb'] = round(time.time() - t1, 1)
        log(f'AXB: {len(sel)} kept ({rec["n_features"]["axb_positive_gain"]} with positive gain, {time.time() - t1:.0f} s)')
    t1 = time.time()
    Xev = np.concatenate([scaler.transform(X[e]).astype(np.float32)[:, sel] if len(e) else np.zeros((0, len(sel)), np.float32) for e in evals]
                         + [np.zeros((0, len(sel)), np.float32)])
    Xs = Xtr[:, sel]
    if 'lnt' in c['proj']:
        ftr, fev, kernels = gl.quiet(BUNDLE.run_groupwise_lnt, Xs, ytr, Xev, cfg)       # [selected, LNT]
        rec['n_features']['lnt'] = int(ftr.shape[1] - len(sel))
    else:
        ftr, fev = Xs, Xev
    rec['seconds']['lnt'] = round(time.time() - t1, 1)
    for kind in (p for p in PROJ if p in c['proj'] and p != 'lnt'):
        t1 = time.time()
        a, b = stacked(kind, Xs, ytr, groups[tr], Xev, seed)
        ftr, fev = np.concatenate([ftr, a], 1), np.concatenate([fev, b], 1)
        rec['n_features'][kind] = int(a.shape[1])
        rec['seconds'][kind] = round(time.time() - t1, 1)
        log(f'{kind}: {a.shape[1]} stacked features ({time.time() - t1:.0f} s)')
    rec['n_features']['final'] = int(ftr.shape[1])
    log(f'classifier input {ftr.shape[1]}')
    t1 = time.time()
    clf = BUNDLE.build_xgb(cfg)
    clf.set_params(n_jobs=threads, device=device, max_depth=c['depth'], n_estimators=c['trees'], learning_rate=c['lr'])
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
