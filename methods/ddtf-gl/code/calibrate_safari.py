"""Which dDTF settings reproduce Safari et al.'s released maps? (their SIFT/MATLAB code is not released; the maps are)

Safari et al. (Sci. Rep. 14:9153, 2024; github.com/safari-mohammadreza/MWL_MachineLearning, commit 604d243) release the dDTF maps they
computed with SIFT for every 6-s window (4-s step) of STEW: files/{hi,lo}_<band>, (48 subjects x 37 windows, 14, 14). This script
computes our maps for the same windows (datasets.Stew6s: the raw recordings, 1-Hz high-pass) under every combination of the settings
SIFT leaves to the user or defines differently from our first choice, and scores the agreement with theirs on 300 windows:

  order         MVAR order p
  zscore        each channel scaled to unit variance per window (True) or only centred (False)
  ref           as recorded, or average reference (Safari et al.'s preprocessing; the residual covariance is then inverted with pinv)
  dtf           'ff' full-frequency DTF (normalised over all frequencies and sources) or 'pf' DTF normalised per frequency (Kaminski)
  pcoh          partial coherence squared (2) or its magnitude (1)
  grid          the frequency grid the full-frequency normalisation runs over: 1 Hz to Nyquist (64) or 1-50 Hz
  agg           band value: mean or sum over the band's 1-Hz bins
  root          the product as is, or its square root (Korzeniewska's chi without squaring)

Agreement: the median over windows of the Pearson correlation between the 182 off-diagonal entries of their map and ours (pattern),
the correlation of the 14 diagonal entries pooled over windows, and the ratio of the mean values (scale), per band and averaged.
usage: python calibrate_safari.py <dir holding Safari's files/> > ../experiments/stew-paper-protocol/results/calibration_safari.txt
"""
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import datasets  # noqa: E402
import ddtf  # noqa: E402

SAF = sys.argv[1]
from safari_maps import load_array as load  # noqa: E402  (a restricted unpickler: numpy arrays only)

BANDS = ddtf.BANDS
src = datasets.Stew6s()
rng = np.random.default_rng(0)
pick = np.sort(rng.choice(len(src), 300, replace=False))
x = src.window(pick).astype(np.float64)                                          # (300, 14, 768)
rec = (src.subject[pick] - 1) * 37 + (np.arange(len(src)) % 37)[pick]             # their index: subject-major, 37 windows
cond = np.where(src.stage[pick] == 1, 'hi', 'lo')
theirs = np.zeros((300, 5, 14, 14))
for b, (name, _, _) in enumerate(BANDS):
    arr = {c: load(os.path.join(SAF, 'files', f'{c}_{name}')) for c in ('hi', 'lo')}
    theirs[:, b] = np.stack([arr[c][r] for c, r in zip(cond, rec)])
off = ~np.eye(14, dtype=bool)


def spectra(x, p, zscore, ref):
    x = x - x.mean(-1, keepdims=True)
    if ref == 'avg':
        x = x - x.mean(1, keepdims=True)
    if zscore:
        x = x / x.std(-1, keepdims=True)
    A, S = ddtf.fit_mvar(x, p)
    f = np.arange(1, 65, dtype=float)
    e = np.exp(-2j * np.pi * np.outer(f, np.arange(1, p + 1)) / 128.0)
    Af = np.eye(14) - np.einsum('nikj,fk->nfij', A, e)
    H2 = np.abs(np.linalg.inv(Af)) ** 2
    Si = np.linalg.pinv(S) if ref == 'avg' else np.linalg.inv(S + 1e-10 * np.trace(S, axis1=1, axis2=2)[:, None, None] / 14 * np.eye(14))
    G = Af.conj().transpose(0, 1, 3, 2) @ Si[:, None] @ Af
    d = np.real(np.diagonal(G, axis1=2, axis2=3))
    pc2 = np.abs(G) ** 2 / (d[..., :, None] * d[..., None, :])
    return f, H2, pc2


def maps(f, H2, pc2, dtf, pcoh, grid, agg, root):
    g = f <= grid
    if dtf == 'ff':
        D = H2 / H2[:, g].sum(axis=(1, 3), keepdims=True)
    else:
        D = H2 / H2.sum(axis=3, keepdims=True)
    v = D * (pc2 if pcoh == 2 else np.sqrt(pc2))
    if root:
        v = np.sqrt(v)
    out = []
    for _, lo, hi in BANDS:
        m = (f >= lo) & (f <= hi)
        out.append(v[:, m].sum(1) if agg == 'sum' else v[:, m].mean(1))
    return np.stack(out, 1)


def score(ours):
    pat = np.array([[np.corrcoef(ours[i, b][off], theirs[i, b][off])[0, 1] for i in range(300)] for b in range(5)])
    dia = [np.corrcoef(np.diagonal(ours[:, b], axis1=1, axis2=2).ravel(), np.diagonal(theirs[:, b], axis1=1, axis2=2).ravel())[0, 1] for b in range(5)]
    scale = [ours[:, b].mean() / theirs[:, b].mean() for b in range(5)]
    return np.median(pat, 1), np.array(dia), np.array(scale)


rows = []
for p, zscore, ref in itertools.product((2, 4, 6, 8, 10, 12, 15), (True, False), ('none', 'avg')):
    f, H2, pc2 = spectra(x, p, zscore, ref)
    for dtf, pcoh, grid, agg, root in itertools.product(('ff', 'pf'), (2, 1), (64, 50), ('mean', 'sum'), (False, True)):
        if dtf == 'pf' and grid == 50:
            continue                                                     # the grid only enters the full-frequency normalisation
        pat, dia, scale = score(maps(f, H2, pc2, dtf, pcoh, grid, agg, root))
        rows.append(dict(order=p, zscore=zscore, ref=ref, dtf=dtf, pcoh=pcoh, grid=grid, agg=agg, root=root,
                         pattern=pat.mean(), diag=dia.mean(), scale=float(np.exp(np.mean(np.log(scale)))), pat_b=pat, dia_b=dia, scale_b=scale))
    print(f'order {p:2d} zscore {zscore!s:5s} ref {ref:4s}: best pattern r so far {max(r["pattern"] for r in rows):.3f}', file=sys.stderr, flush=True)

rows.sort(key=lambda r: -(r['pattern'] + r['diag']))
first = next(r for r in rows if (r['order'], r['zscore'], r['ref'], r['dtf'], r['pcoh'], r['grid'], r['agg'], r['root'])
             == (8, True, 'none', 'ff', 2, 64, 'mean', False))
print('Agreement with Safari et al.\'s released SIFT maps on 300 STEW windows (6 s, 4-s step). pattern = median over windows of the')
print('off-diagonal correlation; diag = correlation of the self-flows; scale = ours / theirs (geometric mean over bands).\n')
print(f'{"order":>5s} {"zscore":>6s} {"ref":>4s} {"dtf":>3s} {"pcoh":>4s} {"grid":>4s} {"agg":>4s} {"root":>5s} {"pattern":>8s} {"diag":>6s} {"scale":>7s}  pattern per band (d t a b g)')
fmt = lambda r: (f'{r["order"]:5d} {r["zscore"]!s:>6s} {r["ref"]:>4s} {r["dtf"]:>3s} {r["pcoh"]:4d} {r["grid"]:4d} {r["agg"]:>4s} {r["root"]!s:>5s} '
                 f'{r["pattern"]:8.3f} {r["diag"]:6.3f} {r["scale"]:7.3g}  ' + ' '.join(f'{v:.2f}' for v in r['pat_b']))
for r in rows[:25]:
    print(fmt(r))
print('\nour first choice (order 8, z-scored, no re-reference, full-frequency DTF, squared pCoh, 1-64 Hz, band mean):')
print(fmt(first))
print(f'\n{len(rows)} combinations tried')
