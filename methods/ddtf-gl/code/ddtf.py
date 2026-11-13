"""dDTF connectivity maps of multichannel EEG windows (the input representation of Cao, Li & Kuo).

The paper computes "dDTF in five standard EEG bands ... following the standard extraction strategy used in [4]" (Safari et al.,
Sci. Rep. 14:9153, 2024), which used SIFT: an MVAR model per window, order by AIC, and dDTF averaged over each band,
delta 2-4, theta 4-8, alpha 8-13, beta 13-32, gamma 32-50 Hz. Neither paper releases its connectivity code, so this is ours:

  1. each channel of a window is centred and scaled to unit variance over time (DTF is not invariant to channel scale);
  2. MVAR(p) by least squares: x(t) = sum_k A_k x(t-k) + e(t), residual covariance Sigma;
  3. A(f) = I - sum_k A_k exp(-i 2 pi f k / fs), H(f) = A(f)^-1, on a 1-Hz grid from 1 Hz to Nyquist;
  4. full-frequency DTF  F2_ij(f) = |H_ij(f)|^2 / sum_f' sum_m |H_im(f')|^2       (Korzeniewska et al. 2003, flow j -> i);
  5. partial coherence    C2_ij(f) = |G_ij(f)|^2 / (G_ii(f) G_jj(f)),  G = S^-1 = A(f)^H Sigma^-1 A(f) (inverse spectral matrix);
  6. dDTF_ij(f) = F2_ij(f) * C2_ij(f); the band value is the mean over the band's 1-Hz bins (edges included in both bands).

Safari et al. do not release their SIFT code but do release the maps it made for STEW. calibrate_safari.py compares ours with theirs
over 672 combinations of settings: these first settings agree weakly (off-diagonal pattern r 0.13); an average reference (Safari et al.'s
preprocessing) and sqrt(F2 * |C|) instead of F2 * C2 raise it to 0.37, the best found. band_ddtf(form='safari') uses those; the first
settings remain available as form='v0'.

A window gives 5 maps of C x C (diagonal included: Safari et al. list self-connections such as O1 => O1), flattened band by
band, row (target) by row: 5 * C * C features.
"""
import os

import numpy as np

BANDS = (('delta', 2, 4), ('theta', 4, 8), ('alpha', 8, 13), ('beta', 13, 32), ('gamma', 32, 50))


def prep(x):
    """(N, C, T) -> float64, every channel of every window centred and scaled to unit variance over time."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(-1, keepdims=True)
    sd = x.std(-1, keepdims=True)
    return x / np.where(sd > 0, sd, 1.0)


def fit_mvar(x, p, ridge=1e-8):
    """Least-squares MVAR(p) of each prepared window x (N, C, T).

    Returns A (N, C, p, C), A[n, i, k, j] = weight of x_j(t - k - 1) in x_i(t), and the residual covariance (N, C, C).
    `ridge` (relative to the mean diagonal of the normal matrix) only guards against numerically singular windows.
    """
    N, C, T = x.shape
    n = T - p
    Z = np.concatenate([x[:, :, p - k - 1:T - k - 1] for k in range(p)], axis=1)    # (N, C p, n): block k = lag k + 1
    Y = x[:, :, p:]                                                               # (N, C, n)
    R = Z @ Z.transpose(0, 2, 1)
    Q = Y @ Z.transpose(0, 2, 1)
    R = R + (ridge * np.trace(R, axis1=1, axis2=2) / R.shape[1])[:, None, None] * np.eye(R.shape[1])
    A = np.linalg.solve(R, Q.transpose(0, 2, 1)).transpose(0, 2, 1)               # Q R^-1 (R is symmetric)
    E = Y - A @ Z
    S = E @ E.transpose(0, 2, 1) / n
    return A.reshape(N, C, p, C), S


def aic(x, p):
    """AIC of MVAR(p) per prepared window: ln det Sigma + 2 p C^2 / (T - p)."""
    _, S = fit_mvar(x, p)
    C, n = x.shape[1], x.shape[2] - p
    return np.linalg.slogdet(S)[1] + 2.0 * p * C * C / n


def max_order(C, T, cap=20):
    """Largest order tried: at least 3 samples per regressor in each equation (T - p >= 3 C p), at most `cap`.

    Fitted by least squares on one short window, ln det Sigma falls without bound as C p approaches T - p, so the AIC's global
    minimum drifts to the largest order allowed (on STEW's 4-s windows: a local minimum at p = 8, then a collapse from p ~ 20 to the
    p = 34 limit). The cap keeps the search where the fit is determined."""
    return int(max(1, min(cap, T // (3 * C + 1))))


def select_order(x, pmax, chunk=50):
    """Order by AIC over p = 1..pmax on raw windows x (N, C, T): each window's first local minimum (the smallest p whose AIC is not
    above the next order's; pmax if the AIC falls throughout), then the median over windows. Returns (order, per-window, AIC table)."""
    table = []
    for s in range(0, len(x), chunk):
        xp = prep(x[s:s + chunk])
        table.append(np.stack([aic(xp, p) for p in range(1, pmax + 1)], 1))
    table = np.concatenate(table)
    if pmax == 1:                                                      # nothing to choose (SEED's 1-s windows of 62 channels)
        return 1, np.ones(len(table), int), table
    rise = np.diff(table, axis=1) >= 0                                 # AIC(p + 1) >= AIC(p)
    best = np.where(rise.any(1), rise.argmax(1) + 1, pmax)
    return int(np.median(best)), best, table


def freq_grid(fs):
    return np.arange(1, int(fs // 2) + 1, dtype=np.float64)


def band_ddtf(x, p, fs, bands=BANDS, ridge=1e-10, ref=None, form='safari'):
    """Raw windows (N, C, T) -> band-mean dDTF (N, n_bands, C, C) float32; entry [b, i, j] = flow j -> i in band b.

    form 'safari' (the default since 2026-11-12): the settings that best reproduce Safari et al.'s released SIFT maps of STEW
      (calibrate_safari.py): the channels in `ref` re-referenced to their average before the per-channel scaling, the residual covariance
      inverted with a pseudo-inverse (an average reference leaves it singular), and the value sqrt(F2_ij(f) * |C_ij(f)|), i.e. the
      full-frequency DTF times the magnitude (not the square) of the partial coherence, square-rooted.
    form 'v0': our first settings (no re-reference, F2_ij(f) * C2_ij(f)); kept as an ablation ("first settings").
    """
    xp = np.asarray(x, dtype=np.float64)
    xp = xp - xp.mean(-1, keepdims=True)
    if ref is not None and len(ref):
        ref = np.asarray(ref)
        xp[:, ref] -= xp[:, ref].mean(1, keepdims=True)
    sd = xp.std(-1, keepdims=True)
    xp = xp / np.where(sd > 0, sd, 1.0)
    A, S = fit_mvar(xp, p)
    N, C = A.shape[:2]
    f = freq_grid(fs)
    masks = [(f >= lo) & (f <= hi) for _, lo, hi in bands]
    assert all(m.any() for m in masks), f'a band lies above Nyquist ({fs / 2} Hz)'
    e = np.exp(-2j * np.pi * np.outer(f, np.arange(1, p + 1)) / fs)               # (F, p)
    Af = np.eye(C) - np.einsum('nikj,fk->nfij', A, e)                             # (N, F, C, C)
    H2 = np.abs(np.linalg.inv(Af)) ** 2
    ff = H2 / H2.sum(axis=(1, 3), keepdims=True)                                  # over all frequencies and sources, per target
    if ref is not None and len(ref):
        Si = np.linalg.pinv(S)
    else:
        Si = np.linalg.inv(S + (ridge * np.trace(S, axis1=1, axis2=2) / C)[:, None, None] * np.eye(C))
    G = Af.conj().transpose(0, 1, 3, 2) @ Si[:, None] @ Af                        # inverse spectral matrix S(f)^-1
    d = np.real(np.diagonal(G, axis1=2, axis2=3))                                 # (N, F, C), positive
    pc2 = np.abs(G) ** 2 / (d[..., :, None] * d[..., None, :])
    dd = np.sqrt(ff * np.sqrt(pc2)) if form == 'safari' else ff * pc2
    return np.stack([dd[:, m].mean(1) for m in masks], 1).astype(np.float32)


def chunk_size(C, fs, budget=6e8):
    """Windows per call so that the (N, F, C, C) complex arrays of band_ddtf stay near `budget` bytes."""
    return int(max(1, budget // (len(freq_grid(fs)) * C * C * 16 * 6)))


def feature_names(channels, bands=BANDS):
    return [f'{b}:{src}->{tgt}' for b, _, _ in bands for tgt in channels for src in channels]


if __name__ == '__main__':
    # Self-check on a simulated 3-channel MVAR(2) with flows 0 -> 1 -> 2 (no direct 0 -> 2): dDTF must show 0 -> 1 and 1 -> 2
    # and suppress the indirect 0 -> 2, which the full-frequency DTF alone still shows.
    rng = np.random.default_rng(0)
    T, fs = 4000, 128.0
    A1 = np.array([[0.5, 0, 0], [0.6, 0.3, 0], [0, 0.6, 0.3]])
    A2 = np.array([[-0.3, 0, 0], [0, -0.2, 0], [0, 0, -0.2]])
    x = np.zeros((3, T))
    for t in range(2, T):
        x[:, t] = A1 @ x[:, t - 1] + A2 @ x[:, t - 2] + rng.standard_normal(3)
    xw = x[None]
    p, best, _ = select_order(xw, 10)
    A, _ = fit_mvar(prep(xw), 2)
    m = band_ddtf(xw, 2, fs, bands=(('all', 1, 64),))[0, 0]
    os.environ.setdefault('PYTHONWARNINGS', 'ignore')
    print('AIC order', p, '(true 2)\nband-mean dDTF (row = target, column = source):\n', np.round(m, 4))
    assert m[1, 0] > 5 * m[2, 0] and m[2, 1] > 5 * m[2, 0], 'direct flows must dominate the indirect 0 -> 2'
    print('ok')
