"""Spectral features of raw windows, for the feature-tuning experiment (extract.py --features spec | spect).

Bands: DGCNN's (methods/dgcnn/code/dgcnn.py, SEED's): delta 1-3, theta 4-7, alpha 8-13, beta 14-30, gamma 31-50 Hz, each capped at
the Nyquist frequency. Power: Hann-windowed periodogram of the mean-removed signal, mean over the band's bins.
"""
import numpy as np

BANDS = (('delta', 1, 3), ('theta', 4, 7), ('alpha', 8, 13), ('beta', 14, 30), ('gamma', 31, 50))


def band_power(x, fs):
    """(N, C, T) -> (N, C, 5) mean periodogram power per band."""
    x = x - x.mean(-1, keepdims=True)
    w = np.hanning(x.shape[-1])
    p = np.abs(np.fft.rfft(x * w, axis=-1)) ** 2 / (fs * (w ** 2).sum())
    f = np.fft.rfftfreq(x.shape[-1], 1 / fs)
    out = []
    for _, lo, hi in BANDS:
        m = (f >= lo) & (f <= min(hi, fs / 2))
        out.append(p[..., m].mean(-1) if m.any() else np.zeros(p.shape[:-1]))
    return np.stack(out, -1)


def window_power(x, fs):
    """(N, C, T) -> (N, C, 10): log10 band power (5) and relative power (5) of the whole window."""
    bp = band_power(x.astype(np.float64), fs)
    rel = bp / np.maximum(bp.sum(-1, keepdims=True), 1e-20)
    return np.concatenate([np.log10(np.maximum(bp, 1e-20)), rel], -1).astype(np.float32)


def segment_power(x, fs):
    """(N, C, T) -> (N, C, S, 5): log10 band power of each 1-s segment."""
    n, c, t = x.shape
    s = int(t // fs)
    seg = x[..., :int(s * fs)].reshape(n, c, s, int(fs)).astype(np.float64)
    bp = band_power(seg.reshape(n, c * s, int(fs)), fs).reshape(n, c, s, len(BANDS))
    return np.log10(np.maximum(bp, 1e-20)).astype(np.float32)


if __name__ == '__main__':      # self-check: a 10-Hz sine puts its power in alpha
    t = np.arange(800) / 200
    x = np.sin(2 * np.pi * 10 * t)[None, None] + 0.01 * np.random.default_rng(0).standard_normal((1, 1, 800))
    w = window_power(x, 200.0)[0, 0]
    assert w[5 + 2] > 0.95, w
    print('ok', np.round(w, 3), segment_power(x, 200.0).shape)
