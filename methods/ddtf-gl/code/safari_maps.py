"""Safari et al.'s released dDTF maps of STEW, as dDTF-GL features for the paper's windows.

The dDTF-GL paper computes its maps "following the standard extraction strategy used in [4]", Safari et al. (Sci. Rep. 14:9153, 2024).
Their repository (github.com/safari-mohammadreza/MWL_MachineLearning, commit 604d243) does not include the code that made the maps
(its README: "We calculated ... by SIFT toolbox in Matlab"), but it releases the maps: files/{hi,lo}_<band>, one pickled numpy array
(48 * 37, 14, 14) per condition and band, subject-major (subjects 1-48, 37 windows of 6 s with a 4-s step each), built by its
notebook's read_mat() from SIFT's per-window output. Those are the windows of datasets.Stew6s, in the same order within a recording.

  python safari_maps.py <Safari's files/ dir> <out dir>     writes X.npy (3552, 980) and meta.npz/info.json in extract.py's layout
                                                            (recordings: subject 1 rest, subject 1 task, subject 2 rest, ...; features
                                                            band-major, then target channel, then source channel)

The pickles are read with a restricted unpickler that can only rebuild numpy arrays (a pickle can otherwise run arbitrary code).
Orientation: entry [i, j] of their matrices is compared with our flow j -> i; calibrate_safari.py checks it (the transposed maps agree
less).
"""
import json
import os
import pickle
import sys

import numpy as np

BANDS = ('delta', 'theta', 'alpha', 'beta', 'gamma')
_ALLOWED = {('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct'), ('numpy', 'ndarray'), ('numpy', 'dtype')}


class _NumpyOnly(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in _ALLOWED:
            raise pickle.UnpicklingError(f'refused to load {module}.{name}')
        if name == '_reconstruct':
            try:
                from numpy._core import multiarray      # numpy >= 2
            except ImportError:
                from numpy.core import multiarray
            return multiarray._reconstruct
        return getattr(np, name)


def load_array(path):
    with open(path, 'rb') as f:
        a = _NumpyOnly(f).load()
    assert isinstance(a, np.ndarray), type(a)
    return a


def maps(files):
    """(96 recordings, 37 windows, 5 bands, 14, 14): recording 2 (s - 1) = subject s at rest ('lo'), 2 (s - 1) + 1 = the task ('hi')."""
    out = np.zeros((96, 37, 5, 14, 14), np.float32)
    for b, name in enumerate(BANDS):
        for st, cond in ((0, 'lo'), (1, 'hi')):
            a = load_array(os.path.join(files, f'{cond}_{name}'))
            assert a.shape == (48 * 37, 14, 14), a.shape
            out[st::2, :, b] = a.reshape(48, 37, 14, 14)
    return out


if __name__ == '__main__':
    files, out = sys.argv[1:3]
    os.makedirs(out, exist_ok=True)
    X = maps(files).reshape(96 * 37, -1)
    subject, stage = np.repeat(np.arange(1, 49), 74), np.tile(np.repeat([0, 1], 37), 48)
    np.save(os.path.join(out, 'X.npy'), X)
    np.savez(os.path.join(out, 'meta.npz'), y=stage.astype(np.int64), subject=subject, stage=stage)
    json.dump(dict(dataset='STEW-6s', features='safari', n=len(X), n_features=X.shape[1], fs=128.0, samples=768,
                   source='Safari et al., github.com/safari-mohammadreza/MWL_MachineLearning @604d243, files/{hi,lo}_<band> (SIFT, MATLAB)',
                   bands=list(BANDS), channels=['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']),
              open(os.path.join(out, 'info.json'), 'w'), indent=1)
    print('wrote', out, X.shape)
