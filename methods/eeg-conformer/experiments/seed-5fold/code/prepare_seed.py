"""SEED -> the per-session arrays conformer_seed_1s_5fold.py reads: the released preprocessing/seed.m and
preprocessing/seed_process_slice_cv.py in Python (no MATLAB on the server).

seed.m: per clip, z-score each channel (population SD), then zero-phase Chebyshev II band-pass 4-47 Hz
(cheby2(6, 60, [8 94] / 200), filtfilt). seed_process_slice_cv.py: cut each clip into 1-s windows, save
S<i>_session<j>.npy and S<i>_session<j>_label.npy.  Subject order: file names sorted numerically (seed.m reads a list
shipped as seed_file/namefile_list.mat, same order).

usage: python prepare_seed.py <Preprocessed_EEG dir> <out dir>
"""
import glob
import os
import re
import sys

import numpy as np
import scipy.io as sio
from scipy.signal import cheby2, filtfilt

src, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
fs = 200
b, a = cheby2(6, 60, [4 * 2 / fs, 47 * 2 / fs], btype='bandpass')       # Wn = [Wl*2 Wh*2]/fc, order 12 band-pass
label = sio.loadmat(os.path.join(src, 'label.mat'))['label'].ravel()
files = [f for f in glob.glob(os.path.join(src, '*.mat')) if re.match(r'\d+_\d+\.mat', os.path.basename(f))]
files.sort(key=lambda f: tuple(int(x) for x in re.match(r'(\d+)_(\d+)', os.path.basename(f)).groups()))
assert len(files) == 45
for n, f in enumerate(files):
    sub, ses = n // 3 + 1, n % 3 + 1
    m = sio.loadmat(f)
    keys = sorted((k for k in m if re.search(r'_eeg\d+$', k)), key=lambda k: int(k.rsplit('eeg', 1)[1]))
    data, lab = [], []
    for t, k in enumerate(keys):
        x = m[k]
        x = (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)
        x = filtfilt(b, a, x, axis=1)
        assert np.isfinite(x).all(), (f, k)
        nw = x.shape[1] // 200
        data.append(np.stack([x[:, i * 200:(i + 1) * 200] for i in range(nw)]).astype(np.float32))
        lab.append(np.full(nw, label[t]))
    arr, arl = np.empty(15, dtype=object), np.empty(15, dtype=object)
    for t in range(15):
        arr[t], arl[t] = data[t], lab[t]
    np.save(os.path.join(out, f'S{sub}_session{ses}.npy'), arr, allow_pickle=True)
    np.save(os.path.join(out, f'S{sub}_session{ses}_label.npy'), arl, allow_pickle=True)
    print(sub, ses, sum(len(d) for d in data), flush=True)
