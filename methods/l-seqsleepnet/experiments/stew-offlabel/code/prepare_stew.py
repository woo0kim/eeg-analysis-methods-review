"""L-SeqSleepNet's input files for STEW: off-label, the released code is a sleep-staging model (see ../../seed-offlabel).

Each 150-s recording (19,200 samples at 128 Hz) is high-pass filtered at 0.3 Hz as for the other models, resampled to 100 Hz (the
spectrogram L-SeqSleepNet expects is 29 x 129 for 30 s at 100 Hz) and cut into its five 30-s windows: one 5-epoch sequence per recording.
Three channels go into the code's three signal slots: T7, T8 and FC6, the temporal and fronto-central electrodes closest to those used for
SEED (Emotiv has no FT8). Labels 1..K with a one-hot y of size 5, the released format. File lists follow the pinned GroupKFold folds of
../../../../cbramod/experiments/finetune-stew/results/folds.json, validation = the next fold, as for every other model.

usage: python prepare_stew.py <dir with sub*_*.txt> <out dir>        then convert.py (hdf5storage) writes the .mat files and lists
"""
import json
import os
import sys

import mne
import numpy as np
from scipy import signal

src, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
CH = {'T7': 4, 'T8': 9, 'FC6': 10}        # AF3 F7 F3 FC5 T7 P7 O1 O2 P8 T8 FC6 F4 F8 AF4
WIN = signal.windows.hamming(200, sym=True)
ratings = {int(r[0]): (int(r[1]), int(r[2])) for r in np.loadtxt(os.path.join(src, 'ratings.txt'), delimiter=',')}


def stft(x1):
    frames = np.stack([x1[..., t * 100:t * 100 + 200] for t in range(29)], axis=-2)
    return (20 * np.log10(np.abs(np.fft.rfft(frames * WIN, 256, axis=-1)) + 1e-10)).astype(np.float32)


index = {}
for s in range(1, 49):
    for st, name in ((0, 'lo'), (1, 'hi')):
        x = np.loadtxt(os.path.join(src, f'sub{s:02d}_{name}.txt')).T[list(CH.values())]
        x = mne.filter.filter_data(x, 128.0, 0.3, None, verbose='ERROR')
        x = signal.resample_poly(x, 25, 32, axis=1)                                   # 128 -> 100 Hz
        assert x.shape[1] == 15000, x.shape
        tag = f'sub{s:02d}_{name}'
        np.savez(os.path.join(out, tag + '.npz'), **{c: stft(x[j].reshape(5, 3000)) for j, c in enumerate(CH)})
        index[tag] = {'subject': s, 'stage': st, 'rating': ratings.get(s, (0, 0))[st]}
json.dump({'files': index, 'channels': list(CH)}, open(os.path.join(out, 'index.json'), 'w'))
print(len(index), 'recordings')
