"""L-SeqSleepNet's input files for SEED / SEED-V: off-label, the released code is a sleep-staging model.

The released network and scripts run unchanged. Everything the sleep setting fixes is adapted in the data here:
  - epochs: L-SeqSleepNet reads one 29 x 129 spectrogram per 30-s epoch, so each clip is cut into non-overlapping 30-s
    windows at 100 Hz (from the 200 Hz samples of the CBraMod-format LMDB, resample_poly 1:2); the rest of a clip is dropped.
    One 1-s sample of the other models has no spectrogram of that shape, so these runs cannot score the same samples:
    compare them at clip level only (collect.py).
  - channels: the released code takes at most three signals, in its EEG, EOG and EMG slots. T7, T8 and FT8, three of the
    temporal electrodes Zheng & Lu (2015) rank highest for SEED, take those slots.
  - labels: 1..K with a one-hot y of size 5, the released format (unused classes stay empty).
  - files: one per session and part (first / middle / last 5 clips), so a sequence never crosses parts. The trial split
    trains on the first parts; the subject split trains on all parts of its training subjects.
Same spectrogram as ../../sleepedf78-cbramod-split/code/prepare_split_data.py: hamming(200), 100 overlap, 256-point FFT, dB.

usage: python prepare_seed.py <SEED|SEED-V>       needs lmdb (reads the LMDB), writes npz; convert.py then writes the .mat
Paths: EEG_DATA_ROOT (default ~/data), LSEQ_SEED (default ~/repro-seed/lseq)
"""
import json
import os
import pickle
import sys

import lmdb
import numpy as np
from scipy import signal

ds = sys.argv[1]
D = os.path.expanduser(os.environ.get('EEG_DATA_ROOT', '~/data'))
OUT = os.path.join(os.path.expanduser(os.environ.get('LSEQ_SEED', '~/repro-seed/lseq')), ds)
os.makedirs(OUT, exist_ok=True)
CH = {'T7': 23, 'T8': 31, 'FT8': 22}                       # indices in SEED's 62-channel order (emo_models.CHANNELS)
WIN = signal.windows.hamming(200, sym=True)
PART = lambda t: 'train' if t < 5 else 'val' if t < 10 else 'test'
S_TEST = {'SEED': range(13, 16), 'SEED-V': range(14, 17)}
S_VAL = {'SEED': range(10, 13), 'SEED-V': range(11, 14)}


def stft(x1):
    frames = np.stack([x1[..., t * 100:t * 100 + 200] for t in range(29)], axis=-2)
    return (20 * np.log10(np.abs(np.fft.rfft(frames * WIN, 256, axis=-1)) + 1e-10)).astype(np.float32)


env = lmdb.open(f'{D}/{ds}/processed_trial555', readonly=True, lock=False)
trials = {}
with env.begin() as txn:
    keys = pickle.loads(txn.get(b'__keys__'))
    for part in ('train', 'val', 'test'):
        for k in keys[part]:
            name, t, i = k.rsplit('-', 2)
            pair = pickle.loads(txn.get(k.encode()))
            trials.setdefault((name, int(t)), {'label': int(pair['label']), 'x': {}})['x'][int(i)] = pair['sample'][list(CH.values()), 0]
index = {}
for name in sorted({n for n, _ in trials}):
    for part in ('train', 'val', 'test'):
        X2 = {c: [] for c in CH}
        lab, clip = [], []
        for t in sorted(t for n, t in trials if n == name and PART(t) == part):
            tr = trials[(name, t)]
            sig = np.concatenate([tr['x'][i] for i in sorted(tr['x'])], axis=1)             # (3, seconds * 200)
            sig = signal.resample_poly(sig, 1, 2, axis=1)                                   # 100 Hz
            n = sig.shape[1] // 3000
            for j, c in enumerate(CH):
                X2[c].append(stft(sig[j, :n * 3000].reshape(n, 3000)))
            lab += [tr['label'] + 1] * n
            clip += [t] * n
        tag = f'{name}_{part}'
        np.savez(os.path.join(OUT, tag + '.npz'), label=np.array(lab), clip=np.array(clip),
                 **{c: np.concatenate(v) for c, v in X2.items()})
        index[tag] = len(lab)
        print(tag, len(lab), flush=True)
json.dump({'files': index, 'channels': list(CH), 'S_VAL': list(S_VAL[ds]), 'S_TEST': list(S_TEST[ds])},
          open(os.path.join(OUT, 'index.json'), 'w'))
