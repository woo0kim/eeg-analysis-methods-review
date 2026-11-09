"""SEED -> LMDBs in the format of CBraMod's SEED-V pipeline, so the released code runs on it unchanged
(finetune_main.py --downstream_dataset SEED-V --num_of_classes 3 --datasets_dir <this LMDB>).

CBraMod has no SEED code. The steps follow its preprocessing_SEEDV.py: 62 channels at 200 Hz, 0.3-75 Hz, 1-s
non-overlapping samples of shape (62, 1, 200) in uV, trials split 5:5:5 in order within each session. The input is
SEED's Preprocessed_EEG (already 200 Hz and low-passed at 75 Hz, stored clip by clip), so the filter runs per clip; SEED
has no continuous 200 Hz recording. Labels -1 / 0 / 1 -> 0 / 1 / 2.

A second LMDB holds the same samples split by subject (9 / 3 / 3), for the subject-independent protocol.

usage: python preprocess_seed.py [n_workers]
Paths: EEG_DATA_ROOT (default ~/data): reads SEED/SEED_EEG/Preprocessed_EEG, writes SEED/processed_trial555, processed_subject.
"""
import glob
import multiprocessing as mp
import os
import pickle
import re
import sys

import lmdb
import mne
import numpy as np
import scipy.io as sio

D = os.path.expanduser(os.environ.get('EEG_DATA_ROOT', '~/data'))
src = f'{D}/SEED/SEED_EEG/Preprocessed_EEG'
LABELS = (sio.loadmat(os.path.join(src, 'label.mat'))['label'].ravel() + 1).tolist()
TRIALS = {'train': range(5), 'val': range(5, 10), 'test': range(10, 15)}
SUBJECTS = {'train': range(1, 10), 'val': range(10, 13), 'test': range(13, 16)}


def load(f):
    m = sio.loadmat(f)
    keys = sorted((k for k in m if re.search(r'_eeg\d+$', k)), key=lambda k: int(k.rsplit('eeg', 1)[1]))
    assert len(keys) == 15, keys
    out = [mne.filter.filter_data(m[k].astype(np.float64), 200, 0.3, 75, verbose='ERROR') for k in keys]
    print(os.path.basename(f), [x.shape[1] // 200 for x in out], flush=True)
    return os.path.basename(f)[:-4], out


def write(path, samples, keys):
    db = lmdb.open(path, map_size=30 * 10 ** 9)
    txn = db.begin(write=True)
    for n, (k, v) in enumerate(samples):
        txn.put(key=k.encode(), value=v)
        if n % 2000 == 1999:
            txn.commit()
            txn = db.begin(write=True)
    txn.put(key='__keys__'.encode(), value=pickle.dumps(keys))
    txn.commit()
    db.close()


if __name__ == '__main__':
    files = [f for f in glob.glob(os.path.join(src, '*.mat')) if re.match(r'\d+_\d+\.mat', os.path.basename(f))]
    files.sort(key=lambda f: tuple(int(x) for x in re.match(r'(\d+)_(\d+)', os.path.basename(f)).groups()))
    assert len(files) == 45, len(files)
    with mp.Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as pool:
        loaded = pool.map(load, files, chunksize=1)
    trial_keys = {m: [] for m in TRIALS}
    subj_keys = {m: [] for m in TRIALS}

    def samples():
        for name, trials in loaded:
            sub = int(name.split('_')[0])
            sub_mode = next(m for m, r in SUBJECTS.items() if sub in r)
            for mode, idx in TRIALS.items():
                for t in idx:
                    x = trials[t]
                    n = x.shape[1] // 200
                    data = x[:, :n * 200].reshape(62, n, 1, 200).transpose(1, 0, 2, 3)
                    for i, s in enumerate(data):
                        key = f'{name}-{t}-{i}'
                        trial_keys[mode].append(key)
                        subj_keys[sub_mode].append(key)
                        yield key, pickle.dumps({'sample': s, 'label': LABELS[t]})

    write(f'{D}/SEED/processed_trial555', samples(), trial_keys)
    print({k: len(v) for k, v in trial_keys.items()})

    def again():
        db = lmdb.open(f'{D}/SEED/processed_trial555', readonly=True, lock=False)
        with db.begin() as txn:
            for keys in trial_keys.values():
                for k in keys:
                    yield k, bytes(txn.get(k.encode()))

    write(f'{D}/SEED/processed_subject', again(), subj_keys)
    print({k: len(v) for k, v in subj_keys.items()})
