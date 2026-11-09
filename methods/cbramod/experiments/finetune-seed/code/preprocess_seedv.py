"""SEED-V -> LMDB: CBraMod's preprocessing/preprocessing_SEEDV.py (trial times, labels, channels, resampling, filter,
1-s samples, 5:5:5 trial split) with paths and workers changed. Changed lines are marked [MOD].

The folder holds 49 files for the 48 sessions: subject 7's first session twice (7_1_20180411.cnt and ..._repaired.cnt).
The paper's 117,744 samples = 7,359 s per subject x 16 subjects, so one of them. The repaired file lacks channels FC4
and M1: the released drop_channels(['M1', ...]) fails on it and it cannot give 62 channels, so the original is the one
that goes through this pipeline. MNE's format detection fails on the original (the file is 34 MB longer than its
header says), hence the explicit int32, the format of every other file.

usage: python preprocess_seedv.py [n_workers]
Paths: EEG_DATA_ROOT (default ~/data): reads SEED-V/EEG_raw, writes SEED-V/processed_trial555 and SEED-V/processed_subject.
"""
import multiprocessing as mp   # [MOD]
import os
import pickle
import sys                     # [MOD]

import lmdb
import mne
import numpy as np

useless_ch = ['M1', 'M2', 'VEO', 'HEO']
trials_of_sessions = {
    '1': {'start': [30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204],
          'end': [102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401, 2607, 2901, 3172, 3359]},

    '2': {'start': [30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966, 2186, 2333, 2490, 2741],
          'end': [267, 488, 614, 773, 967, 1059, 1331, 1622, 1777, 1908, 2153, 2302, 2428, 2709, 2817]},

    '3': {'start': [30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888],
          'end': [321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066]},
}
labels_of_sessions = {
    '1': [4, 1, 3, 2, 0, 4, 1, 3, 2, 0, 4, 1, 3, 2, 0, ],
    '2': [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0, ],
    '3': [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0, ],
}

D = os.path.expanduser(os.environ.get('EEG_DATA_ROOT', '~/data'))             # [MOD]
root_dir = f'{D}/SEED-V/EEG_raw'                                              # [MOD]
SKIP = '7_1_20180411_repaired.cnt'                                            # [MOD] see the docstring
files = sorted(f for f in os.listdir(root_dir) if f.endswith('.cnt') and f != SKIP)   # [MOD]
assert len(files) == 48, len(files)                                           # [MOD]
print(files)

trials_split = {
    'train': range(5),
    'val': range(5, 10),
    'test': range(10, 15),
}
subjects_split = {'train': range(1, 11), 'val': range(11, 14), 'test': range(14, 17)}   # [MOD] 10 / 3 / 3 subjects


def load(file):                                                               # [MOD] one worker per file
    raw = mne.io.read_raw_cnt(os.path.join(root_dir, file), preload=True, verbose='ERROR', data_format='int32')   # [MOD]
    raw.drop_channels(useless_ch)
    # raw.set_eeg_reference(ref_channels='average')
    raw.resample(200, verbose='ERROR')
    raw.filter(l_freq=0.3, h_freq=75, verbose='ERROR')
    data_matrix = raw.get_data(units='uV')
    session_index = file.split('_')[1]
    data_trials = [
        data_matrix[:,
        trials_of_sessions[session_index]['start'][j] * 200:trials_of_sessions[session_index]['end'][j] * 200]
        for j in range(15)]
    print(file, data_matrix.shape, 'std %.1f' % np.median(data_matrix.std(1)), flush=True)
    return file, data_trials


def write(path, samples, keys):                                               # [MOD]
    db = lmdb.open(path, map_size=15614542346 * 2)
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
    trial_keys = {'train': [], 'val': [], 'test': []}                         # [MOD]
    subj_keys = {'train': [], 'val': [], 'test': []}                          # [MOD]
    with mp.Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as pool:       # [MOD]
        loaded = pool.map(load, files, chunksize=1)

    def samples():                                                            # [MOD]
        for file, data_trials in loaded:
            sub = int(file.split('_')[0])
            sub_mode = next(m for m, r in subjects_split.items() if sub in r)
            labels = labels_of_sessions[file.split('_')[1]]
            for mode in trials_split.keys():
                for index in trials_split[mode]:
                    data = data_trials[index]
                    label = labels[index]
                    data = data.reshape(62, -1, 1, 200)
                    data = data.transpose(1, 0, 2, 3)
                    for i, sample in enumerate(data):
                        sample_key = f'{file}-{index}-{i}'
                        trial_keys[mode].append(sample_key)
                        subj_keys[sub_mode].append(sample_key)
                        yield sample_key, pickle.dumps({'sample': sample, 'label': label})

    write(f'{D}/SEED-V/processed_trial555', samples(), trial_keys)
    print({k: len(v) for k, v in trial_keys.items()})

    def again():                                                              # [MOD] same samples, split by subject
        db = lmdb.open(f'{D}/SEED-V/processed_trial555', readonly=True, lock=False)
        with db.begin() as txn:
            for keys in trial_keys.values():
                for k in keys:
                    yield k, bytes(txn.get(k.encode()))

    write(f'{D}/SEED-V/processed_subject', again(), subj_keys)
    print({k: len(v) for k, v in subj_keys.items()})
