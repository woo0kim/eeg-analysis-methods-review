"""The samples every other model in this repository was scored on, read from the shared pipelines' preprocessed data.

A Source holds n samples (C, T) at fs Hz, their labels and grouping metadata, in a fixed order, and reads any subset of them
(`read(idx)`); features are computed once per Source and cached (extract.py). Splits (protocols.py) index into that order.

  STEW          finetune-stew's 4-s windows (windows.npz, x128: 4,704 x 14 x 512 at 128 Hz, high-passed at 0.3 Hz)
  STEW-6s       the paper's windows: 6 s with a 4-s step (37 per recording, 3,552) from the raw files, high-passed at 1 Hz (Safari et al.)
  BCIC-IV-2a, PhysioNet-MI, Mumtaz2016, MentalArithmetic
                CBraMod's LMDBs (the ones every model on these splits read): (C, patches, 200) -> (C, patches * 200) at 200 Hz;
                key order = the LMDB's train, val, test key lists
  SEED, SEED-V  CBraMod-format LMDBs, 1-s samples (62, 200) at 200 Hz; the trial555 and subject LMDBs hold the same samples under
                the same keys, so one Source serves both splits (order: the trial555 LMDB's train, val, test keys)
  SleepEDF      finetune-sleepedf's 30-s epochs (processed_seq20: 20-epoch sequences (20, 2, 6000) at 200 Hz); order: recordings
                sorted, sequence files sorted, epochs in sequence order

Reference for the maps (Source.ref; the calibrated settings, ddtf.band_ddtf): the monopolar montages (STEW, SEED, SEED-V, Mumtaz2016, the 19
scalp channels of MentalArithmetic) are re-referenced to their average, as Safari et al. did; Sleep-EDF's two bipolar derivations are not.

BCIC-IV-2a and PhysioNet-MI are average-referenced in CBraMod's preprocessing (the channels sum to zero), so the C channels have
rank C - 1 and the MVAR residual covariance is singular. The last channel is dropped for the connectivity maps (it is minus the
sum of the others, so no information is lost); `drop_last` records it.
"""
import glob
import json
import os
import pickle
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = os.path.normpath(os.path.join(HERE, '../..'))
E = lambda k, d: os.path.expanduser(os.environ.get(k, d))
D = E('EEG_DATA_ROOT', '~/data')
STEW_WORK = E('STEW_WORK', '~/repro-seed/stew')                       # finetune-stew's windows.npz
STEW_RAW = E('STEW_RAW', f'{D}/STEW/STEW Dataset')                    # sub*_{lo,hi}.txt, ratings.txt
SLEEP = E('SLEEPEDF_PROCESSED', f'{D}/sleep_edfx/processed_seq20')

LMDB = {'BCIC-IV-2a': f'{D}/bciciv2a/processed_inde_avg_filter', 'PhysioNet-MI': f'{D}/physio/processed_average',
        'Mumtaz2016': f'{D}/mumtaz/processed_lmdb_75hz', 'MentalArithmetic': f'{D}/mental_arithmetic/processed',
        'SEED': f'{D}/SEED/processed_trial555', 'SEED-V': f'{D}/SEED-V/processed_trial555'}
CHANNELS = {   # LMDB channel order (methods/dgcnn/experiments/cbramod-splits/code/run_dgcnn.py)
    'PhysioNet-MI': ['FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1',
                     'CPZ', 'CP2', 'CP4', 'CP6', 'FP1', 'FPZ', 'FP2', 'AF7', 'AF3', 'AFZ', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'FZ',
                     'F2', 'F4', 'F6', 'F8', 'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'TP7', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2',
                     'P4', 'P6', 'P8', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2', 'IZ'],
    'BCIC-IV-2a': ['FZ', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPZ', 'CP2',
                   'CP4', 'P1', 'PZ', 'P2', 'POZ'],
    'Mumtaz2016': ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ'],
    'MentalArithmetic': ['FP1', 'FP2', 'F3', 'F4', 'F7', 'F8', 'T3', 'T4', 'C3', 'C4', 'T5', 'T6', 'P3', 'P4', 'O1', 'O2', 'FZ', 'CZ',
                         'PZ', 'A2'],
    'STEW': ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'],   # Emotiv EPOC order
    'SleepEDF': ['Fpz-Cz', 'Pz-Oz'],
}
CHANNELS['STEW-6s'] = CHANNELS['STEW-raw'] = CHANNELS['STEW']
DROP_LAST = {'BCIC-IV-2a', 'PhysioNet-MI'}


class Source:
    name, fs, channels, drop_last = None, None, None, False
    ref = None          # channels re-referenced to their average for the dDTF maps (monopolar montages), after drop_last

    def __len__(self):
        return len(self.y)

    def read(self, idx):
        raise NotImplementedError

    def window(self, idx):
        x = self.read(np.asarray(idx))
        return x[:, :-1] if self.drop_last else x

    def map_channels(self):
        return self.channels[:-1] if self.drop_last else self.channels


class StewWindows(Source):
    """finetune-stew's windows.npz: x128 (4704, 14, 512), subject, stage (0 rest, 1 test), rating (0 = missing)."""
    name, fs = 'STEW', 128.0

    def __init__(self):
        z = np.load(os.path.join(STEW_WORK, 'windows.npz'))
        self.x = z['x128']
        self.subject, self.stage, self.rating = z['subject'], z['stage'], z['rating'].astype(int)
        self.y = self.stage.astype(np.int64)
        self.channels = CHANNELS['STEW']
        self.ref = np.arange(14)

    def read(self, idx):
        return self.x[idx]


class Stew6s(Source):
    """The paper's segmentation: 6-s windows, 4-s step, from the raw recordings (19,200 samples each: 37 windows), after a 1-Hz
    high-pass (Safari et al.'s first step; their line-noise removal, ASR and average reference are not reproduced: no parameters
    are given, and an average reference makes the MVAR residual covariance singular)."""
    name, fs = 'STEW-6s', 128.0

    def __init__(self):
        import mne
        L, STEP = 768, 512
        X, sub, stage = [], [], []
        for s in range(1, 49):
            for st, nm in ((0, 'lo'), (1, 'hi')):
                x = np.loadtxt(os.path.join(STEW_RAW, f'sub{s:02d}_{nm}.txt')).T
                assert x.shape == (14, 19200), x.shape
                x = mne.filter.filter_data(x, self.fs, 1.0, None, verbose='ERROR')
                for i in range(0, x.shape[1] - L + 1, STEP):
                    X.append(x[:, i:i + L])
                    sub.append(s)
                    stage.append(st)
        self.x = np.array(X, dtype=np.float32)
        assert len(self.x) == 96 * 37, len(self.x)
        self.subject, self.stage = np.array(sub), np.array(stage)
        self.y = self.stage.astype(np.int64)
        self.channels = CHANNELS['STEW']
        self.ref = np.arange(14)

    def read(self, idx):
        return self.x[idx]


class StewRaw(Source):
    """The bundle's segmentation of the raw recordings: 4-s windows with a 384-sample step (49 per recording), no filtering at all, in the
    order of finetune-stew's windows.npz (subjects 1-48, rest then test), so the pinned folds and labels apply unchanged."""
    name, fs = 'STEW-raw', 128.0

    def __init__(self):
        X, sub, stage = [], [], []
        for s in range(1, 49):
            for st, nm in ((0, 'lo'), (1, 'hi')):
                x = np.loadtxt(os.path.join(STEW_RAW, f'sub{s:02d}_{nm}.txt'), dtype=np.float32).T   # as mwl_gl_si7.load_dataset_mwl
                for i in range(0, x.shape[1] - 512 + 1, 384):
                    X.append(x[:, i:i + 512])
                    sub.append(s)
                    stage.append(st)
        self.x = np.array(X, dtype=np.float32)
        assert len(self.x) == 96 * 49, len(self.x)
        self.subject, self.stage = np.array(sub), np.array(stage)
        self.y = self.stage.astype(np.int64)
        self.channels = CHANNELS['STEW']
        self.ref = np.arange(14)

    def read(self, idx):
        return self.x[idx]


class LmdbSource(Source):
    """A CBraMod-format LMDB: '__keys__' -> {'train', 'val', 'test'} key lists; each key -> {'sample': (C, P, 200), 'label'}."""
    fs = 200.0

    def __init__(self, name):
        import lmdb
        self.name, self.path = name, LMDB[name]
        self.drop_last = name in DROP_LAST
        env = lmdb.open(self.path, readonly=True, lock=False)
        with env.begin() as txn:
            self.split_keys = pickle.loads(txn.get(b'__keys__'))
            self.keys = [k for part in ('train', 'val', 'test') for k in self.split_keys[part]]
            self.y = np.array([int(pickle.loads(txn.get(k.encode()))['label']) for k in self.keys], dtype=np.int64)
        env.close()
        self.channels = CHANNELS.get(name) or [f'ch{i}' for i in range(62)]
        self._env, self._pid = None, None
        # average reference for the monopolar montages; BCIC-IV-2a and PhysioNet-MI are average-referenced already (drop_last);
        # MentalArithmetic's 20th channel is the ear derivation A2-A1, left out of the average
        self.ref = {'SEED': np.arange(62), 'SEED-V': np.arange(62), 'Mumtaz2016': np.arange(19), 'MentalArithmetic': np.arange(19)}.get(name)

    def close(self):
        """Close this process's environment; call before forking workers (py-lmdb refuses to reopen a path a forked child inherited)."""
        if self._env is not None:
            self._env.close()
        self._env, self._pid = None, None

    def read(self, idx):
        import lmdb
        if self._pid != os.getpid():                        # one environment per process (workers are forked)
            self._env, self._pid = lmdb.open(self.path, readonly=True, lock=False, readahead=False), os.getpid()
        with self._env.begin() as txn:
            out = [np.asarray(pickle.loads(txn.get(self.keys[i].encode()))['sample']) for i in idx]
        out = np.stack(out)
        return out.reshape(out.shape[0], out.shape[1], -1).astype(np.float64)


class SleepEdf(Source):
    """finetune-sleepedf's processed_seq20: seq/<rec>/<rec>-NNNN.npy (20, 2, 6000) float32, labels/<rec>/<rec>-NNNN.npy (20,)."""
    name, fs = 'SleepEDF', 200.0

    def __init__(self):
        self.files, rec, sub, y = [], [], [], []
        for r in sorted(os.listdir(os.path.join(SLEEP, 'seq'))):
            names = sorted(os.listdir(os.path.join(SLEEP, 'seq', r)))
            assert names == sorted(os.listdir(os.path.join(SLEEP, 'labels', r))), r
            for f in names:
                lab = np.load(os.path.join(SLEEP, 'labels', r, f))
                self.files.append(os.path.join(SLEEP, 'seq', r, f))
                y.append(lab)
                rec += [r] * len(lab)
                sub += [int(r[3:5])] * len(lab)
        self.y, self.rec, self.subject = np.concatenate(y).astype(np.int64), np.array(rec), np.array(sub)
        self.seq_len = 20
        assert len(self.y) == len(self.files) * self.seq_len
        self.channels = CHANNELS['SleepEDF']

    def read(self, idx):
        out, cache = [], {}
        for i in idx:
            f = i // self.seq_len
            if f not in cache:
                cache[f] = np.load(self.files[f])
            out.append(cache[f][i % self.seq_len])
        return np.stack(out).astype(np.float64)


def get(name):
    if name == 'STEW':
        return StewWindows()
    if name == 'STEW-6s':
        return Stew6s()
    if name == 'STEW-raw':
        return StewRaw()
    if name == 'SleepEDF':
        return SleepEdf()
    return LmdbSource(name)


def lmdb_split_keys(name, split):
    """{'train', 'val', 'test'} key lists of a SEED / SEED-V LMDB for split 'trial555' or 'subject'."""
    import lmdb
    path = LMDB[name].replace('processed_trial555', f'processed_{split}')
    env = lmdb.open(path, readonly=True, lock=False)
    with env.begin() as txn:
        keys = pickle.loads(txn.get(b'__keys__'))
    env.close()
    return keys


def sleep_split():
    return json.load(open(os.path.join(SLEEP, 'split.json')))
