"""Train / validation / test indices (into a feature cache's sample order, datasets.py) of every protocol run here.

Each protocol is the one every other model in the repository was scored on, so the samples, subject splits and test order match
theirs; dDTF-GL has no epochs to select, so the validation part is only scored, never used for fitting.

  STEW      cbramod  finetune-stew's pinned 7-fold GroupKFold (results/folds.json): test = fold k, validation = fold k + 1 (mod 7),
                     training = the other five folds. Tasks rest-task (2 classes, 48 subjects) and rating3 (ratings 1-3 / 4-6 / 7-9,
                     45 subjects)
  STEW, STEW-raw  si  the bundle's setting on 4-s windows: the pinned rest-task folds, training on the other six (no validation)
  STEW-6s   si       the paper's subject-independent setting: the same pinned rest-task folds, training = the other six (no validation)
            sd       the paper's subject-dependent setting, read as Safari et al.'s sevenfold cross-validation over the pooled windows:
                     StratifiedKFold(7, shuffle=True, random_state=seed); windows of one recording fall on both sides, and neighbouring
                     windows share 2 s of signal
            sd-blocked  control: every recording cut into 7 contiguous blocks of windows (6 or 5); test = block k of every recording,
                     training = the other blocks minus the two windows next to block k, so no training window overlaps a test window
  CBraMod's four datasets    cbramod-split: the LMDB's own train / val / test keys
  SEED, SEED-V   trial555 / subject: the keys of the respective LMDB, in its order (test order = every other model's)
  SleepEDF  split    split.json's 62 / 8 / 8 subjects, in the pipeline's order (subjects as listed, nights and sequence files sorted)
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDS = os.path.normpath(os.path.join(HERE, '../../cbramod/experiments/finetune-stew/results/folds.json'))


def level(v):
    return 0 if v <= 3 else 1 if v <= 6 else 2


def splits(dataset, meta, protocol, task=None, fold=None, seed=0):
    """-> dict(train=idx, val=idx or empty, test=idx, y=labels over the cache order, n_classes)."""
    y = meta['y'].astype(np.int64)
    empty = np.array([], dtype=np.int64)
    if dataset in ('STEW', 'STEW-raw') and protocol == 'si':      # the bundle's setting: test fold k, training the other six, no validation
        folds = json.load(open(FOLDS))['rest-task']['test_subjects']
        subj = meta['subject']
        return dict(train=np.where(~np.isin(subj, folds[fold]))[0], val=empty, test=np.where(np.isin(subj, folds[fold]))[0], y=y)
    if dataset == 'STEW':
        assert protocol == 'cbramod'
        folds = json.load(open(FOLDS))[task]['test_subjects']
        subj = meta['subject']
        if task == 'rating3':
            keep = meta['rating'] > 0
            y = np.array([level(v) if v else -1 for v in meta['rating']], dtype=np.int64)
        else:
            keep = np.ones(len(subj), bool)
        te, va = set(folds[fold]), set(folds[(fold + 1) % 7])
        tr = {s for f in folds for s in f} - te - va
        part = lambda g: np.where(keep & np.isin(subj, sorted(g)))[0]
        return dict(train=part(tr), val=part(va), test=part(te), y=y)
    if dataset == 'STEW-6s':
        subj = meta['subject']
        if protocol == 'si':
            folds = json.load(open(FOLDS))['rest-task']['test_subjects']
            te = np.where(np.isin(subj, folds[fold]))[0]
            return dict(train=np.where(~np.isin(subj, folds[fold]))[0], val=empty, test=te, y=y)
        if protocol == 'sd-blocked':                     # subject-dependent without shared signal: fold k = block k of every recording
            pos = np.arange(len(y)) % 37                   # 37 windows per recording, in time order (datasets.Stew6s)
            b = np.array_split(np.arange(37), 7)[fold]
            buffer = np.isin(pos, [b[0] - 1, b[-1] + 1])   # the neighbours overlapping the block by 2 s are left out of training
            return dict(train=np.where(~np.isin(pos, b) & ~buffer)[0], val=empty, test=np.where(np.isin(pos, b))[0], y=y)
        if protocol == 'sd':
            from sklearn.model_selection import StratifiedKFold
            tr, te = list(StratifiedKFold(7, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))[fold]
            return dict(train=tr, val=empty, test=te, y=y)
    if dataset in ('BCIC-IV-2a', 'PhysioNet-MI', 'Mumtaz2016', 'MentalArithmetic'):
        assert protocol == 'cbramod-split'
        n = [int(meta[f'n_{p}']) for p in ('train', 'val', 'test')]
        return dict(train=np.arange(n[0]), val=np.arange(n[0], n[0] + n[1]), test=np.arange(n[0] + n[1], sum(n)), y=y)
    if dataset in ('SEED', 'SEED-V'):
        pos = {k: i for i, k in enumerate(meta['keys'].tolist())}
        if '_split_keys' in meta:                        # split_keys.json next to the features (written by extract.py)
            keys = meta['_split_keys'][protocol]
        else:
            import datasets
            keys = datasets.lmdb_split_keys(dataset, protocol)
        return dict(**{p: np.array([pos[k] for k in keys[p]]) for p in ('train', 'val', 'test')}, y=y)
    if dataset == 'SleepEDF':
        import datasets
        split = datasets.sleep_split()
        subj = meta['subject']
        part = lambda ss: np.concatenate([np.where(subj == s)[0] for s in ss])
        return dict(train=part(split['train']), val=part(split['val']), test=part(split['test']), y=y)
    raise ValueError((dataset, protocol))


def order_pool(dataset, meta):
    """Samples the MVAR order is chosen from (by AIC, no labels): training samples of the first split, so no test sample is used."""
    if dataset == 'STEW':
        return splits('STEW', meta, 'cbramod', 'rest-task', 0)['train']
    if dataset == 'STEW-6s':
        return splits('STEW-6s', meta, 'si', fold=0)['train']
    if dataset == 'STEW-raw':
        return splits('STEW-raw', meta, 'si', fold=0)['train']
    if dataset in ('SEED', 'SEED-V'):
        a, b = splits(dataset, meta, 'trial555')['train'], splits(dataset, meta, 'subject')['train']
        return np.intersect1d(a, b)
    if dataset == 'SleepEDF':
        return splits(dataset, meta, 'split')['train']
    return splits(dataset, meta, 'cbramod-split')['train']
