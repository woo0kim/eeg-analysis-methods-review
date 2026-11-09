"""Loaders over preprocess_stew.py's windows: the batches of CBraMod's released loader (shuffled training, x / 100, labels long)."""
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def level(v):
    return 0 if v <= 3 else 1 if v <= 6 else 2


def split_indices(z, folds, task, fold):
    """(train, val, test) window indices of fold k: test = fold k, validation = fold k + 1 mod 7, training = the rest."""
    test_s = folds[task]['test_subjects']
    te, va = set(test_s[fold]), set(test_s[(fold + 1) % 7])
    subj, keep = z['subject'], np.ones(len(z['subject']), bool) if task == 'rest-task' else z['rating'] > 0
    all_s = {s for t in test_s for s in t}
    tr = all_s - te - va
    return tuple(np.where(keep & np.isin(subj, sorted(g)))[0] for g in (tr, va, te))


def labels(z, task):
    return z['stage'].astype(np.int64) if task == 'rest-task' else np.array([level(v) if v else -1 for v in z['rating']], dtype=np.int64)


class Windows(Dataset):
    def __init__(self, x, y, idx, patch):
        self.x, self.y, self.idx, self.patch = x, y, idx, patch

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        c, t = self.x[j].shape
        return self.x[j].reshape(c, t // self.patch, self.patch), int(self.y[j])

    @staticmethod
    def collate(batch):
        x = np.stack([b[0] for b in batch]) / 100
        return torch.from_numpy(x).float(), torch.tensor([b[1] for b in batch], dtype=torch.long)


def get_loaders(root, folds_path, task, fold, foundation, batch_size=64):
    z = np.load(os.path.join(root, 'windows.npz'))
    folds = json.load(open(folds_path))
    x = z['x200'] if foundation else z['x128']
    y = labels(z, task)
    tr, va, te = split_indices(z, folds, task, fold)
    print(f'{task} fold {fold}: {len(tr)} train, {len(va)} val, {len(te)} test windows; x {x.shape}')
    p = 200 if foundation else 128
    eb = int(os.environ.get('EVAL_BATCH', batch_size))      # validation / test batch; predictions do not depend on it
    mk = lambda idx, sh, b: DataLoader(Windows(x, y, idx, p), batch_size=b, shuffle=sh, collate_fn=Windows.collate)
    return {'train': mk(tr, True, batch_size), 'val': mk(va, False, eb), 'test': mk(te, False, eb)}
