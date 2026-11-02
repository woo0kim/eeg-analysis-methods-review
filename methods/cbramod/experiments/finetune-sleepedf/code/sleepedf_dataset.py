# Copied from ../../../upstream/datasets/isruc_dataset.py (MIT, CBraMod authors); changed lines marked # [MOD].
import json  # [MOD] the subject split is read from split.json (written by preprocess_sleepedf.py)
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def to_tensor(array):  # [MOD] = utils.util.to_tensor, inlined so the module has no import-path side effects
    return torch.from_numpy(array).float()


class CustomDataset(Dataset):
    def __init__(
            self,
            seqs_labels_path_pair
    ):
        super(CustomDataset, self).__init__()
        self.seqs_labels_path_pair = seqs_labels_path_pair

    def __len__(self):
        return len((self.seqs_labels_path_pair))

    def __getitem__(self, idx):
        seq_path = self.seqs_labels_path_pair[idx][0]
        label_path = self.seqs_labels_path_pair[idx][1]
        seq = np.load(seq_path)
        label = np.load(label_path)
        return seq/100, label

    def collate(self, batch):
        x_seq = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return to_tensor(x_seq), to_tensor(y_label).long()


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.seqs_dir = os.path.join(params.datasets_dir, 'seq')
        self.labels_dir = os.path.join(params.datasets_dir, 'labels')
        self.split = json.load(open(os.path.join(params.datasets_dir, 'split.json')))  # [MOD]
        self.seqs_labels_path_pair = self.load_path()

    def get_data_loader(self):
        train_pairs, val_pairs, test_pairs = self.split_dataset(self.seqs_labels_path_pair)
        train_set = CustomDataset(train_pairs)
        val_set = CustomDataset(val_pairs)
        test_set = CustomDataset(test_pairs)
        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set) + len(val_set) + len(test_set))
        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size // self.params.accum,  # [MOD] micro-batches, see seq_trainer.py
                collate_fn=train_set.collate,
                shuffle=True,
                num_workers=self.params.num_workers,  # [MOD] released: 0; only speeds up file reads
            ),
            'val': DataLoader(
                val_set,
                batch_size=1,
                collate_fn=val_set.collate,
                shuffle=False,
            ),
            'test': DataLoader(
                test_set,
                batch_size=1,
                collate_fn=test_set.collate,
                shuffle=False,
            ),
        }
        return data_loader

    def load_path(self):
        # [MOD] one entry per subject (both nights), in the order of split.json; file lists sorted, where the
        # released code zips two unsorted os.listdir() results
        seqs_labels_path_pair = {}
        for rec in sorted(os.listdir(self.seqs_dir)):
            subject = int(rec[3:5])
            seq_fnames = sorted(os.listdir(os.path.join(self.seqs_dir, rec)))
            label_fnames = sorted(os.listdir(os.path.join(self.labels_dir, rec)))
            assert seq_fnames == label_fnames, rec
            seqs_labels_path_pair.setdefault(subject, []).extend(
                (os.path.join(self.seqs_dir, rec, f), os.path.join(self.labels_dir, rec, f)) for f in seq_fnames)
        return seqs_labels_path_pair

    def split_dataset(self, seqs_labels_path_pair):
        # [MOD] subject lists from split.json instead of index ranges 0-79 / 80-89 / 90-99
        train_pairs = []
        val_pairs = []
        test_pairs = []
        for s in self.split['train']:
            train_pairs.extend(seqs_labels_path_pair[s])
        for s in self.split['val']:
            val_pairs.extend(seqs_labels_path_pair[s])
        for s in self.split['test']:
            test_pairs.extend(seqs_labels_path_pair[s])
        return train_pairs, val_pairs, test_pairs
