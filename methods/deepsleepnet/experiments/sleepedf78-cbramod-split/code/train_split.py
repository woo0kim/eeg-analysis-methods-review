"""DeepSleepNet's released train.py on Sleep-EDF-78 with the subject split of the CBraMod-pipeline experiments.

train.py and the released trainers run unchanged (100 pre-training + 200 fine-tuning epochs, oversampled pre-training
set, final model kept). Only the file selection of the two data loaders' load_train_data() changes, marked [MOD]:
the released code holds out the subject whose number matches --fold_idx (a file-name regex that only knows the E0
and J0 night codes); here the training files are the nights of the 62 training subjects of split.json and the
"validation" files those of its 8 validation subjects. As in the released code, the validation set is only scored
and printed every epoch, never used to choose a model. The rest of both functions is copied from
deepsleep/data_loader.py at eb72f9b (Apache-2.0).

usage: python train_split.py --data_dir <prepare_split_data.py output (153 nights)> --output_dir OUT
                             --fold_idx 0 --n_folds 20 --pretrain_epochs 100 --finetune_epochs 200 --resume False
       SPLIT_JSON = split.json of ../../../../cbramod/experiments/finetune-sleepedf/results (default)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.expanduser(os.environ.get('DSN_SRC', '~/repro-sleepmodels/deepsleepnet-upstream'))
SPLIT = json.load(open(os.environ.get(
    'SPLIT_JSON', os.path.join(HERE, '../../../../cbramod/experiments/finetune-sleepedf/results/split.json'))))
sys.path.insert(0, SRC)
os.chdir(SRC)
import deepsleep.data_loader as dl  # noqa: E402
from deepsleep.sleep_stage import print_n_samples_each_class  # noqa: E402
from deepsleep.utils import get_balance_class_oversample  # noqa: E402


def split_files(data_dir):  # [MOD] nights SC4<ss><n>??.npz of the split's subjects
    npz = sorted(os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz'))
    subj = lambda f: int(os.path.basename(f)[3:5])
    return ([f for f in npz if subj(f) in SPLIT['train']], [f for f in npz if subj(f) in SPLIT['val']])


def nonseq_load_train_data(self, n_files=None):
    train_files, subject_files = split_files(self.data_dir)  # [MOD] replaces the fold_idx regex selection

    # Load training and validation sets
    print("\n========== [Fold-{}] ==========\n".format(self.fold_idx))
    print("Load training set:")
    data_train, label_train = self._load_npz_list_files(npz_files=train_files)
    print(" ")
    print("Load validation set:")
    data_val, label_val = self._load_npz_list_files(npz_files=subject_files)
    print(" ")

    # Reshape the data to match the input of the model - conv2d
    data_train = np.squeeze(data_train)
    data_val = np.squeeze(data_val)
    data_train = data_train[:, :, np.newaxis, np.newaxis]
    data_val = data_val[:, :, np.newaxis, np.newaxis]

    # Casting
    data_train = data_train.astype(np.float32)
    label_train = label_train.astype(np.int32)
    data_val = data_val.astype(np.float32)
    label_val = label_val.astype(np.int32)

    print("Training set: {}, {}".format(data_train.shape, label_train.shape))
    print_n_samples_each_class(label_train)
    print(" ")
    print("Validation set: {}, {}".format(data_val.shape, label_val.shape))
    print_n_samples_each_class(label_val)
    print(" ")

    # Use balanced-class, oversample training set
    x_train, y_train = get_balance_class_oversample(
        x=data_train, y=label_train
    )
    print("Oversampled training set: {}, {}".format(
        x_train.shape, y_train.shape
    ))
    print_n_samples_each_class(y_train)
    print(" ")

    return x_train, y_train, data_val, label_val


def seq_load_train_data(self, n_files=None):
    train_files, subject_files = split_files(self.data_dir)  # [MOD] replaces the fold_idx regex selection

    # Load training and validation sets
    print("\n========== [Fold-{}] ==========\n".format(self.fold_idx))
    print("Load training set:")
    data_train, label_train = self._load_npz_list_files(train_files)
    print(" ")
    print("Load validation set:")
    data_val, label_val = self._load_npz_list_files(subject_files)
    print(" ")

    print("Training set: n_subjects={}".format(len(data_train)))
    n_train_examples = 0
    for d in data_train:
        print(d.shape)
        n_train_examples += d.shape[0]
    print("Number of examples = {}".format(n_train_examples))
    print_n_samples_each_class(np.hstack(label_train))
    print(" ")
    print("Validation set: n_subjects={}".format(len(data_val)))
    n_valid_examples = 0
    for d in data_val:
        print(d.shape)
        n_valid_examples += d.shape[0]
    print("Number of examples = {}".format(n_valid_examples))
    print_n_samples_each_class(np.hstack(label_val))
    print(" ")

    return data_train, label_train, data_val, label_val


dl.NonSeqDataLoader.load_train_data = nonseq_load_train_data
dl.SeqDataLoader.load_train_data = seq_load_train_data

if __name__ == '__main__':
    import tensorflow as tf
    import train  # the released train.py: its flags, pretrain(), finetune() and main()
    tf.compat.v1.app.run(main=train.main)
