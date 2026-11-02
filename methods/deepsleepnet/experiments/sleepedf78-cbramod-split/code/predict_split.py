"""DeepSleepNet's released predict.py for the 8 test subjects of the CBraMod-pipeline split (split.json).

predict.predict() runs unchanged with n_subjects = 8 and n_subjects_per_fold = 8, so every subject is scored with the
one model in <model_dir>/fold0 (train_split.py). Only SeqDataLoader.load_subject_data() changes ([MOD]): subject
index i loads the nights of split.json's i-th test subject instead of matching subject number i with an E0-only
file-name regex. The rest of the function is copied from deepsleep/data_loader.py at eb72f9b (Apache-2.0).
Writes output_subject<i>.npz (released format) and subjects.json (index -> subject number, night files).

usage: python predict_split.py --data_dir <prepare_split_data.py output (153 nights)> --model_dir OUT --output_dir OUT/predict
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


def test_files(data_dir, subject_idx):  # [MOD]
    s = SPLIT['test'][subject_idx]
    return sorted(os.path.join(data_dir, f) for f in os.listdir(data_dir)
                  if f.endswith('.npz') and int(f[3:5]) == s)


def load_subject_data(data_dir, subject_idx):
    subject_files = test_files(data_dir, subject_idx)  # [MOD] replaces the subject-number regex

    # Files for validation sets
    if len(subject_files) == 0 or len(subject_files) > 2:
        raise Exception("Invalid file pattern")

    def load_npz_file(npz_file):
        """Load data and labels from a npz file."""
        with np.load(npz_file) as f:
            data = f["x"]
            labels = f["y"]
            sampling_rate = f["fs"]
        return data, labels, sampling_rate

    def load_npz_list_files(npz_files):
        """Load data and labels from list of npz files."""
        data = []
        labels = []
        fs = None
        for npz_f in npz_files:
            print("Loading {} ...".format(npz_f))
            tmp_data, tmp_labels, sampling_rate = load_npz_file(npz_f)
            if fs is None:
                fs = sampling_rate
            elif fs != sampling_rate:
                raise Exception("Found mismatch in sampling rate.")

            # Reshape the data to match the input of the model - conv2d
            tmp_data = np.squeeze(tmp_data)
            tmp_data = tmp_data[:, :, np.newaxis, np.newaxis]

            # # Reshape the data to match the input of the model - conv1d
            # tmp_data = tmp_data[:, :, np.newaxis]

            # Casting
            tmp_data = tmp_data.astype(np.float32)
            tmp_labels = tmp_labels.astype(np.int32)

            data.append(tmp_data)
            labels.append(tmp_labels)

        return data, labels

    print("Load data from: {}".format(subject_files))
    data, labels = load_npz_list_files(subject_files)

    return data, labels


dl.SeqDataLoader.load_subject_data = staticmethod(load_subject_data)

if __name__ == '__main__':
    import tensorflow as tf
    import predict  # the released predict.py: its flags and predict()
    FLAGS = tf.app.flags.FLAGS

    def main(argv=None):
        os.makedirs(FLAGS.output_dir, exist_ok=True)
        json.dump({i: {'subject': s, 'files': [os.path.basename(f) for f in test_files(FLAGS.data_dir, i)]}
                   for i, s in enumerate(SPLIT['test'])}, open(os.path.join(FLAGS.output_dir, 'subjects.json'), 'w'))
        n = len(SPLIT['test'])
        predict.predict(data_dir=FLAGS.data_dir, model_dir=FLAGS.model_dir, output_dir=FLAGS.output_dir,
                        n_subjects=n, n_subjects_per_fold=n)

    tf.compat.v1.app.run(main=main)
