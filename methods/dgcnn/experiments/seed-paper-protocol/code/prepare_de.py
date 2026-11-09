"""Write the input files of the authors' eeg_DE.py from SEED's distributed DE features.

eeg_DE.py reads DE_<i>.mat ('DE_feature': 62 x 3394 x 5) for i = 1..45 and DE_labels.mat ('de_labels': 3394). The
archive does not include them, nor the code that made them. Its literals fix their shape: 45 sessions, in the order
subject 1 session 1..3, subject 2 session 1..3, ... (accuracy[i*3:3*(i+1)] is one subject), 2,010 training samples
(= the first 9 trials of SEED's 1-s features) and 1,384 test samples (the last 6). Labels are 0/1/2 (C = 3).
Which of SEED's DE smoothings the authors used is not recorded, so both are written.

usage: python prepare_de.py <ExtractedFeatures_1s dir> <out dir> <de_LDS|de_movingAve>
"""
import glob
import os
import re
import sys

import numpy as np
import scipy.io as sio

src, out, key = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
files = [f for f in glob.glob(os.path.join(src, '*.mat')) if re.match(r'\d+_\d+\.mat', os.path.basename(f))]
files.sort(key=lambda f: tuple(int(x) for x in re.match(r'(\d+)_(\d+)', os.path.basename(f)).groups()))
assert len(files) == 45, len(files)
label = sio.loadmat(os.path.join(src, 'label.mat'))['label'].ravel() + 1          # -1 0 1 -> 0 1 2
labels = None
for i, f in enumerate(files):
    m = sio.loadmat(f)
    de = np.concatenate([m[f'{key}{t + 1}'] for t in range(15)], axis=1)           # 62 x time x 5
    y = np.concatenate([np.full(m[f'{key}{t + 1}'].shape[1], label[t]) for t in range(15)])
    assert de.shape == (62, 3394, 5), de.shape
    assert labels is None or (labels == y).all()
    labels = y
    sio.savemat(os.path.join(out, f'DE_{i + 1}.mat'), {'DE_feature': de.astype(np.float32)})
    print(i + 1, os.path.basename(f))
sio.savemat(os.path.join(out, 'DE_labels.mat'), {'de_labels': labels.reshape(-1, 1)})
print('train classes', np.bincount(labels[:2010]), 'test classes', np.bincount(labels[2010:]))
