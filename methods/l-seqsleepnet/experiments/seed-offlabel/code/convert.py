"""npz of prepare_seed.py -> the released .mat format (X2, y, label; v7.3) and the file lists of both splits.

usage: python convert.py <SEED|SEED-V>        needs hdf5storage (tf115 env)
"""
import json
import os
import sys

import hdf5storage
import numpy as np

ds = sys.argv[1]
OUT = os.path.join(os.path.expanduser(os.environ.get('LSEQ_SEED', '~/repro-seed/lseq')), ds)
idx = json.load(open(os.path.join(OUT, 'index.json')))
slots = dict(zip(('eeg', 'eog', 'emg'), idx['channels']))
os.makedirs(os.path.join(OUT, 'mat'), exist_ok=True)
for tag, n in idx['files'].items():
    z = np.load(os.path.join(OUT, tag + '.npz'))
    lab = z['label'].astype(np.float32)
    y = np.eye(5, dtype=np.float32)[z['label'].astype(int) - 1]
    for s, c in slots.items():
        hdf5storage.savemat(os.path.join(OUT, 'mat', f'{tag}_{s}.mat'),
                            {'X2': z[c], 'label': lab[:, None], 'y': y}, format='7.3', matlab_compatible=True)
for split in ('trial555', 'subject'):
    for s in slots:
        L = {'train': [], 'eval': [], 'test': []}
        for tag, n in idx['files'].items():
            name, part = tag.rsplit('_', 1)
            sub = int(name.split('_')[0])
            line = f"{os.path.join(OUT, 'mat', f'{tag}_{s}.mat')}\t{n}"
            if split == 'trial555':
                L[{'train': 'train', 'val': 'eval', 'test': 'test'}[part]].append(line)
            else:
                L['test' if sub in idx['S_TEST'] else 'eval' if sub in idx['S_VAL'] else 'train'].append(line)
        for k, v in L.items():
            open(os.path.join(OUT, f'{split}_{k}_{s}.txt'), 'w').write('\n'.join(v) + '\n')
print(ds, len(idx['files']), 'files')
