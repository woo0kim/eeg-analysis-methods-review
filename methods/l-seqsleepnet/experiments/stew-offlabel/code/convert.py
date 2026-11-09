"""npz of prepare_stew.py -> the released .mat format and the file lists of every task and fold.  usage: python convert.py <dir> <folds.json>"""
import json
import os
import sys

import hdf5storage
import numpy as np

d, folds_path = sys.argv[1:3]
idx, folds = json.load(open(os.path.join(d, 'index.json'))), json.load(open(folds_path))
level = lambda v: 0 if v <= 3 else 1 if v <= 6 else 2
os.makedirs(os.path.join(d, 'mat'), exist_ok=True)
slots = dict(zip(('eeg', 'eog', 'emg'), idx['channels']))
for task in ('rest-task', 'rating3'):
    for tag, m in idx['files'].items():
        if task == 'rating3' and not m['rating']:
            continue
        lab = m['stage'] if task == 'rest-task' else level(m['rating'])
        z = np.load(os.path.join(d, tag + '.npz'))
        for s, c in slots.items():
            hdf5storage.savemat(os.path.join(d, 'mat', f'{task}_{tag}_{s}.mat'), {'X2': z[c], 'label': np.full((5, 1), lab + 1, np.float32),
                                'y': np.tile(np.eye(5, dtype=np.float32)[lab], (5, 1))}, format='7.3', matlab_compatible=True)
    test = folds[task]['test_subjects']
    for k in range(7):
        part = {'test': set(test[k]), 'eval': set(test[(k + 1) % 7])}
        for s in slots:
            L = {'train': [], 'eval': [], 'test': []}
            for tag, m in idx['files'].items():
                if task == 'rating3' and not m['rating']:
                    continue
                p = 'test' if m['subject'] in part['test'] else 'eval' if m['subject'] in part['eval'] else 'train'
                L[p].append(f"{os.path.join(d, 'mat', f'{task}_{tag}_{s}.mat')}\t5")
            for p, v in L.items():
                open(os.path.join(d, f'{task}_fold{k}_{p}_{s}.txt'), 'w').write('\n'.join(v) + '\n')
print('done')
