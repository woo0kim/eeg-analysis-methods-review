"""Write results/test_trials.json: for each dataset and split, the test samples' session and clip, run-length encoded.

The runs save only labels and predictions, in the order of the LMDB's test keys. This file lets collect.py group them
by clip without the LMDB (for the clip-level vote). usage: python export_keys.py <out.json>     (needs $EEG_DATA_ROOT)
"""
import json
import os
import pickle
import sys

import lmdb

D = os.path.expanduser(os.environ.get('EEG_DATA_ROOT', '~/data'))
out = {}
for ds in ('SEED', 'SEED-V'):
    for split in ('trial555', 'subject'):
        env = lmdb.open(f'{D}/{ds}/processed_{split}', readonly=True, lock=False)
        with env.begin() as txn:
            keys = pickle.loads(txn.get(b'__keys__'))['test']
        runs = []
        for k in keys:
            name, trial, _ = k.rsplit('-', 2)
            if runs and runs[-1][:2] == [name, int(trial)]:
                runs[-1][2] += 1
            else:
                runs.append([name, int(trial), 1])
        out[f'{ds}/{split}'] = runs
        print(ds, split, len(keys), 'samples in', len(runs), 'clips')
json.dump(out, open(sys.argv[1], 'w'))
