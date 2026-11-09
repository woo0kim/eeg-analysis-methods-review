"""LaBraM-Base on STEW with its released run_class_finetuning.py and the recipe of ../../finetune-cbramod-splits, one fold per run.

As ../../seed-cbramod-pipeline/code/run_labram_seed.py, with get_dataset() reading the windows of
methods/cbramod/experiments/finetune-stew instead of an LMDB: 14 channels, the 200 Hz windows in uV (LaBraM's engine divides by 100).
Both tasks are multi-class in LaBraM's sense (nb_classes 2 or 3), as they are for every other model on STEW.
With LABRAM_BD=1 the model is braindecode's Labram (../../braindecode-cbramod-splits/code/run_braindecode.py), without --finetune.

usage: python run_labram_stew.py --stew_task T --stew_fold K --stew_root R --stew_folds F <LaBraM's flags> --dataset STEW
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BD = os.environ.get('LABRAM_BD') == '1'
sys.path.insert(0, os.environ.get('LABRAM_CODE', os.path.join(HERE, '../../finetune-cbramod-splits/code')))
if BD:
    sys.path.insert(0, os.path.join(HERE, '../../braindecode-cbramod-splits/code'))
    import run_braindecode as rb  # noqa: E402
    rl = rb.run_labram
    rb.N_TIMES['STEW'] = 800
else:
    import run_labram as rl  # noqa: E402
STEW_CODE = os.path.normpath(os.path.join(HERE, '../../../../cbramod/experiments/finetune-stew/code'))
CH = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
rl.CHANNELS['STEW'] = CH
MC = ['accuracy', 'balanced_accuracy', 'cohen_kappa', 'f1_weighted']


class Windows(torch.utils.data.Dataset):
    def __init__(self, x, y, idx):
        self.x, self.y, self.idx = x, y, idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return torch.FloatTensor(self.x[j]), int(self.y[j])


def get_dataset(args):
    sys.path.insert(0, STEW_CODE)
    import stew_data
    z = np.load(os.path.join(OURS.stew_root, 'windows.npz'))
    folds = json.load(open(OURS.stew_folds))
    x, y = z['x200'], stew_data.labels(z, OURS.stew_task)
    tr, va, te = stew_data.split_indices(z, folds, OURS.stew_task, OURS.stew_fold)
    args.nb_classes = {'rest-task': 2, 'rating3': 3}[OURS.stew_task]
    print(f'STEW {OURS.stew_task} fold {OURS.stew_fold}: train {len(tr)}, val {len(va)}, test {len(te)}')
    return Windows(x, y, tr), Windows(x, y, te), Windows(x, y, va), CH, MC


if __name__ == '__main__':
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for a_ in ('--stew_task', '--stew_root', '--stew_folds'):
        ap.add_argument(a_, required=True)
    ap.add_argument('--stew_fold', type=int, required=True)
    OURS, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    rl.rcf.get_dataset = get_dataset
    rl.RCF_ARGS, ds_init = rl.rcf.get_args()
    os.makedirs(rl.RCF_ARGS.output_dir, exist_ok=True)
    rl.rcf.main(rl.RCF_ARGS, ds_init)
