"""LaBraM-Base in the dDTF-GL paper's STEW settings: methods/labram/experiments/stew-cbramod-pipeline/code/run_labram_stew.py with
get_dataset() reading nets6.py's 6-s windows (200 Hz, 1,200 samples) and splits. [MOD] relative to that file: the data source and
split, and the validation set, which these settings do not have: LaBraM's script needs one, so it gets the first 64 training windows,
whose scores are never used. The final epoch's test probabilities (the last row of preds.npz) are scored, as for the other networks.

usage: python run_labram6.py --p6_protocol si|sd --p6_fold K --p6_seed S --p6_root R <LaBraM's flags> --dataset STEW
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = os.path.normpath(os.path.join(HERE, '../../../..'))
sys.path.insert(0, os.path.join(METHODS, 'labram/experiments/finetune-cbramod-splits/code'))
import run_labram as rl  # noqa: E402
sys.path.insert(0, HERE)
import nets6  # noqa: E402

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
    z = np.load(os.path.join(OURS.p6_root, 'windows6.npz'))
    tr, te, y = nets6.split(z, OURS.p6_protocol, OURS.p6_fold, OURS.p6_seed)            # [MOD]
    x = z['x200']
    args.nb_classes = 2
    print(f'STEW-6s {OURS.p6_protocol} fold {OURS.p6_fold}: train {len(tr)}, test {len(te)} (validation slot: 64 training windows, unused)')
    return Windows(x, y, tr), Windows(x, y, te), Windows(x, y, tr[:64]), CH, MC          # [MOD]


if __name__ == '__main__':
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for a_ in ('--p6_protocol', '--p6_root'):
        ap.add_argument(a_, required=True)
    ap.add_argument('--p6_fold', type=int, required=True)
    ap.add_argument('--p6_seed', type=int, required=True)
    OURS, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    rl.rcf.get_dataset = get_dataset
    rl.RCF_ARGS, ds_init = rl.rcf.get_args()
    os.makedirs(rl.RCF_ARGS.output_dir, exist_ok=True)
    rl.rcf.main(rl.RCF_ARGS, ds_init)
