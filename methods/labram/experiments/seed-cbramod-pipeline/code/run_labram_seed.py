"""LaBraM-Base on SEED / SEED-V with its released run_class_finetuning.py, on CBraMod-format LMDBs.

Everything is ../../finetune-cbramod-splits/code/run_labram.py (the same two hooks and the same recipe); this file only
registers the two datasets: 62 channels (LaBraM's vocabulary holds all of them, CB1 and CB2 included), 3 or 5 classes.

usage: as run_labram.py, with --dataset SEED or SEED-V
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get('LABRAM_CODE', os.path.join(HERE, '../../finetune-cbramod-splits/code')))
import run_labram as rl  # noqa: E402

CH = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1',
      'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1',
      'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ',
      'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']
rl.CHANNELS.update({'SEED': CH, 'SEED-V': CH})
rl.N_CLASSES.update({'SEED': 3, 'SEED-V': 5})
MC = ['accuracy', 'balanced_accuracy', 'cohen_kappa', 'f1_weighted']
rl.METRICS.update({3: MC, 5: MC})

if __name__ == '__main__':
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument('--datasets_dir', required=True)
    rl.OURS, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    rl.RCF_ARGS, ds_init = rl.rcf.get_args()
    os.makedirs(rl.RCF_ARGS.output_dir, exist_ok=True)
    rl.rcf.main(rl.RCF_ARGS, ds_init)
