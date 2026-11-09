"""braindecode's Labram on SEED / SEED-V: ../../braindecode-cbramod-splits/code/run_braindecode.py (same adapters,
weights and recipe) with the two datasets registered as in ../../seed-cbramod-pipeline/code/run_labram_seed.py.

usage: as run_braindecode.py, with --dataset SEED or SEED-V
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '../../braindecode-cbramod-splits/code'))
sys.path.insert(0, os.path.join(HERE, '../../seed-cbramod-pipeline/code'))
import run_labram_seed  # noqa: E402,F401  (registers SEED and SEED-V in run_labram)
import run_braindecode as rb  # noqa: E402

rb.N_TIMES.update({'SEED': 200, 'SEED-V': 200})

if __name__ == '__main__':
    rl = rb.run_labram
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument('--datasets_dir', required=True)
    rl.OURS, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    rl.RCF_ARGS, ds_init = rl.rcf.get_args()
    os.makedirs(rl.RCF_ARGS.output_dir, exist_ok=True)
    rl.rcf.main(rl.RCF_ARGS, ds_init)
