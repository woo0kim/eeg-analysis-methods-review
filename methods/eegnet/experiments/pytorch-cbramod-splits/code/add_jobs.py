"""Enqueue EEGNet-8,2 runs through CBraMod's released fine-tuning pipeline (code/run_eegnet.py).

The data are CBraMod's preprocessed LMDBs, made by ../../../../cbramod/experiments/finetune-public-datasets
(scripts/make_preproc.py and the released preprocessing scripts).
usage: python add_jobs.py <Dataset> <seed> [<seed> ...]

Paths (defaults = the mcl-server layout the 20 logged runs used):
  EEG_DATA_ROOT  preprocessed datasets          (default ~/data)
  CBRAMOD_WORK   runs/ and queue/ are written here (default ~/repro)
  CBRAMOD_SRC    released CBraMod code          (default: this repo's methods/cbramod/upstream)
  CBRAMOD_PY     python of the cbramod env      (default ~/miniforge3/envs/cbramod/bin/python)
"""
import os, shlex, sys

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.expanduser(os.environ.get('EEG_DATA_ROOT', '~/data'))
WORK = os.path.expanduser(os.environ.get('CBRAMOD_WORK', '~/repro'))
SRC = os.environ.get('CBRAMOD_SRC', os.path.normpath(os.path.join(HERE, '../../../../cbramod/upstream')))
# the LMDBs CBraMod was fine-tuned on (../../../../cbramod/experiments/finetune-public-datasets/scripts/add_jobs.py)
DATA = {
    'MentalArithmetic': f'{D}/mental_arithmetic/processed',
    'PhysioNet-MI': f'{D}/physio/processed_average',
    'Mumtaz2016': f'{D}/mumtaz/processed_lmdb_75hz',
    'BCIC-IV-2a': f'{D}/bciciv2a/processed_inde_avg_filter',
}
PY = os.path.expanduser(os.environ.get('CBRAMOD_PY', '~/miniforge3/envs/cbramod/bin/python'))
Q = os.path.join(WORK, 'queue/pending')
os.makedirs(Q, exist_ok=True)

ds, seeds = sys.argv[1], [int(s) for s in sys.argv[2:]]
q = shlex.quote
for seed in seeds:
    name = f'{ds}__eegnet__s{seed}'
    run_dir = os.path.join(WORK, 'runs', name)
    cmd = (f'#!/bin/bash\n# job: {name}\nmkdir -p {q(run_dir)}\ncd {q(SRC)}\n'
           f'CUDA_VISIBLE_DEVICES=$1 {q(PY)} -u {q(os.path.join(HERE, "run_eegnet.py"))} --seed {seed} --cuda 0 '
           f'--downstream_dataset {ds} --datasets_dir {q(DATA[ds])} '
           f'--model_dir {q(run_dir + "/ckpt")} > {q(run_dir + "/log.txt")} 2>&1\n')
    open(os.path.join(Q, f'8_e_{name}.sh'), 'w').write(cmd)
    print('queued', name)
