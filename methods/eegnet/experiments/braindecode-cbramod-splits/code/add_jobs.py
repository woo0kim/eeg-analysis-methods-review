"""Queue braindecode-EEGNet runs through CBraMod's pipeline (code/run_eegnet.py) for tools/run_queue.py.

usage: python code/add_jobs.py <Dataset> <seed> [<seed> ...]

Paths (defaults = the mcl-server layout of the committed runs):
  EEG_DATA_ROOT   CBraMod's preprocessed LMDBs                  (default ~/data)
  BD_EEGNET_WORK  runs/ are written here                        (default ~/repro-braindecode/eegnet)
  BD_QUEUE        queue directory shared by the braindecode runs (default ~/repro-braindecode/queue)
  BD_CBRAMOD_PY   python of the bd-cbramod env                  (default ~/miniforge3/envs/bd-cbramod/bin/python)
"""
import os, shlex, sys

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.expanduser(os.environ.get('EEG_DATA_ROOT', '~/data'))
WORK = os.path.expanduser(os.environ.get('BD_EEGNET_WORK', '~/repro-braindecode/eegnet'))
QUEUE = os.path.expanduser(os.environ.get('BD_QUEUE', '~/repro-braindecode/queue'))
PY = os.path.expanduser(os.environ.get('BD_CBRAMOD_PY', '~/miniforge3/envs/bd-cbramod/bin/python'))
# the LMDBs of CBraMod's released-code runs (../../../cbramod/experiments/finetune-public-datasets/scripts/add_jobs.py)
DATA = {
    'PhysioNet-MI': f'{D}/physio/processed_average',
    'BCIC-IV-2a': f'{D}/bciciv2a/processed_inde_avg_filter',
    'Mumtaz2016': f'{D}/mumtaz/processed_lmdb_75hz',
    'MentalArithmetic': f'{D}/mental_arithmetic/processed',
}
Q = os.path.join(QUEUE, 'pending')
os.makedirs(Q, exist_ok=True)

ds, seeds = sys.argv[1], [int(s) for s in sys.argv[2:]]
q = shlex.quote
for seed in seeds:
    name = f'{ds}__eegnet__s{seed}'
    run_dir = os.path.join(WORK, 'runs', name)
    cmd = (f'#!/bin/bash\n# job: eegnet {name}\nmkdir -p {q(run_dir)}\n'
           f'export CUDA_VISIBLE_DEVICES=$1\n'
           f'{q(PY)} -u {q(os.path.join(HERE, "run_eegnet.py"))} --downstream_dataset {ds} '
           f'--datasets_dir {q(DATA[ds])} --seed {seed} --cuda 0 '
           f'--model_dir {q(run_dir + "/ckpt")} > {q(run_dir + "/log.txt")} 2>&1\n')
    open(os.path.join(Q, f'4_e_{name}.sh'), 'w').write(cmd)
    print('queued', name)
