"""Enqueue DGCNN runs on CBraMod's four public datasets: python add_jobs.py <model> <seed> [<seed> ...]

Paths: EEG_DATA_ROOT (default ~/data), SEED_WORK (default ~/repro-seed), SEED_QUEUE (default $SEED_WORK/queue), SEED_PY,
NVML_FIX, CBRAMOD_SRC as in the SEED experiments. The LMDBs are the ones every other model on these splits used.
"""
import os
import shlex
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
E = lambda k, d: os.path.expanduser(os.environ.get(k, d))
D, WORK = E('EEG_DATA_ROOT', '~/data'), E('SEED_WORK', '~/repro-seed')
PY = E('SEED_PY', '~/miniforge3/envs/bd-labram/bin/python')
NVML = E('NVML_FIX', '~/repro-braindecode/nvml-580.173.02/lib')
Q = os.path.join(E('SEED_QUEUE', os.path.join(WORK, 'queue')), 'pending')
DATA = {
    'MentalArithmetic': f'{D}/mental_arithmetic/processed',
    'PhysioNet-MI': f'{D}/physio/processed_average',
    'Mumtaz2016': f'{D}/mumtaz/processed_lmdb_75hz',
    'BCIC-IV-2a': f'{D}/bciciv2a/processed_inde_avg_filter',
}
q = shlex.quote
model, seeds = sys.argv[1], [int(s) for s in sys.argv[2:]]
for ds, data in DATA.items():
    for seed in seeds:
        name = f'{ds}__{model}__s{seed}'
        run = os.path.join(WORK, 'runs/dgcnn-splits', name)
        cmd = (f'#!/bin/bash\n# job: {name}\nmkdir -p {q(run)}\nexport LD_LIBRARY_PATH={q(NVML)}\n'
               f'CUDA_VISIBLE_DEVICES=$1 {q(PY)} -u {q(os.path.join(HERE, "run_dgcnn.py"))} --downstream_dataset {ds} '
               f'--datasets_dir {q(data)} --model {model} --seed {seed} --model_dir {q(run + "/ckpt")} '
               f'> {q(run + "/log.txt")} 2>&1\n')
        open(os.path.join(Q, f'07_{name}.sh'), 'w').write(cmd)
        print('queued', name)
