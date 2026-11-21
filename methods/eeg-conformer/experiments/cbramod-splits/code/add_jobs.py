"""Enqueue runs of run_models.py on CBraMod's four public datasets for tools/run_queue.py:
python add_jobs.py <model> <seed> [<seed> ...]      models: conformer, bd-conformer, bd-atcnet, bd-deepsleepnet

Paths (defaults = the mcl-server layout the logged runs used):
  EEG_DATA_ROOT  data root with CBraMod's LMDBs, the ones every other model on these splits used   (default ~/data)
  GAPS_WORK      runs/ and queue/ are written here                                                  (default ~/repro-gaps)
  GAPS_QUEUE     queue directory                                                                    (default $GAPS_WORK/queue)
  GAPS_PY        python with torch and braindecode 1.8.1 (default ~/miniforge3/envs/bd-labram/bin/python, as the SEED runs)
  NVML_FIX       LD_LIBRARY_PATH entry for the server's NVML mismatch  (default ~/repro-braindecode/nvml-580.173.02/lib)
"""
import os
import shlex
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
E = lambda k, d: os.path.expanduser(os.environ.get(k, d))  # noqa: E731
D, WORK = E('EEG_DATA_ROOT', '~/data'), E('GAPS_WORK', '~/repro-gaps')
PY = E('GAPS_PY', '~/miniforge3/envs/bd-labram/bin/python')
NVML = E('NVML_FIX', '~/repro-braindecode/nvml-580.173.02/lib')
Q = os.path.join(E('GAPS_QUEUE', os.path.join(WORK, 'queue')), 'pending')
DATA = {   # as methods/dgcnn/experiments/cbramod-splits/code/add_jobs.py
    'MentalArithmetic': f'{D}/mental_arithmetic/processed',
    'PhysioNet-MI': f'{D}/physio/processed_average',
    'Mumtaz2016': f'{D}/mumtaz/processed_lmdb_75hz',
    'BCIC-IV-2a': f'{D}/bciciv2a/processed_inde_avg_filter',
}
q = shlex.quote
os.makedirs(Q, exist_ok=True)
model, seeds = sys.argv[1], [int(s) for s in sys.argv[2:]]
for ds, data in DATA.items():
    for seed in seeds:
        name = f'{ds}__{model}__s{seed}'
        run = os.path.join(WORK, 'runs/cbramod-splits', name)
        cmd = (f'#!/bin/bash\n# job: {name}\nmkdir -p {q(run)}\nexport LD_LIBRARY_PATH={q(NVML)}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}\n'
               f'CUDA_VISIBLE_DEVICES=$1 {q(PY)} -u {q(os.path.join(HERE, "run_models.py"))} --downstream_dataset {ds} '
               f'--datasets_dir {q(data)} --model {model} --seed {seed} --model_dir {q(run + "/ckpt")} '
               f'> {q(run + "/log.txt")} 2>&1\nrm -rf {q(run + "/ckpt")}\n')
        open(os.path.join(Q, f'05_{name}.sh'), 'w').write(cmd)
        print('queued', name)
