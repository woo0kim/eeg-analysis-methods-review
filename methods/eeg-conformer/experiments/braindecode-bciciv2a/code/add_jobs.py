"""Queue the 54 braindecode-EEGConformer runs (9 subjects x 3 seeds x protocols A and B) for tools/run_queue.py.

Protocol A: all 288 trials of session T (--val-frac 0); B: 20% of session T held out (--val-frac 0.2).
usage: python code/add_jobs.py [A|B ...]          # default: A B

Paths (defaults = the mcl-server layout of the committed runs):
  BDCONF_WORK  data_proc/ (preprocessed), results/, logs/      (default ~/repro-braindecode/conformer)
  BD_QUEUE     queue directory shared by the braindecode runs  (default ~/repro-braindecode/queue)
  BDCONF_PY    python of the eegnet env                        (default ~/miniforge3/envs/eegnet/bin/python)
"""
import os, shlex, sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.expanduser(os.environ.get('BDCONF_WORK', '~/repro-braindecode/conformer'))
QUEUE = os.path.expanduser(os.environ.get('BD_QUEUE', '~/repro-braindecode/queue'))
PY = os.path.expanduser(os.environ.get('BDCONF_PY', '~/miniforge3/envs/eegnet/bin/python'))
VAL_FRAC = {'A': 0.0, 'B': 0.2}
SEEDS = (2023, 2024, 2025)          # as in the released-code runs
Q = os.path.join(QUEUE, 'pending')
for d in (Q, os.path.join(WORK, 'results'), os.path.join(WORK, 'logs')):
    os.makedirs(d, exist_ok=True)

q = shlex.quote
for tag in sys.argv[1:] or ['A', 'B']:
    for seed in SEEDS:
        for sub in range(1, 10):
            name = f'{tag}_s{sub}_seed{seed}'
            cmd = (f'#!/bin/bash\n# job: conformer {name}\nexport CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=1\n'
                   f'{q(PY)} -u {q(os.path.join(HERE, "run_braindecode.py"))} --subject {sub} --seed {seed} '
                   f'--epochs 2000 --val-frac {VAL_FRAC[tag]} --tag {tag} --root {q(WORK + "/data_proc/")} '
                   f'--out {q(WORK + "/results")} > {q(f"{WORK}/logs/{name}.txt")} 2>&1\n')
            open(os.path.join(Q, f'5_conformer_{name}.sh'), 'w').write(cmd)
            print('queued', name)
