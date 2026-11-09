"""Queue the 54 DGCNN runs (9 subjects x 3 seeds x protocols A and B), as the EEGNet and EEG Conformer experiments.

Protocol A: all 288 trials of session T (--val-frac 0); B: 20% of session T held out (--val-frac 0.2).
Paths: SEED_WORK (runs in runs/dgcnn-2a; default ~/repro-seed), SEED_QUEUE, SEED_PY, NVML_FIX, and
DATA_2A, the preprocessed .mat files of those experiments (default ~/repro-eegnet/data_proc/).
"""
import os
import shlex

HERE = os.path.dirname(os.path.abspath(__file__))
E = lambda k, d: os.path.expanduser(os.environ.get(k, d))
WORK = E('SEED_WORK', '~/repro-seed')
PY = E('SEED_PY', '~/miniforge3/envs/bd-labram/bin/python')
NVML = E('NVML_FIX', '~/repro-braindecode/nvml-580.173.02/lib')
DATA = E('DATA_2A', '~/repro-eegnet/data_proc/')
Q = os.path.join(E('SEED_QUEUE', os.path.join(WORK, 'queue')), 'pending')
OUT = os.path.join(WORK, 'runs/dgcnn-2a')
os.makedirs(os.path.join(OUT, 'logs'), exist_ok=True)
q = shlex.quote
for tag, frac in (('A', 0.0), ('B', 0.2)):
    for seed in (2023, 2024, 2025):
        for sub in range(1, 10):
            name = f'dgcnn2a_{tag}_s{sub}_seed{seed}'
            cmd = (f'#!/bin/bash\n# job: {name}\nexport CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=1 LD_LIBRARY_PATH={q(NVML)}\n'
                   f'{q(PY)} -u {q(os.path.join(HERE, "run_dgcnn.py"))} --subject {sub} --seed {seed} --epochs 2000 '
                   f'--val-frac {frac} --tag {tag} --root {q(DATA)} --out {q(OUT)} > {q(f"{OUT}/logs/{name}.txt")} 2>&1\n')
            open(os.path.join(Q, f'08_{name}.sh'), 'w').write(cmd)
print('queued 54')
