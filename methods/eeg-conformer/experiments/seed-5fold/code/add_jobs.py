"""Enqueue EEG Conformer's released SEED runs: 15 subjects (session 1, as released) x 5 folds, one seed per subject + fold,
and the same model on a split between clips (--split trial96), 15 subjects x 3 seeds.

usage: python add_jobs.py
Paths: SEED_WORK (default ~/repro-seed), SEED_QUEUE (default $SEED_WORK/queue), SEED_PY, NVML_FIX as in the other SEED experiments.
"""
import os
import shlex

HERE = os.path.dirname(os.path.abspath(__file__))
E = lambda k, d: os.path.expanduser(os.environ.get(k, d))
WORK = E('SEED_WORK', '~/repro-seed')
PY = E('SEED_PY', '~/miniforge3/envs/bd-labram/bin/python')
NVML = E('NVML_FIX', '~/repro-braindecode/nvml-580.173.02/lib')
Q = os.path.join(E('SEED_QUEUE', os.path.join(WORK, 'queue')), 'pending')
ROOT = os.path.join(WORK, 'conformer_data/')
os.makedirs(Q, exist_ok=True)
q = shlex.quote


def job(split, sub, fold, seed):
    name = f'conformer-seed__{split}__S{sub}__f{fold}__s{seed}'
    out = os.path.join(WORK, 'runs/conformer', f'{split}_s{seed}')
    os.makedirs(out, exist_ok=True)
    cmd = (f'#!/bin/bash\n# job: {name}\nexport LD_LIBRARY_PATH={q(NVML)}\n'
           f'cd {q(out)}\nCUDA_VISIBLE_DEVICES=$1 {q(PY)} -u {q(os.path.join(HERE, "conformer_seed_1s_5fold.py"))} '
           f'--subject {sub} --fold {fold} --seed {seed} --split {split} --root {q(ROOT)} --out {q(out)} '
           f'> {q(out + f"/stdout_S{sub}_f{fold}.txt")} 2>&1\n')
    open(os.path.join(Q, f'{5 if split == "kfold" else 6:02d}_{name}.sh'), 'w').write(cmd)


for sub in range(1, 16):
    for fold in range(5):
        job('kfold', sub, fold, 2021 + 100 * sub + fold)
    for seed in (0, 1, 2):
        job('trial96', sub, 0, seed)
print('queued')
