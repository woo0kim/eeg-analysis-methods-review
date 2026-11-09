"""Enqueue the DGCNN port's runs for tools/run_queue.py.

usage: python add_jobs.py [lr|loso-lr]
Paths: SEED_WORK (default ~/repro-seed; DE inputs in de/<key>, runs in runs/, queue in queue/), SEED_PY (python with
torch), NVML_FIX (LD_LIBRARY_PATH entry for the server's NVML driver mismatch).
"""
import os
import shlex
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.expanduser(os.environ.get('SEED_WORK', '~/repro-seed'))
PY = os.path.expanduser(os.environ.get('SEED_PY', '~/miniforge3/envs/bd-labram/bin/python'))
NVML = os.path.expanduser(os.environ.get('NVML_FIX', '~/repro-braindecode/nvml-580.173.02/lib'))
Q = os.path.join(WORK, 'queue/pending')
os.makedirs(Q, exist_ok=True)
q = shlex.quote


def job(prio, name, args):
    out = os.path.join(WORK, 'runs/port', name)
    os.makedirs(out, exist_ok=True)
    cmd = (f'#!/bin/bash\n# job: {name}\nexport LD_LIBRARY_PATH={q(NVML)}\n'
           f'CUDA_VISIBLE_DEVICES=$1 {q(PY)} -u {q(os.path.join(HERE, "run_port.py"))} {args} '
           f'--out {q(out + "/result.npz")} > {q(out + "/log.txt")} 2>&1\n')
    open(os.path.join(Q, f'{prio:02d}_{name}.sh'), 'w').write(cmd)
    print('queued', name)


mode = sys.argv[1] if len(sys.argv) > 1 else 'main'
if mode == 'lr':
    # learning-rate check: eeg_DE.py's 0.07 is unstable (some sessions collapse), so the fixed-step numbers could understate
    # the model. The rate is picked on the validation clips of sd-val (trials 8-9), then read off the 9-trial runs.
    de = q(os.path.join(WORK, 'de', 'de_LDS'))
    for lr in (0.03, 0.01, 0.003, 0.001):
        for seed in range(5):
            job(2, f'sdlr{lr}__de_LDS__s{seed}', f'--protocol sd --de_dir {de} --seed {seed} --lr {lr}')
            job(2, f'sdvallr{lr}__de_LDS__s{seed}', f'--protocol sd-val --de_dir {de} --seed {seed} --lr {lr}')
    sys.exit()
if mode == 'loso-lr':          # LOSO at the rate the subject-dependent validation clips picked
    de = q(os.path.join(WORK, 'de', 'de_LDS'))
    for seed in range(3):
        job(4, f'loso-sessions__de_LDS__zscore-lr0.01__s{seed}',
            f'--protocol loso --de_dir {de} --seed {seed} --norm zscore --lr 0.01')
    sys.exit()

for key in ('de_LDS', 'de_movingAve'):
    de = q(os.path.join(WORK, 'de', key))
    for seed in range(5):
        job(1, f'sd__{key}__s{seed}', f'--protocol sd --de_dir {de} --seed {seed}')
        job(2, f'sdval__{key}__s{seed}', f'--protocol sd-val --de_dir {de} --seed {seed}')
    for seed in range(3):
        for norm in ('authors', 'zscore'):
            job(3, f'loso-sessions__{key}__{norm}__s{seed}', f'--protocol loso --de_dir {de} --seed {seed} --norm {norm}')
            job(4, f'loso-pooled__{key}__{norm}__s{seed}',
                f'--protocol loso --de_dir {de} --seed {seed} --norm {norm} --sessions all')
