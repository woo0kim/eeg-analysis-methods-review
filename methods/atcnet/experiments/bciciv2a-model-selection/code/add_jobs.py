"""Queue the released-code ATCNet runs for tools/run_queue.py: 9 subjects (or LOSO folds) x seeds per protocol.

  traintest, trainvaltest, loso   code/run_released.py (today's released code)
  paper2022                       code/run_paper2022.py (the 2022 code behind the paper's numbers)
usage: python code/add_jobs.py [PROTOCOL[:SEEDS] ...]     e.g. loso:1-3
       default: traintest trainvaltest paper2022 loso:1-3 (297 runs, as committed)

Paths (defaults = the mcl-server layout of the committed runs):
  ATCNET_WORK      outputs: released/<protocol>/{saved models,curves}, released/logs   (default ~/repro-atcnet)
  ATCNET_QUEUE     queue directory                                                    (default ~/repro-atcnet/queue)
  ATCNET_TF_ENV    conda env built by env/setup_env.sh                                (default ~/miniforge3/envs/atcnet-tf)
  ATCNET_NVML_LIB  extra LD_LIBRARY_PATH entry for mcl-server's NVML mismatch, if the directory exists
                   (default ~/repro-braindecode/nvml-580.173.02/lib; see the README, Run configuration)
"""
import os, shlex, sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.expanduser(os.environ.get('ATCNET_WORK', '~/repro-atcnet'))
QUEUE = os.path.expanduser(os.environ.get('ATCNET_QUEUE', '~/repro-atcnet/queue'))
ENV = os.path.expanduser(os.environ.get('ATCNET_TF_ENV', '~/miniforge3/envs/atcnet-tf'))
NVML = os.path.expanduser(os.environ.get('ATCNET_NVML_LIB', '~/repro-braindecode/nvml-580.173.02/lib'))
PRIO = {'traintest': '1_k', 'paper2022': '1_l', 'loso': '1_m', 'trainvaltest': '2_k'}   # queue order (braindecode: 3_*)
DEFAULT = ['traintest', 'trainvaltest', 'paper2022', 'loso:1-3']   # n_train = 10; LOSO runs take ~1.5 h, so 3 seeds
LOGS = os.path.join(WORK, 'released', 'logs')
for d in (os.path.join(QUEUE, 'pending'), LOGS):
    os.makedirs(d, exist_ok=True)

q = shlex.quote
ld = os.path.join(ENV, 'lib') + (':' + NVML if os.path.isdir(NVML) else '')
for arg in sys.argv[1:] or DEFAULT:
    protocol, _, rng = arg.partition(':')
    lo, hi = map(int, (rng or '1-10').split('-'))
    for seed in range(lo, hi + 1):
        for sub in range(1, 10):
            name = f'{protocol}_s{sub}_seed{seed:02d}'
            script, proto_arg = (('run_paper2022.py', '') if protocol == 'paper2022' else
                                 ('run_released.py', f'--protocol {protocol} '))
            cmd = (f'#!/bin/bash\n# job: atcnet released {name}\n'
                   f'export CUDA_VISIBLE_DEVICES=$1 TF_GPU_MEM_MB=5000 TF_CPP_MIN_LOG_LEVEL=2 TF_THREADS=2 '
                   f'OMP_NUM_THREADS=2 MPLBACKEND=Agg LD_LIBRARY_PATH={q(ld)}\n'
                   f'{q(ENV + "/bin/python")} -u {q(os.path.join(HERE, script))} {proto_arg}'
                   f'--subject {sub} --seed {seed} --out {q(WORK + "/released")} > {q(f"{LOGS}/{name}.txt")} 2>&1\n')
            open(os.path.join(QUEUE, 'pending', f'{PRIO[protocol]}_{protocol}_seed{seed:02d}_s{sub}.sh'), 'w').write(cmd)
            print('queued', name)
