"""Enqueue DeepSleepNet runs on Sleep-EDF-78 with the CBraMod-pipeline subject split, for tools/run_queue.py.

One job = one run: train_split.py (the released train.py, 100 + 200 epochs) and predict_split.py on the 8 test
subjects. The released code sets no random seed, so runs differ by their random initialisation and data order only.
usage: python add_jobs.py <run> [<run> ...]          e.g. 0 1 2

Paths: DSN_SRC, TF_PY, QUEUE_DIR, NVML_FIX as in ../../sleepedf20-20fold/code/add_jobs.py, and
  DSN_DATA78  prepare_split_data.py output, 153 nights in prepare_physionet.py's format (default ~/repro-sleepmodels/data/dsn_sc78)
  DSN_WORK78  run outputs                                     (default ~/repro-sleepmodels/runs/dsn_sc78)
"""
import os
import shlex
import sys

E = lambda k, d: os.path.expanduser(os.environ.get(k, d))
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = E('DSN_DATA78', '~/repro-sleepmodels/data/dsn_sc78')
WORK = E('DSN_WORK78', '~/repro-sleepmodels/runs/dsn_sc78')
PY = E('TF_PY', '~/miniforge3/envs/tf115/bin/python')
Q = os.path.join(E('QUEUE_DIR', '~/repro-sleepedf/queue'), 'pending')
NVML = E('NVML_FIX', '~/repro-braindecode/nvml-580.173.02/lib')
q = shlex.quote

os.makedirs(Q, exist_ok=True)
for r in [int(x) for x in sys.argv[1:]]:
    out = f'{WORK}/run{r}'
    cmd = (f'#!/bin/bash\n# job: DeepSleepNet Sleep-EDF-78 split, run {r}\nmkdir -p {q(out)}\n'
           f'export LD_LIBRARY_PATH={q(NVML)}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}} TF_FORCE_GPU_ALLOW_GROWTH=true '
           'TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=$1\n'
           f'{q(PY)} -u {q(os.path.join(HERE, "train_split.py"))} --data_dir {q(DATA)} --output_dir {q(out)} '
           f'--n_folds 20 --fold_idx 0 --pretrain_epochs 100 --finetune_epochs 200 --resume False > {q(out)}/train.txt 2>&1 && '
           f'{q(PY)} -u {q(os.path.join(HERE, "predict_split.py"))} --data_dir {q(DATA)} --model_dir {q(out)} '
           f'--output_dir {q(out)}/predict > {q(out)}/predict.txt 2>&1\n')
    open(os.path.join(Q, f'13_DSN_sc78__run{r}.sh'), 'w').write(cmd)
    print('queued', f'13_DSN_sc78__run{r}.sh')
