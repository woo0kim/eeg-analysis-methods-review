"""Enqueue L-SeqSleepNet's released leave-one-subject-out training on Sleep-EDF-20 for tools/run_queue.py.

Each fold runs the two commands of the released run_training_repeat<r>.sh (train_lseqsleepnet.py, then
test_lseqsleepnet.py, same flags), from the released network/lseqsleepnet directory and on the released data
(sleepedf-20/mat_30min, file_list_20sub). One change, marked [MOD]: the released scripts pass test_list_n1.txt to
test_lseqsleepnet.py for every fold, while evaluate_performance.sh / evaluate.py score fold n against
test_list_n<n>.txt; here fold n is tested on test_list_n<n>.txt, its own held-out subject. As in the released scripts,
two folds share one GPU (--gpu_usage 0.45 each). The scripts run through run_released.py with --mod_eval_factor=10
([MOD] evaluation batch of 80 instead of 640 sequences: the released batch runs out of memory on a 12 GB GPU at
--gpu_usage 0.45; predictions do not depend on it). After all folds: evaluate.py as released (see README).

usage: python add_jobs.py <repeat> <fold> [<fold> ...]       folds 1-20, paired in the order given

Paths (defaults = the mcl-server layout the logged runs used):
  LSEQ_SRC    released repository at the pinned commit   (default ~/repro-sleepmodels/lseqsleepnet-upstream)
  LSEQ_WORK   fold outputs                               (default ~/repro-sleepmodels/runs/lseq_sc20)
  TF_PY, QUEUE_DIR, NVML_FIX as in ../../../../deepsleepnet/experiments/sleepedf20-20fold/code/add_jobs.py
"""
import os
import shlex
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

E = lambda k, d: os.path.expanduser(os.environ.get(k, d))
SRC = E('LSEQ_SRC', '~/repro-sleepmodels/lseqsleepnet-upstream')
WORK = E('LSEQ_WORK', '~/repro-sleepmodels/runs/lseq_sc20')
PY = E('TF_PY', '~/miniforge3/envs/tf115/bin/python')
Q = os.path.join(E('QUEUE_DIR', '~/repro-sleepedf/queue'), 'pending')
NVML = E('NVML_FIX', '~/repro-braindecode/nvml-580.173.02/lib')
FLAGS = ('--dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 '
         '--nsubseq 20 --dualrnn_blocks 1 --gpu_usage 0.45')       # run_training_repeat1.sh
LISTS = '../../file_list_20sub/eeg'
RUN = shlex.quote(os.path.join(HERE, 'run_released.py')) + ' {} --mod_eval_factor=10'   # [MOD] see run_released.py
q = shlex.quote

repeat, folds = int(sys.argv[1]), [int(f) for f in sys.argv[2:]]
os.makedirs(Q, exist_ok=True)
for i in range(0, len(folds), 2):
    pair = folds[i:i + 2]
    lines = [f'#!/bin/bash\n# job: L-SeqSleepNet Sleep-EDF-20 repeat {repeat} folds {pair}',
             f'cd {q(SRC)}/sleepedf-20/network/lseqsleepnet',
             f'export LD_LIBRARY_PATH={q(NVML)}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}} TF_CPP_MIN_LOG_LEVEL=2 '
             'CUDA_VISIBLE_DEVICES=$1']
    for n in pair:
        out = f'{WORK}/repeat{repeat}/n{n}/'
        lines.append(
            f'( mkdir -p {q(out)} && '
            f'{q(PY)} -u {RUN.format("train_lseqsleepnet.py")} --eeg_train_data "{LISTS}/train_list_n{n}.txt" '
            f'--eeg_eval_data "{LISTS}/eval_list_n{n}.txt" --out_dir {q(out)} {FLAGS} --max_eval_steps 110 '
            f'> {q(out)}train.txt 2>&1 && '
            f'{q(PY)} -u {RUN.format("test_lseqsleepnet.py")} --eeg_train_data "{LISTS}/train_list_n{n}.txt" '
            f'--eeg_test_data "{LISTS}/test_list_n{n}.txt" '         # [MOD] released: test_list_n1.txt for every fold
            f'--out_dir {q(out)} {FLAGS} > {q(out)}test.txt 2>&1 ) &')
    lines.append('wait')
    lines.append(' && '.join(f'test -s {q(WORK)}/repeat{repeat}/n{n}/test_ret.mat' for n in pair))
    name = f'12_LSEQ_sc20__r{repeat}_n{"-".join(f"{n:02d}" for n in pair)}.sh'
    open(os.path.join(Q, name), 'w').write('\n'.join(lines) + '\n')
    print('queued', name)
