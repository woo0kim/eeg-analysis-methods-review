"""Enqueue L-SeqSleepNet runs on Sleep-EDF-78 with the CBraMod-pipeline subject split, for tools/run_queue.py.

Each run is the released train_lseqsleepnet.py + test_lseqsleepnet.py with the flags of run_training_repeat1.sh, on
the files of prepare_split_data.py: training on the 62 training subjects, model selection (best validation accuracy)
on the 8 validation subjects, testing once on the 8 test subjects. The scripts run through
../../sleepedf20-loso/code/run_released.py, which changes two settings ([MOD]):
- the training length: the released --max_eval_steps 110 at one validation per 100 steps fixes training at 11,000 steps
  for Sleep-EDF-20 (about 40k training epochs per fold); this training set is about 3.9 times larger, so training runs
  4 times longer, 44,000 steps (same number of passes over the training data). It keeps the released 110 validations
  (best validation accuracy, as released) by validating every 400 steps (--mod_evaluate_every=400): every
  validation scores all 200-epoch windows of the 8 validation subjects, and 440 of them would take ~30 h per run;
- the evaluation batch: 80 instead of 640 sequences (--mod_eval_factor=10); the released batch runs out of memory on
  a 12 GB GPU at --gpu_usage 0.45, and predictions do not depend on it.
Two runs share one GPU (--gpu_usage 0.45 each), as in the released scripts. The released code sets no random seed.

usage: python add_jobs.py <run> [<run> ...]           runs are paired per job in the order given, e.g. 0 1 2
Paths: LSEQ_SRC, TF_PY, QUEUE_DIR, NVML_FIX as in ../../sleepedf20-loso/code/add_jobs.py, and
  LSEQ_DATA78  output of prepare_split_data.py  (default ~/repro-sleepmodels/data/lseq_sc78)
  LSEQ_WORK78  run outputs                       (default ~/repro-sleepmodels/runs/lseq_sc78)
"""
import os
import shlex
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

E = lambda k, d: os.path.expanduser(os.environ.get(k, d))
SRC = E('LSEQ_SRC', '~/repro-sleepmodels/lseqsleepnet-upstream')
DATA = E('LSEQ_DATA78', '~/repro-sleepmodels/data/lseq_sc78')
WORK = E('LSEQ_WORK78', '~/repro-sleepmodels/runs/lseq_sc78')
PY = E('TF_PY', '~/miniforge3/envs/tf115/bin/python')
Q = os.path.join(E('QUEUE_DIR', '~/repro-sleepedf/queue'), 'pending')
NVML = E('NVML_FIX', '~/repro-braindecode/nvml-580.173.02/lib')
FLAGS = ('--dropout_rnn 0.9 --sub_seq_len 10 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 '
         '--nsubseq 20 --dualrnn_blocks 1 --gpu_usage 0.45')       # run_training_repeat1.sh
q = shlex.quote
RUN = q(os.path.normpath(os.path.join(HERE, '../../sleepedf20-loso/code/run_released.py'))) + ' {} --mod_eval_factor=10'  # [MOD]

runs = [int(r) for r in sys.argv[1:]]
os.makedirs(Q, exist_ok=True)
for i in range(0, len(runs), 2):
    pair = runs[i:i + 2]
    lines = [f'#!/bin/bash\n# job: L-SeqSleepNet Sleep-EDF-78 split, runs {pair}',
             f'cd {q(SRC)}/sleepedf-20/network/lseqsleepnet',
             f'export LD_LIBRARY_PATH={q(NVML)}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}} TF_CPP_MIN_LOG_LEVEL=2 '
             'CUDA_VISIBLE_DEVICES=$1']
    for r in pair:
        out = f'{WORK}/run{r}/'
        lines.append(
            f'( mkdir -p {q(out)} && '
            f'{q(PY)} -u {RUN.format("train_lseqsleepnet.py")} --mod_evaluate_every=400 '
            f'--eeg_train_data {q(DATA)}/train_list.txt '
            f'--eeg_eval_data {q(DATA)}/eval_list.txt --out_dir {q(out)} {FLAGS} --max_eval_steps 110 '
            f'> {q(out)}train.txt 2>&1 && '
            f'{q(PY)} -u {RUN.format("test_lseqsleepnet.py")} --eeg_train_data {q(DATA)}/train_list.txt '
            f'--eeg_test_data {q(DATA)}/test_list.txt --out_dir {q(out)} {FLAGS} > {q(out)}test.txt 2>&1 ) &')
    lines.append('wait')
    lines.append(' && '.join(f'test -s {q(WORK)}/run{r}/test_ret.mat' for r in pair))
    name = f'10_LSEQ_sc78__runs{"-".join(map(str, pair))}.sh'
    open(os.path.join(Q, name), 'w').write('\n'.join(lines) + '\n')
    print('queued', name)
