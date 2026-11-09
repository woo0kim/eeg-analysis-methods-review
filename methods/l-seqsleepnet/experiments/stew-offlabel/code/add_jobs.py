"""Enqueue L-SeqSleepNet on STEW (off-label), one fold per job: 3 runs per dataset and split, two runs sharing one GPU, as released.

The released train_lseqsleepnet.py + test_lseqsleepnet.py run through ../../sleepedf20-loso/code/run_released.py, with the
flags of run_training_repeat1.sh except the sequence: 5 x 1 = 5 epochs (released 10 x 20 = 200), a whole STEW recording. Training length: the released 440 passes over the data (11,000 steps x 8 x 200 / 40,000
epochs), at the released 110 validations, so --mod_evaluate_every sets the step count. Selection: best validation
accuracy, as released. The released code sets no seed.

usage: python add_jobs.py <rest-task|rating3> <fold> <run> [<run> ...]
Paths: LSEQ_STEW (default ~/repro-seed/lseq/STEW), LSEQ_SRC, TF_PY, SEED_QUEUE (default ~/repro-seed/queue), NVML_FIX
"""
import os
import shlex
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
E = lambda k, d: os.path.expanduser(os.environ.get(k, d))
DATA = E('LSEQ_STEW', '~/repro-seed/lseq/STEW')
SRC = E('LSEQ_SRC', '~/repro-sleepmodels/lseqsleepnet-upstream')
PY = E('TF_PY', '~/miniforge3/envs/tf115/bin/python')
NVML = E('NVML_FIX', '~/repro-braindecode/nvml-580.173.02/lib')
Q = os.path.join(E('SEED_QUEUE', '~/repro-seed/queue'), 'pending')
FLAGS = ('--dropout_rnn 0.9 --sub_seq_len 5 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 '
         '--nsubseq 1 --dualrnn_blocks 1 --gpu_usage 0.45')
q = shlex.quote
RUN = q(os.path.normpath(os.path.join(HERE, '../../sleepedf20-loso/code/run_released.py'))) + ' {} --mod_eval_factor=10'

ds, split, runs = sys.argv[1], f'fold{sys.argv[2]}', [int(r) for r in sys.argv[3:]]
d = DATA
n_epochs = sum(int(l.split('\t')[1]) for l in open(f'{d}/{ds}_{split}_train_eeg.txt') if l.strip())
every = max(1, round(440 * n_epochs / (8 * 5) / 110))
print(ds, split, n_epochs, 'training epochs; validation every', every, 'steps')
lst = lambda part: ' '.join(f'--{s}_{part}_data {q(f"{d}/{ds}_{split}_{p}_{s}.txt")}' for s in ('eeg', 'eog', 'emg')
                            for p in [{'train': 'train', 'eval': 'eval', 'test': 'test'}[part]])
for i in range(0, len(runs), 2):
    pair = runs[i:i + 2]
    lines = [f'#!/bin/bash\n# job: L-SeqSleepNet {ds} {split}, runs {pair}', f'cd {q(SRC)}/sleepedf-20/network/lseqsleepnet',
             f'export LD_LIBRARY_PATH={q(NVML)} TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=$1']
    for r in pair:
        out = f'{d}/runs/{ds}/{split}/run{r}/'
        lines.append(f'( mkdir -p {q(out)} && {q(PY)} -u {RUN.format("train_lseqsleepnet.py")} --mod_evaluate_every={every} '
                     f'{lst("train")} {lst("eval")} --out_dir {q(out)} {FLAGS} --max_eval_steps 110 > {q(out)}train.txt 2>&1 && '
                     f'{q(PY)} -u {RUN.format("test_lseqsleepnet.py")} {lst("train")} {lst("test")} --out_dir {q(out)} '
                     f'{FLAGS} > {q(out)}test.txt 2>&1 ) &')
    lines += ['wait', ' && '.join(f'test -s {q(f"{d}/runs/{ds}/{split}/run{r}/test_ret.mat")}' for r in pair)]
    name = f'85_LSEQ_STEW_{ds}_{split}__runs{"-".join(map(str, pair))}.sh'
    open(os.path.join(Q, name), 'w').write('\n'.join(lines) + '\n')
    print('queued', name)
