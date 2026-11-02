"""Enqueue DeepSleepNet's released 20-fold cross-validation on Sleep-EDF-20 for tools/run_queue.py.

Each fold runs the README's command unchanged (train.py: 100 pre-training + 200 fine-tuning epochs, the held-out
subject as fold k). One queue job trains two folds side by side on one GPU (TensorFlow memory growth on).
After all folds: predict.py and summary.py as released (see README, "Reproducing").

usage: python add_jobs.py <fold> [<fold> ...]        folds 0-19, paired in the order given

Paths (defaults = the mcl-server layout the logged runs used):
  DSN_SRC     released code at the pinned commit          (default ~/repro-sleepmodels/deepsleepnet-upstream)
  DSN_DATA    output of prepare_physionet.py, Fpz-Cz      (default ~/repro-sleepmodels/data/dsn_eeg_fpz_cz_sc20)
  DSN_WORK    fold outputs and logs                        (default ~/repro-sleepmodels/runs/dsn_sc20)
  TF_PY       python of the TF 1.15 env                    (default ~/miniforge3/envs/tf115/bin/python)
  QUEUE_DIR   tools/run_queue.py queue                     (default ~/repro-sleepedf/queue)
  NVML_FIX    LD_LIBRARY_PATH entry for the server's NVML mismatch (default ~/repro-braindecode/nvml-580.173.02/lib)
"""
import os
import shlex
import sys

E = lambda k, d: os.path.expanduser(os.environ.get(k, d))
SRC = E('DSN_SRC', '~/repro-sleepmodels/deepsleepnet-upstream')
DATA = E('DSN_DATA', '~/repro-sleepmodels/data/dsn_eeg_fpz_cz_sc20')
WORK = E('DSN_WORK', '~/repro-sleepmodels/runs/dsn_sc20')
PY = E('TF_PY', '~/miniforge3/envs/tf115/bin/python')
Q = os.path.join(E('QUEUE_DIR', '~/repro-sleepedf/queue'), 'pending')
NVML = E('NVML_FIX', '~/repro-braindecode/nvml-580.173.02/lib')
q = shlex.quote

folds = [int(f) for f in sys.argv[1:]]
os.makedirs(Q, exist_ok=True)
for i in range(0, len(folds), 2):
    pair = folds[i:i + 2]
    lines = [f'#!/bin/bash\n# job: DeepSleepNet Sleep-EDF-20 folds {pair}', f'mkdir -p {q(WORK)}/logs', f'cd {q(SRC)}',
             f'export LD_LIBRARY_PATH={q(NVML)}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}} TF_FORCE_GPU_ALLOW_GROWTH=true '
             'TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=$1']
    for k in pair:   # the README's command
        lines.append(f'{q(PY)} -u train.py --data_dir {q(DATA)} --output_dir {q(WORK)} --n_folds 20 --fold_idx {k} '
                     f'--pretrain_epochs 100 --finetune_epochs 200 --resume False > {q(WORK)}/logs/fold{k}.txt 2>&1 &')
    lines.append('wait')
    lines.append(' && '.join(f'grep -q "Finish fine-tuning" {q(WORK)}/logs/fold{k}.txt' for k in pair))
    name = f'11_DSN_sc20__folds{"-".join(map(str, pair))}.sh'
    open(os.path.join(Q, name), 'w').write('\n'.join(lines) + '\n')
    print('queued', name)
