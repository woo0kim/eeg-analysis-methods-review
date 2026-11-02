"""Enqueue Sleep-EDF runs for tools/run_queue.py (one job = one model x one seed, 50 epochs on one GPU).

usage: python add_jobs.py <model> <seed> [<seed> ...]       models: see seq_models.build()

--accum is the number of micro-batches per batch of 64 sequences, chosen with smoke_test.py so that a training step
fits in the 12 GB of a Quadro M6000 (it changes memory use, not the update; see seq_trainer.py).
The queue claims job files in sorted order: released implementations first (longest runs first), then braindecode's
DeepSleepNet (the only DeepSleepNet that can run here), then braindecode's versions of the other models.

Paths (defaults = the mcl-server layout the logged runs used):
  EEG_DATA_ROOT     data root; the preprocessed set is $EEG_DATA_ROOT/sleep_edfx/processed_seq20 (default ~/data)
  SLEEPEDF_WORK     runs/ and queue/ are written here       (default ~/repro-sleepedf)
  SLEEPEDF_PY       python with torch 2.1, braindecode 1.8.1, timm 0.4.12 (default ~/miniforge3/envs/bd-labram/bin/python)
  CBRAMOD_WEIGHTS   CBraMod's released pretrained_weights.pth (default ~/CBraMod/pretrained_weights/pretrained_weights.pth)
  NVML_FIX          LD_LIBRARY_PATH entry for the server's NVML driver mismatch (default ~/repro-braindecode/nvml-580.173.02/lib)
"""
import os
import shlex
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.expanduser(os.environ.get('EEG_DATA_ROOT', '~/data'))
DATA = f'{D}/sleep_edfx/processed_seq20'
WORK = os.path.expanduser(os.environ.get('SLEEPEDF_WORK', '~/repro-sleepedf'))
PY = os.path.expanduser(os.environ.get('SLEEPEDF_PY', '~/miniforge3/envs/bd-labram/bin/python'))
WEIGHTS = os.path.expanduser(os.environ.get('CBRAMOD_WEIGHTS', '~/CBraMod/pretrained_weights/pretrained_weights.pth'))
NVML = os.path.expanduser(os.environ.get('NVML_FIX', '~/repro-braindecode/nvml-580.173.02/lib'))
# model -> (micro-batches per batch of 64, queue priority: lower runs first)
CFG = {
    'conformer': (16, 1), 'cbramod': (2, 2), 'cbramod-scratch': (2, 3), 'labram': (2, 4), 'bd-atcnet': (1, 5),
    'eegnet': (1, 6),
    'bd-deepsleepnet': (2, 7),
    'bd-conformer': (2, 27), 'bd-cbramod': (2, 28), 'bd-labram': (2, 29), 'bd-eegnet': (1, 30),
}
Q = os.path.join(WORK, 'queue/pending')
os.makedirs(Q, exist_ok=True)

model, seeds = sys.argv[1], [int(s) for s in sys.argv[2:]]
accum, prio = CFG[model]
q = shlex.quote
for seed in seeds:
    name = f'SleepEDF__{model}__s{seed}'
    run_dir = os.path.join(WORK, 'runs', name)
    cmd = (f'#!/bin/bash\n# job: {name}\nmkdir -p {q(run_dir)}\n'
           f'export LD_LIBRARY_PATH={q(NVML)}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}\n'
           f'export CBRAMOD_WEIGHTS={q(WEIGHTS)}\n'
           f'CUDA_VISIBLE_DEVICES=$1 {q(PY)} -u {q(os.path.join(HERE, "run_seq.py"))} --model {model} --seed {seed} '
           f'--accum {accum} --datasets_dir {q(DATA)} --model_dir {q(run_dir)} > {q(run_dir + "/log.txt")} 2>&1\n')
    open(os.path.join(Q, f'{prio:02d}_{name}.sh'), 'w').write(cmd)
    print('queued', name)
