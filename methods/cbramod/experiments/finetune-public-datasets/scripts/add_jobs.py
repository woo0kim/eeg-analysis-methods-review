"""Enqueue CBraMod fine-tuning runs with the released finetune_main.py.

All hyper-parameters are the released defaults (= paper Table 6): epochs 50, batch 64, lr 1e-4,
wd 5e-2, AdamW, clip 1, dropout 0.1, label smoothing 0.1, classifier all_patch_reps, multi_lr.
usage: python add_jobs.py <Dataset> <pretrained|scratch> <seed> [<seed> ...]
NOTE: finetune_main.py parses booleans with type=bool, so `--use_pretrained_weights False` would be
True; the empty string is the only way to pass False on the command line.

Paths (defaults = the mcl-server layout the 76 logged runs used):
  EEG_DATA_ROOT  preprocessed datasets          (default ~/data)
  CBRAMOD_WORK   runs/ and queue/ are written here (default ~/repro)
  CBRAMOD_SRC    released CBraMod code          (default: this repo's methods/cbramod/upstream)
  CBRAMOD_PY     python of the cbramod env      (default ~/miniforge3/envs/cbramod/bin/python)
"""
import os, shlex, sys

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.expanduser(os.environ.get('EEG_DATA_ROOT', '~/data'))
WORK = os.path.expanduser(os.environ.get('CBRAMOD_WORK', '~/repro'))
SRC = os.environ.get('CBRAMOD_SRC', os.path.normpath(os.path.join(HERE, '../../../upstream')))
CFG = {
    'MentalArithmetic': (f'{D}/mental_arithmetic/processed', 2),
    'PhysioNet-MI': (f'{D}/physio/processed_average', 4),
    'SHU-MI': (f'{D}/shu/processed', 2),
    'Mumtaz2016': (f'{D}/mumtaz/processed_lmdb_75hz', 2),
    'BCIC-IV-2a': (f'{D}/bciciv2a/processed_inde_avg_filter', 4),
    # same dataset preprocessed with the code before the A04T fix (5,088 samples, as in the paper)
    'BCIC-IV-2a@v0616': (f'{D}/bciciv2a/processed_v0616', 4),
}
# Fine-tuning settings. Only 'pretrained' / 'scratch' are compared with the paper; the others are
# robustness checks suggested by the first author in GitHub issues (#6: lr=5e-4 with multi_lr=False).
SETTINGS = {
    'pretrained': '',
    'scratch': " --use_pretrained_weights ''",
    'pt-lr5e-4-single': " --lr 5e-4 --multi_lr ''",
    'eegnet': '',
    'frozen': ' --frozen True',   # paper Table 18 "CBraMod (Fixed)"; bool('True') is True
}
# queue claims files in sorted order -> longest jobs first
PRIO = {'PhysioNet-MI': 1, 'SHU-MI': 2, 'Mumtaz2016': 3, 'BCIC-IV-2a': 4, 'MentalArithmetic': 5}
PY = os.path.expanduser(os.environ.get('CBRAMOD_PY', '~/miniforge3/envs/cbramod/bin/python'))
Q = os.path.join(WORK, 'queue/pending')
os.makedirs(Q, exist_ok=True)

ds, setting, seeds = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3:]]
data_dir, ncls = CFG[ds]
for seed in seeds:
    name = f'{ds}__{setting}__s{seed}'
    run_dir = os.path.join(WORK, 'runs', name)
    extra = SETTINGS[setting]
    if setting == 'eegnet':     # supplementary baseline through the same released Trainer/loaders
        prog = f'{shlex.quote(os.path.join(HERE, "run_eegnet.py"))} --seed {seed} --cuda 0'
        prio = 8
    else:
        prog = f'finetune_main.py --seed {seed} --cuda 0 --num_of_classes {ncls}'
        if '@' in ds:
            prio = 7
        else:
            prio = PRIO[ds] if setting in ('pretrained', 'scratch') else (6 if setting == 'frozen' else 9)
    q = shlex.quote
    cmd = (f'#!/bin/bash\n# job: {name}\nmkdir -p {q(run_dir)}\ncd {q(SRC)}\n'
           f'CUDA_VISIBLE_DEVICES=$1 {q(PY)} -u {prog} '
           f'--downstream_dataset {ds.split("@")[0]} --datasets_dir {q(data_dir)} '
           f'--model_dir {q(run_dir + "/ckpt")}{extra} > {q(run_dir + "/log.txt")} 2>&1\n')
    open(os.path.join(Q, f'{prio}_{setting[0]}_{name}.sh'), 'w').write(cmd)
    print('queued', name)
