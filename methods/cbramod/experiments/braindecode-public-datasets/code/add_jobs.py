"""Queue braindecode-CBraMod fine-tuning runs (code/run_braindecode.py) for tools/run_queue.py.

usage: python code/add_jobs.py <Dataset> <setting> <seed> [<seed> ...]
  pretrained | scratch | frozen     braindecode's CBraMod (code/run_braindecode.py)
  onelayer | onelayer-frozen        control: the *released* model and finetune_main.py with the released
                                    one-layer head (--classifier all_patch_reps_onelayer), i.e. braindecode's head
                                    without the pre-trained proj_out; fine-tuned, or with the backbone frozen

Paths (defaults = the mcl-server layout of the committed runs):
  EEG_DATA_ROOT    CBraMod's preprocessed LMDBs                  (default ~/data)
  BD_CBRAMOD_WORK  runs/ are written here                        (default ~/repro-braindecode/cbramod)
  BD_QUEUE         queue directory shared by the braindecode runs (default ~/repro-braindecode/queue)
  BD_CBRAMOD_PY    python of the bd-cbramod env                  (default ~/miniforge3/envs/bd-cbramod/bin/python)
  CBRAMOD_PY       python of the released runs' cbramod env, for the controls (default ~/miniforge3/envs/cbramod/bin/python)
  CBRAMOD_SRC      released CBraMod code                         (default: this repo's methods/cbramod/upstream)
  CBRAMOD_WEIGHTS  released pretrained_weights.pth, for the controls (default ~/CBraMod/pretrained_weights/pretrained_weights.pth)
"""
import os, shlex, sys

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.expanduser(os.environ.get('EEG_DATA_ROOT', '~/data'))
WORK = os.path.expanduser(os.environ.get('BD_CBRAMOD_WORK', '~/repro-braindecode/cbramod'))
QUEUE = os.path.expanduser(os.environ.get('BD_QUEUE', '~/repro-braindecode/queue'))
PY = os.path.expanduser(os.environ.get('BD_CBRAMOD_PY', '~/miniforge3/envs/bd-cbramod/bin/python'))
REL_PY = os.path.expanduser(os.environ.get('CBRAMOD_PY', '~/miniforge3/envs/cbramod/bin/python'))
SRC = os.environ.get('CBRAMOD_SRC', os.path.normpath(os.path.join(HERE, '../../../upstream')))
WEIGHTS = os.path.expanduser(os.environ.get('CBRAMOD_WEIGHTS', '~/CBraMod/pretrained_weights/pretrained_weights.pth'))
# the LMDBs of the released-code runs (../../finetune-public-datasets/scripts/add_jobs.py)
DATA = {
    'PhysioNet-MI': (f'{D}/physio/processed_average', 4),
    'BCIC-IV-2a': (f'{D}/bciciv2a/processed_inde_avg_filter', 4),
    'Mumtaz2016': (f'{D}/mumtaz/processed_lmdb_75hz', 2),
    'MentalArithmetic': (f'{D}/mental_arithmetic/processed', 2),
}
CONTROL = {'onelayer': '', 'onelayer-frozen': ' --frozen True'}   # bool('True') is True
PRIO = {'PhysioNet-MI': 1, 'Mumtaz2016': 2, 'BCIC-IV-2a': 3, 'MentalArithmetic': 4}   # longest jobs first
Q = os.path.join(QUEUE, 'pending')
os.makedirs(Q, exist_ok=True)

ds, setting, seeds = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3:]]
data_dir, ncls = DATA[ds]
q = shlex.quote
for seed in seeds:
    name = f'{ds}__{setting}__s{seed}'
    run_dir = os.path.join(WORK, 'runs', name)
    if setting in CONTROL:    # released finetune_main.py, run from the released code's directory as in its experiment
        prog = (f'cd {q(SRC)}\n{q(REL_PY)} -u finetune_main.py --seed {seed} --cuda 0 --num_of_classes {ncls} '
                f'--classifier all_patch_reps_onelayer --foundation_dir {q(WEIGHTS)}{CONTROL[setting]}')
    else:
        prog = (f'{q(PY)} -u {q(os.path.join(HERE, "run_braindecode.py"))} --setting {setting} --seed {seed} '
                f'--cuda 0')
    cmd = (f'#!/bin/bash\n# job: cbramod {name}\nmkdir -p {q(run_dir)}\nexport CUDA_VISIBLE_DEVICES=$1\n'
           f'{prog} --downstream_dataset {ds} --datasets_dir {q(data_dir)} '
           f'--model_dir {q(run_dir + "/ckpt")} > {q(run_dir + "/log.txt")} 2>&1\n')
    open(os.path.join(Q, f'{PRIO[ds]}_c_{name}.sh'), 'w').write(cmd)
    print('queued', name)
