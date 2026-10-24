"""Queue braindecode-LaBraM fine-tuning runs (code/run_braindecode.py) for tools/run_queue.py.

Recipe = the released-code runs' (../../finetune-cbramod-splits/code/add_jobs.py: LaBraM README's TUAB command on 1 GPU)
  --weight_decay 0.05 --batch_size 64 --lr 5e-4 --update_freq 1 --warmup_epochs 5 --epochs 50
  --layer_decay 0.65 --drop_path 0.1 --disable_rel_pos_bias --abs_pos_emb --disable_qkv_bias --no_save_ckpt
without --finetune: the pre-trained weights are braindecode's (run_braindecode.py).
usage: python code/add_jobs.py <Dataset> <seed> [<seed> ...]

Paths (defaults = the mcl-server layout of the committed runs):
  EEG_DATA_ROOT    CBraMod's preprocessed LMDBs                  (default ~/data)
  BD_LABRAM_WORK   runs/ are written here                        (default ~/repro-braindecode/labram)
  BD_QUEUE         queue directory shared by the braindecode runs (default ~/repro-braindecode/queue)
  LABRAM_SRC       released LaBraM code                          (default: this repo's methods/labram/upstream)
  BD_LABRAM_PY     python of the bd-labram env                   (default ~/miniforge3/envs/bd-labram/bin/python)
"""
import os, shlex, sys

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.expanduser(os.environ.get('EEG_DATA_ROOT', '~/data'))
WORK = os.path.expanduser(os.environ.get('BD_LABRAM_WORK', '~/repro-braindecode/labram'))
QUEUE = os.path.expanduser(os.environ.get('BD_QUEUE', '~/repro-braindecode/queue'))
SRC = os.environ.get('LABRAM_SRC', os.path.normpath(os.path.join(HERE, '../../../upstream')))
PY = os.path.expanduser(os.environ.get('BD_LABRAM_PY', '~/miniforge3/envs/bd-labram/bin/python'))
DATA = {
    'PhysioNet-MI': f'{D}/physio/processed_average',
    'BCIC-IV-2a': f'{D}/bciciv2a/processed_inde_avg_filter',
    'Mumtaz2016': f'{D}/mumtaz/processed_lmdb_75hz',
    'MentalArithmetic': f'{D}/mental_arithmetic/processed',
}
RECIPE = ('--model labram_base_patch200_200 --weight_decay 0.05 --batch_size 64 --lr 5e-4 --update_freq 1 '
          '--warmup_epochs 5 --epochs 50 --layer_decay 0.65 --drop_path 0.1 --disable_rel_pos_bias '
          '--abs_pos_emb --disable_qkv_bias --no_save_ckpt')
PRIO = {'PhysioNet-MI': 1, 'Mumtaz2016': 2, 'BCIC-IV-2a': 3, 'MentalArithmetic': 4}   # longest jobs first
Q = os.path.join(QUEUE, 'pending')
os.makedirs(Q, exist_ok=True)

ds, seeds = sys.argv[1], [int(s) for s in sys.argv[2:]]
q = shlex.quote
for seed in seeds:
    name = f'{ds}__labram__s{seed}'
    run_dir = os.path.join(WORK, 'runs', name)
    # one-process distributed job, as in the released-code runs: LaBraM then reshuffles the training set every epoch
    cmd = (f'#!/bin/bash\n# job: labram {name}\nmkdir -p {q(run_dir)}\ncd {q(SRC)}\n'
           f'export CUDA_VISIBLE_DEVICES=$1 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 '
           f'MASTER_PORT=$((29500 + $1)) OMP_NUM_THREADS=1\n'
           f'{q(PY)} -u {q(os.path.join(HERE, "run_braindecode.py"))} --dataset {ds} --datasets_dir {q(DATA[ds])} '
           f'{RECIPE} --seed {seed} --output_dir {q(run_dir)} > {q(run_dir + "/stdout.txt")} 2>&1\n')
    open(os.path.join(Q, f'{PRIO[ds]}_l_{name}.sh'), 'w').write(cmd)
    print('queued', name)
