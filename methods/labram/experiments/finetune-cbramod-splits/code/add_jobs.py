"""Queue LaBraM-Base fine-tuning runs for tools/run_queue.py.

Recipe = the fine-tuning command in LaBraM's README (TUAB example), run on 1 GPU instead of 8:
  --weight_decay 0.05 --batch_size 64 --lr 5e-4 --update_freq 1 --warmup_epochs 5 --epochs 50
  --layer_decay 0.65 --drop_path 0.1 --disable_rel_pos_bias --abs_pos_emb --disable_qkv_bias
plus --no_save_ckpt (every epoch's validation/test predictions are saved instead of checkpoints).
usage: python add_jobs.py <Dataset> <seed> [<seed> ...]

Paths (defaults = the mcl-server layout used for the CBraMod runs):
  EEG_DATA_ROOT  CBraMod's preprocessed LMDBs  (default ~/data)
  LABRAM_WORK    runs/ and queue/ are written here (default ~/repro-labram)
  LABRAM_SRC     released LaBraM code          (default: this repo's methods/labram/upstream)
  LABRAM_PY      python of the labram env      (default ~/miniforge3/envs/labram/bin/python)
"""
import os, shlex, sys

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.expanduser(os.environ.get('EEG_DATA_ROOT', '~/data'))
WORK = os.path.expanduser(os.environ.get('LABRAM_WORK', '~/repro-labram'))
SRC = os.environ.get('LABRAM_SRC', os.path.normpath(os.path.join(HERE, '../../../upstream')))
PY = os.path.expanduser(os.environ.get('LABRAM_PY', '~/miniforge3/envs/labram/bin/python'))
# the LMDBs CBraMod was fine-tuned on (methods/cbramod/experiments/finetune-public-datasets/scripts/add_jobs.py)
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
Q = os.path.join(WORK, 'queue/pending')
os.makedirs(Q, exist_ok=True)

ds, seeds = sys.argv[1], [int(s) for s in sys.argv[2:]]
q = shlex.quote
for seed in seeds:
    name = f'{ds}__labram__s{seed}'
    run_dir = os.path.join(WORK, 'runs', name)
    # one-process distributed job: LaBraM then calls set_epoch() and reshuffles every epoch, as with torchrun
    cmd = (f'#!/bin/bash\n# job: {name}\nmkdir -p {q(run_dir)}\ncd {q(SRC)}\n'
           f'export CUDA_VISIBLE_DEVICES=$1 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 '
           f'MASTER_PORT=$((29500 + $1)) OMP_NUM_THREADS=1\n'
           f'{q(PY)} -u {q(os.path.join(HERE, "run_labram.py"))} --dataset {ds} --datasets_dir {q(DATA[ds])} '
           f'--finetune {q(os.path.join(SRC, "checkpoints/labram-base.pth"))} {RECIPE} --seed {seed} '
           f'--output_dir {q(run_dir)} > {q(run_dir + "/stdout.txt")} 2>&1\n')
    open(os.path.join(Q, f'{PRIO[ds]}_l_{name}.sh'), 'w').write(cmd)
    print('queued', name)
