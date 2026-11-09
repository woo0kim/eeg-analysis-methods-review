"""Enqueue SEED / SEED-V runs for tools/run_queue.py (one job = one model x one seed, 50 epochs on one GPU).

usage: python add_jobs.py <SEED|SEED-V> <trial555|subject> <prio> <model> <seed> [<seed> ...]
       models: see emo_models.py, plus labram and bd-labram (LaBraM's released fine-tuning recipe).

Paths (defaults = the mcl-server layout the logged runs used):
  EEG_DATA_ROOT    data root; LMDBs in <dataset>/processed_<split>       (default ~/data)
  SEED_WORK        runs/ and queue/ are written here                     (default ~/repro-seed)
  SEED_PY          python with torch 2.1, braindecode 1.8.1, timm 0.4.12 (default ~/miniforge3/envs/bd-labram/bin/python)
  LABRAM_PY        python of the labram env                              (default ~/miniforge3/envs/labram/bin/python)
  SEED_QUEUE       queue directory                                       (default $SEED_WORK/queue)
  CBRAMOD_WEIGHTS  CBraMod's pretrained_weights.pth   (default ~/CBraMod/pretrained_weights/pretrained_weights.pth)
  NVML_FIX         LD_LIBRARY_PATH entry for the server's NVML mismatch  (default ~/repro-braindecode/nvml-580.173.02/lib)
"""
import os
import shlex
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
E = lambda k, d: os.path.expanduser(os.environ.get(k, d))
D, WORK = E('EEG_DATA_ROOT', '~/data'), E('SEED_WORK', '~/repro-seed')
PY, LPY = E('SEED_PY', '~/miniforge3/envs/bd-labram/bin/python'), E('LABRAM_PY', '~/miniforge3/envs/labram/bin/python')
WEIGHTS = E('CBRAMOD_WEIGHTS', '~/CBraMod/pretrained_weights/pretrained_weights.pth')
NVML = E('NVML_FIX', '~/repro-braindecode/nvml-580.173.02/lib')
METHODS = os.path.normpath(os.path.join(HERE, '../../../..'))
LSRC = os.path.join(METHODS, 'labram/upstream')
RECIPE = ('--model labram_base_patch200_200 --weight_decay 0.05 --batch_size 64 --lr 5e-4 --update_freq 1 '
          '--warmup_epochs 5 --epochs {epochs} --layer_decay 0.65 --drop_path 0.1 --disable_rel_pos_bias '
          '--abs_pos_emb --disable_qkv_bias --no_save_ckpt')
CLASSES = {'SEED': 3, 'SEED-V': 5}
Q = os.path.join(E('SEED_QUEUE', os.path.join(WORK, 'queue')), 'pending')
os.makedirs(Q, exist_ok=True)
q = shlex.quote
EPOCHS = int(os.environ.get('SEED_EPOCHS', 50))         # 50 = released default; fewer only for timing pilots
TAG = os.environ.get('SEED_TAG', '')

ds, split, prio, model, seeds = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], [int(s) for s in sys.argv[5:]]
data = f'{D}/{ds}/processed_{split}'
for seed in seeds:
    name = f'{ds}__{split}__{model}__s{seed}{TAG}'
    run = os.path.join(WORK, 'runs', name)
    head = f'#!/bin/bash\n# job: {name}\nmkdir -p {q(run)}\nexport LD_LIBRARY_PATH={q(NVML)}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}\n'
    if model in ('labram', 'bd-labram'):
        recipe = RECIPE.format(epochs=EPOCHS)
        common = (f'--dataset {ds} --datasets_dir {q(data)} {recipe} --seed {seed} --output_dir {q(run)} '
                  f'> {q(run + "/stdout.txt")} 2>&1\n')
        env = (f'cd {q(LSRC)}\nexport CUDA_VISIBLE_DEVICES=$1 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 '
               'MASTER_PORT=$((20000 + $$ % 20000)) OMP_NUM_THREADS=1\n')
        if model == 'labram':
            cmd = env + (f'{q(LPY)} -u {q(METHODS + "/labram/experiments/seed-cbramod-pipeline/code/run_labram_seed.py")} '
                         f'--finetune {q(LSRC + "/checkpoints/labram-base.pth")} ') + common
        else:
            cmd = env + f'{q(PY)} -u {q(METHODS + "/labram/experiments/braindecode-seed/code/run_bd_labram_seed.py")} ' + common
    else:
        cmd = (f'export CBRAMOD_WEIGHTS={q(WEIGHTS)}\n'
               f'CUDA_VISIBLE_DEVICES=$1 {q(PY)} -u {q(os.path.join(HERE, "run_emo.py"))} --model {model} --seed {seed} '
               f'--datasets_dir {q(data)} --num_of_classes {CLASSES[ds]} --model_dir {q(run)} --epochs {EPOCHS} '
               f'> {q(run + "/log.txt")} 2>&1\n')
    open(os.path.join(Q, f'{prio:02d}_{name}.sh'), 'w').write(head + cmd)
    print('queued', name)
