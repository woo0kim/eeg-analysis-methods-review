"""Enqueue STEW runs for tools/run_queue.py: python add_jobs.py <task> <model> <seed> [<seed> ...]   (all 7 folds)

Paths: STEW_WORK (windows.npz, runs/ and queue/; default ~/repro-seed/stew), STEW_QUEUE (default $STEW_WORK/queue), SEED_PY (python with torch
2.1, braindecode 1.8.1, timm 0.4.12), LABRAM_PY, CBRAMOD_WEIGHTS, NVML_FIX, as in ../../finetune-seed/code/add_jobs.py.
"""
import os
import shlex
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
E = lambda k, d: os.path.expanduser(os.environ.get(k, d))
WORK = E('STEW_WORK', '~/repro-seed/stew')
PY, LPY = E('SEED_PY', '~/miniforge3/envs/bd-labram/bin/python'), E('LABRAM_PY', '~/miniforge3/envs/labram/bin/python')
WEIGHTS = E('CBRAMOD_WEIGHTS', '~/CBraMod/pretrained_weights/pretrained_weights.pth')
NVML = E('NVML_FIX', '~/repro-braindecode/nvml-580.173.02/lib')
METHODS = os.path.normpath(os.path.join(HERE, '../../../..'))
LSRC = os.path.join(METHODS, 'labram/upstream')
FOLDS = os.path.normpath(os.path.join(HERE, '../results/folds.json'))
Q = os.path.join(E('STEW_QUEUE', os.path.join(WORK, 'queue')), 'pending')
EPOCHS = int(os.environ.get('STEW_EPOCHS', 50))
TAG = os.environ.get('STEW_TAG', '')
RECIPE = ('--model labram_base_patch200_200 --weight_decay 0.05 --batch_size 64 --lr 5e-4 --update_freq 1 --warmup_epochs {w} --epochs {e} '
          '--layer_decay 0.65 --drop_path 0.1 --disable_rel_pos_bias --abs_pos_emb --disable_qkv_bias --no_save_ckpt')
os.makedirs(Q, exist_ok=True)
q = shlex.quote
task, model, seeds = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3:]]
PRIO = {'eegnet': 1, 'dgcnn': 1, 'bd-eegnet': 1, 'bd-dgcnn': 1, 'bd-dgcnn-de': 1, 'bd-deepsleepnet': 2, 'conformer': 2, 'bd-conformer': 2,
        'cbramod': 3, 'labram': 3, 'cbramod-scratch': 4, 'bd-cbramod': 4, 'bd-labram': 4, 'bd-atcnet': 5}
for seed in seeds:
    for fold in range(7):
        name = f'{task}__fold{fold}__{model}__s{seed}{TAG}'
        run = os.path.join(WORK, 'runs', name)
        head = (f'#!/bin/bash\n# job: {name}\nmkdir -p {q(run)}\n'      # 2 threads: PyTorch's default of one per core swamps a shared server
                f'export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH={q(NVML)}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}\n')
        if model in ('labram', 'bd-labram'):
            rec = RECIPE.format(e=EPOCHS, w=min(5, EPOCHS - 1))
            bd = model == 'bd-labram'
            cmd = (f'cd {q(LSRC)}\nexport CUDA_VISIBLE_DEVICES=$1 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 '
                   f'MASTER_PORT=$((20000 + $$ % 20000)) OMP_NUM_THREADS=1 LABRAM_BD={int(bd)}\n'
                   f'{q(PY if bd else LPY)} -u {q(METHODS + "/labram/experiments/stew-cbramod-pipeline/code/run_labram_stew.py")} '
                   f'--stew_task {task} --stew_fold {fold} --stew_root {q(WORK)} --stew_folds {q(FOLDS)} --dataset STEW '
                   + ('' if bd else f'--finetune {q(LSRC + "/checkpoints/labram-base.pth")} ') +
                   f'{rec} --seed {seed} --output_dir {q(run)} > {q(run + "/stdout.txt")} 2>&1\n')
        else:
            cmd = (f'export CBRAMOD_WEIGHTS={q(WEIGHTS)}\n'
                   f'CUDA_VISIBLE_DEVICES=$1 {q(PY)} -u {q(os.path.join(HERE, "run_stew.py"))} --model {model} --task {task} --fold {fold} '
                   f'--seed {seed} --root {q(WORK)} --folds {q(FOLDS)} --model_dir {q(run)} --epochs {EPOCHS} > {q(run + "/log.txt")} 2>&1\n')
        open(os.path.join(Q, f'{PRIO[model]}{"0" if task == "rest-task" else "1"}_{name}.sh'), 'w').write(head + cmd)
print('queued', task, model, len(seeds) * 7)
