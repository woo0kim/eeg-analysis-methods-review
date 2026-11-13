"""Enqueue the networks' runs in the dDTF-GL paper's STEW settings for tools/run_queue.py (one GPU per job).

  python add_jobs6.py <model> [<model> ...]       models: eegnet conformer bd-atcnet bd-deepsleepnet dgcnn cbramod labram
                                                  every model: si and sd x folds 0-6 x seeds 0-4
Paths: P6_WORK (windows6.npz, runs/, queue/; default ~/repro-seed/stew6), SEED_PY (torch 2.1, braindecode 1.8.1), LABRAM_PY, CBRAMOD_WEIGHTS,
as in methods/cbramod/experiments/finetune-stew/code/add_jobs.py.
"""
import os
import shlex
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
E = lambda k, d: os.path.expanduser(os.environ.get(k, d))
WORK = E('P6_WORK', '~/repro-seed/stew6')
PY, LPY = E('SEED_PY', '~/miniforge3/envs/bd-labram/bin/python'), E('LABRAM_PY', '~/miniforge3/envs/labram/bin/python')
WEIGHTS = E('CBRAMOD_WEIGHTS', '~/CBraMod/pretrained_weights/pretrained_weights.pth')
METHODS = os.path.normpath(os.path.join(HERE, '../../../..'))
LSRC = os.path.join(METHODS, 'labram/upstream')
Q = os.path.join(WORK, 'queue', 'pending')
RECIPE = ('--model labram_base_patch200_200 --weight_decay 0.05 --batch_size 64 --lr 5e-4 --update_freq 1 --warmup_epochs 5 --epochs 50 '
          '--layer_decay 0.65 --drop_path 0.1 --disable_rel_pos_bias --abs_pos_emb --disable_qkv_bias --no_save_ckpt')   # stew-cbramod-pipeline's
os.makedirs(Q, exist_ok=True)
q = shlex.quote
PRIO = {'eegnet': 1, 'dgcnn': 1, 'conformer': 2, 'bd-deepsleepnet': 2, 'cbramod': 3, 'labram': 3, 'bd-atcnet': 4}
n = 0
for model in sys.argv[1:]:
    for protocol in ('si', 'sd'):
        for seed in range(5):
            for fold in range(7):
                name = f'{protocol}__fold{fold}__{model}__s{seed}'
                run = os.path.join(WORK, 'runs', name)
                head = f'#!/bin/bash\n# job: {name}\nmkdir -p {q(run)}\nexport OMP_NUM_THREADS=2 MKL_NUM_THREADS=2\n'
                if model == 'labram':
                    cmd = (f'cd {q(LSRC)}\nexport CUDA_VISIBLE_DEVICES=$1 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 '
                           f'MASTER_PORT=$((20000 + $$ % 20000)) OMP_NUM_THREADS=1\n'
                           f'{q(LPY)} -u {q(os.path.join(HERE, "run_labram6.py"))} --p6_protocol {protocol} --p6_fold {fold} --p6_seed {seed} '
                           f'--p6_root {q(WORK)} --dataset STEW --finetune {q(LSRC + "/checkpoints/labram-base.pth")} {RECIPE} --seed {seed} '
                           f'--output_dir {q(run)} > {q(run + "/stdout.txt")} 2>&1\n')
                else:
                    cmd = (f'export CBRAMOD_WEIGHTS={q(WEIGHTS)}\n'
                           f'CUDA_VISIBLE_DEVICES=$1 {q(PY)} -u {q(os.path.join(HERE, "nets6.py"))} run --model {model} --protocol {protocol} '
                           f'--fold {fold} --seed {seed} --root {q(WORK)} --out {q(run)} > {q(run + "/log.txt")} 2>&1\n')
                open(os.path.join(Q, f'{PRIO[model]}{"0" if protocol == "si" else "1"}_{name}.sh'), 'w').write(head + cmd)
                n += 1
print('queued', n)
