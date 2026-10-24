"""Queue the braindecode-ATCNet runs for tools/run_queue.py: 9 subjects (or LOSO folds) x seeds per setting.

usage: python code/add_jobs.py [VARIANT:PROTOCOL[:SEEDS] ...]
       default = the committed runs, in queue order:
         default:paper2022 (seeds 1-10), keras2022:paper2022 (1-3), default+<switch>:paper2022 for the 6 switches
         of the ablation (seed 1; dense2022 and denserows, added after the first ablation results, seeds 1-2),
         default:loso (seed 1; ~2 h per run, so queued early), default:trainvaltest (1-5), default:traintest and
         keras:traintest (1-5), faithful:traintest, faithful:trainvaltest and keras:trainvaltest (1-3)

Paths (defaults = the mcl-server layout of the committed runs):
  ATCNET_WORK      outputs: braindecode/<variant>/<protocol>/curves, braindecode/logs   (default ~/repro-atcnet)
  ATCNET_QUEUE     queue directory shared with the released-code runs                  (default ~/repro-atcnet/queue)
  ATCNET_BD_PY     python of the braindecode env (env/setup_env.sh)        (default ~/miniforge3/envs/bd-cbramod/bin/python)
  ATCNET_NVML_LIB  LD_LIBRARY_PATH entry for mcl-server's NVML mismatch, if it exists
                   (default ~/repro-braindecode/nvml-580.173.02/lib)
"""
import os, shlex, sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.expanduser(os.environ.get('ATCNET_WORK', '~/repro-atcnet'))
QUEUE = os.path.expanduser(os.environ.get('ATCNET_QUEUE', '~/repro-atcnet/queue'))
PY = os.path.expanduser(os.environ.get('ATCNET_BD_PY', '~/miniforge3/envs/bd-cbramod/bin/python'))
NVML = os.path.expanduser(os.environ.get('ATCNET_NVML_LIB', '~/repro-braindecode/nvml-580.173.02/lib'))
ABLATION = ['dropout', 'attn', 'bn', 'init', 'dense2022', 'dw2022']
PLAN = [('default:paper2022:1-10', '3_a'), ('keras2022:paper2022:1-3', '3_b')] + \
       [(f'default+{s}:paper2022:1-{2 if s == "dense2022" else 1}', '3_c') for s in ABLATION] + \
       [('default+denserows:paper2022:1-2', '3_c')] + \
       [('default:trainvaltest:1-5', '3_d'), ('default:traintest:1-5', '3_e'), ('keras:traintest:1-5', '3_e'),
        ('faithful:traintest:1-3', '3_f'), ('faithful:trainvaltest:1-3', '3_f'), ('keras:trainvaltest:1-3', '3_f'),
        ('default:loso:1-1', '3_ca')]                 # queue order after the Keras runs (1_*, 2_*)
PRIO = {spec.rsplit(':', 1)[0]: prio for spec, prio in PLAN}
LOGS = os.path.join(WORK, 'braindecode', 'logs')
for d in (os.path.join(QUEUE, 'pending'), LOGS):
    os.makedirs(d, exist_ok=True)

q = shlex.quote
ld = f'export LD_LIBRARY_PATH={q(NVML)}\n' if os.path.isdir(NVML) else ''
for setting in sys.argv[1:] or [spec for spec, _ in PLAN]:
    variant, protocol, rng = (setting + ':1-10').split(':')[:3]
    lo, hi = map(int, rng.split('-'))
    for seed in range(lo, hi + 1):
        for sub in range(1, 10):
            name = f'{variant}_{protocol}_s{sub}_seed{seed:02d}'
            prio = PRIO.get(f'{variant}:{protocol}', '3_z')
            cmd = (f'#!/bin/bash\n# job: atcnet braindecode {name}\nexport CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=2\n{ld}'
                   f'{q(PY)} -u {q(os.path.join(HERE, "run_braindecode.py"))} --variant {variant} --protocol {protocol} '
                   f'--subject {sub} --seed {seed} --out {q(WORK + "/braindecode")} > {q(f"{LOGS}/{name}.txt")} 2>&1\n')
            open(os.path.join(QUEUE, 'pending', f'{prio}_{variant}_{protocol}_seed{seed:02d}_s{sub}.sh'), 'w').write(cmd)
            print('queued', name)
