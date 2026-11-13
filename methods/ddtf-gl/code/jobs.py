"""The runs of each dDTF-GL experiment, one per line: <run name> <run_gl.py arguments>.   python jobs.py <experiment>

Run names follow the convention of the pipeline each experiment joins, so the runs sit next to the other models' runs.
Seeds 0-4 everywhere (XGBoost's row and column subsampling in AXB and the classifier; DFT and LNT are deterministic).
"""
import sys

SEEDS = range(5)
EXPERIMENTS = {}


def exp(name):
    def wrap(fn):
        EXPERIMENTS[name] = fn
        return fn
    return wrap


@exp('stew-cbramod-pipeline')
def stew():
    for variant in ('ddtf-gl', 'ddtf-gl-noaxb', 'dwt-gl', 'dwt-gl-axb', 'ddtf-gl-v0', 'ddtf-gl-noaxb-v0'):
        for task in ('rest-task', 'rating3'):
            for s in SEEDS:
                for k in range(7):
                    yield f'{task}__fold{k}__{variant}__s{s}', f'--dataset STEW --protocol cbramod --task {task} --fold {k} --seed {s} --variant {variant}'


@exp('stew-paper-protocol')
def stew6():
    for variant in ('safari-gl', 'ddtf-gl', 'ddtf-gl-v0', 'safari-gl-gpu'):   # Safari et al.'s maps; ours (calibrated, first); GPU
        for protocol in ('si', 'sd', 'sd-blocked'):
            for s in SEEDS:
                for k in range(7):
                    yield f'{protocol}__fold{k}__{variant}__s{s}', f'--dataset STEW-6s --protocol {protocol} --fold {k} --seed {s} --variant {variant}'
    # controls on the bundle's own 4-s windows and folds (training on six folds): raw (as the bundle) or high-passed at 0.3 Hz (shared windows)
    for ds, tag, variants in (('STEW-raw', 'si4raw', ('dwt-gl', 'dc-gl', 'ddtf-gl', 'ddtf-gl-v0')), ('STEW', 'si4hp', ('dwt-gl', 'dc-gl', 'ddtf-gl', 'ddtf-gl-v0'))):
        for v in variants:
            for s in SEEDS:
                for k in range(7):
                    yield f'{tag}__fold{k}__{v}__s{s}', f'--dataset {ds} --protocol si --fold {k} --seed {s} --variant {v}'


@exp('cbramod-splits')
def splits():
    for v in ('ddtf-gl', 'ddtf-gl-v0'):
        for ds in ('MentalArithmetic', 'Mumtaz2016', 'BCIC-IV-2a', 'PhysioNet-MI'):
            for s in SEEDS:
                yield f'{ds}__{v}__s{s}', f'--dataset {ds} --protocol cbramod-split --seed {s} --variant {v}'


@exp('sleepedf-cbramod-pipeline')
def sleep():
    for v in ('ddtf-gl', 'ddtf-gl-v0'):
        for s in SEEDS:
            yield f'SleepEDF__{v}__s{s}', f'--dataset SleepEDF --protocol split --seed {s} --variant {v}'


@exp('seed-cbramod-pipeline')
def seed():
    for ds in ('SEED', 'SEED-V'):
        for split in ('trial555', 'subject'):
            for s in SEEDS:
                yield f'{ds}__{split}__ddtf-gl__s{s}', f'--dataset {ds} --protocol {split} --seed {s} --variant ddtf-gl'
    # the first map settings (ablation): the runs made before the switch to the calibrated maps, not completed afterwards
    for ds, split, seeds in (('SEED', 'trial555', range(5)), ('SEED', 'subject', range(4)), ('SEED-V', 'subject', range(5))):
        for s in seeds:
            yield f'{ds}__{split}__ddtf-gl-v0__s{s}', f'--dataset {ds} --protocol {split} --seed {s} --variant ddtf-gl-v0'


if __name__ == '__main__':
    for name, args in EXPERIMENTS[sys.argv[1]]():
        print(name, args)
