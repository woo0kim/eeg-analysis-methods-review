"""braindecode's ATCNet vs. the released Keras ATCNet on BCI IV 2a, same data, protocols, seeds and callbacks.

Reads results/curves/<variant>/<protocol>/s<sub>_seed<seed>.npz (code/run_braindecode.py) and the Keras runs of
../../bciciv2a-model-selection/results/curves/<protocol>/ (code/run_released.py there), prints every table of the
README and writes results/results.json. Each braindecode setting is compared with the Keras runs of the same seeds
(paired by subject: per-subject mean over those seeds; two-sided Wilcoxon signed-rank test over the 9 subjects, scipy
defaults). Accuracy in %, kappa = sklearn's cohen_kappa_score of a run, averaged like the released test().
usage: python code/analyze.py [RESULTS_DIR]
"""
import glob, json, os, sys, warnings
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '../results')
KERAS = os.path.join(HERE, '../../bciciv2a-model-selection/results/curves')
SUBJECTS = range(1, 10)
PAPER = {'traintest': 85.38, 'trainvaltest': 85.38, 'loso': 70.97, 'paper2022': 85.38}
VARIANTS = {'default': 'braindecode default', 'faithful': 'braindecode "faithful" options',
            'keras': 'control: braindecode + current Keras training behaviour',
            'keras2022': 'control: braindecode + 2022 Keras training behaviour'}
ABLATION = {'dropout': 'element-wise dropout in the conv block', 'attn': 'attention dropout on the attention weights',
            'bn': 'Keras BatchNorm (eps 1e-3, momentum 0.99)', 'init': 'Keras initialisers',
            'dense2022': 'final max-norm 0.25 per output unit, Keras projection',
            'denserows': "final max-norm 0.25 per output unit, braindecode's parametrization",
            'dw2022': 'depthwise max-norm 1.0'}
VARIANTS.update({f'default+{k}': f'defaults + {v}' for k, v in ABLATION.items()})
PROTOCOLS = {'paper2022': 'the 2022 code behind the paper (session E as validation data, lr 0.0009, no schedule)',
             'traintest': 'main_TrainTest.py (session E as validation data)',
             'trainvaltest': 'main_TrainValTest.py (20% of session T as validation data)',
             'loso': 'LOSO, main_TrainTest.py (held-out subject as validation data)'}


def load(path):
    runs = {}
    for f in glob.glob(os.path.join(path, 's*_seed*.npz')):
        d = dict(np.load(f))
        runs[(int(d['subject']), int(d['seed']))] = d
    return runs


K = {p: load(os.path.join(KERAS, p)) for p in PROTOCOLS}
B = {(v, p): load(os.path.join(RESULTS, 'curves', v, p)) for v in VARIANTS for p in PROTOCOLS}
B = {k: r for k, r in B.items() if r}


def seeds_of(runs):
    """Seeds present for all 9 subjects."""
    return sorted(sd for sd in {k[1] for k in runs} if all((s, sd) in runs for s in SUBJECTS))


def grid(runs, seeds, key='acc', scale=100):
    return np.array([[float(runs[(s, sd)][key]) * scale for sd in seeds] for s in SUBJECTS])


def wilcoxon(a, b):
    if np.allclose(a, b):
        return float('nan')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return float(stats.wilcoxon(a, b).pvalue)


def best_of(acc, kap):
    i = np.argmax(acc, axis=1)
    return acc[np.arange(9), i], kap[np.arange(9), i]


out = {'n_runs': {f'{v}/{p}': len(r) for (v, p), r in B.items()}, 'keras_n_runs': {p: len(r) for p, r in K.items()},
       'comparisons': {}}
for (v, p), r in sorted(B.items()):
    print(f'{VARIANTS[v]} / {p}: {len(r)} runs, seeds {seeds_of(r)}')

print('\nbraindecode vs. Keras, same protocol and seeds (mean over the 9 subjects of the per-subject mean over seeds)\n')
print('| Protocol | Setting | Seeds | Rule | Keras acc. (κ) | braindecode acc. (κ) | braindecode − Keras | '
      'Subjects higher | Wilcoxon p | braindecode − paper |')
print('|---|---|---|---|---|---|---|---|---|---|')
for p in PROTOCOLS:
    for v in ('default', 'faithful', 'keras', 'keras2022'):
        if (v, p) not in B:
            continue
        seeds = [sd for sd in seeds_of(B[(v, p)]) if sd in seeds_of(K[p])]
        if not seeds:
            continue
        ka, kk = grid(K[p], seeds), grid(K[p], seeds, 'kappa', 1)
        ba, bk = grid(B[(v, p)], seeds), grid(B[(v, p)], seeds, 'kappa', 1)
        rules = [('mean of runs', ka.mean(1), kk.mean(1), ba.mean(1), bk.mean(1))]
        if p != 'trainvaltest':
            rules.append((f'best of {len(seeds)} runs (released test())', *best_of(ka, kk), *best_of(ba, bk)))
        for rule, a, ak, b, bkap in rules:
            rec = {'seeds': seeds, 'keras_acc': round(float(a.mean()), 2), 'keras_kappa': round(float(ak.mean()), 3),
                   'bd_acc': round(float(b.mean()), 2), 'bd_kappa': round(float(bkap.mean()), 3),
                   'bd_minus_keras': round(float((b - a).mean()), 2), 'subjects_bd_higher': int((b > a).sum()),
                   'wilcoxon_p': round(wilcoxon(b, a), 4), 'bd_minus_paper': round(float(b.mean() - PAPER[p]), 2),
                   'keras_per_subject': np.round(a, 2).tolist(), 'bd_per_subject': np.round(b, 2).tolist()}
            if rule == 'mean of runs':
                rec['bd_sd_over_seed_means'] = round(float(ba.mean(0).std(ddof=1)), 2) if len(seeds) > 1 else None
                rec['keras_sd_over_seed_means'] = round(float(ka.mean(0).std(ddof=1)), 2) if len(seeds) > 1 else None
            out['comparisons'][f'{v}/{p}/{rule}'] = rec
            print(f"| {p} | {VARIANTS[v]} | {seeds[0]}–{seeds[-1]} | {rule} | {rec['keras_acc']:.2f} ({rec['keras_kappa']:.3f}) | "
                  f"{rec['bd_acc']:.2f} ({rec['bd_kappa']:.3f}) | {rec['bd_minus_keras']:+.2f} | "
                  f"{rec['subjects_bd_higher']} / 9 | {rec['wilcoxon_p']:.4f} | {rec['bd_minus_paper']:+.2f} |")

print('\nPer subject, braindecode default vs. Keras, mean ± SD over seeds (accuracy, %)\n')
cols = [p for p in PROTOCOLS if ('default', p) in B and set(seeds_of(B[('default', p)])) & set(seeds_of(K[p]))]
print('| Subject | ' + ' | '.join(f'{p}: Keras | {p}: braindecode' for p in cols) + ' |')
print('|---|' + '---|---|' * len(cols))
pm = lambda g, i: f'{g[i].mean():.2f} ± {g[i].std(ddof=1):.2f}' if g.shape[1] > 1 else f'{g[i].mean():.2f}'
G = {}
for p in cols:
    seeds = [sd for sd in seeds_of(B[('default', p)]) if sd in seeds_of(K[p])]
    G[p] = (grid(K[p], seeds), grid(B[('default', p)], seeds))
for i, s in enumerate(SUBJECTS):
    print(f'| {s} | ' + ' | '.join(f'{pm(G[p][0], i)} | {pm(G[p][1], i)}' for p in cols) + ' |')
print('| **Avg** | ' + ' | '.join(f'**{G[p][0].mean():.2f}** | **{G[p][1].mean():.2f}**' for p in cols) + ' |')

print('\nTraining behaviour, per run (mean ± SD over runs)\n')
def per_run(runs, fn):
    return np.array([fn(d) for d in runs.values()], float)
qs = {'epochs trained': lambda d: float(d['epochs_run']),
      'epoch of the checkpoint (1-based)': lambda d: float(np.argmin(d['val_loss']) + 1) if str(d['protocol']) == 'trainvaltest'
      else float(np.argmax(d['val_accuracy']) + 1),
      'training accuracy, mean of the last 50 epochs (%)': lambda d: float(np.mean(d['accuracy'][-50:]) * 100),
      'test accuracy at the last epoch (%)': lambda d: float((d['test_acc'] if len(d['test_acc']) else d['val_accuracy'])[-1] * 100),
      'checkpoint − last epoch (pp)': lambda d: float(d['acc'] * 100 - (d['test_acc'] if len(d['test_acc']) else d['val_accuracy'])[-1] * 100),
      'minutes per run': lambda d: float(d['minutes'])}
out['training'] = {}
for p in PROTOCOLS:
    sets = [('Keras', K[p])] + [(VARIANTS[v], B[(v, p)]) for v in ('default', 'faithful', 'keras', 'keras2022')
                                if (v, p) in B]
    if not K[p]:
        continue
    print(f'{p}:\n')
    print('| Quantity | ' + ' | '.join(n for n, _ in sets) + ' |')
    print('|---|' + '---|' * len(sets))
    for qn, fn in qs.items():
        cells = []
        for n, r in sets:
            v = per_run(r, fn)
            out['training'].setdefault(p, {}).setdefault(qn, {})[n] = [round(float(v.mean()), 2), round(float(v.std(ddof=1)), 2)] if len(v) > 1 else None
            cells.append(f'{v.mean():.2f} ± {v.std(ddof=1):.2f}' if len(v) > 1 else '–')
        print(f'| {qn} | ' + ' | '.join(cells) + ' |')
    print()

if ('default', 'paper2022') in B:
    print('Ablation in the 2022 loop: braindecode defaults with one Keras behaviour switched on (seeds of the switch; '
          'per-subject means; Wilcoxon over the 9 subjects)\n')
    print('| Setting | Seeds | Accuracy | − braindecode defaults (p) | − Keras 2022 code (p) |')
    print('|---|---|---|---|---|')
    out['ablation'] = {}
    for v in ['default'] + [f'default+{k}' for k in ABLATION] + ['keras2022']:
        if (v, 'paper2022') not in B:
            continue
        seeds = [sd for sd in seeds_of(B[(v, 'paper2022')]) if sd in seeds_of(B[('default', 'paper2022')])
                 and sd in seeds_of(K['paper2022'])]
        if not seeds:
            continue
        x = grid(B[(v, 'paper2022')], seeds).mean(1)
        d = grid(B[('default', 'paper2022')], seeds).mean(1)
        k = grid(K['paper2022'], seeds).mean(1)
        rec = {'seeds': seeds, 'acc': round(float(x.mean()), 2), 'minus_default': round(float((x - d).mean()), 2),
               'p_default': round(wilcoxon(x, d), 4), 'minus_keras': round(float((x - k).mean()), 2),
               'p_keras': round(wilcoxon(x, k), 4)}
        out['ablation'][v] = rec
        print(f"| {VARIANTS[v]} | {seeds[0]}–{seeds[-1]} | {rec['acc']:.2f} | {rec['minus_default']:+.2f} "
              f"({rec['p_default']:.4f}) | {rec['minus_keras']:+.2f} ({rec['p_keras']:.4f}) |")
    print()

with open(os.path.join(RESULTS, 'results.json'), 'w') as f:
    json.dump(out, f, indent=1)
