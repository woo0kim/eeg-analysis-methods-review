"""Parse the logs of EEGNet-8,2 trained through CBraMod's released pipeline and compare with the CBraMod paper's
EEGNet rows (mean ± std over seeds). Same parsing and results.json schema as
../../../../cbramod/experiments/finetune-public-datasets/scripts/collect.py, where these runs were collected before
2026-10-29 (setting `eegnet`).

usage: python collect.py [RUNS_DIR] [OUT_DIR] > OUT_DIR/results.txt
RUNS_DIR defaults to $CBRAMOD_WORK/runs (~/repro/runs); results.json is written to OUT_DIR (default $CBRAMOD_WORK).
"""
import glob, json, os, re, sys
import numpy as np
from scipy import stats

WORK = os.path.expanduser(os.environ.get('CBRAMOD_WORK', '~/repro'))
RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(WORK, 'runs')
OUT = sys.argv[2] if len(sys.argv) > 2 else WORK
# Paper values (ICLR 2025): mean, std over 5 seeds. Metric order = order printed by the code.
PAPER = {   # EEGNet rows of the CBraMod paper (ICLR 2025): mean, std over 5 seeds; BIOT's code, settings not reported
    ('PhysioNet-MI', 'eegnet'):     ('Table 3', [(0.5814, 0.0125), (0.4468, 0.0199), (0.5796, 0.0115)]),
    ('BCIC-IV-2a', 'eegnet'):       ('Table 15', [(0.4482, 0.0094), (0.2693, 0.0121), (0.4226, 0.0108)]),
    ('Mumtaz2016', 'eegnet'):       ('Table 10', [(0.9232, 0.0104), (0.9626, 0.0095), (0.9639, 0.0093)]),
    ('MentalArithmetic', 'eegnet'): ('Table 12', [(0.6770, 0.0116), (0.5763, 0.0102), (0.7321, 0.0108)]),
}
MC = ['balanced_acc', 'kappa', 'weighted_f1']
BC = ['balanced_acc', 'pr_auc', 'roc_auc']

runs = []
for log in sorted(glob.glob(f'{RUNS}/*/log.txt')):
    name = os.path.basename(os.path.dirname(log))
    ds, setting, seed = name.split('__')
    # Main protocol = seeds 0-4 (5 seeds, as in the paper). Other seeds (e.g. the code default 3407)
    # are reported as separate one-off settings so they don't change the 5-seed statistics.
    if int(seed[1:]) not in range(5):
        setting = f'{setting}@seed{int(seed[1:])}'
    txt = open(log).read()
    m = re.search(r'Test Evaluation: acc: ([\d.]+), (kappa|pr_auc): ([\d.]+), (f1|roc_auc): ([\d.]+)', txt)
    if not m:
        continue
    ep = re.search(r'model save in .*/epoch(\d+)_', txt)
    n_ep = len(re.findall(r'^Epoch \d+ :', txt, re.M))
    vals = [float(m.group(1)), float(m.group(3)), float(m.group(5))]
    runs.append(dict(dataset=ds, setting=setting, seed=int(seed[1:]), metrics=vals,
                     metric_names=MC if m.group(2) == 'kappa' else BC,
                     best_epoch=int(ep.group(1)) if ep else None, epochs_run=n_ep))

summary = []
for key in sorted({(r['dataset'], r['setting']) for r in runs}):
    rs = sorted([r for r in runs if (r['dataset'], r['setting']) == key], key=lambda r: r['seed'])
    arr = np.array([r['metrics'] for r in rs])
    row = dict(dataset=key[0], setting=key[1], n=len(rs), seeds=[r['seed'] for r in rs],
               metric_names=rs[0]['metric_names'], per_seed=arr.tolist(),
               best_epochs=[r['best_epoch'] for r in rs],
               mean=arr.mean(0).tolist(), std=(arr.std(0, ddof=1) if len(rs) > 1 else np.zeros(3)).tolist())
    if key in PAPER:
        tab, pv = PAPER[key]
        row['paper_table'] = tab
        row['paper'] = pv
        row['p_welch'] = [float(stats.ttest_ind_from_stats(mu, sd, len(rs), pm, ps, 5, equal_var=False).pvalue)
                          if len(rs) > 1 else None
                          for mu, sd, (pm, ps) in zip(row['mean'], row['std'], pv)]
    summary.append(row)

json.dump(dict(runs=runs, summary=summary), open(os.path.join(OUT, 'results.json'), 'w'), indent=1)
for row in summary:
    print(f"\n{row['dataset']} [{row['setting']}] n={row['n']} seeds={row['seeds']} best_epochs={row['best_epochs']}")
    for i, mn in enumerate(row['metric_names']):
        s = f"  {mn:13s} ours {row['mean'][i]:.4f} ± {row['std'][i]:.4f}"
        if 'paper' in row:
            pm, ps = row['paper'][i]
            p = row['p_welch'][i]
            s += f" | paper {pm:.4f} ± {ps:.4f} | diff {row['mean'][i] - pm:+.4f}" + (f" | p={p:.3g}" if p is not None else '')
        print(s)
        print('      per-seed:', ' '.join(f'{v[i]:.4f}' for v in row['per_seed']))
