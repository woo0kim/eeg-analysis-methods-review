"""Parse CBraMod fine-tuning logs and compare with the paper (mean ± std over seeds).

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
PAPER = {
    ('PhysioNet-MI', 'pretrained'): ('Table 3', [(0.6417, 0.0091), (0.5222, 0.0169), (0.6427, 0.0100)]),
    ('PhysioNet-MI', 'scratch'):    ('Table 4', [(0.6196, 0.0143), (0.4994, 0.0289), (0.6157, 0.0145)]),
    ('PhysioNet-MI', 'frozen'):     ('Table 18', [(0.3845, 0.0345), (0.2983, 0.0498), (0.3946, 0.0378)]),
    # EEGNet rows of Tables 3/15/10/12 (baseline numbers reported in the paper)
    ('PhysioNet-MI', 'eegnet'):     ('Table 3', [(0.5814, 0.0125), (0.4468, 0.0199), (0.5796, 0.0115)]),
    ('BCIC-IV-2a', 'eegnet'):       ('Table 15', [(0.4482, 0.0094), (0.2693, 0.0121), (0.4226, 0.0108)]),
    ('Mumtaz2016', 'eegnet'):       ('Table 10', [(0.9232, 0.0104), (0.9626, 0.0095), (0.9639, 0.0093)]),
    ('MentalArithmetic', 'eegnet'): ('Table 12', [(0.6770, 0.0116), (0.5763, 0.0102), (0.7321, 0.0108)]),
    ('SHU-MI', 'pretrained'):       ('Table 3', [(0.6370, 0.0151), (0.7139, 0.0088), (0.6988, 0.0068)]),
    ('SHU-MI', 'scratch'):          ('Table 4', [(0.6289, 0.0179), (0.7032, 0.0145), (0.6878, 0.0166)]),
    ('Mumtaz2016', 'pretrained'):   ('Table 10', [(0.9560, 0.0056), (0.9923, 0.0032), (0.9921, 0.0025)]),
    ('MentalArithmetic', 'pretrained'): ('Table 12', [(0.7256, 0.0132), (0.6267, 0.0099), (0.7905, 0.0073)]),
    # Table 15 headers read "AUC-PR / AUROC" but are Cohen's kappa / weighted F1 (author, issue #21)
    ('BCIC-IV-2a', 'pretrained'):   ('Table 15', [(0.5138, 0.0066), (0.3518, 0.0094), (0.4984, 0.0085)]),
}
# Best baseline reported in the paper (LaBraM-Base for all four public datasets)
BASELINE = {
    'PhysioNet-MI': ('LaBraM-Base', [0.6173, 0.4912, 0.6177]),
    'SHU-MI': ('BIOT', [0.6179, 0.6770, 0.6609]),
    'Mumtaz2016': ('LaBraM-Base', [0.9409, 0.9798, 0.9782]),
    'MentalArithmetic': ('LaBraM-Base', [0.6909, 0.5999, 0.7721]),
    'BCIC-IV-2a': ('LaBraM-Base', [0.4869, 0.3159, 0.4758]),
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
    if key[1] == 'pretrained' and key[0] in BASELINE:
        row['paper_best_baseline'] = BASELINE[key[0]]
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
        if 'paper_best_baseline' in row:
            s += f" | {row['paper_best_baseline'][0]} {row['paper_best_baseline'][1][i]:.4f}"
        print(s)
        print('      per-seed:', ' '.join(f'{v[i]:.4f}' for v in row['per_seed']))
