"""Parse the braindecode-EEGNet logs; compare with the CBraMod paper's EEGNet rows and with our PyTorch EEGNet.

Each logs/runs/<dataset>__eegnet__s<seed>/log.txt is the console output of code/run_eegnet.py, which prints CBraMod's
released Trainer's lines. "PyTorch EEGNet" is our own EEGNet-8,2 run through the same pipeline, setting `eegnet` of
../../../cbramod/experiments/finetune-public-datasets/results/results.json (same datasets, seeds, Trainer).
usage: python code/collect.py [RUNS_DIR] [OUT_DIR] > OUT_DIR/results.txt
"""
import glob, json, os, re, sys
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '../logs/runs')
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, '../results')
OURS_PT = os.path.join(HERE, '../../../../cbramod/experiments/finetune-public-datasets/results/results.json')
# EEGNet rows of the CBraMod paper (ICLR 2025; BIOT's code, settings not reported), in the order the code prints
PAPER = {
    'PhysioNet-MI': ('Table 3', [(0.5814, 0.0125), (0.4468, 0.0199), (0.5796, 0.0115)]),
    'BCIC-IV-2a': ('Table 15', [(0.4482, 0.0094), (0.2693, 0.0121), (0.4226, 0.0108)]),
    'Mumtaz2016': ('Table 10', [(0.9232, 0.0104), (0.9626, 0.0095), (0.9639, 0.0093)]),
    'MentalArithmetic': ('Table 12', [(0.6770, 0.0116), (0.5763, 0.0102), (0.7321, 0.0108)]),
}
MC = ['balanced_acc', 'kappa', 'weighted_f1']
BC = ['balanced_acc', 'pr_auc', 'roc_auc']


def welch(m1, s1, n1, m2, s2, n2):
    return float(stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False).pvalue)


runs = []
for log in sorted(glob.glob(f'{RUNS}/*__eegnet__s*/log.txt')):
    ds, _, seed = os.path.basename(os.path.dirname(log)).split('__')
    txt = open(log).read()
    m = re.search(r'Test Evaluation: acc: ([\d.]+), (kappa|pr_auc): ([\d.]+), (f1|roc_auc): ([\d.]+)', txt)
    if not m:
        continue
    ep = re.search(r'model save in .*/epoch(\d+)_', txt)
    n_par = re.search(r'params: (\d+)', txt)
    runs.append(dict(dataset=ds, setting='eegnet-braindecode', seed=int(seed[1:]),
                     metrics=[float(m.group(1)), float(m.group(3)), float(m.group(5))],
                     metric_names=MC if m.group(2) == 'kappa' else BC, best_epoch=int(ep.group(1)) if ep else None,
                     epochs_run=len(re.findall(r'^Epoch \d+ :', txt, re.M)),
                     n_params=int(n_par.group(1)) if n_par else None))

ours_pt = {s['dataset']: s for s in json.load(open(OURS_PT))['summary'] if s['setting'] == 'eegnet'}
summary = []
for ds in sorted({r['dataset'] for r in runs}):
    rs = sorted([r for r in runs if r['dataset'] == ds], key=lambda r: r['seed'])
    arr = np.array([r['metrics'] for r in rs])
    n = len(rs)
    row = dict(dataset=ds, setting='eegnet-braindecode', n=n, seeds=[r['seed'] for r in rs],
               metric_names=rs[0]['metric_names'], per_seed=arr.tolist(), best_epochs=[r['best_epoch'] for r in rs],
               n_params=rs[0]['n_params'], mean=arr.mean(0).tolist(),
               std=(arr.std(0, ddof=1) if n > 1 else np.zeros(3)).tolist())
    row['paper_table'], row['paper'] = PAPER[ds]
    row['p_welch_vs_paper'] = [welch(mu, sd, n, pm, ps, 5) if n > 1 else None
                               for mu, sd, (pm, ps) in zip(row['mean'], row['std'], row['paper'])]
    pt = ours_pt.get(ds)
    if pt:
        row['pytorch_eegnet'] = [list(x) for x in zip(pt['mean'], pt['std'])]
        row['p_welch_vs_pytorch_eegnet'] = [welch(mu, sd, n, m2, s2, pt['n']) if n > 1 else None
                                            for mu, sd, m2, s2 in zip(row['mean'], row['std'], pt['mean'], pt['std'])]
    summary.append(row)

os.makedirs(OUT, exist_ok=True)
json.dump(dict(runs=runs, summary=summary), open(os.path.join(OUT, 'results.json'), 'w'), indent=1)
for row in summary:
    print(f"\n{row['dataset']} n={row['n']} seeds={row['seeds']} best_epochs={row['best_epochs']} params={row['n_params']}")
    for i, mn in enumerate(row['metric_names']):
        s = f"  {mn:13s} braindecode {row['mean'][i]:.4f} ± {row['std'][i]:.4f}"
        if 'pytorch_eegnet' in row:
            m2, s2 = row['pytorch_eegnet'][i]
            p = row['p_welch_vs_pytorch_eegnet'][i]
            s += f" | our PyTorch EEGNet {m2:.4f} ± {s2:.4f} | diff {row['mean'][i] - m2:+.4f}" + \
                 (f" | p={p:.3g}" if p is not None else '')
        pm, ps = row['paper'][i]
        p = row['p_welch_vs_paper'][i]
        s += f" || paper {pm:.4f} ± {ps:.4f} | diff {row['mean'][i] - pm:+.4f}" + (f" | p={p:.3g}" if p is not None else '')
        print(s)
        print('      per-seed:', ' '.join(f'{v[i]:.4f}' for v in row['per_seed']))
