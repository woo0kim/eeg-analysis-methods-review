"""Parse the braindecode-CBraMod fine-tuning logs; compare with the paper and with the released-code runs.

Each logs/runs/<dataset>__<setting>__s<seed>/log.txt is the console output of code/run_braindecode.py, which prints the
released Trainer's lines, so the test metrics are read as ../../finetune-public-datasets/scripts/collect.py reads
them. The released-code numbers come from that experiment's results/results.json (same datasets, settings, seeds); the
controls `onelayer` / `onelayer-frozen` (released model, one-layer head) are compared with its `pretrained` / `frozen`
runs (3-layer head).
usage: python code/collect.py [RUNS_DIR] [OUT_DIR] > OUT_DIR/results.txt
"""
import glob, json, os, re, sys
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '../logs/runs')
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, '../results')
RELEASED = os.path.join(HERE, '../../finetune-public-datasets/results/results.json')
# Paper values (ICLR 2025): mean, std over 5 seeds, in the order the code prints the metrics
PAPER = {
    ('PhysioNet-MI', 'pretrained'): ('Table 3', [(0.6417, 0.0091), (0.5222, 0.0169), (0.6427, 0.0100)]),
    ('PhysioNet-MI', 'scratch'): ('Table 4', [(0.6196, 0.0143), (0.4994, 0.0289), (0.6157, 0.0145)]),
    ('PhysioNet-MI', 'frozen'): ('Table 18', [(0.3845, 0.0345), (0.2983, 0.0498), (0.3946, 0.0378)]),
    ('Mumtaz2016', 'pretrained'): ('Table 10', [(0.9560, 0.0056), (0.9923, 0.0032), (0.9921, 0.0025)]),
    ('MentalArithmetic', 'pretrained'): ('Table 12', [(0.7256, 0.0132), (0.6267, 0.0099), (0.7905, 0.0073)]),
    # Table 15 headers read "AUC-PR / AUROC" but are Cohen's kappa / weighted F1 (author, issue #21)
    ('BCIC-IV-2a', 'pretrained'): ('Table 15', [(0.5138, 0.0066), (0.3518, 0.0094), (0.4984, 0.0085)]),
}
# controls (released model with the released one-layer head) are compared with the released 3-layer-head runs
RELEASED_SETTING = {'onelayer': 'pretrained', 'onelayer-frozen': 'frozen'}
MC = ['balanced_acc', 'kappa', 'weighted_f1']
BC = ['balanced_acc', 'pr_auc', 'roc_auc']


def welch(m1, s1, n1, m2, s2, n2):
    return float(stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False).pvalue)


runs = []
for log in sorted(glob.glob(f'{RUNS}/*/log.txt')):
    ds, setting, seed = os.path.basename(os.path.dirname(log)).split('__')
    txt = open(log).read()
    m = re.search(r'Test Evaluation: acc: ([\d.]+), (kappa|pr_auc): ([\d.]+), (f1|roc_auc): ([\d.]+)', txt)
    if not m:
        continue
    ep = re.search(r'model save in .*/epoch(\d+)_', txt)
    runs.append(dict(dataset=ds, setting=setting, seed=int(seed[1:]),
                     metrics=[float(m.group(1)), float(m.group(3)), float(m.group(5))],
                     metric_names=MC if m.group(2) == 'kappa' else BC, best_epoch=int(ep.group(1)) if ep else None,
                     epochs_run=len(re.findall(r'^Epoch \d+ :', txt, re.M))))

released = {(s['dataset'], s['setting']): s for s in json.load(open(RELEASED))['summary']}
summary = []
for key in sorted({(r['dataset'], r['setting']) for r in runs}):
    rs = sorted([r for r in runs if (r['dataset'], r['setting']) == key], key=lambda r: r['seed'])
    arr = np.array([r['metrics'] for r in rs])
    n = len(rs)
    row = dict(dataset=key[0], setting=key[1], n=n, seeds=[r['seed'] for r in rs], metric_names=rs[0]['metric_names'],
               per_seed=arr.tolist(), best_epochs=[r['best_epoch'] for r in rs], mean=arr.mean(0).tolist(),
               std=(arr.std(0, ddof=1) if n > 1 else np.zeros(3)).tolist())
    if key in PAPER:
        row['paper_table'], row['paper'] = PAPER[key]
        row['p_welch_vs_paper'] = [welch(mu, sd, n, pm, ps, 5) if n > 1 else None
                                   for mu, sd, (pm, ps) in zip(row['mean'], row['std'], row['paper'])]
    rel = released.get((key[0], RELEASED_SETTING.get(key[1], key[1])))
    if rel:
        row['released'] = [list(x) for x in zip(rel['mean'], rel['std'])]
        row['released_per_seed'] = rel['per_seed']
        row['p_welch_vs_released'] = [welch(mu, sd, n, rm, rsd, rel['n']) if n > 1 else None
                                      for mu, sd, rm, rsd in zip(row['mean'], row['std'], rel['mean'], rel['std'])]
    summary.append(row)

os.makedirs(OUT, exist_ok=True)
json.dump(dict(runs=runs, summary=summary), open(os.path.join(OUT, 'results.json'), 'w'), indent=1)
for row in summary:
    print(f"\n{row['dataset']} [{row['setting']}] n={row['n']} seeds={row['seeds']} best_epochs={row['best_epochs']}")
    for i, mn in enumerate(row['metric_names']):
        label = 'control (released model, 1-layer head)' if row['setting'] in RELEASED_SETTING else 'braindecode'
        s = f"  {mn:13s} {label} {row['mean'][i]:.4f} ± {row['std'][i]:.4f}"
        if 'released' in row:
            rm, rsd = row['released'][i]
            p = row['p_welch_vs_released'][i]
            s += f" | released code (3-layer head) {rm:.4f} ± {rsd:.4f} | diff {row['mean'][i] - rm:+.4f}" + \
                 (f" | p={p:.3g}" if p is not None else '')
        if 'paper' in row:
            pm, ps = row['paper'][i]
            p = row['p_welch_vs_paper'][i]
            s += f" || paper {pm:.4f} ± {ps:.4f} | diff {row['mean'][i] - pm:+.4f}" + (f" | p={p:.3g}" if p is not None else '')
        print(s)
        print('      per-seed:', ' '.join(f'{v[i]:.4f}' for v in row['per_seed']))
