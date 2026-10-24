"""Released ATCNet on BCI IV 2a: the paper's numbers against the code that produced them, today's released protocols,
and what the reported number depends on. Needs numpy and scipy only.

Reads results/curves/<protocol>/s<sub>_seed<seed>.npz (one per run; code/run_paper2022.py, code/run_released.py) and
results/authors_2022_log.txt (the authors' own 90 runs behind the paper, upstream commit 3a318b2), prints every table
of the README and writes results/results.json. Accuracy in %; kappa is sklearn's cohen_kappa_score of a run and, as in
the released test(), a subject's kappa is that of the run whose accuracy is reported, averaged over subjects. Paired
comparisons over the 9 subjects use a two-sided Wilcoxon signed-rank test (scipy defaults; with 9 pairs the smallest
attainable p is 0.0039); comparisons of all runs use a two-sided Mann-Whitney U test.
usage: python code/analyze.py [RESULTS_DIR]
"""
import glob, json, os, re, sys, warnings
from math import comb
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '../results')
SUBJECTS, SEEDS = range(1, 10), range(1, 11)
PAPER_ACC = [88.5, 70.5, 97.6, 81.0, 83.0, 73.6, 93.1, 90.3, 91.0]       # Table IV, per subject
PAPER_MEAN, PAPER_K = 85.38, 0.805                                         # Tables II, III, V (Table IV: 85.4 / 0.81)
PAPER_LOSO, PAPER_LOSO_K = 70.97, 0.613                                    # Table VI
RELEASED_WEIGHTS = [86.11, 64.93, 92.01, 80.56, 71.53, 68.75, 95.14, 82.99, 87.85]   # upstream results/log.txt
PROTOCOLS = {
    'paper2022': '2022 code (paper): session E is the validation data; lr 0.0009, no schedule, no weight decay',
    'traintest': 'main_TrainTest.py (today): session E is the validation data; lr 0.001, ReduceLROnPlateau, L2',
    'trainvaltest': 'main_TrainValTest.py (today): 20% of session T is the validation data; 500 epochs',
    'loso': 'main_TrainTest.py with LOSO=True: the held-out subject is the validation data'}


def load(protocol):
    runs = {}
    for f in glob.glob(os.path.join(RESULTS, 'curves', protocol, 's*_seed*.npz')):
        d = dict(np.load(f))
        runs[(int(d['subject']), int(d['seed']))] = d
    return runs


def authors_2022():
    """(9 x 10) accuracy (%) and kappa of the authors' runs, from their log."""
    acc, kap = np.full((9, 10), np.nan), np.full((9, 10), np.nan)
    for s, r, a, k in re.findall(r'Subject: (\d)   Train no\. (\d+)   Time: [\d.]+ m   Test_acc: ([\d.]+)   '
                                 r'Test_kappa: ([\d.]+)', open(os.path.join(RESULTS, 'authors_2022_log.txt')).read()):
        acc[int(s) - 1, int(r) - 1], kap[int(s) - 1, int(r) - 1] = float(a) * 100, float(k)
    return acc, kap


R = {p: load(p) for p in PROTOCOLS}
acc = {p: np.array([[R[p][(s, sd)]['acc'] * 100 if (s, sd) in R[p] else np.nan for sd in SEEDS] for s in SUBJECTS])
       for p in R}
kap = {p: np.array([[R[p][(s, sd)]['kappa'] if (s, sd) in R[p] else np.nan for sd in SEEDS] for s in SUBJECTS])
       for p in R}
acc['authors2022'], kap['authors2022'] = authors_2022()
for p in acc:                                           # keep the seeds present for all 9 subjects
    full = np.isfinite(acc[p]).all(0)
    acc[p], kap[p] = acc[p][:, full], kap[p][:, full]
N = {p: acc[p].shape[1] for p in acc}


def wilcoxon(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a, b):
        return float('nan')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return float(stats.wilcoxon(a, b).pvalue)


def best_of(p):
    """Released test() of main_TrainTest.py / the 2022 main.py: per subject the run with the highest test accuracy."""
    i = np.argmax(acc[p], axis=1)
    return acc[p][np.arange(9), i], kap[p][np.arange(9), i], i + 1


def expected_best_of(row, k):
    """Expected maximum of k runs drawn without replacement from the runs of one subject."""
    v = np.sort(row)
    return sum(v[j] * comb(j, k - 1) for j in range(k - 1, len(v))) / comb(len(v), k) if len(v) >= k else np.nan


out = {'paper': {'per_subject_acc': PAPER_ACC, 'mean_acc': PAPER_MEAN, 'kappa': PAPER_K, 'loso_acc': PAPER_LOSO,
                 'loso_kappa': PAPER_LOSO_K},
       'released_weights': {'per_subject_acc': RELEASED_WEIGHTS, 'mean_acc': round(float(np.mean(RELEASED_WEIGHTS)), 2)},
       'n_runs': {p: int(N[p] * 9) for p in N}, 'headline': {},
       'per_run_acc': {p: np.round(acc[p], 2).tolist() for p in acc if N[p]}}       # (9 subjects x runs)
for p in PROTOCOLS:
    print(f'{p}: {len(R[p])} runs, {N[p]} complete seeds ({PROTOCOLS[p]})')

# ------------------------------------------------------------------ headline
rows = []


def add(key, label, p, a, k, seed_means=None, paper=PAPER_MEAN):
    rec = {'label': label, 'protocol': p, 'per_subject_acc': np.round(a, 2).tolist(),
           'mean_acc': round(float(a.mean()), 2), 'kappa': round(float(k.mean()), 3),
           'minus_paper': round(float(a.mean() - paper), 2) + 0.0}
    if seed_means is not None and len(seed_means) > 1:
        rec['seed_means'] = np.round(seed_means, 2).tolist()
        rec['sd_over_seed_means'] = round(float(np.std(seed_means, ddof=1)), 2)
    if p != 'loso':
        rec['wilcoxon_p_vs_paper'] = round(wilcoxon(a, PAPER_ACC), 4)
    out['headline'][key] = rec
    rows.append(rec)


for p in ('authors2022', 'paper2022', 'traintest', 'loso'):
    if not N[p]:
        continue
    b, bk, _ = best_of(p)
    paper = PAPER_LOSO if p == 'loso' else PAPER_MEAN
    who = "authors' 2022 runs (their log)" if p == 'authors2022' else p
    add(f'{p}_best', f'{who}: best of {N[p]} runs per subject (released test())', p, b, bk, paper=paper)
    add(f'{p}_mean', f'{who}: mean of {N[p]} runs', p, acc[p].mean(1), kap[p].mean(1), acc[p].mean(0), paper=paper)
if N['trainvaltest']:
    add('trainvaltest_mean', f'trainvaltest: mean of {N["trainvaltest"]} seeds', 'trainvaltest',
        acc['trainvaltest'].mean(1), kap['trainvaltest'].mean(1), acc['trainvaltest'].mean(0))
    add('trainvaltest_seed1', 'trainvaltest: seed 1 only (the README\'s configuration)', 'trainvaltest',
        acc['trainvaltest'][:, 0], kap['trainvaltest'][:, 0])

print('\nHeadline: accuracy (%) and kappa, mean over the 9 subjects (LOSO: 9 folds)\n')
print('| | Accuracy | κ | − paper | Wilcoxon p vs. paper (9 subjects) | SD over seed means |')
print('|---|---|---|---|---|---|')
print(f'| Paper, subject-dependent (Tables IV/V) | {PAPER_MEAN:.2f} | {PAPER_K:.3f} | | | |')
print(f'| Authors\' released weights (today, 1 seed) on our data | {np.mean(RELEASED_WEIGHTS):.2f} | 0.748 | '
      f'{np.mean(RELEASED_WEIGHTS) - PAPER_MEAN:+.2f} | {wilcoxon(RELEASED_WEIGHTS, PAPER_ACC):.4f} | |')
for r in rows:
    if r['protocol'] == 'loso' and 'best' in r['label']:
        print(f'| Paper, LOSO (Table VI) | {PAPER_LOSO:.2f} | {PAPER_LOSO_K:.3f} | | | |')
    print(f"| {r['label']} | {r['mean_acc']:.2f} | {r['kappa']:.3f} | {r['minus_paper']:+.2f} | "
          f"{r.get('wilcoxon_p_vs_paper', float('nan')):.4f} | {r.get('sd_over_seed_means', float('nan')):.2f} |")

if N['trainvaltest']:
    s1 = acc['trainvaltest'][:, 0]
    same = int((np.abs(s1 - np.array(RELEASED_WEIGHTS)) < 0.005).sum())
    out['trainvaltest_seed1_identical_subjects'] = same
    print(f'\nmain_TrainValTest.py, seed 1: test accuracy identical to the authors\' README / released weights on '
          f'{same} of 9 subjects (' + ', '.join(f'{a:.2f}/{b:.2f}' for a, b in zip(s1, RELEASED_WEIGHTS)) + ')')

# ------------------------------------------------------------------ per subject
print('\nPer subject, accuracy (%): best of the runs and mean ± SD over runs\n')
cols = [('authors2022', "Authors' 2022 runs"), ('paper2022', '2022 code, ours'), ('traintest', 'main_TrainTest.py'),
        ('trainvaltest', 'main_TrainValTest.py'), ('loso', 'LOSO')]
cols = [(p, n) for p, n in cols if N[p]]
print('| Subject | Paper (Table IV) | ' + ' | '.join(
    (f'{n}: best | {n}: mean ± SD' if p != 'trainvaltest' else f'{n}: mean ± SD') for p, n in cols) + ' |')
print('|---|---|' + ''.join('---|---|' if p != 'trainvaltest' else '---|' for p, _ in cols))
best = {p: best_of(p) for p, _ in cols if p != 'trainvaltest'}
for i, s in enumerate(SUBJECTS):
    cells = []
    for p, _ in cols:
        ms = f'{acc[p][i].mean():.2f} ± {acc[p][i].std(ddof=1):.2f}'
        cells.append(ms if p == 'trainvaltest' else f'{best[p][0][i]:.2f} | {ms}')
    print(f'| {s} | {PAPER_ACC[i]:.1f} | ' + ' | '.join(cells) + ' |')
print(f'| **Avg** | **{PAPER_MEAN:.2f}** | ' + ' | '.join(
    (f'**{acc[p].mean():.2f}**' if p == 'trainvaltest' else f'**{best[p][0].mean():.2f}** | **{acc[p].mean():.2f}**')
    for p, _ in cols) + ' |')
out['per_subject_sd_over_runs'] = {p: np.round(acc[p].std(axis=1, ddof=1), 2).tolist() for p, _ in cols}
out['best_run_index'] = {p: best[p][2].tolist() for p in best}

# ------------------------------------------------------------------ best of k
print('\nExpected best-of-k accuracy (mean over subjects of the expected maximum of k runs drawn from the runs of the '
      'subject; k = 1 is the mean of all runs)\n')
ks = (1, 2, 3, 5, 10)
print('| Runs | ' + ' | '.join(f'k = {k}' for k in ks) + ' |')
print('|---|' + '---|' * len(ks))
out['expected_best_of_k'] = {}
for p, n in cols:
    if p in ('trainvaltest', 'loso'):
        continue
    ek = [float(np.mean([expected_best_of(acc[p][i], k) for i in range(9)])) for k in ks]
    out['expected_best_of_k'][p] = dict(zip(map(str, ks), np.round(ek, 2).tolist()))
    print(f'| {n} | ' + ' | '.join(f'{v:.2f}' for v in ek) + ' |')

# ------------------------------------------------------------------ what the reported number depends on
print('\nPer run (mean ± SD over runs; test accuracy in %)\n')
curve = lambda d: d['test_acc'] if len(d['test_acc']) else d['val_accuracy']   # paper2022/traintest/LOSO: val = test
quant = {
    'test acc. of the checkpoint': lambda d: d['acc'] * 100,
    'test acc. at the last epoch trained': lambda d: curve(d)[-1] * 100,
    'mean test acc. of the last 50 epochs': lambda d: curve(d)[-50:].mean() * 100,
    'highest test acc. of any epoch': lambda d: curve(d).max() * 100,
    'checkpoint − last epoch (pp)': lambda d: d['acc'] * 100 - curve(d)[-1] * 100,
    'checkpoint − mean of the last 50 epochs (pp)': lambda d: d['acc'] * 100 - curve(d)[-50:].mean() * 100,
    'epochs trained': lambda d: float(d['epochs_run']),
    'epoch of the checkpoint (1-based)': lambda d: float(np.argmin(d['val_loss']) + 1) if str(d['protocol']) == 'trainvaltest'
    else float(np.argmax(d['val_accuracy']) + 1),
    'training accuracy, mean of the last 50 epochs': lambda d: float(np.mean(d['accuracy'][-50:]) * 100),
    'minutes per run (2 runs per Quadro M6000)': lambda d: float(d['minutes']),
}
ps = [p for p in PROTOCOLS if R[p]]
print('| Quantity | ' + ' | '.join(ps) + ' |')
print('|---|' + '---|' * len(ps))
out['per_run'] = {}
for qn, fn in quant.items():
    cells = []
    for p in ps:
        v = np.array([fn(d) for d in R[p].values()], float)
        out['per_run'].setdefault(qn, {})[p] = [round(float(v.mean()), 2), round(float(v.std(ddof=1)), 2)]
        cells.append(f'{v.mean():.2f} ± {v.std(ddof=1):.2f}')
    print(f'| {qn} | ' + ' | '.join(cells) + ' |')
chk = [abs(d['acc'] - d['val_accuracy'].max()) for p in ('paper2022', 'traintest', 'loso') for d in R[p].values()]
out['checkpoint_equals_max_val_acc'] = bool(len(chk) == 0 or max(chk) < 1e-6)
print(f'\nIn every run with session E (or the held-out subject) as validation data, the checkpoint\'s test accuracy is '
      f'the highest per-epoch validation accuracy: {out["checkpoint_equals_max_val_acc"]}')

# ------------------------------------------------------------------ paired comparisons
print('\nComparisons (per-subject means unless stated)\n')
print('| Comparison | Difference (pp) | Subjects higher | Wilcoxon p (9 subjects) | Mann-Whitney p (all runs) |')
print('|---|---|---|---|---|')
comps = [('2022 code, ours − authors\' 2022 runs (mean of runs)', 'paper2022', 'authors2022'),
         ('main_TrainTest.py − 2022 code (mean of runs)', 'traintest', 'paper2022'),
         ('main_TrainTest.py − main_TrainValTest.py (mean of runs)', 'traintest', 'trainvaltest'),
         ('2022 code − main_TrainValTest.py (mean of runs)', 'paper2022', 'trainvaltest')]
out['paired'] = {}
for name, p1, p2 in comps:
    if not (N[p1] and N[p2]):
        continue
    a, b = acc[p1].mean(1), acc[p2].mean(1)
    mw = float(stats.mannwhitneyu(acc[p1].ravel(), acc[p2].ravel()).pvalue)
    rec = {'diff': round(float((a - b).mean()), 2), 'higher': int((a > b).sum()), 'p': round(wilcoxon(a, b), 4),
           'mannwhitney_p': round(mw, 4)}
    out['paired'][name] = rec
    print(f"| {name} | {rec['diff']:+.2f} | {rec['higher']} / 9 | {rec['p']:.4f} | {rec['mannwhitney_p']:.4f} |")
for name, p in (('best of 10 − mean of 10, authors\' 2022 runs', 'authors2022'),
                ('best of 10 − mean of 10, 2022 code, ours', 'paper2022'),
                ('best of 10 − mean of 10, main_TrainTest.py', 'traintest')):
    if N[p]:
        a, b = best_of(p)[0], acc[p].mean(1)
        out['paired'][name] = {'diff': round(float((a - b).mean()), 2), 'higher': int((a > b).sum()),
                               'p': round(wilcoxon(a, b), 4)}
        print(f"| {name} | {out['paired'][name]['diff']:+.2f} | {out['paired'][name]['higher']} / 9 | "
              f"{out['paired'][name]['p']:.4f} | |")

with open(os.path.join(RESULTS, 'results.json'), 'w') as f:
    json.dump(out, f, indent=1)
