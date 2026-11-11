"""The authors' eeg_DE.py, run unmodified (code/run_authors.sh), against our PyTorch port under the same rules.

Each authors' run writes accuracy.mat: 45 x 2, per session [test accuracy at the last step, peak test accuracy over the
~80 evaluations] (models.fit returns last_accuracy, peak_accuracy in that order). eeg_DE.py prints the mean of the peak
over each subject's best 2 of 3 sessions (its final_r) and the SD of those 30 values. Our port's sd__<feature>__s<k>/result.npz
holds the 81 test evaluations of every session (the same schedule: every 5 steps of 402, plus the last).

Per rule: mean over runs of the run-level score, the SD of the 30 (or 45) session values as eeg_DE.py prints it, the SD over
runs; authors' code - port: Welch's t-test over runs, and a two-sided Wilcoxon test over the 45 sessions (each session's
mean over runs).
usage: python analyze_authors.py <dir with authors__<feature>__r<k>/accuracy.mat> <port runs dir with sd__*> <out dir>
"""
import glob
import json
import os
import sys

import numpy as np
import scipy.io as sio
from scipy import stats

adir, pdir, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
lines = []


def p(s=''):
    print(s)
    lines.append(s)


def best2(acc):                       # acc: 45 sessions, subject-major -> the 30 values eeg_DE.py averages
    return np.sort(acc.reshape(15, 3), axis=1)[:, 1:].ravel()


def authors(feat):                    # runs x 45 x [last, peak], in %
    fs = sorted(glob.glob(os.path.join(adir, f'authors__{feat}__r*', 'accuracy.mat')))
    return np.stack([sio.loadmat(f)['accuracy'] for f in fs]).astype(float), fs


def port(feat):                       # runs x 45 x [last, peak], in %
    fs = sorted(glob.glob(os.path.join(pdir, f'sd__{feat}__s*', 'result.npz')))
    runs = []
    for f in fs:
        z = np.load(f)
        c = np.stack([z[f's{i}'][0] for i in range(45)])
        runs.append(np.stack([c[:, -1], c.max(1)], 1))
    return np.stack(runs).astype(float) * (100 if np.max(runs) <= 1 else 1), fs


RULES = [('peak on test, best 2 of 3 sessions (the authors\' rule)', 1, True), ('peak on test, all sessions', 1, False),
         ('last step, best 2 of 3 sessions', 0, True), ('last step, all sessions', 0, False)]
res = []
p("DGCNN on SEED, within subject (first 9 clips train, last 6 test), accuracy %. Paper: 90.40 (8.49).")
for feat in ('de_LDS', 'de_movingAve'):
    A, fa = authors(feat)
    P, fp_ = port(feat)
    assert A.shape[1:] == (45, 2) and P.shape[1:] == (45, 2), (A.shape, P.shape)
    p(f'\n{feat}: authors\' code {len(fa)} runs, our port {len(fp_)} runs')
    p(f"{'rule':52s} {'authors (SD sessions; SD runs)':34s} {'port (SD sessions; SD runs)':34s} {'difference':11s} {'Welch p':8s} {'Wilcoxon p':10s}")
    for name, col, b2 in RULES:
        f = (lambda a: best2(a)) if b2 else (lambda a: a)
        ra = np.array([f(r[:, col]).mean() for r in A])
        rp = np.array([f(r[:, col]).mean() for r in P])
        sa = np.mean([f(r[:, col]).std() for r in A])          # eeg_DE.py's np.std, averaged over runs
        sp = np.mean([f(r[:, col]).std() for r in P])
        pw = stats.ttest_ind(ra, rp, equal_var=False).pvalue
        pr = stats.wilcoxon(A[:, :, col].mean(0), P[:, :, col].mean(0)).pvalue
        p(f"{name:52s} {ra.mean():6.2f} ({sa:5.2f}; {ra.std(ddof=1):4.2f})            {rp.mean():6.2f} ({sp:5.2f}; {rp.std(ddof=1):4.2f})"
          f"            {ra.mean() - rp.mean():+6.2f}      {pw:.3f}    {pr:.3f}")
        res.append(dict(feature=feat, rule=name, authors_mean=round(ra.mean(), 2), authors_sd_sessions=round(sa, 2),
                        authors_sd_runs=round(ra.std(ddof=1), 2), authors_per_run=[round(x, 2) for x in ra],
                        port_mean=round(rp.mean(), 2), port_sd_sessions=round(sp, 2), port_sd_runs=round(rp.std(ddof=1), 2),
                        diff=round(ra.mean() - rp.mean(), 2), welch_p=round(float(pw), 4), wilcoxon_sessions_p=round(float(pr), 4)))
json.dump(res, open(os.path.join(out, 'authors.json'), 'w'), indent=1)
open(os.path.join(out, 'authors.txt'), 'w').write('\n'.join(lines) + '\n')
