"""Test balanced accuracy of LaBraM-Base over the 50 fine-tuning epochs, one panel per dataset.

Line + band: mean ± std over the 5 seeds of the test balanced accuracy at every epoch (preds.npz).
Dots: the epoch each seed reports under the headline rule (best validation kappa / AUROC) and its test value.
Dashed line: LaBraM-Base as reported in the CBraMod paper. Same metric code as collect.py.
usage: python code/make_figure.py [RUNS_DIR] [OUT_PNG] [1x4|2x2]   (2x2: the compact version used in the slides)
"""
import glob, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, cohen_kappa_score

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '../logs/runs')
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, '../results/fig_epoch_selection.png')
LAYOUT = sys.argv[3] if len(sys.argv) > 3 else '1x4'
PAPER_BA = {'PhysioNet-MI': 0.6173, 'BCIC-IV-2a': 0.4869, 'Mumtaz2016': 0.9409, 'MentalArithmetic': 0.6909}
ORDER = ['PhysioNet-MI', 'BCIC-IV-2a', 'Mumtaz2016', 'MentalArithmetic']
BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e4e3df'


def ba(prob, y):
    pred = (prob.reshape(-1) > 0.5).astype(int) if prob.shape[-1] == 1 else prob.argmax(-1)
    return balanced_accuracy_score(y, pred)


def monitor(prob, y):   # headline rule: validation kappa (multi-class) / AUROC (binary)
    return roc_auc_score(y, prob.reshape(-1)) if prob.shape[-1] == 1 else cohen_kappa_score(y, prob.argmax(-1))


plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': MUTED,
                     'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED})
fig, axes = (plt.subplots(2, 2, figsize=(6.2, 3.6), constrained_layout=True) if LAYOUT == '2x2' else
             plt.subplots(1, 4, figsize=(10, 2.6), constrained_layout=True))
axes = axes.ravel()
for ax, ds in zip(axes, ORDER):
    curves, picks = [], []
    for d in sorted(glob.glob(os.path.join(RUNS, f'{ds}__labram__s*'))):
        p = np.load(os.path.join(d, 'preds.npz'))
        y_val, y_test = p['y_val'].astype(int), p['y_test'].astype(int)
        n = min(len(p['val']), len(p['test']))
        test = np.array([ba(p['test'][e], y_test) for e in range(n)])
        e = int(np.argmax([monitor(p['val'][e], y_val) for e in range(n)]))
        curves.append(test)
        picks.append((e + 1, test[e]))
    c = np.array(curves)
    x = np.arange(1, c.shape[1] + 1)
    m, s = c.mean(0), c.std(0, ddof=1)
    ax.fill_between(x, m - s, m + s, color=BLUE, alpha=0.15, linewidth=0)
    ax.plot(x, m, color=BLUE, linewidth=1.6, label='test BA per epoch (mean ± sd, 5 seeds)')
    ax.scatter(*zip(*picks), s=22, color=ORANGE, edgecolor='white', linewidth=0.8, zorder=3,
               label='epoch reported (best val. κ / AUROC), one dot per seed')
    ax.axhline(PAPER_BA[ds], color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
    left = ds == 'BCIC-IV-2a'   # keep the label clear of the seed dots
    ax.annotate(f'paper {PAPER_BA[ds]:.3f}', xy=(1 if left else 50, PAPER_BA[ds]), xytext=(0, 3),
                textcoords='offset points', ha='left' if left else 'right', va='bottom', color=MUTED, fontsize=7)
    ax.set_title(ds, color=INK, fontsize=9, fontweight='bold', loc='left')
    if LAYOUT != '2x2' or ds in ORDER[2:]:
        ax.set_xlabel('epoch')
    ax.set_xlim(0, 51)
    lo = min((m - s)[2:].min(), min(v for _, v in picks), PAPER_BA[ds])
    hi = max((m + s)[2:].max(), PAPER_BA[ds])
    ax.set_ylim(lo - 0.02, hi + 0.03)
    ax.grid(axis='y', color=GRID, linewidth=0.6)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
for ax in (axes[::2] if LAYOUT == '2x2' else axes[:1]):
    ax.set_ylabel('test BA' if LAYOUT == '2x2' else 'test balanced accuracy')
handles, labels = axes[0].get_legend_handles_labels()
handles.append(plt.Line2D([], [], color=MUTED, linestyle=(0, (4, 3)), linewidth=1))
labels.append('LaBraM-Base in the CBraMod paper')
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=1 if LAYOUT == '2x2' else 3,
           frameon=False, fontsize=7.5, labelcolor=INK)
fig.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
print('wrote', OUT)
