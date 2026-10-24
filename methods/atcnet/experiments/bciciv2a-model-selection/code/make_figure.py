"""Draw results/fig_reported_vs_reproduced.png.

Left: per protocol, the mean of all runs (circle) and the best-of-N rule of the released test() (diamond), with the
paper's number; right: every run per subject for the authors' 2022 runs and our runs of the same code, with the
paper's Table IV value. Reads results/results.json and the per-run accuracies in it (code/analyze.py).
usage: python code/make_figure.py [RESULTS_DIR]
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '../results')
r = json.load(open(os.path.join(RESULTS, 'results.json')))
H = r['headline']
SURFACE, INK, INK2, MUTED, GRID, BASE = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
COLOR = {'authors2022': MUTED, 'paper2022': '#2a78d6', 'traintest': '#eb6834', 'trainvaltest': '#1f9e7a',
         'loso': '#8b5cc4'}
ROWS = [('authors2022', "Authors' 2022 runs (their log)", 85.38), ('paper2022', '2022 code, our runs', 85.38),
        ('traintest', 'main_TrainTest.py (today)', 85.38), ('trainvaltest', 'main_TrainValTest.py (today)', 85.38),
        ('loso', 'LOSO, main_TrainTest.py', 70.97)]
ROWS = [row for row in ROWS if f'{row[0]}_mean' in H]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 3.9), dpi=200, gridspec_kw={'width_ratios': [1, 1.15]})
fig.patch.set_facecolor(SURFACE)
for a in (ax, bx):
    a.set_facecolor(SURFACE)
    for sp in ('top', 'right', 'left'):
        a.spines[sp].set_visible(False)
    a.spines['bottom'].set_color(BASE)
    a.tick_params(colors=MUTED, labelsize=7.5)
    a.set_axisbelow(True)

# ---- left: mean of runs vs. best of N, per protocol
for k, (p, label, paper) in enumerate(ROWS):
    y = len(ROWS) - 1 - k
    m = H[f'{p}_mean']
    sd = m.get('sd_over_seed_means', 0) or 0
    ax.plot([m['mean_acc'] - sd, m['mean_acc'] + sd], [y, y], color=COLOR[p], lw=2, solid_capstyle='round', zorder=2)
    ax.scatter(m['mean_acc'], y, s=52, color=COLOR[p], edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.annotate(f"{m['mean_acc']:.2f}", (m['mean_acc'], y), xytext=(0, -11), textcoords='offset points',
                ha='center', va='top', fontsize=7, color=INK2)
    if f'{p}_best' in H:
        b = H[f'{p}_best']
        ax.plot([m['mean_acc'], b['mean_acc']], [y, y], color=COLOR[p], lw=1, ls=':', zorder=1)
        ax.scatter(b['mean_acc'], y, s=56, marker='D', color=COLOR[p], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.annotate(f"{b['mean_acc']:.2f}", (b['mean_acc'], y), xytext=(0, 7), textcoords='offset points',
                    ha='center', va='bottom', fontsize=7, color=INK2)
    ax.scatter(paper, y, s=60, marker='|', color=INK, linewidth=1.6, zorder=4)
ax.set_yticks(range(len(ROWS)))
ax.set_yticklabels([lab for _, lab, _ in ROWS][::-1], fontsize=8, color=INK)
ax.tick_params(axis='y', length=0)
ax.set_ylim(-0.7, len(ROWS) - 0.3)
ax.grid(axis='x', color=GRID, lw=0.8)
ax.set_xlabel('Test accuracy (%), mean over the 9 subjects', fontsize=8, color=INK2)
handles = [plt.Line2D([], [], marker='o', ls='-', lw=2, ms=6, color=INK2, label='mean of all runs (± SD over seeds)'),
           plt.Line2D([], [], marker='D', ls='', ms=5, color=INK2, label='best run per subject'),
           plt.Line2D([], [], marker='|', ls='', ms=9, mew=1.6, color=INK, label='paper')]
ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(-0.02, 1.0), ncol=3, frameon=False, fontsize=7.5,
          labelcolor=INK, handletextpad=0.3, columnspacing=1.0)

# ---- right: every run per subject, authors' 2022 runs vs. ours, and the paper's Table IV value
per_run = r['per_run_acc']
for off, p, lab in ((-0.17, 'authors2022', "authors' 2022 runs"), (0.17, 'paper2022', 'our runs of the 2022 code')):
    if p not in per_run:
        continue
    a = np.array(per_run[p])                      # (9, n_runs)
    for i in range(9):
        bx.scatter(np.full(a.shape[1], i + 1 + off), a[i], s=11, color=COLOR[p], alpha=0.8, linewidth=0, zorder=2,
                   label=lab if i == 0 else None)
    bx.scatter(np.arange(1, 10) + off, a.mean(1), s=70, marker='_', color=COLOR[p], linewidth=2, zorder=3)
bx.scatter(np.arange(1, 10), r['paper']['per_subject_acc'], s=46, marker='D', facecolor='none', edgecolor=INK,
           linewidth=1.2, zorder=4, label='paper, Table IV')
bx.set_xticks(range(1, 10))
bx.set_xlabel('Subject', fontsize=8, color=INK2)
bx.set_ylabel('Test accuracy (%)', fontsize=8, color=INK2)
bx.grid(axis='y', color=GRID, lw=0.8)
bx.spines['left'].set_visible(True)
bx.spines['left'].set_color(BASE)
bx.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=7.5, labelcolor=INK,
          handletextpad=0.3, columnspacing=1.0)
fig.suptitle('ATCNet on BCI IV 2a: the paper\'s 85.38% is the best of 10 runs per subject, each checkpointed on the '
             'test set', x=0.01, ha='left', y=0.985, fontsize=9.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
out = os.path.join(RESULTS, 'fig_reported_vs_reproduced.png')
fig.savefig(out, facecolor=SURFACE)
print('wrote', out)
