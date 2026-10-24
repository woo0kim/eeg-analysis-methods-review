"""Draw results/fig_braindecode_vs_released.png: balanced accuracy of braindecode's CBraMod, the released-code runs
and the paper, one panel per dataset (mean ± SD over seeds 0-4).

Reads results/results.json (written by code/collect.py).
usage: python code/make_figure.py [results]
"""
import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS = sys.argv[1] if len(sys.argv) > 1 else 'results'
S = {(s['dataset'], s['setting']): s for s in json.load(open(os.path.join(RESULTS, 'results.json')))['summary']}
SURFACE, INK, INK2, MUTED, GRID, BASE = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
COLOR = {'released': '#2a78d6', 'braindecode': '#eb6834'}
LABEL = {'released': 'released code (3-layer head)', 'braindecode': 'braindecode CBraMod (linear head)'}
DATASETS = ['PhysioNet-MI', 'BCIC-IV-2a', 'Mumtaz2016', 'MentalArithmetic']
SETTINGS = [('pretrained', 'Pre-trained'), ('scratch', 'From scratch'), ('frozen', 'Frozen backbone')]

fig, axes = plt.subplots(1, 4, figsize=(11, 2.9), dpi=200, sharey=True)
fig.patch.set_facecolor(SURFACE)
for ax, ds in zip(axes, DATASETS):
    ax.set_facecolor(SURFACE)
    vals = []
    for j, (st, _) in enumerate(SETTINGS):
        s = S.get((ds, st))
        if not s:
            continue
        y = len(SETTINGS) - 1 - j
        points = [('braindecode', s['mean'][0], s['std'][0], y - 0.17)]
        if 'released' in s:
            points.append(('released', s['released'][0][0], s['released'][0][1], y + 0.17))
        for m, mu, sd, yy in points:
            ax.plot([mu - sd, mu + sd], [yy, yy], color=COLOR[m], lw=2, solid_capstyle='round', zorder=2)
            ax.scatter(mu, yy, s=48, color=COLOR[m], edgecolor=SURFACE, linewidth=2, zorder=3)
            vals += [mu - sd, mu + sd]
        if 'paper' in s:
            ax.scatter(s['paper'][0][0], y, s=44, facecolor='none', edgecolor=INK2, linewidth=1.3, zorder=1,
                       marker='D')       # hollow and underneath, so it never hides a mean
            vals.append(s['paper'][0][0])
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.12
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_title(ds, fontsize=9, color=INK, loc='left')
    ax.tick_params(axis='x', colors=MUTED, labelsize=7.5)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color(BASE)
axes[0].set_yticks(range(len(SETTINGS)))
axes[0].set_yticklabels([lab for _, lab in SETTINGS][::-1], fontsize=8.5, color=INK)
axes[0].set_ylim(-0.6, len(SETTINGS) - 0.4)
fig.supxlabel('Test balanced accuracy, mean ± SD over seeds 0–4 (x-axes differ by dataset)', fontsize=8.5,
              color=INK2)
handles = [plt.Line2D([], [], marker='o', ls='-', lw=2, ms=7, color=COLOR[m], label=LABEL[m]) for m in COLOR]
handles.append(plt.Line2D([], [], marker='D', ls='', ms=6, markerfacecolor='none', markeredgecolor=INK2,
                          label='paper'))
fig.suptitle("CBraMod through its released fine-tuning pipeline: released model vs. braindecode's", x=0.01,
             ha='left', y=0.985, fontsize=9.5, color=INK)
fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.005, 0.93), ncol=3, frameon=False, fontsize=8.5,
           labelcolor=INK)
fig.tight_layout(rect=(0, 0, 1, 0.86))
out = os.path.join(RESULTS, 'fig_braindecode_vs_released.png')
fig.savefig(out, facecolor=SURFACE)
print('wrote', out)
