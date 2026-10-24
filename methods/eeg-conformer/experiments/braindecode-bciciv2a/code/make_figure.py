"""Draw results/fig_braindecode_vs_released.png: test accuracy of the two implementations under each rule.

Reads results/results.json (written by code/analyze.py).
usage: python code/make_figure.py [results]
"""
import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS = sys.argv[1] if len(sys.argv) > 1 else 'results'
r = json.load(open(os.path.join(RESULTS, 'results.json')))
SURFACE, INK, INK2, MUTED, GRID, BASE = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
COLOR = {'released': '#2a78d6', 'braindecode': '#eb6834'}
LABEL = {'released': 'released conformer.py model', 'braindecode': 'braindecode EEGConformer'}

rows = []
for key, label in [('max test acc over 2000 epochs (released rule)', 'Max test acc over 2000 epochs (released rule)'),
                   ('validation-selected epoch (protocol B)', 'Validation-selected epoch (20% of session T held out)'),
                   ('mean of the last 100 epochs', 'Mean of the last 100 epochs'),
                   ('final epoch', 'Final epoch')]:
    rr = r['rules'][key]
    rows.append((label, rr['released']['mean'], rr['braindecode']['mean'], rr['braindecode_minus_released'],
                 rr['wilcoxon_p']))

vals = [v for _, a, b, _, _ in rows for v in (a, b)] + [r['paper']['mean']]
X0, X1 = min(vals) - 3, max(vals) + 7          # room for the value labels and the difference column
fig, ax = plt.subplots(figsize=(9, 3.3), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
n = len(rows) + 1
y = n - 1                                       # paper row on top: one hollow neutral marker
ax.scatter(r['paper']['mean'], y, s=70, facecolor=SURFACE, edgecolor=INK2, linewidth=1.5, zorder=3)
ax.text(r['paper']['mean'] + 0.35, y, f"{r['paper']['mean']:.2f}", ha='left', va='center', fontsize=8, color=INK2)
for i, (label, rel, bd, diff, p) in enumerate(rows):
    y = len(rows) - 1 - i
    dy = 0.1 if abs(bd - rel) < 0.6 else 0     # nudge apart dots that would overlap
    ax.plot([rel, bd], [y + dy, y - dy], color=BASE, lw=2, zorder=1, solid_capstyle='round')
    for v, m, yy in ((rel, 'released', y + dy), (bd, 'braindecode', y - dy)):
        ax.scatter(v, yy, s=70, color=COLOR[m], edgecolor=SURFACE, linewidth=2, zorder=3)
    (lo, ylo), (hi, yhi) = sorted(((rel, y + dy), (bd, y - dy)))
    ax.text(lo - 0.35, ylo, f'{lo:.2f}', ha='right', va='center', fontsize=8, color=INK2)
    ax.text(hi + 0.35, yhi, f'{hi:.2f}', ha='left', va='center', fontsize=8, color=INK2)
    ax.text(X1 - 0.2, y, f'{diff:+.2f} pp (p {p:.2f})'.replace('-', '−'), ha='right', va='center', fontsize=8,
            color=MUTED)
ax.axhline(len(rows) - 0.5, color=GRID, lw=1)
ax.set_yticks(range(n))
ax.set_yticklabels([row[0] for row in rows][::-1] + ['EEG Conformer paper, Table II'], fontsize=8.5, color=INK)
ax.set_xlim(X0, X1)
ax.set_xlabel('Test accuracy (%), mean over 9 subjects × 3 seeds', fontsize=8.5, color=INK2)
ax.text(X1 - 0.2, n - 0.5, 'braindecode − released', ha='right', va='bottom', fontsize=8, color=MUTED)
ax.tick_params(axis='x', colors=MUTED, labelsize=8)
ax.tick_params(axis='y', length=0)
ax.grid(axis='x', color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ('top', 'right', 'left'):
    ax.spines[s].set_visible(False)
ax.spines['bottom'].set_color(BASE)
ax.set_ylim(-0.6, n - 0.25)
handles = [plt.Line2D([], [], marker='o', ls='', ms=8, color=COLOR[m], label=LABEL[m]) for m in COLOR]
fig.suptitle('BCI-IV-2a, within-subject: two EEG Conformer implementations in the same training loop',
             x=0.01, ha='left', y=0.985, fontsize=9.5, color=INK)
fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.005, 0.93), ncol=2, frameon=False, fontsize=8.5,
           labelcolor=INK)
fig.tight_layout(rect=(0, 0, 1, 0.89))
out = os.path.join(RESULTS, 'fig_braindecode_vs_released.png')
fig.savefig(out, facecolor=SURFACE)
print('wrote', out)
