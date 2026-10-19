"""Draw results/fig_epoch_selection.png: EEGNet vs. EEG Conformer test accuracy under each epoch-selection rule.

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
COLOR = {'EEGNet': '#2a78d6', 'EEG Conformer': '#eb6834'}

rows = [('EEG Conformer paper, Table II', r['paper']['EEGNet']['mean'], r['paper']['Conformer']['mean'],
         r['paper']['conformer_minus_eegnet']['mean'])]
for key, label in [('max test acc over 2000 epochs (released rule)', 'Max test acc over 2000 epochs (released rule)'),
                   ('validation-selected epoch (protocol B)', 'Validation-selected epoch (20% of session T held out)'),
                   ('mean of the last 100 epochs', 'Mean of the last 100 epochs'),
                   ('final epoch', 'Final epoch')]:
    rows.append((label, r['rules'][key]['eegnet']['mean'], r['rules'][key]['conformer']['mean'],
                 r['rules'][key]['conformer_minus_eegnet']))   # mean of the per-subject differences, as in the README

vals = [v for _, e, c, _ in rows for v in (e, c)]
X0, X1 = min(vals) - 3, max(vals) + 5          # room for the value labels and the difference column
fig, ax = plt.subplots(figsize=(9, 3.3), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
for i, (label, e, c, diff) in enumerate(rows):
    y = len(rows) - 1 - i
    dy = 0.1 if abs(c - e) < 0.6 else 0          # nudge apart dots that would overlap
    ax.plot([e, c], [y + dy, y - dy], color=BASE, lw=2, zorder=1, solid_capstyle='round')
    for v, m, yy in ((e, 'EEGNet', y + dy), (c, 'EEG Conformer', y - dy)):
        ax.scatter(v, yy, s=70, color=COLOR[m], edgecolor=SURFACE, linewidth=2, zorder=3)
    (lo, ylo), (hi, yhi) = sorted(((e, y + dy), (c, y - dy)))
    ax.text(lo - 0.35, ylo, f'{lo:.2f}', ha='right', va='center', fontsize=8, color=INK2)
    ax.text(hi + 0.35, yhi, f'{hi:.2f}', ha='left', va='center', fontsize=8, color=INK2)
    ax.text(X1 - 0.2, y, f'{diff:+.2f} pp'.replace('-', '−'), ha='right', va='center', fontsize=8, color=MUTED)
ax.axhline(len(rows) - 1.5, color=GRID, lw=1)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([row[0] for row in rows][::-1], fontsize=8.5, color=INK)
ax.set_xlim(X0, X1)
ax.set_xlabel('Test accuracy (%), mean over 9 subjects × 3 seeds', fontsize=8.5, color=INK2)
ax.text(X1 - 0.2, len(rows) - 0.5, 'Conformer − EEGNet', ha='right', va='bottom', fontsize=8, color=MUTED)
ax.tick_params(axis='x', colors=MUTED, labelsize=8)
ax.tick_params(axis='y', length=0)
ax.grid(axis='x', color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ('top', 'right', 'left'):
    ax.spines[s].set_visible(False)
ax.spines['bottom'].set_color(BASE)
ax.set_ylim(-0.6, len(rows) - 0.25)
handles = [plt.Line2D([], [], marker='o', ls='', ms=8, color=COLOR[m], label=m) for m in COLOR]
fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.01, 0.995), ncol=2, frameon=False, fontsize=8.5,
           labelcolor=INK)
fig.suptitle('BCI-IV-2a, within-subject: EEGNet and EEG Conformer in the same training loop',
             x=0.99, ha='right', y=0.985, fontsize=9.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = os.path.join(RESULTS, 'fig_epoch_selection.png')
fig.savefig(out, facecolor=SURFACE)
print('wrote', out)
