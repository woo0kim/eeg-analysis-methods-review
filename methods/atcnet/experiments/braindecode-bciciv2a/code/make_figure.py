"""Draw results/fig_braindecode_vs_keras.png: per protocol, the mean test accuracy of the Keras runs and of each
braindecode setting (± SD over seed means), with the braindecode − Keras difference on the same seeds.

Reads results/results.json (code/analyze.py).
usage: python code/make_figure.py [RESULTS_DIR]
"""
import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '../results')
C = json.load(open(os.path.join(RESULTS, 'results.json')))['comparisons']
SURFACE, INK, INK2, MUTED, GRID, BASE = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
ROWS = [('keras', 'Keras (released code)', '#2a78d6'), ('default', 'braindecode, defaults', '#eb6834'),
        ('faithful', 'braindecode, "faithful" options', '#1f9e7a'),
        ('control', 'control: braindecode + Keras training\n(2022 or current code)', '#8b5cc4')]
PANELS = [('paper2022', '2022 code (paper)'), ('traintest', 'main_TrainTest.py'),
          ('trainvaltest', 'main_TrainValTest.py'), ('loso', 'LOSO')]
PAPER = {'paper2022': 85.38, 'traintest': 85.38, 'trainvaltest': 85.38, 'loso': 70.97}
PANELS = [(p, t) for p, t in PANELS if f'default/{p}/mean of runs' in C]

fig, axes = plt.subplots(1, len(PANELS), figsize=(11.5, 2.9), dpi=200, sharey=True)
axes = axes if len(PANELS) > 1 else [axes]
fig.patch.set_facecolor(SURFACE)
for ax, (p, title) in zip(axes, PANELS):
    ax.set_facecolor(SURFACE)
    vals = [PAPER[p]]
    ref = C[f'default/{p}/mean of runs']
    for k, (v, label, color) in enumerate(ROWS):
        y = len(ROWS) - 1 - k
        if v == 'keras':
            mu, sd, note = ref['keras_acc'], ref.get('keras_sd_over_seed_means') or 0, ''
        else:
            ctrl = 'keras2022' if p == 'paper2022' else 'keras'
            key = f'{ctrl if v == "control" else v}/{p}/mean of runs'
            if key not in C:
                continue
            c = C[key]
            mu, sd = c['bd_acc'], c.get('bd_sd_over_seed_means') or 0
            note = f"{c['bd_minus_keras']:+.2f}" + (' *' if c['wilcoxon_p'] < 0.05 else '')
        ax.plot([mu - sd, mu + sd], [y, y], color=color, lw=2, solid_capstyle='round', zorder=2)
        ax.scatter(mu, y, s=48, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
        if note:
            ax.annotate(note, (mu + sd, y), xytext=(4, 0), textcoords='offset points', va='center', fontsize=7,
                        color=INK2)
        vals += [mu - sd, mu + sd]
    ax.axvline(PAPER[p], color=INK, lw=1, ls='--', zorder=1)
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.25
    ax.set_xlim(lo - pad, hi + pad * 1.6)
    ax.set_title(title, fontsize=9, color=INK, loc='left')
    ax.tick_params(axis='x', colors=MUTED, labelsize=7.5)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color(BASE)
axes[0].set_yticks(range(len(ROWS)))
axes[0].set_yticklabels([lab for _, lab, _ in ROWS][::-1], fontsize=8, color=INK)
axes[0].set_ylim(-0.6, len(ROWS) - 0.4)
fig.supxlabel('Test accuracy (%), mean of runs ± SD over seed means; number = braindecode − Keras on the same seeds '
              '(* Wilcoxon p < 0.05); dashed = paper', fontsize=8, color=INK2)
fig.suptitle("ATCNet on BCI IV 2a: braindecode's ATCNet vs. the released Keras code, same data, protocol and seeds",
             x=0.01, ha='left', y=0.985, fontsize=9.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = os.path.join(RESULTS, 'fig_braindecode_vs_keras.png')
fig.savefig(out, facecolor=SURFACE)
print('wrote', out)
