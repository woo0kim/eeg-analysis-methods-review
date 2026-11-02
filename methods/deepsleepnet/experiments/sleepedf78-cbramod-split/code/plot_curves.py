"""results/fig_curves.png and results/curves.txt: training and validation accuracy per epoch of each run, parsed from the
released trainers' output (logs/runs/run<r>/train.txt). The validation set (the 8 validation subjects) is only
printed by the released code, never used to choose the model; the final fine-tuning epoch is the model.

usage: python code/plot_curves.py [logs/runs] [results]
"""
import glob
import os
import re
import sys

RUNS = sys.argv[1] if len(sys.argv) > 1 else 'logs/runs'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'results'
# one-line epochs, and the multi-line form the trainers print every 10th epoch (confusion matrices in between)
EPOCH = re.compile(r'epoch (\d+):\s*\n?train \([^)]*\): n=\d+, loss=[^,]+, acc=([\d.]+), f1=[\d.]+(?:.|\n)*?'
                   r'valid \([^)]*\): n=\d+, loss=([\d.]+)[^,]*, acc=([\d.]+), f1=([\d.]+)')


def parse(path):
    """-> {'pretrain': [...], 'finetune': [...]}, each a list of (epoch, train acc, valid loss, valid acc, valid f1)."""
    phases, cur = [], []
    for m in EPOCH.finditer(open(path).read()):
        e = int(m.group(1))
        if e == 1 and cur:
            phases.append(cur)
            cur = []
        cur.append((e, float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))))
    phases.append(cur)
    assert len(phases) == 2 and len(phases[0]) == 100 and len(phases[1]) == 200, (path, [len(p) for p in phases])
    return dict(zip(('pretrain', 'finetune'), phases))


def main():
    runs = {int(os.path.basename(os.path.dirname(f))[3:]): parse(f)
            for f in sorted(glob.glob(os.path.join(RUNS, 'run*', 'train.txt')))}
    os.makedirs(OUT, exist_ok=True)
    lines = ['Fine-tuning (200 epochs; the final epoch is the released model). Validation = the 8 validation subjects.',
             '', f'{"run":4s} {"best valid acc (epoch)":>24s} {"valid acc, epoch 200":>21s} {"train acc, epoch 200":>21s} '
             f'{"valid loss, epoch 1 -> 200":>27s}']
    for r, c in runs.items():
        ft = c['finetune']
        best = max(ft, key=lambda x: x[3])
        lines.append(f'{r:<4d} {best[3]:17.3f} ({best[0]:3d}) {ft[-1][3]:21.3f} {ft[-1][1]:21.3f} '
                     f'{ft[0][2]:16.1f} -> {ft[-1][2]:.1f}')
    open(os.path.join(OUT, 'curves.txt'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    TRAIN, VALID = '#2a78d6', '#eb6834'          # validated categorical slots 1-2
    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'font.family': 'DejaVu Sans'})
    fig, ax = plt.subplots(figsize=(7.5, 3.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for r, c in runs.items():
        ep = [x[0] for x in c['finetune']]
        ax.plot(ep, [x[1] for x in c['finetune']], color=TRAIN, lw=1.5, alpha=0.8)
        ax.plot(ep, [x[3] for x in c['finetune']], color=VALID, lw=1.5, alpha=0.8)
        best = max(c['finetune'], key=lambda x: x[3])
        ax.scatter([best[0]], [best[3]], s=36, color=VALID, edgecolors=SURFACE, linewidths=1.5, zorder=3)
    ax.axvline(200, color=MUTED, lw=1, ls='--')
    ax.annotate('released model\n(final epoch)', (200, 0.80), xytext=(-6, 0), textcoords='offset points', ha='right',
                va='center', fontsize=8.5, color=INK)
    ax.set_xlabel('fine-tuning epoch')
    ax.set_ylabel('accuracy')
    ax.set_xlim(0, 205)
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.text(110, 0.955, 'training (62 subjects)', fontsize=9, color=INK, ha='left', va='top')   # direct labels
    ax.text(110, 0.785, 'validation (8 subjects)', fontsize=9, color=INK, ha='left', va='bottom')
    ax.set_title('DeepSleepNet (released code), Sleep-EDF-78: 3 runs; dots: best validation epoch', loc='left',
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_curves.png'), dpi=200, facecolor=SURFACE)


if __name__ == '__main__':
    main()
