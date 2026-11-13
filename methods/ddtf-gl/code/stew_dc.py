"""STEW's raw files keep the Emotiv DC level (about 4,300 uV). Does it differ between the rest and the task recording, consistently
across subjects? If so, a model on unfiltered windows can tell the recordings apart from the offset alone, without any EEG
content (the bundle reads the raw values; the shared STEW pipeline high-passes at 0.3 Hz).

For every subject and channel: mean over the task recording (sub*_hi) minus mean over the rest recording (sub*_lo), in uV; per
channel the median over subjects, how many of the 48 subjects have the same sign as the median, and a two-sided sign test.
Also the within-recording drift: |mean of the last 30 s - mean of the first 30 s|, median over recordings, for scale.

usage: python stew_dc.py > ../experiments/stew-paper-protocol/results/dc_offsets.txt      (needs STEW's raw files, datasets.STEW_RAW)
"""
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import CHANNELS, STEW_RAW  # noqa: E402

lv = {s: {st: np.loadtxt(os.path.join(STEW_RAW, f'sub{s:02d}_{st}.txt'), dtype=np.float64) for st in ('lo', 'hi')} for s in range(1, 49)}
diff = np.array([lv[s]['hi'].mean(0) - lv[s]['lo'].mean(0) for s in range(1, 49)])                      # (48, 14)
level = np.median([lv[s][st].mean() for s in lv for st in ('lo', 'hi')])
drift = np.median([np.abs(lv[s][st][-3840:].mean(0) - lv[s][st][:3840].mean(0)).mean() for s in lv for st in ('lo', 'hi')])
print(f'median DC level of a recording: {level:.0f} uV; median |last 30 s - first 30 s| within a recording: {drift:.1f} uV\n')
print('task - rest, mean level per channel (uV), over the 48 subjects')
print(f'{"channel":8s} {"median":>8s} {"IQR":>17s} {"same sign":>10s} {"sign test p":>12s}')
for c, name in enumerate(CHANNELS['STEW']):
    d = diff[:, c]
    med = np.median(d)
    same = int((np.sign(d) == np.sign(med)).sum())
    p = stats.binomtest(same, 48).pvalue
    q1, q3 = np.percentile(d, [25, 75])
    print(f'{name:8s} {med:8.1f} {f"[{q1:.1f}, {q3:.1f}]":>17s} {f"{same}/48":>10s} {p:12.2g}')
allsame = int((np.sign(np.median(diff, 1)) == np.sign(np.median(diff))).sum())
print(f'\nmedian over channels, per subject: same sign as the overall median in {allsame}/48 subjects '
      f'(sign test p = {stats.binomtest(allsame, 48).pvalue:.2g})')

# Slow activity: the SD over a recording's 49 windows (4 s, the bundle's segmentation) of the window mean, averaged over channels.
wm = {s: {st: np.array([lv[s][st][i:i + 512].mean(0) for i in range(0, 19200 - 511, 384)]) for st in ('lo', 'hi')} for s in lv}
sd = np.array([[wm[s][st].std(0).mean() for st in ('lo', 'hi')] for s in range(1, 49)])               # (48, 2)
more = int((sd[:, 1] > sd[:, 0]).sum())
print(f'\nSD of the 4-s window means within a recording (mean over channels): rest median {np.median(sd[:, 0]):.1f} uV, '
      f'task median {np.median(sd[:, 1]):.1f} uV; larger during the task in {more}/48 subjects (sign test p = {stats.binomtest(more, 48).pvalue:.2g})')
