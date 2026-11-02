"""L-SeqSleepNet's input files for all 153 Sleep-EDF-78 nights, with the epochs of the CBraMod-pipeline experiments.

Port of the released preparation (github.com/pquochuy/xsleepnet, sleepedf-20/data_processing/prepare_raw_data_30min.m
and prepare_data_30min.m, the scripts L-SeqSleepNet's README points to), EEG channel only:
  - Fpz-Cz at 100 Hz in microvolts, unfiltered: the released mat_30min files hold the EDF samples exactly
    (`check` below), although prepare_raw_data_30min.m band-passes 0.3-40 Hz (fir1(100) + filtfilt); the port follows
    the released files, the data L-SeqSleepNet was trained and evaluated on;
  - 30-s epochs; X1 = the epoch (3000 samples); X2 = 20*log10|spectrogram| with a 200-sample Hamming
    window, 100-sample overlap and a 256-point FFT -> 29 frames x 129 bins; label 1..5 (W, N1, N2, N3, REM), y one-hot;
  - unscored / movement epochs deleted, and the remaining epochs concatenated (as the released script does);
  - saved as MATLAB v7.3 files with the released variable names and orientation (hdf5storage).
One deliberate difference: the released script keeps lights-off - 30 min to lights-on + 30 min (times it derives
from the hypnogram files); here each night keeps the epochs of the CBraMod-pipeline experiments (sleep period +- 30 min,
the same selection as ../../../../cbramod/experiments/finetune-sleepedf/code/preprocess_sleepedf.py), so every model is
scored on the same test epochs. keys.json records each night's absolute epoch indices.

  python prepare_split_data.py check      # compares the port with the released mat_30min files (X1, X2) of 2 nights
  python prepare_split_data.py            # writes $OUT/mat/*.mat, $OUT/{train,eval,test}_list.txt, $OUT/keys.json
Env: SLEEPEDF_RAW ($EEG_DATA_ROOT/sleep_edfx), LSEQ_SRC (released repo, for `check`), LSEQ_DATA78 (output dir).
"""
import glob
import json
import os
import sys

import h5py
import hdf5storage
import mne
import numpy as np
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.expanduser(os.environ.get('EEG_DATA_ROOT', '~/data'))
RAW = os.path.expanduser(os.environ.get('SLEEPEDF_RAW', f'{D}/sleep_edfx'))
OUT = os.path.expanduser(os.environ.get('LSEQ_DATA78', '~/repro-sleepmodels/data/lseq_sc78'))
LSEQ_SRC = os.path.expanduser(os.environ.get('LSEQ_SRC', '~/repro-sleepmodels/lseqsleepnet-upstream'))
SPLIT = json.load(open(os.path.join(HERE, '../../../../cbramod/experiments/finetune-sleepedf/results/split.json')))
FS, EPOCH = 100, 3000
STAGE = {'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2, 'Sleep stage 3': 3, 'Sleep stage 4': 3,
         'Sleep stage R': 4}
WIN = signal.windows.hamming(200, sym=True)                                                # = hamming(200)


def fpzcz(psg):
    raw = mne.io.read_raw_edf(psg, preload=True, verbose='ERROR')
    raw.pick_channels(['EEG Fpz-Cz'])
    assert raw.info['sfreq'] == FS
    return raw.get_data()[0] * 1e6                                               # volts -> microvolts


def stft(x1):
    """MATLAB spectrogram(x, hamming(200), 100, 256): 29 frames x 129 bins, log magnitude in dB."""
    frames = np.stack([x1[..., t * 100:t * 100 + 200] for t in range(29)], axis=-2)   # (n, 29, 200)
    return (20 * np.log10(np.abs(np.fft.rfft(frames * WIN, 256, axis=-1)))).astype(np.float32)


def epoch_selection(hyp, n_ep):
    """Kept epochs of preprocess_sleepedf.py: sleep period +- 30 min, scored epochs only (absolute indices)."""
    annot = mne.read_annotations(hyp)
    lab = np.full(n_ep, -1, dtype=np.int64)
    for onset, dur, desc in zip(annot.onset, annot.duration, annot.description):
        i0, n = int(onset // 30), int(dur // 30)
        lab[i0:min(i0 + n, n_ep)] = STAGE.get(desc, -1)
    sleep = np.where(lab > 0)[0]
    first, last = max(sleep[0] - 60, 0), min(sleep[-1] + 60, n_ep - 1)
    idx = np.arange(first, last + 1)
    idx = idx[lab[idx] >= 0]
    return idx, lab[idx]


def check():
    for rec, mat in (('SC4001E0', 'n01_1_eeg.mat'), ('SC4192E0', 'n20_2_eeg.mat')):
        x = fpzcz(f'{RAW}/sleep-cassette/{rec}-PSG.edf')
        f = h5py.File(f'{LSEQ_SRC}/sleepedf-20/mat_30min/{mat}', 'r')
        X1 = np.array(f['X1']).T
        X2 = np.array(f['X2']).transpose(2, 1, 0)
        # locate the released first epoch in the signal (the lights-off trimming is not ported)
        c = signal.correlate(x[:len(x) // 2], X1[0], mode='valid', method='fft')
        off = int(np.argmax(c))
        n = min(len(X1), (len(x) - off) // EPOCH)
        mine = x[off:off + n * EPOCH].reshape(n, EPOCH)
        ok = np.all(np.abs(np.diff(off + np.arange(n) * EPOCH)) == EPOCH)
        d1 = np.abs(mine[:200] - X1[:200]).max()                 # first 200 epochs (no deleted epoch inside)
        d2 = np.abs(stft(mine[:200]) - X2[:200]).max()
        print(f'{rec} vs released {mat}: offset {off} samples ({off / EPOCH:.2f} epochs), contiguous {ok}; '
              f'max |X1 diff| {d1:.2e} uV (signal SD {X1.std():.1f}), max |X2 diff| {d2:.2e} dB')


def main():
    os.makedirs(f'{OUT}/mat', exist_ok=True)
    keys, lists = {}, {'train': [], 'eval': [], 'test': []}
    part_of = {s: p for p, k in (('train', 'train'), ('eval', 'val'), ('test', 'test')) for s in SPLIT[k]}
    for psg in sorted(glob.glob(f'{RAW}/sleep-cassette/SC4*-PSG.edf')):
        rec = os.path.basename(psg)[:6]
        hyp = glob.glob(f'{RAW}/sleep-cassette/{rec}*-Hypnogram.edf')[0]
        x = fpzcz(psg)
        idx, lab = epoch_selection(hyp, len(x) // EPOCH)
        X1 = np.stack([x[i * EPOCH:(i + 1) * EPOCH] for i in idx]).astype(np.float32)
        X2 = stft(X1)
        label = (lab + 1).astype(np.float32)                                  # 1..5 as the released files
        y = np.eye(5, dtype=np.float32)[lab]
        path = f'{OUT}/mat/{rec}_eeg.mat'
        hdf5storage.savemat(path, {'X1': X1, 'X2': X2, 'label': label[:, None], 'y': y},
                            format='7.3', matlab_compatible=True)
        keys[rec] = idx.tolist()
        lists[part_of[int(rec[3:5])]].append(f'{path}\t{len(idx)}')
        print(rec, X2.shape, np.bincount(lab, minlength=5).tolist(), flush=True)
    for p, lines in lists.items():
        open(f'{OUT}/{p}_list.txt', 'w').write('\n'.join(lines) + '\n')
    json.dump(keys, open(f'{OUT}/keys.json', 'w'))


if __name__ == '__main__':
    check() if sys.argv[1:] == ['check'] else main()
