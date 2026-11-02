"""DeepSleepNet's input files (prepare_physionet.py format) for all 153 Sleep-EDF-78 nights, with the epochs of the
CBraMod-pipeline experiments.

The released prepare_physionet.py stops at SC4362F0: that night's hypnogram scores stages until 18,420 s after the
recording ends, its label-trimming step then removes every epoch and np.split fails (82 of 153 nights unwritten;
it was written for the 20-subject Sleep-EDF set). This script writes the same format instead: per night an .npz with
x (epochs, 3000, 1) = unfiltered Fpz-Cz in microvolts at 100 Hz, y = stage 0..4 (W, N1, N2, N3, REM), fs = 100,
named after the PSG file as the released script does (SC4ssNx0.npz). The epochs are those of
../../../../cbramod/experiments/finetune-sleepedf/code/preprocess_sleepedf.py (sleep period +- 30 min, scored epochs,
labels clipped at the end of the signal), the rule prepare_physionet.py also applies. keys.json records each night's
absolute epoch indices.

  python prepare_split_data.py check <released output dir>   # equality with prepare_physionet.py's files it did write
  python prepare_split_data.py                               # writes $DSN_DATA78_OURS/*.npz and keys.json
Env: SLEEPEDF_RAW ($EEG_DATA_ROOT/sleep_edfx), DSN_DATA78_OURS (default ~/repro-sleepmodels/data/dsn_sc78).
"""
import glob
import json
import os
import sys

import mne
import numpy as np

D = os.path.expanduser(os.environ.get('EEG_DATA_ROOT', '~/data'))
RAW = os.path.expanduser(os.environ.get('SLEEPEDF_RAW', f'{D}/sleep_edfx'))
OUT = os.path.expanduser(os.environ.get('DSN_DATA78_OURS', '~/repro-sleepmodels/data/dsn_sc78'))
STAGE = {'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2, 'Sleep stage 3': 3, 'Sleep stage 4': 3,
         'Sleep stage R': 4}


def night(psg):
    rec = os.path.basename(psg)[:6]
    hyp = glob.glob(f'{RAW}/sleep-cassette/{rec}*-Hypnogram.edf')[0]
    raw = mne.io.read_raw_edf(psg, preload=True, verbose='ERROR')
    raw.pick_channels(['EEG Fpz-Cz'])
    assert raw.info['sfreq'] == 100
    x = raw.get_data()[0] * 1e6
    n_ep = len(x) // 3000
    lab = np.full(n_ep, -1, dtype=np.int64)
    a = mne.read_annotations(hyp)
    for onset, dur, desc in zip(a.onset, a.duration, a.description):
        i0, n = int(onset // 30), int(dur // 30)
        lab[i0:min(i0 + n, n_ep)] = STAGE.get(desc, -1)
    sleep = np.where(lab > 0)[0]
    first, last = max(sleep[0] - 60, 0), min(sleep[-1] + 60, n_ep - 1)
    idx = np.arange(first, last + 1)
    idx = idx[lab[idx] >= 0]
    X = np.stack([x[i * 3000:(i + 1) * 3000] for i in idx]).astype(np.float32)[:, :, None]
    return os.path.basename(psg).replace('-PSG.edf', ''), X, lab[idx].astype(np.int32), idx


def check(released_dir):
    for f in sorted(glob.glob(f'{released_dir}/*.npz'))[:6]:
        name = os.path.basename(f)[:-4]
        _, X, y, _ = night(f'{RAW}/sleep-cassette/{name}-PSG.edf')
        r = np.load(f)
        same_n = len(r['y']) == len(y)
        print(f'{name}: released {len(r["y"])} epochs, ours {len(y)}; labels equal {same_n and np.array_equal(r["y"], y)}; '
              f'max |x diff| {np.abs(r["x"].squeeze() - X.squeeze()).max() if same_n else float("nan"):.2e} uV')


def main():
    os.makedirs(OUT, exist_ok=True)
    keys = {}
    for psg in sorted(glob.glob(f'{RAW}/sleep-cassette/SC4*-PSG.edf')):
        name, X, y, idx = night(psg)
        np.savez(f'{OUT}/{name}.npz', x=X, y=y, fs=100.0)
        keys[name[:6]] = idx.tolist()
        print(name, X.shape, np.bincount(y, minlength=5).tolist(), flush=True)
    json.dump(keys, open(f'{OUT}/keys.json', 'w'))


if __name__ == '__main__':
    check(sys.argv[2]) if sys.argv[1:2] == ['check'] else main()
