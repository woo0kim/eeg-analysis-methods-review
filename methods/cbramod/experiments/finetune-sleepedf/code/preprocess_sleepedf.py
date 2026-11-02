"""Sleep-EDF Expanded (Sleep Cassette, 78 subjects, 153 nights) -> the on-disk format of CBraMod's ISRUC pipeline.

CBraMod's released sleep-staging preprocessing (../../../upstream/preprocessing/ISRUC/prepare_ISRUC_1.py) filters
0.3-35 Hz (FIR), notches 50 Hz, cuts 30-s epochs at 200 Hz and saves non-overlapping sequences of 20 consecutive
epochs as seq/<recording>/<recording>-<k>.npy (20, channels, 6000) with labels/<recording>/<recording>-<k>.npy (20,).
This script writes the same layout for Sleep-EDF. What differs from the ISRUC script, and why:

  channels    the two EEG derivations Fpz-Cz and Pz-Oz (ISRUC: six EEG derivations). EOG/EMG are not EEG.
  sampling    100 Hz in the files -> resampled to 200 Hz (CBraMod and LaBraM embed 200-sample patches). The resampling
              comes first, so the 0.3-35 Hz FIR filter is designed at 200 Hz exactly as in the ISRUC script.
  notch       none: the recordings have no content above their 50 Hz Nyquist limit, and the 35 Hz low-pass is applied.
  labels      Rechtschaffen & Kales stages -> AASM: W 0, S1 1, S2 2, S3+S4 3, REM 4 (ISRUC's labels are already
              AASM). 'Movement time' and 'Sleep stage ?' epochs are dropped, and sequences never span such a gap.
  wake        each Sleep Cassette file covers ~20 h; as in DeepSleepNet's prepare_physionet.py and most Sleep-EDF work,
              only the sleep period plus 30 min of wake before the first and after the last sleep epoch is kept.
  units       microvolts, stored as float32 (ISRUC: MNE's to_data_frame(), also microvolts, as float64).
  split       ISRUC splits subjects 1-80 / 81-90 / 91-100 by index. Sleep-EDF subject IDs follow age (IDs 0-19 are
              25-34 years old, IDs 60-77 about 90), so an index split would test only on the oldest subjects. Here
              subjects are sorted by age, cut into blocks of 10 and, within each block, one goes to validation and
              one to test (fixed RNG seed): 62 / 8 / 8 subjects. A subject's two nights stay in one split.

usage: python preprocess_sleepedf.py  [SLEEPEDF_RAW=$EEG_DATA_ROOT/sleep_edfx  SLEEPEDF_OUT=$EEG_DATA_ROOT/sleep_edfx/processed_seq20]
"""
import glob
import json
import os

from collections import Counter

import mne
import numpy as np

D = os.path.expanduser(os.environ.get('EEG_DATA_ROOT', '~/data'))
RAW = os.path.expanduser(os.environ.get('SLEEPEDF_RAW', f'{D}/sleep_edfx'))
OUT = os.path.expanduser(os.environ.get('SLEEPEDF_OUT', f'{RAW}/processed_seq20'))
CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz']
FS, EPOCH_S, SEQ_LEN = 200, 30, 20
WAKE_MARGIN_S = 30 * 60
STAGE = {'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2, 'Sleep stage 3': 3, 'Sleep stage 4': 3,
         'Sleep stage R': 4}                  # 'Movement time' and 'Sleep stage ?' -> dropped (-1)
SPLIT_SEED = 0


def epoch_labels(annot, n_epochs_signal):
    """One label per 30-s epoch from the hypnogram's (onset, duration, description) segments; -1 = not scored."""
    lab = np.full(n_epochs_signal, -1, dtype=np.int64)
    for onset, dur, desc in zip(annot.onset, annot.duration, annot.description):
        assert onset % EPOCH_S == 0 and dur % EPOCH_S == 0, (onset, dur, desc)
        i0, n = int(onset // EPOCH_S), int(dur // EPOCH_S)
        lab[i0:min(i0 + n, n_epochs_signal)] = STAGE.get(desc, -1)
    return lab


def process_night(psg, hyp):
    raw = mne.io.read_raw_edf(psg, include=CHANNELS, preload=True, verbose='ERROR')
    assert raw.ch_names == CHANNELS and raw.info['sfreq'] == 100, (raw.ch_names, raw.info['sfreq'])
    annot = mne.read_annotations(hyp)
    n_ep = int(raw.n_times / raw.info['sfreq'] // EPOCH_S)
    lab = epoch_labels(annot, n_ep)
    sleep = np.where(lab > 0)[0]
    first, last = max(sleep[0] - WAKE_MARGIN_S // EPOCH_S, 0), min(sleep[-1] + WAKE_MARGIN_S // EPOCH_S, n_ep - 1)
    # crop with a 60-s margin on each side so the filter and resampler edges fall outside the kept epochs
    t0 = max(first * EPOCH_S - 60, 0)
    t1 = min((last + 1) * EPOCH_S + 60, raw.n_times / raw.info['sfreq'] - 1 / raw.info['sfreq'])
    raw.crop(t0, t1)
    raw.resample(FS, verbose='ERROR')
    raw.filter(0.3, 35, fir_design='firwin', verbose='ERROR')        # as prepare_ISRUC_1.py
    x = raw.get_data(units='uV').astype(np.float32)                   # (2, samples) at 200 Hz
    off = int(round(t0 * FS))                                          # sample index of the crop start
    seqs, labs, run = [], [], []

    def flush():
        for k in range(len(run) // SEQ_LEN):                           # remainder dropped, as in the ISRUC script
            idx = run[k * SEQ_LEN:(k + 1) * SEQ_LEN]
            seqs.append(np.stack([x[:, i * EPOCH_S * FS - off:(i + 1) * EPOCH_S * FS - off] for i in idx]))
            labs.append(lab[idx])
        run.clear()

    for i in range(first, last + 1):
        if lab[i] < 0:
            flush()                                                    # unscored / movement epoch breaks the sequence
        else:
            run.append(i)
    flush()
    return np.stack(seqs), np.stack(labs), lab[first:last + 1]


def subject_split(meta):
    """Age-stratified fixed split: blocks of 10 subjects by age, one to val and one to test per block."""
    rng = np.random.RandomState(SPLIT_SEED)
    subs = sorted(meta, key=lambda s: (meta[s]['age'], s))
    split = {'train': [], 'val': [], 'test': []}
    for b in range(0, len(subs), 10):
        block = subs[b:b + 10]
        perm = rng.permutation(len(block))
        split['val'].append(block[perm[0]])
        split['test'].append(block[perm[1]])
        split['train'] += [block[j] for j in perm[2:]]
    return {k: sorted(v) for k, v in split.items()}


def read_meta():
    """Age and sex per subject from SC-subjects.xls (needs xlrd), else from the EDF headers' patient field."""
    try:
        import pandas as pd
        d = pd.read_excel(os.path.join(RAW, 'SC-subjects.xls'))
        d.columns = [c.strip().lower() for c in d.columns]
        return {int(r.subject): {'age': int(r.age), 'female': int(r['sex (f=1)'])} for _, r in d.iterrows()}
    except ImportError:
        return json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../results/sc_subjects.json')),
                         object_hook=lambda o: {int(k) if k.isdigit() else k: v for k, v in o.items()})


def main():
    psgs = sorted(glob.glob(os.path.join(RAW, 'sleep-cassette', 'SC4*-PSG.edf')))
    hyps = sorted(glob.glob(os.path.join(RAW, 'sleep-cassette', 'SC4*-Hypnogram.edf')))
    hyp_of = {os.path.basename(h)[:6]: h for h in hyps}                 # SC4ssN: subject ss, night N
    assert len(psgs) == 153 and len(hyps) == 153, (len(psgs), len(hyps))
    meta = read_meta()
    split = subject_split(meta)
    stats = {'nights': {}, 'split': split, 'channels': CHANNELS, 'fs': FS, 'seq_len': SEQ_LEN}
    for psg in psgs:
        rec = os.path.basename(psg)[:6]
        subj = int(rec[3:5])
        seqs, labs, kept = process_night(psg, hyp_of[rec])
        for sub in ('seq', 'labels'):
            os.makedirs(os.path.join(OUT, sub, rec), exist_ok=True)
        for k, (s, l) in enumerate(zip(seqs, labs)):
            np.save(os.path.join(OUT, 'seq', rec, f'{rec}-{k:04d}.npy'), s)
            np.save(os.path.join(OUT, 'labels', rec, f'{rec}-{k:04d}.npy'), l)
        stats['nights'][rec] = {'subject': subj, 'n_seq': len(seqs), 'n_epochs_in_seqs': int(labs.size),
                                'n_epochs_kept_window': int(kept.size),
                                'n_unscored_in_window': int((kept < 0).sum()),
                                'class_counts': {int(c): int(n) for c, n in sorted(Counter(labs.ravel()).items())}}
        print(rec, seqs.shape, stats['nights'][rec]['class_counts'], flush=True)
    for part in ('train', 'val', 'test'):
        c = Counter()
        for rec, v in stats['nights'].items():
            if v['subject'] in split[part]:
                c.update({int(k): n for k, n in v['class_counts'].items()})
        stats[f'{part}_class_counts'] = dict(sorted(c.items()))
        stats[f'{part}_n_seq'] = sum(v['n_seq'] for v in stats['nights'].values() if v['subject'] in split[part])
        print(part, len(split[part]), 'subjects', stats[f'{part}_n_seq'], 'sequences', stats[f'{part}_class_counts'])
    stats['ages'] = {p: [meta[s]['age'] for s in split[p]] for p in split}
    json.dump(stats, open(os.path.join(OUT, 'stats.json'), 'w'), indent=1)
    json.dump(split, open(os.path.join(OUT, 'split.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
