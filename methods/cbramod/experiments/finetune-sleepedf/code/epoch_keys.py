"""Which 30-s epochs each Sleep-EDF experiment scores, as (recording, absolute epoch index) keys.

The CBraMod-pipeline experiments score the epochs of complete 20-epoch sequences (preprocess_sleepedf.py drops
each run's remainder and breaks runs at unscored epochs), in the loader's order: test subjects in split.json order,
each subject's nights sorted, sequence files sorted. L-SeqSleepNet on the same split scores every kept epoch of the
test nights; DeepSleepNet's released predict.py scores the first floor(n / 25) * 25 kept epochs of each night (it
predicts in 25-epoch sequences and drops the rest). common_test_keys() is the set every model scores: the pipeline's
test epochs minus those DeepSleepNet leaves out.
Recomputed from the hypnograms alone (no signal), with the same rules as preprocess_sleepedf.py.
`python epoch_keys.py` writes ../results/test_epochs.json (pipeline test keys in test_preds.npz order, their labels and
the common-set mask), which the analysis scripts read, so they need no raw data.
"""
import glob
import json
import os

import mne
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.expanduser(os.environ.get('EEG_DATA_ROOT', '~/data'))
RAW = os.path.expanduser(os.environ.get('SLEEPEDF_RAW', f'{D}/sleep_edfx'))
SPLIT = json.load(open(os.path.join(HERE, '../results/split.json')))
STAGE = {'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2, 'Sleep stage 3': 3, 'Sleep stage 4': 3,
         'Sleep stage R': 4}


def night_labels(rec):
    """Per-epoch labels (-1 = not scored) of a night, clipped to the signal length, and the kept window."""
    psg = glob.glob(f'{RAW}/sleep-cassette/{rec}*-PSG.edf')[0]
    hyp = glob.glob(f'{RAW}/sleep-cassette/{rec}*-Hypnogram.edf')[0]
    n_ep = int(mne.io.read_raw_edf(psg, verbose='ERROR').n_times / 100 // 30)
    lab = np.full(n_ep, -1, dtype=np.int64)
    a = mne.read_annotations(hyp)
    for onset, dur, desc in zip(a.onset, a.duration, a.description):
        i0, n = int(onset // 30), int(dur // 30)
        lab[i0:min(i0 + n, n_ep)] = STAGE.get(desc, -1)
    sleep = np.where(lab > 0)[0]
    return lab, max(sleep[0] - 60, 0), min(sleep[-1] + 60, n_ep - 1)


def pipeline_keys(rec, seq_len=20):
    """Epoch indices of the night's 20-epoch sequences, in file order (preprocess_sleepedf.py)."""
    lab, first, last = night_labels(rec)
    keys, run = [], []
    for i in range(first, last + 2):
        if i <= last and lab[i] >= 0:
            run.append(i)
            continue
        keys += run[:len(run) // seq_len * seq_len]
        run = []
    return keys, lab


def kept_keys(rec):
    """All scored epochs of the night's kept window, in order (the epochs DeepSleepNet / L-SeqSleepNet receive)."""
    lab, first, last = night_labels(rec)
    idx = [i for i in range(first, last + 1) if lab[i] >= 0]
    return idx, lab


def test_records():
    recs = sorted({os.path.basename(p)[:6] for p in glob.glob(f'{RAW}/sleep-cassette/SC4*-PSG.edf')})
    return [r for s in SPLIT['test'] for r in recs if int(r[3:5]) == s]


def pipeline_test_keys():
    """(rec, epoch) of every epoch in a CBraMod-pipeline run's test_preds.npz, in its order, and the labels."""
    keys, labels = [], []
    for rec in test_records():
        k, lab = pipeline_keys(rec)
        keys += [(rec, i) for i in k]
        labels += lab[k].tolist()
    return keys, np.array(labels)


def common_test_keys():
    """Mask over pipeline_test_keys() of the epochs every model scores, and those (rec, epoch) keys."""
    covered = set()
    for rec in test_records():
        idx, _ = kept_keys(rec)
        covered |= {(rec, i) for i in idx[:len(idx) // 25 * 25]}                # DeepSleepNet's predict.py
    keys, labels = pipeline_test_keys()
    mask = np.array([k in covered for k in keys])
    return mask, [k for k, m in zip(keys, mask) if m], labels[mask]


TEST_EPOCHS = os.path.join(HERE, '../results/test_epochs.json')


def load_test_epochs():
    """(pipeline keys, labels, common mask) from ../results/test_epochs.json."""
    d = json.load(open(TEST_EPOCHS))
    return [tuple(k) for k in d['pipeline_keys']], np.array(d['labels']), np.array(d['common_mask'], dtype=bool)


if __name__ == '__main__':
    k, y = pipeline_test_keys()
    print(len(k), 'test epochs in 20-epoch sequences;', np.bincount(y, minlength=5).tolist())
    m, ck, cy = common_test_keys()
    print(len(ck), 'of them also scored by DeepSleepNet (common test set);', np.bincount(cy, minlength=5).tolist())
    json.dump({'pipeline_keys': [list(x) for x in k], 'labels': y.tolist(), 'common_mask': m.astype(int).tolist()},
              open(TEST_EPOCHS, 'w'))
