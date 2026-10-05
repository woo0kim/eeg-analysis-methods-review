"""Fine-tune LaBraM-Base with its released run_class_finetuning.py on CBraMod's preprocessed data and splits.

The released script (../../../upstream/run_class_finetuning.py) runs unmodified. Two hooks are installed
around it:
  1. get_dataset() reads the LMDB files written by CBraMod's preprocessing, i.e. the same samples and the
     same subject-independent train/val/test split CBraMod is evaluated on, and gives LaBraM the channel
     names it needs for its electrode embedding.
  2. The full-set probabilities that LaBraM's evaluate() scores at the end of every validation/test pass are
     also saved to <output_dir>/preds.npz, so the test metrics can be computed with CBraMod's metric code at
     the epoch chosen by CBraMod's validation monitor (code/collect.py).

The only extra argument is --datasets_dir (the LMDB directory); --dataset takes a CBraMod dataset name.
Single-GPU runs are launched as a one-process distributed job (see add_jobs.py), so that LaBraM reshuffles
the training set every epoch as it does in its multi-GPU recipe; without RANK/WORLD_SIZE its
DistributedSampler never gets set_epoch() and repeats the same order every epoch.
"""
import argparse
import os
import pickle
import sys

import lmdb
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get('LABRAM_SRC', os.path.normpath(os.path.join(HERE, '../../../upstream')))
sys.path.insert(0, SRC)
import run_class_finetuning as rcf  # noqa: E402  (LaBraM's released fine-tuning script)
import utils  # noqa: E402  (LaBraM's utils, the module engine_for_finetuning calls)

# Channel order of each LMDB, copied from methods/cbramod/upstream/preprocessing/preprocessing_*.py and
# written in LaBraM's standard_1020 vocabulary (upper case, no reference suffix).
CHANNELS = {
    # preprocessing_physio.py selected_channels ('Fc5.' -> 'FC5', ...)
    'PhysioNet-MI': ['FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6',
                     'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'FP1', 'FPZ', 'FP2', 'AF7', 'AF3', 'AFZ',
                     'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FT8', 'T7', 'T8',
                     'T9', 'T10', 'TP7', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7',
                     'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2', 'IZ'],
    # preprocessing_bciciv2a.py keeps the first 22 columns of BNCI 001-2014, i.e. its documented EEG montage
    'BCIC-IV-2a': ['FZ', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'CP3',
                   'CP1', 'CPZ', 'CP2', 'CP4', 'P1', 'PZ', 'P2', 'POZ'],
    # preprocessing_mumtaz.py selected_channels ('EEG Fp1-LE' -> 'FP1', ...)
    'Mumtaz2016': ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5',
                   'T6', 'FZ', 'CZ', 'PZ'],
    # preprocessing_stress.py selected_channels minus its 20th channel 'EEG A2-A1': an ear-reference
    # derivation with no entry in LaBraM's electrode vocabulary, so it is dropped (19 of CBraMod's 20 channels)
    'MentalArithmetic': ['FP1', 'FP2', 'F3', 'F4', 'F7', 'F8', 'T3', 'T4', 'C3', 'C4', 'T5', 'T6', 'P3', 'P4',
                         'O1', 'O2', 'FZ', 'CZ', 'PZ'],
}
N_CLASSES = {'PhysioNet-MI': 4, 'BCIC-IV-2a': 4, 'Mumtaz2016': 1, 'MentalArithmetic': 1}   # 1 = binary (BCE)
# LaBraM's own metric lists for multi-class (TUEV) and binary (TUAB) tasks
METRICS = {4: ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"],
           1: ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]}


class CBraModLMDB(torch.utils.data.Dataset):
    """Same LMDB access as methods/cbramod/upstream/datasets/*_dataset.py, without CBraMod's division by 100:
    LaBraM's engine divides by 100 itself, so both models see the signal in units of 0.1 mV."""

    def __init__(self, path, mode, n_channels):
        self.db = lmdb.open(path, readonly=True, lock=False, readahead=True, meminit=False)
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get('__keys__'.encode()))[mode]
        self.n_channels = n_channels

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(self.keys[idx].encode()))
        x = pair['sample'][:self.n_channels]                       # (channels, patches, 200) at 200 Hz, in uV
        x = torch.FloatTensor(np.ascontiguousarray(x).reshape(x.shape[0], -1))   # LaBraM input: (channels, time)
        return x, pair['label']


def get_dataset(args):
    ch_names = CHANNELS[args.dataset]
    args.nb_classes = N_CLASSES[args.dataset]
    train, val, test = (CBraModLMDB(OURS.datasets_dir, mode, len(ch_names)) for mode in ('train', 'val', 'test'))
    print(f'{args.dataset}: train {len(train)}, val {len(val)}, test {len(test)}, {len(ch_names)} channels')
    return train, test, val, ch_names, METRICS[args.nb_classes]


# --- record what evaluate() scores -------------------------------------------------------------------------
_last = {}
_get_metrics = utils.get_metrics


def get_metrics(output, target, metrics, is_binary, threshold=None):
    # evaluate() calls get_metrics once per batch (no threshold) and once on the whole set with threshold=0.5
    if threshold is None:
        return _get_metrics(output, target, metrics, is_binary)
    _last['prob'], _last['y'] = np.asarray(output), np.asarray(target)
    return _get_metrics(output, target, metrics, is_binary, threshold)


_evaluate = rcf.evaluate
PREDS = {'val': [], 'test': []}


def evaluate(data_loader, model, device, header='Test:', **kw):
    stats = _evaluate(data_loader, model, device, header=header, **kw)
    split = 'val' if header.startswith('Val') else 'test'
    PREDS[split].append(_last['prob'].astype(np.float32))
    PREDS['y_' + split] = _last['y'].reshape(-1)
    np.savez_compressed(os.path.join(RCF_ARGS.output_dir, 'preds.npz'),
                        val=np.stack(PREDS['val']), test=np.stack(PREDS['test']) if PREDS['test'] else np.zeros(0),
                        y_val=PREDS['y_val'], y_test=PREDS.get('y_test', np.zeros(0)))
    return stats


utils.get_metrics = get_metrics
rcf.get_dataset = get_dataset
rcf.evaluate = evaluate

if __name__ == '__main__':
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)   # else '--dataset' would match --datasets_dir
    ap.add_argument('--datasets_dir', required=True)
    OURS, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    RCF_ARGS, ds_init = rcf.get_args()
    assert RCF_ARGS.output_dir, '--output_dir is required (preds.npz and LaBraM log.txt go there)'
    os.makedirs(RCF_ARGS.output_dir, exist_ok=True)
    rcf.main(RCF_ARGS, ds_init)
