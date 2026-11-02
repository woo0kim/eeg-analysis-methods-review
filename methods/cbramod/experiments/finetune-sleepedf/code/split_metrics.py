"""Scoring for sleep models run outside the CBraMod pipeline on the same subject split (DeepSleepNet, L-SeqSleepNet).

Their predictions cover the kept epochs of the 8 test subjects' nights (DeepSleepNet: all but each night's last
n mod 25); the CBraMod-pipeline runs score the epochs inside complete 20-epoch sequences (epoch_keys.py).
score_run() reports a model on all of its own test epochs and on the common test set (epoch_keys.common_test_keys(),
the epochs every model scores), with the metric definitions of collect.py. The common-set numbers are the ones
compared across models; collect.py reports the pipeline runs on the same set.
"""
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score

import epoch_keys

STAGES = ['W', 'N1', 'N2', 'N3', 'REM']
_COMMON = None


def metrics(y, p):
    m = {'balanced_acc': balanced_accuracy_score(y, p), 'kappa': cohen_kappa_score(y, p),
         'weighted_f1': f1_score(y, p, average='weighted'), 'accuracy': accuracy_score(y, p),
         'macro_f1': f1_score(y, p, average='macro'), 'n_epochs': int(len(y))}
    for s, f in zip(STAGES, f1_score(y, p, average=None, labels=range(5))):
        m['f1_' + s] = f
    return m


def common_keys():
    global _COMMON
    if _COMMON is None:
        keys, labels, mask = epoch_keys.load_test_epochs()
        _COMMON = ([k for k, m in zip(keys, mask) if m], labels[mask])
    return _COMMON


def score_run(pred_by_key, label_by_key):
    """pred_by_key / label_by_key: {(rec, epoch index): stage} over all of the model's test epochs."""
    keys = sorted(pred_by_key)
    allm = metrics(np.array([label_by_key[k] for k in keys]), np.array([pred_by_key[k] for k in keys]))
    ck, cl = common_keys()
    missing = [k for k in ck if k not in pred_by_key]
    assert not missing, f'{len(missing)} common test epochs without a prediction, e.g. {missing[:3]}'
    y = np.array([label_by_key[k] for k in ck])
    assert np.array_equal(y, cl), 'labels differ from the common test set'
    return {'common': metrics(y, np.array([pred_by_key[k] for k in ck])), 'all': allm}
