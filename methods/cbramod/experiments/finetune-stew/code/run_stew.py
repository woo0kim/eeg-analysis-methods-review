"""Train one model on one STEW fold through CBraMod's released multi-class training path (../../finetune-seed/code/trainer.py).

finetune_main.py's defaults, as in every CBraMod-pipeline run here: 50 epochs, batch 64, AdamW lr 1e-4 (backbone) / 5e-4 (rest,
multi_lr), weight decay 5e-2, clipping 1, label smoothing 0.1, cosine schedule, checkpoint = best validation kappa. Both tasks
use this path, the two-class one too, so kappa and accuracy (the paper's metrics) exist for all.

usage: python run_stew.py --model M --task rest-task --fold 0 --seed 0 --root R --folds results/folds.json --model_dir OUT
"""
import argparse
import os
import random
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stew_data  # noqa: E402
import stew_models  # noqa: E402
sys.path.insert(0, stew_models.em.CBRAMOD_SRC)
from trainer import Trainer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    for a_ in ('--model', '--task', '--root', '--folds', '--model_dir'):
        ap.add_argument(a_, required=True)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--epochs', type=int, default=50)
    a = ap.parse_args()
    k = {'rest-task': 2, 'rating3': 3}[a.task]
    params = argparse.Namespace(seed=a.seed, cuda=0, epochs=a.epochs, batch_size=64, lr=1e-4, weight_decay=5e-2, optimizer='AdamW',
                                clip_value=1, dropout=0.1, classifier='all_patch_reps', downstream_dataset='SEED-V', datasets_dir=a.root,
                                num_of_classes=k, model_dir=a.model_dir, num_workers=0, label_smoothing=0.1, multi_lr=True, frozen=False,
                                use_pretrained_weights=False, foundation_dir='', accum=1, model=a.model)
    print(params)
    os.makedirs(a.model_dir, exist_ok=True)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(0)
    loader = stew_data.get_loaders(a.root, a.folds, a.task, a.fold, a.model in stew_models.FOUNDATION)
    model = stew_models.build(a.model, k)
    if hasattr(model, 'fit_norm'):
        model.fit_norm(loader['train'], n_batches=30)
    Trainer(params, loader, model).train_for_multiclass()
    print('Done!!!!!')


if __name__ == '__main__':
    main()
