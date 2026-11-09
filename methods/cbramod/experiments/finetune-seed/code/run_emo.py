"""Train one model on SEED or SEED-V through CBraMod's released pipeline (its SEED-V path).

Hyper-parameters are finetune_main.py's defaults (= CBraMod paper Table 6), as in every CBraMod-pipeline run in this
repository: 50 epochs, batch 64, AdamW lr 1e-4 (backbone) / 5e-4 (rest, multi_lr), weight decay 5e-2, gradient clipping
1, label smoothing 0.1, cosine schedule, checkpoint = best validation kappa. Loader: released seedv_dataset.

usage: python run_emo.py --model <name> --seed S --datasets_dir D --num_of_classes K --model_dir OUT

EVAL_BATCH (default 64, the released batch) sets the validation / test batch. ATCNet's later runs use 512: it is limited by kernel
launches, scoring took longer than training at 64, and in eval mode a prediction does not depend on the batch.
"""
import argparse
import os
import random
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import emo_models  # noqa: E402
sys.path.insert(0, emo_models.CBRAMOD_SRC)     # finetune_evaluator (trainer.py), datasets, models
from datasets import seedv_dataset  # noqa: E402
from trainer import Trainer  # noqa: E402


def setup_seed(seed):   # = finetune_main.setup_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--datasets_dir', required=True)
    ap.add_argument('--num_of_classes', type=int, required=True)
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--accum', type=int, default=1)
    ap.add_argument('--cuda', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=50)
    a = ap.parse_args()
    params = argparse.Namespace(   # finetune_main.py defaults
        seed=a.seed, cuda=a.cuda, epochs=a.epochs, batch_size=64 // a.accum * a.accum, lr=1e-4, weight_decay=5e-2,
        optimizer='AdamW', clip_value=1, dropout=0.1, classifier='all_patch_reps', downstream_dataset='SEED-V',
        datasets_dir=a.datasets_dir, num_of_classes=a.num_of_classes, model_dir=a.model_dir, num_workers=0,
        label_smoothing=0.1, multi_lr=True, frozen=False, use_pretrained_weights=False, foundation_dir='',
        accum=a.accum, model=a.model)
    print(params)
    os.makedirs(a.model_dir, exist_ok=True)
    setup_seed(a.seed)
    torch.cuda.set_device(a.cuda)
    loader = seedv_dataset.LoadDataset(params).get_data_loader()
    eval_batch = int(os.environ.get('EVAL_BATCH', 64))      # validation / test batch; predictions do not depend on it
    if eval_batch != params.batch_size:
        from torch.utils.data import DataLoader
        for k in ('val', 'test'):
            ds = loader[k].dataset
            loader[k] = DataLoader(ds, batch_size=eval_batch, collate_fn=ds.collate, shuffle=False)
        print('evaluation batch', eval_batch)
    model = emo_models.build(a.model, params)
    if hasattr(model, 'fit_norm'):
        model.fit_norm(loader['train'])
    t = Trainer(params, loader, model)
    t.train_for_multiclass()
    print('Done!!!!!')


if __name__ == '__main__':
    main()
