"""Train one model on Sleep-EDF through CBraMod's released sleep-staging pipeline (its ISRUC path).

Hyper-parameters are finetune_main.py's defaults (= CBraMod paper Table 6), the values every earlier CBraMod-pipeline
run in this repository used: 50 epochs, batch 64, AdamW lr 1e-4 (backbone) / 5e-4 (rest, multi_lr), weight decay
5e-2, gradient clipping 1, label smoothing 0.1, cosine schedule, checkpoint = best validation kappa. The seed setup is
finetune_main.setup_seed().

usage: python run_seq.py --model <name> --seed S --datasets_dir D --model_dir OUT [--accum K] [--cuda 0]
       names: see seq_models.build()
"""
import argparse
import os
import random
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import seq_models  # noqa: E402
sys.path.insert(0, seq_models.CBRAMOD_SRC)     # finetune_evaluator (imported by seq_trainer)
from seq_trainer import Trainer  # noqa: E402
import sleepedf_dataset  # noqa: E402


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
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--accum', type=int, default=1, help='micro-batches per batch of 64 (memory only)')
    ap.add_argument('--cuda', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--num_workers', type=int, default=4)
    a = ap.parse_args()
    params = argparse.Namespace(   # finetune_main.py defaults
        seed=a.seed, cuda=a.cuda, epochs=a.epochs, batch_size=64, lr=1e-4, weight_decay=5e-2, optimizer='AdamW',
        clip_value=1, dropout=0.1, classifier='all_patch_reps', downstream_dataset='ISRUC',
        datasets_dir=a.datasets_dir, num_of_classes=5, model_dir=a.model_dir, num_workers=a.num_workers,
        label_smoothing=0.1, multi_lr=True, frozen=False, use_pretrained_weights=a.model != 'cbramod-scratch',
        foundation_dir='', accum=a.accum, model=a.model)
    assert params.batch_size % a.accum == 0
    print(params)
    os.makedirs(a.model_dir, exist_ok=True)
    setup_seed(a.seed)
    torch.cuda.set_device(a.cuda)
    data_loader = sleepedf_dataset.LoadDataset(params).get_data_loader()
    model = seq_models.build(a.model)
    encoder = getattr(model, model.enc_name)
    if hasattr(encoder, 'fit_norm'):             # DGCNN: band statistics of the training set
        encoder.fit_norm(data_loader['train'])
    t = Trainer(params, data_loader, model)
    t.train_for_multiclass()
    print('Done!!!!!')


if __name__ == '__main__':
    main()
