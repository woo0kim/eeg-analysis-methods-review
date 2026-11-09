# Copied from ../../../upstream/finetune_trainer.py (MIT, CBraMod authors) through ../../finetune-sleepedf/code/seq_trainer.py:
# Trainer.__init__ and train_for_multiclass, the path the released code takes for SEED-V. Changed lines are marked # [MOD]:
#   - the test set is also scored after every epoch and every metric is written to metrics.json, so that the effect
#     of the epoch-selection rule can be measured; the reported number is the released one (test score at the epoch
#     with the best validation kappa), and test scores never influence training or selection.
#   - the selected epoch's test predictions are saved (test_preds.npz) instead of a checkpoint.
#   - the best validation kappa starts below any possible value, so a model that never beats chance still gets scored;
#     for every run whose kappa ever exceeds 0 this changes nothing.
#   - gradient accumulation (accum micro-batches per optimizer step): kept from seq_trainer.py, for the models a batch
#     of 64 does not fit on 12 GB. With accum = 1, the released update.
import copy
import json  # [MOD]
import os
from timeit import default_timer as timer

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from finetune_evaluator import Evaluator


def _predict(model, loader):  # [MOD] predictions for test_preds.npz (same forward pass as Evaluator)
    model.eval()
    truths, preds = [], []
    for x, y in loader:
        truths.append(y.numpy().reshape(-1))
        preds.append(torch.max(model(x.cuda()), dim=-1)[1].cpu().numpy().reshape(-1))
    return np.concatenate(truths), np.concatenate(preds)


class Trainer(object):
    def __init__(self, params, data_loader, model):
        self.params = params
        self.data_loader = data_loader

        self.val_eval = Evaluator(params, self.data_loader['val'])
        self.test_eval = Evaluator(params, self.data_loader['test'])

        self.model = model.cuda()
        self.criterion = CrossEntropyLoss(label_smoothing=self.params.label_smoothing).cuda()  # SEED-V branch

        self.best_model_states = None

        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)

                if params.frozen:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                other_params.append(param)

        if self.params.optimizer == 'AdamW':
            if self.params.multi_lr: # set different learning rates for different modules
                self.optimizer = torch.optim.AdamW([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': 0.001*(self.params.batch_size/256)**0.5}
                ], weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr,
                                                   weight_decay=self.params.weight_decay)

        self.data_length = len(self.data_loader['train']) // self.params.accum  # [MOD] optimizer steps per epoch
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.params.epochs * self.data_length, eta_min=1e-6
        )
        print(self.model)
        n_bb = sum(p.numel() for p in backbone_params)  # [MOD]
        n_other = sum(p.numel() for p in other_params)  # [MOD]
        print(f'parameters: backbone {n_bb:,}, other {n_other:,}, total {n_bb + n_other:,}')  # [MOD]

    def train_for_multiclass(self):
        f1_best = 0
        kappa_best = -2  # [MOD] released: 0, which keeps no checkpoint and crashes if validation kappa never exceeds 0
        acc_best = 0
        cm_best = None
        history = []  # [MOD]
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            self.optimizer.zero_grad()  # [MOD] zero once per optimizer step (below), not per micro-batch
            for step, (x, y) in enumerate(tqdm(self.data_loader['train'], mininterval=10)):
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)
                loss = self.criterion(pred, y)

                (loss / self.params.accum).backward()  # [MOD]
                losses.append(loss.data.cpu().numpy())
                if (step + 1) % self.params.accum:  # [MOD] accumulate until a full batch of 64 sequences
                    continue
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()
                self.optimizer.zero_grad()  # [MOD]
            self.optimizer.zero_grad()  # [MOD] an incomplete last batch is dropped

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                acc, kappa, f1, cm = self.val_eval.get_metrics_for_multiclass(self.model)
                t_acc, t_kappa, t_f1, t_cm = self.test_eval.get_metrics_for_multiclass(self.model)  # [MOD]
                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        kappa,
                        f1,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                print(cm)
                history.append({'epoch': epoch + 1, 'train_loss': float(np.mean(losses)),  # [MOD]
                                'val': {'balanced_acc': acc, 'kappa': kappa, 'weighted_f1': f1, 'cm': cm.tolist()},
                                'test': {'balanced_acc': t_acc, 'kappa': t_kappa, 'weighted_f1': t_f1,
                                         'cm': t_cm.tolist()},
                                'minutes': (timer() - start_time) / 60})
                json.dump(history, open(os.path.join(self.params.model_dir, 'metrics.json'), 'w'))  # [MOD]
                if kappa > kappa_best:
                    print("kappa increasing....saving weights !! ")
                    print("Val Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                        acc,
                        kappa,
                        f1,
                    ))
                    best_f1_epoch = epoch + 1
                    acc_best = acc
                    kappa_best = kappa
                    f1_best = f1
                    cm_best = cm
                    self.best_model_states = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            acc, kappa, f1, cm = self.test_eval.get_metrics_for_multiclass(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                    acc,
                    kappa,
                    f1,
                )
            )
            print(cm)
            truths, preds = _predict(self.model, self.data_loader['test'])  # [MOD] replaces torch.save(checkpoint)
            np.savez_compressed(os.path.join(self.params.model_dir, 'test_preds.npz'), y=truths, pred=preds,
                                best_epoch=best_f1_epoch)
            print("best epoch {} (validation kappa {:.5f})".format(best_f1_epoch, kappa_best))  # [MOD]
