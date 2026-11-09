# Copied from methods/eeg-conformer/upstream/conformer_seed_1s_5fold.py at 9ae149b (GPL-3.0; this file is GPL-3.0 as well).
# Changed lines are marked # [MOD]:
#   - paths, one (subject, fold) per process, a fixed seed, and imports the model does not use removed
#   - the per-epoch test accuracies are saved to curves/ (the released log has them too)
#   - --split trial96 replaces the within-clip folds by a split between clips: first 9 clips train, last 6 test
#   - the per-epoch test pass runs under no_grad (memory only)
# With --split kfold everything that affects training is the released code.
"""
EEG conformer 

Test SEED data 1 second
perform strict 5-fold cross validation 
"""


import argparse
import os
gpus = [0]  # [MOD] the queue sets CUDA_VISIBLE_DEVICES
import numpy as np
import math
import glob
import random
import itertools
import datetime
import time
import datetime
import sys
import scipy.io

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.autograd as autograd

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init

from torch.utils.data import Dataset

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()

        self.eegnet = nn.Sequential(
            nn.Conv2d(1, 8, (1, 125), (1, 1)),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (22, 1), (1, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4), (1, 4)),
            nn.Dropout(0.5),
            nn.Conv2d(16, 16, (1, 16), (1, 1)),
            nn.BatchNorm2d(16), 
            nn.ELU(),
            nn.AvgPool2d((1, 8), (1, 8)),
            nn.Dropout2d(0.5)
        )

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (62, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # 5 is better than 1
            Rearrange('b e (h) (w) -> b (h w) e'),
        )


    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape

        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=5,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.cov = nn.Sequential(
            nn.Conv1d(190, 1, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.clshead_fc = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, 32),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(32, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(280, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        
        return x, out


# ! Rethink the use of Transformer for EEG signal
class ViT(nn.Sequential):
    def __init__(self, emb_size=40, depth=6, n_classes=4, **kwargs):
        super().__init__(

            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )


class ExGAN():
    def __init__(self, nsub, fold, root, out):  # [MOD]
        super(ExGAN, self).__init__()
        self.batch_size = 200
        self.n_epochs = 600  #1000
        self.img_height = 22
        self.img_width = 600
        self.channels = 1
        self.c_dim = 4
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.alpha = 0.0002
        self.dimension = (190, 50)
        self.nSub = nsub

        self.start_epoch = 0
        self.root = root  # [MOD]

        self.pretrain = False

        self.log_write = open(os.path.join(out, "log_subject%d_fold%d.txt" % (self.nSub, fold+1)), "w")  # [MOD]

        self.img_shape = (self.channels, self.img_height, self.img_width)

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.model = ViT().cuda()
        self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = self.model.cuda()

        self.centers = {}

    def interaug(self, timg, label):
        aug_data = []
        aug_label = []
        for cls4aug in range(3):
            cls_idx = np.where(label == cls4aug)
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]

            tmp_aug_data = np.zeros((int(self.batch_size / 3), 1, 62, 200))
            for ri in range(int(self.batch_size / 3)):
                for rj in range(8):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                    tmp_aug_data[ri, :, :, rj * 25:(rj + 1) * 25] = tmp_data[rand_idx[rj], :, :,
                                                                      rj * 25:(rj + 1) * 25]

            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label[:int(self.batch_size / 3)])
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data).cuda()
        aug_data = aug_data.float()
        aug_label = torch.from_numpy(aug_label).cuda()
        aug_label = aug_label.long()
        return aug_data, aug_label

    def get_source_data(self, fold):

        self.all_data = np.load(self.root + 'S%d_session1.npy' % self.nSub, allow_pickle=True)
        self.all_label = np.load(self.root + 'S%d_session1_label.npy' % self.nSub, allow_pickle=True)
        self.train_data = []
        self.train_label = []
        self.test_data = []
        self.test_label = []

        for tri in range(np.shape(self.all_data)[0]):
            tmp_tri = np.array(self.all_data[tri])
            tmp_tri_label = np.array(self.all_label[tri])

            one_fold_num = np.shape(tmp_tri)[0] // 5
            tri_num =  one_fold_num * 5
            tmp_tri_idx = np.arange(tri_num)
            test_idx = np.arange(one_fold_num * fold, one_fold_num * (fold+1))
            train_idx = np.delete(tmp_tri_idx, test_idx)
            if SPLIT == 'trial96':  # [MOD] whole clips: first 9 train, last 6 test
                all_idx = np.arange(np.shape(tmp_tri)[0])  # [MOD]
                train_idx, test_idx = (all_idx, all_idx[:0]) if tri < 9 else (all_idx[:0], all_idx)  # [MOD]

            self.train_data.append(tmp_tri[train_idx])
            self.train_label.append(tmp_tri_label[train_idx])
            self.test_data.append(tmp_tri[test_idx])
            self.test_label.append(tmp_tri_label[test_idx])
        
        self.train_data = np.concatenate(self.train_data)
        self.train_data = np.expand_dims(self.train_data, axis=1)
        self.train_label = np.concatenate(self.train_label)
        self.test_data = np.concatenate(self.test_data)
        self.test_data = np.expand_dims(self.test_data, axis=1)
        self.test_label = np.concatenate(self.test_label)

        shuffle_num = np.random.permutation(len(self.train_data))
        self.train_data = self.train_data[shuffle_num, :, :, :]
        self.train_label = self.train_label[shuffle_num]

        # standardize
        target_mean = np.mean(self.train_data)
        target_std = np.std(self.train_data)
        self.train_data = (self.train_data - target_mean) / target_std
        self.test_data = (self.test_data - target_mean) / target_std

        return self.train_data, self.train_label, self.test_data, self.test_label

    def update_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def aug(self, img, label):
        aug_data = []
        aug_label = []
        for cls4aug in range(4):
            cls_idx = np.where(label == cls4aug + 1)
            tmp_data = img[cls_idx]
            tmp_label = label[cls_idx]

            tmp_aug_data = np.zeros(tmp_data.shape)
            for ri in range(tmp_data.shape[0]):
                for rj in range(8):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                    tmp_aug_data[ri, :, :, rj * 125:(rj + 1) * 125] = tmp_data[rand_idx[rj], :, :, rj * 125:(rj + 1) * 125]

            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label)
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        return aug_data, aug_label

    def update_centers(self, feature, label):
            deltac = {}
            count = {}
            count[0] = 0
            for i in range(len(label)):
                l = label[i]
                if l in deltac:
                    deltac[l] += self.centers[l]-feature[i]
                else:
                    deltac[l] = self.centers[l]-feature[i]
                if l in count:
                    count[l] += 1
                else:
                    count[l] = 1

            for ke in deltac.keys():
                deltac[ke] = deltac[ke]/(count[ke]+1)

            return deltac

    def train(self, fold):

        img, label, test_data, test_label = self.get_source_data(fold)

        img = torch.from_numpy(img)
        label = torch.from_numpy(label + 1)

        dataset = torch.utils.data.TensorDataset(img, label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label + 1)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

        for i in range(self.c_dim):
            self.centers[i] = torch.randn(self.dimension)
            self.centers[i] = self.centers[i].cuda()

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))

        bestAcc = 0
        averAcc = 0
        num = 0
        Y_true = 0
        Y_pred = 0

        # Train the cnn model
        total_step = len(self.dataloader)
        curr_lr = self.lr

        for e in range(self.n_epochs):
            in_epoch = time.time()
            self.model.train()
            for i, (img, label) in enumerate(self.dataloader):

                img = Variable(img.cuda().type(self.Tensor))
                # img = self.active_function(img)
                label = Variable(label.cuda().type(self.LongTensor))

                tok, outputs = self.model(img)

                # Central loss
                cen_feature = tok
                cen_label = label
                nplabela = cen_label.cpu().numpy()


                loss = self.criterion_cls(outputs, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            out_epoch = time.time()

            if (e + 1) % 1 == 0:
                self.model.eval()
                with torch.no_grad():  # [MOD] released: no no_grad, a graph over the whole test set (6.4 GB for 1,384 windows); scores unchanged
                    Tok, Cls = self.model(test_data)

                loss_test = self.criterion_cls(Cls, test_label)
                y_pred = torch.max(Cls, 1)[1]
                acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))
                # print('The epoch is:', e, '  The accuracy is:', acc)
                print('Epoch:', e,
                      '  Train loss: %.4f' % loss.detach().cpu().numpy(),
                      '  Test loss: %.4f' % loss_test.detach().cpu().numpy(),
                      '  Train acc: %.4f' % train_acc,
                      '  Test acc: %.4f' % acc)
                self.log_write.write(str(e) + "    " + str(acc) + "\n")
                CURVE.append(acc)  # [MOD]
                num = num + 1
                averAcc = averAcc + acc
                if acc > bestAcc:
                    bestAcc = acc
                    Y_true = test_label
                    Y_pred = y_pred

        averAcc = averAcc / num
        print('The average accuracy of fold%d is:' %(fold+1), averAcc)
        print('The best accuracy of fold%d is:' %(fold+1), bestAcc)
        self.log_write.write('The average accuracy of fold%d is: ' %(fold+1) + str(averAcc) + "\n")
        self.log_write.write('The best accuracy fold%d is: ' %(fold+1) + str(bestAcc) + "\n")
        return bestAcc, averAcc, Y_true, Y_pred
        # writer.close()


def main():  # [MOD] one (subject, fold) per process; the released main() loops over them and averages the two numbers
    global SPLIT
    ap = argparse.ArgumentParser()
    ap.add_argument('--subject', type=int, required=True)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--split', default='kfold', choices=['kfold', 'trial96'])
    a = ap.parse_args()
    SPLIT = a.split
    os.makedirs(os.path.join(a.out, 'curves'), exist_ok=True)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    exgan = ExGAN(a.subject, a.fold, a.root, a.out)
    ba, aa, _, _ = exgan.train(a.fold)
    np.save(os.path.join(a.out, 'curves', 'S%d_f%d_%s.npy' % (a.subject, a.fold, a.split)), np.array(CURVE, dtype=np.float32))
    print('best', ba, 'mean', aa, 'last', CURVE[-1])


SPLIT, CURVE = 'kfold', []  # [MOD]


if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))
