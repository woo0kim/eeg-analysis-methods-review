"""The repository's networks in the dDTF-GL paper's two STEW settings, on exactly the windows and splits dDTF-GL ran on there.

  python nets6.py build <out dir>                      windows6.npz: the 3,552 6-s windows of methods/ddtf-gl/code/datasets.Stew6s
                                                       (1-Hz high-pass, 4-s step) at 128 Hz (768 samples) and resampled to 200 Hz
                                                       (1,200 samples, for CBraMod and LaBraM), with subject and label
  python nets6.py run --model M --protocol si|sd --fold K --seed S --root <dir with windows6.npz> --out <run dir>

Splits: methods/ddtf-gl/code/protocols.py, the ones dDTF-GL used: SI = the pinned rest-task subject folds, training on the other six;
SD = StratifiedKFold(7, shuffle=True, random_state=seed) over all windows. The paper has no validation set, so nothing is selected:
each network trains for the shared recipe's 50 epochs on the whole training part and the final epoch is scored (as chosen for this
comparison). Models: the shared STEW zoo (methods/cbramod/experiments/finetune-stew/code/stew_models.py) sized for 6-s windows
(changes marked [MOD]); LaBraM-Base runs through run_labram6.py. Training: the shared trainer
(methods/cbramod/experiments/finetune-seed/code/trainer.py, CBraMod's released loop), with its epoch loop copied below because that
loop keeps the best-validation epoch; changed lines are marked [MOD].
"""
import argparse
import json
import os
import random
import sys
from timeit import default_timer as timer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = os.path.normpath(os.path.join(HERE, '../../../..'))
GL_CODE = os.path.join(METHODS, 'ddtf-gl/code')
STEW_CODE = os.path.join(METHODS, 'cbramod/experiments/finetune-stew/code')
SEC, N_CH, FS = 6, 14, 128
N_T = FS * SEC                       # 768
FOUNDATION = ('cbramod', 'labram')


# ------------------------------------------------------------------------------------------------------------------------- data
def build(out):
    from scipy import signal
    sys.path.insert(0, GL_CODE)
    import datasets
    s = datasets.Stew6s()
    x200 = signal.resample_poly(s.x.astype(np.float64), 25, 16, axis=2).astype(np.float32)
    assert s.x.shape == (3552, 14, 768) and x200.shape == (3552, 14, 1200), (s.x.shape, x200.shape)
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, 'windows6.npz'), x128=s.x, x200=x200, subject=s.subject, stage=s.stage)
    print('wrote', os.path.join(out, 'windows6.npz'), s.x.shape, x200.shape)


def split(z, protocol, fold, seed):
    sys.path.insert(0, GL_CODE)
    import protocols
    sp = protocols.splits('STEW-6s', {'y': z['stage'].astype(np.int64), 'subject': z['subject']}, protocol, fold=fold, seed=seed)
    return sp['train'], sp['test'], sp['y']


def loaders(root, protocol, fold, seed, foundation, batch_size=64):
    import torch
    from torch.utils.data import DataLoader, Dataset
    z = np.load(os.path.join(root, 'windows6.npz'))
    x = z['x200'] if foundation else z['x128']
    tr, te, y = split(z, protocol, fold, seed)
    p = 200 if foundation else FS

    class Windows(Dataset):    # as stew_data.Windows: (channels, seconds, samples per second), x / 100
        def __init__(self, idx):
            self.idx = idx

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            j = self.idx[i]
            return x[j].reshape(N_CH, SEC, p), int(y[j])

    def collate(batch):
        return torch.from_numpy(np.stack([b[0] for b in batch]) / 100).float(), torch.tensor([b[1] for b in batch], dtype=torch.long)

    print(f'STEW-6s {protocol} fold {fold} seed {seed}: train {len(tr)}, test {len(te)}; x {x.shape}')
    mk = lambda idx, sh: DataLoader(Windows(idx), batch_size=batch_size, shuffle=sh, collate_fn=collate)
    return {'train': mk(tr, True), 'val': mk(te[:0], False), 'test': mk(te, False)}


# ----------------------------------------------------------------------------------------------------------------------- models
def build_model(name, k):
    """stew_models.build for 6-s windows. [MOD] relative to stew_models.py: N_T 512 -> 768; CBraMod's head sized for 6 patches; the
    Conformer's first head layer for 45 tokens ((768 - 99) // 15 + 1, as 28 for 512); DGCNN on DE of each of the 6 seconds (30 values)."""
    import torch.nn as nn
    from einops.layers.torch import Rearrange
    sys.path.insert(0, STEW_CODE)
    import stew_models as sm
    em = sm.em
    if name == 'cbramod':
        import torch
        sys.path.insert(0, em.CBRAMOD_SRC)
        from models.cbramod import CBraMod

        class CBraModSTEW6(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8)
                self.backbone.load_state_dict(torch.load(em.CBRAMOD_WEIGHTS, map_location='cpu'))
                self.backbone.proj_out = nn.Identity()
                self.classifier = nn.Sequential(Rearrange('b c s d -> b (c s d)'), nn.Linear(N_CH * SEC * 200, SEC * 200), nn.ELU(),  # [MOD] 4 -> SEC
                                                nn.Dropout(0.1), nn.Linear(SEC * 200, 200), nn.ELU(), nn.Dropout(0.1), nn.Linear(200, k))

            def forward(self, x):
                return self.classifier(self.backbone(x))
        return CBraModSTEW6()
    if name == 'eegnet':
        sys.path.insert(0, os.path.join(METHODS, 'eegnet/experiments/pytorch-cbramod-splits/code'))
        sys.path.insert(0, em.CBRAMOD_SRC)
        from run_eegnet import EEGNet
        return EEGNet(N_CH, N_T, k, kern=64)                                                                          # [MOD] N_T
    if name == 'conformer':
        sys.path.insert(0, os.path.join(METHODS, 'cbramod/experiments/finetune-sleepedf/code'))
        from conformer_released import PatchEmbedding, TransformerEncoder
        tokens = (N_T - 99) // 15 + 1                                                                                  # [MOD] 28 -> 45

        class ConformerSTEW6(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch, self.transformer = PatchEmbedding(40, n_chans=N_CH), TransformerEncoder(6, 40)
                self.fc = nn.Sequential(nn.Linear(tokens * 40, 256), nn.ELU(), nn.Dropout(0.5), nn.Linear(256, 32), nn.ELU(), nn.Dropout(0.3),
                                        nn.Linear(32, k))

            def forward(self, x):
                return self.fc(self.transformer(self.patch(x.reshape(x.shape[0], 1, N_CH, -1))).flatten(1))
        return ConformerSTEW6()
    if name == 'dgcnn':
        return sm.WithDE(sm.DGCNN(N_CH, 5 * SEC, k), N_CH, FS, SEC)                                                   # [MOD] 20, 4 -> 30, 6
    from braindecode import models as bd
    if name == 'bd-atcnet':
        return em.Flat(bd.ATCNet(n_chans=N_CH, n_outputs=k, n_times=N_T, sfreq=FS))
    if name == 'bd-deepsleepnet':
        return em.Flat(bd.DeepSleepNet(n_chans=N_CH, n_outputs=k, n_times=N_T, sfreq=FS))
    raise ValueError(name)


# ---------------------------------------------------------------------------------------------------------------------- training
def train_final(params, loader, model):
    """Trainer.train_for_multiclass (finetune-seed/code/trainer.py) without validation: same optimizer, schedule, loss, clipping and
    batches; the test set is scored after every epoch for the record (metrics.json) and never used; the final epoch's predictions are
    saved."""
    import torch
    sys.path.insert(0, os.path.join(METHODS, 'cbramod/experiments/finetune-seed/code'))
    sys.path.insert(0, os.path.join(METHODS, 'cbramod/upstream'))
    from trainer import Trainer, _predict
    t = Trainer(params, loader, model)            # optimizer (multi_lr), cosine schedule, label-smoothed loss, as every STEW run
    history = []
    for epoch in range(params.epochs):
        t.model.train()
        start, losses = timer(), []
        t.optimizer.zero_grad()
        for x, y in loader['train']:
            loss = t.criterion(t.model(x.cuda()), y.cuda())
            loss.backward()
            losses.append(loss.item())
            if params.clip_value > 0:
                torch.nn.utils.clip_grad_norm_(t.model.parameters(), params.clip_value)
            t.optimizer.step()
            t.optimizer_scheduler.step()
            t.optimizer.zero_grad()
        with torch.no_grad():                                                                       # [MOD] no validation set
            acc, kappa, f1, cm = t.test_eval.get_metrics_for_multiclass(t.model)
        history.append({'epoch': epoch + 1, 'train_loss': float(np.mean(losses)),
                        'test': {'balanced_acc': acc, 'kappa': kappa, 'weighted_f1': f1, 'cm': cm.tolist()}, 'minutes': (timer() - start) / 60})
        print(f'Epoch {epoch + 1}: training loss {np.mean(losses):.5f}, test (recorded, not used) acc {acc:.4f} kappa {kappa:.4f}', flush=True)
        json.dump(history, open(os.path.join(params.model_dir, 'metrics.json'), 'w'))
    with torch.no_grad():
        truths, preds = _predict(t.model, loader['test'])                                           # [MOD] the final epoch, no selection
    np.savez_compressed(os.path.join(params.model_dir, 'test_preds.npz'), y=truths.astype(np.int8), pred=preds.astype(np.int8), epoch=params.epochs)
    print(f'final epoch {params.epochs}: test accuracy {np.mean(truths == preds):.4f}')


def run(a):
    import torch
    k = 2
    params = argparse.Namespace(seed=a.seed, cuda=0, epochs=a.epochs, batch_size=64, lr=1e-4, weight_decay=5e-2, optimizer='AdamW',  # run_stew.py
                                clip_value=1, dropout=0.1, classifier='all_patch_reps', downstream_dataset='SEED-V', datasets_dir=a.root,
                                num_of_classes=k, model_dir=a.out, num_workers=0, label_smoothing=0.1, multi_lr=True, frozen=False,
                                use_pretrained_weights=False, foundation_dir='', accum=1, model=a.model)
    print(params)
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(0)
    loader = loaders(a.root, a.protocol, a.fold, a.seed, a.model in FOUNDATION)
    model = build_model(a.model, k)
    if hasattr(model, 'fit_norm'):
        model.fit_norm(loader['train'], n_batches=30)
    train_final(params, loader, model)


if __name__ == '__main__':
    if sys.argv[1] == 'build':
        build(sys.argv[2])
        sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd')
    for a_ in ('--model', '--protocol', '--root', '--out'):
        ap.add_argument(a_, required=True)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--epochs', type=int, default=50)
    run(ap.parse_args())
