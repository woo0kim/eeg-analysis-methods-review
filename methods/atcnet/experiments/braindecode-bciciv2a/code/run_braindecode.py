"""braindecode's ATCNet trained by a PyTorch port of the released ATCNet training loop, one subject and one seed.

The released code is TensorFlow, so the model cannot be swapped into it; instead this script ports the loop and keeps
everything else of the released runs (../../bciciv2a-model-selection/code/run_released.py):
  data       the released preprocess.get_data(), imported unchanged (its tensorflow.keras.utils.to_categorical import
             is served by a numpy one-hot, so TensorFlow is not needed): 22 channels, 1.5-6 s, z-scored per channel
             with training-set statistics, the same shuffles and, for trainvaltest, the same train_test_split
  training   Keras model.fit semantics: reshuffle every epoch, batches of 64 including the last partial one, mean
             cross-entropy, Adam(lr 1e-3, betas 0.9/0.999, eps 1e-7 = the Keras defaults); after every epoch, loss
             and accuracy on the validation data in eval mode
  callbacks  as in main_TrainTest.py (traintest, loso) or main_TrainValTest.py (trainvaltest): ModelCheckpoint
             (best val_accuracy, strictly greater / best val_loss, strictly smaller), ReduceLROnPlateau(val_loss,
             factor 0.9, patience 20, min_delta 1e-4, min_lr 1e-4), EarlyStopping(val_accuracy, patience 300) for
             traintest and loso; then the checkpoint is evaluated on the test set
  protocols  traintest (session E is the validation data, as in the paper), trainvaltest (20% of session T),
             loso; epochs 1000 / 500 / 1000; seeds 1..10 like the released loop
             paper2022: the loop of the 2022 code behind the paper's numbers (../../bciciv2a-model-selection/code/
             run_paper2022.py): as traintest, but Adam lr 0.0009, no ReduceLROnPlateau, and Adam's bias correction
             off (step counter started at 100000) except for subject 1 / seed 1, as with the Adam instance the 2022
             loop shares across runs; its get_data() returns the same trials, unshuffled

Variants of the model:
  default   ATCNet(n_chans=22, n_outputs=4, n_times=1125) and plain Adam: braindecode out of the box
  faithful  ATCNet(..., conv_max_norm_const=0.6) and Adam(model.source_optimizer_param_groups()): the options
            braindecode documents for matching the official code
  keras     control for today's code: braindecode's ATCNet with every training-time difference to the current Keras
            model switched to the Keras behaviour (SWITCHES below): element-wise dropout in the conv block, attention
            dropout on the attention weights, BatchNorm eps 1e-3 / momentum 0.99, no max-norm on the final layer, the
            Keras initialisers, the L2 penalties (0.009 conv, 0.5 dense) in the loss and the max-norm (0.6)
            projections after every step. If this reproduces the Keras runs, the port of the loop is faithful and
            any gap of `default` comes from the model.
  keras2022 the same for the 2022 code: as keras, but the final layer keeps a Keras max-norm of 0.25 per output
            unit, the depthwise kernel has a max-norm of 1.0, and there is no L2 and no conv max-norm 0.6
  default+<switch>[+<switch>...]   braindecode's defaults with single Keras behaviours switched on (ablation)
For trainvaltest the checkpoint and the learning-rate schedule follow val_loss, which in Keras includes the L2
penalties; val_loss here includes the penalty the variant trains with (none for default, braindecode's weight decay
as an L2 term for faithful, the Keras terms for keras).

Writes <out>/<variant>/<protocol>/curves/s<sub>_seed<seed>.npz (same keys as the Keras runs).
usage: python code/run_braindecode.py --variant default --protocol traintest --subject 1 --seed 1
"""
import argparse, contextlib, io, math, os, sys, time, types
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import parametrize
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '../../bciciv2a-model-selection/code'))
import paths

TF_STUB = ('tensorflow', 'tensorflow.keras', 'tensorflow.keras.utils')      # preprocess.py imports to_categorical
for name in TF_STUB:
    sys.modules[name] = types.ModuleType(name)
sys.modules['tensorflow.keras.utils'].to_categorical = lambda y: np.eye(int(np.max(y)) + 1, dtype='float32')[y]
sys.path.insert(0, paths.UPSTREAM)
from preprocess import get_data
for name in TF_STUB:          # drop the stub again (torch._dynamo probes for a real tensorflow package)
    del sys.modules[name]
with contextlib.redirect_stderr(io.StringIO()):
    from braindecode.models import ATCNet

EPOCHS = {'traintest': 1000, 'trainvaltest': 500, 'loso': 1000, 'paper2022': 1000}
LR = {'paper2022': 0.0009}            # all others: 0.001
PATIENCE = 300                        # EarlyStopping of main_TrainTest.py (main_TrainValTest.py has none)
L2_CONV, L2_DENSE, MAX_NORM = 0.009, 0.5, 0.6      # models.ATCNet_: conv_weightDecay, dense_weightDecay, conv_maxNorm


# ---------------------------------------------------------------- control variant: Keras training-time behaviour
def _mha_forward_keras(self, Q, K, V):
    """braindecode's _MHA.forward with the dropout moved onto the attention weights, as in Keras MultiHeadAttention."""
    b = Q.shape[0]
    Q, K, V = self.fc_q(Q), self.fc_k(K), self.fc_v(V)
    Q_, K_, V_ = (torch.cat(t.split(self.head_dim, -1), 0) for t in (Q, K, V))
    W = torch.softmax(Q_.bmm(K_.transpose(-2, -1)) / math.sqrt(self.head_dim), -1)
    H = torch.cat(self.dropout(W).bmm(V_).split(b, 0), -1)
    return self.fc_o(H)


def _glorot_(w, fan_in, fan_out):
    lim = math.sqrt(6 / (fan_in + fan_out))
    nn.init.uniform_(w, -lim, lim)


def _leaf(module):
    """The trainable weight of a layer, under braindecode's max-norm parametrization if it has one."""
    return module.parametrizations.weight.original if parametrize.is_parametrized(module, 'weight') else module.weight


# One Keras behaviour each; the presets combine them. A variant is a preset or 'default+<switch>[+<switch>...]'.
SWITCHES = {
    'dropout': 'element-wise dropout in the conv block (Keras Dropout; braindecode uses Dropout2d)',
    'attn': 'attention dropout on the attention weights (Keras MultiHeadAttention; braindecode: on the output)',
    'bn': 'BatchNorm eps 1e-3 and momentum 0.99 (Keras defaults)',
    'init': 'Keras initialisers (Glorot uniform with Keras fan computation, He uniform in the TCN, zero biases)',
    'dense': 'final dense layers without max-norm (current code)',
    'dense2022': 'final dense layers with Keras max_norm(0.25) per output unit (2022 code), not per input feature',
    'denserows': "braindecode's own max-norm parametrization of the final layers (0.25), but per output unit (dim 1)",
    'dw2022': 'Keras max_norm(1.0) on the depthwise kernel (2022 code)',
    'l2': 'L2 penalties 0.009 (conv kernels) and 0.5 (dense kernels) in the loss (current code)',
    'conv': 'Keras max_norm(0.6) projections on all conv kernels (current code)'}
PRESETS = {'default': [], 'keras': ['dropout', 'attn', 'bn', 'dense', 'init', 'l2', 'conv'],
           'keras2022': ['dropout', 'attn', 'bn', 'dense2022', 'init', 'dw2022']}


class _MaxNormRows(nn.Module):
    """braindecode's MaxNorm parametrization (modules/parametrization.py) with the norm over dim 1, i.e. one norm per
    output unit of a (out, in) Linear weight, the axis of Keras' max_norm on a Dense kernel."""
    def __init__(self, max_norm_val=0.25, eps=1e-5):
        super().__init__()
        self.max_norm_val, self.eps = max_norm_val, eps

    def forward(self, X):
        norm = X.norm(2, dim=1, keepdim=True)
        denom = norm.clamp(min=self.max_norm_val / 2)
        number = denom.clamp(max=self.max_norm_val)
        return X * (number / (denom + self.eps))


def switches_of(variant):
    if variant == 'faithful':
        return []
    base, *extra = variant.split('+')
    out = PRESETS[base] + extra
    assert set(out) <= set(SWITCHES), f'unknown switch in {variant}'
    return out


def keras_match(m, sw):
    """Switch braindecode's ATCNet to the Keras behaviours in sw. Returns the max-norm projections to apply after
    every optimizer step as (weight, kind, max norm)."""
    cb = m.conv_block
    if 'dropout' in sw:
        cb.drop2, cb.drop3 = nn.Dropout(cb.drop2.p), nn.Dropout(cb.drop3.p)      # Keras Dropout, not Dropout2d
    if 'attn' in sw:
        for blk in m.attention_blocks:
            blk.mha.forward = types.MethodType(_mha_forward_keras, blk.mha)
    if 'bn' in sw:
        for mod in m.modules():
            if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                mod.eps, mod.momentum = 1e-3, 0.01                                  # Keras eps 1e-3, momentum 0.99
    if 'dense' in sw or 'dense2022' in sw or 'denserows' in sw:
        for lin in m.final_layer:
            parametrize.remove_parametrizations(lin, 'weight', leave_parametrized=False)
            if 'denserows' in sw:
                parametrize.register_parametrization(lin, 'weight', _MaxNormRows(0.25))
    if 'init' in sw:
        # glorot_uniform with Keras' fan computation (kernel shapes in Keras layout), zero biases; he_uniform in the TCN
        with torch.no_grad():
            _glorot_(cb.conv1.weight, 64, 64 * 16)                 # Conv2D kernel (64, 1, 1, 16)
            _glorot_(cb.conv2.weight, 22 * 16, 22 * 2)             # DepthwiseConv2D kernel (1, 22, 16, 2)
            _glorot_(cb.conv3.weight, 16 * 32, 16 * 32)            # Conv2D kernel (16, 1, 32, 32)
            for blk in m.attention_blocks:
                for fc in (blk.mha.fc_q, blk.mha.fc_k, blk.mha.fc_v):
                    _glorot_(fc.weight, 32 * 2, 32 * 8)            # EinsumDense kernel (32, 2, 8)
                    fc.bias.zero_()
                _glorot_(blk.mha.fc_o.weight, 2 * 8, 2 * 32)       # EinsumDense kernel (2, 8, 32)
                blk.mha.fc_o.bias.zero_()
            for tcn in m.temporal_conv_nets:
                for blk in tcn:
                    for conv in (blk.conv1, blk.conv2):
                        nn.init.kaiming_uniform_(conv.weight)        # = he_uniform, fan_in = kernel * in_channels
                        conv.bias.zero_()
            for lin in m.final_layer:
                _glorot_(_leaf(lin), 32, 4)
                lin.bias.zero_()
    constrained = []
    if 'conv' in sw:
        constrained += [(cb.conv1.weight, 'filter', MAX_NORM), (cb.conv2.weight, 'depthwise', MAX_NORM),
                        (cb.conv3.weight, 'filter', MAX_NORM)]
        constrained += [(c.weight, 'filter', MAX_NORM) for tcn in m.temporal_conv_nets for blk in tcn
                        for c in (blk.conv1, blk.conv2)]
    if 'dw2022' in sw:
        constrained.append((cb.conv2.weight, 'depthwise', 1.0))
    if 'dense2022' in sw:
        constrained += [(lin.weight, 'filter', 0.25) for lin in m.final_layer]
    return constrained


@torch.no_grad()
def keras_max_norm_(constrained):
    """Keras MaxNorm after each update: w * clip(n, 0, max) / (1e-7 + n). Conv kernels and Dense kernels: one norm per
    output filter / unit; DepthwiseConv2D with axis=[0, 1, 2]: one norm per depth-multiplier index over all channels
    and input filters."""
    for w, kind, mx in constrained:
        if kind == 'depthwise':                                # torch (F1*D, 1, 1, C), output f*D + d
            v = w.view(16, 2, -1)
            n = v.pow(2).sum(dim=(0, 2), keepdim=True).sqrt()
            v.mul_(n.clamp(0, mx) / (1e-7 + n))
        else:
            n = w.pow(2).sum(dim=tuple(range(1, w.dim())), keepdim=True).sqrt()
            w.mul_(n.clamp(0, mx) / (1e-7 + n))


def l2_terms(m, variant, sw):
    """[(coefficient, weights)] of the L2 penalty the variant trains with (also added to val_loss, as Keras does)."""
    if 'l2' in sw:
        conv = [m.conv_block.conv1.weight, m.conv_block.conv2.weight, m.conv_block.conv3.weight] + \
               [c.weight for tcn in m.temporal_conv_nets for blk in tcn for c in (blk.conv1, blk.conv2)]
        return [(L2_CONV, conv), (L2_DENSE, [lin.weight for lin in m.final_layer])]
    if variant == 'faithful':   # torch weight_decay wd on Adam = gradient of (wd / 2) * ||w||^2
        return [(g['weight_decay'] / 2, g['params']) for g in m.source_optimizer_param_groups() if g['weight_decay']]
    return []


def penalty(terms):
    return sum(c * sum(w.pow(2).sum() for w in ws) for c, ws in terms) if terms else 0.0


# ---------------------------------------------------------------- training loop (Keras fit + released callbacks)
def evaluate(model, X, y, terms):
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(X[i:i + 288]) for i in range(0, len(X), 288)])
        loss = float(F.cross_entropy(logits, y)) + float(penalty(terms))
    return loss, float((logits.argmax(1) == y).float().mean()), logits.argmax(1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, help='default | faithful | keras | keras2022 | default+<switch>[+...]')
    ap.add_argument('--protocol', choices=sorted(EPOCHS), required=True)
    ap.add_argument('--subject', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--epochs', type=int, help='override (smoke tests only)')
    ap.add_argument('--out', default=os.path.join(paths.WORK, 'braindecode'))
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs, sub = a.epochs or EPOCHS[a.protocol], a.subject - 1

    conf = paths.dataset_conf(loso=a.protocol == 'loso')
    X_train, _, y_train_onehot, X_test, _, y_test_onehot = get_data(conf['data_path'], sub, 'BCI2a',
                                                                    LOSO=conf['LOSO'], isStandard=True)
    if a.protocol == 'trainvaltest':
        X_train, X_val, y_train_onehot, y_val_onehot = train_test_split(X_train, y_train_onehot, test_size=0.2,
                                                                        random_state=42)
    else:
        X_val, y_val_onehot = X_test, y_test_onehot
    T = lambda x: torch.as_tensor(x[:, 0], dtype=torch.float32, device=dev)            # (N, 22, 1125)
    L = lambda y: torch.as_tensor(y.argmax(-1), dtype=torch.long, device=dev)
    Xtr, ytr, Xva, yva, Xte, yte = T(X_train), L(y_train_onehot), T(X_val), L(y_val_onehot), T(X_test), L(y_test_onehot)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    g = torch.Generator().manual_seed(a.seed)
    with contextlib.redirect_stderr(io.StringIO()):
        model = ATCNet(n_chans=22, n_outputs=4, n_times=1125,
                       **({'conv_max_norm_const': MAX_NORM} if a.variant == 'faithful' else {}))
    sw = switches_of(a.variant)
    constrained = keras_match(model, sw)
    model = model.to(dev)
    params = model.source_optimizer_param_groups() if a.variant == 'faithful' else model.parameters()
    opt = torch.optim.Adam(params, lr=LR.get(a.protocol, 1e-3), betas=(0.9, 0.999), eps=1e-7)
    if a.protocol == 'paper2022' and (a.subject, a.seed) != (1, 1):     # bias correction off, as in the 2022 loop
        for pg in opt.param_groups:
            for prm in pg['params']:
                opt.state[prm] = {'step': torch.tensor(100000.), 'exp_avg': torch.zeros_like(prm),
                                  'exp_avg_sq': torch.zeros_like(prm)}
    terms = l2_terms(model, a.variant, sw)
    train_terms = terms if 'l2' in sw else []      # faithful: the optimizer applies the decay itself

    by_acc = a.protocol != 'trainvaltest'
    best_ckpt, best_state, best_epoch = (-np.inf if by_acc else np.inf), None, -1
    rlr_best, rlr_wait = np.inf, 0
    es_best, es_wait = -np.inf, 0
    h = {k: [] for k in ('loss', 'accuracy', 'val_loss', 'val_accuracy', 'lr', 'test_acc')}
    t0 = time.time()
    for epoch in range(epochs):
        h['lr'].append(opt.param_groups[0]['lr'])
        model.train()
        perm, tl, tc = torch.randperm(len(Xtr), generator=g).to(dev), 0.0, 0
        for i in range(0, len(Xtr), 64):
            idx = perm[i:i + 64]
            out = model(Xtr[idx])
            loss = F.cross_entropy(out, ytr[idx]) + penalty(train_terms)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            keras_max_norm_(constrained)
            tl += float(loss) * len(idx)
            tc += int((out.argmax(1) == ytr[idx]).sum())
        h['loss'].append(tl / len(Xtr))
        h['accuracy'].append(tc / len(Xtr))
        vl, va, _ = evaluate(model, Xva, yva, terms)
        h['val_loss'].append(vl)
        h['val_accuracy'].append(va)
        if a.protocol == 'trainvaltest':
            h['test_acc'].append(evaluate(model, Xte, yte, [])[1])
        # ModelCheckpoint(save_best_only=True)
        cur = va if by_acc else vl
        if (cur > best_ckpt) if by_acc else (cur < best_ckpt):
            best_ckpt, best_epoch = cur, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        # ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=20, min_lr=1e-4), Keras min_delta 1e-4
        if a.protocol == 'paper2022':
            pass                                  # the 2022 loop has no learning-rate schedule
        elif vl < rlr_best - 1e-4:
            rlr_best, rlr_wait = vl, 0
        else:
            rlr_wait += 1
            if rlr_wait >= 20:
                for pg in opt.param_groups:
                    if pg['lr'] > np.float32(1e-4):
                        pg['lr'] = max(pg['lr'] * 0.9, 1e-4)
                rlr_wait = 0
        # EarlyStopping(monitor='val_accuracy', mode='max', patience=300)
        if by_acc:
            es_wait += 1
            if va > es_best:
                es_best, es_wait = va, 0
            if es_wait >= PATIENCE:
                print(f'Epoch {epoch + 1:05d}: early stopping')
                break

    model.load_state_dict(best_state)
    _, acc, y_pred = evaluate(model, Xte, yte, [])
    labels = yte.cpu().numpy()
    acc, kappa = accuracy_score(labels, y_pred), cohen_kappa_score(labels, y_pred)
    minutes = (time.time() - t0) / 60
    extra = {}
    info = (f'Subject: {sub + 1}   Train no. {a.seed}   Time: {minutes:.1f} m   Test_acc: {acc:.4f}   '
            f'Test_kappa: {kappa:.4f}   best epoch {best_epoch + 1} of {len(h["loss"])}')
    if a.protocol == 'trainvaltest':
        extra['val_acc_at_checkpoint'] = evaluate(model, Xva, yva, [])[1]
        info += f'   valid_acc: {extra["val_acc_at_checkpoint"]:.4f}   valid_loss: {min(h["val_loss"]):.3f}'
    print(info)
    od = os.path.join(a.out, a.variant, a.protocol, 'curves')
    os.makedirs(od, exist_ok=True)
    np.savez_compressed(os.path.join(od, f's{sub + 1}_seed{a.seed}.npz'), protocol=a.protocol, variant=a.variant,
                        subject=sub + 1, seed=a.seed, **{k: np.array(v, dtype=np.float32) for k, v in h.items()},
                        y_pred=y_pred.astype(np.int8), labels=labels.astype(np.int8), acc=acc, kappa=kappa,
                        minutes=minutes, epochs_run=len(h['loss']), best_epoch=best_epoch, n_train=len(Xtr),
                        n_val=len(Xva), n_test=len(Xte), torch_version=torch.__version__, **extra)


if __name__ == '__main__':
    main()
