"""Inference FLOPs and parameters of every model in every shared pipeline, per sample, and the training compute of one run.
Feeds SUMMARY.md (Section 3, "Model size and compute").

Counting. PyTorch models: torch.utils.flop_counter.FlopCounterMode on one sample (batch 1, eval mode, gradients enabled so that
nn.MultiheadAttention does not take its fused fast path, which the counter misses), the counter the CBraMod report used (CBraMod's
backbone on 16 channels x 10 s: 1,596M, as in that report) (2 FLOPs per multiply-add of every matmul, convolution and attention product; elementwise operations, normalisation and
activations are not counted). Two things it misses are added analytically: LSTM layers (2 x 4 gates x hidden x (input + hidden) per
step and direction) and DGCNN's FFT for differential entropy (2.5 n log2 n per 1-s segment and channel). The two TensorFlow sleep
models (DeepSleepNet and L-SeqSleepNet as released) and dDTF-GL's map computation are counted analytically from their layer sizes and
MVAR orders (functions below). Each model is the one in the shared pipeline's rank grid: the released implementation where it is
PyTorch, braindecode's for ATCNet and DeepSleepNet; CBraMod without its weights (same graph) and LaBraM-Base as braindecode's Labram
(identical to the released model, methods/labram/experiments/braindecode-cbramod-splits), so no weights or data are needed.

Training compute of one run = 3 x forward FLOPs x training samples x epochs (forward + backward), the usual estimate; 50 epochs in
every shared pipeline. Sleep-EDF counts one 30-s epoch as a sample (its models see 20-epoch sequences).

usage: python tools/count_flops.py [out.json]     (torch >= 2.1, braindecode 1.8.1, einops, timm 0.4.12; mcl-server: env bd-labram)
"""
import argparse
import json
import math
import os
import sys
import warnings

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = os.path.join(ROOT, 'methods')
for p in ('cbramod/upstream', 'dgcnn/code', 'cbramod/experiments/finetune-seed/code', 'cbramod/experiments/finetune-stew/code',
          'cbramod/experiments/finetune-sleepedf/code', 'eeg-conformer/experiments/cbramod-splits/code',
          'eegnet/experiments/pytorch-cbramod-splits/code'):
    sys.path.insert(0, os.path.join(M, p))

EPOCHS = 50
# pipeline -> (channels, samples, sampling rate, classes, training samples per run, sample described)
PIPE = {
    'BCI-IV-2a': (22, 800, 200, 4, 2880, '4-s trial'),
    'PhysioNet-MI': (64, 800, 200, 4, 6300, '4-s trial'),
    'Mumtaz2016': (19, 1000, 200, 2, 4891, '5-s window'),
    'MentalArithmetic': (20, 1000, 200, 2, 1343, '5-s window'),
    'SEED': (62, 200, 200, 3, 91638, '1-s window (subject split)'),
    'SEED-V': (62, 200, 200, 5, 73590, '1-s window (subject split)'),
    'STEW': (14, 512, 128, 2, 3430, '4-s window (one fold, rest vs. task)'),
    'Sleep-EDF-78': (2, 6000, 200, 5, 153100, '30-s epoch'),
}
CBRAMOD_HEAD = {'BCI-IV-2a': 'model_for_bciciv2a', 'PhysioNet-MI': 'model_for_physio', 'Mumtaz2016': 'model_for_mumtaz',
                'MentalArithmetic': 'model_for_stress'}


def lstm_hook(store):
    def hook(m, inp, out):
        x = inp[0]
        T = x.shape[1] if m.batch_first else x.shape[0]
        d = 2 if m.bidirectional else 1
        f, i = 0, m.input_size
        for _ in range(m.num_layers):
            f += 2 * 4 * m.hidden_size * (i + m.hidden_size) * T * d
            i = m.hidden_size * d
        store.append(f)
    return hook


def fft_hook(store):
    def hook(m, inp, out):
        x = inp[0]
        C, T = x.shape[-2], x.shape[-1]
        seg = int(m.T // max(1, round(m.T / 200))) if hasattr(m, 'T') else T
        n_seg = max(1, T // seg)
        store.append(C * n_seg * 2.5 * seg * math.log2(seg))
    return hook


def count(model, x):
    model.eval()
    extra, hooks = [], []
    from dgcnn import WithDE
    for mod in model.modules():
        if isinstance(mod, (nn.LSTM, nn.GRU)):
            hooks.append(mod.register_forward_hook(lstm_hook(extra)))
        if isinstance(mod, WithDE):
            hooks.append(mod.register_forward_hook(fft_hook(extra)))
    with FlopCounterMode(display=False) as fc:      # gradients on: in eval mode without them nn.MultiheadAttention takes a fused
        model(x)                                    # fast path the counter does not see (0 FLOPs for attention)
    for h in hooks:
        h.remove()
    return fc.get_total_flops() + sum(extra), sum(p.numel() for p in model.parameters())


class _Labram(nn.Module):
    """braindecode's Labram with channel names passed to forward (any C canonical names: the count does not depend on them)."""

    def __init__(self, C, T, k):
        super().__init__()
        from braindecode.models import labram
        self.m = labram.Labram(n_times=T, n_chans=C, n_outputs=k, patch_size=200, use_mean_pooling=True)
        self.names = list(labram.LABRAM_CHANNEL_ORDER)[:C]

    def forward(self, x):
        return self.m(x, ch_names=self.names)


def bd_labram(C, T, k):
    return _Labram(C, T, k)


# ---------------------------------------------------------------------------- PyTorch models, by pipeline
def cbramod_splits(pipe):
    C, T, _, k, *_ = PIPE[pipe]
    import importlib
    from run_models import build as bm
    from run_eegnet import EEGNet
    from dgcnn import DGCNN, WithDE
    out_k = 1 if k == 2 else k
    params = argparse.Namespace(use_pretrained_weights=False, foundation_dir='', classifier='all_patch_reps', dropout=0.1,
                                num_of_classes=k, cuda=0)
    head = importlib.import_module(f'models.{CBRAMOD_HEAD[pipe]}')
    x4 = torch.randn(1, C, T // 200, 200)
    flat = lambda m: (lambda x: m(x.reshape(1, C, -1)))   # noqa: E731
    yield 'CBraMod', head.Model(params), x4
    yield 'LaBraM-Base', bd_labram(C, T, out_k), torch.randn(1, C, T)
    yield 'EEGNet', EEGNet(C, T, out_k), x4
    for name, key in (('EEG Conformer', 'conformer'), ('ATCNet (bd)', 'bd-atcnet'), ('DeepSleepNet (bd)', 'bd-deepsleepnet')):
        yield name, bm(key, C, T, out_k), torch.randn(1, C, T)
    yield 'DGCNN', WithDE(DGCNN(C, 5 * (T // 200), out_k), C, 200, T // 200), torch.randn(1, C, T)


def seed(pipe):
    C, T, _, k, *_ = PIPE[pipe]
    import emo_models as em
    params = argparse.Namespace(num_of_classes=k, use_pretrained_weights=False, foundation_dir='', classifier='all_patch_reps',
                                dropout=0.1, cuda=0)
    x = torch.randn(1, C, 1, T)
    for name, key in (('CBraMod', 'cbramod-scratch'), ('EEGNet', 'eegnet'), ('EEG Conformer', 'conformer'), ('DGCNN', 'dgcnn'),
                      ('ATCNet (bd)', 'bd-atcnet'), ('DeepSleepNet (bd)', 'bd-deepsleepnet')):
        yield name, em.build(key, params), x
    yield 'LaBraM-Base', bd_labram(C, T, k), torch.randn(1, C, T)


def stew(pipe):
    C, T, _, k, *_ = PIPE[pipe]
    import stew_models as sm
    for name, key in (('EEGNet', 'eegnet'), ('EEG Conformer', 'conformer'), ('DGCNN', 'dgcnn'), ('ATCNet (bd)', 'bd-atcnet'),
                      ('DeepSleepNet (bd)', 'bd-deepsleepnet')):
        yield name, sm.build(key, k), torch.randn(1, C, 4, T // 4)
    yield 'CBraMod', sm.build('cbramod-scratch', k), torch.randn(1, C, 4, 200)          # 200 Hz, as run
    yield 'LaBraM-Base', bd_labram(C, 800, k), torch.randn(1, C, 800)


class _SeqHead(nn.Module):
    """The sleep pipeline's sequence head on given epoch features, for LaBraM (whose encoder is counted separately)."""

    def __init__(self, feat_dim):
        super().__init__()
        import seq_models as sq
        self.m = sq.SeqModel(nn.Identity(), feat_dim, 5, is_backbone=False)

    def forward(self, f):
        e = self.m.head(f)
        return self.m.classifier(self.m.sequence_encoder(e))


def sleep(pipe):
    import seq_models as sq
    x = torch.randn(1, 20, 2, 6000)
    for name, key in (('CBraMod', 'cbramod-scratch'), ('EEGNet', 'eegnet'), ('EEG Conformer', 'conformer'), ('DGCNN', 'dgcnn'),
                      ('ATCNet (bd)', 'bd-atcnet'), ('DeepSleepNet (bd)', 'bd-deepsleepnet')):
        yield name, sq.build(key), x


SEQ = 20     # epochs per Sleep-EDF sequence in the shared pipeline


# ---------------------------------------------------------------------------- analytic counts
def deepsleepnet_released():
    """Supratak et al. 2017 as released (deepsleep/model.py): 30-s Fpz-Cz at 100 Hz, 'SAME' padding; per epoch."""
    mac = 0
    L = math.ceil(3000 / 6); mac += L * 64 * 50 * 1                     # small path: conv 50 / 6
    L = math.ceil(L / 8); mac += L * 128 * 8 * (64 + 128 + 128)          # pool 8, three convs of 8
    Ls = math.ceil(L / 4)
    L = math.ceil(3000 / 50); mac += L * 64 * 400                        # large path: conv 400 / 50
    L = math.ceil(L / 4); mac += L * 128 * 6 * (64 + 128 + 128)          # pool 4, three convs of 6
    Ll = math.ceil(L / 2)
    feat = 128 * (Ls + Ll)
    mac += feat * 1024                                                   # residual FC
    mac += 4 * 512 * (feat + 512) * 2 + 4 * 512 * (1024 + 512) * 2       # 2-layer BiLSTM, 512 units
    mac += 2048 * 5
    return 2 * mac, None


def lseqsleepnet_released():
    """Phan et al. 2023 as released (config.py: 29 x 129 spectrogram, 32 filters, BiLSTMs of 64 units, attention 64); per epoch."""
    mac = 29 * 129 * 32                                                  # learnable filterbank
    mac += 4 * 64 * (32 + 64) * 29 * 2                                   # epoch-level BiLSTM over 29 frames
    mac += 29 * 128 * 64 + 29 * 64                                       # attention
    mac += 2 * 4 * 64 * (128 + 64) * 2                                   # BiLSTMs within and across subsequences, per epoch
    mac += 128 * 5
    return 2 * mac, None


DDTF = {  # pipeline -> (channels used, samples, fs, MVAR order) from methods/ddtf-gl/README.md
    'BCI-IV-2a': (21, 800, 200, 12), 'PhysioNet-MI': (63, 800, 200, 4), 'Mumtaz2016': (19, 1000, 200, 17),
    'MentalArithmetic': (20, 1000, 200, 16), 'SEED': (62, 200, 200, 1), 'SEED-V': (62, 200, 200, 1), 'STEW': (14, 512, 128, 8),
    'Sleep-EDF-78': (2, 6000, 200, 9)}


def ddtf_maps(pipe):
    """One window's dDTF maps: least-squares MVAR fit (normal equations and solve), then per 1-Hz frequency the transfer matrix
    A(f), its inverse H(f) and the inverse spectral matrix for the partial coherence (complex C x C products, 8 real FLOPs per
    complex multiply-add); the band means and elementwise steps are not counted, as for the networks."""
    C, T, fs, p = DDTF[pipe]
    n, q = T - p, C * p
    fit = 2 * n * q * q + (2 / 3) * q ** 3 + 2 * n * q * C + 2 * n * C * C
    nf = int(fs // 2)
    spec = nf * (8 * p * C * C + 8 * C ** 3 + 2 * 8 * C ** 3)
    return fit + spec


PLUS_RUN = {'BCI-IV-2a': 'cbramod-splits/logs/runs/BCIC-IV-2a__ddtf-gl-plus__s0', 'PhysioNet-MI': 'cbramod-splits/logs/runs/PhysioNet-MI__ddtf-gl-plus__s0',
            'Mumtaz2016': 'cbramod-splits/logs/runs/Mumtaz2016__ddtf-gl-plus__s0',
            'MentalArithmetic': 'cbramod-splits/logs/runs/MentalArithmetic__ddtf-gl-plus__s0',
            'STEW': 'stew-cbramod-pipeline/logs/runs/rest-task__fold0__ddtf-gl-plus__s0',
            'SEED': 'seed-cbramod-pipeline/logs/runs/SEED__subject__ddtf-gl-plus__s0',
            'SEED-V': 'seed-cbramod-pipeline/logs/runs/SEED-V__subject__ddtf-gl-plus__s0',
            'Sleep-EDF-78': 'sleepedf-cbramod-pipeline/logs/runs/SleepEDF__ddtf-gl-plus__s0'}


def ddtf_plus(pipe):
    """dDTF-GL+ per window, from the configuration and feature counts of its first final run (methods/ddtf-gl/experiments/*): the
    dDTF maps if the configuration uses them (or their graph summaries); the band-power blocks' FFTs (2.5 n log2 n per channel and
    window or 1-s segment); standardisation (2 per feature); LNT (16 per output); LDA and linear SVM (2 per selected feature and
    output); the RBF SVM bounded above (exact SVC: every training sample a support vector; Nystroem: 1,000 components); the trees
    (one comparison per level). Returns (FLOPs, configuration) or None if the run is not there."""
    path = os.path.join(M, 'ddtf-gl/experiments', PLUS_RUN[pipe], 'metrics.json')
    if not os.path.exists(path):
        return None
    m = json.load(open(path))
    feat = dict(kv.split('=') for kv in m['config'].split(';'))
    blocks, proj = feat['feat'].split('+'), [q for q in feat['proj'].split('+') if q != 'none']
    C, T, fs, _ = DDTF[pipe]
    nf = m['n_features']
    sel = nf.get('axb', nf['dft'])
    f = ddtf_maps(pipe) if ('ddtf' in blocks or 'graph' in blocks) else 0
    if 'spec' in blocks:
        f += C * 2.5 * T * math.log2(T)
    if 'spect' in blocks:
        f += C * (T // fs) * 2.5 * fs * math.log2(fs)
    f += 2 * nf['input'] + 16 * nf.get('lnt', 0)
    for q in ('lda', 'svml'):
        f += 2 * sel * nf.get(q, 0)
    if 'svmr' in proj:
        n_sv = m['n_train'] if m['n_train'] <= 8000 else 1000
        f += 2 * n_sv * sel + 2 * n_sv * nf['svmr']
    f += m['model']['trees'] * int(feat['depth'])
    return f, m['config']


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'tools', 'flops.json')
    rows = []
    gens = {p: cbramod_splits for p in CBRAMOD_HEAD} | {'SEED': seed, 'SEED-V': seed, 'STEW': stew, 'Sleep-EDF-78': sleep}
    for pipe, gen in gens.items():
        C, T, fs, k, n_train, what = PIPE[pipe]
        for name, model, x in gen(pipe):
            try:
                f, n_par = count(model, x)
            except Exception as e:                                        # report, do not stop
                print(f'{pipe:17s} {name:18s} FAILED: {type(e).__name__}: {e}')
                continue
            per = SEQ if pipe == 'Sleep-EDF-78' else 1
            rows.append(dict(pipeline=pipe, model=name, sample=what, gflops=f / per / 1e9, params=n_par, counted='FlopCounterMode'))
        if pipe == 'Sleep-EDF-78':
            enc_f, enc_p = count(bd_labram(2, 3000, 1), torch.randn(1, 2, 3000))
            head_f, head_p = count(_SeqHead(400), torch.randn(1, SEQ, 400))
            rows.append(dict(pipeline=pipe, model='LaBraM-Base', sample=what, gflops=(2 * enc_f + head_f / SEQ) / 1e9,
                             params=enc_p + head_p, counted='FlopCounterMode (two 15-s halves + sequence head)'))
            for name, fn in (('DeepSleepNet (released)', deepsleepnet_released), ('L-SeqSleepNet (released)', lseqsleepnet_released)):
                f, _ = fn()
                rows.append(dict(pipeline=pipe, model=name, sample=what, gflops=f / 1e9, params=None, counted='analytic'))
        rows.append(dict(pipeline=pipe, model='dDTF-GL', sample=what, gflops=ddtf_maps(pipe) / 1e9, params=None,
                         counted='analytic: dDTF maps (the XGBoost classifier adds < 0.0001 GFLOPs)'))
        plus = ddtf_plus(pipe)
        if plus:
            rows.append(dict(pipeline=pipe, model='dDTF-GL+', sample=what, gflops=plus[0] / 1e9, params=None,
                             counted=f'analytic (an upper bound with an RBF SVM): {plus[1]}'))
    for r in rows:
        C, T, fs, k, n_train, _ = PIPE[r['pipeline']]
        r['train_pflops_per_run'] = 3 * r['gflops'] * 1e9 * n_train * EPOCHS / 1e15 if not r['model'].startswith('dDTF-GL') else None
        r['n_train'] = n_train
        print(f"{r['pipeline']:17s} {r['model']:26s} {r['gflops']:10.4f} GFLOPs  "
              f"{(str(round(r['params'] / 1e6, 3)) + 'M') if r['params'] else '–':>10s}  "
              f"{('%.3f PFLOPs' % r['train_pflops_per_run']) if r['train_pflops_per_run'] is not None else ''}")
    json.dump(dict(epochs=EPOCHS, rows=rows), open(out, 'w'), indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
