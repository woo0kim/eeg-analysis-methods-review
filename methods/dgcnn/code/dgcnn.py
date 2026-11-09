"""DGCNN (Song, Zheng, Song, Cui, IEEE TAC 2020), PyTorch.

Written from the authors' TensorFlow code (lib/models.py, class cgcnn, and EEG/eeg_DE.py; methods/dgcnn/fetch_upstream.sh)
with the settings eeg_DE.py uses on SEED. Their code is a modified cnn_graph (Defferrard et al., 2016):

  W        = ReLU(A + b), A a free N x N matrix (asymmetric, diagonal kept), b one scalar
  L        = I - D W D, D = diag(1 / (sqrt(sum_j W_ij) + 1e-5))          no rescaling by lambda_max
  features = concat(T_0(L) x, ..., T_{K-1}(L) x)  per node               T_0 = I, T_1 = L, T_k = 2 L T_{k-1} - T_{k-2}
  y        = features @ G, G of shape (F_in * K, F)                      shared by all nodes: the paper's 1 x 1 convolution
  h        = ReLU(y + c), c one bias per filter
  out      = FC(256) + ReLU + dropout, then a linear layer to the classes

Initialisation (eeg_DE.py / models.py): A ~ truncated normal, std 0.3 off the diagonal and 0.6 on it; b ~ truncated
normal, std 0.1; G and the FC weights ~ truncated normal, std 0.1; every bias = 0.1. TensorFlow's truncated normal
redraws outside two standard deviations.

The loss adds regularization * sum(0.5 * ||p||^2) over A, b, G and both FC layers (weights and biases), not c.

DEFeatures computes the input the paper uses (differential entropy in five bands) from raw windows, for pipelines that
feed signals. encoder=True returns the flattened graph output (N * F values), the input of the model's own classifier.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

BANDS = ((1, 3), (4, 7), (8, 13), (14, 30), (31, 50))       # delta, theta, alpha, beta, gamma: SEED's bands, Hz


def _truncnorm(shape, std):
    t = nn.init.trunc_normal_(torch.empty(shape), mean=0.0, std=1.0, a=-2.0, b=2.0)
    return t * std


class DGCNN(nn.Module):
    def __init__(self, n_nodes, n_feat, n_classes, K=2, n_filters=64, fc=256, drop=0.5, encoder=False):
        super().__init__()
        self.K, self.encoder = K, encoder
        self.A = nn.Parameter(_truncnorm((n_nodes, n_nodes), 0.3 * torch.eye(n_nodes) + 0.3))
        self.b_A = nn.Parameter(_truncnorm((1,), 0.1))
        self.G = nn.Parameter(_truncnorm((n_feat * K, n_filters), 0.1))
        self.c = nn.Parameter(torch.full((1, 1, n_filters), 0.1))
        self.feat_dim = n_nodes * n_filters
        if not encoder:
            self.fc_w = nn.Parameter(_truncnorm((n_nodes * n_filters, fc), 0.1))
            self.fc_b = nn.Parameter(torch.full((fc,), 0.1))
            self.out_w = nn.Parameter(_truncnorm((fc, n_classes), 0.1))
            self.out_b = nn.Parameter(torch.full((n_classes,), 0.1))
            self.drop = nn.Dropout(drop)

    def laplacian(self):
        W = F.relu(self.A + self.b_A)
        d = 1.0 / (torch.sqrt(W.sum(dim=1)) + 1e-5)
        return torch.eye(W.shape[0], device=W.device) - d[:, None] * W * d[None, :]

    def graph_features(self, x):                             # x: (batch, nodes, features)
        L = self.laplacian()
        t = [x]
        if self.K > 1:
            t.append(torch.matmul(L, x))
        for _ in range(2, self.K):
            t.append(2 * torch.matmul(L, t[-1]) - t[-2])
        z = torch.stack(t, dim=3).flatten(2)                 # (batch, nodes, F_in * K), features first as in chebyshev5
        return F.relu(torch.matmul(z, self.G) + self.c).flatten(1)

    def forward(self, x):
        h = self.graph_features(x)
        if self.encoder:
            return h
        h = self.drop(F.relu(torch.matmul(h, self.fc_w) + self.fc_b))
        return torch.matmul(h, self.out_w) + self.out_b

    def l2(self):
        names = ['A', 'b_A', 'G'] + ([] if self.encoder else ['fc_w', 'fc_b', 'out_w', 'out_b'])
        return sum(0.5 * getattr(self, n).pow(2).sum() for n in names)


class DEFeatures(nn.Module):
    """Differential entropy of consecutive 1-s segments: 0.5 * log(2 * pi * e * band power), per channel and band.

    (batch, channels, time) -> (batch, channels, 5 * seconds). A segment's band power is the mean squared band-passed
    signal, computed from a Hann-windowed FFT by Parseval, the same quantity SEED's short-time features estimate.
    mean / scale standardise each band, from the training set (set_norm); they stay 0 / 1 otherwise.
    """

    def __init__(self, sfreq=200, seconds=1):
        super().__init__()
        self.n, self.seconds = int(sfreq), int(seconds)
        freqs = torch.fft.rfftfreq(self.n, 1.0 / sfreq)
        self.register_buffer('win', torch.hann_window(self.n, periodic=True))
        self.register_buffer('masks', torch.stack([((freqs >= lo) & (freqs <= hi)).float() for lo, hi in BANDS]))
        self.register_buffer('mean', torch.zeros(1, 1, 5 * self.seconds))
        self.register_buffer('scale', torch.ones(1, 1, 5 * self.seconds))

    def set_norm(self, mean, scale):
        self.mean[:] = mean
        self.scale[:] = scale

    def raw(self, x):
        b, c, t = x.shape
        assert t == self.n * self.seconds, (t, self.n, self.seconds)
        seg = x.reshape(b, c, self.seconds, self.n) * self.win
        p = torch.fft.rfft(seg, dim=-1).abs().pow(2) / self.win.pow(2).sum()        # power per frequency bin
        band = 2 * torch.einsum('bcsf,qf->bcsq', p, self.masks) / self.n            # (batch, channels, seconds, 5)
        return 0.5 * torch.log(2 * math.pi * math.e * band + 1e-12).flatten(2)

    def forward(self, x):
        return (self.raw(x) - self.mean) / self.scale


class WithDE(nn.Module):
    """DEFeatures then a graph model: (..., channels, time) -> net((batch, channels, 5 * seconds)).

    Each band is standardised with training-set statistics. fit_norm reads them from the first batches of a shuffled
    loader; without a loader, the first training batch sets them (lazy=True), for training loops with no hook. (The
    authors' 2 (x - mean) / mean needs positive means, which the units of a pipeline do not guarantee.)
    """

    def __init__(self, net, n_ch, sfreq=200, seconds=1, lazy=False):
        super().__init__()
        self.de, self.net, self.n_ch, self.lazy = DEFeatures(sfreq, seconds), net, n_ch, lazy
        self.T = int(sfreq) * int(seconds)

    def set_stats(self, z):
        self.de.set_norm(z.mean((0, 1), keepdim=True), z.std((0, 1), keepdim=True))

    @torch.no_grad()
    def fit_norm(self, loader, n_batches=200):
        zs = []
        for i, (x, _) in enumerate(loader):
            if i == n_batches:
                break
            zs.append(self.de.raw(x.float().reshape(-1, self.n_ch, self.T)))
        self.set_stats(torch.cat(zs))

    def forward(self, x):
        x = x.reshape(-1, self.n_ch, self.T)
        if self.lazy and self.training:
            self.lazy = False
            with torch.no_grad():
                self.set_stats(self.de.raw(x))
        return self.net(self.de(x))


def chs_info(names, sfreq=200):
    """MNE channel info with positions from channel names, for braindecode's DGCNN, whose initial graph needs them.
    CB1 / CB2 and the ear channel A2 have no entry in MNE's standard_1005: the nearest named electrode stands in."""
    import mne
    swap = {'CB1': 'I1', 'CB2': 'I2', 'A2': 'TP10'}
    fixed = [swap.get(c, c).replace('FP', 'Fp').replace('Z', 'z') for c in names]
    info = mne.create_info(fixed, sfreq, 'eeg')
    info.set_montage('standard_1005', match_case=False)
    return info['chs']
