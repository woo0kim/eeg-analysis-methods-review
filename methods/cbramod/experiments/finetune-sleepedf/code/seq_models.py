"""Every model as an epoch encoder inside CBraMod's released sleep-staging model.

CBraMod's ISRUC model (../../../upstream/models/model_for_isruc.py) encodes each 30-s epoch of a 20-epoch sequence
with the backbone, maps the flattened epoch features to 512 dimensions (Linear + GELU), runs one Transformer
encoder layer across the 20 epochs and classifies every epoch. SeqModel below is that model with the backbone
swapped; the head, sequence encoder and classifier are copied unchanged. Each encoder returns the features its own
model feeds to its own classifier:

  cbramod        released CBraMod backbone, proj_out removed (as model_for_isruc)  -> channels x 30 x 200
  labram         released LaBraM-Base, mean-pooled + fc_norm tokens (its fine-tuning head input) -> 200 per 15 s
  eegnet         our PyTorch EEGNet-8,2 (methods/eegnet/experiments/pytorch-cbramod-splits), flattened block-2 output
  conformer      released EEG Conformer convolution + 6 Transformer blocks (conformer_released.py), flattened tokens
  bd-*           braindecode 1.8.1's CBraMod / Labram / EEGNet / EEGConformer / ATCNet / DeepSleepNet, the same features

LaBraM's pre-trained time embedding has 16 positions (16 one-second patches), so a 30-s epoch is encoded as two 15-s
halves whose pooled features are concatenated; the pre-trained weights are used unchanged. ATCNet's released code is
TensorFlow; only braindecode's ATCNet (same layers and weights as the released network, see
methods/atcnet/experiments/braindecode-bciciv2a) can be run here. Its features are the five sliding windows' TCN
outputs, concatenated (its own final layer, whose max-norm axis differed from Keras', is replaced by the head).

Foundation-model encoders are registered as `backbone`, so CBraMod's Trainer gives them the backbone learning rate
(multi_lr: 1e-4) and everything else the head rate (5e-4 at batch 64); the supervised models have no `backbone`
and train entirely at the head rate, as EEGNet did in methods/eegnet/experiments/pytorch-cbramod-splits.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CBRAMOD_SRC = os.environ.get('CBRAMOD_SRC', os.path.normpath(os.path.join(HERE, '../../../upstream')))
LABRAM_SRC = os.environ.get('LABRAM_SRC', os.path.normpath(os.path.join(HERE, '../../../../labram/upstream')))
CBRAMOD_WEIGHTS = os.environ.get('CBRAMOD_WEIGHTS', os.path.join(CBRAMOD_SRC, 'pretrained_weights/pretrained_weights.pth'))
LABRAM_WEIGHTS = os.environ.get('LABRAM_WEIGHTS', os.path.join(LABRAM_SRC, 'checkpoints/labram-base.pth'))
# pinned revisions of braindecode's Hugging Face weights, as in the earlier braindecode experiments
BD_CBRAMOD = ('braindecode/cbramod-pretrained', '584cdc415913739a05d84bf0c1cb3db397764507')
BD_LABRAM = ('braindecode/labram-pretrained', '0563b6c626e7b40d9a36653b763715db94d945d7')
CH_LABRAM = ['FPZ', 'PZ']     # Fpz-Cz, Pz-Oz: the active electrode of each bipolar derivation
FS, EPOCH = 200, 6000


class SeqModel(nn.Module):
    """model_for_isruc.Model with the backbone replaced by `encoder` (feat_dim features per epoch)."""

    def __init__(self, encoder, feat_dim, num_of_classes, is_backbone):
        super().__init__()
        self.enc_name = 'backbone' if is_backbone else 'encoder'
        setattr(self, self.enc_name, encoder)
        # --- copied from model_for_isruc.Model (6*30*200 -> feat_dim) ---
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=4, dim_feedforward=2048, batch_first=True, activation=F.gelu, norm_first=True
        )
        self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1, enable_nested_tensor=False)
        self.classifier = nn.Linear(512, num_of_classes)

    def forward(self, x):
        bz, seq_len, ch_num, epoch_size = x.shape
        epoch_features = getattr(self, self.enc_name)(x.contiguous().view(bz * seq_len, ch_num, epoch_size))
        epoch_features = epoch_features.contiguous().view(bz, seq_len, -1)
        epoch_features = self.head(epoch_features)
        seq_features = self.sequence_encoder(epoch_features)
        out = self.classifier(seq_features)
        return out


# ---------------------------------------------------------------- released implementations
class CBraModEncoder(nn.Module):
    def __init__(self, n_ch, pretrained):
        super().__init__()
        sys.path.insert(0, CBRAMOD_SRC)
        from models.cbramod import CBraMod
        self.m = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8)
        if pretrained:
            self.m.load_state_dict(torch.load(CBRAMOD_WEIGHTS, map_location='cpu'))
        self.m.proj_out = nn.Identity()
        self.feat_dim = n_ch * 30 * 200

    def forward(self, x):                                  # (N, ch, 6000) -> (N, ch*30*200)
        return self.m(x.view(x.shape[0], x.shape[1], 30, 200)).flatten(1)


def _labram_input_chans():
    sys.path.insert(0, LABRAM_SRC)
    src = open(os.path.join(LABRAM_SRC, 'utils.py')).read()  # LaBraM's utils.standard_1020, without importing utils
    i = src.index('standard_1020 = [')
    vocab = eval(src[i + len('standard_1020 = '):src.index(']', i) + 1])
    return [0] + [vocab.index(c) + 1 for c in CH_LABRAM]   # = utils.get_input_chans(CH_LABRAM)


class LaBraMEncoder(nn.Module):
    """LaBraM-Base as run_class_finetuning.py builds and loads it with the README recipe's flags."""

    def __init__(self):
        super().__init__()
        self.input_chans = _labram_input_chans()
        import modeling_finetune  # noqa: F401  (registers labram_base_patch200_200 with timm)
        from timm.models import create_model
        self.m = create_model('labram_base_patch200_200', pretrained=False, num_classes=0, drop_rate=0.0,
                              drop_path_rate=0.1, attn_drop_rate=0.0, drop_block_rate=None, use_mean_pooling=True,
                              init_scale=0.001, use_rel_pos_bias=False, use_abs_pos_emb=True, init_values=0.1,
                              qkv_bias=False)
        ckpt = torch.load(LABRAM_WEIGHTS, map_location='cpu')['model']
        state = {k[8:]: v for k, v in ckpt.items() if k.startswith('student.')}
        state = {k: v for k, v in state.items() if 'relative_position_index' not in k}
        missing, unexpected = self.m.load_state_dict(state, strict=False)
        print(f'LaBraM: loaded {len(state)} student tensors; missing {missing}; unexpected {unexpected}')
        assert set(missing) <= {'fc_norm.weight', 'fc_norm.bias'}, missing
        self.feat_dim = 2 * 200

    def forward(self, x):                                  # (N, ch, 6000) -> two 15-s halves -> (N, 400)
        n, c, _ = x.shape
        halves = torch.cat([x[..., :3000], x[..., 3000:]], 0).view(2 * n, c, 15, 200)
        f = self.m.forward_features(halves, input_chans=self.input_chans)    # (2N, 200)
        return torch.cat([f[:n], f[n:]], 1)


class EEGNet(nn.Module):
    """Copied from methods/eegnet/experiments/pytorch-cbramod-splits/code/run_eegnet.py (EEGNet-8,2, 0.5-s
    temporal kernel); returns the flattened block-2 output its classifier reads."""

    def __init__(self, n_ch, n_times, F1=8, D=2, F2=16, kern=100, drop=0.25):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern), padding=(0, kern // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),   # depthwise spatial filter
            nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(drop))
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),  # separable conv
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(drop))
        with torch.no_grad():
            self.feat_dim = self.block2(self.block1(torch.zeros(1, 1, n_ch, n_times))).numel()

    def forward(self, x):                                  # (N, ch, T)
        return self.block2(self.block1(x.unsqueeze(1))).flatten(1)


class ConformerEncoder(nn.Module):
    def __init__(self, n_ch, n_times):
        super().__init__()
        from conformer_released import PatchEmbedding, TransformerEncoder
        self.patch = PatchEmbedding(40, n_chans=n_ch)
        self.transformer = TransformerEncoder(6, 40)
        with torch.no_grad():
            self.eval()
            self.feat_dim = self.patch(torch.zeros(1, 1, n_ch, n_times)).numel()
            self.train()

    def forward(self, x):                                  # released input: (N, 1, ch, T)
        return self.transformer(self.patch(x.unsqueeze(1))).flatten(1)


# ---------------------------------------------------------------- braindecode 1.8.1
class BDCBraModEncoder(nn.Module):
    """braindecode's CBraMod with its pre-trained weights; features = its return_features output (after proj_out,
    which braindecode keeps)."""

    def __init__(self, n_ch, pretrained):
        super().__init__()
        from braindecode.models import CBraMod
        kw = dict(n_chans=n_ch, n_times=EPOCH, n_outputs=5)
        self.m = CBraMod.from_pretrained(BD_CBRAMOD[0], revision=BD_CBRAMOD[1], **kw) if pretrained else CBraMod(**kw)
        self.m.final_layer = nn.Identity()
        self.feat_dim = n_ch * 30 * 200

    def forward(self, x):
        return self.m(x, return_features=True)['features'].flatten(1)


class BDLabramEncoder(nn.Module):
    """braindecode's Labram (15-s input, the window its weights fit) with the settings and weight loading of
    methods/labram/experiments/braindecode-cbramod-splits/code/run_braindecode.py; two halves as LaBraMEncoder."""

    def __init__(self):
        super().__init__()
        from braindecode.models import Labram
        from braindecode.models.labram import _LABRAM_CANONICAL_INDEX
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        self.m = Labram(n_times=3000, n_chans=len(CH_LABRAM), n_outputs=5, drop_prob=0.0, attn_drop_prob=0.0,
                        drop_path_prob=0.1, use_mean_pooling=True, init_scale=0.001, use_abs_pos_emb=True,
                        init_values=0.1, qkv_bias=False)
        state = load_file(hf_hub_download(BD_LABRAM[0], 'model.safetensors', revision=BD_LABRAM[1]))
        state['temporal_embedding'] = state['temporal_embedding'][:, :self.m.temporal_embedding.shape[1]]
        missing, unexpected = self.m.load_state_dict(state, strict=False)
        print(f'braindecode Labram: {len(state)} tensors; missing {missing}; unexpected {unexpected}')
        assert set(missing) == {'final_layer.weight', 'final_layer.bias', 'fc_norm.weight', 'fc_norm.bias'}, missing
        self.m.final_layer = nn.Identity()
        self.register_buffer('input_chans', torch.tensor([0] + [_LABRAM_CANONICAL_INDEX[c] + 1 for c in CH_LABRAM]),
                             persistent=False)
        self.feat_dim = 2 * 200

    def forward(self, x):
        n = x.shape[0]
        halves = torch.cat([x[..., :3000], x[..., 3000:]], 0)
        f = self.m.forward_features(halves, input_chans=self.input_chans)    # mean-pooled tokens + fc_norm
        return torch.cat([f[:n], f[n:]], 1)


class BDEEGNetEncoder(nn.Module):
    """braindecode's EEGNet with the settings of methods/eegnet/experiments/braindecode-cbramod-splits."""

    def __init__(self, n_ch):
        super().__init__()
        from braindecode.models import EEGNet as BDEEGNet
        self.m = BDEEGNet(n_chans=n_ch, n_outputs=5, n_times=EPOCH, F1=8, D=2, kernel_length=100, drop_prob=0.25,
                          final_layer_with_constraint=True)
        self.m.final_layer = nn.Identity()
        with torch.no_grad():
            self.eval()
            self.feat_dim = self(torch.zeros(1, n_ch, EPOCH)).shape[1]
            self.train()

    def forward(self, x):
        return self.m(x).flatten(1)


class BDConformerEncoder(nn.Module):
    def __init__(self, n_ch):
        super().__init__()
        from braindecode.models import EEGConformer
        self.m = EEGConformer(n_chans=n_ch, n_outputs=5, n_times=EPOCH, sfreq=FS, return_features=True)
        self.m.fc, self.m.final_layer = nn.Identity(), nn.Identity()   # unused with return_features=True
        with torch.no_grad():
            self.eval()
            self.feat_dim = self(torch.zeros(1, n_ch, EPOCH)).shape[1]
            self.train()

    def forward(self, x):
        return self.m(x).flatten(1)


class BDATCNetEncoder(nn.Module):
    """braindecode's ATCNet; ATCNet.forward up to the per-window TCN outputs (before its final layer)."""

    def __init__(self, n_ch):
        super().__init__()
        from braindecode.models import ATCNet
        self.m = ATCNet(n_chans=n_ch, n_outputs=5, n_times=EPOCH, sfreq=FS)
        self.m.final_layer = nn.ModuleList()
        self.feat_dim = self.m.n_windows * self.m.F2

    def forward(self, X):
        m = self.m
        X = m.dimshuffle(m.ensuredims(X))
        conv_feat = m.conv_block(X).view(-1, m.F2, m.Tc)
        feats = [tcn(att(conv_feat[..., i:i + m.Tw]))[..., -1]
                 for i, (att, tcn) in enumerate(zip(m.attention_blocks, m.temporal_conv_nets))]
        return torch.cat(feats, 1)


class BDDeepSleepNetEncoder(nn.Module):
    """braindecode's DeepSleepNet: its two CNN paths, the BiLSTM over a sequence of length 1 (braindecode's design) and
    the residual shortcut -> 1024 features per epoch. DeepSleepNet sets its first-layer sizes from the sampling rate
    (small path: kernel Fs/2, stride Fs/16; large path: kernel 4 Fs, stride Fs/2) and braindecode's defaults are those
    values at 100 Hz; here they are given for 200 Hz, so every later feature map has the original size."""

    def __init__(self, n_ch):
        super().__init__()
        from braindecode.models import DeepSleepNet
        self.m = DeepSleepNet(n_chans=n_ch, n_outputs=5, n_times=EPOCH, sfreq=FS,
                              small_first_kernel_size=100, small_first_stride=12, small_first_padding=44,
                              large_first_kernel_size=800, large_first_stride=100, large_first_padding=350)
        self.m.final_layer = nn.Identity()
        with torch.no_grad():
            self.eval()
            self.feat_dim = self(torch.zeros(2, n_ch, EPOCH)).shape[1]
            self.train()

    def forward(self, x):
        return self.m(x).flatten(1)


def build(name, n_ch=2, n_classes=5):
    """name -> SeqModel. Foundation-model encoders are the `backbone`."""
    if name == 'cbramod':
        enc, bb = CBraModEncoder(n_ch, pretrained=True), True
    elif name == 'cbramod-scratch':
        enc, bb = CBraModEncoder(n_ch, pretrained=False), True
    elif name == 'labram':
        enc, bb = LaBraMEncoder(), True
    elif name == 'eegnet':
        enc, bb = EEGNet(n_ch, EPOCH), False
    elif name == 'conformer':
        enc, bb = ConformerEncoder(n_ch, EPOCH), False
    elif name == 'bd-cbramod':
        enc, bb = BDCBraModEncoder(n_ch, pretrained=True), True
    elif name == 'bd-labram':
        enc, bb = BDLabramEncoder(), True
    elif name == 'bd-eegnet':
        enc, bb = BDEEGNetEncoder(n_ch), False
    elif name == 'bd-conformer':
        enc, bb = BDConformerEncoder(n_ch), False
    elif name == 'bd-atcnet':
        enc, bb = BDATCNetEncoder(n_ch), False
    elif name == 'bd-deepsleepnet':
        enc, bb = BDDeepSleepNetEncoder(n_ch), False
    else:
        raise ValueError(name)
    return SeqModel(enc, enc.feat_dim, n_classes, is_backbone=bb)
