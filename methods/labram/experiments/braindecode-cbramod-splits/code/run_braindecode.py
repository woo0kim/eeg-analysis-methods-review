"""LaBraM-Base as implemented in braindecode, fine-tuned by LaBraM's released run_class_finetuning.py on CBraMod's data.

Everything except the model is the released-code experiment ../../finetune-cbramod-splits: its code/run_labram.py is
imported, which installs the same two hooks into LaBraM's released script (CBraMod's LMDB data and splits; the
validation / test probabilities of every epoch saved to preds.npz), and the script runs with the same recipe
(code/add_jobs.py). Only get_models() is replaced: it builds braindecode's Labram with the settings the released
get_models() passes to LaBraM's own model, and loads braindecode's pre-trained weights.

Three adapters, none of which changes the computation:
  - forward(x, input_chans): LaBraM's engine passes (batch, channels, patches, 200) and LaBraM's channel indices;
    braindecode takes (batch, channels, time) and channel names, which it maps to the same indices (checked in
    code/check_equivalence.py).
  - no_weight_decay(): LaBraM's optimizer asks the model which tensors to exempt from weight decay; braindecode's
    Labram has no such method, so it returns braindecode's names for LaBraM's {'pos_embed', 'cls_token', 'time_embed'}.
  - layer-wise lr decay: LaBraM's optim_factory.get_num_layer_for_vit() assigns a layer to each parameter by name.
    braindecode's position_embedding and temporal_embedding are looked up under LaBraM's names (pos_embed,
    time_embed), so every tensor gets the same learning-rate scale as in the released runs.
Pre-trained weights: braindecode/labram-pretrained (pinned revision), the same tensors as checkpoints/labram-base.pth
(check_equivalence.py). from_pretrained() cannot load them into a model for 4- or 5-s windows, because braindecode
sizes temporal_embedding by n_times (16 rows saved for 15 patches; 5 or 6 needed here), so the file is loaded
directly and temporal_embedding keeps its first rows. Both implementations only use rows 0..n_patches-1.

usage: as ../../finetune-cbramod-splits/code/run_labram.py, without --finetune (see code/add_jobs.py)
"""
import argparse
import os
import sys

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from braindecode.models import Labram

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get('LABRAM_CODE', os.path.join(HERE, '../../finetune-cbramod-splits/code')))
import run_labram  # noqa: E402  (puts LaBraM's code on sys.path and installs the data / prediction hooks)
import optim_factory  # noqa: E402  (LaBraM's)

rcf = run_labram.rcf
HF_REPO = 'braindecode/labram-pretrained'
HF_REVISION = '0563b6c626e7b40d9a36653b763715db94d945d7'   # 2026-02-06
N_TIMES = {'PhysioNet-MI': 800, 'BCIC-IV-2a': 800, 'Mumtaz2016': 1000, 'MentalArithmetic': 1000}   # 200 Hz
TO_LABRAM_NAME = {'position_embedding': 'pos_embed', 'temporal_embedding': 'time_embed'}


class LabramForEngine(Labram):
    """braindecode's Labram with the two methods LaBraM's training code calls on its own model."""

    ch_names = None

    def forward(self, x, input_chans=None, **kwargs):
        if x.ndim == 4:                    # LaBraM's engine; 3-D input comes from braindecode's own model summary
            x = x.reshape(x.shape[0], x.shape[1], -1)
        return super().forward(x, ch_names=self.ch_names)

    def no_weight_decay(self):
        return {'position_embedding', 'cls_token', 'temporal_embedding'}


def build(dataset, nb_classes, args):
    """braindecode's Labram with the settings the released get_models() gives create_model(), and the weights."""
    names = run_labram.CHANNELS[dataset]
    model = LabramForEngine(
        n_times=N_TIMES[dataset], n_chans=len(names), n_outputs=nb_classes,
        drop_prob=args.drop, attn_drop_prob=args.attn_drop_rate, drop_path_prob=args.drop_path,
        use_mean_pooling=args.use_mean_pooling, init_scale=args.init_scale, use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value, qkv_bias=args.qkv_bias)
    model.ch_names = names
    state = load_file(hf_hub_download(HF_REPO, 'model.safetensors', revision=HF_REVISION))
    state['temporal_embedding'] = state['temporal_embedding'][:, :model.temporal_embedding.shape[1]]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'Load {HF_REPO}@{HF_REVISION[:7]}: {len(state)} tensors; not in the file (trained from scratch): '
          f'{missing}; in the file but unused: {unexpected}')
    # as in the released runs: a new head and fc_norm (mean pooling); the pre-training `norm` is unused
    assert set(missing) == {'final_layer.weight', 'final_layer.bias', 'fc_norm.weight', 'fc_norm.bias'}, missing
    assert set(unexpected) <= {'norm.weight', 'norm.bias'}, unexpected
    return model


def get_models(args):
    assert args.model == 'labram_base_patch200_200', args.model
    assert not args.rel_pos_bias, 'braindecode has no relative position bias in this mode; use --disable_rel_pos_bias'
    return build(args.dataset, args.nb_classes, args)


_get_num_layer_for_vit = optim_factory.get_num_layer_for_vit


def get_num_layer_for_vit(var_name, num_max_layer):
    return _get_num_layer_for_vit(TO_LABRAM_NAME.get(var_name, var_name), num_max_layer)


optim_factory.get_num_layer_for_vit = get_num_layer_for_vit
rcf.get_models = get_models

if __name__ == '__main__':
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)   # else '--dataset' would match --datasets_dir
    ap.add_argument('--datasets_dir', required=True)
    run_labram.OURS, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    run_labram.RCF_ARGS, ds_init = rcf.get_args()
    assert run_labram.RCF_ARGS.output_dir, '--output_dir is required (preds.npz and LaBraM log.txt go there)'
    assert not run_labram.RCF_ARGS.finetune, 'weights come from braindecode; do not pass --finetune'
    os.makedirs(run_labram.RCF_ARGS.output_dir, exist_ok=True)
    rcf.main(run_labram.RCF_ARGS, ds_init)
