"""Is braindecode's Labram the released LaBraM-Base, and are its Hugging Face weights the released weights?

1. Compare braindecode's pinned Hugging Face weights (run_braindecode.HF_REPO / HF_REVISION) tensor by tensor with the
   `student.` tensors of the released checkpoints/labram-base.pth, after translating braindecode's names.
2. Check that braindecode maps each dataset's channel names to the position-embedding rows LaBraM's
   utils.get_input_chans() selects.
3. Build both fine-tuning models as the runs do: LaBraM's own model through the released get_models() with the README
   recipe flags and the released checkpoint-loading code, and braindecode's through run_braindecode.build(). Copy the
   released model's freshly initialised head and fc_norm into braindecode's, then feed the same input (eval mode,
   float64) and compare the logits.
CPU only. usage: python code/check_equivalence.py > results/equivalence.txt
"""
import os, re, sys
from collections import OrderedDict
import torch
import braindecode
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from braindecode.models.labram import LABRAM_CHANNEL_ORDER

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get('LABRAM_CODE', os.path.join(HERE, '../../finetune-cbramod-splits/code')))
import run_labram  # noqa: E402
rcf = run_labram.rcf
released_get_models = rcf.get_models            # run_braindecode replaces rcf.get_models on import
sys.path.insert(0, HERE)
import run_braindecode as bd  # noqa: E402
import utils  # noqa: E402  (LaBraM's)

CKPT = os.path.join(bd.run_labram.SRC, 'checkpoints/labram-base.pth')
RENAME = [(r'^pos_embed$', 'position_embedding'), (r'^time_embed$', 'temporal_embedding'),
          (r'^patch_embed\.', 'patch_embed.temporal_conv.'), (r'\.mlp\.fc1\.', '.mlp.0.'), (r'\.mlp\.fc2\.', '.mlp.2.')]


def to_braindecode(name):
    for pat, rep in RENAME:
        name = re.sub(pat, rep, name)
    return name


print(f'torch {torch.__version__}, braindecode {braindecode.__version__}')
# 1. weights
ck = torch.load(CKPT, map_location='cpu')['model']
student = OrderedDict((k[len('student.'):], v) for k, v in ck.items() if k.startswith('student.'))
hf = load_file(hf_hub_download(bd.HF_REPO, 'model.safetensors', revision=bd.HF_REVISION))
print(f'\nreleased labram-base.pth: {len(student)} student tensors ({sum(v.numel() for v in student.values()):,} values)')
print(f'{bd.HF_REPO}@{bd.HF_REVISION[:7]} model.safetensors: {len(hf)} tensors ({sum(v.numel() for v in hf.values()):,} values)')
mapped = {to_braindecode(k): k for k in student}
print('released tensors with no braindecode counterpart (pre-training only):', [mapped[k] for k in mapped if k not in hf])
print('braindecode tensors with no released counterpart:', [k for k in hf if k not in mapped] or 'none')
common = [k for k in hf if k in mapped]
diff = [k for k in common if student[mapped[k]].shape != hf[k].shape or not torch.equal(student[mapped[k]], hf[k])]
generic = lambda k: re.sub(r'blocks\.[0-9]+', 'blocks.*', re.sub(r'\.(weight|bias)$', '', k))  # noqa: E731
print('renamed (released -> braindecode):', sorted({f'{generic(mapped[k])} -> {generic(k)}' for k in common if mapped[k] != k}))
print(f'tensors compared: {len(common)}; identical: {len(common) - len(diff)}; different: {diff or "none"}')

# 2. channel indices
print('\nPosition-embedding rows (index 0 = [CLS]) for each dataset\'s channels:')
for ds, names in run_labram.CHANNELS.items():
    lab = utils.get_input_chans(names)
    order = [c.upper() for c in LABRAM_CHANNEL_ORDER]
    bdi = [0] + [1 + order.index(n) for n in names]          # what Labram.forward(ch_names=...) selects
    print(f'  {ds:16s} {len(names)} channels: LaBraM {lab[:6]}... braindecode {bdi[:6]}... identical: {lab == bdi}')

# 3. forward pass of the two fine-tuning models
sys.argv = ['x', '--model', 'labram_base_patch200_200', '--disable_rel_pos_bias', '--abs_pos_emb', '--disable_qkv_bias',
            '--drop_path', '0.1', '--layer_decay', '0.65', '--dataset', 'BCIC-IV-2a']
args, _ = rcf.get_args()
print(f'\nRecipe model settings: use_mean_pooling={args.use_mean_pooling} drop={args.drop} attn_drop={args.attn_drop_rate} '
      f'drop_path={args.drop_path} init_scale={args.init_scale} layer_scale={args.layer_scale_init_value} '
      f'qkv_bias={args.qkv_bias} rel_pos_bias={args.rel_pos_bias} abs_pos_emb={args.abs_pos_emb}')
print('Same input, eval mode, float64: released model (released weights) vs. braindecode (braindecode weights), '
      'with the released head and fc_norm copied into braindecode')
for ds, names in run_labram.CHANNELS.items():
    args.nb_classes = run_labram.N_CLASSES[ds]
    rel = released_get_models(args)
    sd = OrderedDict((k, v) for k, v in student.items() if 'relative_position_index' not in k)
    utils.load_state_dict(rel, sd, prefix='')          # what run_class_finetuning.main() does with --finetune
    model = bd.build(ds, args.nb_classes, args)
    model.final_layer.load_state_dict(rel.head.state_dict())
    model.fc_norm.load_state_dict(rel.fc_norm.state_dict())
    rel, model = rel.double().eval(), model.double().eval()
    n_patch = bd.N_TIMES[ds] // 200
    x = torch.randn(16, len(names), n_patch, 200, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        a = rel(x, input_chans=utils.get_input_chans(names))
        b = model(x, input_chans=None)
        fa = rel.forward_features(x, input_chans=utils.get_input_chans(names))
        fb = model.forward_features(x.reshape(16, len(names), -1), input_chans=torch.tensor(utils.get_input_chans(names)))
    n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    print(f'  {ds:16s} input {tuple(x.shape)}: max |pooled-feature difference| {(fa - fb).abs().max().item():.3g}, '
          f'max |logit difference| {(a - b).abs().max().item():.3g}; parameters released {n(rel):,}, braindecode {n(model):,}')
