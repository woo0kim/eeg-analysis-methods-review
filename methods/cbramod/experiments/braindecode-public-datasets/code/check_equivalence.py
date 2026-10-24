"""Is braindecode's CBraMod the released model, and are its Hugging Face weights the released weights?

1. Compare braindecode's pinned Hugging Face weights (run_braindecode.HF_REPO / HF_REVISION) tensor by tensor with
   the released pretrained_weights.pth (HuggingFace weighting666/CBraMod), and check that from_pretrained() really
   loaded every tensor (braindecode loads with strict=False, which would skip mismatched names silently).
2. Feed the same input through the released backbone (upstream/models/cbramod.py, released weights) and through
   braindecode's encoder (from_pretrained, task head removed), in eval mode and float64, and compare the outputs.
   Both end with the pre-training projection `proj_out`; the released fine-tuning model then replaces it with
   Identity and adds its 3-layer `all_patch_reps` head, braindecode keeps it and adds one Linear layer.
3. Count parameters of the two fine-tuning models (backbone + head) per dataset.
CPU only. usage: python code/check_equivalence.py > results/equivalence.txt
Released weights: $CBRAMOD_WEIGHTS (default ~/CBraMod/pretrained_weights/pretrained_weights.pth)
"""
import argparse, os, sys
import torch
import braindecode
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_braindecode import CFG, HF_REPO, HF_REVISION, BraindecodeCBraMod  # noqa: E402  (puts upstream on sys.path)
from models.cbramod import CBraMod as ReleasedCBraMod  # noqa: E402
from models import model_for_physio, model_for_bciciv2a, model_for_mumtaz, model_for_stress  # noqa: E402

WEIGHTS = os.path.expanduser(os.environ.get('CBRAMOD_WEIGHTS', '~/CBraMod/pretrained_weights/pretrained_weights.pth'))
print(f'torch {torch.__version__}, braindecode {braindecode.__version__}')

# 1. weights
released = torch.load(WEIGHTS, map_location='cpu')
hf = load_file(hf_hub_download(HF_REPO, 'model.safetensors', revision=HF_REVISION))
print(f'\nreleased {os.path.basename(WEIGHTS)}: {len(released)} tensors, '
      f'{sum(v.numel() for v in released.values()):,} values')
print(f'{HF_REPO}@{HF_REVISION[:7]} model.safetensors: {len(hf)} tensors, {sum(v.numel() for v in hf.values()):,} values')
only_rel, only_hf = sorted(set(released) - set(hf)), sorted(set(hf) - set(released))
print('names only in the released file:', only_rel or 'none')
print('names only in the braindecode file:', only_hf or 'none')
common = sorted(set(released) & set(hf))
diff = [k for k in common if released[k].shape != hf[k].shape or not torch.equal(released[k].float(), hf[k].float())]
print(f'tensors with the same name: {len(common)}; identical: {len(common) - len(diff)}; different: {diff or "none"}')

model = BraindecodeCBraMod(22, 800, 4, pretrained=True)
loaded = model.backbone.state_dict()
not_loaded = [k for k, v in hf.items() if k not in loaded or not torch.equal(loaded[k].float(), v.float())]
print(f'from_pretrained(): {len(hf) - len(not_loaded)} of {len(hf)} file tensors are in the model; '
      f'not loaded: {not_loaded or "none"}')
print('model tensors not in the file (trained from scratch in fine-tuning):',
      [k for k in loaded if k not in hf] + [f'head.{k}' for k in model.head.state_dict()])

# 2. forward pass, released weights in the released backbone vs. braindecode's encoder
rel = ReleasedCBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8)
rel.load_state_dict(released)
print('\nSame input, eval mode, float64: released backbone (released weights) vs. braindecode encoder (from_pretrained)')
for ds, (_, n_ch, n_times, n_out, _) in CFG.items():
    bd = BraindecodeCBraMod(n_ch, n_times, n_out, pretrained=True).double().eval()
    x = torch.randn(16, n_ch, n_times // 200, 200, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        a = rel.double().eval()(x)
        b = bd.backbone(x.reshape(16, n_ch, n_times))
    print(f'  {ds:16s} input {tuple(x.shape)}: output {tuple(b.shape)}, max |difference| '
          f'{(a - b).abs().max().item():.3g} (output scale: max |value| {a.abs().max().item():.3g})')

# 3. parameter counts of the fine-tuning models
print('\nParameters as fine-tuned: released Model (backbone with proj_out -> Identity + all_patch_reps head) vs. '
      'braindecode (backbone with proj_out + final_layer)')
released_models = {'PhysioNet-MI': model_for_physio, 'BCIC-IV-2a': model_for_bciciv2a, 'Mumtaz2016': model_for_mumtaz,
                   'MentalArithmetic': model_for_stress}
for ds, (_, n_ch, n_times, n_out, _) in CFG.items():
    p = argparse.Namespace(use_pretrained_weights=False, classifier='all_patch_reps', dropout=0.1,
                           num_of_classes=max(n_out, 2), cuda=0)
    rm = released_models[ds].Model(p)
    bd = BraindecodeCBraMod(n_ch, n_times, n_out, pretrained=False)
    n = lambda mod: sum(q.numel() for q in mod.parameters())  # noqa: E731
    print(f'  {ds:16s} released: backbone {n(rm.backbone):,} + head {n(rm.classifier):,} = {n(rm):,} | '
          f'braindecode: backbone {n(bd.backbone):,} + head {n(bd.head):,} = {n(bd):,}')
