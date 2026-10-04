# CBraMod

> J. Wang et al., *CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding*, ICLR 2025.
> [OpenReview](https://openreview.net/forum?id=NPNUHgHF2w)

- Released code: [wjq-learning/CBraMod](https://github.com/wjq-learning/CBraMod), mirrored in
  [woo0kim/CBraMod](https://github.com/woo0kim/CBraMod)
- Released weights: [weighting666/CBraMod](https://huggingface.co/weighting666/CBraMod) (`pretrained_weights.pth`,
  md5 355de9c3…); place it at `upstream/pretrained_weights/pretrained_weights.pth`
- `upstream/`: git submodule pinned at `b9e9610` (2026-08-07), the code that was evaluated
- License of the released code: MIT

## Experiments

| Experiment | Datasets | Question | Finding |
|---|---|---|---|
| [finetune-public-datasets](experiments/finetune-public-datasets) | PhysioNet-MI, BCIC-IV-2a, Mumtaz2016, MentalArithmetic | Do the released code and weights reproduce the paper's fine-tuning results? | No: all 12 metrics fall below the paper (balanced accuracy −0.015 to −0.137). EEGNet trained through the same pipeline beats CBraMod on BCIC-IV-2a. |

## Not yet covered

Public but not yet run: ISRUC, CHB-MIT. Access-gated: FACED, SEED-V, SEED-VIG, SHU-MI, TUAB, TUEV,
BCIC2020-3 (see [datasets](../../datasets/README.md)).
