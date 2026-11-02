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
| [finetune-public-datasets](experiments/finetune-public-datasets) | PhysioNet-MI, BCIC-IV-2a, Mumtaz2016, MentalArithmetic | Do the released code and weights reproduce the paper's fine-tuning results? | No: all 12 metrics fall below the paper (balanced accuracy −0.015 to −0.137). EEGNet trained through the same pipeline ([eegnet/pytorch-cbramod-splits](../eegnet/experiments/pytorch-cbramod-splits)) beats CBraMod on BCIC-IV-2a. Against [LaBraM-Base run the same way](../labram/experiments/finetune-cbramod-splits), CBraMod wins only on PhysioNet-MI. |
| [braindecode-public-datasets](experiments/braindecode-public-datasets) | same four | Does braindecode's `CBraMod` (braindecode 1.8.1, its pre-trained weights), fine-tuned through the same released pipeline, reproduce the released code? | Its weights and encoder are the released ones (identical tensors and outputs), but its task head is not: the pre-trained `proj_out` is kept and followed by one linear layer instead of the released 3-layer MLP. Within noise of the released code on BCIC-IV-2a, Mumtaz2016 and MentalArithmetic; lower on PhysioNet-MI (fine-tuned −0.020, from scratch −0.072, frozen −0.092). Controls trace the fine-tuned gap to the head's depth and the frozen gap to `proj_out`. Below the paper on all 12 metrics. |
| [finetune-sleepedf](experiments/finetune-sleepedf) | Sleep-EDF-78 | How do the released code and weights do on Sleep-EDF, with its released sleep-staging (ISRUC) pipeline? This experiment also holds the code every Sleep-EDF experiment shares, where each model replaces CBraMod's backbone. | There is no paper number to reproduce: CBraMod's sleep benchmark is ISRUC. κ 0.768 and balanced accuracy 0.754; pre-training adds 0.023 κ (p = 0.009). On the common test set CBraMod ties LaBraM-Base (κ 0.773), L-SeqSleepNet with its own released training (0.773) and braindecode's ATCNet (0.771): none of the differences is significant. |
| [braindecode-sleepedf](experiments/braindecode-sleepedf) | Sleep-EDF-78 | Does braindecode's `CBraMod`, in the same sleep-staging pipeline, give the released code's numbers? | Yes. Here both use the released sequence head, and the encoders differ only by the pre-trained `proj_out` braindecode keeps (checked: identical features once it is applied). κ 0.768 vs. 0.768 (p = 0.99); balanced accuracy −0.014 (p = 0.17), within the seed spread. |

## Not yet covered

Public but not yet run: ISRUC, CHB-MIT. Access-gated: FACED, SEED-V, SEED-VIG, SHU-MI, TUAB, TUEV,
BCIC2020-3 (see [datasets](../../datasets/README.md)).
