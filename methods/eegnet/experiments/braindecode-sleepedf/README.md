# EEGNet (braindecode) on Sleep-EDF, in CBraMod's sleep-staging pipeline

Does braindecode's EEGNet score like our PyTorch EEGNet on Sleep-EDF when both replace CBraMod's backbone in CBraMod's
released sleep-staging pipeline? Only the encoder changes: the data, subject split, sequence head, trainer, checkpoint
rule, seeds and metric code are those of the runs in [sleepedf-cbramod-pipeline](../sleepedf-cbramod-pipeline). The
pipeline, data and split are described in [cbramod/finetune-sleepedf](../../../cbramod/experiments/finetune-sleepedf),
which holds the shared code. This experiment holds the 5 runs (seeds 0–4) of braindecode 1.8.1's `EEGNet`.

EEGNet has no released PyTorch code. Under CBraMod's splits the two implementations agreed on all four public
datasets ([braindecode-cbramod-splits](../braindecode-cbramod-splits)). They have the same layers and the same
parameter count, and differ in max-norm constraints, batch-norm defaults and initialisation (below).

## Results

Test set of the 8 test subjects (21,080 epochs), scored at the epoch with the best validation κ; mean ± SD over
seeds 0–4 (`results/results.txt`).

| | Balanced acc. | Cohen's κ | Weighted F1 | Accuracy | Macro-F1 |
|---|---|---|---|---|---|
| EEGNet (braindecode) | 0.6868 ± 0.0159 | 0.6813 ± 0.0155 | 0.7719 ± 0.0090 | 0.7713 ± 0.0118 | 0.6818 ± 0.0115 |
| EEGNet (ours, [sleepedf-cbramod-pipeline](../sleepedf-cbramod-pipeline)) | 0.6855 ± 0.0139 | 0.6819 ± 0.0155 | 0.7694 ± 0.0087 | 0.7724 ± 0.0126 | 0.6785 ± 0.0076 |

| Per-stage F1 | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|
| EEGNet (braindecode) | 0.9065 | 0.3336 | 0.7941 | 0.7684 | 0.6064 |
| EEGNet (ours) | 0.9004 | 0.3113 | 0.8011 | 0.7860 | 0.5937 |

On the common test set (20,976 epochs every Sleep-EDF model scores; `python tools/compare_sleepedf.py`):

- **The two implementations agree.** braindecode − ours is −0.0007 κ (p = 0.95), +0.0013 balanced accuracy
  (p = 0.89) and +0.0032 macro-F1 (p = 0.62). No per-stage F1 differs significantly (closest: N2, −0.007,
  p = 0.071). The seed ranges overlap: κ 0.655–0.697 against 0.664–0.706.
- **Either way EEGNet is the weakest model on Sleep-EDF**, 0.087 κ below CBraMod (p < 0.001), with the weakest REM
  and N1 stages. Under CBraMod's splits it beat both foundation models on BCIC-IV-2a, in both implementations.
- Validation selection picks epochs 10–21 of 50 (ours: 4–12); the best test epoch would add 0.021 κ, the last
  epoch scores 0.6856.

## Paper vs. this reproduction

EEGNet's paper evaluates no sleep data, so the reference here is our PyTorch EEGNet on the same pipeline.

### What differs from our PyTorch EEGNet, and why

| | Our PyTorch EEGNet ([sleepedf-cbramod-pipeline](../sleepedf-cbramod-pipeline)) | This experiment | Why |
|---|---|---|---|
| Implementation | `code/run_eegnet.py` of [pytorch-cbramod-splits](../pytorch-cbramod-splits), copied in the shared `seq_models.py` | braindecode 1.8.1 `EEGNet` | the question of this experiment |
| Architecture | EEGNet-8,2: temporal kernel 100 (0.5 s at 200 Hz), depthwise spatial, separable kernel 16, average pooling 4 and 8, dropout 0.25 | the same values | – |
| Max-norm constraint | none | depthwise spatial filter, max norm 1 (the classifier's max norm 0.25 goes with the classifier the sequence head replaces) | EEGNet paper (Lawhern et al. 2018); braindecode's default |
| Batch norm | PyTorch defaults (momentum 0.1, ε 1e-5) | braindecode defaults (momentum 0.01, ε 1e-3, the Keras values of the original code) | braindecode's implementation |
| Initialisation | PyTorch defaults (Kaiming uniform) | Glorot uniform, zero bias (the Keras default of the original code) | braindecode's implementation |
| Sequence head, trainer, data, seeds | CBraMod's released sequence head and trainer | the same, shared code | only the encoder may differ |

### Settings we had to choose

| Setting | Value | Why |
|---|---|---|
| Variant | EEGNet-8,2 (F1 = 8, D = 2, F2 = 16), temporal kernel 100, dropout 0.25 | as our PyTorch runs and [braindecode-cbramod-splits](../braindecode-cbramod-splits) |
| Features | the flattened block-2 output (16 × 187), before braindecode's final layer | EEGNet's own classifier input, as in our PyTorch runs |
| Other layers | braindecode defaults | – |
| Seeds | 0–4 | as every CBraMod-pipeline experiment |

### Run configuration

The shared run configuration ([cbramod/finetune-sleepedf](../../../cbramod/experiments/finetune-sleepedf#run-configuration)):
50 epochs, batch 64 sequences (one micro-batch), AdamW (weight decay 5e-2), lr 5e-4 for the whole model (no
`backbone`), cosine schedule, label smoothing 0.1. 4,688,789 parameters (encoder 1,424), as our PyTorch runs. About
45 min per run on one Quadro M6000 12 GB. Environment: conda env `bd-labram`, braindecode 1.8.1
(`env/pip_freeze.txt`).

## Layout

```
code/reproduce.sh        collect (recompute results/ from logs/runs) or train (enqueue the 5 runs), with the
                         shared code in ../../../cbramod/experiments/finetune-sleepedf/code
logs/runs/<run>/         log.txt (progress bars removed), metrics.json (per-epoch validation and test metrics),
                         test_preds.npz (test labels and predictions at the selected epoch)
results/                 results.txt, results.json (with common_* metrics on the common test set), summary.csv
env/pip_freeze.txt       packages of the runs
```

## Reproducing

```bash
code/reproduce.sh collect        # numpy, scipy, scikit-learn; diffs against results/results.txt
code/reproduce.sh train          # needs the preprocessed data (shared README, "Reproducing")
```

## Provenance

Runs: 2026-11-01 on mcl-server (`~/repro-sleepedf/runs/SleepEDF__bd-eegnet__s*`). braindecode: the pinned submodule
`methods/eegnet/upstream` (1.8.1).
