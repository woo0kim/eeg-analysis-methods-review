# EEGNet on Sleep-EDF, in CBraMod's sleep-staging pipeline

How does EEGNet-8,2 score on Sleep-EDF when it replaces CBraMod's backbone in CBraMod's released sleep-staging
pipeline? Only the encoder changes: the data, subject split, sequence head, trainer, checkpoint rule, seeds and metric
code are those of every model in the comparison. The pipeline, data and split are described in
[cbramod/finetune-sleepedf](../../../cbramod/experiments/finetune-sleepedf), which holds the shared code. This
experiment holds the 5 runs (seeds 0–4) of our PyTorch EEGNet, the implementation of
[pytorch-cbramod-splits](../pytorch-cbramod-splits).

EEGNet's paper evaluates P300, ERN, MRCP and motor imagery, not sleep, so this compares models; it reproduces no
published number.

## Results

Test set of the 8 test subjects (21,080 epochs), scored at the epoch with the best validation κ; mean ± SD over
seeds 0–4 (`results/results.txt`).

| Balanced acc. | Cohen's κ | Weighted F1 | Accuracy | Macro-F1 |
|---|---|---|---|---|
| 0.6855 ± 0.0139 | 0.6819 ± 0.0155 | 0.7694 ± 0.0087 | 0.7724 ± 0.0126 | 0.6785 ± 0.0076 |

| Per-stage F1 | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|
| EEGNet | 0.9004 | 0.3113 | 0.8011 | 0.7860 | 0.5937 |

On the common test set (20,976 epochs every Sleep-EDF model scores; `python tools/compare_sleepedf.py`):

- **EEGNet has the lowest κ of the original implementations** (0.6815 ± 0.0154); braindecode's EEGNet is level
  with it (0.6808, [braindecode-sleepedf](../braindecode-sleepedf)). It trails CBraMod by 0.086 κ and 0.068
  balanced accuracy (both p < 0.001).
- **REM and N1 are its weak stages**, with the lowest F1 of all models: REM 0.59 (braindecode's EEGNet 0.61, the
  others 0.66–0.83) and N1 0.31.
  Its N3 F1 (0.79) is in line with the stronger models, unlike EEG Conformer's.
- Its balanced accuracy (0.685) is above the released EEG Conformer's (0.645), the other small supervised model
  (braindecode's EEG Conformer: 0.680).
- Validation selection picks epochs 4–12 of 50. The best test epoch would add 0.025 κ.

On CBraMod's motor-imagery and clinical datasets the same EEGNet was competitive and beat CBraMod on BCIC-IV-2a
([pytorch-cbramod-splits](../pytorch-cbramod-splits)). On Sleep-EDF it is the weakest model.

## Paper vs. this reproduction

### What differs from EEGNet's paper, and why

| | EEGNet paper (Lawhern et al. 2018) | This experiment | Why |
|---|---|---|---|
| Data | ERP and motor-imagery trials, 128 Hz, 1–4 s | Sleep-EDF Fpz-Cz and Pz-Oz at 200 Hz, 30-s epochs in sequences of 20 | the comparison task |
| Model | EEGNet-8,2 with a dense softmax classifier | blocks 1–2 of EEGNet-8,2 (`code/run_eegnet.py` of [pytorch-cbramod-splits](../pytorch-cbramod-splits), copied in the shared `seq_models.py`); the flattened block-2 output (16 × 187) feeds CBraMod's sequence head | only the encoder differs between models |
| Temporal kernel | 64 samples (0.5 s at 128 Hz) | 100 samples (0.5 s at 200 Hz) | the same duration |
| Training | Adam, 500 epochs, the weights with the lowest validation loss | CBraMod's trainer: AdamW lr 5e-4, batch 64, cosine schedule, 50 epochs, best validation κ | one trainer for all models |

### Settings we had to choose

| Setting | Value | Why |
|---|---|---|
| Variant | EEGNet-8,2 (F1 8, D 2, F2 16), dropout 0.25 | the paper's cross-subject model, as in [pytorch-cbramod-splits](../pytorch-cbramod-splits) |
| Implementation | our PyTorch EEGNet (no max-norm constraints, PyTorch initialisation) | the implementation used in the other CBraMod-pipeline experiment; braindecode's runs in [braindecode-sleepedf](../braindecode-sleepedf) |
| Seeds | 0–4 | as every CBraMod-pipeline experiment |

### Run configuration

The shared run configuration ([cbramod/finetune-sleepedf](../../../cbramod/experiments/finetune-sleepedf#run-configuration)):
50 epochs, batch 64 sequences (one micro-batch), AdamW (weight decay 5e-2), lr 5e-4 for the whole model (no
`backbone`), cosine schedule, label smoothing 0.1. 4,688,789 parameters (encoder 1,424). About 45 min per run on
one Quadro M6000 12 GB. Environment: conda env `bd-labram` (`env/pip_freeze.txt`).

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

Runs: 2026-10-30 on mcl-server (`~/repro-sleepedf/runs/SleepEDF__eegnet__s*`).
