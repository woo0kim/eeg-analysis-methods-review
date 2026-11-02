# EEG Conformer on Sleep-EDF, in CBraMod's sleep-staging pipeline

How does the released EEG Conformer score on Sleep-EDF when it replaces CBraMod's backbone in CBraMod's released
sleep-staging pipeline? Only the encoder changes: the data, subject split, sequence head, trainer, checkpoint rule,
seeds and metric code are those of every model in the comparison. The pipeline, data and split are described in
[cbramod/finetune-sleepedf](../../../cbramod/experiments/finetune-sleepedf), which holds the shared code. This
experiment holds the 5 EEG Conformer runs (seeds 0–4).

EEG Conformer's paper evaluates motor imagery (BCI IV 2a, 2b) and emotion (SEED), not sleep, so this compares
models; it reproduces no published number.

## Results

Test set of the 8 test subjects (21,080 epochs), scored at the epoch with the best validation κ; mean ± SD over
seeds 0–4 (`results/results.txt`).

| Balanced acc. | Cohen's κ | Weighted F1 | Accuracy | Macro-F1 |
|---|---|---|---|---|
| 0.6449 ± 0.0266 | 0.6880 ± 0.0188 | 0.7711 ± 0.0127 | 0.7804 ± 0.0131 | 0.6583 ± 0.0234 |

| Per-stage F1 | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|
| EEG Conformer | 0.9082 | 0.3469 | 0.8127 | 0.5611 | 0.6626 |

On the common test set (20,976 epochs every Sleep-EDF model scores; `python tools/compare_sleepedf.py`):

- **Among the original implementations EEG Conformer is second to last** (κ 0.6876 ± 0.0190), ahead only of
  EEGNet. It trails CBraMod by 0.080 κ and 0.109 balanced accuracy (both p < 0.001). Its balanced accuracy (0.645)
  is the lowest of all models.
- **Deep sleep is its weak stage.** N3 F1 is 0.56, against 0.71 for braindecode's EEG Conformer and 0.77–0.81 for
  every other pipeline model, and REM F1 is 0.66. braindecode's EEG Conformer is within noise overall (κ 0.701,
  p = 0.47) but higher on N3 (p = 0.047; [braindecode-sleepedf](../braindecode-sleepedf)).
- **The two Conformers gain the most from test-set epoch selection:** +0.034 κ at the best test epoch (braindecode:
  +0.037), against +0.005 to +0.025 for the other models. They also have the largest seed spreads in balanced
  accuracy (SD 0.027; braindecode: 0.030). Validation selection picks epochs 10–18 of 50.

## Paper vs. this reproduction

### What differs from EEG Conformer's paper and released code, and why

| | EEG Conformer paper / released `conformer.py` | This experiment | Why |
|---|---|---|---|
| Data | BCI IV 2a: 22 channels at 250 Hz, 4-s trials | Sleep-EDF Fpz-Cz and Pz-Oz at 200 Hz, 30-s epochs in sequences of 20 | the comparison task |
| Model | convolution + 6 attention blocks + its classification head | the released convolution and 6 attention blocks (`../../../cbramod/experiments/finetune-sleepedf/code/conformer_released.py`, copied); its head replaced by CBraMod's sequence head on the flattened tokens (394 × 40) | only the encoder differs between models |
| Kernel sizes | temporal conv 25 samples, pooling 75 / stride 15 (at 250 Hz) | the same sample counts at 200 Hz (0.125 s conv, 0.375 s pooling) | the released layer sizes |
| Training | Adam (lr 2e-4, β1 0.5), batch 72, 2,000 epochs, segmentation-and-reconstruction augmentation | CBraMod's trainer: AdamW lr 5e-4, batch 64, cosine schedule, 50 epochs, no augmentation | one trainer for all models |
| Model selection | the released code reports the best test accuracy over epochs ([bciciv2a-epoch-selection](../bciciv2a-epoch-selection)) | the epoch with the best validation κ | the pipeline's rule |

### Settings we had to choose

| Setting | Value | Why |
|---|---|---|
| Embedding | 40, 6 blocks, 10 heads, dropout 0.5 as released | the released model |
| Spatial convolution | over the 2 channels | the released layer for any channel count |
| Micro-batches | 16 micro-batches of 4 sequences per batch of 64 (`add_jobs.py`) | 12 GB GPU; the patch embedding's batch normalisation sees 4 sequences (80 epochs) per micro-batch instead of 64, the one way the update differs from a full batch |
| Seeds | 0–4 | as every CBraMod-pipeline experiment |

### Run configuration

The shared run configuration ([cbramod/finetune-sleepedf](../../../cbramod/experiments/finetune-sleepedf#run-configuration)):
50 epochs, batch 64 sequences, AdamW (weight decay 5e-2), lr 5e-4 for the whole model (no `backbone`), cosine
schedule, label smoothing 0.1. 11,348,901 parameters (encoder 124,320; the head's first Linear holds 8.1M). About
13 h per run on one Quadro M6000 12 GB. Environment: conda env `bd-labram` (`env/pip_freeze.txt`).

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

Runs: 2026-10-29 on mcl-server (`~/repro-sleepedf/runs/SleepEDF__conformer__s*`). `conformer_released.py` copies
GPL-3.0 code from EEG Conformer's repository.
