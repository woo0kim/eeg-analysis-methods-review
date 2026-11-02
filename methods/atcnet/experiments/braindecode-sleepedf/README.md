# ATCNet (braindecode) on Sleep-EDF, in CBraMod's sleep-staging pipeline

How does ATCNet score on Sleep-EDF when it replaces CBraMod's backbone in CBraMod's released sleep-staging pipeline?
Only the encoder changes: the data, subject split, sequence head, trainer, checkpoint rule, seeds and metric code are
those of every model in the comparison. The pipeline, data and split are described in
[cbramod/finetune-sleepedf](../../../cbramod/experiments/finetune-sleepedf), which holds the shared code. This
experiment holds the 5 ATCNet runs (seeds 0–4).

The released ATCNet is TensorFlow, so the encoder is braindecode 1.8.1's `ATCNet`. It has the released network's
layers and loads its weights ([braindecode-bciciv2a](../braindecode-bciciv2a)). The one consequential difference
found there, the max-norm axis of the final layer, is in the layer the sequence head replaces here. ATCNet's paper
evaluates motor imagery (BCI IV 2a), not sleep, so this compares models; it reproduces no published number.

## Results

Test set of the 8 test subjects (21,080 epochs), scored at the epoch with the best validation κ; mean ± SD over
seeds 0–4 (`results/results.txt`).

| Balanced acc. | Cohen's κ | Weighted F1 | Accuracy | Macro-F1 |
|---|---|---|---|---|
| 0.7431 ± 0.0078 | 0.7710 ± 0.0039 | 0.8298 ± 0.0032 | 0.8385 ± 0.0026 | 0.7495 ± 0.0053 |

| Per-stage F1 | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|
| ATCNet | 0.9458 | 0.3833 | 0.8416 | 0.7982 | 0.7786 |

On the common test set (20,976 epochs every Sleep-EDF model scores; `python tools/compare_sleepedf.py`):

- **ATCNet's κ (0.7709) is level with the pre-trained foundation models.** CBraMod − ATCNet is −0.0033 κ
  (p = 0.56) and +0.0107 balanced accuracy (p = 0.076). ATCNet has an encoder of 112k parameters and no
  pre-training; CBraMod's has 4.9M and is pre-trained.
- **It is the most stable model**: the smallest seed spread (κ SD 0.004) and the smallest gain from test-set epoch
  selection (+0.005 κ).
- **It trains slowly**: validation picks epochs 28–39 of 50, where the other models peak by epoch 28. At about
  24 h per run it is also the slowest model here.
- N1 is its weak stage (F1 0.383, against 0.42–0.43 for the foundation models).

## Paper vs. this reproduction

### What differs from ATCNet's paper and released code, and why

| | ATCNet paper / released code (Keras) | This experiment | Why |
|---|---|---|---|
| Implementation | TensorFlow / Keras | braindecode 1.8.1 `ATCNet` | the pipeline is PyTorch; braindecode's network is the released one ([braindecode-bciciv2a](../braindecode-bciciv2a)) |
| Data | BCI IV 2a: 22 channels at 250 Hz, 4.5-s trials | Sleep-EDF Fpz-Cz and Pz-Oz at 200 Hz, 30-s epochs in sequences of 20 | the comparison task |
| Model on top | per-window dense layers, averaged | CBraMod's sequence head on the five sliding windows' TCN outputs, concatenated (5 × 32) | only the encoder differs between models |
| Training | Adam lr 1e-3, batch 64, 1,000 epochs, early stopping on validation | CBraMod's trainer: AdamW lr 5e-4, batch 64, cosine schedule, 50 epochs, best validation κ | one trainer for all models |

### Settings we had to choose

| Setting | Value | Why |
|---|---|---|
| Layer sizes | braindecode's defaults (the paper's: F1 16, D 2, kernel 64, pooling 8 and 7, 5 windows, 2 attention heads, TCN depth 2), `sfreq = 200`, `n_times = 6000` | the released configuration; no tuning for any model |
| Features | the five windows' last TCN outputs, concatenated, before ATCNet's final layer | ATCNet's own classifier input |
| Seeds | 0–4 | as every CBraMod-pipeline experiment |

### Run configuration

The shared run configuration ([cbramod/finetune-sleepedf](../../../cbramod/experiments/finetune-sleepedf#run-configuration)):
50 epochs, batch 64 sequences (one micro-batch), AdamW (weight decay 5e-2), lr 5e-4 for the whole model (no
`backbone`), cosine schedule, label smoothing 0.1. 3,349,813 parameters (encoder 112,432). About 24 h per run
(≈ 30 min per epoch) on one Quadro M6000 12 GB. Environment: conda env `bd-labram`, braindecode 1.8.1
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

Runs: 2026-10-30 / 31 on mcl-server (`~/repro-sleepedf/runs/SleepEDF__bd-atcnet__s*`). braindecode: the pinned
submodule `methods/eegnet/upstream` (1.8.1).
