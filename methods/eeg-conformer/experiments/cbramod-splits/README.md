# EEG Conformer on CBraMod's four public datasets

EEG Conformer through CBraMod's released fine-tuning pipeline on BCI-IV-2a, PhysioNet-MI, Mumtaz2016 and MentalArithmetic: the
same LMDBs and subject splits, `Trainer`, loss, optimizer, schedule, checkpoint rule (best validation κ or AUROC) and `Evaluator` as
every other model run on these splits (EEGNet, CBraMod, LaBraM-Base, DGCNN, dDTF-GL). Only the model differs. Two implementations,
seeds 0–4 each:

| Setting | Model |
|---|---|
| `conformer` | the released model of `conformer.py` (the BCI-IV-2a script): patch embedding, 6 encoder blocks with 10 heads, its 3-layer head, whose first layer is sized to the tokens of the input (47 × 40 for 800 samples; 61 × 40 = 2,440 for 1,000, the released size). Blocks from `cbramod/experiments/finetune-sleepedf/code/conformer_released.py`, which matches the released file except for the channel count |
| `bd-conformer` | braindecode 1.8.1 `EEGConformer(n_chans, n_outputs, n_times, sfreq=200)` |

The CBraMod paper reports EEG Conformer on all four datasets (Tables 3, 10, 12, 15; its own runs with BIOT's code, settings not
given), so these runs also check those rows. The code in `code/` is shared with
[atcnet/braindecode-cbramod-splits](../../../atcnet/experiments/braindecode-cbramod-splits) and
[deepsleepnet/braindecode-cbramod-splits](../../../deepsleepnet/experiments/braindecode-cbramod-splits).

## Results

Mean ± SD over seeds 0–4 (`results/results.txt`); Welch's t-test against the CBraMod paper's mean ± SD over its 5 seeds. Rank: by
mean balanced accuracy among the 8 models run on these splits (`python tools/compare_methods.py`, last table).

| Dataset | Metric | Paper (CBraMod's Table) | Released model | p | braindecode | p | Rank of 8 |
|---|---|---|---|---|---|---|---|
| BCI-IV-2a | balanced acc. | 0.4696 ± 0.0106 (T15) | **0.5042 ± 0.0142** | 0.003 | 0.5080 ± 0.0272 | 0.031 | 3 |
| | κ | 0.2924 ± 0.0141 | 0.3389 ± 0.0190 | 0.003 | 0.3440 ± 0.0362 | 0.030 | |
| PhysioNet-MI | balanced acc. | 0.6049 ± 0.0104 (T3) | 0.5787 ± 0.0148 | 0.014 | 0.5861 ± 0.0074 | 0.013 | 6 |
| | κ | 0.4736 ± 0.0171 | 0.4381 ± 0.0197 | 0.017 | 0.4482 ± 0.0099 | 0.026 | |
| Mumtaz2016 | balanced acc. | 0.9308 ± 0.0117 (T10) | 0.8747 ± 0.0302 | 0.011 | 0.8904 ± 0.0253 | 0.019 | 7 |
| | AUROC | 0.9702 ± 0.0101 | 0.9629 ± 0.0031 | 0.19 | 0.9637 ± 0.0131 | 0.41 | |
| MentalArithmetic | balanced acc. | 0.6805 ± 0.0123 (T12) | 0.5438 ± 0.0501 | 0.003 | 0.5500 ± 0.0727 | 0.015 | 8 |
| | AUROC | 0.7424 ± 0.0128 | 0.6486 ± 0.0718 | 0.042 | 0.6592 ± 0.0520 | 0.021 | |

- **The CBraMod paper's EEG Conformer rows do not reproduce, in either direction.** Above the paper on BCI-IV-2a (+0.035, p = 0.003),
  below it on the other three (−0.026 to −0.137 balanced accuracy, all p ≤ 0.014). The same pattern as the paper's EEGNet rows
  ([eegnet/pytorch-cbramod-splits](../../../eegnet/experiments/pytorch-cbramod-splits)) and CBraMod's own numbers: the paper's baselines
  ran with BIOT's code under settings it does not give.
- **Against the other models:** on BCI-IV-2a it is level with EEGNet, DeepSleepNet and ATCNet (p ≥ 0.12) and above both foundation
  models; on the other three datasets it is in the bottom three.
- **braindecode's `EEGConformer` is within noise of the released model** on every metric (balanced accuracy: +0.004 to +0.016, all p ≥ 0.35; `tools/compare_methods.py`, second table).

All metrics per seed are in `results/results.txt`.

## Paper vs. this reproduction

There is no EEG Conformer paper number to reproduce here (its paper has no cross-subject results on these datasets); the reference is
the CBraMod paper's EEG Conformer baseline.

### What differs, and why

| | CBraMod paper's EEG Conformer | EEG Conformer's own code | This experiment | Why |
|---|---|---|---|---|
| Code | BIOT's re-implementation | `conformer.py` | the released model in CBraMod's pipeline | released model, the pipeline every model shares |
| Sampling rate | 200 Hz (CBraMod's LMDBs) | 250 Hz | 200 Hz, the released kernel sizes (25-sample convolution, 75 / 15 pooling) unchanged | the shared samples; the kernels are not rescaled, as on Sleep-EDF, SEED and STEW |
| Head size | – | Linear(2440, 256) for 4 s at 250 Hz | sized to the tokens (47 or 61) | the input length differs |
| Training | not given | Adam 2e-4, 2000 epochs, S&R augmentation, best test epoch | CBraMod's trainer: AdamW 5e-4, 50 epochs, cosine, label smoothing 0.1, best validation κ / AUROC | one trainer for all models; no test-set selection |

### Settings we had to choose

| Setting | Value | Why |
|---|---|---|
| Seeds | 0–4 | as every model on these splits |
| Learning rate | 5e-4 for the whole model | CBraMod's `multi_lr` gives non-backbone parameters 5 × 1e-4; the Conformer has no `backbone`, as EEGNet and DGCNN |

### Run configuration

`run_models.py` with `finetune_main.py`'s defaults: 50 epochs, batch 64, AdamW, weight decay 5e-2, cosine to 1e-6, clipping 1, label
smoothing 0.1. PyTorch 2.1.2, braindecode 1.8.1 (env `bd-labram` on mcl-server, as the SEED runs), Quadro M6000. `code/smoke_test.py`
builds every model of `run_models.py` for every dataset (parameters and peak GPU memory for a batch of 64).

## Reproducing

```bash
code/reproduce.sh collect    # results/ from logs/runs into results_rerun/ (numpy, scipy), then diff
code/reproduce.sh train      # enqueue the runs for tools/run_queue.py; needs CBraMod's LMDBs (see code/add_jobs.py)
```

## Provenance

Runs: 2026-11-20 on mcl-server (`~/repro-gaps/runs/cbramod-splits`), 8 × Quadro M6000 shared with the SEED runs.
