# LaBraM-Base on Sleep-EDF, in CBraMod's sleep-staging pipeline

How does the released LaBraM-Base score on Sleep-EDF when it replaces CBraMod's backbone in CBraMod's released
sleep-staging pipeline? Only the encoder changes: the data, subject split, sequence head, trainer, checkpoint rule,
seeds and metric code are those of every model in the comparison. The pipeline, data and split are described in
[cbramod/finetune-sleepedf](../../../cbramod/experiments/finetune-sleepedf), which holds the shared code. This
experiment holds the 5 LaBraM runs (seeds 0–4).

LaBraM's paper evaluates no sleep data (TUAB, TUEV, SEED-V, MoBI), so this compares models; it reproduces no
published number.

## Results

Test set of the 8 test subjects (21,080 epochs), scored at the epoch with the best validation κ; mean ± SD over
seeds 0–4 (`results/results.txt`).

| Balanced acc. | Cohen's κ | Weighted F1 | Accuracy | Macro-F1 |
|---|---|---|---|---|
| 0.7613 ± 0.0185 | 0.7730 ± 0.0086 | 0.8347 ± 0.0090 | 0.8384 ± 0.0057 | 0.7619 ± 0.0154 |

| Per-stage F1 | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|
| LaBraM-Base | 0.9413 | 0.4346 | 0.8437 | 0.8049 | 0.7849 |

On the common test set (20,976 epochs every Sleep-EDF model scores; `python tools/compare_sleepedf.py`):

- **LaBraM-Base has the highest κ of the original implementations** (0.7727 ± 0.0085), but it is not significantly
  ahead of CBraMod (+0.0051, p = 0.44) or L-SeqSleepNet with its own released training (+0.0002, p = 0.96).
  braindecode's LaBraM, the same encoder, scores 0.7773 (+0.0046, p = 0.36;
  [braindecode-sleepedf](../braindecode-sleepedf)).
- **It is 0.028 κ above CBraMod without pre-training** (p = 0.001). The pre-trained encoder is worth as much here as
  CBraMod's own pre-training (+0.023).
- Its N1 F1 (0.435) is the best of the pipeline models after braindecode's LaBraM (0.442).
- Validation selection picks epochs 6–17 of 50. The best test epoch would add 0.0131 κ; the last epoch scores 0.7715.

## Paper vs. this reproduction

### What differs from LaBraM's paper and released code, and why

| | LaBraM paper / released fine-tuning (`run_class_finetuning.py`) | This experiment | Why |
|---|---|---|---|
| Task and data | single-segment classification (TUAB, TUEV, …), up to 16 one-second patches | 30-s epochs of Sleep-EDF Fpz-Cz and Pz-Oz at 200 Hz, classified in sequences of 20 | the comparison task |
| Model on top | LaBraM's linear head on mean-pooled tokens | CBraMod's released sequence head (Linear → 512, one Transformer layer across 20 epochs, linear classifier) on LaBraM's pooled features | only the encoder differs between models |
| 30-s epochs | n/a (≤ 16 s) | two 15-s halves, each through LaBraM (time positions 0–14); pooled features (2 × 200) concatenated | the pre-trained time embedding has 16 positions; splitting keeps the weights unchanged |
| Channels | 10-20 electrode names | Fpz-Cz → FPZ, Pz-Oz → PZ (the active electrode) | its electrode embedding has no bipolar derivations |
| Optimizer | AdamW, lr 5e-4 with layer-wise decay 0.65, 5 warm-up epochs, 50 epochs, batch 64 (README recipe) | CBraMod's trainer: AdamW, backbone 1e-4, head 5e-4, cosine schedule, 50 epochs | one trainer for all models |
| Units | 0.1 mV (data / 100) | the same (`sleepedf_dataset.py`: µV / 100) | as LaBraM and CBraMod expect |

### Settings we had to choose

| Setting | Value | Why |
|---|---|---|
| Weights | `labram-base.pth` from `methods/labram/upstream/checkpoints`, the `student.` tensors | the released LaBraM-Base, loaded as `run_class_finetuning.py` does |
| Model flags | the README recipe's (`drop_path 0.1`, `init_values 0.1`, absolute position embedding, mean pooling, no relative position bias) | as LaBraM's fine-tuning recipe |
| Features | mean-pooled tokens after `fc_norm` | LaBraM's own classifier input |
| Seeds | 0–4 | as every CBraMod-pipeline experiment |

### Run configuration

The shared run configuration ([cbramod/finetune-sleepedf](../../../cbramod/experiments/finetune-sleepedf#run-configuration)):
50 epochs, batch 64 sequences (2 micro-batches of 32), AdamW (weight decay 5e-2), lr 1e-4 for the LaBraM encoder
and 5e-4 for the head, cosine schedule, label smoothing 0.1. The model has 9,180,197 parameters (encoder
5,819,936). About 4.6 h per run on one Quadro M6000 12 GB. Environment: conda env `bd-labram` (`env/pip_freeze.txt`).

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

Runs: 2026-10-29 / 30 on mcl-server (`~/repro-sleepedf/runs/SleepEDF__labram__s*`). LaBraM's code: the pinned
submodule `methods/labram/upstream`.
