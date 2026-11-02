# DeepSleepNet (braindecode) on Sleep-EDF, in CBraMod's sleep-staging pipeline

How does DeepSleepNet score on Sleep-EDF when it replaces CBraMod's backbone in CBraMod's released sleep-staging
pipeline, like every other model in the comparison? Only the encoder changes: the data, subject split, sequence
head, trainer, checkpoint rule, seeds and metric code are shared. The pipeline, data and split are described in
[cbramod/finetune-sleepedf](../../../cbramod/experiments/finetune-sleepedf), which holds the shared code. This
experiment holds the 5 runs (seeds 0–4) of braindecode 1.8.1's `DeepSleepNet`.

braindecode's `DeepSleepNet` is single-epoch: its BiLSTM runs over a sequence of length 1, where the released model
runs it across 25 epochs (its docstring says so). Here the cross-epoch context comes from CBraMod's sequence
Transformer instead. The released TensorFlow DeepSleepNet, with its own training, runs on the same split in
[sleepedf78-cbramod-split](../sleepedf78-cbramod-split). Its Sleep-EDF-20 result is reproduced in
[sleepedf20-20fold](../sleepedf20-20fold).

## Results

Test set of the 8 test subjects (21,080 epochs), scored at the epoch with the best validation κ; mean ± SD over
seeds 0–4 (`results/results.txt`).

| Balanced acc. | Cohen's κ | Weighted F1 | Accuracy | Macro-F1 |
|---|---|---|---|---|
| 0.7663 ± 0.0053 | 0.7623 ± 0.0213 | 0.8259 ± 0.0151 | 0.8291 ± 0.0170 | 0.7517 ± 0.0158 |

| Per-stage F1 | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|
| DeepSleepNet (braindecode) | 0.9367 | 0.4181 | 0.8351 | 0.8096 | 0.7591 |

On the common test set (20,976 epochs every Sleep-EDF model scores; `python tools/compare_sleepedf.py`):

- **It ties braindecode's LaBraM for the highest balanced accuracy of all models** (0.7666 ± 0.0053 against
  0.7670, p = 0.93), 0.013 above CBraMod (p = 0.027). In κ it is level with CBraMod (−0.005, p = 0.65): its seed
  spread in κ is the largest of the strong models (0.732–0.780).
- **It beats the released DeepSleepNet on the same split** by 0.060 κ (p = 0.002) and 0.042 balanced accuracy
  (p = 0.011). The released code keeps its final, overfitted model
  ([sleepedf78-cbramod-split](../sleepedf78-cbramod-split#results)); this pipeline keeps the epoch with the best
  validation κ, and its sequence model is a Transformer over 20 epochs rather than a BiLSTM over 25.
- Validation selection picks epochs 9–28 of 50. The best test epoch would add 0.020 κ.

## Paper vs. this reproduction

### What differs from DeepSleepNet's paper and released code, and why

| | DeepSleepNet paper / released code | This experiment | Why |
|---|---|---|---|
| Implementation | TensorFlow 1.x | braindecode 1.8.1 `DeepSleepNet` | the pipeline is PyTorch |
| Sequence model | two BiLSTM layers (512) across 25 epochs + residual shortcut | braindecode: the BiLSTM sees one epoch (a nonlinear transform with the residual); across epochs, CBraMod's Transformer over 20 epochs | braindecode's design; the pipeline's head is the same for every model |
| Input | one channel (Fpz-Cz) at 100 Hz, unfiltered | Fpz-Cz and Pz-Oz at 200 Hz, 0.3–35 Hz; each channel through the CNN paths separately, features concatenated (2 × 3,072) | the pipeline's data; braindecode's handling of several channels |
| First layers | small path kernel Fs/2, stride Fs/16; large path kernel 4·Fs, stride Fs/2 | the same rule at 200 Hz: 100 / 12 and 800 / 100 (padding 44 and 350), so every later feature map has the original size | braindecode's defaults are the 100 Hz values |
| LSTM cells | peephole connections | standard gates (`torch.nn.LSTM` has no peepholes) | braindecode's implementation |
| Training | CNN pre-training on oversampled data (100 epochs), then 200 epochs of sequence fine-tuning; final model | one stage, CBraMod's trainer: AdamW lr 5e-4, cosine schedule, 50 epochs, best validation κ | one trainer for all models |

### Settings we had to choose

| Setting | Value | Why |
|---|---|---|
| Features | the residual-sum output (1,024 per epoch), before braindecode's final layer | DeepSleepNet's own classifier input |
| Micro-batches | 2 of 32 sequences per batch of 64 | 12 GB GPU; batch normalisation in the CNN paths sees 32 sequences (640 epochs) at a time |
| Seeds | 0–4 | as every CBraMod-pipeline experiment |

### Run configuration

The shared run configuration ([cbramod/finetune-sleepedf](../../../cbramod/experiments/finetune-sleepedf#run-configuration)):
50 epochs, batch 64 sequences, AdamW (weight decay 5e-2), lr 5e-4 for the whole model (no `backbone`), cosine
schedule, label smoothing 0.1. 44,176,901 parameters (encoder 40,497,152, mostly the BiLSTM on 6,144 inputs).
About 50 min per run on one Quadro M6000 12 GB. Environment: conda env `bd-labram`, braindecode 1.8.1
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

Runs: 2026-10-30 on mcl-server (`~/repro-sleepedf/runs/SleepEDF__bd-deepsleepnet__s*`). braindecode: the pinned
submodule `methods/eegnet/upstream` (1.8.1).
