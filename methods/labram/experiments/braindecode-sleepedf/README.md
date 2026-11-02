# LaBraM-Base (braindecode) on Sleep-EDF, in CBraMod's sleep-staging pipeline

Does braindecode's LaBraM-Base give the released LaBraM-Base's Sleep-EDF result when both replace CBraMod's backbone
in CBraMod's released sleep-staging pipeline? Only the encoder changes: the data, subject split, sequence head,
trainer, checkpoint rule, seeds and metric code are those of the released-code runs in
[sleepedf-cbramod-pipeline](../sleepedf-cbramod-pipeline). The pipeline, data and split are described in
[cbramod/finetune-sleepedf](../../../cbramod/experiments/finetune-sleepedf), which holds the shared code. This
experiment holds the 5 runs (seeds 0–4) of braindecode 1.8.1's `Labram`, fine-tuned from braindecode's copy of the
released LaBraM-Base weights.

**The two encoders compute the same function.** `code/check_equivalence.py` (output: `results/equivalence.txt`)
builds both as the runs do, each from its own weight file, and copies the released encoder's freshly initialised
`fc_norm` (the one layer neither file holds) into braindecode's. Both map Fpz-Cz and Pz-Oz (as FPZ and PZ) to the
same rows of the position embedding, have 5,819,936 parameters and give identical features for 30-s epochs (max
|difference| 0 in float64). The weights themselves are identical tensors
([braindecode-cbramod-splits](../braindecode-cbramod-splits#is-braindecodes-labram-the-released-model)). So the two
sets of runs differ only by random draws: the initial weights of the head and `fc_norm`, data order and drop path.

## Results

Test set of the 8 test subjects (21,080 epochs), scored at the epoch with the best validation κ; mean ± SD over
seeds 0–4 (`results/results.txt`).

| | Balanced acc. | Cohen's κ | Weighted F1 | Accuracy | Macro-F1 |
|---|---|---|---|---|---|
| LaBraM-Base (braindecode) | 0.7670 ± 0.0106 | 0.7775 ± 0.0057 | 0.8383 ± 0.0036 | 0.8414 ± 0.0042 | 0.7667 ± 0.0062 |
| LaBraM-Base (released code, [sleepedf-cbramod-pipeline](../sleepedf-cbramod-pipeline)) | 0.7613 ± 0.0185 | 0.7730 ± 0.0086 | 0.8347 ± 0.0090 | 0.8384 ± 0.0057 | 0.7619 ± 0.0154 |

| Per-stage F1 | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|
| LaBraM-Base (braindecode) | 0.9454 | 0.4418 | 0.8427 | 0.8040 | 0.7995 |
| LaBraM-Base (released code) | 0.9413 | 0.4346 | 0.8437 | 0.8049 | 0.7849 |

On the common test set (20,976 epochs every Sleep-EDF model scores; `python tools/compare_sleepedf.py`):

- **braindecode's LaBraM reproduces the released-code runs.** braindecode − released is +0.0046 κ (p = 0.36),
  +0.0059 balanced accuracy (p = 0.56) and +0.0049 macro-F1 (p = 0.53). No per-stage F1 differs significantly
  (largest: REM, +0.015, p = 0.23). The seed ranges overlap: κ 0.768–0.783 against 0.761–0.781.
- **It has the highest κ (0.7773) and balanced accuracy (0.7670) of all Sleep-EDF models**, but neither lead is
  significant: +0.0046 κ over the released LaBraM, +0.0048 over L-SeqSleepNet (p = 0.23) and +0.0097 over CBraMod
  (p = 0.13); +0.0005 balanced accuracy over braindecode's DeepSleepNet (p = 0.93).
- Validation selection picks epochs 11–21 of 50 (released code: 6–17); the best test epoch would add 0.0081 κ, the
  last epoch scores 0.7721.

## Paper vs. this reproduction

LaBraM's paper evaluates no sleep data, so the reference here is the released code on the same pipeline.

### What differs from the released-code runs, and why

| | Released code ([sleepedf-cbramod-pipeline](../sleepedf-cbramod-pipeline)) | This experiment | Why |
|---|---|---|---|
| Model code | `modeling_finetune.py` (`labram_base_patch200_200`), built as `run_class_finetuning.py` builds it | braindecode 1.8.1 `Labram`, the same flags (mean pooling, drop path 0.1, layer scale 0.1, absolute position embedding, no qkv bias, no relative position bias) | the question of this experiment |
| Pre-trained weights | `labram-base.pth`, the `student.` tensors | `braindecode/labram-pretrained` @ `0563b6c`: the same 221 tensors under braindecode's names | braindecode's own distribution |
| Time embedding | 16 rows; 15-s halves use rows 0–14 | braindecode sizes it by the window (15 s); the runs load the file and keep its first rows, because `from_pretrained()` raises a size mismatch | braindecode's implementation; the rows used are the same |
| Channel indices | LaBraM's `utils.get_input_chans()` | braindecode's `_LABRAM_CANONICAL_INDEX` | the same rows (`results/equivalence.txt`) |
| Sequence head, trainer, data, seeds | CBraMod's released sequence head and trainer | the same, shared code | only the encoder may differ |

### Settings we had to choose

| Setting | Value | Why |
|---|---|---|
| 30-s epochs | two 15-s halves through the encoder, pooled features (2 × 200) concatenated | as the released-code runs: the pre-trained time embedding covers 16 s |
| Features | mean-pooled tokens after `fc_norm` (`forward_features`) | LaBraM's own classifier input |
| Micro-batches | 2 of 32 sequences per batch of 64 | as the released-code runs; LaBraM has no batch normalisation, so the update is the full-batch one |
| Seeds | 0–4 | as every CBraMod-pipeline experiment |

### Run configuration

The shared run configuration ([cbramod/finetune-sleepedf](../../../cbramod/experiments/finetune-sleepedf#run-configuration)):
50 epochs, batch 64 sequences, AdamW (weight decay 5e-2), lr 1e-4 for the encoder (`backbone`) and 5e-4 for the rest
(`multi_lr`), cosine schedule, label smoothing 0.1. 9,180,197 parameters (encoder 5,819,936), as the released-code
runs. About 4.6 h per run on one Quadro M6000 12 GB. Environment: conda env `bd-labram`, braindecode 1.8.1
(`env/pip_freeze.txt`).

## Layout

```
code/reproduce.sh         collect (recompute results/ from logs/runs) or train (enqueue the 5 runs), with the
                          shared code in ../../../cbramod/experiments/finetune-sleepedf/code
code/check_equivalence.py the two encoders on Sleep-EDF input (-> results/equivalence.txt)
logs/runs/<run>/          log.txt (progress bars removed), metrics.json (per-epoch validation and test metrics),
                          test_preds.npz (test labels and predictions at the selected epoch)
results/                  results.txt, results.json (with common_* metrics on the common test set), summary.csv,
                          equivalence.txt
env/pip_freeze.txt        packages of the runs
```

## Reproducing

```bash
code/reproduce.sh collect        # numpy, scipy, scikit-learn; diffs against results/results.txt
code/reproduce.sh train          # needs the preprocessed data and braindecode's weights (shared README, "Reproducing")
python code/check_equivalence.py # torch, timm 0.4.12, braindecode 1.8.1; LaBraM's released code and labram-base.pth
```

## Provenance

Runs: 2026-11-01 on mcl-server (`~/repro-sleepedf/runs/SleepEDF__bd-labram__s*`). braindecode: the pinned submodule
`methods/eegnet/upstream` (1.8.1).
