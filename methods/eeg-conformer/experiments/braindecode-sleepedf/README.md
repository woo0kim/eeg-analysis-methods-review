# EEG Conformer (braindecode) on Sleep-EDF, in CBraMod's sleep-staging pipeline

Does braindecode's EEG Conformer score like the released EEG Conformer on Sleep-EDF when both replace CBraMod's
backbone in CBraMod's released sleep-staging pipeline? Only the encoder changes: the data, subject split, sequence
head, trainer, checkpoint rule, seeds and metric code are those of the released-model runs in
[sleepedf-cbramod-pipeline](../sleepedf-cbramod-pipeline). The pipeline, data and split are described in
[cbramod/finetune-sleepedf](../../../cbramod/experiments/finetune-sleepedf), which holds the shared code. This
experiment holds the 5 runs (seeds 0–4) of braindecode 1.8.1's `EEGConformer`.

braindecode's `EEGConformer` has the released model's layers, and with the released weights copied in it differs in
one place, the attention temperature ([braindecode-bciciv2a](../braindecode-bciciv2a#is-braindecodes-eegconformer-the-released-model)).
On BCI IV 2a that made no measurable difference.

## Results

Test set of the 8 test subjects (21,080 epochs), scored at the epoch with the best validation κ; mean ± SD over
seeds 0–4 (`results/results.txt`).

| | Balanced acc. | Cohen's κ | Weighted F1 | Accuracy | Macro-F1 |
|---|---|---|---|---|---|
| EEG Conformer (braindecode) | 0.6802 ± 0.0304 | 0.7016 ± 0.0344 | 0.7843 ± 0.0246 | 0.7879 ± 0.0256 | 0.6918 ± 0.0326 |
| EEG Conformer (released model, [sleepedf-cbramod-pipeline](../sleepedf-cbramod-pipeline)) | 0.6449 ± 0.0266 | 0.6880 ± 0.0188 | 0.7711 ± 0.0127 | 0.7804 ± 0.0131 | 0.6583 ± 0.0234 |

| Per-stage F1 | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|
| EEG Conformer (braindecode) | 0.8993 | 0.3675 | 0.8272 | 0.7063 | 0.6584 |
| EEG Conformer (released model) | 0.9082 | 0.3469 | 0.8127 | 0.5611 | 0.6626 |

On the common test set (20,976 epochs every Sleep-EDF model scores; `python tools/compare_sleepedf.py`):

- **The two implementations do not differ significantly overall.** braindecode − released is +0.0135 κ (p = 0.47),
  +0.0350 balanced accuracy (p = 0.089) and +0.0332 macro-F1 (p = 0.11).
- **Deep sleep is where they differ.** braindecode's N3 F1 is 0.706 against 0.562 (+0.145, p = 0.047). The
  released model's N3 F1 swings from 0.44 to 0.71 across seeds, braindecode's from 0.66 to 0.75. No other stage
  differs (p ≥ 0.095).
- **Either way EEG Conformer stays near the bottom.** CBraMod leads braindecode's Conformer by 0.066 κ (p = 0.010)
  and 0.074 balanced accuracy (p = 0.004). Both Conformers have the largest seed spreads in balanced accuracy (SD
  0.030 here, 0.027 released) and the largest gains from test-set epoch selection (+0.037 κ here, +0.034
  released).
- Validation selection picks epochs 10–21 of 50; the last epoch scores 0.7133 κ.

The runs do not separate the two causes of the N3 difference below (attention temperature and micro-batch size).

## Paper vs. this reproduction

EEG Conformer's paper evaluates motor imagery and emotion data, not sleep, so the reference here is the released model
on the same pipeline.

### What differs from the released-model runs, and why

| | Released model ([sleepedf-cbramod-pipeline](../sleepedf-cbramod-pipeline)) | This experiment | Why |
|---|---|---|---|
| Model code | `conformer.py`'s convolution and 6 attention blocks (`conformer_released.py`) | braindecode 1.8.1 `EEGConformer`, `return_features=True` | the question of this experiment |
| Attention temperature | query–key products divided by √(embedding size) = √40 | divided by √(head dimension) = √4 (PyTorch's `scaled_dot_product_attention`): a softmax √10 ≈ 3.2× sharper, in all 6 layers | braindecode's implementation ([braindecode-bciciv2a](../braindecode-bciciv2a#is-braindecodes-eegconformer-the-released-model)) |
| Micro-batches | 16 of 4 sequences: the patch embedding's batch normalisation sees 80 epochs at a time | 2 of 32 sequences: 640 epochs at a time | chosen per implementation with `smoke_test.py` to fit in 12 GB |
| Random draws | – | braindecode also builds its own classifier layers (replaced by the head) and draws attention dropout inside `scaled_dot_product_attention`, so a seed gives different initial weights and dropout masks | the runs are not seed-for-seed comparable |
| Sequence head, trainer, data, seeds | CBraMod's released sequence head and trainer | the same, shared code | only the encoder may differ |

### Settings we had to choose

| Setting | Value | Why |
|---|---|---|
| Layer sizes | braindecode's defaults: 40 filters, temporal kernel 25, pooling 75 / stride 15, 6 blocks, 10 heads, dropout 0.5 (the released values), `sfreq = 200`, `n_times = 6000` | the released configuration |
| Features | the `return_features` output, flattened: 394 tokens × 40 | the released model's classifier input |
| Seeds | 0–4 | as every CBraMod-pipeline experiment |

### Run configuration

The shared run configuration ([cbramod/finetune-sleepedf](../../../cbramod/experiments/finetune-sleepedf#run-configuration)):
50 epochs, batch 64 sequences, AdamW (weight decay 5e-2), lr 5e-4 for the whole model (no `backbone`), cosine
schedule, label smoothing 0.1. 11,348,901 parameters (encoder 124,320), as the released-model runs. About 13 h per
run on one Quadro M6000 12 GB. Environment: conda env `bd-labram`, braindecode 1.8.1 (`env/pip_freeze.txt`).

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

Runs: 2026-10-30 to 11-01 on mcl-server (`~/repro-sleepedf/runs/SleepEDF__bd-conformer__s*`). braindecode: the pinned
submodule `methods/eegnet/upstream` (1.8.1).
