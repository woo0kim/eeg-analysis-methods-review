# CBraMod (braindecode) on Sleep-EDF, in CBraMod's sleep-staging pipeline

Does braindecode's CBraMod give the released CBraMod's Sleep-EDF result when it runs in the released sleep-staging
pipeline? Only the encoder changes: the data, subject split, sequence head, trainer, checkpoint rule, seeds and metric
code are those of the released-code runs in [finetune-sleepedf](../finetune-sleepedf), which also holds the shared
code. This experiment holds the 5 runs (seeds 0–4) of braindecode 1.8.1's `CBraMod`, fine-tuned from braindecode's
copy of the released weights.

On CBraMod's public datasets braindecode's model comes with its own one-layer task head, and that head cost 0.02
balanced accuracy on PhysioNet-MI ([braindecode-public-datasets](../braindecode-public-datasets)). Here the head is
the released sequence head for both implementations, so the two runs differ in one layer of the encoder.
`code/check_equivalence.py` (output: `results/equivalence.txt`) builds both encoders as the runs do. braindecode's
encoder ends with CBraMod's pre-trained `proj_out` (200 → 200; the released weights, identical), which the released
sleep model removes. Passed through that `proj_out`, the released encoder's features equal braindecode's (max
|difference| 0 in float64, 2-channel 30-s epochs); without it they differ.

## Results

Test set of the 8 test subjects (21,080 epochs), scored at the epoch with the best validation κ; mean ± SD over
seeds 0–4 (`results/results.txt`).

| | Balanced acc. | Cohen's κ | Weighted F1 | Accuracy | Macro-F1 |
|---|---|---|---|---|---|
| CBraMod (braindecode) | 0.7394 ± 0.0182 | 0.7680 ± 0.0125 | 0.8302 ± 0.0081 | 0.8368 ± 0.0087 | 0.7529 ± 0.0116 |
| CBraMod (released code, [finetune-sleepedf](../finetune-sleepedf)) | 0.7536 ± 0.0088 | 0.7676 ± 0.0109 | 0.8300 ± 0.0080 | 0.8350 ± 0.0081 | 0.7533 ± 0.0097 |

| Per-stage F1 | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|
| CBraMod (braindecode) | 0.9457 | 0.4150 | 0.8368 | 0.7939 | 0.7733 |
| CBraMod (released code) | 0.9461 | 0.4155 | 0.8345 | 0.7949 | 0.7753 |

On the common test set (20,976 epochs every Sleep-EDF model scores; `python tools/compare_sleepedf.py`):

- **braindecode's CBraMod reproduces the released-code runs.** braindecode − released is +0.0001 κ (p = 0.99),
  −0.0006 macro-F1 (p = 0.93) and −0.0144 balanced accuracy (p = 0.17). No per-stage F1 differs by more than 0.003
  (all p ≥ 0.71).
- The balanced-accuracy gap is seed spread, not a shift: braindecode's seeds range 0.715–0.758, the released code's
  0.741–0.763. κ ranges 0.752–0.782 against 0.749–0.777.
- Validation selection picks epochs 10–36 of 50; the best test epoch would add 0.0055 κ, the last epoch scores
  0.7567.

## Paper vs. this reproduction

CBraMod's paper reports no Sleep-EDF result (its sleep benchmark is ISRUC), so the reference here is the released
code on the same pipeline.

### What differs from the released-code runs, and why

| | Released code ([finetune-sleepedf](../finetune-sleepedf)) | This experiment | Why |
|---|---|---|---|
| Model code | `models/cbramod.py`, as `model_for_isruc.py` builds it | braindecode 1.8.1 `CBraMod` | the question of this experiment |
| Pre-trained weights | `pretrained_weights.pth` | `braindecode/cbramod-pretrained` @ `584cdc4`: the same 211 tensors, identical ([braindecode-public-datasets](../braindecode-public-datasets#is-braindecodes-cbramod-the-released-model)) | braindecode's own distribution |
| Encoder output | `proj_out` replaced by `Identity`, as `model_for_isruc.py` does | `return_features` output, which comes after the pre-trained `proj_out` (200 → 200, 40,200 parameters, trained at the backbone learning rate) | braindecode's `return_features` keeps `proj_out`; up to it the encoders give identical outputs (`results/equivalence.txt`) |
| Sequence head, trainer, data, seeds | `model_for_isruc.py` head, `finetune_trainer.py` (ISRUC branch) | the same, shared code | only the encoder may differ |

### Settings we had to choose

| Setting | Value | Why |
|---|---|---|
| Features | the `return_features` output, flattened: 2 channels × 30 patches × 200 | the released model's classifier input, plus `proj_out` |
| Micro-batches | 2 of 32 sequences per batch of 64 | as the released-code runs; CBraMod has no batch normalisation, so the update is the full-batch one |
| Seeds | 0–4 | as every CBraMod-pipeline experiment |

### Run configuration

The shared run configuration ([finetune-sleepedf](../finetune-sleepedf#run-configuration)): 50 epochs, batch 64
sequences, AdamW (weight decay 5e-2), lr 1e-4 for the encoder (`backbone`) and 5e-4 for the rest (`multi_lr`),
cosine schedule, label smoothing 0.1. 14,223,461 parameters (encoder 4,924,000; the released runs' 14,183,261 and
4,883,800 without `proj_out`). About 8.6 h per run on one Quadro M6000 12 GB. Environment: conda env `bd-labram`,
braindecode 1.8.1 (`env/pip_freeze.txt`).

## Layout

```
code/reproduce.sh        collect (recompute results/ from logs/runs) or train (enqueue the 5 runs), with the
                         shared code in ../finetune-sleepedf/code
code/check_equivalence.py the two encoders on Sleep-EDF input (-> results/equivalence.txt)
logs/runs/<run>/         log.txt (progress bars removed), metrics.json (per-epoch validation and test metrics),
                         test_preds.npz (test labels and predictions at the selected epoch)
results/                 results.txt, results.json (with common_* metrics on the common test set), summary.csv,
                         equivalence.txt
env/pip_freeze.txt       packages of the runs
```

## Reproducing

```bash
code/reproduce.sh collect        # numpy, scipy, scikit-learn; diffs against results/results.txt
code/reproduce.sh train          # needs the preprocessed data and braindecode's weights (shared README, "Reproducing")
python code/check_equivalence.py # torch, braindecode 1.8.1; CBraMod's released weights (CBRAMOD_WEIGHTS)
```

## Provenance

Runs: 2026-10-31 / 11-01 on mcl-server (`~/repro-sleepedf/runs/SleepEDF__bd-cbramod__s*`). braindecode: the pinned
submodule `methods/eegnet/upstream` (1.8.1).
