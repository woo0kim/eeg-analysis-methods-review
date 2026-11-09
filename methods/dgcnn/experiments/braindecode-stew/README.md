# braindecode's DGCNN, raw input, braindecode's DGCNN, DE input on STEW

braindecode's DGCNN, raw input, braindecode's DGCNN, DE input on STEW (48 subjects, Emotiv, 14 channels at 128 Hz, rest vs. the SIMKAP multitasking test), in the pipeline every model shares there: 4-s windows with 25% overlap, 7-fold GroupKFold across subjects, CBraMod's training loop with only the model swapped. Data, windows, folds and run configuration are in [cbramod/finetune-stew](../../../cbramod/experiments/finetune-stew), which holds the shared code. Implementation: braindecode's DGCNN, raw input: braindecode 1.8.1's `DGCNN` as it ships (raw samples as node features); braindecode's DGCNN, DE input: braindecode 1.8.1's `DGCNN` on the same DE input as ours. Seeds 0–4 × 7 folds per task.

No paper among these methods reports STEW, and the dataset paper's 69% / κ 0.46 (3 levels) comes from its own feature pipeline: the comparison is between models. Each seed's score is computed on the pooled predictions of its 7 test folds; mean ± SD over seeds (`results/results.txt`).

## Results

| Task | Model | Seeds × folds | Accuracy | Bal. acc | κ | Weighted F1 | Fold SD (accuracy) |
|---|---|---|---|---|---|---|---|
| rest vs. task | braindecode's DGCNN, raw input | 5 × 7 | 0.6691 ± 0.0039 | 0.6691 ± 0.0039 | 0.3383 ± 0.0077 | 0.6691 ± 0.0038 | 0.0496 |
| rest vs. task | braindecode's DGCNN, DE input | 5 × 7 | 0.7833 ± 0.0091 | 0.7833 ± 0.0091 | 0.5666 ± 0.0182 | 0.7821 ± 0.0098 | 0.0549 |
| 3 rating levels | braindecode's DGCNN, raw input | 5 × 7 | 0.4876 ± 0.0074 | 0.4361 ± 0.0091 | 0.1814 ± 0.0120 | 0.4718 ± 0.0075 | 0.0546 |
| 3 rating levels | braindecode's DGCNN, DE input | 5 × 7 | 0.5532 ± 0.0110 | 0.5035 ± 0.0113 | 0.2966 ± 0.0146 | 0.5449 ± 0.0075 | 0.0696 |

Against the other models (`python tools/compare_stew.py`; Welch's t-test over seeds):

- **On DE input, level with ours** (+0.0086, p = 0.14); **on raw samples, 0.1055 lower** (p < 0.001): the same input effect as on SEED and Sleep-EDF.

## Paper vs. this reproduction

Every difference from the released pipeline, and every setting chosen for all models, is listed once in [the shared README](../../../cbramod/experiments/finetune-stew).
Model-specific settings are in `../../../cbramod/experiments/finetune-stew/code/stew_models.py` and summarised above.
Hardware: RTX A6000 (mcl-server2, shared with other users).

## Reproducing

`code/reproduce.sh collect` recomputes `results/` from `logs/runs` (numpy, scikit-learn); `code/reproduce.sh train` enqueues the runs.
`logs/runs/<run>/` holds `log.txt` (progress output removed), `metrics.json` (validation and test scores of every epoch) and
`test_preds.npz` (test labels and predictions at the selected epoch).

## Provenance

Runs: 2026-11-05 to 11-08, `~/repro-seed/stew/runs` on mcl-server and mcl-server2.
