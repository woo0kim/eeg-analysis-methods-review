# EEGNet on STEW

EEGNet on STEW (48 subjects, Emotiv, 14 channels at 128 Hz, rest vs. the SIMKAP multitasking test), in the pipeline every model shares there: 4-s windows with 25% overlap, 7-fold GroupKFold across subjects, CBraMod's training loop with only the model swapped. Data, windows, folds and run configuration are in [cbramod/finetune-stew](../../../cbramod/experiments/finetune-stew), which holds the shared code. Implementation: EEGNet: our PyTorch EEGNet-8,2 (of [pytorch-cbramod-splits](../pytorch-cbramod-splits)). Seeds 0–4 × 7 folds per task.

No paper among these methods reports STEW, and the dataset paper's 69% / κ 0.46 (3 levels) comes from its own feature pipeline: the comparison is between models. Each seed's score is computed on the pooled predictions of its 7 test folds; mean ± SD over seeds (`results/results.txt`).

## Results

| Task | Model | Seeds × folds | Accuracy | Bal. acc | κ | Weighted F1 | Fold SD (accuracy) |
|---|---|---|---|---|---|---|---|
| rest vs. task | EEGNet | 5 × 7 | 0.6648 ± 0.0061 | 0.6648 ± 0.0061 | 0.3297 ± 0.0122 | 0.6561 ± 0.0054 | 0.0622 |
| 3 rating levels | EEGNet | 5 × 7 | 0.4975 ± 0.0200 | 0.4089 ± 0.0260 | 0.1387 ± 0.0459 | 0.4378 ± 0.0302 | 0.0539 |

Against the other models (`python tools/compare_stew.py`; Welch's t-test over seeds):

- **The weakest model on both tasks:** 0.6648 on rest vs. task, κ 0.139 on 3 levels; DGCNN beats it by 0.110 (p < 0.001).

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
