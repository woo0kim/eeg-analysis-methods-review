# EEG Conformer on STEW

EEG Conformer on STEW (48 subjects, Emotiv, 14 channels at 128 Hz, rest vs. the SIMKAP multitasking test), in the pipeline every model shares there: 4-s windows with 25% overlap, 7-fold GroupKFold across subjects, CBraMod's training loop with only the model swapped. Data, windows, folds and run configuration are in [cbramod/finetune-stew](../../../cbramod/experiments/finetune-stew), which holds the shared code. Implementation: EEG Conformer: the released model: for SEED / SEED-V the model of `conformer_seed_1s_5fold.py` (5 heads, its SEED head); for STEW the model of `conformer.py` (10 heads, its 3-layer head sized for 28 tokens). Seeds 0–4 × 7 folds per task.

No paper among these methods reports STEW, and the dataset paper's 69% / κ 0.46 (3 levels) comes from its own feature pipeline: the comparison is between models. Each seed's score is computed on the pooled predictions of its 7 test folds; mean ± SD over seeds (`results/results.txt`).

## Results

| Task | Model | Seeds × folds | Accuracy | Bal. acc | κ | Weighted F1 | Fold SD (accuracy) |
|---|---|---|---|---|---|---|---|
| rest vs. task | EEG Conformer | 5 × 7 | 0.7364 ± 0.0151 | 0.7364 ± 0.0151 | 0.4728 ± 0.0301 | 0.7349 ± 0.0151 | 0.0554 |
| 3 rating levels | EEG Conformer | 5 × 7 | 0.5118 ± 0.0173 | 0.4792 ± 0.0129 | 0.2459 ± 0.0229 | 0.4975 ± 0.0173 | 0.0833 |

Against the other models (`python tools/compare_stew.py`; Welch's t-test over seeds):

- 0.7364 on rest vs. task and 0.5118 on 3 levels: behind DGCNN (−0.0383, p = 0.002) and the foundation models, ahead of EEGNet.

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
