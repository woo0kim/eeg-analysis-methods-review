# braindecode's DeepSleepNet on STEW

braindecode's DeepSleepNet on STEW (48 subjects, Emotiv, 14 channels at 128 Hz, rest vs. the SIMKAP multitasking test), in the pipeline every model shares there: 4-s windows with 25% overlap, 7-fold GroupKFold across subjects, CBraMod's training loop with only the model swapped. Data, windows, folds and run configuration are in [cbramod/finetune-stew](../../../cbramod/experiments/finetune-stew), which holds the shared code. Implementation: braindecode's DeepSleepNet: braindecode 1.8.1's `DeepSleepNet` (the released code is TensorFlow and sleep-specific), braindecode's default first layers. Seeds 0–4 × 7 folds per task.

No paper among these methods reports STEW, and the dataset paper's 69% / κ 0.46 (3 levels) comes from its own feature pipeline: the comparison is between models. Each seed's score is computed on the pooled predictions of its 7 test folds; mean ± SD over seeds (`results/results.txt`).

## Results

| Task | Model | Seeds × folds | Accuracy | Bal. acc | κ | Weighted F1 | Fold SD (accuracy) |
|---|---|---|---|---|---|---|---|
| rest vs. task | braindecode's DeepSleepNet | 5 × 7 | 0.7168 ± 0.0100 | 0.7168 ± 0.0100 | 0.4336 ± 0.0200 | 0.7154 ± 0.0100 | 0.0407 |
| 3 rating levels | braindecode's DeepSleepNet | 5 × 7 | 0.5468 ± 0.0090 | 0.4899 ± 0.0107 | 0.2700 ± 0.0153 | 0.5263 ± 0.0102 | 0.0750 |

Against the other models (`python tools/compare_stew.py`; Welch's t-test over seeds):

- 0.7168 on rest vs. task, 0.5468 on 3 levels: between EEG Conformer and CBraMod without pre-training. Not a reproduction: DeepSleepNet's papers have no workload result.

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
