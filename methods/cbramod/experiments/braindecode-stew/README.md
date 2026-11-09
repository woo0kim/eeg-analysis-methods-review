# braindecode's CBraMod on STEW

braindecode's CBraMod on STEW (48 subjects, Emotiv, 14 channels at 128 Hz, rest vs. the SIMKAP multitasking test), in the pipeline every model shares there: 4-s windows with 25% overlap, 7-fold GroupKFold across subjects, CBraMod's training loop with only the model swapped. Data, windows, folds and run configuration are in [cbramod/finetune-stew](../../../cbramod/experiments/finetune-stew), which holds the shared code. Implementation: braindecode's CBraMod: braindecode 1.8.1's `CBraMod` with its pre-trained weights and its own head (as [braindecode-public-datasets](../braindecode-public-datasets)). Seeds 0–4 × 7 folds per task.

No paper among these methods reports STEW, and the dataset paper's 69% / κ 0.46 (3 levels) comes from its own feature pipeline: the comparison is between models. Each seed's score is computed on the pooled predictions of its 7 test folds; mean ± SD over seeds (`results/results.txt`).

## Results

| Task | Model | Seeds × folds | Accuracy | Bal. acc | κ | Weighted F1 | Fold SD (accuracy) |
|---|---|---|---|---|---|---|---|
| rest vs. task | braindecode's CBraMod | 5 × 7 | 0.7977 ± 0.0073 | 0.7977 ± 0.0073 | 0.5955 ± 0.0145 | 0.7974 ± 0.0073 | 0.0473 |
| 3 rating levels | braindecode's CBraMod | 5 × 7 | 0.5447 ± 0.0053 | 0.4945 ± 0.0067 | 0.2842 ± 0.0079 | 0.5378 ± 0.0050 | 0.0598 |

Against the other models (`python tools/compare_stew.py`; Welch's t-test over seeds):

- **Level with the released CBraMod** on rest vs. task (−0.0021, p = 0.71); **0.021 lower on 3 levels** (balanced accuracy, p = 0.021), as on SEED-V (−0.018, p = 0.011), where braindecode's one-layer head also replaces the released three-layer head.

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
