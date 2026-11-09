# DGCNN on STEW

DGCNN on STEW (48 subjects, Emotiv, 14 channels at 128 Hz, rest vs. the SIMKAP multitasking test), in the pipeline every model shares there: 4-s windows with 25% overlap, 7-fold GroupKFold across subjects, CBraMod's training loop with only the model swapped. Data, windows, folds and run configuration are in [cbramod/finetune-stew](../../../cbramod/experiments/finetune-stew), which holds the shared code. Implementation: DGCNN: our DGCNN ([../../code/dgcnn.py](../../code/dgcnn.py)) on 5-band differential entropy of each 1-s segment, standardised per band with training statistics. Seeds 0–4 × 7 folds per task.

No paper among these methods reports STEW, and the dataset paper's 69% / κ 0.46 (3 levels) comes from its own feature pipeline: the comparison is between models. Each seed's score is computed on the pooled predictions of its 7 test folds; mean ± SD over seeds (`results/results.txt`).

## Results

| Task | Model | Seeds × folds | Accuracy | Bal. acc | κ | Weighted F1 | Fold SD (accuracy) |
|---|---|---|---|---|---|---|---|
| rest vs. task | DGCNN | 5 × 7 | 0.7747 ± 0.0073 | 0.7747 ± 0.0073 | 0.5493 ± 0.0145 | 0.7735 ± 0.0075 | 0.0496 |
| 3 rating levels | DGCNN | 5 × 7 | 0.5517 ± 0.0074 | 0.5017 ± 0.0068 | 0.2971 ± 0.0110 | 0.5455 ± 0.0072 | 0.0744 |

Against the other models (`python tools/compare_stew.py`; Welch's t-test over seeds):

- **DGCNN is the best model without pre-training** (ours 0.7747 on rest vs. task, braindecode's on the same DE input 0.7833): 0.0383 above EEG Conformer (p = 0.002) and 0.110 above EEGNet; 0.0252 below CBraMod (p = 0.002).
- **Close to the foundation models on 3 levels:** 0.5517 against CBraMod's 0.5661 (κ p = 0.096).

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
