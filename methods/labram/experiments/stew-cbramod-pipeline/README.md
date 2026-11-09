# LaBraM-Base on STEW

LaBraM-Base on STEW (48 subjects, Emotiv, 14 channels at 128 Hz, rest vs. the SIMKAP multitasking test), in the pipeline every model shares there: 4-s windows with 25% overlap, 7-fold GroupKFold across subjects, CBraMod's training loop with only the model swapped. Data, windows, folds and run configuration are in [cbramod/finetune-stew](../../../cbramod/experiments/finetune-stew), which holds the shared code. Implementation: LaBraM-Base: the released code and weights, fine-tuned with LaBraM's own recipe (run_class_finetuning.py, as in [finetune-cbramod-splits](../finetune-cbramod-splits)). Seeds 0–4 × 7 folds per task.

No paper among these methods reports STEW, and the dataset paper's 69% / κ 0.46 (3 levels) comes from its own feature pipeline: the comparison is between models. Each seed's score is computed on the pooled predictions of its 7 test folds; mean ± SD over seeds (`results/results.txt`).

## Results

| Task | Model | Seeds × folds | Accuracy | Bal. acc | κ | Weighted F1 | Fold SD (accuracy) |
|---|---|---|---|---|---|---|---|
| rest vs. task | LaBraM-Base | 5 × 7 | 0.8179 ± 0.0107 | 0.8179 ± 0.0107 | 0.6359 ± 0.0214 | 0.8177 ± 0.0108 | 0.0453 |
| 3 rating levels | LaBraM-Base | 5 × 7 | 0.5670 ± 0.0082 | 0.5085 ± 0.0150 | 0.3132 ± 0.0182 | 0.5502 ± 0.0147 | 0.0390 |

Against the other models (`python tools/compare_stew.py`; Welch's t-test over seeds):

- **The best model on rest vs. task:** 0.8179 accuracy (κ 0.636), ahead of CBraMod by 0.0181 (p = 0.022) and DGCNN by 0.0433 (p < 0.001).
- **Tied with CBraMod on 3 rating levels** (0.5670 against 0.5661, p = 0.87).

## Paper vs. this reproduction

Every difference from the released pipeline, and every setting chosen for all models, is listed once in [the shared README](../../../cbramod/experiments/finetune-stew).
Model-specific settings are in `../../../cbramod/experiments/finetune-stew/code/stew_models.py` and summarised above.
Hardware: RTX A6000 (mcl-server2, shared with other users).

## Reproducing

`code/reproduce.sh collect` recomputes `results/` from `logs/runs` (numpy, scikit-learn); `code/reproduce.sh train` enqueues the runs.
`logs/runs/<run>/` holds `log.txt` (progress output removed), `metrics.json` (validation and test scores of every epoch) and
`test_preds.npz` (test labels and predictions at the selected epoch); LaBraM runs also keep its `stdout.txt` (per-batch warnings removed).

## Provenance

Runs: 2026-11-05 to 11-08, `~/repro-seed/stew/runs` on mcl-server and mcl-server2.
