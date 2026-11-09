# braindecode's Labram on STEW

braindecode's Labram on STEW (48 subjects, Emotiv, 14 channels at 128 Hz, rest vs. the SIMKAP multitasking test), in the pipeline every model shares there: 4-s windows with 25% overlap, 7-fold GroupKFold across subjects, CBraMod's training loop with only the model swapped. Data, windows, folds and run configuration are in [cbramod/finetune-stew](../../../cbramod/experiments/finetune-stew), which holds the shared code. Implementation: braindecode's Labram: braindecode 1.8.1's `Labram` with its pre-trained weights, fine-tuned with LaBraM's recipe (as [braindecode-cbramod-splits](../braindecode-cbramod-splits)). Seeds 0–4 × 7 folds per task.

No paper among these methods reports STEW, and the dataset paper's 69% / κ 0.46 (3 levels) comes from its own feature pipeline: the comparison is between models. Each seed's score is computed on the pooled predictions of its 7 test folds; mean ± SD over seeds (`results/results.txt`).

## Results

| Task | Model | Seeds × folds | Accuracy | Bal. acc | κ | Weighted F1 | Fold SD (accuracy) |
|---|---|---|---|---|---|---|---|
| rest vs. task | braindecode's Labram | 5 × 7 | 0.8153 ± 0.0134 | 0.8153 ± 0.0134 | 0.6307 ± 0.0268 | 0.8149 ± 0.0136 | 0.0520 |
| 3 rating levels | braindecode's Labram | 5 × 7 | 0.5739 ± 0.0133 | 0.5263 ± 0.0125 | 0.3317 ± 0.0184 | 0.5649 ± 0.0065 | 0.0512 |

Against the other models (`python tools/compare_stew.py`; Welch's t-test over seeds):

- **Level with the released LaBraM-Base** on both tasks (rest vs. task −0.0026, p = 0.75); on 3 rating levels +0.018 balanced accuracy (p = 0.077) and the top κ of all models (0.332).

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
