# braindecode's ATCNet on STEW

braindecode's ATCNet on STEW (48 subjects, Emotiv, 14 channels at 128 Hz, rest vs. the SIMKAP multitasking test), in the pipeline every model shares there: 4-s windows with 25% overlap, 7-fold GroupKFold across subjects, CBraMod's training loop with only the model swapped. Data, windows, folds and run configuration are in [cbramod/finetune-stew](../../../cbramod/experiments/finetune-stew), which holds the shared code. Implementation: braindecode's ATCNet: braindecode 1.8.1's `ATCNet` (the released code is TensorFlow); on 1-s SEED samples `conv_block_pool_size_2=2` keeps the five sliding windows. Seeds 0–4 × 7 folds per task.

No paper among these methods reports STEW, and the dataset paper's 69% / κ 0.46 (3 levels) comes from its own feature pipeline: the comparison is between models. Each seed's score is computed on the pooled predictions of its 7 test folds; mean ± SD over seeds (`results/results.txt`).

## Results

| Task | Model | Seeds × folds | Accuracy | Bal. acc | κ | Weighted F1 | Fold SD (accuracy) |
|---|---|---|---|---|---|---|---|
| rest vs. task | braindecode's ATCNet | 5 × 7 | 0.7257 ± 0.0054 | 0.7257 ± 0.0054 | 0.4514 ± 0.0109 | 0.7246 ± 0.0058 | 0.0623 |
| 3 rating levels | braindecode's ATCNet | 5 × 7 | 0.5290 ± 0.0057 | 0.4849 ± 0.0091 | 0.2568 ± 0.0120 | 0.5014 ± 0.0041 | 0.0702 |

Against the other models (`python tools/compare_stew.py`; Welch's t-test over seeds):

- **Mid-table on both tasks:** 0.7257 on rest vs. task (κ 0.451) and 0.5290 on 3 levels (κ 0.257); level with EEG Conformer (−0.011, p = 0.20; +0.006, p = 0.45) and braindecode's DeepSleepNet, above EEGNet (+0.061 and +0.076 balanced accuracy, p ≤ 0.002), below DGCNN (−0.049, p < 0.001; −0.017, p = 0.012) and CBraMod.
- **On 4-s windows it fits the training data,** unlike on SEED's 1-s samples ([braindecode-seed](../braindecode-seed)): training loss 0.48 at epoch 50 on rest vs. task (chance level ln 2 = 0.69; EEGNet 0.53, CBraMod 0.20; fold 0, seed 0).
- Runs took about 30 minutes each on a Quadro M6000, one per GPU.

## Paper vs. this reproduction

Every difference from the released pipeline, and every setting chosen for all models, is listed once in [the shared README](../../../cbramod/experiments/finetune-stew).
Model-specific settings are in `../../../cbramod/experiments/finetune-stew/code/stew_models.py` and summarised above.
Hardware: RTX A6000 (mcl-server2, shared with other users); ATCNet on mcl-server (Quadro M6000).

## Reproducing

`code/reproduce.sh collect` recomputes `results/` from `logs/runs` (numpy, scikit-learn); `code/reproduce.sh train` enqueues the runs.
`logs/runs/<run>/` holds `log.txt` (progress output removed), `metrics.json` (validation and test scores of every epoch) and
`test_preds.npz` (test labels and predictions at the selected epoch).

## Provenance

Runs: 2026-11-05 to 11-08, `~/repro-seed/stew/runs` on mcl-server and mcl-server2.
