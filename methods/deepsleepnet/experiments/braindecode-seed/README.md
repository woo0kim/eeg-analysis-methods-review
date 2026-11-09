# braindecode's DeepSleepNet on SEED and SEED-V, in CBraMod's SEED-V pipeline

braindecode's DeepSleepNet through the pipeline every model shares on SEED and SEED-V: CBraMod's released SEED-V code (its loader, 1-s samples, trainer, checkpoint on the best validation κ and metric code), with only the model swapped. Data, splits, pipeline and run configuration are in [cbramod/finetune-seed](../../../cbramod/experiments/finetune-seed), which holds the shared code. Implementation: braindecode's DeepSleepNet: braindecode 1.8.1's `DeepSleepNet` (the released code is TensorFlow and sleep-specific), braindecode's default first layers. Seeds 0–4 on each dataset and split.

No paper reports this model in this pipeline on these datasets: the comparison is between models. Each seed's score is the test score at the epoch with the best validation κ; mean ± SD over seeds (`results/results.txt`).
The clip vote is the accuracy of a majority vote over the 1-s predictions of each clip, the level at which SEED labels exist.

## Results

| Dataset | Split | Model | Runs | Bal. acc | κ | Weighted F1 | Accuracy | Clip vote |
|---|---|---|---|---|---|---|---|---|
| SEED | trial | braindecode's DeepSleepNet | 5 | 0.5912 ± 0.0094 | 0.3775 ± 0.0132 | 0.6097 ± 0.0078 | 0.6095 ± 0.0080 | 0.6702 ± 0.0106 |
| SEED | subject | braindecode's DeepSleepNet | 5 | 0.5034 ± 0.0293 | 0.2592 ± 0.0444 | 0.5023 ± 0.0301 | 0.5060 ± 0.0299 | 0.5363 ± 0.0580 |
| SEED-V | trial | braindecode's DeepSleepNet | 5 | 0.3995 ± 0.0030 | 0.2531 ± 0.0040 | 0.4076 ± 0.0035 | 0.4064 ± 0.0037 | 0.4783 ± 0.0157 |
| SEED-V | subject | braindecode's DeepSleepNet | 5 | 0.2421 ± 0.0117 | 0.0544 ± 0.0163 | 0.2283 ± 0.0176 | 0.2443 ± 0.0228 | 0.2593 ± 0.0262 |

Against the other models (`python tools/compare_seed.py`; Welch's t-test over seeds):

- **A sleep model level with CBraMod on SEED-V's trial split:** 0.3995 against 0.3962, third after the two LaBraMs; on SEED 0.5912, ahead of DGCNN and CBraMod without pre-training.
- **Weakest across subjects on SEED** (0.5034), near chance on SEED-V (0.2421) as every model.
- Not a reproduction: DeepSleepNet's papers have no emotion result. On a 1-s window its first layers use braindecode's defaults (kernels 0.25 s and 2 s at 200 Hz); the 200-Hz sizes used on Sleep-EDF do not fit.

## Paper vs. this reproduction

Every difference from the released pipeline, and every setting chosen for all models, is listed once in [the shared README](../../../cbramod/experiments/finetune-seed).
Model-specific settings are in `../../../cbramod/experiments/finetune-seed/code/emo_models.py` and summarised above.
Hardware: Quadro M6000 (mcl-server) for the trial split; RTX A6000 (mcl-server2, shared with other users) for the subject split.

## Reproducing

`code/reproduce.sh collect` recomputes `results/` from `logs/runs` (numpy, scikit-learn); `code/reproduce.sh train` enqueues the runs.
`logs/runs/<run>/` holds `log.txt` (progress output removed), `metrics.json` (validation and test scores of every epoch) and
`test_preds.npz` (test labels and predictions at the selected epoch).

## Provenance

Runs: 2026-11-05 to 11-08, `~/repro-seed/runs` on mcl-server and mcl-server2.
