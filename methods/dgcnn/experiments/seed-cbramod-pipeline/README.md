# DGCNN on SEED and SEED-V, in CBraMod's SEED-V pipeline

DGCNN through the pipeline every model shares on SEED and SEED-V: CBraMod's released SEED-V code (its loader, 1-s samples, trainer, checkpoint on the best validation κ and metric code), with only the model swapped. Data, splits, pipeline and run configuration are in [cbramod/finetune-seed](../../../cbramod/experiments/finetune-seed), which holds the shared code. Implementation: DGCNN: our DGCNN ([../../code/dgcnn.py](../../code/dgcnn.py)) on 5-band differential entropy of each 1-s segment, standardised per band with training statistics. Seeds 0–4 on each dataset and split.

No paper reports this model in this pipeline on these datasets: the comparison is between models. Each seed's score is the test score at the epoch with the best validation κ; mean ± SD over seeds (`results/results.txt`).
The clip vote is the accuracy of a majority vote over the 1-s predictions of each clip, the level at which SEED labels exist.

## Results

| Dataset | Split | Model | Runs | Bal. acc | κ | Weighted F1 | Accuracy | Clip vote |
|---|---|---|---|---|---|---|---|---|
| SEED | trial | DGCNN | 5 | 0.5741 ± 0.0175 | 0.3380 ± 0.0413 | 0.5807 ± 0.0305 | 0.5801 ± 0.0303 | 0.6622 ± 0.0441 |
| SEED | subject | DGCNN | 5 | 0.5215 ± 0.0139 | 0.2861 ± 0.0210 | 0.5234 ± 0.0126 | 0.5243 ± 0.0142 | 0.5393 ± 0.0379 |
| SEED-V | trial | DGCNN | 5 | 0.3866 ± 0.0051 | 0.2346 ± 0.0066 | 0.3930 ± 0.0051 | 0.3909 ± 0.0052 | 0.4717 ± 0.0162 |
| SEED-V | subject | DGCNN | 5 | 0.2563 ± 0.0097 | 0.0708 ± 0.0096 | 0.2453 ± 0.0056 | 0.2498 ± 0.0065 | 0.2874 ± 0.0097 |

Against the other models (`python tools/compare_seed.py`; Welch's t-test over seeds):

- **Ahead of EEG Conformer and EEGNet on both trial splits:** SEED 0.5741 (+0.0381 over EEG Conformer, p = 0.005; +0.083 over EEGNet, p < 0.001), SEED-V 0.3866 (+0.066, +0.097, p < 0.001). Behind braindecode's DeepSleepNet (SEED 0.5912, SEED-V 0.3995) and the foundation models: CBraMod −0.042 (SEED, p = 0.003), −0.0096 (SEED-V, p = 0.027).
- **Across subjects on SEED it scores 0.52 here against 77.6% in its own protocol** ([seed-paper-protocol](../seed-paper-protocol)). There it reads SEED's distributed `de_LDS` features (smoothed over each clip, so every second carries information from the whole clip) and z-scores every session with its own statistics (including the test subject's); here each 1-s window is described by its own DE only, as every other model sees only its own window.
- **On SEED-V across subjects it is the least bad model** (0.2563, κ 0.071), where every model is near chance.

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
