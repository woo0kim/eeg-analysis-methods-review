# braindecode's EEGNet on SEED and SEED-V, in CBraMod's SEED-V pipeline

braindecode's EEGNet through the pipeline every model shares on SEED and SEED-V: CBraMod's released SEED-V code (its loader, 1-s samples, trainer, checkpoint on the best validation κ and metric code), with only the model swapped. Data, splits, pipeline and run configuration are in [cbramod/finetune-seed](../../../cbramod/experiments/finetune-seed), which holds the shared code. Implementation: braindecode's EEGNet: braindecode 1.8.1's `EEGNet` (EEGNet-8,2, max-norm classifier, as [braindecode-cbramod-splits](../braindecode-cbramod-splits)). Seeds 0–4 on each dataset and split.

No paper reports this model in this pipeline on these datasets: the comparison is between models. Each seed's score is the test score at the epoch with the best validation κ; mean ± SD over seeds (`results/results.txt`).
The clip vote is the accuracy of a majority vote over the 1-s predictions of each clip, the level at which SEED labels exist.

## Results

| Dataset | Split | Model | Runs | Bal. acc | κ | Weighted F1 | Accuracy | Clip vote |
|---|---|---|---|---|---|---|---|---|
| SEED | trial | braindecode's EEGNet | 5 | 0.4748 ± 0.0158 | 0.1909 ± 0.0235 | 0.4905 ± 0.0183 | 0.4962 ± 0.0140 | 0.5333 ± 0.0319 |
| SEED-V | trial | braindecode's EEGNet | 5 | 0.2809 ± 0.0124 | 0.1031 ± 0.0153 | 0.2577 ± 0.0176 | 0.3015 ± 0.0111 | 0.2883 ± 0.0238 |

Against the other models (`python tools/compare_seed.py`; Welch's t-test over seeds):

- **Level with our PyTorch EEGNet:** −0.0161 on SEED (p = 0.093), −0.0085 on SEED-V (p = 0.26).

## Paper vs. this reproduction

Every difference from the released pipeline, and every setting chosen for all models, is listed once in [the shared README](../../../cbramod/experiments/finetune-seed).
Model-specific settings are in `../../../cbramod/experiments/finetune-seed/code/emo_models.py` and summarised above.
Hardware: Quadro M6000 (mcl-server); trial split only.

## Reproducing

`code/reproduce.sh collect` recomputes `results/` from `logs/runs` (numpy, scikit-learn); `code/reproduce.sh train` enqueues the runs.
`logs/runs/<run>/` holds `log.txt` (progress output removed), `metrics.json` (validation and test scores of every epoch) and
`test_preds.npz` (test labels and predictions at the selected epoch).

## Provenance

Runs: 2026-11-05 to 11-08, `~/repro-seed/runs` on mcl-server and mcl-server2.
