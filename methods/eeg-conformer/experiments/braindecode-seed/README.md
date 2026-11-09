# braindecode's EEGConformer on SEED and SEED-V, in CBraMod's SEED-V pipeline

braindecode's EEGConformer through the pipeline every model shares on SEED and SEED-V: CBraMod's released SEED-V code (its loader, 1-s samples, trainer, checkpoint on the best validation κ and metric code), with only the model swapped. Data, splits, pipeline and run configuration are in [cbramod/finetune-seed](../../../cbramod/experiments/finetune-seed), which holds the shared code. Implementation: braindecode's EEGConformer: braindecode 1.8.1's `EEGConformer` with its defaults. Seeds 0–4 on each dataset and split.

No paper reports this model in this pipeline on these datasets: the comparison is between models. Each seed's score is the test score at the epoch with the best validation κ; mean ± SD over seeds (`results/results.txt`).
The clip vote is the accuracy of a majority vote over the 1-s predictions of each clip, the level at which SEED labels exist.

## Results

| Dataset | Split | Model | Runs | Bal. acc | κ | Weighted F1 | Accuracy | Clip vote |
|---|---|---|---|---|---|---|---|---|
| SEED | trial | braindecode's EEGConformer | 5 | 0.5259 ± 0.0095 | 0.2445 ± 0.0151 | 0.5071 ± 0.0168 | 0.5114 ± 0.0131 | 0.5547 ± 0.0158 |
| SEED-V | trial | braindecode's EEGConformer | 5 | 0.2962 ± 0.0123 | 0.1134 ± 0.0147 | 0.2696 ± 0.0248 | 0.2966 ± 0.0146 | 0.3358 ± 0.0210 |

Against the other models (`python tools/compare_seed.py`; Welch's t-test over seeds):

- **Slightly below the released model:** −0.0101 on SEED (p = 0.099), −0.0243 on SEED-V (p = 0.019).

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
