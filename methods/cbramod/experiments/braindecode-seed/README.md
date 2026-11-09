# braindecode's CBraMod on SEED and SEED-V, in CBraMod's SEED-V pipeline

braindecode's CBraMod through the pipeline every model shares on SEED and SEED-V: CBraMod's released SEED-V code (its loader, 1-s samples, trainer, checkpoint on the best validation κ and metric code), with only the model swapped. Data, splits, pipeline and run configuration are in [cbramod/finetune-seed](../../../cbramod/experiments/finetune-seed), which holds the shared code. Implementation: braindecode's CBraMod: braindecode 1.8.1's `CBraMod` with its pre-trained weights and its own head (as [braindecode-public-datasets](../braindecode-public-datasets)). Seeds 0–4 on each dataset and split.

No paper reports this model in this pipeline on these datasets: the comparison is between models. Each seed's score is the test score at the epoch with the best validation κ; mean ± SD over seeds (`results/results.txt`).
The clip vote is the accuracy of a majority vote over the 1-s predictions of each clip, the level at which SEED labels exist.

## Results

| Dataset | Split | Model | Runs | Bal. acc | κ | Weighted F1 | Accuracy | Clip vote |
|---|---|---|---|---|---|---|---|---|
| SEED | trial | braindecode's CBraMod | 5 | 0.6198 ± 0.0029 | 0.4069 ± 0.0041 | 0.6214 ± 0.0031 | 0.6212 ± 0.0025 | 0.7022 ± 0.0147 |
| SEED-V | trial | braindecode's CBraMod | 5 | 0.3786 ± 0.0096 | 0.2219 ± 0.0122 | 0.3806 ± 0.0093 | 0.3797 ± 0.0092 | 0.4775 ± 0.0303 |

Against the other models (`python tools/compare_seed.py`; Welch's t-test over seeds):

- **Level with the released CBraMod on SEED** (+0.0036, p = 0.59), **0.0176 lower on SEED-V** (p = 0.011), where braindecode's one-layer head replaces the released three-layer head (as on PhysioNet-MI in [braindecode-public-datasets](../braindecode-public-datasets)).

## Paper vs. this reproduction

Every difference from the released pipeline, and every setting chosen for all models, is listed once in [the shared README](../../../cbramod/experiments/finetune-seed).
Model-specific settings are in `../../../cbramod/experiments/finetune-seed/code/emo_models.py` and summarised above.
Hardware: Quadro M6000 (mcl-server) for SEED-V, RTX A6000 (mcl-server2, shared with other users) for SEED; trial split only.

## Reproducing

`code/reproduce.sh collect` recomputes `results/` from `logs/runs` (numpy, scikit-learn); `code/reproduce.sh train` enqueues the runs.
`logs/runs/<run>/` holds `log.txt` (progress output removed), `metrics.json` (validation and test scores of every epoch) and
`test_preds.npz` (test labels and predictions at the selected epoch).

## Provenance

Runs: 2026-11-05 to 11-08, `~/repro-seed/runs` on mcl-server and mcl-server2.
