# braindecode's Labram on SEED and SEED-V, in CBraMod's SEED-V pipeline

braindecode's Labram through the pipeline every model shares on SEED and SEED-V: CBraMod's released SEED-V code (its loader, 1-s samples, trainer, checkpoint on the best validation κ and metric code), with only the model swapped. Data, splits, pipeline and run configuration are in [cbramod/finetune-seed](../../../cbramod/experiments/finetune-seed), which holds the shared code. Implementation: braindecode's Labram: braindecode 1.8.1's `Labram` with its pre-trained weights, fine-tuned with LaBraM's recipe (as [braindecode-cbramod-splits](../braindecode-cbramod-splits)). Seeds 0–4 on each dataset and split.

No paper reports this model in this pipeline on these datasets: the comparison is between models. Each seed's score is the test score at the epoch with the best validation κ; mean ± SD over seeds (`results/results.txt`).
The clip vote is the accuracy of a majority vote over the 1-s predictions of each clip, the level at which SEED labels exist.

## Results

| Dataset | Split | Model | Runs | Bal. acc | κ | Weighted F1 | Accuracy | Clip vote |
|---|---|---|---|---|---|---|---|---|
| SEED | trial | braindecode's Labram | 5 | 0.6161 ± 0.0121 | 0.3981 ± 0.0278 | 0.6137 ± 0.0217 | 0.6144 ± 0.0212 | 0.6933 ± 0.0270 |
| SEED-V | trial | braindecode's Labram | 5 | 0.4106 ± 0.0062 | 0.2618 ± 0.0085 | 0.4138 ± 0.0074 | 0.4102 ± 0.0072 | 0.5008 ± 0.0130 |

Against the other models (`python tools/compare_seed.py`; Welch's t-test over seeds):

- **Level with the released LaBraM-Base**: −0.0139 balanced accuracy on SEED (p = 0.061), −0.0030 on SEED-V (p = 0.565).

## Paper vs. this reproduction

Every difference from the released pipeline, and every setting chosen for all models, is listed once in [the shared README](../../../cbramod/experiments/finetune-seed).
Model-specific settings are in `../../../cbramod/experiments/finetune-seed/code/emo_models.py` and summarised above.
Hardware: Quadro M6000 (mcl-server) for SEED-V, RTX A6000 (mcl-server2, shared with other users) for SEED; trial split only.

## Reproducing

`code/reproduce.sh collect` recomputes `results/` from `logs/runs` (numpy, scikit-learn); `code/reproduce.sh train` enqueues the runs.
`logs/runs/<run>/` holds `log.txt` (progress output removed), `metrics.json` (validation and test scores of every epoch) and
`test_preds.npz` (test labels and predictions at the selected epoch); LaBraM runs also keep its `stdout.txt` (per-batch warnings removed).

## Provenance

Runs: 2026-11-05 to 11-08, `~/repro-seed/runs` on mcl-server and mcl-server2.
