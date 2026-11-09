# braindecode's DGCNN, raw input, braindecode's DGCNN, DE input on SEED and SEED-V, in CBraMod's SEED-V pipeline

braindecode's DGCNN, raw input, braindecode's DGCNN, DE input through the pipeline every model shares on SEED and SEED-V: CBraMod's released SEED-V code (its loader, 1-s samples, trainer, checkpoint on the best validation κ and metric code), with only the model swapped. Data, splits, pipeline and run configuration are in [cbramod/finetune-seed](../../../cbramod/experiments/finetune-seed), which holds the shared code. Implementation: braindecode's DGCNN, raw input: braindecode 1.8.1's `DGCNN` as it ships (raw samples as node features); braindecode's DGCNN, DE input: braindecode 1.8.1's `DGCNN` on the same DE input as ours. Seeds 0–4 on each dataset and split.

No paper reports this model in this pipeline on these datasets: the comparison is between models. Each seed's score is the test score at the epoch with the best validation κ; mean ± SD over seeds (`results/results.txt`).
The clip vote is the accuracy of a majority vote over the 1-s predictions of each clip, the level at which SEED labels exist.

## Results

| Dataset | Split | Model | Runs | Bal. acc | κ | Weighted F1 | Accuracy | Clip vote |
|---|---|---|---|---|---|---|---|---|
| SEED | trial | braindecode's DGCNN, raw input | 5 | 0.5191 ± 0.0047 | 0.2510 ± 0.0106 | 0.5249 ± 0.0089 | 0.5245 ± 0.0087 | 0.6027 ± 0.0247 |
| SEED | trial | braindecode's DGCNN, DE input | 5 | 0.5749 ± 0.0121 | 0.3412 ± 0.0256 | 0.5844 ± 0.0181 | 0.5836 ± 0.0178 | 0.6524 ± 0.0330 |
| SEED-V | trial | braindecode's DGCNN, raw input | 5 | 0.2846 ± 0.0009 | 0.1041 ± 0.0014 | 0.2897 ± 0.0019 | 0.2896 ± 0.0014 | 0.3675 ± 0.0090 |
| SEED-V | trial | braindecode's DGCNN, DE input | 5 | 0.3752 ± 0.0039 | 0.2194 ± 0.0045 | 0.3793 ± 0.0033 | 0.3779 ± 0.0033 | 0.4742 ± 0.0215 |

Against the other models (`python tools/compare_seed.py`; Welch's t-test over seeds):

- **braindecode's DGCNN on our DE input matches ours on SEED** (+0.0008, p = 0.94) and is 0.0114 lower on SEED-V (p = 0.005).
- **As it ships, with raw samples as node features, it loses 0.055 (SEED) and 0.102 (SEED-V) balanced accuracy** (p ≤ 0.001): the input, not the graph, carries DGCNN.

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
