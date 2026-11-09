# EEGNet on SEED and SEED-V, in CBraMod's SEED-V pipeline

EEGNet through the pipeline every model shares on SEED and SEED-V: CBraMod's released SEED-V code (its loader, 1-s samples, trainer, checkpoint on the best validation κ and metric code), with only the model swapped. Data, splits, pipeline and run configuration are in [cbramod/finetune-seed](../../../cbramod/experiments/finetune-seed), which holds the shared code. Implementation: EEGNet: our PyTorch EEGNet-8,2 (of [pytorch-cbramod-splits](../pytorch-cbramod-splits)). Seeds 0–4 on each dataset and split.

The SEED-V trial split is the CBraMod paper's protocol (Table 2), which reports EEGNet 0.2961 / 0.1006 / 0.2749 (balanced accuracy / κ / weighted F1). Each seed's score is the test score at the epoch with the best validation κ; mean ± SD over seeds (`results/results.txt`).
The clip vote is the accuracy of a majority vote over the 1-s predictions of each clip, the level at which SEED labels exist.

## Results

| Dataset | Split | Model | Runs | Bal. acc | κ | Weighted F1 | Accuracy | Clip vote |
|---|---|---|---|---|---|---|---|---|
| SEED | trial | EEGNet | 5 | 0.4909 ± 0.0089 | 0.2053 ± 0.0203 | 0.4929 ± 0.0124 | 0.4978 ± 0.0158 | 0.5298 ± 0.0310 |
| SEED | subject | EEGNet | 5 | 0.5227 ± 0.0240 | 0.2882 ± 0.0371 | 0.5157 ± 0.0219 | 0.5264 ± 0.0248 | 0.5481 ± 0.0448 |
| SEED-V | trial | EEGNet | 5 | 0.2894 ± 0.0094 | 0.1133 ± 0.0097 | 0.2738 ± 0.0115 | 0.3080 ± 0.0064 | 0.3067 ± 0.0260 |
| SEED-V | subject | EEGNet | 5 | 0.2468 ± 0.0084 | 0.0555 ± 0.0099 | 0.2116 ± 0.0199 | 0.2387 ± 0.0080 | 0.2607 ± 0.0364 |

Against the other models (`python tools/compare_seed.py`; Welch's t-test over seeds):

- **Matches its number in the CBraMod paper on SEED-V:** 0.2894 against 0.2961 balanced accuracy (p = 0.31), κ 0.1133 against 0.1006 (p = 0.15).
- **The weakest original implementation on the trial splits** (SEED 0.4909, SEED-V 0.2894); DGCNN beats it by 0.083 and 0.097 (p < 0.001).

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
