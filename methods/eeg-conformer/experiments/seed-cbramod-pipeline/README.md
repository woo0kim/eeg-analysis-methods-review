# EEG Conformer on SEED and SEED-V, in CBraMod's SEED-V pipeline

EEG Conformer through the pipeline every model shares on SEED and SEED-V: CBraMod's released SEED-V code (its loader, 1-s samples, trainer, checkpoint on the best validation κ and metric code), with only the model swapped. Data, splits, pipeline and run configuration are in [cbramod/finetune-seed](../../../cbramod/experiments/finetune-seed), which holds the shared code. Implementation: EEG Conformer: the released model: for SEED / SEED-V the model of `conformer_seed_1s_5fold.py` (5 heads, its SEED head); for STEW the model of `conformer.py` (10 heads, its 3-layer head sized for 28 tokens). Seeds 0–4 on each dataset and split.

The SEED-V trial split is the CBraMod paper's protocol (Table 2), which reports EEG Conformer 0.3537 / 0.1772 / 0.3487 (balanced accuracy / κ / weighted F1). Each seed's score is the test score at the epoch with the best validation κ; mean ± SD over seeds (`results/results.txt`).
The clip vote is the accuracy of a majority vote over the 1-s predictions of each clip, the level at which SEED labels exist.

## Results

| Dataset | Split | Model | Runs | Bal. acc | κ | Weighted F1 | Accuracy | Clip vote |
|---|---|---|---|---|---|---|---|---|
| SEED | trial | EEG Conformer | 5 | 0.5360 ± 0.0074 | 0.2640 ± 0.0146 | 0.5244 ± 0.0131 | 0.5256 ± 0.0116 | 0.5591 ± 0.0225 |
| SEED | subject | EEG Conformer | 5 | 0.5378 ± 0.0105 | 0.3122 ± 0.0157 | 0.5134 ± 0.0112 | 0.5433 ± 0.0102 | 0.5644 ± 0.0062 |
| SEED-V | trial | EEG Conformer | 5 | 0.3205 ± 0.0137 | 0.1464 ± 0.0196 | 0.3135 ± 0.0229 | 0.3249 ± 0.0170 | 0.3642 ± 0.0337 |
| SEED-V | subject | EEG Conformer | 5 | 0.2366 ± 0.0089 | 0.0420 ± 0.0116 | 0.1888 ± 0.0196 | 0.2181 ± 0.0120 | 0.2252 ± 0.0084 |

Against the other models (`python tools/compare_seed.py`; Welch's t-test over seeds):

- **Below its number in the CBraMod paper on SEED-V:** 0.3205 against 0.3537 balanced accuracy (p = 0.003), and lower κ and weighted F1 (p = 0.031, 0.023).
- **On SEED it trails DGCNN** (0.5360 against 0.5741, p = 0.005) and CBraMod (−0.0802, p < 0.001), and it is third of six on the subject split (0.5378).
- Its own SEED protocol (5 folds inside clips) gives 98.24%, and 71.6% once clips are held out ([seed-5fold](../seed-5fold)).

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
