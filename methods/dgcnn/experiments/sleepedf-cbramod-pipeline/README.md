# DGCNN on Sleep-EDF, in CBraMod's sleep-staging pipeline

DGCNN as the epoch encoder of the shared Sleep-EDF pipeline: same data, 62 / 8 / 8 subject split, sequence head, trainer and
seeds as every other model. The pipeline is documented in
[cbramod/finetune-sleepedf](../../../cbramod/experiments/finetune-sleepedf). 2 models × seeds 0–4.

| Model | Implementation | Encoder output per 30-s epoch |
|---|---|---|
| `dgcnn` | ours, on 5-band DE of each of the 30 seconds (150 values per channel) | flattened graph output, 2 × 64 |
| `bd-dgcnn` | braindecode 1.8.1, as it ships, raw samples as node features | its 256-unit layer |

## Results

The common test set of 20,976 epochs (`python ../../../../tools/compare_sleepedf.py`):

| Model | Bal. acc | κ | Macro F1 | Accuracy |
|---|---|---|---|---|
| DGCNN, DE input | 0.6983 ± 0.0100 | 0.6941 ± 0.0081 | 0.6917 ± 0.0117 | 0.7806 ± 0.0060 |
| DGCNN, braindecode, raw input | 0.6017 ± 0.0260 | 0.5956 ± 0.0192 | 0.6059 ± 0.0175 | 0.7114 ± 0.0146 |

- DGCNN sits with EEG Conformer and EEGNet (κ 0.69, 0.68) and below the foundation models (0.77): CBraMod − DGCNN = +0.0734 κ
  (p < 0.001).
- **The input decides here too:** raw samples cost 0.0985 κ against DE (p < 0.001). With two channels the graph has one possible
  edge, so these runs test the features and the head more than the graph.

## Paper vs. this reproduction

Not a reproduction: DGCNN's paper has no sleep result. Differences from the paper are those of
[cbramod-splits](../cbramod-splits), with the shared Sleep-EDF pipeline in place of CBraMod's fine-tuning path.

## Reproducing

`code/reproduce.sh collect` recomputes `results/` from `logs/runs`; `code/reproduce.sh train` enqueues the runs (20–28 minutes each on a Quadro M6000 shared by 2 jobs).

## Provenance

Runs: 2026-11-06 on mcl-server, `~/repro-seed/runs`.
