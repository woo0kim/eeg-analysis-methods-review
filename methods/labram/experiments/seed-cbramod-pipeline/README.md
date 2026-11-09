# LaBraM-Base on SEED and SEED-V, in CBraMod's SEED-V pipeline

LaBraM-Base through the pipeline every model shares on SEED and SEED-V: CBraMod's released SEED-V code (its loader, 1-s samples, trainer, checkpoint on the best validation κ and metric code), with only the model swapped. Data, splits, pipeline and run configuration are in [cbramod/finetune-seed](../../../cbramod/experiments/finetune-seed), which holds the shared code. Implementation: LaBraM-Base: the released code and weights, fine-tuned with LaBraM's own recipe (run_class_finetuning.py, as in [finetune-cbramod-splits](../finetune-cbramod-splits)). Seeds 0–4 on each dataset and split.

The SEED-V trial split is the CBraMod paper's protocol (Table 2), which reports LaBraM-Base 0.3976 / 0.2386 / 0.3974 (balanced accuracy / κ / weighted F1). Each seed's score is the test score at the epoch with the best validation κ; mean ± SD over seeds (`results/results.txt`).
The clip vote is the accuracy of a majority vote over the 1-s predictions of each clip, the level at which SEED labels exist.

## Results

| Dataset | Split | Model | Runs | Bal. acc | κ | Weighted F1 | Accuracy | Clip vote |
|---|---|---|---|---|---|---|---|---|
| SEED | trial | LaBraM-Base | 5 | 0.6300 ± 0.0008 | 0.4260 ± 0.0032 | 0.6348 ± 0.0034 | 0.6347 ± 0.0035 | 0.7120 ± 0.0101 |
| SEED | subject | LaBraM-Base | 5 | 0.5822 ± 0.0105 | 0.3788 ± 0.0159 | 0.5820 ± 0.0103 | 0.5863 ± 0.0107 | 0.6415 ± 0.0193 |
| SEED-V | trial | LaBraM-Base | 5 | 0.4136 ± 0.0094 | 0.2661 ± 0.0119 | 0.4162 ± 0.0092 | 0.4133 ± 0.0105 | 0.4908 ± 0.0242 |
| SEED-V | subject | LaBraM-Base | 5 | 0.2214 ± 0.0105 | 0.0257 ± 0.0128 | 0.1973 ± 0.0104 | 0.2196 ± 0.0071 | 0.2193 ± 0.0124 |

Against the other models (`python tools/compare_seed.py`; Welch's t-test over seeds):

- **The best model on SEED-V's trial split, and above its number in the CBraMod paper.** Balanced accuracy 0.4136 against the paper's 0.3976 (p = 0.069); κ +0.0275 (p = 0.041) and weighted F1 +0.0188 (p = 0.020). It beats CBraMod by 0.0174 balanced accuracy (p = 0.011), the opposite of the paper's ranking.
- **Also the best on SEED** (0.6300 trial split, 0.5822 subject split); against CBraMod −0.0138 (p = 0.087) and +0.0380 (p < 0.001).
- **At chance across subjects on SEED-V:** 0.2214 balanced accuracy (chance 0.20), κ 0.026, as every model on that split.

## Paper vs. this reproduction

Every difference from the released pipeline, and every setting chosen for all models, is listed once in [the shared README](../../../cbramod/experiments/finetune-seed).
Model-specific settings are in `../../../cbramod/experiments/finetune-seed/code/emo_models.py` and summarised above.
Hardware: Quadro M6000 (mcl-server) for the trial split; RTX A6000 (mcl-server2, shared with other users) for the subject split.

## Reproducing

`code/reproduce.sh collect` recomputes `results/` from `logs/runs` (numpy, scikit-learn); `code/reproduce.sh train` enqueues the runs.
`logs/runs/<run>/` holds `log.txt` (progress output removed), `metrics.json` (validation and test scores of every epoch) and
`test_preds.npz` (test labels and predictions at the selected epoch); LaBraM runs also keep its `stdout.txt` (per-batch warnings removed).

## Provenance

Runs: 2026-11-05 to 11-08, `~/repro-seed/runs` on mcl-server and mcl-server2.
