# EEGNet (our PyTorch implementation) through CBraMod's fine-tuning pipeline

The CBraMod paper compares against EEGNet (Tables 3, 10, 12, 15). Its EEGNet rows come from BIOT's code, with
settings it does not report. This experiment asks:

> How does EEGNet-8,2 score when trained exactly as CBraMod is fine-tuned, with only the model swapped?

`code/run_eegnet.py` runs our PyTorch EEGNet-8,2 through CBraMod's *released* fine-tuning pipeline
(`methods/cbramod/upstream`, commit `b9e9610`). Everything else is that pipeline's: dataset loaders and subject
splits, `Trainer` (loss, AdamW, cosine schedule, gradient clipping, label smoothing, checkpoint on the best validation
κ / AUROC, one test evaluation) and `Evaluator`. 4 datasets × seeds 0–4, 20 runs, on 2026-09-30 / 10-01.

These runs were made and first reported inside CBraMod's
[finetune-public-datasets](../../../cbramod/experiments/finetune-public-datasets) (setting `eegnet`). They moved
here on 2026-10-29, so that each method's experiments sit under that method. The logs are the same files, and
`results/` is recomputed from them (bit-identical numbers).

## Results

Mean ± SD over seeds 0–4; p from Welch's t-test against the paper's mean ± SD.

| Dataset (CBraMod paper table) | Metric | Paper (EEGNet row) | Our EEGNet | Difference (p) |
|---|---|---|---|---|
| PhysioNet-MI (T3) | balanced acc. | 0.5814 ± 0.0125 | 0.5948 ± 0.0028 | +0.0134 (0.074) |
|  | Cohen's κ | 0.4468 ± 0.0199 | 0.4597 ± 0.0037 | +0.0129 (0.22) |
|  | weighted F1 | 0.5796 ± 0.0115 | 0.5955 ± 0.0023 | +0.0159 (0.035) |
| BCIC-IV-2a (T15) | balanced acc. | 0.4482 ± 0.0094 | 0.5288 ± 0.0394 | +0.0806 (0.009) |
|  | Cohen's κ | 0.2693 ± 0.0121 | 0.3718 ± 0.0525 | +0.1025 (0.011) |
|  | weighted F1 | 0.4226 ± 0.0108 | 0.5094 ± 0.0491 | +0.0868 (0.015) |
| Mumtaz2016 (T10) | balanced acc. | 0.9232 ± 0.0104 | 0.8828 ± 0.0295 | −0.0404 (0.034) |
|  | AUC-PR | 0.9626 ± 0.0095 | 0.9596 ± 0.0096 | −0.0030 (0.63) |
|  | AUROC | 0.9639 ± 0.0093 | 0.9540 ± 0.0073 | −0.0099 (0.10) |
| MentalArithmetic (T12) | balanced acc. | 0.6770 ± 0.0116 | 0.5569 ± 0.0247 | −0.1201 (< 0.001) |
|  | AUC-PR | 0.5763 ± 0.0102 | 0.6104 ± 0.1208 | +0.0341 (0.56) |
|  | AUROC | 0.7321 ± 0.0108 | 0.7794 ± 0.0866 | +0.0473 (0.29) |

Balanced accuracy, compared with CBraMod in the same pipeline
([finetune-public-datasets](../../../cbramod/experiments/finetune-public-datasets)):

| Dataset | CBraMod, pre-trained | CBraMod, from scratch | EEGNet |
|---|---|---|---|
| PhysioNet-MI | 0.6265 ± 0.0054 | 0.5738 ± 0.0082 | 0.5948 ± 0.0028 |
| BCIC-IV-2a | 0.4571 ± 0.0399 | 0.4561 ± 0.0195 | **0.5288 ± 0.0394** |
| Mumtaz2016 | 0.8933 ± 0.0101 | 0.8973 ± 0.0115 | 0.8828 ± 0.0295 |
| MentalArithmetic | 0.5889 ± 0.0209 | 0.5924 ± 0.0403 | 0.5569 ± 0.0247 |

- **The paper's EEGNet rows hold within noise on 6 of 12 metrics.** BCIC-IV-2a is far above the paper (+0.08
  balanced accuracy) and MentalArithmetic balanced accuracy far below (−0.12).
- **On BCIC-IV-2a, EEGNet beats the reproduced CBraMod.** On PhysioNet-MI it lands between pre-trained and scratch
  CBraMod. On Mumtaz2016 the three are within noise.
- braindecode's `EEGNet` in the same pipeline gives the same numbers: none of the 12 metrics differs significantly
  ([braindecode-cbramod-splits](../braindecode-cbramod-splits)).

`results/results.txt` has every metric with per-seed values and the selected epochs.

## Paper vs. this reproduction

### What differs, and why

| | CBraMod paper, EEGNet rows | This experiment | Why |
|---|---|---|---|
| Implementation | BIOT's code, settings not reported | `code/run_eegnet.py`: our PyTorch EEGNet-8,2 | an implementation whose settings are known |
| Pipeline | not reported | CBraMod's released loaders, splits, `Trainer` (every Table 6 setting) and `Evaluator` | only the model differs from the CBraMod runs |
| Learning rate | not reported | 5e-4 for the whole model: the `Trainer`'s head rate, since EEGNet has no `backbone` parameters | the released `multi_lr` rule |
| Seeds | not reported | 0–4 | as the CBraMod runs |

### Settings we had to choose

| Setting | Value | Why |
|---|---|---|
| Variant | EEGNet-8,2: F1 = 8, D = 2, F2 = 16 | the EEGNet paper's cross-subject model |
| Temporal kernel | 100 samples (0.5 s at 200 Hz) | the EEGNet paper's 64 samples at 128 Hz |
| Dropout | 0.25 | Lawhern et al.'s value for cross-subject training |
| Max-norm constraints, initialisation, batch norm | none; PyTorch defaults | a plain PyTorch port; braindecode's version, which has them, scores the same |

### Run configuration

CBraMod's `finetune_main.py` defaults (paper Table 6): AdamW, weight decay 5e-2, cosine schedule, 50 epochs,
label smoothing 0.1, gradient clipping 1, batch 64, lr 1e-4 (head 5e-4 under `multi_lr`). Environment: conda env `cbramod`
(Python 3.11.7, PyTorch 2.1.2 + CUDA 12.1; `env/setup_env.sh`, `env/pip_freeze.txt`). Hardware: Quadro M6000
12 GB, one run per GPU.

## Layout

```
code/run_eegnet.py       EEGNet-8,2 through CBraMod's released Trainer and loaders
code/add_jobs.py         enqueue runs for tools/run_queue.py
code/collect.py          logs/runs -> results.txt / results.json, Welch tests vs. the paper
code/export_summary.py   results/summary.csv (feeds RESULTS.md)
logs/runs/<run>/log.txt  the 20 run logs
results/                 results.txt, results.json, summary.csv
env/                     setup_env.sh, pip_freeze.txt
```

## Reproducing

From the committed logs (Python standard library + scipy):

```bash
mkdir -p results_rerun && python code/collect.py logs/runs results_rerun > results_rerun/results.txt \
  && diff results_rerun/results.txt results/results.txt
```

Training needs CBraMod's preprocessed LMDBs (made as in
[finetune-public-datasets](../../../cbramod/experiments/finetune-public-datasets#reproducing)):

```bash
for ds in PhysioNet-MI BCIC-IV-2a Mumtaz2016 MentalArithmetic; do python code/add_jobs.py $ds 0 1 2 3 4; done
RUN_QUEUE_DIR=~/repro/queue python ../../../../tools/run_queue.py 0 1 2 3 4 5 6 7
```

## Provenance

Runs: 2026-09-30 / 10-01 on mcl-server (`~/repro/runs/*__eegnet__s*`), with CBraMod's released code at `b9e9610`.
