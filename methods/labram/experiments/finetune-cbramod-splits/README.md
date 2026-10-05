# LaBraM-Base under CBraMod's splits: the baseline, run on our end

The CBraMod paper (ICLR 2025) compares CBraMod with LaBraM-Base, its strongest baseline, using LaBraM numbers
from its own runs ("we fine-tune BIOT and LaBraM based on their public code and pre-trained weights";
"strict consistency in the splits … for every method"). Our CBraMod runs
([finetune-public-datasets](../../../cbramod/experiments/finetune-public-datasets)) came out below the
paper, so comparing them with the paper's LaBraM numbers would be unfair. This experiment re-runs
LaBraM-Base with its released code and weights under exactly the conditions of our CBraMod runs:

- the same preprocessed samples (CBraMod's LMDB files) and subject-independent train/val/test splits,
- seeds 0–4,
- the same checkpoint rule (best validation Cohen's κ for multi-class, AUROC for binary; first on ties),
- the same metric code (CBraMod's `finetune_evaluator.py` formulas),
- the same GPUs (mcl-server, Quadro M6000), one run per GPU.

Runs were made on 2026-10-04: 20 runs (4 datasets × 5 seeds), all exited normally, ~1.2 h wall time
on 8 GPUs (PhysioNet-MI ~67 min per run).

## Results

### LaBraM-Base on our end vs. the CBraMod paper's LaBraM-Base numbers

| Dataset (CBraMod paper table) | Metric | Paper | Ours (5 seeds) | Δ | p (Welch) |
|---|---|---|---|---|---|
| PhysioNet-MI (T3) | balanced acc. | 0.6173 ± 0.0122 | 0.6098 ± 0.0098 | −0.0075 | 0.32 |
| | Cohen's κ | 0.4912 ± 0.0192 | 0.4797 ± 0.0131 | −0.0115 | 0.30 |
| | weighted F1 | 0.6177 ± 0.0141 | 0.6129 ± 0.0095 | −0.0048 | 0.54 |
| BCIC-IV-2a (T15) | balanced acc. | 0.4869 ± 0.0085 | 0.4477 ± 0.0421 | −0.0392 | 0.11 |
| | Cohen's κ | 0.3159 ± 0.0154 | 0.2637 ± 0.0562 | −0.0522 | 0.11 |
| | weighted F1 | 0.4758 ± 0.0103 | 0.3911 ± 0.0534 | −0.0847 | 0.023 |
| Mumtaz2016 (T10) | balanced acc. | 0.9409 ± 0.0079 | 0.8905 ± 0.0041 | −0.0504 | <0.001 |
| | AUC-PR | 0.9798 ± 0.0093 | 0.9711 ± 0.0089 | −0.0087 | 0.17 |
| | AUROC | 0.9782 ± 0.0057 | 0.9664 ± 0.0166 | −0.0118 | 0.19 |
| MentalArithmetic (T12) | balanced acc. | 0.6909 ± 0.0125 | 0.6562 ± 0.0397 | −0.0346 | 0.13 |
| | AUC-PR | 0.5999 ± 0.0155 | 0.6146 ± 0.0793 | +0.0147 | 0.70 |
| | AUROC | 0.7721 ± 0.0093 | 0.7791 ± 0.0608 | +0.0070 | 0.81 |

10 of the 12 metrics are within noise of the paper's LaBraM numbers. The exceptions are Mumtaz2016
balanced accuracy (−0.050) and BCIC-IV-2a weighted F1 (−0.085). The Mumtaz2016 shortfall is shared by
all three models we ran through CBraMod's released data pipeline (balanced accuracy: CBraMod −0.063,
EEGNet −0.040, LaBraM −0.050), whose subject split differs from the one the paper describes; it is a
property of the pipeline, not of one model.

### CBraMod vs. LaBraM-Base, both run on our end

| Dataset | Metric | CBraMod (ours) | LaBraM-Base (ours) | CBraMod − LaBraM | p (Welch) | Paper: CBraMod − LaBraM |
|---|---|---|---|---|---|---|
| PhysioNet-MI | balanced acc. | 0.6265 ± 0.0054 | 0.6098 ± 0.0098 | +0.0167 | 0.015 | +0.0244 |
| | Cohen's κ | 0.5019 ± 0.0072 | 0.4797 ± 0.0131 | +0.0222 | 0.015 | +0.0310 |
| | weighted F1 | 0.6271 ± 0.0051 | 0.6129 ± 0.0095 | +0.0142 | 0.025 | +0.0250 |
| BCIC-IV-2a | balanced acc. | 0.4571 ± 0.0399 | 0.4477 ± 0.0421 | +0.0094 | 0.73 | +0.0269 |
| | Cohen's κ | 0.2762 ± 0.0532 | 0.2637 ± 0.0562 | +0.0125 | 0.73 | +0.0359 |
| | weighted F1 | 0.4138 ± 0.0673 | 0.3911 ± 0.0534 | +0.0227 | 0.57 | +0.0226 |
| Mumtaz2016 | balanced acc. | 0.8933 ± 0.0101 | 0.8905 ± 0.0041 | +0.0028 | 0.59 | +0.0151 |
| | AUC-PR | 0.9780 ± 0.0049 | 0.9711 ± 0.0089 | +0.0068 | 0.18 | +0.0125 |
| | AUROC | 0.9770 ± 0.0059 | 0.9664 ± 0.0166 | +0.0107 | 0.23 | +0.0139 |
| MentalArithmetic | balanced acc. | 0.5889 ± 0.0209 | 0.6562 ± 0.0397 | −0.0674 | 0.015 | +0.0347 |
| | AUC-PR | 0.5110 ± 0.1096 | 0.6146 ± 0.0793 | −0.1036 | 0.13 | +0.0268 |
| | AUROC | 0.7344 ± 0.0515 | 0.7791 ± 0.0608 | −0.0447 | 0.25 | +0.0184 |

In the paper CBraMod beats LaBraM-Base on all 12 metrics. Run the same way on our end, CBraMod is
significantly better on PhysioNet-MI (all three metrics, by about two-thirds of the paper's margin),
indistinguishable from LaBraM-Base on BCIC-IV-2a and Mumtaz2016, and worse on MentalArithmetic (balanced
accuracy −0.067, p = 0.015). The paper's ranking holds on one of the four datasets. Caveats: 5 seeds per
model, Welch tests without multiple-comparison correction, and one untuned recipe per model.

### LaBraM's own checkpoint rule

Picking the epoch by validation accuracy instead (`labram@val-acc`, same runs) chooses the same epoch in
every PhysioNet-MI and BCIC-IV-2a run. On the binary datasets it gives MentalArithmetic 0.6632 ± 0.0988
balanced accuracy (AUROC 0.7835 ± 0.0241) and Mumtaz2016 0.8919 ± 0.0030 (AUROC 0.9669 ± 0.0169); the
conclusions above do not change. `results/results.txt` has every metric with per-seed values and the
selected epochs; LaBraM's own PyHealth metrics for every epoch are in `logs/runs/*/log.txt`.

## Paper vs. this reproduction

### What differs, and why

| | CBraMod paper (how it ran LaBraM) | Released LaBraM code | This experiment | Why |
|---|---|---|---|---|
| Training code | "public code and pre-trained weights" | `run_class_finetuning.py`, `checkpoints/labram-base.pth` | the same script, unmodified, started through `code/run_labram.py`, which only replaces `get_dataset()` and records the probabilities `evaluate()` scores | what the paper says it did |
| Data and splits | same splits for every method | loaders for TUAB and TUEV only | CBraMod's LMDB files: identical samples and subject splits to our CBraMod runs | the comparison is only fair on identical data |
| Input scale | – | the engine divides the input by 100 (µV → 0.1 mV) | LMDB values (µV) are passed unscaled, so LaBraM's own division gives the 0.1 mV units CBraMod's loaders produce | identical signal for both models |
| Channel names | – | required, in LaBraM's `standard_1020` vocabulary (its README: "significant to obtain normal performance") | copied from CBraMod's preprocessing scripts, see `CHANNELS` in `code/run_labram.py`; MentalArithmetic drops `EEG A2-A1` (19 of CBraMod's 20 channels) | A2−A1 is an ear-reference derivation with no electrode embedding in LaBraM |
| Checkpoint rule | κ (multi-class) / AUROC (binary) as the monitor score | best validation accuracy | headline: CBraMod's rule; LaBraM's own rule is reported as `labram@val-acc`, read from the same runs | one rule for both models; LaBraM's rule shown for transparency |
| Metrics | balanced acc., κ, weighted F1 / balanced acc., AUC-PR, AUROC | PyHealth; AUC-PR = average precision | CBraMod's formulas on LaBraM's saved probabilities; AUC-PR = trapezoidal area under the PR curve | one metric implementation for both models (other metrics are identical in the two codes) |
| GPUs | not stated | README example: 8 GPUs with `torchrun` (effective batch 512) | 1 GPU, batch 64, launched as a one-process distributed job (`RANK=0 WORLD_SIZE=1`) | one run per GPU, as for CBraMod; without the distributed variables LaBraM never calls `set_epoch()`, and its sampler would repeat one data order every epoch |
| `deepspeed==0.4.0` | – | in `requirements.txt` | not installed | only imported with `--enable_deepspeed`, which the recipe does not use |
| Checkpoints | – | saved every 5 epochs and at the best epoch | `--no_save_ckpt`; validation and test probabilities of every epoch saved instead (`preds.npz`) | every number can be recomputed without a GPU, and no checkpoint files are needed |

### Settings we had to choose (the papers and the code are silent)

| Setting | Value | Why |
|---|---|---|
| Fine-tuning hyperparameters | LaBraM README's TUAB command: lr 5e-4, 5 warm-up epochs, 50 epochs, layer decay 0.65, drop path 0.1, weight decay 0.05, batch 64, `--disable_rel_pos_bias --abs_pos_emb --disable_qkv_bias`; script defaults for the rest | the CBraMod paper gives no LaBraM settings; this is the authors' only published fine-tuning recipe. Not tuned per dataset, just as CBraMod uses one setting everywhere (the README notes lr and warm-up "strongly affect results") |
| Effective batch size | 64 (1 GPU × 64), not 512 | same as CBraMod's batch; at 512, MentalArithmetic (1,343 training samples) would get 2 updates per epoch |
| Seeds | 0–4 | same as our CBraMod runs |
| Epoch budget | 50 | same as CBraMod (Table 6) and the README recipe |

### Run configuration

Model `labram_base_patch200_200` (5.8M parameters) initialised from
`checkpoints/labram-base.pth`; mean pooling + linear head; AdamW (ε 1e-8, default β), peak lr 5e-4 with
layer-wise decay 0.65, linear warm-up from 0 over 5 epochs (the `--warmup_lr` argument is never used) then
cosine to 1e-6 per step, weight decay 0.05 (constant),
drop path 0.1, no gradient clipping, label smoothing 0.1 for multi-class and BCE for binary, mixed
precision (`torch.cuda.amp`), `cudnn.benchmark = True` (so re-runs are not bit-identical), 10 loader
workers. Environment: Python 3.11.16, PyTorch 2.0.1 + CUDA 11.8, timm 0.4.12, PyHealth 1.1.4, numpy
1.26.4, lmdb 1.4.1 (`env/setup_env.sh`, `env/pip_freeze.txt`) — the versions LaBraM's README names.
Hardware: 8× Quadro M6000 12 GB.

## Layout

```
code/run_labram.py      hooks CBraMod's data into LaBraM's released run_class_finetuning.py; saves preds.npz
code/add_jobs.py        queues the runs for tools/run_queue.py (recipe and launch variables)
code/collect.py         scores every run with CBraMod's rule and metric code -> results.txt / results.json
code/export_summary.py  results.json -> results/summary.csv (feeds the repo-wide RESULTS.md)
env/                    setup_env.sh, pip_freeze.txt
results/                results.txt, results.json, summary.csv
logs/runs/<dataset>__labram__s<seed>/
    preds.npz           validation and test probabilities for each of the 50 epochs, with labels
    log.txt             LaBraM's per-epoch JSON log (its own metrics)
    stdout.txt          full console output
logs/queue.log          start and end of every job
```

## Reproducing

From the committed predictions, with no GPU (numpy, scipy, scikit-learn):

```bash
mkdir -p results_rerun
python code/collect.py logs/runs results_rerun > results_rerun/results.txt
diff results_rerun/results.txt results/results.txt
python code/export_summary.py results
```

Re-running the training on a Linux GPU server, after the CBraMod data have been preprocessed (step 2 of
the [CBraMod experiment](../../../cbramod/experiments/finetune-public-datasets#reproducing)):

```bash
git clone --recurse-submodules https://github.com/woo0kim/eeg-analysis-methods-review
cd eeg-analysis-methods-review/methods/labram/experiments/finetune-cbramod-splits
bash env/setup_env.sh
for ds in PhysioNet-MI Mumtaz2016 BCIC-IV-2a MentalArithmetic; do python code/add_jobs.py $ds 0 1 2 3 4; done
RUN_QUEUE_DIR=${LABRAM_WORK:-~/repro-labram}/queue python ../../../../tools/run_queue.py 0 1 2 3 4 5 6 7
python code/collect.py ${LABRAM_WORK:-~/repro-labram}/runs results_rerun > results_rerun/results.txt
```

Paths come from `EEG_DATA_ROOT` (`~/data`), `LABRAM_WORK` (`~/repro-labram`), `LABRAM_SRC`
(`../../upstream`) and `LABRAM_PY` (`~/miniforge3/envs/labram/bin/python`).
