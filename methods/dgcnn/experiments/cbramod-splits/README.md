# DGCNN on CBraMod's four public datasets

How does DGCNN do outside emotion recognition, on the splits every other model in this repository ran on?
`code/run_dgcnn.py` keeps everything of CBraMod's released pipeline (`methods/cbramod/upstream`, commit `b9e9610`): loaders and
subject splits, `Trainer`, `Evaluator`, seeds 0–4. Only the model differs. DGCNN's paper does not report these datasets.

| Setting | Model | Node features |
|---|---|---|
| `dgcnn` | ours (`../../code/dgcnn.py`) | 5-band DE of each 1-s segment (the paper's input) |
| `bd-dgcnn-de` | braindecode 1.8.1 `DGCNN` | the same |
| `bd-dgcnn` | braindecode 1.8.1 `DGCNN`, as it ships | raw samples |

## Results

Balanced accuracy, mean ± SD over seeds 0–4. Other models from [SUMMARY.md](../../../../SUMMARY.md), same splits and scoring.

| Dataset | DGCNN (DE) | DGCNN (bd, DE) | DGCNN (bd, raw) | CBraMod | LaBraM-Base | EEGNet |
|---|---|---|---|---|---|---|
| PhysioNet-MI | 0.4822 ± 0.0178 | 0.4914 ± 0.0098 | **0.5781 ± 0.0047** | 0.6265 | 0.6098 | 0.5948 |
| BCI-IV-2a | 0.3458 ± 0.0338 | 0.3563 ± 0.0373 | **0.4524 ± 0.0072** | 0.4571 | 0.4477 | 0.5288 |
| Mumtaz2016 | **0.9619 ± 0.0063** | **0.9704 ± 0.0040** | 0.8710 ± 0.0058 | 0.8933 | 0.8905 | 0.8828 |
| MentalArithmetic | 0.6236 ± 0.0665 | **0.6368 ± 0.0175** | 0.5590 ± 0.0179 | 0.5889 | 0.6562 | 0.5569 |

- **The input decides.** Against raw samples, DE (band power, no timing inside a second) gains 0.09–0.10 on Mumtaz2016 and
  0.06–0.08 on MentalArithmetic, and loses 0.09–0.11 on the two motor-imagery datasets, where raw samples keep the timing of
  the event.
- **On Mumtaz2016, DE-input DGCNN is the best model in this repository,** 0.96–0.97 against 0.89 for the foundation models, and
  above the CBraMod paper's 0.9560 for CBraMod. Spectral features alone separate these groups in 11 test subjects; whether
  that reflects depression or differences between the recordings cannot be told from this split.
- **The two implementations agree** when given the same input (within one SD on three datasets, 0.0085 apart on Mumtaz2016), which
  supports the port from another direction than the numeric check in
  [seed-paper-protocol](../seed-paper-protocol).

AUC-PR and AUROC (binary) and κ, weighted F1 (multi-class) are in `results/results.txt`.

## Paper vs. this reproduction

There is no paper number to reproduce: this experiment compares models on identical splits.

### What differs from DGCNN's paper, and why

| | Paper | This experiment | Why |
|---|---|---|---|
| Datasets | SEED, DREAMER | CBraMod's four | the question |
| Trainer | SGD, lr 0.07, 20 epochs, L2 | CBraMod's: AdamW 5e-4, 50 epochs, cosine, label smoothing | one trainer for all models |
| Input | SEED's DE, smoothed | DE computed in the model from each raw window, no smoothing | the pipeline feeds signals; smoothing across samples would cross the split |
| Normalisation | 2(x − μ)/μ | per-band standardisation with training statistics | the authors' form needs positive means, which these units do not guarantee |

### Settings we had to choose

| Setting | Value | Why |
|---|---|---|
| Node features for windows longer than 1 s | 5 bands × seconds, 1-s segments | keeps the paper's 1-s DE |
| Bands | δ 1–3, θ 4–7, α 8–13, β 14–30, γ 31–50 Hz | SEED's |
| Model size | K = 2, 64 filters, 256-unit layer, dropout 0.5 | the authors' SEED values, no tuning |
| Initial graph, braindecode | Gaussian kernel on standard-montage positions; TP10 stands in for the ear derivation A2-A1 | braindecode's default needs positions |

### Run configuration

`finetune_main.py` defaults (CBraMod Table 6), as for EEGNet on these splits: 50 epochs, batch 64, AdamW lr 5e-4 for the whole
model (no `backbone` parameters), weight decay 5e-2, cosine to 1e-6, clipping 1, label smoothing 0.1, best validation κ or AUROC.
PyTorch 2.1.2, braindecode 1.8.1 (`env/pip_freeze.txt`), Quadro M6000.

## Reproducing

```bash
mkdir -p results_rerun && python code/collect.py logs/runs results_rerun > results_rerun/results.txt \
  && diff results_rerun/results.txt results/results.txt
for m in dgcnn bd-dgcnn bd-dgcnn-de; do python code/add_jobs.py $m 0 1 2 3 4; done    # training, needs CBraMod's LMDBs
```

## Provenance

Runs: 2026-11-06 on mcl-server, `~/repro-seed/runs/dgcnn-splits`.
