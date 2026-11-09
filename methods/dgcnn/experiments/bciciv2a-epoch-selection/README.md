# DGCNN on BCI-IV-2a, within subject

DGCNN through EEG Conformer's released training loop, as [eegnet/bciciv2a-epoch-selection](../../../eegnet/experiments/bciciv2a-epoch-selection)
did for EEGNet: `code/run_dgcnn.py` rebinds the model and nothing else, so data, S&R augmentation, optimizer, 2000 epochs, seeds
and per-epoch scoring are Conformer's. 9 subjects × 3 seeds × 2 protocols: 54 runs.

## Results

Accuracy %, mean over subjects of the mean over seeds 2023–2025; two-sided Wilcoxon over the 9 subjects.

| Epoch rule | DGCNN | EEG Conformer | EEGNet | Conformer − DGCNN (p) | EEGNet − DGCNN (p) |
|---|---|---|---|---|---|
| Best test epoch of 2000 (Conformer's released rule) | 50.40 | 78.72 | 72.62 | +28.32 (0.004) | +22.22 (0.004) |
| Mean of the last 100 epochs | 47.19 | 68.77 | 69.53 | +21.58 (0.004) | +22.34 (0.004) |
| Final epoch | 47.17 | 69.21 | 69.35 | +22.04 (0.004) | +22.18 (0.004) |
| Validation-selected epoch | 44.42 | 70.01 | 67.36 | +25.59 (0.004) | +22.94 (0.004) |

DGCNN on band power is more than 20 points below both raw-signal models under every rule (chance is 25%). Motor imagery depends
on the timing of a band's change within a trial, which DE over 1-s segments keeps only coarsely. DGCNN was not built for this
task and its paper does not report it.

## Paper vs. this reproduction

Not a reproduction: DGCNN's paper has no motor-imagery result. Differences from the paper are those of
[cbramod-splits](../cbramod-splits), with Conformer's training loop in place of CBraMod's.

| Setting we chose | Value | Why |
|---|---|---|
| Node features | 5-band DE of the 4 consecutive 1-s segments of a trial at 250 Hz: 20 per channel | keeps the paper's 1-s DE |
| Normalisation | per-band standardisation from the first training batch | that loop has no hook for a pass over the data |
| Model | K = 2, 64 filters, 256-unit layer, dropout 0.5 | the authors' SEED values, no tuning |
| Seeds, protocols | 2023–2025; A: all of session T, B: 20% held out | as the Conformer and EEGNet experiments |

Hardware and packages: Quadro M6000, PyTorch 2.1.2 (`env/pip_freeze.txt`); 2000 epochs in about 13 minutes per run on a GPU shared by 2 jobs.

## Reproducing

```bash
python code/analyze.py results                                  # from the committed curves
python code/add_jobs.py                                         # training; DATA_2A is the Conformer experiment's data_proc/
```

## Provenance

Runs: 2026-11-06 on mcl-server, `~/repro-seed/runs/dgcnn-2a`. `code/run_dgcnn.py` imports GPL-3.0 code and is GPL-3.0.
