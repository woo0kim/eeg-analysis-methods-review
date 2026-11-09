# EEG Conformer on SEED: the released 5-fold code, and the same model on held-out clips

The EEG Conformer README reports 95.30% on SEED (5-fold). This experiment runs the released `conformer_seed_1s_5fold.py` and
asks what that number measures. The script takes session 1 of each subject, cuts each 4-minute clip into 1-s windows and
makes 5 folds **inside each clip**, so the test windows of a clip sit next to training windows of the same clip. The
second protocol keeps everything the same and holds out whole clips.

## Results

Session 1 of 15 subjects, accuracy % (SD over subjects). 75 runs inside clips (15 subjects × 5 folds), 45 on held-out clips
(15 subjects × 3 seeds).

| Epoch rule | 5 folds inside clips | First 9 clips → last 6 | Difference | p |
|---|---|---|---|---|
| Best test epoch of 600 (released) | **98.24 (1.06)** | 79.12 (11.91) | +19.12 | 0.0001 |
| Mean of all epochs (released second number) | 96.10 (1.82) | 71.36 (12.20) | +24.74 | 0.0001 |
| Mean of the last 100 epochs | 96.86 (1.59) | 71.70 (12.32) | +25.16 | 0.0001 |
| Final epoch | **96.82 (1.71)** | **71.59 (12.54)** | +25.23 | 0.0001 |

- **The released code reproduces the number and exceeds it:** 98.24% under its rule against 95.30%.
- **Holding out whole clips costs 25 points.** The same model, session and training drop from 96.8% to 71.6% at the final
  epoch (two-sided Wilcoxon over the 15 subjects). In the 5-fold split a window's neighbours, one second away in the same
  clip of the same film, are in the training set, so the split measures recognising a clip more than decoding an emotion.
  SEED's labels belong to clips, and the literature's standard split (first 9 clips → last 6) keeps them apart.
- **Epoch selection adds little inside clips (1.4 pp) and much more between clips (7.5 pp).**
- On a split between clips, Conformer on one session's raw signal (71.6) is close to DGCNN and SVM on DE features over three
  sessions (77.8, 79.4; [dgcnn/seed-paper-protocol](../../../dgcnn/experiments/seed-paper-protocol)); these differ in sessions,
  so the comparison is rough.

## Paper vs. this reproduction

### What differs, and why

| | Released code | This experiment | Why |
|---|---|---|---|
| Preprocessing | `preprocessing/seed.m` (MATLAB) and `seed_process_slice_cv.py` | the same steps in Python (`code/prepare_seed.py`): per-clip z-score, Chebyshev II 4–47 Hz, filtfilt, 1-s windows | no MATLAB on the server; `filtfilt` padding differs slightly at clip edges |
| Paths, main loop | hard-coded, loops over subjects and folds | one subject and fold per process, marked `# [MOD]` in `code/conformer_seed_1s_5fold.py` | to run through the queue |
| Test pass | no `no_grad` | under `no_grad` | 6.4 GB for 1,384 windows; memory only, scores unchanged |
| Split | 5 folds inside clips | that, and `--split trial96` | the question of this experiment |
| Curves | written to a log | also saved per run (`results/curves`) | every number recomputes without a GPU |

### Settings we had to choose

| Setting | Value | Why |
|---|---|---|
| Seeds | fixed per subject and fold (5-fold), 0–2 (held-out clips) | the released code draws a random one |
| Held-out split | first 9 clips → last 6, session 1 | the literature's split, and DGCNN's paper's; the session is the released script's |

### Run configuration

Released values: 600 epochs, batch 200, Adam lr 2e-4, β 0.5 / 0.999, 5 attention heads (the BCI script uses 10), 6 blocks,
head Linear(280, 32) + ELU + Dropout 0.3 + Linear(32, 3), training-set standardisation. PyTorch 2.1.2 on Quadro M6000s
(`env/pip_freeze.txt`), 2 jobs per GPU: about 39 minutes per 5-fold run, 23 per held-out run.

## Reproducing

```bash
python code/analyze.py <dir with kfold_s*/curves, trial96_s*/curves> results_rerun     # layout of the runs, see add_jobs.py
python code/prepare_seed.py $EEG_DATA_ROOT/SEED/SEED_EEG/Preprocessed_EEG ~/repro-seed/conformer_data
python code/add_jobs.py && RUN_QUEUE_DIR=~/repro-seed/queue python ../../../../tools/run_queue.py 0 1 2 3
```

`results/curves` holds each run's 600 test accuracies; `results/results.json` every number above.

## Provenance

Runs: 2026-11-05 and 06 on mcl-server, `~/repro-seed/runs/conformer`. `code/conformer_seed_1s_5fold.py` copies GPL-3.0 code and is
GPL-3.0.
