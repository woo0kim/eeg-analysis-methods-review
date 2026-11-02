# DeepSleepNet on Sleep-EDF-20: the released code, 20-fold cross-validation

Does DeepSleepNet's released code reproduce the paper's Sleep-EDF Fpz-Cz result (Table IV: accuracy 82.0%,
macro-F1 76.9, κ 0.76) with the paper's protocol? The protocol is 20 subjects and 20 folds, one held-out subject
per fold, with all folds' predictions pooled. Every step ran as released, on mcl-server: `prepare_physionet.py`,
`train.py` for folds 0–19, `predict.py` and `summary.py`.

## Results

| Metric | This run | 95% CI over subjects | Paper (Table IV) | Difference |
|---|---|---|---|---|
| Accuracy | 0.8181 | [0.7839, 0.8479] | 0.820 | −0.0019 |
| Macro-F1 | 0.7672 | [0.7345, 0.7945] | 0.769 | −0.0018 |
| Cohen's κ | 0.7543 | [0.7109, 0.7928] | 0.76 | −0.0057 |

| Per-stage F1 | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|
| This run | 0.8604 | 0.4570 | 0.8518 | 0.8435 | 0.8236 |
| Paper (Table IV) | 0.847 | 0.466 | 0.859 | 0.848 | 0.824 |

**The released code reproduces the paper.** All three overall metrics are within 0.6 points, and per-stage F1 is
within 1.4 points. The test set has 41,950 epochs, the paper's count exactly. The confusion matrix
(`results/results.txt`) has the same row totals as the paper's Table III. Most cells are within a few percent of
the paper's; the largest gaps are REM scored as W (220 here vs. 360 in the paper) and N2 scored as REM (876 vs. 775).

Subjects differ more than the paper suggests. The paper reports no spread; per-subject accuracy ranges from 0.606
(subject 11) and 0.694 (subject 8) to 0.911 (subject 15). L-SeqSleepNet's reproduction also scores subject 11
lowest ([sleepedf20-loso](../../../l-seqsleepnet/experiments/sleepedf20-loso)). The 95% intervals come from 10,000
bootstrap resamples of subjects.

### Why the epoch count matches

`predict.py` classifies each night in sequences of 25 epochs and drops the last n mod 25 epochs. Pooled over the 39
nights, that leaves exactly the 41,950 epochs the paper reports. The paper's class counts (W 7,927 …) are the counts
after this drop, so the released prediction code is the code behind the paper's table.

## Paper vs. this reproduction

### What differs, and why

| | Paper / released code | This experiment | Why |
|---|---|---|---|
| Data | Sleep-EDF (2013 release), SC subjects 0–19, downloaded by `data/download_physionet.sh` | the same 39 recordings (the 39 PSG files that script lists), from Sleep-EDF Expanded 1.0.0 (`datasets/download.sh sleepedf`) | the script's physiobank URLs predate PhysioNet's current layout; the SC4001–SC4192 files are the ones it lists |
| Code | as released | unchanged: the README's commands | |
| Environment | README: Python 3.5.4, tensorflow-gpu 1.15.2, CUDA 10.0, cuDNN 7.6.4 | Python 3.7.12, TensorFlow 1.15.0 (GPU), CUDA 10.0, cuDNN 7.6.5 (conda) | the closest conda build available; the code runs unchanged |
| Hardware | not stated | two folds side by side on one 12 GB Quadro M6000 (memory growth on) | throughput |

### Settings we had to choose

| Setting | Value | Why |
|---|---|---|
| Channel | Fpz-Cz (`--select_ch 'EEG Fpz-Cz'`) | the paper's headline channel |
| Seeds | none (the released code sets none) | as released; one run of the 20 folds |
| Number of runs | 1 | the paper reports one cross-validation |

### Run configuration

`train.py --n_folds 20 --fold_idx k --pretrain_epochs 100 --finetune_epochs 200 --resume False` for k = 0–19. The
released trainers pre-train the two CNNs for 100 epochs on an oversampled training set, then fine-tune the whole
network on 25-epoch sequences for 200 epochs (Adam; lr 1e-4, 1e-6 for the CNNs). The final model is kept, with no
validation-based selection. About 2.5 h per pair of folds, ≈ 25 GPU-hours in total. Environment: conda env `tf115`
with Python 3.7.12, TensorFlow 1.15.0 (GPU; CUDA 10.0, cuDNN 7.6.5), NumPy 1.18.5 and MNE 0.20.0
(`env/conda_list.txt`).

## Layout

```
code/add_jobs.py        enqueue the README's train.py command, two folds per job (tools/run_queue.py)
code/export_preds.py    predict.py's output_subject<i>.npz -> logs/preds/subject<i>.npz (labels, predictions only)
code/collect.py         every number above from logs/preds
code/export_summary.py  results/summary.csv (feeds RESULTS.md)
logs/runs/fold<k>.txt   train.py output per fold
logs/runs/predict.txt, summary.txt, predict_rc.txt   predict.py and summary.py output (summary.py as released)
logs/preds/             subject<i>.npz
results/                results.txt, results.json, summary.csv
env/conda_list.txt      packages of the runs
```

## Reproducing

From the committed predictions (numpy, scikit-learn):

```bash
mkdir -p results_rerun && python code/collect.py logs/preds results_rerun > results_rerun/results.txt \
  && diff results_rerun/results.txt results/results.txt
```

Training (Linux, TF 1.15 env; `DSN_SRC` = the released code, e.g. `../../upstream`):

```bash
cd $DSN_SRC && python prepare_physionet.py --data_dir <39 SC files> --output_dir $DSN_DATA --select_ch 'EEG Fpz-Cz'
python code/add_jobs.py $(seq 0 19)                     # 10 jobs
RUN_QUEUE_DIR=~/repro-sleepedf/queue python ../../../../tools/run_queue.py 0 1 2 3 4 5 6 7
cd $DSN_SRC && python predict.py --data_dir $DSN_DATA --model_dir $DSN_WORK --output_dir $DSN_WORK/predict \
  && python summary.py --data_dir $DSN_WORK/predict
python code/export_preds.py $DSN_WORK/predict logs/preds
```

## Provenance

Runs: 2026-10-30 on mcl-server (`~/repro-sleepmodels/runs/dsn_sc20`), released code at `eb72f9b`.
