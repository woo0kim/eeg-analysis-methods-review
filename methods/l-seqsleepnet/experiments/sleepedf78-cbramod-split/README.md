# L-SeqSleepNet on Sleep-EDF-78, on the subject split of the CBraMod-pipeline experiments

How does L-SeqSleepNet, trained with its own released code, compare with the other models on the same 62 / 8 / 8
subjects and the same test epochs? The released training and test scripts run on Sleep-EDF-78 (Sleep Cassette,
153 nights of 78 subjects, Fpz-Cz). They train on the 62 training subjects of
[cbramod/finetune-sleepedf](../../../cbramod/experiments/finetune-sleepedf) (`results/split.json`), select the
checkpoint on its 8 validation subjects, and test once on its 8 test subjects. 3 runs.

The paper reports Sleep-EDF-20 only (reproduced in [sleepedf20-loso](../sleepedf20-loso)). No published number
exists for this split.

## Results

The common test set holds the 20,976 test epochs that every Sleep-EDF model in this repository scores
(`results/test_epochs.json` of the CBraMod experiment). Values are mean ± SD over 3 runs.

| Metric | L-SeqSleepNet | per run |
|---|---|---|
| Balanced accuracy | 0.7558 ± 0.0182 | 0.7435 0.7767 0.7471 |
| Cohen's κ | 0.7725 ± 0.0040 | 0.7682 0.7762 0.7731 |
| Weighted F1 | 0.8314 ± 0.0042 | 0.8273 0.8358 0.8312 |
| Accuracy | 0.8387 ± 0.0024 | 0.8360 0.8403 0.8398 |
| Macro-F1 | 0.7583 ± 0.0096 | 0.7508 0.7692 0.7549 |

Per-stage F1 is W 0.9287, N1 0.4057, N2 0.8467, N3 0.7761 and REM 0.8343. On all 21,320 epochs of the 16 test
nights, κ is 0.7715 ± 0.0037 (`results/results.txt`).

On the common test set, against the other models (`python tools/compare_sleepedf.py`; Welch's t-test over runs):

- **L-SeqSleepNet ties the two foundation models.** In κ it is −0.0002 against LaBraM-Base (p = 0.96) and
  +0.0049 against CBraMod (p = 0.40). The two foundation models run in CBraMod's sequence pipeline with pre-trained
  encoders.
- L-SeqSleepNet has the best REM F1 of any model: 0.834, where the others score 0.785 or less. Its N3 F1 (0.776) is
  below the foundation models' (0.80).
- **It beats the released DeepSleepNet** on the same split
  ([deepsleepnet/sleepedf78-cbramod-split](../../../deepsleepnet/experiments/sleepedf78-cbramod-split)) by +0.0695
  κ (p = 0.003) and +0.0443 macro-F1 (p = 0.009).
- **κ is lower here than on Sleep-EDF-20** (0.811 there). The test subjects differ in number and age: Sleep-EDF-78
  goes up to age 101, and the stratified test set spans ages 27–97.

The checkpoint was chosen early: the best validation accuracy came at validation 14, 22 and 13 of 110 (steps
5,600, 8,800 and 5,200 of 44,000). Validation never improved after that.

## Paper vs. this reproduction

### What differs, and why

| | Paper / released code | This experiment | Why |
|---|---|---|---|
| Data | Sleep-EDF-20 (39 nights), released mat files; preparation in xsleepnet (MATLAB) | all 153 Sleep-EDF-78 nights, prepared by `code/prepare_split_data.py`: a Python port of the released preparation (raw Fpz-Cz at 100 Hz, 29 × 129 log spectrogram with a 2-s Hamming window, 50% overlap and a 256-point FFT; MATLAB v7.3 files) | the released mat files cover 20 subjects. The port matches the released files of two nights: raw signal to 7e-6 µV, spectrogram exactly (`prepare_split_data.py check`) |
| Epochs kept | lights-off − 30 min to lights-on + 30 min | the epochs of the CBraMod-pipeline experiments (sleep period ± 30 min, scored epochs) | every model scored on the same test epochs |
| Split | LOSO, 4 validation subjects per fold | 62 train / 8 validation / 8 test subjects (`split.json`, age-stratified), one model per run | the comparison split |
| Training length | 11,000 steps (`--max_eval_steps 110`, a validation every 100 steps) | 44,000 steps, a validation every 400 (`code/run_released.py --mod_evaluate_every=400`, `--max_eval_steps 110`) | the training set is 3.9 times Sleep-EDF-20's, so 4 times the steps keeps the number of passes over the data. Validating every 400 steps keeps the released 110 validations: one validation scores every stride-1 200-epoch window of 16 nights, and 440 of them would add about 30 h per run |
| Evaluation batch | 640 sequences | 80 (`--mod_eval_factor=10`) | out of memory on 12 GB. Prediction-neutral: inference mode, see [sleepedf20-loso](../sleepedf20-loso) |
| Runs | 5 repetitions of LOSO | 3 runs on one split | 3 is the minimum for an SD. The released code sets no seed, so runs differ by initialisation and data order |

### Settings we had to choose

| Setting | Value | Why |
|---|---|---|
| Flags | `run_training_repeat1.sh`'s: 20 subsequences × 10 epochs, dropout 0.1, etc. | as released |
| Checkpoint | best validation accuracy (released) on the 8 validation subjects | as released |
| GPUs | runs 0 and 1 share one GPU (`--gpu_usage 0.45`), run 2 has one | as the released scripts pair folds |
| Common test set | test epochs inside complete 20-epoch sequences of the CBraMod pipeline, minus each night's last n mod 25 epochs that DeepSleepNet's predict.py drops | the epochs every model scores (`epoch_keys.py` of the CBraMod experiment) |

### Run configuration

Released flags: `--dropout_rnn 0.9 --sub_seq_len 10 --nsubseq 20 --nfilter 32 --nhidden1 64 --nhidden2 64
--attention_size 64 --dualrnn_blocks 1 --gpu_usage 0.45 --max_eval_steps 110`, plus Adam with lr 1e-4, batch 8 and
ℓ2 1e-4 (`config.py`). With the two `[MOD]`s above, each run took 19–20 h. Environment: conda env `tf115`
(TensorFlow 1.15.0 GPU, Python 3.7.12; `env/conda_list.txt`). Hardware: Quadro M6000 12 GB.

## Layout

```
code/prepare_split_data.py   153 nights -> mat files, train/eval/test lists, keys.json (needs MNE, hdf5storage)
code/add_jobs.py             enqueue the released train + test commands (tools/run_queue.py)
code/collect.py              export: test_ret.mat -> per-epoch predictions; collect: every number above
code/export_summary.py       results/summary.csv (feeds RESULTS.md)
logs/runs/run<r>/            train.txt (per-step lines removed; they are in train_log.txt), train_log.txt,
                             eval_result_log.txt, current_best.txt, test.txt, test_time.txt
logs/preds/                  run<r>.npz (y_true, y_pred, nights), keys.json (absolute epoch indices per night)
results/                     results.txt, results.json, summary.csv
env/conda_list.txt           packages of the runs
```

`code/run_released.py` and `code/export_preds.py` are in [sleepedf20-loso](../sleepedf20-loso/code).

## Reproducing

From the committed predictions (numpy, scikit-learn; the common test set comes from the CBraMod experiment):

```bash
mkdir -p results_rerun && python code/collect.py logs/preds results_rerun > results_rerun/results.txt \
  && diff results_rerun/results.txt results/results.txt
```

Training (Linux, TF 1.15 env):

```bash
python code/prepare_split_data.py                          # -> $LSEQ_DATA78 (~/repro-sleepmodels/data/lseq_sc78)
python code/add_jobs.py 0 1 2                              # 2 jobs: runs (0, 1) and (2)
RUN_QUEUE_DIR=~/repro-sleepedf/queue python ../../../../tools/run_queue.py 0 1 2 3 4 5 6 7
python code/collect.py export ~/repro-sleepmodels/runs/lseq_sc78 logs/preds && cp $LSEQ_DATA78/keys.json logs/preds/
```

## Provenance

Runs: 2026-10-31 to 2026-11-01 on mcl-server (`~/repro-sleepmodels/runs/lseq_sc78`), released code at `06c89f7`.
