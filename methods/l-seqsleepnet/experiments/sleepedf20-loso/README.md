# L-SeqSleepNet on Sleep-EDF-20: the released code and data, leave-one-subject-out

Do L-SeqSleepNet's released code and its released Sleep-EDF-20 data reproduce the paper's random-initialisation
result (Table III: accuracy 86.3 ± 0.2%, κ 0.813 ± 0.003, MF1 79.3 ± 0.4, mean ± SD over 5 repetitions of LOSO CV)?
One repetition (repeat 1: 20 folds) was run with the released scripts on mcl-server.

## Results

Repeat 1, the 20 folds' predictions pooled and scored as the released `evaluate.py` does (it gives the same numbers:
`logs/evaluate_released.txt`); 46,241 test epochs.

| Metric | This run (repeat 1) | 95% CI over subjects | Paper (mean ± SD, 5 repetitions) | Difference |
|---|---|---|---|---|
| Accuracy | 0.8611 | [0.8294, 0.8893] | 0.863 ± 0.002 | −0.0019 |
| Cohen's κ | 0.8108 | [0.7675, 0.8478] | 0.813 ± 0.003 | −0.0022 |
| Macro-F1 | 0.7944 | [0.7626, 0.8203] | 0.793 ± 0.004 | +0.0014 |
| Sensitivity (macro) | 0.7911 | | 0.788 ± 0.005 | +0.0031 |
| Specificity (macro) | 0.9631 | | 0.963 ± 0.001 | +0.0001 |

| Per-stage F1 | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|
| This run | 0.9207 | 0.4609 | 0.8819 | 0.8623 | 0.8463 |
| Paper | 0.916 | 0.453 | 0.885 | 0.862 | 0.852 |

**The released code reproduces the paper.** Every overall metric is within about one paper SD of the published mean,
and the per-stage F1 scores are within 0.6 points. One repetition cannot reproduce the paper's SD across
repetitions. The spread between subjects is far larger: fold accuracy ranges from 0.670 (fold 12, subject 11) to
0.950 (fold 7). DeepSleepNet's reproduction also scores subject 11 lowest
([sleepedf20-20fold](../../../deepsleepnet/experiments/sleepedf20-20fold)). The 95% intervals come from 10,000
bootstrap resamples of subjects.

The released run scripts differ from the paper in two ways (next section). The run follows the scripts.

## Paper vs. this reproduction

### What differs, and why

| | Paper | Released code | This experiment | Why |
|---|---|---|---|---|
| Data | Sleep-EDF-20 Fpz-Cz, lights-off − 30 min to lights-on + 30 min | `sleepedf-20/mat_30min` (39 nights); the preparation scripts are in a separate repository (xsleepnet, MATLAB) | the released mat files | the data the paper was evaluated on |
| Filtering | not stated | the xsleepnet script band-passes 0.3–40 Hz, but the released mat files hold the raw EDF samples (checked with `../sleepedf78-cbramod-split/code/prepare_split_data.py check`) | released files, unfiltered | run the released data |
| Sequence folding | L = 200 as B = 10 subsequences × K = 20 epochs (Sec. IV-B) | `run_training_repeat*.sh`: `--nsubseq 20 --sub_seq_len 10`, i.e. 20 × 10 | 20 × 10 | run the released script; the paper's Sec. IV-D.3 reports that 20 × 10 performs better than 10 × 20 |
| Stopping | "10 training epochs", early stopping after 50 validations without improvement (Sec. IV-B) | `--max_eval_steps 110`, early stopping off: a fixed 11,000 steps, validating every 100 | as released | run the released script |
| Test list | each fold tested on its held-out subject | `test_lseqsleepnet.py` gets `test_list_n1.txt` for **every** fold, while `evaluate.py` scores fold n against `test_list_n<n>.txt` | fold n tested on `test_list_n<n>.txt` (`[MOD]` in `code/add_jobs.py`) | as released, folds 2–20 would score subject 0, who is in their training data, and `evaluate.py` would compare those predictions with the wrong subject's labels |
| Evaluation batch | n/a | `factor = 20*4`: 640 sequences × 200 epochs per forward pass | 80 sequences (`code/run_released.py --mod_eval_factor=10`) | the released batch runs out of memory on a 12 GB GPU at `--gpu_usage 0.45` (all 12 first attempts failed in their first validation); evaluation runs with dropout off and batch normalisation in inference mode, so predictions do not depend on the batch size |
| Repetitions | 5 | `run_training_repeat1.sh` … `repeat5.sh` (same file lists, unseeded) | repeat 1 | GPU time (≈ 80 GPU-hours per repetition here); the other repetitions differ only by random initialisation |

### Settings we had to choose

| Setting | Value | Why |
|---|---|---|
| Repetition | repeat 1 (released fold lists `file_list_20sub/eeg/*_n<n>.txt`; 4 validation subjects per fold, as Table II) | the released lists of the first repetition |
| Seeds | none (the released code sets none) | run as released |
| GPUs | two folds per GPU (`--gpu_usage 0.45`), as the released scripts pair them | as released |
| Aggregation | `evaluate.py`'s default, `multiplication`: log-probabilities of the up to 200 overlapping windows that contain an epoch are summed | as released; `code/export_preds.py` copies the function |

### Run configuration

Released flags (`run_training_repeat1.sh`): `--dropout_rnn 0.9` (keep probability, i.e. dropout 0.1),
`--sub_seq_len 10 --nsubseq 20 --nfilter 32 --nhidden1 64 --nhidden2 64 --attention_size 64 --dualrnn_blocks 1
--gpu_usage 0.45 --max_eval_steps 110`. `config.py` sets Adam with lr 1e-4, batch 8 sequences, ℓ2 1e-4, and a
validation every 100 steps. The checkpoint is the best validation accuracy: the last improvement came between
validation 15 and 107 (median 38.5). Each run took about 8 h per pair of folds, 0.57 s per training step plus about
2.3 min per validation, so the 20 folds took ≈ 80 GPU-hours. Environment: conda env `tf115`, with Python 3.7.12,
TensorFlow 1.15.0 (GPU; CUDA 10.0, cuDNN 7.6.5), NumPy 1.18.5, scikit-learn 0.22.2, imbalanced-learn 0.6.2 and
hdf5storage 0.1.18 (`env/conda_list.txt`). Hardware: Quadro M6000 12 GB.

## Layout

```
code/add_jobs.py        enqueue the released train + test commands per pair of folds (tools/run_queue.py)
code/run_released.py    runs a released script with the evaluation batch changed ([MOD], see above)
code/export_preds.py    test_ret.mat -> per-epoch predictions (evaluate.py's aggregation, copied)
code/collect.py         every number above from logs/preds
code/export_summary.py  results/summary.csv (feeds RESULTS.md)
logs/runs/n<n>/         train.txt (stdout; per-step lines removed, they are in train_log.txt), train_log.txt
                        (released per-step log), eval_result_log.txt (released per-validation log),
                        current_best.txt, test.txt, test_time.txt
logs/preds/             repeat1_n<n>.npz: y_true, y_pred per fold (small; recompute every metric without TF)
logs/evaluate_released.txt   the released evaluate.py on the 20 test_ret.mat files (--num_repeat 1)
results/                results.txt, results.json, summary.csv
env/conda_list.txt      packages of the runs
```

## Reproducing

From the committed predictions (numpy, scikit-learn):

```bash
mkdir -p results_rerun && python code/collect.py logs/preds results_rerun > results_rerun/results.txt \
  && diff results_rerun/results.txt results/results.txt
```

Training (Linux, TF 1.15 env, one 12 GB GPU per pair of folds):

```bash
python code/add_jobs.py 1 $(seq 1 20)                     # 10 jobs: folds (1,2), (3,4), ...
RUN_QUEUE_DIR=~/repro-sleepedf/queue python ../../../../tools/run_queue.py 0 1 2 3 4 5 6 7
python code/export_preds.py ~/repro-sleepmodels/runs/lseq_sc20 logs/preds 1
cd ../../upstream/sleepedf-20/network/lseqsleepnet && python evaluate.py --out_dir ~/repro-sleepmodels/runs/lseq_sc20/ \
  --out_filename test_ret.mat --datalist_dir ../../file_list_20sub/eeg/ --num_fold 20 --num_repeat 1 --subseqlen 10 --nsubseq 20
```

`LSEQ_SRC` points to the released repository at the pinned commit (`../../upstream`, or a full clone: the submodule
is shallow but holds the data).

## Provenance

Runs: 2026-10-31 to 2026-11-01 on mcl-server (`~/repro-sleepmodels/runs/lseq_sc20`), released code at `06c89f7`.
`code/export_preds.py` copies two functions of the released `evaluate.py` (CC-BY-NC-4.0).
