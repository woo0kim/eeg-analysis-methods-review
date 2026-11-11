# DGCNN on SEED, in the paper's two protocols

Does DGCNN reach the numbers of

> T. Song, W. Zheng, P. Song, Z. Cui, *EEG Emotion Recognition Using Dynamical Graph Convolutional Neural Networks*,
> IEEE Transactions on Affective Computing 11(3), 2020

on SEED, and how are those numbers computed? The paper's abstract gives 90.40% subject-dependent and 79.95%
subject-independent (SDs 8.49 and 9.02 as other papers quote them).

**What was run.** Our PyTorch implementation (`../../code/dgcnn.py`), written from the authors' TensorFlow code, with their
settings; all 45 sessions train in about 3 minutes. Then the authors' `eeg_DE.py` itself, unmodified (TensorFlow 1.15,
`code/run_authors.sh`), on the same features: it gives the same numbers as the port (87.59% vs. 87.69% under its own rule), so
the gap to the paper's 90.40% is not in the port ([below](#is-the-implementation-right)).

## Results

Subject-dependent: each of the 45 sessions alone, first 9 clips → last 6, `de_LDS` features, 5 runs, accuracy % (SD).

| Checkpoint | Best 2 of each subject's 3 sessions | All 3 sessions |
|---|---|---|
| Peak of ~80 test evaluations (**the authors' rule, with the left column**) | **87.69 (9.06)** | 82.97 (11.11) |
| Last step, lr 0.07 (authors') | 81.39 (11.90) | 74.62 (16.05) |
| Last step, lr 0.01 (picked on clips 8–9) | 83.12 | **77.79** |
| Step picked on clips 8–9, lr 0.07, trained on 7 clips | 78.35 (10.65) | 71.23 (15.59) |

Paper: 90.40 (8.49). Its number is the top-left cell's rule (below).

- **The published figure selects twice on the test set.** The authors' `eeg_DE.py` sets `val_data = test_data`, keeps the
  highest test accuracy over about 80 evaluations during training, and then averages only each subject's two best
  sessions of three (`sorted(accuracy[i*3:3*(i+1),1])`, entries 1 and 2). With neither choice the same runs give 74.62%, or
  77.79% at a learning rate picked without the test set.
- **The authors' own code falls 2.7 pp short of the paper too** (87.59 vs. 90.40; 86.90–88.22 over 5 runs), level with our
  port (87.69; p = 0.75 over runs, 0.66 over the 45 sessions). The gap is therefore not in the port: it lies in what the
  archive does not hold, most likely the DE files they trained on (their computation, smoothing or normalisation).
- **On these features a linear SVM ties DGCNN within subject.** With nothing chosen on the test set (Wilcoxon over the 15
  subjects): DGCNN 77.79, SVM 79.42, p = 0.33. Under the authors' rule the SVM gives 87.89 (`code/svm_baseline.py`). The
  paper's advantage over SVM (its literature value, 83.99) does not appear on identical data and selection.
- **Across subjects DGCNN is far ahead.** Leave one subject out, one session at a time, last step, nothing chosen on the
  test set: DGCNN **77.58 (SD 5.14)**, SVM 58.85 (9.86), p = 0.0001. That is within 2.4 pp of the paper's 79.95 with no test
  selection, and the SVM is close to the 56.73 (16.29) quoted for SVM in the literature. Both models z-score each session
  on its own, which uses the test subject's unlabeled data.
- **The smoothing matters more than the rule.** `de_LDS` beats `de_movingAve` by 7.4 pp under the authors' rule (87.69 vs.
  80.32) for DGCNN and by 11 pp for the SVM. Both come with the dataset; the authors' archive does not say which they used.

Leave one subject out, mean over the 15 test subjects (SD over subjects), 3 runs:

| Data | Features | Normalisation, lr | Peak on test | Last step |
|---|---|---|---|---|
| one session at a time | `de_LDS` | each session z-scored, 0.01 | 82.47 (5.77) | **77.58 (5.14)** |
| | `de_LDS` | each session z-scored, 0.07 | 84.15 (4.73) | 67.89 (11.17) |
| | `de_LDS` | authors' 2(x − μ)/μ, 0.07 | 70.77 (8.94) | 46.26 (10.10) |
| | `de_movingAve` | each session z-scored, 0.07 | 69.76 (5.82) | 57.48 (6.77) |
| | `de_movingAve` | authors', 0.07 | 64.55 (6.79) | 40.63 (5.47) |
| all 3 sessions pooled | `de_LDS` | each session z-scored, 0.07 | 84.47 (4.80) | 60.77 (22.37) |
| | `de_LDS` | authors', 0.07 | 69.07 (8.05) | 47.54 (14.70) |

Learning rates, `de_LDS`, all 3 sessions (best 2 of 3 in parentheses):

| lr | Validation clips, last step | Test, last step | Test, peak |
|---|---|---|---|
| 0.07 (authors') | 67.67 | 74.62 (81.39) | 82.97 (87.69) |
| 0.03 | 71.29 | 76.87 (82.74) | 83.63 (88.61) |
| **0.01 (picked)** | 71.67 | 77.79 (83.12) | 83.84 (88.56) |
| 0.003 | 64.30 | 76.53 (81.53) | 82.09 (86.37) |
| 0.001 | 44.85 | 73.38 (78.63) | 79.34 (84.15) |

`results/results.txt` has every table, including `de_movingAve`.

## Is the implementation right?

`code/check_equivalence.py` copies a braindecode 1.8.1 `DGCNN` into ours (braindecode's cites the same archive): identical
parameter counts and outputs to 1.2e-07 for Chebyshev orders 2, 3 and 4. Other public implementations differ from the
authors' code in ways that change the model: torcheeg's `DGCNN` uses powers of the adjacency matrix, not Chebyshev
polynomials of a Laplacian, with no dropout; xueyunlong12589/DGCNN and xujiayang530/DGCNN add BatchNorm
and drop the 256-unit layer.

**The authors' code, unmodified** (`code/run_authors.sh`, TensorFlow 1.15, 5 runs per feature; `code/analyze_authors.py`,
`results/authors.txt`). Accuracy %, mean over runs (SD over the 30 or 45 session values, as `eeg_DE.py` prints it); authors'
code − port with Welch's t-test over runs and a two-sided Wilcoxon test over the 45 sessions:

| Features | Rule | Authors' code | Our port | Difference | p (runs / sessions) |
|---|---|---|---|---|---|
| `de_LDS` | peak on test, best 2 of 3 sessions (**the authors' rule**) | **87.59** (9.00) | 87.69 (9.06) | −0.10 | 0.75 / 0.66 |
| | peak on test, all sessions | 82.87 (11.36) | 82.97 (11.11) | −0.10 | 0.77 / 0.66 |
| | last step, best 2 of 3 sessions | 81.45 (11.96) | 81.39 (11.90) | +0.07 | 0.90 / 0.75 |
| | last step, all sessions | 75.05 (15.55) | 74.62 (16.05) | +0.43 | 0.63 / 0.75 |
| `de_movingAve` | peak on test, best 2 of 3 sessions | 80.11 (8.85) | 80.32 (8.72) | −0.22 | 0.15 / 0.018 |
| | peak on test, all sessions | 75.62 (10.82) | 76.12 (10.65) | −0.49 | 0.065 / 0.018 |
| | last step, best 2 of 3 sessions | 71.31 (13.94) | 74.03 (11.42) | −2.72 | 0.13 / 0.036 |
| | last step, all sessions | 64.21 (17.43) | 67.13 (15.94) | −2.91 | 0.21 / 0.036 |

On `de_LDS` the two implementations agree on every rule. On `de_movingAve` the authors' code is 0.2–0.5 lower at the peak and
about 2.8 lower at the last step, where its runs also vary more (SD over runs 3.1–4.0 vs. 1.7–2.5): learning rate 0.07 makes
some sessions collapse late in training, and the two frameworks' random initialisations differ. Neither implementation reaches
90.40 on either feature; the best single authors' run gives 88.22.

## Paper vs. this reproduction

The paper is closed access and IEEE refuses scripted downloads, so it was not read in full. Paper numbers here are the
abstract's; the settings are the authors' code, and the paper's equations are as quoted in braindecode's `dgcnn.py`.

### What differs, and why

| | Paper | Authors' code | This experiment | Why |
|---|---|---|---|---|
| Implementation | – | TensorFlow 1.x, modified cnn_graph (`EEG-TAC.zip`) | PyTorch port; the authors' code also run unmodified | the archive has no license: fetched by script, not committed |
| Input | DE features | `DE_<i>.mat`, not in the archive | SEED's distributed `de_LDS` and `de_movingAve` (`code/prepare_de.py`) | shapes match exactly: 3,394 samples per session, 2,010 in the first 9 clips |
| Initial graph | spatial (Eq. 1, per braindecode's quote) | random, std 0.3 off the diagonal and 0.6 on it, the spatial `adjacency()` unused | as the code | run the code; the paper and code disagree |
| Validation set | – | the test set | as the code for the "authors' rule" rows; clips 8–9 elsewhere | to measure the effect |
| Reported number | 90.40 | peak over test evaluations, best 2 of 3 sessions | that, and the alternatives above | to measure the effect |
| Subject-independent code | LOSO | none in the archive | `code/run_port.py --protocol loso` | the settings below are ours |

### Settings we had to choose

| Setting | Value | Why |
|---|---|---|
| DE smoothing | both | not recorded by the authors |
| Runs | 5 (subject-dependent), 3 (LOSO), seeds 0–4 | the code sets no seed |
| LOSO data | one session index at a time; also pooled | not recorded |
| LOSO normalisation | the code's 2(x − μ)/μ from training data; each session z-scored | not recorded; z-scoring works far better |
| Learning-rate check | 0.07 to 0.001, picked on clips 8–9 trained on 1–7 | 0.07 is unstable: some sessions collapse |
| SVM | C from 2^−10 to 2^10 (SD), 2^−14 to 2^−4 (LOSO, picked on an inner subject); per-band standardisation | as Zheng & Lu (2015) for the range; no test selection |

### Run configuration

`eeg_DE.py`: one Chebyshev layer (K = 2, 64 filters), bias + ReLU, 256-unit layer with dropout 0.5, linear output. SGD with
momentum 0.9, lr 0.07 × 0.95 every 3 epochs (staircase), batch 100, 20 epochs, L2 5e-4 on everything but the filter bias,
per-band normalisation 2(x − μ)/μ with training means, evaluation every 5 steps. PyTorch 2.1.2, one Quadro M6000 shared by
2 jobs (`env/pip_freeze.txt`). The authors' code: TensorFlow 1.15.0, Python 3.7 (`env/authors_tf115_conda_list.txt`), one run of
45 sessions per Quadro M6000, about 5 minutes.

## Reproducing

```bash
python code/analyze.py logs/runs results_rerun > /dev/null      # every table, from the committed curves (numpy, scipy)
python code/check_equivalence.py                                # needs torch, braindecode 1.8.1, mne
# training: SEED's ExtractedFeatures_1s (license-gated, datasets/README.md)
python code/prepare_de.py $EEG_DATA_ROOT/SEED/SEED_EEG/ExtractedFeatures_1s ~/repro-seed/de/de_LDS de_LDS
python code/prepare_de.py $EEG_DATA_ROOT/SEED/SEED_EEG/ExtractedFeatures_1s ~/repro-seed/de/de_movingAve de_movingAve
python code/add_jobs.py && python code/add_jobs.py lr && python code/add_jobs.py loso-lr
RUN_QUEUE_DIR=~/repro-seed/queue python ../../../../tools/run_queue.py 0 1 2 3
python code/svm_baseline.py ~/repro-seed/de/de_LDS results_rerun/svm_de_LDS.json
python code/svm_loso.py ~/repro-seed/de/de_LDS results_rerun/svm_loso.json
# the authors' code, unmodified: fetch it (pinned SHA-256), then one run per call (TensorFlow 1.15 env)
bash ../../fetch_upstream.sh
bash code/run_authors.sh <gpu> de_LDS <run> ~/repro-seed/dgcnn_authors ~/repro-seed/de ../../upstream/EEG-TAC <tf1.15 python>
python code/analyze_authors.py logs/authors logs/runs results_rerun          # from the committed accuracy.mat files
```

`logs/authors/authors__<features>__r<k>/` holds each authors' run's `accuracy.mat` (per session: last-step and peak test
accuracy) and a trimmed log ending in `eeg_DE.py`'s own printed summary.

## Provenance

Runs: 2026-11-05 and 06 on mcl-server, `~/repro-seed/runs/port`. The authors' code is the `[code]` link beside the paper on
Tengfei Song's page, now offline, read from the Wayback Machine's capture of 2022-11-22 (`../../fetch_upstream.sh`, pinned by
SHA-256). Its `eeg_DE.py` was run unmodified on 2026-11-09, with the repository owner's approval, on mcl-server
(`~/repro-seed/dgcnn_authors`), after checking the sources: no shell, network or out-of-folder file operations; it deletes and
rewrites only its own `summaries/` and `checkpoints/`, which is why `code/run_authors.sh` gives every run its own copy. The
script ends in `pdb.set_trace()`, which exits on the closed input after everything is saved and printed.
