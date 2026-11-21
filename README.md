# EEG analysis methods review

Independent validation of published EEG decoding methods. Each method is run from its authors' released
code (pinned as a submodule) on public datasets, and the reproduced numbers are compared with the
paper's. Every method is also re-run with its [braindecode](methods/eegnet/upstream) implementation, swapped into
the same pipeline, to check whether the library's version reproduces the released code. Every experiment keeps enough
raw output (training logs, per-epoch curves) to recompute its numbers without a GPU.

**[RESULTS.md](RESULTS.md)** collects every reproduced number by method and dataset.
**[SUMMARY.md](SUMMARY.md)** summarizes them: results by task category and dataset (domain-specific vs. general
models), methods grouped by characteristics, each method's results across datasets and the patterns they show (e.g. spectral-input
models lead on depression and workload but trail on motor imagery), model size and FLOPs (`tools/count_flops.py`), datasets,
baselines, and all models compared on identical subject-independent splits: CBraMod's four public datasets, sleep staging on
Sleep-EDF, emotion recognition on SEED and SEED-V, and mental workload on STEW, every model through one pipeline per dataset.

## Methods

**Scope: subject-independent protocols only (since 2026-11-19).** Every reported number comes from test subjects that no part of
training saw; the subject-dependent experiments stay in `methods/`, marked in each method's README, but are no longer reported.

| Method | Paper | Experiments (subject-independent) | Finding so far |
|---|---|---|---|
| [EEG Conformer](methods/eeg-conformer) | Song et al., IEEE TNSRE 2023 | CBraMod's 4 datasets: [cbramod-splits](methods/eeg-conformer/experiments/cbramod-splits); Sleep-EDF: [sleepedf-cbramod-pipeline](methods/eeg-conformer/experiments/sleepedf-cbramod-pipeline), [braindecode-sleepedf](methods/eeg-conformer/experiments/braindecode-sleepedf); SEED / SEED-V subject split: [seed-cbramod-pipeline](methods/eeg-conformer/experiments/seed-cbramod-pipeline); STEW: [stew-cbramod-pipeline](methods/eeg-conformer/experiments/stew-cbramod-pipeline), [braindecode-stew](methods/eeg-conformer/experiments/braindecode-stew) | Its own paper is within-subject only. Across subjects the CBraMod paper's EEG Conformer rows do not reproduce, in either direction: above on BCI-IV-2a (0.504 vs. 0.470), below on the other three. Level with EEGNet on BCI-IV-2a, in the bottom three elsewhere; κ 0.688 on Sleep-EDF. braindecode's `EEGConformer` is within noise of the released model. |
| [CBraMod](methods/cbramod) | Wang et al., ICLR 2025 | [finetune-public-datasets](methods/cbramod/experiments/finetune-public-datasets), [braindecode-public-datasets](methods/cbramod/experiments/braindecode-public-datasets), Sleep-EDF: [finetune-sleepedf](methods/cbramod/experiments/finetune-sleepedf), [braindecode-sleepedf](methods/cbramod/experiments/braindecode-sleepedf), SEED / SEED-V subject split: [finetune-seed](methods/cbramod/experiments/finetune-seed), STEW: [finetune-stew](methods/cbramod/experiments/finetune-stew), [braindecode-stew](methods/cbramod/experiments/braindecode-stew) | Released code and weights fall short of the paper on all four public datasets tested. On Sleep-EDF pre-training adds 0.023 κ and CBraMod ties LaBraM-Base, L-SeqSleepNet and ATCNet (κ 0.768–0.773). braindecode's `CBraMod` has the released encoder and weights but a different task head: within noise of the released code on three datasets, 0.02 lower on PhysioNet-MI. Its paper's SEED-V protocol is the subject-dependent trial split (not reported); across subjects SEED-V is at chance for every model. |
| [LaBraM](methods/labram) | Jiang et al., ICLR 2024 | [finetune-cbramod-splits](methods/labram/experiments/finetune-cbramod-splits), [braindecode-cbramod-splits](methods/labram/experiments/braindecode-cbramod-splits), Sleep-EDF: [sleepedf-cbramod-pipeline](methods/labram/experiments/sleepedf-cbramod-pipeline), [braindecode-sleepedf](methods/labram/experiments/braindecode-sleepedf), SEED / SEED-V: [seed-cbramod-pipeline](methods/labram/experiments/seed-cbramod-pipeline), STEW: [stew-cbramod-pipeline](methods/labram/experiments/stew-cbramod-pipeline), [braindecode-stew](methods/labram/experiments/braindecode-stew) | CBraMod's strongest baseline, run under CBraMod's splits, seeds and scoring: it matches the CBraMod paper's LaBraM numbers on 10 of 12 metrics. braindecode's `Labram` is the same model and reproduces these runs. The best model across subjects on SEED (0.582) and on STEW (0.818), tied first on Sleep-EDF (κ 0.773); sixth of eight on BCI-IV-2a. |
| [ATCNet](methods/atcnet) | Altaheri et al., IEEE TII 2023 | BCI-IV-2a leave one subject out: [bciciv2a-model-selection](methods/atcnet/experiments/bciciv2a-model-selection), [braindecode-bciciv2a](methods/atcnet/experiments/braindecode-bciciv2a); CBraMod's 4 datasets: [braindecode-cbramod-splits](methods/atcnet/experiments/braindecode-cbramod-splits); Sleep-EDF: [braindecode-sleepedf](methods/atcnet/experiments/braindecode-sleepedf); SEED / SEED-V and STEW: [braindecode-seed](methods/atcnet/experiments/braindecode-seed), [braindecode-stew](methods/atcnet/experiments/braindecode-stew) | Its LOSO number (70.97%) is checkpointed on the held-out subject: the code gives 69.06% that way and 60.23% with a validation split. In the shared pipelines braindecode's 114K-parameter `ATCNet` is level with the foundation models: second on PhysioNet-MI (0.614), first by mean on MentalArithmetic (0.695), κ 0.771 on Sleep-EDF; on STEW mid-table. Its final max-norm is taken over another axis than the Keras model's. |
| [EEGNet](methods/eegnet) (baseline) | Lawhern et al., J. Neural Eng. 2018 | our PyTorch implementation: [pytorch-cbramod-splits](methods/eegnet/experiments/pytorch-cbramod-splits), Sleep-EDF: [sleepedf-cbramod-pipeline](methods/eegnet/experiments/sleepedf-cbramod-pipeline), SEED / SEED-V: [seed-cbramod-pipeline](methods/eegnet/experiments/seed-cbramod-pipeline), STEW: [stew-cbramod-pipeline](methods/eegnet/experiments/stew-cbramod-pipeline); braindecode: [braindecode-cbramod-splits](methods/eegnet/experiments/braindecode-cbramod-splits), [braindecode-sleepedf](methods/eegnet/experiments/braindecode-sleepedf), [braindecode-seed](methods/eegnet/experiments/braindecode-seed), [braindecode-stew](methods/eegnet/experiments/braindecode-stew) | Cross-subject, in CBraMod's pipeline, the best model by mean on BCI-IV-2a (0.529), above both foundation models (p ≤ 0.021), with either implementation. On Sleep-EDF it is the weakest network (κ 0.681); last on STEW. |
| [DeepSleepNet](methods/deepsleepnet) (sleep) | Supratak et al., IEEE TNSRE 2017 | Sleep-EDF: [sleepedf20-20fold](methods/deepsleepnet/experiments/sleepedf20-20fold), [sleepedf78-cbramod-split](methods/deepsleepnet/experiments/sleepedf78-cbramod-split), [braindecode-sleepedf](methods/deepsleepnet/experiments/braindecode-sleepedf); off-label: [braindecode-cbramod-splits](methods/deepsleepnet/experiments/braindecode-cbramod-splits), [braindecode-seed](methods/deepsleepnet/experiments/braindecode-seed), [braindecode-stew](methods/deepsleepnet/experiments/braindecode-stew) | The released code reproduces the paper on Sleep-EDF-20 (81.8% vs. 82.0%, κ 0.754 vs. 0.76) on exactly its 41,950 epochs. On Sleep-EDF-78 its final-model rule overfits (κ 0.703). braindecode's single-epoch DeepSleepNet in CBraMod's pipeline scores 0.762. Off-label it is level with the best model on BCI-IV-2a (0.525). |
| [L-SeqSleepNet](methods/l-seqsleepnet) (sleep) | Phan et al., IEEE JBHI 2023 | Sleep-EDF: [sleepedf20-loso](methods/l-seqsleepnet/experiments/sleepedf20-loso), [sleepedf78-cbramod-split](methods/l-seqsleepnet/experiments/sleepedf78-cbramod-split); STEW (off-label): [stew-offlabel](methods/l-seqsleepnet/experiments/stew-offlabel) | The released code and data reproduce the paper on Sleep-EDF-20: 86.1% vs. 86.3%, κ 0.811 vs. 0.813, in one of its five repetitions, once the run scripts' test list is fixed. On Sleep-EDF-78 it ties the two foundation models (κ 0.773). Not in braindecode; it cannot run on 4–5-s trials. |
| [DGCNN](methods/dgcnn) | Song et al., IEEE TAC 2020 | SEED leave one subject out: [seed-paper-protocol](methods/dgcnn/experiments/seed-paper-protocol); [cbramod-splits](methods/dgcnn/experiments/cbramod-splits), Sleep-EDF: [sleepedf-cbramod-pipeline](methods/dgcnn/experiments/sleepedf-cbramod-pipeline), SEED / SEED-V: [seed-cbramod-pipeline](methods/dgcnn/experiments/seed-cbramod-pipeline), STEW: [stew-cbramod-pipeline](methods/dgcnn/experiments/stew-cbramod-pipeline); braindecode: [braindecode-seed](methods/dgcnn/experiments/braindecode-seed), [braindecode-stew](methods/dgcnn/experiments/braindecode-stew) | Written from the authors' TensorFlow code (archived; no license, so fetched by script). Across subjects the paper holds: 77.6% against 79.95% (linear SVM 58.9%). On band-power input it is the best model on Mumtaz2016 (0.962, p ≤ 0.003) and on STEW the best without pre-training; on motor imagery near the bottom. |
| [dDTF-GL](methods/ddtf-gl) (no network) | Cao, Li & Kuo, manuscript (USC) | STEW, the paper's subject-independent setting with every network alongside: [stew-paper-protocol](methods/ddtf-gl/experiments/stew-paper-protocol); [stew-cbramod-pipeline](methods/ddtf-gl/experiments/stew-cbramod-pipeline), [cbramod-splits](methods/ddtf-gl/experiments/cbramod-splits), [sleepedf-cbramod-pipeline](methods/ddtf-gl/experiments/sleepedf-cbramod-pipeline), [seed-cbramod-pipeline](methods/ddtf-gl/experiments/seed-cbramod-pipeline); tuned: [feature-tuning](methods/ddtf-gl/experiments/feature-tuning) | dDTF connectivity maps, feature selection and XGBoost. On Safari et al.'s released maps its subject-independent result reproduces exactly (0.774 vs. 0.766); in that setting LaBraM-Base, CBraMod, DGCNN and EEG Conformer score higher. Elsewhere mid-table on STEW, above both foundation models on Mumtaz2016, near chance on motor imagery, last on Sleep-EDF. Tuned on validation data (dDTF-GL+: band power added to or replacing the maps, LDA / SVM projection features) it ties the best model on STEW (0.819) and in the paper's own setting (0.863), is first by mean on MentalArithmetic (0.710), second on SEED (0.547) and reaches κ 0.721 on Sleep-EDF; motor imagery stays far behind the networks. |

## braindecode re-implementations

Seven of the eight models exist in [braindecode](https://github.com/braindecode/braindecode) (v1.8.1, pinned as
`methods/eegnet/upstream`); L-SeqSleepNet does not. Each was swapped into the pipeline of the released-code experiment, with
everything else unchanged (for ATCNet, whose released code is TensorFlow, into a PyTorch port of its training loops), and checked
first against the released model with identical weights (`code/check_equivalence.py` in each experiment):

| Model | braindecode class | Same as the released model? | braindecode vs. released code, same pipeline | Experiment |
|---|---|---|---|---|
| EEG Conformer | `EEGConformer` | same layers; attention divides by √(head dim) = 2 instead of √(embedding) ≈ 6.3, so outputs differ for the same weights | within noise on CBraMod's 4 datasets (all p ≥ 0.35); Sleep-EDF-78: κ 0.701 vs. 0.688 (p = 0.47), N3 F1 higher (0.71 vs. 0.56) | [cbramod-splits](methods/eeg-conformer/experiments/cbramod-splits), [braindecode-sleepedf](methods/eeg-conformer/experiments/braindecode-sleepedf) |
| CBraMod | `CBraMod` | encoder and weights identical; task head different (keeps the pre-training `proj_out`, then one linear layer instead of a 3-layer MLP) | within noise on 3 of 4 datasets; PhysioNet-MI −0.020 fine-tuned, −0.072 from scratch, −0.092 frozen. Sleep-EDF-78, where both feed the released sequence head: κ 0.768 vs. 0.768 (p = 0.99) | [braindecode-public-datasets](methods/cbramod/experiments/braindecode-public-datasets), [braindecode-sleepedf](methods/cbramod/experiments/braindecode-sleepedf) |
| LaBraM-Base | `Labram` | identical (weights and logits) | within noise on all 12 metrics; Sleep-EDF-78: κ 0.777 vs. 0.773 (p = 0.36), encoder features identical | [braindecode-cbramod-splits](methods/labram/experiments/braindecode-cbramod-splits), [braindecode-sleepedf](methods/labram/experiments/braindecode-sleepedf) |
| ATCNet | `ATCNet` | same layers and weights; identical predictions once BatchNorm eps is set to Keras' 1e-3; final-layer max-norm taken per input feature instead of per output unit | leave one subject out: 67.36% vs. 69.12% (−1.8 pp, p = 0.20) | [braindecode-bciciv2a](methods/atcnet/experiments/braindecode-bciciv2a) |
| EEGNet | `EEGNet` | not compared (the released Keras code is not used); same architecture size as our PyTorch EEGNet | within noise of our PyTorch EEGNet on all 12 metrics; Sleep-EDF-78: κ 0.6808 vs. 0.6815 (p = 0.95) | [braindecode-cbramod-splits](methods/eegnet/experiments/braindecode-cbramod-splits), [braindecode-sleepedf](methods/eegnet/experiments/braindecode-sleepedf) |
| DeepSleepNet | `DeepSleepNet` | not the released model: single-epoch (its BiLSTM sees one epoch, where the released model's runs across 25), standard LSTM gates instead of peepholes | not run in the released pipeline (TensorFlow). In CBraMod's sleep pipeline it scores κ 0.762 on Sleep-EDF-78, against 0.703 for the released code with its own training on the same split | [braindecode-sleepedf](methods/deepsleepnet/experiments/braindecode-sleepedf) |
| DGCNN | `DGCNN` | the same function as our port of the authors' code (outputs equal to 1.2e-07 with the same weights); it ships with raw samples as node features and a spatial initial graph | on our DE input within noise of ours on STEW and Mumtaz2016; as shipped, raw input costs 0.08–0.11 on every dataset but motor imagery, where raw is better | [braindecode-stew](methods/dgcnn/experiments/braindecode-stew), [cbramod-splits](methods/dgcnn/experiments/cbramod-splits) |

No braindecode version is systematically closer to its paper than the released code. The model ranking under CBraMod's splits
changes in one place: with both foundation models from braindecode, CBraMod no longer beats LaBraM-Base on PhysioNet-MI
([SUMMARY.md](SUMMARY.md#the-same-comparison-with-braindecodes-implementations)). On Sleep-EDF-78, in CBraMod's sleep-staging
pipeline, all four re-runs are within noise of the originals; there both CBraMods feed the same released sequence head and match
(κ +0.0001), consistent with braindecode's own head causing the PhysioNet-MI gap. On STEW's rest vs. task every braindecode model
is within noise of the original; on its 3 levels braindecode's CBraMod (its own head again) and EEG Conformer are about 0.02 lower.

## Layout

```
.
├── RESULTS.md                      generated index of all results (tools/build_results_index.py)
├── SUMMARY.md                      hand-written summary: results by dataset, methods, datasets, baselines, comparisons
├── datasets/
│   ├── README.md                   dataset registry: source, access, which experiments use it
│   └── download.sh                 shared downloader -> $EEG_DATA_ROOT (default ~/data)
├── methods/
│   └── <method>/
│       ├── README.md               paper, released code / weights, pinned commit, license, experiments
│       ├── upstream/               git submodule: the authors' code at the evaluated commit (never edited)
│       └── experiments/
│           └── <experiment>/
│               ├── README.md       question, results, paper vs. this reproduction, how to reproduce
│               ├── code/ | scripts/
│               ├── env/            environment setup / pip freeze (or requirements.txt)
│               ├── results/        numbers, figures, summary.csv
│               └── logs/           raw training logs
├── tools/
│   ├── run_queue.py                file-based multi-GPU job queue
│   ├── build_results_index.py      methods/*/experiments/*/results/summary.csv -> RESULTS.md
│   ├── compare_methods.py          Welch tests between models run on CBraMod's splits (SUMMARY.md)
│   ├── compare_sleepedf.py         every Sleep-EDF model on the common test set, Welch tests, figure (SUMMARY.md)
│   ├── compare_seed.py             every SEED / SEED-V model, both splits, against the CBraMod paper, Welch tests
│   ├── compare_stew.py             every STEW model, both tasks, Welch tests
│   └── count_flops.py              FLOPs and parameters of every model in every pipeline -> tools/flops.json
└── docs/
    ├── adding-an-experiment.md     conventions, the "Paper vs. this reproduction" section, summary.csv schema
    ├── literature/                 literature investigations (e.g. how review papers report others' results)
    ├── reports/<method>/           weekly reports and slides (YYYY-MM-DD_<method>_weekly-report.docx | -slides.pptx)
    └── templates/                  blank weekly report template
```

Methods are the top level because each one brings its own code, environment and preprocessing.
Datasets are shared through `datasets/` and `$EEG_DATA_ROOT`, and `RESULTS.md` gives the
method × dataset view.

## Quick start

```bash
git clone --recurse-submodules https://github.com/woo0kim/eeg-analysis-methods-review
cd eeg-analysis-methods-review
# already cloned without submodules?  git submodule update --init

# Recompute the committed numbers on a laptop (numpy, scipy, matplotlib)
cd methods/eeg-conformer/experiments/bciciv2a-epoch-selection && python code/analyze.py results && cd -
cd methods/eegnet/experiments/bciciv2a-epoch-selection && python code/analyze.py results && cd -
cd methods/cbramod/experiments/finetune-public-datasets && mkdir -p results_rerun \
  && python scripts/collect.py logs/runs results_rerun > results_rerun/results.txt \
  && diff results_rerun/results.txt results/results.txt && cd -
# the braindecode experiments, e.g.
cd methods/cbramod/experiments/braindecode-public-datasets && mkdir -p results_rerun \
  && python code/collect.py logs/runs results_rerun > results_rerun/results.txt \
  && diff results_rerun/results.txt results/results.txt && cd -
cd methods/atcnet/experiments/bciciv2a-model-selection && mkdir -p results_rerun \
  && cp -r results/curves results/authors_2022_log.txt results_rerun/ \
  && python code/analyze.py results_rerun > results_rerun/results.txt && diff results_rerun/results.txt results/results.txt && cd -
python tools/build_results_index.py
python tools/compare_methods.py
python tools/compare_sleepedf.py
python tools/compare_seed.py
python tools/compare_stew.py
python tools/count_flops.py       # needs torch >= 2.1 and braindecode (no data, no GPU)
```

Re-running training needs a GPU; each experiment README gives the exact commands, hardware and
run time. Raw data, checkpoints and re-run outputs are kept out of git (`.gitignore`).

## Reports

Weekly reports and slides live in [docs/reports](docs/README.md), one folder per method.

## Adding work

See [docs/adding-an-experiment.md](docs/adding-an-experiment.md): pin the upstream code as a submodule,
put the experiment under `methods/<method>/experiments/<name>/`, export `results/summary.csv`, and
rebuild `RESULTS.md`.

## Compute

- mcl-server: 8× NVIDIA Quadro M6000 (CBraMod, LaBraM, EEGNet, ATCNet, DeepSleepNet, L-SeqSleepNet, DGCNN, all Sleep-EDF
  runs, the SEED / SEED-V trial split, the braindecode runs, and the 2026-11-20 runs: EEG Conformer, ATCNet and DeepSleepNet on
  CBraMod's four datasets, CBraMod without pre-training on the SEED / SEED-V subject split)
- mcl-server2: 8× NVIDIA RTX A6000, shared with the lab (STEW, the SEED / SEED-V subject split; each model–dataset–split group on
  one machine, recorded in each README)
- dDTF-GL needs no GPU: its runs used the CPUs of mcl-server (2× Xeon E5-2620 v3) and, for SEED-V's subject split, mcl-server2
  (EPYC 7543); its GPU variant (STEW, the paper's settings) ran on mcl-server's Quadro M6000; the feature-tuning search (dDTF-GL+)
  on mcl-server's CPUs; `metrics.json` records the host of each run
- FLOPs and parameters of every model in every pipeline: `python tools/count_flops.py` (no GPU, no data; SUMMARY.md Section 3)
- USC CARC: NVIDIA A40, Slurm (EEG Conformer, released code)

## License

Code under `methods/*/upstream/` keeps its authors' license (EEG Conformer: GPL-3.0, CBraMod and LaBraM: MIT,
ATCNet: Apache-2.0, DeepSleepNet: Apache-2.0 with an academic-use note, L-SeqSleepNet: CC-BY-NC-4.0, braindecode:
BSD-3-Clause), as do files that copy from it (marked `# [MOD]`). The DGCNN authors' archive carries no license, so it is not
committed: `methods/dgcnn/fetch_upstream.sh` downloads it into the git-ignored `methods/dgcnn/upstream/`. The scripts that import EEG Conformer's
GPL-3.0 training loop (`methods/eegnet/experiments/bciciv2a-epoch-selection/code/run_eegnet.py`,
`methods/eeg-conformer/experiments/braindecode-bciciv2a/code/run_braindecode.py`) are GPL-3.0 as well. No license has
been chosen yet for the rest of this repository.
