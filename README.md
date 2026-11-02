# EEG analysis methods review

Independent validation of published EEG decoding methods. Each method is run from its authors' released
code (pinned as a submodule) on public datasets, and the reproduced numbers are compared with the
paper's. Every method is also re-run with its [braindecode](methods/eegnet/upstream) implementation, swapped into
the same pipeline, to check whether the library's version reproduces the released code. Every experiment keeps enough
raw output (training logs, per-epoch curves) to recompute its numbers without a GPU.

**[RESULTS.md](RESULTS.md)** collects every reproduced number by method and dataset.
**[SUMMARY.md](SUMMARY.md)** summarizes them: results by task category and dataset (domain-specific vs. general
models), methods grouped by characteristics, datasets, baselines, and all models compared on identical splits,
including sleep staging on Sleep-EDF, where every model runs through one pipeline next to two sleep-specific models.

## Methods

| Method | Paper | Experiments | Finding so far |
|---|---|---|---|
| [EEG Conformer](methods/eeg-conformer) | Song et al., IEEE TNSRE 2023 | [bciciv2a-epoch-selection](methods/eeg-conformer/experiments/bciciv2a-epoch-selection), Sleep-EDF: [sleepedf-cbramod-pipeline](methods/eeg-conformer/experiments/sleepedf-cbramod-pipeline); braindecode: [braindecode-bciciv2a](methods/eeg-conformer/experiments/braindecode-bciciv2a), [braindecode-sleepedf](methods/eeg-conformer/experiments/braindecode-sleepedf) | The reported accuracy is the best of 2000 test-set evaluations; with a validation split it drops by ~9 pp. braindecode's `EEGConformer` reproduces the released model's numbers under every rule. In CBraMod's sleep-staging pipeline on Sleep-EDF it is second to last (κ 0.688). |
| [CBraMod](methods/cbramod) | Wang et al., ICLR 2025 | [finetune-public-datasets](methods/cbramod/experiments/finetune-public-datasets), Sleep-EDF: [finetune-sleepedf](methods/cbramod/experiments/finetune-sleepedf); braindecode: [braindecode-public-datasets](methods/cbramod/experiments/braindecode-public-datasets), [braindecode-sleepedf](methods/cbramod/experiments/braindecode-sleepedf) | Released code and weights fall short of the paper on all four public datasets tested. On Sleep-EDF (no paper number; its sleep pipeline hosts every model there) pre-training adds 0.023 κ, and CBraMod ties LaBraM-Base, L-SeqSleepNet and ATCNet (κ 0.768–0.773). braindecode's `CBraMod` has the released encoder and weights but a different task head: within noise of the released code on three datasets, 0.02 lower on PhysioNet-MI. |
| [LaBraM](methods/labram) | Jiang et al., ICLR 2024 | [finetune-cbramod-splits](methods/labram/experiments/finetune-cbramod-splits), Sleep-EDF: [sleepedf-cbramod-pipeline](methods/labram/experiments/sleepedf-cbramod-pipeline); braindecode: [braindecode-cbramod-splits](methods/labram/experiments/braindecode-cbramod-splits), [braindecode-sleepedf](methods/labram/experiments/braindecode-sleepedf) | CBraMod's strongest baseline, run under CBraMod's splits, seeds and scoring: it matches the CBraMod paper's LaBraM numbers on 10 of 12 metrics, and CBraMod beats it only on PhysioNet-MI (tie on BCIC-IV-2a and Mumtaz2016, LaBraM ahead on MentalArithmetic). braindecode's `Labram` is the same model and reproduces these runs on all 12 metrics. On Sleep-EDF it has the highest κ of all models (0.773; braindecode's re-run 0.777), within noise of CBraMod and L-SeqSleepNet. |
| [ATCNet](methods/atcnet) | Altaheri et al., IEEE TII 2023 | [bciciv2a-model-selection](methods/atcnet/experiments/bciciv2a-model-selection); braindecode: [braindecode-bciciv2a](methods/atcnet/experiments/braindecode-bciciv2a), Sleep-EDF: [braindecode-sleepedf](methods/atcnet/experiments/braindecode-sleepedf) | The reported 85.38% is the best of 10 runs per subject, each checkpointed on the test set (the authors' 2022 log and saved models; both reproduce exactly). Single runs of that code average 81.5%, and 79.5% with a validation split (the protocol the authors now recommend). braindecode's `ATCNet` is the same network but scores 5 pp lower when trained the same way (paper's protocol), because its final layer's max-norm is taken over the wrong axis. On Sleep-EDF (braindecode, CBraMod's pipeline) its 112k-parameter encoder ties the foundation models in κ (0.771). |
| [EEGNet](methods/eegnet) (baseline) | Lawhern et al., J. Neural Eng. 2018 | braindecode: [bciciv2a-epoch-selection](methods/eegnet/experiments/bciciv2a-epoch-selection), [braindecode-cbramod-splits](methods/eegnet/experiments/braindecode-cbramod-splits), [braindecode-sleepedf](methods/eegnet/experiments/braindecode-sleepedf); our PyTorch implementation: [pytorch-cbramod-splits](methods/eegnet/experiments/pytorch-cbramod-splits), Sleep-EDF: [sleepedf-cbramod-pipeline](methods/eegnet/experiments/sleepedf-cbramod-pipeline) | Run through EEG Conformer's own training loop, EEGNet trails Conformer only when each model's best test-set epoch is reported (−6.1 pp); at the last-100-epoch mean they are level, and with a validation split the gap (−2.65 pp) is not significant. Cross-subject, in CBraMod's pipeline, it is the best model on BCI-IV-2a, with either implementation. On Sleep-EDF it is the weakest model (κ 0.681). |
| [DeepSleepNet](methods/deepsleepnet) (sleep) | Supratak et al., IEEE TNSRE 2017 | Sleep-EDF: [sleepedf20-20fold](methods/deepsleepnet/experiments/sleepedf20-20fold), [sleepedf78-cbramod-split](methods/deepsleepnet/experiments/sleepedf78-cbramod-split); braindecode: [braindecode-sleepedf](methods/deepsleepnet/experiments/braindecode-sleepedf) | The released code reproduces the paper on Sleep-EDF-20 (81.8% vs. 82.0%, κ 0.754 vs. 0.76) on exactly its 41,950 epochs. On Sleep-EDF-78 its final-model rule overfits and it is the weakest sequence model (κ 0.703). braindecode's single-epoch DeepSleepNet in CBraMod's pipeline scores 0.762, with the highest balanced accuracy of all models alongside braindecode's LaBraM. |
| [L-SeqSleepNet](methods/l-seqsleepnet) (sleep) | Phan et al., IEEE JBHI 2023 | Sleep-EDF: [sleepedf20-loso](methods/l-seqsleepnet/experiments/sleepedf20-loso), [sleepedf78-cbramod-split](methods/l-seqsleepnet/experiments/sleepedf78-cbramod-split) | The released code and data reproduce the paper on Sleep-EDF-20: 86.1% vs. 86.3%, κ 0.811 vs. 0.813, in one of its five repetitions, once the run scripts' test list is fixed. On Sleep-EDF-78 it ties the two foundation models (κ 0.773). Not in braindecode. |

## braindecode re-implementations

Six of the seven models exist in [braindecode](https://github.com/braindecode/braindecode) (v1.8.1, pinned as
`methods/eegnet/upstream`); L-SeqSleepNet does not. Each of the first five was swapped into the pipeline of the
released-code experiment, with everything else unchanged (for ATCNet, whose released code is TensorFlow, into a PyTorch port of its training loops), and checked
first against the released model with identical weights (`code/check_equivalence.py` in each experiment):

| Model | braindecode class | Same as the released model? | braindecode vs. released code, same pipeline | Experiment |
|---|---|---|---|---|
| EEG Conformer | `EEGConformer` | same layers; attention divides by √(head dim) = 2 instead of √(embedding) ≈ 6.3, so outputs differ for the same weights | within noise under every epoch-selection rule (released rule 78.56% vs. 78.72%, p = 0.57); Sleep-EDF-78: κ 0.701 vs. 0.688 (p = 0.47), N3 F1 higher (0.71 vs. 0.56) | [braindecode-bciciv2a](methods/eeg-conformer/experiments/braindecode-bciciv2a), [braindecode-sleepedf](methods/eeg-conformer/experiments/braindecode-sleepedf) |
| CBraMod | `CBraMod` | encoder and weights identical; task head different (keeps the pre-training `proj_out`, then one linear layer instead of a 3-layer MLP) | within noise on 3 of 4 datasets; PhysioNet-MI −0.020 fine-tuned, −0.072 from scratch, −0.092 frozen. Sleep-EDF-78, where both feed the released sequence head: κ 0.768 vs. 0.768 (p = 0.99) | [braindecode-public-datasets](methods/cbramod/experiments/braindecode-public-datasets), [braindecode-sleepedf](methods/cbramod/experiments/braindecode-sleepedf) |
| LaBraM-Base | `Labram` | identical (weights and logits) | within noise on all 12 metrics; Sleep-EDF-78: κ 0.777 vs. 0.773 (p = 0.36), encoder features identical | [braindecode-cbramod-splits](methods/labram/experiments/braindecode-cbramod-splits), [braindecode-sleepedf](methods/labram/experiments/braindecode-sleepedf) |
| ATCNet | `ATCNet` | same layers and weights; identical predictions once BatchNorm eps is set to Keras' 1e-3 (braindecode: 1e-4 / 1e-5, which turns the paper's 85.38% models into 84.53%); final-layer max-norm taken per input feature instead of per output unit | 5.2 pp lower in the paper's protocol (76.26 vs. 81.47%, 0 of 9 subjects higher), 10–13 pp lower in today's; fixing the max-norm axis recovers 3.7–4.6 pp | [braindecode-bciciv2a](methods/atcnet/experiments/braindecode-bciciv2a) |
| EEGNet | `EEGNet` | not compared (the released Keras code is not used); same architecture size as our PyTorch EEGNet | within noise of our PyTorch EEGNet on all 12 metrics; Sleep-EDF-78: κ 0.6808 vs. 0.6815 (p = 0.95) | [braindecode-cbramod-splits](methods/eegnet/experiments/braindecode-cbramod-splits), [bciciv2a-epoch-selection](methods/eegnet/experiments/bciciv2a-epoch-selection), [braindecode-sleepedf](methods/eegnet/experiments/braindecode-sleepedf) |
| DeepSleepNet | `DeepSleepNet` | not the released model: single-epoch (its BiLSTM sees one epoch, where the released model's runs across 25), standard LSTM gates instead of peepholes | not run in the released pipeline (TensorFlow). In CBraMod's sleep pipeline it scores κ 0.762 on Sleep-EDF-78, against 0.703 for the released code with its own training on the same split | [braindecode-sleepedf](methods/deepsleepnet/experiments/braindecode-sleepedf) |

None of the braindecode versions comes closer to the papers than the released code. ATCNet's falls furthest short of
its released code: subject-dependent, −5 pp in the paper's protocol and −10 to −13 pp in today's, lower on every
subject. In the paper's protocol the whole gap traces to an axis difference in braindecode's `MaxNormLinear` (a layer
its `EEGTCNet` uses too); today's Keras code adds regularization that braindecode's defaults lack. The model ranking
under CBraMod's splits changes in one place: with both foundation models from braindecode, CBraMod no longer beats LaBraM-Base on
PhysioNet-MI ([SUMMARY.md](SUMMARY.md#the-same-comparison-with-braindecodes-implementations)). On Sleep-EDF-78, in CBraMod's
sleep-staging pipeline, all four re-runs are within noise of the originals. There both CBraMods feed the same released
sequence head and match (κ +0.0001), consistent with braindecode's own head causing the PhysioNet-MI gap.

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
│   └── compare_sleepedf.py         every Sleep-EDF model on the common test set, Welch tests, figure (SUMMARY.md)
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

- mcl-server: 8× NVIDIA Quadro M6000 (CBraMod, LaBraM, EEGNet, ATCNet, DeepSleepNet, L-SeqSleepNet, all Sleep-EDF
  runs, and all braindecode runs)
- USC CARC: NVIDIA A40, Slurm (EEG Conformer, released code)

## License

Code under `methods/*/upstream/` keeps its authors' license (EEG Conformer: GPL-3.0, CBraMod and LaBraM: MIT,
ATCNet: Apache-2.0, DeepSleepNet: Apache-2.0 with an academic-use note, L-SeqSleepNet: CC-BY-NC-4.0, braindecode:
BSD-3-Clause), as do files that copy from it (marked `# [MOD]`). The scripts that import EEG Conformer's
GPL-3.0 training loop (`methods/eegnet/experiments/bciciv2a-epoch-selection/code/run_eegnet.py`,
`methods/eeg-conformer/experiments/braindecode-bciciv2a/code/run_braindecode.py`) are GPL-3.0 as well. No license has
been chosen yet for the rest of this repository.
