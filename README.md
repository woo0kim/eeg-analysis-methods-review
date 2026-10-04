# EEG analysis methods review

Independent validation of published EEG decoding methods. Each method is run from its authors' released
code (pinned as a submodule) on public datasets, and the reproduced numbers are compared with the
paper's. Every experiment keeps enough raw output (training logs, per-epoch curves) to recompute its
numbers without a GPU.

**[RESULTS.md](RESULTS.md)** collects every reproduced number by method and dataset.

## Methods

| Method | Paper | Experiments | Finding so far |
|---|---|---|---|
| [EEG Conformer](methods/eeg-conformer) | Song et al., IEEE TNSRE 2023 | [bciciv2a-epoch-selection](methods/eeg-conformer/experiments/bciciv2a-epoch-selection) | The reported accuracy is the best of 2000 test-set evaluations; with a validation split it drops by ~9 pp. |
| [CBraMod](methods/cbramod) | Wang et al., ICLR 2025 | [finetune-public-datasets](methods/cbramod/experiments/finetune-public-datasets) | Released code and weights fall short of the paper on all four public datasets tested; EEGNet (same pipeline) is included as a baseline. |

## Layout

```
.
├── RESULTS.md                      generated index of all results (tools/build_results_index.py)
├── datasets/
│   ├── README.md                   dataset registry: source, access, which experiments use it
│   └── download.sh                 shared downloader -> $EEG_DATA_ROOT (default ~/data)
├── methods/
│   └── <method>/
│       ├── README.md               paper, released code / weights, pinned commit, license, experiments
│       ├── upstream/               git submodule: the authors' code at the evaluated commit (never edited)
│       └── experiments/
│           └── <experiment>/
│               ├── README.md       question, setup, results, how to reproduce, provenance
│               ├── code/ | scripts/
│               ├── env/            environment setup / pip freeze (or requirements.txt)
│               ├── results/        numbers, figures, summary.csv
│               └── logs/           raw training logs
├── tools/
│   ├── run_queue.py                file-based multi-GPU job queue
│   └── build_results_index.py      methods/*/experiments/*/results/summary.csv -> RESULTS.md
└── docs/
    └── adding-an-experiment.md     conventions and the summary.csv schema
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
cd methods/cbramod/experiments/finetune-public-datasets && mkdir -p results_rerun \
  && python scripts/collect.py logs/runs results_rerun > results_rerun/results.txt \
  && diff results_rerun/results.txt results/results.txt && cd -
python tools/build_results_index.py
```

Re-running training needs a GPU; each experiment README gives the exact commands, hardware and
run time. Raw data, checkpoints and re-run outputs are kept out of git (`.gitignore`).

## Adding work

See [docs/adding-an-experiment.md](docs/adding-an-experiment.md): pin the upstream code as a submodule,
put the experiment under `methods/<method>/experiments/<name>/`, export `results/summary.csv`, and
rebuild `RESULTS.md`.

## Compute

- mcl-server: 8× NVIDIA Quadro M6000 (CBraMod)
- USC CARC: NVIDIA A40, Slurm (EEG Conformer)

## License

Code under `methods/*/upstream/` keeps its authors' license (EEG Conformer: GPL-3.0, CBraMod: MIT), as
do files that copy from it (marked `# [MOD]`). No license has been chosen yet for the rest of this
repository.
