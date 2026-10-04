# Adding a method or an experiment

## 1. A new method

```bash
M=methods/<method-name>                     # kebab-case, e.g. labram, eegnet
git submodule add https://github.com/<owner>/<repo>.git $M/upstream
git -C $M/upstream checkout <commit-you-evaluate>
git add $M/upstream
```

Pin the exact commit you ran, and never edit files inside `upstream/`. If you need a modified copy of
an upstream file, copy it into your experiment and mark every changed line with `# [MOD]`, as
`methods/eeg-conformer/experiments/bciciv2a-epoch-selection/code/run_validation.py` does.

Then write `$M/README.md`: citation, released-code and weights links, pinned commit, upstream license,
and a table of experiments with a one-line finding each (see `methods/cbramod/README.md`).

## 2. A new experiment

```
methods/<method>/experiments/<name>/
├── README.md        question, setup, results, how to reproduce, environment, provenance
├── code/ or scripts/
├── env/             setup script and/or pip freeze of the runs (or requirements.txt)
├── results/         every number the README quotes, plus summary.csv (below)
└── logs/            raw training logs (text only; no checkpoints)
```

Name experiments after the dataset and the question (`bciciv2a-epoch-selection`), or only the question
when one pipeline spans several datasets (`finetune-public-datasets`).

Rules:

- Commit enough raw output (per-epoch curves, full logs) that every number in the README can be
  recomputed without a GPU, and say how in the README.
- Never commit data or checkpoints. Read data from `$EEG_DATA_ROOT` and add a key to
  `datasets/download.sh` and a row to `datasets/README.md` for any new dataset.
- Hard-coded paths become environment variables whose defaults match the machine the logged runs used.
- Record hardware, seeds and package versions.
- Write re-runs to a separate directory (`results_rerun/`, ignored by git) so they never overwrite the
  committed results.

## 3. `results/summary.csv`

Each experiment exports its numbers in one schema, so `tools/build_results_index.py` can collect them
into `RESULTS.md`. Write it with a small script next to your analysis code (see the two existing
`export_summary.py`), not by hand.

| Column | Meaning |
|---|---|
| `method` | model being evaluated (a baseline run inside another method's experiment gets its own name, e.g. `EEGNet`) |
| `dataset` | name used in `datasets/README.md` (`BCI-IV-2a`, `PhysioNet-MI`, …) |
| `protocol` | evaluation split, e.g. `within-subject, session T -> E`, `cross-subject 5/2/2`; only rows with the same protocol are comparable across methods |
| `setting` | what was run, in words |
| `metric` | `accuracy`, `balanced_acc`, `kappa`, `weighted_f1`, `roc_auc`, `pr_auc`, … |
| `paper_mean`, `paper_std`, `paper_source` | the paper's number and its table; empty when the paper has none |
| `ours_mean`, `ours_std`, `n_runs` | reproduced value as a fraction in [0, 1], SD over seeds, number of runs |
| `headline` | `1` for the rows shown in RESULTS.md's headline table, else `0` |

```bash
python methods/<method>/experiments/<name>/code/export_summary.py methods/<method>/experiments/<name>/results
python tools/build_results_index.py
```
