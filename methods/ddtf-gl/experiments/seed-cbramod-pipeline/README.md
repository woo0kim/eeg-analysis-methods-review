# dDTF-GL on SEED and SEED-V

dDTF-GL on the 1-s samples of the shared SEED / SEED-V pipeline ([cbramod/finetune-seed](../../../cbramod/experiments/finetune-seed): 62
channels at 200 Hz, 0.3–75 Hz; SEED 152,730 samples, SEED-V 117,744), under both splits every other model ran: CBraMod's **trial**
split (clips 1–5 / 6–10 / 11–15 of every session, all subjects on both sides) and the **subject** split (SEED 9 / 3 / 3, SEED-V
10 / 3 / 3). Seeds 0–4; trained on the training part, the validation part scored only. The paper has no emotion data.

A 1-s window of 62 channels has 200 samples, which leaves room for an MVAR model of order 1 only (at least three samples per
regressor: [../../README.md](../../README.md#the-method-as-run-here)); the maps are 62 × 62 per band, 19,220 features, the largest
input of any dataset here.

## Results

Balanced accuracy and κ, mean ± SD over seeds 0–4, and the clip vote (majority over a clip's 1-s predictions); every model:
`python ../../../../tools/compare_seed.py` (`results/results.txt`). Maps with the settings calibrated on Safari et al.'s maps.

| Dataset, split | dDTF-GL: bal. acc | κ | clip vote | DGCNN | EEG Conformer | EEGNet | CBraMod | LaBraM-Base | chance |
|---|---|---|---|---|---|---|---|---|---|
| SEED, trial | 0.5609 ± 0.0014 | 0.3206 ± 0.0022 | 0.6480 | 0.5741 | 0.5360 | 0.4909 | 0.6162 | 0.6300 | 0.333 |
| SEED, subject | 0.5237 ± 0.0015 | 0.2897 ± 0.0022 | 0.5407 | 0.5215 | 0.5378 | 0.5227 | 0.5442 | 0.5822 | 0.333 |
| SEED-V, trial | 0.3245 ± 0.0012 | 0.1583 ± 0.0016 | 0.4333 | 0.3866 | 0.3205 | 0.2894 | 0.3962 | 0.4136 | 0.200 |
| SEED-V, subject | 0.2315 ± 0.0019 | 0.0387 ± 0.0019 | 0.2607 | 0.2563 | 0.2366 | 0.2468 | 0.2223 | 0.2214 | 0.200 |

dDTF-GL − model, balanced accuracy, Welch p over seeds:

| | − DGCNN | − EEGNet | − CBraMod | − LaBraM-Base |
|---|---|---|---|---|
| SEED, trial | −0.013 (p = 0.17) | +0.070 (p < 0.001) | −0.055 (p < 0.001) | −0.069 (p < 0.001) |
| SEED, subject | +0.002 (p = 0.74) | +0.001 (p = 0.93) | −0.021 (p = 0.007) | −0.059 (p < 0.001) |
| SEED-V, trial | −0.062 (p < 0.001) | +0.035 (p = 0.001) | −0.072 (p < 0.001) | −0.089 (p < 0.001) |
| SEED-V, subject | −0.025 (p = 0.004) | −0.015 (p = 0.013) | +0.009 (p = 0.040) | +0.010 (p = 0.097) |

- **With the supervised models, behind the foundation models.** On SEED's trial split dDTF-GL is level with DGCNN (p = 0.17), above EEG
  Conformer (+0.025, p = 0.001), EEGNet and ATCNet, and below DeepSleepNet (braindecode, −0.030, p = 0.002) and both foundation models.
  Across subjects it is level with DGCNN, EEGNet and DeepSleepNet and below EEG Conformer (−0.014, p = 0.039), ATCNet and the foundation
  models. On SEED-V's trial split it is level with EEG Conformer (p = 0.55), above EEGNet and ATCNet, 0.06–0.09 below DGCNN,
  DeepSleepNet and the foundation models.
- **Across subjects SEED-V is at chance for dDTF-GL too** (κ 0.04), as for every model.
- **The largest input of any dataset here:** 62 channels give 19,220 values per 1-s sample from an MVAR model of order 1; DFT keeps
  9,610, AXB 3,800–4,800, and the classifier sees 4,700–6,000 features. A run takes 13–45 minutes on 20–32 CPU threads.
- **Our first map settings** scored slightly higher on SEED (trial 0.5686, subject 0.5500 over 4 seeds) and the same on SEED-V's subject
  split (0.2328).

## Paper vs. this reproduction

Not a reproduction: the paper has no SEED result. The method's settings are in [../../README.md](../../README.md#the-method-as-run-here);
differences from the other models in the pipeline:

| | Every other model here | dDTF-GL | Why |
|---|---|---|---|
| Model selection | best validation κ | none; validation scored only | no epochs |
| Input | 1-s samples (÷ 100) | the same samples, re-referenced to their average and z-scored per channel for the MVAR fit (the calibrated settings); MVAR order 1 | a 1-s window of 62 channels allows no higher order |
| Hardware | GPUs | CPU: mcl-server (2× Xeon E5-2620 v3, 20–22 threads) for SEED, mcl-server2 (EPYC 7543, 32 threads) for SEED-V; `metrics.json` records the host and threads of every run | 19,220 features × up to 92k samples take 25–45 min per run on mcl-server, 15–20 min on mcl-server2 |

History of the runs: with our first map settings, the SEED subject split and SEED-V were first run on mcl-server2 from a copy of the
features in shared memory, which the server cleared before the run directories were saved; SEED-V's subject split was repeated there
from disk. The maps were then switched to the settings calibrated on Safari et al.'s maps and every calibrated run made from scratch:
SEED on mcl-server, SEED-V on mcl-server2. The first-settings ablation covers the runs completed before the switch (SEED: trial split 5
seeds, subject split 4; SEED-V: subject split 5).

## Reproducing

```bash
code/reproduce.sh collect                       # results/ from logs/runs (clip vote from finetune-seed/results/test_trials.json)
python ../../../../tools/compare_seed.py        # every model
code/reproduce.sh train                         # needs the SEED / SEED-V LMDBs of finetune-seed under $EEG_DATA_ROOT
```

## Provenance

Runs: 2026-11-11 to 11-13, `~/repro-gl/runs/seed-cbramod-pipeline` on mcl-server (SEED) and mcl-server2 (SEED-V).
