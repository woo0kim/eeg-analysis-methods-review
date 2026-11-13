# dDTF-GL on STEW, in the pipeline every model shares there

dDTF-GL on exactly the windows, folds and tasks every other model ran on STEW ([cbramod/finetune-stew](../../../cbramod/experiments/finetune-stew):
4-s windows with 25% overlap at 128 Hz after a 0.3-Hz high-pass, 4,704 windows; the 7 pinned GroupKFold folds; rest vs. task and 3 rating
levels), and what its two distinctive parts, the dDTF input and the auxiliary XGBoost (AXB), add. No network and no epochs: the
validation fold (k + 1) is scored but not used, and the model is fitted once on the five training folds. Seeds 0–4 × 7 folds per task and
variant; a seed's score is computed on its 7 pooled test folds, as for every model here.

| Variant | Input | AXB | |
|---|---|---|---|
| `ddtf-gl` | dDTF maps (980), settings calibrated on Safari et al.'s maps | yes | the paper's method ([../../README.md](../../README.md)) |
| `ddtf-gl-noaxb` | the same maps | no | the bundle's chain on dDTF |
| `ddtf-gl-v0`, `ddtf-gl-noaxb-v0` | dDTF maps with our first settings | yes / no | before the calibration (ablation) |
| `dwt-gl` | the bundle's DWT statistics (710) | no | the authors' bundle (`mwl_gl_si7.py`'s features and chain) on the shared windows and folds |
| `dwt-gl-axb` | DWT statistics | yes | |

## Results

Mean ± SD over seeds 0–4, each seed scored on its 7 pooled test folds (`results/results.txt`; every model:
`python ../../../../tools/compare_stew.py`). dDTF-GL's maps use the settings calibrated on Safari et al.'s released maps
([../../README.md](../../README.md#the-method-as-run-here)); "first map settings" is our implementation before that calibration.

| Task | Variant | Accuracy | Bal. acc | κ | Weighted F1 | Fold SD (accuracy) |
|---|---|---|---|---|---|---|
| rest vs. task | **dDTF-GL** | 0.7398 ± 0.0043 | 0.7398 ± 0.0043 | 0.4796 ± 0.0085 | 0.7393 ± 0.0043 | 0.0464 |
| rest vs. task | dDTF-GL, no AXB | 0.7429 ± 0.0018 | 0.7429 ± 0.0018 | 0.4858 ± 0.0036 | 0.7424 ± 0.0018 | 0.0402 |
| rest vs. task | dDTF-GL, first map settings | 0.7623 ± 0.0034 | 0.7623 ± 0.0034 | 0.5247 ± 0.0068 | 0.7622 ± 0.0034 | 0.0441 |
| rest vs. task | dDTF-GL, first map settings, no AXB | 0.7646 ± 0.0055 | 0.7646 ± 0.0055 | 0.5293 ± 0.0110 | 0.7645 ± 0.0055 | 0.0454 |
| rest vs. task | DWT-GL (the bundle's chain) | 0.7773 ± 0.0038 | 0.7773 ± 0.0038 | 0.5546 ± 0.0076 | 0.7766 ± 0.0038 | 0.0520 |
| rest vs. task | DWT-GL + AXB | 0.7817 ± 0.0067 | 0.7817 ± 0.0067 | 0.5634 ± 0.0135 | 0.7812 ± 0.0069 | 0.0505 |
| 3 rating levels | **dDTF-GL** | 0.5185 ± 0.0052 | 0.4626 ± 0.0059 | 0.2365 ± 0.0075 | 0.5091 ± 0.0051 | 0.0304 |
| 3 rating levels | dDTF-GL, no AXB | 0.5260 ± 0.0045 | 0.4676 ± 0.0045 | 0.2465 ± 0.0073 | 0.5149 ± 0.0047 | 0.0294 |
| 3 rating levels | dDTF-GL, first map settings | 0.5303 ± 0.0017 | 0.4715 ± 0.0028 | 0.2520 ± 0.0034 | 0.5174 ± 0.0024 | 0.0430 |
| 3 rating levels | dDTF-GL, first map settings, no AXB | 0.5338 ± 0.0021 | 0.4716 ± 0.0024 | 0.2550 ± 0.0037 | 0.5188 ± 0.0024 | 0.0482 |
| 3 rating levels | DWT-GL (the bundle's chain) | 0.5559 ± 0.0017 | 0.5058 ± 0.0015 | 0.3056 ± 0.0026 | 0.5551 ± 0.0017 | 0.1056 |
| 3 rating levels | DWT-GL + AXB | 0.5525 ± 0.0068 | 0.5050 ± 0.0066 | 0.3014 ± 0.0091 | 0.5529 ± 0.0054 | 0.1138 |

Against the other models (accuracy, A − B, Welch p):

| | rest vs. task | 3 rating levels |
|---|---|---|
| dDTF-GL − LaBraM-Base | −0.0781 (p < 0.001) | −0.0485 (p < 0.001) |
| dDTF-GL − CBraMod | −0.0600 (p < 0.001) | −0.0476 (p < 0.001) |
| dDTF-GL − DGCNN | −0.0349 (p < 0.001) | −0.0332 (p < 0.001) |
| dDTF-GL − EEG Conformer | +0.0034 (p = 0.65) | +0.0067 (p = 0.45) |
| dDTF-GL − EEGNet | +0.0750 (p < 0.001) | +0.0210 (p = 0.077) |
| dDTF-GL − dDTF-GL, no AXB | −0.0031 (p = 0.19) | −0.0075 (p = 0.040) |
| dDTF-GL − dDTF-GL, first map settings | −0.0225 (p < 0.001) | −0.0118 (p = 0.005) |
| dDTF-GL − DWT-GL | −0.0375 (p < 0.001) | −0.0374 (p < 0.001) |

- **Level with EEG Conformer, below DGCNN and the foundation models**, above EEGNet, with no network and about 7,000 tree nodes
  (21,800 for 3 classes). On rest vs. task ATCNet (0.726) and braindecode's DeepSleepNet (0.717) are below it, L-SeqSleepNet off-label
  (0.767) above.
- **The calibrated maps score lower than our first settings** here (−0.023 on rest vs. task, −0.012 on 3 levels), although they
  reproduce Safari et al.'s maps three times better. Closer to the paper's input is not better on these 4-s windows.
- **AXB adds nothing** (−0.003, p = 0.19; −0.008 on 3 levels), with either maps or with DWT statistics.
- **The bundle's own DWT statistics beat the paper's dDTF maps** on the same windows by 0.037 (both tasks, p < 0.001). The paper
  motivates dDTF maps by deep networks doing better on them than on raw EEG (its Tables 1–2); for its own pipeline, wavelet
  statistics do better.
- The fold spread (SD 0.03–0.05 accuracy, DWT 0.05–0.11) is ten times the seed spread.

## Paper vs. this reproduction

Not a reproduction of a paper number: the paper's own STEW protocol (6-s windows, its SD and SI settings) is in
[stew-paper-protocol](../stew-paper-protocol). Here the windows, folds, tasks and scoring are the shared pipeline's; the method's
settings are listed in [../../README.md](../../README.md#the-method-as-run-here) (MVAR order 8 by AIC on fold 0's training windows; average reference).

| | Every other model here | dDTF-GL | Why |
|---|---|---|---|
| Model selection | the epoch with the best validation κ (validation = fold k + 1) | none: one fit on the training folds, validation scored only | the method has no epochs or early stopping; the validation fold is not added to training, so every model trains on the same five folds |
| 3 rating levels | – | the same chain; LNT gives 2 components per group (1 for rest vs. task) | the bundle's `select_comb` / `lnt_kernel` |

Run configuration: the bundle's DFT, LNT and XGBoost settings; 10–16 threads (`metrics.json` records host and threads of each run);
environment [../../env/conda_list.txt](../../env/conda_list.txt); mcl-server, 2× Xeon E5-2620 v3, CPU only.
`code/check_equivalence.py` output: `results/equivalence.txt`.

## Reproducing

```bash
code/reproduce.sh collect                     # results/ from logs/runs
python ../../../../tools/compare_stew.py      # every model
code/reproduce.sh train                       # features and the 420 runs (needs finetune-stew's windows.npz)
```

## Provenance

Runs: 2026-11-11, `~/repro-gl/runs/stew-cbramod-pipeline` on mcl-server.
