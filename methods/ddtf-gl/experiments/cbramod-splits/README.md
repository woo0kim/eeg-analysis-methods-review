# dDTF-GL on CBraMod's four public datasets

dDTF-GL on exactly the preprocessed samples, subject splits and metrics every other model in this repository was scored on for
CBraMod's four public datasets (CBraMod's LMDBs: BCI-IV-2a, PhysioNet-MI, Mumtaz2016, MentalArithmetic; [cbramod/finetune-public-datasets](../../../cbramod/experiments/finetune-public-datasets)).
The paper reports none of them; the question is how a method built and tuned on STEW's 14-channel workload recordings does on motor
imagery, depression and mental-stress data. Seeds 0–4; trained on the training subjects, the validation subjects scored only.

| Dataset | Samples (train / val / test) | Channels used | MVAR order | dDTF features |
|---|---|---|---|---|
| BCI-IV-2a | 2,880 / 1,152 / 1,152 trials of 4 s | 21 of 22 (average-referenced already) | 12 (AIC still falling at the cap) | 2,205 |
| PhysioNet-MI | 6,300 / 1,734 / 1,803 trials of 4 s | 63 of 64 (average-referenced already) | 4 (the cap) | 19,845 |
| Mumtaz2016 | 4,891 / 1,041 / 1,211 windows of 5 s | 19, re-referenced to their average | 17 (cap) | 1,805 |
| MentalArithmetic | 1,343 / 172 / 192 windows of 5 s | 20; the 19 scalp channels re-referenced to their average, A2-A1 as is | 16 (cap) | 2,000 |

## Results

Balanced accuracy, mean ± SD over seeds 0–4 (`results/results.txt`); the other models from [SUMMARY.md](../../../../SUMMARY.md), same
samples, splits and scoring (DGCNN on 5-band DE). dDTF-GL's maps use the settings calibrated on Safari et al.'s maps; the first map
settings are an ablation:

| Dataset | dDTF-GL | dDTF-GL, first map settings | EEGNet | CBraMod | LaBraM-Base | DGCNN | Chance |
|---|---|---|---|---|---|---|---|
| PhysioNet-MI | 0.3274 ± 0.0055 | 0.3247 ± 0.0058 | 0.5948 | 0.6265 | 0.6098 | 0.4822 | 0.25 |
| BCI-IV-2a | 0.3276 ± 0.0084 | 0.3125 ± 0.0073 | 0.5288 | 0.4571 | 0.4477 | 0.3458 | 0.25 |
| Mumtaz2016 | **0.9192 ± 0.0025** (AUROC 0.9816) | 0.9433 ± 0.0057 (AUROC 0.9910) | 0.8828 | 0.8933 | 0.8905 | 0.9619 | 0.50 |
| MentalArithmetic | 0.5833 ± 0.0282 (AUROC 0.7149) | 0.6326 ± 0.0513 (AUROC 0.7133) | 0.5569 | 0.5889 | 0.6562 | 0.6236 | 0.50 |

dDTF-GL − each model, Welch p over seeds (`python ../../../../tools/compare_methods.py`, last table; pass `roc_auc` or `kappa` for those):

| Dataset | − EEGNet | − CBraMod | − LaBraM-Base | − DGCNN |
|---|---|---|---|---|
| PhysioNet-MI | −0.2674 (p < 0.001) | −0.2991 (p < 0.001) | −0.2824 (p < 0.001) | −0.1548 (p < 0.001) |
| BCI-IV-2a | −0.2012 (p < 0.001) | −0.1295 (p = 0.001) | −0.1201 (p = 0.003) | −0.0182 (p = 0.30) |
| Mumtaz2016 | +0.0364 (p = 0.051) | +0.0258 (p = 0.004) | +0.0286 (p < 0.001) | −0.0428 (p < 0.001) |
| MentalArithmetic | +0.0264 (p = 0.16) | −0.0056 (p = 0.73) | −0.0729 (p = 0.012) | −0.0403 (p = 0.26) |

- **Motor imagery is out of reach for connectivity maps:** 0.33 on both datasets against chance 0.25 (κ 0.10), the weakest model on
  both, level with DGCNN's DE input on BCI-IV-2a (p = 0.30). Like band power, a dDTF map averaged over a 4-s trial keeps no timing
  of the imagery.
- **Mumtaz2016: above both foundation models** (0.919, +0.026 and +0.029), 0.043 behind DGCNN's DE input; the first map settings did
  better still (0.943). As there, spectral features alone separate these 11 test subjects.
- **MentalArithmetic: level with CBraMod and EEGNet** (0.583), below LaBraM-Base (−0.073, p = 0.012); the first map settings scored 0.633.
  4 test subjects make the seed spread large.
- **The calibration costs here:** the maps that agree better with Safari et al.'s score lower on the two non-motor datasets
  (−0.024 and −0.049) and the same on motor imagery.
- **Cost grows with the square of the channels.** PhysioNet-MI's 63 channels give 19,845 features; a run takes 15–18 minutes on
  20 threads (AXB about half of it), against 15–110 s on the other three datasets.

## Paper vs. this reproduction

Not a reproduction: the paper evaluates only STEW. The method's settings are in [../../README.md](../../README.md#the-method-as-run-here);
what differs from the other models on these splits:

| | Every other model here | dDTF-GL | Why |
|---|---|---|---|
| Input | the LMDB samples (÷ 100) | the same samples, each channel z-scored per window for the MVAR fit; BCI-IV-2a and PhysioNet-MI without their last channel | dDTF needs a full-rank signal; the dropped channel is minus the sum of the others |
| Model selection | best validation κ (multi-class) or AUROC (binary) | none: one fit on the training subjects; validation scored only | no epochs |
| Binary scores | sigmoid of the logit, threshold 0.5 | XGBoost's class-1 probability, threshold 0.5 (its predicted class) | – |
| Metrics | CBraMod's evaluator: balanced accuracy, κ, weighted F1; or balanced accuracy, AUC-PR (`auc(recall, precision)`), AUROC | the same functions (`code/run_gl.py`) | – |

Run configuration: 10 threads; environment [../../env/conda_list.txt](../../env/conda_list.txt); mcl-server (2× Xeon E5-2620 v3).

## Reproducing

```bash
code/reproduce.sh collect                     # results/ from logs/runs
python ../../../../tools/compare_methods.py   # last table: dDTF-GL against EEGNet, CBraMod, LaBraM-Base and DGCNN
code/reproduce.sh train                       # needs CBraMod's LMDBs under $EEG_DATA_ROOT (datasets.py)
```

## Provenance

Runs: 2026-11-11, `~/repro-gl/runs/cbramod-splits` on mcl-server.
