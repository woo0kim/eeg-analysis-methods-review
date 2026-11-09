# CBraMod on SEED and SEED-V, and the pipeline every model shares there

Emotion recognition on SEED (15 subjects, 3 classes) and SEED-V (16 subjects, 5 classes) with CBraMod's released SEED-V code
(`upstream/finetune_main.py`, `datasets/seedv_dataset.py`, `models/model_for_seedv.py`, `finetune_trainer.py`, commit `b9e9610`).
This experiment holds the CBraMod runs, fine-tuned from the released weights and trained from random initialisation, and the code every
other model uses on these datasets: preprocessing, the model zoo, the run script, collection and the comparison.

Every other model runs through the same pipeline with only the model swapped: the 1-s samples, splits, seeds (0–4), optimizer, schedule,
checkpoint rule and metric code are the same. Each model's runs sit in that model's own experiment:

| Model | Experiment | Implementation | Splits |
|---|---|---|---|
| CBraMod (pre-trained, and no pre-training) | this experiment | released code and weights | trial (both), subject (pre-trained) |
| LaBraM-Base | [labram/seed-cbramod-pipeline](../../../labram/experiments/seed-cbramod-pipeline) | released code and weights, LaBraM's fine-tuning recipe | trial, subject |
| EEG Conformer | [eeg-conformer/seed-cbramod-pipeline](../../../eeg-conformer/experiments/seed-cbramod-pipeline) | released SEED model (`conformer_seed_1s_5fold.py`) | trial, subject |
| EEGNet | [eegnet/seed-cbramod-pipeline](../../../eegnet/experiments/seed-cbramod-pipeline) | our PyTorch EEGNet-8,2 | trial, subject |
| DGCNN | [dgcnn/seed-cbramod-pipeline](../../../dgcnn/experiments/seed-cbramod-pipeline) | ours, on 5-band DE of each window | trial, subject |
| ATCNet | [atcnet/braindecode-seed](../../../atcnet/experiments/braindecode-seed) | braindecode (released code is TensorFlow) | trial, subject |
| DeepSleepNet | [deepsleepnet/braindecode-seed](../../../deepsleepnet/experiments/braindecode-seed) | braindecode | trial, subject |
| L-SeqSleepNet | [l-seqsleepnet/seed-offlabel](../../../l-seqsleepnet/experiments/seed-offlabel) | released code (TF 1.15), off-label: 30-s windows, 3 channels | trial, subject |
| CBraMod, LaBraM-Base, EEG Conformer, EEGNet, DGCNN | `braindecode-seed` of each | braindecode 1.8.1 | trial |

The CBraMod paper reports SEED-V (Table 2: CBraMod, LaBraM-Base, EEG Conformer, EEGNet) under the trial split, so that cell reproduces
published numbers. SEED appears in two other papers here under other protocols: EEG Conformer's 5-fold code
([eeg-conformer/seed-5fold](../../../eeg-conformer/experiments/seed-5fold)) and DGCNN's paper
([dgcnn/seed-paper-protocol](../../../dgcnn/experiments/seed-paper-protocol)).

## Two splits

- **trial** (CBraMod's SEED-V split): one model for all subjects; in every session clips 1–5 train, 6–10 validate, 11–15 test. The same
  subjects are in all three parts, the clips differ. Applied to SEED unchanged.
- **subject**: the same 1-s samples split by subject: SEED 9 / 3 / 3, SEED-V 10 / 3 / 3, all sessions of a subject together. Every
  clip appears on both sides, from different people. Released and baseline implementations only: the braindecode twins ran under the
  trial split.

## Results

### CBraMod

| Dataset | Split | Model | Bal. acc | κ | Weighted F1 | Accuracy | Clip vote |
|---|---|---|---|---|---|---|---|
| SEED-V | trial | CBraMod | 0.3962 ± 0.0060 | 0.2501 ± 0.0071 | 0.4019 ± 0.0079 | 0.4056 ± 0.0047 | 0.5050 ± 0.0175 |
| SEED-V | trial | CBraMod, no pre-training | 0.2744 ± 0.1019 | 0.0949 ± 0.1300 | 0.2110 ± 0.1674 | 0.2989 ± 0.0890 | 0.3167 ± 0.1602 |
| SEED-V | subject | CBraMod | 0.2223 ± 0.0070 | 0.0230 ± 0.0083 | 0.1887 ± 0.0166 | 0.2100 ± 0.0097 | 0.2178 ± 0.0144 |
| SEED | trial | CBraMod | 0.6162 ± 0.0137 | 0.4020 ± 0.0278 | 0.6183 ± 0.0207 | 0.6189 ± 0.0197 | 0.6969 ± 0.0447 |
| SEED | trial | CBraMod, no pre-training | 0.5872 ± 0.0045 | 0.3630 ± 0.0080 | 0.5961 ± 0.0056 | 0.5973 ± 0.0052 | 0.6800 ± 0.0133 |
| SEED | subject | CBraMod | 0.5442 ± 0.0092 | 0.3211 ± 0.0143 | 0.5415 ± 0.0081 | 0.5479 ± 0.0096 | 0.5852 ± 0.0196 |

- **SEED-V is CBraMod's closest match to its paper in this repository, but not a clean one:** κ and weighted F1 are within noise (−0.0068,
  p = 0.38; −0.0082, p = 0.21), balanced accuracy is 0.0129 lower (0.3962 against 0.4091, p = 0.041). On its four other public datasets 9 of
  12 metrics were significantly below the paper ([finetune-public-datasets](../finetune-public-datasets)).
- **LaBraM-Base, not CBraMod, is the best model on SEED-V,** by 0.0174 balanced accuracy (p = 0.011), and is above its own number in the
  CBraMod paper (significantly for κ and weighted F1, p = 0.041 and 0.020). The paper ranks CBraMod first (0.4091 against 0.3976).
- **Without pre-training, 3 of 5 seeds on SEED-V never leave chance** (balanced accuracy 0.2000, κ 0.000: one class for every sample); the two
  that train reach 0.3908 and 0.3812, level with the pre-trained model. Pre-training makes training reliable here more than it raises the
  ceiling. On SEED all five train, and pre-training adds 0.0290 (p = 0.007).
- **Across subjects, SEED-V is at chance for every model,** CBraMod included (κ 0.023). Its published SEED-V number depends on the same
  subjects being in the training set. On SEED the subject split costs CBraMod 0.072 balanced accuracy against the trial split.
- **Validation selection costs up to 0.076 κ on SEED** (seed 0: 0.3527 selected, 0.4283 at the best test epoch) and 0.005–0.035 on SEED-V.

### All models

Every model on the same samples (`python tools/compare_seed.py`). Clip vote: majority vote over a clip's 1-s predictions.

<!-- COMMON (python tools/compare_seed.py) -->
### SEED, trial split, as CBraMod on SEED-V (chance: 0.333)

| Model | Implementation | Runs | bal. acc | κ | weighted F1 | accuracy | clip vote |
|---|---|---|---|---|---|---|---|
| LaBraM-Base | released | 5 | 0.6300 ± 0.0008 | 0.4260 ± 0.0032 | 0.6348 ± 0.0034 | 0.6347 ± 0.0035 | 0.7120 ± 0.0101 |
| CBraMod (bd) | braindecode | 5 | 0.6198 ± 0.0029 | 0.4069 ± 0.0041 | 0.6214 ± 0.0031 | 0.6212 ± 0.0025 | 0.7022 ± 0.0147 |
| CBraMod | released | 5 | 0.6162 ± 0.0137 | 0.4020 ± 0.0278 | 0.6183 ± 0.0207 | 0.6189 ± 0.0197 | 0.6969 ± 0.0447 |
| LaBraM-Base (bd) | braindecode | 5 | 0.6161 ± 0.0121 | 0.3981 ± 0.0278 | 0.6137 ± 0.0217 | 0.6144 ± 0.0212 | 0.6933 ± 0.0270 |
| DeepSleepNet (bd) | braindecode | 5 | 0.5912 ± 0.0094 | 0.3775 ± 0.0132 | 0.6097 ± 0.0078 | 0.6095 ± 0.0080 | 0.6702 ± 0.0106 |
| CBraMod scratch | released, no pre-training | 5 | 0.5872 ± 0.0045 | 0.3630 ± 0.0080 | 0.5961 ± 0.0056 | 0.5973 ± 0.0052 | 0.6800 ± 0.0133 |
| DGCNN (bd, DE) | braindecode, DE input | 5 | 0.5749 ± 0.0121 | 0.3412 ± 0.0256 | 0.5844 ± 0.0181 | 0.5836 ± 0.0178 | 0.6524 ± 0.0330 |
| DGCNN | ours (PyTorch, DE input) | 5 | 0.5741 ± 0.0175 | 0.3380 ± 0.0413 | 0.5807 ± 0.0305 | 0.5801 ± 0.0303 | 0.6622 ± 0.0441 |
| EEG Conformer | released | 5 | 0.5360 ± 0.0074 | 0.2640 ± 0.0146 | 0.5244 ± 0.0131 | 0.5256 ± 0.0116 | 0.5591 ± 0.0225 |
| DGCNN (bd, raw) | braindecode, raw input | 5 | 0.5191 ± 0.0047 | 0.2510 ± 0.0106 | 0.5249 ± 0.0089 | 0.5245 ± 0.0087 | 0.6027 ± 0.0247 |
| EEG Conformer (bd) | braindecode | 5 | 0.5259 ± 0.0095 | 0.2445 ± 0.0151 | 0.5071 ± 0.0168 | 0.5114 ± 0.0131 | 0.5547 ± 0.0158 |
| EEGNet | ours (PyTorch) | 5 | 0.4909 ± 0.0089 | 0.2053 ± 0.0203 | 0.4929 ± 0.0124 | 0.4978 ± 0.0158 | 0.5298 ± 0.0310 |
| ATCNet (bd) | braindecode | 5 | 0.4892 ± 0.0184 | 0.1989 ± 0.0265 | 0.4741 ± 0.0252 | 0.4917 ± 0.0173 | 0.5076 ± 0.0234 |
| EEGNet (bd) | braindecode | 5 | 0.4748 ± 0.0158 | 0.1909 ± 0.0235 | 0.4905 ± 0.0183 | 0.4962 ± 0.0140 | 0.5333 ± 0.0319 |
| L-SeqSleepNet | released, off-label | 3 | (30-s windows) | (30-s windows) | (30-s windows) | (30-s windows) | 0.5126 ± 0.0245 |

L-SeqSleepNet (off-label) classifies 30-s windows in sequences of 20; only its clip vote compares.

pre-training: A − B, mean over runs, Welch p

| A − B | bal. acc | κ | clip vote |
|---|---|---|---|
| CBraMod − CBraMod scratch | +0.0290 (p = 0.007) | +0.0390 (p = 0.032) | +0.0169 (p = 0.457) |

foundation models: A − B, mean over runs, Welch p

| A − B | bal. acc | κ | clip vote |
|---|---|---|---|
| CBraMod − LaBraM-Base | -0.0138 (p = 0.087) | -0.0240 (p = 0.125) | -0.0151 (p = 0.498) |

foundation models vs. supervised models: A − B, mean over runs, Welch p

| A − B | bal. acc | κ | clip vote |
|---|---|---|---|
| CBraMod − EEG Conformer | +0.0802 (p < 0.001) | +0.1380 (p < 0.001) | +0.1378 (p < 0.001) |
| CBraMod − EEGNet | +0.1253 (p < 0.001) | +0.1967 (p < 0.001) | +0.1671 (p < 0.001) |
| CBraMod − DGCNN | +0.0421 (p = 0.003) | +0.0640 (p = 0.024) | +0.0347 (p = 0.252) |
| LaBraM-Base − DGCNN | +0.0559 (p = 0.002) | +0.0880 (p = 0.009) | +0.0498 (p = 0.064) |

DGCNN vs. the other supervised models: A − B, mean over runs, Welch p

| A − B | bal. acc | κ | clip vote |
|---|---|---|---|
| DGCNN − EEGNet | +0.0833 (p < 0.001) | +0.1327 (p < 0.001) | +0.1324 (p < 0.001) |
| DGCNN − EEG Conformer | +0.0381 (p = 0.005) | +0.0740 (p = 0.013) | +0.1031 (p = 0.004) |
| DGCNN − ATCNet (bd) | +0.0849 (p < 0.001) | +0.1391 (p < 0.001) | +0.1547 (p < 0.001) |

braindecode − original implementation, same pipeline: A − B, mean over runs, Welch p

| A − B | bal. acc | κ | clip vote |
|---|---|---|---|
| CBraMod (bd) − CBraMod | +0.0036 (p = 0.593) | +0.0049 (p = 0.716) | +0.0053 (p = 0.810) |
| LaBraM-Base (bd) − LaBraM-Base | -0.0139 (p = 0.061) | -0.0279 (p = 0.088) | -0.0187 (p = 0.207) |
| EEG Conformer (bd) − EEG Conformer | -0.0101 (p = 0.099) | -0.0195 (p = 0.072) | -0.0044 (p = 0.729) |
| EEGNet (bd) − EEGNet | -0.0161 (p = 0.093) | -0.0144 (p = 0.329) | +0.0036 (p = 0.863) |
| DGCNN (bd, DE) − DGCNN | +0.0008 (p = 0.935) | +0.0032 (p = 0.887) | -0.0098 (p = 0.703) |
| DGCNN (bd, raw) − DGCNN | -0.0550 (p = 0.001) | -0.0870 (p = 0.008) | -0.0596 (p = 0.037) |

### SEED, subject-independent split (chance: 0.333)

| Model | Implementation | Runs | bal. acc | κ | weighted F1 | accuracy | clip vote |
|---|---|---|---|---|---|---|---|
| LaBraM-Base | released | 5 | 0.5822 ± 0.0105 | 0.3788 ± 0.0159 | 0.5820 ± 0.0103 | 0.5863 ± 0.0107 | 0.6415 ± 0.0193 |
| CBraMod | released | 5 | 0.5442 ± 0.0092 | 0.3211 ± 0.0143 | 0.5415 ± 0.0081 | 0.5479 ± 0.0096 | 0.5852 ± 0.0196 |
| ATCNet (bd) | braindecode | 5 | 0.5410 ± 0.0057 | 0.3165 ± 0.0086 | 0.5186 ± 0.0083 | 0.5461 ± 0.0056 | 0.5881 ± 0.0285 |
| EEG Conformer | released | 5 | 0.5378 ± 0.0105 | 0.3122 ± 0.0157 | 0.5134 ± 0.0112 | 0.5433 ± 0.0102 | 0.5644 ± 0.0062 |
| EEGNet | ours (PyTorch) | 5 | 0.5227 ± 0.0240 | 0.2882 ± 0.0371 | 0.5157 ± 0.0219 | 0.5264 ± 0.0248 | 0.5481 ± 0.0448 |
| DGCNN | ours (PyTorch, DE input) | 5 | 0.5215 ± 0.0139 | 0.2861 ± 0.0210 | 0.5234 ± 0.0126 | 0.5243 ± 0.0142 | 0.5393 ± 0.0379 |
| DeepSleepNet (bd) | braindecode | 5 | 0.5034 ± 0.0293 | 0.2592 ± 0.0444 | 0.5023 ± 0.0301 | 0.5060 ± 0.0299 | 0.5363 ± 0.0580 |
| L-SeqSleepNet | released, off-label | 3 | (30-s windows) | (30-s windows) | (30-s windows) | (30-s windows) | 0.7877 ± 0.0280 |

L-SeqSleepNet (off-label) classifies 30-s windows in sequences of 20; only its clip vote compares. On this split its sequences span clips shown to every subject in the same order, so they can read a clip's identity from its position: the number is not valid.

foundation models: A − B, mean over runs, Welch p

| A − B | bal. acc | κ | clip vote |
|---|---|---|---|
| CBraMod − LaBraM-Base | -0.0380 (p < 0.001) | -0.0577 (p < 0.001) | -0.0563 (p = 0.002) |

foundation models vs. supervised models: A − B, mean over runs, Welch p

| A − B | bal. acc | κ | clip vote |
|---|---|---|---|
| CBraMod − EEG Conformer | +0.0064 (p = 0.337) | +0.0089 (p = 0.378) | +0.0207 (p = 0.076) |
| CBraMod − EEGNet | +0.0215 (p = 0.119) | +0.0329 (p = 0.122) | +0.0370 (p = 0.146) |
| CBraMod − DGCNN | +0.0227 (p = 0.019) | +0.0349 (p = 0.018) | +0.0459 (p = 0.053) |
| LaBraM-Base − DGCNN | +0.0607 (p < 0.001) | +0.0926 (p < 0.001) | +0.1022 (p = 0.002) |

DGCNN vs. the other supervised models: A − B, mean over runs, Welch p

| A − B | bal. acc | κ | clip vote |
|---|---|---|---|
| DGCNN − EEGNet | -0.0012 (p = 0.926) | -0.0021 (p = 0.917) | -0.0089 (p = 0.744) |
| DGCNN − EEG Conformer | -0.0163 (p = 0.072) | -0.0261 (p = 0.060) | -0.0252 (p = 0.213) |
| DGCNN − ATCNet (bd) | -0.0195 (p = 0.031) | -0.0304 (p = 0.028) | -0.0489 (p = 0.053) |

### SEED-V, trial split, as CBraMod on SEED-V (chance: 0.200)

| Model | Implementation | Runs | bal. acc | κ | weighted F1 | accuracy | clip vote |
|---|---|---|---|---|---|---|---|
| LaBraM-Base | released | 5 | 0.4136 ± 0.0094 | 0.2661 ± 0.0119 | 0.4162 ± 0.0092 | 0.4133 ± 0.0105 | 0.4908 ± 0.0242 |
| LaBraM-Base (bd) | braindecode | 5 | 0.4106 ± 0.0062 | 0.2618 ± 0.0085 | 0.4138 ± 0.0074 | 0.4102 ± 0.0072 | 0.5008 ± 0.0130 |
| DeepSleepNet (bd) | braindecode | 5 | 0.3995 ± 0.0030 | 0.2531 ± 0.0040 | 0.4076 ± 0.0035 | 0.4064 ± 0.0037 | 0.4783 ± 0.0157 |
| CBraMod | released | 5 | 0.3962 ± 0.0060 | 0.2501 ± 0.0071 | 0.4019 ± 0.0079 | 0.4056 ± 0.0047 | 0.5050 ± 0.0175 |
| DGCNN | ours (PyTorch, DE input) | 5 | 0.3866 ± 0.0051 | 0.2346 ± 0.0066 | 0.3930 ± 0.0051 | 0.3909 ± 0.0052 | 0.4717 ± 0.0162 |
| CBraMod (bd) | braindecode | 5 | 0.3786 ± 0.0096 | 0.2219 ± 0.0122 | 0.3806 ± 0.0093 | 0.3797 ± 0.0092 | 0.4775 ± 0.0303 |
| DGCNN (bd, DE) | braindecode, DE input | 5 | 0.3752 ± 0.0039 | 0.2194 ± 0.0045 | 0.3793 ± 0.0033 | 0.3779 ± 0.0033 | 0.4742 ± 0.0215 |
| EEG Conformer | released | 5 | 0.3205 ± 0.0137 | 0.1464 ± 0.0196 | 0.3135 ± 0.0229 | 0.3249 ± 0.0170 | 0.3642 ± 0.0337 |
| EEG Conformer (bd) | braindecode | 5 | 0.2962 ± 0.0123 | 0.1134 ± 0.0147 | 0.2696 ± 0.0248 | 0.2966 ± 0.0146 | 0.3358 ± 0.0210 |
| EEGNet | ours (PyTorch) | 5 | 0.2894 ± 0.0094 | 0.1133 ± 0.0097 | 0.2738 ± 0.0115 | 0.3080 ± 0.0064 | 0.3067 ± 0.0260 |
| DGCNN (bd, raw) | braindecode, raw input | 5 | 0.2846 ± 0.0009 | 0.1041 ± 0.0014 | 0.2897 ± 0.0019 | 0.2896 ± 0.0014 | 0.3675 ± 0.0090 |
| EEGNet (bd) | braindecode | 5 | 0.2809 ± 0.0124 | 0.1031 ± 0.0153 | 0.2577 ± 0.0176 | 0.3015 ± 0.0111 | 0.2883 ± 0.0238 |
| CBraMod scratch | released, no pre-training | 5 | 0.2744 ± 0.1019 | 0.0949 ± 0.1300 | 0.2110 ± 0.1674 | 0.2989 ± 0.0890 | 0.3167 ± 0.1602 |
| ATCNet (bd) | braindecode | 5 | 0.2701 ± 0.0043 | 0.0861 ± 0.0040 | 0.2362 ± 0.0074 | 0.2897 ± 0.0026 | 0.2817 ± 0.0140 |
| L-SeqSleepNet | released, off-label | 3 | (30-s windows) | (30-s windows) | (30-s windows) | (30-s windows) | 0.3583 ± 0.1343 |

L-SeqSleepNet (off-label) classifies 30-s windows in sequences of 20; only its clip vote compares.

Against the CBraMod paper (Table 2): ours − paper, Welch p from the two means, SDs and n = 5

| Model | bal. acc: paper / ours / Δ (p) | κ: paper / ours / Δ (p) | weighted F1: paper / ours / Δ (p) |
|---|---|---|---|
| CBraMod | 0.4091 / 0.3962 / -0.0129 (p = 0.041) | 0.2569 / 0.2501 / -0.0068 (p = 0.377) | 0.4101 / 0.4019 / -0.0082 (p = 0.210) |
| LaBraM-Base | 0.3976 / 0.4136 / +0.0160 (p = 0.069) | 0.2386 / 0.2661 / +0.0275 (p = 0.041) | 0.3974 / 0.4162 / +0.0188 (p = 0.020) |
| EEG Conformer | 0.3537 / 0.3205 / -0.0332 (p = 0.003) | 0.1772 / 0.1464 / -0.0308 (p = 0.031) | 0.3487 / 0.3135 / -0.0352 (p = 0.023) |
| EEGNet | 0.2961 / 0.2894 / -0.0067 (p = 0.313) | 0.1006 / 0.1133 / +0.0127 (p = 0.145) | 0.2749 / 0.2738 / -0.0011 (p = 0.871) |

pre-training: A − B, mean over runs, Welch p

| A − B | bal. acc | κ | clip vote |
|---|---|---|---|
| CBraMod − CBraMod scratch | +0.1218 (p = 0.056) | +0.1551 (p = 0.056) | +0.1883 (p = 0.058) |

foundation models: A − B, mean over runs, Welch p

| A − B | bal. acc | κ | clip vote |
|---|---|---|---|
| CBraMod − LaBraM-Base | -0.0174 (p = 0.011) | -0.0160 (p = 0.038) | +0.0142 (p = 0.323) |

foundation models vs. supervised models: A − B, mean over runs, Welch p

| A − B | bal. acc | κ | clip vote |
|---|---|---|---|
| CBraMod − EEG Conformer | +0.0758 (p < 0.001) | +0.1036 (p < 0.001) | +0.1408 (p < 0.001) |
| CBraMod − EEGNet | +0.1068 (p < 0.001) | +0.1368 (p < 0.001) | +0.1983 (p < 0.001) |
| CBraMod − DGCNN | +0.0096 (p = 0.027) | +0.0155 (p = 0.007) | +0.0333 (p = 0.014) |
| LaBraM-Base − DGCNN | +0.0270 (p = 0.001) | +0.0316 (p = 0.002) | +0.0192 (p = 0.185) |

DGCNN vs. the other supervised models: A − B, mean over runs, Welch p

| A − B | bal. acc | κ | clip vote |
|---|---|---|---|
| DGCNN − EEGNet | +0.0972 (p < 0.001) | +0.1213 (p < 0.001) | +0.1650 (p < 0.001) |
| DGCNN − EEG Conformer | +0.0662 (p < 0.001) | +0.0881 (p < 0.001) | +0.1075 (p < 0.001) |
| DGCNN − ATCNet (bd) | +0.1165 (p < 0.001) | +0.1485 (p < 0.001) | +0.1900 (p < 0.001) |

braindecode − original implementation, same pipeline: A − B, mean over runs, Welch p

| A − B | bal. acc | κ | clip vote |
|---|---|---|---|
| CBraMod (bd) − CBraMod | -0.0176 (p = 0.011) | -0.0282 (p = 0.004) | -0.0275 (p = 0.126) |
| LaBraM-Base (bd) − LaBraM-Base | -0.0030 (p = 0.565) | -0.0043 (p = 0.526) | +0.0100 (p = 0.446) |
| EEG Conformer (bd) − EEG Conformer | -0.0243 (p = 0.019) | -0.0330 (p = 0.018) | -0.0283 (p = 0.156) |
| EEGNet (bd) − EEGNet | -0.0085 (p = 0.257) | -0.0101 (p = 0.252) | -0.0183 (p = 0.278) |
| DGCNN (bd, DE) − DGCNN | -0.0114 (p = 0.005) | -0.0152 (p = 0.004) | +0.0025 (p = 0.841) |
| DGCNN (bd, raw) − DGCNN | -0.1021 (p < 0.001) | -0.1304 (p < 0.001) | -0.1042 (p < 0.001) |

### SEED-V, subject-independent split (chance: 0.200)

| Model | Implementation | Runs | bal. acc | κ | weighted F1 | accuracy | clip vote |
|---|---|---|---|---|---|---|---|
| DGCNN | ours (PyTorch, DE input) | 5 | 0.2563 ± 0.0097 | 0.0708 ± 0.0096 | 0.2453 ± 0.0056 | 0.2498 ± 0.0065 | 0.2874 ± 0.0097 |
| ATCNet (bd) | braindecode | 5 | 0.2528 ± 0.0071 | 0.0641 ± 0.0081 | 0.2324 ± 0.0100 | 0.2434 ± 0.0059 | 0.2696 ± 0.0249 |
| EEGNet | ours (PyTorch) | 5 | 0.2468 ± 0.0084 | 0.0555 ± 0.0099 | 0.2116 ± 0.0199 | 0.2387 ± 0.0080 | 0.2607 ± 0.0364 |
| DeepSleepNet (bd) | braindecode | 5 | 0.2421 ± 0.0117 | 0.0544 ± 0.0163 | 0.2283 ± 0.0176 | 0.2443 ± 0.0228 | 0.2593 ± 0.0262 |
| EEG Conformer | released | 5 | 0.2366 ± 0.0089 | 0.0420 ± 0.0116 | 0.1888 ± 0.0196 | 0.2181 ± 0.0120 | 0.2252 ± 0.0084 |
| LaBraM-Base | released | 5 | 0.2214 ± 0.0105 | 0.0257 ± 0.0128 | 0.1973 ± 0.0104 | 0.2196 ± 0.0071 | 0.2193 ± 0.0124 |
| CBraMod | released | 5 | 0.2223 ± 0.0070 | 0.0230 ± 0.0083 | 0.1887 ± 0.0166 | 0.2100 ± 0.0097 | 0.2178 ± 0.0144 |
| L-SeqSleepNet | released, off-label | 3 | (30-s windows) | (30-s windows) | (30-s windows) | (30-s windows) | 0.4074 ± 0.0370 |

L-SeqSleepNet (off-label) classifies 30-s windows in sequences of 20; only its clip vote compares. On this split its sequences span clips shown to every subject in the same order, so they can read a clip's identity from its position: the number is not valid.

foundation models: A − B, mean over runs, Welch p

| A − B | bal. acc | κ | clip vote |
|---|---|---|---|
| CBraMod − LaBraM-Base | +0.0010 (p = 0.868) | -0.0027 (p = 0.708) | -0.0015 (p = 0.866) |

foundation models vs. supervised models: A − B, mean over runs, Welch p

| A − B | bal. acc | κ | clip vote |
|---|---|---|---|
| CBraMod − EEG Conformer | -0.0142 (p = 0.024) | -0.0189 (p = 0.020) | -0.0074 (p = 0.358) |
| CBraMod − EEGNet | -0.0244 (p = 0.001) | -0.0324 (p < 0.001) | -0.0430 (p = 0.056) |
| CBraMod − DGCNN | -0.0340 (p < 0.001) | -0.0478 (p < 0.001) | -0.0696 (p < 0.001) |
| LaBraM-Base − DGCNN | -0.0349 (p < 0.001) | -0.0451 (p < 0.001) | -0.0681 (p < 0.001) |

DGCNN vs. the other supervised models: A − B, mean over runs, Welch p

| A − B | bal. acc | κ | clip vote |
|---|---|---|---|
| DGCNN − EEGNet | +0.0095 (p = 0.135) | +0.0154 (p = 0.037) | +0.0267 (p = 0.180) |
| DGCNN − EEG Conformer | +0.0197 (p = 0.010) | +0.0288 (p = 0.003) | +0.0622 (p < 0.001) |
| DGCNN − ATCNet (bd) | +0.0035 (p = 0.536) | +0.0067 (p = 0.268) | +0.0178 (p = 0.195) |
<!-- /COMMON -->

## Data

| | SEED | SEED-V |
|---|---|---|
| Recordings | 15 subjects × 3 sessions × 15 film clips of about 4 min (positive, neutral, negative; 5 each) | 16 subjects × 3 sessions × 15 clips (happy, sad, neutral, disgust, fear; 3 each) |
| Source files | `Preprocessed_EEG` (200 Hz, 0–75 Hz, clip by clip), 62 channels | `EEG_raw` (Neuroscan `.cnt`, 1,000 Hz), 62 EEG channels after dropping M1, M2, VEO, HEO |
| Preprocessing | `code/preprocess_seed.py`: 0.3–75 Hz per clip (no continuous 200 Hz recording exists), 1-s windows | `code/preprocess_seedv.py`: the released `preprocessing_SEEDV.py` (resample to 200 Hz, 0.3–75 Hz, released clip times, 1-s windows) with paths and workers changed (`# [MOD]`) |
| 1-s samples | 152,730 | 117,744, the paper's count |
| trial split (train / val / test) | 49,365 / 51,750 / 51,615 | 34,432 / 42,960 / 40,352 |
| subject split | 91,638 / 30,546 / 30,546 | 73,590 / 22,077 / 22,077 |
| Test clips (clip vote) | 225 (trial), 135 (subject) | 240 (trial), 135 (subject) |

Checks: SEED's preprocessed signal is in µV (median |x| 12.9 against 9.0 µV in subject 1's raw `.cnt`), which the released loader's
division by 100 assumes. SEED-V holds subject 7's first session twice; `7_1_20180411_repaired.cnt` lacks FC4 and M1, so the released
`drop_channels` fails on it and it cannot give 62 channels: the original file is used (with an explicit `int32` format, which MNE's
detection misses because the file is 34 MB longer than its header says). The clip of every test sample is in `results/test_trials.json`.

## Paper vs. this experiment

### What differs from CBraMod's released SEED-V code, and why

| | Released | Here | Why |
|---|---|---|---|
| `model_for_seedv.Model.forward` | `feats.view(bz, -1)` before a classifier whose first layer rearranges 4-D input | the classifier gets the backbone output, as in every other `model_for_*.py` (`code/emo_models.py`, `# [MOD]`) | **the released forward raises `KeyError` with the default head**, also in CBraMod's own environment |
| SEED | not supported | `code/preprocess_seed.py` writes SEED-V's LMDB format; the released code then runs with `--num_of_classes 3` | the dataset of this experiment |
| Subject split | none | 9 / 3 / 3 and 10 / 3 / 3 | to test across subjects |
| Trainer | released | `code/trainer.py` (`# [MOD]`): test scored after every epoch (recorded, never used to select), test predictions saved instead of a checkpoint, best validation κ starts at −2 | to measure epoch selection; the released start (0) keeps no checkpoint and crashes when validation κ never exceeds 0, as for 3 seeds of CBraMod without pre-training on SEED-V |
| Evaluation batch | 64 | 512 for ATCNet's later runs (`EVAL_BATCH`) | ATCNet is limited by kernel launches and scored slower than it trained; predictions do not depend on the batch |

### Settings we had to choose

| Setting | Value | Why |
|---|---|---|
| Models other than CBraMod | each model with its own classifier, in place of `model_for_seedv.Model` (`code/emo_models.py`) | only the model differs |
| Hyper-parameters | `finetune_main.py` defaults (CBraMod Table 6) for every model; LaBraM with its own README recipe | as every CBraMod-pipeline experiment |
| Seeds | 0–4 | as every CBraMod-pipeline experiment |
| EEG Conformer | the released SEED model (5 heads, Linear(280, 32) head) | the authors' model for 62 × 200 input |
| ATCNet on 1 s | `conv_block_pool_size_2=2` | braindecode otherwise rescales silently to one sliding window |
| DeepSleepNet on 1 s | braindecode's default first layers | the 200-Hz sizes used on Sleep-EDF exceed a 1-s window |
| DGCNN | DE of the window in 5 bands (SEED's), standardised per band with training statistics | the paper's input, computable from any window |

### Run configuration

`finetune_main.py` defaults: 50 epochs, batch 64, AdamW (weight decay 5e-2), lr 1e-4 for the backbone and 5e-4 for everything else,
cosine schedule to 1e-6, gradient clipping 1, label smoothing 0.1, checkpoint at the best validation κ. Environment `bd-labram`
(PyTorch 2.1.2, braindecode 1.8.1; LaBraM in its own `labram` env; `env/pip_freeze.txt`). Hardware: the trial split on 8 Quadro M6000
(mcl-server); the subject split, CBraMod without pre-training and braindecode's CBraMod and Labram on SEED on RTX A6000 (mcl-server2,
shared with other users, one model–dataset–split group per machine). CBraMod takes about 4 h (SEED-V) to 5.5 h (SEED) per run on an M6000.

## Reproducing

From the committed runs, without a GPU (numpy, scipy, scikit-learn):

```bash
code/reproduce.sh collect                      # this experiment's results/ (every pipeline experiment has the same script)
python ../../../../tools/compare_seed.py       # all models
```

Training (SEED and SEED-V are license-gated; see datasets/README.md):

```bash
python code/preprocess_seed.py 8; python code/preprocess_seedv.py 8     # -> $EEG_DATA_ROOT/{SEED,SEED-V}/processed_{trial555,subject}
python code/export_keys.py results/test_trials.json
code/enqueue_all.sh && RUN_QUEUE_DIR=~/repro-seed/queue2 python ../../../../tools/run_queue.py 0 1 2 3 4 5 6 7
```

## Layout

```
code/preprocess_seed.py, preprocess_seedv.py   SEED / SEED-V -> LMDBs in the released SEED-V format (trial and subject splits)
code/emo_models.py       every model on 62 x 200 samples
code/trainer.py          the released Trainer's SEED-V path ([MOD], above)
code/run_emo.py          one run; code/add_jobs.py, enqueue_all.sh: the queue; code/smoke_test.py: parameters and memory
code/collect.py, export_summary.py, export_keys.py    results of a pipeline experiment; code/reproduce.sh
results/test_trials.json the clip of every test sample (for the clip vote)
```

## Provenance

Runs: 2026-11-05 to 11-08 on mcl-server and mcl-server2. Data from the BCMI download links (license-gated; not redistributed).
