# CBraMod on STEW, and the pipeline every model shares there

Mental workload on STEW (Simultaneous Task EEG Workload; W. L. Lim, O. Sourina, L. Wang, IEEE TNSRE 26(11), 2018,
[doi:10.1109/TNSRE.2018.2872924](https://doi.org/10.1109/TNSRE.2018.2872924)): 48 subjects, Emotiv EPOC, 14 channels at 128 Hz, 150 s at
rest and 150 s during the SIMKAP multitasking test, with a 1–9 workload rating after each. This experiment holds the CBraMod runs and the
code every model uses on STEW. Every model runs through the same windows, folds, seeds, training loop and metric code; only the model differs.

| Model | Experiment | Implementation |
|---|---|---|
| CBraMod (pre-trained, and no pre-training) | this experiment | released code and weights |
| LaBraM-Base | [labram/stew-cbramod-pipeline](../../../labram/experiments/stew-cbramod-pipeline) | released code and weights, LaBraM's recipe |
| EEG Conformer | [eeg-conformer/stew-cbramod-pipeline](../../../eeg-conformer/experiments/stew-cbramod-pipeline) | released model (`conformer.py`, 14 channels) |
| EEGNet | [eegnet/stew-cbramod-pipeline](../../../eegnet/experiments/stew-cbramod-pipeline) | our PyTorch EEGNet-8,2 |
| DGCNN | [dgcnn/stew-cbramod-pipeline](../../../dgcnn/experiments/stew-cbramod-pipeline) | ours, on 5-band DE |
| ATCNet, DeepSleepNet | [atcnet](../../../atcnet/experiments/braindecode-stew), [deepsleepnet](../../../deepsleepnet/experiments/braindecode-stew) `/braindecode-stew` | braindecode |
| L-SeqSleepNet | [l-seqsleepnet/stew-offlabel](../../../l-seqsleepnet/experiments/stew-offlabel) | released code, off-label: five 30-s windows per recording |
| dDTF-GL (and ablations: no AXB; the authors' DWT features) | [ddtf-gl/stew-cbramod-pipeline](../../../ddtf-gl/experiments/stew-cbramod-pipeline) | ours (no network: dDTF maps, feature selection, XGBoost), the same windows and folds, no epoch selection |
| CBraMod, LaBraM-Base, EEG Conformer, EEGNet, DGCNN | `braindecode-stew` of each | braindecode 1.8.1 |

No paper among these methods reports STEW. The dataset paper reports 69% accuracy and κ 0.46 for 3 workload levels from the rating scale,
with support vector regression on features chosen by neighbourhood component analysis; its abstract gives neither the cross-validation nor the
cut-offs, so that number is context, not a target.

## The protocol

| | |
|---|---|
| Windows | as specified for this experiment: sample_rate 128 Hz, window_sec 4.0, overlap_ratio 0.25: 512 samples, stride 384, 49 windows per recording, 4,704 in all |
| Signal | each recording high-pass filtered at 0.3 Hz first: Emotiv's raw output rides on an offset of about 4,300 µV; nothing else |
| Sampling rate per model | 128 Hz for every model except CBraMod and LaBraM, whose pre-trained 1-s patches fix 200 Hz: they read the same windows resampled to 200 Hz (800 samples) |
| Tasks | **rest vs. task**: the recording, 2 classes, 48 subjects, 2,352 windows each. **3 rating levels**: 1–3, 4–6, 7–9 of the recording's rating, 45 subjects (no ratings for 5, 24, 42), 2,058 / 1,127 / 1,225 windows. Every subject rated the test above rest; low is almost only rest |
| Folds | `GroupKFold(n_splits=7)` with the subject as group: test subjects per fold 7, 7, 7, 7, 7, 7, 6 (rest vs. task) and 7, 7, 7, 6, 6, 6, 6 (3 levels) |
| Validation | fold k + 1 (mod 7): the checkpoint is chosen on subjects never in training or test; training on the other five folds |
| Seeds | 0–4 per fold; a seed's score is computed on the pooled test predictions of its 7 folds, which cover every window once |

**Does GroupKFold need a random seed?** It takes none: `GroupKFold(n_splits=7)` does not shuffle, and passing `random_state` without
`shuffle=True` raises `ValueError`. It is deterministic in one environment, but **its folds are not portable**: every STEW subject has the
same number of windows, and with such ties the order in which equal groups are assigned depends on the numpy / scikit-learn version. The
same call gives fold 0 = subjects 3, 10, 17, 31, 38, 45, 48 with scikit-learn 1.6.1 / numpy 2.0.2 and 6, 13, 20, 27, 34, 41, 48 with
scikit-learn 1.9.1 / numpy 1.26.4. The folds were therefore computed once and pinned in `results/folds.json`; every run reads them.
Training seeds are still needed (model initialisation and data order): 5 per fold here.

## Results

### CBraMod

| Task | Model | Accuracy | Bal. acc | κ | Weighted F1 | Fold SD (accuracy) |
|---|---|---|---|---|---|---|
| rest vs. task | CBraMod | 0.7998 ± 0.0095 | 0.7998 ± 0.0095 | 0.5997 ± 0.0189 | 0.7993 ± 0.0095 | 0.0472 |
| rest vs. task | CBraMod, no pre-training | 0.6986 ± 0.0061 | 0.6986 ± 0.0061 | 0.3972 ± 0.0122 | 0.6982 ± 0.0060 | 0.0566 |
| 3 rating levels | CBraMod | 0.5661 ± 0.0086 | 0.5150 ± 0.0131 | 0.3147 ± 0.0172 | 0.5510 ± 0.0105 | 0.0556 |
| 3 rating levels | CBraMod, no pre-training | 0.4965 ± 0.0081 | 0.4527 ± 0.0063 | 0.2047 ± 0.0102 | 0.4842 ± 0.0055 | 0.0614 |

- **Pre-training matters most on STEW of any dataset here:** +0.1012 accuracy on rest vs. task and +0.0696 on 3 levels (p < 0.001), with
  a small dataset (about 3,300 training windows per fold).
- **LaBraM-Base is again ahead on rest vs. task** (0.8179, +0.0181, p = 0.022); on 3 levels the foundation models and DGCNN are within
  0.015 of each other.
- **The spread across folds (SD 0.05–0.06 accuracy) is four to eight times the spread across seeds:** which subjects are tested matters more than
  the seed.
- **No model approaches the dataset paper's 69% / κ 0.46 on 3 levels;** the best is 0.574 / 0.332 (braindecode's Labram). The protocols
  differ (cross-subject here; the paper's is not stated in its abstract).

### All models

Every model on the same windows and folds (`python tools/compare_stew.py`).

<!-- COMMON (python tools/compare_stew.py) -->
### STEW, rest vs. task, 48 subjects (chance accuracy 0.500)

| Model | Implementation | Seeds | accuracy | bal. acc | κ | weighted F1 |
|---|---|---|---|---|---|---|
| LaBraM-Base | released | 5 | 0.8179 ± 0.0107 | 0.8179 ± 0.0107 | 0.6359 ± 0.0214 | 0.8177 ± 0.0108 |
| LaBraM-Base (bd) | braindecode | 5 | 0.8153 ± 0.0134 | 0.8153 ± 0.0134 | 0.6307 ± 0.0268 | 0.8149 ± 0.0136 |
| CBraMod | released | 5 | 0.7998 ± 0.0095 | 0.7998 ± 0.0095 | 0.5997 ± 0.0189 | 0.7993 ± 0.0095 |
| CBraMod (bd) | braindecode | 5 | 0.7977 ± 0.0073 | 0.7977 ± 0.0073 | 0.5955 ± 0.0145 | 0.7974 ± 0.0073 |
| DGCNN (bd, DE) | braindecode, DE input | 5 | 0.7833 ± 0.0091 | 0.7833 ± 0.0091 | 0.5666 ± 0.0182 | 0.7821 ± 0.0098 |
| DWT-GL + AXB | DWT statistics with the auxiliary XGBoost | 5 | 0.7817 ± 0.0067 | 0.7817 ± 0.0067 | 0.5634 ± 0.0135 | 0.7812 ± 0.0069 |
| DWT-GL (bundle) | the authors' bundle chain: DWT statistics, no AXB | 5 | 0.7773 ± 0.0038 | 0.7773 ± 0.0038 | 0.5546 ± 0.0076 | 0.7766 ± 0.0038 |
| DGCNN | ours (PyTorch, DE input) | 5 | 0.7747 ± 0.0073 | 0.7747 ± 0.0073 | 0.5493 ± 0.0145 | 0.7735 ± 0.0075 |
| dDTF-GL, no AXB | ours, without the auxiliary XGBoost | 5 | 0.7646 ± 0.0055 | 0.7646 ± 0.0055 | 0.5293 ± 0.0110 | 0.7645 ± 0.0055 |
| dDTF-GL | ours (dDTF maps, XGBoost; no network) | 5 | 0.7623 ± 0.0034 | 0.7623 ± 0.0034 | 0.5247 ± 0.0068 | 0.7622 ± 0.0034 |
| EEG Conformer | released model | 5 | 0.7364 ± 0.0151 | 0.7364 ± 0.0151 | 0.4728 ± 0.0301 | 0.7349 ± 0.0151 |
| EEG Conformer (bd) | braindecode | 5 | 0.7325 ± 0.0111 | 0.7325 ± 0.0111 | 0.4650 ± 0.0222 | 0.7317 ± 0.0115 |
| ATCNet (bd) | braindecode | 5 | 0.7257 ± 0.0054 | 0.7257 ± 0.0054 | 0.4514 ± 0.0109 | 0.7246 ± 0.0058 |
| DeepSleepNet (bd) | braindecode | 5 | 0.7168 ± 0.0100 | 0.7168 ± 0.0100 | 0.4336 ± 0.0200 | 0.7154 ± 0.0100 |
| CBraMod scratch | released, no pre-training | 5 | 0.6986 ± 0.0061 | 0.6986 ± 0.0061 | 0.3972 ± 0.0122 | 0.6982 ± 0.0060 |
| DGCNN (bd, raw) | braindecode, raw input | 5 | 0.6691 ± 0.0039 | 0.6691 ± 0.0039 | 0.3383 ± 0.0077 | 0.6691 ± 0.0038 |
| EEGNet | ours (PyTorch) | 5 | 0.6648 ± 0.0061 | 0.6648 ± 0.0061 | 0.3297 ± 0.0122 | 0.6561 ± 0.0054 |
| EEGNet (bd) | braindecode | 5 | 0.6521 ± 0.0243 | 0.6521 ± 0.0243 | 0.3042 ± 0.0487 | 0.6371 ± 0.0290 |
| L-SeqSleepNet | released, off-label | 3 | 0.7667 ± 0.0150 | 0.7667 ± 0.0150 | 0.5333 ± 0.0300 | 0.7665 ± 0.0149 |

L-SeqSleepNet (off-label, 3 runs) classifies 30-s windows, five per recording in one sequence, not the 4-s windows of the other models; its recording vote is in l-seqsleepnet/experiments/stew-offlabel.

pre-training: A − B, mean over seeds, Welch p

| A − B | accuracy | κ |
|---|---|---|
| CBraMod − CBraMod scratch | +0.1012 (p < 0.001) | +0.2025 (p < 0.001) |

foundation models: A − B, mean over seeds, Welch p

| A − B | accuracy | κ |
|---|---|---|
| LaBraM-Base − CBraMod | +0.0181 (p = 0.022) | +0.0362 (p = 0.022) |

foundation models vs. supervised models: A − B, mean over seeds, Welch p

| A − B | accuracy | κ |
|---|---|---|
| CBraMod − DGCNN | +0.0252 (p = 0.002) | +0.0503 (p = 0.002) |
| CBraMod − EEG Conformer | +0.0634 (p < 0.001) | +0.1269 (p < 0.001) |
| CBraMod − EEGNet | +0.1350 (p < 0.001) | +0.2700 (p < 0.001) |
| LaBraM-Base − DGCNN | +0.0433 (p < 0.001) | +0.0866 (p < 0.001) |

DGCNN vs. the other supervised models: A − B, mean over seeds, Welch p

| A − B | accuracy | κ |
|---|---|---|
| DGCNN − EEG Conformer | +0.0383 (p = 0.002) | +0.0765 (p = 0.002) |
| DGCNN − EEGNet | +0.1098 (p < 0.001) | +0.2196 (p < 0.001) |
| DGCNN − ATCNet (bd) | +0.0489 (p < 0.001) | +0.0979 (p < 0.001) |

dDTF-GL against the other models: A − B, mean over seeds, Welch p

| A − B | accuracy | κ |
|---|---|---|
| dDTF-GL − LaBraM-Base | -0.0556 (p < 0.001) | -0.1112 (p < 0.001) |
| dDTF-GL − CBraMod | -0.0375 (p < 0.001) | -0.0750 (p < 0.001) |
| dDTF-GL − DGCNN | -0.0123 (p = 0.015) | -0.0247 (p = 0.015) |
| dDTF-GL − EEG Conformer | +0.0259 (p = 0.017) | +0.0519 (p = 0.017) |
| dDTF-GL − EEGNet | +0.0975 (p < 0.001) | +0.1950 (p < 0.001) |

dDTF-GL: input and AXB: A − B, mean over seeds, Welch p

| A − B | accuracy | κ |
|---|---|---|
| dDTF-GL − dDTF-GL, no AXB | -0.0023 (p = 0.454) | -0.0046 (p = 0.454) |
| dDTF-GL − DWT-GL (bundle) | -0.0150 (p < 0.001) | -0.0299 (p < 0.001) |
| DWT-GL + AXB − DWT-GL (bundle) | +0.0044 (p = 0.250) | +0.0088 (p = 0.250) |

braindecode − original implementation, same pipeline: A − B, mean over seeds, Welch p

| A − B | accuracy | κ |
|---|---|---|
| CBraMod (bd) − CBraMod | -0.0021 (p = 0.707) | -0.0042 (p = 0.707) |
| LaBraM-Base (bd) − LaBraM-Base | -0.0026 (p = 0.745) | -0.0052 (p = 0.745) |
| EEG Conformer (bd) − EEG Conformer | -0.0039 (p = 0.654) | -0.0078 (p = 0.654) |
| EEGNet (bd) − EEGNet | -0.0128 (p = 0.313) | -0.0255 (p = 0.313) |
| DGCNN (bd, DE) − DGCNN | +0.0086 (p = 0.138) | +0.0173 (p = 0.138) |
| DGCNN (bd, raw) − DGCNN | -0.1055 (p < 0.001) | -0.2111 (p < 0.001) |

### STEW, 3 rating levels, 45 subjects (chance accuracy 0.333)

| Model | Implementation | Seeds | accuracy | bal. acc | κ | weighted F1 |
|---|---|---|---|---|---|---|
| LaBraM-Base (bd) | braindecode | 5 | 0.5739 ± 0.0133 | 0.5263 ± 0.0125 | 0.3317 ± 0.0184 | 0.5649 ± 0.0065 |
| CBraMod | released | 5 | 0.5661 ± 0.0086 | 0.5150 ± 0.0131 | 0.3147 ± 0.0172 | 0.5510 ± 0.0105 |
| LaBraM-Base | released | 5 | 0.5670 ± 0.0082 | 0.5085 ± 0.0150 | 0.3132 ± 0.0182 | 0.5502 ± 0.0147 |
| DWT-GL (bundle) | the authors' bundle chain: DWT statistics, no AXB | 5 | 0.5559 ± 0.0017 | 0.5058 ± 0.0015 | 0.3056 ± 0.0026 | 0.5551 ± 0.0017 |
| DWT-GL + AXB | DWT statistics with the auxiliary XGBoost | 5 | 0.5525 ± 0.0068 | 0.5050 ± 0.0066 | 0.3014 ± 0.0091 | 0.5529 ± 0.0054 |
| DGCNN | ours (PyTorch, DE input) | 5 | 0.5517 ± 0.0074 | 0.5017 ± 0.0068 | 0.2971 ± 0.0110 | 0.5455 ± 0.0072 |
| DGCNN (bd, DE) | braindecode, DE input | 5 | 0.5532 ± 0.0110 | 0.5035 ± 0.0113 | 0.2966 ± 0.0146 | 0.5449 ± 0.0075 |
| CBraMod (bd) | braindecode | 5 | 0.5447 ± 0.0053 | 0.4945 ± 0.0067 | 0.2842 ± 0.0079 | 0.5378 ± 0.0050 |
| DeepSleepNet (bd) | braindecode | 5 | 0.5468 ± 0.0090 | 0.4899 ± 0.0107 | 0.2700 ± 0.0153 | 0.5263 ± 0.0102 |
| ATCNet (bd) | braindecode | 5 | 0.5290 ± 0.0057 | 0.4849 ± 0.0091 | 0.2568 ± 0.0120 | 0.5014 ± 0.0041 |
| dDTF-GL, no AXB | ours, without the auxiliary XGBoost | 5 | 0.5338 ± 0.0021 | 0.4716 ± 0.0024 | 0.2550 ± 0.0037 | 0.5188 ± 0.0024 |
| dDTF-GL | ours (dDTF maps, XGBoost; no network) | 5 | 0.5303 ± 0.0017 | 0.4715 ± 0.0028 | 0.2520 ± 0.0034 | 0.5174 ± 0.0024 |
| EEG Conformer | released model | 5 | 0.5118 ± 0.0173 | 0.4792 ± 0.0129 | 0.2459 ± 0.0229 | 0.4975 ± 0.0173 |
| EEG Conformer (bd) | braindecode | 5 | 0.4896 ± 0.0108 | 0.4573 ± 0.0099 | 0.2124 ± 0.0163 | 0.4747 ± 0.0099 |
| CBraMod scratch | released, no pre-training | 5 | 0.4965 ± 0.0081 | 0.4527 ± 0.0063 | 0.2047 ± 0.0102 | 0.4842 ± 0.0055 |
| DGCNN (bd, raw) | braindecode, raw input | 5 | 0.4876 ± 0.0074 | 0.4361 ± 0.0091 | 0.1814 ± 0.0120 | 0.4718 ± 0.0075 |
| EEGNet | ours (PyTorch) | 5 | 0.4975 ± 0.0200 | 0.4089 ± 0.0260 | 0.1387 ± 0.0459 | 0.4378 ± 0.0302 |
| EEGNet (bd) | braindecode | 5 | 0.4990 ± 0.0144 | 0.4001 ± 0.0188 | 0.1229 ± 0.0335 | 0.4195 ± 0.0211 |
| L-SeqSleepNet | released, off-label | 3 | 0.5407 ± 0.0238 | 0.4852 ± 0.0313 | 0.2715 ± 0.0431 | 0.5311 ± 0.0282 |

L-SeqSleepNet (off-label, 3 runs) classifies 30-s windows, five per recording in one sequence, not the 4-s windows of the other models; its recording vote is in l-seqsleepnet/experiments/stew-offlabel.

pre-training: A − B, mean over seeds, Welch p

| A − B | accuracy | κ |
|---|---|---|
| CBraMod − CBraMod scratch | +0.0696 (p < 0.001) | +0.1100 (p < 0.001) |

foundation models: A − B, mean over seeds, Welch p

| A − B | accuracy | κ |
|---|---|---|
| LaBraM-Base − CBraMod | +0.0009 (p = 0.869) | -0.0016 (p = 0.892) |

foundation models vs. supervised models: A − B, mean over seeds, Welch p

| A − B | accuracy | κ |
|---|---|---|
| CBraMod − DGCNN | +0.0144 (p = 0.023) | +0.0177 (p = 0.096) |
| CBraMod − EEG Conformer | +0.0542 (p < 0.001) | +0.0688 (p < 0.001) |
| CBraMod − EEGNet | +0.0686 (p < 0.001) | +0.1760 (p < 0.001) |
| LaBraM-Base − DGCNN | +0.0153 (p = 0.015) | +0.0161 (p = 0.137) |

DGCNN vs. the other supervised models: A − B, mean over seeds, Welch p

| A − B | accuracy | κ |
|---|---|---|
| DGCNN − EEG Conformer | +0.0399 (p = 0.004) | +0.0511 (p = 0.005) |
| DGCNN − EEGNet | +0.0542 (p = 0.002) | +0.1583 (p = 0.001) |
| DGCNN − ATCNet (bd) | +0.0227 (p < 0.001) | +0.0402 (p < 0.001) |

dDTF-GL against the other models: A − B, mean over seeds, Welch p

| A − B | accuracy | κ |
|---|---|---|
| dDTF-GL − LaBraM-Base | -0.0366 (p < 0.001) | -0.0611 (p = 0.001) |
| dDTF-GL − CBraMod | -0.0357 (p < 0.001) | -0.0627 (p < 0.001) |
| dDTF-GL − DGCNN | -0.0214 (p = 0.002) | -0.0450 (p < 0.001) |
| dDTF-GL − EEG Conformer | +0.0185 (p = 0.075) | +0.0061 (p = 0.588) |
| dDTF-GL − EEGNet | +0.0329 (p = 0.021) | +0.1133 (p = 0.005) |

dDTF-GL: input and AXB: A − B, mean over seeds, Welch p

| A − B | accuracy | κ |
|---|---|---|
| dDTF-GL − dDTF-GL, no AXB | -0.0034 (p = 0.023) | -0.0030 (p = 0.221) |
| dDTF-GL − DWT-GL (bundle) | -0.0255 (p < 0.001) | -0.0536 (p < 0.001) |
| DWT-GL + AXB − DWT-GL (bundle) | -0.0034 (p = 0.335) | -0.0042 (p = 0.368) |

braindecode − original implementation, same pipeline: A − B, mean over seeds, Welch p

| A − B | accuracy | κ |
|---|---|---|
| CBraMod (bd) − CBraMod | -0.0214 (p = 0.002) | -0.0305 (p = 0.013) |
| LaBraM-Base (bd) − LaBraM-Base | +0.0069 (p = 0.360) | +0.0186 (p = 0.147) |
| EEG Conformer (bd) − EEG Conformer | -0.0222 (p = 0.047) | -0.0335 (p = 0.031) |
| EEGNet (bd) − EEGNet | +0.0016 (p = 0.889) | -0.0159 (p = 0.551) |
| DGCNN (bd, DE) − DGCNN | +0.0015 (p = 0.814) | -0.0004 (p = 0.958) |
| DGCNN (bd, raw) − DGCNN | -0.0641 (p < 0.001) | -0.1157 (p < 0.001) |
<!-- /COMMON -->

## Paper vs. this experiment

There is no paper result for these models on STEW; differences from the released pipelines are those of
[finetune-seed](../finetune-seed) (the same `trainer.py`), plus:

| | Here | Why |
|---|---|---|
| Data loading | `code/stew_data.py`: in-memory windows, batches as CBraMod's released loader makes them (shuffled training, x / 100) | 4,704 windows fit in memory; the 7 folds share them |
| CBraMod head | the released `all_patch_reps` head sized for 14 × 4 × 200 features (`code/stew_models.py`) | CBraMod has no STEW head |
| Both tasks | CBraMod's multi-class path (cross-entropy with label smoothing, best validation κ), the 2-class task too | κ and accuracy for every model and task; the released binary path monitors AUROC |
| EEG Conformer | `conformer.py`'s model (10 heads), first head layer sized for 28 tokens | 4-s windows, as the BCI-IV-2a model it was built for |
| EEGNet temporal kernel | 64 samples (0.5 s at 128 Hz, the EEGNet paper's) | the sampling rate of this data |
| Threads | 2 per job (1 for ATCNet) | mcl-server2 is shared; PyTorch's default of one thread per core swamped it |

Run configuration: `finetune_main.py` defaults as on SEED (50 epochs, batch 64, AdamW, lr 1e-4 / 5e-4, cosine, clipping 1, label smoothing
0.1). LaBraM with its own recipe. Hardware: RTX A6000 (mcl-server2, shared), ATCNet on Quadro M6000 (mcl-server). A CBraMod run takes 4–5 min
on an A6000.

## Reproducing

```bash
code/reproduce.sh collect                     # this experiment's results/ (every STEW experiment has the same script)
python ../../../../tools/compare_stew.py      # all models
# training: STEW Dataset.zip from IEEE DataPort (free IEEE account; datasets/README.md)
python code/preprocess_stew.py "<dir with sub*_*.txt>" ~/repro-seed/stew results_rerun/folds.json   # windows.npz; the runs read the pinned
                                              # results/folds.json, not the recomputed one (GroupKFold's ties depend on library versions)
for t in rest-task rating3; do for m in cbramod cbramod-scratch; do python code/add_jobs.py $t $m 0 1 2 3 4; done; done
```

## Layout

```
code/preprocess_stew.py   STEW -> windows.npz (128 and 200 Hz) and the GroupKFold folds
code/stew_data.py         loaders over the windows (train / validation / test of a fold)
code/stew_models.py       every model on 14 x 512 (CBraMod, LaBraM: 14 x 800)
code/run_stew.py          one run; code/add_jobs.py: the queue; code/collect.py, export_summary.py, reproduce.sh
results/folds.json        the pinned folds
```

## Provenance

Runs: 2026-11-06 to 11-08 on mcl-server2 (and ATCNet on mcl-server). Data: IEEE DataPort, doi:10.21227/44r8-ya50 (CC BY), not redistributed.
