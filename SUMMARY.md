# Summary of results

Everything validated so far, as of 2026-11-20: 5 methods (EEG Conformer, CBraMod, LaBraM, ATCNet, DGCNN), 2 sleep-staging
methods (DeepSleepNet, L-SeqSleepNet), 1 method without a neural network (dDTF-GL: connectivity maps, feature selection and
XGBoost; a USC manuscript, run on every dataset, and a tuned version of it, dDTF-GL+), 1 baseline added by us (EEGNet), 8 datasets
with every model on one pipeline per dataset, braindecode's implementations re-run in the same pipelines (Section 6), model size and
compute (Section 3), and the weekly reports of 2026-09-24 to 11-20. "Paper" is the number printed in the paper (cited); "ours" is a
re-run in this repository. Every metric and setting is in [RESULTS.md](RESULTS.md), which is generated from the experiments and is
authoritative if the two disagree.

**Scope: subject-independent protocols only (since 2026-11-19).** Every number below comes from test subjects that no part of
training saw. The subject-dependent experiments (within-subject BCI-IV-2a for EEG Conformer, ATCNet, EEGNet and DGCNN; DGCNN's and
EEG Conformer's within-subject SEED protocols; SEED's and SEED-V's trial split, which tests on unseen clips of training subjects; the
dDTF-GL paper's subject-dependent STEW setting) stay in `methods/`, marked in each method's README, but are no longer reported.

## 1. Findings at a glance

| Method | What we ran (subject-independent) | Verdict | Key numbers |
|---|---|---|---|
| [CBraMod](methods/cbramod) | Released code and weights, 4 public datasets, cross-subject, seeds 0–4 | Does not reproduce: all 12 metrics are below the paper, 9 of them significantly | balanced accuracy −0.015 (PhysioNet-MI) to −0.137 (MentalArithmetic) |
| [LaBraM-Base](methods/labram) | Released code and weights on CBraMod's 4 datasets, splits, seeds and scoring | Reproduces the CBraMod paper's LaBraM-Base numbers on 10 of 12 metrics | ties or beats CBraMod on 3 of the 4 datasets; the best model across subjects on SEED and on STEW |
| [EEG Conformer](methods/eeg-conformer) | Released model in every shared pipeline, including CBraMod's 4 datasets (new), seeds 0–4 | The CBraMod paper's EEG Conformer rows do not reproduce, in either direction | BCI-IV-2a 0.504 vs. 0.470 (p = 0.003); PhysioNet-MI, Mumtaz2016, MentalArithmetic 0.026–0.137 below (p ≤ 0.014) |
| [ATCNet](methods/atcnet) | Authors' code, BCI-IV-2a leave one subject out, 9 folds × 3 seeds with each of today's scripts; braindecode's `ATCNet` in every shared pipeline (new: CBraMod's 4 datasets) | The LOSO number is checkpointed on the held-out subject; with a validation split it drops by 9 pp. Across subjects in the shared pipelines, a 114K-parameter model level with the foundation models | LOSO 69.06% (best of 3: 70.16%) vs. 70.97%, 60.23% with a validation split; PhysioNet-MI 0.614 (2nd of 8), MentalArithmetic 0.695 (1st by mean), Sleep-EDF κ 0.771 |
| [EEGNet](methods/eegnet) (baseline) | EEGNet-8,2 through CBraMod's fine-tuning pipeline and every other shared pipeline | With 2–4K parameters, the best model by mean on BCI-IV-2a | 0.5288, level with DeepSleepNet, EEG Conformer and ATCNet (p ≥ 0.12), above CBraMod (p = 0.021) and LaBraM-Base (p = 0.014) |
| braindecode implementations | `EEGConformer`, `CBraMod`, `Labram`, `EEGNet`, `ATCNet`, `DeepSleepNet`, `DGCNN` swapped into the same pipelines, same seeds; braindecode's pre-trained weights | `Labram`, `EEGNet` and `EEGConformer` reproduce our runs; `CBraMod` (different task head) does on 3 of 4 datasets; `ATCNet` is the same network with its final max-norm over another axis. None is systematically closer to a paper than the released code | CBraMod, PhysioNet-MI: 0.6068 vs. 0.6265 (p = 0.02); LaBraM-Base: all p ≥ 0.12; EEG Conformer on CBraMod's 4 datasets: all p ≥ 0.35; ATCNet LOSO −1.8 pp (p = 0.20); Sleep-EDF κ: all p ≥ 0.36 |
| [DeepSleepNet](methods/deepsleepnet) (sleep) | Released code (TensorFlow 1.15): Sleep-EDF-20 with the paper's 20-fold subject-wise CV; Sleep-EDF-78 on the shared 62 / 8 / 8 split, 3 runs | Reproduces the paper. On Sleep-EDF-78, with a training set 3.5× larger, the released final-model rule overfits | 81.81% / macro-F1 76.72 / κ 0.754 vs. 82.0 / 76.9 / 0.76, on exactly the paper's 41,950 epochs; Sleep-EDF-78: κ 0.703 |
| [L-SeqSleepNet](methods/l-seqsleepnet) (sleep) | Released code and data: Sleep-EDF-20, leave-one-subject-out, 1 of the paper's 5 repetitions; Sleep-EDF-78 on the shared split, 3 runs | Reproduces the paper once the run scripts' test list is fixed (they score every fold on subject 0) | 86.11% / κ 0.811 / macro-F1 79.44 vs. 86.3 ± 0.2 / 0.813 / 79.3; Sleep-EDF-78: κ 0.773, tied for first |
| All models, sleep staging (Sleep-EDF-78) | Every model as the epoch encoder of CBraMod's released sleep-staging pipeline, the two sleep models also with their own code; one 62 / 8 / 8 split, the same 20,976 test epochs (Section 5) | No paper reports this split. Two foundation models, L-SeqSleepNet and ATCNet tie at the top; the two small supervised models trail | κ: LaBraM-Base 0.773, L-SeqSleepNet 0.773, ATCNet 0.771, CBraMod 0.768, DeepSleepNet (bd) 0.762, DeepSleepNet (released) 0.703, EEG Conformer 0.688, EEGNet 0.681 |
| [DGCNN](methods/dgcnn) | Our PyTorch port of the authors' TensorFlow code (numerically identical to braindecode's `DGCNN`); SEED leave one subject out, 5 runs; then every other dataset here | The paper's subject-independent result holds up; on band-power input DGCNN is the best model on Mumtaz2016 | leave one subject out 77.58% vs. 79.95% (linear SVM 58.85%); Mumtaz2016 0.962, above every other model (p ≤ 0.003) |
| All models, emotion (SEED, SEED-V) | Every model through CBraMod's released SEED-V pipeline with test subjects held out (9 / 3 / 3, 10 / 3 / 3), seeds 0–4 | LaBraM-Base leads on SEED; across subjects SEED-V is at chance for every model | SEED: LaBraM-Base 0.582, CBraMod 0.544, ATCNet 0.541; SEED-V: 0.22–0.26 (chance 0.20) |
| All models, mental workload (STEW) | Every model, 7-fold GroupKFold across subjects (folds pinned), 4-s windows, two tasks | Foundation models lead among the published methods; pre-training adds 0.10 | rest vs. task: LaBraM-Base 0.818, CBraMod 0.800, DGCNN 0.775; 3 levels: at most 0.574 (majority class 0.467) |
| [dDTF-GL](methods/ddtf-gl) (no network) | Our implementation of the paper's method (the authors' bundle computes wavelet statistics instead); STEW in the paper's subject-independent setting on Safari et al.'s released dDTF maps, every network on the same windows and splits; then every dataset here | Its subject-independent number reproduces exactly; its claim of the highest accuracy does not: in its own setting both foundation models, DGCNN and EEG Conformer score higher. Elsewhere mid-table on STEW, strong on Mumtaz2016, near chance on motor imagery, last on Sleep-EDF | 0.774 (SD over subjects 0.147) vs. 0.766 ± 0.148; same setting: LaBraM-Base 0.863, CBraMod 0.845, DGCNN 0.805, EEG Conformer 0.784; shared STEW 0.740; Mumtaz2016 0.919 |
| [dDTF-GL+](methods/ddtf-gl/experiments/feature-tuning) (tuned here) | dDTF-GL with its features, feature counts, projection features (LNT, LDA, linear and RBF SVMs) and classifier chosen per dataset on validation data only | Tuning lifts dDTF-GL on 8 of 10 targets (p < 0.001; not on Mumtaz2016, where validation AUROC saturates, nor on BCI-IV-2a); band power added to (or replacing) the dDTF maps does most of it. It ties the best model on STEW, also in the dDTF-GL paper's own setting, and is first by mean on MentalArithmetic. It had a per-dataset search the networks did not | STEW rest vs. task 0.740 → 0.819 (LaBraM-Base 0.818); the paper's setting 0.774 → 0.863 (LaBraM-Base 0.863); MentalArithmetic 0.583 → 0.710; Sleep-EDF κ 0.560 → 0.721; SEED 0.524 → 0.547; PhysioNet-MI 0.327 → 0.490; Mumtaz2016 0.919 → 0.921; BCI-IV-2a 0.328 → 0.335 |

## 2. Results by dataset

Every model run so far on a subject-independent protocol, grouped by task category and dataset. "Paper" is the number the method's
paper reports; "ours" are our re-runs: with the authors' released code, and with braindecode's implementation in the same pipeline.

| Category | Dataset | Subjects | Labels | Popularity: citations · method papers here | Protocol (metric) | Model | Model type | Paper | Ours: released code | Ours: braindecode |
|---|---|---|---|---|---|---|---|---|---|---|
| **Motor imagery classification** | **BCI-IV-2a** | 9 | 4: left hand, right hand, feet, tongue | 1,287 · 4 of 8 ᶠ | leave-one-subject-out, 9 folds (accuracy, %) | ATCNet | domain-specific | 70.97 | 70.16 (best of 3) · 69.06 (mean) / 60.23 ᵇ | 67.36 (1 run) |
|  |  |  |  |  | cross-subject, 5 / 2 / 2 subjects (balanced accuracy) | EEGNet | domain-specific | 0.4482 ᶜ | 0.5288↑ ᵈ | 0.5255↑ |
|  |  |  |  |  |  | EEG Conformer | domain-specific | 0.4696 ᶜ | 0.5042↑ | 0.5080↑ |
|  |  |  |  |  |  | ATCNet | domain-specific | – | – | 0.4898 |
|  |  |  |  |  |  | DeepSleepNet | domain-specific (sleep), off-label | – | – | 0.5250 |
|  |  |  |  |  |  | LaBraM-Base | general | 0.4869 ᶜ | 0.4477 | 0.4375 |
|  |  |  |  |  |  | CBraMod | general | 0.5138 | 0.4571↓ | 0.4703↓ |
|  |  |  |  |  |  | DGCNN | domain-specific | – | 0.3458 | 0.3563 (DE input) · 0.4524 (raw) |
|  |  |  |  |  |  | dDTF-GL | no network ᵒ | – | 0.3276 | – |
|  |  |  |  |  |  | dDTF-GL+ | no network, tuned here ᵠ | – | 0.3347 | – |
|  | **PhysioNet-MI** | 109 | 4: left fist, right fist, both fists, both feet | 500 · 1 of 8 ᶠ | cross-subject, 70 / 19 / 20 subjects (balanced accuracy) | EEGNet | domain-specific | 0.5814 ᶜ | 0.5948 ᵈ | 0.5824 |
|  |  |  |  |  |  | EEG Conformer | domain-specific | 0.6049 ᶜ | 0.5787↓ | 0.5861↓ |
|  |  |  |  |  |  | ATCNet | domain-specific | – | – | 0.6142 |
|  |  |  |  |  |  | DeepSleepNet | domain-specific (sleep), off-label | – | – | 0.5877 |
|  |  |  |  |  |  | LaBraM-Base | general | 0.6173 ᶜ | 0.6098 | 0.6175 |
|  |  |  |  |  |  | CBraMod | general | 0.6417 | 0.6265↓ | 0.6068↓ |
|  |  |  |  |  |  | DGCNN | domain-specific | – | 0.4822 | 0.4914 (DE input) · 0.5781 (raw) |
|  |  |  |  |  |  | dDTF-GL | no network ᵒ | – | 0.3274 | – |
|  |  |  |  |  |  | dDTF-GL+ | no network, tuned here ᵠ | – | 0.4904 | – |
| **Mental disorder diagnosis** | **Mumtaz2016** | 64 (34 MDD, 30 healthy); 63 used ᵉ | 2: major depressive disorder, healthy | 269 · 1 of 8 ᶠ | cross-subject, 43 / 9 / 11 subjects (balanced accuracy) | EEGNet | domain-specific | 0.9232 ᶜ | 0.8828↓ ᵈ | 0.8931 |
|  |  |  |  |  |  | EEG Conformer | domain-specific | 0.9308 ᶜ | 0.8747↓ | 0.8904↓ |
|  |  |  |  |  |  | ATCNet | domain-specific | – | – | 0.9077 |
|  |  |  |  |  |  | DeepSleepNet | domain-specific (sleep), off-label | – | – | 0.8391 |
|  |  |  |  |  |  | LaBraM-Base | general | 0.9409 ᶜ | 0.8905↓ | 0.8894↓ |
|  |  |  |  |  |  | CBraMod | general | 0.9560 | 0.8933↓ | 0.8827↓ |
|  |  |  |  |  |  | DGCNN | domain-specific | – | 0.9619 | 0.9704 (DE input) · 0.8710 (raw) |
|  |  |  |  |  |  | dDTF-GL | no network ᵒ | – | 0.9192 | – |
|  |  |  |  |  |  | dDTF-GL+ | no network, tuned here ᵠ | – | 0.9213 | – |
| **Mental stress detection** | **MentalArithmetic** | 36 | 2: rest, mental arithmetic | 315 · 1 of 8 ᶠ | cross-subject, 28 / 4 / 4 subjects (balanced accuracy) | EEGNet | domain-specific | 0.6770 ᶜ | 0.5569↓ ᵈ | 0.5437↓ |
|  |  |  |  |  |  | EEG Conformer | domain-specific | 0.6805 ᶜ | 0.5438↓ | 0.5500↓ |
|  |  |  |  |  |  | ATCNet | domain-specific | – | – | 0.6951 |
|  |  |  |  |  |  | DeepSleepNet | domain-specific (sleep), off-label | – | – | 0.5917 |
|  |  |  |  |  |  | LaBraM-Base | general | 0.6909 ᶜ | 0.6562 | 0.6451 |
|  |  |  |  |  |  | CBraMod | general | 0.7256 | 0.5889↓ | 0.6042↓ |
|  |  |  |  |  |  | DGCNN | domain-specific | – | 0.6236 | 0.6368 (DE input) · 0.5590 (raw) |
|  |  |  |  |  |  | dDTF-GL | no network ᵒ | – | 0.5833 | – |
|  |  |  |  |  |  | dDTF-GL+ | no network, tuned here ᵠ | – | 0.7097 | – |
| **Sleep staging** | **Sleep-EDF** | 78 (Sleep-EDF-20: the first 20) | 5: W, N1, N2, N3, REM | 966 · 2 of 8 ᶠ | Sleep-EDF-20, 20-fold subject CV (accuracy, %) | DeepSleepNet | domain-specific | 82.0 | 81.81 ᵍ | – |
|  |  |  |  |  | Sleep-EDF-20, leave-one-subject-out (accuracy, %) | L-SeqSleepNet | domain-specific | 86.3 ± 0.2 | 86.11 ᵍ | – ʰ |
|  |  |  |  |  | Sleep-EDF-78, cross-subject, 62 / 8 / 8 subjects, common test set (κ) ⁱ | EEGNet | domain-specific | – | 0.6815 ᵈ | 0.6808 |
|  |  |  |  |  |  | EEG Conformer | domain-specific | – | 0.6876 | 0.7011 |
|  |  |  |  |  |  | ATCNet | domain-specific | – | – | 0.7709 |
|  |  |  |  |  |  | DeepSleepNet | domain-specific | – | 0.7030 ⁱ | 0.7625 |
|  |  |  |  |  |  | L-SeqSleepNet | domain-specific | – | 0.7725 | – ʰ |
|  |  |  |  |  |  | LaBraM-Base | general | – | 0.7727 | 0.7773 |
|  |  |  |  |  |  | CBraMod | general | – | 0.7676 | 0.7677 |
|  |  |  |  |  |  | DGCNN | domain-specific | – | 0.6941 | 0.5956 (raw) |
|  |  |  |  |  |  | dDTF-GL | no network ᵒ | – | 0.5595 | – |
|  |  |  |  |  |  | dDTF-GL+ | no network, tuned here ᵠ | – | 0.7207 | – |
| **Emotion recognition** | **SEED** | 15, 3 sessions each | 3: positive, neutral, negative | 2,735 · 2 of 8 ᶠ | leave one subject out (accuracy, %) | DGCNN | domain-specific | 79.95 | 77.58 | – |
|  |  |  |  |  | CBraMod's SEED-V pipeline, subject split 9 / 3 / 3 (balanced accuracy) | EEGNet | domain-specific | – | 0.5227 ᵈ | – |
|  |  |  |  |  |  | EEG Conformer | domain-specific | – | 0.5378 | – |
|  |  |  |  |  |  | ATCNet | domain-specific | – | – | 0.5410 |
|  |  |  |  |  |  | DeepSleepNet | domain-specific (sleep), off-label | – | – | 0.5034 |
|  |  |  |  |  |  | L-SeqSleepNet | domain-specific (sleep) | – | not valid ʳ | – ʰ |
|  |  |  |  |  |  | DGCNN | domain-specific | – | 0.5215 | – |
|  |  |  |  |  |  | LaBraM-Base | general | – | 0.5822 | – |
|  |  |  |  |  |  | CBraMod | general | – | 0.5442 | – |
|  |  |  |  |  |  | dDTF-GL | no network ᵒ | – | 0.5237 | – |
|  |  |  |  |  |  | dDTF-GL+ | no network, tuned here ᵠ | – | 0.5473 | – |
|  | **SEED-V** | 16, 3 sessions each | 5: happy, sad, neutral, disgust, fear | 486 · 2 of 8 ᶠ | CBraMod's SEED-V pipeline, subject split 10 / 3 / 3 (balanced accuracy) | EEGNet | domain-specific | – | 0.2468 ᵈ | – |
|  |  |  |  |  |  | EEG Conformer | domain-specific | – | 0.2366 | – |
|  |  |  |  |  |  | ATCNet | domain-specific | – | – | 0.2528 |
|  |  |  |  |  |  | DeepSleepNet | domain-specific (sleep), off-label | – | – | 0.2421 |
|  |  |  |  |  |  | L-SeqSleepNet | domain-specific (sleep) | – | not valid ʳ | – ʰ |
|  |  |  |  |  |  | DGCNN | domain-specific | – | 0.2563 | – |
|  |  |  |  |  |  | LaBraM-Base | general | – | 0.2214 | – |
|  |  |  |  |  |  | CBraMod | general | – | 0.2223 | – |
|  |  |  |  |  |  | dDTF-GL | no network ᵒ | – | 0.2315 | – |
|  |  |  |  |  |  | dDTF-GL+ | no network, tuned here ᵠ | – | 0.2578 | – |
| **Mental workload** | **STEW** | 48 | 2: rest, multitasking test; or 3 rating levels | 219 · 0 of 8 ᶠ | 7-fold GroupKFold across subjects, rest vs. task, 48 subjects (accuracy; tests between models: Section 5) | EEGNet | domain-specific | – | 0.6648 ᵈ | 0.6521 |
|  |  |  |  |  |  | EEG Conformer | domain-specific | – | 0.7364 | 0.7325 |
|  |  |  |  |  |  | ATCNet | domain-specific | – | – | 0.7257 |
|  |  |  |  |  |  | DeepSleepNet | domain-specific (sleep), off-label | – | – | 0.7168 |
|  |  |  |  |  |  | L-SeqSleepNet | domain-specific (sleep) | – | 0.7667 (off-label: 30-s windows) | – ʰ |
|  |  |  |  |  |  | DGCNN | domain-specific | – | 0.7747 | 0.7833 (DE input) · 0.6691 (raw) |
|  |  |  |  |  |  | LaBraM-Base | general | – | 0.8179 | 0.8153 |
|  |  |  |  |  |  | CBraMod | general | – | 0.7998 | 0.7977 |
|  |  |  |  |  |  | dDTF-GL | no network ᵒ | – | 0.7398 | – |
|  |  |  |  |  |  | dDTF-GL+ | no network, tuned here ᵠ | – | 0.8191 | – |
|  |  |  |  |  | 7-fold GroupKFold across subjects, 3 rating levels, 45 subjects (accuracy) | EEGNet | domain-specific | – | 0.4975 ᵈ | 0.4990 |
|  |  |  |  |  |  | EEG Conformer | domain-specific | – | 0.5118 | 0.4896 |
|  |  |  |  |  |  | ATCNet | domain-specific | – | – | 0.5290 |
|  |  |  |  |  |  | DeepSleepNet | domain-specific (sleep), off-label | – | – | 0.5468 |
|  |  |  |  |  |  | L-SeqSleepNet | domain-specific (sleep) | – | 0.5407 (off-label) | – ʰ |
|  |  |  |  |  |  | DGCNN | domain-specific | – | 0.5517 | 0.5532 (DE input) · 0.4876 (raw) |
|  |  |  |  |  |  | LaBraM-Base | general | – | 0.5670 | 0.5739 |
|  |  |  |  |  |  | CBraMod | general | – | 0.5661 | 0.5447 |
|  |  |  |  |  |  | dDTF-GL | no network ᵒ | – | 0.5185 | – |
|  |  |  |  |  |  | dDTF-GL+ | no network, tuned here ᵠ | – | 0.5636 | – |
|  |  |  |  |  | the dDTF-GL paper's subject-independent setting: 7 subject folds, 6-s windows, no validation (accuracy; networks: final epoch) | EEGNet | domain-specific | 0.683 ᵘ | 0.6647 ᵈ | – |
|  |  |  |  |  |  | EEG Conformer | domain-specific | – | 0.7842 | – |
|  |  |  |  |  |  | ATCNet | domain-specific | – | – | 0.7675 |
|  |  |  |  |  |  | DeepSleepNet | domain-specific (sleep), off-label | – | – | 0.7408 |
|  |  |  |  |  |  | DGCNN | domain-specific | – | 0.8046 | – |
|  |  |  |  |  |  | LaBraM-Base | general | – | 0.8626 | – |
|  |  |  |  |  |  | CBraMod | general | – | 0.8451 | – |
|  |  |  |  |  |  | dDTF-GL | no network ᵒ | 0.766 | 0.7744 ᵖ | – |
|  |  |  |  |  |  | dDTF-GL, XGBoost on the GPU | no network ᵒ | 0.765 | 0.7726 ᵖ | – |
|  |  |  |  |  |  | dDTF-GL+ | no network, tuned here ᵠ | – | 0.8632 (STEW 4-s configuration) | – |

- **Model type.** *Domain-specific*: a supervised model trained from scratch on the task's own data (EEGNet, EEG Conformer, ATCNet,
  DGCNN, and the sleep models DeepSleepNet and L-SeqSleepNet, here also run off-label). *General*: an EEG foundation model,
  pre-trained on large unlabeled EEG corpora and fine-tuned for each task (LaBraM-Base, CBraMod). *No network*: dDTF-GL.
- **↓ / ↑**: significantly below / above the paper (p < 0.05); no mark: within noise. Welch's t-test, 5 seeds against the paper's
  mean ± SD over 5 seeds (LOSO: not tested, the paper gives no per-subject numbers).
- **Comparisons hold within a protocol block only.** Each cross-subject block ran every model through one pipeline (same samples,
  splits, seeds, checkpoint rule and metric code; tests between models in Section 5). LOSO, Sleep-EDF-20 and the dDTF-GL paper's
  setting are the papers' own protocols.
- ᵇ ATCNet LOSO: x / y = checkpoint on the held-out subject (`main_TrainTest.py`, the paper's) / on 20% of the 8 training subjects'
  trials (`main_TrainValTest.py`), mean of 3 seeds. ᶜ From the CBraMod paper's baseline rows (LaBraM's and EEG Conformer's papers do
  not evaluate these datasets). ᵈ EEGNet has no released PyTorch code: this column is our own PyTorch EEGNet-8,2. ᵉ 34 patients and 30
  healthy controls per the dataset's paper (Mumtaz et al. 2017); the data after CBraMod's preprocessing hold 63 (33 MDD, 30 healthy).
  ᶠ Next table. ᵍ One run, not tested against the paper: one 20-fold cross-validation (DeepSleepNet, which reports no spread) and one
  of L-SeqSleepNet's five repetitions. ʰ Not in braindecode. ⁱ The models and braindecode's DeepSleepNet ran as the epoch encoder of
  CBraMod's released sleep-staging pipeline (seeds 0–4); the "released code" cells of DeepSleepNet and L-SeqSleepNet are their own
  released training on the same split (3 runs). Every model is scored on the same 20,976 test epochs.
- ᵒ dDTF-GL: our implementation of the paper's method; the authors' code bundle computes wavelet statistics instead. Its maps: on
  STEW in the paper's setting Safari et al.'s released SIFT maps (the paper's stated source); elsewhere our generator with the
  settings calibrated on those maps. Run through each shared pipeline with the same samples, splits and seeds, without epoch
  selection, and on Sleep-EDF with each epoch classified alone ([methods/ddtf-gl](methods/ddtf-gl)). ᵖ STEW, the dDTF-GL paper's
  setting ([stew-paper-protocol](methods/ddtf-gl/experiments/stew-paper-protocol)): every model on the same windows and splits; the
  GPU row is the paper's "Our Method (GPU)". ᵘ The dDTF-GL paper's own EEGNet on raw EEG (its Tables 1–2).
- ᵠ dDTF-GL+: dDTF-GL with its feature blocks (dDTF maps, graph summaries of them, band power, per-second band power), feature counts,
  projection features (LNT, LDA, linear and RBF SVMs) and classifier settings chosen per dataset on validation data only; the chosen
  configuration on seeds 0–4 ([feature-tuning](methods/ddtf-gl/experiments/feature-tuning); Section 3).
- ʳ L-SeqSleepNet on SEED / SEED-V is off-label: it classifies 30-s windows of three channels, and on the subject split its sequences
  span clips shown to every subject in the same order, so it can read a clip's identity from its position: its numbers are not valid
  ([l-seqsleepnet/seed-offlabel](methods/l-seqsleepnet/experiments/seed-offlabel)).

### How widely each dataset is used

Two objective measures, retrieved on 2026-10-23 (Sleep-EDF: 2026-11-01; SEED, SEED-V, STEW: 2026-11-08): citations of the publication each dataset asks its users to cite
([OpenAlex](https://openalex.org) / [Semantic Scholar](https://www.semanticscholar.org); Google Scholar, which counts
more sources, is typically higher), and how many of the eight method papers in this repository evaluate on it
(Section 4). The headline number in the table above is the most dataset-specific of the citation counts.

| Dataset | Publication users are asked to cite | Citations, OpenAlex / Semantic Scholar | Dataset-specific count | Method papers here (of 8) |
|---|---|---|---|---|
| BCI-IV-2a | Tangermann et al. 2012, *Review of the BCI Competition IV*, Front. Neurosci. 6:55 ([doi](https://doi.org/10.3389/fnins.2012.00055)); the data set's own description (Brunner et al. 2008) is a technical report that OpenAlex does not index | **1,287** / 1,405, shared by all five BCI IV data sets | 1,032 works name the dataset in their title or abstract (OpenAlex, any of 8 spellings, e.g. "BCI Competition IV 2a", "BCIC IV 2a"); 306 of the review's citers mention "2a" there | 4: EEGNet, EEG Conformer, ATCNet, CBraMod |
| PhysioNet-MI | the dataset DOI (Schalk 2009, [doi](https://doi.org/10.13026/C28G6P)) and the BCI2000 paper (Schalk et al. 2004, IEEE TBME, [doi](https://doi.org/10.1109/TBME.2004.827072)) | dataset DOI 16 / –; BCI2000 3,032 / 3,266, which also counts every other use of the BCI2000 software | **500** works cite both BCI2000 and PhysioNet's standard reference (Goldberger et al. 2000); 261 name the dataset in their title or abstract (any of 7 spellings, e.g. "EEG Motor Movement/Imagery", "eegmmidb") | 1: CBraMod |
| Mumtaz2016 | the figshare dataset (Mumtaz 2016, [doi](https://doi.org/10.6084/m9.figshare.4244171)) and the paper whose data statement points to it (Mumtaz et al. 2017, PLoS ONE 12:e0171409, [doi](https://doi.org/10.1371/journal.pone.0171409)) | dataset 34 / 77; paper **269** / 236 | – (the name is too common to count mentions) | 1: CBraMod |
| MentalArithmetic | Zyma et al. 2019, *Electroencephalograms during Mental Arithmetic Task Performance*, Data 4:14 ([doi](https://doi.org/10.3390/data4010014)), the dataset's descriptor | **315** / 332 | – (the name is too common to count mentions) | 1: CBraMod |
| Sleep-EDF | Kemp et al. 2000, *Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG*, IEEE TBME 47:1185 ([doi](https://doi.org/10.1109/10.867928)), the reference PhysioNet's `sleep-edfx` page asks for | **966** / 1,089 | 761 works name the dataset in their title or abstract (OpenAlex, any of 6 spellings, e.g. "Sleep-EDF", "SleepEDF", "Sleep-EDFx") | 2: DeepSleepNet, L-SeqSleepNet |
| SEED | Zheng & Lu 2015, *Investigating Critical Frequency Bands and Channels for EEG-Based Emotion Recognition with Deep Neural Networks*, IEEE TAMD 7:162 ([doi](https://doi.org/10.1109/TAMD.2015.2431497)) | 2,735 (OpenAlex) | **2,735** (the paper is the dataset's reference) | 2 (EEG Conformer, DGCNN) |
| SEED-V | Liu et al. 2022, *Comparing Recognition Performance and Robustness of Multimodal Deep Learning Models for Multimodal Emotion Recognition*, IEEE TCDS 14:715 ([doi](https://doi.org/10.1109/TCDS.2021.3071170)) | 486 (OpenAlex) | **486** | 2 (LaBraM, appendix; CBraMod) |
| STEW | Lim, Sourina & Wang 2018, *STEW: Simultaneous Task EEG Workload Data Set*, IEEE TNSRE 26:2106 ([doi](https://doi.org/10.1109/TNSRE.2018.2872924)) | 219 (OpenAlex) | **219** | 0 |

SEED's reference paper is the most cited (2,735), ahead of BCI-IV-2a, which four of the eight method papers here evaluate on. Sleep-EDF is
third (966 citations, 761 works naming it), the standard benchmark of sleep staging. The other three are in the same
range, a few hundred works each. Citation counts are a proxy: a citation
is not always a use of the data and a use is not always cited, and the dataset DOIs themselves are cited far less than
the papers (16 and 34 in OpenAlex).

Every metric, setting and standard deviation is in [RESULTS.md](RESULTS.md); κ, weighted F1, AUROC and AUC-PR in the
experiment READMEs. Which datasets each method's own paper uses, and which of them we could not run, is in Section 4.

## 3. Methods by characteristics

| | EEGNet | EEG Conformer | ATCNet | LaBraM | CBraMod | DeepSleepNet | L-SeqSleepNet |
|---|---|---|---|---|---| --- | --- |
| Paper | Lawhern et al., J. Neural Eng. 2018 | Song et al., IEEE TNSRE 2023 | Altaheri et al., IEEE TII 2023 | Jiang et al., ICLR 2024 | Wang et al., ICLR 2025 | Supratak et al., IEEE TNSRE 2017 | Phan et al., IEEE JBHI 2023 |
| Family | task-specific, supervised | task-specific, supervised | task-specific, supervised | EEG foundation model | EEG foundation model | task-specific (sleep staging), supervised | task-specific (sleep staging), supervised |
| Architecture | compact CNN: temporal, depthwise spatial and separable convolutions | CNN (temporal + spatial convolution, pooling) → 6 self-attention blocks → MLP head | EEGNet-like conv block → 5 overlapping windows, each through multi-head self-attention and a dilated causal TCN; window outputs averaged | Transformer over channel patches, with a vector-quantized neural tokenizer | criss-cross Transformer: spatial and temporal attention in parallel over channel patches | two CNN paths (small and large filters) on the raw 30-s epoch → two BiLSTM layers across 25 epochs, with a residual shortcut | per epoch, a spectrogram through a learnable filterbank, a BiLSTM and attention → BiLSTMs within and across subsequences of a 200-epoch sequence |
| Pre-training | none | none | none | ~2,500 h from ~20 datasets; predicts the neural codes of masked patches | TUEG (TUH EEG Corpus); reconstructs masked patches | none (its two-step training pre-trains the CNNs on the same training data) | none (a variant pre-trained on SHHS is in the paper) |
| Parameters | 2,193–4,020 (CBraMod pipeline, by dataset) | 789,816 (BCI-IV-2a) | 113,732 (paper: 115.2K, incl. BatchNorm statistics) | 5.8M (Base) | backbone 4.88M (paper: 4.0M); 19–46M as fine-tuned, with the default head | not counted (braindecode's single-epoch version with 2 channels: 40.5M) | not counted |
| Released weights | – | – (trained per subject) | per subject: today's code (1 seed); the paper's best-of-10 models in the repository history | Base only; Large and Huge not released | yes | – | SHHS-pretrained weights |
| Paper's own benchmarks | P300, ERN, MRCP, SMR (BCI-IV-2a) | BCI-IV-2a, BCI-IV-2b, SEED | BCI-IV-2a | TUAB, TUEV (+ SEED-V, MoBI in the appendix) | 13 downstream datasets | MASS, Sleep-EDF (20 subjects) | SHHS, Sleep-EDF (20 subjects), ear-EEG, cEEGrid |
| Evaluation protocol | within-subject (4-fold block-wise cross-validation) and cross-subject | within-subject, session T → E | within-subject, session T → E; leave-one-subject-out | cross-subject | cross-subject: subject-independent train / val / test | subject-wise cross-validation (Sleep-EDF: 20 folds), predictions pooled | leave-one-subject-out with held-out validation subjects, 5 repetitions (SHHS: 70 / 30 split) |
| Epoch selection in the released code | lowest validation loss (paper) | maximum test accuracy over 2000 epochs; no validation set | maximum test accuracy (the test session is the validation data), then the best of 10 runs by test accuracy; `main_TrainValTest.py`: lowest validation loss on 20% of session T | best validation accuracy (the paper states κ / AUROC) | best validation κ (multi-class) or AUROC (binary) | none: the final model after 200 fine-tuning epochs; validation is printed, not used | best validation accuracy, every 100 steps (the paper adds early stopping after 50 validations; the scripts run a fixed 110) |
| Seeds in the paper | – | not reported; a random seed every run | not reported; 10 runs per subject (no seed set in 2022, seeds 1–10 today) | 5, not named | 5, not named | not reported; none set | not reported; none set (5 repetitions) |
| Role here | baseline we added | method validated | method validated | method validated; CBraMod's strongest baseline | method validated | method validated (sleep model) | method validated (sleep model) |
| braindecode class (v1.8.1) | `EEGNet` | `EEGConformer`: same layers, attention scaled by 1/√(head dim) instead of 1/√(embedding size) | `ATCNet`: same layers and weights; identical predictions with Keras' BatchNorm eps (1e-3; braindecode: 1e-4 / 1e-5); its final `MaxNormLinear` limits the norm per input feature, not per output unit | `Labram`: identical model and weights | `CBraMod`: identical encoder and weights; task head = pre-training `proj_out` + one linear layer, not the released 3-layer MLP | `DeepSleepNet`: single-epoch (its BiLSTM sees one epoch), standard LSTM gates instead of peepholes | – (not in braindecode) |
| Our experiment | our own PyTorch re-implementation through CBraMod's pipeline: [pytorch-cbramod-splits](methods/eegnet/experiments/pytorch-cbramod-splits), Sleep-EDF: [sleepedf-cbramod-pipeline](methods/eegnet/experiments/sleepedf-cbramod-pipeline) | [cbramod-splits](methods/eeg-conformer/experiments/cbramod-splits), Sleep-EDF: [sleepedf-cbramod-pipeline](methods/eeg-conformer/experiments/sleepedf-cbramod-pipeline) | [bciciv2a-model-selection](methods/atcnet/experiments/bciciv2a-model-selection) (its leave-one-subject-out runs) | [finetune-cbramod-splits](methods/labram/experiments/finetune-cbramod-splits), Sleep-EDF: [sleepedf-cbramod-pipeline](methods/labram/experiments/sleepedf-cbramod-pipeline) | [finetune-public-datasets](methods/cbramod/experiments/finetune-public-datasets), Sleep-EDF: [finetune-sleepedf](methods/cbramod/experiments/finetune-sleepedf) | [sleepedf20-20fold](methods/deepsleepnet/experiments/sleepedf20-20fold), [sleepedf78-cbramod-split](methods/deepsleepnet/experiments/sleepedf78-cbramod-split) | [sleepedf20-loso](methods/l-seqsleepnet/experiments/sleepedf20-loso), [sleepedf78-cbramod-split](methods/l-seqsleepnet/experiments/sleepedf78-cbramod-split) |
| braindecode experiment | [braindecode-cbramod-splits](methods/eegnet/experiments/braindecode-cbramod-splits) (cross-subject), [braindecode-sleepedf](methods/eegnet/experiments/braindecode-sleepedf) | [cbramod-splits](methods/eeg-conformer/experiments/cbramod-splits) (`bd-conformer`), [braindecode-sleepedf](methods/eeg-conformer/experiments/braindecode-sleepedf) | [braindecode-cbramod-splits](methods/atcnet/experiments/braindecode-cbramod-splits), [braindecode-bciciv2a](methods/atcnet/experiments/braindecode-bciciv2a) (LOSO), Sleep-EDF: [braindecode-sleepedf](methods/atcnet/experiments/braindecode-sleepedf) | [braindecode-cbramod-splits](methods/labram/experiments/braindecode-cbramod-splits), [braindecode-sleepedf](methods/labram/experiments/braindecode-sleepedf) | [braindecode-public-datasets](methods/cbramod/experiments/braindecode-public-datasets), [braindecode-sleepedf](methods/cbramod/experiments/braindecode-sleepedf) | [braindecode-sleepedf](methods/deepsleepnet/experiments/braindecode-sleepedf) (in CBraMod's sleep-staging pipeline), [braindecode-cbramod-splits](methods/deepsleepnet/experiments/braindecode-cbramod-splits) | – |

Parameter counts are from the method READMEs and run logs; every model's parameters and FLOPs in every pipeline are in
[Model size and compute](#model-size-and-compute) below. The "Evaluation protocol" row describes each paper; this repository reports
only the subject-independent ones (scope note at the top).

### Model size and compute

Inference FLOPs per sample (MFLOPs; a sample is one trial or window, on Sleep-EDF one 30-s epoch), counted by
`python tools/count_flops.py` (output `tools/flops.json`) with PyTorch's FLOP counter, the counter of the CBraMod report: 2 FLOPs per
multiply-add of every matmul, convolution and attention product. LSTMs and DGCNN's FFT are added analytically; DeepSleepNet and
L-SeqSleepNet as released (TensorFlow) and dDTF-GL's maps are counted analytically from their layer sizes and MVAR orders;
dDTF-GL+ from the configuration and feature counts of its final runs (maps if used, band-power FFTs, projections, trees; the RBF
SVM bounded above by every training sample as a support vector). Each model
is the one in its pipeline (Sleep-EDF: with the shared 20-epoch sequence head, per epoch). Counted with gradients enabled: in eval
mode without them `nn.MultiheadAttention` takes a fused path the counter does not see (0 FLOPs for attention); so counted, CBraMod's
backbone on 16 channels × 10 s gives 1,596M, as in the 2026-10-01 report, against 318.9M in its paper (thop).

| Model | BCI-IV-2a | PhysioNet-MI | Mumtaz2016 | MentalArithmetic | STEW | SEED | SEED-V | Sleep-EDF-78 |
|---|---|---|---|---|---|---|---|---|
| LaBraM-Base | 1,104 | 3,601 | 1,197 | 1,264 | 689 | 765 | 765 | 741 |
| CBraMod | 900 | 2,668 | 978 | 1,030 | 570 | 645 | 645 | 614 |
| EEGNet | 29.0 | 83.9 | 31.3 | 32.9 | 7.7 | 20.4 | 20.4 | 30.5 |
| EEG Conformer | 103 | 272 | 116 | 121 | 43.4 | 58.4 | 58.4 | 326 |
| ATCNet (bd) | 43.1 | 114 | 47.4 | 49.5 | 16.8 | 29.2 | 29.2 | 79.5 |
| DeepSleepNet (bd) | 544 | 1,550 | 564 | 593 | 238 | 398 | 398 | 146 |
| DeepSleepNet (released) | – | – | – | – | – | – | – | 103 |
| L-SeqSleepNet (released) | – | – | – | – | – | – | – | 4.0 |
| DGCNN | 1.3 | 3.8 | 1.2 | 1.3 | 0.59 | 2.2 | 2.2 | 6.6 |
| dDTF-GL | 146 | 756 | 262 | 261 | 20.4 | 580 | 580 | 4.4 |
| dDTF-GL+ (tuned per dataset) | 146 | 2.2 | 262 | 2.8 | 20.7 | 0.24 | 580 | 5.8 |

Parameters (with each pipeline's task head; Sleep-EDF: with the sequence head, which alone has 4.7M):

| Model | BCI-IV-2a | PhysioNet-MI | Mumtaz2016 | MentalArithmetic | STEW | SEED | SEED-V | Sleep-EDF-78 |
|---|---|---|---|---|---|---|---|---|
| LaBraM-Base | 5.82M | 5.82M | 5.82M | 5.82M | 5.82M | 5.82M | 5.82M | 9.18M |
| CBraMod | 19.1M | 46M | 24.1M | 25.1M | 14M | 15M | 15M | 14.2M |
| EEGNet | 3.35K | 4.02K | 2.19K | 2.21K | 1.84K | 2.67K | 2.87K | 4.69M |
| EEG Conformer | 646K | 713K | 785K | 786K | 439K | 229K | 229K | 11.3M |
| ATCNet (bd) | 114K | 115K | 113K | 113K | 74.5K | 115K | 115K | 3.35M |
| DeepSleepNet (bd) | 110M | 303M | 109M | 114M | 54.9M | 90.3M | 90.3M | 44.2M |
| DGCNN | 365K | 1.06M | 315K | 332K | 233K | 1.02M | 1.02M | 3.24M |

Training compute of one run, PFLOPs ≈ 3 × forward FLOPs × training samples × 50 epochs (one fold on STEW; Sleep-EDF: 153,100
training epochs):

| Model | BCI-IV-2a | PhysioNet-MI | Mumtaz2016 | MentalArithmetic | STEW | SEED | SEED-V | Sleep-EDF-78 |
|---|---|---|---|---|---|---|---|---|
| LaBraM-Base | 0.48 | 3.40 | 0.88 | 0.25 | 0.35 | 10.52 | 8.45 | 17.02 |
| CBraMod | 0.39 | 2.52 | 0.72 | 0.21 | 0.29 | 8.86 | 7.12 | 14.11 |
| EEGNet | 0.013 | 0.079 | 0.023 | 0.007 | 0.004 | 0.28 | 0.23 | 0.70 |
| EEG Conformer | 0.044 | 0.26 | 0.085 | 0.024 | 0.022 | 0.80 | 0.65 | 7.48 |
| ATCNet (bd) | 0.019 | 0.11 | 0.035 | 0.010 | 0.009 | 0.40 | 0.32 | 1.83 |
| DeepSleepNet (bd) | 0.23 | 1.46 | 0.41 | 0.12 | 0.12 | 5.47 | 4.39 | 3.35 |
| DeepSleepNet (released) | – | – | – | – | – | – | – | 2.37 |
| L-SeqSleepNet (released) | – | – | – | – | – | – | – | 0.091 |
| DGCNN | < 0.001 | 0.004 | < 0.001 | < 0.001 | < 0.001 | 0.030 | 0.024 | 0.15 |

- **The foundation models cost 2–90× the small supervised models per sample** (0.6–3.6 GFLOPs against 8–330 MFLOPs for EEGNet,
  EEG Conformer and ATCNet; 2× only against EEG Conformer on Sleep-EDF), and on Sleep-EDF and SEED one fine-tuning run costs 7–17 PFLOPs. Where they lead (SEED, STEW, sleep)
  they lead by 0.01–0.05; where a small model leads (BCI-IV-2a, Mumtaz2016) the cheap model is also the better one.
- **ATCNet ties the foundation models on Sleep-EDF at 1/9 of LaBraM's FLOPs** (80 vs. 741 MFLOPs per epoch), and on PhysioNet-MI is
  second at 114 MFLOPs against CBraMod's 2,668.
- **dDTF-GL is not cheap where channels are many.** Its maps cost C³ per frequency: 756 MFLOPs per PhysioNet-MI window (63 channels),
  580 on SEED (62), more than every small network and close to the foundation models; on STEW's 14 channels 20 MFLOPs and on
  Sleep-EDF's 2 channels 4 MFLOPs. The classifier (XGBoost, a few thousand comparisons) is negligible. The paper's claim of an
  efficient model therefore holds for few channels only.
- **dDTF-GL+ is cheap where it drops the maps.** On MentalArithmetic, PhysioNet-MI and SEED the search kept band power only: 0.2–3
  MFLOPs per sample, the cheapest model on those datasets with DGCNN, and first by mean on MentalArithmetic. Where it keeps the maps
  (BCI-IV-2a, Mumtaz2016, STEW, Sleep-EDF, SEED-V) it costs what dDTF-GL costs. Its training has no epochs: one run takes from half
  a minute (MentalArithmetic, Mumtaz2016) to 40 minutes (Sleep-EDF, stacked SVMs on 153,100 epochs) on 2–5 CPU threads, against 0.004–17 PFLOPs of GPU training
  for the networks; the validation search (20–140 fits per dataset) is the expensive part.
- **braindecode's DeepSleepNet grows with the channels** (its recurrent layers take every channel's features): 303M parameters and
  1.5 GFLOPs on PhysioNet-MI, as much as a foundation model, for a sleep model run off-label.

### Grouped by characteristic

| Characteristic | Group | Methods | What it means for comparisons |
|---|---|---|---|
| Model family | task-specific, trained from scratch | EEGNet, EEG Conformer, ATCNet; for sleep staging DeepSleepNet, L-SeqSleepNet | no pre-training data or released weights involved |
| | pre-trained foundation model, fine-tuned | LaBraM, CBraMod | results depend on released weights; LaBraM-Large/-Huge cannot be checked |
| | feature learning without a network ("green learning") | dDTF-GL | hand-defined connectivity features, feature selection and XGBoost; trains on a CPU in seconds to minutes; no epochs, so nothing is selected on validation data |
| Evaluation protocol (all subject-independent) | cross-subject subject splits, one shared pipeline per dataset | every model on CBraMod's 4 datasets, Sleep-EDF-78, SEED, SEED-V and STEW | directly comparable within a dataset: identical samples, splits, seeds, checkpoint rule and metric code (Section 5) |
| | sleep staging, Sleep-EDF-78, one 62 / 8 / 8 subject split | every model: in CBraMod's sleep-staging pipeline, DeepSleepNet and L-SeqSleepNet also with their own released code | comparable on the common test set (20,976 epochs every model scores; Section 5) |
| | leave one subject out, the papers' own | ATCNet (BCI-IV-2a), DGCNN (SEED) | reproductions of the papers' subject-independent numbers |
| | Sleep-EDF-20, the sleep papers' own protocols | DeepSleepNet (20-fold CV), L-SeqSleepNet (leave-one-subject-out) | reproductions of the papers; the two protocols and epoch sets differ |
| Epoch selection in released code | on the test set | EEG Conformer, ATCNet (which also keeps the best of 10 runs by test accuracy) | inflates the reported number: ATCNet's LOSO checkpoint on the held-out subject gives 8.8 pp more than a validation split (EEG Conformer's evidence is within-subject and no longer reported) |
| | on a validation set | CBraMod, LaBraM, L-SeqSleepNet | no test-set leakage |
| | none (final model) | DeepSleepNet | no leakage, but the fixed training length overfits on a larger training set (Sleep-EDF-78) |
| Reproduction outcome | reproduces | LaBraM-Base (10 / 12 metrics); ATCNet's LOSO under its own rule (69.06–70.16 vs. 70.97%); DGCNN's LOSO (77.58 vs. 79.95%); DeepSleepNet and L-SeqSleepNet on Sleep-EDF-20; dDTF-GL's subject-independent STEW number (our implementation; the authors' bundle is a different method and scores higher) | |
| | does not reproduce | CBraMod (3 / 12 metrics within noise); the CBraMod paper's EEG Conformer rows (above on BCI-IV-2a, below on the other three) | |
| | mixed | EEGNet (6 / 12 within noise: above the paper on all BCI-IV-2a metrics and PhysioNet-MI weighted F1, below on MentalArithmetic and Mumtaz2016 balanced accuracy) | the paper's EEGNet came from BIOT's code with unreported settings |

### Results by method

Section 2 turned around: each method, then its results on every dataset it ran on, all on test subjects no part of training saw.
**Shared pipeline** is the comparable number: the model inside the one pipeline every model shares on that dataset (CBraMod's for
the four cross-subject datasets, Sleep-EDF-78, SEED and SEED-V; the shared STEW recipe; the dDTF-GL paper's subject-independent
STEW setting; Section 5), with its **rank by mean** among the models run there (one entry per model, dDTF-GL+ included; 1 = best of
n). Ranks do not mark ties; the tests are in Section 5 and in the patterns below. SEED-V's subject split is at chance for every
model, so its ranks are noise. **Paper's protocol** is the reproduction of the method's own paper, paper → ours, where that protocol
is subject-independent.

- Metrics: balanced accuracy (motor imagery, Mumtaz2016, MentalArithmetic, SEED, SEED-V), κ (Sleep-EDF-78), accuracy (STEW). STEW
  settings: 2-class (rest vs. task) and 3-level on the shared 4-s windows and folds, and SI, the dDTF-GL paper's subject-independent
  setting on 6-s windows. "Paper" in the shared-pipeline column is the method's own paper (CBraMod) or the CBraMod paper's baseline
  row.
- bd: braindecode's implementation, the only one of ATCNet and DeepSleepNet that runs in the shared pipelines (their released
  code is TensorFlow). braindecode re-runs of the other models are in Sections 2 and 6.
- dDTF-GL+ is dDTF-GL with its features, feature counts, projection features and classifier chosen per dataset on validation data
  ([feature-tuning](methods/ddtf-gl/experiments/feature-tuning)). The networks run with each pipeline's fixed settings, so dDTF-GL+
  had more tuning than any of them.

| Method | Task | Dataset | Shared pipeline: ours (rank / n) | Paper's protocol: paper → ours |
|---|---|---|---|---|
| **LaBraM-Base**<br>*foundation model*: raw signal, Transformer, 5.8M, pre-trained | motor imagery | BCI-IV-2a | 0.4477 (6 / 9); CBraMod paper 0.4869 | its paper's TUAB and TUEV not run (access-gated) |
|  | motor imagery | PhysioNet-MI | 0.6098 (3 / 9); CBraMod paper 0.6173 | – |
|  | depression | Mumtaz2016 | 0.8905 (6 / 9); CBraMod paper 0.9409 | – |
|  | mental stress | MentalArithmetic | 0.6562 (3 / 9); CBraMod paper 0.6909 | – |
|  | sleep staging | Sleep-EDF-78 | **κ 0.7727 (1 / 10)** | – |
|  | emotion | SEED | subject **0.5822 (1 / 9)** | – |
|  | emotion | SEED-V | subject 0.2214 (9 / 9) | – |
|  | workload | STEW | 2-class 0.8179 (2 / 9) · 3-level **0.5670 (1 / 9)** · SI 0.8626 (2 / 9) | – |
| **CBraMod**<br>*foundation model*: raw signal, criss-cross Transformer, 4.9M backbone, pre-trained | motor imagery | BCI-IV-2a | 0.4571 (5 / 9); paper 0.5138 | its paper's protocol is the shared pipeline (paper numbers in the previous column) |
|  | motor imagery | PhysioNet-MI | **0.6265 (1 / 9)**; paper 0.6417 | – |
|  | depression | Mumtaz2016 | 0.8933 (5 / 9); paper 0.9560 | – |
|  | mental stress | MentalArithmetic | 0.5889 (6 / 9); paper 0.7256 | – |
|  | sleep staging | Sleep-EDF-78 | κ 0.7676 (4 / 10) | – |
|  | emotion | SEED | subject 0.5442 (3 / 9) | – |
|  | emotion | SEED-V | subject 0.2223 (8 / 9) | its paper's SEED-V benchmark is the trial split (out of scope) |
|  | workload | STEW | 2-class 0.7998 (3 / 9) · 3-level 0.5661 (2 / 9) · SI 0.8451 (3 / 9) | – |
| **EEGNet**<br>*small, raw input*: compact CNN, 2–4K, from scratch | motor imagery | BCI-IV-2a | **0.5288 (1 / 9)**; CBraMod paper 0.4482 | its paper's protocols are within-subject (out of scope) |
|  | motor imagery | PhysioNet-MI | 0.5948 (4 / 9); CBraMod paper 0.5814 | – |
|  | depression | Mumtaz2016 | 0.8828 (7 / 9); CBraMod paper 0.9232 | – |
|  | mental stress | MentalArithmetic | 0.5569 (8 / 9); CBraMod paper 0.6770 | – |
|  | sleep staging | Sleep-EDF-78 | κ 0.6815 (9 / 10) | – |
|  | emotion | SEED | subject 0.5227 (7 / 9) | – |
|  | emotion | SEED-V | subject 0.2468 (4 / 9) | – |
|  | workload | STEW | 2-class 0.6648 (9 / 9) · 3-level 0.4975 (9 / 9) · SI 0.6647 (9 / 9) | the dDTF-GL paper's EEGNet, SI: 0.683 → 0.6647 |
| **EEG Conformer**<br>*small, raw input*: CNN + self-attention, 230–790K, from scratch | motor imagery | BCI-IV-2a | 0.5042 (3 / 9); CBraMod paper 0.4696 | its paper's protocols (BCI-IV-2a session T → E, SEED session 1) are within-subject (out of scope) |
|  | motor imagery | PhysioNet-MI | 0.5787 (6 / 9); CBraMod paper 0.6049 | – |
|  | depression | Mumtaz2016 | 0.8747 (8 / 9); CBraMod paper 0.9308 | – |
|  | mental stress | MentalArithmetic | 0.5438 (9 / 9); CBraMod paper 0.6805 | – |
|  | sleep staging | Sleep-EDF-78 | κ 0.6876 (8 / 10) | – |
|  | emotion | SEED | subject 0.5378 (5 / 9) | – |
|  | emotion | SEED-V | subject 0.2366 (6 / 9) | – |
|  | workload | STEW | 2-class 0.7364 (6 / 9) · 3-level 0.5118 (8 / 9) · SI 0.7842 (5 / 9) | – |
| **ATCNet**<br>*small, raw input*: CNN + attention + TCN, 114K, from scratch; bd in the shared pipelines | motor imagery | BCI-IV-2a | 0.4898 (4 / 9), bd | leave one subject out: 70.97 → 69.06 % (the paper's rule: checkpoint on the held-out subject) / 60.23 % (validation split) |
|  | motor imagery | PhysioNet-MI | 0.6142 (2 / 9), bd | – |
|  | depression | Mumtaz2016 | 0.9077 (4 / 9), bd | – |
|  | mental stress | MentalArithmetic | 0.6951 (2 / 9), bd | – |
|  | sleep staging | Sleep-EDF-78 | κ 0.7709 (3 / 10), bd | – |
|  | emotion | SEED | subject 0.5410 (4 / 9), bd | – |
|  | emotion | SEED-V | subject 0.2528 (3 / 9), bd | – |
|  | workload | STEW | 2-class 0.7257 (7 / 9) · 3-level 0.5290 (6 / 9) · SI 0.7675 (7 / 9), bd | – |
| **DeepSleepNet**<br>*sleep model*: raw 30-s epochs, two-scale CNN + BiLSTM, from scratch; bd in the shared pipelines | motor imagery | BCI-IV-2a | 0.5250 (2 / 9), bd, off-label | – |
|  | motor imagery | PhysioNet-MI | 0.5877 (5 / 9), bd, off-label | – |
|  | depression | Mumtaz2016 | 0.8391 (9 / 9), bd, off-label | – |
|  | mental stress | MentalArithmetic | 0.5917 (5 / 9), bd, off-label | – |
|  | sleep staging | Sleep-EDF | Sleep-EDF-78: κ 0.7625 (5 / 10), bd; its released training on the same split 0.7030 | Sleep-EDF-20, 20-fold CV: 82.0 → 81.81 % (κ 0.76 → 0.754) |
|  | emotion | SEED | subject 0.5034 (9 / 9), bd, off-label | – |
|  | emotion | SEED-V | subject 0.2421 (5 / 9), bd, off-label | – |
|  | workload | STEW | 2-class 0.7168 (8 / 9) · 3-level 0.5468 (5 / 9) · SI 0.7408 (8 / 9), bd, off-label | – |
| **L-SeqSleepNet**<br>*sleep model*: spectrograms through a learned filterbank, BiLSTMs over 200-epoch sequences, from scratch | sleep staging | Sleep-EDF | Sleep-EDF-78: κ 0.7725 (2 / 10), its own released training | Sleep-EDF-20, leave one subject out: 86.3 → 86.11 % (κ 0.813 → 0.811) |
|  | emotion | SEED, SEED-V | off-label on 30-s windows, not ranked; on the subject split its sequences can read a clip from its position, so the number is not valid | – |
|  | workload | STEW | off-label on 30-s windows, not ranked: 2-class 0.7667 · 3-level 0.5407 | – |
| **DGCNN**<br>*spectral input*: 5-band differential entropy per electrode, graph network, from scratch | motor imagery | BCI-IV-2a | 0.3458 (7 / 9) | – |
|  | motor imagery | PhysioNet-MI | 0.4822 (8 / 9) | – |
|  | depression | Mumtaz2016 | **0.9619 (1 / 9)** | – |
|  | mental stress | MentalArithmetic | 0.6236 (4 / 9) | – |
|  | sleep staging | Sleep-EDF-78 | κ 0.6941 (7 / 10) | – |
|  | emotion | SEED | subject 0.5215 (8 / 9) | leave one subject out: 79.95 → 77.58 % |
|  | emotion | SEED-V | subject 0.2563 (2 / 9) | – |
|  | workload | STEW | 2-class 0.7747 (4 / 9) · 3-level 0.5517 (4 / 9) · SI 0.8046 (4 / 9) | – |
| **dDTF-GL**<br>*spectral input*: dDTF connectivity in frequency bands, feature selection + XGBoost, no network | motor imagery | BCI-IV-2a | 0.3276 (9 / 9) | – |
|  | motor imagery | PhysioNet-MI | 0.3274 (9 / 9) | – |
|  | depression | Mumtaz2016 | 0.9192 (3 / 9) | – |
|  | mental stress | MentalArithmetic | 0.5833 (7 / 9) | – |
|  | sleep staging | Sleep-EDF-78 | κ 0.5595 (10 / 10), each epoch alone | – |
|  | emotion | SEED | subject 0.5237 (6 / 9) | – |
|  | emotion | SEED-V | subject 0.2315 (7 / 9) | – |
|  | workload | STEW | 2-class 0.7398 (5 / 9) · 3-level 0.5185 (7 / 9) · SI 0.7744 (6 / 9) | SI 0.766 → 0.7744, on Safari et al.'s maps |
| **dDTF-GL+**<br>*spectral input, tuned here*: dDTF maps, their graph summaries and band power chosen per dataset on validation data; LNT or stacked LDA / SVM scores + XGBoost, no network | motor imagery | BCI-IV-2a | 0.3347 (8 / 9) | tuned here (methods/ddtf-gl/experiments/feature-tuning) |
|  | motor imagery | PhysioNet-MI | 0.4904 (7 / 9) | – |
|  | depression | Mumtaz2016 | 0.9213 (2 / 9) | – |
|  | mental stress | MentalArithmetic | **0.7097 (1 / 9)** | – |
|  | sleep staging | Sleep-EDF-78 | κ 0.7207 (6 / 10), each epoch alone | – |
|  | emotion | SEED | subject 0.5473 (2 / 9) | – |
|  | emotion | SEED-V | subject 0.2578 (1 / 9) | – |
|  | workload | STEW | 2-class **0.8191 (1 / 9)** · 3-level 0.5636 (3 / 9) · SI **0.8632 (1 / 9)** | – |

### Patterns across datasets, by characteristic

The methods grouped by what they take as input and whether they are pre-trained, then their ranks from the table above side
by side. The columns run from cue-locked trials (motor imagery) through sustained states (a diagnosis from resting EEG,
stress, workload, emotion) to sleep. Tests: `python tools/compare_methods.py` (the last table: every model with Welch p against the
best; the fourth: DGCNN, raw against DE input), `tools/compare_seed.py`, `tools/compare_stew.py`, `tools/compare_sleepedf.py`.

| Group | Methods | Input | Pre-training | Parameters | Built for |
|---|---|---|---|---|---|
| Foundation models | LaBraM-Base, CBraMod | raw signal, cut into per-channel patches | LaBraM: ~2,500 h from ~20 datasets; CBraMod: the TUH EEG corpus | 5.8M; 4.9M backbone | any EEG task, fine-tuned per task |
| Small supervised models, raw input | EEGNet, EEG Conformer, ATCNet | raw signal; temporal and spatial filters learned | none | 2–4K; 230–790K; 75–115K | motor imagery (EEGNet also ERPs; EEG Conformer also SEED) |
| Sleep models | DeepSleepNet, L-SeqSleepNet | raw 30-s epochs; spectrograms through a learned filterbank | none | 40.5M (braindecode's single-epoch version, 2 channels); not counted | sleep staging, over sequences of 25 / 200 epochs |
| Spectral-feature models | DGCNN, dDTF-GL | features fixed before the model: band power (5-band differential entropy per electrode and second); directed connectivity (dDTF) in frequency bands per window | none | 0.2–3.2M; no network (XGBoost) | emotion (SEED); mental workload (STEW) |
| Tuned here | dDTF-GL+ | per dataset: dDTF maps, their graph summaries, band power per window and per second | none | no network (XGBoost, stacked LDA / SVMs) | – |

Rank by mean in each shared pipeline (1 = best of n):

| Group | Model | BCI-IV-2a | PhysioNet-MI | Mumtaz2016 | MentalArithmetic | STEW 2-class | STEW 3-level | STEW SI | SEED | SEED-V (chance) | Sleep-EDF-78 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Foundation | LaBraM-Base | 6 / 9 | 3 / 9 | 6 / 9 | 3 / 9 | 2 / 9 | **1** / 9 | 2 / 9 | **1** / 9 | 9 / 9 | **1** / 10 |
|  | CBraMod | 5 / 9 | **1** / 9 | 5 / 9 | 6 / 9 | 3 / 9 | 2 / 9 | 3 / 9 | 3 / 9 | 8 / 9 | 4 / 10 |
| Small, raw input | EEGNet | **1** / 9 | 4 / 9 | 7 / 9 | 8 / 9 | 9 / 9 | 9 / 9 | 9 / 9 | 7 / 9 | 4 / 9 | 9 / 10 |
|  | EEG Conformer | 3 / 9 | 6 / 9 | 8 / 9 | 9 / 9 | 6 / 9 | 8 / 9 | 5 / 9 | 5 / 9 | 6 / 9 | 8 / 10 |
|  | ATCNet (bd) | 4 / 9 | 2 / 9 | 4 / 9 | 2 / 9 | 7 / 9 | 6 / 9 | 7 / 9 | 4 / 9 | 3 / 9 | 3 / 10 |
| Sleep | DeepSleepNet (bd) | 2 / 9 | 5 / 9 | 9 / 9 | 5 / 9 | 8 / 9 | 5 / 9 | 8 / 9 | 9 / 9 | 5 / 9 | 5 / 10 |
|  | L-SeqSleepNet | n.a. | n.a. | n.a. | n.a. | n.a. | n.a. | n.a. | n.a. | n.a. | 2 / 10 |
| Spectral input | DGCNN | 7 / 9 | 8 / 9 | **1** / 9 | 4 / 9 | 4 / 9 | 4 / 9 | 4 / 9 | 8 / 9 | 2 / 9 | 7 / 10 |
|  | dDTF-GL | 9 / 9 | 9 / 9 | 3 / 9 | 7 / 9 | 5 / 9 | 7 / 9 | 6 / 9 | 6 / 9 | 7 / 9 | 10 / 10 |
| Tuned here | dDTF-GL+ | 8 / 9 | 7 / 9 | 2 / 9 | **1** / 9 | **1** / 9 | 3 / 9 | **1** / 9 | 2 / 9 | 1 / 9 | 6 / 10 |

1. **The input decides more than the task a model was built for.** DGCNN was built for emotion and dDTF-GL for workload; what
   they share is spectral input, band power per electrode (DGCNN) or frequency-domain connectivity (dDTF-GL, dDTF-GL+).
   - They are **the top three on Mumtaz2016** (depression vs. healthy, resting EEG): DGCNN 0.9619 is above every other model
     (p ≤ 0.003), and dDTF-GL+ 0.9213 and dDTF-GL 0.9192 come next, above both foundation models (dDTF-GL: p ≤ 0.004).
   - DGCNN is **the best network without pre-training on all three STEW settings** (fourth of nine, behind the foundation models
     and dDTF-GL+; 2-class: above EEG Conformer, ATCNet, EEGNet and dDTF-GL, p ≤ 0.002).
   - They are **the bottom three on both motor-imagery datasets** (0.33–0.49, chance 0.25).
   - The same network with the other input repeats the split: braindecode's DGCNN on raw samples, against DE input
     (braindecode's DGCNN on DE for the four cross-subject datasets, our port elsewhere; the two agree, Section 6), gains 0.087
     (PhysioNet-MI) and 0.096 (BCI-IV-2a) on motor imagery and loses 0.064–0.106 on every other dataset (Mumtaz2016,
     MentalArithmetic, STEW 2-class and 3-level, Sleep-EDF κ), each p ≤ 0.004. The model's original task does not predict any of
     this; its input does.
   - **On MentalArithmetic band power with a tuned classifier is first by mean:** dDTF-GL+ (log band power per window and per
     second, stacked LDA / SVM scores, no dDTF maps) 0.7097, level with ATCNet (0.6951, p = 0.68) and above LaBraM-Base and DGCNN
     (p = 0.040, 0.041). DGCNN alone does not separate from the networks there (SD over seeds 0.067).
   - **Sleep staging is a state too, but spectral input is not enough there.** DGCNN (κ 0.6941) sits with EEG Conformer and
     EEGNet, 0.073 below CBraMod (p < 0.001); dDTF-GL+, which classifies each epoch alone, reaches κ 0.7207, above EEGNet and EEG
     Conformer (p ≤ 0.018) but 0.047 below CBraMod (p < 0.001). Raw-input encoders take the top places (LaBraM-Base, ATCNet,
     CBraMod, DeepSleepNet) next to L-SeqSleepNet.
   - A reading, not tested here: on Mumtaz2016, STEW and SEED the class is a state held for seconds to minutes, which shifts band
     power across the whole window. Motor imagery is a cue-locked event whose class lies in when and where power changes inside
     the trial, which raw input keeps and a per-second band-power summary blurs. Sleep stages are scored partly on transient
     waveforms (spindles, K-complexes), which a band-power summary also averages out.
2. **Pre-training buys consistency, not a win everywhere.** Among the eight published methods a foundation model is first by mean
   on 6 of the 10 settings (LaBraM-Base on STEW's three settings, SEED and Sleep-EDF-78; CBraMod on PhysioNet-MI). The exceptions are
   BCI-IV-2a, where EEGNet beats both (p ≤ 0.021); Mumtaz2016, where DGCNN beats both (p < 0.001); MentalArithmetic, where ATCNet
   (0.6951) is level with LaBraM-Base (p = 0.32); and SEED-V across subjects, at chance for every model. First is not always
   significant: on Sleep-EDF-78 four models tie (p ≥ 0.40), and dDTF-GL+ ties LaBraM-Base on STEW 2-class (+0.001, p = 0.82) and in
   the SI setting (0.8632 vs. 0.8626). Pre-training itself (CBraMod against CBraMod without it) helps on STEW (+0.101), PhysioNet-MI
   (+0.053) and Sleep-EDF (+0.023 κ); on SEED (+0.030, p = 0.086) and SEED-V (+0.018, p = 0.034) both are near chance, and on
   BCI-IV-2a, Mumtaz2016 and MentalArithmetic it changes nothing (p ≥ 0.58).
3. **The small raw-input models depend on the task.**
   - EEGNet (2–4K parameters) is the best model on BCI-IV-2a across subjects, and last or second to last on 5 of the other 9
     settings (MentalArithmetic, STEW's three, Sleep-EDF).
   - ATCNet (114K) is the steadiest of them: in the top four on 6 of the 9 settings that are above chance (second on PhysioNet-MI,
     below CBraMod only, p = 0.006; second on MentalArithmetic; third on Sleep-EDF, level with CBraMod, p = 0.56, with 114K
     parameters against 4.9M), but sixth or seventh on STEW.
   - EEG Conformer ranges from third (BCI-IV-2a) to last (MentalArithmetic). Size alone does not explain the sleep ranking: EEG
     Conformer (11.3M with the sequence head) sits with EEGNet (κ 0.6876 and 0.6815), while ATCNet (3.35M) ties the foundation
     models.
4. **A feature pipeline tuned on validation data moves from mid-table to the top on state tasks, not on motor imagery.** dDTF-GL+
   against dDTF-GL: STEW 2-class +0.079, SI +0.089, 3-level +0.045, MentalArithmetic +0.126, Sleep-EDF κ +0.161, SEED +0.024,
   PhysioNet-MI +0.163, SEED-V +0.026, still at chance (each p < 0.001); BCI-IV-2a +0.007 (p = 0.16) and Mumtaz2016 +0.002
   (p = 0.30) do not move. It is first by mean on STEW 2-class, SI and MentalArithmetic, second on Mumtaz2016 and SEED. Band power
   does most of it: the search keeps it alone on MentalArithmetic, PhysioNet-MI and SEED, adds it to the maps on STEW, Sleep-EDF and
   SEED-V, and keeps the maps without it only on Mumtaz2016 and BCI-IV-2a. The comparison favours dDTF-GL+, the only model whose
   design was searched per dataset; it shows how far a tuned no-network baseline goes, not that the networks are worse.
5. **Cost does not order the results** ([Model size and compute](#model-size-and-compute)). The two cheapest models lead where
   their input fits: DGCNN (about 1 MFLOP per sample) on Mumtaz2016 and dDTF-GL+ (3 MFLOPs) on MentalArithmetic. ATCNet ties the
   foundation models on Sleep-EDF at 1/9 of LaBraM-Base's FLOPs. Where the foundation models lead (STEW, SEED, Sleep-EDF) they
   lead by at most 0.04, at 0.6–3.6 GFLOPs per sample.
6. **SEED-V across subjects separates nothing:** balanced accuracy 0.22–0.26 for every model (chance 0.20), so its ranks,
   including dDTF-GL+'s first place (0.2578, DGCNN 0.2563), are noise.

What this means for comparisons:

- **A ranking on one task category does not carry over to another.** EEGNet goes from first (BCI-IV-2a) to last (STEW), DGCNN
  from first (Mumtaz2016) to seventh and eighth of nine (motor imagery).
- **Input type is a variable to report next to model family.** Without pre-training, a band-power model (DGCNN on DE, dDTF-GL+ on
  band power) is the one to beat on the resting-state, stress and workload datasets; on motor imagery it tells little.
- **A foundation model is the safest default,** first on 6 of 10 settings among the published methods, but not the best on every
  dataset, and its lead over the next model is often within noise.
- **How much each model was tuned belongs in the comparison.** The same feature chain goes from fifth to first on STEW and from
  seventh to first on MentalArithmetic when its inputs are chosen on validation data.

## 4. Datasets

### Datasets run here

| Category | Dataset | Task (classes) | Subjects: train / val / test | Channels | Input as run | Samples | Metrics (checkpoint metric in bold) | Run with | Paper tables |
|---|---|---|---|---|---|---|---|---|---|
| Motor imagery classification | BCI-IV-2a | left hand, right hand, feet, tongue (4) | cross-subject 5 / 2 / 2; leave one subject out (ATCNet's protocol) | 22 | 4 s at 200 Hz (CBraMod pipeline); 4.5 s at 250 Hz, unfiltered, z-scored (ATCNet's LOSO) | 5,184 (CBraMod pipeline); 288 + 288 trials per subject (ATCNet's LOSO) | balanced acc., **κ**, weighted F1; EEG Conformer, ATCNet: accuracy (κ) | every model except L-SeqSleepNet; ATCNet also in its LOSO | CBraMod T15; EEG Conformer T II; ATCNet T IV–VI |
| Motor imagery classification | PhysioNet-MI | left fist, right fist, both fists, both feet (4) | 70 / 19 / 20 | 64 | 4 s at 200 Hz | 9,837 | balanced acc., **κ**, weighted F1 | every model except L-SeqSleepNet | CBraMod T3, T4, T18 |
| Mental disorder diagnosis | Mumtaz2016 | major depressive disorder vs. healthy (2) | 43 / 9 / 11 | 19 | 5 s at 200 Hz | 7,143 | balanced acc., AUC-PR, **AUROC** | every model except L-SeqSleepNet | CBraMod T10 |
| Mental stress detection | MentalArithmetic | rest vs. mental arithmetic ("mental stress") (2) | 28 / 4 / 4 | 20 (LaBraM: 19, no A2-A1) | 5 s at 200 Hz | 1,707 | balanced acc., AUC-PR, **AUROC** | every model except L-SeqSleepNet | CBraMod T12 |
| Sleep staging | Sleep-EDF (Expanded, Sleep Cassette) | W, N1, N2, N3, REM (5) | Sleep-EDF-78: 62 / 8 / 8, age-stratified; Sleep-EDF-20: 20-fold CV (DeepSleepNet), leave-one-subject-out (L-SeqSleepNet) | 2 (Fpz-Cz, Pz-Oz); the sleep models' released code: Fpz-Cz | CBraMod pipeline: 30-s epochs at 200 Hz, 0.3–35 Hz, in sequences of 20, sleep period ± 30 min; DeepSleepNet: raw 30-s epochs at 100 Hz in sequences of 25; L-SeqSleepNet: spectrograms of 30-s epochs at 100 Hz in sequences of 200 | 192,840 epochs (pipeline, Sleep-EDF-78); 41,950 (DeepSleepNet) and 46,241 (L-SeqSleepNet) test epochs on Sleep-EDF-20 | balanced acc., **κ**, weighted F1, accuracy, macro-F1 | all seven, dDTF-GL | DeepSleepNet T III–IV; L-SeqSleepNet T III |
| Emotion recognition | SEED | positive, neutral, negative (3) | subject split 9 / 3 / 3; DGCNN's leave one subject out | 62 | 1 s at 200 Hz, 0.3–75 Hz (CBraMod pipeline); DE features (DGCNN's LOSO) | 152,730 | balanced acc., **κ**, weighted F1, accuracy, clip vote | all eight, dDTF-GL | DGCNN abstract; EEG Conformer README |
| Emotion recognition | SEED-V | happy, sad, neutral, disgust, fear (5) | subject split 10 / 3 / 3 | 62 | 1 s at 200 Hz, 0.3–75 Hz | 117,744 | balanced acc., **κ**, weighted F1 | all eight, dDTF-GL | CBraMod T2 |
| Mental workload | STEW | rest vs. task (2); rating 1–3 / 4–6 / 7–9 (3) | 7-fold GroupKFold across subjects (48; 45 with ratings), validation = next fold | 14 | 4-s windows, 25% overlap, 128 Hz (CBraMod, LaBraM: 200 Hz); dDTF-GL's paper protocol: 6-s windows, 4-s step | 4,704 | accuracy, balanced acc., **κ**, weighted F1 | all eight, dDTF-GL | – |

### Which datasets each method's paper evaluates on

● = the paper evaluates its own method on this dataset; ✓ = we ran this model on it (results in Section 2);
●✓ = both. Download and access details: [datasets/README.md](datasets/README.md).

| Category | Dataset | EEGNet | EEG Conformer | ATCNet | LaBraM | CBraMod | DeepSleepNet | L-SeqSleepNet | DGCNN | dDTF-GL | Status here |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Motor imagery classification** | BCI-IV-2a | ●✓ ᵃ | ●✓ | ●✓ | ✓ | ●✓ |  |  | ✓ | ✓ | run: leave-one-subject-out, cross-subject (within-subject runs kept, not reported) |
|  | BCI-IV-2b |  | ● |  |  |  |  |  |  |  | public, not run yet (no download key) |
|  | PhysioNet-MI | ✓ |  |  | ✓ | ●✓ |  |  | ✓ | ✓ | run: cross-subject |
|  | SHU-MI |  |  |  |  | ● |  |  |  |  | not run: files are password-protected |
| **Emotion recognition** | SEED | ✓ | ●✓ | ✓ ˡ | ✓ | ✓ | ✓ | ✓ ᵐ | ●✓ | ✓ | run: all models in CBraMod's pipeline (subject split); DGCNN's leave one subject out |
|  | SEED-V | ✓ | ✓ | ✓ ˡ | ●✓ ᵇ | ●✓ | ✓ | ✓ ᵐ | ✓ | ✓ | run: all models in CBraMod's pipeline (subject split; CBraMod's own trial split is subject-dependent, not reported) |
|  | FACED |  |  |  |  | ● |  |  |  |  | not run: Synapse account |
|  | DREAMER |  |  |  |  |  |  |  | ● |  | not run: access by request, not requested |
| **Mental workload** | STEW | ✓ | ✓ | ✓ ˡ | ✓ | ✓ | ✓ | ✓ ᵐ | ✓ | ●✓ | run: all models, 7-fold GroupKFold across subjects; dDTF-GL also in its paper's subject-independent setting (6-s windows) |
| **Mental disorder diagnosis** | Mumtaz2016 | ✓ |  |  | ✓ | ●✓ |  |  | ✓ | ✓ | run: cross-subject |
| **Mental stress detection** | MentalArithmetic | ✓ |  |  | ✓ | ●✓ |  |  | ✓ | ✓ | run: cross-subject |
| **Vigilance estimation** | SEED-VIG |  |  |  |  | ● |  |  |  |  | not run: license application |
| **Sleep staging** | Sleep-EDF | ✓ | ✓ | ✓ | ✓ | ✓ | ●✓ ᶜ | ●✓ ᶜ | ✓ | ✓ | run: Sleep-EDF-20 with the sleep papers' protocols; Sleep-EDF-78 cross-subject, all models |
|  | MASS |  |  |  |  |  | ● |  |  |  | not run: access request |
|  | SHHS |  |  |  |  |  |  | ● |  |  | not run: NSRR access |
|  | ear-EEG, cEEGrid |  |  |  |  |  |  | ● |  |  | not run; access not checked |
|  | ISRUC |  |  |  |  | ● |  |  |  |  | public, not run yet (too large for one round) |
| **Seizure detection** | CHB-MIT |  |  |  |  | ● |  |  |  |  | public, not run yet (too large for one round) |
| **Imagined speech classification** | BCIC2020-3 |  |  |  |  | ● |  |  |  |  | not run: test labels not public |
| **Abnormal EEG detection** | TUAB |  |  |  | ● | ● |  |  |  |  | not run: TUH registration |
| **Event type classification** | TUEV |  |  |  | ● | ● |  |  |  |  | not run: TUH registration |
| **Event-related potentials** | P300 (visual evoked potentials) | ● |  |  |  |  |  |  |  |  | not run; access not checked |
|  | ERN (feedback error-related negativity) | ● |  |  |  |  |  |  |  |  | not run; access not checked |
|  | MRCP (movement-related cortical potentials) | ● |  |  |  |  |  |  |  |  | not run; access not checked |
| **Gait prediction (regression)** | MoBI |  |  |  | ● ᵇ |  |  |  |  |  | not run; access not checked |
|  | **Datasets in the paper / run here** | **4 / 1** | **3 / 2** | **1 / 1** | **4 / 1** | **13 / 5** | **2 / 1** | **4 / 1** | **2 / 1** | **1 / 1** |  |

- ᵃ EEGNet's paper uses BCI-IV-2a (its "SMR" dataset) with its own protocol (within-subject 4-fold block-wise
  cross-validation, and cross-subject). Here EEGNet ran only inside EEG Conformer's and CBraMod's pipelines, as their
  baseline.
- ᵇ In LaBraM's appendix; its main text evaluates TUAB and TUEV. LaBraM-Base ran here on CBraMod's datasets, as
  CBraMod's strongest baseline, on Sleep-EDF and on SEED-V; its own SEED-V result uses a 20-subject version that is not public (the
  public one has 16), TUAB and TUEV are access-gated, and MoBI was not tried.
- ˡ ATCNet: braindecode's implementation (the released code is TensorFlow), as on Sleep-EDF.
- ᵐ Off-label: the sleep model on 30-s windows of three channels.
- ᶜ DeepSleepNet's and L-SeqSleepNet's papers use Sleep-EDF-20 (the first 20 subjects, 39 nights); every model here
  also ran on Sleep-EDF-78 (all 78 subjects).
- **Baseline rows.** The papers also report other methods on their datasets; these are the "paper" numbers of the
  baselines in Section 2. CBraMod's paper reports EEGNet, EEG Conformer and LaBraM-Base on its datasets (its own runs,
  the supervised baselines with BIOT's code; the TUAB/TUEV rows copied from LaBraM; Section 7); EEG Conformer's
  Table II and ATCNet's Table IV report EEGNet on BCI-IV-2a.
- **Pre-training data is not evaluation data.** LaBraM pre-trains on about 2,500 hours from about 20 datasets, CBraMod
  on the TUH EEG corpus (TUEG); neither appears in the table.

## 5. Cross-subject results: all models on identical splits

On CBraMod's four public datasets, every model ran on CBraMod's preprocessed samples and subject splits, with seeds 0–4. The
checkpoint was the best validation κ / AUROC epoch, the test set was scored once, and the metric code was the same for every model.
The first tables keep the four models the CBraMod paper reports; each cell is **ours, mean ± std over 5 seeds (paper)**. Every paper
number is from the CBraMod paper; its EEGNet and LaBraM-Base rows are the CBraMod authors' runs. [All nine models](#all-nine-models-on-the-four-datasets)
follow, then sleep staging ([every model on one split](#sleep-staging-on-sleep-edf-78-every-model-on-one-split)) and emotion and
workload.

### Balanced accuracy

| Category | Dataset | EEGNet | CBraMod, no pre-training | CBraMod | LaBraM-Base | Best, ours | Best, paper |
|---|---|---|---|---|---|---|---|
| Motor imagery | PhysioNet-MI | 0.5948 ± 0.0028 (0.5814) | 0.5738 ± 0.0082 (0.6196) | **0.6265 ± 0.0054** (0.6417) | 0.6098 ± 0.0098 (0.6173) | CBraMod | CBraMod |
| Motor imagery | BCI-IV-2a | **0.5288 ± 0.0394** (0.4482) | 0.4561 ± 0.0195 (–) | 0.4571 ± 0.0399 (0.5138) | 0.4477 ± 0.0421 (0.4869) | EEGNet | CBraMod |
| Mental disorder diagnosis | Mumtaz2016 | 0.8828 ± 0.0295 (0.9232) | 0.8973 ± 0.0115 (–) | 0.8933 ± 0.0101 (0.9560) | 0.8905 ± 0.0041 (0.9409) | tie (no tested pair differs, p ≥ 0.48) | CBraMod |
| Mental stress detection | MentalArithmetic | 0.5569 ± 0.0247 (0.6770) | 0.5924 ± 0.0403 (–) | 0.5889 ± 0.0209 (0.7256) | **0.6562 ± 0.0397** (0.6909) | LaBraM-Base | CBraMod |

### Checkpoint metric: κ (4-class) or AUROC (binary)

| Dataset | Metric | EEGNet | CBraMod, no pre-training | CBraMod | LaBraM-Base |
|---|---|---|---|---|---|
| PhysioNet-MI | κ | 0.4597 ± 0.0037 (0.4468) | 0.4317 ± 0.0109 (0.4994) | 0.5019 ± 0.0072 (0.5222) | 0.4797 ± 0.0131 (0.4912) |
| BCI-IV-2a | κ | 0.3718 ± 0.0525 (0.2693) | 0.2748 ± 0.0260 (–) | 0.2762 ± 0.0532 (0.3518) | 0.2637 ± 0.0562 (0.3159) |
| Mumtaz2016 | AUROC | 0.9540 ± 0.0073 (0.9639) | 0.9546 ± 0.0073 (–) | 0.9770 ± 0.0059 (0.9921) | 0.9664 ± 0.0166 (0.9782) |
| MentalArithmetic | AUROC | 0.7794 ± 0.0866 (0.7321) | 0.6922 ± 0.0146 (–) | 0.7344 ± 0.0515 (0.7905) | 0.7791 ± 0.0608 (0.7721) |

### Pairwise tests on our runs (balanced accuracy)

A − B, mean over seeds 0–4, Welch p (`python tools/compare_methods.py`; pass `kappa` or `roc_auc` for the other
metrics):

| Dataset | CBraMod − CBraMod scratch | CBraMod − LaBraM-Base | CBraMod − EEGNet | LaBraM-Base − EEGNet |
|---|---|---|---|---|
| PhysioNet-MI | +0.0527 (p < 0.001) | +0.0167 (p = 0.015) | +0.0317 (p < 0.001) | +0.0150 (p = 0.024) |
| BCI-IV-2a | +0.0010 (p = 0.960) | +0.0094 (p = 0.727) | -0.0717 (p = 0.021) | -0.0811 (p = 0.014) |
| Mumtaz2016 | -0.0040 (p = 0.579) | +0.0028 (p = 0.592) | +0.0105 (p = 0.485) | +0.0077 (p = 0.591) |
| MentalArithmetic | -0.0035 (p = 0.870) | -0.0674 (p = 0.015) | +0.0319 (p = 0.060) | +0.0993 (p = 0.002) |

### All nine models on the four datasets

Balanced accuracy, best first: mean over seeds 0–4 and Welch p against the best (`python tools/compare_methods.py`, last table).
ATCNet and DeepSleepNet are braindecode's; dDTF-GL+ is dDTF-GL tuned on validation data
([feature-tuning](methods/ddtf-gl/experiments/feature-tuning)).

| Rank | BCI-IV-2a | PhysioNet-MI | Mumtaz2016 | MentalArithmetic |
|---|---|---|---|---|
| 1 | **EEGNet 0.5288** | **CBraMod 0.6265** | **DGCNN 0.9619** | **dDTF-GL+ 0.7097** |
| 2 | DeepSleepNet (bd) 0.5250 (p = 0.851) | ATCNet (bd) 0.6142 (p = 0.006) | dDTF-GL+ 0.9213 (p < 0.001) | ATCNet (bd) 0.6951 (p = 0.683) |
| 3 | EEG Conformer 0.5042 (p = 0.245) | LaBraM-Base 0.6098 (p = 0.015) | dDTF-GL 0.9192 (p < 0.001) | LaBraM-Base 0.6562 (p = 0.040) |
| 4 | ATCNet (bd) 0.4898 (p = 0.116) | EEGNet 0.5948 (p < 0.001) | ATCNet (bd) 0.9077 (p < 0.001) | DGCNN 0.6236 (p = 0.041) |
| 5 | CBraMod 0.4571 (p = 0.021) | DeepSleepNet (bd) 0.5877 (p < 0.001) | CBraMod 0.8933 (p < 0.001) | DeepSleepNet (bd) 0.5917 (p < 0.001) |
| 6 | LaBraM-Base 0.4477 (p = 0.014) | EEG Conformer 0.5787 (p = 0.001) | LaBraM-Base 0.8905 (p < 0.001) | CBraMod 0.5889 (p < 0.001) |
| 7 | DGCNN 0.3458 (p < 0.001) | dDTF-GL+ 0.4904 (p < 0.001) | EEGNet 0.8828 (p = 0.003) | dDTF-GL 0.5833 (p < 0.001) |
| 8 | dDTF-GL+ 0.3347 (p < 0.001) | DGCNN 0.4822 (p < 0.001) | EEG Conformer 0.8747 (p = 0.002) | EEGNet 0.5569 (p < 0.001) |
| 9 | dDTF-GL 0.3276 (p < 0.001) | dDTF-GL 0.3274 (p < 0.001) | DeepSleepNet (bd) 0.8391 (p < 0.001) | EEG Conformer 0.5438 (p < 0.001) |

- **No model is first on two of the four datasets.** The best by mean is EEGNet (BCI-IV-2a, level with braindecode's
  DeepSleepNet, EEG Conformer and ATCNet), CBraMod (PhysioNet-MI), DGCNN (Mumtaz2016) and dDTF-GL+ (MentalArithmetic, level with
  ATCNet).
- **ATCNet is the steadiest:** second on PhysioNet-MI and MentalArithmetic, fourth on BCI-IV-2a and Mumtaz2016, with 114K
  parameters.
- **The foundation models are never first on the three small datasets** (5th and 6th on BCI-IV-2a and Mumtaz2016, 3rd and 6th on
  MentalArithmetic).

### What the cross-subject results show

- **The paper's ranking (CBraMod first everywhere) holds on one dataset of four.** CBraMod is best on PhysioNet-MI,
  EEGNet on BCI-IV-2a and LaBraM-Base on MentalArithmetic, and all four models tie on Mumtaz2016.
- **Pre-training helps CBraMod only on PhysioNet-MI:** +0.053 balanced accuracy, more than the paper's +0.022. It
  changes nothing on the other three datasets. On AUROC it helps on Mumtaz2016 (+0.022, p < 0.001) but not
  significantly on MentalArithmetic (+0.042, p = 0.14).
- **A 2–4K-parameter baseline is competitive.** EEGNet beats both foundation models on BCI-IV-2a and ties them on
  Mumtaz2016. It loses to both on PhysioNet-MI and to LaBraM-Base on MentalArithmetic.
- **The Mumtaz2016 shortfall belongs to the data pipeline, not to a model.** All three models with a paper number
  fall 0.040–0.063 below it. The released code's subject split differs from the one the paper describes.
- **The seed spread is larger than published.** On BCI-IV-2a our std is 4–6× the paper's for every model (e.g.
  LaBraM-Base ±0.0421 vs. ±0.0085). Caveats for every row above: 5 seeds per model, Welch tests without
  multiple-comparison correction, and one untuned recipe per model.

Other CBraMod settings (PhysioNet-MI unless noted):

| Setting | Balanced acc., ours | Paper |
|---|---|---|
| Frozen backbone (head trained) | 0.5370 ± 0.0120 | 0.3845 ± 0.0345 (T18) |
| MentalArithmetic, code-default seed 3407 (n = 1) | 0.6146 | 0.7256 |
| MentalArithmetic, lr 5e-4 without `multi_lr` (author's tip, issue #6) | 0.5361 ± 0.0560 | 0.7256 |
| BCI-IV-2a, paper-era preprocessing (`@v0616`, 5,088 samples) | 0.4446 ± 0.0186 | 0.5138 |

None of the robustness checks closes the gap.

### dDTF-GL on the same splits

dDTF-GL (no network: dDTF connectivity maps, feature selection, XGBoost; [ddtf-gl/cbramod-splits](methods/ddtf-gl/experiments/cbramod-splits))
on the same samples, splits, seeds and metric code, but without epoch selection. Balanced accuracy, mean ± SD over seeds 0–4, and
dDTF-GL − each model (Welch p; `python tools/compare_methods.py`, last table):

| Dataset | dDTF-GL | − EEGNet | − CBraMod | − LaBraM-Base | − DGCNN (DE) |
|---|---|---|---|---|---|
| PhysioNet-MI | 0.3274 ± 0.0055 | −0.2674 (p < 0.001) | −0.2991 (p < 0.001) | −0.2824 (p < 0.001) | −0.1548 (p < 0.001) |
| BCI-IV-2a | 0.3276 ± 0.0084 | −0.2012 (p < 0.001) | −0.1295 (p = 0.001) | −0.1201 (p = 0.003) | −0.0182 (p = 0.301) |
| Mumtaz2016 | 0.9192 ± 0.0025 | +0.0364 (p = 0.051) | +0.0258 (p = 0.004) | +0.0286 (p < 0.001) | −0.0428 (p < 0.001) |
| MentalArithmetic | 0.5833 ± 0.0282 | +0.0264 (p = 0.155) | −0.0056 (p = 0.733) | −0.0729 (p = 0.012) | −0.0403 (p = 0.264) |

- Like DGCNN's band-power input, connectivity maps averaged over a trial carry almost nothing for motor imagery (chance 0.25), and
  are enough for depression vs. healthy (above both foundation models) and mental arithmetic (level with CBraMod and EEGNet).
- Maps with our first settings, before the calibration on Safari et al.'s maps, scored higher on Mumtaz2016 (0.943) and
  MentalArithmetic (0.633) and the same on motor imagery ([ddtf-gl/cbramod-splits](methods/ddtf-gl/experiments/cbramod-splits)).
- **dDTF-GL+, tuned on validation data,** drops the maps for band power on MentalArithmetic (0.7097, first of nine; +0.126 over
  dDTF-GL) and PhysioNet-MI (0.4904; +0.163, still 0.14 below CBraMod), and keeps them on Mumtaz2016 (0.9213) and BCI-IV-2a (0.3347),
  where it does not move (p ≥ 0.16).

### The same comparison with braindecode's implementations

The four models were re-run with braindecode 1.8.1's `EEGNet`, `CBraMod` and `Labram` (braindecode's pre-trained
weights for the foundation models), each swapped into the same pipeline as above, with the same splits, seeds,
checkpoint rule and metric code (Section 6). Balanced accuracy, mean ± SD over seeds 0–4; in parentheses the
released-code run of the same model from the table above:

| Dataset | EEGNet (bd) | CBraMod, no pre-training (bd) | CBraMod (bd) | LaBraM-Base (bd) | Best, braindecode | Best, released code |
|---|---|---|---|---|---|---|
| PhysioNet-MI | 0.5824 ± 0.0103 (0.5948) | 0.5022 ± 0.0121 (0.5738) | 0.6068 ± 0.0124 (0.6265) | **0.6175 ± 0.0042** (0.6098) | LaBraM-Base and CBraMod (p = 0.13) | CBraMod |
| BCI-IV-2a | **0.5255 ± 0.0212** (0.5288) | 0.4688 ± 0.0181 (0.4561) | 0.4703 ± 0.0159 (0.4571) | 0.4375 ± 0.0435 (0.4477) | EEGNet | EEGNet |
| Mumtaz2016 | 0.8931 ± 0.0293 (0.8828) | 0.8823 ± 0.0244 (0.8973) | 0.8827 ± 0.0204 (0.8933) | 0.8894 ± 0.0119 (0.8905) | tie (p ≥ 0.54) | tie |
| MentalArithmetic | 0.5437 ± 0.0467 (0.5569) | 0.5708 ± 0.0419 (0.5924) | 0.6042 ± 0.0551 (0.5889) | **0.6451 ± 0.0613** (0.6562) | LaBraM-Base, ahead of EEGNet only (p = 0.02) | LaBraM-Base |

Pairwise tests among the braindecode runs (balanced accuracy, A − B, Welch p; `python tools/compare_methods.py`,
third table):

| Dataset | CBraMod − CBraMod scratch | CBraMod − LaBraM-Base | CBraMod − EEGNet | LaBraM-Base − EEGNet |
|---|---|---|---|---|
| PhysioNet-MI | +0.1046 (p < 0.001) | -0.0107 (p = 0.129) | +0.0244 (p = 0.010) | +0.0351 (p < 0.001) |
| BCI-IV-2a | +0.0016 (p = 0.889) | +0.0328 (p = 0.173) | -0.0552 (p = 0.002) | -0.0880 (p = 0.007) |
| Mumtaz2016 | +0.0004 (p = 0.976) | -0.0067 (p = 0.550) | -0.0103 (p = 0.538) | -0.0037 (p = 0.805) |
| MentalArithmetic | +0.0333 (p = 0.315) | -0.0410 (p = 0.299) | +0.0604 (p = 0.099) | +0.1014 (p = 0.020) |

- **With braindecode's implementations CBraMod is not the best model on any dataset.** Its one released-code win,
  PhysioNet-MI (+0.017 over LaBraM-Base, p = 0.015), becomes −0.011 (p = 0.13): braindecode's CBraMod head costs
  0.02 there, while braindecode's LaBraM-Base is 0.008 above the released one (Section 6).
- **EEGNet's lead on BCI-IV-2a does not depend on the implementation** (braindecode: +0.055 over CBraMod, p = 0.002;
  +0.088 over LaBraM-Base, p = 0.007), and neither does the Mumtaz2016 tie.
- **Pre-training helps braindecode's CBraMod on PhysioNet-MI by +0.105** (released code: +0.053), because
  braindecode's from-scratch model is the weaker one there (0.5022 vs. 0.5738); elsewhere it changes nothing
  (p ≥ 0.32).

### Sleep staging on Sleep-EDF-78: every model on one split

One age-stratified 62 / 8 / 8 subject split of Sleep-EDF-78 (Fpz-Cz and Pz-Oz). The five models, and braindecode's
DeepSleepNet, ran as the epoch encoder of CBraMod's released sleep-staging pipeline (20-epoch sequences, the released
sequence head and trainer, best validation κ, seeds 0–4); ATCNet and DeepSleepNet with braindecode's networks, as
their released code is TensorFlow. DeepSleepNet and L-SeqSleepNet also trained with their own released code on the
same split (3 runs each; the code sets no seed). Every model is scored on the same 20,976 test epochs
([cbramod/finetune-sleepedf](methods/cbramod/experiments/finetune-sleepedf#all-models-on-the-common-test-set);
`python tools/compare_sleepedf.py`). No paper reports this split. Mean ± SD over runs, sorted by κ:

| Model | Implementation | Training | κ | Balanced acc. | Macro-F1 | N1 F1 | N3 F1 | REM F1 |
|---|---|---|---|---|---|---|---|---|
| LaBraM-Base | released | CBraMod pipeline | **0.7727 ± 0.0085** | 0.7612 ± 0.0184 | **0.7619 ± 0.0153** | **0.434** | 0.805 | 0.785 |
| L-SeqSleepNet | released (TF 1.15) | its own | 0.7725 ± 0.0040 | 0.7558 ± 0.0182 | 0.7583 ± 0.0096 | 0.406 | 0.776 | **0.834** |
| ATCNet | braindecode | CBraMod pipeline | 0.7709 ± 0.0039 | 0.7430 ± 0.0078 | 0.7496 ± 0.0053 | 0.383 | 0.798 | 0.779 |
| CBraMod | released | CBraMod pipeline | 0.7676 ± 0.0110 | 0.7537 ± 0.0087 | 0.7536 ± 0.0098 | 0.416 | 0.795 | 0.776 |
| DeepSleepNet | braindecode | CBraMod pipeline | 0.7625 ± 0.0214 | **0.7666 ± 0.0053** | 0.7522 ± 0.0158 | 0.419 | 0.810 | 0.760 |
| CBraMod, no pre-training | released | CBraMod pipeline | 0.7446 ± 0.0099 | 0.7316 ± 0.0155 | 0.7344 ± 0.0152 | 0.384 | **0.812** | 0.719 |
| dDTF-GL+ | ours, tuned on validation data | each epoch alone | 0.7207 ± 0.0009 | 0.7072 ± 0.0012 | 0.7023 ± 0.0014 | 0.308 | 0.797 | 0.657 |
| DeepSleepNet | released (TF 1.15) | its own | 0.7030 ± 0.0103 | 0.7247 ± 0.0104 | 0.7140 ± 0.0121 | 0.427 | 0.697 | 0.764 |
| EEG Conformer | released | CBraMod pipeline | 0.6876 ± 0.0190 | 0.6448 ± 0.0267 | 0.6585 ± 0.0235 | 0.347 | 0.562 | 0.663 |
| EEGNet | ours (PyTorch) | CBraMod pipeline | 0.6815 ± 0.0154 | 0.6852 ± 0.0137 | 0.6785 ± 0.0073 | 0.311 | 0.787 | 0.594 |
| dDTF-GL | ours (no network) | each epoch alone | 0.5595 ± 0.0006 | 0.5219 ± 0.0012 | 0.5229 ± 0.0013 | 0.086 | 0.521 | 0.414 |

Pairwise tests (A − B, mean over runs, Welch p):

| A − B | κ | Balanced acc. |
|---|---|---|
| CBraMod − CBraMod, no pre-training | +0.0230 (p = 0.009) | +0.0221 (p = 0.030) |
| LaBraM-Base − CBraMod, no pre-training | +0.0281 (p = 0.001) | +0.0296 (p = 0.026) |
| LaBraM-Base − CBraMod | +0.0051 (p = 0.435) | +0.0075 (p = 0.443) |
| L-SeqSleepNet − CBraMod | +0.0049 (p = 0.404) | +0.0021 (p = 0.864) |
| L-SeqSleepNet − LaBraM-Base | −0.0002 (p = 0.964) | −0.0054 (p = 0.704) |
| CBraMod − ATCNet | −0.0033 (p = 0.556) | +0.0107 (p = 0.076) |
| DeepSleepNet (braindecode) − CBraMod | −0.0051 (p = 0.652) | +0.0129 (p = 0.027) |
| L-SeqSleepNet − DeepSleepNet (released) | +0.0695 (p = 0.003) | +0.0311 (p = 0.078) |
| DeepSleepNet (braindecode) − DeepSleepNet (released) | +0.0595 (p = 0.002) | +0.0418 (p = 0.011) |
| CBraMod − EEGNet | +0.0861 (p < 0.001) | +0.0684 (p < 0.001) |
| CBraMod − EEG Conformer | +0.0800 (p < 0.001) | +0.1089 (p < 0.001) |
| CBraMod − dDTF-GL+ | +0.0469 (p < 0.001) | +0.0465 (p < 0.001) |
| dDTF-GL+ − DeepSleepNet (released) | +0.0177 (p = 0.097) | −0.0175 (p = 0.098) |
| dDTF-GL+ − EEGNet | +0.0392 (p = 0.005) | +0.0220 (p = 0.023) |
| dDTF-GL+ − EEG Conformer | +0.0331 (p = 0.018) | +0.0624 (p = 0.006) |
| dDTF-GL+ − dDTF-GL | +0.1612 (p < 0.001) | +0.1853 (p < 0.001) |

- **Four models tie at the top.** LaBraM-Base, L-SeqSleepNet, ATCNet and CBraMod lie within 0.005 κ, and none of the
  tested pairs among them differs (p ≥ 0.40). braindecode's LaBraM-Base (Section 6) scores 0.777, the highest κ of all
  runs, and is within noise of all four (p ≥ 0.08).
- **Pre-training helps on sleep staging.** Both foundation models beat CBraMod without pre-training (+0.023 and
  +0.028 κ). On CBraMod's public datasets pre-training helped only on PhysioNet-MI.
- **The foundation models do not beat the specialised models.** L-SeqSleepNet, trained with its own released code, is
  level with them. ATCNet, whose encoder has 112K parameters and no pre-training, is level with CBraMod (4.9M,
  pre-trained).
- **DeepSleepNet's training rule costs more than its architecture.** The released code keeps the final fine-tuning
  epoch. On this split its validation accuracy peaks at epochs 7–10 of 200 and then falls while training accuracy
  reaches 0.99, giving κ 0.703. braindecode's DeepSleepNet in the pipeline, which keeps the best validation epoch,
  gains 0.060 κ and, with braindecode's LaBraM (Section 6), has the highest balanced accuracy of all models (0.767).
- **The two small supervised models trail by 0.08–0.09 κ** (both p < 0.001). On BCI-IV-2a EEGNet beat both
  foundation models. Here EEGNet is weakest on REM and N1, and EEG Conformer on N3.
- **dDTF-GL, without a network, is the weakest model** (κ 0.560; EEGNet − dDTF-GL = +0.122, p < 0.001). Two channels give 20 dDTF
  values per epoch, of which its selection keeps 7, and it classifies each epoch alone where the pipeline models see 20-epoch
  sequences; it recognises wake (F1 0.87) and almost never N1 (0.09). **dDTF-GL+** (band power per epoch and per second added to
  the maps, stacked LDA / SVM scores, still each epoch alone) gains 0.161 κ: 0.721, above EEGNet and EEG Conformer, level with
  DeepSleepNet's released training (p = 0.10), 0.047 below CBraMod; N1 F1 0.31.
- Caveats: one split; 5 seeds or 3 runs per model; Welch tests without multiple-comparison correction; one untuned
  recipe per model. The two sleep models with their own code follow their own training and checkpoint rules.

### Emotion (SEED, SEED-V) and mental workload (STEW): all models on one pipeline per dataset

SEED and SEED-V run through CBraMod's released SEED-V pipeline (1-s samples of 62 channels, its trainer and metric code) on a subject
split (SEED 9 / 3 / 3, SEED-V 10 / 3 / 3 subjects; CBraMod's own trial split, with every subject in training and test, is out of
scope). STEW runs 4-s windows through the same trainer with 7-fold GroupKFold across subjects. Seeds 0–4; checkpoint at the best
validation κ. Balanced accuracy (SEED, SEED-V) and accuracy (STEW), mean ± SD over seeds; best per column in bold. Details:
[cbramod/finetune-seed](methods/cbramod/experiments/finetune-seed), [cbramod/finetune-stew](methods/cbramod/experiments/finetune-stew);
every test: `python tools/compare_seed.py`, `python tools/compare_stew.py`.

| Model | SEED | SEED-V | STEW, rest vs. task | STEW, 3 levels |
|---|---|---|---|---|
| CBraMod | 0.544 ± 0.009 | 0.222 ± 0.007 | 0.800 ± 0.009 | 0.566 ± 0.009 |
| CBraMod scratch | 0.515 ± 0.029 | 0.205 ± 0.013 | 0.699 ± 0.006 | 0.496 ± 0.008 |
| LaBraM-Base | **0.582 ± 0.011** | 0.221 ± 0.010 | 0.818 ± 0.011 | **0.567 ± 0.008** |
| EEG Conformer | 0.538 ± 0.010 | 0.237 ± 0.009 | 0.736 ± 0.015 | 0.512 ± 0.017 |
| EEGNet | 0.523 ± 0.024 | 0.247 ± 0.008 | 0.665 ± 0.006 | 0.497 ± 0.020 |
| DGCNN | 0.521 ± 0.014 | 0.256 ± 0.010 | 0.775 ± 0.007 | 0.552 ± 0.007 |
| ATCNet (bd) | 0.541 ± 0.006 | 0.253 ± 0.007 | 0.726 ± 0.005 | 0.529 ± 0.006 |
| DeepSleepNet (bd) | 0.503 ± 0.029 | 0.242 ± 0.012 | 0.717 ± 0.010 | 0.547 ± 0.009 |
| dDTF-GL (no network) | 0.524 ± 0.002 | 0.232 ± 0.002 | 0.740 ± 0.004 | 0.519 ± 0.005 |
| dDTF-GL+ (tuned on validation data) | 0.547 ± 0.005 | **0.258 ± 0.004** | **0.819 ± 0.002** | 0.564 ± 0.002 |
| chance (majority class) | 0.333 | 0.200 | 0.500 | 0.467 |

- **LaBraM-Base is first on SEED and STEW's 3 levels, and level with dDTF-GL+ on rest vs. task** (0.818 vs. 0.819, p = 0.82). On
  SEED it is 0.035 above the next model (dDTF-GL+, p < 0.001); CBraMod, ATCNet, dDTF-GL+ and EEG Conformer follow within 0.01.
- **Across subjects, SEED-V is at chance for every model** (balanced accuracy 0.20–0.26 for 5 classes; κ 0.01–0.07). Published SEED-V
  results, CBraMod's included, use the trial split, with the test subjects in the training set.
- **Pre-training pays most on STEW (+0.10 accuracy)** and little on the emotion datasets across subjects: SEED +0.030 (p = 0.086),
  SEED-V +0.018 (p = 0.034), both near chance.
- **DGCNN** is the best network without pre-training on STEW (0.775 and 0.552); on SEED it is level with EEGNet and dDTF-GL (0.521).
- **dDTF-GL, without a network,** is level with DGCNN and EEGNet on SEED and with EEG Conformer on STEW (0.740, p = 0.65), below
  DGCNN ([ddtf-gl](methods/ddtf-gl)). **dDTF-GL+** gains 0.079 on STEW rest vs. task, 0.045 on 3 levels and 0.024 on SEED (band power
  alone; second there, level with CBraMod, p = 0.54, and ATCNet, p = 0.10). Its own paper's STEW protocol, with every network on the
  same windows, is in Sections 1–3.
- **ATCNet (braindecode)** is level with CBraMod on SEED (0.541, p = 0.53), and on STEW's 4-s windows mid-table (0.726, level with
  EEG Conformer, 0.061 above EEGNet); on Sleep-EDF's 30-s epochs, in CBraMod's sleep-staging trainer, it tied CBraMod.
- **STEW's 3 levels are barely decodable across subjects:** at most 0.574 accuracy (braindecode's LaBraM-Base; majority class 0.467),
  κ 0.33. The dataset paper's 69% / κ 0.46 comes from another model and an unstated protocol. L-SeqSleepNet, off-label on 30-s
  windows of three channels, reaches 0.767 on rest vs. task and 0.541 (κ 0.27) on 3 levels
  ([l-seqsleepnet/stew-offlabel](methods/l-seqsleepnet/experiments/stew-offlabel)).
- DGCNN's own subject-independent SEED protocol (leave one subject out: 79.95% → 77.58%) is in Sections 1–2.

## 6. braindecode re-implementations

Each model was also run with braindecode 1.8.1's implementation (pinned as `methods/eegnet/upstream`), swapped into
the pipeline of its released-code experiment with everything else unchanged: data, splits, training loop or script,
seeds, checkpoint rule, metric code and GPUs. ATCNet's released code is TensorFlow, so its training loops were ported
to PyTorch instead, with control runs (braindecode's network with the Keras training behaviour) showing that the port
reproduces the Keras runs. Before training, each experiment's `code/check_equivalence.py` loads the same weights into
both implementations and compares them.

### Is it the same model?

| Model | Pre-trained weights | Forward pass with the same weights | What differs in training |
|---|---|---|---|
| EEG Conformer | – (trained from scratch) | logits differ (max 8.6e-4 at initialisation): braindecode's attention divides the query–key products by √(head dim) = 2, the released code by √(embedding) = √40 ≈ 6.3; identical once the released scaling is patched in | that scaling (a 3.2× sharper softmax); the unused `clshead` branch (244 parameters) is not built |
| CBraMod | braindecode's Hugging Face file = the released `pretrained_weights.pth` (211 of 211 tensors identical) | encoder outputs identical (difference 0) | the task head: braindecode keeps the pre-training `proj_out` and adds one linear layer (19K–205K parameters); the released fine-tuning model drops `proj_out` and adds a 3-layer MLP (14–41M). On Sleep-EDF both use the released sequence head, so only `proj_out` differs (features identical once it is applied) |
| LaBraM-Base | braindecode's file = the released checkpoint's 221 `student.` tensors | logits identical (difference 0), with the same head; on Sleep-EDF input (FPZ, PZ, 15-s halves) features identical | nothing; braindecode's `from_pretrained()` cannot load the weights for 4–5-s windows, and the model lacks the `no_weight_decay()` LaBraM's optimizer calls (both handled in the runner, with identical parameter groups) |
| ATCNet | – (trained per subject); the authors' trained Keras weights load one to one (195 tensors) | identical predictions once BatchNorm's eps is Keras' 1e-3 (with braindecode's 1e-4 / 1e-5, 34 of 2,592 predictions change) | the final `MaxNormLinear` limits the norm per input feature, where the 2022 Keras model limits it per output unit (today's Keras model has no max-norm there, but L2 and a conv max-norm); `Dropout2d` in the conv block; attention dropout on the output instead of the weights; PyTorch BatchNorm momentum and initialisation |
| EEGNet | – | not compared with the authors' Keras code | against our PyTorch EEGNet: max-norm constraints, Keras batch-norm defaults and Glorot initialisation; same parameter count |

### braindecode − original implementation, same pipeline

Cross-subject, balanced accuracy, Welch p (`python tools/compare_methods.py`, second table; EEGNet is compared with
our PyTorch implementation):

| Dataset | CBraMod | CBraMod, no pre-training | LaBraM-Base | EEGNet | EEG Conformer |
|---|---|---|---|---|---|
| PhysioNet-MI | -0.0197 (p = 0.020) | -0.0716 (p < 0.001) | +0.0077 (p = 0.163) | -0.0124 (p = 0.053) | +0.0075 (p = 0.352) |
| BCI-IV-2a | +0.0132 (p = 0.521) | +0.0127 (p = 0.317) | -0.0102 (p = 0.715) | -0.0033 (p = 0.874) | +0.0038 (p = 0.790) |
| Mumtaz2016 | -0.0106 (p = 0.339) | -0.0150 (p = 0.262) | -0.0012 (p = 0.845) | +0.0103 (p = 0.596) | +0.0157 (p = 0.400) |
| MentalArithmetic | +0.0153 (p = 0.586) | -0.0215 (p = 0.432) | -0.0111 (p = 0.744) | -0.0132 (p = 0.597) | +0.0062 (p = 0.879) |

Sleep staging, Sleep-EDF-78 in CBraMod's sleep-staging pipeline, on the common test set (A − B, Welch p;
`python tools/compare_sleepedf.py`, last table). Every model there feeds the released sequence head, so braindecode's
CBraMod differs from the released one only by the `proj_out` layer it keeps:

| Model | κ, original | κ, braindecode | braindecode − original, κ | braindecode − original, balanced acc. |
|---|---|---|---|---|
| CBraMod | 0.7676 | 0.7677 | +0.0001 (p = 0.991) | −0.0144 (p = 0.166) |
| LaBraM-Base | 0.7727 | 0.7773 | +0.0046 (p = 0.356) | +0.0059 (p = 0.559) |
| EEG Conformer | 0.6876 | 0.7011 | +0.0135 (p = 0.470) | +0.0350 (p = 0.089) |
| EEGNet (vs. our PyTorch EEGNet) | 0.6815 | 0.6808 | −0.0007 (p = 0.947) | +0.0013 (p = 0.894) |

ATCNet, BCI-IV-2a leave one subject out (released Keras code vs. braindecode's `ATCNet` in the PyTorch port of the same loop,
same seed; accuracy %, Wilcoxon over the 9 folds; [braindecode-bciciv2a](methods/atcnet/experiments/braindecode-bciciv2a)): Keras
69.12, braindecode 67.36, −1.76 pp (p = 0.20).

- **LaBraM-Base, EEGNet and EEG Conformer: braindecode reproduces our runs.** No metric differs significantly (LaBraM-Base: all
  p ≥ 0.12; EEGNet: closest PhysioNet-MI, −0.012, p = 0.053; EEG Conformer on CBraMod's four datasets: all p ≥ 0.35). EEG Conformer's
  different attention scaling changes nothing measurable.
- **ATCNet: the same network, with its final max-norm over another axis.** braindecode's `MaxNormLinear` limits the norm per input
  feature where the Keras model limits it per output unit. Across subjects (LOSO) braindecode's model is 1.8 pp below the Keras code
  (p = 0.20); in the shared pipelines it is level with the foundation models (Section 5). The within-subject runs that measure the
  axis's effect are subject-dependent and no longer reported ([braindecode-bciciv2a](methods/atcnet/experiments/braindecode-bciciv2a)).
- **On Sleep-EDF every braindecode re-run is within noise of the original** (κ: all p ≥ 0.36). One stage
  differs: braindecode's EEG Conformer scores N3 F1 0.71 against 0.56 (p = 0.047). Its attention scaling and its
  larger micro-batch (batch norm over 32 sequences instead of 4) both differ there, and the runs do not separate the
  two.
- **CBraMod: braindecode's head costs 0.02 balanced accuracy on PhysioNet-MI** (fine-tuned), within noise on the
  other datasets. On Sleep-EDF, where both implementations feed the same released sequence head, the two match
  (κ +0.0001). Two control runs of the released model with its own one-layer head (PhysioNet-MI) locate the gap:
  fine-tuned, the one-layer head scores like braindecode's (0.6042 vs. 0.6068), so the head's depth explains it; with
  the backbone frozen, depth does not matter (0.5491 vs. 0.5370 for the 3-layer head) but probing `proj_out`'s output,
  as braindecode does, costs 0.10 (0.4450).
- **No braindecode version is systematically closer to its paper.** Metrics within noise of the paper (CBraMod's datasets):
  CBraMod 0 of 12 (released code 3), LaBraM-Base 9 of 12 (released 10), EEGNet 8 of 12 (our PyTorch EEGNet 6), EEG Conformer
  3 of 12 (released 2). In balanced accuracy the two implementations differ significantly only for CBraMod on PhysioNet-MI
  (its head).
- **Pitfalls for braindecode users found on the way:** braindecode 1.8.1 declares `torch>=2.0` but fails to import
  below 2.1; `Labram.from_pretrained()` raises a size mismatch for windows other than 15 s; `CBraMod.from_pretrained()`
  loads with `strict=False`, so a renamed tensor would be skipped silently (it is not, here); `EEGConformer`'s
  attention is not the released model's; `MaxNormLinear` (in `ATCNet` and `EEGTCNet`) is documented as the Keras
  `max_norm` equivalent but normalizes over the other axis; `ATCNet`'s BatchNorm eps differs from the official
  model's, and `source_optimizer_param_groups()` applies half its L2 penalty.

### STEW

On STEW's rest vs. task every braindecode model is within noise of the original implementation (all p ≥ 0.06). On the 3 levels two
fall behind: braindecode's CBraMod (its one-layer head again) by 0.0205 (p = 0.021) and EEG Conformer by 0.0219 (p = 0.018).
braindecode's `DGCNN` computes the same function as our port of the authors' code; given our DE input it matches it on STEW (+0.009),
but as it ships, with raw samples as node features, it loses 0.106 there and 0.078–0.099 on Mumtaz2016 and MentalArithmetic, and gains
0.087–0.096 on motor imagery ([dgcnn/cbramod-splits](methods/dgcnn/experiments/cbramod-splits); Section 3).

## 7. Baselines: what each paper compared against, and what we re-ran

| Paper | Baselines it reports | Where its baseline numbers come from | Re-run here | Outcome |
|---|---|---|---|---|
| EEG Conformer | FBCSP, ConvNet, EEGNet, C2CM, FBCNet, DRDA (BCI-IV-2a, Table II) | not stated, and no baseline code is released; the FBCSP and DRDA rows look copied from their own papers | EEGNet, through Conformer's released training loop (within-subject: kept, no longer reported) | – (subject-dependent; across subjects EEG Conformer and EEGNet are level on BCI-IV-2a, p = 0.25, Section 5) |
| ATCNet | EEGNet, EEG-TCNet, TCNet_Fusion (Table IV, per subject), and 6 more models in Table V | the three in Table IV are the authors' runs, "following the same procedure defined in this article" (Sec. III-G), i.e. with the same test-set checkpoint and best-of-10 rule; the others are copied from their papers (marked as not reproduced) | not yet (the three are in the released `models.py`) | – |
| LaBraM | BIOT and the baselines of BIOT's paper (SPaRCNet, ContraWR, CNN-Transformer, …) on TUAB and TUEV | copied from BIOT Tables 4–5, taking the best BIOT variant ("we choose the best results") | no (needs TUH access) | two benchmarks re-ran LaBraM-Base: TUEV 0.5958 and 0.5905 vs. 0.6409 in the paper |
| CBraMod | supervised baselines re-implemented with BIOT's code (incl. EEGNet and EEG Conformer), BIOT, LaBraM-Base | its own runs on most datasets; the TUAB/TUEV rows are copied from LaBraM, unmarked | EEGNet, EEG Conformer and LaBraM-Base, under CBraMod's exact pipeline (Section 5) | LaBraM-Base matches the paper on 10 of 12 metrics; the EEGNet and EEG Conformer rows do not reproduce in either direction (both above the paper on BCI-IV-2a, below on the other datasets); EEGNet is best by mean on BCI-IV-2a |
| CBraMod, ablations | no pre-training (T4), frozen backbone (T18) | its own runs | both | pre-training +0.053 balanced accuracy on PhysioNet-MI (paper +0.022); frozen backbone 0.5370 (paper 0.3845) |
| DeepSleepNet | earlier Sleep-EDF and MASS methods (hand-crafted features, CNNs) | copied from their papers (the released `summary.py` prints their confusion matrices) | no | – |
| L-SeqSleepNet | SeqSleepNet (its main baseline) and about 20 published Sleep-EDF systems, DeepSleepNet among them (Table III) | SeqSleepNet: the authors' runs; the others copied. DeepSleepNet's per-stage F1 there (W 86.7, N1 45.5, N2 85.1, N3 83.3, REM 82.6) differ from DeepSleepNet's own Table IV (84.7, 46.6, 85.9, 84.8, 82.4) | DeepSleepNet with its released code on the same Sleep-EDF-78 split (Section 5) | L-SeqSleepNet leads the released DeepSleepNet by 0.070 κ (p = 0.003); braindecode's DeepSleepNet in CBraMod's pipeline is 0.010 κ behind it |
| dDTF-GL | LSTM, TSception, EEGNet, ShallowFBCSPNet, Deep4Net, EEG-ITNet, each on raw EEG and on dDTF maps (Tables 1–4), and Safari et al.'s method (Table 3) | the networks: its own runs, with no settings given; Safari's row (0.895 ± 0.014) is Safari et al.'s published 89.53 ± 1.36%, copied and unmarked | the repository's seven networks in the paper's subject-independent setting, on the same windows and splits (raw input, final epoch); EEGNet is the one model in both | our EEGNet on raw EEG matches the paper's (0.665 vs. 0.683); LaBraM-Base (0.863), CBraMod (0.845), DGCNN (0.805) and EEG Conformer (0.784) score above dDTF-GL (0.774); the paper's networks on dDTF maps not re-run |

## 8. Issues found, by type

| Type | EEG Conformer | ATCNet | CBraMod | LaBraM | DeepSleepNet | L-SeqSleepNet |
|---|---|---|---|---| --- | --- |
| Test-set model selection | yes: the released code reports the maximum over 2000 test evaluations (its protocol is within-subject; the effect is no longer reported) | yes: the checkpoint is the held-out subject's best test epoch (LOSO: +8.8 pp over a validation-split checkpoint), and within-subject `test()` also keeps each subject's best of 10 runs; neither is stated in the paper | no | no | no: the final model, no selection | no: best validation accuracy |
| Seeds | not reported; a random seed every run | not reported; none set in the 2022 code | 5, not named; code default 3407 | 5, not named; README uses seed 0 | not reported; none set | not reported; none set |
| Paper vs. released code | 3 FC layers, not 2; other scripts use a different depth and number of heads; S&R augmentation never called for SEED; `BCIIV2a.m` hard-codes subject 6 | the paper-era code and results were deleted and the model rewritten in Dec. 2023 (today's `main_TrainTest.py` scores +5.4 pp per run under the old protocol); windows averaged, not concatenated; an extra dropout layer; the 2022 LOSO loader cannot run (arguments in the wrong order); Table IV subject 4: 81.0 vs. 80.90 in the log | classifier head trains at 5× the stated lr (`multi_lr`); BCI-IV-2a has 5,184 samples, not 5,088; no 0.5–45 Hz filter on MentalArithmetic; Mumtaz2016 split differs | checkpoint chosen on validation accuracy, not κ / AUROC; single-GPU runs never reshuffle; 50 Hz notch on US (60 Hz) recordings | `prepare_physionet.py` stops at SC4362F0 on Sleep-EDF-78 (written for the 20-subject set; 82 of 153 nights unwritten); `predict.py` drops each night's last n mod 25 epochs, and the paper's epoch counts (41,950) are after that drop | the run scripts test every fold on subject 0's list (`test_list_n1.txt`), while `evaluate.py` scores fold n against its own subject; the scripts fold the sequence 20 × 10 and train a fixed 11,000 steps, where the paper describes 10 × 20 with early stopping; the evaluation batch (640 sequences × 200 epochs) does not fit a 12 GB GPU; the released data are unfiltered although the preparation script band-passes 0.3–40 Hz |
| Baseline numbers | source not stated | EEGNet, EEG-TCNet, TCNet_Fusion re-run by the authors under the same test-selected protocol; others copied | TUAB/TUEV rows copied from LaBraM, unmarked | copied from BIOT, best variant | copied from their papers | SeqSleepNet re-run; DeepSleepNet's per-stage F1 cited differently from its own paper |
| Model size and cost | exact (789,816 parameters, +17.62% for attention) | 115.2K counts the 1,440 BatchNorm statistics; 113,732 trainable | under-counted: 4.88M vs. 4.0M parameters, 1,596M vs. 318.9M FLOPs (thop misses attention; Section 3) | Large and Huge cannot be checked (weights not released) | – | – |
| Statistics | 3 of 9 significance claims fail on the paper's own per-subject numbers (C2CM, FBCNet, DRDA; within-subject) | no tests reported | – | – | no spread reported (one cross-validation) | mean ± SD over 5 repetitions |
| braindecode implementation | attention scaled by 1/√(head dim), not 1/√(embedding); no measurable effect | final-layer max-norm over the wrong axis: −5.2 pp in the paper's protocol, recovered by fixing the axis; BatchNorm eps differs (the paper's models: 85.38 → 84.53%); `source_optimizer_param_groups()` applies half the Keras L2 penalty | task head differs (pre-training `proj_out` kept, one linear layer); −0.02 on PhysioNet-MI fine-tuned, −0.09 frozen | identical; `from_pretrained()` fails for windows other than 15 s; no `no_weight_decay()` | single-epoch: its BiLSTM sees one epoch, not 25; no peepholes | not in braindecode |

Found in the 2026-11-08 round:

- **DGCNN (released code):** the authors' archive holds no subject-independent script (our LOSO runs use the port) and does not say
  which DE smoothing it used. Its within-subject script selects on the test set (subject-dependent; no longer reported).
- **CBraMod (released code):** `model_for_seedv.Model.forward` flattens the features before a classifier that needs 4-D input and
  raises `KeyError` with its default head; the released `Trainer` keeps no checkpoint and crashes when validation κ never exceeds 0
  (3 seeds of CBraMod without pre-training on SEED-V). Both worked around without editing `upstream/`.
- **Datasets:** SEED-V ships subject 7's first session twice; the `_repaired` file lacks two channels, so only the original gives
  the paper's 117,744 samples. `GroupKFold` folds on STEW differ between scikit-learn / numpy versions (equal group sizes): pinned.
- **Implementations:** torcheeg's DGCNN uses powers of the adjacency matrix, not Chebyshev polynomials of a Laplacian; two GitHub
  re-implementations add BatchNorm and drop the 256-unit layer. braindecode's matches the authors' code.

Found in the 2026-11-12 round (dDTF-GL):

- **Released code vs. paper:** the authors' code bundle computes wavelet (DWT) statistics of the raw signal, not dDTF maps, and has no
  auxiliary XGBoost; it scores 0.845 subject-independent on STEW, above the paper's 0.766 for the dDTF method.
- **The paper's input:** "following [4]": Safari et al. release their SIFT dDTF maps of STEW but not the code that made them. On those
  maps dDTF-GL reproduces the paper's subject-independent result (0.774, SD over subjects 0.147, against 0.766 ± 0.148); its GPU
  variant gives the same (0.773; the paper: 0.765). Our own generator agrees with their maps only at r = 0.37 after calibration (0.13 before).
- **The paper's claim of the highest accuracy:** in its own subject-independent setting, on identical windows and splits, LaBraM-Base, CBraMod, DGCNN and
  EEG Conformer are ahead (SI 0.863, 0.845, 0.805, 0.784); the paper's baselines are weaker models, and its raw-EEG EEGNet reproduces.
- **STEW's raw files:** the Emotiv DC offset (about 4,300 µV) and slow drift are in them. The 14 channel means of a 4-s window alone
  separate rest from task at 0.703 in unseen subjects; a 0.3-Hz high-pass (as in the shared STEW pipeline) costs the bundle 0.048.
- **Protocol and spread:** the subject-independent split is not described and its ± is a spread over subjects. Safari et al.'s
  notebook averages window pairs across subject boundaries.
- **Baseline and claims:** "Safari's" row is the other paper's published number; the announced comparison of FLOPs, model size and
  carbon is not in the manuscript (Section 3 counts its FLOPs). In our ablation the auxiliary XGBoost changes nothing and dDTF maps lose to the bundle's DWT
  statistics.

## 9. Where this sits in the literature

Of 21 review, survey, benchmark and method papers, none of the six reviews without their own experiments re-runs
a method; they tabulate self-reported numbers. All ten benchmarks re-run methods, but none puts a re-run next to
the number the method's own paper reports. A per-method paper vs. re-run table, as in Sections 2 and 5 and
[RESULTS.md](RESULTS.md), is therefore new. See
[docs/literature/review-papers-reproduction-vs-citation.md](docs/literature/review-papers-reproduction-vs-citation.md).

## 10. Open items

- Apply for TUH access, then re-run LaBraM-Base and CBraMod on LaBraM's own TUAB/TUEV splits.
- Run the remaining public CBraMod datasets (ISRUC, CHB-MIT); ISRUC is CBraMod's own sleep benchmark, which the
  Sleep-EDF pipeline is built from. BCI-IV-2b (EEG Conformer's) across subjects.
- DGCNN: DREAMER (access by request), across subjects.
- Sleep-EDF: L-SeqSleepNet's repetitions 2–5 on Sleep-EDF-20 (to compare with the paper's SD); DeepSleepNet on
  Sleep-EDF-78 with validation-based checkpoint selection, to confirm that the final-model rule causes its gap.
- ATCNet: its Table IV baselines (EEGNet, EEG-TCNet, TCNet_Fusion, in the released `models.py`) in its LOSO protocol with a
  validation split.
- braindecode: the from-scratch gap of braindecode's CBraMod on PhysioNet-MI (−0.072) was not controlled (a released
  one-layer head trained from scratch would test the head's depth there). The attention scaling of `EEGConformer`,
  the window-length limit of `Labram.from_pretrained()`, the `torch>=2.0` requirement, and `MaxNormLinear`'s axis,
  `ATCNet`'s BatchNorm eps and the factor 2 in `ATCNet.source_optimizer_param_groups()` could be reported to
  braindecode (the axis fix is one argument: `dim=1`).
- dDTF-GL: run deep networks on dDTF maps (the paper's Tables 1–2 claim that connectivity input beats raw EEG; its other baselines,
  ShallowFBCSPNet, Deep4Net, TSception, EEG-ITNet, LSTM, are not in this repository); ask Safari et al. for their SIFT settings (or the
  code) so that maps for the other datasets can match theirs. dDTF-GL+: the search covered four stages greedily on 1–2 seeds; a joint
  search, and the same feature blocks fed to the networks, are untested.

## Sources

| Section | Source |
|---|---|
| Numbers in Sections 1, 2, 5, 7, 8 | [RESULTS.md](RESULTS.md) and the experiment READMEs; tests from `tools/compare_methods.py` (its last table ranks every model on CBraMod's datasets), `compare_sleepedf.py`, `compare_seed.py`, `compare_stew.py` |
| braindecode re-implementations (Sections 5, 6) | [eeg-conformer/cbramod-splits](methods/eeg-conformer/experiments/cbramod-splits), [braindecode-public-datasets](methods/cbramod/experiments/braindecode-public-datasets), [labram/braindecode-cbramod-splits](methods/labram/experiments/braindecode-cbramod-splits), [eegnet/braindecode-cbramod-splits](methods/eegnet/experiments/braindecode-cbramod-splits), [atcnet/braindecode-bciciv2a](methods/atcnet/experiments/braindecode-bciciv2a) (`code/check_equivalence.py`, `results/equivalence.txt`); `tools/compare_methods.py` |
| Robustness checks, frozen backbone | [finetune-public-datasets](methods/cbramod/experiments/finetune-public-datasets) |
| EEG Conformer code mismatches | [2026-09-24 report](docs/reports/eeg-conformer/2026-09-24_eeg-conformer_weekly-report.docx) |
| Parameters and FLOPs (Section 3) | `tools/count_flops.py` → `tools/flops.json`; CBraMod's paper counts: [2026-10-01 report](docs/reports/cbramod/2026-10-01_cbramod_weekly-report.docx) |
| ATCNet: LOSO protocols; across subjects | [bciciv2a-model-selection](methods/atcnet/experiments/bciciv2a-model-selection) (`code/analyze.py`), [braindecode-cbramod-splits](methods/atcnet/experiments/braindecode-cbramod-splits); [2026-10-22 report](docs/reports/atcnet/2026-10-22_atcnet_weekly-report.docx) |
| Sleep-EDF (Sections 1, 2, 5) | [cbramod/finetune-sleepedf](methods/cbramod/experiments/finetune-sleepedf) (shared pipeline, data, common test set) and each model's Sleep-EDF experiment; [deepsleepnet](methods/deepsleepnet), [l-seqsleepnet](methods/l-seqsleepnet) experiments; tests from `tools/compare_sleepedf.py`; citation counts from OpenAlex and Semantic Scholar (2026-11-01) |
| LaBraM code issues, TUAB/TUEV benchmark re-runs | [2026-10-08 report](docs/reports/labram/2026-10-08_labram_weekly-report.docx), [literature note](docs/literature/review-papers-reproduction-vs-citation.md) |
| dDTF-GL (Sections 1, 2, 3, 5, 7, 8) | [methods/ddtf-gl](methods/ddtf-gl) and its experiments, including [feature-tuning](methods/ddtf-gl/experiments/feature-tuning) (dDTF-GL+); tests from `tools/compare_stew.py`, `compare_methods.py`, `compare_sleepedf.py`, `compare_seed.py`; [2026-11-15 report](docs/reports/ddtf-gl/2026-11-15_ddtf-gl_weekly-report.docx) |
| Scope (subject-independent only), new cross-subject runs, FLOPs, dDTF-GL+ | [2026-11-20 report](docs/reports/review-paper/2026-11-20_review-paper_weekly-report.docx) |
