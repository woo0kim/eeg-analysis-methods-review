# Summary of results

Everything validated so far, as of 2026-11-08: 5 methods (EEG Conformer, CBraMod, LaBraM, ATCNet, DGCNN), 2 sleep-staging
methods (DeepSleepNet, L-SeqSleepNet), 1 baseline added by us (EEGNet, run inside two of those methods' pipelines),
8 datasets (Sleep-EDF, SEED, SEED-V and STEW with every model on one pipeline per dataset), braindecode's
implementations re-run in the same pipelines (Section 7), and the weekly reports of 2026-09-24, 10-01, 10-08, 10-22,
11-01 and 11-08. "Paper" is the number printed in the paper (cited); "ours" is a
re-run in this repository. Every metric and setting is in [RESULTS.md](RESULTS.md), which is generated from the
experiments and is authoritative if the two disagree.

## 1. Findings at a glance

| Method | What we ran | Verdict | Key numbers |
|---|---|---|---|
| [EEG Conformer](methods/eeg-conformer) | Released code, BCI-IV-2a, within-subject, 9 subjects × 3 seeds × 2 protocols | The pipeline reproduces, but the reported number is the maximum over 2000 test-set evaluations | 78.72% under the released rule vs. 78.66% in the paper; 70.01% when the epoch is chosen on a validation split |
| [CBraMod](methods/cbramod) | Released code and weights, 4 public datasets, cross-subject, seeds 0–4 | Does not reproduce: all 12 metrics are below the paper, 9 of them significantly | balanced accuracy −0.015 (PhysioNet-MI) to −0.137 (MentalArithmetic) |
| [LaBraM-Base](methods/labram) | Released code and weights on CBraMod's 4 datasets, splits, seeds and scoring | Reproduces the CBraMod paper's LaBraM-Base numbers on 10 of 12 metrics | ties or beats CBraMod on 3 of the 4 datasets |
| [ATCNet](methods/atcnet) | Authors' code, BCI-IV-2a: the 2022 code behind the paper (from the repository history) and today's two scripts, within-subject, 9 subjects × 10 seeds each; LOSO, 9 folds × 3 seeds with each of today's scripts | The code reproduces, but 85.38% is the best of 10 runs per subject, each checkpointed on the test set | the paper's saved models give exactly 85.38%; single runs of that code 81.47% (authors' own runs 81.98%); with a validation split 79.52%; LOSO 69.06% (best of 3: 70.16%) vs. 70.97%, and 60.23% with a validation split |
| [EEGNet](methods/eegnet) (baseline), cross-subject | EEGNet-8,2 through CBraMod's own fine-tuning pipeline, same 4 datasets | With 2–4K parameters, the best model on BCI-IV-2a | 0.5288 balanced accuracy vs. 0.4571 (CBraMod, p = 0.021) and 0.4477 (LaBraM-Base, p = 0.014) |
| [EEGNet](methods/eegnet) (baseline), within-subject | braindecode EEGNet-8,2 through EEG Conformer's released training loop, BCI-IV-2a, 9 subjects × 3 seeds × 2 protocols | Conformer's lead over EEGNet (Table II) is significant only when each model's best test-set epoch is reported | Conformer − EEGNet: +6.10 pp under the released rule (p = 0.0039); −0.76 pp at the last-100-epoch mean; +2.65 pp with a validation split (p = 0.16) |
| braindecode implementations of all five models | braindecode 1.8.1's `EEGConformer`, `CBraMod`, `Labram`, `EEGNet`, `ATCNet` swapped into the same pipelines (ATCNet: a PyTorch port of its TensorFlow loops), same seeds and protocols; braindecode's pre-trained weights | `Labram` and `EEGNet` reproduce our runs on every metric, `EEGConformer` under every epoch-selection rule; `CBraMod` (different task head) does on 3 of 4 datasets; `ATCNet` falls short on every subject (its final max-norm is taken over the wrong axis). On Sleep-EDF-78, where all feed the same sequence head, `CBraMod`, `Labram`, `EEGConformer` and `EEGNet` match the originals. None comes closer to a paper than the released code | EEG Conformer, released rule: 78.56% vs. 78.72%; CBraMod, PhysioNet-MI: 0.6068 vs. 0.6265 (p = 0.02); LaBraM-Base: all p ≥ 0.12; ATCNet, paper's protocol: 76.26% vs. 81.47%; Sleep-EDF κ, braindecode − original: all p ≥ 0.36 |
| [DeepSleepNet](methods/deepsleepnet) (sleep) | Released code (TensorFlow 1.15): Sleep-EDF-20 with the paper's 20-fold CV; Sleep-EDF-78 on the shared 62 / 8 / 8 split, 3 runs | Reproduces the paper. On Sleep-EDF-78, with a training set 3.5× larger, the released final-model rule overfits | 81.81% / macro-F1 76.72 / κ 0.754 vs. 82.0 / 76.9 / 0.76, on exactly the paper's 41,950 epochs; Sleep-EDF-78: κ 0.703, the weakest sequence model |
| [L-SeqSleepNet](methods/l-seqsleepnet) (sleep) | Released code and data: Sleep-EDF-20, leave-one-subject-out, 1 of the paper's 5 repetitions; Sleep-EDF-78 on the shared split, 3 runs | Reproduces the paper once the run scripts' test list is fixed (they score every fold on subject 0) | 86.11% / κ 0.811 / macro-F1 79.44 vs. 86.3 ± 0.2 / 0.813 / 79.3; Sleep-EDF-78: κ 0.773, tied for first |
| All models, sleep staging (Sleep-EDF-78) | The five models as the epoch encoder of CBraMod's released sleep-staging pipeline (plus braindecode's DeepSleepNet), and the two sleep models with their own released code; one 62 / 8 / 8 split, all scored on the same 20,976 test epochs (Section 5) | No paper reports this split. Two foundation models, L-SeqSleepNet and ATCNet tie at the top; the two small supervised models trail | κ: LaBraM-Base 0.773, L-SeqSleepNet 0.773, ATCNet 0.771, CBraMod 0.768, DeepSleepNet (bd) 0.762, CBraMod without pre-training 0.745, DeepSleepNet (released) 0.703, EEG Conformer 0.688, EEGNet 0.681 |
| [DGCNN](methods/dgcnn) | Our PyTorch port of the authors' TensorFlow code (archived; numerically identical to braindecode's `DGCNN`); SEED in the paper's two protocols, 5 runs; then every other dataset here | The paper's 90.40% is the peak of ~80 test evaluations over each subject's best 2 of 3 sessions. The across-subject result holds up | 87.69% under the paper's rule, 77.79% with nothing chosen on the test set, level with a linear SVM (79.42%, p = 0.33); leave one subject out 77.58% vs. 79.95% (SVM 58.85%) |
| [EEG Conformer](methods/eeg-conformer) on SEED | Released `conformer_seed_1s_5fold.py`, session 1, 15 subjects × 5 folds; the same with whole clips held out, 3 seeds | Reproduces (98.24% vs. 95.30%), but its folds sit inside 4-minute clips | final epoch 96.82% inside clips, 71.59% with clips held out (p = 0.0001) |
| All models, emotion (SEED, SEED-V) | Every model (braindecode for ATCNet and DeepSleepNet) through CBraMod's released SEED-V pipeline, seeds 0–4: its trial split and a subject split | SEED-V's trial split reproduces the CBraMod paper's numbers and ranking within noise except that LaBraM-Base beats CBraMod (p = 0.011); across subjects SEED-V is at chance for every model | SEED-V, CBraMod: κ 0.250 vs. 0.257 (p = 0.38); subject split: balanced accuracy 0.22–0.26 (chance 0.20) |
| All models, mental workload (STEW) | Every model, 7-fold GroupKFold across subjects (folds pinned: they depend on library versions), 4-s windows, two tasks | Foundation models lead; pre-training adds 0.10 | rest vs. task: LaBraM-Base 0.818, CBraMod 0.800, DGCNN 0.775; 3 levels: at most 0.574 (majority class 0.467) |

## 2. Results by dataset

Every model run so far, grouped by task category and dataset. "Paper" is the number the method's paper reports;
"ours" are our re-runs: with the authors' released code, and with braindecode's implementation in the same pipeline.

| Category | Dataset | Subjects | Labels | Popularity: citations · method papers here | Protocol (metric) | Model | Model type | Paper | Ours: released code | Ours: braindecode |
|---|---|---|---|---|---|---|---|---|---|---|
| **Motor imagery classification** | **BCI-IV-2a** | 9 | 4: left hand, right hand, feet, tongue | 1,287 · 4 of 8 ᶠ | within-subject, session T → E (accuracy, %) | EEGNet | domain-specific | 74.50 ᵃ | – ᵈ | 72.62 / 67.36↓ |
| | | | | | | EEG Conformer | domain-specific | 78.66 | 78.72 / 70.01↓ | 78.56 / 69.32↓ |
| | | | | | | ATCNet | domain-specific | 85.38 | 84.18↓ / 79.52↓ ᵇ | 79.48↓ / 67.57↓ |
| | | | | | leave-one-subject-out, 9 folds (accuracy, %) | ATCNet | domain-specific | 70.97 | 70.16 (best of 3) · 69.06 (mean) / 60.23 | 67.36 (1 run) |
| | | | | | cross-subject, 5 / 2 / 2 subjects (balanced accuracy) | EEGNet | domain-specific | 0.4482 ᶜ | 0.5288↑ ᵈ | 0.5255↑ |
| | | | | | | LaBraM-Base | general | 0.4869 ᶜ | 0.4477 | 0.4375 |
| | | | | | | CBraMod | general | 0.5138 | 0.4571↓ | 0.4703↓ |
| | **PhysioNet-MI** | 109 | 4: left fist, right fist, both fists, both feet | 500 · 1 of 8 ᶠ | cross-subject, 70 / 19 / 20 subjects (balanced accuracy) | EEGNet | domain-specific | 0.5814 ᶜ | 0.5948 ᵈ | 0.5824 |
| | | | | | | LaBraM-Base | general | 0.6173 ᶜ | 0.6098 | 0.6175 |
| | | | | | | CBraMod | general | 0.6417 | 0.6265↓ | 0.6068↓ |
| **Mental disorder diagnosis** | **Mumtaz2016** | 64 (34 MDD, 30 healthy); 63 used ᵉ | 2: major depressive disorder, healthy | 269 · 1 of 8 ᶠ | cross-subject, 43 / 9 / 11 subjects (balanced accuracy) | EEGNet | domain-specific | 0.9232 ᶜ | 0.8828↓ ᵈ | 0.8931 |
| | | | | | | LaBraM-Base | general | 0.9409 ᶜ | 0.8905↓ | 0.8894↓ |
| | | | | | | CBraMod | general | 0.9560 | 0.8933↓ | 0.8827↓ |
| **Mental stress detection** | **MentalArithmetic** | 36 | 2: rest, mental arithmetic | 315 · 1 of 8 ᶠ | cross-subject, 28 / 4 / 4 subjects (balanced accuracy) | EEGNet | domain-specific | 0.6770 ᶜ | 0.5569↓ ᵈ | 0.5437↓ |
| | | | | | | LaBraM-Base | general | 0.6909 ᶜ | 0.6562 | 0.6451 |
| | | | | | | CBraMod | general | 0.7256 | 0.5889↓ | 0.6042↓ |
| **Sleep staging** | **Sleep-EDF** | 78 (Sleep-EDF-20: the first 20) | 5: W, N1, N2, N3, REM | 966 · 2 of 8 ᶠ | Sleep-EDF-20, 20-fold subject CV (accuracy, %) | DeepSleepNet | domain-specific | 82.0 | 81.81 ᵍ | – |
| | | | | | Sleep-EDF-20, leave-one-subject-out (accuracy, %) | L-SeqSleepNet | domain-specific | 86.3 ± 0.2 | 86.11 ᵍ | – ʰ |
| | | | | | Sleep-EDF-78, cross-subject, 62 / 8 / 8 subjects, common test set (κ) ⁱ | EEGNet | domain-specific | – | 0.6815 ᵈ | 0.6808 |
| | | | | | | EEG Conformer | domain-specific | – | 0.6876 | 0.7011 |
| | | | | | | ATCNet | domain-specific | – | – | 0.7709 |
| | | | | | | DeepSleepNet | domain-specific | – | 0.7030 ⁱ | 0.7625 |
| | | | | | | L-SeqSleepNet | domain-specific | – | 0.7725 | – ʰ |
| | | | | | | LaBraM-Base | general | – | 0.7727 | 0.7773 |
| | | | | | | CBraMod | general | – | 0.7676 | 0.7677 |
| **Emotion recognition** | **SEED** | 15, 3 sessions each | 3: positive, neutral, negative | 2,735 · 2 of 8 ᶠ | within-subject, session by session, 9 → 6 clips (accuracy, %) | DGCNN | domain-specific | 90.40 | 87.69 ʲ / 77.79 | – |
| | | | | | leave one subject out (accuracy, %) | DGCNN | domain-specific | 79.95 | 77.58 | – |
| | | | | | within-subject, session 1, 5 folds inside clips (accuracy, %) | EEG Conformer | domain-specific | 95.30 | 98.24↑ / 71.59 ᵏ | – |
| | | | | | CBraMod's SEED-V pipeline, trial split (balanced accuracy) | LaBraM-Base | general | – | 0.6300 | 0.6161 |
| | | | | | | CBraMod | general | – | 0.6162 | 0.6198 |
| | **SEED-V** | 16, 3 sessions each | 5: happy, sad, neutral, disgust, fear | 486 · 2 of 8 ᶠ | CBraMod's trial split, 5 / 5 / 5 clips per session (balanced accuracy) | EEGNet | domain-specific | 0.2961 ᶜ | 0.2894 | 0.2809 |
| | | | | | | EEG Conformer | domain-specific | 0.3537 ᶜ | 0.3205↓ | 0.2962↓ |
| | | | | | | LaBraM-Base | general | 0.3976 ᶜ | 0.4136 | 0.4106 |
| | | | | | | CBraMod | general | 0.4091 | 0.3962↓ | 0.3786↓ |
| **Mental workload** | **STEW** | 48 | 2: rest, multitasking test; or 3 rating levels | 219 · 0 of 8 ᶠ | 7-fold GroupKFold across subjects, rest vs. task (accuracy) | all models: Section 5 | | – | LaBraM-Base 0.818, CBraMod 0.800 | 0.815, 0.798 |

- **Model type.** *Domain-specific*: a supervised model trained from scratch on the task's own data (EEGNet, EEG
  Conformer, ATCNet, and the sleep models DeepSleepNet and L-SeqSleepNet). *General*: an EEG foundation model, pre-trained on large unlabeled EEG corpora and fine-tuned for
  each task (LaBraM-Base, CBraMod).
- **Within-subject cells "x / y"**: x follows the paper's own reporting rule, y picks the epoch on a 20% validation
  split of session T. The reporting rules both select on the test set: EEG Conformer (and EEGNet in its training loop)
  report each run's best of 2000 test-set evaluations; ATCNet reports each subject's best of 10 runs, each run
  checkpointed on the test set (2022 code; y: `main_TrainValTest.py`, mean of runs). Runs: 9 subjects × 3 seeds
  (EEG Conformer, EEGNet), × 10 seeds (ATCNet; braindecode: 10 for x, 5 for y). ATCNet's LOSO cell is read the
  same way: x checkpoints on the held-out subject (`main_TrainTest.py`), y on 20% of the 8 training subjects' trials
  (`main_TrainValTest.py`, mean of 3 seeds).
- **↓ / ↑**: significantly below / above the paper (p < 0.05); no mark: within noise. Cross-subject: Welch's t-test,
  5 seeds against the paper's mean ± SD over 5 seeds. Within-subject: Wilcoxon signed-rank test over the 9 subjects
  against the paper's per-subject numbers. LOSO: not tested (the paper gives no per-subject numbers).
- **Comparisons hold within a protocol block only.** The cross-subject blocks ran all three models through CBraMod's
  pipeline (same samples, splits, seeds, checkpoint rule and metric code; tests between models in Section 5). In the
  within-subject block EEGNet and EEG Conformer share EEG Conformer's training loop; ATCNet ran its own (4.5-s
  unfiltered trials). Within-subject, LOSO and cross-subject numbers are not comparable with each other.
- ᵃ EEGNet's within-subject number is from EEG Conformer's Table II (EEGNet's own paper uses another protocol).
  ᵇ The authors' saved best-of-10 models give exactly 85.38% on our data; 84.18% is the best of 10 of our own runs,
  a maximum of noisy runs, while the mean of our runs matches the authors' (81.47% vs. 81.98%). ᶜ From the CBraMod
  paper's baseline rows (LaBraM's paper does not evaluate these datasets). ᵈ EEGNet has no released PyTorch code:
  cross-subject, this column is our own PyTorch EEGNet-8,2; within-subject only braindecode's EEGNet was run.
  ᵉ 34 patients and 30 healthy controls per the dataset's paper (Mumtaz et al. 2017); the data after CBraMod's
  preprocessing hold 63 (33 MDD, 30 healthy), counted from our preprocessing log. ᶠ Next table.
  ᵍ One run, not tested against the paper: one 20-fold cross-validation (DeepSleepNet, which reports no spread) and
  one of L-SeqSleepNet's five repetitions. ʰ Not in braindecode. ⁱ The five general-purpose models and braindecode's
  DeepSleepNet ran as the epoch encoder of CBraMod's released sleep-staging pipeline (seeds 0–4). The "released code"
  cells of DeepSleepNet and L-SeqSleepNet are their own released training on the same split (3 runs). Every model is
  scored on the same 20,976 test epochs; tests between models are in Section 5.

- ʲ DGCNN, SEED: 87.69 under the paper's own rule (peak of ~80 test evaluations, each subject's best 2 of 3 sessions), 77.79 with nothing
  chosen on the test set ([seed-paper-protocol](methods/dgcnn/experiments/seed-paper-protocol)). No significance test: the paper gives a single mean.
- ᵏ EEG Conformer, SEED: x / y = the released 5-fold split (best epoch) / whole clips held out (final epoch), the same model and session
  ([seed-5fold](methods/eeg-conformer/experiments/seed-5fold)).

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
| Parameters | 2,193–4,020 (CBraMod pipeline, by dataset); 3,444 (braindecode, BCI-IV-2a within-subject) | 789,816 (BCI-IV-2a) | 113,732 (paper: 115.2K, incl. BatchNorm statistics) | 5.8M (Base) | backbone 4.88M (paper: 4.0M); 19–46M as fine-tuned, with the default head | not counted (braindecode's single-epoch version with 2 channels: 40.5M) | not counted |
| Released weights | – | – (trained per subject) | per subject: today's code (1 seed); the paper's best-of-10 models in the repository history | Base only; Large and Huge not released | yes | – | SHHS-pretrained weights |
| Paper's own benchmarks | P300, ERN, MRCP, SMR (BCI-IV-2a) | BCI-IV-2a, BCI-IV-2b, SEED | BCI-IV-2a | TUAB, TUEV (+ SEED-V, MoBI in the appendix) | 13 downstream datasets | MASS, Sleep-EDF (20 subjects) | SHHS, Sleep-EDF (20 subjects), ear-EEG, cEEGrid |
| Evaluation protocol | within-subject (4-fold block-wise cross-validation) and cross-subject | within-subject, session T → E | within-subject, session T → E; leave-one-subject-out | cross-subject | cross-subject: subject-independent train / val / test | subject-wise cross-validation (Sleep-EDF: 20 folds), predictions pooled | leave-one-subject-out with held-out validation subjects, 5 repetitions (SHHS: 70 / 30 split) |
| Epoch selection in the released code | lowest validation loss (paper) | maximum test accuracy over 2000 epochs; no validation set | maximum test accuracy (the test session is the validation data), then the best of 10 runs by test accuracy; `main_TrainValTest.py`: lowest validation loss on 20% of session T | best validation accuracy (the paper states κ / AUROC) | best validation κ (multi-class) or AUROC (binary) | none: the final model after 200 fine-tuning epochs; validation is printed, not used | best validation accuracy, every 100 steps (the paper adds early stopping after 50 validations; the scripts run a fixed 110) |
| Seeds in the paper | – | not reported; a random seed every run | not reported; 10 runs per subject (no seed set in 2022, seeds 1–10 today) | 5, not named | 5, not named | not reported; none set | not reported; none set (5 repetitions) |
| Role here | baseline we added | method validated | method validated | method validated; CBraMod's strongest baseline | method validated | method validated (sleep model) | method validated (sleep model) |
| braindecode class (v1.8.1) | `EEGNet` | `EEGConformer`: same layers, attention scaled by 1/√(head dim) instead of 1/√(embedding size) | `ATCNet`: same layers and weights; identical predictions with Keras' BatchNorm eps (1e-3; braindecode: 1e-4 / 1e-5); its final `MaxNormLinear` limits the norm per input feature, not per output unit | `Labram`: identical model and weights | `CBraMod`: identical encoder and weights; task head = pre-training `proj_out` + one linear layer, not the released 3-layer MLP | `DeepSleepNet`: single-epoch (its BiLSTM sees one epoch), standard LSTM gates instead of peepholes | – (not in braindecode) |
| Our experiment | braindecode `EEGNet` (pinned in [methods/eegnet](methods/eegnet)) inside EEG Conformer's loop: [bciciv2a-epoch-selection](methods/eegnet/experiments/bciciv2a-epoch-selection); our own PyTorch re-implementation through CBraMod's pipeline: [pytorch-cbramod-splits](methods/eegnet/experiments/pytorch-cbramod-splits), Sleep-EDF: [sleepedf-cbramod-pipeline](methods/eegnet/experiments/sleepedf-cbramod-pipeline) | [bciciv2a-epoch-selection](methods/eeg-conformer/experiments/bciciv2a-epoch-selection), Sleep-EDF: [sleepedf-cbramod-pipeline](methods/eeg-conformer/experiments/sleepedf-cbramod-pipeline) | [bciciv2a-model-selection](methods/atcnet/experiments/bciciv2a-model-selection) | [finetune-cbramod-splits](methods/labram/experiments/finetune-cbramod-splits), Sleep-EDF: [sleepedf-cbramod-pipeline](methods/labram/experiments/sleepedf-cbramod-pipeline) | [finetune-public-datasets](methods/cbramod/experiments/finetune-public-datasets), Sleep-EDF: [finetune-sleepedf](methods/cbramod/experiments/finetune-sleepedf) | [sleepedf20-20fold](methods/deepsleepnet/experiments/sleepedf20-20fold), [sleepedf78-cbramod-split](methods/deepsleepnet/experiments/sleepedf78-cbramod-split) | [sleepedf20-loso](methods/l-seqsleepnet/experiments/sleepedf20-loso), [sleepedf78-cbramod-split](methods/l-seqsleepnet/experiments/sleepedf78-cbramod-split) |
| braindecode experiment | [braindecode-cbramod-splits](methods/eegnet/experiments/braindecode-cbramod-splits) (cross-subject), [braindecode-sleepedf](methods/eegnet/experiments/braindecode-sleepedf); the within-subject runs above already use braindecode | [braindecode-bciciv2a](methods/eeg-conformer/experiments/braindecode-bciciv2a), [braindecode-sleepedf](methods/eeg-conformer/experiments/braindecode-sleepedf) | [braindecode-bciciv2a](methods/atcnet/experiments/braindecode-bciciv2a), Sleep-EDF: [braindecode-sleepedf](methods/atcnet/experiments/braindecode-sleepedf) | [braindecode-cbramod-splits](methods/labram/experiments/braindecode-cbramod-splits), [braindecode-sleepedf](methods/labram/experiments/braindecode-sleepedf) | [braindecode-public-datasets](methods/cbramod/experiments/braindecode-public-datasets), [braindecode-sleepedf](methods/cbramod/experiments/braindecode-sleepedf) | [braindecode-sleepedf](methods/deepsleepnet/experiments/braindecode-sleepedf) (in CBraMod's sleep-staging pipeline) | – |

CBraMod's parameter and FLOP counts are from the 2026-10-01 report, EEGNet's from our run logs, and the others
from the method READMEs.

### Grouped by characteristic

| Characteristic | Group | Methods | What it means for comparisons |
|---|---|---|---|
| Model family | task-specific, trained from scratch | EEGNet, EEG Conformer, ATCNet; for sleep staging DeepSleepNet, L-SeqSleepNet | no pre-training data or released weights involved |
| | pre-trained foundation model, fine-tuned | LaBraM, CBraMod | results depend on released weights; LaBraM-Large/-Huge cannot be checked |
| Evaluation protocol | within-subject (train on session T, test on session E of the same subject) | EEG Conformer, ATCNet, EEGNet (our runs in Conformer's loop) | its BCI-IV-2a numbers (0.70–0.79 accuracy) are **not comparable** with the cross-subject rows (0.45–0.53 balanced accuracy) |
| | cross-subject (subject-independent splits) | CBraMod, LaBraM, EEGNet (our runs in CBraMod's pipeline) | directly comparable: identical samples, splits, seeds, checkpoint rule and metric code (Section 5) |
| | sleep staging, Sleep-EDF-78, one 62 / 8 / 8 subject split | all seven: five in CBraMod's sleep-staging pipeline, DeepSleepNet and L-SeqSleepNet also with their own released code | comparable on the common test set (20,976 epochs every model scores; Section 5) |
| | Sleep-EDF-20, the sleep papers' own protocols | DeepSleepNet (20-fold CV), L-SeqSleepNet (leave-one-subject-out) | reproductions of the papers; the two protocols and epoch sets differ |
| Epoch selection in released code | on the test set | EEG Conformer, ATCNet (which also keeps the best of 10 runs by test accuracy) | inflates the reported number, and Conformer's more than EEGNet's (Section 6) |
| | on a validation set | CBraMod, LaBraM, L-SeqSleepNet | no test-set leakage |
| | none (final model) | DeepSleepNet | no leakage, but the fixed training length overfits on a larger training set (Sleep-EDF-78) |
| Reproduction outcome | reproduces | LaBraM-Base (10 / 12 metrics); EEG Conformer under its own rule; ATCNet under its own rule (the paper's models and the 2022 code); DeepSleepNet and L-SeqSleepNet on Sleep-EDF-20 | |
| | does not reproduce | CBraMod (3 / 12 metrics within noise) | |
| | mixed | EEGNet (6 / 12 within noise: above the paper on all BCI-IV-2a metrics and PhysioNet-MI weighted F1, below on MentalArithmetic and Mumtaz2016 balanced accuracy) | the paper's EEGNet came from BIOT's code with unreported settings |

## 4. Datasets

### Datasets run here

| Category | Dataset | Task (classes) | Subjects: train / val / test | Channels | Input as run | Samples | Metrics (checkpoint metric in bold) | Run with | Paper tables |
|---|---|---|---|---|---|---|---|---|---|
| Motor imagery classification | BCI-IV-2a | left hand, right hand, feet, tongue (4) | cross-subject 5 / 2 / 2; within-subject: each of 9 subjects, session T → E | 22 | 4 s at 200 Hz (CBraMod pipeline); 4 s at 250 Hz (EEG Conformer's pipeline, also for EEGNet); 4.5 s at 250 Hz, unfiltered, z-scored (ATCNet) | 5,184 (CBraMod pipeline); 288 + 288 trials per subject (EEG Conformer, ATCNet) | balanced acc., **κ**, weighted F1; EEG Conformer, ATCNet: accuracy (κ) | EEG Conformer, ATCNet, CBraMod, LaBraM-Base, EEGNet | CBraMod T15; EEG Conformer T II; ATCNet T IV–VI |
| Motor imagery classification | PhysioNet-MI | left fist, right fist, both fists, both feet (4) | 70 / 19 / 20 | 64 | 4 s at 200 Hz | 9,837 | balanced acc., **κ**, weighted F1 | CBraMod, LaBraM-Base, EEGNet | CBraMod T3, T4, T18 |
| Mental disorder diagnosis | Mumtaz2016 | major depressive disorder vs. healthy (2) | 43 / 9 / 11 | 19 | 5 s at 200 Hz | 7,143 | balanced acc., AUC-PR, **AUROC** | CBraMod, LaBraM-Base, EEGNet | CBraMod T10 |
| Mental stress detection | MentalArithmetic | rest vs. mental arithmetic ("mental stress") (2) | 28 / 4 / 4 | 20 (LaBraM: 19, no A2-A1) | 5 s at 200 Hz | 1,707 | balanced acc., AUC-PR, **AUROC** | CBraMod, LaBraM-Base, EEGNet | CBraMod T12 |
| Sleep staging | Sleep-EDF (Expanded, Sleep Cassette) | W, N1, N2, N3, REM (5) | Sleep-EDF-78: 62 / 8 / 8, age-stratified; Sleep-EDF-20: 20-fold CV (DeepSleepNet), leave-one-subject-out (L-SeqSleepNet) | 2 (Fpz-Cz, Pz-Oz); the sleep models' released code: Fpz-Cz | CBraMod pipeline: 30-s epochs at 200 Hz, 0.3–35 Hz, in sequences of 20, sleep period ± 30 min; DeepSleepNet: raw 30-s epochs at 100 Hz in sequences of 25; L-SeqSleepNet: spectrograms of 30-s epochs at 100 Hz in sequences of 200 | 192,840 epochs (pipeline, Sleep-EDF-78); 41,950 (DeepSleepNet) and 46,241 (L-SeqSleepNet) test epochs on Sleep-EDF-20 | balanced acc., **κ**, weighted F1, accuracy, macro-F1 | all seven | DeepSleepNet T III–IV; L-SeqSleepNet T III |
| Emotion recognition | SEED | positive, neutral, negative (3) | trial split: all 15, clips 1–5 / 6–10 / 11–15; subject split 9 / 3 / 3; DGCNN's protocols: each session, 9 → 6 clips, and leave one subject out | 62 | 1 s at 200 Hz, 0.3–75 Hz (CBraMod pipeline); DE features (DGCNN's protocols); 1 s, 4–47 Hz, z-scored (EEG Conformer's code) | 152,730 | balanced acc., **κ**, weighted F1, accuracy, clip vote | all eight | DGCNN abstract; EEG Conformer README |
| Emotion recognition | SEED-V | happy, sad, neutral, disgust, fear (5) | trial split: all 16, 5 / 5 / 5 clips; subject split 10 / 3 / 3 | 62 | 1 s at 200 Hz, 0.3–75 Hz | 117,744 | balanced acc., **κ**, weighted F1 | all eight | CBraMod T2 |
| Mental workload | STEW | rest vs. task (2); rating 1–3 / 4–6 / 7–9 (3) | 7-fold GroupKFold across subjects (48; 45 with ratings), validation = next fold | 14 | 4-s windows, 25% overlap, 128 Hz (CBraMod, LaBraM: 200 Hz) | 4,704 | accuracy, balanced acc., **κ**, weighted F1 | all eight | – |

### Which datasets each method's paper evaluates on

● = the paper evaluates its own method on this dataset; ✓ = we ran this model on it (results in Section 2);
●✓ = both. Download and access details: [datasets/README.md](datasets/README.md).

| Category | Dataset | EEGNet | EEG Conformer | ATCNet | LaBraM | CBraMod | DeepSleepNet | L-SeqSleepNet | DGCNN | Status here |
|---|---|---|---|---|---|---|---|---|---|---|
| **Motor imagery classification** | BCI-IV-2a | ●✓ ᵃ | ●✓ | ●✓ | ✓ | ●✓ |  |  | ✓ | run: within-subject, leave-one-subject-out, cross-subject |
|  | BCI-IV-2b |  | ● |  |  |  |  |  |  | public, not run yet (no download key) |
|  | PhysioNet-MI | ✓ |  |  | ✓ | ●✓ |  |  | ✓ | run: cross-subject |
|  | SHU-MI |  |  |  |  | ● |  |  |  | not run: files are password-protected |
| **Emotion recognition** | SEED | ✓ | ●✓ | ✓ ˡ | ✓ | ✓ | ✓ | ✓ ᵐ | ●✓ | run: all models in CBraMod's pipeline (trial and subject splits); EEG Conformer's and DGCNN's own protocols |
|  | SEED-V | ✓ | ✓ | ✓ ˡ | ●✓ ᵇ | ●✓ | ✓ | ✓ ᵐ | ✓ | run: all models in CBraMod's pipeline (trial and subject splits) |
|  | FACED |  |  |  |  | ● |  |  |  | not run: Synapse account |
|  | DREAMER |  |  |  |  |  |  |  | ● | not run: access by request, not requested |
| **Mental workload** | STEW | ✓ | ✓ | ✓ ˡ | ✓ | ✓ | ✓ | ✓ ᵐ | ✓ | run: all models, 7-fold GroupKFold across subjects |
| **Mental disorder diagnosis** | Mumtaz2016 | ✓ |  |  | ✓ | ●✓ |  |  | ✓ | run: cross-subject |
| **Mental stress detection** | MentalArithmetic | ✓ |  |  | ✓ | ●✓ |  |  | ✓ | run: cross-subject |
| **Vigilance estimation** | SEED-VIG |  |  |  |  | ● |  |  |  | not run: license application |
| **Sleep staging** | Sleep-EDF | ✓ | ✓ | ✓ | ✓ | ✓ | ●✓ ᶜ | ●✓ ᶜ | ✓ | run: Sleep-EDF-20 with the sleep papers' protocols; Sleep-EDF-78 cross-subject, all models |
|  | MASS |  |  |  |  |  | ● |  |  | not run: access request |
|  | SHHS |  |  |  |  |  |  | ● |  | not run: NSRR access |
|  | ear-EEG, cEEGrid |  |  |  |  |  |  | ● |  | not run; access not checked |
|  | ISRUC |  |  |  |  | ● |  |  |  | public, not run yet (too large for one round) |
| **Seizure detection** | CHB-MIT |  |  |  |  | ● |  |  |  | public, not run yet (too large for one round) |
| **Imagined speech classification** | BCIC2020-3 |  |  |  |  | ● |  |  |  | not run: test labels not public |
| **Abnormal EEG detection** | TUAB |  |  |  | ● | ● |  |  |  | not run: TUH registration |
| **Event type classification** | TUEV |  |  |  | ● | ● |  |  |  | not run: TUH registration |
| **Event-related potentials** | P300 (visual evoked potentials) | ● |  |  |  |  |  |  |  | not run; access not checked |
|  | ERN (feedback error-related negativity) | ● |  |  |  |  |  |  |  | not run; access not checked |
|  | MRCP (movement-related cortical potentials) | ● |  |  |  |  |  |  |  | not run; access not checked |
| **Gait prediction (regression)** | MoBI |  |  |  | ● ᵇ |  |  |  |  | not run; access not checked |
|  | **Datasets in the paper / run here** | **4 / 1** | **3 / 2** | **1 / 1** | **4 / 1** | **13 / 5** | **2 / 1** | **4 / 1** | **2 / 1** |  |

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
  the supervised baselines with BIOT's code; the TUAB/TUEV rows copied from LaBraM; Section 8); EEG Conformer's
  Table II and ATCNet's Table IV report EEGNet on BCI-IV-2a.
- **Pre-training data is not evaluation data.** LaBraM pre-trains on about 2,500 hours from about 20 datasets, CBraMod
  on the TUH EEG corpus (TUEG); neither appears in the table.

## 5. Cross-subject results: all models on identical splits

On CBraMod's four public datasets, all four models ran on CBraMod's preprocessed samples and subject splits, with
seeds 0–4. The checkpoint was the best validation κ / AUROC epoch, the test set was scored once, and the metric code
was the same for every model. Each cell is **ours, mean ± std over 5 seeds (paper)**. Every paper number is from the
CBraMod paper; its EEGNet and LaBraM-Base rows are the CBraMod authors' runs. Sleep staging, with all seven models on
one split, is the [last subsection](#sleep-staging-on-sleep-edf-78-all-seven-models-on-one-split).

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

### The same comparison with braindecode's implementations

The four models were re-run with braindecode 1.8.1's `EEGNet`, `CBraMod` and `Labram` (braindecode's pre-trained
weights for the foundation models), each swapped into the same pipeline as above, with the same splits, seeds,
checkpoint rule and metric code (Section 7). Balanced accuracy, mean ± SD over seeds 0–4; in parentheses the
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
  0.02 there, while braindecode's LaBraM-Base is 0.008 above the released one (Section 7).
- **EEGNet's lead on BCI-IV-2a does not depend on the implementation** (braindecode: +0.055 over CBraMod, p = 0.002;
  +0.088 over LaBraM-Base, p = 0.007), and neither does the Mumtaz2016 tie.
- **Pre-training helps braindecode's CBraMod on PhysioNet-MI by +0.105** (released code: +0.053), because
  braindecode's from-scratch model is the weaker one there (0.5022 vs. 0.5738); elsewhere it changes nothing
  (p ≥ 0.32).

### Sleep staging on Sleep-EDF-78: all seven models on one split

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
| DeepSleepNet | released (TF 1.15) | its own | 0.7030 ± 0.0103 | 0.7247 ± 0.0104 | 0.7140 ± 0.0121 | 0.427 | 0.697 | 0.764 |
| EEG Conformer | released | CBraMod pipeline | 0.6876 ± 0.0190 | 0.6448 ± 0.0267 | 0.6585 ± 0.0235 | 0.347 | 0.562 | 0.663 |
| EEGNet | ours (PyTorch) | CBraMod pipeline | 0.6815 ± 0.0154 | 0.6852 ± 0.0137 | 0.6785 ± 0.0073 | 0.311 | 0.787 | 0.594 |

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

- **Four models tie at the top.** LaBraM-Base, L-SeqSleepNet, ATCNet and CBraMod lie within 0.005 κ, and none of the
  tested pairs among them differs (p ≥ 0.40). braindecode's LaBraM-Base (Section 7) scores 0.777, the highest κ of all
  runs, and is within noise of all four (p ≥ 0.08).
- **Pre-training helps on sleep staging.** Both foundation models beat CBraMod without pre-training (+0.023 and
  +0.028 κ). On CBraMod's public datasets pre-training helped only on PhysioNet-MI.
- **The foundation models do not beat the specialised models.** L-SeqSleepNet, trained with its own released code, is
  level with them. ATCNet, whose encoder has 112K parameters and no pre-training, is level with CBraMod (4.9M,
  pre-trained).
- **DeepSleepNet's training rule costs more than its architecture.** The released code keeps the final fine-tuning
  epoch. On this split its validation accuracy peaks at epochs 7–10 of 200 and then falls while training accuracy
  reaches 0.99, giving κ 0.703. braindecode's DeepSleepNet in the pipeline, which keeps the best validation epoch,
  gains 0.060 κ and, with braindecode's LaBraM (Section 7), has the highest balanced accuracy of all models (0.767).
- **The two small supervised models trail by 0.08–0.09 κ** (both p < 0.001). On BCI-IV-2a EEGNet beat both
  foundation models. Here EEGNet is weakest on REM and N1, and EEG Conformer on N3.
- Caveats: one split; 5 seeds or 3 runs per model; Welch tests without multiple-comparison correction; one untuned
  recipe per model. The two sleep models with their own code follow their own training and checkpoint rules.

### Emotion (SEED, SEED-V) and mental workload (STEW): all models on one pipeline per dataset

SEED and SEED-V run through CBraMod's released SEED-V pipeline (1-s samples of 62 channels, its trainer and metric code) under two
splits: CBraMod's **trial** split (every subject in training and test, clips 1–5 / 6–10 / 11–15 of each session) and a **subject** split
(SEED 9 / 3 / 3, SEED-V 10 / 3 / 3). STEW runs 4-s windows through the same trainer with 7-fold GroupKFold across subjects. Seeds 0–4;
checkpoint at the best validation κ. Balanced accuracy (SEED, SEED-V) and accuracy (STEW), mean ± SD over seeds; best per column in bold.
Details: [cbramod/finetune-seed](methods/cbramod/experiments/finetune-seed), [cbramod/finetune-stew](methods/cbramod/experiments/finetune-stew);
every test: `python tools/compare_seed.py`, `python tools/compare_stew.py`.

| Model | SEED, trial | SEED, subject | SEED-V, trial | SEED-V, subject | STEW, rest vs. task | STEW, 3 levels |
|---|---|---|---|---|---|---|
| CBraMod | 0.616 ± 0.014 | 0.544 ± 0.009 | 0.396 ± 0.006 | 0.222 ± 0.007 | 0.800 ± 0.009 | 0.566 ± 0.009 |
| CBraMod scratch | 0.587 ± 0.005 | – | 0.274 ± 0.102 | – | 0.699 ± 0.006 | 0.496 ± 0.008 |
| LaBraM-Base | **0.630 ± 0.001** | **0.582 ± 0.011** | **0.414 ± 0.009** | 0.221 ± 0.010 | **0.818 ± 0.011** | **0.567 ± 0.008** |
| EEG Conformer | 0.536 ± 0.007 | 0.538 ± 0.010 | 0.320 ± 0.014 | 0.237 ± 0.009 | 0.736 ± 0.015 | 0.512 ± 0.017 |
| EEGNet | 0.491 ± 0.009 | 0.523 ± 0.024 | 0.289 ± 0.009 | 0.247 ± 0.008 | 0.665 ± 0.006 | 0.497 ± 0.020 |
| DGCNN | 0.574 ± 0.017 | 0.521 ± 0.014 | 0.387 ± 0.005 | **0.256 ± 0.010** | 0.775 ± 0.007 | 0.552 ± 0.007 |
| ATCNet (bd) | 0.489 ± 0.018 | 0.541 ± 0.006 | 0.270 ± 0.004 | 0.253 ± 0.007 | 0.726 ± 0.005 | 0.529 ± 0.006 |
| DeepSleepNet (bd) | 0.591 ± 0.009 | 0.503 ± 0.029 | 0.400 ± 0.003 | 0.242 ± 0.012 | 0.717 ± 0.010 | 0.547 ± 0.009 |
| chance (majority class) | 0.333 | 0.333 | 0.200 | 0.200 | 0.500 | 0.467 |

- **LaBraM-Base is the best model in five of the six columns.** On SEED-V, the CBraMod paper's own benchmark, it beats CBraMod by
  0.017 (p = 0.011); the paper ranks CBraMod first. CBraMod itself is within noise of its paper there in κ and weighted F1 and 0.013 lower
  in balanced accuracy (p = 0.041), its closest match in this repository. EEG Conformer is 0.033 below the paper's number for it (p = 0.003),
  EEGNet matches.
- **Across subjects, SEED-V is at chance for every model** (balanced accuracy 0.22–0.26 for 5 classes; κ 0.02–0.07). Published SEED-V
  results, which use the trial split, depend on the test subjects being in the training set. On SEED the subject split costs the
  foundation models, DGCNN and DeepSleepNet 0.05–0.09, but not EEG Conformer, EEGNet or ATCNet (+0.002, +0.032, +0.052): for the
  models that barely fit 1-s samples, unseen clips are as hard as unseen people.
- **Pre-training pays most on STEW (+0.10 accuracy) and least on SEED (+0.029).** On SEED-V, CBraMod without pre-training never leaves
  chance in 3 of 5 seeds; the two that train match the pre-trained model.
- **DGCNN** is the best model without pre-training on STEW and beats EEG Conformer and EEGNet on both trial splits and STEW (not on
  SEED's subject split, 0.521 against 0.538 and 0.523); braindecode's DeepSleepNet, a sleep model run off-label, is ahead of it on both
  trial splits and level with CBraMod on SEED-V's.
- **ATCNet (braindecode) underfits 1-s samples:** 0.270 on SEED-V's trial split, the lowest of all models, and 0.489 on SEED's,
  level with EEGNet (p = 0.87). On SEED-V its training loss falls only from 1.60 to 1.42 in 50 epochs (chance level ln 5 = 1.61;
  EEGNet 1.41, DGCNN 0.81, CBraMod 0.40): under CBraMod's training settings the two small CNNs barely fit 1-s windows. Across
  subjects it loses nothing: 0.541 on SEED (level with CBraMod, p = 0.53; second to LaBraM-Base) and 0.253 on SEED-V. On STEW's
  4-s windows it fits and is mid-table (0.726, level with EEG Conformer, 0.061 above EEGNet); on Sleep-EDF's 30-s epochs, in
  CBraMod's sleep-staging trainer, it tied CBraMod.
- **STEW's 3 levels are barely decodable across subjects:** at most 0.574 accuracy (majority class 0.467), κ 0.33. The dataset paper's
  69% / κ 0.46 comes from another model and an unstated protocol. L-SeqSleepNet, off-label on 30-s windows of three channels, reaches
  0.767 on rest vs. task and 0.541 (κ 0.27) on 3 levels ([l-seqsleepnet/stew-offlabel](methods/l-seqsleepnet/experiments/stew-offlabel)).
- The two papers' own SEED protocols are in Sections 1–2: EEG Conformer's folds inside clips (98.2% → 71.6% with clips held out) and
  DGCNN's test-selected rule (90.4% reported, 77.8% without test selection).

## 6. Within-subject results on BCI-IV-2a: EEG Conformer, EEGNet and ATCNet

The same 27 runs per model (9 subjects × 3 seeds), with the reported epoch chosen by different rules. The
validation-selected row is a separate set of 27 runs per model with 20% of session T held out. EEGNet
(braindecode EEGNet-8,2, 3,444 parameters) ran through EEG Conformer's released training loop with only the model
swapped: same data, augmentation, optimizer, 2000 epochs, seeds and validation trials. p is a two-sided Wilcoxon
signed-rank test over the 9 subjects, the paper's test; for the paper row it is recomputed from Table II.

| How the reported epoch is chosen | EEG Conformer, acc. % (κ) | EEGNet, acc. % (κ) | Conformer − EEGNet (p) | EEG Conformer, braindecode, acc. % (κ) | braindecode Conformer − EEGNet (p) |
|---|---|---|---|---|---|
| Paper, Table II | 78.66 (0.7155) | 74.50 (0.6600) | +4.17 pp (0.0497) | – | – |
| Released rule: maximum test accuracy over 2000 epochs | 78.72 (0.7162) | 72.62 (0.6349) | +6.10 pp (0.0039) | 78.56 (0.7142) | +5.94 pp (0.0039) |
| Validation-selected epoch | 70.01 (0.6001) | 67.36 (0.5648) | +2.65 pp (0.16) | 69.32 (0.5909) | +1.95 pp (0.43) |
| Final epoch | 69.21 (0.5895) | 69.35 (0.5914) | −0.14 pp (1.00) | 68.84 (0.5845) | −0.51 pp (0.91) |
| Mean of the last 100 epochs | 68.77 (0.5836) | 69.53 (0.5937) | −0.76 pp (0.65) | 67.96 (0.5728) | −1.57 pp (0.36) |
| Control: scrambled training labels, released rule (subjects 1–6; chance 25.00) | 35.15 (–) | – | (report only) | – | – |

- **The released rule reproduces Conformer's number** (+0.06 pp). Every rule that does not look at the test set
  gives 8.7–9.9 pp less.
- **Conformer's lead over EEGNet is significant only under that rule.** At the last-100-epoch mean or the
  final epoch of the same runs, EEGNet is level or slightly ahead. With a validation split Conformer leads by
  2.65 pp on 6 of 9 subjects, which is not significant with 9 subjects. On those same runs EEGNet's
  last-100-epoch mean is 1.07 pp higher, so the lead comes from the validation selection itself.
- **The maximum inflates Conformer more because its test accuracy fluctuates more.** Over the last 100 epochs
  of a run its SD is 5.52 pp, against 1.10 pp for EEGNet. The maximum therefore sits 9.95 pp above Conformer's
  last-100 mean and 3.09 pp above EEGNet's.
- **None of this depends on the Conformer implementation.** braindecode's `EEGConformer` in the same loop
  (Section 7) gives the same numbers under every rule and leads EEGNet significantly only under the released rule
  (+5.94 pp, 9 of 9 subjects, p = 0.0039).
- Caveat: the loop is Conformer's (Adam lr 2e-4), not EEGNet's own recipe, and it was not tuned for EEGNet.

The scrambled-label control is from the 2026-09-24 report. Its code was later removed, so it cannot be
recomputed from this repository. The same "best test epoch" rule applied to our LaBraM-Base runs would add +0.006
(PhysioNet-MI) to +0.065 (MentalArithmetic) balanced accuracy
([selection analysis](methods/labram/experiments/finetune-cbramod-splits#how-much-the-reported-number-depends-on-the-reporting-rule)).

### ATCNet

ATCNet ran through its own released code, 9 subjects × 10 seeds per protocol. Its pipeline is not EEG Conformer's
(4.5-s unfiltered, z-scored trials instead of 4-s band-passed ones; its own optimizer, early stopping and validation
split), so the two columns compare protocols, not identical pipelines.

| How the reported number is chosen | EEG Conformer, acc. % | ATCNet, 2022 code (paper), acc. % (κ) | ATCNet, today's code, acc. % (κ) |
|---|---|---|---|
| Paper | 78.66 (Table II) | 85.38 (0.805) (Tables IV–V) | – |
| Best test epoch of each run, then the best of 10 runs per subject (ATCNet's released `test()`) | – | 84.18 (0.789); the authors' own runs: 85.38 | 88.50 (0.847) |
| Best test epoch of each run, mean of runs (EEG Conformer's released rule) | 78.72 | 81.47 (0.753); the authors' own runs: 81.98 | 86.89 (0.825) |
| Validation-selected epoch | 70.01 | – | 79.52 (0.727) (`main_TrainValTest.py`) |

- **Both papers report a number chosen on the test set; ATCNet's adds a second choice.** On top of the best test
  epoch of each run, ATCNet's `test()` keeps each subject's best of 10 runs: +2.72 pp in our runs of the 2022 code,
  +3.40 pp in the authors' (9 of 9 subjects, p = 0.0039 each).
- **ATCNet's lead over EEG Conformer survives a validation split** (79.52% vs. 70.01%; paper: 85.38 vs. 78.66), with
  the caveat that the pipelines differ. Moving to validation selection costs Conformer 8.71 pp and ATCNet's current
  code 7.37 pp.
- **The authors' current code is not the code behind the paper.** Under the same test-set protocol, today's
  `main_TrainTest.py` (rewritten 2023-12-28: L2 weight decay, max-norm, lr schedule) scores 5.42 pp more per run than
  the 2022 code (9 of 9 subjects).

## 7. braindecode re-implementations

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
| ATCNet | – (trained per subject); the authors' trained Keras weights load one to one (195 tensors) | identical predictions once BatchNorm's eps is Keras' 1e-3; with braindecode's 1e-4 / 1e-5 the paper's best-of-10 models score 84.53% instead of 85.38% (34 of 2,592 predictions change) | the final `MaxNormLinear` limits the norm per input feature, where the 2022 Keras model limits it per output unit (today's Keras model has no max-norm there, but L2 and a conv max-norm); `Dropout2d` in the conv block; attention dropout on the output instead of the weights; PyTorch BatchNorm momentum and initialisation |
| EEGNet | – | not compared with the authors' Keras code | against our PyTorch EEGNet: max-norm constraints, Keras batch-norm defaults and Glorot initialisation; same parameter count |

### braindecode − original implementation, same pipeline

Cross-subject, balanced accuracy, Welch p (`python tools/compare_methods.py`, second table; EEGNet is compared with
our PyTorch implementation):

| Dataset | CBraMod | CBraMod, no pre-training | LaBraM-Base | EEGNet |
|---|---|---|---|---|
| PhysioNet-MI | -0.0197 (p = 0.020) | -0.0716 (p < 0.001) | +0.0077 (p = 0.163) | -0.0124 (p = 0.053) |
| BCI-IV-2a | +0.0132 (p = 0.521) | +0.0127 (p = 0.317) | -0.0102 (p = 0.715) | -0.0033 (p = 0.874) |
| Mumtaz2016 | -0.0106 (p = 0.339) | -0.0150 (p = 0.262) | -0.0012 (p = 0.845) | +0.0103 (p = 0.596) |
| MentalArithmetic | +0.0153 (p = 0.586) | -0.0215 (p = 0.432) | -0.0111 (p = 0.744) | -0.0132 (p = 0.597) |

Sleep staging, Sleep-EDF-78 in CBraMod's sleep-staging pipeline, on the common test set (A − B, Welch p;
`python tools/compare_sleepedf.py`, last table). Every model there feeds the released sequence head, so braindecode's
CBraMod differs from the released one only by the `proj_out` layer it keeps:

| Model | κ, original | κ, braindecode | braindecode − original, κ | braindecode − original, balanced acc. |
|---|---|---|---|---|
| CBraMod | 0.7676 | 0.7677 | +0.0001 (p = 0.991) | −0.0144 (p = 0.166) |
| LaBraM-Base | 0.7727 | 0.7773 | +0.0046 (p = 0.356) | +0.0059 (p = 0.559) |
| EEG Conformer | 0.6876 | 0.7011 | +0.0135 (p = 0.470) | +0.0350 (p = 0.089) |
| EEGNet (vs. our PyTorch EEGNet) | 0.6815 | 0.6808 | −0.0007 (p = 0.947) | +0.0013 (p = 0.894) |

Within-subject, EEG Conformer on BCI-IV-2a (27 runs per protocol, accuracy %, Wilcoxon over the 9 subjects):

| How the reported epoch is chosen | braindecode | released model | braindecode − released (p) |
|---|---|---|---|
| Released rule: maximum test accuracy over 2000 epochs | 78.56 | 78.72 | −0.15 pp (0.57) |
| Mean of the last 100 epochs | 67.96 | 68.77 | −0.81 pp (0.20) |
| Final epoch | 68.84 | 69.21 | −0.37 pp (0.73) |
| Validation-selected epoch | 69.32 | 70.01 | −0.69 pp (0.57) |

Within-subject, ATCNet on BCI-IV-2a (released Keras code vs. braindecode's `ATCNet` in the PyTorch port of the same
loop, same seeds; accuracy %, Wilcoxon over the 9 subjects; [braindecode-bciciv2a](methods/atcnet/experiments/braindecode-bciciv2a)):

| Protocol | Keras | braindecode, defaults | − Keras (p) | control: braindecode + Keras training, − Keras (p) |
|---|---|---|---|---|
| 2022 loop (paper), mean of 10 runs | 81.47 | 76.26 | −5.20 pp (0.0039) | +0.72 pp (0.57) |
| 2022 loop, best of 10 per subject (paper's rule) | 84.18 | 79.48 | −4.71 pp (0.012) | – |
| `main_TrainTest.py`, mean of 5 runs | 86.76 | 76.41 | −10.35 pp (0.0039) | +0.58 pp (0.055) |
| `main_TrainValTest.py`, mean of 5 runs | 80.29 | 67.57 | −12.72 pp (0.0039) | −1.12 pp (0.13) |
| LOSO, 1 run | 69.12 | 67.36 | −1.76 pp (0.20) | – |

- **ATCNet: the same network, trained differently.** In the 2022 loop an ablation switches single Keras behaviours on
  in braindecode's defaults. Only the final layer's max-norm axis matters: limiting the norm per output unit (as Keras
  does) instead of per input feature recovers +3.68 pp with braindecode's own parametrization (+4.57 pp with a Keras
  projection), leaving −0.89 pp (p = 0.43). Dropout2d, attention-dropout placement, BatchNorm and initialisation each
  move the result by ≤ 1.5 pp. In today's loops the Keras model also has L2 and a conv max-norm; braindecode's
  documented "faithful" options (`conv_max_norm_const=0.6`, `source_optimizer_param_groups()`) close two thirds of the
  gap (−3.29 / −4.63 pp remain). With `main_TrainValTest.py` the validation loss has no L2 term for braindecode's
  defaults, so its checkpoint lands early (epoch 187 vs. 444).
- **LaBraM-Base and EEGNet: braindecode reproduces our runs.** No metric differs significantly (LaBraM-Base: all
  p ≥ 0.12; EEGNet: closest PhysioNet-MI, −0.012, p = 0.053).
- **EEG Conformer: the different attention scaling changes nothing measurable.** braindecode's model reaches the
  paper's number under the released rule (78.56% vs. 78.66%) and is level with the released model under every rule;
  its test accuracy fluctuates as much over the last 100 epochs (SD 5.71 vs. 5.52 pp), so the released rule inflates
  it just as much (+10.61 vs. +9.95 pp over the last-100 mean). A validation split gives 69.32% (released model:
  70.01%), and the comparison with EEGNet comes out as with the released model (Section 6).
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
- **No braindecode version comes closer to its paper.** Metrics within noise of the paper: CBraMod 0 of 12 (released
  code 3), LaBraM-Base 9 of 12 (released 10), EEGNet 8 of 12 (our PyTorch EEGNet 6); ATCNet under the paper's
  protocol and rule 79.48% (Keras 84.18%, paper 85.38%).
- **Pitfalls for braindecode users found on the way:** braindecode 1.8.1 declares `torch>=2.0` but fails to import
  below 2.1; `Labram.from_pretrained()` raises a size mismatch for windows other than 15 s; `CBraMod.from_pretrained()`
  loads with `strict=False`, so a renamed tensor would be skipped silently (it is not, here); `EEGConformer`'s
  attention is not the released model's; `MaxNormLinear` (in `ATCNet` and `EEGTCNet`) is documented as the Keras
  `max_norm` equivalent but normalizes over the other axis; `ATCNet`'s BatchNorm eps differs from the official
  model's, and `source_optimizer_param_groups()` applies half its L2 penalty.

### SEED, SEED-V and STEW

On SEED's trial split and STEW's rest vs. task every braindecode model is within noise of the original implementation (all
p ≥ 0.06). On the two harder tasks two models fall behind: braindecode's CBraMod (its one-layer head again) by 0.0176 on SEED-V
(p = 0.011) and 0.0205 on STEW's 3 levels (p = 0.021), EEG Conformer by 0.0243 (p = 0.019) and 0.0219 (p = 0.018); on SEED-V
also DGCNN on our DE input, by 0.0114 (p = 0.005). Balanced accuracy, Welch's t-test over 5 seeds. braindecode's `DGCNN` computes the same function as our port of the authors' code; given our DE input it
matches it on SEED and STEW (+0.001, +0.009), but as it
ships, with raw samples as node features, it loses 0.055 (SEED), 0.102 (SEED-V) and 0.106 (STEW), and gains 0.096–0.107 on motor
imagery ([dgcnn/cbramod-splits](methods/dgcnn/experiments/cbramod-splits)).

## 8. Baselines: what each paper compared against, and what we re-ran

| Paper | Baselines it reports | Where its baseline numbers come from | Re-run here | Outcome |
|---|---|---|---|---|
| EEG Conformer | FBCSP, ConvNet, EEGNet, C2CM, FBCNet, DRDA (BCI-IV-2a, Table II) | not stated, and no baseline code is released; the FBCSP and DRDA rows look copied from their own papers | EEGNet, through Conformer's released training loop with the same seeds, protocols and epoch-selection rules (Section 6); ConvNet not yet | Conformer's lead over EEGNet is significant only when each model's best test-set epoch is reported (+6.10 pp, p = 0.0039); with a validation split it is +2.65 pp, p = 0.16. EEGNet's own paper chose the epoch on validation data |
| ATCNet | EEGNet, EEG-TCNet, TCNet_Fusion (Table IV, per subject), and 6 more models in Table V | the three in Table IV are the authors' runs, "following the same procedure defined in this article" (Sec. III-G), i.e. with the same test-set checkpoint and best-of-10 rule; the others are copied from their papers (marked as not reproduced) | not yet (the three are in the released `models.py`) | – |
| LaBraM | BIOT and the baselines of BIOT's paper (SPaRCNet, ContraWR, CNN-Transformer, …) on TUAB and TUEV | copied from BIOT Tables 4–5, taking the best BIOT variant ("we choose the best results") | no (needs TUH access) | two benchmarks re-ran LaBraM-Base: TUEV 0.5958 and 0.5905 vs. 0.6409 in the paper |
| CBraMod | supervised baselines re-implemented with BIOT's code (incl. EEGNet and EEG Conformer), BIOT, LaBraM-Base | its own runs on most datasets; the TUAB/TUEV rows are copied from LaBraM, unmarked | EEGNet and LaBraM-Base, under CBraMod's exact pipeline (Section 5) | LaBraM-Base matches the paper on 10 of 12 metrics; EEGNet is best on BCI-IV-2a |
| CBraMod, ablations | no pre-training (T4), frozen backbone (T18) | its own runs | both | pre-training +0.053 balanced accuracy on PhysioNet-MI (paper +0.022); frozen backbone 0.5370 (paper 0.3845) |
| DeepSleepNet | earlier Sleep-EDF and MASS methods (hand-crafted features, CNNs) | copied from their papers (the released `summary.py` prints their confusion matrices) | no | – |
| L-SeqSleepNet | SeqSleepNet (its main baseline) and about 20 published Sleep-EDF systems, DeepSleepNet among them (Table III) | SeqSleepNet: the authors' runs; the others copied. DeepSleepNet's per-stage F1 there (W 86.7, N1 45.5, N2 85.1, N3 83.3, REM 82.6) differ from DeepSleepNet's own Table IV (84.7, 46.6, 85.9, 84.8, 82.4) | DeepSleepNet with its released code on the same Sleep-EDF-78 split (Section 5) | L-SeqSleepNet leads the released DeepSleepNet by 0.070 κ (p = 0.003); braindecode's DeepSleepNet in CBraMod's pipeline is 0.010 κ behind it |

## 9. Issues found, by type

| Type | EEG Conformer | ATCNet | CBraMod | LaBraM | DeepSleepNet | L-SeqSleepNet |
|---|---|---|---|---| --- | --- |
| Test-set model selection | yes: maximum over 2000 test evaluations (≈ +9 pp); it also creates the significant lead over EEGNet (Section 6) | yes, twice: the checkpoint is the best test epoch (+3.6 pp over the last-50-epoch mean; LOSO +8.6 pp, and +8.8 pp over a validation-split checkpoint) and `test()` keeps each subject's best of 10 runs (+2.7 to +3.4 pp); neither is stated in the paper | no | no | no: the final model, no selection | no: best validation accuracy |
| Seeds | not reported; a random seed every run | not reported; none set in the 2022 code | 5, not named; code default 3407 | 5, not named; README uses seed 0 | not reported; none set | not reported; none set |
| Paper vs. released code | 3 FC layers, not 2; other scripts use a different depth and number of heads; S&R augmentation never called for SEED; `BCIIV2a.m` hard-codes subject 6 | the paper-era code and results were deleted and the model rewritten in Dec. 2023 (today's `main_TrainTest.py` scores +5.4 pp per run under the old protocol); windows averaged, not concatenated; an extra dropout layer; the 2022 LOSO loader cannot run (arguments in the wrong order); Table IV subject 4: 81.0 vs. 80.90 in the log | classifier head trains at 5× the stated lr (`multi_lr`); BCI-IV-2a has 5,184 samples, not 5,088; no 0.5–45 Hz filter on MentalArithmetic; Mumtaz2016 split differs | checkpoint chosen on validation accuracy, not κ / AUROC; single-GPU runs never reshuffle; 50 Hz notch on US (60 Hz) recordings | `prepare_physionet.py` stops at SC4362F0 on Sleep-EDF-78 (written for the 20-subject set; 82 of 153 nights unwritten); `predict.py` drops each night's last n mod 25 epochs, and the paper's epoch counts (41,950) are after that drop | the run scripts test every fold on subject 0's list (`test_list_n1.txt`), while `evaluate.py` scores fold n against its own subject; the scripts fold the sequence 20 × 10 and train a fixed 11,000 steps, where the paper describes 10 × 20 with early stopping; the evaluation batch (640 sequences × 200 epochs) does not fit a 12 GB GPU; the released data are unfiltered although the preparation script band-passes 0.3–40 Hz |
| Baseline numbers | source not stated | EEGNet, EEG-TCNet, TCNet_Fusion re-run by the authors under the same test-selected protocol; others copied | TUAB/TUEV rows copied from LaBraM, unmarked | copied from BIOT, best variant | copied from their papers | SeqSleepNet re-run; DeepSleepNet's per-stage F1 cited differently from its own paper |
| Model size and cost | exact (789,816 parameters, +17.62% for attention) | 115.2K counts the 1,440 BatchNorm statistics; 113,732 trainable | under-counted: 4.88M vs. 4.0M parameters, 1,596M vs. 318.9M FLOPs | Large and Huge cannot be checked (weights not released) | – | – |
| Statistics | 3 of 9 significance claims fail on the paper's own per-subject numbers (C2CM, FBCNet, DRDA) | no tests reported | – | – | no spread reported (one cross-validation) | mean ± SD over 5 repetitions |
| braindecode implementation | attention scaled by 1/√(head dim), not 1/√(embedding); no measurable effect | final-layer max-norm over the wrong axis: −5.2 pp in the paper's protocol, recovered by fixing the axis; BatchNorm eps differs (the paper's models: 85.38 → 84.53%); `source_optimizer_param_groups()` applies half the Keras L2 penalty | task head differs (pre-training `proj_out` kept, one linear layer); −0.02 on PhysioNet-MI fine-tuned, −0.09 frozen | identical; `from_pretrained()` fails for windows other than 15 s; no `no_weight_decay()` | single-epoch: its BiLSTM sees one epoch, not 25; no peepholes | not in braindecode |

Found in the 2026-11-08 round:

- **DGCNN (reported number):** the authors' `eeg_DE.py` validates on the test set, keeps the peak of ~80 test evaluations and averages
  each subject's best 2 of 3 sessions; 87.69% that way against 77.79% without test selection. The archive holds no subject-independent
  script and does not say which DE smoothing it used.
- **EEG Conformer (protocol):** `conformer_seed_1s_5fold.py` makes its folds inside 4-minute clips, so test windows have training
  neighbours one second away: 96.8% against 71.6% with clips held out.
- **CBraMod (released code):** `model_for_seedv.Model.forward` flattens the features before a classifier that needs 4-D input and
  raises `KeyError` with its default head; the released `Trainer` keeps no checkpoint and crashes when validation κ never exceeds 0
  (3 seeds of CBraMod without pre-training on SEED-V). Both worked around without editing `upstream/`.
- **Datasets:** SEED-V ships subject 7's first session twice; the `_repaired` file lacks two channels, so only the original gives
  the paper's 117,744 samples. `GroupKFold` folds on STEW differ between scikit-learn / numpy versions (equal group sizes): pinned.
- **Implementations:** torcheeg's DGCNN uses powers of the adjacency matrix, not Chebyshev polynomials of a Laplacian; two GitHub
  re-implementations add BatchNorm and drop the 256-unit layer. braindecode's matches the authors' code.

## 10. Where this sits in the literature

Of 21 review, survey, benchmark and method papers, none of the six reviews without their own experiments re-runs
a method; they tabulate self-reported numbers. All ten benchmarks re-run methods, but none puts a re-run next to
the number the method's own paper reports. A per-method paper vs. re-run table, as in Sections 5–6 and
[RESULTS.md](RESULTS.md), is therefore new. See
[docs/literature/review-papers-reproduction-vs-citation.md](docs/literature/review-papers-reproduction-vs-citation.md).

## 11. Open items

- Apply for TUH access, then re-run LaBraM-Base and CBraMod on LaBraM's own TUAB/TUEV splits.
- Re-run EEG Conformer on BCI-IV-2b and SEED with a validation split. Run ConvNet (its other CNN baseline)
  through the same loop as EEGNet. Optionally, run EEGNet with its own recipe (Adam lr 1e-3, 128 Hz) to see how
  much the shared loop costs it.
- Run the remaining public CBraMod datasets (ISRUC, CHB-MIT); ISRUC is CBraMod's own sleep benchmark, which the
  Sleep-EDF pipeline is built from.
- DGCNN: run the authors' archived TensorFlow code unmodified on SEED (`methods/dgcnn/experiments/seed-paper-protocol/code/run_authors.sh`,
  awaiting approval) to separate a port difference from a data difference in the 2.7 pp gap; read the paper itself (closed access); DREAMER.
- SEED: L-SeqSleepNet with sequences confined to one clip, to confirm that its subject-split score reads the clip order.
- Sleep-EDF: L-SeqSleepNet's repetitions 2–5 on Sleep-EDF-20 (to compare with the paper's SD); DeepSleepNet on
  Sleep-EDF-78 with validation-based checkpoint selection, to confirm that the final-model rule causes its gap.
- ATCNet: re-run its Table IV baselines (EEGNet, EEG-TCNet, TCNet_Fusion, all in the released `models.py`) under
  `main_TrainValTest.py`, to see whether its 4.7–5.8 pp lead over them (Table V) survives without test-set
  selection; the ablations
  (Table II, Figs. 7–8) with a validation split.
- braindecode: the from-scratch gap of braindecode's CBraMod on PhysioNet-MI (−0.072) was not controlled (a released
  one-layer head trained from scratch would test the head's depth there). The attention scaling of `EEGConformer`,
  the window-length limit of `Labram.from_pretrained()`, the `torch>=2.0` requirement, and `MaxNormLinear`'s axis,
  `ATCNet`'s BatchNorm eps and the factor 2 in `ATCNet.source_optimizer_param_groups()` could be reported to
  braindecode (the axis fix is one argument: `dim=1`).

## Sources

| Section | Source |
|---|---|
| Numbers in Sections 1, 2, 5, 6, 8 | [RESULTS.md](RESULTS.md) and the experiment READMEs; pairwise tests from `tools/compare_methods.py` |
| braindecode re-implementations (Sections 5, 7) | [braindecode-bciciv2a](methods/eeg-conformer/experiments/braindecode-bciciv2a), [braindecode-public-datasets](methods/cbramod/experiments/braindecode-public-datasets), [labram/braindecode-cbramod-splits](methods/labram/experiments/braindecode-cbramod-splits), [eegnet/braindecode-cbramod-splits](methods/eegnet/experiments/braindecode-cbramod-splits), [atcnet/braindecode-bciciv2a](methods/atcnet/experiments/braindecode-bciciv2a) (`code/check_equivalence.py`, `results/equivalence.txt`); `tools/compare_methods.py` |
| Robustness checks, frozen backbone | [finetune-public-datasets](methods/cbramod/experiments/finetune-public-datasets) |
| EEGNet vs. EEG Conformer within-subject, Wilcoxon tests | [methods/eegnet/experiments/bciciv2a-epoch-selection](methods/eegnet/experiments/bciciv2a-epoch-selection) (`code/analyze.py`) |
| EEG Conformer scrambled labels, significance re-tests, code mismatches | [2026-09-24 report](docs/reports/eeg-conformer/2026-09-24_eeg-conformer_weekly-report.docx) |
| CBraMod parameters and FLOPs | [2026-10-01 report](docs/reports/cbramod/2026-10-01_cbramod_weekly-report.docx) |
| ATCNet: paper-era code, log and weights; protocols, best-of-10 analysis | [bciciv2a-model-selection](methods/atcnet/experiments/bciciv2a-model-selection) (`code/analyze.py`, `results/authors_2022_log.txt`); upstream commit `3a318b2`; [2026-10-22 report](docs/reports/atcnet/2026-10-22_atcnet_weekly-report.docx) |
| Sleep-EDF (Sections 1, 2, 5) | [cbramod/finetune-sleepedf](methods/cbramod/experiments/finetune-sleepedf) (shared pipeline, data, common test set) and each model's Sleep-EDF experiment; [deepsleepnet](methods/deepsleepnet), [l-seqsleepnet](methods/l-seqsleepnet) experiments; tests from `tools/compare_sleepedf.py`; citation counts from OpenAlex and Semantic Scholar (2026-11-01) |
| LaBraM code issues, TUAB/TUEV benchmark re-runs | [2026-10-08 report](docs/reports/labram/2026-10-08_labram_weekly-report.docx), [literature note](docs/literature/review-papers-reproduction-vs-citation.md) |
