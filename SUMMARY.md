# Summary of results

Everything validated so far, as of 2026-10-19: 3 methods, 1 baseline added by us (EEGNet, run inside two of those methods' pipelines), 4 public datasets, and the
weekly reports of 2026-09-24, 10-01 and 10-08. "Paper" is the number printed in the paper (cited); "ours" is a
re-run in this repository. Every metric and setting is in [RESULTS.md](RESULTS.md), which is generated from the
experiments and is authoritative if the two disagree.

## 1. Findings at a glance

| Method | What we ran | Verdict | Key numbers |
|---|---|---|---|
| [EEG Conformer](methods/eeg-conformer) | Released code, BCI-IV-2a, within-subject, 9 subjects × 3 seeds × 2 protocols | The pipeline reproduces, but the reported number is the maximum over 2000 test-set evaluations | 78.72% under the released rule vs. 78.66% in the paper; 70.01% when the epoch is chosen on a validation split |
| [CBraMod](methods/cbramod) | Released code and weights, 4 public datasets, cross-subject, seeds 0–4 | Does not reproduce: all 12 metrics are below the paper, 9 of them significantly | balanced accuracy −0.015 (PhysioNet-MI) to −0.137 (MentalArithmetic) |
| [LaBraM-Base](methods/labram) | Released code and weights on CBraMod's 4 datasets, splits, seeds and scoring | Reproduces the CBraMod paper's LaBraM-Base numbers on 10 of 12 metrics | ties or beats CBraMod on 3 of the 4 datasets |
| [EEGNet](methods/eegnet) (baseline), cross-subject | EEGNet-8,2 through CBraMod's own fine-tuning pipeline, same 4 datasets | With 2–4K parameters, the best model on BCI-IV-2a | 0.5288 balanced accuracy vs. 0.4571 (CBraMod, p = 0.021) and 0.4477 (LaBraM-Base, p = 0.014) |
| [EEGNet](methods/eegnet) (baseline), within-subject | braindecode EEGNet-8,2 through EEG Conformer's released training loop, BCI-IV-2a, 9 subjects × 3 seeds × 2 protocols | Conformer's lead over EEGNet (Table II) is significant only when each model's best test-set epoch is reported | Conformer − EEGNet: +6.10 pp under the released rule (p = 0.0039); −0.76 pp at the last-100-epoch mean; +2.65 pp with a validation split (p = 0.16) |

## 2. Methods by characteristics

| | EEGNet | EEG Conformer | LaBraM | CBraMod |
|---|---|---|---|---|
| Paper | Lawhern et al., J. Neural Eng. 2018 | Song et al., IEEE TNSRE 2023 | Jiang et al., ICLR 2024 | Wang et al., ICLR 2025 |
| Family | task-specific, supervised | task-specific, supervised | EEG foundation model | EEG foundation model |
| Architecture | compact CNN: temporal, depthwise spatial and separable convolutions | CNN (temporal + spatial convolution, pooling) → 6 self-attention blocks → MLP head | Transformer over channel patches, with a vector-quantized neural tokenizer | criss-cross Transformer: spatial and temporal attention in parallel over channel patches |
| Pre-training | none | none | ~2,500 h from ~20 datasets; predicts the neural codes of masked patches | TUEG (TUH EEG Corpus); reconstructs masked patches |
| Parameters | 2,193–4,020 (CBraMod pipeline, by dataset); 3,444 (braindecode, BCI-IV-2a within-subject) | 789,816 (BCI-IV-2a) | 5.8M (Base) | backbone 4.88M (paper: 4.0M); 19–46M as fine-tuned, with the default head |
| Released weights | – | – (trained per subject) | Base only; Large and Huge not released | yes |
| Paper's own benchmarks | P300, ERN, MRCP, SMR (BCI-IV-2a) | BCI-IV-2a, BCI-IV-2b, SEED | TUAB, TUEV (+ SEED-V, MoBI in the appendix) | 13 downstream datasets |
| Evaluation protocol | within-subject (4-fold block-wise cross-validation) and cross-subject | within-subject, session T → E | cross-subject | cross-subject: subject-independent train / val / test |
| Epoch selection in the released code | lowest validation loss (paper) | maximum test accuracy over 2000 epochs; no validation set | best validation accuracy (the paper states κ / AUROC) | best validation κ (multi-class) or AUROC (binary) |
| Seeds in the paper | – | not reported; a random seed every run | 5, not named | 5, not named |
| Role here | baseline we added | method validated | method validated; CBraMod's strongest baseline | method validated |
| Our experiment | braindecode `EEGNet` (pinned in [methods/eegnet](methods/eegnet)) inside EEG Conformer's loop: [bciciv2a-epoch-selection](methods/eegnet/experiments/bciciv2a-epoch-selection); our own PyTorch re-implementation ([`scripts/run_eegnet.py`](methods/cbramod/experiments/finetune-public-datasets/scripts/run_eegnet.py)) inside CBraMod's [finetune-public-datasets](methods/cbramod/experiments/finetune-public-datasets) | [bciciv2a-epoch-selection](methods/eeg-conformer/experiments/bciciv2a-epoch-selection) | [finetune-cbramod-splits](methods/labram/experiments/finetune-cbramod-splits) | [finetune-public-datasets](methods/cbramod/experiments/finetune-public-datasets) |

CBraMod's parameter and FLOP counts are from the 2026-10-01 report, EEGNet's from our run logs, and the others
from the method READMEs.

### Grouped by characteristic

| Characteristic | Group | Methods | What it means for comparisons |
|---|---|---|---|
| Model family | task-specific, trained from scratch | EEGNet, EEG Conformer | no pre-training data or released weights involved |
| | pre-trained foundation model, fine-tuned | LaBraM, CBraMod | results depend on released weights; LaBraM-Large/-Huge cannot be checked |
| Evaluation protocol | within-subject (train on session T, test on session E of the same subject) | EEG Conformer, EEGNet (our runs in Conformer's loop) | its BCI-IV-2a numbers (0.70–0.79 accuracy) are **not comparable** with the cross-subject rows (0.45–0.53 balanced accuracy) |
| | cross-subject (subject-independent splits) | CBraMod, LaBraM, EEGNet (our runs in CBraMod's pipeline) | directly comparable: identical samples, splits, seeds, checkpoint rule and metric code (Section 4) |
| Epoch selection in released code | on the test set | EEG Conformer | inflates the reported number, and Conformer's more than EEGNet's (Section 5) |
| | on a validation set | CBraMod, LaBraM | no test-set leakage |
| Reproduction outcome | reproduces | LaBraM-Base (10 / 12 metrics); EEG Conformer under its own rule | |
| | does not reproduce | CBraMod (3 / 12 metrics within noise) | |
| | mixed | EEGNet (6 / 12 within noise: above the paper on all BCI-IV-2a metrics and PhysioNet-MI weighted F1, below on MentalArithmetic and Mumtaz2016 balanced accuracy) | the paper's EEGNet came from BIOT's code with unreported settings |

## 3. Datasets

| Category | Dataset | Task (classes) | Subjects: train / val / test | Channels | Input as run | Samples | Metrics (checkpoint metric in bold) | Run with | Paper tables |
|---|---|---|---|---|---|---|---|---|---|
| Motor imagery | BCI-IV-2a | left hand, right hand, feet, tongue (4) | cross-subject 5 / 2 / 2; within-subject: each of 9 subjects, session T → E | 22 | 4 s at 200 Hz (CBraMod pipeline); 4 s at 250 Hz (EEG Conformer's pipeline, also for EEGNet) | 5,184 (CBraMod pipeline); 288 + 288 trials per subject (EEG Conformer) | balanced acc., **κ**, weighted F1; EEG Conformer: accuracy | EEG Conformer, CBraMod, LaBraM-Base, EEGNet | CBraMod T15; EEG Conformer T II |
| Motor imagery | PhysioNet-MI | left fist, right fist, both fists, both feet (4) | 70 / 19 / 20 | 64 | 4 s at 200 Hz | 9,837 | balanced acc., **κ**, weighted F1 | CBraMod, LaBraM-Base, EEGNet | CBraMod T3, T4, T18 |
| Clinical | Mumtaz2016 | major depressive disorder vs. healthy (2) | 43 / 9 / 11 | 19 | 5 s at 200 Hz | 7,143 | balanced acc., AUC-PR, **AUROC** | CBraMod, LaBraM-Base, EEGNet | CBraMod T10 |
| Cognitive state | MentalArithmetic | rest vs. mental arithmetic ("mental stress") (2) | 28 / 4 / 4 | 20 (LaBraM: 19, no A2-A1) | 5 s at 200 Hz | 1,707 | balanced acc., AUC-PR, **AUROC** | CBraMod, LaBraM-Base, EEGNet | CBraMod T12 |

Not run yet (details in [datasets/README.md](datasets/README.md)):

| Status | Datasets |
|---|---|
| Public, not run yet | ISRUC, CHB-MIT (CBraMod; too large for one round), BCI-IV-2b (EEG Conformer; no download key yet) |
| Access-gated | TUAB, TUEV (TUH registration; LaBraM's own benchmarks), SEED, SEED-V, SEED-VIG (license application), FACED (Synapse account), SHU-MI (password-protected), BCIC2020-3 (test labels not public) |

## 4. Cross-subject results: all models on identical splits

All four models ran on CBraMod's preprocessed samples and subject splits, with seeds 0–4. The checkpoint was the
best validation κ / AUROC epoch, the test set was scored once, and the metric code was the same for every model.
Each cell is **ours, mean ± std over 5 seeds (paper)**. Every paper number is from the CBraMod paper; its EEGNet
and LaBraM-Base rows are the CBraMod authors' runs.

### Balanced accuracy

| Category | Dataset | EEGNet | CBraMod, no pre-training | CBraMod | LaBraM-Base | Best, ours | Best, paper |
|---|---|---|---|---|---|---|---|
| Motor imagery | PhysioNet-MI | 0.5948 ± 0.0028 (0.5814) | 0.5738 ± 0.0082 (0.6196) | **0.6265 ± 0.0054** (0.6417) | 0.6098 ± 0.0098 (0.6173) | CBraMod | CBraMod |
| Motor imagery | BCI-IV-2a | **0.5288 ± 0.0394** (0.4482) | 0.4561 ± 0.0195 (–) | 0.4571 ± 0.0399 (0.5138) | 0.4477 ± 0.0421 (0.4869) | EEGNet | CBraMod |
| Clinical | Mumtaz2016 | 0.8828 ± 0.0295 (0.9232) | 0.8973 ± 0.0115 (–) | 0.8933 ± 0.0101 (0.9560) | 0.8905 ± 0.0041 (0.9409) | tie (no tested pair differs, p ≥ 0.48) | CBraMod |
| Cognitive state | MentalArithmetic | 0.5569 ± 0.0247 (0.6770) | 0.5924 ± 0.0403 (–) | 0.5889 ± 0.0209 (0.7256) | **0.6562 ± 0.0397** (0.6909) | LaBraM-Base | CBraMod |

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

## 5. Within-subject results on BCI-IV-2a: EEG Conformer and EEGNet

The same 27 runs per model (9 subjects × 3 seeds), with the reported epoch chosen by different rules. The
validation-selected row is a separate set of 27 runs per model with 20% of session T held out. EEGNet
(braindecode EEGNet-8,2, 3,444 parameters) ran through EEG Conformer's released training loop with only the model
swapped: same data, augmentation, optimizer, 2000 epochs, seeds and validation trials. p is a two-sided Wilcoxon
signed-rank test over the 9 subjects, the paper's test; for the paper row it is recomputed from Table II.

| How the reported epoch is chosen | EEG Conformer, acc. % (κ) | EEGNet, acc. % (κ) | Conformer − EEGNet (p) |
|---|---|---|---|
| Paper, Table II | 78.66 (0.7155) | 74.50 (0.6600) | +4.17 pp (0.0497) |
| Released rule: maximum test accuracy over 2000 epochs | 78.72 (0.7162) | 72.62 (0.6349) | +6.10 pp (0.0039) |
| Validation-selected epoch | 70.01 (0.6001) | 67.36 (0.5648) | +2.65 pp (0.16) |
| Final epoch | 69.21 (0.5895) | 69.35 (0.5914) | −0.14 pp (1.00) |
| Mean of the last 100 epochs | 68.77 (0.5836) | 69.53 (0.5937) | −0.76 pp (0.65) |
| Control: scrambled training labels, released rule (subjects 1–6; chance 25.00) | 35.15 (–) | – | (report only) |

- **The released rule reproduces Conformer's number** (+0.06 pp). Every rule that does not look at the test set
  gives 8.7–9.9 pp less.
- **Conformer's lead over EEGNet is significant only under that rule.** At the last-100-epoch mean or the
  final epoch of the same runs, EEGNet is level or slightly ahead. With a validation split Conformer leads by
  2.65 pp on 6 of 9 subjects, which is not significant with 9 subjects. On those same runs EEGNet's
  last-100-epoch mean is 1.07 pp higher, so the lead comes from the validation selection itself.
- **The maximum inflates Conformer more because its test accuracy fluctuates more.** Over the last 100 epochs
  of a run its SD is 5.52 pp, against 1.10 pp for EEGNet. The maximum therefore sits 9.95 pp above Conformer's
  last-100 mean and 3.09 pp above EEGNet's.
- Caveat: the loop is Conformer's (Adam lr 2e-4), not EEGNet's own recipe, and it was not tuned for EEGNet.

The scrambled-label control is from the 2026-09-24 report. Its code was later removed, so it cannot be
recomputed from this repository. The same "best test epoch" rule applied to our LaBraM-Base runs would add +0.006
(PhysioNet-MI) to +0.065 (MentalArithmetic) balanced accuracy
([selection analysis](methods/labram/experiments/finetune-cbramod-splits#how-much-the-reported-number-depends-on-the-reporting-rule)).

## 6. Baselines: what each paper compared against, and what we re-ran

| Paper | Baselines it reports | Where its baseline numbers come from | Re-run here | Outcome |
|---|---|---|---|---|
| EEG Conformer | FBCSP, ConvNet, EEGNet, C2CM, FBCNet, DRDA (BCI-IV-2a, Table II) | not stated, and no baseline code is released; the FBCSP and DRDA rows look copied from their own papers | EEGNet, through Conformer's released training loop with the same seeds, protocols and epoch-selection rules (Section 5); ConvNet not yet | Conformer's lead over EEGNet is significant only when each model's best test-set epoch is reported (+6.10 pp, p = 0.0039); with a validation split it is +2.65 pp, p = 0.16. EEGNet's own paper chose the epoch on validation data |
| LaBraM | BIOT and the baselines of BIOT's paper (SPaRCNet, ContraWR, CNN-Transformer, …) on TUAB and TUEV | copied from BIOT Tables 4–5, taking the best BIOT variant ("we choose the best results") | no (needs TUH access) | two benchmarks re-ran LaBraM-Base: TUEV 0.5958 and 0.5905 vs. 0.6409 in the paper |
| CBraMod | supervised baselines re-implemented with BIOT's code (incl. EEGNet and EEG Conformer), BIOT, LaBraM-Base | its own runs on most datasets; the TUAB/TUEV rows are copied from LaBraM, unmarked | EEGNet and LaBraM-Base, under CBraMod's exact pipeline (Section 4) | LaBraM-Base matches the paper on 10 of 12 metrics; EEGNet is best on BCI-IV-2a |
| CBraMod, ablations | no pre-training (T4), frozen backbone (T18) | its own runs | both | pre-training +0.053 balanced accuracy on PhysioNet-MI (paper +0.022); frozen backbone 0.5370 (paper 0.3845) |

## 7. Issues found, by type

| Type | EEG Conformer | CBraMod | LaBraM |
|---|---|---|---|
| Test-set model selection | yes: maximum over 2000 test evaluations (≈ +9 pp); it also creates the significant lead over EEGNet (Section 5) | no | no |
| Seeds | not reported; a random seed every run | 5, not named; code default 3407 | 5, not named; README uses seed 0 |
| Paper vs. released code | 3 FC layers, not 2; other scripts use a different depth and number of heads; S&R augmentation never called for SEED; `BCIIV2a.m` hard-codes subject 6 | classifier head trains at 5× the stated lr (`multi_lr`); BCI-IV-2a has 5,184 samples, not 5,088; no 0.5–45 Hz filter on MentalArithmetic; Mumtaz2016 split differs | checkpoint chosen on validation accuracy, not κ / AUROC; single-GPU runs never reshuffle; 50 Hz notch on US (60 Hz) recordings |
| Baseline numbers | source not stated | TUAB/TUEV rows copied from LaBraM, unmarked | copied from BIOT, best variant |
| Model size and cost | exact (789,816 parameters, +17.62% for attention) | under-counted: 4.88M vs. 4.0M parameters, 1,596M vs. 318.9M FLOPs | Large and Huge cannot be checked (weights not released) |
| Statistics | 3 of 9 significance claims fail on the paper's own per-subject numbers (C2CM, FBCNet, DRDA) | – | – |

## 8. Where this sits in the literature

Of 21 review, survey, benchmark and method papers, none of the six reviews without their own experiments re-runs
a method; they tabulate self-reported numbers. All ten benchmarks re-run methods, but none puts a re-run next to
the number the method's own paper reports. A per-method paper vs. re-run table, as in Sections 4–5 and
[RESULTS.md](RESULTS.md), is therefore new. See
[docs/literature/review-papers-reproduction-vs-citation.md](docs/literature/review-papers-reproduction-vs-citation.md).

## 9. Open items

- Apply for TUH access, then re-run LaBraM-Base and CBraMod on LaBraM's own TUAB/TUEV splits.
- Re-run EEG Conformer on BCI-IV-2b and SEED with a validation split. Run ConvNet (its other CNN baseline)
  through the same loop as EEGNet. Optionally, run EEGNet with its own recipe (Adam lr 1e-3, 128 Hz) to see how
  much the shared loop costs it.
- Run the remaining public CBraMod datasets (ISRUC, CHB-MIT).

## Sources

| Section | Source |
|---|---|
| Numbers in Sections 1, 4, 5, 6 | [RESULTS.md](RESULTS.md) and the experiment READMEs; pairwise tests from `tools/compare_methods.py` |
| Robustness checks, frozen backbone | [finetune-public-datasets](methods/cbramod/experiments/finetune-public-datasets) |
| EEGNet vs. EEG Conformer within-subject, Wilcoxon tests | [methods/eegnet/experiments/bciciv2a-epoch-selection](methods/eegnet/experiments/bciciv2a-epoch-selection) (`code/analyze.py`) |
| EEG Conformer scrambled labels, significance re-tests, code mismatches | [2026-09-24 report](docs/reports/eeg-conformer/2026-09-24_eeg-conformer_weekly-report.docx) |
| CBraMod parameters and FLOPs | [2026-10-01 report](docs/reports/cbramod/2026-10-01_cbramod_weekly-report.docx) |
| LaBraM code issues, TUAB/TUEV benchmark re-runs | [2026-10-08 report](docs/reports/labram/2026-10-08_labram_weekly-report.docx), [literature note](docs/literature/review-papers-reproduction-vs-citation.md) |
