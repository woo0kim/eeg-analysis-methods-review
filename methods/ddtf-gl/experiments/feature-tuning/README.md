# dDTF-GL+: tuning dDTF-GL's features on validation data

How far does dDTF-GL get when its features, feature counts, projection features and classifier are tuned per dataset? dDTF-GL+ keeps
the paper's chain (standardise → DFT selection → AXB → projection features → XGBoost) and opens four choices, searched on the
validation part of each subject-independent protocol only; the selected configuration then runs on seeds 0–4 and is scored on the test
subjects exactly as dDTF-GL and every network in that pipeline (the final runs sit in each pipeline experiment's `logs/runs` as
variant `ddtf-gl-plus`, so its collector scores them). Code: [code/gl_plus.py](../../code/gl_plus.py) (the chain),
[run_gl_plus.py](../../code/run_gl_plus.py) (one run), [tune.py](../../code/tune.py) (the search), [spectral.py](../../code/spectral.py)
(band-power features), [tune_report.py](../../code/tune_report.py) (this README's tables). With its default configuration `gl_plus`
gives `gl.fit_predict`'s predictions exactly (checked on STEW: identical predictions and probabilities).

## What can change

| Choice | Options | The paper's chain |
|---|---|---|
| Feature blocks | `ddtf`: the band-mean dDTF maps (5 · C²); `graph`: per band and channel the maps' out-strength, in-strength, net flow and self-flow (4 · 5 · C); `spec`: log band power and relative band power of each channel over the window (10 · C); `spect`: log band power of each channel in each 1-s segment (5 · C · seconds). Bands: DGCNN's (1–3, 4–7, 8–13, 14–30, 31–50 Hz) | `ddtf` |
| Feature counts | DFT keeps 0.1, 0.25, 0.5 or all of the features; AXB on or off | DFT 0.5, AXB on |
| Projection features ("LNT features") | any of `lnt` (the bundle's groupwise LNT), `lda` (shrinkage LDA's decision values), `svml` (a linear SVM's), `svmr` (an RBF SVM's; Nystroem approximation above 8,000 training samples). LDA and the SVMs are stacked: their training values are out-of-fold predictions of a 5-fold GroupKFold over the training subjects, so no model scores samples of subjects it was fitted on | `lnt` |
| Classifier | XGBoost (depth, trees, learning rate) = (3, 500, 0.1), (6, 500, 0.1), (3, 1000, 0.05), (6, 1000, 0.05) | (3, 500, 0.1) |

## Search

Greedy, in four stages (features → projections → counts → classifier), each starting from the best configuration so far; a candidate
replaces it only with a higher validation score. The score is the checkpoint metric every model of that pipeline is selected on: κ
(multi-class datasets and STEW) or AUROC (Mumtaz2016, MentalArithmetic), on the validation predictions pooled over folds (STEW) and
averaged over the search seeds. No test sample enters a choice. Targets and seeds:

| Target | Protocol | Validation part | Search seeds |
|---|---|---|---|
| STEW-rest, STEW-rating3 | shared 7-fold GroupKFold, 4-s windows | the next fold | 0, all 7 folds |
| BCIC-IV-2a, Mumtaz2016, MentalArithmetic | CBraMod's subject splits | CBraMod's validation subjects | 0, 1 |
| PhysioNet-MI | CBraMod's subject split | its validation subjects | 0 |
| SleepEDF | the shared 62 / 8 / 8 split | the 8 validation subjects | 0, 1 |
| SEED, SEED-V | the subject split | 3 validation subjects | 0 (search runs train on every 4th training window: consecutive 1-s windows of a 4-minute clip are near-duplicates; the final runs use all) |
| STEW-6s-SI | the dDTF-GL paper's subject-independent setting (6-s windows, no validation part) | – | not searched: STEW-rest's configuration runs as it is |

SEED-V ran stages 1–2 only: every candidate was at chance (validation κ ≤ 0.085, as every model on SEED-V across subjects), so the
remaining two stages (about 13 hours of CPU) could not change anything meaningful; its final runs use the stage-2 selection.

## Results

dDTF-GL+ (the selected configuration, seeds 0–4) against dDTF-GL (the paper's chain), both scored by the pipeline's collector; mean ±
SD over seeds, Welch's t-test (`results/results.txt`). Best other model: the best of the repository's models in that pipeline
(SUMMARY.md, Sections 2 and 5).

| Target | Selected configuration | dDTF-GL | dDTF-GL+ | Δ (p) | Best other model |
|---|---|---|---|---|---|
| STEW rest vs. task (accuracy) | all four blocks; DFT and AXB off; LNT + LDA | 0.7398 ± 0.0043 | **0.8191 ± 0.0024** | +0.079 (< 0.001) | LaBraM-Base 0.8179 (level, p = 0.82) |
| STEW, the paper's setting (accuracy) | STEW rest vs. task's | 0.7744 (Safari et al.'s maps) | **0.8632 ± 0.0020** | +0.089 (< 0.001) | LaBraM-Base 0.8626 |
| STEW 3 levels (accuracy) | all four blocks; DFT and AXB off; LNT | 0.5185 ± 0.0052 | 0.5636 ± 0.0019 | +0.045 (< 0.001) | LaBraM-Base 0.5670 |
| MentalArithmetic (balanced acc.) | band power (`spec`, `spect`); no DFT, no AXB; LDA + linear + RBF SVM; depth 6 | 0.5833 ± 0.0282 | **0.7097 ± 0.0259** | +0.126 (< 0.001) | ATCNet 0.6951 |
| Mumtaz2016 (balanced acc.) | `ddtf` + `graph`; the paper's counts and LNT | 0.9192 ± 0.0025 | 0.9213 ± 0.0035 | +0.002 (0.30) | DGCNN 0.9619 |
| Sleep-EDF-78 (κ, common test set) | all four blocks; DFT and AXB off; LDA + linear + RBF SVM; depth 6 | 0.5595 ± 0.0006 | 0.7207 ± 0.0009 | +0.161 (< 0.001) | LaBraM-Base 0.7727 |
| SEED, subject split (balanced acc.) | `spec` only; the paper's counts and LNT | 0.5237 ± 0.0015 | 0.5473 ± 0.0052 | +0.024 (< 0.001) | LaBraM-Base 0.5822 |
| PhysioNet-MI (balanced acc.) | band power (`spec`, `spect`); AXB; LNT; 1,000 trees at 0.05 | 0.3274 ± 0.0055 | 0.4904 ± 0.0072 | +0.163 (< 0.001) | CBraMod 0.6265 |
| BCI-IV-2a (balanced acc.) | `ddtf` + `graph`; DFT and AXB off; LNT; depth 6 | 0.3276 ± 0.0084 | 0.3347 ± 0.0054 | +0.007 | EEGNet 0.5288 |
| SEED-V, subject split (balanced acc.) | `ddtf` + `spec`; the paper's counts and LNT | 0.2315 ± 0.0019 | 0.2578 ± 0.0036 | +0.026 (< 0.001) | every model at chance (0.20); DGCNN 0.2563 |

- **Band power does most of the work.** Every target's first stage moved most: adding `spec` / `spect` to the dDTF maps (STEW, Sleep-EDF)
  or replacing the maps with them (MentalArithmetic, PhysioNet-MI, SEED). The maps survive where connectivity carries the class
  (Mumtaz2016, the depression dataset where the original dDTF-GL was already second) and are dropped where it does not.
- **Fewer features never helped.** On every target the selection kept all features (DFT off) or the paper's half; AXB was switched off
  on STEW, MentalArithmetic, Sleep-EDF and BCI-IV-2a.
- **Projection features matter little.** Adding LDA or SVM scores to LNT changed validation κ by at most +0.02 (STEW rest vs. task
  +0.018); only MentalArithmetic and Sleep-EDF replaced LNT with stacked LDA / SVM scores.
- **Where the search cannot see, it does not help.** On Mumtaz2016 validation AUROC is 1.000 for almost every candidate (9 validation
  subjects separate perfectly), so the selection is arbitrary: dDTF-GL+ is level with dDTF-GL in balanced accuracy and 0.005 lower in
  AUROC (p = 0.028). On SEED-V every candidate is at chance.
- **Motor imagery stays out of reach.** BCI-IV-2a is unchanged (0.335, chance 0.25) and PhysioNet-MI rises to 0.49 but stays far below
  the networks (0.58–0.63).
- **Cheaper as well as better where the maps are dropped.** Without the dDTF maps a window costs 0.2–3 MFLOPs instead of 260–760
  (PhysioNet-MI, MentalArithmetic, SEED; `tools/count_flops.py`); training (search excluded) takes seconds to minutes on a CPU.

Every candidate's validation score is recomputed from the committed search runs (`logs/<target>/`) by `tune_report.py` (largest
difference to `results/selection/*.json`: 0).

## Paper vs. this experiment

| | The dDTF-GL paper | dDTF-GL+ | Why |
|---|---|---|---|
| Features | dDTF maps | dDTF maps, their graph summaries, band power (window, 1-s segments): chosen per dataset | the user's question: which features, how many, which projections |
| Selection | DFT 50%, AXB | chosen per dataset (DFT fraction, AXB on/off) | as above |
| Projections | LNT | LNT, LDA, linear SVM, RBF SVM (stacked out-of-fold by subject) | "LNT features → [lnt, lda, svm_linear, svm_rbf]" |
| Classifier | XGBoost 500 / 3 / 0.1 (the bundle's) | one of four settings | as above |
| Model selection | none stated | validation data only; test sets never consulted | the repository's rule |

### Settings we had to choose

| Setting | Value | Why |
|---|---|---|
| Search order and greediness | features → projections → counts → classifier, one stage at a time | a full grid (192 configurations × seeds × folds) does not fit the CPUs; features first because they matter most |
| Search seeds | 1–2 (STEW: 1 seed × 7 folds) | XGBoost's subsampling is the only randomness; the final runs use 5 seeds |
| SVM settings | LinearSVC C = 0.01; SVC C = 1, gamma 'scale'; Nystroem 1,000 components, LinearSVC C = 0.1 | standard values for standardised high-dimensional input, not tuned |
| Out-of-fold groups | 5-fold GroupKFold over training subjects (keys or metadata) | stacked scores must come from subjects the model did not see, as the test subjects |
| SEED / SEED-V search subsample | every 4th training window | runtime (a full SEED run with stacked SVMs takes over an hour); the final runs use all windows |

### Run configuration

CPU only: mcl-server (2× Xeon E5-2620 v3, shared), 2–5 threads per run with BLAS capped to the same; Python env `ddtf-gl` (`../../env`;
scikit-learn 1.9.1, XGBoost 3.4.2, numpy 2.4.6). Features cached under `$GL_WORK/features/<dataset>/{ddtf,spec,spect}` (extract.py).

## Reproducing

```bash
code/reproduce.sh collect    # results/ from results/selection and the pipeline experiments' results; validation scores recomputed from logs/
code/reproduce.sh train      # features, every search and the final runs (CPU; see the script)
```

## Layout

`results/selection/<target>.json` (every stage's candidates and validation scores, the selection), `results/results.txt` and
`results.json` (the table above, per seed); `logs/<target>/<config hash>__fold<k>__s<seed>/` (`metrics.json`, `test_preds.npz` with the
validation predictions the selection used, `log.txt`); the final runs: `../<pipeline experiment>/logs/runs/*ddtf-gl-plus*`.

## Provenance

Runs: 2026-11-20 to 11-21 (UTC) on mcl-server (`~/repro-gl/runs-plus`); SEED-V's final runs took 2.4 h each (stacked LNT over 9,920
features of 73,590 windows, 2,500 trees in total).
