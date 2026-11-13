# dDTF-GL

> Q. Cao, H. Li, C.-C. J. Kuo, *A New Method for Mental Workload Classification with EEG Connectivity Maps*, University of Southern
> California, manuscript (the PDF we received is named *Efficient and green mental workload classification using EEG connectivity
> maps*; not yet published, SHA-256 a58c7fc7…).

A "green learning" pipeline without a neural network: each EEG window becomes directed-connectivity maps (dDTF, the direct
Directed Transfer Function) in five bands, flattened into one vector; a two-stage selection keeps the most discriminant entries
(the Discriminant Feature Test, DFT, then the gain importance of an auxiliary XGBoost, AXB); the Least-squares Normal Transform
(LNT) adds a few supervised linear projections; an XGBoost classifies. The paper evaluates only STEW (rest vs. SIMKAP multitasking):
0.898 accuracy subject-dependent and 0.766 subject-independent, ahead of six deep networks it trains on the same dDTF maps.

- **Authors' code:** `mwl_gl_si7_txt_bundle.zip`, shared with us on 2026-11-10 (SHA-256 cd995bec…); every file carries an extra
  `.txt`. It has no license and is not public, so it is not committed: `./setup_upstream.sh <zip>` unpacks it into the git-ignored
  `upstream/`. **The bundle is not the paper's method.** Its script `mwl_gl_si7.py` computes DWT statistics, not dDTF, and its
  README says "No auxiliary branch, band-aware preprocessing, or mRMR in this version" and that the parameters "have NOT been
  verified against the historical 76%+ run". It does hold the DFT, LNT and XGBoost stages (`lib/`, `lnt_generate.py`,
  `run_groupwise_lnt`, `build_xgb`) with their settings, which the paper does not give.
- **The maps' source:** the paper computes dDTF "following [4]", Safari et al. 2024, whose repository
  ([safari-mohammadreza/MWL_MachineLearning](https://github.com/safari-mohammadreza/MWL_MachineLearning), commit `604d243`) releases the
  SIFT/MATLAB **maps** of STEW's 6-s windows but not the code that made them (its README: "by SIFT toolbox in Matlab"). On STEW in the
  paper's protocol dDTF-GL runs on those released maps (`safari-gl`; [code/safari_maps.py](code/safari_maps.py) reads their pickles with an
  unpickler restricted to numpy arrays). For every other dataset the maps come from our generator ([code/ddtf.py](code/ddtf.py), numpy),
  whose settings were calibrated on Safari et al.'s maps ([code/calibrate_safari.py](code/calibrate_safari.py), below).
- **Our implementation** ([code/](code)): the map generator, the AXB stage, and the chain around the bundle's own functions, which are
  imported unchanged ([code/gl.py](code/gl.py)). `code/check_equivalence.py` checks the wrappers against the bundle (identical DFT losses and
  selection; LNT rows equal to 6e-8; deterministic runs).
- **braindecode:** not applicable (no network).

## Experiments

| Experiment | Question | Finding |
|---|---|---|
| [stew-paper-protocol](experiments/stew-paper-protocol) | Do the paper's STEW numbers reproduce, in its own windows and settings? How do the repository's networks do there? What does the authors' bundle give? | On Safari et al.'s released maps the subject-independent result reproduces exactly (0.774, SD over subjects 0.147, vs. 0.766 ± 0.148); the subject-dependent one does not (0.872 vs. 0.898). In the same settings LaBraM-Base, CBraMod, DGCNN and EEG Conformer score higher (SI 0.863–0.784); EEGNet on raw EEG matches the paper's (0.665 vs. 0.683). The bundle (DWT, no AXB) scores 0.845, 0.048 of it from content below 0.3 Hz; the 14 channel means alone reach 0.703. |
| [stew-cbramod-pipeline](experiments/stew-cbramod-pipeline) | dDTF-GL against every other model, on STEW's shared windows and folds; what do the maps' settings, dDTF and AXB add? | 0.740 on rest vs. task, level with EEG Conformer, below DGCNN (0.775) and the foundation models; 0.519 on 3 levels. The calibrated maps score 0.023 below our first settings; AXB adds nothing; the bundle's DWT statistics beat the dDTF maps by 0.037. |
| [cbramod-splits](experiments/cbramod-splits) | CBraMod's four public datasets, on its samples and splits | Near chance on motor imagery (0.33 on both; chance 0.25), the weakest model there; 0.919 on Mumtaz2016, above both foundation models and below DGCNN; 0.583 on MentalArithmetic, level with CBraMod and EEGNet. Our first map settings did better on the last two (0.943, 0.633). |
| [sleepedf-cbramod-pipeline](experiments/sleepedf-cbramod-pipeline) | Sleep-EDF-78, the shared split and common test set | The weakest model: κ 0.560 (EEGNet 0.682, LaBraM-Base 0.773). Two channels give 20 dDTF values, of which the classifier sees 7; each epoch is classified alone, without the sequence head the other models use. |
| [seed-cbramod-pipeline](experiments/seed-cbramod-pipeline) | SEED and SEED-V, trial and subject splits | With the supervised models, behind the foundation models: SEED trial split 0.561 (DGCNN 0.574, p = 0.17; LaBraM-Base 0.630), subject split 0.524 (level with DGCNN and EEGNet, below EEG Conformer); SEED-V trial 0.325 (level with EEG Conformer), at chance across subjects like every model. 19,220 features per 1-s sample. |

## The method as run here

| Stage | Paper | Here | Source |
|---|---|---|---|
| Windows | 6 s, 4-s step (STEW) | the paper's on STEW ([stew-paper-protocol](experiments/stew-paper-protocol)); elsewhere each shared pipeline's samples (4-s STEW windows, CBraMod's 4–5-s trials, SEED's 1-s samples, Sleep-EDF's 30-s epochs) | – |
| Connectivity | dDTF per window in δ, θ, α, β, γ, "following the standard extraction strategy used in [4]" (Safari et al., Sci. Rep. 14:9153, 2024: SIFT, MVAR order by AIC, bands 2–4, 4–8, 8–13, 13–32, 32–50 Hz, band average) | STEW, the paper's protocol: Safari et al.'s released maps. Elsewhere our generator with the calibrated settings: monopolar montages re-referenced to their average (Safari et al.'s preprocessing; not Sleep-EDF's bipolar pair; BCI-IV-2a and PhysioNet-MI come average-referenced), each channel z-scored over the window, MVAR by least squares, order by AIC (below), value √(F²ᵢⱼ(f) · \|Cᵢⱼ(f)\|) (full-frequency DTF times the partial-coherence magnitude, Korzeniewska et al. 2003) on a 1-Hz grid from 1 Hz to Nyquist, mean over each band's bins; all C × C entries | Safari et al.'s maps; our `code/ddtf.py` |
| Fusion | band maps concatenated | 5 · C² features (STEW 980, SEED 19,220) | ours |
| Standardisation | – | StandardScaler fitted on the training samples | bundle |
| DFT | rank, keep "a subset of high-ranking features" | `feature_selection(FStype='DFT_entropy', B=32)`, keep 50% | bundle (`fs_thrs` 0.5, `fs_B` 32) |
| AXB | XGBoost importance "used to select features further"; no rule given | XGBoost with the classifier's settings on the DFT set; keep the features with positive gain, at most half of the DFT set, in DFT order | ours |
| LNT | LNT features concatenated with the selected ones | `run_groupwise_lnt`: groups of 8, 2 components requested (1 for 2 classes), concatenated | bundle |
| Classifier | XGBoost | `build_xgb`: 500 trees, depth 3, learning rate 0.1, subsample 0.8, colsample 0.8, `random_state` = seed | bundle |

**MVAR order.** Fitted by least squares on one short window, ln det Σ falls without bound as the regressors per equation (C · p)
approach the samples (T − p), so the AIC's global minimum drifts to the largest order tried. On STEW's 4-s windows the mean AIC has a
real minimum at p = 8 and then collapses towards p = 34. The order is therefore searched over p ≤ T / (3C + 1) (at least three
samples per regressor, at most 20), each window's first local AIC minimum is taken, and the median over 300 training windows (no
labels) is used for the whole dataset:

| Dataset | Channels × samples (Hz) | Orders searched | Order | Note |
|---|---|---|---|---|
| STEW, 4-s windows | 14 × 512 (128) | 1–11 | 8 | 254 of 300 windows at 7–8 |
| STEW, 6-s windows | 14 × 768 (128) | 1–17 | 8 | |
| BCI-IV-2a | 21 × 800 (200) | 1–12 | 12 | AIC still falling at the cap; one of 22 channels dropped (average reference) |
| PhysioNet-MI | 63 × 800 (200) | 1–4 | 4 | still falling at the cap; one of 64 channels dropped (average reference) |
| Mumtaz2016 | 19 × 1000 (200) | 1–17 | 17 | still falling at the cap |
| MentalArithmetic | 20 × 1000 (200) | 1–16 | 16 | still falling at the cap |
| Sleep-EDF | 2 × 6000 (200) | 1–20 | 9 | |
| SEED, SEED-V | 62 × 200 (200) | 1 | 1 | a 1-s window allows no more |

**Calibration on Safari et al.'s maps.** Their SIFT settings are not published beyond "AIC" and the bands, so our generator was run on the same
300 STEW windows under 672 combinations of the settings SIFT leaves open or defines differently (MVAR order; per-window scaling; average
reference; full-frequency vs. per-frequency DTF normalisation; partial coherence squared or not; frequency grid; band mean or sum; square
root) and compared with their maps (`../experiments/stew-paper-protocol/results/calibration_safari.txt`):

| Settings | Off-diagonal pattern r (median over windows) | Self-flow r | Mean value, ours / theirs |
|---|---|---|---|
| our first settings (no re-reference, F² · C²) | 0.129 | 0.418 | 0.02 |
| calibrated: average reference, √(F² · \|C\|) | **0.367** | 0.484 | 0.70 |

The order barely matters (0.36–0.37 for p = 6–15). The remaining disagreement is not recoverable from the publications (artifact
subspace reconstruction, line-noise removal, SIFT's MVAR estimator and windowing). Results with the first settings are kept as an
ablation (`ddtf-gl-v0`) in every experiment.

**Average reference.** CBraMod's BCI-IV-2a and PhysioNet-MI samples are average-referenced (the channels sum to zero), so their
rank is C − 1 and the MVAR residual covariance is singular. The last channel is dropped before the maps are computed; it is minus
the sum of the others, so nothing is lost.

## Issues found

| Type | Finding |
|---|---|
| Map source | The paper's maps follow Safari et al., whose repository releases their SIFT maps of STEW but not the code; on those maps the paper's subject-independent number reproduces exactly and its subject-dependent number does not (−0.026). Their notebook averages window pairs across subject boundaries (`n_avg = 2`) |
| Claim of the best result | In the paper's own settings, on identical windows and splits, LaBraM-Base, CBraMod, DGCNN and EEG Conformer score above dDTF-GL; the paper's baselines are weaker networks, and its raw-EEG EEGNet reproduces ([stew-paper-protocol](experiments/stew-paper-protocol)) |
| Released code vs. paper | The bundle is a different method: DWT statistics of the raw signal instead of dDTF maps, and no AXB. It scores 0.845 subject-independent, above the paper's 0.766; its README says its parameters were not checked against the "historical 76%+ run" |
| Signal | The bundle reads STEW's raw values, DC offset (about 4,300 µV) and slow drift included. The 14 channel means of a window alone classify rest vs. task at 0.703 on unseen subjects; high-passing at 0.3 Hz costs the bundle 0.048 ([stew-paper-protocol](experiments/stew-paper-protocol)) |
| Protocol | Neither setting's split is described. The SD number matches pooled 7-fold cross-validation, which puts windows sharing 2 s of signal on both sides (+0.018 here over holding out whole blocks, which also matches it); the SI ± (0.148) is a spread over subjects, not over folds |
| Baseline numbers | "Safari's" row in Table 3 (0.895 ± 0.014) is Safari et al.'s own published 89.53 ± 1.36%, obtained with their preprocessing, their feature selection and an unstated split, not a run "under the same dDTF input setting" |
| Unspecified settings | MVAR order and estimator, frequency grid, AXB selection rule, DFT fraction, LNT and XGBoost hyperparameters (the last three are in the bundle) |
| Efficiency claims | Section 4.2 announces a comparison of FLOPs, model size and carbon per inference; no table or number in the manuscript reports it |
| Ablation | On STEW's shared windows AXB changes nothing (±0.008) and dDTF maps are 0.037 worse than the bundle's DWT statistics ([stew-cbramod-pipeline](experiments/stew-cbramod-pipeline)) |

## License

Our code here has no license chosen yet, as the rest of this repository. The authors' bundle carries no license and is not
redistributed.
