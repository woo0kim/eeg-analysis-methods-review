# dDTF-GL on STEW, the paper's protocol

Do the paper's STEW numbers reproduce, under its own segmentation and its two settings and on the maps it says it follows? How do the
repository's networks do in the same settings? And what does the authors' bundle give when run as released?

The paper evaluates only STEW (48 subjects, Emotiv EPOC, 14 channels at 128 Hz, 2.5 min at rest and 2.5 min of the SIMKAP
multitasking test; rest = low workload, test = high), "segmented into overlapping temporal windows ... 6 seconds long, with a
stride of 4 seconds", in a subject-dependent (SD) and a subject-independent (SI) setting (Tables 3–4: accuracy, F1, precision,
recall, mean ± SD; "Our Method (CPU)" is the headline row). It states neither how either setting splits the data nor what the ± is
taken over.

| Setting | What was run | Runs |
|---|---|---|
| SI, dDTF-GL | the paper's method ([../../README.md](../../README.md)) on 6-s windows with a 4-s step (37 per recording, 3,552 in all) with three sources of maps: **Safari et al.'s released SIFT maps** of exactly these windows (`safari-gl`: the paper's input, "following [4]"); our generator with the settings calibrated on them (`ddtf-gl`, after a 1-Hz high-pass); our first settings (`ddtf-gl-v0`). The 7 subject folds pinned for STEW in this repository ([finetune-stew](../../../cbramod/experiments/finetune-stew), `results/folds.json`, rest vs. task), training on the other six | 5 seeds × 7 folds per source |
| SD, dDTF-GL | the same windows and map sources; 7-fold cross-validation over the pooled windows of all subjects, stratified by class and shuffled with the run's seed. Safari et al.'s notebook does the same (`KFold(n_splits=7, shuffle=True, random_state=0)` over the pooled windows), and the paper's SD table quotes their result. Control: 7 contiguous blocks per recording, the neighbours of the test block left out of training (`sd-blocked`) | 5 seeds × 7 folds per source and split |
| SI, the bundle as released | `upstream/mwl_gl_si7.py` with its own config: DWT statistics → DFT → LNT → XGBoost, 4-s windows with 25% overlap of the unfiltered recordings, its own `GroupKFold(n_splits=7)` (`code/run_bundle.sh`) | 5 seeds × 7 folds |
| SI and SD, the repository's networks | EEGNet, EEG Conformer, ATCNet and DeepSleepNet (braindecode), DGCNN, CBraMod, LaBraM-Base on the same windows and splits as dDTF-GL (`code/nets6.py`, `code/run_labram6.py`): each model and recipe of the shared STEW pipeline ([finetune-stew](../../../cbramod/experiments/finetune-stew)), sized for 6-s windows; 50 epochs on the whole training part, **the final epoch scored** (the settings have no validation set, so nothing is selected) | 5 seeds × 7 folds × 2 settings per model |

## Results

### The paper's two settings

Mean over the 7 test folds, ± SD over the folds (averaged over seeds 0–4); in brackets the SD of that mean over seeds; for SI also the SD
over the 48 subjects. Paper: Tables 3 and 4, "Our Method (CPU)" (`results/results.txt`).

| Setting | Maps | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|
| SI | **Safari et al.'s released maps** | **0.7744** ± 0.0683 [0.0052]; subjects ± **0.1474** | 0.7732 | 0.7816 | 0.7744 |
| SI | ours, calibrated | 0.7778 ± 0.0569 [0.0032]; subjects ± 0.1321 | 0.7763 | 0.7856 | 0.7778 |
| SI | ours, first settings | 0.8000 ± 0.0430 [0.0016]; subjects ± 0.1297 | 0.7991 | 0.8044 | 0.8000 |
| SI | *paper* | *0.766 ± 0.148* | *0.756 ± 0.161* | *0.781 ± 0.148* | *0.766 ± 0.148* |
| SI | the authors' bundle as released (DWT, 4-s windows) | 0.8450 ± 0.0368 (ddof 0) [0.0035] | 0.8436 | – | – |
| SD, pooled folds | **Safari et al.'s released maps** | **0.8716** ± 0.0119 [0.0022] | 0.8715 | 0.8721 | 0.8716 |
| SD, pooled folds | ours, calibrated | 0.9126 ± 0.0130 [0.0039] | 0.9125 | 0.9129 | 0.9126 |
| SD, pooled folds | ours, first settings | 0.9153 ± 0.0119 [0.0020] | 0.9152 | 0.9156 | 0.9153 |
| SD | *paper* | *0.898 ± 0.016* | *0.898 ± 0.016* | *0.899 ± 0.015* | *0.898 ± 0.015* |
| SD, whole blocks held out (control) | Safari et al.'s released maps | 0.8613 ± 0.0241 [0.0030] | 0.8612 | 0.8624 | 0.8613 |
| SD, whole blocks held out (control) | ours, calibrated | 0.8885 ± 0.0188 [0.0048] | 0.8885 | 0.8892 | 0.8885 |
| SD, whole blocks held out (control) | ours, first settings | 0.8975 ± 0.0180 [0.0007] | 0.8974 | 0.8983 | 0.8975 |

- **On Safari et al.'s maps the paper's subject-independent result reproduces exactly:** 0.774 against 0.766, F1 0.773 against 0.756,
  precision 0.782 against 0.781, and a spread over the 48 subjects of 0.147 against the paper's 0.148. The paper's ± is a spread over
  subjects, not folds (ours over folds: 0.068), and its input was most likely these released maps.
- **Its subject-dependent result does not:** 0.872 on the same maps with pooled folds, 0.026 below the paper's 0.898; our own maps give
  0.913–0.915 there. The paper does not describe this split; Safari et al.'s notebook pools the windows as we do. The SD spreads
  (0.012–0.013) are fold SDs like the paper's (0.016).
- **Pooled folds overlap.** Windows of a recording share 2 s of signal with their neighbours, and pooled folds put neighbours on both
  sides: holding out whole blocks costs 0.010 (Safari's maps) to 0.024 (our calibrated maps).
- **The authors' bundle scores higher than the paper:** 0.845 subject-independent (0.941 when each recording is labelled by the
  majority of its windows), with DWT statistics instead of dDTF and no AXB. Its README mentions a "historical 76%+ run" that it does
  not claim to reproduce; the 0.766 of the paper is what Safari et al.'s maps give.
- **Safari et al.'s notebook also averages consecutive window pairs over the stacked array** (`n_avg = 2`); with 37 windows per
  recording this pairs one subject's last window with the next subject's first. It is not used here.

### The repository's networks in the same settings

Every network of the shared STEW pipeline on exactly these windows and splits (raw 6-s windows after the 1-Hz high-pass; 200 Hz for
CBraMod and LaBraM), 50 epochs on the whole training part, the final epoch scored (`logs/runs_networks`, `code/nets6.py`). Accuracy,
mean over folds ± SD over folds [SD over seeds]; dDTF-GL on Safari et al.'s maps; dDTF-GL − network, Welch p over seeds
(`results/results.txt`, last section):

| Model | SI | SI, SD over subjects | dDTF-GL − it, SI | SD (pooled folds) | dDTF-GL − it, SD |
|---|---|---|---|---|---|
| LaBraM-Base | **0.8626** ± 0.0537 [0.0033] | 0.130 | −0.088 (p < 0.001) | **0.9733** ± 0.0078 [0.0016] | −0.102 (p < 0.001) |
| CBraMod | 0.8451 ± 0.0588 [0.0067] | 0.133 | −0.071 (p < 0.001) | 0.9619 ± 0.0076 [0.0007] | −0.090 (p < 0.001) |
| DGCNN | 0.8046 ± 0.0415 [0.0043] | 0.163 | −0.030 (p < 0.001) | 0.9557 ± 0.0083 [0.0048] | −0.084 (p < 0.001) |
| EEG Conformer | 0.7842 ± 0.0476 [0.0064] | 0.149 | −0.010 (p = 0.030) | 0.8926 ± 0.0271 [0.0091] | −0.021 (p = 0.006) |
| **dDTF-GL** (Safari et al.'s maps) | 0.7744 ± 0.0683 [0.0052] | 0.147 | – | 0.8716 ± 0.0119 [0.0022] | – |
| ATCNet (braindecode) | 0.7675 ± 0.0559 [0.0075] | 0.158 | +0.007 (p = 0.13) | 0.8360 ± 0.0226 [0.0042] | +0.036 (p < 0.001) |
| DeepSleepNet (braindecode) | 0.7408 ± 0.0434 [0.0191] | 0.171 | +0.034 (p = 0.015) | 0.8495 ± 0.0419 [0.0405] | +0.022 (p = 0.29) |
| EEGNet | 0.6647 ± 0.0670 [0.0256] | 0.172 | +0.110 (p < 0.001) | 0.6814 ± 0.0357 [0.0554] | +0.190 (p = 0.002) |
| *paper, "Our Method (CPU)"* | *0.766 ± 0.148* | | | *0.898 ± 0.016* | |
| *paper, EEGNet on raw EEG (Tables 1–2)* | *0.683* | | | *0.692* | |

- **The paper's "highest accuracy" does not hold against these networks.** In its own two settings, on identical samples and splits and
  with no epoch selection, both foundation models, DGCNN and EEG Conformer score above dDTF-GL; it is level with ATCNet and above
  DeepSleepNet and EEGNet.
- **Our EEGNet agrees with the paper's.** On raw EEG it scores 0.665 (SI) and 0.681 (SD) against the paper's 0.683 and 0.692 (Tables
  1–2): the paper's raw-EEG baselines are reproduced, and they are the weakest networks here. Its dDTF-input baselines (Tables 3–4) were
  not re-run.
- The spread over subjects is similar for every model (0.13–0.17): which subjects are tested matters more than the model.

### What the bundle's number rests on

Controls on the bundle's own 4-s windows and folds, training on the other six folds as it does; window accuracy, mean over folds ±
SD over folds, seeds 0–4 (`results/results.txt`, "Controls"):

| Input | Raw recordings, as the bundle reads them | High-passed at 0.3 Hz (the shared STEW windows) |
|---|---|---|
| DWT statistics, the bundle's chain (`dwt-gl`) | 0.8469 ± 0.0377 | 0.7990 ± 0.0562 |
| dDTF maps, the paper's method (`ddtf-gl`, our calibrated maps) | 0.7389 ± 0.0582 | 0.7485 ± 0.0441 |
| dDTF maps, our first settings (`ddtf-gl-v0`) | 0.7542 ± 0.0408 | 0.7806 ± 0.0470 |
| Only each channel's mean over the window (`dc-gl`, 14 numbers) | **0.7029** ± 0.0480 | 0.5537 ± 0.0322 |

- **Our wrapper reproduces the bundle:** 0.8469 against its 0.84 on the same windows and folds. They are not bit-identical: the bundle
  orders each subject's task recording before the rest recording, which changes XGBoost's row subsampling.
- **Part of the bundle's lead comes from below 0.3 Hz.** High-passing the same windows costs the DWT features 0.048. On the raw
  recordings, the 14 channel means of a window, which carry the Emotiv offset (about 4,300 µV) and its slow drift but no EEG
  rhythm, classify rest against task at 0.703 in subjects never seen in training. After the high-pass they fall to 0.554.
- **Which part of the offset carries it is not a simple shift.** The mean level of the task recording is not consistently above or
  below the rest recording's (same sign in 24–29 of 48 subjects for 13 channels; T8 in 35 of 48, p = 0.002), and the slow
  variation within a recording is not larger during the task (19 of 48 subjects) (`results/dc_offsets.txt`, `code/stew_dc.py`).
  Whatever it is, it is not EEG, and it is present in the raw files the bundle reads.
- **dDTF prefers the filtered signal** (calibrated maps 0.749 against 0.739; first settings 0.781 against 0.754), DWT statistics the raw
  one. Even high-passed, the bundle's DWT statistics are 0.050 ahead of the calibrated dDTF maps.
- **Training size:** dDTF-GL in the shared pipeline ([stew-cbramod-pipeline](../stew-cbramod-pipeline)) trains on five folds and scores
  0.740 pooled; with six folds on the same windows, 0.749.

Cost (the paper's "green" claim; mcl-server, 16–20 threads): one of our dDTF maps takes 8.5 ms of one CPU core per 4-s window and 14 ms
per 6-s window (Safari et al.'s SIFT maps came precomputed); fitting DFT, AXB, LNT and XGBoost on ~3,000 windows takes 8–18 s; the
classifier has 500 trees (about 7,000 nodes) on 276 features and predicts in 4–7 µs per window. The paper's text announces FLOPs, model size and carbon per inference, but no table
in the manuscript reports them.

## Paper vs. this reproduction

### What differs, and why

| | Paper | Authors' bundle | Here | Why |
|---|---|---|---|---|
| Features | dDTF maps in 5 bands, "following [4]" (Safari et al.) | DWT statistics (Haar level 4: 9 statistics per channel and level, 16 cross-channel summaries per level; 710) | Safari et al.'s released SIFT maps of these windows (980; headline); our generator calibrated on them, and with its first settings | the paper's input; their code is not released, only the maps (github.com/safari-mohammadreza/MWL_MachineLearning @604d243). The bundle is also run as released |
| AXB | yes; no selection rule given | no ("No auxiliary branch ... in this version") | yes, our rule (positive gain, at most half the DFT set) | the paper's method |
| DFT, LNT, XGBoost settings | not given | 50% kept (B 32); LNT groups of 8, 2 components; 500 trees, depth 3, lr 0.1, subsample and colsample 0.8 | the bundle's | the only source |
| Preprocessing | not given; dDTF "following [4]" (Safari et al.: 1-Hz high-pass, line-noise removal, artifact subspace reconstruction, average reference) | none (raw values) | Safari et al.'s maps: theirs. Our maps: 1-Hz high-pass (FIR, MNE) and average reference (calibrated maps; the residual covariance is then inverted with a pseudo-inverse) | ASR and their line-noise removal have no published parameters; Emotiv's headset already notch-filters |
| Windows | 6 s, 4-s step | 4 s, 25% overlap (3-s step), 49 per recording | 6 s, 4-s step | the paper's |
| SI folds | not given | `GroupKFold(7)` over the 96 recordings grouped by subject | the 7 pinned folds of [finetune-stew](../../../cbramod/experiments/finetune-stew) | `GroupKFold`'s assignment of equal-sized groups depends on the numpy / scikit-learn version; the pinned folds are the ones every model here uses. The bundle ran its own (fold 0's test subjects are in `results/results.json`) |
| SD split | not given | – | 7-fold stratified over pooled windows, shuffled by seed | the reading under which the paper's SD row is comparable with Safari et al.'s, which it quotes |
| ± | not given | SD over the 7 folds (numpy, ddof 0) | SD over the 7 folds (ddof 1), mean over seeds; SI also SD over the 48 subjects | on Safari et al.'s maps the SD over subjects (0.147) matches the paper's SI SD (0.148); the fold SD (0.068) does not |
| F1, precision, recall | not defined | macro F1 | macro (the paper's SI recall equals its accuracy, as macro recall does on balanced folds) | – |
| GPU row | XGBoost on GPU | – | not run | the CPU row is the paper's headline; the environment's xgboost is the CPU build |

### Settings we had to choose

| Setting | Value | Why |
|---|---|---|
| Seeds | 0–4 | as everywhere in this repository; the paper does not report seeds or repetitions |
| MVAR order | our maps: 8, by AIC on 300 training windows (fold 0's training subjects); Safari et al.'s: their AIC choice, not reported | [../../README.md](../../README.md#the-method-as-run-here) |
| AXB rule | positive gain, at most half of the DFT set | the paper gives none |
| Threads | 16 for XGBoost and DFT | shared server; the results do not depend on it |
| Networks: checkpoint | the final of 50 epochs, trained on the whole training part | the paper's settings have no validation set; keeping the final epoch uses the same training data as dDTF-GL and selects nothing (chosen over carving a validation fold out of the training subjects) |
| Networks: input and recipe | the 6-s windows dDTF-GL saw (1-Hz high-pass; 200 Hz for CBraMod and LaBraM), each model with the shared STEW pipeline's recipe | identical samples for every model in these settings; the models and recipes are the ones every model used on STEW |
| LaBraM's validation slot | the first 64 training windows, never used | LaBraM's script requires a validation set; its final-epoch test probabilities are scored |

### Run configuration

The bundle's settings for DFT, LNT and XGBoost (table above); dDTF as in [../../README.md](../../README.md). Environment
[../../env/conda_list.txt](../../env/conda_list.txt) (Python 3.11, numpy 2.4.6, scikit-learn 1.9.1, xgboost 3.4.2 CPU, pywavelets
1.9.0, opencv 5.0.0). Hardware: dDTF-GL and the bundle on mcl-server, 2× Xeon E5-2620 v3 (24 threads), CPU only; the networks on
mcl-server2's RTX A6000 GPUs (shared with other users), with the shared STEW pipeline's environments (PyTorch 2.1.2, braindecode 1.8.1).

## Reproducing

```bash
code/reproduce.sh collect          # results/ from logs/runs (numpy, scikit-learn)
code/reproduce.sh train            # features, the dDTF-GL runs and the 5 bundle runs (STEW raw files; methods/ddtf-gl/upstream)
# Safari et al.'s maps (github.com/safari-mohammadreza/MWL_MachineLearning @604d243, its files/ folder) as features, and the calibration:
python ../../code/safari_maps.py <their files/> ~/repro-gl/features/STEW-6s/safari
python ../../code/calibrate_safari.py <dir holding their files/> > results/calibration_safari.txt
# the networks (GPU; the environments of the shared STEW pipeline, P6_WORK as in code/add_jobs6.py):
python code/nets6.py build ~/repro-seed/stew6                       # windows6.npz: the same 6-s windows
python code/add_jobs6.py eegnet conformer bd-atcnet bd-deepsleepnet dgcnn cbramod labram
RUN_QUEUE_DIR=~/repro-seed/stew6/queue python ../../../../tools/run_queue.py 0 1 2 3 4 5 6 7
```
The networks' runs are in `logs/runs_networks/<si|sd>__fold<k>__<model>__s<seed>/`: `log.txt`, `metrics.json` (training loss and the test
scores of every epoch, recorded and never used) and `test_preds.npz` (final epoch). LaBraM writes the probabilities of every epoch
(`preds.npz`); `code/reproduce.sh collect` turned them into the same two files, and `preds.npz` is not committed (too large).

`logs/runs/<setting>__fold<k>__<variant>__s<seed>/` (settings `si`, `sd`, `sd-blocked`, and the controls `si4raw`, `si4hp`; variants
`safari-gl`, `ddtf-gl`, `ddtf-gl-v0`, `dwt-gl`, `dc-gl`) holds `log.txt`, `metrics.json` (scores, feature counts per stage, model size,
timings) and `test_preds.npz` (labels, predictions, probabilities); `logs/runs/bundle__s<seed>/` the bundle's own report
(`mwl_gl_si7.json`), its log and the four lines of its config that were changed (`config.diff`).

## Provenance

Runs: 2026-11-11 to 11-13, `~/repro-gl/runs/stew-paper-protocol` on mcl-server; the networks `~/repro-seed/stew6/runs` on mcl-server2.
Data: IEEE DataPort, doi:10.21227/44r8-ya50 (CC BY); Safari et al.'s maps from their public repository (no license stated; not
redistributed here, only the scores computed from them).
