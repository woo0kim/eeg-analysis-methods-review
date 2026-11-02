# EEGNet

> V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, B. J. Lance, *EEGNet: a compact
> convolutional neural network for EEG-based brain–computer interfaces*, J. Neural Eng. 15:056013, 2018.
> doi:[10.1088/1741-2552/aace8c](https://doi.org/10.1088/1741-2552/aace8c)

- Authors' code: [vlawhern/arl-eegmodels](https://github.com/vlawhern/arl-eegmodels) (Keras / TensorFlow). Not used here.
- Implementation evaluated: braindecode's `EEGNet` (named `EEGNetv4` before braindecode 1.x), in PyTorch.
  `upstream/` is a git submodule of [braindecode/braindecode](https://github.com/braindecode/braindecode) pinned at
  `85b8b5c` (release v1.8.1, 2026-08-31), installed into the experiment's environment from that directory.
  braindecode was chosen over the authors' Keras code so that EEGNet runs inside the other methods' PyTorch
  pipelines unchanged.
- License: braindecode is BSD-3-Clause. `experiments/bciciv2a-epoch-selection/code/run_eegnet.py` imports the EEG
  Conformer experiment's GPL-3.0 training loop and is GPL-3.0 as well.

EEGNet is not validated here against its own paper. It is the baseline that two of the papers we validate compare
against: EEG Conformer (Table II) and CBraMod (Tables 3, 10, 12, 15). Each experiment runs it inside that paper's
own pipeline so that only the model differs. Both experiments in this folder use braindecode's `EEGNet`; the older
cross-subject runs inside CBraMod's experiment used our own PyTorch implementation, and
[braindecode-cbramod-splits](experiments/braindecode-cbramod-splits) repeats them with braindecode's.

## Experiments

| Experiment | Dataset | Question | Finding |
|---|---|---|---|
| [bciciv2a-epoch-selection](experiments/bciciv2a-epoch-selection) | BCI IV 2a, within-subject | Does EEG Conformer's lead over EEGNet survive when both are trained by Conformer's released loop and their reported epoch is chosen by the same rule? | Only under the test-set maximum. Under that rule Conformer leads by 6.10 pp (78.72 vs. 72.62%, all 9 subjects, Wilcoxon p = 0.0039). At the last-100-epoch mean or the final epoch the two are level (EEGNet ahead by 0.76 and 0.14 pp). With a validation split Conformer leads by 2.65 pp (70.01 vs. 67.36%), not significantly (p = 0.16). Conformer's test accuracy fluctuates about 5× more from epoch to epoch, so the maximum inflates it more. |
| [pytorch-cbramod-splits](experiments/pytorch-cbramod-splits) | PhysioNet-MI, BCI-IV-2a, Mumtaz2016, MentalArithmetic, cross-subject | How does EEGNet do through CBraMod's released fine-tuning pipeline? | It is the best of all models on BCI-IV-2a (0.5288 balanced accuracy vs. 0.4571 for CBraMod and 0.4477 for LaBraM-Base). It ties on Mumtaz2016 and is below both foundation models on PhysioNet-MI. Our own PyTorch EEGNet-8,2; the runs were made inside CBraMod's [finetune-public-datasets](../cbramod/experiments/finetune-public-datasets) and moved here on 2026-10-29. |
| [braindecode-cbramod-splits](experiments/braindecode-cbramod-splits) | PhysioNet-MI, BCI-IV-2a, Mumtaz2016, MentalArithmetic, cross-subject | Does braindecode's `EEGNet`, through the same CBraMod pipeline, give the same numbers as our PyTorch EEGNet? | Yes: none of the 12 metrics differs significantly (closest: PhysioNet-MI, −0.012, p = 0.053). Still the best model on BCI-IV-2a (0.5255) and tied on Mumtaz2016; 8 of 12 metrics within noise of the CBraMod paper's EEGNet rows (our PyTorch EEGNet: 6). |
| [sleepedf-cbramod-pipeline](experiments/sleepedf-cbramod-pipeline) | Sleep-EDF-78 | How does our PyTorch EEGNet-8,2 do as the epoch encoder of CBraMod's sleep-staging pipeline, against every other model under the same conditions? | It is the weakest model, with braindecode's EEGNet: κ 0.681 on the common test set, with CBraMod 0.086 higher (p < 0.001). REM (F1 0.59) and N1 (0.31) are its weakest stages. Unlike on BCIC-IV-2a, where it beat both foundation models, it is no match for them here. |
| [braindecode-sleepedf](experiments/braindecode-sleepedf) | Sleep-EDF-78 | Does braindecode's `EEGNet`, in the same pipeline, give our implementation's numbers? | Yes: κ 0.6808 vs. 0.6815 (p = 0.95), balanced accuracy +0.001 (p = 0.89), no stage differs significantly, despite braindecode's max-norm constraint, Keras batch-norm defaults and Glorot initialisation. |

## Not yet covered

The EEGNet paper's own results (P300, ERN, MRCP and SMR, within- and cross-subject) and its own training recipe
(128 Hz, Adam with default settings, 500 epochs, the epoch with the lowest validation loss); BCI IV 2b and SEED
(the other EEG Conformer datasets).
