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
own pipeline so that only the model differs.

## Experiments

| Experiment | Dataset | Question | Finding |
|---|---|---|---|
| [bciciv2a-epoch-selection](experiments/bciciv2a-epoch-selection) | BCI IV 2a, within-subject | Does EEG Conformer's lead over EEGNet survive when both are trained by Conformer's released loop and their reported epoch is chosen by the same rule? | Only under the test-set maximum. Under that rule Conformer leads by 6.10 pp (78.72 vs. 72.62%, all 9 subjects, Wilcoxon p = 0.0039). At the last-100-epoch mean or the final epoch the two are level (EEGNet ahead by 0.76 and 0.14 pp). With a validation split Conformer leads by 2.65 pp (70.01 vs. 67.36%), not significantly (p = 0.16). Conformer's test accuracy fluctuates about 5× more from epoch to epoch, so the maximum inflates it more. |
| EEGNet baseline inside CBraMod's [finetune-public-datasets](../cbramod/experiments/finetune-public-datasets) | PhysioNet-MI, BCI-IV-2a, Mumtaz2016, MentalArithmetic, cross-subject | How does EEGNet do through CBraMod's released fine-tuning pipeline? | Best of all models on BCI-IV-2a (0.5288 balanced accuracy vs. 0.4571 CBraMod, 0.4477 LaBraM-Base); ties on Mumtaz2016; below both foundation models on PhysioNet-MI. These runs predate this folder and use our own PyTorch EEGNet-8,2 (`scripts/run_eegnet.py` there), not braindecode. |

## Not yet covered

The EEGNet paper's own results (P300, ERN, MRCP and SMR, within- and cross-subject) and its own training recipe
(128 Hz, Adam with default settings, 500 epochs, the epoch with the lowest validation loss); BCI IV 2b and SEED
(the other EEG Conformer datasets).
