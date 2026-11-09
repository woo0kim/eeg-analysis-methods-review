# DeepSleepNet

> A. Supratak, H. Dong, C. Wu, Y. Guo, *DeepSleepNet: a Model for Automatic Sleep Stage Scoring based on Raw
> Single-Channel EEG*, IEEE TNSRE 25(11):1998–2008, 2017. doi:[10.1109/TNSRE.2017.2721116](https://doi.org/10.1109/TNSRE.2017.2721116)

- Released code: [akaraspt/deepsleepnet](https://github.com/akaraspt/deepsleepnet) (TensorFlow 1.15, Python 3.5 per its
  README), Apache-2.0 ("for academic and non-commercial use only" in its README)
- `upstream/`: git submodule pinned at `eb72f9b` (2024-07-19), the current code. Its last code change
  (`58fafa4`, 2023-11-15) only replaces `np.int` by `int` in `prepare_physionet.py`.
- No released weights.
- braindecode: `DeepSleepNet` (braindecode 1.8.1, pinned as [methods/eegnet/upstream](../eegnet/upstream)). Its
  docstring states that it processes single epochs: its BiLSTM runs over a sequence of length 1, where the released
  model runs it across sequences of 25 epochs.

## The model

Raw single-channel EEG (Fpz-Cz, 100 Hz) per 30-s epoch → two CNN paths (small filters: kernel Fs/2, stride Fs/16;
large filters: kernel 4·Fs, stride Fs/2; four convolutions each), concatenated (3,072 features) → two bidirectional
LSTM layers (512 units) across a sequence of 25 epochs, with a residual fully-connected shortcut → softmax per epoch.
Two-step training: the CNNs are pre-trained on a class-balanced (oversampled) training set for 100 epochs, then the
whole network is fine-tuned on sequences for 200 epochs (Adam; lr 1e-6 for the CNNs, 1e-4 for the rest). No
validation set and no early stopping: the final model is reported.

## Paper's Sleep-EDF result

Sleep-EDF (version of 2013): the 20 Sleep Cassette subjects (39 nights), Fpz-Cz, sleep period ± 30 min of wake,
41,950 epochs; 20-fold cross-validation (one subject per fold), predictions pooled: accuracy 82.0%, macro-F1 76.9,
κ 0.76; per-stage F1 W 84.7, N1 46.6, N2 85.9, N3 84.8, REM 82.4 (Tables III–IV). Pz-Oz: 79.8%, 73.1, 0.72.

## Experiments

| Experiment | Dataset | Question | Finding |
|---|---|---|---|
| [sleepedf20-20fold](experiments/sleepedf20-20fold) | Sleep-EDF-20 | Does the released code reproduce 82.0% / MF1 76.9 / κ 0.76 with the paper's 20-fold protocol? | Yes: 81.8% / 76.7 / 0.754 on exactly the paper's 41,950 epochs, with per-stage F1 within 1.4 points. The count matches because `predict.py` drops each night's last n mod 25 epochs: the paper's class counts are those after the drop. |
| [sleepedf78-cbramod-split](experiments/sleepedf78-cbramod-split) | Sleep-EDF-78 | How does the released DeepSleepNet compare with the other models on the same 62 / 8 / 8 subject split and test epochs? | It is the weakest sequence model: κ 0.703 ± 0.010 over 3 runs on the common test set, 0.070 below L-SeqSleepNet (p = 0.003) and 0.060 below braindecode's DeepSleepNet in CBraMod's pipeline (p = 0.002). The released final-model rule overfits on this larger training set: validation accuracy peaks at fine-tuning epoch 7–10 (0.78–0.79) and ends at 0.75–0.76. The released `prepare_physionet.py` crashes on Sleep-EDF-78 (SC4362F0), so the data are written by a port. |
| [braindecode-sleepedf](experiments/braindecode-sleepedf) | Sleep-EDF-78 | braindecode's `DeepSleepNet` as the epoch encoder of CBraMod's sleep-staging pipeline, like every other model | It ties braindecode's LaBraM for the highest balanced accuracy of all models (0.767), 0.013 above CBraMod (p = 0.027), and its κ (0.762) is level with CBraMod's. That is 0.060 κ above the released DeepSleepNet on the same split. |
| [braindecode-seed](experiments/braindecode-seed) | SEED, SEED-V | braindecode's `DeepSleepNet` (off-label) in the shared SEED pipeline | Level with CBraMod on SEED-V's trial split (0.400 vs. 0.396); the weakest across subjects on SEED |
| [braindecode-stew](experiments/braindecode-stew) | STEW | braindecode's `DeepSleepNet` on STEW | 0.717 on rest vs. task, between EEG Conformer and CBraMod without pre-training |
