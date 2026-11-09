# L-SeqSleepNet

> H. Phan, K. P. Lorenzen, E. Heremans, O. Y. Chén, M. C. Tran, P. Koch, A. Mertins, M. Baumert, K. B. Mikkelsen,
> M. De Vos, *L-SeqSleepNet: Whole-cycle Long Sequence Modelling for Automatic Sleep Staging*, IEEE JBHI 27(10),
> 2023. [IEEE Xplore 10210638](https://ieeexplore.ieee.org/document/10210638),
> [arXiv:2301.03441](https://arxiv.org/abs/2301.03441)

- Released code: [pquochuy/L-SeqSleepNet](https://github.com/pquochuy/L-SeqSleepNet) (TensorFlow 1.x), CC-BY-NC-4.0.
  The repository ships the prepared Sleep-EDF-20 data (`sleepedf-20/mat_30min`, 39 nights), the leave-one-subject-out
  file lists, run scripts for 5 repetitions, and SHHS-pretrained weights.
- `upstream/`: git submodule pinned at `06c89f7` (2023-12-07), shallow (the data make the repository ~0.9 GB).
- Data preparation: not in the repository; its README points to
  [pquochuy/xsleepnet](https://github.com/pquochuy/xsleepnet) (MATLAB).
- Not in braindecode.

## The model

Each 30-s Fpz-Cz epoch (100 Hz) becomes a log-magnitude spectrogram (2-s Hamming windows, 50% overlap, 256-point FFT:
29 × 129). Epoch encoder (from SeqSleepNet): a learnable filterbank (32 filters), a BLSTM over the 29 frames and
attention pooling. Long-sequence model: a sequence of L = 200 epochs is folded into subsequences, a BLSTM runs within
each subsequence and another across subsequences (with fully-connected layers, layer normalisation and residuals),
then two fully-connected layers classify every epoch. Adam (lr 1e-4), batch 8 sequences, ℓ2 1e-4, dropout 0.1; the
checkpoint is the best validation accuracy, evaluated every 100 steps. At test time each epoch is classified in up to
200 overlapping sequences and the outputs are fused.

## Paper's Sleep-EDF result

"SleepEDF" = Sleep-EDF-20 (Sleep Cassette, 20 subjects, 39 nights, Fpz-Cz), lights-off − 30 min to lights-on + 30 min;
leave-one-subject-out with 4 validation subjects per fold, repeated 5 times (Table II). Random initialisation:
accuracy 86.3 ± 0.2%, κ 0.813 ± 0.003, MF1 79.3 ± 0.4; per-stage F1 W 91.6, N1 45.3, N2 88.5, N3 86.2, REM 85.2
(Table III). Pre-trained on SHHS (not checkable here: SHHS needs NSRR access): 88.6 ± 0.1%, κ 0.845, MF1 82.9.

## Experiments

| Experiment | Dataset | Question | Finding |
|---|---|---|---|
| [sleepedf20-loso](experiments/sleepedf20-loso) | Sleep-EDF-20 | Do the released code and data reproduce 86.3% / κ 0.813 / MF1 79.3 (random initialisation)? | Yes, in one of the 5 repetitions: 86.1% / 0.811 / 79.4, with per-stage F1 within 0.6 points. The released scripts test every fold on subject 0's list (fixed here); they fold the sequence 20 × 10 and train a fixed 11,000 steps, where the paper describes 10 × 20 with early stopping. The released evaluation batch needs more than 12 GB, so it was reduced (prediction-neutral). |
| [sleepedf78-cbramod-split](experiments/sleepedf78-cbramod-split) | Sleep-EDF-78 | How does the released L-SeqSleepNet compare with the other models on the same 62 / 8 / 8 subject split and test epochs? | κ 0.773 ± 0.004 over 3 runs on the common test set ties it with LaBraM-Base and CBraMod (p ≥ 0.40). It is 0.070 above the released DeepSleepNet (p = 0.003) and has the best REM F1 of all models (0.834). |
| [seed-offlabel](experiments/seed-offlabel) | SEED, SEED-V | The released code on emotion data (30-s windows, 3 channels) | Trial split: SEED clip vote 0.51, near the small supervised models. Subject split not valid: its sequences can read the fixed clip order |
| [stew-offlabel](experiments/stew-offlabel) | STEW | The released code on STEW (one 5-window sequence per recording) | Rest vs. task 0.767 from 3 channels on 30-s windows (between DGCNN and EEG Conformer on 4-s windows); 3 levels 0.541, κ 0.27, inside the other models' range |
