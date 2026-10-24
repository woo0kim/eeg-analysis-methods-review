# ATCNet

> H. Altaheri, G. Muhammad, M. Alsulaiman, *Physics-Informed Attention Temporal Convolutional Network for EEG-Based
> Motor Imagery Classification*, IEEE Transactions on Industrial Informatics 19(2):2249–2258, 2023.
> doi:[10.1109/TII.2022.3197419](https://doi.org/10.1109/TII.2022.3197419)

- Released code: [Altaheri/EEG-ATCNet](https://github.com/Altaheri/EEG-ATCNet) (TensorFlow/Keras), Apache-2.0
- `upstream/`: git submodule pinned at `65162fb` (2026-08-16), the current code. Its history also holds the code
  and results behind the paper: commit `3a318b2` (2022-08-18, nine days after the paper went online) uploaded
  `main.py`, `models.py`, the training log of the paper's 90 runs and the best run's weights per subject. That
  `results/` folder was deleted on 2023-12-31, and the training code was rewritten on 2023-12-28 (`a4a68f9`).
  Both versions are evaluated here.
- Released weights: `upstream/results/saved models/run-1/` (current code, `main_TrainValTest.py`, one seed; the
  README reports 81.10% for them) and, in the history at `3a318b2`, the best-of-10 models behind the paper's 85.38%.
- braindecode: `ATCNet` (braindecode 1.8.1, pinned as [methods/eegnet/upstream](../eegnet/upstream)); the upstream
  README points to it as the PyTorch version.

## The model

A convolutional block (EEGNet-like: temporal conv 16 × 64 taps, depthwise spatial conv ×2, average pooling 8 and
7) turns a 22 × 1125 trial (4.5 s at 250 Hz) into 20 time steps of 32 features. Five overlapping windows of 16 steps
each go through their own multi-head self-attention (2 heads of size 8) and a two-block dilated causal TCN; the last
step of each window is mapped to the 4 classes and the five outputs are averaged (the paper says they are
concatenated). 113,732 trainable parameters (the paper's 115.2K includes the BatchNorm moving statistics).

## Experiments

| Experiment | Dataset | Question | Finding |
|---|---|---|---|
| [bciciv2a-model-selection](experiments/bciciv2a-model-selection) | BCI IV 2a | Does the released code reproduce 85.38% (subject-dependent) and 70.97% (LOSO), and what does the number depend on? | 85.38% is the authors' best of 10 runs per subject, each run checkpointed on the test set: their 2022 log says so, and their saved best-of-10 models give exactly 85.38% on our data. Our 90 runs of that 2022 code average 81.47% (the authors' runs: 81.98%). With a validation split, the authors' current `main_TrainValTest.py` gives 79.52% (10 seeds, −5.86 pp). LOSO: 69.06% mean, 70.16% best of 3 (paper 70.97%). Today's `main_TrainTest.py` is a rewritten model (L2, max-norm, lr schedule) and scores 86.89% per run under the old test-set protocol. |
| [braindecode-bciciv2a](experiments/braindecode-bciciv2a) | BCI IV 2a | Is braindecode's `ATCNet` the released network, and does it reproduce the released code's numbers when trained the same way? | Same network: the authors' weights map one to one and give identical predictions once BatchNorm's eps is Keras' 1e-3 (with braindecode's eps the paper's models score 84.53%, not 85.38%). Trained by a PyTorch port of the released loops, braindecode's defaults score 5.2 pp below the 2022 code in the paper's protocol (76.26 vs. 81.47%) and 10–13 pp below today's code. The cause is its `MaxNormLinear`, which limits the final layer's norm per input feature instead of per output unit: fixing the axis recovers 3.7–4.6 pp. A control with the Keras training behaviour matches the Keras runs; braindecode's documented "faithful" options close two thirds of the gap. |

## Not yet covered

The paper's ablations (Table II, Figs. 7–8, Table III) and the High Gamma Dataset numbers in the upstream README.
