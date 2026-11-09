# DGCNN

> T. Song, W. Zheng, P. Song, Z. Cui, *EEG Emotion Recognition Using Dynamical Graph Convolutional Neural Networks*, IEEE
> Transactions on Affective Computing 11(3), 532–541, 2020 (online 2018). [doi:10.1109/TAFFC.2018.2817622](https://doi.org/10.1109/TAFFC.2018.2817622)

A graph network for multichannel EEG: electrodes are nodes, each with a feature vector (five-band differential entropy in the
paper), and the adjacency matrix is a trained parameter instead of being fixed by electrode positions. One Chebyshev graph
convolution, then fully connected layers. Paper: 90.40% subject-dependent and 79.95% subject-independent on SEED.

- **Authors' code:** `EEG-TAC.zip`, TensorFlow 1.x, a modified [cnn_graph](https://github.com/mdeff/cnn_graph), the `[code]` link beside
  the paper on Tengfei Song's page, now offline. `fetch_upstream.sh` reads the Wayback Machine's capture of 2022-11-22, pinned by
  SHA-256 (15e4df18…). The archive has no license, so it is not committed and `upstream/` is ignored by git. It has not been run here.
- **Our implementation:** [`code/dgcnn.py`](code/dgcnn.py), PyTorch, written from that code, plus differential entropy from raw
  windows for pipelines that feed signals. The paper itself is closed access and was not read in full: paper numbers are the
  abstract's, settings the authors' code.
- **braindecode:** `DGCNN` (v1.8.1, `methods/eegnet/upstream`), the same model on the same archive; same weights give the same outputs
  as ours to 1.2e-07. It ships with raw samples as node features, a spatial initial graph and no L2 term.

## Experiments

| Experiment | Question | Finding |
|---|---|---|
| [seed-paper-protocol](experiments/seed-paper-protocol) | Do the paper's SEED numbers reproduce, and how are they computed? | 90.40% is the peak of ~80 test evaluations over each subject's best 2 of 3 sessions. Under that rule we get 87.69%; with no test selection, 77.79%, level with a linear SVM (79.42%, p = 0.33). Across subjects the paper holds: 77.58% against its 79.95%, with no test selection, and 58.85% for the SVM. |
| [cbramod-splits](experiments/cbramod-splits) | How does it do on CBraMod's four public datasets? | Best model on Mumtaz2016 (0.96, DE input), second to LaBraM-Base on MentalArithmetic, last on both motor-imagery datasets, where raw samples beat DE by 0.09–0.11. |
| [bciciv2a-epoch-selection](experiments/bciciv2a-epoch-selection) | BCI-IV-2a within subject, through EEG Conformer's loop | 44–50%, over 20 points below EEG Conformer and EEGNet under every epoch rule. |
| [sleepedf-cbramod-pipeline](experiments/sleepedf-cbramod-pipeline) | Sleep-EDF-78 in the shared pipeline | κ 0.694 (DE), 0.596 (raw), level with EEG Conformer and EEGNet, below the foundation models. |
| [seed-cbramod-pipeline](experiments/seed-cbramod-pipeline) | SEED and SEED-V in the shared pipeline (1-s samples, no DE smoothing) | Ahead of EEG Conformer and EEGNet on both trial splits, behind braindecode's DeepSleepNet, CBraMod and LaBraM-Base. Across subjects on SEED 0.52, against 77.6% in its own protocol, which uses SEED's clip-smoothed features and per-session z-scoring. |
| [braindecode-seed](experiments/braindecode-seed) | braindecode's `DGCNN` there, as shipped (raw) and on DE | On DE level with ours on SEED; raw input costs 0.055 (SEED) and 0.102 (SEED-V). |
| [stew-cbramod-pipeline](experiments/stew-cbramod-pipeline) | STEW, 7-fold GroupKFold | The best model without pre-training (0.775 rest vs. task; braindecode's DGCNN on DE 0.783); near the foundation models on 3 levels. |
| [braindecode-stew](experiments/braindecode-stew) | braindecode's `DGCNN` on STEW | On DE level with ours; raw input costs 0.106. |

## License

Our code here has no license chosen yet, as the rest of this repository. The experiment `bciciv2a-epoch-selection` imports GPL-3.0 code.
