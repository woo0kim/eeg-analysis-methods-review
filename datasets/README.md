# Datasets

Raw and preprocessed data are never committed. Every experiment reads them from `$EEG_DATA_ROOT`
(default `~/data`), so methods evaluated on the same dataset share one download.

```bash
datasets/download.sh <key>      # Linux; needs wget, curl and GNU coreutils (sha256sum / md5sum)
```

Preprocessing is method-specific and lives with each experiment, since reproducing a paper means
using its own pipeline.

| Dataset | Task | Source | `download.sh` key | Raw files under `$EEG_DATA_ROOT` | Used by |
|---|---|---|---|---|---|
| BCI Competition IV-2a | motor imagery, 4-class, 9 subjects | [BNCI Horizon 2020, 001-2014](https://bnci-horizon-2020.eu/database/data-sets) | `bciciv2a` | `bciciv2a/data_mat/A0{1..9}{T,E}.mat` | [EEG-Conformer](../methods/eeg-conformer/experiments/bciciv2a-epoch-selection), [CBraMod](../methods/cbramod/experiments/finetune-public-datasets), [LaBraM](../methods/labram/experiments/finetune-cbramod-splits), [EEGNet](../methods/eegnet/experiments/bciciv2a-epoch-selection), [ATCNet](../methods/atcnet/experiments/bciciv2a-model-selection); braindecode re-implementations: [EEG-Conformer](../methods/eeg-conformer/experiments/braindecode-bciciv2a), [ATCNet](../methods/atcnet/experiments/braindecode-bciciv2a), [CBraMod](../methods/cbramod/experiments/braindecode-public-datasets), [LaBraM](../methods/labram/experiments/braindecode-cbramod-splits), [EEGNet](../methods/eegnet/experiments/braindecode-cbramod-splits) |
| PhysioNet-MI | motor imagery, 4-class (imagery runs R04/06/08/10/12/14) | [PhysioNet eegmmidb 1.0.0](https://physionet.org/content/eegmmidb/1.0.0/) | `physio` | `physio/files/S*/S*R*.edf` | [CBraMod](../methods/cbramod/experiments/finetune-public-datasets), [LaBraM](../methods/labram/experiments/finetune-cbramod-splits); braindecode re-implementations: [CBraMod](../methods/cbramod/experiments/braindecode-public-datasets), [LaBraM](../methods/labram/experiments/braindecode-cbramod-splits), [EEGNet](../methods/eegnet/experiments/braindecode-cbramod-splits) |
| Mumtaz2016 | depression (MDD) vs. healthy | [figshare 4244171](https://doi.org/10.6084/m9.figshare.4244171) | `mumtaz` | `mumtaz/files/` | [CBraMod](../methods/cbramod/experiments/finetune-public-datasets), [LaBraM](../methods/labram/experiments/finetune-cbramod-splits); braindecode re-implementations: [CBraMod](../methods/cbramod/experiments/braindecode-public-datasets), [LaBraM](../methods/labram/experiments/braindecode-cbramod-splits), [EEGNet](../methods/eegnet/experiments/braindecode-cbramod-splits) |
| MentalArithmetic | mental stress, 2-class | [PhysioNet eegmat 1.0.0](https://physionet.org/content/eegmat/1.0.0/) | `mental` | `mental_arithmetic/edf/` | [CBraMod](../methods/cbramod/experiments/finetune-public-datasets), [LaBraM](../methods/labram/experiments/finetune-cbramod-splits); braindecode re-implementations: [CBraMod](../methods/cbramod/experiments/braindecode-public-datasets), [LaBraM](../methods/labram/experiments/braindecode-cbramod-splits), [EEGNet](../methods/eegnet/experiments/braindecode-cbramod-splits) |
| SHU-MI | motor imagery, 2-class | [figshare 19228725](https://doi.org/10.6084/m9.figshare.19228725) | `shu` | `shu/mat_unzip/` | – (files are password-protected) |
| Sleep-EDF (Sleep-EDF Expanded, Sleep Cassette study) | sleep staging, 5 AASM stages, 78 subjects, 153 nights | [PhysioNet sleep-edfx 1.0.0](https://physionet.org/content/sleep-edfx/1.0.0/) | `sleepedf` | `sleep_edfx/sleep-cassette/SC4*-{PSG,Hypnogram}.edf`, `sleep_edfx/SC-subjects.xls` | in CBraMod's sleep-staging pipeline: [CBraMod](../methods/cbramod/experiments/finetune-sleepedf), [LaBraM](../methods/labram/experiments/sleepedf-cbramod-pipeline), [EEG-Conformer](../methods/eeg-conformer/experiments/sleepedf-cbramod-pipeline), [EEGNet](../methods/eegnet/experiments/sleepedf-cbramod-pipeline); braindecode re-implementations: [CBraMod](../methods/cbramod/experiments/braindecode-sleepedf), [LaBraM](../methods/labram/experiments/braindecode-sleepedf), [EEG-Conformer](../methods/eeg-conformer/experiments/braindecode-sleepedf), [EEGNet](../methods/eegnet/experiments/braindecode-sleepedf), [ATCNet](../methods/atcnet/experiments/braindecode-sleepedf), [DeepSleepNet](../methods/deepsleepnet/experiments/braindecode-sleepedf); the sleep models' released code: DeepSleepNet on Sleep-EDF-20 ([sleepedf20-20fold](../methods/deepsleepnet/experiments/sleepedf20-20fold), the 39 nights of subjects 0–19) and Sleep-EDF-78 ([sleepedf78-cbramod-split](../methods/deepsleepnet/experiments/sleepedf78-cbramod-split)), L-SeqSleepNet on Sleep-EDF-78 ([sleepedf78-cbramod-split](../methods/l-seqsleepnet/experiments/sleepedf78-cbramod-split)) |

The script verifies the PhysioNet SHA-256 sums, the figshare MD5 sums and the SHU-MI archive MD5. Sleep-EDF's
Sleep Telemetry study (44 nights of a temazepam trial) is not downloaded. L-SeqSleepNet's Sleep-EDF-20 experiment
([sleepedf20-loso](../methods/l-seqsleepnet/experiments/sleepedf20-loso)) uses the prepared files shipped in its
repository (`methods/l-seqsleepnet/upstream/sleepedf-20/mat_30min`), not this download.

## Not downloadable by script

Used by papers in this repo but gated behind an account, license or request: FACED (Synapse),
SEED, SEED-V and SEED-VIG (license application), TUAB and TUEV (TUH registration), BCIC2020-3 (test
labels not public). BCI Competition IV-2b (EEG Conformer) is public but has no download key yet.
