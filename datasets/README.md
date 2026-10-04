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
| BCI Competition IV-2a | motor imagery, 4-class, 9 subjects | [BNCI Horizon 2020, 001-2014](https://bnci-horizon-2020.eu/database/data-sets) | `bciciv2a` | `bciciv2a/data_mat/A0{1..9}{T,E}.mat` | [EEG-Conformer](../methods/eeg-conformer/experiments/bciciv2a-epoch-selection), [CBraMod](../methods/cbramod/experiments/finetune-public-datasets) |
| PhysioNet-MI | motor imagery, 4-class (imagery runs R04/06/08/10/12/14) | [PhysioNet eegmmidb 1.0.0](https://physionet.org/content/eegmmidb/1.0.0/) | `physio` | `physio/files/S*/S*R*.edf` | [CBraMod](../methods/cbramod/experiments/finetune-public-datasets) |
| Mumtaz2016 | depression (MDD) vs. healthy | [figshare 4244171](https://doi.org/10.6084/m9.figshare.4244171) | `mumtaz` | `mumtaz/files/` | [CBraMod](../methods/cbramod/experiments/finetune-public-datasets) |
| MentalArithmetic | mental stress, 2-class | [PhysioNet eegmat 1.0.0](https://physionet.org/content/eegmat/1.0.0/) | `mental` | `mental_arithmetic/edf/` | [CBraMod](../methods/cbramod/experiments/finetune-public-datasets) |
| SHU-MI | motor imagery, 2-class | [figshare 19228725](https://doi.org/10.6084/m9.figshare.19228725) | `shu` | `shu/mat_unzip/` | – (files are password-protected) |

The script verifies the PhysioNet SHA-256 sums, the figshare MD5 sums and the SHU-MI archive MD5.

## Not downloadable by script

Used by papers in this repo but gated behind an account, license or request: FACED (Synapse),
SEED, SEED-V and SEED-VIG (license application), TUAB and TUEV (TUH registration), BCIC2020-3 (test
labels not public). BCI Competition IV-2b (EEG Conformer) is public but has no download key yet.
