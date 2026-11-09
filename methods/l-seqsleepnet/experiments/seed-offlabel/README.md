# L-SeqSleepNet on SEED and SEED-V (off-label)

L-SeqSleepNet is a sleep-staging model: it classifies every 30-s epoch of a long sequence from its spectrogram, with recurrent layers inside
each epoch and across the sequence. Its papers report sleep only. Asked to run every method on SEED and SEED-V, we ran its released code
(train_lseqsleepnet.py, test_lseqsleepnet.py, unchanged, through [../sleepedf20-loso/code/run_released.py](../sleepedf20-loso/code/run_released.py)) on emotion data
adapted to its input. **The adaptation changes what the model sees, so these runs do not score the same samples as the other models; only
the clip vote compares, and only on the trial split.**

## Adaptation (`code/prepare_seed.py`, `code/convert.py`, `code/add_jobs.py`)

| | Released (sleep) | Here | Why |
|---|---|---|---|
| Epoch | 30 s at 100 Hz, spectrogram 29 × 129 | the same: each clip cut into 30-s windows (the rest dropped), resampled from 200 to 100 Hz | the network's input shape is fixed |
| Channels | up to 3 (EEG, EOG, EMG slots) | T7, T8 and FT8, temporal electrodes Zheng & Lu (2015) rank highest on SEED | at most three signals |
| Sequence | 10 × 20 = 200 epochs | 5 × 4 = 20 epochs | the shortest SEED-V part has 20 windows |
| Files | one per night | one per session and part (clips 1–5, 6–10, 11–15) | a sequence never crosses parts |
| Training length | 11,000 steps, 110 validations | the released 440 passes over the data, 110 validations | scaled to the data size, as on Sleep-EDF-78 |
| Labels | 5 stages | 1..3 (SEED) or 1..5 (SEED-V), one-hot over 5 | the released format |
| Splits | – | the trial and subject splits of [cbramod/finetune-seed](../../../cbramod/experiments/finetune-seed) | as every model |

3 runs per dataset and split (the released code sets no seed), on Quadro M6000s (mcl-server), 25–53 minutes per pair of runs.

## Results

Clip vote (majority over a clip's 30-s windows) and window-level scores, mean ± SD over 3 runs (`results/results.txt`).

| Dataset | Split | Clip vote | Window bal. acc | Window κ | Per run (clip vote) |
|---|---|---|---|---|---|
| SEED | trial | 0.5126 ± 0.0245 | 0.4523 ± 0.0128 | 0.1650 ± 0.0107 | 0.538, 0.511, 0.489 |
| SEED-V | trial | 0.3583 ± 0.1343 | 0.3346 ± 0.0963 | 0.1535 ± 0.1060 | 0.450, 0.204, 0.421 |
| SEED | subject (not valid, below) | 0.7877 ± 0.0280 | 0.7547 ± 0.0267 | 0.6361 ± 0.0394 | 0.756, 0.807, 0.800 |
| SEED-V | subject (not valid, below) | 0.4074 ± 0.0370 | 0.3971 ± 0.0323 | 0.2551 ± 0.0405 | 0.407, 0.444, 0.370 |

- **On the trial split L-SeqSleepNet is near the small supervised models:** a SEED clip vote of 0.51 (EEGNet 0.53, EEG Conformer 0.56, CBraMod
  0.70) from three channels; on SEED-V one of three runs never left chance (0.20), the other two reach 0.45 and 0.42.
- **The subject-split numbers are not valid comparisons, most likely because of the clip order.** Every subject saw the clips in the same
  order, so within a file the sequence of labels is the same for every subject; a model reading 20 consecutive windows across clip
  boundaries can learn that sequence. On the trial split, test clips (11–15) come in a different label order from training clips (1–5), and
  L-SeqSleepNet scores 0.51; on the subject split it scores 0.79, while every 1-s model, which sees one window at a time, gets at most 0.64
  on the same split. Testing this directly (sequences confined to one clip) is an open item.

## Reproducing

```bash
python code/collect.py logs/preds results_rerun > results_rerun/results.txt        # from the committed per-window predictions
# training (TF 1.15, the released repository at LSEQ_SRC): prepare_seed.py (LMDB of finetune-seed) -> convert.py -> add_jobs.py
```

`logs/preds/<dataset>__<split>__l-seqsleepnet__s<run>.npz` holds every test window's label, prediction, clip and file.

## Provenance

Runs: 2026-11-06 on mcl-server, `~/repro-seed/lseq`. The released code is CC-BY-NC-4.0.
