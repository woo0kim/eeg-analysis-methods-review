# L-SeqSleepNet on STEW (off-label)

L-SeqSleepNet is a sleep-staging model: it classifies every 30-s epoch of a sequence from its spectrogram, with recurrent layers inside
each epoch and across the sequence. Its papers report sleep only. Asked to run every method on STEW, we ran its released code
(train_lseqsleepnet.py, test_lseqsleepnet.py, unchanged, through [../sleepedf20-loso/code/run_released.py](../sleepedf20-loso/code/run_released.py)) on
workload data adapted to its input, as for SEED ([seed-offlabel](../seed-offlabel)). **It classifies 30-s windows, not the 4-s windows every
other model sees, so its numbers indicate what the model can do here; they are not a like-for-like comparison.** Folds, tasks and the
high-pass filter are those of every other model ([cbramod/finetune-stew](../../../cbramod/experiments/finetune-stew)).

## Adaptation (`code/prepare_stew.py`, `code/convert.py`, `code/add_jobs.py`)

| | Released (sleep) | Here | Why |
|---|---|---|---|
| Epoch | 30 s at 100 Hz, spectrogram 29 × 129 | the same: each 150-s recording high-passed at 0.3 Hz, resampled from 128 to 100 Hz and cut into its five 30-s windows | the network's input shape is fixed |
| Channels | up to 3 (EEG, EOG, EMG slots) | T7, T8 and FC6, the temporal and fronto-central Emotiv electrodes closest to those used for SEED (Emotiv has no FT8) | at most three signals |
| Sequence | 10 × 20 = 200 epochs | 5 × 1 = 5 epochs: one whole recording | a recording has five windows |
| Files | one per night | one per recording | |
| Training length | 11,000 steps, 110 validations | the released 440 passes over the data, 110 validations | scaled to the data size, as on Sleep-EDF-78 and SEED |
| Labels | 5 stages | rest vs. task (1..2) or 3 rating levels 1–3 / 4–6 / 7–9 (1..3), one-hot over 5 | the released format |
| Splits | – | the pinned 7-fold GroupKFold of [finetune-stew](../../../cbramod/experiments/finetune-stew/results/folds.json); validation = the next fold | as every model |
| Selection | best validation accuracy | the same | released |

3 runs per task and fold (the released code sets no seed), two runs sharing one Quadro M6000 (mcl-server), 19–24 minutes per pair.
Each run is scored on the pooled predictions of its 7 test folds, as every STEW model.

## Results

30-s windows, mean ± SD over 3 runs (`results/results.txt`). Recording vote: majority over a recording's five windows. Fold SD: SD of
accuracy across the 7 folds, mean over runs.

| Task | Accuracy | Bal. acc | κ | Recording vote | Fold SD | Per run (accuracy) |
|---|---|---|---|---|---|---|
| rest vs. task (48 subjects, chance 0.500) | 0.7667 ± 0.0150 | 0.7667 ± 0.0150 | 0.5333 ± 0.0300 | 0.7674 ± 0.0159 | 0.081 | 0.771, 0.750, 0.779 |
| 3 rating levels (45 subjects, majority class 0.467) | 0.5407 ± 0.0238 | 0.4852 ± 0.0313 | 0.2715 ± 0.0431 | 0.5444 ± 0.0294 | 0.095 | 0.553, 0.513, 0.556 |

- **Rest vs. task: 0.767 from three channels,** between DGCNN (0.775) and EEG Conformer (0.736) on 4-s windows of all 14
  channels. A 30-s window holds 7.5 times the signal of a 4-s window, so this does not rank the architectures.
- **3 rating levels: 0.541 (κ 0.27),** inside the range of the other models (accuracy 0.49–0.57, κ 0.12–0.33): no model decodes the
  three self-rated levels well across subjects.
- **The runs fit their training data almost completely:** the last logged training batch is at accuracy 1.0 in 38 of 42 runs (0.925–0.975
  in the others); the checkpoint with the best validation accuracy is kept, as released. The five windows of a recording almost always get one label: the recording vote equals the
  window accuracy within 0.01.

## Reproducing

```bash
python code/collect.py logs/preds results_rerun > results_rerun/results.txt        # from the committed per-window predictions
# training (TF 1.15, the released repository at LSEQ_SRC): prepare_stew.py (STEW txt files) -> convert.py -> add_jobs.py
# predictions from the runs: python code/collect.py export <lseq STEW dir> logs/preds  (hdf5storage)
```

`logs/preds/<task>__fold<k>__l-seqsleepnet__s<run>.npz` holds every test window's label, prediction, subject and recording.

## Provenance

Runs: 2026-11-08 on mcl-server, `~/repro-seed/lseq/STEW`. The released code is CC-BY-NC-4.0.
