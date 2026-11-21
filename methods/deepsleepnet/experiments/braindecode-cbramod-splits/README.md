# DeepSleepNet (braindecode), off-label, on CBraMod's four public datasets

DeepSleepNet is a sleep-staging model; this runs it off-label across subjects on BCI-IV-2a, PhysioNet-MI, Mumtaz2016 and
MentalArithmetic, as it already ran on SEED, SEED-V and STEW, so that its row of the rank grid (SUMMARY.md, Section 3) is complete.
Pipeline: CBraMod's released fine-tuning code (same LMDBs and subject splits, `Trainer`, checkpoint on the best validation κ or AUROC,
`Evaluator`) with only the model swapped. Model: braindecode 1.8.1's single-epoch `DeepSleepNet(n_chans, n_outputs, n_times,
sfreq=200)` with its default kernel sizes (0.25 s and 2 s at 200 Hz), as on SEED and STEW. Seeds 0–4. Shared code:
[eeg-conformer/cbramod-splits/code](../../../eeg-conformer/experiments/cbramod-splits/code) (`run_models.py --model bd-deepsleepnet`).

braindecode's DeepSleepNet feeds every channel's features into its recurrent layers, so its size grows with the channels: 110M
parameters on BCI-IV-2a's 22 channels, 303M on PhysioNet-MI's 64 (`code/smoke_test.py` there; FLOPs in `tools/flops.json`).

## Results

Mean ± SD over seeds 0–4 (`results/results.txt`); rank by mean balanced accuracy among the 8 models on these splits and Welch p against
the best (`python tools/compare_methods.py`, last table).

| Dataset | Balanced acc. | κ / AUROC | Weighted F1 / AUC-PR | Rank of 8 | Against the best |
|---|---|---|---|---|---|
| BCI-IV-2a | 0.5250 ± 0.0181 | κ 0.3667 ± 0.0241 | 0.4918 ± 0.0295 | 2 | EEGNet 0.5288, p = 0.85 (level) |
| PhysioNet-MI | 0.5877 ± 0.0067 | κ 0.4501 ± 0.0090 | 0.5892 ± 0.0066 | 5 | CBraMod 0.6265, p < 0.001 |
| Mumtaz2016 | 0.8391 ± 0.0072 | AUROC 0.9433 ± 0.0110 | AUC-PR 0.9473 ± 0.0083 | 8 | DGCNN 0.9619, p < 0.001 |
| MentalArithmetic | 0.5917 ± 0.0241 | AUROC 0.6779 ± 0.0396 | AUC-PR 0.3856 ± 0.0554 | 4 | ATCNet 0.6951, p = 0.028 |

- **Off-label, it is level with the best model on BCI-IV-2a** (0.525 against EEGNet's 0.529, p = 0.85) and above both foundation
  models there; it is last on Mumtaz2016.

## Paper vs. this reproduction

There is no paper number: DeepSleepNet's paper evaluates sleep staging only (MASS, Sleep-EDF; [sleepedf20-20fold](../sleepedf20-20fold)).

| | DeepSleepNet's paper and code | This experiment | Why |
|---|---|---|---|
| Implementation | TensorFlow, a BiLSTM across 25 epochs | braindecode's single-epoch version | the shared pipelines are PyTorch; one sample is one trial here |
| Input | one EEG channel, 30-s epochs at 100 Hz | every channel of CBraMod's 4–5-s samples at 200 Hz | off-label use on the shared samples |
| Training | two-step (CNN pre-training, then fine-tuning), final model kept | CBraMod's trainer, 50 epochs, best validation κ / AUROC | one trainer for all models |

Run configuration as in [eeg-conformer/cbramod-splits](../../../eeg-conformer/experiments/cbramod-splits).

## Reproducing

```bash
code/reproduce.sh collect    # results/ from logs/runs into results_rerun/, then diff
code/reproduce.sh train      # enqueue the runs for tools/run_queue.py; needs CBraMod's LMDBs
```

## Provenance

Runs: 2026-11-20 on mcl-server (`~/repro-gaps/runs/cbramod-splits`).
