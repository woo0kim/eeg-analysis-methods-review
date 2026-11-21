# ATCNet (braindecode) on CBraMod's four public datasets

ATCNet across subjects on BCI-IV-2a, PhysioNet-MI, Mumtaz2016 and MentalArithmetic, through CBraMod's released fine-tuning pipeline: the
same LMDBs and subject splits, `Trainer`, loss, optimizer, schedule, checkpoint rule (best validation κ or AUROC) and `Evaluator` as
every other model run on these splits. The released ATCNet is Keras, so the model is braindecode 1.8.1's `ATCNet(n_chans, n_outputs,
n_times, sfreq=200)` with its defaults, as on Sleep-EDF, SEED and STEW: 800 and 1,000 samples are above its 616-sample minimum, so
nothing is rescaled (5 sliding windows). Seeds 0–4. The code is shared:
[eeg-conformer/cbramod-splits/code](../../../eeg-conformer/experiments/cbramod-splits/code) (`run_models.py --model bd-atcnet`).

ATCNet's paper has no cross-subject results on these datasets, so there is no paper number to reproduce; this experiment fills its
row of the rank grid (SUMMARY.md, Section 3).

## Results

Mean ± SD over seeds 0–4 (`results/results.txt`); rank by mean balanced accuracy among the 8 models on these splits, and Welch p
against the best (`python tools/compare_methods.py`, last table).

| Dataset | Balanced acc. | κ / AUROC | Weighted F1 / AUC-PR | Rank of 8 | Against the best |
|---|---|---|---|---|---|
| BCI-IV-2a | 0.4898 ± 0.0293 | κ 0.3197 ± 0.0391 | 0.4579 ± 0.0381 | 4 | EEGNet 0.5288, p = 0.12; above both foundation models by mean |
| PhysioNet-MI | 0.6142 ± 0.0049 | κ 0.4856 ± 0.0065 | 0.6153 ± 0.0052 | 2 | CBraMod 0.6265, p = 0.006; above LaBraM-Base by mean |
| Mumtaz2016 | 0.9077 ± 0.0163 | AUROC 0.9599 ± 0.0075 | AUC-PR 0.9664 ± 0.0097 | 3 | DGCNN 0.9619, p < 0.001 |
| MentalArithmetic | **0.6951 ± 0.0709** | AUROC 0.7864 ± 0.0431 | AUC-PR 0.6300 ± 0.1017 | 1 | LaBraM-Base 0.6562, p = 0.32 (level) |

- **A 114K-parameter network is level with the foundation models across subjects.** Second on PhysioNet-MI (behind CBraMod only,
  0.012), first by mean on MentalArithmetic (level with LaBraM-Base, p = 0.32), third on Mumtaz2016, and within noise of the best on
  BCI-IV-2a.
- On SEED's 1-s windows the same model underfits ([braindecode-seed](../braindecode-seed)); on these 4–5-s trials it trains normally.

## Paper vs. this reproduction

### What differs, and why

| | ATCNet's paper and code | This experiment | Why |
|---|---|---|---|
| Implementation | Keras (TensorFlow) | braindecode 1.8.1 `ATCNet` | the shared pipelines are PyTorch; its final max-norm axis differs from Keras' ([braindecode-bciciv2a](../braindecode-bciciv2a)) |
| Data | BCI-IV-2a, 4.5-s unfiltered trials at 250 Hz, z-scored | CBraMod's samples: 4–5 s at 200 Hz | the pipeline every model shares |
| Protocol | within-subject and leave one subject out | CBraMod's subject splits | subject-independent, identical for every model |
| Training | Adam 1e-3, up to 1,000 epochs, checkpoint on the test set | CBraMod's trainer, 50 epochs, best validation κ / AUROC | one trainer for all models; no test-set selection |

### Settings we had to choose

| Setting | Value | Why |
|---|---|---|
| Model arguments | braindecode's defaults with `sfreq=200` | as on Sleep-EDF and STEW; no rescaling needed at these lengths |
| Learning rate | 5e-4 for the whole model | no `backbone` parameters, as EEGNet and DGCNN on these splits |

### Run configuration

As [eeg-conformer/cbramod-splits](../../../eeg-conformer/experiments/cbramod-splits): `finetune_main.py`'s defaults, 50 epochs, batch 64,
AdamW, PyTorch 2.1.2, braindecode 1.8.1, Quadro M6000.

## Reproducing

```bash
code/reproduce.sh collect    # results/ from logs/runs into results_rerun/, then diff
code/reproduce.sh train      # enqueue the runs for tools/run_queue.py; needs CBraMod's LMDBs
```

## Provenance

Runs: 2026-11-20 on mcl-server (`~/repro-gaps/runs/cbramod-splits`).
