# dDTF-GL on Sleep-EDF-78

dDTF-GL on the 30-s epochs of the shared Sleep-EDF pipeline ([cbramod/finetune-sleepedf](../../../cbramod/experiments/finetune-sleepedf):
Fpz-Cz and Pz-Oz at 200 Hz, 0.3–35 Hz, the age-stratified 62 / 8 / 8 subject split), scored on the same test epochs as every other
model, including the common test set of 20,976 epochs. Seeds 0–4. The paper has no sleep data; this asks how the method transfers.

**Not the same task as the other pipeline models.** They classify each epoch inside a 20-epoch sequence, through the released
Transformer head; dDTF-GL classifies every epoch on its own, as the method has no sequence model. With two channels a window gives
2 × 2 × 5 = 20 dDTF values (two self-flows and two cross-flows per band; partial coherence of two channels is their coherence).

## Results

Common test set (20,976 epochs; `python ../../../../tools/compare_sleepedf.py`), mean ± SD over seeds 0–4:

| Model | Training | κ | Balanced acc. | Macro-F1 | Accuracy | N1 F1 | N3 F1 | REM F1 |
|---|---|---|---|---|---|---|---|---|
| dDTF-GL | each epoch alone | 0.5595 ± 0.0006 | 0.5219 ± 0.0012 | 0.5229 ± 0.0013 | 0.6948 ± 0.0003 | 0.086 | 0.521 | 0.414 |
| dDTF-GL, first map settings | each epoch alone | 0.5682 ± 0.0062 | 0.5267 ± 0.0042 | 0.5281 ± 0.0037 | 0.7011 ± 0.0042 | 0.061 | 0.553 | 0.420 |
| DGCNN, braindecode, raw input (weakest network) | CBraMod pipeline | 0.5956 ± 0.0192 | 0.6017 ± 0.0260 | 0.6059 ± 0.0175 | 0.7114 ± 0.0146 | | | |
| EEGNet | CBraMod pipeline | 0.6815 ± 0.0154 | 0.6852 ± 0.0137 | 0.6785 ± 0.0073 | 0.7719 ± 0.0126 | 0.311 | 0.787 | 0.594 |
| LaBraM-Base (best) | CBraMod pipeline | 0.7727 ± 0.0085 | 0.7612 ± 0.0184 | 0.7619 ± 0.0153 | 0.8381 ± 0.0056 | 0.434 | 0.805 | 0.785 |

- **dDTF-GL is the weakest model on Sleep-EDF:** 0.122 κ below EEGNet and 0.213 below LaBraM-Base (both p < 0.001). Wake is
  recognised (F1 0.87), N1 almost never (0.09).
- **The map settings hardly matter here:** the calibrated maps (two bipolar channels, so no average reference; only the
  √(F² · |C|) form changes) score 0.009 κ below the first settings.
- **Little is left to select from:** DFT keeps 10 of the 20 values and AXB 5, LNT adds 2, so the classifier sees 7 numbers per epoch.
  The method's selection stages assume many redundant features; two channels give few.
- On the pipeline's own test epochs (21,080): κ 0.5590 ± 0.0006, balanced accuracy 0.5216 ± 0.0012, weighted F1 0.6701 ± 0.0006
  (`results/results.txt`).

## Paper vs. this reproduction

Not a reproduction: the paper has no sleep result. The method's settings are in [../../README.md](../../README.md#the-method-as-run-here)
(MVAR order 9 by AIC on 300 training epochs; no re-reference: the two channels are bipolar). Differences from the other models in the pipeline:

| | Every other pipeline model | dDTF-GL | Why |
|---|---|---|---|
| Context | 20-epoch sequences through the released Transformer head | each 30-s epoch alone | the method has no sequence model; adding one would no longer be the method |
| Model selection | best validation κ | none; the validation subjects are scored only | no epochs |
| Training data | the 62 training subjects' sequences | the same epochs (153,100) | – |

Run configuration: 10 threads; environment [../../env/conda_list.txt](../../env/conda_list.txt); mcl-server (2× Xeon E5-2620 v3),
about one minute per run.

## Reproducing

```bash
code/reproduce.sh collect                            # results/ from logs/runs (checks the test order against the pipeline's)
python ../../../../tools/compare_sleepedf.py         # every model on the common test set
code/reproduce.sh train                              # needs the pipeline's processed_seq20 (finetune-sleepedf README)
```

## Provenance

Runs: 2026-11-11, `~/repro-gl/runs/sleepedf-cbramod-pipeline` on mcl-server.
