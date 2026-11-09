# braindecode's ATCNet on SEED and SEED-V, in CBraMod's SEED-V pipeline

braindecode's ATCNet through the pipeline every model shares on SEED and SEED-V: CBraMod's released SEED-V code (its loader, 1-s samples, trainer, checkpoint on the best validation κ and metric code), with only the model swapped. Data, splits, pipeline and run configuration are in [cbramod/finetune-seed](../../../cbramod/experiments/finetune-seed), which holds the shared code. Implementation: braindecode's ATCNet: braindecode 1.8.1's `ATCNet` (the released code is TensorFlow); on 1-s SEED samples `conv_block_pool_size_2=2` keeps the five sliding windows. Seeds 0–4 on each dataset and split.

No paper reports this model in this pipeline on these datasets: the comparison is between models. Each seed's score is the test score at the epoch with the best validation κ; mean ± SD over seeds (`results/results.txt`).
The clip vote is the accuracy of a majority vote over the 1-s predictions of each clip, the level at which SEED labels exist.

## Results

| Dataset | Split | Model | Runs | Bal. acc | κ | Weighted F1 | Accuracy | Clip vote |
|---|---|---|---|---|---|---|---|---|
| SEED | trial | braindecode's ATCNet | 5 | 0.4892 ± 0.0184 | 0.1989 ± 0.0265 | 0.4741 ± 0.0252 | 0.4917 ± 0.0173 | 0.5076 ± 0.0234 |
| SEED | subject | braindecode's ATCNet | 5 | 0.5410 ± 0.0057 | 0.3165 ± 0.0086 | 0.5186 ± 0.0083 | 0.5461 ± 0.0056 | 0.5881 ± 0.0285 |
| SEED-V | trial | braindecode's ATCNet | 5 | 0.2701 ± 0.0043 | 0.0861 ± 0.0040 | 0.2362 ± 0.0074 | 0.2897 ± 0.0026 | 0.2817 ± 0.0140 |
| SEED-V | subject | braindecode's ATCNet | 5 | 0.2528 ± 0.0071 | 0.0641 ± 0.0081 | 0.2324 ± 0.0100 | 0.2434 ± 0.0059 | 0.2696 ± 0.0249 |

Against the other models (`python tools/compare_seed.py`; Welch's t-test over seeds):

- **The weakest model on SEED-V's trial split:** balanced accuracy 0.2701 (κ 0.086), below EEGNet (0.2894) and EEG Conformer (0.3205); 5 classes, chance 0.20. On SEED's trial split 0.4892, level with EEGNet (0.4909, p = 0.87) and 0.047 below EEG Conformer (p = 0.003).
- **It underfits rather than overfits:** on SEED-V its training loss falls only from 1.60 to 1.42 over the 50 epochs (chance level ln 5 = 1.61; EEGNet 1.41, EEG Conformer 1.25, DGCNN 0.81, DeepSleepNet (bd) 0.55, CBraMod 0.40; seed 0, `log.txt`). CBraMod's settings (AdamW, learning rate 5e-4 for non-pre-trained parameters, weight decay 0.05, label smoothing 0.1) train the small CNNs slowly on 1-s windows; on Sleep-EDF's 30-s epochs, in CBraMod's sleep-staging trainer, ATCNet tied CBraMod.
- **Across subjects on SEED-V, 0.2528 (κ 0.064):** inside the 0.22–0.26 of every model, where the 5 classes are near chance.
- **Across subjects on SEED it loses nothing:** 0.5410, 0.052 above its trial split (p = 0.002) and level with CBraMod (0.5442, p = 0.53), second to LaBraM-Base (0.5822); EEGNet gains too (+0.032), while the foundation models, DGCNN and DeepSleepNet lose 0.05–0.09. A model that barely fits 1-s samples finds unseen clips as hard as unseen people.
- Runs took 12–21 h each on a Quadro M6000 (SEED-V subject split 12 h, SEED trial split 17–21 h), bound by kernel-launch latency (one run per GPU).

## Paper vs. this reproduction

Every difference from the released pipeline, and every setting chosen for all models, is listed once in [the shared README](../../../cbramod/experiments/finetune-seed).
Model-specific settings are in `../../../cbramod/experiments/finetune-seed/code/emo_models.py` and summarised above.
Hardware: Quadro M6000 (mcl-server), one run per GPU.

## Reproducing

`code/reproduce.sh collect` recomputes `results/` from `logs/runs` (numpy, scikit-learn); `code/reproduce.sh train` enqueues the runs.
`logs/runs/<run>/` holds `log.txt` (progress output removed), `metrics.json` (validation and test scores of every epoch) and
`test_preds.npz` (test labels and predictions at the selected epoch).

## Provenance

Runs: 2026-11-05 to 11-08, `~/repro-seed/runs` on mcl-server and mcl-server2.
