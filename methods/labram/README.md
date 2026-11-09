# LaBraM

> W.-B. Jiang, L.-M. Zhao, B.-L. Lu, *Large Brain Model for Learning Generic Representations with
> Tremendous EEG Data in BCI*, ICLR 2024. [OpenReview](https://openreview.net/forum?id=QzTpTRVtrP)

- Released code: [935963004/LaBraM](https://github.com/935963004/LaBraM)
- Released weights: `upstream/checkpoints/labram-base.pth` (LaBraM-Base, 5.8M parameters, md5 808a646a…),
  shipped inside the repository. LaBraM-Large and -Huge weights are not public.
- `upstream/`: git submodule pinned at `c431221` (2025-09-29)
- License of the released code: MIT

## Experiments

| Experiment | Datasets | Question | Finding |
|---|---|---|---|
| [finetune-cbramod-splits](experiments/finetune-cbramod-splits) | PhysioNet-MI, BCIC-IV-2a, Mumtaz2016, MentalArithmetic | How does LaBraM-Base do under exactly the data, splits, seeds and scoring of our CBraMod runs? This is the comparison the CBraMod paper makes with LaBraM's numbers from its own runs. | LaBraM-Base reproduces the CBraMod paper's LaBraM numbers on 10 of 12 metrics. Under identical conditions CBraMod is better only on PhysioNet-MI; the two tie on BCIC-IV-2a and Mumtaz2016, and LaBraM is better on MentalArithmetic (balanced accuracy +0.067, p = 0.015). |
| [braindecode-cbramod-splits](experiments/braindecode-cbramod-splits) | same four | Does braindecode's `Labram` (braindecode 1.8.1, its pre-trained weights), fine-tuned by the same released script, reproduce those numbers? | Yes. Its weights are the released checkpoint's and its forward pass is identical to the released model's (logit difference 0). None of the 12 metrics differs from the released-code runs (p ≥ 0.12); 9 of 12 are within noise of the CBraMod paper's LaBraM numbers. braindecode's `from_pretrained()` cannot load the weights for 4–5-s windows, and its model lacks the `no_weight_decay()` LaBraM's optimizer calls; both are handled in the runner. |
| [sleepedf-cbramod-pipeline](experiments/sleepedf-cbramod-pipeline) | Sleep-EDF-78 | How does the released LaBraM-Base do as the epoch encoder of CBraMod's sleep-staging pipeline, against every other model under the same conditions? | It has the highest κ of the original implementations (0.773 on the common test set; braindecode's LaBraM 0.777) and, with braindecode's LaBraM, the best N1 F1 of the pipeline models, but it is within noise of CBraMod (+0.005, p = 0.44) and of L-SeqSleepNet (p = 0.96). It is 0.028 κ above CBraMod without pre-training (p = 0.001). LaBraM's paper has no sleep data. |
| [braindecode-sleepedf](experiments/braindecode-sleepedf) | Sleep-EDF-78 | Does braindecode's `Labram`, in the same pipeline, give the released model's numbers? | Yes. Its encoder gives the released encoder's features on Sleep-EDF input (max difference 0), so only random draws differ. κ 0.777 vs. 0.773 (p = 0.36), balanced accuracy +0.006 (p = 0.56), no stage differs significantly. It has the highest κ and balanced accuracy of all Sleep-EDF models, within noise of the released LaBraM, CBraMod and L-SeqSleepNet. |
| [seed-cbramod-pipeline](experiments/seed-cbramod-pipeline) | SEED, SEED-V | LaBraM-Base in the shared SEED pipeline, with its own recipe | The best model on both datasets' trial split and on SEED across subjects; on SEED-V above its number in the CBraMod paper (κ p = 0.041) and above CBraMod (p = 0.011). At chance across subjects on SEED-V |
| [braindecode-seed](experiments/braindecode-seed) | SEED, SEED-V | Does braindecode's `Labram` match? | Yes (−0.014, p = 0.061; −0.003, p = 0.57) |
| [stew-cbramod-pipeline](experiments/stew-cbramod-pipeline) | STEW | LaBraM-Base on STEW | The best model on rest vs. task (0.818); tied with CBraMod on 3 levels |
| [braindecode-stew](experiments/braindecode-stew) | STEW | Does braindecode's `Labram` match? | Yes on both tasks (rest vs. task −0.003, p = 0.75; 3 levels +0.018, p = 0.077) |

## Not yet covered

LaBraM's own downstream benchmarks: TUAB and TUEV are access-gated, MoBI is not yet downloaded, and its SEED-V result uses a
20-subject version that is not public (the public SEED-V has 16; CBraMod paper, footnote 1).
