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

## Not yet covered

LaBraM's own downstream benchmarks (TUAB, TUEV, SEED-V, MoBI) are access-gated or not yet downloaded.
