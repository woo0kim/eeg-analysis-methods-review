# EEG Conformer

> Y. Song, Q. Zheng, B. Liu, X. Gao, *EEG Conformer: Convolutional Transformer for EEG Decoding and
> Visualization*, IEEE TNSRE 31:710–719, 2023. doi:[10.1109/TNSRE.2022.3230250](https://doi.org/10.1109/TNSRE.2022.3230250)

- Released code: [eeyhsong/EEG-Conformer](https://github.com/eeyhsong/EEG-Conformer), mirrored in
  [woo0kim/EEG-Conformer](https://github.com/woo0kim/EEG-Conformer)
- `upstream/`: git submodule pinned at `9ae149b` (2024-07-24), the code that was evaluated
- License of the released code: GPL-3.0. `experiments/*/code/run_validation.py` copies parts of
  `conformer.py` and is GPL-3.0 as well.

## Experiments

| Experiment | Dataset | Question | Finding |
|---|---|---|---|
| [bciciv2a-epoch-selection](experiments/bciciv2a-epoch-selection) | BCI IV 2a | How much of the reported accuracy comes from picking the best of 2000 test-set evaluations? | The released rule reproduces the paper (78.72% vs. 78.66%), but a held-out validation split gives 70.01% and the last-100-epoch mean 68.77%. |

## Not yet covered

The paper also reports BCI IV 2b and SEED (`upstream/conformer_BCIIV2b.py`,
`upstream/conformer_seed_1s_5fold.py`).
