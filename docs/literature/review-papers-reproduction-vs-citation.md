# Do EEG review papers reproduce existing methods' results, or cite them?

Investigation for the 2026-10-08 weekly report ([LaBraM](../reports/labram/2026-10-08_labram_weekly-report.docx),
Section III.6). The question is narrow: when a review, survey or benchmark presents the performance of existing
methods, do its authors **re-run** the methods (reproduction) or **copy** the numbers the original papers
report (citation)?

**Method.** Each paper was downloaded from arXiv (version given below) and read in full. A verdict is recorded
only when a sentence, table or figure in the paper settles it; that evidence is quoted verbatim with its
location. Page numbers are PDF pages of the version read. Where a paper reprints another paper's numbers, the
digits were checked against the source table.

## Summary

| Group | Papers | Re-run methods | Copy reported numbers |
|---|---|---|---|
| Reviews and surveys without their own experiments | 6 | 0 | 6 (Lai et al. present no numbers and say they are incomparable) |
| Survey + benchmark | 1 | yes (benchmark part) | yes (survey part), kept separate |
| Method papers (the source of the numbers reviews copy) | 4 | BIOT | LaBraM, NeuroLM; CBraMod mixes both in one table |
| Benchmarks and re-evaluations | 10 | 10 | Gemein et al. also cite literature values, labelled as such |

1. **No review or survey without its own benchmark re-ran a method.** They quote or tabulate each paper's
   self-reported numbers, sometimes reading them off figures (Kuruppu et al.) or relabelling the metric (Zhou
   et al.). Roy et al. say so themselves: their pooled gain over baselines "might be biased positively".
2. **Re-running happens only in papers that present themselves as benchmarks**, and all ten of those re-run.
   None of them puts a method's re-run number next to the number the method's own paper reports. They rank
   models, so a reader cannot tell which published claims held up.
3. **Copied numbers travel without a label.** One set of TUAB/TUEV numbers passes through five papers
   ([below](#a-citation-chain-tuab-and-tuev)) without being re-run, and no later table marks it as copied.
4. **Where benchmarks did re-run LaBraM and CBraMod, the margins are smaller than reported**, as in our own runs
   ([methods/labram](../../methods/labram/experiments/finetune-cbramod-splits)).

Not classified because the full text is paywalled: Craik, He & Contreras-Vidal 2019 (J. Neural Eng. 16:031001),
Lotte et al. 2018 (J. Neural Eng. 15:031005), Altaheri et al. 2023 (Neural Comput. Appl. 35:14681), Yao et al.
2025 (J. Neural Eng. 22:061002; EEG-FM-Compass's Table 59 lists it as a review with a "unified comparison" on
6 datasets, but whether those numbers were re-run could not be checked).

## Reviews and surveys

**Roy et al., Deep learning-based electroencephalography analysis: a systematic review**, J. Neural Eng. 16:051001
(2019). [arXiv:1901.05498v2](https://arxiv.org/abs/1901.05498). *Cited (meta-analysis of self-reported results).*
- p. 21, Sec. 3.7.1: "Points of comparison are typically obtained in two different ways: 1) (re)implementing
  standard models or 2) referring to published models."
- p. 24, Fig. 14: "Difference in accuracy between each proposed DL model and corresponding baseline model for
  studies reporting accuracy". The median gain of 5.4% (p. 24) is pooled from the 102 studies' own numbers.
- p. 28, Sec. 4.5: "very few articles reported negative improvements, which could be explained by a publication
  bias towards positive results … the values that we report might be biased positively."
- p. 25, Sec. 3.8: "only 11 out of 156 studies (7%) could easily be reproduced using both the same data and code".

**Zhang et al., A survey on deep learning-based non-invasive brain signals: recent advances and new frontiers**,
J. Neural Eng. (2021). [arXiv:1905.04149v5](https://arxiv.org/abs/1905.04149). *Cited.*
- Results appear as quoted figures in the text, e.g. p. 7: "… achieved an accuracy of 86%". Table 4 (p. 8)
  groups studies by model type and lists references only.

**Gu et al., EEG-based brain-computer interfaces (BCIs): a survey of recent studies …**, IEEE/ACM TCBB (2021).
[arXiv:2001.11337v1](https://arxiv.org/abs/2001.11337). *Cited.*
- p. 11: "experiments with ESTCNN reached an accuracy rate of 97.37%". All performance figures are quoted from
  the surveyed studies.

**Kuruppu et al., EEG foundation models: a critical review of current progress and future directions** (2025).
[arXiv:2507.11783v3](https://arxiv.org/abs/2507.11783). *Cited, partly read off figures.*
- p. 14, Fig. 3: "In Figure 3a-3b, we compare the performances reported on these common tasks."
- p. 35, Suppl. Table 2: "Evaluation scores estimated based on the graphics in the paper are denoted using
  superscript∗."
- p. 15 notes the comparison holds "notwithstanding potential confounders such as preprocessing, data splits,
  and model adaptation strategies".

**Zhou et al., Brain foundation models: a survey on advancements in neural signal processing and brain
discovery** (2025). [arXiv:2503.00580v2](https://arxiv.org/abs/2503.00580). *Cited, metric relabelled.*
- p. 19, Table III ("ACC (Accuracy)"): LaBraM TUAB 0.8140 ± 0.0019 / AUROC 0.9022 ± 0.0009, TUEV
  0.6409 ± 0.0065 / F1 0.8312 ± 0.0052 are LaBraM's own Tables 1–2, where 0.8140 and 0.6409 are *balanced*
  accuracy. The NeuroLM column (0.7969, 0.7884, 0.4679, 0.7359) and LaBraM's SEED value (0.7318 ± 0.0019) are
  NeuroLM's Tables 2–3 (NeuroLM-XL and NeuroLM's own LaBraM-Base run).

**Lai et al., A simple review of EEG foundation models: datasets, advancements and future perspectives** (2025).
[arXiv:2504.20069v2](https://arxiv.org/abs/2504.20069). *No performance numbers.*
- Tables I–IX cover datasets, preprocessing, parameter counts, objectives and hardware. p. 10: "the datasets,
  preprocessing pipelines, and data partitioning strategies … used can vary, making performance results
  incomparable."

**EEG-FM-Compass: progress, benchmarking, and future directions for EEG foundation models**, Natl. Sci. Rev.
(2026). [arXiv:2601.17883v3](https://arxiv.org/abs/2601.17883). *Survey cites; benchmark re-runs; kept apart.*
- p. 4, Table 1: "'—' indicates that the corresponding information was not reported in the original paper"
  (survey part).
- Abstract: "We first review 55 representative models … We then evaluate 12 open source FMs and competitive
  specialist baselines across 13 EEG datasets" (benchmark part).
- p. 86, Suppl. Table 59 compares itself with other surveys and marks Lai et al., Zhou et al., Kuruppu et al.,
  Altaheri et al. 2025, Gu et al. 2025, Lee et al. 2025 and Kwon & Shin 2026 as providing no benchmark
  experiments ("–").

## Method papers: where the reprinted numbers come from

**BIOT** (Yang, Westover & Sun), NeurIPS 2023. [arXiv:2305.10351v1](https://arxiv.org/abs/2305.10351).
*Reproduced (baselines re-implemented).*
- p. 15, App. A.2: "the SPaRCNet code is requested from the authors …, the ContraWR code is downloaded and
  modified upon the github, CNN-Transformer is easily implemented following the Fig. 3 of the original paper".
- p. 7: "we obtain five sets of results with different random seeds". Results: p. 16, Tables 4 (TUAB) and 5
  (TUEV). Input: "16 montages" (p. 6, Table 1).

**LaBraM** (Jiang, Zhao & Lu), ICLR 2024. [arXiv:2405.18765v1](https://arxiv.org/abs/2405.18765). *Cited.*
- p. 7, Sec. 3.2: "The baselines are from Yang et al. (2023a), where we choose the best results to compare with."
- p. 8, Tables 1–2: every baseline cell equals BIOT Tables 4–5. The BIOT row is the best of 4 TUAB and 13 TUEV
  BIOT variants: "Pre-trained BIOT (6 EEG datasets)" for TUAB (best AUROC; the PREST+SHHS variant has the higher
  balanced accuracy, 0.8019 ± 0.0021) and "Pre-trained BIOT (ultimate)" for TUEV.
- LaBraM's own TUEV inputs differ from BIOT's: `dataset_maker/make_TUEV.py` line 157 comments out the bipolar
  conversion it inherited from BIOT (`#signals = convert_signals(signals, Rawdata)`), so LaBraM uses 23
  monopolar channels.

**NeuroLM** (Jiang, Wang, Lu & Li), ICLR 2025. [arXiv:2409.00101v3](https://arxiv.org/abs/2409.00101).
*Cited, unmarked.*
- p. 7: "The average and standard deviation values are reported using three random seeds".
- p. 8, Table 2: the TUAB/TUEV rows for SPaRCNet … BIOT and LaBraM-Base are LaBraM's Tables 1–2, which are
  five-seed results.

**CBraMod** (Wang et al.), ICLR 2025. [arXiv:2412.07236v6](https://arxiv.org/abs/2412.07236).
*Mixed in one table, unmarked.*
- p. 7, Sec. 3.2: "We re-implemented above baselines based on the public code provided by BIOT … unless their
  experimental results have already been reported in existing studies. … We fine-tune BIOT and LaBraM based on
  their public code and pre-trained weights, unless their experimental results have already been reported in
  the original papers."
- p. 26, Tables 13 (TUEV) and 14 (TUAB): the EEGNet and EEG Conformer rows are new; the other nine rows,
  including LaBraM-Large and LaBraM-Huge, are copied from LaBraM, with nothing in the tables marking which is
  which. CBraMod's own TUEV runs use "the common 16 bipolar montage channels" and a 60 Hz notch (p. 26), while
  the copied LaBraM rows come from 23 monopolar channels and a 50 Hz notch. On its other datasets (e.g. Tables
  3, 10, 12, 15) CBraMod ran LaBraM itself; our re-run of those numbers is in
  [methods/labram](../../methods/labram/experiments/finetune-cbramod-splits).

## Benchmarks and re-evaluations

| Paper | What was re-run | Evidence |
|---|---|---|
| Jayaram & Barachant, **MOABB**, J. Neural Eng. 15:066011 (2018), [arXiv:1805.06427v1](https://arxiv.org/abs/1805.06427) | BCI pipelines on 12 open datasets | p. 4: "the world of CSP regularization literature does not appear to have the effect that was originally claimed" |
| Chevallier et al., **The largest EEG-based BCI reproducibility study for open science: the MOABB benchmark** (2024), [arXiv:2404.15319v1](https://arxiv.org/abs/2404.15319) | 30 pipelines, 36 datasets | Abstract: "30 machine learning pipelines … are meticulously re-implemented and evaluated across 36 publicly available datasets"; p. 6: "reporting classification performances on EEG-based BCI tasks is not standardized" |
| Gemein et al., **Machine-learning-based diagnostics of EEG pathology**, NeuroImage 220:117021 (2020), [arXiv:2002.05115v1](https://arxiv.org/abs/2002.05115) | Feature-based, Riemannian and Braindecode models on TUAB; literature values shown separately | p. 5, Table 1: related works' reported accuracies; p. 15: own results compared "to the result of 86.57% accuracy previously reported using ChronoNet" (re-run and cited kept apart) |
| Rommel et al., **Data augmentation for learning predictive models on EEG: a systematic comparison**, J. Neural Eng. (2022), [arXiv:2206.14483v2](https://arxiv.org/abs/2206.14483) | 13 augmentations, two tasks | Abstract: augmentations' "positive impact on performance is often evaluated on a single dataset … We compare quantitatively 13 different augmentations" |
| Lee et al., **Are large brainwave foundation models capable yet? Insights from fine-tuning** (2025), [arXiv:2507.01196v1](https://arxiv.org/abs/2507.01196) | LaBraM, NeuroGPT vs. EEGNet, EEG-Inception | p. 4, Table 1: mean accuracy LaBraM 0.742 (5,854,288 params), EEGNet 0.731 (2,394), EEG-Inception 0.733; abstract: "only marginal improvements (0.9%-1.2%)" |
| Wu et al., **AdaBrain-Bench** (2025), [arXiv:2507.09882v2](https://arxiv.org/abs/2507.09882) | 8 models, 13 datasets | p. 5: "a unified foundation model adaptation pipeline, integrating uniform data splitting, standardized data preprocessing"; p. 6, Table 2: LaBraM TUAB 81.50, TUEV 59.05 BA; macro-average LaBraM 64.61 vs. CBraMod 62.66 |
| Xiong et al., **EEG-FM-Bench** (2025), [arXiv:2508.17742v3](https://arxiv.org/abs/2508.17742) | 7 FMs, 14 datasets | Abstract: "under released checkpoints and a matched downstream protocol"; p. 22, Table 2: LaBraM TUAB 79.50 ± 0.81, TUEV 59.58 ± 4.31 BA |
| Kastrati et al., **EEG-Bench** (2025), [arXiv:2512.08959v1](https://arxiv.org/abs/2512.08959) | 3 FMs + classical baselines, 14 clinical datasets | p. 4: "We use the publicly released LaBraM-Base and fine-tune it end-to-end"; abstract: "simpler models often remain competitive" |
| Shen et al., **Brain4FMs** (2026), [arXiv:2602.11558v2](https://arxiv.org/abs/2602.11558) | 17 FMs, 21 EEG/iEEG datasets | p. 2: "We include BFMs from peer-reviewed studies or widely cited preprints with publicly available code and pretrained checkpoints. All models use the same dataset-specific preprocessing pipeline" |
| Lu et al., **OmniEEG-Bench** (2026), [arXiv:2606.00815v1](https://arxiv.org/abs/2606.00815) | 10 FMs | p. 16: "Models were included if they provide accessible pretrained weights or reproducible implementations" |

None of these ten papers has a table putting a method's re-run number next to the number its own paper reports
(searched for "originally / previously reported", "original paper", "reported in" and read the results
sections).

## A citation chain: TUAB and TUEV

The same digits, table by table. "–" means the paper does not print that row.

| Row | BIOT T4/T5 (2023) | LaBraM T1/T2 (2024) | NeuroLM T2 (2025) | CBraMod T13/T14 (2025) | Zhou et al. T III (2025) |
|---|---|---|---|---|---|
| SPaRCNet, TUAB BA | 0.7896 ± 0.0018 (re-implemented) | 0.7896 ± 0.0018 | 0.7896 ± 0.0018 | 0.7896 ± 0.0018 | – |
| SPaRCNet, TUEV BA | 0.4161 ± 0.0262 (re-implemented) | 0.4161 ± 0.0262 | 0.4161 ± 0.0262 | 0.4161 ± 0.0262 | – |
| BIOT, TUAB BA | 0.7959 ± 0.0057 ("6 EEG datasets") | 0.7959 ± 0.0057 | 0.7959 ± 0.0057 | 0.7959 ± 0.0057 | – |
| BIOT, TUEV BA | 0.5281 ± 0.0225 ("ultimate") | 0.5281 ± 0.0225 | 0.5281 ± 0.0225 | 0.5281 ± 0.0225 | – |
| LaBraM-Base, TUAB BA | – | 0.8140 ± 0.0019 (own runs) | 0.8140 ± 0.0019 | 0.8140 ± 0.0019 | 0.8140 ± 0.0019 ("ACC") |
| LaBraM-Base, TUEV BA | – | 0.6409 ± 0.0065 (own runs) | 0.6409 ± 0.0065 | 0.6409 ± 0.0065 | 0.6409 ± 0.0065 ("ACC") |
| LaBraM-Huge, TUEV BA | – | 0.6616 ± 0.0170 (weights not released) | – | 0.6616 ± 0.0170 | – |

Re-run from the released LaBraM-Base checkpoint by the two benchmarks that report TUAB/TUEV: TUAB 0.7950 ±
0.0081 (EEG-FM-Bench) and 0.8150 (AdaBrain-Bench); TUEV 0.5958 ± 0.0431 and 0.5905. Both benchmarks use their own
preprocessing and splits, so they support, but do not replace, a controlled re-run on LaBraM's own splits.

## What this means for our review

- A **paper vs. re-run** comparison per method is new. Reviews only tabulate claims, and benchmarks only rank
  re-runs.
- Every number in our tables should say whether we re-ran it or cited it, and if cited, give the paper and table.
  This repository already does this (`RESULTS.md`: "Paper" vs. "Reproduced" columns with the source table).
