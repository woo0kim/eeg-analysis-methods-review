# Docs

```
docs/
├── adding-an-experiment.md          conventions for new methods and experiments
├── literature/                      literature investigations backing the reports (quotes, pages, links)
├── reports/<method>/                weekly reports and slides, one folder per method
│   └── YYYY-MM-DD_<method>_weekly-report.docx | _weekly-slides.pptx
├── reports/sleepedf/                the Sleep-EDF report, which covers every method on one dataset
├── reports/seed-stew/               the SEED, SEED-V and STEW report (every method), with DGCNN added
└── templates/                       blank weekly report template
```

File names start with the report date so they sort chronologically and stay identifiable when
downloaded on their own. The experiment folders hold the code and numbers; the reports are the
narrative that went to the weekly meeting.

## Reports

| Date | Method | Report | Slides | Experiments it covers |
|---|---|---|---|---|
| 2026-09-24 | EEG Conformer | [report](reports/eeg-conformer/2026-09-24_eeg-conformer_weekly-report.docx) | [slides](reports/eeg-conformer/2026-09-24_eeg-conformer_weekly-slides.pptx) | [bciciv2a-epoch-selection](../methods/eeg-conformer/experiments/bciciv2a-epoch-selection) |
| 2026-10-01 | CBraMod | [report](reports/cbramod/2026-10-01_cbramod_weekly-report.docx) | [slides](reports/cbramod/2026-10-01_cbramod_weekly-slides.pptx) | [finetune-public-datasets](../methods/cbramod/experiments/finetune-public-datasets), [LaBraM baseline](../methods/labram/experiments/finetune-cbramod-splits) |
| 2026-10-08 | LaBraM | [report](reports/labram/2026-10-08_labram_weekly-report.docx) | [slides](reports/labram/2026-10-08_labram_weekly-slides.pptx) | [finetune-cbramod-splits](../methods/labram/experiments/finetune-cbramod-splits), [review papers: reproduction vs. citation](literature/review-papers-reproduction-vs-citation.md) |
| 2026-10-22 | ATCNet; braindecode re-implementations of all methods | [report](reports/atcnet/2026-10-22_atcnet_weekly-report.docx) | [slides](reports/atcnet/2026-10-22_atcnet_weekly-slides.pptx) | [atcnet/bciciv2a-model-selection](../methods/atcnet/experiments/bciciv2a-model-selection), [atcnet/braindecode-bciciv2a](../methods/atcnet/experiments/braindecode-bciciv2a), [eeg-conformer/braindecode-bciciv2a](../methods/eeg-conformer/experiments/braindecode-bciciv2a), [cbramod/braindecode-public-datasets](../methods/cbramod/experiments/braindecode-public-datasets), [labram/braindecode-cbramod-splits](../methods/labram/experiments/braindecode-cbramod-splits), [eegnet/braindecode-cbramod-splits](../methods/eegnet/experiments/braindecode-cbramod-splits) |
| 2026-11-01 | Sleep-EDF: all seven methods; DeepSleepNet and L-SeqSleepNet added | [report](reports/sleepedf/2026-11-01_sleepedf_weekly-report.docx) | [slides](reports/sleepedf/2026-11-01_sleepedf_weekly-slides.pptx) | [cbramod/finetune-sleepedf](../methods/cbramod/experiments/finetune-sleepedf) (shared pipeline and common test set) and the `sleepedf-cbramod-pipeline` / `braindecode-sleepedf` experiments of each method; [deepsleepnet/sleepedf20-20fold](../methods/deepsleepnet/experiments/sleepedf20-20fold), [deepsleepnet/sleepedf78-cbramod-split](../methods/deepsleepnet/experiments/sleepedf78-cbramod-split), [l-seqsleepnet/sleepedf20-loso](../methods/l-seqsleepnet/experiments/sleepedf20-loso), [l-seqsleepnet/sleepedf78-cbramod-split](../methods/l-seqsleepnet/experiments/sleepedf78-cbramod-split) |
| 2026-11-08 | SEED, SEED-V and STEW: all methods; DGCNN added and run on every dataset | [report](reports/seed-stew/2026-11-08_seed-stew_weekly-report.docx) | [slides](reports/seed-stew/2026-11-08_seed-stew_weekly-slides.pptx) | [cbramod/finetune-seed](../methods/cbramod/experiments/finetune-seed) and [cbramod/finetune-stew](../methods/cbramod/experiments/finetune-stew) (shared pipelines) and the `seed-cbramod-pipeline`, `stew-cbramod-pipeline`, `braindecode-seed` and `braindecode-stew` experiments of each method; [dgcnn/seed-paper-protocol](../methods/dgcnn/experiments/seed-paper-protocol), [eeg-conformer/seed-5fold](../methods/eeg-conformer/experiments/seed-5fold), [dgcnn](../methods/dgcnn) (all its experiments), [l-seqsleepnet/seed-offlabel](../methods/l-seqsleepnet/experiments/seed-offlabel), [l-seqsleepnet/stew-offlabel](../methods/l-seqsleepnet/experiments/stew-offlabel) |

Notes on how the reports relate to the committed code:

- **EEG Conformer, 2026-09-24.** The report also covers a third protocol (training on scrambled labels,
  subjects 1–6) that was later removed from the code (`f7f0412` in woo0kim/EEG-Conformer), so its
  numbers are not in this repository.
- **CBraMod, 2026-10-01.** Revised on 2026-10-04 after LaBraM-Base, the paper's strongest baseline, was
  re-run under CBraMod's splits, seeds and scoring ([methods/labram](../methods/labram)). Report: a new
  section 3 (CBraMod vs. LaBraM-Base, both run on our end), a "LaBraM-Base: paper / mine" column in the
  main table, one more task bullet in section I, and the sentences and conclusion that relied on the
  paper's LaBraM numbers. Slides: a new slide 4 with the same comparison, and the slide-2 takeaway. The
  "Plan for next week" section was left as written.
- **LaBraM, 2026-10-08.** No new training: the numbers come from the 20 committed runs of
  [finetune-cbramod-splits](../methods/labram/experiments/finetune-cbramod-splits) (2026-10-04), plus
  `code/selection_analysis.py` and `code/make_figure.py`, which recompute them from the saved predictions. The
  TUAB/TUEV re-runs it quotes are from two published benchmarks (EEG-FM-Bench, AdaBrain-Bench), not ours. Section
  III.6 and slides 5–6 summarize [literature/review-papers-reproduction-vs-citation.md](literature/review-papers-reproduction-vs-citation.md).
  Both files were generated from the CBraMod report and deck as templates.
- **ATCNet and braindecode, 2026-10-22.** Covers the braindecode re-runs of the four earlier methods and the new
  ATCNet experiments; every number is in the experiment READMEs above. The report fills
  `templates/weekly-report-template.docx`; the deck reuses the title, closing slide and layout of the LaBraM deck
  (there is no blank slide template).
- **Sleep-EDF, 2026-11-01.** SEED was the requested dataset, but it is license-gated and not yet available, so the
  week went to Sleep-EDF Expanded, a public dataset on which all seven methods can run (the two sleep models only
  do sleep staging). Every number comes from the committed
  `results.json` files through `tools/compare_sleepedf.py` (common test set, pairwise tests, figure). The report and
  deck were generated from the ATCNet report and deck (2026-10-22) as templates.
- **SEED, SEED-V and STEW, 2026-11-08.** Written from the results available on Sunday afternoon, 2026-11-08, and revised
  on 2026-11-09 with ATCNet's SEED / SEED-V subject-split runs, which finished after the meeting, and with the DGCNN authors'
  own TensorFlow code, run unmodified once approved (section I says so). Every number comes from the committed
  `results.json` files through `tools/compare_seed.py`, `tools/compare_stew.py` and `tools/fig_seed_stew.py`. The report
  and deck were generated from the Sleep-EDF report and deck (2026-11-01) as templates. SEED and SEED-V are
  license-gated: neither file contains data from them, only results.

## Templates

`templates/weekly-report-template.docx` is the blank lab report template. There is no blank slide
template: the file called "Weekly Slide Template.pptx" in the original CBraMod folder is the EEG Conformer
deck, filed above as `2026-09-24_eeg-conformer_weekly-slides.pptx`.
