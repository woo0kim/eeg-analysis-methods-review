# Docs

```
docs/
├── adding-an-experiment.md          conventions for new methods and experiments
├── reports/<method>/                weekly reports and slides, one folder per method
│   └── YYYY-MM-DD_<method>_weekly-report.docx | _weekly-slides.pptx
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

## Templates

`templates/weekly-report-template.docx` is the blank lab report template. There is no blank slide
template: the file called "Weekly Slide Template.pptx" in the original CBraMod folder is the EEG Conformer
deck, filed above as `2026-09-24_eeg-conformer_weekly-slides.pptx`.
