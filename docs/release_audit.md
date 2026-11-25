# Release audit — 25 November 2026

The reviewed folder contains no empirical person-level tables, identifiers, linked behavioural records, individual longitudinal sequences, or saved per-agent simulation records. The review covers release inputs and generated outputs.

## Data and notebook review

- All 82 data files were reviewed: 77 aggregate CSV files, three topic-level numeric arrays and two settings/label files. Each tabular file now has explicit columns, row counts and unique aggregate grouping keys in `data_schema.json`.
- All six notebooks were inspected across 267 cells. Saved outputs, attachments, execution counts and cell metadata are empty. Source inspection found no pasted person-level tables, identifier lists or encoded data payloads.
- The original 653,486 participant identifiers were compared in memory against file contents and decoded PDF streams. No matches were found, and no identifiers were written to the audit files.
- The 68 PDF/Illustrator files have no embedded file attachments or form fields. The 43 generated PNG files contain only standard image metadata.
- Distribution data use fixed-grid group CDFs or group quantiles. Model input arrays describe 35 topics; the historical name `user_alpha_mean_video_v3.npy` does not indicate person-level contents.

## English comments and source preservation

Comments, Markdown explanations and printed diagnostic labels in six notebooks, plus one simulator comment, have been translated into English. No Chinese text remains in release text files. The topic-label table retains only English labels; its original-language label column has been removed from the release copy, without changing the original research file.

All 262 notebook code cells were compared with their source versions. Calculations and formatting expressions are unchanged; only comments and diagnostic text differ. Hash checks confirm that the 49 referenced original research files were not modified.

## README and reproduction mapping

The README and machine-readable index cover four main figures, 17 supplementary figures and 24 supplementary table numbers. Table S5 has three CSV panels, so there are 26 table CSV files.

All 21 main/supplementary plotting functions were executed while recording their actual input reads. All 128 checked file/function references in the README and linked figure, reproduction and data guides resolve. An independent table export reproduces all 26 CSV files byte for byte. The review corrected the following documentation details:

- Figure S11 also reads `model_initialization.csv` to map the topic matrix to display order.
- Quintile cut points, common-profile counts and editable artwork are now marked as supporting files when the plotting function does not read them directly.
- Figure 1d–g draws age risks and normalized gaps; risk ratios are provided in the associated threshold table.
- Table S21 includes stationary entropy in its displayed CSV. The unrounded `population_status.csv` contains persistence and depth summaries only.
- The model section now identifies the four actual input files and explains that `settings.json` documents parameters rather than configuring the runner.

`verify_release.py` checks the reviewed schemas, grouping keys, notebook payloads, English comments, code/function references, dependency map and release checksums. Generated table copies must match the public data; generated simulation CSVs must contain group counts and risks. Regression and mediation coefficients remain aggregate estimates: private individual-level models were not re-estimated during this audit.
