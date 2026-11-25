# Reproduction notes

The [README](../README.md) provides installation and run commands. The [figure guide](FIGURE_GUIDE.md) maps each result to its code and data.

## Model settings

The simulator uses 35 topics, 10,000 candidate items, ten recommendations per step, 5,000 interaction steps and 100-step consumption windows. The time step is 0.01, and the positive and negative feedback weights are 1 and −0.1.

The model reads four files from `data/model/inputs/`:

| File | Input |
|---|---|
| `phi_video_v2.npy` | Correlations between the 35 topics |
| `item_popularity_video_v2.npy` | Topic-level item frequencies |
| `user_alpha_mean_video_v3.npy` | 35 topic-level preference parameters |
| `age_behaviour_distribution.csv` | Behavioural-profile frequencies by age group |

`data/model/inputs/settings.json` documents the parameter values implemented in `code/model/run.py`.

### Behavioural parameters and interventions

Exploration score `e` and exploitation score `x` range from 0 to 9. Their calibrated mappings are:

```text
sigma0 = 0.018 + 0.0003458 * e
alpha0 = 5.6 + 0.1228 * x
beta0  = 7
```

Each intervention applies a normalized adjustment `delta` from 0 to 1:

| Intervention | Parameter rule |
|---|---|
| Weaker recommendation matching | `beta = 7 - delta` |
| Lower exploitation | `alpha = alpha0 - 1.228 * delta` |
| Higher exploration | `sigma = sigma0 + 0.003458 * delta` |

The 30% adjustment corresponds to `delta = 0.3` on this normalized scale. The rules above specify the corresponding change in each raw parameter.

The model classifies deep information cocoons using final-window relative entropy; the empirical analysis uses minimum relative entropy over follow-up. Their common numerical criterion is approximately 0.719032.

### Simulation outputs

New runs use seed 4 by default and sample behavioural profiles from the supplied age-group distributions. They generate new simulated populations. The saved results in `data/model/outputs/` provide the study's original numerical results for figure reproduction.

Each run writes a group-summary CSV and a JSON file containing its settings and elapsed runtime to `outputs/simulation/`. The documented parameter scan covers `beta = 1,...,9`. The saved grid also includes the historical `beta = 0` slice used in the Figure 4 cross-sections.

## Figure data and plotting conventions

- **Distributions:** CDFs use fixed grids, and box plots use the 5th, 25th, 50th, 75th and 95th group percentiles.
- **Mean error bars:** Reproduction plots use normal intervals computed from group SDs and sample sizes. Some original bar plots used unseeded bootstrap intervals. The assembled manuscript figures are available in `reference/figures/`.
- **Risk and regression intervals:** Risk plots use the interval methods documented in the data dictionary; regression plots use the saved estimates and intervals.
- **Figure 1b:** The blue curve is the minimum-entropy criterion used in the original plotting notebook. The orange curve is mean follow-up relative entropy among classified users. The conditional mean minimum is supplied as a separate column.
- **Figure 1h,i:** Saved plotting coordinates preserve the original age-axis transformation and the envelope constructed from intercept and age-coefficient bounds.
- **Calibration:** The saved 100-cell empirical grid is paired with its age-group distributions. Reconstructing deciles from the rounded research CSV moves three users between tied cells; figure reproduction uses the saved calibration grid.
- **Supplementary Figure S3:** The input values were digitized from archived pilot-sample figures.
- **Illustrations:** Figure 2f,g and Supplementary Figure S10 are supplied as authored vector graphics. Numerical panels are generated using portable fonts.

## Tables and analysis sources

`python code/reproduce.py --tables` exports the supplementary tables as CSV and Markdown files. `python code/export_tables.py` regenerates the table CSV files from the included LaTeX sources. Table S5 has three panels, giving 26 CSV files for 24 table numbers.

The original empirical analysis is documented in `reference/analysis/`. The author-side exporter, `code/prepare_aggregates.py`, prepares aggregate files from a research directory supplied with `--private-root`. The public figure commands read the aggregate files in `data/`.

File sources and schemas are recorded in `docs/data_provenance.json`, `docs/code_provenance.json` and `docs/data_schema.json`. To check the package and its numerical cross-references, run:

```sh
python code/verify_release.py
```
