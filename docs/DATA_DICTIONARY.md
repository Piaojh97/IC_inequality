# Aggregate data dictionary

This package contains group statistics and model inputs. No row represents an empirical individual, an individual interaction, or a person's sequence of observations. Saved model outputs are aggregated too. `file_inventory.csv` lists every public data file, its columns and its row count.

## Shared fields

| Field | Meaning / unit |
|---|---|
| `age` | One of six age categories: 19–23, 24–30, 31–40, 41–50, 51–60 and 61–79 |
| `n`, `count`, `cases` | Number of observations in a statistical group, cell or bin; `cases` counts deep ICs |
| `risk`, `proportion`, `cdf` | Fractions on a 0–1 scale unless the header explicitly says percent |
| `percent`, `share (%)`, fields ending `_pp` | Percentages or percentage-point differences, as indicated |
| `deep` | Group label: 1 for deep IC, 0 for other users; never an individual outcome record |
| `alpha` in empirical threshold files | One-sided null tail probability used to define the deep-IC criterion |
| `criterion` | Relative-entropy boundary: `1 + norm.ppf(alpha) * cohort_SD(minimum_relative_entropy)` |
| `alpha` in model files | Exploitation intensity; distinct from the empirical tail probability |
| `beta`, `sigma` | Recommendation-matching strength and exploration intensity |
| `exploration`, `exploitation` | Integer behavioural scores from 0 to 9; each row is a group or grid cell |
| `delta` | Normalized intervention dose from 0 to 1; see [reproduction notes](REPRODUCTION.md) for parameter-specific conversion |
| `mean`, `sd`, `se` | Group mean, sample SD and standard error of the mean |
| `lo`, `hi` | Lower and upper interval endpoints; interval methods are documented per file family |
| `q05`, `q25`, `median`, `q75`, `q95` | Group percentiles used for box plots; no individual fliers are provided |

## Empirical figure data

- `min_entropy_histogram.csv`: 100 bins. `density = count / (cohort_n * bin_width)`. It contains a histogram, not individual minimum-entropy values.
- `thresholds.csv`: one row per tail probability. `disparity` is the oldest-minus-youngest risk difference divided by cohort risk at the same threshold. `risk_ratio` is oldest risk divided by youngest risk.
- `age_risks.csv`: one row per age category and tail probability. Intervals use the normal binomial approximation, as in the main plotting notebook.
- `threshold_depth.csv`: the original Figure 1b plotting quantities. `criterion` is the boundary, `conditional_mean_average` is mean follow-up relative entropy among classified users, and `conditional_mean_minimum` is separately named to prevent confusing it with the criterion curve.
- `persistence_by_status.csv`: two group summaries. Persistence is each person's fraction of observed follow-up windows below relative entropy 1, then averaged across the group. No person-level proportions or sequences are published. The t statistic is recovered from group means, SDs and counts. A stored p value of zero reflects floating-point underflow.
- `behaviour_by_age_status.csv`: age-by-status summaries of searches per hour (`watched_search1`), interactions per hour (`activeness`), mean duration on interested videos (`attention_span1`, seconds), and duration on the longest-watched tenth (`top10avg`, seconds). Mean intervals are deterministic normal approximations.
- `boxplot_statistics.csv`: group quantiles for the four behaviours and first-hour entropy, using the same 5th/95th-percentile whiskers as the supplement.
- `distribution_cdfs.csv`: cumulative counts at fixed grids. The minimum-entropy grid is 0–2 in steps of 0.005; content-share grids are 0–1 in steps of 0.005. Values outside a displayed grid are not released as individual endpoints.
- `indicator_rank_pairs.csv`: two 10×10 contingency tables, normalized by the full cohort count. Each cell contains a count and percentage, not linked records.

## Saved statistical estimates

- `regression_all_specifications.csv` contains the five models for each outcome. Logistic estimates are on the log-odds scale; exponentiate coefficients and interval endpoints for odds ratios. Linear estimates are in relative-entropy units. The coefficient intervals use robust standard errors.
- `regression_curves.csv` contains already evaluated plotting coordinates. It retains the original age-axis transformation and the envelope constructed from intercept/age coefficient bounds. It does not contain individual fitted probabilities or residuals.
- `regression_paths.csv` contains coefficients for the exploration-score model, exploitation-score model and deep-IC model. No design matrix is included.
- `joint_mediation_risks.csv` contains five standardized population risks. Age codes 0 and 5 denote the youngest and oldest categories. The three digits select the age setting for the outcome, exploration and exploitation equations.
- `joint_mediation.csv` reports standardized differences as fractions, contributions to the total gap, and intervals. Multiplying risk differences by 100 gives percentage points. The joint share is approximately 0.6931.
- `fig_s2_groups.csv` uses Wilson intervals for the supplementary group risk bars. Other interval methods are not substituted for these saved values.
- `fig_s1_hourly_digitized.csv` contains aggregate pilot-figure readings. Its provenance is digitization, not access to an individual hourly dataset.

## Model inputs and outputs

- Topic input tables contain 35 topics. `model_topic_correlations.csv` is a 35×35 matrix with row labels. The NPY inputs are numeric arrays of topic probabilities, correlations and Dirichlet inputs, not user matrices.
- `age_behaviour_distribution.csv` contains 600 age-by-exploration-by-exploitation cells. `proportion` sums to one per age group, and `count` sums to 653,486 across groups. These are the saved calibration-aligned distributions.
- `phase_grid.csv` has axes beta, exploitation and exploration; `risk` is the saved cell-level deep-IC fraction. The documented scan uses beta=1,...,9. The beta=0 slice is retained only for the historical main-figure cross-sections.
- `population_status.csv` and `population_boxplots.csv` contain summaries of stationary relative entropy, persistence (`timeInDEEP`, fraction of 50 windows) and depth (`narrowing`, one minus mean relative entropy).
- `intervention_curves.csv` contains eleven doses for each intervention. `n=1596` at each setting. Normal binomial intervals reproduce the main risk-curve method.
- Model CDF files use fixed grids and grouped cumulative counts. There are no per-agent rows, assignments, or traces in the release.

## Displayed tables

`data/tables/` preserves the rounded values and mathematical text in the current supplement, including confidence intervals and notation. Numeric analysis-ready versions are also supplied under `data/statistical_results/` where available. LaTeX notation in cells is retained intentionally; table labels are resolved through `data/table_labels.json`.
