"""Estimate a ten-year age contrast for the two fully adjusted Fig. 1 outcomes.

Fit both models to the full cohort. For the logistic model, average the
difference between predictions at a user's recorded age and ten years older.
Both ages must lie in the cohort's observed age range. The interval for this
standardized contrast conditions on the observed covariates.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

from build_figures import RESULTS, read_cohort


def fit_models(cohort, design):
    risk = sm.Logit(cohort.deep, design, missing="raise").fit(disp=False, cov_type="HC0")
    depth = sm.OLS(cohort.m_ratio, design, missing="raise").fit(cov_type="HC0")
    assert risk.mle_retvals["converged"]
    return risk, depth


def decade_risk_difference(model, observed):
    older = observed.assign(age_decades=observed.age_decades + 1)
    p0 = model.predict(observed).to_numpy()
    p1 = model.predict(older).to_numpy()
    difference = (p1 - p0).mean()

    # Derivative of the mean difference with respect to all logit coefficients.
    gradient = (older.to_numpy().T @ (p1 * (1 - p1))
                - observed.to_numpy().T @ (p0 * (1 - p0))) / len(observed)
    se = np.sqrt(gradient @ model.cov_params().to_numpy() @ gradient)
    margin = norm.ppf(0.975) * se
    return 100 * np.array([difference, difference - margin, difference + margin])


def main():
    cohort = read_cohort()
    reference = pd.read_csv(RESULTS / "regression_full.csv").set_index(["model", "term"])
    factors = ["gender", "fre_city_level_eng", "fre_community_level_eng"]
    design = pd.get_dummies(cohort[factors], dtype=float)
    numeric = ["age_conti", "entertainment_ratio", "news_ratio", "watch_entropy_1st"]
    design[numeric] = cohort[numeric]
    design["mod_price_range_conti"] = cohort.price_conti
    design["const"] = 1.0
    design = design.loc[:, reference.loc["risk"].index]

    # Recover the archived coefficients before substituting age in years.
    for name, model in zip(["risk", "depth"], fit_models(cohort, design)):
        expected = reference.loc[name].loc[design.columns, "estimate"]
        np.testing.assert_allclose(model.params, expected, atol=1e-8, rtol=1e-8)
    assert cohort.age.between(19, 79).all()
    design = design.rename(columns={"age_conti": "age_decades"})
    design["age_decades"] = cohort.age / 10
    risk, depth = fit_models(cohort, design)

    eligible = design.loc[cohort.age + 10 <= cohort.age.max()]
    risk_contrast = decade_risk_difference(risk, eligible)
    depth_contrast = [depth.params["age_decades"], *depth.conf_int().loc["age_decades"]]
    results = pd.DataFrame([
        ["Minimum relative entropy", "entropy ratio units", len(cohort), len(cohort), *depth_contrast],
        ["Deep IC probability", "percentage points", len(cohort), len(eligible), *risk_contrast],
    ], columns=["outcome", "unit", "n_fit", "n_contrast", "older_minus_younger", "lo95", "hi95"])
    results.to_csv(RESULTS / "age_in_years.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
