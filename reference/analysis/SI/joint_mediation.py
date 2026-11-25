"""Standardised, additive decomposition of the 0-to-5 age-band risk difference.

Run from any directory; keep the modelling data and subject-level calculations local.
The linear score and logistic outcome specifications follow mediation_si.R.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SCORES = ["explore_q", "exploit_q"]
NUMERIC = ["age_conti", "price_conti", "entertainment_ratio", "knowledge_ratio", "watch_entropy_1st"]
FACTORS = ["gender", "fre_city_level_eng", "fre_community_level_eng"]
# Each tuple sets the age code for (outcome, exploration score, exploitation score).
STATES = [(0, 0, 0), (5, 0, 0), (5, 5, 0), (5, 0, 5), (5, 5, 5)]
CONTRASTS = pd.DataFrame(
    [[0, -.5, .5, -.5, .5], [0, -.5, -.5, .5, .5],
     [0, -1, 0, 0, 1], [-1, 1, 0, 0, 0], [-1, 0, 0, 0, 1]],
    index=["Exploration", "Exploitation", "Both behaviours", "Remaining", "Total"],
)


def cohort(named=False):
    columns = ["user_id", "age", "m_ratio"] + NUMERIC + FACTORS + SCORES
    data = pd.read_csv(ROOT.parent / "cleaned_data_v2.csv", usecols=columns, dtype={"user_id": str})
    assert len(data) == 653486 and data.user_id.is_unique and not data.isna().any().any()
    assert set(data.age_conti) == set(range(6))
    assert data.groupby("age_conti").age.agg(["min","max"]).to_numpy()[[0,-1]].tolist() == [[19,23],[61,79]]
    cut = 1 + norm.ppf(.10) * data.m_ratio.std(ddof=0)
    y = (data.m_ratio < cut).astype(float).to_numpy()
    assert np.isclose(y.mean(), .3735076191379769)
    dummies = pd.get_dummies(data[FACTORS], dtype=float).drop(
        columns=["gender_M", "fre_city_level_eng_fifth", "fre_community_level_eng_City"])
    design = sm.add_constant(pd.concat([data[NUMERIC], dummies], axis=1))
    x = design if named else np.asarray(design, dtype=float, order="C")
    m = data[SCORES].to_numpy(dtype=float)
    assert np.all((m >= 0) & (m <= 9))
    return x, m, y


def standardise(x, m, y, weight=None, influence=True):
    """Fit all three equations and return cohort-averaged risks at the five settings."""
    n = len(y)
    w = np.ones(n) if weight is None else weight
    # Columns 0,1 are the intercept and the ordered age-band code.
    a = x[:, 1]
    xwx = x.T @ (w[:,None] * x)
    score_coef = np.linalg.solve(xwx, x.T @ (w[:,None] * m))
    design = np.column_stack((x, m))
    model = sm.GLM(y, design, freq_weights=w, family=sm.families.Binomial())
    fit = model.fit(maxiter=50, tol=1e-10)
    assert fit.converged
    beta = fit.params
    logit = design @ beta
    if influence:
        p = expit(logit)
        # Unnormalised sums are used below: each row is its O(1/n) contribution.
        info = design.T @ ((p*(1-p))[:,None] * design)
        y_res = y-p
        age_leverage = x @ np.linalg.inv(xwx)[:,1]
        score_slope_if = age_leverage[:,None]*(m-x@score_coef)
    risks, contributions = [], []
    for y_age, e_age, u_age in STATES:
        step = np.column_stack((e_age-a, u_age-a))
        latent = m + step*score_coef[1]
        shifted = np.clip(latent, 0, 9)
        prob = expit(logit + beta[1]*(y_age-a) + (shifted-m) @ beta[-2:])
        risk = np.average(prob, weights=w)
        risks.append(risk)
        if influence:
            # Derivatives of the mean standardised risk with respect to the outcome
            # coefficients and the two score-model age slopes. Pairing each user's
            # residuals retains dependence between scores and between fitted equations.
            dp = prob*(1-prob)
            grad_beta = design.T @ dp/n
            grad_beta[1] = y_age*dp.mean()
            grad_beta[-2:] = shifted.T @ dp/n
            slope_grad = ((dp[:,None]*step)*((latent>0)&(latent<9))).mean(axis=0)*beta[-2:]
            contrib = (prob-risk)/n
            contrib += y_res*(design @ np.linalg.solve(info, grad_beta))
            contrib += score_slope_if @ slope_grad
            contributions.append(contrib)
    return np.array(risks), np.array(contributions)


def main():
    x, m, y = cohort()
    risk, risk_if = standardise(x, m, y)
    delta = CONTRASTS.to_numpy() @ risk
    effect_if = CONTRASTS.to_numpy() @ risk_if
    total = delta[-1]
    assert total > 0 and np.isclose(delta[:2].sum(), delta[2])
    assert np.isclose(delta[2]+delta[3], total)
    assert np.allclose(effect_if[0]+effect_if[1],effect_if[2]) and np.allclose(effect_if[2]+effect_if[3],effect_if[-1])
    assert np.max(abs(risk_if.sum(axis=1))) < 1e-7
    share = delta/total
    share_if = (effect_if - share[:,None]*effect_if[-1])/total
    z = norm.ppf(.975)
    half = z*np.linalg.norm(effect_if,axis=1)
    share_half = z*np.linalg.norm(share_if,axis=1)
    rows = pd.DataFrame(dict(component=CONTRASTS.index, risk_difference=delta,
                             lo=delta-half, hi=delta+half, share=share,
                             share_lo=share-share_half, share_hi=share+share_half))
    RESULTS.mkdir(exist_ok=True)
    rows.to_csv(RESULTS/"joint_mediation.csv",index=False)
    checks = pd.DataFrame(STATES,columns=["outcome_age","exploration_age","exploitation_age"])
    checks["standardised_risk"] = risk
    checks.to_csv(RESULTS/"joint_mediation_risks.csv",index=False)
    print(rows.to_string(index=False))
    print(checks.to_string(index=False))
    if "--verify" in sys.argv:
        # An independent numerical check of the stacked subject-level influence
        # calculation, perturbing the weights then refitting every equation.
        rng=np.random.default_rng(5271)
        direction=rng.choice([-1.,1.],len(y))
        eps=.005
        plus,_=standardise(x,m,y,1+eps*direction,False)
        minus,_=standardise(x,m,y,1-eps*direction,False)
        measured=(plus-minus)/(2*eps)
        predicted=risk_if@direction
        np.testing.assert_allclose(measured,predicted,atol=2e-8,rtol=2e-4)
        print("Finite-difference check (max absolute error)",abs(measured-predicted).max())


if __name__ == "__main__":
    main()
