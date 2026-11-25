"""Report the three equations underlying the standardized decomposition in Table S3."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.patches import FancyBboxPatch

from build_figures import BLUE, ORANGE, finish, plt
from joint_mediation import RESULTS, SCORES, cohort


def coefficients():
    x, m, y = cohort(named=True)
    rows = []
    for i, score in enumerate(SCORES):
        model = sm.OLS(m[:, i], x).fit(cov_type="HC0")
        rows.extend(model_rows(score, model))
    predictors = pd.concat([x, pd.DataFrame(m, columns=SCORES)], axis=1)
    model = sm.GLM(y, predictors, family=sm.families.Binomial()).fit(cov_type="HC0")
    assert model.converged
    rows.extend(model_rows("risk", model))
    return pd.DataFrame(rows, columns=["model", "term", "estimate", "lo", "hi", "n"])


def model_rows(name, fit):
    ci = fit.conf_int()
    return [(name, term, value, *ci.loc[term], int(fit.nobs)) for term, value in fit.params.items()]


def diagram(data):
    coef = data.set_index(["model", "term"]).estimate
    fig, ax = plt.subplots(figsize=(7.0, 1.8))
    fig.subplots_adjust(left=.025, right=.975, bottom=.04, top=.98)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    def node(x, y, w, h, label, border):
        ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=.004,rounding_size=.018",
                                    facecolor="white", edgecolor=border, linewidth=1.3))
        ax.text(x, y, label, ha="center", va="center", fontsize=10)
    def arrow(start, end, text, label_at):
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", color="#464E55",
                    lw=1.1, mutation_scale=11, shrinkA=0, shrinkB=0))
        ax.text(*label_at, text, ha="center", va="center", fontsize=8.5,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.1))
    node(.095,.50,.17,.19,"Age band\n(0–5)", "#49535C")
    node(.50,.83,.22,.21,"Exploration score\n(0–9)", BLUE)
    node(.50,.17,.22,.21,"Exploitation score\n(0–9)", ORANGE)
    node(.905,.50,.17,.19,"Deep IC\n(yes/no)", "#49535C")
    arrow((.18,.56),(.385,.81),f"β = {coef['explore_q', 'age_conti']:+.3f}",(.265,.77))
    arrow((.18,.44),(.385,.19),f"β = {coef['exploit_q', 'age_conti']:+.3f}",(.265,.24))
    arrow((.615,.81),(.816,.56),f"OR = {np.exp(coef['risk', 'explore_q']):.3f}",(.74,.77))
    arrow((.615,.19),(.816,.44),f"OR = {np.exp(coef['risk', 'exploit_q']):.3f}",(.74,.24))
    arrow((.18,.50),(.816,.50),f"Adjusted age association   OR = {np.exp(coef['risk', 'age_conti']):.3f}",(.50,.565))
    finish(fig, "s6")


if __name__ == "__main__":
    data = coefficients()
    data.to_csv(RESULTS / "regression_paths.csv", index=False)
    diagram(data)
