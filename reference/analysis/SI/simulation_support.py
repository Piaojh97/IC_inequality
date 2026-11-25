"""Summarize the saved simulations used in main Figures 3 and 4."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "processed_data"
OUT = HERE / "results"
AGES = ["18-23", "24-30", "31-40", "41-50", "51-60", "60+"]
LABELS = ["19–23", "24–30", "31–40", "41–50", "51–60", "61–79"]


def main():
    empirical = pd.read_csv(HERE.parent / "cleaned_data_v2.csv", usecols=["age_range", "m_ratio"])
    cutoff = 1 + norm.ppf(.1) * empirical.m_ratio.std(ddof=0)
    empirical["deep"] = empirical.m_ratio < cutoff
    simulated = pd.read_pickle(DATA / "samples_10000_model.pkl")
    assert len(empirical) == 653486 and len(simulated) == 10000
    np.testing.assert_array_equal(simulated.indicator, simulated.entropy < cutoff)
    with np.load(HERE / "source_data" / "simulation_baseline_entropy.npz") as trajectories:
        assert len(trajectories.files) == 5
        for age in trajectories.files:
            trajectory = trajectories[age]
            group = simulated.loc[simulated.age_range == age]
            assert trajectory.shape == (50, len(group))
            np.testing.assert_array_equal(trajectory[-1], group.entropy)
            np.testing.assert_array_equal(trajectory.mean(axis=0), group.depth)
            np.testing.assert_array_equal((trajectory < 1).mean(axis=0), group.timeInDEEP)
    e = empirical.groupby("age_range").deep.mean()
    s = simulated.groupby("age_range").indicator.agg(["size", "sum", "mean"])
    rows = []
    for key, label in zip(AGES, LABELS):
        rows.append({"age": label, "n": int(s.loc[key, "size"]),
                     "deep_n": int(s.loc[key, "sum"]), "empirical_risk": e.loc[key],
                     "simulated_risk": s.loc[key, "mean"]})
    pd.DataFrame(rows).to_csv(OUT / "simulation_population_summary.csv", index=False)
    simulated.groupby("indicator")[["timeInDEEP", "depth"]].agg(["size", "mean"]).to_csv(
        OUT / "simulation_persistence_summary.csv")

    grid = pd.read_pickle(DATA / "intervention_beta.pkl")
    assert grid.groupby(["sigma", "alpha", "beta"]).size().eq(1000).all()
    np.testing.assert_allclose(
        grid.loc[grid.beta == 7].groupby(["sigma", "alpha"]).indicator.mean().unstack(),
        pd.read_pickle(DATA / "heatmap_model.pkl")["mat"], atol=0, rtol=0)
    rows = []
    for parameter, field, final_setting in [
        ("beta", "reduced_beta", 1), ("alpha", "reduced_alpha", 10),
        ("sigma", "increased_sigma", 10),
    ]:
        records = pd.read_pickle(DATA / f"varying_{parameter}.pkl")
        summary = records.groupby(field).indicator.agg(["size", "mean"])
        assert summary["size"].eq(1596).all() and len(summary) == 11
        np.testing.assert_allclose(summary["mean"].iloc[0], s.loc["60+", "mean"])
        np.testing.assert_allclose(summary.index / final_setting, np.linspace(0, 1, 11))
        rows.append({"parameter": parameter, "n": 1596, "dose_fraction": .3,
                     "risk_at_30": summary["mean"].iloc[3],
                     "risk_at_100": summary["mean"].iloc[-1]})
    pd.DataFrame(rows).to_csv(OUT / "intervention_dose_definition.csv", index=False)
    print("Verified the 100-cell grid, 10,000-user population and three intervention curves.")


if __name__ == "__main__":
    main()
