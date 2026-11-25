"""Write LaTeX table rows directly from the aggregate analysis outputs."""
from pathlib import Path
import numpy as np
import pandas as pd
base = Path(__file__).resolve().parent
res = base/"results"
rows = {}
age = pd.read_csv(res/"table_s1.csv")
rows["AgeRows"] = [f"{r.age.replace('–','--')} & {r.n:,} & {r.percent:.2f}" for r in age.itertuples()]
rows["AgeRows"].append(r"\midrule Total & 653,486 & 100.00")
decade = pd.read_csv(res/"age_in_years.csv").set_index("outcome")
rows["DecadeRows"] = []
for key, label, digits in [
    ("Deep IC probability", "Deep-IC risk (percentage points)", 2),
    ("Minimum relative entropy", "Minimum relative entropy", 4),
]:
    r = decade.loc[key]
    rows["DecadeRows"].append(
        f"{label} & {r.older_minus_younger:.{digits}f} "
        f"[{r.lo95:.{digits}f}, {r.hi95:.{digits}f}] & {int(r.n_contrast):,}"
    )
population = pd.read_csv(res/"simulation_population_summary.csv")
rows["SimulationAgeRows"] = [
    f"{r.age.replace('–','--')} & {r.n:,} & {r.empirical_risk*100:.1f} & {r.simulated_risk*100:.1f}"
    for r in population.itertuples()
]
doses = pd.read_csv(res/"intervention_dose_definition.csv").set_index("parameter")
rows["DoseRows"] = []
for parameter, label, change in [
    ("beta", "Weaker matching", r"$\beta=7-\delta$"),
    ("alpha", "Lower exploitation", r"$\alpha_l=\alpha_{l,0}-1.228\delta$"),
    ("sigma", "Higher exploration", r"$\sigma_l=\sigma_{l,0}+0.003458\delta$"),
]:
    r = doses.loc[parameter]
    rows["DoseRows"].append(
        f"{label} & {change} & {r.risk_at_30*100:.1f} & {r.risk_at_100*100:.1f}"
    )
reg = pd.read_csv(res/"table_s2.csv")
rows["RegressionRows"] = []
for label, group in reg.groupby("adjustment",sort=False):
    ors = group[group.metric=="risk"].iloc[0]
    slope = group[group.metric=="depth"].iloc[0]
    rows["RegressionRows"].append(f"{label} & {ors.estimate:.3f} [{ors.lo:.3f}, {ors.hi:.3f}] & "
                                  f"{slope.estimate:.4f} [{slope.lo:.4f}, {slope.hi:.4f}]")
med = pd.read_csv(res/"joint_mediation.csv")
rows["MediationRows"] = []
for row in med.itertuples(index=False):
    effect = f"{100*row.risk_difference:.2f} [{100*row.lo:.2f}, {100*row.hi:.2f}]"
    share = f"{100*row.share:.1f} [{100*row.share_lo:.1f}, {100*row.share_hi:.1f}]"
    if row.component == "Total":
        share = "100.0"
    label = row.component
    if label in ("Both behaviours", "Total"):
        label = r"\midrule " + label
    rows["MediationRows"].append(f"{label} & {effect} & {share}")
# Full regression outputs retain the original factor coding. Shares are shown
# per 10 percentage points to give their coefficients an interpretable unit.
terms = [
    ("age_conti", "Age band"), ("explore_q", "Exploration score"), ("exploit_q", "Exploitation score"),
    ("gender_F", "Female"), ("price_conti", "Handset price band"),
    *[(f"fre_city_level_eng_{tier}", f"City tier {i}") for i,tier in enumerate(
        ["first", "second", "third", "fourth"], start=1)],
    ("fre_community_level_eng_County", "County"), ("fre_community_level_eng_Countryside", "Countryside"),
    ("entertainment_ratio", "Entertainment share"), ("news_ratio", "News share"),
    ("knowledge_ratio", "Knowledge share"), ("watch_entropy_1st", "First-hour entropy"),
    ("const", "Intercept (link scale)"),
]
for filename, name, models in [
    ("regression_full", "FullAgeRows", ["risk","depth"]),
    ("regression_paths", "PathRows", ["explore_q","exploit_q","risk"]),
]:
    saved = pd.read_csv(res / (filename + ".csv")).replace({"mod_price_range_conti":"price_conti"})
    saved = saved.set_index(["model","term"])
    assert set(saved.index.get_level_values("term")) <= {k for k,_ in terms}
    rows[name] = []
    for key,label in terms:
        if key not in saved.index.get_level_values("term"):
            continue
        cells = [label]
        for model in models:
            if (model,key) not in saved.index:
                cells.append(r"\textemdash")
                continue
            v = saved.loc[(model,key),["estimate","lo","hi"]].to_numpy(dtype=float)
            if key.endswith("_ratio"):
                v = v * .1
            if model == "risk" and key != "const":
                v = np.exp(v)
            digits = 4 if model == "depth" else 3
            cells.append(f"{v[0]:.{digits}f} [{v[1]:.{digits}f}, {v[2]:.{digits}f}]")
        rows[name].append(" & ".join(cells))

with (base/"overleaf"/"ic-tables.tex").open("w") as output:
    for name,values in rows.items():
        output.write(r"\newcommand{" + "\\" + name + "}{%\n")
        output.write("\n".join(v+r" \\" for v in values)+"\n}\n")
