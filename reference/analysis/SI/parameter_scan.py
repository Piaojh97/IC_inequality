"""Render saved parameter-grid outcomes; do not run new simulations.

The array axes are beta, alpha, sigma. The overlapping beta=4,...,8
slices are verified against the individual-user outputs. The fine-grained
run scripts specify beta=1,2,3,9, 1,000 users and 5,000 steps per cell.
The all-zero beta=0 slice is not included: no corresponding fine-grained
run specification was located. No missing or synthetic values are added.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns

from build_figures import SIM, OUT, RESULTS, panel


def main():
    path = SIM / "phase_diagram.pkl"
    risk = pd.read_pickle(path)
    assert risk.shape == (10, 10, 10)
    assert np.isfinite(risk).all() and ((risk >= 0) & (risk <= 1)).all()
    detail = pd.read_pickle(SIM / "intervention_beta.pkl")
    grouped = detail.groupby(["beta", "alpha", "sigma"]).indicator.agg(["mean", "size"])
    assert grouped["size"].eq(1000).all()
    np.testing.assert_array_equal(grouped["mean"].to_numpy().reshape(5,10,10), risk[4:9])
    np.testing.assert_array_equal(risk[7].T, pd.read_pickle(SIM / "heatmap_model.pkl")["mat"])

    alpha = 5.6 + .1228 * np.arange(10)
    sigma = .018 + .0003458 * np.arange(10)
    rows = [(b, a, s, alpha[a], sigma[s], 1000, risk[b,a,s])
            for b in range(1,10) for a in range(10) for s in range(10)]
    source = pd.DataFrame(rows, columns=["beta", "exploitation_score", "exploration_score",
                                        "alpha", "sigma", "n", "risk"])
    source.to_csv(RESULTS / "fig_s7_parameter_scan.csv", index=False)
    source.groupby("beta").risk.agg(["min","max","mean"]).to_csv(
        RESULTS / "fig_s7_parameter_summary.csv")

    fig, axes = plt.subplots(3,3,figsize=(180/25.4,165/25.4),layout="constrained")
    cmap = sns.color_palette("vlag_r", as_cmap=True)
    # White marks the observed cohort risk, using the main figures' color direction.
    norm = TwoSlopeNorm(vmin=0, vcenter=37.4, vmax=70)
    edges = np.arange(11) - .5
    for b, ax in zip(range(1,10), axes.flat):
        im = ax.pcolormesh(edges, edges, risk[b].T * 100, cmap=cmap, norm=norm,
                           edgecolors="#B8B8B8", linewidth=.25, rasterized=False)
        ax.set_aspect("equal")
        ax.set_xticks([0,4,9], [f"{alpha[i]:.2f}" for i in [0,4,9]])
        ax.set_yticks([0,4,9], [f"{sigma[i]*1000:.1f}" for i in [0,4,9]])
        ax.tick_params(length=3, width=.6, labelsize=7)
        title = f"β = {b}" + (" (baseline)" if b == 7 else "")
        ax.set_title(title,fontsize=7,pad=7,weight="bold" if b==7 else "normal")
        if b > 6:
            ax.set_xlabel("Exploitation, α",fontsize=7)
        if (b-1) % 3 == 0:
            ax.set_ylabel(r"Exploration, σ ($\times 10^{-3}$)",fontsize=7)
        panel(ax, "abcdefghi"[b-1],fontsize=8)
    cb=fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.78, fraction=.04, pad=.025,
                    ticks=[0,20,37.4,50,70])
    cb.set_label("Deep-IC risk (%)",fontsize=7)
    cb.ax.tick_params(labelsize=7,length=3,width=.6)
    cb.outline.set_linewidth(.6)
    fig.savefig(OUT / "ic-s7.pdf")
    plt.close(fig)
    (RESULTS / "fig_s7_provenance.json").write_text(json.dumps({
        "source": str(path.relative_to(Path(__file__).resolve().parent.parent)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "array_axes": ["beta", "alpha", "sigma"],
        "plotted_beta": list(range(1,10)),
        "settings": len(source), "users_per_setting": 1000,
        "verified_exact_slices": list(range(4,9)),
        "baseline_matches_calibration": True,
        "note": "beta=0 omitted because no corresponding fine-grained run specification was located."
    },ensure_ascii=False,indent=2) + "\n")
    print("Rendered nine beta slices (900 settings); 500 overlapping cells and the baseline match exactly.")


if __name__ == "__main__":
    main()
