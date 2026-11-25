"""Verify initialization inputs and report the implemented calibration settings."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
INPUT = HERE.parent / "server_backup" / "fine_grained" / "Data"
SIM = HERE.parent / "processed_data"


def main():
    empirical = pd.read_pickle(INPUT / "first_hour_watch_avg_vetor.pkl")
    prior = np.load(INPUT / "user_alpha_mean_video_v3.npy")
    items = pd.read_pickle(INPUT / "items_num.pkl")
    item_input = np.load(INPUT / "item_popularity_video_v2.npy")
    phi = np.load(INPUT / "phi_video_v2.npy")
    correlations = pd.read_pickle(INPUT / "corr_matrix.pkl")
    np.testing.assert_array_equal(prior, .65 * empirical.to_numpy())
    np.testing.assert_array_equal(item_input[:, 1], items.items_num / items.items_num.sum())
    np.testing.assert_array_equal(phi, correlations.to_numpy())
    np.testing.assert_allclose(phi, phi.T)
    np.testing.assert_allclose(np.diag(phi), 1)
    assert len(prior) == 35 and (prior > 0).all()
    np.testing.assert_allclose((15 * prior).sum(), 9.75, atol=1e-10)
    topic_ids = [int(str(i).split("_")[0]) for i in empirical.index]
    assert topic_ids == items.first_level_category_id.tolist()
    pd.DataFrame({"topic_id": topic_ids, "mean_first_hour_share": empirical.to_numpy(),
                  "candidate_topic_probability": item_input[:, 1],
                  "dirichlet_concentration": 15 * prior}).to_csv(
        HERE / "source_data" / "model_initialization.csv", index=False)
    correlations.to_csv(HERE / "source_data" / "model_topic_correlations.csv")

    obs = pd.read_pickle(SIM / "heatmap_em_new.pkl")
    sim = pd.read_pickle(SIM / "heatmap_model.pkl")
    x = np.indices((10,10)).reshape(2,-1).T
    fitted = np.linalg.lstsq(np.c_[np.ones(100),x],obs.ravel(),rcond=None)[0]
    retained = np.ones((10,10), dtype=bool)
    retained[np.ix_([0,9],[0,9])] = False
    r2 = 1 - np.sum((sim["mat"] - obs)**2) / np.sum((obs - obs.mean())**2)
    mae = np.abs(sim["mat"] - obs)[retained].mean() * 100
    np.testing.assert_allclose(sim["beta"], 5.6 + .1228 * np.arange(10))
    np.testing.assert_allclose(sim["sigma"], .018 + .0003458 * np.arange(10))
    metadata = {
        "initialization": {
            "topics": 35, "candidate_items": 10000,
            "user_prior_multiplier_in_input": .65,
            "simulator_multiplier": 15, "total_concentration": 9.75,
            "intrinsic_and_observed_draws": "independent",
            "intrinsic_preferences": "fixed during each run",
            "topic_correlations": "unmodified empirical correlation matrix",
        },
        "calibration": {
            "author_confirmation": "2026-11-24: four corners set the range; slopes set the mapping increments.",
            "empirical_plane_intercept_zero_based": float(fitted[0]),
            "empirical_slopes": fitted[1:].tolist(),
            "stored_rounded_empirical_slopes": [-.01645, .01893],
            "stored_coarse_model_slopes_per_grid_step": [-.01903, .01541],
            "coarse_grid_steps_from_run_scripts": [.0004, .1],
            "slope_ratio_increments": [.0004*.01645/.01903, .1*.01893/.01541],
            "implemented_increments": [.0003458, .1228],
            "note": "The exploratory mapping cell writes 0.0005 for the sigma step; the server's nine-level 0.018-0.0212 grid has step 0.0004, which reproduces the implemented mapping.",
            "beta": 7, "r_squared_100_cells": r2, "mae_96_cells_pp": mae,
        },
        "input_sha256": {name: hashlib.sha256((INPUT/name).read_bytes()).hexdigest()
                         for name in ["user_alpha_mean_video_v3.npy", "item_popularity_video_v2.npy", "phi_video_v2.npy"]},
    }
    (HERE / "results" / "model_details.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(f"Verified 35-topic initialization (concentration 9.75), empirical inputs and mappings; R²={r2:.6f}, MAE={mae:.4f} pp.")


if __name__ == "__main__":
    main()
