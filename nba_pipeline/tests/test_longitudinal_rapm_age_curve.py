import numpy as np
import pandas as pd

from nba_pipeline.scripts.fit_longitudinal_rapm_age_curve import (
    fit_longitudinal_curves,
    main,
)


def synthetic_player_season_panel() -> pd.DataFrame:
    season_levels = {
        2010: 0.0,
        2011: 0.30,
        2012: 0.10,
        2013: 0.20,
        2014: 0.00,
        2015: 0.10,
        2016: -0.10,
        2017: 0.00,
        2018: 0.00,
    }
    rows = []
    for player_index in range(72):
        first_age = 19.0 + (player_index % 12)
        off_level = -2.0 + 0.08 * player_index
        def_level = 1.5 - 0.05 * player_index
        for season in range(2010, 2019):
            age = first_age + (season - 2010)
            rows.append(
                {
                    "player_id": str(1000 + player_index),
                    "season": season,
                    "age": age,
                    "orapm": off_level + 0.25 * (age - 27.0) + season_levels[season],
                    "drapm": def_level - 0.10 * (age - 27.0) - 0.5 * season_levels[season],
                    "off_poss": 2000,
                    "def_poss": 2000,
                }
            )
    return pd.DataFrame(rows)


def test_longitudinal_curve_recovers_age_change_separately_from_calendar_shocks():
    fitted = fit_longitudinal_curves(
        synthetic_player_season_panel(),
        age_min=19,
        age_max=39,
        reference_age=27,
        smoothing_alpha=1e-8,
        bootstrap_reps=0,
    )

    curve = fitted.curve.set_index("age")

    assert curve.loc[27, "orapm_age_effect"] == 0.0
    assert curve.loc[27, "drapm_age_effect"] == 0.0
    assert np.isclose(curve.loc[30, "orapm_age_effect"], 0.75, atol=0.04)
    assert np.isclose(curve.loc[30, "drapm_age_effect"], -0.30, atol=0.04)
    assert fitted.receipt["identification"] == "calendar_pair_effects_weighted_mean_zero"


def test_cli_writes_frozen_curve_and_audit_artifacts(tmp_path):
    panel_path = tmp_path / "annual_orapm_drapm.csv"
    output_dir = tmp_path / "age_curve"
    synthetic_player_season_panel().to_csv(panel_path, index=False)

    assert main(
        [
            "--panel",
            str(panel_path),
            "--output-dir",
            str(output_dir),
            "--age-min",
            "19",
            "--age-max",
            "39",
            "--smoothing-alpha",
            "0.00000001",
            "--bootstrap-reps",
            "0",
        ]
    ) == 0

    assert (output_dir / "age_curve.csv").exists()
    assert (output_dir / "consecutive_player_season_pairs.csv").exists()
    assert (output_dir / "calendar_pair_effects.csv").exists()
    assert (output_dir / "fit_receipt.json").exists()


def test_player_cluster_bootstrap_reports_uncertainty_and_keeps_reference_fixed():
    fitted = fit_longitudinal_curves(
        synthetic_player_season_panel(),
        age_min=19,
        age_max=39,
        reference_age=27,
        smoothing_alpha=0.01,
        bootstrap_reps=8,
        random_seed=7,
    )
    curve = fitted.curve.set_index("age")

    for column in [
        "orapm_ci_low",
        "orapm_ci_high",
        "drapm_ci_low",
        "drapm_ci_high",
    ]:
        assert column in curve.columns
        assert curve[column].notna().all()
        assert curve.loc[27, column] == 0.0
    assert (curve["orapm_ci_low"] <= curve["orapm_age_effect"]).all()
    assert (curve["orapm_age_effect"] <= curve["orapm_ci_high"]).all()


def test_smoothness_is_selected_separately_with_grouped_player_cv_minimum():
    grid = [0.01, 1.0, 100.0]
    fitted = fit_longitudinal_curves(
        synthetic_player_season_panel(),
        age_min=19,
        age_max=39,
        reference_age=27,
        smoothing_alpha=None,
        smoothing_grid=grid,
        cv_folds=4,
        bootstrap_reps=0,
        random_seed=11,
    )

    assert fitted.receipt["smoothing_selection"] == "grouped_player_minimum_mse"
    assert fitted.receipt["off_smoothing_alpha"] in grid
    assert fitted.receipt["def_smoothing_alpha"] in grid
    assert set(fitted.cv_results["side"]) == {"o", "d"}
    assert set(fitted.cv_results["alpha"]) == set(grid)
    for _, side_rows in fitted.cv_results.groupby("side"):
        selected = side_rows[side_rows["selected"]].iloc[0]
        assert selected["mean_mse"] == side_rows["mean_mse"].min()


def test_curve_reports_local_transition_support():
    fitted = fit_longitudinal_curves(
        synthetic_player_season_panel(),
        age_min=18,
        age_max=40,
        reference_age=27,
        smoothing_alpha=1.0,
        bootstrap_reps=0,
    )
    curve = fitted.curve.set_index("age")

    assert {"transition_pairs", "transition_players", "supported_age"}.issubset(curve.columns)
    assert not bool(curve.loc[18, "supported_age"])
    assert bool(curve.loc[27, "supported_age"])
