"""Behavior tests for the scalar Title Lift over Contract challenger."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_pipeline.scripts.analyze_contract_title_curve import (
    attach_realized_surplus_points,
    build_career_table,
    build_leave_one_season_out_challenger,
    evaluate_predictions,
    fit_monotone_contract_curve,
    fit_zero_anchored_title_curve,
    summarize_player_possessions,
)


def test_scalar_curves_are_monotone_and_title_lift_is_zero_at_zero_surplus() -> None:
    salary_share = np.array([0.05, 0.10, 0.20, 0.30, 0.35])
    peer_epm = np.array([-0.5, 0.0, 0.8, 2.0, 1.8])
    contract_curve = fit_monotone_contract_curve(salary_share, peer_epm)

    salary_grid = np.linspace(0.05, 0.35, 31)
    expected_epm = contract_curve.predict(salary_grid)
    assert np.all(np.diff(expected_epm) >= -1e-12)

    surplus_epm = np.array([-3.0, -1.0, 0.5, 2.0, 4.0])
    full_model_lift = np.array([-0.08, -0.03, 0.01, 0.09, 0.20])
    title_curve = fit_zero_anchored_title_curve(surplus_epm, full_model_lift)

    surplus_grid = np.linspace(-3.0, 4.0, 71)
    predicted_lift = title_curve.predict(surplus_grid)
    assert np.all(np.diff(predicted_lift) >= -1e-12)
    assert title_curve.predict(np.array([0.0]))[0] == 0.0


def test_challenger_predictions_do_not_train_on_the_held_out_season() -> None:
    rows = []
    for season, offset in [(2022, 0.00), (2023, 0.01), (2024, -0.01)]:
        for player_id, (share, epm, peer, lift) in enumerate(
            [
                (0.05, -1.0, -0.5, -0.02),
                (0.12, 0.5, 0.0, 0.02),
                (0.22, 2.0, 1.0, 0.07),
                (0.34, 5.0, 2.2, 0.18),
            ],
            start=1,
        ):
            rows.append(
                {
                    "season": season,
                    "player_id": player_id,
                    "salary_share": share,
                    "own_dpm": epm,
                    "salary_peer_mean_epm": peer + offset,
                    "salary_peer_mean_workload_share": 0.20 + share,
                    "title_lift_over_contract": lift + offset,
                }
            )
    frame = pd.DataFrame(rows)
    original = build_leave_one_season_out_challenger(frame)

    changed = frame.copy()
    held_out = changed["season"] == 2024
    changed.loc[held_out, "salary_peer_mean_epm"] += 20.0
    changed.loc[held_out, "title_lift_over_contract"] += 0.5
    rescored = build_leave_one_season_out_challenger(changed)

    original_2024 = original.loc[original["season"] == 2024, "scalar_title_lift"]
    rescored_2024 = rescored.loc[rescored["season"] == 2024, "scalar_title_lift"]
    np.testing.assert_allclose(original_2024, rescored_2024)
    np.testing.assert_allclose(
        original.loc[original["season"] == 2024, "role_adjusted_title_lift"],
        rescored.loc[rescored["season"] == 2024, "role_adjusted_title_lift"],
    )
    assert np.all(original["scalar_expected_workload"] > 0.0)


def test_prediction_metrics_report_probability_errors_in_percentage_points() -> None:
    metrics = evaluate_predictions(
        actual=np.array([-0.02, 0.01, 0.05]),
        predicted=np.array([-0.01, 0.00, 0.04]),
    )

    assert metrics["n"] == 3
    assert metrics["mae_pp"] == 1.0
    assert metrics["rmse_pp"] == 1.0
    assert metrics["bias_actual_minus_predicted_pp"] == 0.333333333333


def test_exported_surplus_points_use_realized_possessions_and_divide_by_100() -> None:
    scored = pd.DataFrame(
        {
            "season": [2025, 2025],
            "player_id": [1, 2],
            "scalar_epm_surplus": [4.0, -1.5],
        }
    )
    possessions = pd.DataFrame(
        {
            "season": [2025, 2025],
            "player_id": [1, 2],
            "regular_season_possessions": [5_000.0, 4_000.0],
        }
    )

    exported = attach_realized_surplus_points(scored, possessions)

    assert exported.loc[0, "regular_season_possessions"] == 5_000.0
    assert exported.loc[0, "realized_rs_points_over_contract"] == 200.0
    assert exported.loc[1, "realized_rs_points_over_contract"] == -60.0


def test_player_possessions_average_offense_and_defense_instead_of_double_counting() -> None:
    lineups = pd.DataFrame(
        {
            "O1": [1, 2],
            "O2": [2, 4],
            "O3": [5, 5],
            "O4": [6, 6],
            "O5": [7, 7],
            "D1": [3, 1],
            "D2": [8, 3],
            "D3": [9, 9],
            "D4": [10, 10],
            "D5": [11, 11],
        }
    )

    possessions = summarize_player_possessions(lineups, season=2025).set_index(
        "player_id"
    )

    assert possessions.loc[1, "offensive_possessions"] == 1
    assert possessions.loc[1, "defensive_possessions"] == 1
    assert possessions.loc[1, "regular_season_possessions"] == 1.0
    assert possessions.loc[2, "offensive_possessions"] == 2
    assert possessions.loc[2, "defensive_possessions"] == 0
    assert possessions.loc[2, "regular_season_possessions"] == 1.0


def test_career_export_sums_realized_points_instead_of_epm_rates() -> None:
    scored = pd.DataFrame(
        {
            "season": [2024, 2025, 2025],
            "player_id": [1, 1, 2],
            "player_name": ["Volume Star", "Volume Star", "Rate Star"],
            "title_lift_over_contract": [0.1, 0.2, 0.15],
            "scalar_title_lift": [0.1, 0.2, 0.15],
            "role_adjusted_title_lift": [0.1, 0.2, 0.15],
            "role_adjusted_title_lift_unanchored": [0.1, 0.2, 0.15],
            "scalar_epm_surplus": [4.0, 4.0, 8.0],
            "regular_season_possessions": [5_000.0, 2_500.0, 2_500.0],
            "realized_rs_points_over_contract": [200.0, 100.0, 200.0],
        }
    )

    career = build_career_table(scored).set_index("player_id")

    assert career.loc[1, "realized_rs_points_over_contract"] == 300.0
    assert career.loc[1, "surplus_epm_season_sum"] == 8.0
    assert career.loc[1, "realized_rs_points_rank"] == 1
    assert career.loc[2, "realized_rs_points_rank"] == 2
