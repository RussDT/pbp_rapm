from __future__ import annotations

import numpy as np
import pandas as pd

from nba_pipeline.scripts.build_second_chance_opportunity_prior import (
    aggregate_player_zone_accounting,
    game_cluster_bootstrap_mean,
    reporting_shot_zone,
    role_swap_counterfactual,
    second_chance_ev_per_attempt,
)


def test_role_swap_counterfactual_reassigns_shooter_and_crash_roles_exactly():
    result = role_swap_counterfactual(
        actual_raw_prediction=0.20,
        shooter_id=10,
        non_shooter_ids=[20, 30, 40, 50],
        shot_zone="long_mid",
        non_shooter_coefficients={
            10: 0.01,
            20: 0.03,
            30: -0.01,
            40: 0.02,
            50: 0.00,
        },
        shooter_zone_coefficients={
            (10, "long_mid"): -0.04,
            (20, "long_mid"): 0.02,
            (30, "long_mid"): 0.00,
            (40, "long_mid"): -0.01,
            (50, "long_mid"): 0.01,
        },
        clip_bounds=(0.0, 1.0),
    )

    # Swapping teammate 20 into the shot removes 20 from the crash group,
    # adds shooter 10 to it, and replaces the shooter-zone effect:
    # 0.20 + (0.01 - 0.03) + (0.02 - -0.04) = 0.24.
    assert np.isclose(result.swap_predictions[0], 0.24)
    assert np.isclose(result.actual_probability, 0.20)
    assert np.isclose(result.equal_teammate_swap_probability, 0.245)

    # The two named components must reconcile on the unbounded linear scale.
    assert np.isclose(
        result.actual_minus_swap_raw,
        result.crash_group_delta_raw + result.shooter_zone_delta_raw,
    )
    assert np.isclose(result.actual_minus_swap_raw, -0.045)


def test_second_chance_ev_identity_prices_only_the_miss_branch():
    actual_ev = second_chance_ev_per_attempt(
        make_probability=0.45,
        offensive_rebound_probability=0.16,
        points_per_offensive_rebound=1.10,
    )
    swap_ev = second_chance_ev_per_attempt(
        make_probability=0.45,
        offensive_rebound_probability=0.20,
        points_per_offensive_rebound=1.10,
    )

    assert np.isclose(actual_ev, 0.55 * 0.16 * 1.10)
    assert np.isclose(swap_ev, 0.55 * 0.20 * 1.10)
    assert np.isclose(actual_ev - swap_ev, -0.0242)


def test_reporting_zone_combines_both_long_two_geometry_buckets():
    zones = pd.Series(["rim", "short_mid", "long_mid", "deep_two", "three"])
    assert reporting_shot_zone(zones).tolist() == [
        "rim",
        "short_mid",
        "long_two",
        "long_two",
        "three",
    ]


def test_game_bootstrap_keeps_clustered_positive_gain_positive():
    result = game_cluster_bootstrap_mean(
        game_ids=["a", "a", "b", "b", "c", "c"],
        row_values=[0.01, 0.02, 0.03, 0.04, 0.02, 0.01],
        repetitions=200,
        seed=7,
    )
    assert np.isclose(result["point"], 0.021666666666666667)
    assert result["ci95_low"] > 0.0
    assert result["ci95_high"] > result["ci95_low"]


def test_player_zone_accounting_closes_to_attempt_weighted_ev():
    opportunities = pd.DataFrame(
        {
            "shooter_id": [10, 10],
            "shooter_name": ["Example", "Example"],
            "shot_zone": ["long_mid", "long_mid"],
            "actual_oreb": [0.0, 1.0],
            "actual_config_probability_oof": [0.16, 0.18],
            "role_swap_probability_oof": [0.20, 0.22],
            "context_lineup_probability_oof": [0.19, 0.21],
            "crash_group_delta_raw": [-0.01, -0.01],
            "shooter_zone_delta_raw": [-0.03, -0.03],
            "points_per_oreb_oof": [1.10, 1.10],
        }
    )
    attempts = pd.DataFrame(
        {
            "shooter_id": [10],
            "shooter_name": ["Example"],
            "shot_zone": ["long_mid"],
            "attempts": [5],
            "shot_quality_prior_make_prob": [0.40],
            "shot_quality_season_make_prob": [0.44],
        }
    )

    out = aggregate_player_zone_accounting(opportunities, attempts)
    row = out.iloc[0]

    assert row["attempts"] == 5
    assert row["rebound_opportunity_misses"] == 2
    assert np.isclose(row["actual_config_oreb_probability_oof"], 0.17)
    assert np.isclose(row["role_swap_oreb_probability_oof"], 0.21)
    assert np.isclose(row["actual_config_sc_ev_per_attempt"], 0.60 * 0.17 * 1.10)
    assert np.isclose(row["role_swap_sc_ev_per_attempt"], 0.60 * 0.21 * 1.10)
    assert np.isclose(
        row["role_vacancy_sc_points_total"],
        row["role_vacancy_sc_ev_per_attempt"] * 5,
    )
    assert np.isclose(
        row["actual_config_sc_ev_per_attempt_season_sq"],
        (1.0 - 0.44) * 0.17 * 1.10,
    )
    assert np.isclose(
        row["role_vacancy_sc_points_total_season_sq"],
        row["role_vacancy_sc_ev_per_attempt_season_sq"] * 5,
    )
