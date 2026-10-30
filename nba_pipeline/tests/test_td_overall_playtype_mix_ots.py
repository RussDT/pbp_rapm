import numpy as np
import pandas as pd
import pytest

from nba_pipeline.scripts.analyze_td_overall_playtype_mix_ots import (
    MIX_CATEGORIES,
    classify_mix_shots,
    compute_weighted_player_season_shares,
    estimate_lag_prices,
    time_decay_weights,
)


def test_transition_precedes_description_shot_family():
    frame = pd.DataFrame(
        {
            "home_description": [
                "Player 2' Cutting Layup Shot",
                "Player 26' Pullup 3PT Jump Shot",
            ],
            "visitor_description": ["", ""],
        }
    )
    result = classify_mix_shots(
        frame,
        pd.Series([False, True]),
        pd.Series([True, False]),
    )
    assert result.tolist() == ["TRANSITION", "SELF_CREATED_3"]


def test_lag_prices_use_only_supplied_pricing_outcomes():
    pricing = pd.DataFrame(
        {
            "mix_category": ["TRANSITION"] * 100 + ["OTHER_2"] * 100,
            "actual_points": [2.0] * 100 + [0.0] * 100,
        }
    )
    prices = estimate_lag_prices(
        pricing,
        pricing_season=2024,
        feature_season=2025,
        shrink_fga=0,
    ).set_index("mix_category")
    assert set(prices.index) == set(MIX_CATEGORIES)
    assert prices.loc["TRANSITION", "difficulty_pp"] == pytest.approx(-50.0)
    assert prices.loc["OTHER_2", "difficulty_pp"] == pytest.approx(50.0)


def test_time_decay_weighted_shot_share_centers_usage_mechanically():
    frame = pd.DataFrame(
        {
            "Season": [2025, 2025, 2025, 2025],
            "Focal_Player": [1, 1, 2, 2],
            "O1": [2, 2, 1, 1],
            "O2": [3, 3, 3, 3],
            "O3": [4, 4, 4, 4],
            "O4": [5, 5, 5, 5],
        }
    )
    weights = np.array([1.0, 0.5, 1.0, 0.5])
    shares = compute_weighted_player_season_shares(frame, weights).set_index(
        "player_id"
    )
    assert shares.loc[1, "shot_share"] == pytest.approx(0.5)
    assert shares.loc[2, "shot_share"] == pytest.approx(0.5)
    for player_id in [1, 2, 3, 4, 5]:
        interaction_sum = 0.0
        for row, weight in zip(frame.itertuples(index=False), weights):
            offense = [row.Focal_Player, row.O1, row.O2, row.O3, row.O4]
            if player_id in offense:
                interaction_sum += weight * (
                    float(row.Focal_Player == player_id)
                    - shares.loc[player_id, "shot_share"]
                )
        assert interaction_sum == pytest.approx(0.0)


def test_half_life_weight_is_one_half():
    dates = pd.Series(["2025-01-01", "2026-01-01"])
    weights = time_decay_weights(dates, 365.0)
    assert weights.tolist() == pytest.approx([0.5, 1.0])

