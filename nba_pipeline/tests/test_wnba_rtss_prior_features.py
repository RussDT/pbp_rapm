import json

import numpy as np
import pandas as pd
import pytest

from wnba_test.build_wnba_rtss_playmaking_roles import (
    _conditional_role_frame,
    _extract_game,
    _q_probability,
)
from wnba_test.build_wnba_rtss_prior_features import (
    _aggregate_game,
    _feature_surface,
)
from wnba_test.evaluate_wnba_rtss_priors import Q_FEATURES, _weighted_metrics


def test_qsq_and_qsp_minus_qsq_are_mid_three_only(tmp_path):
    player = "shooter"
    payload = {
        "chances": [{"season": 2024}],
        "chance_players": [
            {"playerId": player, "offense": True, "involved": True},
            {"playerId": player, "offense": False, "involved": True},
        ],
        "shots": [
            {
                "shooterId": player,
                "three": False,
                "region": "left elbow",
                "qSQ": 40.0,
                "qSP": 45.0,
                "assistOpp": False,
            },
            {
                "shooterId": player,
                "three": True,
                "region": "left wing three",
                "qSQ": 35.0,
                "qSP": 37.0,
                "assistOpp": True,
            },
            {
                "shooterId": player,
                "three": False,
                "region": "ra",
                "qSQ": 60.0,
                "qSP": 70.0,
                "assistOpp": True,
                "assisted": True,
                "outcome": True,
            },
        ],
    }
    path = tmp_path / "basketball-markings.json"
    path.write_text(json.dumps(payload))
    raw = pd.DataFrame(_aggregate_game(str(path))).fillna(0.0)
    surface = _feature_surface(raw).set_index("provider_player_id").loc[player]

    assert surface["rtss_mid_qsq_epv"] == pytest.approx(0.8)
    assert surface["rtss_mid_shooter_adjustment_epv"] == pytest.approx(0.1)
    assert surface["rtss_three_qsq_epv"] == pytest.approx(1.05)
    assert surface["rtss_three_shooter_adjustment_epv"] == pytest.approx(0.06)
    assert surface["rtss_mid_assist_opp_share"] == pytest.approx(0.0)
    assert surface["rtss_three_assist_opp_share"] == pytest.approx(1.0)
    assert surface["rtss_rim_assist_opp_share"] == pytest.approx(1.0)
    assert surface["rtss_mid_self_created_rate100"] == pytest.approx(100.0)
    assert surface["rtss_mid_self_created_fg_pct"] == pytest.approx(0.0)
    assert surface["rtss_rim_assist_opp_fg_pct"] == pytest.approx(1.0)
    assert surface["rtss_three_assist_opp_qsq_epv"] == pytest.approx(1.05)
    assert all("RIM" not in prefix for prefix, _side in Q_FEATURES)


def test_weighted_metrics_reward_exact_predictions():
    result = _weighted_metrics(
        actual=pd.Series([1.0, 2.0, 3.0]).to_numpy(),
        prediction=pd.Series([1.0, 2.0, 3.0]).to_numpy(),
        weight=pd.Series([1.0, 2.0, 3.0]).to_numpy(),
    )
    assert result["rmse"] == pytest.approx(0.0)
    assert result["correlation"] == pytest.approx(1.0)


def test_on_ball_time_uses_exact_offensive_chance_seconds(tmp_path):
    payload = {
        "chances": [
            {
                "id": "chance",
                "season": 2024,
                "startGameClock": 100.0,
                "endGameClock": 90.0,
            }
        ],
        "chance_players": [
            {
                "chanceId": "chance",
                "playerId": "handler",
                "offense": True,
            }
        ],
        "touches": [
            {
                "playerId": "handler",
                "touchTime": 2.0,
            }
        ],
    }
    path = tmp_path / "basketball-markings.json"
    path.write_text(json.dumps(payload))
    raw = pd.DataFrame(_aggregate_game(str(path))).fillna(0.0)
    surface = _feature_surface(raw).set_index("provider_player_id").loc["handler"]

    assert surface["off_chance_seconds"] == pytest.approx(10.0)
    assert surface["rtss_on_ball_time_pct"] == pytest.approx(20.0)


def test_playmaking_extract_keeps_assisted_misses_and_exact_lineups(tmp_path):
    offense = [f"o{number}" for number in range(5)]
    defense = [f"d{number}" for number in range(5)]
    payload = {
        "chances": [{"id": "chance", "season": 2024}],
        "chance_players": [
            *[
                {
                    "chanceId": "chance",
                    "playerId": player_id,
                    "offense": True,
                }
                for player_id in offense
            ],
            *[
                {
                    "chanceId": "chance",
                    "playerId": player_id,
                    "offense": False,
                }
                for player_id in defense
            ],
        ],
        "shots": [
            {
                "id": "shot",
                "gameId": "game",
                "chanceId": "chance",
                "shooterId": "o0",
                "passerId": "o1",
                "assistOpp": True,
                "outcome": False,
                "three": True,
                "region": "middle three",
                "qSQ": 40.0,
            }
        ],
        "passes": [
            {
                "passerId": "o1",
                "assistOpp": True,
                "turnover": True,
            }
        ],
    }
    path = tmp_path / "basketball-markings.json"
    path.write_text(json.dumps(payload))

    result = _extract_game(str(path))

    assert result["shots"][0]["passer_id"] == "o1"
    assert result["shots"][0]["actual_points"] == pytest.approx(0.0)
    assert result["shots"][0]["provider_qsq_points"] == pytest.approx(1.2)
    assert result["diagnostics"]["potential_assisted_shots"] == 1
    assert result["pass_stats"][0]["pass_turnovers"] == 1


def test_centered_shooter_passer_presence_roles_close_to_parent():
    # One player's four coefficient blocks: offense, defense, shooter, passer.
    coefficients = np.array([0.4, -0.2, 0.8, 0.3])
    frame = _conditional_role_frame(
        season=2024,
        target_name="actual_points",
        players=["player"],
        exposure=np.array([100.0]),
        shooter_share=np.array([0.25]),
        passer_share=np.array([0.15]),
        coefficients=coefficients,
        intercept=1.0,
        base_alpha=3000.0,
        role_alpha=1000.0,
    ).iloc[0]

    assert frame["presence_share"] == pytest.approx(0.60)
    assert frame["shooter_delta_vs_presence"] == pytest.approx(0.8)
    assert frame["passer_delta_vs_presence"] == pytest.approx(0.3)
    assert (
        frame["shooter_contribution_per_shot"]
        + frame["passer_contribution_per_shot"]
        + frame["presence_contribution_per_shot"]
    ) == pytest.approx(frame["offensive_parent_per_shot"])
    assert frame["role_identity_gap"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(0.4, 0.4), (40.0, 0.4), (120.0, None), (-1.0, None)],
)
def test_provider_q_probability_normalizes_both_scales(raw, expected):
    assert _q_probability(raw) == expected
