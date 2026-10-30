from __future__ import annotations

import numpy as np
import pandas as pd

from nba_pipeline.scripts.build_epm_game_value import (
    WinCurve,
    build_matched_pairs,
    fit_wins_per_point,
)


def test_wins_per_point_uses_centered_team_season_wins() -> None:
    rows = pd.DataFrame(
        {
            "gp": [82, 82, 82, 82],
            "wins": [51, 31, 46, 36],
            "point_diff": [300, -300, 150, -150],
        }
    )

    assert np.isclose(fit_wins_per_point(rows), 1.0 / 30.0)


def test_win_curve_is_centered_and_linear_mapping_is_bounded() -> None:
    curve = WinCurve(
        margin_intercept=2.3,
        strength_coefficient=0.9,
        logistic_k=0.1445,
        probit_sigma=11.4,
        linear_slope=0.0308,
    )

    assert np.isclose(curve.logistic(0.0), 0.5)
    assert np.isclose(curve.probit(0.0), 0.5)
    assert np.isclose(curve.linear(0.0), 0.5)
    assert curve.linear(100.0) == 1.0
    assert curve.linear(-100.0) == 0.0


def test_matched_pairs_requires_higher_impact_and_lower_minutes() -> None:
    players = pd.DataFrame(
        {
            "player_name": ["High", "Volume", "Wrong direction"],
            "epm": [6.0, 3.0, 1.0],
            "mpg": [28.0, 35.0, 20.0],
            "gp": [40, 55, 80],
            "war": [5.0, 5.04, 5.02],
            "game_margin": [3.5, 2.2, 0.4],
            "net_win_probability": [0.62, 0.58, 0.51],
            "seasontype": [2, 2, 2],
        }
    )

    pairs = build_matched_pairs(players)

    assert len(pairs) == 1
    assert pairs.iloc[0]["higher_impact_player"] == "High"
    assert pairs.iloc[0]["lower_impact_player"] == "Volume"
