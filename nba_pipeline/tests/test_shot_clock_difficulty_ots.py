import numpy as np
import pandas as pd

from nba_pipeline.scripts.analyze_shot_clock_difficulty_ots import (
    _prepare_outcomes,
    aggregate_player_clock_stats,
    build_zts_role_numbers,
    crossfit_clock_expectation,
)


def test_prepare_outcomes_floors_calibrated_tail_only():
    outcomes = pd.DataFrame(
        {
            "game_id": ["1", "2"],
            "season": [2024, 2024],
            "shooter_id": [10, 11],
            "shooter_name": ["A", "B"],
            "points": [2, 0],
            "tsa": [1.0, 1.0],
            "shot_clock_est": [1.0, 12.0],
            "clock": [0.5, 13.25],
            "rule_reset_len": [24, 14],
        }
    )
    prepared = _prepare_outcomes(outcomes)
    assert prepared["raw_rule_clock"].tolist() == [1.0, 12.0]
    assert prepared["calibrated_floor4_clock"].tolist() == [4.0, 13.25]


def test_crossfit_and_player_adjusted_ts_identity():
    rows = []
    for game in range(10):
        for event in range(20):
            clock = float(event % 10 + 1)
            rows.append(
                {
                    "game_id": str(game),
                    "season": 2024,
                    "shooter_id": 10 + event % 2,
                    "shooter_name": "A" if event % 2 == 0 else "B",
                    "points": 2.0 if clock >= 6 else 0.0,
                    "tsa": 1.0,
                    "shot_clock_est": clock,
                    "clock": clock,
                    "rule_reset_len": 24,
                }
            )
    prepared = _prepare_outcomes(pd.DataFrame(rows))
    scored = crossfit_clock_expectation(
        prepared, clock_column="raw_rule_clock", n_splits=5
    )
    assert scored[["expected_tsx2", "league_tsx2"]].notna().all().all()
    scored["clock_used"] = scored["raw_rule_clock"]
    players = aggregate_player_clock_stats(scored, "raw")
    gap = (
        players["raw_raw_ts_pct"]
        + players["raw_shot_clock_difficulty_pp"]
        - players["raw_clock_adjusted_ts_pct"]
    )
    assert np.allclose(gap, 0.0, atol=1e-10)


def test_zts_role_numbers_preserve_role_volume_and_weighted_rts(tmp_path):
    playtype_path = tmp_path / "playtype_rows.csv"
    zts_path = tmp_path / "zts.csv"
    pd.DataFrame(
        {
            "year": [2026, 2026, 2026],
            "nba_id": [10, 10, 10],
            "player_name": ["A", "A", "A"],
            "playtype_group": ["creation", "creation", "transition"],
            "TSA": [30.0, 10.0, 20.0],
            "TSA100": [6.0, 2.0, 4.0],
            "ts_percent": [0.50, 0.70, 0.80],
            "playtype_avg_ts": [0.55, 0.60, 0.65],
            "playtype_rTS": [-0.05, 0.10, 0.15],
        }
    ).to_csv(playtype_path, index=False)
    pd.DataFrame(
        {
            "year": [2026],
            "nba_id": [10],
            "playtype_weighted_rTS": [0.05],
            "playtype_diff": [1.25],
            "pt_adj_rTS": [3.5],
        }
    ).to_csv(zts_path, index=False)

    result = build_zts_role_numbers(
        playtype_path, zts_path, start_year=2026, end_year=2026
    )
    creation = result[result["role"].eq("creation")].iloc[0]
    assert creation["role_tsa"] == 40.0
    assert creation["role_tsa100"] == 8.0
    assert np.isclose(creation["role_attempt_share"], 2 / 3)
    assert np.isclose(creation["role_actual_ts_pct"], 55.0)
    assert np.isclose(creation["role_rts_pct_pts"], -1.25)
    assert creation["pt_adj_rTS"] == 3.5
