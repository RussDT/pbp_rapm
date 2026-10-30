from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_darko_teammate_help.py"
SPEC = importlib.util.spec_from_file_location("build_darko_teammate_help", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def history(dates, seasons, values):
    return MODULE.PlayerRatingHistory(
        dates=np.asarray(dates, dtype="datetime64[D]"),
        seasons=np.asarray(seasons, dtype=np.int16),
        values=np.asarray(values, dtype=float),
    )


def test_lookup_is_strictly_asof_and_carries_prior_snapshot():
    histories = {
        1: history(
            ["2023-05-01", "2024-05-01"],
            [2023, 2024],
            [[1.0, 0.6, 0.4], [3.0, 2.0, 1.0]],
        )
    }
    values, dates, seasons = MODULE.lookup_asof_ratings(
        np.array([1, 1, 1]),
        np.array(["2023-04-30", "2023-12-01", "2024-06-01"], dtype="datetime64[D]"),
        histories,
    )

    assert np.isnan(values[0]).all()
    np.testing.assert_allclose(values[1], [1.0, 0.6, 0.4])
    np.testing.assert_allclose(values[2], [3.0, 2.0, 1.0])
    assert np.isnat(dates[0])
    assert dates[1] == np.datetime64("2023-05-01")
    assert seasons.tolist() == [-1, 2023, 2024]


def test_lookup_same_season_applies_later_row_back_to_full_season():
    histories = {
        1: history(
            ["2023-05-01", "2024-05-01"],
            [2023, 2024],
            [[1.0, 0.6, 0.4], [3.0, 2.0, 1.0]],
        ),
        2: history(
            ["2023-05-01"],
            [2023],
            [[2.0, 1.0, 1.0]],
        ),
    }
    values, dates, seasons = MODULE.lookup_same_season_ratings(
        np.array([1, 1, 2]),
        2024,
        histories,
    )

    np.testing.assert_allclose(values[:2], [[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    assert np.isnan(values[2]).all()
    assert dates[0] == np.datetime64("2024-05-01")
    assert np.isnat(dates[2])
    assert seasons.tolist() == [2024, 2024, -1]


def test_help_is_other_four_sum_and_pairs_reconcile_exactly():
    exposures = pd.DataFrame(
        {
            "game_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "P1": [1, 1],
            "P2": [2, 2],
            "P3": [3, 3],
            "P4": [4, 4],
            "P5": [5, 5],
            "exposures": [2.0, 3.0],
        }
    )
    histories = {
        player_id: history(
            ["2024-05-01"],
            [2024],
            [[float(player_id), player_id / 2.0, player_id / 2.0]],
        )
        for player_id in range(1, 6)
    }

    player, pair = MODULE.summarize_lineup_exposures(exposures, histories, season_end_year=2024)
    anchor = player.set_index("player_id").loc[1]
    assert anchor["total_exposures"] == 5.0
    assert anchor["rated_exposures"] == 5.0
    assert anchor["help_dpm"] == 14.0
    assert anchor["help_o_dpm"] == 7.0
    assert anchor["help_d_dpm"] == 7.0
    assert anchor["prior_snapshot_pct"] == 0.0
    assert np.isnan(anchor["avg_rating_lag_days"])

    anchor_pairs = pair[pair["player_id"] == 1]
    assert anchor_pairs["shared_exposures"].sum() == 20.0
    assert abs(anchor_pairs["contribution_dpm"].sum() - anchor["help_dpm"]) < 1e-12
    assert abs(anchor_pairs["contribution_o_dpm"].sum() - anchor["help_o_dpm"]) < 1e-12
    assert abs(anchor_pairs["contribution_d_dpm"].sum() - anchor["help_d_dpm"]) < 1e-12


def test_asof_audit_mode_retains_prior_season_and_lag_diagnostics():
    exposures = pd.DataFrame(
        {
            "game_date": pd.to_datetime(["2024-01-01"]),
            "P1": [1],
            "P2": [2],
            "P3": [3],
            "P4": [4],
            "P5": [5],
            "exposures": [2.0],
        }
    )
    histories = {
        player_id: history(
            ["2023-12-31"],
            [2023],
            [[float(player_id), player_id / 2.0, player_id / 2.0]],
        )
        for player_id in range(1, 6)
    }

    player, _ = MODULE.summarize_lineup_exposures(
        exposures,
        histories,
        season_end_year=2024,
        rating_timing="asof",
    )
    anchor = player.set_index("player_id").loc[1]
    assert anchor["help_dpm"] == 14.0
    assert anchor["prior_snapshot_pct"] == 100.0
    assert anchor["avg_rating_lag_days"] == 1.0


def test_missing_teammate_invalidates_only_anchors_who_need_that_rating():
    exposures = pd.DataFrame(
        {
            "game_date": pd.to_datetime(["2024-01-01"]),
            "P1": [1],
            "P2": [2],
            "P3": [3],
            "P4": [4],
            "P5": [5],
            "exposures": [7.0],
        }
    )
    histories = {
        player_id: history(
            ["2024-05-01"],
            [2024],
            [[float(player_id), player_id / 2.0, player_id / 2.0]],
        )
        for player_id in range(1, 5)
    }

    player, pair = MODULE.summarize_lineup_exposures(exposures, histories)
    indexed = player.set_index("player_id")
    assert indexed.loc[5, "rated_exposures"] == 7.0
    assert indexed.loc[5, "help_dpm"] == 10.0
    assert indexed.loc[1:4, "rated_exposures"].sum() == 0.0
    assert pair[pair["player_id"] == 5]["contribution_dpm"].sum() == 10.0


def test_collapse_team_exposures_rejects_partial_and_duplicate_lineups():
    possessions = pd.DataFrame(
        {
            "game_date": ["2024-01-01"] * 3,
            "O1": [1, 1, 1],
            "O2": [2, 2, 2],
            "O3": [3, 3, 3],
            "O4": [4, 4, 4],
            "O5": [5, 0, 4],
            "D1": [6, 6, 6],
            "D2": [7, 7, 7],
            "D3": [8, 8, 8],
            "D4": [9, 9, 9],
            "D5": [10, 10, 10],
        }
    )
    collapsed = MODULE.collapse_team_exposures(possessions)
    assert collapsed["exposures"].sum() == 4.0
    assert len(collapsed) == 2


def test_career_pair_summary_pools_rs_and_playoff_exposure():
    player = pd.DataFrame(
        [
            {
                "season": 2024,
                "season_type": "rs",
                "player_id": 1,
                "total_exposures": 100.0,
                "rated_exposures": 80.0,
            },
            {
                "season": 2024,
                "season_type": "ps",
                "player_id": 1,
                "total_exposures": 20.0,
                "rated_exposures": 20.0,
            },
        ]
    )
    pair = pd.DataFrame(
        [
            {
                "season": 2024,
                "season_type": "rs",
                "player_id": 1,
                "teammate_id": 2,
                "teammate_name": "Teammate",
                "shared_exposures": 50.0,
                "rated_exposures": 40.0,
                "weighted_dpm": 80.0,
                "weighted_o_dpm": 48.0,
                "weighted_d_dpm": 32.0,
            },
            {
                "season": 2024,
                "season_type": "ps",
                "player_id": 1,
                "teammate_id": 2,
                "teammate_name": "Teammate",
                "shared_exposures": 10.0,
                "rated_exposures": 10.0,
                "weighted_dpm": 30.0,
                "weighted_o_dpm": 20.0,
                "weighted_d_dpm": 10.0,
            },
        ]
    )

    regular = MODULE.build_career_pair_summary(player, pair, "rs").iloc[0]
    combined = MODULE.build_career_pair_summary(player, pair, "combined").iloc[0]

    assert regular["shared_exposures"] == 50.0
    assert combined["shared_exposures"] == 60.0
    assert combined["rated_exposures"] == 50.0
    assert combined["share_of_player_exposures_pct"] == 50.0
    assert combined["avg_teammate_dpm"] == 2.2
    assert combined["contribution_dpm"] == 1.1
