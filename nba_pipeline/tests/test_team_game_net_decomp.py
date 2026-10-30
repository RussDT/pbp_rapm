from __future__ import annotations

import numpy as np
import pandas as pd

import nba_pipeline.scripts.build_team_game_net_decomp as builder


def _component_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    first_chance = pd.DataFrame(
        {
            "game_id": [42500301, 42500301, 42500301],
            "period": [1, 1, 1],
            "time_quarter": ["11:30", "11:00", "10:30"],
            "event_num": [10, 20, 30],
            "game_date": ["2026-05-19"] * 3,
            "O1": [1, 11, 1],
            "O2": [2, 12, 2],
            "O3": [3, 13, 3],
            "O4": [4, 14, 4],
            "O5": [5, 15, 5],
            "Net_Diff": [2.0, 1.0, 0.0],
            "Is_Turnover": [0, 0, 1],
            "FC_TS_NET_Diff": [2.0, 1.0, 0.0],
            "FC_DECOMP_NET_RIM_FREQ_Diff": [0.2, 0.0, 0.0],
            "FC_DECOMP_NET_RIM_FG_Diff": [0.3, 0.2, 0.0],
            "FC_DECOMP_NET_MID_FG_Diff": [0.5, 0.4, 0.0],
            "FC_DECOMP_NET_THREE_FREQ_Diff": [0.1, 0.1, 0.0],
            "FC_DECOMP_NET_THREE_FG_Diff": [0.4, 0.1, 0.0],
            "FC_FT_Diff": [0.5, 0.2, 0.0],
            "FC_RUSSELL_SQ_V2_Diff": [1.2, 0.7, 0.0],
            "FC_RUSSELL_MAKE_V2_Diff": [0.3, 0.1, 0.0],
        }
    )
    second_chance = pd.DataFrame(
        {
            "game_id": [42500301, 42500301, 42500301],
            "period": [1, 1, 1],
            "time_quarter": ["11:30", "11:00", "10:30"],
            "event_num": [10, 20, 30],
            "Off_Diff": [0.0, 1.0, 0.0],
            "SC_NET_CONT_Diff": [0.0, 1.0, 0.0],
        }
    )
    return first_chance, second_chance


def test_row_components_close_both_views() -> None:
    first_chance, second_chance = _component_fixture()
    rows = builder.build_row_components(first_chance, second_chance)

    net_cols = [column for column in rows if column.startswith("netcont__")]
    sq_cols = [column for column in rows if column.startswith("sq__")]
    np.testing.assert_allclose(rows[net_cols].sum(axis=1), rows["modeled_points"], atol=1e-12)
    np.testing.assert_allclose(rows[sq_cols].sum(axis=1), rows["modeled_points"], atol=1e-12)

    # A turnover receives the typical component opportunity and then loses the
    # same full counterfactual in the turnover bucket.
    turnover_row = rows.loc[rows["Is_Turnover"].eq(1)].iloc[0]
    assert np.isclose(turnover_row["netcont__turnover_value"], -1.5)
    assert np.isclose(turnover_row[net_cols].drop("netcont__turnover_value").sum(), 1.5)


def test_russell_v2_first_fga_swap_preserves_first_chance_identity() -> None:
    first_chance = pd.DataFrame(
        {
            "game_id": [42500301],
            "event_num": [20],
            "Has_FC_FGA": [1],
            "Net_Diff": [2.0],
            "FC_FT_Diff": [0.0],
            "FC_DECOMP_EFG_Diff": [2.0],
            "FC_DECOMP_MID_VALUE_Diff": [0.0],
            "FC_DECOMP_RIM_FREQ_Diff": [0.0],
            "FC_DECOMP_RIM_FG_Diff": [1.0],
            "FC_DECOMP_MID_FG_Diff": [1.0],
            "FC_DECOMP_THREE_FREQ_Diff": [0.0],
            "FC_DECOMP_THREE_FG_Diff": [0.0],
        }
    )
    raw = pd.DataFrame(
        {
            "game_id": [42500301, 42500301],
            "period": [1, 1],
            "time_quarter": ["11:30", "11:20"],
            "minute_remaining_quarter": [11, 11],
            "seconds_remaining_quarter": [30, 20],
            "event_num": [10, 20],
            "event_type": ["MAKE", "Rebound"],
            "home_description": ["Player 2PT Shot (2 PTS)", "Rebound"],
            "visitor_description": [None, None],
        }
    )
    shots = pd.DataFrame(
        {
            "game_id": [42500301],
            "event_num": [10],
            "shot_quality_season_ev": [1.2],
        }
    )

    out, stats = builder.attach_russell_v2_first_chance(first_chance, raw, shots)

    assert out.loc[0, "FC_RUSSELL_SQ_V2_Diff"] == 1.2
    assert out.loc[0, "FC_RUSSELL_MAKE_V2_Diff"] == 0.8
    assert stats["russell_v2_values_attached"] == 1
    assert stats["max_abs_russell_v2_first_chance_gap"] < 1e-12


def test_game_metadata_and_offense_assignment_use_raw_lineup_sides() -> None:
    raw = pd.DataFrame(
        {
            "game_id": ["0042500301", "0042500301"],
            "home_description": ["Home shot", None],
            "visitor_description": [None, "Away shot"],
            "player1_team_abbreviation": ["NYK", "CLE"],
            "away_score": [0.0, 2.0],
            "home_score": [2.0, 2.0],
            **{f"home_player{i}": [i, i] for i in range(1, 6)},
            **{f"away_player{i}": [i + 10, i + 10] for i in range(1, 6)},
        }
    )
    metadata = builder.build_game_metadata(raw)
    assert metadata["0042500301"].home_team == "NYK"
    assert metadata["0042500301"].away_team == "CLE"

    first_chance, _ = _component_fixture()
    teams = builder.assign_offense_teams(
        first_chance.assign(game_id_norm="0042500301"), metadata
    )
    assert teams.tolist() == ["NYK", "CLE", "NYK"]


def test_matchup_outputs_close_to_actual_score_with_ft_bridge() -> None:
    first_chance, second_chance = _component_fixture()
    rows = builder.build_row_components(first_chance, second_chance)
    raw = pd.DataFrame(
        {
            "game_id": ["0042500301", "0042500301"],
            "home_description": ["Home shot", None],
            "visitor_description": [None, "Away shot"],
            "player1_team_abbreviation": ["NYK", "CLE"],
            "away_score": [0.0, 2.0],
            "home_score": [2.0, 3.0],
            **{f"home_player{i}": [i, i] for i in range(1, 6)},
            **{f"away_player{i}": [i + 10, i + 10] for i in range(1, 6)},
        }
    )
    metadata = builder.build_game_metadata(raw)
    summary, components, series = builder.build_matchup_outputs(rows, metadata, "NYK", "CLE")

    assert summary.loc[0, "score_margin"] == 1
    assert np.isclose(summary.loc[0, "ft_conversion_bridge_margin"], 1.0)
    for view in ("net_continuation", "shot_quality"):
        game_total = components.loc[components["view"].eq(view), "net_points"].sum()
        series_total = series.loc[series["view"].eq(view), "net_points"].sum()
        assert np.isclose(game_total, 1.0)
        assert np.isclose(series_total, 1.0)
