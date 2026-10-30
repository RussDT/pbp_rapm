from __future__ import annotations

import numpy as np
import pandas as pd

from nba_pipeline.scripts.compare_sq_second_chance_forward import (
    add_release_value_predictions,
    build_release_map_from_frames,
    derive_non_shooter_lineups,
    train_only_conversion_by_segment,
)


def test_release_map_assigns_first_and_second_fga_to_same_terminal():
    raw = pd.DataFrame(
        {
            "game_id": ["1"] * 5,
            "event_num": [1, 2, 3, 4, 5],
            "event_type": [2, 4, 2, 4, 1],
            "period": [1] * 5,
            "minute_remaining_quarter": [11] * 5,
            "seconds_remaining_quarter": [50, 48, 42, 40, 30],
        }
    )
    terminals = pd.DataFrame({"game_id": ["1", "1"], "event_num": [4, 5]})
    second_chance = pd.DataFrame(
        {
            "game_id": ["1", "1"],
            "event_num": [4, 5],
            "Off_Diff": [2.0, 0.0],
        }
    )

    release, audit = build_release_map_from_frames(
        raw, terminals, second_chance
    )

    assert release["event_num"].tolist() == [1, 3, 5]
    assert release["terminal_event_num"].tolist() == [4, 4, 5]
    assert release["fga_rank"].tolist() == [0, 1, 0]
    assert release["realized_second_chance_points"].tolist() == [2.0, 2.0, 0.0]
    assert audit["terminal_rows_found"] == 2


def test_non_shooter_derivation_removes_shooter_and_rejects_bad_defense():
    shots = pd.DataFrame(
        {
            "shooter_id": [10, 20],
            "O1": [10, 20],
            "O2": [20, 30],
            "O3": [30, 40],
            "O4": [40, 50],
            "O5": [50, 60],
            "D1": [101, 101],
            "D2": [102, 102],
            "D3": [103, 103],
            "D4": [104, 104],
            "D5": [105, 0],
        }
    )

    valid, audit = derive_non_shooter_lineups(shots)

    assert len(valid) == 1
    assert valid[["O1", "O2", "O3", "O4"]].iloc[0].tolist() == [
        20,
        30,
        40,
        50,
    ]
    assert audit["invalid_lineup_rows"] == 1


def test_train_only_conversion_prices_realized_points_per_initial_oreb():
    training = pd.DataFrame(
        {
            "shot_zone": ["three", "three", "long_mid"],
            "actual_oreb": [1.0, 0.0, 1.0],
            "realized_second_chance_points": [1.2, 0.0, 1.4],
        }
    )
    conversion = train_only_conversion_by_segment(training)
    assert np.isclose(conversion["three"], 1.2)
    assert np.isclose(conversion["mid"], 1.4)


def test_release_value_adds_only_miss_weighted_continuation():
    shots = pd.DataFrame(
        {
            "segment": ["three"],
            "made": [0],
            "shot_value": [3.0],
            "realized_second_chance_points": [2.0],
            "oreb_probability_context_lineup": [0.20],
            "oreb_probability_actual_config": [0.16],
            "oreb_probability_role_swap": [0.24],
            "p_initial_ev": [0.40],
            "p_context_epm": [0.42],
            "p_context_epm_def_geometry_identity": [0.41],
        }
    )

    out = add_release_value_predictions(
        shots, conversions={"three": 1.10}
    )
    row = out.iloc[0]

    assert np.isclose(row["immediate_ev::initial_ev"], 1.20)
    assert np.isclose(
        row["second_chance_ev::initial_ev::actual_config"],
        0.60 * 0.16 * 1.10,
    )
    assert np.isclose(
        row["release_ev::initial_ev::actual_config"],
        1.20 + 0.60 * 0.16 * 1.10,
    )
    assert np.isclose(
        row["role_vacancy_sc_ev::initial_ev"],
        0.60 * (0.16 - 0.24) * 1.10,
    )
