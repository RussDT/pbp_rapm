import numpy as np
import pandas as pd
import pytest

from wnba_test.build_wnba_sq_net_targets import (
    MID_RESIDUAL_COLUMN,
    NET_SHOT_COLUMNS,
    THREE_RESIDUAL_COLUMN,
    attach_sq_net_targets,
    prepare_sq_table,
)
from wnba_test.tune_wnba_daily_net_hyperparams import build_component_targets


def _first_chance_fixture() -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "game_id": [1, 1, 1, 1],
            "event_num": [20, 40, 60, 80],
            "Is_Turnover": [0, 0, 0, 0],
            "Net_Diff": [0.0, 2.0, 0.0, 3.0],
            "FC_TS_NET_Diff": [0.3, 2.0, 0.25, 3.0],
            "FC_CONT_KAPPA_Diff": [0.3, 0.0, 0.25, 0.0],
            "FC_FT_Diff": [0.0, 0.0, 0.0, 0.0],
            "FC_NET_RIM_EFG_FREQ_Diff": [0.0] * 4,
            "FC_NET_RIM_EFG_FG_Diff": [0.0] * 4,
            "FC_NET_MID_EFG_FREQ_Diff": [0.0] * 4,
            "FC_NET_MID_EFG_FG_Diff": [-0.6, 1.1, 0.0, 0.0],
            "FC_NET_THREE_EFG_FREQ_Diff": [0.0] * 4,
            "FC_NET_THREE_EFG_FG_Diff": [0.0, 0.0, -0.95, 1.8],
        }
    )
    return out


def _raw_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [1, 1, 1, 1, 1, 1, 1, 1],
            "event_num": [10, 20, 30, 40, 50, 60, 70, 80],
            "period": [1] * 8,
            "minute_remaining_quarter": [11, 11, 10, 10, 9, 9, 8, 8],
            "seconds_remaining_quarter": [40, 35, 40, 35, 40, 35, 40, 35],
            "event_type": [
                "MISS",
                "Turnover",
                "MAKE",
                "Turnover",
                "MISS",
                "Turnover",
                "MAKE",
                "Turnover",
            ],
            "home_description": [
                "MISS Player 15' Jump Shot",
                "Turnover",
                "Player 12' Jump Shot (2 PTS)",
                "Turnover",
                "MISS Player 25' 3PT Jump Shot",
                "Turnover",
                "Player 25' 3PT Jump Shot (3 PTS)",
                "Turnover",
            ],
            "visitor_description": [None] * 8,
        }
    )


def _sq_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [1, 1, 1, 1],
            "event_num": [10, 30, 50, 70],
            "shot_zone": ["short_mid", "long_mid", "three", "three"],
            "shot_value": [2.0, 2.0, 3.0, 3.0],
            "shot_context_ev": [0.8, 1.0, 1.1, 1.2],
            "shot_quality_prior_ev": [0.85, 1.05, 1.15, 1.25],
        }
    )


def test_wnba_sq_net_split_maps_raw_first_fga_and_closes_exactly() -> None:
    original = _first_chance_fixture()
    swapped, stats = attach_sq_net_targets(
        original,
        _raw_fixture(),
        _sq_fixture(),
        ev_column="shot_context_ev",
    )

    original_sum = original[NET_SHOT_COLUMNS].sum(axis=1)
    swapped_sum = (
        swapped[NET_SHOT_COLUMNS].sum(axis=1)
        + swapped[MID_RESIDUAL_COLUMN]
        + swapped[THREE_RESIDUAL_COLUMN]
    )
    np.testing.assert_allclose(swapped_sum, original_sum, atol=1e-12)
    assert stats["sq_attached_rows"] == 4
    assert stats["max_abs_net_sq_closure_gap"] < 1e-12
    np.testing.assert_allclose(
        swapped["FC_NET_RIM_EFG_FG_Diff"],
        original["FC_NET_RIM_EFG_FG_Diff"],
    )


def test_wnba_sq_conflicting_keys_fail_loudly() -> None:
    duplicated = pd.concat(
        [_sq_fixture().iloc[[0]], _sq_fixture().iloc[[0]].assign(shot_context_ev=1.1)],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="conflicting duplicate"):
        prepare_sq_table(duplicated, "shot_context_ev")


def test_structural_tuning_targets_keep_tov_and_close_to_actual() -> None:
    frame = _first_chance_fixture()
    frame["_sc_net"] = [0.1, 0.0, 0.2, 0.0]
    frame.loc[0, "Is_Turnover"] = 1
    frame.loc[0, "FC_TS_NET_Diff"] = 0.0
    frame["_actual_points"] = frame["FC_TS_NET_Diff"] + frame["_sc_net"]

    prepared, targets = build_component_targets(frame, "net")

    np.testing.assert_allclose(
        prepared[targets].sum(axis=1),
        prepared["_actual_points"],
        atol=1e-12,
    )
    assert "target_TOV" in targets
    assert "target_NET_BASE" in targets
    assert prepared.loc[0, "target_TOV"] < 0.0
