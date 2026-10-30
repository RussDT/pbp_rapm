import numpy as np
import pandas as pd
import pytest

from nba_pipeline.scripts.build_vpm_target_swap import (
    PUBLIC_SHOT_COLUMNS,
    attach_ev_targets,
    prepare_ev_table,
)
from nba_pipeline.scripts.rapm import (
    compute_alt_first_chance_baselines,
    detect_file_type_and_prepare,
)


def _first_chance_fixture():
    children = {
        "FC_DECOMP_RIM_FREQ_Diff": [0.0, 0.0],
        "FC_DECOMP_RIM_FG_Diff": [0.0, 0.0],
        "FC_DECOMP_MID_FG_Diff": [0.0, 0.8],
        "FC_DECOMP_THREE_FREQ_Diff": [0.0, 0.3],
        "FC_DECOMP_THREE_FG_Diff": [0.0, 1.9],
    }
    out = pd.DataFrame(
        {
            "game_id": ["1", "1"],
            "event_num": [200, 50],
            "Net_Diff": [0.0, 3.0],
            "Is_Turnover": [0, 0],
            "FC_FT_Diff": [0.0, 0.0],
            **children,
        }
    )
    out["FC_DECOMP_EFG_Diff"] = out[PUBLIC_SHOT_COLUMNS].sum(axis=1)
    out["FC_DECOMP_MID_VALUE_Diff"] = 0.0
    return out


def _raw_fixture():
    # Event 50 is a corrected low-number terminal action after event 300.
    # Clock-first assignment must still map event 300 to that possession.
    return pd.DataFrame(
        {
            "game_id": ["1", "1", "1", "1"],
            "event_num": [10, 200, 300, 50],
            "period": [1, 1, 1, 1],
            "minute_remaining_quarter": [11, 11, 10, 10],
            "seconds_remaining_quarter": [30, 20, 55, 50],
            "event_type": [2, 5, 1, 5],
            "event_action_type": [0, 0, 0, 0],
            "home_description": [
                "MISS Player 15' Pullup Jump Shot",
                "Turnover",
                "Player 25' 3PT Jump Shot (3 PTS)",
                "Turnover",
            ],
            "visitor_description": [None, None, None, None],
        }
    )


def _ev_fixture():
    return pd.DataFrame(
        {
            "game_id": ["0000000001", "0000000001"],
            "event_num": [10, 300],
            "zone": ["mid", "three"],
            "ev_pts": [0.8, 1.2],
        }
    )


def test_ev_target_swap_uses_clock_first_possessions_and_closes_exactly():
    original = _first_chance_fixture()
    out, stats = attach_ev_targets(original, _raw_fixture(), _ev_fixture())

    np.testing.assert_allclose(out["VPM_MID_RESID_Diff"], [-0.8, 0.0])
    np.testing.assert_allclose(out["VPM_THREE_RESID_Diff"], [0.0, 1.8])
    np.testing.assert_allclose(out["FC_DECOMP_MID_FG_Diff"], [0.8, 0.8])
    np.testing.assert_allclose(out["FC_DECOMP_THREE_FG_Diff"], [0.0, 0.1])

    original_sum = original[PUBLIC_SHOT_COLUMNS].sum(axis=1)
    vpm_sum = (
        out[PUBLIC_SHOT_COLUMNS].sum(axis=1)
        + out["FC_DECOMP_EFG_Diff"]
        + out["FC_DECOMP_MID_VALUE_Diff"]
    )
    np.testing.assert_allclose(vpm_sum, original_sum, atol=1e-12)
    assert stats["terminal_keys_found_in_raw"] == 2
    assert stats["ev_attached_rows"] == 2
    assert stats["max_abs_vpm_shot_closure_gap"] < 1e-12


def test_missing_ev_keeps_actual_target_and_zero_residual():
    original = _first_chance_fixture()
    out, stats = attach_ev_targets(
        original,
        _raw_fixture(),
        _ev_fixture().iloc[:1],
    )

    assert out.loc[1, "FC_DECOMP_THREE_FG_Diff"] == pytest.approx(1.9)
    assert out.loc[1, "FC_DECOMP_EFG_Diff"] == pytest.approx(0.0)
    assert stats["three_swapped"] == 0


def test_conflicting_duplicate_ev_keys_fail_loudly():
    ev = pd.concat(
        [
            _ev_fixture().iloc[:1],
            _ev_fixture().iloc[:1].assign(ev_pts=1.1),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="conflicting duplicate"):
        prepare_ev_table(ev)


def test_solver_facing_vpm_components_still_sum_to_actual_first_chance():
    out, _ = attach_ev_targets(
        _first_chance_fixture(),
        _raw_fixture(),
        _ev_fixture(),
    )
    turnover = out.iloc[[0]].copy()
    turnover["event_num"] = 999
    turnover["Net_Diff"] = 0.0
    turnover["Is_Turnover"] = 1
    turnover[PUBLIC_SHOT_COLUMNS] = 0.0
    turnover["FC_FT_Diff"] = 0.0
    turnover["FC_DECOMP_EFG_Diff"] = 0.0
    turnover["FC_DECOMP_MID_VALUE_Diff"] = 0.0
    out = pd.concat([out, turnover], ignore_index=True)

    components = [
        "DECOMP_RIM_FREQ",
        "DECOMP_RIM_FG",
        "DECOMP_MID_FG",
        "DECOMP_THREE_FREQ",
        "DECOMP_THREE_FG",
        "ALT_FT",
        "ALT_TOV_VALUE",
        "DECOMP_EFG",
        "DECOMP_MID_VALUE",
    ]
    baselines = compute_alt_first_chance_baselines(
        out,
        requested_prefixes=components,
    )
    component_targets = [
        detect_file_type_and_prepare(
            out,
            prefix=component,
            alt_baselines=baselines,
        )["Off_Diff"]
        for component in components
    ]

    np.testing.assert_allclose(sum(component_targets), out["Net_Diff"], atol=1e-12)
