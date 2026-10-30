from __future__ import annotations

import numpy as np
import pandas as pd

from nba_pipeline.scripts.run_primary_defender_net_decomp import (
    DEFENSE_COLS,
    ROLE_COLS,
    build_centered_primary_roles,
    build_design_matrix,
    build_reader_summary,
    decompose_defensive_role,
    select_paired_role_alpha_one_se,
)


def assignment_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "_processed_row_id": [0, 1, 2, 3],
            "game_id": ["1", "1", "2", "2"],
            "primary_defender_id": [1, 2, 1, 2],
            "D1": [1, 1, 1, 1],
            "D2": [2, 2, 2, 2],
            "D3": [3, 3, 3, 3],
            "D4": [4, 4, 4, 4],
            "D5": [5, 5, 5, 5],
        }
    )


def test_primary_roles_center_exactly_for_every_defender():
    long, shares, max_gap = build_centered_primary_roles(assignment_fixture())
    assert max_gap < 1e-12
    totals = long.groupby("player_id")["primary_role"].sum()
    np.testing.assert_allclose(totals, 0.0, atol=1e-12)
    share_lookup = shares.set_index("player_id")["primary_share"]
    assert share_lookup.loc[1] == 0.5
    assert share_lookup.loc[2] == 0.5
    assert share_lookup.loc[3] == 0.0


def test_primary_and_nonprimary_contributions_close_to_defensive_parent():
    result = decompose_defensive_role(
        parent_allowed=-0.8,
        contrast_allowed=-1.2,
        primary_share=0.25,
    )
    assert abs(result["decomposition_gap"]) < 1e-12
    assert np.isclose(
        result["primary_conditional_impact"]
        - result["nonprimary_conditional_impact"],
        result["primary_vs_nonprimary_contrast"],
    )
    assert result["def_parent_impact"] == 0.8


def test_design_matrix_uses_centered_role_values_on_defense_slots():
    frame = assignment_fixture()
    long, _, _ = build_centered_primary_roles(frame)
    wide = long.pivot(
        index="_processed_row_id", columns="slot", values="primary_role"
    ).rename(columns={slot: f"primary_role_D{slot}" for slot in range(1, 6)})
    frame = frame.merge(wide.reset_index(), on="_processed_row_id")
    for index, column in enumerate(["O1", "O2", "O3", "O4", "O5"], start=6):
        frame[column] = index
    player_ids = np.arange(1, 11, dtype=np.int64)
    matrix = build_design_matrix(
        frame,
        player_ids,
        off_alpha=1.0,
        def_alpha=1.0,
        role_alpha=1.0,
    )
    role_block = matrix[:, 2 * len(player_ids) :].toarray()
    for slot, defense_column in enumerate(DEFENSE_COLS):
        player_positions = frame[defense_column].to_numpy(dtype=int) - 1
        np.testing.assert_allclose(
            role_block[np.arange(len(frame)), player_positions],
            frame[ROLE_COLS[slot]].to_numpy(float),
        )


def test_paired_one_se_uses_fold_differences_not_absolute_fold_levels():
    cv = pd.DataFrame(
        {
            "role_alpha": [1000.0, 3000.0, 10000.0, 1.0e12],
            "mean_mse": [1.10002, 1.10000, 1.10008, 1.10018],
            "fold_1_mse": [1.00002, 1.00000, 1.00008, 1.00018],
            "fold_2_mse": [1.20002, 1.20000, 1.20008, 1.20018],
            "fold_3_mse": [1.10002, 1.10000, 1.10008, 1.10018],
        }
    )
    selected, annotated = select_paired_role_alpha_one_se(cv)
    assert selected == 3000.0
    lookup = annotated.set_index("role_alpha")
    assert lookup.loc[3000.0, "within_paired_one_se"]
    assert not lookup.loc[1.0e12, "within_paired_one_se"]


def test_reader_summary_filters_sorts_and_rounds_for_humans():
    summary = pd.DataFrame(
        {
            "player_id": [1, 2],
            "player_name": ["Full Name", "Small Sample"],
            "def_parent_impact": [1.23456, 9.87654],
            "primary_conditional_impact": [2.22222, 8.88888],
            "nonprimary_conditional_impact": [0.11111, 7.77777],
            "primary_vs_nonprimary_contrast": [2.11111, 1.11111],
            "primary_share": [0.23456, 0.5],
            "primary_assignments": [900, 50],
            "assignment_exposures": [4000, 100],
            "assignment_seasons": [4, 1],
        }
    )
    reader = build_reader_summary(summary)
    assert reader.to_dict("records") == [
        {
            "rank": 1,
            "player_id": 1,
            "player": "Full Name",
            "overall_net_shot_defense": 1.23,
            "primary_defense": 2.22,
            "nonprimary_defense": 0.11,
            "primary_advantage": 2.11,
            "primary_share_pct": 23.5,
            "primary_assignments": 900,
            "assignment_exposures": 4000,
            "seasons": 4,
        }
    ]
