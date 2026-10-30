from __future__ import annotations

import pandas as pd

from nba_pipeline.scripts.process_rapm_blocks.process_assist_point_roles import (
    expand_role_possessions,
    parse_assist_name,
    resolve_assister_id,
)
from nba_pipeline.scripts.run_assist_points_hierarchy import (
    build_role_contrast_input,
    reconcile_role_components,
)


def test_parse_and_resolve_assister_against_offensive_lineup():
    aliases = {
        1: {"nikola jokic", "jokic"},
        2: {"jamal murray", "murray"},
        3: {"aaron gordon", "gordon"},
    }
    description = "Gordon Cutting Dunk Shot (2 PTS) (Jokic 8 AST)"
    assert parse_assist_name(description) == "Jokic"
    assert resolve_assister_id("Jokic", [1, 2, 3, 4, 5], aliases) == 1


def test_expand_role_possessions_reconciles_parent_target():
    terminal = pd.DataFrame(
        {
            "game_id": ["0000000001", "0000000001"],
            "poss_group": [0, 1],
            "O1": [1, 6],
            "O2": [2, 7],
            "O3": [3, 8],
            "O4": [4, 9],
            "O5": [5, 10],
            "D1": [11, 1],
            "D2": [12, 2],
            "D3": [13, 3],
            "D4": [14, 4],
            "D5": [15, 5],
            "Assist_Points": [3, 0],
        }
    )
    role_points = pd.DataFrame(
        {
            "game_id": ["0000000001"],
            "poss_group": [0],
            "Focal_Player": [3],
            "Role_Assist_Points": [3],
        }
    )
    expanded = expand_role_possessions(
        terminal, role_points, "Receiver_Assist_Points"
    )
    assert len(expanded) == 10
    assert expanded.groupby(["game_id", "poss_group"])["Receiver_Assist_Points"].sum().tolist() == [3, 0]
    assert expanded.groupby(["game_id", "poss_group"])["Assist_Points"].first().tolist() == [3, 0]
    assert expanded.groupby(["game_id", "poss_group"])["Unresolved_Assist_Points"].first().tolist() == [0, 0]
    assert expanded.loc[expanded["Focal_Player"].eq(3), "Receiver_Assist_Points"].iloc[0] == 3
    assert all(
        int(row.Focal_Player) not in {int(row.O1), int(row.O2), int(row.O3), int(row.O4)}
        for row in expanded.itertuples()
    )


def test_role_contrast_uses_common_fully_resolved_sample():
    common = {
        "source_file": ["ASSIST_POINTS_ROLE26.parquet"] * 2,
        "game_id": ["0000000001"] * 2,
        "poss_group": [0, 1],
        "Focal_Player": [1, 1],
        "O1": [2, 2],
        "O2": [3, 3],
        "O3": [4, 4],
        "O4": [5, 5],
        "D1": [6, 6],
        "D2": [7, 7],
        "D3": [8, 8],
        "D4": [9, 9],
        "D5": [10, 10],
    }
    receiver = pd.DataFrame(
        {
            **common,
            "Receiver_Assist_Points": [0, 2],
            "Role_Resolution_Complete": [True, False],
        }
    )
    passer = pd.DataFrame(
        {
            **common,
            "Passer_Assist_Points": [3, 0],
            "Role_Resolution_Complete": [True, True],
        }
    )
    contrast = build_role_contrast_input(receiver, passer)
    assert len(contrast) == 1
    assert contrast["Passer_Receiver_Contrast"].iloc[0] == 3


def test_reconciled_role_components_close_to_parent_exactly():
    hierarchy = pd.DataFrame(
        {
            "assist_points_off_rapm": [4.459, -1.25],
            "passer_minus_receiver_rapm": [8.0, -0.75],
        }
    )
    result = reconcile_role_components(hierarchy)
    assert result["passer_assist_points_rapm"].round(10).tolist() == [6.2295, -1.0]
    assert result["receiver_assist_points_rapm"].round(10).tolist() == [-1.7705, -0.25]
    assert result["assist_points_decomp_gap"].abs().max() < 1e-12
