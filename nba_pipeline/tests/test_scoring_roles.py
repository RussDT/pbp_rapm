from __future__ import annotations

import pandas as pd

from nba_pipeline.scripts.process_rapm_blocks.process_scoring_roles import (
    build_scoring_role_rows,
)
from nba_pipeline.scripts.run_scoring_role_hierarchy import (
    build_scoring_contrast_input,
    reconcile_scoring_components,
)


def test_scoring_role_rows_reconcile_onball_and_offball_to_parent():
    terminal = pd.DataFrame(
        {
            "game_id": ["0000000001"],
            "poss_group": [0],
            "O1": [1],
            "O2": [2],
            "O3": [3],
            "O4": [4],
            "O5": [5],
            "D1": [6],
            "D2": [7],
            "D3": [8],
            "D4": [9],
            "D5": [10],
            "Scoring_Points": [3.0],
        }
    )
    scorer_points = pd.DataFrame(
        {
            "game_id": ["0000000001"],
            "poss_group": [0],
            "Focal_Player": [3],
            "OnBall_Scoring_Points": [3.0],
        }
    )
    rows = build_scoring_role_rows(terminal, scorer_points)
    assert len(rows) == 5
    assert rows["OnBall_Scoring_Points"].sum() == 3.0
    assert rows["OffBall_Scoring_Points"].sum() == 3.0
    assert rows.loc[rows["Focal_Player"].eq(3), "OffBall_Scoring_Points"].iloc[0] == 0
    assert rows.loc[~rows["Focal_Player"].eq(3), "OffBall_Scoring_Points"].eq(0.75).all()
    assert rows["Role_Resolution_Complete"].all()


def test_scoring_contrast_and_parent_reconciliation():
    common = {
        "game_id": ["0000000001"],
        "poss_group": [0],
        "Focal_Player": [1],
        "O1": [2],
        "O2": [3],
        "O3": [4],
        "O4": [5],
        "D1": [6],
        "D2": [7],
        "D3": [8],
        "D4": [9],
        "D5": [10],
        "Role_Resolution_Complete": [True],
    }
    onball = pd.DataFrame({**common, "OnBall_Scoring_Points": [3.0]})
    offball = pd.DataFrame({**common, "OffBall_Scoring_Points": [0.0]})
    contrast = build_scoring_contrast_input(onball, offball)
    assert contrast["OnBall_OffBall_Contrast"].iloc[0] == 3.0

    result = reconcile_scoring_components(
        pd.DataFrame(
            {
                "offensive_rapm": [5.0],
                "onball_minus_offball_rapm": [3.0],
            }
        )
    )
    assert result["onball_scoring_rapm"].iloc[0] == 4.0
    assert result["offball_scoring_rapm"].iloc[0] == 1.0
    assert result["scoring_role_decomp_gap"].iloc[0] == 0.0
