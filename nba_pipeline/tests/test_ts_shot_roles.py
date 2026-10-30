from __future__ import annotations

import numpy as np
import pandas as pd

from nba_pipeline.scripts.process_rapm_blocks.process_ts_shot_roles import (
    build_ts_shot_role_rows,
)
from nba_pipeline.scripts.run_ts_shot_role_hierarchy import (
    anchor_conditional_ts_effects,
    reconcile_ts_hierarchy,
    shapley_ts_components,
)


def test_ts_attempt_expands_to_one_self_and_four_teammate_rows():
    ts = pd.DataFrame(
        {
            "game_id": [22500001],
            "event_num": [14],
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
            "Net_Diff": [2.8],
            "Season": [2026],
        }
    )
    raw = pd.DataFrame(
        {
            "game_id": ["0022500001"],
            "event_num": [14],
            "player1_id": [3],
        }
    )
    self_rows, teammate_rows = build_ts_shot_role_rows(ts, raw)

    assert len(self_rows) == 1
    assert len(teammate_rows) == 4
    assert self_rows["Focal_Player"].tolist() == [3]
    assert set(self_rows[["O1", "O2", "O3", "O4"]].iloc[0]) == {1, 2, 4, 5}
    assert set(teammate_rows["Focal_Player"]) == {1, 2, 4, 5}
    assert teammate_rows["TS_Points"].eq(2.8).all()
    for row in teammate_rows.itertuples():
        context = {row.O1, row.O2, row.O3, row.O4}
        assert row.Focal_Player not in context
        assert 3 in context


def test_conditional_anchor_and_shapley_components_close_exactly():
    parent = pd.Series([8.0, -3.0])
    contrast = pd.Series([10.0, -4.0])
    share = pd.Series([0.30, 0.10])
    self_rate, teammate_rate = anchor_conditional_ts_effects(
        parent, contrast, share
    )
    anchored = share * self_rate + (1.0 - share) * teammate_rate
    assert np.allclose(anchored, parent)

    self_component, teammate_component, mix_component = shapley_ts_components(
        self_rate, teammate_rate, share, reference_share=0.2
    )
    assert np.allclose(
        self_component + teammate_component + mix_component, parent
    )


def test_reconciled_ts_hierarchy_reports_zero_gaps():
    result = reconcile_ts_hierarchy(
        pd.DataFrame(
            {
                "ots_rapm": [8.0],
                "self_minus_teammate_ts_rapm": [10.0],
                "shot_share": [0.30],
            }
        )
    )
    assert abs(result["conditional_anchor_gap"].iloc[0]) < 1e-12
    assert abs(result["ts_role_decomp_gap"].iloc[0]) < 1e-12
