import numpy as np
import pandas as pd

from nba_pipeline.scripts.train_sq_walk_forward_challenger import (
    add_game_shifted_prior,
    metric_row,
    resolve_shot_rows,
)


def test_resolve_shot_rows_collapses_exact_and_excludes_conflicts():
    frame = pd.DataFrame(
        {
            "game_id": ["1", "1", "1", "2"],
            "GAME_EVENT_ID": [10, 10, 11, 20],
            "made": [1, 1, 0, 1],
            "value": [3.0, 3.0, 2.0, 2.0],
        }
    )
    conflict = pd.DataFrame(
        {
            "game_id": ["1"],
            "GAME_EVENT_ID": [11],
            "made": [0],
            "value": [9.0],
        }
    )
    resolved, audit = resolve_shot_rows(pd.concat([frame, conflict]))

    assert resolved[["game_id", "GAME_EVENT_ID"]].values.tolist() == [["1", 10], ["2", 20]]
    assert audit["exact_duplicate_rows_removed"] == 1
    assert audit["conflicting_event_keys_removed"] == 1
    assert audit["conflicting_rows_removed"] == 2


def test_game_shifted_prior_excludes_all_current_game_shots():
    frame = pd.DataFrame(
        {
            "personId": [7, 7, 7, 7],
            "segment": ["three"] * 4,
            "game_date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-03", "2024-01-03"]
            ),
            "game_id_normalized": ["1", "1", "2", "2"],
            "made": [1, 1, 0, 0],
        }
    )
    out = add_game_shifted_prior(
        frame,
        player_column="personId",
        prefix="shooter",
        shrink_attempts=2.0,
        baseline_by_segment={"three": 0.5},
    )

    np.testing.assert_allclose(
        out.loc[:1, "shooter_prior_make_probability"], [0.5, 0.5]
    )
    np.testing.assert_allclose(
        out.loc[2:, "shooter_prior_make_probability"], [0.75, 0.75]
    )
    np.testing.assert_allclose(out.loc[2:, "shooter_prior_log_attempts"], np.log1p(2))


def test_metric_row_reports_points_per_shot_not_only_percentages():
    frame = pd.DataFrame(
        {
            "season": [2026, 2026],
            "segment": ["three", "three"],
            "game_id_normalized": ["1", "2"],
            "made": [1, 0],
            "shot_value": [3.0, 3.0],
        }
    )
    row = metric_row(
        frame,
        np.array([0.6, 0.4]),
        model="test",
        coverage_slice="all_shots",
    )

    assert row["actual_pps"] == 1.5
    assert row["predicted_pps"] == 1.5
    assert row["pps_bias"] == 0.0
