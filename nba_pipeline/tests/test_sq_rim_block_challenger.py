import numpy as np
import pandas as pd

from nba_pipeline.scripts.train_sq_rim_block_challenger import (
    add_game_shifted_binary_prior,
    attach_block_labels_and_lineups,
)


def test_game_shifted_binary_prior_excludes_the_whole_current_game():
    frame = pd.DataFrame(
        {
            "personId": [7, 7, 7, 7],
            "game_date": pd.to_datetime(
                ["2025-01-01", "2025-01-01", "2025-01-03", "2025-01-03"]
            ),
            "game_id_normalized": ["1", "1", "2", "2"],
            "blocked": [1, 0, 1, 1],
        }
    )
    result = add_game_shifted_binary_prior(
        frame,
        player_column="personId",
        target_column="blocked",
        prefix="shooter_prior_block",
        shrink_attempts=2.0,
        baseline=0.25,
    )

    assert np.allclose(
        result.loc[:1, "shooter_prior_block_probability"], 0.25
    )
    # Game 2 sees Game 1's one block in two attempts, plus the two-attempt
    # baseline prior. Neither outcome from Game 2 enters either Game 2 row.
    assert np.allclose(
        result.loc[2:, "shooter_prior_block_probability"],
        (1.0 + 2.0 * 0.25) / 4.0,
    )


def test_block_join_uses_opposing_lineup_and_drops_ambiguous_outcome():
    shots = pd.DataFrame(
        {
            "game_id_normalized": ["22500001", "22500001"],
            "GAME_EVENT_ID": pd.Series([10, 11], dtype="Int64"),
            "personId": [101, 101],
            "made": [False, True],
        }
    )
    raw = pd.DataFrame(
        {
            "game_id_normalized": ["22500001", "22500001"],
            "GAME_EVENT_ID": pd.Series([10, 11], dtype="Int64"),
            "player1_id": [101, 101],
            "blocked": [True, True],
            **{f"home_player{i}": [100 + i, 100 + i] for i in range(1, 6)},
            **{f"away_player{i}": [200 + i, 200 + i] for i in range(1, 6)},
        }
    )

    result, audit = attach_block_labels_and_lineups(shots, raw)

    assert len(result) == 1
    assert bool(result.loc[0, "blocked"])
    assert result.loc[0, [f"D{i}" for i in range(1, 6)]].tolist() == [
        201,
        202,
        203,
        204,
        205,
    ]
    assert audit["rim_rows_made_and_blocked_removed"] == 1
    assert audit["rim_defense_lineup_coverage"] == 1.0
