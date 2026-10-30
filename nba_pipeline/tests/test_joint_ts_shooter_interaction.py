from __future__ import annotations

import numpy as np
import pandas as pd

from nba_pipeline.scripts.run_joint_ts_shooter_interaction import (
    OFFENSE_COLS,
    build_joint_design_matrix,
    build_results,
    collect_player_ids,
    compute_player_season_shot_shares,
    season_centered_target,
    select_role_alpha_one_se,
)


def fixture_attempts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [1, 1],
            "Focal_Player": [1, 2],
            "O1": [2, 1],
            "O2": [3, 3],
            "O3": [4, 4],
            "O4": [5, 5],
            "D1": [6, 6],
            "D2": [7, 7],
            "D3": [8, 8],
            "D4": [9, 9],
            "D5": [10, 10],
            "TS_Points": [2.0, 0.0],
            "Season": [2026, 2026],
        }
    )


def test_centered_shooter_interactions_sum_to_zero_by_player_season():
    df = fixture_attempts()
    shares = compute_player_season_shot_shares(df)
    player_ids = collect_player_ids(df)
    _, interaction = build_joint_design_matrix(
        df,
        player_ids,
        shares,
        off_alpha=3000,
        def_alpha=5000,
        role_alpha=3000,
    )

    audits = []
    for idx, column in enumerate(OFFENSE_COLS):
        audits.append(
            pd.DataFrame(
                {
                    "Season": df["Season"],
                    "player_id": df[column],
                    "interaction": interaction[:, idx],
                }
            )
        )
    totals = (
        pd.concat(audits)
        .groupby(["Season", "player_id"])["interaction"]
        .sum()
    )
    assert totals.abs().max() < 1e-12


def test_joint_parent_reconciles_to_direct_role_contributions():
    df = fixture_attempts()
    player_ids = collect_player_ids(df)
    n_players = len(player_ids)
    beta = np.zeros(3 * n_players)
    player_one = int(np.where(player_ids == 1)[0][0])
    beta[player_one] = 0.02 * np.sqrt(3000)
    beta[2 * n_players + player_one] = 0.06 * np.sqrt(3000)

    result, _ = build_results(
        df,
        player_ids,
        beta,
        off_alpha=3000,
        def_alpha=5000,
        role_alpha=3000,
    )
    row = result[result["player_id"].eq(1)].iloc[0]
    assert abs(row["joint_ots_decomp_gap"]) < 1e-12
    assert np.isclose(
        row["conditional_self_shot_ots"]
        - row["conditional_teammate_shot_ots"],
        row["shooter_role_contrast_rapm"],
    )


def test_season_target_is_centered_independently():
    df = fixture_attempts()
    df.loc[1, "Season"] = 2025
    centered, baselines = season_centered_target(df)
    assert np.allclose(centered, 0.0)
    assert baselines.set_index("Season")["baseline_per_attempt"].to_dict() == {
        2025: 0.0,
        2026: 2.0,
    }


def test_one_se_rule_prefers_stronger_indistinguishable_shrinkage():
    cv = pd.DataFrame(
        {
            "role_alpha": [100.0, 500.0, 3000.0],
            "mean_mse": [1.0000, 1.0004, 1.0020],
            "std_mse": [0.0010, 0.0010, 0.0010],
        }
    )
    selected, annotated = select_role_alpha_one_se(cv, folds=4)
    assert selected == 500.0
    assert annotated.set_index("role_alpha").loc[500.0, "within_one_se"]
    assert not annotated.set_index("role_alpha").loc[3000.0, "within_one_se"]
