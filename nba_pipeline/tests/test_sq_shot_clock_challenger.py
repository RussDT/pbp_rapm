from __future__ import annotations

import numpy as np
import pandas as pd

from nba_pipeline.scripts.train_sq_shot_clock_challenger import (
    attach_clock_value_columns,
    attach_possession_value_columns,
    build_clock_feature_matrix,
    summarize_player_clock_burden,
)


def test_neutral_clock_counterfactual_changes_clock_features_not_shot_anchor():
    shots = pd.DataFrame(
        {
            "shot_quality_prior_make_prob": [0.40, 0.50],
            "shot_clock_calibrated": [2.0, 18.0],
            "shot_clock_calibrated_lower": [0.0, 16.0],
            "shot_clock_calibrated_upper": [5.0, 20.0],
            "confidence_score": [0.9, 0.9],
            "needs_review": [False, False],
            "game_clock_below_24": [False, False],
            "shot_value": [2.0, 3.0],
            "shot_zone": ["long_mid", "three"],
            "reset_reason": ["made_fg", "def_rebound"],
        }
    )

    actual = build_clock_feature_matrix(shots)
    neutral = build_clock_feature_matrix(shots, clock_override=14.0)

    np.testing.assert_allclose(actual["baseline_logit"], neutral["baseline_logit"])
    np.testing.assert_allclose(neutral["clock_centered"], 0.0)
    np.testing.assert_allclose(neutral["late_7_seconds"], 0.0)
    np.testing.assert_allclose(neutral["grenade_4_seconds"], 0.0)
    assert actual.loc[0, "grenade_4_seconds"] == 2.0
    assert actual.loc[1, "clock_centered"] == 4.0


def test_clock_value_columns_preserve_conditioned_plus_burden_identity():
    shots = pd.DataFrame(
        {
            "shot_value": [2.0, 3.0],
            "shot_clock_calibrated": [3.5, 12.0],
        }
    )
    out = attach_clock_value_columns(
        shots,
        conditioned_probability=np.array([0.35, 0.40]),
        neutral_probability=np.array([0.42, 0.39]),
        grenade_threshold=4.0,
    )

    np.testing.assert_allclose(out["clock_conditioned_ev"], [0.70, 1.20])
    np.testing.assert_allclose(out["neutral_clock_ev"], [0.84, 1.17])
    np.testing.assert_allclose(out["clock_drag_ev"], [-0.14, 0.03])
    np.testing.assert_allclose(
        out["clock_conditioned_ev"] + out["grenade_burden_ev"],
        out["neutral_clock_ev"],
    )
    assert out["is_grenade"].tolist() == [True, False]


def test_player_summary_keeps_grenade_burden_separate_from_all_clock_drag():
    shots = pd.DataFrame(
        {
            "shooter_id": [10, 10, 10, 20],
            "shooter_name": ["Example", "Example", "Example", "Other"],
            "is_make": [0, 1, 1, 0],
            "shot_value": [2.0, 2.0, 3.0, 2.0],
            "is_grenade": [True, True, False, True],
            "clock_conditioned_ev": [0.6, 0.8, 1.2, 0.7],
            "neutral_clock_ev": [0.8, 0.9, 1.1, 0.8],
            "clock_drag_ev": [-0.2, -0.1, 0.1, -0.1],
            "grenade_burden_ev": [0.2, 0.1, -0.1, 0.1],
        }
    )

    summary = summarize_player_clock_burden(shots)
    row = summary[summary["shooter_id"].eq(10)].iloc[0]

    assert row["attempts"] == 3
    assert row["grenade_attempts"] == 2
    assert np.isclose(row["grenade_rate"], 2 / 3)
    assert np.isclose(row["clock_drag_ev_total"], -0.2)
    assert np.isclose(row["grenade_burden_ev_total"], 0.3)
    assert np.isclose(row["grenade_burden_ev_per_grenade"], 0.15)


def test_possession_value_adds_expected_rebound_continuation_and_closes():
    shots = pd.DataFrame(
        {
            "clock_conditioned_make_probability_oof": [0.35],
            "neutral_clock_make_probability_oof": [0.42],
            "shot_value": [2.0],
            "actual_config_oreb_probability_oof": [0.20],
            "points_per_oreb_oof": [1.10],
        }
    )
    out = attach_possession_value_columns(shots)
    row = out.iloc[0]

    assert np.isclose(row["clock_conditioned_second_chance_ev"], 0.143)
    assert np.isclose(row["neutral_clock_second_chance_ev"], 0.1276)
    assert np.isclose(row["clock_conditioned_possession_ev"], 0.843)
    assert np.isclose(row["neutral_clock_possession_ev"], 0.9676)
    assert np.isclose(row["grenade_shot_burden_ev"], 0.14)
    assert np.isclose(row["second_chance_recovery_offset_ev"], 0.0154)
    assert np.isclose(row["grenade_possession_burden_ev"], 0.1246)
    assert np.isclose(
        row["clock_conditioned_possession_ev"]
        + row["grenade_possession_burden_ev"],
        row["neutral_clock_possession_ev"],
    )
