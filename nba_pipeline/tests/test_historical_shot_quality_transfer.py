import numpy as np
import pandas as pd

from nba_pipeline.scripts.build_historical_shot_quality_transfer import (
    HISTORICAL_CATEGORICAL,
    HISTORICAL_NUMERIC,
    apply_probability_calibrator,
    chronological_game_split,
    fit_probability_calibrator,
    paired_game_loss_intervals,
    score_student_rows,
    train_transfer_models,
)


def _synthetic_rows() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(42)
    for bucket in ["midrange", "three"]:
        for game_idx in range(12):
            for shot_idx in range(20):
                distance = (16 + shot_idx % 6) if bucket == "midrange" else (23 + shot_idx % 5)
                initial = 0.42 if bucket == "midrange" else 0.36
                special = (shot_idx + game_idx) % 3 == 0
                true_p = initial + (0.08 if special else -0.02)
                made = rng.random() < true_p
                rows.append(
                    {
                        "game_id": f"g{game_idx:02d}",
                        "season_end_year": 2026,
                        "season_type": "RS",
                        "model_bucket": bucket,
                        "is_make": made,
                        "shot_value": 2.0 if bucket == "midrange" else 3.0,
                        "initial_make_prob": initial,
                        "dynamic_make_prob": initial + (0.02 if special else 0.0),
                        "shot_zone": "long_mid" if bucket == "midrange" else "three",
                        "distance_bin": "15_18" if bucket == "midrange" else "23_25",
                        "shot_family": "jumper" if bucket == "midrange" else "three",
                        "modifier_bundle": "pullup" if special else "none",
                        "event_action_type_str": "1",
                        "distance_source": "parsed_description",
                        "offense_side": "Home",
                        "shot_clock_reset_reason": "turnover",
                        "shot_clock_confidence": "high",
                        "shot_distance_ft": float(distance),
                        "shot_clock_est": float(18 - shot_idx % 12),
                        "shot_clock_fraction": float(18 - shot_idx % 12) / 24.0,
                        "elapsed_since_reset": float(6 + shot_idx % 12),
                        "effective_reset_len": 24.0,
                        "period": 1 + game_idx % 4,
                        "seconds_remaining_quarter": float(700 - shot_idx * 20),
                        "sq_desc_pull_up": float(special),
                        "shooter_id": 100 + shot_idx % 8,
                        "source_file": "NBA26.parquet",
                    }
                )
    return pd.DataFrame(rows)


def test_temporal_split_keeps_games_intact_and_latest_games_held_out():
    rows = _synthetic_rows()
    train, test = chronological_game_split(rows, test_fraction=0.25)
    train_games = set(rows.loc[train, "game_id"])
    test_games = set(rows.loc[test, "game_id"])
    assert train_games.isdisjoint(test_games)
    assert min(test_games) > max(train_games)


def test_probability_calibrator_is_bounded():
    raw = np.array([0.1, 0.2, 0.7, 0.9])
    y = np.array([0, 0, 1, 1])
    calibrator = fit_probability_calibrator(raw, y)
    calibrated = apply_probability_calibrator(raw, calibrator)
    assert np.all(calibrated > 0)
    assert np.all(calibrated < 1)
    assert calibrated[0] < calibrated[-1]


def test_paired_intervals_cluster_on_complete_games():
    rows = pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g2", "g2"],
            "is_make": [1, 0, 1, 0],
            "initial_probability": [0.5, 0.5, 0.5, 0.5],
            "candidate": [0.8, 0.2, 0.8, 0.2],
        }
    )
    intervals = paired_game_loss_intervals(rows, "candidate", bootstrap_repetitions=100)
    assert intervals["brier_improvement_ci_low"] > 0
    assert intervals["log_loss_improvement_ci_low"] > 0


def test_transfer_models_are_separate_and_student_scores_without_sq_fields():
    rows = _synthetic_rows()
    artifact, heldout, metrics = train_transfer_models(rows)
    assert set(artifact["models"]) == {"midrange", "three"}
    assert set(metrics["model_bucket"]) == {"midrange", "three"}
    assert {"initial_probability", "teacher_probability", "student_probability"}.issubset(
        heldout.columns
    )
    for bucket in artifact["models"].values():
        assert "is_make" not in bucket["teacher_numeric"]
        assert "made_assisted_tag" not in bucket["teacher_numeric"]
        assert bucket["student_categorical"] == HISTORICAL_CATEGORICAL
        assert bucket["student_numeric"] == HISTORICAL_NUMERIC

    historical_only = rows.drop(
        columns=[
            "initial_make_prob",
            "dynamic_make_prob",
            "sq_desc_pull_up",
            "is_make",
        ]
    )
    scored = score_student_rows(historical_only, artifact)
    assert scored["historical_sq_make_prob"].between(0, 1).all()
    assert scored["historical_sq_ev"].notna().all()
    assert scored["historical_sq_direct_make_prob"].between(0, 1).all()
    assert scored["historical_sq_distilled_make_prob"].between(0, 1).all()
    assert scored["historical_sq_component_gap"].ge(0).all()
