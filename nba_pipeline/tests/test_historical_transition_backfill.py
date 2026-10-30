import numpy as np
import pandas as pd
from pathlib import Path

from nba_pipeline.scripts.score_historical_transition import (
    apply_logit_calibrator,
    build_era_audit,
    build_group_crossfit_calibration,
    era_label,
    feature_drift_rows,
    fit_logit_calibrator,
    parse_raw_filename,
    retrospective_calibration,
)


def test_parse_raw_filename_and_eras():
    assert parse_raw_filename(Path("NBA97.parquet")) == (1997, "rs")
    assert parse_raw_filename(Path("NBA00_PS.parquet")) == (2000, "ps")
    assert parse_raw_filename(Path("NBA26.parquet")) == (2026, "rs")
    assert era_label(1999) == "1997-2003"
    assert era_label(2018) == "2017-2020"
    assert era_label(2026) == "2025-2026"


def test_logit_calibration_moves_mean_toward_observed_rate():
    raw_probability = np.array([0.1, 0.2, 0.3, 0.4] * 50)
    target = np.array([0, 0, 1, 1] * 50)
    params = fit_logit_calibrator(raw_probability, target)
    calibrated = apply_logit_calibrator(raw_probability, params)
    assert 0 <= calibrated.min() <= calibrated.max() <= 1
    assert abs(calibrated.mean() - target.mean()) < abs(
        raw_probability.mean() - target.mean()
    )


def test_retrospective_policy_never_claims_historical_season_calibration():
    artifact = {
        "targets": {
            "is_transition": {
                "pooled_retrospective": {
                    "parameters": {"intercept": 5.0, "slope": 6.0},
                    "operating_point": {"threshold": 0.42},
                },
            }
        }
    }
    params, scope, threshold = retrospective_calibration(
        artifact, "is_transition"
    )
    assert params["intercept"] == 5.0
    assert scope == "retrospective_pooled_2025_2026_for_unlabeled_season"
    assert threshold == 0.42


def test_crossfit_calibrator_never_fits_same_game_outcomes():
    games = np.repeat([f"game_{index}" for index in range(10)], 4)
    frame = pd.DataFrame(
        {
            "calibration_season": 2026,
            "game_id": games,
            "target": np.tile([0, 1, 0, 1], 10),
            "raw_probability": np.tile([0.1, 0.7, 0.2, 0.8], 10),
        }
    )
    predictions, assignment, folds = build_group_crossfit_calibration(
        frame, folds=5, seed=7
    )
    assert np.isfinite(predictions).all()
    assert (assignment >= 0).all()
    for payload in folds:
        assert set(payload["fit_game_ids"]).isdisjoint(
            payload["validation_game_ids"]
        )
        for prefixed_game_id in payload["validation_game_ids"]:
            game_id = prefixed_game_id.split("::", 1)[1]
            row_indices = frame.index[frame["game_id"].eq(game_id)]
            assert set(assignment[row_indices]) == {payload["fold"]}


def test_era_audit_uses_shot_weighting():
    files = pd.DataFrame(
        {
            "era": ["1997-2003", "1997-2003"],
            "season_type": ["rs", "rs"],
            "season_end_year": [1997, 1998],
            "shot_rows": [100, 300],
            "games": [10, 20],
            "severe_drift_features": [2, 3],
            "mean_feature_missing_rate": [0.2, 0.4],
            "transition_probability_mean": [0.1, 0.2],
            "transition_label_050_rate": [0.05, 0.10],
            "transition_label_operating_rate": [0.10, 0.20],
            "transition_rim_probability_mean": [0.04, 0.08],
            "transition_rim_label_050_rate": [0.02, 0.04],
            "transition_rim_label_operating_rate": [0.03, 0.06],
        }
    )
    era = build_era_audit(files).iloc[0]
    assert era["shot_rows"] == 400
    assert era["games"] == 30
    assert era["severe_drift_feature_file_pairs"] == 5
    assert np.isclose(era["transition_probability_mean"], 0.175)


def test_feature_drift_marks_constant_reference_change_without_huge_z_score():
    features = pd.DataFrame({"action_code": [3.0, 4.0, 5.0]})
    reference = pd.DataFrame(
        {
            "feature": ["action_code"],
            "reference_missing_rate": [0.0],
            "reference_mean": [0.0],
            "reference_std": [0.0],
        }
    )
    row = feature_drift_rows(
        features,
        reference,
        source_file="NBA97.parquet",
        year=1997,
        season_type="rs",
    )[0]
    assert row["standardized_mean_shift"] is None
    assert row["reference_constant_change"] is True
    assert row["severe_drift_flag"] is True
