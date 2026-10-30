import json

import numpy as np
import pandas as pd
import pytest

from wnba_test.train_wnba_rtss_pbp_student import (
    CONFIRMATION_SEASON,
    FORBIDDEN_STUDENT_FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    exact_teacher_join,
    extract_rtss_teacher_game,
    fit_student,
    add_2024_selected_blends,
    predict_student,
    q_expected_points,
)


def _student_frame() -> pd.DataFrame:
    rows = []
    for idx in range(240):
        shot_value = 3.0 if idx % 2 else 2.0
        row = {column: 0.0 for column in NUMERIC_FEATURES}
        row.update(
            {
                "shot_value": shot_value,
                "shot_distance": 24.0 if shot_value == 3.0 else 15.0,
                "period": 1 + idx % 4,
                "seconds_remaining_quarter": float(600 - idx % 600),
                "shot_zone": "three" if shot_value == 3.0 else "long_mid",
                "distance_bin": "23_25" if shot_value == 3.0 else "15_18",
                "shot_family": "jumper",
                "clock_bucket": "normal",
                "location_side": "center",
                "teacher_qsq_ev": 1.0 if shot_value == 3.0 else 0.8,
                "actual_points": shot_value if idx % 3 == 0 else 0.0,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def test_qsq_is_expected_efg_and_converts_to_expected_points():
    assert q_expected_points(50.0) == pytest.approx(1.0)
    assert q_expected_points(0.50) == pytest.approx(1.0)
    assert q_expected_points(None) is None


def test_rtss_extractor_keeps_only_mid_three_and_fences_2026(tmp_path):
    payload = {
        "shots": [
            {
                "id": "three",
                "season": 2025,
                "pbpId": 10,
                "period": 1,
                "three": True,
                "shotType": "jumper",
                "qSQ": 50.0,
                "qSP": 52.0,
                "outcome": True,
            },
            {
                "id": "rim",
                "season": 2025,
                "pbpId": 20,
                "period": 1,
                "three": False,
                "shotType": "layup",
                "qSQ": 60.0,
                "qSP": 61.0,
                "outcome": False,
            },
        ]
    }
    path = tmp_path / "basketball-markings.json"
    path.write_text(json.dumps(payload))
    rows = extract_rtss_teacher_game((str(path), "1022500001", 2025))
    assert [row["provider_shot_id"] for row in rows] == ["three"]
    assert rows[0]["teacher_qsq_ev"] == pytest.approx(1.0)

    payload["shots"][0]["season"] = CONFIRMATION_SEASON
    path.write_text(json.dumps(payload))
    assert extract_rtss_teacher_game((str(path), "1022600001", 2026)) == []


def test_exact_join_rejects_conflicts_instead_of_keep_first():
    teacher = pd.DataFrame(
        [
            {
                "season": 2025,
                "game_id": "1022500001",
                "event_num": 10,
                "period_rtss": 1,
                "shot_value_rtss": 3.0,
                "outcome_rtss": True,
                "teacher_qsq_ev": 1.0,
                "provider_qsp_ev": 1.1,
            },
            {
                "season": 2025,
                "game_id": "1022500001",
                "event_num": 10,
                "period_rtss": 1,
                "shot_value_rtss": 3.0,
                "outcome_rtss": True,
                "teacher_qsq_ev": 1.0,
                "provider_qsp_ev": 1.1,
            },
            {
                "season": 2025,
                "game_id": "1022500001",
                "event_num": 20,
                "period_rtss": 1,
                "shot_value_rtss": 2.0,
                "outcome_rtss": False,
                "teacher_qsq_ev": 0.8,
                "provider_qsp_ev": 0.8,
            },
        ]
    )
    pbp = pd.DataFrame(
        [
            {
                "game_id": "1022500001",
                "event_num": 10,
                "period": 1,
                "shot_value": 3.0,
                "is_make": True,
                "season_end_year": 2025,
            },
            {
                "game_id": "1022500001",
                "event_num": 20,
                "period": 1,
                "shot_value": 2.0,
                "is_make": False,
                "season_end_year": 2025,
            },
        ]
    )
    joined, audit = exact_teacher_join(teacher, pbp)
    assert joined["event_num"].tolist() == [20]
    assert audit["rtss_ambiguous_game_event_rows"] == 2


def test_student_features_exclude_outcomes_and_provider_fields():
    features = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    assert features.isdisjoint(FORBIDDEN_STUDENT_FEATURES)

    frame = _student_frame()
    model = fit_student(frame, "teacher_qsq_ev")
    prediction = predict_student(model, frame.iloc[:4])
    assert len(prediction) == 4
    assert np.isfinite(prediction).all()
    assert (prediction > 0).all()
    assert (prediction <= frame.iloc[:4]["shot_value"].to_numpy()).all()


def test_blend_weight_is_selected_on_2024_and_applied_to_2025():
    predictions = pd.DataFrame(
        {
            "season_end_year": [2024, 2024, 2025, 2025],
            "game_id": ["g1", "g2", "g3", "g4"],
            "event_num": [1, 2, 3, 4],
            "actual_points": [1.0, 1.0, 1.0, 1.0],
            "shot_value": [2.0, 2.0, 2.0, 2.0],
            "student_qsq_ev": [1.0, 1.0, 1.0, 1.0],
            "incumbent_context_ev": [0.0, 0.0, 0.0, 0.0],
            "incumbent_shooter_ev": [0.0, 0.0, 0.0, 0.0],
            "rtss_matched": [True, True, True, True],
        }
    )
    metrics, comparisons, scored, selections = add_2024_selected_blends(
        pd.DataFrame(),
        pd.DataFrame(),
        predictions,
        bootstrap_draws=20,
    )
    assert {row["student_weight"] for row in selections} == {1.0}
    assert scored.loc[
        scored["season_end_year"].eq(2025), "blend_context_student_ev"
    ].eq(1.0).all()
    assert metrics["validation_year"].eq(2025).all()
    assert comparisons["mse_gain"].gt(0).all()
