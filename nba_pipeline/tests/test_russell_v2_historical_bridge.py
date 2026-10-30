from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_pipeline.scripts.build_russell_v2_historical_bridge import (
    BUCKETS,
    STUDENT_CATEGORICAL,
    STUDENT_NUMERIC,
    choose_bridge_source,
    fit_distiller,
    fit_outcome_stack,
    load_v2_labels,
    score_bridge_rows,
    season_seam_audit,
)


def _student_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for bucket_index, bucket in enumerate(BUCKETS):
        for index in range(60):
            portable = 0.38 + 0.06 * ((index % 5) / 4.0)
            row: dict[str, object] = {
                "source_file": "NBA15.parquet",
                "season_end_year": 2015,
                "season_type": "rs",
                "game_id": f"g{bucket_index}{index:03d}",
                "event_num": index,
                "shooter_id": 100 + index % 8,
                "model_bucket": bucket,
                "shot_value": 2.0 if bucket == "midrange" else 3.0,
                "is_make": bool(index % 3 == 0),
                "p_v2": min(0.95, portable + 0.02 * (index % 2)),
                "historical_sq_make_prob": portable,
                "historical_sq_direct_make_prob": portable + 0.01,
                "historical_sq_distilled_make_prob": portable - 0.01,
                "historical_sq_component_gap": 0.02,
                "transition_probability": 0.1 + 0.01 * (index % 4),
            }
            for column in STUDENT_CATEGORICAL:
                row.setdefault(column, f"{column}_{index % 3}")
            for column in STUDENT_NUMERIC:
                row.setdefault(column, float(index % 7))
            rows.append(row)
    return pd.DataFrame(rows)


def test_overlapping_v2_labels_must_be_identical(tmp_path):
    first = pd.DataFrame(
        {
            "season_end_year": [2021],
            "game_id": ["123"],
            "event_num": [7],
            "zone": ["mid"],
            "made": [1],
            "shot_value": [2.0],
            "p_v2": [0.44],
        }
    )
    path_a = tmp_path / "a.parquet"
    path_b = tmp_path / "b.parquet"
    first.to_parquet(path_a, index=False)
    first.to_parquet(path_b, index=False)
    labels, audit = load_v2_labels([path_a, path_b])
    assert len(labels) == 1
    assert labels["zone"].iloc[0] == "midrange"
    assert audit["overlap_duplicate_rows"] == 2

    conflicting = first.copy()
    conflicting["p_v2"] = 0.45
    conflicting.to_parquet(path_b, index=False)
    with pytest.raises(ValueError, match="disagree"):
        load_v2_labels([path_a, path_b])


def test_bridge_gate_requires_accuracy_and_teacher_fidelity():
    selected, gates = choose_bridge_source(
        portable_brier=0.24,
        bridge_brier=0.239,
        portable_rmse_vs_v2=0.08,
        bridge_rmse_vs_v2=0.06,
    )
    assert selected == "v2_distilled_stack"
    assert all(gates.values())

    fallback, gates = choose_bridge_source(
        portable_brier=0.24,
        bridge_brier=0.241,
        portable_rmse_vs_v2=0.08,
        bridge_rmse_vs_v2=0.06,
    )
    assert fallback == "portable_v3_fallback"
    assert not gates["actual_brier_no_regression"]


def test_student_contract_excludes_current_outcome_and_scores_bounded():
    assert "is_make" not in STUDENT_NUMERIC
    assert "made" not in STUDENT_NUMERIC
    assert "p_v2" not in STUDENT_NUMERIC
    assert "is_make" not in STUDENT_CATEGORICAL

    rows = _student_rows()
    models: dict[str, object] = {}
    for bucket in BUCKETS:
        bucket_rows = rows[rows["model_bucket"].eq(bucket)]
        distiller = fit_distiller(bucket_rows, max_iter=5)
        student = 1.0 / (1.0 + np.exp(-distiller.predict(bucket_rows)))
        stack = fit_outcome_stack(
            bucket_rows["historical_sq_make_prob"],
            student,
            bucket_rows["is_make"],
        )
        models[bucket] = {
            "distiller": distiller,
            "stack": stack,
            "selected_source": (
                "v2_distilled_stack" if bucket == "midrange" else "portable_v3_fallback"
            ),
        }

    scored = score_bridge_rows(rows, models)
    assert scored["p_make"].between(0.0, 1.0).all()
    assert scored["p_v2_historical_student"].between(0.0, 1.0).all()
    assert np.allclose(scored["ev_pts"], scored["p_make"] * scored["shot_value"])
    assert set(scored["bridge_source"]) == {
        "v2_distilled_stack",
        "portable_v3_fallback",
    }
    three = scored[scored["zone"].eq("three")]
    assert np.allclose(three["p_make"], three["p_portable_v3"])


def test_seam_audit_keeps_bridge_and_native_levels_explicit():
    bridge = pd.DataFrame(
        {
            "season_end_year": [2015, 2015],
            "zone": ["mid", "three"],
            "made": [1, 0],
            "p_make": [0.40, 0.35],
            "bridge_source": ["v2_distilled_stack", "portable_v3_fallback"],
        }
    )
    native = pd.DataFrame(
        {
            "season_end_year": [2016, 2016],
            "zone": ["midrange", "three"],
            "made": [0, 1],
            "p_v2": [0.42, 0.36],
        }
    )
    audit, seam = season_seam_audit(bridge, native)
    assert set(audit["series_source"]) == {
        "v2_distilled_stack",
        "portable_v3_fallback",
        "native_v2",
    }
    assert seam["mid"]["probability_change_2016_minus_2015"] == pytest.approx(0.02)
    assert seam["three"]["probability_change_2016_minus_2015"] == pytest.approx(0.01)
