from __future__ import annotations

import json

import numpy as np
import pandas as pd

from nba_pipeline.scripts.build_historical_shot_quality_transfer import (
    HISTORICAL_CATEGORICAL,
    HISTORICAL_NUMERIC,
)
from nba_pipeline.scripts.score_historical_shot_quality_transfer import (
    OUTPUT_COLUMNS,
    OUTPUT_SCHEMA_VERSION,
    add_streaming_pregame_shooter_priors,
    calibration_proxy_summary,
    missingness_records,
    output_is_resumable,
    sidecar_path,
)


def test_streaming_priors_exclude_current_game_and_continue_across_files():
    state = {}
    first = pd.DataFrame(
        {
            "shooter_id": [1, 1, 1, 1],
            "model_bucket": ["three"] * 4,
            "game_id": ["g1", "g1", "g2", "g2"],
            "is_make": [1, 0, 1, 1],
        }
    )
    scored_first = add_streaming_pregame_shooter_priors(first, state, shrink=0)
    assert scored_first.loc[scored_first["game_id"].eq("g1"), "shooter_pregame_log_attempts"].eq(
        0
    ).all()
    g2 = scored_first.loc[scored_first["game_id"].eq("g2")]
    assert np.allclose(g2["shooter_pregame_make_rate"], 0.5)
    assert state[(1, "three")] == (4.0, 3.0)

    second = pd.DataFrame(
        {
            "shooter_id": [1],
            "model_bucket": ["three"],
            "game_id": ["g3"],
            "is_make": [0],
        }
    )
    scored_second = add_streaming_pregame_shooter_priors(second, state, shrink=0)
    assert scored_second["shooter_pregame_make_rate"].iloc[0] == 0.75
    assert state[(1, "three")] == (5.0, 3.0)


def test_resume_requires_matching_sidecar_and_valid_probabilities(tmp_path):
    output = tmp_path / "season.parquet"
    frame = pd.DataFrame({column: [None, None] for column in OUTPUT_COLUMNS})
    for column in [
        "historical_sq_make_prob",
        "historical_sq_direct_make_prob",
        "historical_sq_distilled_make_prob",
    ]:
        frame[column] = [0.3, 0.4]
    frame.to_parquet(output, index=False)
    source = {"source_file": "NBA97.parquet", "source_size": 10, "source_mtime_ns": 20}
    sidecar_path(output).write_text(
        json.dumps(
            {
                "schema_version": OUTPUT_SCHEMA_VERSION,
                "model_sha256": "abc",
                "rows": 2,
                **source,
            }
        )
    )

    valid, reason = output_is_resumable(
        output, expected_rows=2, model_sha256="abc", source=source
    )
    assert valid, reason
    invalid, reason = output_is_resumable(
        output, expected_rows=3, model_sha256="abc", source=source
    )
    assert not invalid
    assert reason == "sidecar_rows_mismatch"


def test_calibration_proxy_does_not_invent_outcome_metrics_without_labels():
    rows = pd.DataFrame(
        {
            "historical_sq_make_prob": [0.3, 0.4],
            "historical_sq_direct_make_prob": [0.31, 0.39],
            "historical_sq_distilled_make_prob": [0.29, 0.41],
            "is_make": [np.nan, np.nan],
        }
    )
    summary = calibration_proxy_summary(
        rows,
        scope="era",
        scope_value="1997-2003",
        season_type="RS",
        bucket="three",
        support={"three": (0.25, 0.50)},
    )
    assert summary["outcome_audit_kind"] == "distribution_proxy_only"
    assert summary["labeled_rows"] == 0
    assert np.isnan(summary["outcome_brier"])
    assert np.isnan(summary["outcome_calibration_bias"])


def test_literal_none_modifier_is_not_counted_as_missing():
    row = {
        "model_bucket": "midrange",
        **{feature: "known" for feature in HISTORICAL_CATEGORICAL},
        **{feature: 1.0 for feature in HISTORICAL_NUMERIC},
    }
    row["modifier_bundle"] = "none"
    records = missingness_records(pd.DataFrame([row]), 1997, "RS")
    modifier = next(record for record in records if record["feature"] == "modifier_bundle")
    assert modifier["missing_rows"] == 0
