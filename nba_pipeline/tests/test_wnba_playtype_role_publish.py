"""Offline coverage for the WNBA Role / zTS publish bridge.

The publisher is the boundary where a research artifact becomes a reader-facing
column, so the invariants tested here are the ones a downstream consumer cannot
re-check: only modern-descriptor seasons ship, the published `role` is the
scaled column rather than the raw one, playoff rows never appear, and the
reader identity `zTS = rTS + role` closes on the published numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
WNBA_TEST_DIR = ROOT / "wnba_test"
if str(WNBA_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(WNBA_TEST_DIR))

import publish_wnba_playtype_role as publish  # noqa: E402

COEFFICIENT = 0.81


def _source_frame() -> pd.DataFrame:
    raw = [3.0, -2.0, 1.0, 5.0, 4.0]
    return pd.DataFrame(
        {
            "feature_season": [2026, 2026, 2026, 2015, 2026],
            "player_id": [1, 2, 3, 4, 5],
            "player_name": ["Modern A", "Modern B", "Thin C", "Sparse D", "Modern E"],
            "pbp_fga": [200, 150, 10, 300, 60],
            "playtype_diff": raw,
            "playtype_diff_scaled": [COEFFICIENT * value for value in raw],
            "descriptor_regime": [
                "modern",
                "modern",
                "modern",
                "sparse_descriptions",
                "modern",
            ],
        }
    )


def test_sparse_descriptor_seasons_never_publish():
    published = publish.build_publish_frame(_source_frame(), min_fga=50)
    assert 2015 not in set(published["season"])
    assert "Sparse D" not in set(published["player_name"])


def test_thin_samples_are_dropped_by_the_fga_floor():
    published = publish.build_publish_frame(_source_frame(), min_fga=50)
    assert "Thin C" not in set(published["player_name"])
    assert published["pbp_fga"].min() >= 50


def test_published_role_is_the_scaled_column_not_the_raw_one():
    published = publish.build_publish_frame(_source_frame(), min_fga=50)
    assert published["role"].tolist() == pytest.approx(
        (COEFFICIENT * published["playtype_diff_raw"]).tolist(), abs=5e-4
    )
    # A raw column that leaked through would be strictly larger in magnitude.
    nonzero = published[published["playtype_diff_raw"].abs() > 0]
    assert (nonzero["role"].abs() < nonzero["playtype_diff_raw"].abs()).all()


def test_publish_frame_is_regular_season_only_and_unique():
    published = publish.build_publish_frame(_source_frame(), min_fga=50)
    assert not published["is_playoffs"].any()
    assert not published.duplicated(subset=["season", "is_playoffs", "player_id"]).any()


def test_validation_rejects_a_raw_role_column():
    published = publish.build_publish_frame(_source_frame(), min_fga=50)
    leaked = published.copy()
    leaked["role"] = leaked["playtype_diff_raw"]
    with pytest.raises(ValueError, match="not 0.81 x raw"):
        publish.validate_publish_frame(leaked, COEFFICIENT)


def test_validation_accepts_the_real_publish_frame():
    published = publish.build_publish_frame(_source_frame(), min_fga=50)
    report = publish.validate_publish_frame(published, COEFFICIENT)
    assert report["rows"] == len(published)
    assert report["max_role_minus_scaled_raw"] < 5e-4


def test_reader_identity_closes_and_the_raw_column_would_break_it():
    """zTS is summed downstream, so prove which column the sum must use."""
    published = publish.build_publish_frame(_source_frame(), min_fga=50)
    r_ts = pd.Series(2.5, index=published.index)

    z_ts = r_ts + published["role"]
    assert (z_ts - r_ts).tolist() == pytest.approx(published["role"].tolist(), abs=1e-9)

    # Summing the raw column instead would over-adjust by (1 - 0.81) x raw,
    # which is exactly the mistake the scaled column exists to prevent.
    z_ts_raw = r_ts + published["playtype_diff_raw"]
    over_adjustment = (z_ts_raw - z_ts).abs()
    expected = (1 - COEFFICIENT) * published["playtype_diff_raw"].abs()
    assert over_adjustment.tolist() == pytest.approx(expected.tolist(), abs=5e-4)
    assert over_adjustment.max() > 0


def test_shipped_artifact_matches_the_publish_contract():
    csv_path = (
        ROOT
        / "wnba_test"
        / "Results"
        / "playtype_difficulty"
        / "publish"
        / publish.PUBLISH_BASENAME
    )
    if not csv_path.exists():
        pytest.skip("publish artifact not built in this checkout")

    frame = pd.read_csv(csv_path)
    assert list(frame.columns) == publish.PUBLISH_COLUMNS
    assert frame["season"].min() >= 2018
    assert not frame["is_playoffs"].any()
    report = publish.validate_publish_frame(frame, COEFFICIENT)
    assert report["rows"] > 0
