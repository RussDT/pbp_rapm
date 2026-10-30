import json

import pandas as pd
import pytest

from nba_pipeline.scripts.fetch_epm_data import (
    fetch_snapshot,
    normalize_snapshot,
    snapshot_paths,
    write_snapshot,
)


def _row(player_id=203999, *, game_optimized=0):
    return {
        "date": "2026-06-13",
        "game_dt": "2026-04-30",
        "game_optimized": game_optimized,
        "player_id": player_id,
        "player_name": "Nikola Jokic",
        "p_fg3pct": 0.365,
        "p_fgpct_mid": 0.548,
        "p_fg3a_100": 6.1,
        "p_fga_mid_100": 9.8,
        "p_orb_100": 3.45,
        "oepm": 6.39,
        "depm": 0.94,
        "epm": 7.33,
    }


class _Response:
    def __init__(self, rows):
        self._rows = rows

    def raise_for_status(self):
        return None

    def json(self):
        return self._rows


class _Session:
    def __init__(self, rows):
        self.rows = rows
        self.call = None

    def get(self, url, **kwargs):
        self.call = (url, kwargs)
        return _Response(self.rows)


def test_fetch_uses_authorization_header_without_putting_key_in_query():
    session = _Session([_row()])
    rows = fetch_snapshot(
        "2026-07-23",
        game_optimized=0,
        api_key="secret-token",
        session=session,
    )

    assert len(rows) == 1
    _, kwargs = session.call
    assert kwargs["headers"]["Authorization"] == "secret-token"
    assert "secret-token" not in kwargs["params"].values()
    assert kwargs["params"] == {"date": "2026-07-23", "game_optimized": 0}


def test_normalize_preserves_requested_and_provider_effective_dates():
    frame = normalize_snapshot(
        [_row(2), _row(1)],
        requested_date="2026-07-23",
        game_optimized=0,
    )

    assert frame["player_id"].tolist() == [1, 2]
    assert frame["requested_date"].unique().tolist() == ["2026-07-23"]
    assert frame["date"].unique().tolist() == ["2026-06-13"]


def test_normalize_rejects_schema_drift_and_duplicate_players():
    missing = _row()
    missing.pop("p_fg3pct")
    with pytest.raises(ValueError, match="missing required columns"):
        normalize_snapshot(
            [missing],
            requested_date="2026-07-23",
            game_optimized=0,
        )

    with pytest.raises(ValueError, match="duplicate player_id"):
        normalize_snapshot(
            [_row(), _row()],
            requested_date="2026-07-23",
            game_optimized=0,
        )


def test_write_snapshot_emits_checksum_manifest_and_refuses_overwrite(tmp_path):
    frame = normalize_snapshot(
        [_row()],
        requested_date="2026-07-23",
        game_optimized=0,
    )
    csv_path, manifest_path, manifest = write_snapshot(
        frame,
        cache_dir=tmp_path,
        requested_date="2026-07-23",
        game_optimized=0,
    )

    expected_csv, expected_manifest = snapshot_paths(tmp_path, "2026-07-23", 0)
    assert (csv_path, manifest_path) == (expected_csv, expected_manifest)
    assert pd.read_csv(csv_path).loc[0, "requested_date"] == "2026-07-23"
    persisted = json.loads(manifest_path.read_text())
    assert persisted["provider_effective_dates"] == ["2026-06-13"]
    assert persisted["row_count"] == 1
    assert persisted["csv_sha256"] == manifest["csv_sha256"]

    with pytest.raises(FileExistsError, match="snapshot already exists"):
        write_snapshot(
            frame,
            cache_dir=tmp_path,
            requested_date="2026-07-23",
            game_optimized=0,
        )
