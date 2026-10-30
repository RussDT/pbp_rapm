from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "01b_enrich_pbp_shotquality.py"
)
SPEC = importlib.util.spec_from_file_location("enrich_pbp_shotquality", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _sq_row(**overrides):
    row = {
        "game_id": "22500001",
        "event_num": 10,
        "initial_ev": 1.1,
        "is_transition": False,
        "sq_shooter_id": 1,
        "sq_primary_defender_id": 2,
    }
    row.update(overrides)
    return row


def test_exact_provider_duplicates_collapse_without_arbitrary_selection():
    sq = pd.DataFrame([_sq_row(), _sq_row()])

    resolved = MODULE.resolve_shotquality_candidates(sq)

    assert len(resolved) == 1
    assert resolved.loc[0, "sq_match_status"] == "exact_event_key"
    assert resolved.loc[0, "sq_match_candidate_count"] == 1
    assert resolved.loc[0, "sq_source_row_count"] == 2
    assert resolved.loc[0, "sq_primary_defender_id"] == 2


def test_conflicting_provider_rows_remain_explicitly_ambiguous():
    sq = pd.DataFrame(
        [
            _sq_row(sq_primary_defender_id=2),
            _sq_row(sq_primary_defender_id=3),
        ]
    )

    resolved = MODULE.resolve_shotquality_candidates(sq)

    assert len(resolved) == 1
    assert resolved.loc[0, "sq_match_status"] == "ambiguous_provider_event_key"
    assert resolved.loc[0, "sq_match_candidate_count"] == 2
    assert pd.isna(resolved.loc[0, "sq_primary_defender_id"])


def test_duplicate_raw_event_keys_do_not_receive_the_same_provider_assignment():
    raw = pd.DataFrame(
        {
            "game_id": ["0022500001", "0022500001"],
            "event_num": [10, 10],
            "description": ["shot", "auxiliary row"],
        }
    )
    sq = pd.DataFrame([_sq_row()])

    enriched = MODULE.attach_shotquality_rows(raw, sq)

    assert enriched["sq_match_status"].eq("ambiguous_raw_event_key").all()
    assert enriched["sq_primary_defender_id"].isna().all()
    assert "sq_defender_id" not in enriched.columns


def test_unique_raw_key_gets_provider_specific_primary_defender():
    raw = pd.DataFrame(
        {
            "game_id": ["0022500001", "0022500001"],
            "event_num": [10, 11],
        }
    )
    sq = pd.DataFrame([_sq_row()])

    enriched = MODULE.attach_shotquality_rows(raw, sq)

    matched = enriched.loc[enriched["event_num"].eq(10)].iloc[0]
    unmatched = enriched.loc[enriched["event_num"].eq(11)].iloc[0]
    assert matched["sq_provider"] == "shotquality"
    assert matched["sq_match_status"] == "exact_event_key"
    assert matched["sq_primary_defender_id"] == 2
    assert unmatched["sq_match_status"] == "unmatched"


def test_parquet_enrichment_preserves_game_id_and_removes_legacy_field(tmp_path):
    raw_path = tmp_path / "NBA25.parquet"
    pd.DataFrame(
        {
            "game_id": ["0022500001"],
            "event_num": [10],
            "sq_defender_id": [999],
        }
    ).to_parquet(raw_path, index=False)

    assert MODULE.enrich_parquet(raw_path, pd.DataFrame([_sq_row()]))
    enriched = pd.read_parquet(raw_path)

    assert enriched.loc[0, "game_id"] == "0022500001"
    assert enriched.loc[0, "sq_primary_defender_id"] == 2
    assert "sq_defender_id" not in enriched.columns
