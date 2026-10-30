import json
from pathlib import Path

import pandas as pd
import pytest
import httpx

from nba_pipeline.scripts.publish_supabase_results import (
    load_registry,
    execute_with_transport_retry,
    prepare_dataset,
    publication_receipt,
)


def config(source: str = "result.csv") -> dict:
    return {
        "slug": "test_board",
        "title": "Test board",
        "description": "Test result data.",
        "status": "research",
        "is_public": True,
        "schema_version": 1,
        "source": source,
        "as_of_field": "as_of",
        "row_key_fields": ["player_id", "season"],
        "minimum_rows": 2,
        "expected_columns": ["player_id", "season", "value", "as_of"],
        "metadata": {},
    }


def write_source(root: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(root / "result.csv", index=False)


def test_prepare_dataset_is_typed_deterministic_and_content_addressed(tmp_path: Path) -> None:
    rows = [
        {"player_id": 1, "season": 2026, "value": 1.25, "as_of": "2026-08-25"},
        {"player_id": 2, "season": 2026, "value": None, "as_of": "2026-08-25"},
    ]
    write_source(tmp_path, rows)

    first = prepare_dataset(config(), repo_root=tmp_path)
    second = prepare_dataset(config(), repo_root=tmp_path)

    assert first.content_sha256 == second.content_sha256
    assert first.version_id == second.version_id
    assert first.as_of_date == "2026-08-25"
    assert first.row_keys == ["[1,2026]", "[2,2026]"]
    assert first.records[0]["value"] == 1.25
    assert first.records[1]["value"] is None
    assert publication_receipt(first, published=False)["rows"] == 2

    changed_metadata = config()
    changed_metadata["metadata"] = {"method": "v2"}
    third = prepare_dataset(changed_metadata, repo_root=tmp_path)
    assert third.content_sha256 != first.content_sha256
    assert third.version_id != first.version_id


def test_prepare_dataset_rejects_schema_drift(tmp_path: Path) -> None:
    write_source(
        tmp_path,
        [
            {"player_id": 1, "season": 2026, "renamed": 1.0, "as_of": "2026-08-25"},
            {"player_id": 2, "season": 2026, "renamed": 2.0, "as_of": "2026-08-25"},
        ],
    )
    with pytest.raises(ValueError, match="schema drift"):
        prepare_dataset(config(), repo_root=tmp_path)


def test_prepare_dataset_rejects_duplicate_keys(tmp_path: Path) -> None:
    write_source(
        tmp_path,
        [
            {"player_id": 1, "season": 2026, "value": 1.0, "as_of": "2026-08-25"},
            {"player_id": 1, "season": 2026, "value": 2.0, "as_of": "2026-08-25"},
        ],
    )
    with pytest.raises(ValueError, match="duplicate row keys"):
        prepare_dataset(config(), repo_root=tmp_path)


def test_load_registry_requires_unique_slugs(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"schema_version": 1, "datasets": [config(), config()]}))
    with pytest.raises(ValueError, match="unique"):
        load_registry(path)


def test_transport_retry_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nba_pipeline.scripts.publish_supabase_results.time.sleep", lambda _: None)
    attempts = 0

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.WriteError("broken pipe")
        return "ok"

    assert execute_with_transport_retry(operation, label="test") == "ok"
    assert attempts == 3
