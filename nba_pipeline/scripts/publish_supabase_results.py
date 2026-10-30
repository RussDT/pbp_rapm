#!/usr/bin/env python3
"""Validate and publish registered result CSVs to the Supabase results hub.

The command is dry-run by default. Use ``--publish`` for the explicit remote
write step. Content-addressed versions make retries idempotent, and a version
becomes site-visible only after every expected row has been uploaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "nba_pipeline" / "config" / "supabase_publications.json"
VERSION_NAMESPACE = uuid.UUID("a2eeb1fb-bb2a-43af-b3ad-4133e2a782d4")
DATASET_TABLE = "analytics_datasets"
VERSION_TABLE = "analytics_dataset_versions"
ROW_TABLE = "analytics_dataset_rows"
CURRENT_VIEW = "analytics_current_rows"


@dataclass(frozen=True)
class PreparedDataset:
    config: dict[str, Any]
    source: Path
    records: list[dict[str, Any]]
    row_keys: list[str]
    content_sha256: str
    version_id: uuid.UUID
    as_of_date: str | None

    @property
    def slug(self) -> str:
        return str(self.config["slug"])


def load_registry(path: Path) -> dict[str, Any]:
    registry = json.loads(path.read_text())
    if registry.get("schema_version") != 1:
        raise ValueError(f"unsupported registry schema_version in {path}")
    datasets = registry.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(f"registry has no datasets: {path}")
    slugs = [item.get("slug") for item in datasets]
    if any(not slug for slug in slugs) or len(slugs) != len(set(slugs)):
        raise ValueError("dataset slugs must be present and unique")
    return registry


def _records_from_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    # pandas emits JSON-native numbers, booleans and nulls here; using to_dict
    # directly would leak numpy scalar types and NaN into the API payload.
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _row_key(record: dict[str, Any], fields: list[str]) -> str:
    values = [record[field] for field in fields]
    if any(value is None for value in values):
        raise ValueError(f"null row-key value for fields {fields}: {values}")
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def _as_of_date(config: dict[str, Any], records: list[dict[str, Any]]) -> str | None:
    static_value = config.get("as_of_date")
    field = config.get("as_of_field")
    if static_value and field:
        raise ValueError(f"{config['slug']} sets both as_of_date and as_of_field")
    if field:
        values = {record.get(field) for record in records}
        values.discard(None)
        if len(values) != 1:
            raise ValueError(f"{config['slug']} expected one {field} value, found {values}")
        static_value = values.pop()
    if static_value is None:
        return None
    value = str(static_value)[:10]
    date.fromisoformat(value)
    return value


def prepare_dataset(config: dict[str, Any], repo_root: Path = REPO_ROOT) -> PreparedDataset:
    source = repo_root / config["source"]
    if not source.is_file():
        raise FileNotFoundError(f"dataset source does not exist: {source}")

    frame = pd.read_csv(source, low_memory=False)
    expected_columns = list(config["expected_columns"])
    actual_columns = frame.columns.tolist()
    if actual_columns != expected_columns:
        missing = [column for column in expected_columns if column not in actual_columns]
        extra = [column for column in actual_columns if column not in expected_columns]
        raise ValueError(
            f"{config['slug']} schema drift: missing={missing}, extra={extra}, "
            f"order_matches={actual_columns == expected_columns}"
        )

    minimum_rows = int(config.get("minimum_rows", 1))
    if len(frame) < minimum_rows:
        raise ValueError(
            f"{config['slug']} has {len(frame)} rows; minimum is {minimum_rows}"
        )

    key_fields = list(config["row_key_fields"])
    missing_key_fields = [field for field in key_fields if field not in frame.columns]
    if missing_key_fields:
        raise ValueError(f"{config['slug']} missing row-key fields {missing_key_fields}")

    records = _records_from_frame(frame)
    row_keys = [_row_key(record, key_fields) for record in records]
    if len(row_keys) != len(set(row_keys)):
        duplicate_count = len(row_keys) - len(set(row_keys))
        raise ValueError(f"{config['slug']} has {duplicate_count} duplicate row keys")

    digest = hashlib.sha256()
    version_contract = {
        "schema_version": int(config["schema_version"]),
        "expected_columns": expected_columns,
        "row_key_fields": key_fields,
        "as_of_date": _as_of_date(config, records),
        "metadata": dict(config.get("metadata", {})),
    }
    digest.update(
        json.dumps(
            version_contract,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    digest.update(b"\n")
    for ordinal, (row_key, record) in enumerate(zip(row_keys, records)):
        canonical = json.dumps(
            {"ordinal": ordinal, "row_key": row_key, "payload": record},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        digest.update(canonical.encode("utf-8"))
        digest.update(b"\n")
    content_sha256 = digest.hexdigest()
    version_id = uuid.uuid5(
        VERSION_NAMESPACE, f"{config['slug']}:{content_sha256}"
    )

    return PreparedDataset(
        config=config,
        source=source,
        records=records,
        row_keys=row_keys,
        content_sha256=content_sha256,
        version_id=version_id,
        as_of_date=version_contract["as_of_date"],
    )


def _chunks(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def execute_with_transport_retry(
    operation: Any,
    *,
    label: str,
    attempts: int = 3,
) -> Any:
    """Retry only transient HTTP transport failures with a short backoff."""
    import httpx

    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except httpx.TransportError:
            if attempt == attempts:
                raise
            time.sleep(0.5 * (2 ** (attempt - 1)))
            print(f"retrying {label} after transient transport failure ({attempt}/{attempts})")


def publish_dataset(client: Any, prepared: PreparedDataset, batch_size: int = 500) -> dict[str, Any]:
    config = prepared.config
    dataset_record = {
        "slug": prepared.slug,
        "title": config["title"],
        "description": config["description"],
        "status": config["status"],
        "is_public": bool(config["is_public"]),
        "schema_version": int(config["schema_version"]),
        "row_key_fields": list(config["row_key_fields"]),
        "source_path": config["source"],
    }
    execute_with_transport_retry(
        lambda: client.table(DATASET_TABLE)
        .upsert(dataset_record, on_conflict="slug")
        .execute(),
        label=f"{prepared.slug} registry upsert",
    )

    version_record = {
        "id": str(prepared.version_id),
        "dataset_slug": prepared.slug,
        "content_sha256": prepared.content_sha256,
        "as_of_date": prepared.as_of_date,
        "row_count": len(prepared.records),
        "column_names": list(config["expected_columns"]),
        "source_path": config["source"],
        "metadata": dict(config.get("metadata", {})),
    }
    execute_with_transport_retry(
        lambda: client.table(VERSION_TABLE)
        .upsert(version_record, on_conflict="id")
        .execute(),
        label=f"{prepared.slug} version upsert",
    )

    rows = [
        {
            "version_id": str(prepared.version_id),
            "row_key": row_key,
            "row_ordinal": ordinal,
            "payload": record,
        }
        for ordinal, (row_key, record) in enumerate(
            zip(prepared.row_keys, prepared.records)
        )
    ]
    for batch_number, batch in enumerate(_chunks(rows, batch_size), start=1):
        execute_with_transport_retry(
            lambda batch=batch: client.table(ROW_TABLE)
            .upsert(batch, on_conflict="version_id,row_key")
            .execute(),
            label=f"{prepared.slug} row batch {batch_number}",
        )

    execute_with_transport_retry(
        lambda: client.rpc(
            "promote_analytics_dataset_version",
            {
                "p_dataset_slug": prepared.slug,
                "p_version_id": str(prepared.version_id),
                "p_expected_rows": len(rows),
            },
        ).execute(),
        label=f"{prepared.slug} promotion",
    )

    from postgrest.types import CountMethod

    response = execute_with_transport_retry(
        lambda: client.table(CURRENT_VIEW)
        .select("row_key", count=CountMethod.exact, head=True)
        .eq("dataset_slug", prepared.slug)
        .execute(),
        label=f"{prepared.slug} verification",
    )
    if response.count != len(rows):
        raise RuntimeError(
            f"{prepared.slug} site-facing verification returned "
            f"{response.count} rows, expected {len(rows)}"
        )
    return publication_receipt(prepared, published=True)


def publication_receipt(prepared: PreparedDataset, *, published: bool) -> dict[str, Any]:
    return {
        "dataset": prepared.slug,
        "status": prepared.config["status"],
        "public": bool(prepared.config["is_public"]),
        "source": prepared.config["source"],
        "rows": len(prepared.records),
        "columns": len(prepared.config["expected_columns"]),
        "as_of_date": prepared.as_of_date,
        "content_sha256": prepared.content_sha256,
        "version_id": str(prepared.version_id),
        "published": published,
    }


def _selected_configs(registry: dict[str, Any], requested: list[str]) -> list[dict[str, Any]]:
    configs = {config["slug"]: config for config in registry["datasets"]}
    if not requested or requested == ["all"]:
        return list(configs.values())
    unknown = [slug for slug in requested if slug not in configs]
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}; choices: {sorted(configs)}")
    return [configs[slug] for slug in requested]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset slug to validate/publish; repeat or use 'all' (default: all).",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Write validated datasets to Supabase. Without this flag, dry-run only.",
    )
    parser.add_argument("--batch-size", type=int, default=500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.batch_size > 1000:
        raise ValueError("--batch-size must be between 1 and 1000")

    registry = load_registry(args.registry)
    prepared = [
        prepare_dataset(config) for config in _selected_configs(registry, args.dataset)
    ]

    if not args.publish:
        print(json.dumps([publication_receipt(item, published=False) for item in prepared], indent=2))
        return 0

    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    supabase_url = os.getenv("SUPABASE_URL")
    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not service_role_key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required for --publish"
        )

    from supabase import create_client

    client = create_client(supabase_url, service_role_key)
    receipts = [
        publish_dataset(client, item, batch_size=args.batch_size) for item in prepared
    ]
    print(json.dumps(receipts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
