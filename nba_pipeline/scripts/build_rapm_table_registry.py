#!/usr/bin/env python3
"""Build an API-ready inventory from the curated RAPM table registry.

The checked-in JSON defines meaning and ownership. This script adds current
artifact paths, schemas, row counts, sizes, timestamps, and overnight job
status without changing the curated definitions.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PIPELINE_ROOT.parent
DEFAULT_REGISTRY = PIPELINE_ROOT / "config" / "rapm_table_registry.json"
DEFAULT_OUTPUT_DIR = PIPELINE_ROOT / "results" / "rapm_registry"
MECHANISM_MANIFEST = PIPELINE_ROOT / "results" / "mechanism_rapm_registry" / "run_manifest.json"
CURRENT_MECHANISM_MANIFEST = (
    PIPELINE_ROOT / "results" / "mechanism_td550_current" / "run_manifest.json"
)
OVERNIGHT_STATE = PIPELINE_ROOT / "results" / "rapm_overnight" / "run_state.json"


def validate_registry(registry: dict) -> None:
    if registry.get("schema_version") != 1:
        raise ValueError("Unsupported RAPM registry schema_version")
    series_ids = [item["series_id"] for item in registry["series"]]
    mechanism_ids = [item["mechanism_id"] for item in registry["mechanism_rapms"]]
    if len(series_ids) != len(set(series_ids)):
        raise ValueError("Duplicate series_id in RAPM registry")
    if len(mechanism_ids) != len(set(mechanism_ids)):
        raise ValueError("Duplicate mechanism_id in RAPM registry")
    current = registry.get("current_mechanism_view")
    if current is not None:
        if current.get("prior_scale") != 0:
            raise ValueError("Current mechanism view must remain zero-prior")
        if current.get("half_life_days") != 550:
            raise ValueError("Current mechanism view must use the owner-selected TD550 clock")
        alpha = current.get("alpha_selection", {})
        if alpha.get("method") != "weighted_grouped_game_cv":
            raise ValueError("Current mechanism alphas must use weighted grouped-game CV")
        requested_ids = [item["mechanism_id"] for item in current.get("mechanisms", [])]
        unknown = set(requested_ids) - set(mechanism_ids)
        if unknown:
            raise ValueError(f"Unknown current mechanism ids: {sorted(unknown)}")
        if len(requested_ids) != len(set(requested_ids)):
            raise ValueError("Duplicate mechanism_id in current mechanism view")


def _csv_metadata(path: Path) -> tuple[int | None, list[str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        try:
            columns = next(reader)
        except StopIteration:
            return 0, []
        rows = sum(1 for _ in reader)
    return rows, columns


def _parquet_metadata(path: Path) -> tuple[int | None, list[str]]:
    try:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        return parquet.metadata.num_rows, parquet.schema.names
    except Exception:
        frame = pd.read_parquet(path)
        return len(frame), list(frame.columns)


def artifact_metadata(path: Path, *, series_id: str, variant_id: str) -> dict[str, object]:
    if path.suffix.lower() == ".csv":
        row_count, columns = _csv_metadata(path)
    elif path.suffix.lower() in {".parquet", ".parq"}:
        row_count, columns = _parquet_metadata(path)
    elif path.suffix.lower() == ".json":
        row_count, columns = None, []
    else:
        row_count, columns = None, []
    stat = path.stat()
    return {
        "artifact_id": f"{series_id}:{variant_id}:{path.name}",
        "series_id": series_id,
        "variant_id": variant_id,
        "path": str(path.relative_to(REPO_ROOT)),
        "format": path.suffix.lower().lstrip("."),
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "row_count": row_count,
        "columns": columns,
    }


def expected_csv_count(series: dict) -> int | None:
    release = series.get("release_windows")
    if release == "current_1y_5y":
        return 5
    if release == "all_complete_rolling_5y_1997_2026":
        return 26
    if release == "current_1y_5y_plus_all_complete_rolling_5y_1997_2026":
        # The current five-year cut is also the final historical rolling window.
        return 26 + 4
    if release == "all_rolling_1y_5y_1997_2026":
        return sum(min(5, season_count) for season_count in range(1, 31))
    return None


def collect_series_artifacts(registry: dict) -> tuple[list[dict], list[dict]]:
    artifacts: list[dict] = []
    series_status: list[dict] = []
    for series in registry["series"]:
        paths: set[Path] = set()
        for pattern in series.get("output_globs", []):
            paths.update(path for path in REPO_ROOT.glob(pattern) if path.is_file())
        for path in sorted(paths):
            artifacts.append(
                artifact_metadata(
                    path,
                    series_id=series["series_id"],
                    variant_id=series["variant_id"],
                )
            )
        format_counts = Counter(path.suffix.lower().lstrip(".") for path in paths)
        csv_count = format_counts["csv"]
        expected = expected_csv_count(series)
        required_formats = series.get("required_formats", ["csv"])
        if expected is None:
            status = "present" if paths else "missing"
        elif all(format_counts[fmt] >= expected for fmt in required_formats):
            status = "complete"
        elif any(format_counts[fmt] for fmt in required_formats):
            status = "partial"
        else:
            status = "missing"
        series_status.append(
            {
                "series_id": series["series_id"],
                "variant_id": series["variant_id"],
                "artifact_status": status,
                "csv_count": csv_count,
                "expected_csv_count": expected,
                "required_formats": required_formats,
                "format_counts": dict(sorted(format_counts.items())),
                "artifact_count_all_formats": len(paths),
            }
        )
    return artifacts, series_status


def load_json_if_present(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def collect_mechanism_artifacts(registry: dict) -> tuple[list[dict], list[dict]]:
    manifest = load_json_if_present(MECHANISM_MANIFEST)
    by_id = {item["mechanism_id"]: item for item in registry["mechanism_rapms"]}
    artifacts: list[dict] = []
    statuses: list[dict] = []
    records_by_id: dict[str, list[dict]] = {mechanism_id: [] for mechanism_id in by_id}
    for record in (manifest or {}).get("records", []):
        records_by_id.setdefault(record["mechanism_id"], []).append(record)
        output = record.get("output")
        if output and Path(output).exists():
            mechanism = by_id[record["mechanism_id"]]
            artifacts.append(
                artifact_metadata(
                    Path(output),
                    series_id=f"mechanism:{record['mechanism_id']}",
                    variant_id=f"{record['window']}:{mechanism['season_type'].lower()}",
                )
            )
    for mechanism_id, mechanism in by_id.items():
        records = records_by_id.get(mechanism_id, [])
        specialized_paths: set[Path] = set()
        for pattern in mechanism.get("output_globs", []):
            specialized_paths.update(path for path in REPO_ROOT.glob(pattern) if path.is_file())
        for path in sorted(specialized_paths):
            artifacts.append(
                artifact_metadata(
                    path,
                    series_id=f"mechanism:{mechanism_id}",
                    variant_id=f"specialized:{mechanism['season_type'].lower()}",
                )
            )
        if mechanism.get("specialized_runner"):
            status = "complete" if specialized_paths else "not_run"
        elif any(record.get("status") == "failed" for record in records):
            status = "failed"
        elif records and all(record.get("status") in {"built", "existing"} for record in records):
            status = "complete"
        elif records:
            status = "partial"
        else:
            status = "not_run"
        statuses.append(
            {
                "mechanism_id": mechanism_id,
                "artifact_status": status,
                "artifact_count": len(records) + len(specialized_paths),
            }
        )
    return artifacts, statuses


def collect_current_mechanism_artifacts(registry: dict) -> tuple[list[dict], list[dict]]:
    spec = registry.get("current_mechanism_view") or {}
    manifest = load_json_if_present(CURRENT_MECHANISM_MANIFEST)
    records = (manifest or {}).get("records", [])
    records_by_id = {record["mechanism_id"]: record for record in records}
    artifacts: list[dict] = []
    statuses: list[dict] = []
    board_id = str(spec.get("board_id", "current_mechanisms"))
    for requested in spec.get("mechanisms", []):
        mechanism_id = requested["mechanism_id"]
        record = records_by_id.get(mechanism_id)
        paths: list[Path] = []
        if record:
            for field in ("output", "alpha_cv"):
                value = record.get(field)
                if value and Path(value).exists():
                    paths.append(Path(value))
        for path in paths:
            artifacts.append(
                artifact_metadata(
                    path,
                    series_id=f"current_mechanism:{mechanism_id}",
                    variant_id=f"{board_id}:{record['window']}",
                )
            )
        status = "not_run"
        if record:
            status = "failed" if record.get("status") == "failed" else "complete"
        statuses.append(
            {
                "mechanism_id": mechanism_id,
                "board_id": board_id,
                "artifact_status": status,
                "coverage_start_year": requested.get("start_year", spec.get("default_start_year")),
                "coverage_end_year": spec.get("season_end_year"),
                "coverage_note": requested.get("coverage_note"),
                "selected_off_alpha": record.get("selected_off_alpha") if record else None,
                "selected_def_alpha": record.get("selected_def_alpha") if record else None,
                "alpha_grid_boundary": record.get("alpha_grid_boundary") if record else None,
                "artifact_count": len(paths),
            }
        )
    return artifacts, statuses


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    flattened = []
    for row in rows:
        flattened.append(
            {
                key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
        )
    pd.DataFrame(flattened).to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    registry = json.loads(args.registry.read_text())
    validate_registry(registry)
    series_artifacts, series_status = collect_series_artifacts(registry)
    mechanism_artifacts, mechanism_status = collect_mechanism_artifacts(registry)
    current_artifacts, current_status = collect_current_mechanism_artifacts(registry)
    artifacts = series_artifacts + mechanism_artifacts + current_artifacts
    output = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "registry_source": str(args.registry.relative_to(REPO_ROOT)),
        "api_contract": registry["api_contract"],
        "current_mechanism_view": registry.get("current_mechanism_view"),
        "series": registry["series"],
        "series_status": series_status,
        "mechanism_rapms": registry["mechanism_rapms"],
        "mechanism_status": mechanism_status,
        "current_mechanism_status": current_status,
        "artifacts": artifacts,
        "overnight_state": load_json_if_present(OVERNIGHT_STATE),
        "summary": {
            "series_count": len(registry["series"]),
            "mechanism_count": len(registry["mechanism_rapms"]),
            "artifact_count": len(artifacts),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "rapm_table_registry.json"
    json_path.write_text(json.dumps(output, indent=2) + "\n")
    write_csv(args.output_dir / "series_status.csv", series_status)
    write_csv(args.output_dir / "mechanism_status.csv", mechanism_status)
    write_csv(args.output_dir / "current_mechanism_status.csv", current_status)
    write_csv(args.output_dir / "artifacts.csv", artifacts)
    print(
        f"Wrote {json_path}: series={len(registry['series'])} "
        f"mechanisms={len(registry['mechanism_rapms'])} artifacts={len(artifacts)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
