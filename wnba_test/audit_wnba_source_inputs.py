"""Audit Gabriel WNBA play-by-play and proprietary daily rating inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_ROOT = Path("nba_pipeline/external/gabriel_wnba")
DEFAULT_RATINGS_ROOT = Path("nba_pipeline/external/wnba_darko_daily")
DEFAULT_REPORT = Path("nba_pipeline/validation/wnba_source_inputs/audit.json")

PBP_CONFIG = {
    "classic": {
        "directory": "pbp_data",
        "key": "EVENTNUM",
        "required": {
            "GAME_ID",
            "EVENTNUM",
            "EVENTMSGTYPE",
            "EVENTMSGACTIONTYPE",
            "PERIOD",
            "PCTIMESTRING",
            "HOMEDESCRIPTION",
            "NEUTRALDESCRIPTION",
            "VISITORDESCRIPTION",
            "PLAYER1_ID",
            "PLAYER2_ID",
            "PLAYER3_ID",
        },
        "actor_fields": {"PLAYER1_ID", "PLAYER2_ID", "PLAYER3_ID"},
    },
    "modern": {
        "directory": "pbp_modern",
        "key": "actionNumber",
        "required": {
            "actionNumber",
            "clock",
            "period",
            "actionType",
            "subType",
            "description",
            "personId",
            "teamId",
            "teamTricode",
        },
        "actor_fields": {
            "personId",
            "jumpBallRecoverdPersonId",
            "jumpBallWonPersonId",
            "jumpBallLostPersonId",
            "foulDrawnPersonId",
            "blockPersonId",
            "assistPersonId",
            "stealPersonId",
        },
    },
}


def parse_game_id(game_id: str) -> tuple[int, str]:
    value = str(game_id).strip()
    if len(value) != 10 or not value.isdigit():
        raise ValueError(f"invalid WNBA game id: {game_id!r}")
    season_type = {"102": "RS", "104": "PS"}.get(value[:3])
    if season_type is None:
        raise ValueError(f"unsupported WNBA game-id prefix: {value[:3]}")
    return 2000 + int(value[3:5]), season_type


def positive_int(value: Any) -> int | None:
    try:
        result = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def descriptions(row: dict[str, str]) -> str:
    return " ".join(
        str(row.get(column) or "")
        for column in ("HOMEDESCRIPTION", "NEUTRALDESCRIPTION", "VISITORDESCRIPTION")
    )


def is_complete_last_row(flavor: str, row: dict[str, str] | None) -> bool:
    if not row:
        return False
    if flavor == "modern":
        return (
            str(row.get("actionType") or "").lower() == "game"
            and str(row.get("subType") or "").lower() == "end"
        ) or str(row.get("description") or "").strip().lower() == "game end"
    period = positive_int(row.get("PERIOD")) or 0
    return (
        str(row.get("EVENTMSGTYPE") or "") == "13"
        and period >= 4
        and "end of" in descriptions(row).lower()
    )


def header_digest(fieldnames: list[str]) -> str:
    payload = "\x1f".join(fieldnames).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def scan_pbp_directory(
    source_root: Path,
    flavor: str,
) -> tuple[dict[str, Any], set[int], set[str]]:
    config = PBP_CONFIG[flavor]
    paths = sorted((source_root / config["directory"]).glob("*.csv"))
    season_files: Counter[int] = Counter()
    season_rows: Counter[int] = Counter()
    season_complete: Counter[int] = Counter()
    season_empty: Counter[int] = Counter()
    season_types: Counter[str] = Counter()
    header_variants: Counter[str] = Counter()
    field_presence: Counter[str] = Counter()
    actor_ids: set[int] = set()
    game_ids: set[str] = set()
    total_rows = 0
    exact_duplicates = 0
    duplicate_keys = 0
    conflicting_duplicate_keys = 0
    malformed_rows = 0
    game_id_mismatches = 0

    for path in paths:
        season, season_type = parse_game_id(path.stem)
        game_ids.add(path.stem)
        season_files[season] += 1
        season_types[season_type] += 1
        row_count = 0
        last_row: dict[str, str] | None = None
        seen_rows: set[tuple[str, ...]] = set()
        seen_keys: dict[str, tuple[str, ...]] = {}
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            header_variants[header_digest(fieldnames)] += 1
            field_presence.update(fieldnames)
            for row in reader:
                row_count += 1
                total_rows += 1
                if None in row:
                    malformed_rows += 1
                values = tuple(str(row.get(column) or "") for column in fieldnames)
                if values in seen_rows:
                    exact_duplicates += 1
                else:
                    seen_rows.add(values)
                event_key = str(row.get(config["key"]) or "")
                if event_key in seen_keys:
                    duplicate_keys += 1
                    if seen_keys[event_key] != values:
                        conflicting_duplicate_keys += 1
                else:
                    seen_keys[event_key] = values
                if flavor == "classic" and str(row.get("GAME_ID") or "") != path.stem:
                    game_id_mismatches += 1
                for field in config["actor_fields"]:
                    actor_id = positive_int(row.get(field))
                    if actor_id is not None:
                        actor_ids.add(actor_id)
                last_row = row
        season_rows[season] += row_count
        if row_count == 0:
            season_empty[season] += 1
        if is_complete_last_row(flavor, last_row):
            season_complete[season] += 1

    missing_required = sorted(
        field for field in config["required"] if field_presence[field] != len(paths)
    )
    summary = {
        "directory": str(source_root / config["directory"]),
        "files": len(paths),
        "rows": total_rows,
        "season_type_files": dict(sorted(season_types.items())),
        "files_by_season": {str(key): value for key, value in sorted(season_files.items())},
        "rows_by_season": {str(key): value for key, value in sorted(season_rows.items())},
        "complete_files_by_season": {
            str(key): value for key, value in sorted(season_complete.items())
        },
        "empty_files_by_season": {
            str(key): value for key, value in sorted(season_empty.items()) if value
        },
        "header_variants": len(header_variants),
        "largest_header_variant_file_counts": sorted(
            header_variants.values(), reverse=True
        )[:10],
        "required_fields_missing_from_some_files": missing_required,
        "actor_ids": len(actor_ids),
        "exact_duplicate_rows": exact_duplicates,
        "duplicate_event_keys": duplicate_keys,
        "conflicting_duplicate_event_keys": conflicting_duplicate_keys,
        "malformed_rows": malformed_rows,
        "filename_game_id_mismatches": game_id_mismatches,
    }
    return summary, actor_ids, game_ids


def scan_schedule(source_root: Path) -> tuple[dict[str, Any], set[str]]:
    path = source_root / "data" / "wnba_game_dates.csv"
    game_ids: set[str] = set()
    seasons: Counter[int] = Counter()
    season_types: Counter[str] = Counter()
    dates: list[date] = []
    duplicate_ids = 0
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_id = str(row.get("gameId") or "").strip()
            if raw_id.endswith(".0"):
                raw_id = raw_id[:-2]
            raw_id = raw_id.zfill(10)
            if raw_id in game_ids:
                duplicate_ids += 1
            game_ids.add(raw_id)
            try:
                season, _ = parse_game_id(raw_id)
                seasons[season] += 1
            except ValueError:
                pass
            season_types[str(row.get("seasonType") or "")] += 1
            raw_date = str(row.get("date") or "")[:10]
            try:
                dates.append(datetime.strptime(raw_date, "%Y-%m-%d").date())
            except ValueError:
                continue
    return (
        {
            "path": str(path),
            "rows": sum(seasons.values()),
            "unique_game_ids": len(game_ids),
            "duplicate_game_ids": duplicate_ids,
            "files_by_season": {str(key): value for key, value in sorted(seasons.items())},
            "season_types": dict(sorted(season_types.items())),
            "date_min": min(dates).isoformat() if dates else None,
            "date_max": max(dates).isoformat() if dates else None,
        },
        game_ids,
    )


def scan_lineup_aggregates(source_root: Path) -> tuple[dict[str, Any], set[int]]:
    paths = sorted((source_root / "lineup_data").glob("*/*.csv"))
    categories: Counter[str] = Counter()
    files_by_season: Counter[int] = Counter()
    entity_lengths: Counter[int] = Counter()
    header_variants: Counter[str] = Counter()
    player_ids: set[int] = set()
    total_rows = 0
    for path in paths:
        season = int(path.parent.name)
        files_by_season[season] += 1
        category = "PS" if "_ps" in path.stem else "RS"
        category += "_OPPONENT" if "_vs" in path.stem else "_TEAM"
        categories[category] += 1
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            header_variants[header_digest(list(reader.fieldnames or []))] += 1
            for row in reader:
                total_rows += 1
                entity_ids = [value for value in str(row.get("EntityId") or "").split("-") if value]
                entity_lengths[len(entity_ids)] += 1
                for value in entity_ids:
                    player_id = positive_int(value)
                    if player_id is not None:
                        player_ids.add(player_id)
    return (
        {
            "directory": str(source_root / "lineup_data"),
            "grain": "season-team five-player aggregate; not event-level or paired opponent stints",
            "files": len(paths),
            "rows": total_rows,
            "categories": dict(sorted(categories.items())),
            "files_by_season": {
                str(key): value for key, value in sorted(files_by_season.items())
            },
            "header_variants": len(header_variants),
            "entity_id_player_counts": {
                str(key): value for key, value in sorted(entity_lengths.items())
            },
            "player_ids": len(player_ids),
        },
        player_ids,
    )


def dataframe_rating_summary(path: Path, as_of: date) -> tuple[dict[str, Any], Any]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "rating audit requires pandas plus a parquet engine; use the repo .venv"
        ) from exc

    frame = pd.read_parquet(path)
    date_column = pd.to_datetime(frame["date"], errors="coerce")
    key = ["date", "nba_id", "tm_id"] if "tm_id" in frame.columns else ["date", "nba_id"]
    summary: dict[str, Any] = {
        "path": str(path),
        "rows": len(frame),
        "columns": list(frame.columns),
        "date_min": date_column.min().date().isoformat(),
        "date_max": date_column.max().date().isoformat(),
        "dates": int(date_column.nunique()),
        "player_ids": int(frame["nba_id"].nunique()),
        "player_names": int(frame["player_name"].nunique()),
        "duplicate_primary_key_rows": int(frame.duplicated(key, keep=False).sum()),
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "null_cells": int(frame.isna().sum().sum()),
        "future_rows_after_as_of": int((date_column.dt.date > as_of).sum()),
        "future_dates_after_as_of": sorted(
            date_column.loc[date_column.dt.date > as_of]
            .dt.date.astype(str)
            .drop_duplicates()
            .tolist()
        ),
        "rows_per_date_min": int(frame.groupby("date").size().min()),
        "rows_per_date_median": float(frame.groupby("date").size().median()),
        "rows_per_date_max": int(frame.groupby("date").size().max()),
    }
    numeric_columns = [
        column
        for column in frame.columns
        if column not in {"date", "nba_id", "player_name", "tm_id"}
    ]
    summary["nonfinite_numeric_values"] = int(
        sum(
            (~frame[column].map(lambda value: math.isfinite(float(value)))).sum()
            for column in numeric_columns
        )
    )
    if {"o_dpm", "d_dpm", "dpm"}.issubset(frame.columns):
        summary["dpm_closure_max_abs_gap"] = float(
            (frame["dpm"] - frame["o_dpm"] - frame["d_dpm"]).abs().max()
        )
    return summary, frame[["nba_id", "player_name"]].drop_duplicates()


def scan_ratings(ratings_root: Path, as_of: date) -> dict[str, Any]:
    filenames = [
        "dpm_cols.parq",
        "pi_rapm_ratings.parq",
        "npi_rapm_ratings.parq",
    ]
    tables: dict[str, Any] = {}
    identities = []
    for filename in filenames:
        path = ratings_root / filename
        if not path.exists():
            tables[filename] = {"path": str(path), "missing": True}
            continue
        summary, identity = dataframe_rating_summary(path, as_of)
        tables[filename] = summary
        identities.append(identity)
    if identities:
        import pandas as pd

        combined = pd.concat(identities, ignore_index=True).drop_duplicates()
        ids_multiple_names = int(
            (combined.groupby("nba_id")["player_name"].nunique() > 1).sum()
        )
        names_multiple_ids = int(
            (combined.groupby("player_name")["nba_id"].nunique() > 1).sum()
        )
    else:
        ids_multiple_names = 0
        names_multiple_ids = 0
    return {
        "directory": str(ratings_root),
        "as_of_date": as_of.isoformat(),
        "tables": tables,
        "identity_conflicts": {
            "player_ids_with_multiple_name_strings": ids_multiple_names,
            "player_names_with_multiple_ids": names_multiple_ids,
        },
        "validation_join_rule": (
            "Use each player's latest rating_date strictly before game_date until "
            "the provider confirms whether date-D values are pregame or postgame."
        ),
    }


def git_metadata(source_root: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(source_root), *args],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()

    return {
        "origin": git("config", "--get", "remote.origin.url"),
        "commit": git("rev-parse", "HEAD"),
        "commit_time": git("log", "-1", "--format=%cI"),
        "dirty": bool(git("status", "--porcelain")),
    }


def build_audit(source_root: Path, ratings_root: Path, as_of: date) -> dict[str, Any]:
    classic, classic_actor_ids, classic_game_ids = scan_pbp_directory(
        source_root, "classic"
    )
    modern, modern_actor_ids, modern_game_ids = scan_pbp_directory(source_root, "modern")
    schedule, schedule_ids = scan_schedule(source_root)
    lineups, lineup_player_ids = scan_lineup_aggregates(source_root)
    all_actor_ids = classic_actor_ids | modern_actor_ids
    ratings = scan_ratings(ratings_root, as_of)
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "source_checkout": git_metadata(source_root),
        "play_by_play": {
            "classic": classic,
            "modern": modern,
            "game_overlap": {
                "both": len(classic_game_ids & modern_game_ids),
                "classic_only": len(classic_game_ids - modern_game_ids),
                "modern_only": len(modern_game_ids - classic_game_ids),
                "modern_share_of_classic_pct": round(
                    100 * len(classic_game_ids & modern_game_ids) / len(classic_game_ids),
                    2,
                )
                if classic_game_ids
                else 0.0,
            },
            "schedule_coverage": {
                "classic_games_in_schedule": len(classic_game_ids & schedule_ids),
                "classic_games_missing_schedule": len(classic_game_ids - schedule_ids),
                "modern_games_in_schedule": len(modern_game_ids & schedule_ids),
                "modern_games_missing_schedule": len(modern_game_ids - schedule_ids),
            },
        },
        "schedule": schedule,
        "lineup_aggregates": lineups,
        "identity_coverage": {
            "pbp_actor_ids": len(all_actor_ids),
            "lineup_player_ids": len(lineup_player_ids),
            "intersection": len(all_actor_ids & lineup_player_ids),
            "pbp_actor_ids_in_lineup_data_pct": round(
                100 * len(all_actor_ids & lineup_player_ids) / len(all_actor_ids), 2
            )
            if all_actor_ids
            else 0.0,
        },
        "ratings": ratings,
        "ingest_guardrails": [
            "Drop zero-row schedule placeholders and require a completed final event.",
            "Normalize modern CSVs by column name because headers evolve across games.",
            "Collapse exact duplicate classic rows but fail on conflicting duplicate event keys.",
            "Do not treat lineup_data as event lineups; it is season-team five-player aggregate data.",
            "Reconstruct modern event lineups from boxscore starters plus grouped substitutions.",
            "Use strict prior-date rating joins until rating timestamp semantics are confirmed.",
        ],
    }


def print_summary(audit: dict[str, Any]) -> None:
    classic = audit["play_by_play"]["classic"]
    modern = audit["play_by_play"]["modern"]
    ratings = audit["ratings"]["tables"]
    print(
        f"Classic PBP: {classic['files']:,} files, {classic['rows']:,} rows, "
        f"{classic['exact_duplicate_rows']:,} exact duplicate rows"
    )
    print(
        f"Modern PBP: {modern['files']:,} files, {modern['rows']:,} rows, "
        f"{modern['header_variants']:,} header variants"
    )
    for filename, table in ratings.items():
        if table.get("missing"):
            print(f"Ratings: {filename} missing")
        else:
            print(
                f"Ratings: {filename} {table['rows']:,} rows, "
                f"{table['date_min']} through {table['date_max']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--ratings-root", type=Path, default=DEFAULT_RATINGS_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--as-of-date",
        type=date.fromisoformat,
        default=date.today(),
        help="Flag rating rows later than this YYYY-MM-DD date",
    )
    args = parser.parse_args()

    audit = build_audit(args.source_root, args.ratings_root, args.as_of_date)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print_summary(audit)
    print(f"Audit report: {args.report}")


if __name__ == "__main__":
    main()
