#!/usr/bin/env python3
"""Merge the standalone DARKO TLOC challenger into Help Index app data."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "nba_pipeline" / "results" / "darko_playoff_title_lift"
DEFAULT_APP_DIR = REPO_ROOT / "teammate-help-app" / "public" / "data" / "championship-help"
APP_SHARDS = 64


def read_gzip_json(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_gzip_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as handle:
        json.dump(payload, handle, separators=(",", ":"), allow_nan=False)


def clean_record(row: dict[str, object]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in row.items():
        if pd.isna(value):
            output[key] = None
        elif isinstance(value, (np.integer,)):
            output[key] = int(value)
        elif isinstance(value, (np.floating,)):
            output[key] = round(float(value), 6)
        elif isinstance(value, (np.bool_,)):
            output[key] = bool(value)
        else:
            output[key] = value
    return output


def merge_rows(
    base: list[dict[str, object]],
    additions: list[dict[str, object]],
    key_fields: tuple[str, ...],
) -> list[dict[str, object]]:
    index = {tuple(row.get(field) for field in key_fields): row for row in base}
    for addition in additions:
        key = tuple(addition.get(field) for field in key_fields)
        target = index.get(key)
        if target is None:
            target = {field: addition.get(field) for field in key_fields}
            base.append(target)
            index[key] = target
        target.update(addition)
    return sorted(base, key=lambda row: tuple(row.get(field) or 0 for field in key_fields))


def publish(results_dir: Path, app_dir: Path) -> dict[str, int]:
    seasons = pd.read_csv(results_dir / "player_season_estimates.csv")
    careers = pd.read_csv(results_dir / "career_player_estimates.csv")
    metadata = json.loads((results_dir / "metadata.json").read_text())
    seasons = seasons.rename(
        columns={
            "era_local_salary_peer_placement_count": "darko_placement_count",
            "era_local_budget_mean_signed_salary_gap_cap_points": (
                "darko_mean_signed_salary_gap_cap_points"
            ),
        }
    )
    season_fields = [
        "season", "player_id", "player_name", "team_id", "team_abbr",
        "is_playoff_participant", "salary", "salary_cap", "salary_share",
        "darko_placement_count", "darko_mean_signed_salary_gap_cap_points",
        *[column for column in seasons.columns if column.startswith("darko_")],
    ]
    season_fields = list(dict.fromkeys(field for field in season_fields if field in seasons.columns))
    career_fields = [
        "player_id", "player_name",
        *[column for column in careers.columns if column.startswith("darko_")],
    ]
    season_records = [clean_record(row) for row in seasons[season_fields].to_dict("records")]
    career_records = [clean_record(row) for row in careers[career_fields].to_dict("records")]

    career_path = app_dir / "players.json.gz"
    season_path = app_dir / "season-players.json.gz"
    merged_careers = merge_rows(read_gzip_json(career_path), career_records, ("player_id",))
    merged_seasons = merge_rows(
        read_gzip_json(season_path), season_records, ("season", "player_id")
    )
    write_gzip_json(career_path, merged_careers)
    write_gzip_json(season_path, merged_seasons)

    season_by_shard: dict[int, list[dict[str, object]]] = {shard: [] for shard in range(APP_SHARDS)}
    for row in season_records:
        season_by_shard[int(row["player_id"]) % APP_SHARDS].append(row)
    for shard in range(APP_SHARDS):
        path = app_dir / "players" / f"{shard:02d}.json.gz"
        payload = read_gzip_json(path)
        payload["seasons"] = merge_rows(
            payload.get("seasons", []),
            season_by_shard[shard],
            ("season", "player_id"),
        )
        write_gzip_json(path, payload)

    app_metadata_path = app_dir / "metadata.json"
    app_metadata = json.loads(app_metadata_path.read_text())
    contract = app_metadata.setdefault("model", {}).setdefault(
        "title_lift_over_contract", {}
    )
    contract["darko_playoff_prior_challenger"] = metadata
    app_metadata_path.write_text(json.dumps(app_metadata, indent=2))
    return {
        "season_rows": len(season_records),
        "career_rows": len(career_records),
        "season_2026_rows": int(seasons["season"].eq(2026).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--app-dir", type=Path, default=DEFAULT_APP_DIR)
    args = parser.parse_args()
    print(publish(args.results_dir, args.app_dir))


if __name__ == "__main__":
    main()
