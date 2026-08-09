#!/usr/bin/env python3
"""Fetch private monthly EPM-only snapshots for retrospective title attribution."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
import requests

try:
    from nba_pipeline.scripts.fetch_epm_data import (
        fetch_snapshot,
        load_api_key,
        snapshot_paths,
        write_snapshot,
    )
except ModuleNotFoundError:
    from fetch_epm_data import (  # type: ignore[no-redef]
        fetch_snapshot,
        load_api_key,
        snapshot_paths,
        write_snapshot,
    )


DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "cache" / "epm_title"
TITLE_COLUMNS = (
    "date",
    "game_dt",
    "game_optimized",
    "player_id",
    "player_name",
    "oepm",
    "depm",
    "epm",
)


def normalize_title_snapshot(
    rows: list[dict[str, object]], requested_date: str
) -> tuple[pd.DataFrame, int]:
    """Keep player EPM only; collapse duplicate stint rows only when EPM is identical."""
    frame = pd.DataFrame(rows)
    missing = sorted(set(TITLE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"EPM title response is missing columns: {missing}")
    frame = frame.loc[:, TITLE_COLUMNS].copy()
    frame["player_id"] = pd.to_numeric(frame["player_id"], errors="raise").astype("int64")
    modes = set(pd.to_numeric(frame["game_optimized"], errors="raise").astype(int))
    if modes != {0}:
        raise ValueError(f"EPM title snapshots require game_optimized=0, got {sorted(modes)}")
    for column in ("oepm", "depm", "epm"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    for column in ("date", "game_dt"):
        parsed = pd.to_datetime(frame[column], errors="coerce")
        invalid = frame[column].notna() & parsed.isna()
        if invalid.any():
            raise ValueError(f"EPM title response contains invalid {column}")
        frame[column] = parsed.dt.strftime("%Y-%m-%d")

    duplicate_rows = int(frame["player_id"].duplicated(keep=False).sum())
    for player_id, group in frame[frame["player_id"].duplicated(keep=False)].groupby(
        "player_id"
    ):
        for column in ("player_name", "date", "game_dt", "oepm", "depm", "epm"):
            if group[column].fillna("<NA>").astype(str).nunique() != 1:
                raise ValueError(
                    f"Conflicting EPM duplicate for player_id {player_id}: {column}"
                )
    frame = frame.drop_duplicates("player_id", keep="last")
    frame.insert(0, "requested_date", requested_date)
    return frame.sort_values("player_id", kind="stable").reset_index(drop=True), duplicate_rows


def monthly_dates(start: str, end: str) -> list[str]:
    first = date.fromisoformat(start)
    last = date.fromisoformat(end)
    year, month = first.year, first.month
    values: list[str] = []
    while (year, month) <= (last.year, last.month):
        values.append(date(year, month, 1).isoformat())
        month += 1
        if month == 13:
            year += 1
            month = 1
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2001-11-01")
    parser.add_argument("--end", default=date.today().replace(day=1).isoformat())
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--key-file", type=Path)
    args = parser.parse_args()
    api_key = load_api_key(args.key_file)
    session = requests.Session()
    fetched = skipped = duplicate_rows = 0
    for requested_date in monthly_dates(args.start, args.end):
        csv_path, manifest_path = snapshot_paths(args.cache_dir, requested_date, 0)
        if csv_path.exists() and manifest_path.exists():
            skipped += 1
            continue
        rows = fetch_snapshot(
            requested_date,
            game_optimized=0,
            api_key=api_key,
            session=session,
        )
        frame, collapsed = normalize_title_snapshot(rows, requested_date)
        write_snapshot(
            frame,
            cache_dir=args.cache_dir,
            requested_date=requested_date,
            game_optimized=0,
        )
        fetched += 1
        duplicate_rows += collapsed
        if fetched % 25 == 0:
            print(f"fetched={fetched} latest={requested_date}", flush=True)
    print(
        f"complete fetched={fetched} skipped={skipped} "
        f"duplicate_rows_audited={duplicate_rows}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
