#!/usr/bin/env python3
"""Fetch date-stamped Dunks & Threes EPM/E-Skills snapshots.

The provider's date contract is predictive: a snapshot requested for date D
does not include games played on D. Generated CSV/manifest files live under
``nba_pipeline/cache/epm`` by default and are intentionally gitignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import pandas as pd
import requests


API_URL = "https://dunksandthrees.com/api/v1/epm-all"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "cache" / "epm"
REQUIRED_COLUMNS = {
    "date",
    "game_dt",
    "game_optimized",
    "player_id",
    "player_name",
    "p_fg3pct",
    "p_fgpct_mid",
    "p_fg3a_100",
    "p_fga_mid_100",
    "p_orb_100",
    "oepm",
    "depm",
    "epm",
}


def parse_iso_date(value: str) -> str:
    """Return a canonical YYYY-MM-DD date or raise argparse's value error."""
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid ISO date {value!r}; expected YYYY-MM-DD"
        ) from exc


def load_api_key(key_file: Optional[Path] = None) -> str:
    """Load the API key without ever logging its value."""
    token = os.environ.get("EPM_API_KEY", "").strip()
    if not token and key_file is not None and key_file.exists():
        token = key_file.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError(
            "missing EPM API credential: set EPM_API_KEY or pass --key-file"
        )
    return token


def fetch_snapshot(
    requested_date: str,
    *,
    game_optimized: int,
    api_key: str,
    session: Optional[requests.Session] = None,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """Fetch one snapshot and validate its top-level response contract."""
    client = session or requests.Session()
    response = client.get(
        API_URL,
        params={
            "date": requested_date,
            "game_optimized": int(game_optimized),
        },
        headers={
            "Authorization": api_key,
            "Accept": "application/json",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    rows = response.json()
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            f"EPM response for {requested_date} must be a non-empty JSON array"
        )
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"EPM response for {requested_date} contains non-object rows")
    return rows


def normalize_snapshot(
    rows: Iterable[dict[str, Any]],
    *,
    requested_date: str,
    game_optimized: int,
) -> pd.DataFrame:
    """Normalize a response and fail loudly on schema or identity drift."""
    frame = pd.DataFrame(list(rows))
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"EPM response is missing required columns: {missing}")

    frame["player_id"] = pd.to_numeric(frame["player_id"], errors="raise").astype("int64")
    if frame["player_id"].duplicated().any():
        duplicates = (
            frame.loc[frame["player_id"].duplicated(keep=False), "player_id"]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(f"EPM response has duplicate player_id values: {duplicates[:10]}")

    modes = set(pd.to_numeric(frame["game_optimized"], errors="raise").astype(int))
    if modes != {int(game_optimized)}:
        raise ValueError(
            f"EPM response game_optimized values {sorted(modes)} do not match "
            f"requested mode {game_optimized}"
        )

    parsed_date = pd.to_datetime(frame["date"], errors="coerce")
    if parsed_date.isna().any():
        raise ValueError("EPM response contains invalid date values")
    frame["date"] = parsed_date.dt.strftime("%Y-%m-%d")

    # A player without a completed NBA game can legitimately have no game_dt.
    parsed_game_date = pd.to_datetime(frame["game_dt"], errors="coerce")
    invalid_game_date = frame["game_dt"].notna() & parsed_game_date.isna()
    if invalid_game_date.any():
        raise ValueError("EPM response contains invalid game_dt values")
    frame["game_dt"] = parsed_game_date.dt.strftime("%Y-%m-%d")

    frame.insert(0, "requested_date", requested_date)
    return frame.sort_values(["player_id"], kind="stable").reset_index(drop=True)


def snapshot_paths(
    cache_dir: Path,
    requested_date: str,
    game_optimized: int,
) -> Tuple[Path, Path]:
    stem = f"epm_all_{requested_date}_go{int(game_optimized)}"
    return cache_dir / f"{stem}.csv", cache_dir / f"{stem}.manifest.json"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(text)
    os.replace(temp_path, path)


def write_snapshot(
    frame: pd.DataFrame,
    *,
    cache_dir: Path,
    requested_date: str,
    game_optimized: int,
    overwrite: bool = False,
) -> Tuple[Path, Path, dict[str, Any]]:
    """Atomically write a canonical CSV plus provenance/checksum manifest."""
    csv_path, manifest_path = snapshot_paths(
        cache_dir, requested_date, game_optimized
    )
    if not overwrite and (csv_path.exists() or manifest_path.exists()):
        raise FileExistsError(
            f"snapshot already exists for {requested_date}; pass --overwrite "
            "only after auditing the existing cache"
        )

    csv_text = frame.to_csv(index=False, lineterminator="\n")
    digest = hashlib.sha256(csv_text.encode("utf-8")).hexdigest()
    effective_dates = sorted(frame["date"].dropna().astype(str).unique().tolist())
    game_dates = sorted(frame["game_dt"].dropna().astype(str).unique().tolist())
    manifest = {
        "schema_version": 1,
        "source": "Dunks & Threes /api/v1/epm-all",
        "requested_date": requested_date,
        "provider_effective_dates": effective_dates,
        "provider_latest_game_date": max(game_dates) if game_dates else None,
        "game_optimized": int(game_optimized),
        "row_count": int(len(frame)),
        "columns": frame.columns.tolist(),
        "csv_sha256": digest,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "date_contract": (
            "Provider predictions requested for date D exclude games played on D."
        ),
    }

    _atomic_write_text(csv_path, csv_text)
    _atomic_write_text(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return csv_path, manifest_path, manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dates",
        nargs="+",
        type=parse_iso_date,
        help="one or more prediction dates; defaults to today's local date",
    )
    parser.add_argument(
        "--game-optimized",
        type=int,
        choices=(0, 1),
        default=0,
        help="0 for game-agnostic E-Skills/EPM; 1 for game-optimized values",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--key-file",
        type=Path,
        help="optional legacy key file; EPM_API_KEY is preferred",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing requested-date snapshot after validation",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dates = args.dates or [date.today().isoformat()]
    api_key = load_api_key(args.key_file)
    session = requests.Session()

    for requested_date in dates:
        rows = fetch_snapshot(
            requested_date,
            game_optimized=args.game_optimized,
            api_key=api_key,
            session=session,
            timeout=args.timeout,
        )
        frame = normalize_snapshot(
            rows,
            requested_date=requested_date,
            game_optimized=args.game_optimized,
        )
        csv_path, manifest_path, manifest = write_snapshot(
            frame,
            cache_dir=args.cache_dir,
            requested_date=requested_date,
            game_optimized=args.game_optimized,
            overwrite=args.overwrite,
        )
        print(
            f"{requested_date}: wrote {len(frame)} players to {csv_path}; "
            f"provider effective date(s)={manifest['provider_effective_dates']}; "
            f"manifest={manifest_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
