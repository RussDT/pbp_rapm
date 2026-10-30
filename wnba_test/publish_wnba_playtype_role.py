#!/usr/bin/env python3
"""Publish the WNBA playtype role/zTS surface for the Databallr stat table.

This is the only sanctioned bridge from the research artifact built by
`build_wnba_playtype_difficulty.py` to the reader-facing `Role` and `zTS`
columns on databallr's WNBA stats table.

Three rules are enforced here rather than trusted downstream:

1. Only `descriptor_regime == "modern"` seasons ship. WNBA descriptor
   vocabulary drifts hard enough before 2018 that sparse-description seasons
   read as league-wide difficulty rather than player role.
2. The published `role` is the *scaled* column `playtype_diff_scaled`
   (`ROLE_DISPLAY_COEFFICIENT` x raw), so the reader identity
   `zTS = rTS + role` closes on the number actually shown. The raw column
   ships alongside it as `playtype_diff_raw` for audit, never for display.
3. `zTS` is deliberately NOT computed here. Databallr adds `role` to its own
   `rTS%` so the displayed identity closes exactly against the displayed rTS.

`zTS` here is the WNBA reader label for `rTS + role`. It is not the NBA
production `zTS`, whose chain (`+4` shift, `-1.211295`, `0.800618`) is
NBA-fitted and carries coefficient exactly 1.0. Never route these numbers
into the NBA `SV/100` formula.

Examples:
    .venv/bin/python wnba_test/publish_wnba_playtype_role.py
    .venv/bin/python wnba_test/publish_wnba_playtype_role.py --rebuild \
        --stage-databallr ../../products/databallr --upload
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "wnba_test" / "Results" / "playtype_difficulty"
DEFAULT_OUTPUT = DEFAULT_SOURCE / "publish"
PUBLISH_BASENAME = "wnba_playtype_role_current.csv"
SUPABASE_TABLE = "wnba_playtype_role"

# Matches the 50-attempt floor databallr already uses to qualify a player for
# overall-shooting percentiles. Below it the six-family mix is noise: a 1-FGA
# player can post a -12 role.
DEFAULT_MIN_FGA = 50

PUBLISH_COLUMNS = [
    "season",
    "is_playoffs",
    "player_id",
    "player_name",
    "pbp_fga",
    "role",
    "playtype_diff_raw",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rebuild_source(source: Path, season_type: str) -> None:
    """Refresh the difficulty artifact so an in-season 2026 role is current."""
    cmd = [
        sys.executable,
        str(ROOT / "wnba_test" / "build_wnba_playtype_difficulty.py"),
        "--output",
        str(source),
        "--season-type",
        season_type,
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def load_source(source: Path) -> tuple[pd.DataFrame, dict]:
    player_season = source / "wnba_playtype_player_season.csv"
    if not player_season.exists():
        raise FileNotFoundError(
            f"missing {player_season}; run build_wnba_playtype_difficulty.py "
            "or pass --rebuild"
        )
    frame = pd.read_csv(player_season)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    return frame, manifest


def build_publish_frame(frame: pd.DataFrame, min_fga: int) -> pd.DataFrame:
    required = {
        "feature_season",
        "player_id",
        "player_name",
        "pbp_fga",
        "playtype_diff",
        "playtype_diff_scaled",
        "descriptor_regime",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"source artifact is missing {sorted(missing)}; rebuild it with the "
            "current build_wnba_playtype_difficulty.py"
        )

    modern = frame[frame["descriptor_regime"] == "modern"].copy()
    if modern.empty:
        raise ValueError("no modern-regime rows to publish")

    qualified = modern[modern["pbp_fga"] >= min_fga].copy()
    if qualified.empty:
        raise ValueError(f"no rows clear the {min_fga} FGA floor")

    publish = pd.DataFrame(
        {
            "season": qualified["feature_season"].astype(int),
            # The artifact is regular-season descriptions only. Playoff rows on
            # databallr must not silently inherit a regular-season role.
            "is_playoffs": False,
            "player_id": qualified["player_id"].astype("int64").astype(str),
            "player_name": qualified["player_name"].astype(str),
            "pbp_fga": qualified["pbp_fga"].astype(int),
            "role": qualified["playtype_diff_scaled"].astype(float).round(4),
            "playtype_diff_raw": qualified["playtype_diff"].astype(float).round(4),
        }
    )
    publish = publish.sort_values(["season", "player_name"]).reset_index(drop=True)
    return publish[PUBLISH_COLUMNS]


def validate_publish_frame(frame: pd.DataFrame, coefficient: float) -> dict:
    """Prove the published surface before it can reach a reader."""
    dupes = frame.duplicated(subset=["season", "is_playoffs", "player_id"]).sum()
    if dupes:
        raise ValueError(f"{dupes} duplicate (season, is_playoffs, player_id) rows")

    if frame[["role", "playtype_diff_raw"]].isna().any().any():
        raise ValueError("null role or raw difficulty in the publish frame")

    # role must be exactly the scaled column, not the raw one.
    gap = (frame["role"] - coefficient * frame["playtype_diff_raw"]).abs().max()
    if gap > 5e-4:
        raise ValueError(
            f"role is not {coefficient} x raw difficulty (max gap {gap:.6f})"
        )

    nonzero = frame[frame["playtype_diff_raw"].abs() > 1e-9]
    if not nonzero.empty:
        shrunk = (nonzero["role"].abs() < nonzero["playtype_diff_raw"].abs()).all()
        if not shrunk:
            raise ValueError("role is not strictly inside raw difficulty")

    if bool(frame["is_playoffs"].any()):
        raise ValueError("playoff rows are not supported by this artifact")

    return {
        "rows": int(len(frame)),
        "players": int(frame["player_id"].nunique()),
        "seasons": [int(frame["season"].min()), int(frame["season"].max())],
        "max_role_minus_scaled_raw": float(gap),
        "role_min": float(frame["role"].min()),
        "role_max": float(frame["role"].max()),
        "rows_by_season": {
            str(season): int(count)
            for season, count in frame["season"].value_counts().sort_index().items()
        },
    }


def stage_databallr(path: Path, databallr_root: Path) -> list[str]:
    public_dirs = [
        databallr_root / "public",
        databallr_root / "new next js migration" / "public",
    ]
    staged: list[str] = []
    for public_dir in public_dirs:
        if not public_dir.exists():
            continue
        target = public_dir / path.name
        shutil.copyfile(path, target)
        staged.append(str(target))
    if not staged:
        raise RuntimeError(f"no Databallr public folders found under {databallr_root}")
    return staged


def upload_supabase(frame: pd.DataFrame, as_of: str) -> dict:
    """Upsert the board so the live site refreshes without a databallr deploy."""
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    from supabase import create_client

    url = os.getenv("SUPABASE_URL")
    # The service role key is required: the publishable key cannot write here.
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set to upload"
        )

    client = create_client(url, key)
    records = []
    for row in frame.to_dict(orient="records"):
        records.append(
            {
                "season": int(row["season"]),
                "is_playoffs": bool(row["is_playoffs"]),
                "player_id": str(row["player_id"]),
                "player_name": row["player_name"],
                "pbp_fga": int(row["pbp_fga"]),
                "role": float(row["role"]),
                "playtype_diff_raw": float(row["playtype_diff_raw"]),
                "updated_at": as_of,
            }
        )

    uploaded = 0
    chunk = 500
    for start in range(0, len(records), chunk):
        batch = records[start : start + chunk]
        client.table(SUPABASE_TABLE).upsert(
            batch, on_conflict="season,is_playoffs,player_id"
        ).execute()
        uploaded += len(batch)
    return {"table": SUPABASE_TABLE, "uploaded_rows": uploaded}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-fga", type=int, default=DEFAULT_MIN_FGA)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Refresh the difficulty artifact first (needed for a live season).",
    )
    parser.add_argument("--season-type", choices=["RS", "PS", "ALL"], default="RS")
    parser.add_argument(
        "--stage-databallr",
        type=Path,
        default=None,
        help="Copy the publish CSV into databallr public folders (no commit).",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help=f"Upsert into the Supabase {SUPABASE_TABLE} table.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    as_of = datetime.now(timezone.utc).isoformat()

    if args.rebuild:
        rebuild_source(args.source, args.season_type)

    frame, source_manifest = load_source(args.source)
    coefficient = float(source_manifest.get("role_display_coefficient", 0.81))
    publish = build_publish_frame(frame, args.min_fga)
    validation = validate_publish_frame(publish, coefficient)

    args.output.mkdir(parents=True, exist_ok=True)
    publish_csv = args.output / PUBLISH_BASENAME
    publish.to_csv(publish_csv, index=False)

    manifest = {
        "as_of": as_of,
        "source": str(args.source),
        "publish_csv": str(publish_csv),
        "publish_csv_sha256": sha256_file(publish_csv),
        "min_pbp_fga": args.min_fga,
        "descriptor_regime": "modern",
        "season_type": args.season_type,
        "role_display_coefficient": coefficient,
        "role_definition": (
            "role = role_display_coefficient x playtype_diff, in TS percentage "
            "points. Positive means a harder-than-average shot role."
        ),
        "zts_definition": (
            "zTS is computed downstream as rTS + role so the reader identity "
            "closes on the displayed rTS. This is the WNBA reader object, not "
            "the NBA production zTS chain."
        ),
        "validation": validation,
    }

    if args.stage_databallr:
        manifest["staged"] = stage_databallr(publish_csv, args.stage_databallr)
    if args.upload:
        manifest["supabase"] = upload_supabase(publish, as_of)

    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
