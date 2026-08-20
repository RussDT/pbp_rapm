#!/usr/bin/env python3
"""Prepare the WNBA VPM (factor-specific box-prior board) for Databallr.

Databallr's `/api/supabase/wnba_rapm` endpoint and every WNBA consumer behind it
(stats table, hub ticker, dashboard, player profile) read one CSV with the
`weighted_factors_alt3_efg_value` column schema and assume **one row per
player** keyed by `player_id` with a `Latest_Year` label. This script turns the
promoted current-season board (`wnba_test/Results/current/wnba_current_metrics.csv`,
built by `update_wnba_metrics.py`) into that publish shape and validates it.

Outputs (default `wnba_test/Results/current/publish/`):

- `wnba_vpm_boxprior_current.csv` — the drop-in file: one row per 2026 player,
  same column order as the legacy Databallr WNBA RAPM CSV, plus the prior/update
  audit columns and `season`, `season_type`, `model_variant`,
  `box_prior_source`, `as_of` metadata columns.
- `wnba_vpm_boxprior_history_2022_2026.csv` — the per-season 2022-2026 reader
  history (`build_wnba_metric_history.py` output) with the 2026 rows replaced by
  the refreshed current board. It has multiple rows per player and is **not**
  drop-in compatible with the current Databallr consumers; it is kept for a
  season-aware endpoint.
- `publish_manifest.json` — provenance (source paths, sha256, as-of, game
  count), validation report, top-10, and the Databallr staging checklist.

- `wnba_vpm_rack_payload.json` — the payload behind databallr's `/wnba/vpm`
  page (board + meta + the 2022-2026 player-profile box rows). The page fetches
  this file directly, the same way `/wnba/galaxy` fetches its JSON; it is a
  mutable pointer file, so it must not be cached immutably.

`--stage-databallr PATH` copies the drop-in file and the rack payload into
`PATH/public/` and `PATH/new next js migration/public/` when those folders
exist. It never commits or deploys anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wnba_test.update_wnba_metrics import validate_current_board  # noqa: E402

RESULTS_DIR = ROOT / "wnba_test" / "Results"
CURRENT_DIR = RESULTS_DIR / "current"
DEFAULT_BOARD = CURRENT_DIR / "wnba_current_metrics.csv"
DEFAULT_MANIFEST = CURRENT_DIR / "latest_manifest.json"
DEFAULT_PRIORS = RESULTS_DIR / "box_prior_true_prior" / "box_priors_2026.csv"
DEFAULT_HISTORY = (
    RESULTS_DIR
    / "box_prior_true_prior"
    / "history"
    / "wnba_single_season_metrics_2022_2026.csv"
)
DEFAULT_OUTPUT_DIR = CURRENT_DIR / "publish"
PUBLISH_BASENAME = "wnba_vpm_boxprior_current.csv"
HISTORY_BASENAME = "wnba_vpm_boxprior_history_2022_2026.csv"
RACK_PAYLOAD_BASENAME = "wnba_vpm_rack_payload.json"
MODEL_VARIANT = "single_season_box_trueprior_v1"
QUALIFYING_POSSESSIONS = 1000

# Column order of the legacy Databallr file
# `wnba_weighted_factors_alt3_efg_value_21_26_all_td700_a1750_4000.csv`.
LEGACY_COLUMNS = [
    "player_id", "player_name", "Latest_Year",
    "oALT_EFG_VALUE", "oALT_EFG_RIM", "oALT_EFG_RIM_FREQ", "oALT_EFG_RIM_FG",
    "oALT_EFG_MID", "oALT_EFG_MID_FREQ", "oALT_EFG_MID_FG",
    "oALT_EFG_THREE", "oALT_EFG_THREE_FREQ", "oALT_EFG_THREE_FG",
    "oALT_FT", "oALT_FT_FREQ", "oALT_FT_SEVERITY", "oALT_TOV_VALUE", "oSC", "oFC",
    "oDISPLAY_SUM", "oRESID", "oALT_TOV_LOSS_VALUE", "oALT_BADPASS_TOV_VALUE",
    "oALT_SCORING_TOV_VALUE", "oALT_TOV_VALUE_split",
    "dALT_EFG_VALUE", "dALT_EFG_RIM", "dALT_EFG_RIM_FREQ", "dALT_EFG_RIM_FG",
    "dALT_EFG_MID", "dALT_EFG_MID_FREQ", "dALT_EFG_MID_FG",
    "dALT_EFG_THREE", "dALT_EFG_THREE_FREQ", "dALT_EFG_THREE_FG",
    "dALT_FT", "dALT_FT_FREQ", "dALT_FT_SEVERITY", "dALT_TOV_VALUE", "dSC", "dFC",
    "dDISPLAY_SUM", "dRESID", "dALT_TOV_LOSS_VALUE", "dALT_BADPASS_TOV_VALUE",
    "dALT_SCORING_TOV_VALUE", "dALT_TOV_VALUE_split",
    "nALT_EFG_VALUE", "nALT_FT", "nALT_FT_FREQ", "nALT_FT_SEVERITY",
    "nALT_TOV_VALUE", "nSC", "nDISPLAY_SUM", "RESID",
    "off", "def", "net_rapm", "possessions", "off_poss", "def_poss",
]
AUDIT_COLUMNS = [
    "prior_off", "prior_def", "prior_net_rapm",
    "delta_off", "delta_def", "delta_net_rapm", "has_prior",
    "box_off", "box_def", "box_net", "box_prior_source",
]
META_COLUMNS = ["team", "season", "season_type", "model_variant", "qualified", "as_of"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def load_board(board_path: Path) -> pd.DataFrame:
    board = pd.read_csv(board_path, dtype={"player_id": str})
    board["player_id"] = _normalize_id(board["player_id"])
    return board


def attach_box_prior_audit(board: pd.DataFrame, priors_path: Path | None) -> pd.DataFrame:
    out = board.copy()
    for column in ("box_off", "box_def", "box_net"):
        if column not in out.columns:
            out[column] = np.nan
    if "box_prior_source" not in out.columns:
        out["box_prior_source"] = pd.NA
    if priors_path is None or not priors_path.exists():
        return out
    priors = pd.read_csv(priors_path, dtype={"player_id": str})
    priors["player_id"] = _normalize_id(priors["player_id"])
    keep = ["player_id", "box_prior_source", "box_off", "box_def", "box_net"]
    priors = priors[keep].drop_duplicates("player_id", keep="last")
    merged = out.drop(columns=["box_prior_source", "box_off", "box_def", "box_net"]).merge(
        priors, on="player_id", how="left", validate="one_to_one"
    )
    return merged


def attach_team(board: pd.DataFrame, totals_path: Path | None) -> pd.DataFrame:
    """Join the PBP Stats current-season team abbreviation as a reader convenience.

    The Databallr consumers take team from the advanced-stats rows, so this
    column is informational only and stays blank when the totals cache is
    unavailable.
    """
    out = board.copy()
    out["team"] = pd.NA
    if totals_path is None or not totals_path.exists():
        return out
    totals = pd.read_csv(totals_path, dtype={"EntityId": str}, usecols=["EntityId", "TeamAbbreviation"])
    totals["EntityId"] = _normalize_id(totals["EntityId"])
    totals = totals.drop_duplicates("EntityId", keep="last").rename(
        columns={"EntityId": "player_id", "TeamAbbreviation": "team"}
    )
    out = out.drop(columns=["team"]).merge(totals, on="player_id", how="left", validate="one_to_one")
    return out


def build_publish_frame(
    board: pd.DataFrame,
    *,
    season: int,
    season_type: str,
    as_of: str,
) -> pd.DataFrame:
    missing = [column for column in LEGACY_COLUMNS if column not in board.columns]
    if missing:
        raise RuntimeError(f"current board is missing legacy publish columns: {missing}")
    out = board.copy()
    if "team" not in out.columns:
        out["team"] = pd.NA
    out["Latest_Year"] = int(season)
    out["season"] = int(season)
    out["season_type"] = season_type
    out["model_variant"] = MODEL_VARIANT
    out["qualified"] = pd.to_numeric(out["possessions"], errors="coerce") >= QUALIFYING_POSSESSIONS
    out["as_of"] = as_of
    for column in AUDIT_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
    ordered = LEGACY_COLUMNS + AUDIT_COLUMNS + META_COLUMNS
    out = out[ordered].sort_values(["net_rapm", "possessions"], ascending=[False, False])
    return out.reset_index(drop=True)


def validate_publish_frame(frame: pd.DataFrame, tolerance: float = 0.0021) -> dict[str, Any]:
    if frame["player_id"].duplicated().any():
        raise RuntimeError("publish frame has duplicate player ids")
    if (frame["Latest_Year"].astype(int) != frame["season"].astype(int)).any():
        raise RuntimeError("Latest_Year must equal season on every publish row")
    numeric_columns = [
        column
        for column in LEGACY_COLUMNS
        if column not in {"player_id", "player_name", "Latest_Year"}
    ]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all():
        raise RuntimeError("publish frame has missing or nonfinite legacy values")

    identities = {
        "net = off + def": ("net_rapm", ["off", "def"]),
        "off = oDISPLAY_SUM + oRESID": ("off", ["oDISPLAY_SUM", "oRESID"]),
        "def = dDISPLAY_SUM + dRESID": ("def", ["dDISPLAY_SUM", "dRESID"]),
        "nDISPLAY_SUM = oDISPLAY_SUM + dDISPLAY_SUM": (
            "nDISPLAY_SUM",
            ["oDISPLAY_SUM", "dDISPLAY_SUM"],
        ),
        "oDISPLAY_SUM = oEFG + oFT + oTOV + oSC": (
            "oDISPLAY_SUM",
            ["oALT_EFG_VALUE", "oALT_FT", "oALT_TOV_VALUE", "oSC"],
        ),
        "dDISPLAY_SUM = dEFG + dFT + dTOV + dSC": (
            "dDISPLAY_SUM",
            ["dALT_EFG_VALUE", "dALT_FT", "dALT_TOV_VALUE", "dSC"],
        ),
        "oEFG = rim + mid + three": (
            "oALT_EFG_VALUE",
            ["oALT_EFG_RIM", "oALT_EFG_MID", "oALT_EFG_THREE"],
        ),
        "dEFG = rim + mid + three": (
            "dALT_EFG_VALUE",
            ["dALT_EFG_RIM", "dALT_EFG_MID", "dALT_EFG_THREE"],
        ),
        "nEFG = oEFG + dEFG": ("nALT_EFG_VALUE", ["oALT_EFG_VALUE", "dALT_EFG_VALUE"]),
        "nFT = oFT + dFT": ("nALT_FT", ["oALT_FT", "dALT_FT"]),
        "nTOV = oTOV + dTOV": ("nALT_TOV_VALUE", ["oALT_TOV_VALUE", "dALT_TOV_VALUE"]),
        "nSC = oSC + dSC": ("nSC", ["oSC", "dSC"]),
    }
    for side in ("o", "d"):
        for zone in ("RIM", "MID", "THREE"):
            identities[f"{side}{zone} = freq + fg"] = (
                f"{side}ALT_EFG_{zone}",
                [f"{side}ALT_EFG_{zone}_FREQ", f"{side}ALT_EFG_{zone}_FG"],
            )
        identities[f"{side}FT = freq + severity"] = (
            f"{side}ALT_FT",
            [f"{side}ALT_FT_FREQ", f"{side}ALT_FT_SEVERITY"],
        )
        identities[f"{side}TOV loss = badpass + scoring"] = (
            f"{side}ALT_TOV_LOSS_VALUE",
            [f"{side}ALT_BADPASS_TOV_VALUE", f"{side}ALT_SCORING_TOV_VALUE"],
        )
    gaps: dict[str, float] = {}
    for name, (parent, children) in identities.items():
        gap = (numeric[parent] - sum(numeric[child] for child in children)).abs()
        gaps[name] = float(gap.max())
    max_gap = max(gaps.values())
    if max_gap > tolerance:
        worst = max(gaps, key=gaps.get)
        raise RuntimeError(f"publish identity failed: {worst} max gap {max_gap:.6f}")
    return {
        "rows": int(len(frame)),
        "unique_players": int(frame["player_id"].nunique()),
        "qualified_rows": int(frame["qualified"].sum()),
        "max_identity_gap": max_gap,
        "identity_gaps": gaps,
        "tolerance": tolerance,
    }


def build_history_frame(
    history_path: Path,
    publish: pd.DataFrame,
    *,
    season: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    history = pd.read_csv(history_path, dtype={"player_id": str})
    history["player_id"] = _normalize_id(history["player_id"])
    prior_seasons = history[history["season"].astype(int) < int(season)].copy()
    replaced = int((history["season"].astype(int) == int(season)).sum())

    current = publish.copy()
    current["wnba_impact"] = current["net_rapm"]
    current["wnba_offense"] = current["off"]
    current["wnba_defense"] = current["def"]
    current["rank"] = current["wnba_impact"].rank(method="min", ascending=False).astype("Int64")
    qualified_rank = current.loc[current["qualified"], "wnba_impact"].rank(
        method="min", ascending=False
    )
    current["qualified_rank"] = pd.Series(pd.NA, index=current.index, dtype="Int64")
    current.loc[current["qualified"], "qualified_rank"] = qualified_rank.astype("Int64")
    if "minutes" not in current.columns:
        current["minutes"] = np.nan

    columns = list(prior_seasons.columns)
    for column in columns:
        if column not in current.columns:
            current[column] = np.nan
    combined = pd.concat([prior_seasons[columns], current[columns]], ignore_index=True)
    combined = combined.sort_values(
        ["season", "wnba_impact", "possessions"], ascending=[False, False, False]
    ).reset_index(drop=True)

    if combined.duplicated(["season", "player_id"]).any():
        raise RuntimeError("history frame has duplicate season/player keys")
    net_gap = (
        pd.to_numeric(combined["wnba_impact"], errors="coerce")
        - pd.to_numeric(combined["wnba_offense"], errors="coerce")
        - pd.to_numeric(combined["wnba_defense"], errors="coerce")
    ).abs()
    if float(net_gap.max()) > 0.0021:
        raise RuntimeError(f"history net identity failed: {float(net_gap.max()):.6f}")
    report = {
        "rows": int(len(combined)),
        "replaced_current_season_rows": replaced,
        "season_rows": {
            str(int(year)): int(count) for year, count in combined.groupby("season").size().items()
        },
        "max_net_identity_gap": float(net_gap.max()),
        "note": (
            "Multiple rows per player; not drop-in compatible with the current "
            "Databallr wnba_rapm consumers, which assume one row per player."
        ),
    }
    return combined, report


def top_rows(frame: pd.DataFrame, n: int = 10) -> list[dict[str, Any]]:
    cols = ["player_id", "player_name", "team", "off", "def", "net_rapm", "possessions", "box_prior_source"]
    return [
        {
            key: (None if (isinstance(value, float) and not np.isfinite(value)) else value)
            for key, value in row.items()
        }
        for row in frame[cols].head(n).to_dict(orient="records")
    ]


def build_rack_payload(
    publish_path: Path, manifest_path: Path, history_path: Path, output_dir: Path
) -> Path:
    """Write the JSON payload that databallr's `/wnba/vpm` page fetches.

    It is exactly the object the self-contained research rack embeds: the published
    board, the run meta, and the 2022-2026 per-player box rows behind the profile
    modules. The page owns the presentation; this repo owns the numbers.
    """
    from wnba_test import build_wnba_vpm_rack as rack

    payload = rack.build_payload(publish_path, manifest_path)
    board_ids = {p["id"] for p in payload["players"]}
    payload["profile"] = rack.build_profile(rack.BOX_DATA_DIR, history_path, board_ids)
    out = output_dir / RACK_PAYLOAD_BASENAME
    out.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return out


def stage_databallr(paths: list[Path], databallr_root: Path) -> list[str]:
    public_dirs = [
        databallr_root / "public",
        databallr_root / "new next js migration" / "public",
    ]
    staged: list[str] = []
    for public_dir in public_dirs:
        if not public_dir.exists():
            continue
        for path in paths:
            target = public_dir / path.name
            shutil.copyfile(path, target)
            staged.append(str(target))
    if not staged:
        raise RuntimeError(f"no Databallr public folders found under {databallr_root}")
    return staged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--board", type=Path, default=DEFAULT_BOARD)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--priors", type=Path, default=DEFAULT_PRIORS)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument(
        "--totals",
        type=Path,
        default=None,
        help="PBP Stats totals CSV for the team column; defaults to the manifest box_totals path",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--season", type=int, default=None, help="Defaults to the manifest season")
    parser.add_argument("--season-type", default="RS")
    parser.add_argument("--as-of", default=None, help="Defaults to the manifest as_of")
    parser.add_argument("--skip-history", action="store_true")
    parser.add_argument(
        "--skip-rack-payload", action="store_true",
        help="do not build wnba_vpm_rack_payload.json (the /wnba/vpm page payload)",
    )
    parser.add_argument(
        "--stage-databallr",
        type=Path,
        default=None,
        help="Databallr repo root; copies the drop-in CSV into its public folders",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest: dict[str, Any] = {}
    if args.manifest.exists():
        manifest = json.loads(args.manifest.read_text())
    if manifest and manifest.get("status") not in (None, "complete"):
        raise RuntimeError(f"latest manifest status is {manifest.get('status')!r}, refusing to publish")
    season = args.season or int(manifest.get("season") or date.today().year)
    as_of = args.as_of or str(manifest.get("as_of") or date.today().isoformat())

    totals_path: Path | None = args.totals
    if totals_path is None:
        manifest_totals = (manifest.get("box_totals") or {}).get("path")
        totals_path = Path(manifest_totals) if manifest_totals else None

    board_report = validate_current_board(args.board)
    board = attach_box_prior_audit(load_board(args.board), args.priors)
    board = attach_team(board, totals_path)
    publish = build_publish_frame(board, season=season, season_type=args.season_type, as_of=as_of)
    publish_report = validate_publish_frame(publish)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    publish_path = args.output_dir / PUBLISH_BASENAME
    publish.to_csv(publish_path, index=False)

    history_report: dict[str, Any] | None = None
    history_path: Path | None = None
    if not args.skip_history and args.history.exists():
        history, history_report = build_history_frame(args.history, publish, season=season)
        history_path = args.output_dir / HISTORY_BASENAME
        history.to_csv(history_path, index=False)

    rack_payload_path: Path | None = None
    if not args.skip_rack_payload and history_path is not None:
        rack_payload_path = build_rack_payload(publish_path, args.manifest, history_path, args.output_dir)

    staged: list[str] = []
    if args.stage_databallr is not None:
        stage_paths = [publish_path] + ([rack_payload_path] if rack_payload_path else [])
        staged = stage_databallr(stage_paths, args.stage_databallr)

    source_summary = {}
    for step in manifest.get("steps_run", []) or []:
        if step.get("name") == "fetch_completed_games":
            source_summary["fetch_elapsed_seconds"] = step.get("elapsed_seconds")
    for key in ("source_games", "processed_possessions", "board_players", "latest_game_id", "promoted"):
        if key in manifest:
            source_summary[key] = manifest[key]

    out_manifest = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of": as_of,
        "season": season,
        "season_type": args.season_type,
        "model_variant": MODEL_VARIANT,
        "solve": {"off_alpha": 5000, "def_alpha": 40000, "prior_scale": 1.5,
                  "policy": manifest.get("model_policy", "reuse_frozen_pre_2026_selection")},
        "sources": {
            "board": {"path": str(args.board), "sha256": sha256_file(args.board)},
            "manifest": {"path": str(args.manifest), "status": manifest.get("status")},
            "priors": {"path": str(args.priors), "exists": args.priors.exists()},
            "history": {"path": str(args.history), "exists": args.history.exists()},
        },
        "source_summary": source_summary,
        "board_validation": board_report,
        "publish_validation": publish_report,
        "history_validation": history_report,
        "outputs": {
            "publish_csv": {"path": str(publish_path), "sha256": sha256_file(publish_path), "rows": int(len(publish))},
            "history_csv": (
                {"path": str(history_path), "sha256": sha256_file(history_path)} if history_path else None
            ),
            "rack_payload_json": (
                {"path": str(rack_payload_path), "sha256": sha256_file(rack_payload_path)}
                if rack_payload_path else None
            ),
        },
        "staged_databallr_files": staged,
        "top_10": top_rows(publish),
        "databallr_checklist": [
            "Copy publish_csv and rack_payload_json into databallr/public/ and 'new next js migration/public/' (or use --stage-databallr).",
            "Point backend/app/routes/supabase/wnba.py WNBA_RAPM_CSV and the WNBA_RAPM_CSV_URL default at the new basename.",
            "Bump the Flask cache key prefix (wnba_rapm_v2 -> v3) or wait out the 3600s cache after deploy.",
            "Relabel the headline WNBA 'RAPM' metric strings to 'VPM' on the surfaces that read /wnba_rapm.",
            "Deploy frontend (serves public/ CSV) before or with the backend so the URL fallback resolves.",
        ],
        "consumer_contract": (
            "One row per player, keyed by player_id, Latest_Year = season of the row. "
            "Consumers rank by net_rapm without deduping, and the dashboard requests limit=500."
        ),
    }
    manifest_path = args.output_dir / "publish_manifest.json"
    manifest_path.write_text(json.dumps(out_manifest, indent=2, default=str) + "\n")

    print(json.dumps({
        "publish_csv": str(publish_path),
        "rows": int(len(publish)),
        "max_identity_gap": publish_report["max_identity_gap"],
        "history_csv": str(history_path) if history_path else None,
        "rack_payload_json": str(rack_payload_path) if rack_payload_path else None,
        "staged": staged,
        "manifest": str(manifest_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
