#!/usr/bin/env python3
"""Update the current WNBA box-prior impact board from completed games."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
WNBA_ROOT = ROOT / "wnba_test"
NORMALIZED_DIR = WNBA_ROOT / "GabrielNormalized"
CLEAN_DIR = WNBA_ROOT / "GabrielNormalizedClean"
PROCESSED_DIR = WNBA_ROOT / "Processed"
RESULTS_DIR = WNBA_ROOT / "Results"
BOX_PRIOR_DIR = RESULTS_DIR / "box_prior_true_prior"
CURRENT_DIR = RESULTS_DIR / "current"
DAILY_DIR = RESULTS_DIR / "daily_updates"

UPDATE_STEPS = [
    "stage_current_source",
    "fetch_completed_games",
    "validate_lineups",
    "refresh_box_totals",
    "rebuild_processed_surfaces",
    "rebuild_clean_decomposition",
    "build_current_box_prior_board",
    "validate_current_board",
    "promote_source_and_board",
]


def build_plan(as_of: date) -> dict[str, Any]:
    return {
        "as_of": as_of.isoformat(),
        "season": as_of.year,
        "season_type": "Regular Season",
        "model_policy": "reuse_frozen_pre_2026_selection",
        "steps": UPDATE_STEPS,
    }


def validate_current_board(path: Path, tolerance: float = 0.0021) -> dict[str, Any]:
    board = pd.read_csv(path, dtype={"player_id": str})
    if board.empty:
        raise RuntimeError(f"current board is empty: {path}")
    if board["player_id"].duplicated().any():
        raise RuntimeError("current board has duplicate player ids")

    identities = {
        "net": ("net_rapm", ["off", "def"]),
        "prior_net": ("prior_net_rapm", ["prior_off", "prior_def"]),
        "delta_net": ("delta_net_rapm", ["delta_off", "delta_def"]),
    }
    for side in ("o", "d"):
        for zone in ("RIM", "MID", "THREE"):
            identities[f"{side.lower()}_{zone.lower()}"] = (
                f"{side}ALT_EFG_{zone}",
                [f"{side}ALT_EFG_{zone}_FREQ", f"{side}ALT_EFG_{zone}_FG"],
            )
        identities[f"{side.lower()}_ft"] = (
            f"{side}ALT_FT",
            [f"{side}ALT_FT_FREQ", f"{side}ALT_FT_SEVERITY"],
        )
        identities[f"{side.lower()}_tov"] = (
            f"{side}ALT_TOV_LOSS_VALUE",
            [f"{side}ALT_BADPASS_TOV_VALUE", f"{side}ALT_SCORING_TOV_VALUE"],
        )

    required = {
        "player_id",
        "player_name",
        "possessions",
        *(column for parent, children in identities.values() for column in [parent, *children]),
    }
    missing = sorted(required.difference(board.columns))
    if missing:
        raise RuntimeError(f"current board is missing required columns: {missing}")

    numeric = board[list(required.difference({"player_id", "player_name"}))].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy()).all():
        raise RuntimeError("current board has missing or nonfinite required values")

    gaps = {}
    for name, (parent, children) in identities.items():
        gap = (
            pd.to_numeric(board[parent], errors="coerce")
            - sum(pd.to_numeric(board[child], errors="coerce") for child in children)
        ).abs()
        gaps[name] = float(gap.max())
    max_gap = max(gaps.values())
    if max_gap > tolerance:
        raise RuntimeError(
            f"current board additive identity failed: max gap {max_gap:.6f}"
        )
    return {
        "players": int(len(board)),
        "max_identity_gap": max_gap,
        "identity_gaps": gaps,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_summary(path: Path, game_column: str | None = None) -> dict[str, Any]:
    usecols = [game_column] if game_column else None
    frame = pd.read_csv(
        path,
        usecols=usecols,
        dtype={game_column: str} if game_column else None,
        low_memory=False,
    )
    summary = {
        "path": str(path),
        "rows": int(len(frame)),
        "sha256": sha256_file(path),
    }
    if game_column:
        game_ids = (
            frame[game_column]
            .dropna()
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .sort_values()
        )
        summary["games"] = int(game_ids.nunique())
        summary["first_game_id"] = game_ids.iloc[0] if not game_ids.empty else None
        summary["last_game_id"] = game_ids.iloc[-1] if not game_ids.empty else None
    return summary


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_command(
    name: str,
    command: list[str],
    run_dir: Path,
    steps: list[dict[str, Any]],
) -> None:
    log_path = run_dir / "logs" / f"{len(steps) + 1:02d}_{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] starting", flush=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.monotonic() - started
    step = {
        "name": name,
        "command": command,
        "returncode": int(result.returncode),
        "elapsed_seconds": round(elapsed, 3),
        "log": str(log_path),
    }
    steps.append(step)
    if result.returncode != 0:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-30:])
        raise RuntimeError(f"{name} failed with exit {result.returncode}\n{tail}")
    print(f"[{name}] complete in {elapsed:.1f}s", flush=True)


def processed_paths(season_suffix: str) -> list[Path]:
    return [
        PROCESSED_DIR / f"{prefix}{season_suffix}.csv"
        for prefix in (
            "RAPM",
            "TS",
            "REB",
            "TOV",
            "FIRST_CHANCE",
            "SECOND_CHANCE_CLEAN",
        )
    ]


def backup_files(paths: list[Path], backup_dir: Path) -> dict[Path, Path | None]:
    backups: dict[Path, Path | None] = {}
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.exists():
            backup = backup_dir / path.name
            shutil.copy2(path, backup)
            backups[path] = backup
        else:
            backups[path] = None
    return backups


def restore_files(backups: dict[Path, Path | None]) -> None:
    for destination, backup in backups.items():
        if backup is None:
            if destination.exists():
                destination.unlink()
        else:
            atomic_copy(backup, destination)


def top_players(path: Path, limit: int = 10) -> list[dict[str, Any]]:
    board = pd.read_csv(path)
    columns = [
        "player_id",
        "player_name",
        "net_rapm",
        "off",
        "def",
        "possessions",
        "prior_net_rapm",
        "delta_net_rapm",
    ]
    return (
        board.sort_values("net_rapm", ascending=False)
        .head(limit)[columns]
        .replace({np.nan: None})
        .to_dict("records")
    )


def execute_update(args: argparse.Namespace) -> dict[str, Any]:
    today = date.today()
    if args.as_of != today:
        raise RuntimeError(
            f"execution only supports today's completed CDN state ({today}); "
            f"use --plan-only for another date"
        )
    season = args.as_of.year
    if season != 2026:
        raise RuntimeError("the frozen daily WNBA lane currently supports 2026 only")

    run_dir = DAILY_DIR / args.as_of.isoformat()
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    staging_dir = Path(
        tempfile.mkdtemp(prefix="staging_", dir=run_dir)
    )
    staged_raw_dir = staging_dir / "raw"
    staged_clean_dir = staging_dir / "clean"
    staged_raw_dir.mkdir(parents=True)
    staged_clean_dir.mkdir(parents=True)

    season_suffix = f"{season % 100:02d}"
    raw_name = f"WNBA{season_suffix}.csv"
    production_raw = NORMALIZED_DIR / raw_name
    production_clean = CLEAN_DIR / raw_name
    staged_raw = staged_raw_dir / raw_name
    staged_clean = staged_clean_dir / raw_name
    selection_report = args.selection_report or (BOX_PRIOR_DIR / "report.json")
    stable_board = CURRENT_DIR / "wnba_current_metrics.csv"
    prior_stable_board = (
        csv_summary(stable_board) if stable_board.exists() else None
    )
    steps: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        **build_plan(args.as_of),
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "run_directory": str(run_dir),
        "selection_report": str(selection_report),
        "previous_current_board": prior_stable_board,
        "steps_run": steps,
    }
    processed = processed_paths(season_suffix)
    backups = backup_files(processed, staging_dir / "processed_backup")

    try:
        if not production_raw.exists():
            raise FileNotFoundError(production_raw)
        shutil.copy2(production_raw, staged_raw)
        source_before = csv_summary(staged_raw, "game_id")

        run_command(
            "fetch_completed_games",
            [
                sys.executable,
                str(WNBA_ROOT / "fetch_wnba_cdn.py"),
                "--season",
                str(season),
                "--season-type",
                "Regular Season",
                "--max-misses",
                str(args.max_misses),
                "--sleep",
                str(args.sleep),
                "--output",
                str(staged_raw),
            ],
            run_dir,
            steps,
        )
        source_after = csv_summary(staged_raw, "game_id")
        if source_after["games"] < source_before["games"]:
            raise RuntimeError("staged CDN source lost previously available games")

        run_command(
            "validate_lineups",
            [
                sys.executable,
                str(WNBA_ROOT / "validate_wnba_normalized_lineups.py"),
                "--input-dir",
                str(staged_raw_dir),
                "--output-dir",
                str(staged_clean_dir),
                "--write-clean",
            ],
            run_dir,
            steps,
        )
        lineup_report_path = staged_clean_dir / "lineup_validation_report.json"
        lineup_report = json.loads(lineup_report_path.read_text())
        current_lineup = lineup_report["files"][raw_name]
        if current_lineup["invalid_game_count"]:
            raise RuntimeError(
                "current-season lineup validation rejected games: "
                f"{current_lineup['invalid_games']}"
            )
        if not staged_clean.exists():
            raise RuntimeError("lineup validator did not write the staged clean season")

        from wnba_test.fetch_wnba_cdn import update_player_index_map

        update_player_index_map(
            pd.read_csv(staged_raw, dtype={"game_id": str}, low_memory=False),
            WNBA_ROOT / "player_index_map.csv",
        )

        run_command(
            "refresh_box_totals",
            [
                sys.executable,
                str(WNBA_ROOT / "wnba_stats_source.py"),
                "--refresh",
                str(season),
            ],
            run_dir,
            steps,
        )
        box_cache = ROOT.parent / "wnba" / "data" / f"{season}_pbp.csv"
        if not box_cache.exists():
            raise RuntimeError(f"box totals refresh did not write {box_cache}")

        run_command(
            "rebuild_processed_surfaces",
            [
                sys.executable,
                str(WNBA_ROOT / "process_rapm_wnba.py"),
                str(staged_clean),
                "--year",
                str(season),
                "--season",
                str(season),
            ],
            run_dir,
            steps,
        )
        run_command(
            "rebuild_clean_decomposition",
            [
                sys.executable,
                str(WNBA_ROOT / "clean_decomp_wnba.py"),
                "process",
                str(staged_clean),
            ],
            run_dir,
            steps,
        )
        missing_processed = [str(path) for path in processed if not path.exists()]
        if missing_processed:
            raise RuntimeError(
                f"current processing did not produce required CSVs: {missing_processed}"
            )

        output_label = f"box_trueprior_daily_{args.as_of:%Y%m%d}"
        run_command(
            "build_current_box_prior_board",
            [
                sys.executable,
                str(WNBA_ROOT / "build_wnba_box_priors.py"),
                "--output-dir",
                str(BOX_PRIOR_DIR),
                "--current-only",
                "--selection-report",
                str(selection_report),
                "--output-label",
                output_label,
            ],
            run_dir,
            steps,
        )
        current_build_report = json.loads(
            (BOX_PRIOR_DIR / "current_build_report.json").read_text()
        )
        built_board = Path(current_build_report["board_path"])
        if not built_board.is_absolute():
            built_board = ROOT / built_board
        board_validation = validate_current_board(built_board)

        atomic_copy(staged_raw, production_raw)
        atomic_copy(staged_clean, production_clean)
        atomic_copy(built_board, stable_board)
        source_before["path"] = str(production_raw)
        source_before["state"] = "pre_update_snapshot"
        source_after["path"] = str(production_raw)
        source_after["state"] = "promoted_current_source"
        current_lineup["path"] = str(production_clean)

        manifest.update(
            {
                "status": "complete",
                "completed_at": datetime.now().astimezone().isoformat(),
                "source": {
                    "before": source_before,
                    "after": source_after,
                    "new_games": source_after["games"] - source_before["games"],
                    "new_rows": source_after["rows"] - source_before["rows"],
                },
                "lineup_validation": current_lineup,
                "box_totals": csv_summary(box_cache),
                "processed_surfaces": [csv_summary(path) for path in processed],
                "model": {
                    "policy": "reuse_frozen_pre_2026_selection",
                    "selected_spec": current_build_report["selected_spec"],
                    "box_prior_training_years": current_build_report[
                        "box_prior_training_years"
                    ],
                    "box_prior_source_counts": current_build_report[
                        "box_prior_source_counts"
                    ],
                    "box_exposure_audit": current_build_report[
                        "box_exposure_audit"
                    ],
                },
                "board": {
                    **csv_summary(stable_board),
                    **board_validation,
                    "built_path": str(built_board),
                    "top_players": top_players(stable_board),
                },
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, default=float) + "\n")
        atomic_copy(manifest_path, CURRENT_DIR / "latest_manifest.json")
        return manifest
    except Exception as exc:
        restore_files(backups)
        manifest.update(
            {
                "status": "failed",
                "failed_at": datetime.now().astimezone().isoformat(),
                "error": str(exc),
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, default=float) + "\n")
        raise
    finally:
        if not args.keep_staging:
            shutil.rmtree(staging_dir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--max-misses", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--selection-report", type=Path)
    parser.add_argument("--keep-staging", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = build_plan(args.as_of)
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return 0
    manifest = execute_update(args)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "as_of": manifest["as_of"],
                "new_games": manifest["source"]["new_games"],
                "games": manifest["source"]["after"]["games"],
                "board": manifest["board"]["path"],
                "players": manifest["board"]["players"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
