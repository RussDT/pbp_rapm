#!/usr/bin/env python3
"""Run the restartable RAPM table build queue.

The queue deliberately keeps explanatory Net DECOMP, its ShotQuality split,
and the TD550 research VPM in separate stages and directories.  A failure in
one stage is recorded and does not prevent independent later stages from
running.  Re-running the queue is safe: the underlying runners reuse complete
artifacts unless ``--force`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PIPELINE_ROOT.parent
STATE_DIR = PIPELINE_ROOT / "results" / "rapm_overnight"
STATE_PATH = STATE_DIR / "run_state.json"
LOG_DIR = PIPELINE_ROOT / "logs" / "rapm_overnight" / "stages"
HISTORICAL_EV = (
    PIPELINE_ROOT
    / "results"
    / "historical_shot_quality"
    / "russell_v2_historical_bridge"
    / "russell_v2_historical_bridge_ev_1997_2015.parquet"
)
TARGET_DIR = PIPELINE_ROOT / "results" / "vpm_target_swap_netcont" / "processed"
MECHANISM_ROOT = PIPELINE_ROOT / "results" / "mechanism_rapm_registry"


@dataclass(frozen=True)
class Stage:
    stage_id: str
    description: str
    commands: tuple[tuple[str, ...], ...]
    depends_on: tuple[str, ...] = ()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def python_command(script: str, *args: str) -> tuple[str, ...]:
    return (sys.executable, str(SCRIPT_DIR / script), *map(str, args))


def current_parent_path(mechanism_id: str, metric: str, end_year: int) -> Path:
    start_year = end_year - 4
    compact = f"{start_year % 100:02d}_{end_year % 100:02d}"
    return (
        MECHANISM_ROOT
        / mechanism_id
        / compact
        / f"{metric.lower()}_{compact}_rs_rb_se_results.csv"
    )


def build_stages(args: argparse.Namespace) -> list[Stage]:
    shared = (
        "--release-end-year",
        str(args.release_end_year),
        "--workers",
        str(args.workers),
        "--cores-per-rapm",
        str(args.cores_per_rapm),
    )
    force = ("--force",) if args.force else ()
    end_yy = str(args.release_end_year % 100)
    start_yy = str((args.release_end_year - 4) % 100)

    assist_parent = current_parent_path(
        "assist_points_role_parent_rs", "ASSIST_POINTS", args.release_end_year
    )
    scoring_parent = current_parent_path(
        "scoring_role_parent_rs", "SCORING_PARENT", args.release_end_year
    )
    ts_parent = current_parent_path("ts_role_parent_rs", "TS", args.release_end_year)

    if getattr(args, "full_window_grid", False):
        all_grid = (
            "--all-one-to-five-season-windows",
            "--release-end-year",
            str(args.release_end_year),
            "--workers",
            str(args.workers),
            "--cores-per-rapm",
            str(args.cores_per_rapm),
        )
        return [
            Stage(
                "historical_sq_targets",
                "Attach the frozen 1997-2015 SQ-v2 bridge to net-continuation targets",
                (
                    python_command(
                        "build_vpm_target_swap.py",
                        "--ev-parquet",
                        str(HISTORICAL_EV),
                        "--years",
                        "1997-2015",
                        "--season-types",
                        "all",
                        "--processed-dir",
                        str(PIPELINE_ROOT / "processed_netcont_lane"),
                        "--output-dir",
                        str(TARGET_DIR),
                        "--rapm-source-dir",
                        str(PIPELINE_ROOT / "processed"),
                        "--link-unchanged-families",
                        "--net-lane",
                        "--overwrite",
                    ),
                ),
            ),
            Stage(
                "net_actual_full_grid",
                "All 140 rolling 1Y-5Y actual Net DECOMP windows, 1997-2026",
                (
                    python_command(
                        "run_net_decomp_rolling.py", *all_grid, *force
                    ),
                ),
            ),
            Stage(
                "net_sq_full_grid",
                "All 140 rolling 1Y-5Y explanatory SQ-v2 plus residual Net windows",
                (
                    python_command(
                        "run_net_sq_decomp_rolling.py",
                        *all_grid,
                        "--profile",
                        "explanatory",
                        *force,
                    ),
                ),
                depends_on=("historical_sq_targets",),
            ),
            Stage(
                "net_sq_td550_full_grid",
                "All 140 rolling 1Y-5Y TD550 SQ-v2 Net VPM research windows",
                (
                    python_command(
                        "run_net_sq_decomp_rolling.py",
                        *all_grid,
                        "--profile",
                        "vpm",
                        "--timedecay",
                        "--half-life",
                        "550",
                        *force,
                    ),
                ),
                depends_on=("historical_sq_targets",),
            ),
            Stage(
                "registry_snapshot",
                "Build the API-ready artifact registry snapshot",
                (python_command("build_rapm_table_registry.py"),),
            ),
        ]

    stages = [
        Stage(
            "net_actual_current",
            "Current 1Y-5Y actual Net DECOMP release family",
            (python_command("run_net_decomp_rolling.py", "--current-release", *shared, *force),),
        ),
        Stage(
            "net_sq_current",
            "Current 1Y-5Y explanatory Net DECOMP with SQ v2 plus residuals",
            (
                python_command(
                    "run_net_sq_decomp_rolling.py",
                    "--current-release",
                    "--profile",
                    "explanatory",
                    *shared,
                    *force,
                ),
            ),
        ),
        Stage(
            "net_sq_td550_current",
            "Current 1Y-5Y TD550 SQ v2 Net VPM research family",
            (
                python_command(
                    "run_net_sq_decomp_rolling.py",
                    "--current-release",
                    "--profile",
                    "vpm",
                    "--timedecay",
                    "--half-life",
                    "550",
                    *shared,
                    *force,
                ),
            ),
        ),
        Stage(
            "mechanisms_generic",
            "Registry-driven current mechanism RAPMs",
            (python_command("run_mechanism_rapm_registry.py", *shared, *force),),
        ),
        Stage(
            "mechanisms_specialized",
            "Current specialized shooter, passer/receiver, and scoring-role RAPMs",
            (
                python_command(
                    "run_shooter_oreb_rapm.py", start_yy, end_yy, "ALL", "--alpha", "3000"
                ),
                python_command(
                    "run_assist_points_hierarchy.py",
                    start_yy,
                    end_yy,
                    "RS",
                    "--parent-results",
                    str(assist_parent),
                ),
                python_command(
                    "run_scoring_role_hierarchy.py",
                    start_yy,
                    end_yy,
                    "RS",
                    "--parent-results",
                    str(scoring_parent),
                ),
                python_command(
                    "run_ts_shot_role_hierarchy.py",
                    start_yy,
                    end_yy,
                    "RS",
                    "--parent-results",
                    str(ts_parent),
                ),
                python_command(
                    "run_joint_ts_shooter_interaction.py",
                    start_yy,
                    end_yy,
                    "RS",
                    "--parent-results",
                    str(ts_parent),
                ),
            ),
            depends_on=("mechanisms_generic",),
        ),
        Stage(
            "historical_sq_targets",
            "Attach the frozen 1997-2015 SQ-v2 bridge to net-continuation targets",
            (
                python_command(
                    "build_vpm_target_swap.py",
                    "--ev-parquet",
                    str(HISTORICAL_EV),
                    "--years",
                    "1997-2015",
                    "--season-types",
                    "all",
                    "--processed-dir",
                    str(PIPELINE_ROOT / "processed_netcont_lane"),
                    "--output-dir",
                    str(TARGET_DIR),
                    "--rapm-source-dir",
                    str(PIPELINE_ROOT / "processed"),
                    "--link-unchanged-families",
                    "--net-lane",
                    "--overwrite",
                ),
            ),
        ),
        Stage(
            "net_sq_td550_history",
            "All 26 completed five-season TD550 SQ-v2 Net VPM research windows",
            (
                python_command(
                    "run_net_sq_decomp_rolling.py",
                    "--all-five-season-windows",
                    "--profile",
                    "vpm",
                    "--timedecay",
                    "--half-life",
                    "550",
                    "--workers",
                    str(args.workers),
                    "--cores-per-rapm",
                    str(args.cores_per_rapm),
                    *force,
                ),
            ),
            depends_on=("historical_sq_targets",),
        ),
        Stage(
            "registry_snapshot",
            "Build the API-ready artifact registry snapshot",
            (python_command("build_rapm_table_registry.py"),),
        ),
    ]
    return stages


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    os.replace(temporary, STATE_PATH)


def command_text(command: Iterable[str]) -> str:
    return " ".join(map(str, command))


def run_stage(stage: Stage, state: dict) -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{stage.stage_id}.log"
    record = {
        "stage_id": stage.stage_id,
        "description": stage.description,
        "status": "running",
        "started_at": utc_now(),
        "log": str(log_path.relative_to(REPO_ROOT)),
        "commands": [list(command) for command in stage.commands],
    }
    state["stages"][stage.stage_id] = record
    save_state(state)
    with log_path.open("w") as log:
        for index, command in enumerate(stage.commands, 1):
            log.write(f"COMMAND {index}/{len(stage.commands)}: {command_text(command)}\n")
            log.flush()
            result = subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                record.update(
                    {
                        "status": "failed",
                        "returncode": result.returncode,
                        "failed_command_index": index,
                        "finished_at": utc_now(),
                    }
                )
                return record
    record.update({"status": "complete", "returncode": 0, "finished_at": utc_now()})
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-end-year", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cores-per-rapm", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-history", action="store_true")
    parser.add_argument(
        "--full-window-grid",
        action="store_true",
        help="Build all rolling 1Y-5Y Net families from 1997 through the release year.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.full_window_grid and args.skip_history:
        parser.error("--full-window-grid requires historical SQ targets; omit --skip-history")

    stages = build_stages(args)
    if args.skip_history:
        stages = [
            stage
            for stage in stages
            if stage.stage_id not in {"historical_sq_targets", "net_sq_td550_history"}
        ]
    if args.dry_run:
        for stage in stages:
            print(f"STAGE {stage.stage_id}: {stage.description}")
            for command in stage.commands:
                print(f"  {command_text(command)}")
        return 0

    state = {
        "schema_version": 1,
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "status": "running",
        "started_at": utc_now(),
        "release_end_year": args.release_end_year,
        "pid": os.getpid(),
        "stages": {},
    }
    save_state(state)
    failed = False
    for stage in stages:
        dependency_failures = [
            dependency
            for dependency in stage.depends_on
            if state["stages"].get(dependency, {}).get("status") != "complete"
        ]
        if dependency_failures:
            record = {
                "stage_id": stage.stage_id,
                "description": stage.description,
                "status": "skipped_dependency",
                "dependencies": dependency_failures,
                "finished_at": utc_now(),
            }
        else:
            record = run_stage(stage, state)
        state["stages"][stage.stage_id] = record
        save_state(state)
        print(f"{stage.stage_id}: {record['status']}", flush=True)
        failed = failed or record["status"] in {"failed", "skipped_dependency"}

    state["status"] = "complete_with_errors" if failed else "complete"
    state["finished_at"] = utc_now()
    save_state(state)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
