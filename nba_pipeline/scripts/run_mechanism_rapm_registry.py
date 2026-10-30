#!/usr/bin/env python3
"""Solve the generic mechanism RAPMs declared in the table registry.

The runner schedules only windows whose processed RS/PS inputs are complete.
Unavailable provider-era windows are recorded as skipped instead of being
silently represented by a shorter sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
DEFAULT_REGISTRY = PIPELINE_ROOT / "config" / "rapm_table_registry.json"
DEFAULT_PROCESSED_DIR = PIPELINE_ROOT / "processed"
DEFAULT_OUTPUT_ROOT = PIPELINE_ROOT / "results" / "mechanism_rapm_registry"
DEFAULT_CURRENT_OUTPUT_ROOT = PIPELINE_ROOT / "results" / "mechanism_td550_current"
DEFAULT_FUTURE_OUTPUT_ROOT = PIPELINE_ROOT / "results" / "mechanism_td550_future_selected"


@dataclass(frozen=True)
class Window:
    start_year: int
    end_year: int

    @property
    def compact(self) -> str:
        if self.start_year == self.end_year:
            return f"{self.end_year % 100:02d}"
        return f"{self.start_year % 100:02d}_{self.end_year % 100:02d}"


@dataclass(frozen=True)
class Job:
    mechanism_id: str
    metric: str
    processed_prefix: str
    season_type: str
    window: Window
    off_alpha: int
    def_alpha: int
    timedecay: bool = False
    half_life_days: float | None = None
    cv_alpha: bool = False
    cv_folds: int = 5
    cv_passes: int = 2
    cv_alpha_grid: tuple[float, ...] = ()


ALPHA_SELECTION_RE = re.compile(
    r"Alpha CV selected off_alpha=([0-9.]+) and def_alpha=([0-9.]+)"
)


def current_windows(end_year: int = 2026) -> list[Window]:
    return [Window(end_year - seasons + 1, end_year) for seasons in range(1, 6)]


def windows_for_rule(rule: str, end_year: int) -> list[Window]:
    if rule in {"current_up_to_5", "available_current"}:
        return current_windows(end_year)
    if rule == "current_5y":
        return [Window(end_year - 4, end_year)]
    raise ValueError(f"Unknown mechanism release window rule: {rule}")


def expected_inputs(processed_dir: Path, job: Job) -> list[Path]:
    suffixes = {
        "RS": ("",),
        "PS": ("_PS",),
        "ALL": ("", "_PS"),
    }[job.season_type]
    return [
        processed_dir / f"{job.processed_prefix}{year % 100:02d}{suffix}.parquet"
        for year in range(job.window.start_year, job.window.end_year + 1)
        for suffix in suffixes
    ]


def build_plan(
    registry: dict,
    processed_dir: Path,
    end_year: int,
) -> tuple[list[Job], list[dict[str, object]]]:
    jobs: list[Job] = []
    skipped: list[dict[str, object]] = []
    for mechanism in registry["mechanism_rapms"]:
        metric = mechanism.get("metric")
        if not metric:
            skipped.append(
                {
                    "mechanism_id": mechanism["mechanism_id"],
                    "status": "specialized",
                    "runner": mechanism.get("specialized_runner"),
                }
            )
            continue
        off_alpha, def_alpha = mechanism["alphas"]
        for window in windows_for_rule(mechanism["release_windows"], end_year):
            job = Job(
                mechanism_id=mechanism["mechanism_id"],
                metric=metric,
                processed_prefix=mechanism["processed_prefix"],
                season_type=mechanism["season_type"],
                window=window,
                off_alpha=int(off_alpha),
                def_alpha=int(def_alpha),
            )
            missing = [path for path in expected_inputs(processed_dir, job) if not path.exists()]
            if missing:
                skipped.append(
                    {
                        "mechanism_id": job.mechanism_id,
                        "metric": job.metric,
                        "window": job.window.compact,
                        "status": "missing_inputs",
                        "missing_count": len(missing),
                        "first_missing": str(missing[0]),
                    }
                )
            else:
                jobs.append(job)
    return jobs, skipped


def build_current_td_plan(
    registry: dict,
    processed_dir: Path,
    end_year: int,
) -> tuple[list[Job], list[dict[str, object]], dict[str, object]]:
    """Build the owner-curated, no-prior current mechanism board."""
    spec = registry.get("current_mechanism_view")
    if not isinstance(spec, dict):
        raise ValueError("Registry is missing current_mechanism_view")

    alpha_spec = spec.get("alpha_selection", {})
    if alpha_spec.get("method") != "weighted_grouped_game_cv":
        raise ValueError("current_mechanism_view requires weighted_grouped_game_cv")
    alpha_grid = tuple(float(value) for value in alpha_spec.get("grid", []))
    if not alpha_grid or any(value <= 0 for value in alpha_grid):
        raise ValueError("current_mechanism_view alpha grid must contain positive values")

    by_id = {item["mechanism_id"]: item for item in registry["mechanism_rapms"]}
    jobs: list[Job] = []
    skipped: list[dict[str, object]] = []
    default_start = int(spec["default_start_year"])
    half_life = float(spec["half_life_days"])
    season_type = str(spec.get("season_type", "ALL"))

    for requested in spec.get("mechanisms", []):
        mechanism_id = str(requested["mechanism_id"])
        mechanism = by_id.get(mechanism_id)
        if mechanism is None:
            raise ValueError(f"Unknown current mechanism_id: {mechanism_id}")
        metric = mechanism.get("metric")
        if not metric:
            skipped.append(
                {
                    "mechanism_id": mechanism_id,
                    "status": "specialized",
                    "runner": mechanism.get("specialized_runner"),
                }
            )
            continue

        start_year = int(requested.get("start_year", default_start))
        off_alpha, def_alpha = mechanism["alphas"]
        job = Job(
            mechanism_id=mechanism_id,
            metric=str(metric),
            processed_prefix=str(mechanism["processed_prefix"]),
            season_type=season_type,
            window=Window(start_year, end_year),
            off_alpha=int(off_alpha),
            def_alpha=int(def_alpha),
            timedecay=True,
            half_life_days=half_life,
            cv_alpha=True,
            cv_folds=int(alpha_spec.get("folds", 5)),
            cv_passes=int(alpha_spec.get("passes", 2)),
            cv_alpha_grid=alpha_grid,
        )
        missing = [path for path in expected_inputs(processed_dir, job) if not path.exists()]
        if missing:
            skipped.append(
                {
                    "mechanism_id": mechanism_id,
                    "metric": metric,
                    "window": job.window.compact,
                    "status": "missing_inputs",
                    "missing_count": len(missing),
                    "first_missing": str(missing[0]),
                    "coverage_note": requested.get("coverage_note"),
                }
            )
        else:
            jobs.append(job)
    return jobs, skipped, spec


def build_future_selected_td_plan(
    registry: dict,
    processed_dir: Path,
    end_year: int,
) -> tuple[list[Job], list[dict[str, object]], dict[str, object]]:
    """Build the zero-prior TD550 board using future-screen recommendations."""
    spec = registry.get("current_mechanism_view")
    if not isinstance(spec, dict):
        raise ValueError("Registry is missing current_mechanism_view")
    future = spec.get("future_validation", {})
    recommendations = future.get("recommended_tested_pairs", {})
    if not recommendations:
        raise ValueError("Registry has no future-selected mechanism alpha recommendations")

    by_id = {item["mechanism_id"]: item for item in registry["mechanism_rapms"]}
    jobs: list[Job] = []
    skipped: list[dict[str, object]] = []
    start_year = int(spec["default_start_year"])
    half_life = float(spec["half_life_days"])
    season_type = str(spec.get("season_type", "ALL"))
    for mechanism_id, pair in recommendations.items():
        mechanism = by_id.get(str(mechanism_id))
        if mechanism is None:
            raise ValueError(f"Unknown future-selected mechanism_id: {mechanism_id}")
        if not mechanism.get("metric"):
            raise ValueError(f"Future-selected mechanism requires a generic metric: {mechanism_id}")
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"Invalid future-selected alpha pair for {mechanism_id}: {pair}")
        job = Job(
            mechanism_id=str(mechanism_id),
            metric=str(mechanism["metric"]),
            processed_prefix=str(mechanism["processed_prefix"]),
            season_type=season_type,
            window=Window(start_year, end_year),
            off_alpha=int(pair[0]),
            def_alpha=int(pair[1]),
            timedecay=True,
            half_life_days=half_life,
        )
        missing = [path for path in expected_inputs(processed_dir, job) if not path.exists()]
        if missing:
            skipped.append(
                {
                    "mechanism_id": mechanism_id,
                    "metric": mechanism["metric"],
                    "window": job.window.compact,
                    "status": "missing_inputs",
                    "missing_count": len(missing),
                    "first_missing": str(missing[0]),
                }
            )
        else:
            jobs.append(job)
    return jobs, skipped, future


def _existing_result(output_dir: Path, metric: str, timedecay: bool = False) -> Path | None:
    matches = list(output_dir.rglob(f"{metric.lower()}_*_results.csv"))
    if timedecay:
        matches = [path for path in matches if "_td" in path.stem]
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def build_command(
    job: Job,
    processed_dir: Path,
    output_dir: Path,
    cores: int,
) -> tuple[list[str], dict[str, str]]:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "rapm.py"),
        job.metric,
        str(job.window.start_year % 100),
        str(job.window.end_year % 100),
        job.season_type,
        "--rubberband",
        "--season-effects",
        "--off-alpha",
        str(job.off_alpha),
        "--def-alpha",
        str(job.def_alpha),
        "--cores",
        str(cores),
    ]
    if job.timedecay:
        if job.half_life_days is None:
            raise ValueError("A time-decay job requires half_life_days")
        cmd.extend(["--half-life", f"{job.half_life_days:g}"])
    if job.cv_alpha:
        if not job.cv_alpha_grid:
            raise ValueError("An alpha-CV job requires cv_alpha_grid")
        cmd.extend(
            [
                "--cv-alpha",
                "--cv-folds",
                str(job.cv_folds),
                "--cv-passes",
                str(job.cv_passes),
                "--cv-alpha-grid",
                ",".join(f"{alpha:g}" for alpha in job.cv_alpha_grid),
            ]
        )
    env = os.environ.copy()
    env["NBA_PIPELINE_PROCESSED_DIR"] = str(processed_dir)
    env["NBA_PIPELINE_RESULTS_DIR"] = str(output_dir)
    return cmd, env


def parse_selected_alphas(output: str) -> tuple[float, float] | None:
    match = ALPHA_SELECTION_RE.search(output)
    return (float(match.group(1)), float(match.group(2))) if match else None


def alpha_cv_path(result_path: Path) -> Path:
    return result_path.with_name(result_path.name.replace("_results.csv", "_alpha_cv.csv"))


def alpha_cv_split_types(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return sorted({str(row.get("split_type", "")) for row in csv.DictReader(handle)})


def merge_partial_manifest(
    manifest_path: Path,
    records: list[dict[str, object]],
    skipped: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Preserve unaffected records when a filtered restart rewrites the manifest."""
    if not manifest_path.exists():
        return records, skipped
    previous = json.loads(manifest_path.read_text())
    record_map = {
        (str(row["mechanism_id"]), str(row["window"])): row
        for row in previous.get("records", [])
    }
    for row in records:
        record_map[(str(row["mechanism_id"]), str(row["window"]))] = row
    skipped_map = {
        (str(row.get("mechanism_id")), str(row.get("window"))): row
        for row in previous.get("skipped", [])
    }
    for row in skipped:
        skipped_map[(str(row.get("mechanism_id")), str(row.get("window")))] = row
    return list(record_map.values()), list(skipped_map.values())


def run_job(
    job: Job,
    processed_dir: Path,
    output_root: Path,
    cores: int,
    force: bool,
) -> dict[str, object]:
    output_dir = output_root / job.mechanism_id / job.window.compact
    log_path = output_dir / "solve.log"
    existing = _existing_result(output_dir, job.metric, job.timedecay)
    if existing is not None and not force:
        cv_path = alpha_cv_path(existing)
        selected = parse_selected_alphas(log_path.read_text()) if log_path.exists() else None
        reported_pair = selected if job.cv_alpha else (float(job.off_alpha), float(job.def_alpha))
        boundary = None
        if selected and job.cv_alpha_grid:
            low, high = min(job.cv_alpha_grid), max(job.cv_alpha_grid)
            boundary = {
                "offense": selected[0] in {low, high},
                "defense": selected[1] in {low, high},
            }
        return {
            "mechanism_id": job.mechanism_id,
            "metric": job.metric,
            "window": job.window.compact,
            "status": "existing",
            "output": str(existing),
            "selected_off_alpha": reported_pair[0] if reported_pair else None,
            "selected_def_alpha": reported_pair[1] if reported_pair else None,
            "alpha_grid_boundary": boundary,
            "alpha_cv": str(cv_path) if cv_path.exists() else None,
            "alpha_cv_split_types": alpha_cv_split_types(cv_path),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd, env = build_command(job, processed_dir, output_dir, cores)
    started = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.time() - started
    log_path.write_text(
        "$ "
        + " ".join(cmd)
        + f"\n\nelapsed_seconds={elapsed:.3f}"
        + "\n\nSTDOUT:\n"
        + result.stdout
        + "\n\nSTDERR:\n"
        + result.stderr
    )
    if result.returncode:
        return {
            "mechanism_id": job.mechanism_id,
            "metric": job.metric,
            "window": job.window.compact,
            "status": "failed",
            "returncode": result.returncode,
            "log": str(log_path),
            "elapsed_seconds": round(elapsed, 3),
        }
    output = _existing_result(output_dir, job.metric, job.timedecay)
    if output is None:
        raise FileNotFoundError(f"Solver succeeded without a result for {job.metric}: {output_dir}")
    selected = parse_selected_alphas(result.stdout + "\n" + result.stderr)
    reported_pair = selected if job.cv_alpha else (float(job.off_alpha), float(job.def_alpha))
    cv_path = alpha_cv_path(output)
    boundary = None
    if selected and job.cv_alpha_grid:
        low, high = min(job.cv_alpha_grid), max(job.cv_alpha_grid)
        boundary = {
            "offense": selected[0] in {low, high},
            "defense": selected[1] in {low, high},
        }
    return {
        "mechanism_id": job.mechanism_id,
        "metric": job.metric,
        "window": job.window.compact,
        "status": "built",
        "output": str(output),
        "selected_off_alpha": reported_pair[0] if reported_pair else None,
        "selected_def_alpha": reported_pair[1] if reported_pair else None,
        "alpha_grid_boundary": boundary,
        "alpha_cv": str(cv_path) if cv_path.exists() else None,
        "alpha_cv_split_types": alpha_cv_split_types(cv_path),
        "elapsed_seconds": round(elapsed, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--release-end-year", type=int, default=2026)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--current-td550",
        action="store_true",
        help="Run the curated zero-prior TD550 current mechanism board with alpha CV.",
    )
    mode.add_argument(
        "--future-selected",
        action="store_true",
        help="Run the zero-prior TD550 current board with future-screen alpha recommendations.",
    )
    parser.add_argument(
        "--mechanism-id",
        action="append",
        help="Restrict the plan to one mechanism id; repeat for multiple mechanisms.",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cores-per-rapm", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    registry = json.loads(args.registry.read_text())
    current_spec = None
    if args.current_td550:
        jobs, skipped, current_spec = build_current_td_plan(
            registry, args.processed_dir, args.release_end_year
        )
    elif args.future_selected:
        jobs, skipped, current_spec = build_future_selected_td_plan(
            registry, args.processed_dir, args.release_end_year
        )
    else:
        jobs, skipped = build_plan(registry, args.processed_dir, args.release_end_year)
    if args.mechanism_id:
        selected_ids = set(args.mechanism_id)
        known_ids = {job.mechanism_id for job in jobs} | {
            str(record.get("mechanism_id")) for record in skipped
        }
        unknown = selected_ids - known_ids
        if unknown:
            raise ValueError(f"Unknown or unavailable mechanism ids: {sorted(unknown)}")
        jobs = [job for job in jobs if job.mechanism_id in selected_ids]
        skipped = [
            record for record in skipped if record.get("mechanism_id") in selected_ids
        ]
    output_root = args.output_root or (
        DEFAULT_CURRENT_OUTPUT_ROOT
        if args.current_td550
        else DEFAULT_FUTURE_OUTPUT_ROOT
        if args.future_selected
        else DEFAULT_OUTPUT_ROOT
    )
    print(f"mechanism_jobs={len(jobs)} skipped_or_specialized={len(skipped)}")
    if args.dry_run:
        for job in jobs:
            td = f" TD{job.half_life_days:g}" if job.timedecay else ""
            cv = (
                f" CV[{','.join(f'{value:g}' for value in job.cv_alpha_grid)}]"
                if job.cv_alpha
                else ""
            )
            print(
                f"PLAN {job.mechanism_id} {job.metric} {job.window.compact} "
                f"{job.season_type}{td}{cv}"
            )
        for record in skipped:
            print(f"SKIP {record}")
        return 0

    records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_job,
                job,
                args.processed_dir,
                output_root,
                args.cores_per_rapm,
                args.force,
            ): job
            for job in jobs
        }
        for completed, future in enumerate(as_completed(futures), 1):
            record = future.result()
            records.append(record)
            print(
                f"[{completed}/{len(jobs)}] {record['mechanism_id']} "
                f"{record['window']} {record['status']}",
                flush=True,
            )

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "run_manifest.json"
    if args.mechanism_id:
        records, skipped = merge_partial_manifest(manifest_path, records, skipped)
    manifest = {
        "schema_version": 2,
        "registry": str(args.registry),
        "release_end_year": args.release_end_year,
        "mode": (
            "current_td550"
            if args.current_td550
            else "future_selected_td550"
            if args.future_selected
            else "fixed_windows"
        ),
        "current_mechanism_view": current_spec,
        "records": sorted(records, key=lambda row: (str(row["mechanism_id"]), str(row["window"]))),
        "skipped": skipped,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {manifest_path}")
    return 1 if any(record["status"] == "failed" for record in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
