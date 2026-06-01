#!/usr/bin/env python3
"""Build rolling NBA Alt3 EFG-value weighted-factor bundles.

This script assumes the standard Alt3 weighted-factor base files already exist
for most windows. It solves missing EFG-value components, assembles the display
bundle, writes weighted-factor CSVs to `nba_pipeline/master_results`, and can
publish the generated files to the downstream `rapms` repo.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from build_alt3_efg_value_display_bundle import build_bundle
from build_alt3_efg_value_weighted_factors import build_weighted_factors


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_ROOT.parent
RESULTS_DIR = PIPELINE_ROOT / "results"
MASTER_RESULTS_DIR = PIPELINE_ROOT / "master_results"
RAPMS_DIR = PROJECT_ROOT.parent / "rapms"
RAPMS_MASTER_RESULTS_DIR = RAPMS_DIR / "master_results"
LOG_DIR = PIPELINE_ROOT / "logs" / "alt3_efg_value_rolling"

BASE_SUFFIX_PARTS = ["all", "rb", "se"]
OFF_ALPHA = 2000
DEF_ALPHA = 4000

COMPONENT_PREFIXES = [
    "RAPM",
    "ALT_EFG_RIM_FREQ",
    "ALT_EFG_RIM_FG",
    "ALT_EFG_MID_FREQ",
    "ALT_EFG_MID_FG",
    "ALT_EFG_THREE_FREQ",
    "ALT_EFG_THREE_FG",
    "ALT_FT",
    "ALT_FT_FREQ",
    "ALT_FT_SEVERITY",
    "ALT_TOV_VALUE",
    "SECOND_CHANCE_CLEAN",
    "ALT_BADPASS_TOV_VALUE",
    "ALT_SCORING_TOV_VALUE",
    "ASSIST_POINTS",
    "RIM_ASSIST",
]


@dataclass(frozen=True)
class Window:
    start_year: int
    end_year: int

    @property
    def start_arg(self) -> int:
        return self.start_year % 100

    @property
    def end_arg(self) -> int:
        return self.end_year % 100

    @property
    def year_range(self) -> str:
        start = self.start_year % 100
        end = self.end_year % 100
        return f"{end:02d}" if start == end else f"{start:02d}_{end:02d}"


def rolling_windows(include_full: bool) -> list[Window]:
    windows = [
        Window(start, start + length - 1)
        for length in range(1, 6)
        for start in range(1997, 2027 - length + 1)
    ]
    if include_full:
        windows.append(Window(1997, 2026))
    return windows


def suffix(age_dummies: bool = False) -> str:
    parts = [*BASE_SUFFIX_PARTS]
    if age_dummies:
        parts.append("agefe")
    parts.append(f"a{OFF_ALPHA}_{DEF_ALPHA}")
    return "_".join(parts)


def parse_year_range(value: str) -> tuple[int, int]:
    if "-" in value:
        start, end = value.split("-", 1)
    elif "_" in value:
        start, end = value.split("_", 1)
    else:
        start = end = value
    start_year = int(start)
    end_year = int(end)
    if start_year < 100:
        start_year = 2000 + start_year if start_year <= 50 else 1900 + start_year
    if end_year < 100:
        end_year = 2000 + end_year if end_year <= 50 else 1900 + end_year
    return start_year, end_year


def result_filename(prefix: str, year_range: str, output_suffix: str) -> str:
    if prefix == "RAPM":
        return f"rapm_{year_range}_{output_suffix.replace('all', 'all_pure', 1)}_results.csv"
    return f"{prefix.lower()}_{year_range}_{output_suffix}_results.csv"


def base_filename(year_range: str, output_suffix: str) -> str:
    return f"weighted_factors_alt3_{year_range}_{output_suffix}.csv"


def bundle_filename(year_range: str, output_suffix: str) -> str:
    return f"alt_efg_value_display_bundle_{year_range}_{output_suffix}_results.csv"


def weighted_filename(year_range: str, output_suffix: str) -> str:
    return f"weighted_factors_alt3_efg_value_{year_range}_{output_suffix}.csv"


def find_result_file(prefix: str, year_range: str, output_suffix: str) -> Path | None:
    name = result_filename(prefix, year_range, output_suffix)
    direct = RESULTS_DIR / name
    if direct.exists():
        return direct
    matches = sorted(RESULTS_DIR.glob(f"**/{name}"))
    return matches[0] if matches else None


def command_for_component(
    prefix: str,
    window: Window,
    cores_per_rapm: int | None,
    age_dummies: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "rapm.py"),
        prefix,
        str(window.start_arg),
        str(window.end_arg),
        "ALL",
        "--rubberband",
        "--season-effects",
        "--off-alpha",
        str(OFF_ALPHA),
        "--def-alpha",
        str(DEF_ALPHA),
    ]
    if prefix == "RAPM":
        cmd.append("--pure")
    if age_dummies:
        cmd.append("--age-dummies")
    if cores_per_rapm is not None:
        cmd.extend(["--cores", str(cores_per_rapm)])
    return cmd


def run_component(
    prefix: str,
    window: Window,
    cores_per_rapm: int | None,
    age_dummies: bool,
) -> tuple[str, str, float]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cmd = command_for_component(prefix, window, cores_per_rapm, age_dummies)
    output_suffix = suffix(age_dummies)
    log_path = LOG_DIR / f"{prefix.lower()}_{window.year_range}_{output_suffix}.log"
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    log_path.write_text(
        "$ " + " ".join(cmd) + "\n\nSTDOUT:\n" + result.stdout + "\n\nSTDERR:\n" + result.stderr
    )
    if result.returncode != 0:
        tail = "\n".join((result.stderr or result.stdout).splitlines()[-40:])
        raise RuntimeError(
            f"{prefix} {window.year_range} failed after {elapsed:.1f}s; see {log_path}\n{tail}"
        )
    return prefix, window.year_range, elapsed


def run_base_if_missing(
    window: Window,
    rapm_workers: int,
    cores_per_rapm: int | None,
    age_dummies: bool,
    force_base: bool,
) -> None:
    output_suffix = suffix(age_dummies)
    base = MASTER_RESULTS_DIR / base_filename(window.year_range, output_suffix)
    if base.exists() and not force_base:
        return
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "03_run_rapm_analysis.py"),
        str(window.start_arg),
        str(window.end_arg),
        "ALL",
        "--rubberband",
        "--season-effects",
        "--off-alpha",
        str(OFF_ALPHA),
        "--def-alpha",
        str(DEF_ALPHA),
        "--rapm-workers",
        str(rapm_workers),
    ]
    if cores_per_rapm is not None:
        cmd.extend(["--cores-per-rapm", str(cores_per_rapm)])
    if age_dummies:
        cmd.append("--age-dummies")
    print(f"Building {'forced' if force_base else 'missing'} base: {base.name}", flush=True)
    subprocess.run(cmd, check=True)


def stage_results(window: Window, output_suffix: str) -> Path:
    stage = RESULTS_DIR / "alt3_efg_value_staging" / f"{window.year_range}_{output_suffix}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for prefix in COMPONENT_PREFIXES:
        source = find_result_file(prefix, window.year_range, output_suffix)
        if source is None:
            raise FileNotFoundError(f"Missing {result_filename(prefix, window.year_range, output_suffix)}")
        target = stage / result_filename(prefix, window.year_range, output_suffix)
        if target.exists():
            target.unlink()
        target.symlink_to(source.resolve())
    return stage


def build_window(window: Window, age_dummies: bool) -> list[Path]:
    output_suffix = suffix(age_dummies)
    stage = stage_results(window, output_suffix)
    bundle_path = RESULTS_DIR / bundle_filename(window.year_range, output_suffix)
    base_path = MASTER_RESULTS_DIR / base_filename(window.year_range, output_suffix)
    if not base_path.exists():
        raise FileNotFoundError(base_path)

    build_bundle(
        year_range=window.year_range,
        suffix=output_suffix,
        results_dir=stage,
        output=bundle_path,
        decimals=3,
    )
    output_path, master_path, parquet_path, master_parquet_path, out = build_weighted_factors(
        bundle_path=bundle_path,
        base_path=base_path,
        output_name=weighted_filename(window.year_range, output_suffix),
        decimals=3,
        copy_to_master=True,
        write_parquet=True,
    )
    assert master_path is not None
    assert master_parquet_path is not None
    max_resid = float(out[["oRESID", "dRESID", "RESID"]].abs().max().max())
    print(
        f"Built {master_path.name} (+ parquet): rows={len(out):,} max_resid={max_resid:.6f}",
        flush=True,
    )
    return [master_path, master_parquet_path]


def publish_to_rapms(paths: list[Path], commit_message: str) -> bool:
    if not RAPMS_DIR.exists():
        raise FileNotFoundError(RAPMS_DIR)
    preexisting_staged = subprocess.run(
        ["git", "-C", str(RAPMS_DIR), "diff", "--cached", "--name-only"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if preexisting_staged:
        raise RuntimeError("rapms has preexisting staged changes:\n" + preexisting_staged)

    RAPMS_MASTER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    for source in paths:
        dest = RAPMS_MASTER_RESULTS_DIR / source.name
        shutil.copy2(source, dest)
        copied.append(dest.relative_to(RAPMS_DIR))

    subprocess.run(["git", "-C", str(RAPMS_DIR), "add", *map(str, copied)], check=True)
    diff = subprocess.run(["git", "-C", str(RAPMS_DIR), "diff", "--cached", "--quiet"])
    if diff.returncode == 0:
        print("No rapms changes to commit.", flush=True)
        return False
    if diff.returncode != 1:
        raise RuntimeError("git diff --cached --quiet failed")
    subprocess.run(
        ["git", "-C", str(RAPMS_DIR), "commit", "-m", commit_message],
        check=True,
    )
    subprocess.run(["git", "-C", str(RAPMS_DIR), "push", "origin", "main"], check=True)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-full", action="store_true", help="Also build 1997-2026.")
    parser.add_argument("--full-only", action="store_true", help="Only build 1997-2026.")
    parser.add_argument(
        "--intersect-years",
        default="",
        help="Only build rolling windows that intersect this year range, e.g. 1997-2013.",
    )
    parser.add_argument("--publish-to-rapms", action="store_true")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent missing component solves.")
    parser.add_argument("--rapm-workers", type=int, default=4, help="Workers for missing base 03 runs.")
    parser.add_argument("--cores-per-rapm", type=int, default=2)
    parser.add_argument("--only-missing-components", action="store_true")
    parser.add_argument("--force-base", action="store_true", help="Rerun matching legacy alt3 base files even if present.")
    parser.add_argument("--age-dummies", action="store_true", help="Pass --age-dummies through to component and base solves.")
    parser.add_argument(
        "--force-components",
        default="",
        help=(
            "Comma-separated component prefixes to rerun even when result files already exist, "
            "for example ALT_BADPASS_TOV_VALUE,ALT_SCORING_TOV_VALUE."
        ),
    )
    parser.add_argument(
        "--publish-message",
        default="Publish NBA alt3 EFG value rolling factors",
        help="Commit message used when --publish-to-rapms creates a downstream commit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.full_only:
        windows = [Window(1997, 2026)]
    else:
        windows = rolling_windows(include_full=args.include_full)
    if args.intersect_years:
        intersect_start, intersect_end = parse_year_range(args.intersect_years)
        windows = [
            window
            for window in windows
            if not (window.end_year < intersect_start or window.start_year > intersect_end)
        ]
    force_components = {
        item.strip().upper()
        for item in args.force_components.split(",")
        if item.strip()
    }
    unknown_force = sorted(force_components - (set(COMPONENT_PREFIXES) | {"ALL"}))
    if unknown_force:
        raise ValueError(f"Unknown --force-components values: {unknown_force}")

    if "ALL" in force_components:
        force_components = set(COMPONENT_PREFIXES)

    output_suffix = suffix(args.age_dummies)
    print(
        f"Selected windows: {len(windows)}; suffix={output_suffix}; "
        f"force_base={args.force_base}; age_dummies={args.age_dummies}",
        flush=True,
    )

    for window in windows:
        run_base_if_missing(
            window,
            args.rapm_workers,
            args.cores_per_rapm,
            args.age_dummies,
            args.force_base,
        )

    missing: list[tuple[str, Window]] = []
    for window in windows:
        for prefix in COMPONENT_PREFIXES:
            if prefix in force_components or find_result_file(prefix, window.year_range, output_suffix) is None:
                missing.append((prefix, window))

    force_msg = f"; forced={sorted(force_components)}" if force_components else ""
    print(f"Windows: {len(windows)}; component solves queued: {len(missing)}{force_msg}", flush=True)
    if missing:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_component, prefix, window, args.cores_per_rapm, args.age_dummies): (prefix, window)
                for prefix, window in missing
            }
            completed = 0
            for future in as_completed(futures):
                prefix, year_range, elapsed = future.result()
                completed += 1
                print(
                    f"[{completed}/{len(missing)}] {prefix} {year_range} solved in {elapsed:.1f}s",
                    flush=True,
                )

    if args.only_missing_components:
        return 0

    built = [path for window in windows for path in build_window(window, args.age_dummies)]
    print(f"Built weighted-factor bundles: {len(built)}", flush=True)

    if args.publish_to_rapms:
        pushed = publish_to_rapms(built, args.publish_message)
        print(f"Publish complete; pushed={pushed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
