#!/usr/bin/env python3
"""Run net-continuation DECOMP windows.

The historical default remains every complete rolling five-season window.
Use ``--current-release`` for the forward 1Y-5Y release family,
``--all-one-to-five-season-windows`` for the full 1997-through-current grid,
or pass explicit 1Y-5Y windows with ``--windows``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from build_net_decomp_weighted_factors import (
    OTHER_COMPONENTS,
    SHOT_COMPONENTS,
    build_net_decomp_weighted_factors,
    result_filename,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROCESSED_DIR = PIPELINE_ROOT / "processed"
NET_PROCESSED_DIR = PIPELINE_ROOT / "processed_netcont_lane"
RESULTS_DIR = PIPELINE_ROOT / "results" / "net_decomp_rolling" / "components"
MASTER_RESULTS_DIR = PIPELINE_ROOT / "master_results" / "net_decomp_rolling"
LOG_DIR = PIPELINE_ROOT / "logs" / "net_decomp_rolling"
OFF_ALPHA = 2000
DEF_ALPHA = 4000
SUFFIX = "all_rb_se_a2000_4000"
COMPONENTS = ["RAPM", *SHOT_COMPONENTS.values(), *OTHER_COMPONENTS.values()]


@dataclass(frozen=True)
class Window:
    start_year: int
    end_year: int

    @property
    def compact(self) -> str:
        if self.start_year == self.end_year:
            return f"{self.end_year % 100:02d}"
        return f"{self.start_year % 100:02d}_{self.end_year % 100:02d}"

    @property
    def seasons(self) -> int:
        return self.end_year - self.start_year + 1


def parse_window(value: str) -> Window:
    normalized = value.strip().replace("_", "-")
    if "-" in normalized:
        start, end = normalized.split("-", 1)
    else:
        start = end = normalized
    start_year, end_year = int(start), int(end)
    if start_year < 100:
        start_year = 2000 + start_year if start_year <= 50 else 1900 + start_year
    if end_year < 100:
        end_year = 2000 + end_year if end_year <= 50 else 1900 + end_year
    seasons = end_year - start_year + 1
    if seasons < 1 or seasons > 5:
        raise ValueError(f"Net DECOMP window must contain 1-5 seasons: {value}")
    return Window(start_year, end_year)


def all_windows() -> list[Window]:
    return [Window(start, start + 4) for start in range(1997, 2023)]


def current_release_windows(end_year: int = 2026) -> list[Window]:
    return [Window(end_year - seasons + 1, end_year) for seasons in range(1, 6)]


def all_one_to_five_season_windows(
    end_year: int = 2026, first_year: int = 1997
) -> list[Window]:
    if end_year < first_year:
        raise ValueError(f"release end year {end_year} precedes first year {first_year}")
    return [
        Window(window_end - seasons + 1, window_end)
        for window_end in range(first_year, end_year + 1)
        for seasons in range(1, min(5, window_end - first_year + 1) + 1)
    ]


def solve(prefix: str, window: Window, cores: int) -> tuple[str, str, float]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(SCRIPT_DIR / "rapm.py"), prefix,
        str(window.start_year % 100), str(window.end_year % 100), "ALL",
        "--rubberband", "--season-effects", "--off-alpha", str(OFF_ALPHA),
        "--def-alpha", str(DEF_ALPHA), "--cores", str(cores),
    ]
    if prefix == "RAPM":
        cmd.append("--pure")
    env = os.environ.copy()
    env["NBA_PIPELINE_PROCESSED_DIR"] = str(
        PROCESSED_DIR if prefix == "RAPM" else NET_PROCESSED_DIR
    )
    env["NBA_PIPELINE_RESULTS_DIR"] = str(RESULTS_DIR)
    started = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.time() - started
    log = LOG_DIR / f"{prefix.lower()}_{window.compact}_{SUFFIX}.log"
    log.write_text("$ " + " ".join(cmd) + "\n\nSTDOUT:\n" + result.stdout + "\n\nSTDERR:\n" + result.stderr)
    if result.returncode:
        tail = "\n".join((result.stderr or result.stdout).splitlines()[-40:])
        raise RuntimeError(f"{prefix} {window.compact} failed; see {log}\n{tail}")
    return prefix, window.compact, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--windows", default="", help="Comma-separated 1Y-5Y windows")
    group.add_argument(
        "--current-release",
        action="store_true",
        help="Build the current 1Y-5Y family ending in --release-end-year.",
    )
    group.add_argument(
        "--all-one-to-five-season-windows",
        action="store_true",
        help="Build every rolling 1Y-5Y window from 1997 through --release-end-year.",
    )
    parser.add_argument("--release-end-year", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cores-per-rapm", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.windows:
        windows = [parse_window(value.strip()) for value in args.windows.split(",") if value.strip()]
    elif args.all_one_to_five_season_windows:
        windows = all_one_to_five_season_windows(args.release_end_year)
    elif args.current_release:
        windows = current_release_windows(args.release_end_year)
    else:
        windows = all_windows()

    queued: list[tuple[str, Window]] = []
    for window in windows:
        for prefix in COMPONENTS:
            path = RESULTS_DIR / result_filename(prefix, window.compact, SUFFIX)
            if args.force or not path.exists():
                queued.append((prefix, window))
    print(f"Net DECOMP windows={len(windows)}; component solves queued={len(queued)}", flush=True)
    if args.dry_run:
        for prefix, window in queued:
            print(f"PLAN {window.compact} {prefix}")
        return 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(solve, prefix, window, args.cores_per_rapm): (prefix, window)
            for prefix, window in queued
        }
        for completed, future in enumerate(as_completed(futures), 1):
            prefix, year_range, elapsed = future.result()
            print(f"[{completed}/{len(queued)}] {prefix} {year_range}: {elapsed:.1f}s", flush=True)

    for completed, window in enumerate(windows, 1):
        name = f"weighted_factors_decomp_net_{window.compact}_{SUFFIX}.csv"
        paths = build_net_decomp_weighted_factors(
            window.compact,
            SUFFIX,
            name,
            results_dir=RESULTS_DIR,
            master_results_dir=MASTER_RESULTS_DIR,
            latest_year=window.end_year,
        )
        out = paths[-1]
        max_row_gap = out[["oDECOMP_EFG_PARENT_GAP", "dDECOMP_EFG_PARENT_GAP"]].abs().max().max()
        print(f"[{completed}/{len(windows)}] built {paths[1]} rows={len(out):,} max_parent_gap={max_row_gap:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
