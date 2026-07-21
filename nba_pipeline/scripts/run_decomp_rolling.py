#!/usr/bin/env python3
"""Solve and assemble rolling public DECOMP weighted-factor windows."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:
    from .build_decomp_weighted_factors import (
        build_decomp_weighted_factors,
        find_result,
    )
except ImportError:  # Direct script execution adds scripts/ to sys.path.
    from build_decomp_weighted_factors import (
        build_decomp_weighted_factors,
        find_result,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PIPELINE_ROOT / "results"
MASTER_RESULTS_DIR = PIPELINE_ROOT / "master_results"
LOG_DIR = PIPELINE_ROOT / "logs" / "decomp_rolling"
DEFAULT_TRANSFER_MANIFEST = PIPELINE_ROOT / "config" / "decomp_tov_sc_transfer_multipliers.json"

OFF_ALPHA = 2000
DEF_ALPHA = 4000
COMPONENT_PREFIXES = [
    "RAPM",
    "DECOMP_EFG",
    "DECOMP_RIM_FREQ",
    "DECOMP_RIM_FG",
    "DECOMP_MID_FG",
    "DECOMP_THREE_FREQ",
    "DECOMP_THREE_FG",
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
        start, end = self.start_arg, self.end_arg
        return f"{end:02d}" if start == end else f"{start:02d}_{end:02d}"


def parse_window(value: str) -> Window:
    parts = value.replace("_", "-").split("-", 1)
    start = int(parts[0])
    end = int(parts[-1])
    if start < 100:
        start = 2000 + start if start <= 50 else 1900 + start
    if end < 100:
        end = 2000 + end if end <= 50 else 1900 + end
    if start > end:
        raise ValueError(f"Invalid window: {value}")
    return Window(start, end)


def rolling_windows(include_full: bool) -> list[Window]:
    windows = [
        Window(start, start + length - 1)
        for length in range(1, 6)
        for start in range(1997, 2027 - length + 1)
    ]
    if include_full:
        windows.append(Window(1997, 2026))
    return windows


def result_suffix(age_dummies: bool) -> str:
    parts = ["all", "rb", "se"]
    if age_dummies:
        parts.append("agefe")
    parts.append(f"a{OFF_ALPHA}_{DEF_ALPHA}")
    return "_".join(parts)


def component_command(prefix: str, window: Window, cores: int | None, age_dummies: bool) -> list[str]:
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
    if cores is not None:
        cmd.extend(["--cores", str(cores)])
    return cmd


def solve_component(prefix: str, window: Window, cores: int | None, age_dummies: bool) -> tuple[str, str, float]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cmd = component_command(prefix, window, cores, age_dummies)
    log_path = LOG_DIR / f"{prefix.lower()}_{window.year_range}_{result_suffix(age_dummies)}.log"
    started = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - started
    log_path.write_text(
        "$ " + " ".join(cmd) + "\n\nSTDOUT:\n" + result.stdout + "\n\nSTDERR:\n" + result.stderr
    )
    if result.returncode:
        tail = "\n".join((result.stderr or result.stdout).splitlines()[-40:])
        raise RuntimeError(f"{prefix} {window.year_range} failed; see {log_path}\n{tail}")
    return prefix, window.year_range, elapsed


def ensure_base(window: Window, suffix: str, workers: int, cores: int | None, age_dummies: bool) -> Path:
    base = MASTER_RESULTS_DIR / f"weighted_factors_alt3_{window.year_range}_{suffix}.csv"
    if base.exists():
        return base
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
        str(workers),
    ]
    if cores is not None:
        cmd.extend(["--cores-per-rapm", str(cores)])
    if age_dummies:
        cmd.append("--age-dummies")
    subprocess.run(cmd, check=True)
    if not base.exists():
        raise FileNotFoundError(f"Base suite completed without producing {base}")
    return base


def load_transfer_manifest(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError("JSON transfer manifest must be a year-range to multiplier object")
        return {
            str(key).strip().replace("-", "_"): float(value)
            for key, value in payload.items()
        }
    df = pd.read_csv(path)
    required = {"year_range", "tov_sc_transfer_multiplier"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Transfer manifest is missing columns: {sorted(missing)}")
    values = {}
    for row in df.itertuples(index=False):
        key = str(row.year_range).strip().replace("-", "_")
        values[key] = float(row.tov_sc_transfer_multiplier)
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", default="", help="Comma-separated explicit windows, e.g. 2022-2026")
    parser.add_argument("--include-full", action="store_true")
    parser.add_argument("--full-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rapm-workers", type=int, default=4)
    parser.add_argument("--cores-per-rapm", type=int, default=2)
    parser.add_argument("--age-dummies", action="store_true")
    parser.add_argument("--force-components", default="")
    parser.add_argument("--only-missing-components", action="store_true")
    parser.add_argument(
        "--tov-sc-transfer-manifest",
        type=Path,
        default=DEFAULT_TRANSFER_MANIFEST,
        help="JSON mapping or CSV with sample-specific full_possession_v1 multipliers.",
    )
    parser.add_argument(
        "--allow-unallocated-tov",
        action="store_true",
        help="Build with multiplier 1.0 when a window is absent from the transfer manifest.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.windows:
        windows = [parse_window(value.strip()) for value in args.windows.split(",") if value.strip()]
    elif args.full_only:
        windows = [Window(1997, 2026)]
    else:
        windows = rolling_windows(args.include_full)
    windows = list(dict.fromkeys(windows))
    suffix = result_suffix(args.age_dummies)
    transfer = load_transfer_manifest(args.tov_sc_transfer_manifest)
    if not args.only_missing_components and not args.allow_unallocated_tov:
        missing_transfer = [w.year_range for w in windows if w.year_range not in transfer]
        if missing_transfer:
            raise ValueError(
                "Missing sample-specific turnover/second-chance multipliers for windows: "
                f"{missing_transfer[:10]}. Supply --tov-sc-transfer-manifest or explicitly "
                "use --allow-unallocated-tov."
            )

    bases = {
        window: ensure_base(
            window, suffix, args.rapm_workers, args.cores_per_rapm, args.age_dummies
        )
        for window in windows
    }
    forced = {value.strip().upper() for value in args.force_components.split(",") if value.strip()}
    unknown = forced - set(COMPONENT_PREFIXES) - {"ALL"}
    if unknown:
        raise ValueError(f"Unknown forced components: {sorted(unknown)}")
    if "ALL" in forced:
        forced = set(COMPONENT_PREFIXES)

    queued = []
    for window in windows:
        for prefix in COMPONENT_PREFIXES:
            try:
                find_result(prefix, window.year_range, suffix, RESULTS_DIR)
                exists = True
            except FileNotFoundError:
                exists = False
            if prefix in forced or not exists:
                queued.append((prefix, window))
    print(f"Windows: {len(windows)}; component solves queued: {len(queued)}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(solve_component, prefix, window, args.cores_per_rapm, args.age_dummies): (prefix, window)
            for prefix, window in queued
        }
        for completed, future in enumerate(as_completed(futures), 1):
            prefix, year_range, elapsed = future.result()
            print(f"[{completed}/{len(queued)}] {prefix} {year_range}: {elapsed:.1f}s", flush=True)
    if args.only_missing_components:
        return 0

    for window in windows:
        multiplier = transfer.get(window.year_range, 1.0)
        name = f"weighted_factors_decomp_{window.year_range}_{suffix}.csv"
        _, master, _, master_parquet, out = build_decomp_weighted_factors(
            year_range=window.year_range,
            suffix=suffix,
            base_path=bases[window],
            output_name=name,
            transfer_multiplier=multiplier,
        )
        print(
            f"Built {master} and {master_parquet}: rows={len(out):,}, "
            f"multiplier={multiplier:.12g}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
