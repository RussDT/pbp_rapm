#!/usr/bin/env python3
"""Patch published alt3 weighted-factor files with point-valued TOV splits.

The legacy alt3 files originally wrote `oALT_TOV`, `oALT_TOV_bp`, and
`oALT_TOV_sc` from first-chance turnover-rate aliases (`-1` per turnover).
This helper uses the point-valued aliases:

- ALT_BADPASS_TOV_VALUE
- ALT_SCORING_TOV_VALUE

Those aliases value turnover rows against the same-sample first-chance scoring
baseline. The patched public columns remain centered by possession-weighted
player mean, and `oALT_TOV` / `dALT_TOV` are rebuilt as the exact sum of their
bad-pass and scoring-TOV children.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PIPELINE_ROOT.parent
RESULTS_DIR = PIPELINE_ROOT / "results"
MASTER_RESULTS_DIR = PIPELINE_ROOT / "master_results"
RAPMS_MASTER_RESULTS_DIR = PROJECT_ROOT.parent / "rapms" / "master_results"
RAPM_SCRIPT = PIPELINE_ROOT / "scripts" / "rapm.py"

ALT3_PATTERN = re.compile(
    r"^weighted_factors_alt3_(\d{2})(?:_(\d{2}))?_all_rb_se_a2000_4000\.csv$"
)


def normalize_year(two_digit: int) -> int:
    return 2000 + two_digit if two_digit <= 26 else 1900 + two_digit


def window_from_file_name(file_name: str) -> tuple[int, int, str] | None:
    match = ALT3_PATTERN.match(file_name)
    if not match:
        return None
    start = int(match.group(1))
    end = int(match.group(2) or match.group(1))
    start_full = normalize_year(start)
    end_full = normalize_year(end)
    if end_full < start_full:
        end_full += 100
    if not 1 <= end_full - start_full + 1 <= 5:
        return None
    year_range = f"{start:02d}" if match.group(2) is None else f"{start:02d}_{end:02d}"
    return start, end, year_range


def discover_windows(master_dir: Path) -> list[tuple[int, int, str]]:
    windows: list[tuple[int, int, str]] = []
    for path in sorted(master_dir.glob("weighted_factors_alt3_*_all_rb_se_a2000_4000.csv")):
        parsed = window_from_file_name(path.name)
        if parsed is not None:
            windows.append(parsed)
    return windows


def result_path(prefix: str, year_range: str) -> Path:
    return RESULTS_DIR / f"{prefix.lower()}_{year_range}_all_rb_se_a2000_4000_results.csv"


def run_alias(prefix: str, start: int, end: int, cores: int, force: bool) -> None:
    path = result_path(prefix, f"{start:02d}" if start == end else f"{start:02d}_{end:02d}")
    if path.exists() and not force:
        return
    command = [
        "python",
        str(RAPM_SCRIPT),
        prefix,
        str(start),
        str(end),
        "ALL",
        "--rubberband",
        "--season-effects",
        "--off-alpha",
        "2000",
        "--def-alpha",
        "4000",
        "--cores",
        str(cores),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def run_alias_task(task: tuple[str, int, int, int, bool]) -> str:
    prefix, start, end, cores, force = task
    year_range = f"{start:02d}" if start == end else f"{start:02d}_{end:02d}"
    run_alias(prefix, start, end, cores, force)
    return f"{prefix} {year_range}"


def weighted_mean(df: pd.DataFrame, col: str, weight_col: str) -> float:
    values = pd.to_numeric(df[col], errors="coerce")
    weights = pd.to_numeric(df[weight_col], errors="coerce")
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        raise ValueError(f"Cannot center {col}: no positive weights in {weight_col}")
    return float(np.average(values[mask], weights=weights[mask]))


def center(df: pd.DataFrame, col: str, weight_col: str, decimals: int) -> None:
    df[col] = (
        pd.to_numeric(df[col], errors="coerce") - weighted_mean(df, col, weight_col)
    ).round(decimals)


def load_component(prefix: str, year_range: str) -> pd.DataFrame:
    path = result_path(prefix, year_range)
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    required = {"player_id", "off", "def"}
    df = pd.read_csv(path)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing required columns: {sorted(missing)}")
    return df[["player_id", "off", "def"]].copy()


def patch_one_file(path: Path, year_range: str, decimals: int) -> dict[str, float]:
    df = pd.read_csv(path)
    bp = load_component("ALT_BADPASS_TOV_VALUE", year_range).rename(
        columns={"off": "_bp_off", "def": "_bp_def"}
    )
    sc = load_component("ALT_SCORING_TOV_VALUE", year_range).rename(
        columns={"off": "_sc_off", "def": "_sc_def"}
    )
    merged = df.merge(bp, on="player_id", how="left").merge(sc, on="player_id", how="left")
    for col in ["_bp_off", "_bp_def", "_sc_off", "_sc_def"]:
        if merged[col].isna().any():
            missing = int(merged[col].isna().sum())
            raise ValueError(f"{path.name}: {missing} rows missing {col} component")

    merged["oALT_TOV_bp"] = merged["_bp_off"]
    merged["oALT_TOV_sc"] = merged["_sc_off"]
    merged["dALT_TOV_bp"] = -merged["_bp_def"]
    merged["dALT_TOV_sc"] = -merged["_sc_def"]

    center(merged, "oALT_TOV_bp", "off_poss", decimals)
    center(merged, "oALT_TOV_sc", "off_poss", decimals)
    center(merged, "dALT_TOV_bp", "def_poss", decimals)
    center(merged, "dALT_TOV_sc", "def_poss", decimals)

    merged["oALT_TOV"] = (merged["oALT_TOV_bp"] + merged["oALT_TOV_sc"]).round(decimals)
    merged["dALT_TOV"] = (merged["dALT_TOV_bp"] + merged["dALT_TOV_sc"]).round(decimals)

    for side, total_col, fc_col, sc_col in [
        ("o", "off", "oFC", "oSC"),
        ("d", "def", "dFC", "dSC"),
    ]:
        merged[f"{side}ALT_TS"] = (merged[f"{side}ALT_EFG"] + merged[f"{side}ALT_FT"]).round(decimals)
        merged[fc_col] = (merged[f"{side}ALT_TS"] + merged[f"{side}ALT_TOV"]).round(decimals)
        merged[sc_col] = (
            pd.to_numeric(merged[total_col], errors="coerce") - merged[fc_col]
        ).round(decimals)

    component_cols = ["oALT_TS", "oALT_TOV", "oSC", "dALT_TS", "dALT_TOV", "dSC"]
    if "RESID" in merged.columns and all(col in merged.columns for col in component_cols):
        merged["RESID"] = (
            pd.to_numeric(merged["net_rapm"], errors="coerce") - merged[component_cols].sum(axis=1)
        ).round(decimals)

    drop_cols = ["_bp_off", "_bp_def", "_sc_off", "_sc_def"]
    output = merged.drop(columns=drop_cols)
    output.to_csv(path, index=False, float_format=f"%.{decimals}f")

    return {
        "o_split_resid": float((output["oALT_TOV_bp"] + output["oALT_TOV_sc"] - output["oALT_TOV"]).abs().max()),
        "d_split_resid": float((output["dALT_TOV_bp"] + output["dALT_TOV_sc"] - output["dALT_TOV"]).abs().max()),
        "resid_abs_max": float(output["RESID"].abs().max()) if "RESID" in output.columns else float("nan"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-solves", action="store_true")
    parser.add_argument("--force-solves", action="store_true")
    parser.add_argument("--solve-workers", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--decimals", type=int, default=3)
    parser.add_argument("--patch-downstream", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    windows = discover_windows(MASTER_RESULTS_DIR)
    if len(windows) != 140:
        raise RuntimeError(f"Expected 140 rolling alt3 windows, found {len(windows)}")

    if args.run_solves:
        tasks = [
            (prefix, start, end, args.cores, args.force_solves)
            for start, end, _ in windows
            for prefix in ["ALT_BADPASS_TOV_VALUE", "ALT_SCORING_TOV_VALUE"]
        ]
        completed = 0
        with ProcessPoolExecutor(max_workers=args.solve_workers) as executor:
            futures = [executor.submit(run_alias_task, task) for task in tasks]
            for future in as_completed(futures):
                completed += 1
                label = future.result()
                print(f"[{completed:03d}/{len(tasks)}] solved/skipped {label}", flush=True)

    target_dirs = [MASTER_RESULTS_DIR]
    if args.patch_downstream:
        target_dirs.append(RAPMS_MASTER_RESULTS_DIR)

    patched = 0
    for target_dir in target_dirs:
        for _, _, year_range in windows:
            path = target_dir / f"weighted_factors_alt3_{year_range}_all_rb_se_a2000_4000.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            if args.dry_run:
                continue
            summary = patch_one_file(path, year_range, args.decimals)
            patched += 1
            print(f"PATCHED {path}: {summary}", flush=True)

    print(f"windows={len(windows)} patched_files={patched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
