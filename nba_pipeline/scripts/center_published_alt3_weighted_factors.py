#!/usr/bin/env python3
"""
One-off helper to center published alt3 weighted-factor columns in-place.

Default target:
  /Users/russellthomas/Docs/rapms/master_results/weighted_factors_alt3_*_all_rb_se.csv

This is meant for the 4-year rolling ALL rubberband + season-effects alt3 files.
It shifts the existing public columns, not new display columns:
  - oALT_EFG/oALT_FT/oALT_TS/oALT_TOV/oSC/oFC using off_poss
  - dALT_EFG/dALT_FT/dALT_TS/dALT_TOV/dSC/dFC using def_poss

Raw total columns off/def/net_rapm stay unchanged. Second chance is rebuilt as
the residual so the published public decomposition remains additive.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RAPMS_MASTER_RESULTS = Path("/Users/russellthomas/Docs/rapms/master_results")
DEFAULT_PATTERN = "weighted_factors_alt3_*_all_rb_se.csv"


def weighted_mean(df: pd.DataFrame, col: str, weight_col: str) -> float | None:
    if col not in df.columns or weight_col not in df.columns:
        return None
    values = pd.to_numeric(df[col], errors="coerce")
    weights = pd.to_numeric(df[weight_col], errors="coerce")
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return None
    return float(np.average(values[mask], weights=weights[mask]))


def center_column(df: pd.DataFrame, col: str, weight_col: str) -> bool:
    mean = weighted_mean(df, col, weight_col)
    if mean is None:
        return False
    df[col] = (pd.to_numeric(df[col], errors="coerce") - mean).round(2)
    return True


def set_column(df: pd.DataFrame, col: str, value: pd.Series) -> None:
    df[col] = value.round(2)


def center_alt3_columns(df: pd.DataFrame) -> bool:
    if "off_poss" not in df.columns or "def_poss" not in df.columns:
        return False

    for side, weight_col, total_col, fc_col, sc_col in [
        ("o", "off_poss", "off", "oFC", "oSC"),
        ("d", "def_poss", "def", "dFC", "dSC"),
    ]:
        has_sq_make = all(f"{side}ALT_{part}" in df.columns for part in ["SQ", "MAKE"])
        if has_sq_make:
            center_column(df, f"{side}ALT_SQ", weight_col)
            center_column(df, f"{side}ALT_MAKE", weight_col)
            set_column(df, f"{side}ALT_EFG", df[f"{side}ALT_SQ"] + df[f"{side}ALT_MAKE"])
        else:
            center_column(df, f"{side}ALT_EFG", weight_col)

        center_column(df, f"{side}ALT_FT", weight_col)
        if f"{side}ALT_EFG" in df.columns and f"{side}ALT_FT" in df.columns:
            set_column(df, f"{side}ALT_TS", df[f"{side}ALT_EFG"] + df[f"{side}ALT_FT"])
        else:
            center_column(df, f"{side}ALT_TS", weight_col)

        center_column(df, f"{side}ALT_TOV", weight_col)
        center_column(df, f"{side}ALT_TOV_bp", weight_col)
        center_column(df, f"{side}ALT_TOV_sc", weight_col)

        if f"{side}ALT_TS" in df.columns and f"{side}ALT_TOV" in df.columns:
            set_column(df, fc_col, df[f"{side}ALT_TS"] + df[f"{side}ALT_TOV"])
        else:
            center_column(df, fc_col, weight_col)

        if total_col in df.columns and fc_col in df.columns:
            set_column(df, sc_col, pd.to_numeric(df[total_col], errors="coerce") - df[fc_col])
        else:
            center_column(df, sc_col, weight_col)

    component_cols = ["oALT_TS", "oALT_TOV", "oSC", "dALT_TS", "dALT_TOV", "dSC"]
    if "net_rapm" in df.columns and all(col in df.columns for col in component_cols):
        set_column(df, "RESID", pd.to_numeric(df["net_rapm"], errors="coerce") - df[component_cols].sum(axis=1))
    return True


def weighted_summary(df: pd.DataFrame) -> dict[str, float]:
    checks = {}
    for col in ["oALT_TS", "oALT_EFG", "oALT_FT", "oALT_TOV", "oSC"]:
        mean = weighted_mean(df, col, "off_poss")
        if mean is not None:
            checks[col] = round(mean, 6)
    for col in ["dALT_TS", "dALT_EFG", "dALT_FT", "dALT_TOV", "dSC"]:
        mean = weighted_mean(df, col, "def_poss")
        if mean is not None:
            checks[col] = round(mean, 6)
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Center published alt3 weighted-factor files in-place.")
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_RAPMS_MASTER_RESULTS)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(args.target_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched {args.target_dir / args.pattern}")

    updated = 0
    for path in paths:
        df = pd.read_csv(path)
        if not center_alt3_columns(df):
            print(f"SKIP {path.name}: missing alt3 possession columns")
            continue
        if not args.dry_run:
            df.to_csv(path, index=False)
        updated += 1
        print(f"{'DRY ' if args.dry_run else ''}CENTERED {path.name}: {weighted_summary(df)}")

    print(f"centered_files={updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
