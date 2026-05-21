#!/usr/bin/env python3
"""Normalize mixed-regime xFG data onto a single expected-points scale.

The source parquet appears to switch semantics by season:
- early seasons behave like xFG ~= make probability
- modern seasons behave more like an eFG-style value field

This script infers the regime season by season, converts xFG into expected
points per shot, then season-centers that expectation so league-average
expected PPS matches league-average actual PPS.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("/Users/russellthomas/Docs/pbp_rapm/all_shots_with_xfg.parquet")
DEFAULT_OUTPUT = Path("/Users/russellthomas/Docs/pbp_rapm/all_shots_with_xfg_fixed.parquet")
DEFAULT_SUMMARY = Path("/Users/russellthomas/Docs/pbp_rapm/all_shots_with_xfg_fixed_summary.csv")


USECOLS = [
    "GAME_ID",
    "GAME_DATE",
    "SEASON",
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_NAME",
    "PERIOD",
    "MINUTES_REMAINING",
    "SECONDS_REMAINING",
    "ACTION_TYPE",
    "SHOT_TYPE",
    "SHOT_ZONE_BASIC",
    "SHOT_ZONE_AREA",
    "SHOT_DISTANCE",
    "LOC_X",
    "LOC_Y",
    "SHOT_MADE_FLAG",
    "xFG",
]


SHOT_VALUE_MAP = {
    "2PT Field Goal": 2.0,
    "3PT Field Goal": 3.0,
}


def infer_regime(three_pt_xfg_mean: float) -> str:
    # A league-average expected 3P make probability above ~0.40 is not plausible.
    # In those seasons the field behaves like an eFG-style value surface instead.
    return "efg_value" if three_pt_xfg_mean > 0.40 else "fg_probability"


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for season, group in df.groupby("SEASON", dropna=False):
        three_mask = group["SHOT_TYPE"].eq("3PT Field Goal")
        three_mean = float(group.loc[three_mask, "xFG"].mean())
        regime = infer_regime(three_mean)

        actual_pps = float(group["actual_points"].mean())
        if regime == "fg_probability":
            raw_expected_pps = group["xFG"] * group["shot_value"]
        else:
            raw_expected_pps = 2.0 * group["xFG"]

        raw_mean = float(raw_expected_pps.mean())
        offset = actual_pps - raw_mean

        rows.append(
            {
                "SEASON": season,
                "shots": int(len(group)),
                "three_pt_xfg_mean": three_mean,
                "regime": regime,
                "actual_pps": actual_pps,
                "raw_expected_pps": raw_mean,
                "season_offset_pps": offset,
                "corrected_expected_pps": raw_mean + offset,
            }
        )

    return pd.DataFrame(rows).sort_values("SEASON").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    df = pd.read_parquet(args.input, columns=USECOLS).copy()
    df["shot_value"] = df["SHOT_TYPE"].map(SHOT_VALUE_MAP).astype(float)
    df["actual_points"] = df["SHOT_MADE_FLAG"].astype(float) * df["shot_value"]

    summary = build_summary(df)
    df = df.merge(summary[["SEASON", "regime", "season_offset_pps"]], on="SEASON", how="left")

    prob_mask = df["regime"].eq("fg_probability")
    df["xfg_expected_pps_raw"] = np.where(prob_mask, df["xFG"] * df["shot_value"], 2.0 * df["xFG"])
    df["xfg_expected_pps_fixed"] = df["xfg_expected_pps_raw"] + df["season_offset_pps"]
    df["xfg_diff_pps_fixed"] = df["actual_points"] - df["xfg_expected_pps_fixed"]

    # TS-style percentage-point scale for downstream comparisons.
    df["xfg_expected_ts_pts_fixed"] = df["xfg_expected_pps_fixed"] * 50.0
    df["xfg_diff_ts_pts_fixed"] = df["xfg_diff_pps_fixed"] * 50.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    summary.to_csv(args.summary, index=False)

    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
