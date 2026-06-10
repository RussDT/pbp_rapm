#!/usr/bin/env python3
"""Regenerate PureRAPMPeaks.csv from the published PureRAPM.csv export."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = PROJECT_ROOT.parent / "csvs" / "PureRAPM.csv"
DEFAULT_TARGET = PROJECT_ROOT.parent / "csvs" / "PureRAPMPeaks.csv"
DEFAULT_SEASON_LIMITS = PROJECT_ROOT.parent / "REPLIT_NBA_RAPM" / "first_last_season.csv"
RAPM_LENGTHS = (2, 3, 4, 5)


def load_season_limits(path: Path) -> dict[int, int]:
    if not path.exists():
        print(f"Warning: {path} not found; peak rows will not be season-limited")
        return {}

    df = pd.read_csv(path)
    missing = {"nba_id", "last_season"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    limits = df[["nba_id", "last_season"]].copy()
    limits["nba_id"] = pd.to_numeric(limits["nba_id"], errors="coerce")
    limits["last_season"] = pd.to_numeric(limits["last_season"], errors="coerce")
    limits = limits.dropna(subset=["nba_id", "last_season"])
    return dict(zip(limits["nba_id"].astype(int), limits["last_season"].astype(int)))


def normalize_purerapm(df: pd.DataFrame) -> pd.DataFrame:
    required = ["nba_id", "player", "length_of_rapm", "ending_season", "off", "def", "rapm"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"PureRAPM source missing required columns: {sorted(missing)}")

    out = df[required].copy()
    for column in ["nba_id", "length_of_rapm", "ending_season", "off", "def", "rapm"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["nba_id", "length_of_rapm", "ending_season", "off", "def", "rapm"])
    out["nba_id"] = out["nba_id"].astype(int)
    out["length_of_rapm"] = out["length_of_rapm"].astype(int)
    out["ending_season"] = out["ending_season"].astype(int)
    out = out.rename(columns={"player": "player_name"})
    return out


def apply_season_limits(df: pd.DataFrame, season_limits: dict[int, int]) -> pd.DataFrame:
    if not season_limits:
        return df

    allowed_last = df["nba_id"].map(season_limits)
    valid = allowed_last.isna() | (df["ending_season"] <= allowed_last) | (allowed_last < 2001)
    return df[valid].copy()


def build_window_peaks(df: pd.DataFrame, length: int) -> pd.DataFrame:
    window = df[df["length_of_rapm"] == length].copy()
    if window.empty:
        raise ValueError(f"No PureRAPM rows found for {length}Y peak build")

    identity = window[["nba_id", "player_name"]].drop_duplicates(subset=["nba_id"], keep="last")

    total_peak = window.loc[window.groupby("nba_id")["rapm"].idxmax()].copy()
    total_peak = total_peak[["nba_id", "off", "def", "rapm"]].rename(
        columns={
            "rapm": f"peak_{length}y_tot_tot",
            "off": f"peak_{length}y_tot_off",
            "def": f"peak_{length}y_tot_def",
        }
    )
    total_peak[f"peak_{length}y_tot_rank"] = (
        total_peak[f"peak_{length}y_tot_tot"].rank(ascending=False, method="min").astype(int)
    )
    total_peak[f"peak_{length}y_tot_off_rank"] = (
        total_peak[f"peak_{length}y_tot_off"].rank(ascending=False, method="min").astype(int)
    )
    total_peak[f"peak_{length}y_tot_def_rank"] = (
        total_peak[f"peak_{length}y_tot_def"].rank(ascending=False, method="min").astype(int)
    )

    off_peak = window.loc[window.groupby("nba_id")["off"].idxmax(), ["nba_id", "off"]].rename(
        columns={"off": f"offpeak_{length}y"}
    )
    off_peak[f"offpeak_{length}y_rank"] = (
        off_peak[f"offpeak_{length}y"].rank(ascending=False, method="min").astype(int)
    )

    def_peak = window.loc[window.groupby("nba_id")["def"].idxmax(), ["nba_id", "def"]].rename(
        columns={"def": f"defpeak_{length}y"}
    )
    def_peak[f"defpeak_{length}y_rank"] = (
        def_peak[f"defpeak_{length}y"].rank(ascending=False, method="min").astype(int)
    )

    return (
        identity.merge(total_peak, on="nba_id", how="inner")
        .merge(off_peak, on="nba_id", how="inner")
        .merge(def_peak, on="nba_id", how="inner")
    )


def build_peaks(source: Path, season_limits_path: Path) -> pd.DataFrame:
    pure = normalize_purerapm(pd.read_csv(source))
    pure = apply_season_limits(pure, load_season_limits(season_limits_path))

    result = build_window_peaks(pure, RAPM_LENGTHS[0])
    for length in RAPM_LENGTHS[1:]:
        result = result.merge(
            build_window_peaks(pure, length),
            on=["nba_id", "player_name"],
            how="outer",
        )

    result = result.sort_values(
        ["peak_5y_tot_tot", "peak_4y_tot_tot", "peak_3y_tot_tot", "peak_2y_tot_tot"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    value_cols = [col for col in result.columns if col not in {"nba_id", "player_name"}]
    result[value_cols] = result[value_cols].round(1)
    rank_cols = [col for col in value_cols if col.endswith("_rank")]
    for column in rank_cols:
        result[column] = result[column].astype("Int64")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--season-limits", type=Path, default=DEFAULT_SEASON_LIMITS)
    args = parser.parse_args()

    peaks = build_peaks(args.source, args.season_limits)
    args.target.parent.mkdir(parents=True, exist_ok=True)
    peaks.to_csv(args.target, index=False)
    print(f"Wrote {len(peaks)} rows and {len(peaks.columns)} columns to {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
