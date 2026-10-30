#!/usr/bin/env python3
"""Build a PBP-compatible historical playtype-value mechanism.

This is not exact provider PLAYTYPE_PROXY_PTS history. It uses the frozen
seven-family PBP taxonomy plus the labeled transition backcast, then prices
each phase's categories by shrunk actual points per FGA relative to its phase
mean. The separate BACKCAST prefix keeps that semantic seam explicit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from analyze_td_overall_playtype_mix_ots import MIX_CATEGORIES, classify_mix_shots
except ImportError:  # pragma: no cover
    from nba_pipeline.scripts.analyze_td_overall_playtype_mix_ots import (
        MIX_CATEGORIES,
        classify_mix_shots,
    )


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED = PIPELINE_ROOT / "processed"
DEFAULT_RAW = PIPELINE_ROOT / "raw_data"
DEFAULT_TRANSITION = PIPELINE_ROOT / "results" / "sq_transition_backfill"
SHRINK_FGA = 100.0


def normalize_game_id(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype("Int64")
    return numeric.astype(str).str.replace("<NA>", "", regex=False).str.zfill(10)


def unique_key_frame(
    frame: pd.DataFrame, *, label: str, allow_conflicting_keep_last: bool = False
) -> pd.DataFrame:
    keys = ["game_id", "event_num"]
    frame = frame.copy()
    frame["game_id"] = normalize_game_id(frame["game_id"])
    frame["event_num"] = pd.to_numeric(frame["event_num"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["event_num"])
    duplicates = frame.duplicated(keys, keep=False)
    if duplicates.any():
        value_columns = [column for column in frame.columns if column not in keys]
        conflicts = frame.loc[duplicates].groupby(keys, dropna=False)[value_columns].nunique(dropna=False)
        if (conflicts > 1).any(axis=None) and not allow_conflicting_keep_last:
            raise ValueError(f"{label}: conflicting duplicate event keys")
        # Modern corrected feed rows can repeat an event key. The correction
        # is appended, so raw descriptions deliberately retain the last row.
        frame = frame.drop_duplicates(keys, keep="last")
    return frame


def price_categories(category: pd.Series, actual_points: pd.Series) -> pd.DataFrame:
    global_mean = float(actual_points.mean())
    stats = (
        pd.DataFrame({"mix_category": category, "actual_points": actual_points})
        .groupby("mix_category")["actual_points"]
        .agg(["mean", "count"])
        .reindex(MIX_CATEGORIES)
    )
    stats["mean"] = stats["mean"].fillna(global_mean)
    stats["count"] = stats["count"].fillna(0.0)
    stats["shrunk_points_per_fga"] = (
        stats["count"] * stats["mean"] + SHRINK_FGA * global_mean
    ) / (stats["count"] + SHRINK_FGA)
    stats["playtype_proxy_points"] = stats["shrunk_points_per_fga"] - global_mean
    stats["global_points_per_fga"] = global_mean
    return stats.reset_index()


def build_one(
    *, year: int, phase: str, processed_dir: Path, raw_dir: Path, transition_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    suffix = f"{year % 100:02d}" + ("_PS" if phase == "ps" else "")
    base = pd.read_parquet(processed_dir / f"RIM_FREQ{suffix}.parquet")
    base = unique_key_frame(base, label=f"RIM_FREQ {year} {phase}")
    components = {
        "THREE_FREQ": "Is_Three_Attempt",
        "RIM_FG_PCT": "Is_Rim_Make",
        "MIDRANGE_FG_PCT": "Is_Midrange_Make",
        "THREE_FG_PCT": "Is_Three_Make",
    }
    merged = base
    for prefix, value_column in components.items():
        part = pd.read_parquet(
            processed_dir / f"{prefix}{suffix}.parquet",
            columns=["game_id", "event_num", value_column],
        )
        part = unique_key_frame(part, label=f"{prefix} {year} {phase}")
        merged = merged.merge(part, on=["game_id", "event_num"], how="left", validate="one_to_one")
        if value_column.endswith("_Make"):
            # Zone conversion files contain only that zone's attempts; a
            # missing key therefore means zero makes in that zone.
            merged[value_column] = merged[value_column].fillna(0).astype(int)

    raw_path = raw_dir / f"NBA{year % 100:02d}{'_PS' if phase == 'ps' else ''}.parquet"
    descriptions = pd.read_parquet(
        raw_path, columns=["game_id", "event_num", "home_description", "visitor_description"]
    )
    descriptions = unique_key_frame(
        descriptions,
        label=f"raw descriptions {year} {phase}",
        allow_conflicting_keep_last=True,
    )
    merged = merged.merge(descriptions, on=["game_id", "event_num"], how="left", validate="one_to_one")
    for column in ("home_description", "visitor_description"):
        merged[column] = merged[column].fillna("").astype(str)

    transition = pd.read_parquet(
        transition_dir / f"transition_backfill_{year}_{phase}.parquet",
        columns=["game_id", "event_num", "transition_label_operating"],
    )
    transition = unique_key_frame(transition, label=f"transition {year} {phase}")
    merged = merged.merge(transition, on=["game_id", "event_num"], how="left", validate="one_to_one")
    required = ["Is_Three_Attempt", "transition_label_operating"]
    if merged[required].isna().any().any():
        missing = merged[required].isna().sum().to_dict()
        raise ValueError(f"{year} {phase}: incomplete PBP backcast inputs {missing}")

    is_three = merged["Is_Three_Attempt"].astype(bool)
    category = classify_mix_shots(
        merged,
        is_three,
        merged["transition_label_operating"].astype(bool),
    )
    actual_points = (
        2.0 * (merged["Is_Rim_Make"] + merged["Is_Midrange_Make"])
        + 3.0 * merged["Is_Three_Make"]
    )
    prices = price_categories(category, actual_points)
    price_lookup = prices.set_index("mix_category")["playtype_proxy_points"]
    merged["Playtype_Proxy_PTS_Backcast"] = category.map(price_lookup).astype(float)
    merged["Playtype_Proxy_Category_Backcast"] = category
    merged["target_provenance"] = "pbp_seven_family_plus_frozen_transition_backcast"
    output_columns = [
        "game_id", "O1", "O2", "O3", "O4", "O5", "D1", "D2", "D3", "D4", "D5",
        "score", "period", "time_quarter", "away_score", "home_score", "score_margin",
        "event_num", "game_date", "Season",
        "Playtype_Proxy_PTS_Backcast",
        "Playtype_Proxy_Category_Backcast",
        "target_provenance",
    ]
    prices.insert(0, "season", year)
    prices.insert(1, "phase", phase)
    return merged[output_columns].copy(), prices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=1997)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--transition-dir", type=Path, default=DEFAULT_TRANSITION)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, price_parts = [], []
    for year in range(args.start_year, args.end_year + 1):
        for phase in ("rs", "ps"):
            suffix = f"{year % 100:02d}" + ("_PS" if phase == "ps" else "")
            destination = args.processed_dir / f"PLAYTYPE_PROXY_PTS_BACKCAST{suffix}.parquet"
            if destination.exists() and not args.overwrite:
                records.append({"year": year, "phase": phase, "status": "existing"})
                continue
            output, prices = build_one(
                year=year,
                phase=phase,
                processed_dir=args.processed_dir,
                raw_dir=args.raw_dir,
                transition_dir=args.transition_dir,
            )
            output.to_parquet(destination, index=False)
            price_parts.append(prices)
            records.append({"year": year, "phase": phase, "status": "built", "rows": len(output)})
    if price_parts:
        pd.concat(price_parts, ignore_index=True).to_csv(
            args.processed_dir / "playtype_proxy_points_backcast_prices.csv", index=False
        )
    manifest = {
        "series": "pbp_playtype_proxy_points_backcast",
        "warning": "PBP seven-family compatible backcast; not exact provider PLAYTYPE_PROXY_PTS history.",
        "records": records,
    }
    (args.processed_dir / "playtype_proxy_points_backcast_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
