#!/usr/bin/env python3
"""Validate historical RAPM EOP/offense flow without touching production files."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from process_rapm_blocks import common
from process_rapm_blocks import process_rapm


def parse_years(value: str) -> list[int]:
    years: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            years.extend(range(int(start), int(end) + 1))
        else:
            years.append(int(part))
    return sorted(dict.fromkeys(years))


def season_label(year: int) -> str:
    return f"{year - 1}-{year % 100:02d}"


def raw_path(raw_dir: Path, year: int) -> Path:
    return raw_dir / f"NBA{year % 100:02d}.parquet"


def processed_path(processed_dir: Path, year: int) -> Path:
    return processed_dir / f"RAPM{year % 100:02d}.parquet"


def prefinal_path(processed_dir: Path, year: int) -> Path:
    return processed_dir / f"RAPM{year % 100:02d}_prefinal_eop_debug.parquet"


def player_cols(prefix: str) -> list[str]:
    return [f"{prefix}{i}" for i in range(1, 6)]


def normalize_player_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in player_cols("O") + player_cols("D"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
    return df


def overlap_counts(df: pd.DataFrame) -> dict[str, int]:
    df = normalize_player_cols(df)
    o = df[player_cols("O")].to_numpy()
    d = df[player_cols("D")].to_numpy()
    nonzero_o = o > 0
    nonzero_d = d > 0
    same = (
        (o[:, :, None] == d[:, None, :])
        & nonzero_o[:, :, None]
        & nonzero_d[:, None, :]
    ).any(axis=(1, 2))
    incomplete = (nonzero_o.sum(axis=1) < 4) | (nonzero_d.sum(axis=1) < 4)
    return {
        "same_player_overlap_rows": int(same.sum()),
        "incomplete_rows": int(incomplete.sum()),
    }


def removed_reason_summary(prefinal: pd.DataFrame, year: int) -> pd.DataFrame:
    prefinal = normalize_player_cols(prefinal)
    known_o = (prefinal[player_cols("O")] > 0).sum(axis=1)
    known_d = (prefinal[player_cols("D")] > 0).sum(axis=1)
    removed = prefinal[(known_o < 4) | (known_d < 4)].copy()
    if removed.empty:
        return pd.DataFrame(columns=[
            "year",
            "reason",
            "event_type",
            "poss_offense",
            "known_o",
            "known_d",
            "rows",
            "points",
        ])

    removed["known_o"] = known_o.loc[removed.index]
    removed["known_d"] = known_d.loc[removed.index]
    removed["reason"] = np.select(
        [
            removed["event_type"].eq("EndOfPeriod"),
            removed["event_type"].eq("Turnover")
            & removed["neutral_description"].fillna("").astype(str).str.contains(
                "Turnover", case=False, regex=False
            ),
            removed["event_type"].eq("FreeThrow"),
            removed["PotentialFoul"].fillna(False).astype(bool),
        ],
        ["end_of_period", "neutral_turnover", "free_throw_terminal", "potential_foul_terminal"],
        default="unknown",
    )
    summary = (
        removed.groupby(["reason", "event_type", "poss_offense", "known_o", "known_d"], dropna=False)
        .agg(rows=("game_id", "size"), points=("Off_Diff", "sum"))
        .reset_index()
        .sort_values(["rows", "points"], ascending=[False, False])
    )
    summary.insert(0, "year", year)
    return summary


def run_one_year(
    year: int,
    raw_dir: str,
    processed_dir: str,
    write_prefinal: bool,
    use_player_stats: bool,
) -> dict:
    # Force validation mode to use local/default stats fallback. With no stats,
    # missing_ft_fallback="actual" makes FT rows actual-scoring comparisons.
    os.environ["SUPABASE_URL"] = ""
    os.environ["SUPABASE_SERVICE_ROLE_KEY"] = ""
    os.environ["SUPABASE_KEY"] = ""

    raw = raw_path(Path(raw_dir), year)
    out_dir = Path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        return {"year": year, "skipped": True, "reason": f"missing raw {raw}"}

    original_common_finalize = common._finalize_df
    original_process_finalize = process_rapm._finalize_df
    original_fetch_player_stats = process_rapm.fetch_player_stats_supabase

    def identity_finalize(df: pd.DataFrame, diff_cols: list[str], finalize_year: int) -> pd.DataFrame:
        keep = df.copy()
        keep["Season"] = finalize_year
        return keep

    common._finalize_df = identity_finalize
    process_rapm._finalize_df = identity_finalize
    if not use_player_stats:
        process_rapm.fetch_player_stats_supabase = lambda _year, _is_playoffs: None
    try:
        prefinal, _ = process_rapm.process_rapm_py(
            str(raw),
            year,
            season_label(year),
            o_luck=0,
            d_luck=0,
            missing_ft_fallback="actual",
        )
    finally:
        common._finalize_df = original_common_finalize
        process_rapm._finalize_df = original_process_finalize
        process_rapm.fetch_player_stats_supabase = original_fetch_player_stats

    if prefinal is None:
        return {"year": year, "skipped": True, "reason": "processing returned None"}

    final = original_common_finalize(prefinal, ["Net_Diff", "Off_Diff", "Def_Diff"], year)
    final.to_parquet(processed_path(out_dir, year), index=False)
    if write_prefinal:
        prefinal.to_parquet(prefinal_path(out_dir, year), index=False)

    prefinal_norm = normalize_player_cols(prefinal)
    known_o = (prefinal_norm[player_cols("O")] > 0).sum(axis=1)
    known_d = (prefinal_norm[player_cols("D")] > 0).sum(axis=1)
    removed = prefinal_norm[(known_o < 4) | (known_d < 4)]
    raw_rows = pd.read_parquet(raw, columns=["game_id"]).shape[0]
    final_overlap = overlap_counts(final)
    prefinal_overlap = overlap_counts(prefinal)

    return {
        "year": year,
        "raw_rows": int(raw_rows),
        "prefinal_rows": int(len(prefinal)),
        "final_rows": int(len(final)),
        "final_filter_removed_rows": int(len(prefinal) - len(final)),
        "incomplete_prefinal_rows": int(len(removed)),
        "final_filter_removed_points": float(removed["Off_Diff"].sum()) if not removed.empty else 0.0,
        "prefinal_points": float(prefinal["Off_Diff"].sum()),
        "final_points": float(final["Off_Diff"].sum()),
        "prefinal_same_player_overlap_rows": prefinal_overlap["same_player_overlap_rows"],
        "prefinal_incomplete_rows": prefinal_overlap["incomplete_rows"],
        "final_same_player_overlap_rows": final_overlap["same_player_overlap_rows"],
        "final_incomplete_rows": final_overlap["incomplete_rows"],
        "processed_path": str(processed_path(out_dir, year)),
        "prefinal_path": str(prefinal_path(out_dir, year)) if write_prefinal else "",
        "skipped": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", default="1997-2013")
    parser.add_argument("--raw-dir", type=Path, default=Path("nba_pipeline/raw_data"))
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--write-prefinal", action="store_true")
    parser.add_argument("--use-player-stats", action="store_true")
    args = parser.parse_args()

    years = parse_years(args.years)
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    if args.workers <= 1:
        for year in years:
            results.append(
                run_one_year(
                    year,
                    str(args.raw_dir),
                    str(args.processed_dir),
                    args.write_prefinal,
                    args.use_player_stats,
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    run_one_year,
                    year,
                    str(args.raw_dir),
                    str(args.processed_dir),
                    args.write_prefinal,
                    args.use_player_stats,
                ): year
                for year in years
            }
            for future in as_completed(futures):
                results.append(future.result())

    results = sorted(results, key=lambda item: item["year"])
    summary = pd.DataFrame(results)
    summary.to_csv(args.report_dir / "historical_eop_flow_summary.csv", index=False)

    removed_summaries = []
    if args.write_prefinal:
        for year in years:
            path = prefinal_path(args.processed_dir, year)
            if path.exists():
                removed_summaries.append(removed_reason_summary(pd.read_parquet(path), year))
    removed_summaries = [df for df in removed_summaries if not df.empty]
    if removed_summaries:
        pd.concat(removed_summaries, ignore_index=True).to_csv(
            args.report_dir / "historical_eop_flow_removed_summary.csv", index=False
        )

    meta = {
        "years": years,
        "raw_dir": str(args.raw_dir),
        "processed_dir": str(args.processed_dir),
        "workers": args.workers,
        "write_prefinal": args.write_prefinal,
        "use_player_stats": args.use_player_stats,
    }
    (args.report_dir / "historical_eop_flow_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
