#!/usr/bin/env python3
"""
Build regular-season processed parquet average audit files.

Outputs:
  - results/processed_rs_parquet_averages_97_26.csv
  - results/processed_rs_parquet_averages_per100_97_26.csv
  - results/processed_rs_parquet_row_counts_97_26.csv
  - results/processed_rs_parquet_average_sources_97_26.csv
  - results/processed_rs_parquet_average_skipped_97_26.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROCESSED_DIR = PIPELINE_ROOT / "processed"
RESULTS_DIR = PIPELINE_ROOT / "results"

DEFAULT_START_YEAR = 1997
DEFAULT_END_YEAR = 2026

VALUE_COLUMNS = {
    "ASSIST_POINTS": "Assist_Points",
    "BLOCK_RECOVERY": "Block_Recovered_By_Defense",
    "CONTEST": "Net_Diff",
    "EV_RAPM": "Off_Diff",
    "FIRST_CHANCE": "Net_Diff",
    "FT_PREMIUM": "Net_Diff",
    "INITIAL_EV": "Initial_EV",
    "INITIAL_EV_BETA": "Off_Diff",
    "LA_RAPM": "Off_Diff",
    "MIDRANGE_FG_PCT": "Is_Midrange_Make",
    "MIDRANGE_FREQ": "Is_Midrange_Attempt",
    "PLAYTYPE_PROXY_PTS": "Playtype_Proxy_PTS",
    "PLAYTYPE_TS_MIX": "Playtype_Exp_PTS",
    "RAPM": "Off_Diff",
    "RAPM_state_": "Off_Diff",
    "REB": "Offensive_Rebound",
    "DUNK": "Is_Dunk",
    "DUNK_ASSIST": "Is_Dunk_Assist",
    "RIM_ASSIST": "Is_Rim_Assist",
    "RIM_FG_PCT": "Is_Rim_Make",
    "RIM_FREQ": "Is_Rim_Attempt",
    "SECOND_CHANCE": "Off_Diff",
    "SECOND_CHANCE_CLEAN": "Off_Diff",
    "SPECIAL_RAPM": "Off_Diff",
    "SQ_POSS": "Net_Diff",
    "THREE_FG_PCT": "Is_Three_Make",
    "THREE_FREQ": "Is_Three_Attempt",
    "TOV": "Is_Turnover",
    "TRANSITION_FREQ": "Is_Transition",
    "TRANSITION_RIM": "Is_Transition_Rim",
    "TS": "Net_Diff",
}

ALT_ALIAS_INSERT_AFTER = "FIRST_CHANCE"
ALT_ALIAS_ORDER = [
    "ALT_TS",
    "ALT_EFG",
    "ALT_FT",
    "ALT_TOV",
    "ALT_BADPASS_TOV",
    "ALT_SCORING_TOV",
    "ALT_SQ",
    "ALT_MAKE",
]
ALT_BASELINE_COLUMNS = {
    "ALT_TS": "Net_Diff",
    "ALT_EFG": "FC_EFG_Diff",
    "ALT_FT": "FC_FT_Diff",
    "ALT_SQ": "FC_SQ_Diff",
    "ALT_MAKE": "FC_MAKE_Diff",
}


def end_year_from_suffix(suffix: str) -> int:
    year = int(suffix)
    return 1900 + year if year >= 97 else 2000 + year


def metric_order(existing_per100_path: Path) -> list[str]:
    if existing_per100_path.exists():
        columns = pd.read_csv(existing_per100_path, nrows=0).columns.tolist()
        metrics = [col for col in columns if col != "year"]
    else:
        metrics = sorted(VALUE_COLUMNS)

    missing_aliases = [metric for metric in ALT_ALIAS_ORDER if metric not in metrics]
    if missing_aliases:
        if ALT_ALIAS_INSERT_AFTER in metrics:
            insert_at = metrics.index(ALT_ALIAS_INSERT_AFTER) + 1
            metrics[insert_at:insert_at] = missing_aliases
        else:
            metrics.extend(missing_aliases)
    return metrics


def parse_processed_file(path: Path) -> tuple[str, int] | None:
    if path.name.endswith("_PS.parquet"):
        return None
    match = re.fullmatch(r"(.+?)(\d{2})\.parquet", path.name)
    if not match:
        return None
    metric, suffix = match.groups()
    return metric, end_year_from_suffix(suffix)


def requested_first_chance_columns(metrics: list[str], schema_names: list[str]) -> list[str]:
    columns = {"Net_Diff"}
    requested_aliases = set(metrics).intersection(ALT_ALIAS_ORDER)
    if requested_aliases:
        columns.add("Is_Turnover")
        for alias in requested_aliases.intersection(ALT_BASELINE_COLUMNS):
            source_col = ALT_BASELINE_COLUMNS[alias]
            if source_col in schema_names:
                columns.add(source_col)
        if {"ALT_BADPASS_TOV", "ALT_SCORING_TOV"}.intersection(requested_aliases):
            if "Is_BadPass_TOV" in schema_names:
                columns.add("Is_BadPass_TOV")
    return [col for col in columns if col in schema_names]


def write_metric_value(
    *,
    averages: pd.DataFrame,
    averages_per100: pd.DataFrame,
    row_counts: pd.DataFrame,
    sources: list[dict],
    metric: str,
    year: int,
    path: Path,
    value_col: str,
    mean: float,
    rows_used: int,
) -> None:
    row_idx = averages.index[averages["year"].eq(year)]
    if row_idx.empty:
        return
    idx = row_idx[0]
    averages.at[idx, metric] = mean
    averages_per100.at[idx, metric] = mean * 100
    row_counts.at[idx, metric] = rows_used
    sources.append(
        {
            "metric": metric,
            "year": year,
            "file": path.name,
            "value_column": value_col,
            "rows_used": rows_used,
            "mean_per_possession": mean,
            "mean_per_100": mean * 100,
        }
    )


def append_first_chance_alias_values(
    *,
    df: pd.DataFrame,
    metrics: list[str],
    year: int,
    path: Path,
    averages: pd.DataFrame,
    averages_per100: pd.DataFrame,
    row_counts: pd.DataFrame,
    sources: list[dict],
    skipped: list[dict],
) -> None:
    requested_aliases = [metric for metric in ALT_ALIAS_ORDER if metric in metrics]
    if not requested_aliases:
        return
    if "Is_Turnover" not in df.columns:
        skipped.append({"file": path.name, "reason": "missing Is_Turnover for ALT aliases"})
        return

    is_turnover = pd.to_numeric(df["Is_Turnover"], errors="coerce").fillna(0).astype(int)
    non_turnover_mask = is_turnover.eq(0)
    if not non_turnover_mask.any():
        skipped.append({"file": path.name, "reason": "no non-turnover FIRST_CHANCE rows for ALT baselines"})
        return

    for alias, source_col in ALT_BASELINE_COLUMNS.items():
        if alias not in metrics:
            continue
        if source_col not in df.columns:
            continue
        values = pd.to_numeric(df.loc[non_turnover_mask, source_col], errors="coerce")
        mean = values.mean()
        rows_used = int(values.notna().sum())
        write_metric_value(
            averages=averages,
            averages_per100=averages_per100,
            row_counts=row_counts,
            sources=sources,
            metric=alias,
            year=year,
            path=path,
            value_col=f"{source_col}|Is_Turnover=0",
            mean=mean,
            rows_used=rows_used,
        )

    if "ALT_TOV" in metrics:
        values = pd.to_numeric(df["Is_Turnover"], errors="coerce")
        write_metric_value(
            averages=averages,
            averages_per100=averages_per100,
            row_counts=row_counts,
            sources=sources,
            metric="ALT_TOV",
            year=year,
            path=path,
            value_col="Is_Turnover",
            mean=values.mean(),
            rows_used=int(values.notna().sum()),
        )

    if "ALT_BADPASS_TOV" in metrics and "Is_BadPass_TOV" in df.columns:
        values = pd.to_numeric(df["Is_BadPass_TOV"], errors="coerce")
        write_metric_value(
            averages=averages,
            averages_per100=averages_per100,
            row_counts=row_counts,
            sources=sources,
            metric="ALT_BADPASS_TOV",
            year=year,
            path=path,
            value_col="Is_BadPass_TOV",
            mean=values.mean(),
            rows_used=int(values.notna().sum()),
        )

    if "ALT_SCORING_TOV" in metrics:
        badpass = (
            pd.to_numeric(df["Is_BadPass_TOV"], errors="coerce").fillna(0)
            if "Is_BadPass_TOV" in df.columns
            else 0
        )
        values = pd.to_numeric(df["Is_Turnover"], errors="coerce") - badpass
        write_metric_value(
            averages=averages,
            averages_per100=averages_per100,
            row_counts=row_counts,
            sources=sources,
            metric="ALT_SCORING_TOV",
            year=year,
            path=path,
            value_col="Is_Turnover-Is_BadPass_TOV",
            mean=values.mean(),
            rows_used=int(values.notna().sum()),
        )


def build_reports(processed_dir: Path, start_year: int, end_year: int, metrics: list[str]):
    years = list(range(start_year, end_year + 1))
    averages = pd.DataFrame({"year": years})
    averages_per100 = pd.DataFrame({"year": years})
    row_counts = pd.DataFrame({"year": years})
    for metric in metrics:
        averages[metric] = pd.NA
        averages_per100[metric] = pd.NA
        row_counts[metric] = pd.NA

    sources: list[dict] = []
    skipped: list[dict] = []

    for path in sorted(processed_dir.glob("*.parquet")):
        parsed = parse_processed_file(path)
        if parsed is None:
            skipped.append({"file": path.name, "reason": "filename did not match metricYY.parquet"})
            continue

        metric, year = parsed
        if metric not in VALUE_COLUMNS:
            skipped.append({"file": path.name, "reason": "no recognized primary value column"})
            continue
        if year < start_year or year > end_year:
            skipped.append({"file": path.name, "reason": "outside requested year range"})
            continue

        value_col = VALUE_COLUMNS[metric]
        try:
            parquet_schema = pq.ParquetFile(path).schema.names
        except Exception as exc:
            skipped.append({"file": path.name, "reason": f"read failed: {exc}"})
            continue
        if value_col not in parquet_schema:
            skipped.append({"file": path.name, "reason": f"missing value column {value_col}"})
            continue

        columns_to_read = [value_col]
        if metric == "FIRST_CHANCE":
            columns_to_read = requested_first_chance_columns(metrics, parquet_schema)

        df = pd.read_parquet(path, columns=columns_to_read, engine="pyarrow")
        values = pd.to_numeric(df[value_col], errors="coerce")
        mean = values.mean()
        rows_used = int(values.notna().sum())
        write_metric_value(
            averages=averages,
            averages_per100=averages_per100,
            row_counts=row_counts,
            sources=sources,
            metric=metric,
            year=year,
            path=path,
            value_col=value_col,
            mean=mean,
            rows_used=rows_used,
        )
        if metric == "FIRST_CHANCE":
            append_first_chance_alias_values(
                df=df,
                metrics=metrics,
                year=year,
                path=path,
                averages=averages,
                averages_per100=averages_per100,
                row_counts=row_counts,
                sources=sources,
                skipped=skipped,
            )

    return averages, averages_per100, row_counts, pd.DataFrame(sources), pd.DataFrame(skipped)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    per100_path = args.results_dir / "processed_rs_parquet_averages_per100_97_26.csv"
    metrics = metric_order(per100_path)

    averages, averages_per100, row_counts, sources, skipped = build_reports(
        args.processed_dir,
        args.start_year,
        args.end_year,
        metrics,
    )

    suffix = f"{args.start_year % 100:02d}_{args.end_year % 100:02d}"
    averages.to_csv(args.results_dir / f"processed_rs_parquet_averages_{suffix}.csv", index=False)
    averages_per100.to_csv(args.results_dir / f"processed_rs_parquet_averages_per100_{suffix}.csv", index=False)
    row_counts.to_csv(args.results_dir / f"processed_rs_parquet_row_counts_{suffix}.csv", index=False)
    sources.to_csv(args.results_dir / f"processed_rs_parquet_average_sources_{suffix}.csv", index=False)
    skipped.to_csv(args.results_dir / f"processed_rs_parquet_average_skipped_{suffix}.csv", index=False)

    print(f"Wrote processed RS parquet average reports for {args.start_year}-{args.end_year}")
    print(f"Recognized source files: {len(sources):,}")
    print(f"Skipped files: {len(skipped):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
