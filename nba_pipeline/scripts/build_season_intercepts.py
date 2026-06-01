#!/usr/bin/env python3
"""
Build season-level parquet averages for use as hardcoded intercept inputs.

This scans the processed parquet surface, computes:
1. raw numeric column means per parquet
2. solver-facing intercept means for each metric family
3. a small audit table showing how each file's season was resolved

Outputs land in nba_pipeline/results/season_intercepts/ by default.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).parent
PIPELINE_ROOT = SCRIPT_DIR.parent
DEFAULT_PROCESSED_DIR = PIPELINE_ROOT / "processed"
DEFAULT_OUTPUT_DIR = PIPELINE_ROOT / "results" / "season_intercepts"

FILENAME_RE = re.compile(r"^(?P<prefix>.+?)(?P<year>\d{2})(?P<playoffs>_PS)?\.parquet$")
SKIP_PREFIXES = ("LINEUP_STATS", "RAPM_state_")
CONTROL_COLUMNS = {
    "game_id",
    "O1",
    "O2",
    "O3",
    "O4",
    "O5",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "score",
    "period",
    "time_quarter",
    "away_score",
    "home_score",
    "score_margin",
    "event_num",
    "game_date",
    "Season",
}
MIRROR_COLUMNS = [
    "Offensive_Rebound",
    "Assist_Points",
    "Is_Rim_Assist",
    "Is_Rim_Attempt",
    "Is_Rim_Make",
    "Is_Midrange_Attempt",
    "Is_Midrange_Make",
    "Is_Three_Attempt",
    "Is_Three_Make",
    "Playtype_Exp_PTS",
    "Is_Transition",
    "Is_Transition_Rim",
    "Initial_EV",
]


def normalize_season_end_year(year_value) -> int | None:
    if pd.isna(year_value):
        return None
    value = int(year_value)
    if value < 100:
        return 2000 + value
    return value


def parse_processed_filename(path: Path) -> tuple[str, int, str]:
    match = FILENAME_RE.match(path.name)
    if not match:
        raise ValueError(f"Could not parse processed parquet filename: {path.name}")
    prefix = match.group("prefix")
    season_end_year = normalize_season_end_year(int(match.group("year")))
    season_type = "PS" if match.group("playoffs") else "RS"
    return prefix, season_end_year, season_type


def should_skip_file(path: Path) -> bool:
    return any(path.name.startswith(prefix) for prefix in SKIP_PREFIXES)


def coerce_mean(series: pd.Series) -> tuple[float | None, int]:
    numeric = pd.to_numeric(series, errors="coerce")
    sample_count = int(numeric.notna().sum())
    if sample_count == 0:
        return None, 0
    return float(numeric.mean()), sample_count


def append_raw_rows(raw_rows: list[dict], df: pd.DataFrame, *, source_prefix: str, source_file: str,
                    season_end_year: int, season_type: str, row_count: int) -> list[str]:
    raw_metric_columns: list[str] = []
    for column in df.columns:
        if column == "Season":
            continue
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
        raw_metric_columns.append(column)
        mean_value, sample_count = coerce_mean(df[column])
        raw_rows.append({
            "artifact": "raw_column_mean",
            "metric_key": source_prefix,
            "stat_name": "mean",
            "source_column": column,
            "mean_value": mean_value,
            "sample_count": sample_count,
            "row_count": row_count,
            "season_end_year": season_end_year,
            "season_type": season_type,
            "source_prefix": source_prefix,
            "source_file": source_file,
        })
    return raw_metric_columns


def add_solver_stat(rows: list[dict], *, metric_key: str, stat_name: str, mean_value: float | None,
                    sample_count: int, row_count: int, season_end_year: int, season_type: str,
                    source_prefix: str, source_file: str, source_column: str) -> None:
    rows.append({
        "artifact": "solver_intercept",
        "metric_key": metric_key,
        "stat_name": stat_name,
        "source_column": source_column,
        "mean_value": mean_value,
        "sample_count": sample_count,
        "row_count": row_count,
        "season_end_year": season_end_year,
        "season_type": season_type,
        "source_prefix": source_prefix,
        "source_file": source_file,
    })


def append_signed_metric_rows(rows: list[dict], *, metric_key: str, series: pd.Series,
                              season_end_year: int, season_type: str, row_count: int,
                              source_prefix: str, source_file: str, source_column: str) -> None:
    base_mean, sample_count = coerce_mean(series)
    add_solver_stat(
        rows,
        metric_key=metric_key,
        stat_name="off_diff_mean",
        mean_value=base_mean,
        sample_count=sample_count,
        row_count=row_count,
        season_end_year=season_end_year,
        season_type=season_type,
        source_prefix=source_prefix,
        source_file=source_file,
        source_column=source_column,
    )
    add_solver_stat(
        rows,
        metric_key=metric_key,
        stat_name="def_diff_mean",
        mean_value=(-base_mean if base_mean is not None else None),
        sample_count=sample_count,
        row_count=row_count,
        season_end_year=season_end_year,
        season_type=season_type,
        source_prefix=source_prefix,
        source_file=source_file,
        source_column=source_column,
    )


def append_existing_diff_rows(rows: list[dict], df: pd.DataFrame, *, metric_key: str,
                              season_end_year: int, season_type: str, row_count: int,
                              source_prefix: str, source_file: str) -> None:
    off_mean, off_count = coerce_mean(df["Off_Diff"])
    def_mean, def_count = coerce_mean(df["Def_Diff"])
    add_solver_stat(
        rows,
        metric_key=metric_key,
        stat_name="off_diff_mean",
        mean_value=off_mean,
        sample_count=off_count,
        row_count=row_count,
        season_end_year=season_end_year,
        season_type=season_type,
        source_prefix=source_prefix,
        source_file=source_file,
        source_column="Off_Diff",
    )
    add_solver_stat(
        rows,
        metric_key=metric_key,
        stat_name="def_diff_mean",
        mean_value=def_mean,
        sample_count=def_count,
        row_count=row_count,
        season_end_year=season_end_year,
        season_type=season_type,
        source_prefix=source_prefix,
        source_file=source_file,
        source_column="Def_Diff",
    )
    if "Net_Diff" in df.columns:
        net_mean, net_count = coerce_mean(df["Net_Diff"])
        add_solver_stat(
            rows,
            metric_key=metric_key,
            stat_name="net_diff_mean",
            mean_value=net_mean,
            sample_count=net_count,
            row_count=row_count,
            season_end_year=season_end_year,
            season_type=season_type,
            source_prefix=source_prefix,
            source_file=source_file,
            source_column="Net_Diff",
        )


def append_first_chance_rows(rows: list[dict], df: pd.DataFrame, *, season_end_year: int,
                             season_type: str, row_count: int, source_prefix: str,
                             source_file: str) -> list[str]:
    emitted_metrics = ["FIRST_CHANCE", "ALT_TS", "ALT_TOV", "ALT_BADPASS_TOV", "ALT_SCORING_TOV"]
    append_signed_metric_rows(
        rows,
        metric_key="FIRST_CHANCE",
        series=df["Net_Diff"],
        season_end_year=season_end_year,
        season_type=season_type,
        row_count=row_count,
        source_prefix=source_prefix,
        source_file=source_file,
        source_column="Net_Diff",
    )
    net_mean, net_count = coerce_mean(df["Net_Diff"])
    add_solver_stat(
        rows,
        metric_key="FIRST_CHANCE",
        stat_name="net_diff_mean",
        mean_value=net_mean,
        sample_count=net_count,
        row_count=row_count,
        season_end_year=season_end_year,
        season_type=season_type,
        source_prefix=source_prefix,
        source_file=source_file,
        source_column="Net_Diff",
    )

    turnover_mask = pd.to_numeric(df["Is_Turnover"], errors="coerce").fillna(0).astype(int) == 1
    non_turnover_values = pd.to_numeric(df.loc[~turnover_mask, "Net_Diff"], errors="coerce")
    alt_baseline, baseline_count = coerce_mean(non_turnover_values)
    turnover_rate = float(turnover_mask.mean()) if len(turnover_mask) else None
    non_turnover_share = float((~turnover_mask).mean()) if len(turnover_mask) else None

    add_solver_stat(
        rows,
        metric_key="ALT_TS",
        stat_name="alt_non_turnover_baseline",
        mean_value=alt_baseline,
        sample_count=baseline_count,
        row_count=row_count,
        season_end_year=season_end_year,
        season_type=season_type,
        source_prefix=source_prefix,
        source_file=source_file,
        source_column="Net_Diff|Is_Turnover=0",
    )
    add_solver_stat(
        rows,
        metric_key="ALT_TS",
        stat_name="off_diff_mean",
        mean_value=alt_baseline,
        sample_count=baseline_count,
        row_count=row_count,
        season_end_year=season_end_year,
        season_type=season_type,
        source_prefix=source_prefix,
        source_file=source_file,
        source_column="Net_Diff|Is_Turnover=0",
    )
    add_solver_stat(
        rows,
        metric_key="ALT_TS",
        stat_name="def_diff_mean",
        mean_value=(-alt_baseline if alt_baseline is not None else None),
        sample_count=baseline_count,
        row_count=row_count,
        season_end_year=season_end_year,
        season_type=season_type,
        source_prefix=source_prefix,
        source_file=source_file,
        source_column="Net_Diff|Is_Turnover=0",
    )
    add_solver_stat(
        rows,
        metric_key="ALT_TS",
        stat_name="turnover_rate",
        mean_value=turnover_rate,
        sample_count=row_count,
        row_count=row_count,
        season_end_year=season_end_year,
        season_type=season_type,
        source_prefix=source_prefix,
        source_file=source_file,
        source_column="Is_Turnover",
    )

    append_signed_metric_rows(
        rows,
        metric_key="ALT_TOV",
        series=-pd.to_numeric(df["Is_Turnover"], errors="coerce"),
        season_end_year=season_end_year,
        season_type=season_type,
        row_count=row_count,
        source_prefix=source_prefix,
        source_file=source_file,
        source_column="Is_Turnover",
    )
    add_solver_stat(
        rows,
        metric_key="ALT_TOV",
        stat_name="turnover_rate",
        mean_value=turnover_rate,
        sample_count=row_count,
        row_count=row_count,
        season_end_year=season_end_year,
        season_type=season_type,
        source_prefix=source_prefix,
        source_file=source_file,
        source_column="Is_Turnover",
    )
    add_solver_stat(
        rows,
        metric_key="ALT_TOV",
        stat_name="non_turnover_share",
        mean_value=non_turnover_share,
        sample_count=row_count,
        row_count=row_count,
        season_end_year=season_end_year,
        season_type=season_type,
        source_prefix=source_prefix,
        source_file=source_file,
        source_column="Is_Turnover",
    )

    badpass_series = pd.to_numeric(df.get("Is_BadPass_TOV", 0), errors="coerce").fillna(0)
    badpass_rate, badpass_count = coerce_mean(badpass_series)
    append_signed_metric_rows(
        rows,
        metric_key="ALT_BADPASS_TOV",
        series=-badpass_series,
        season_end_year=season_end_year,
        season_type=season_type,
        row_count=row_count,
        source_prefix=source_prefix,
        source_file=source_file,
        source_column="Is_BadPass_TOV",
    )
    add_solver_stat(
        rows,
        metric_key="ALT_BADPASS_TOV",
        stat_name="turnover_rate",
        mean_value=badpass_rate,
        sample_count=badpass_count,
        row_count=row_count,
        season_end_year=season_end_year,
        season_type=season_type,
        source_prefix=source_prefix,
        source_file=source_file,
        source_column="Is_BadPass_TOV",
    )

    scoring_series = pd.to_numeric(df["Is_Turnover"], errors="coerce").fillna(0) - badpass_series
    scoring_rate, scoring_count = coerce_mean(scoring_series)
    append_signed_metric_rows(
        rows,
        metric_key="ALT_SCORING_TOV",
        series=-scoring_series,
        season_end_year=season_end_year,
        season_type=season_type,
        row_count=row_count,
        source_prefix=source_prefix,
        source_file=source_file,
        source_column="Is_Turnover-Is_BadPass_TOV",
    )
    add_solver_stat(
        rows,
        metric_key="ALT_SCORING_TOV",
        stat_name="turnover_rate",
        mean_value=scoring_rate,
        sample_count=scoring_count,
        row_count=row_count,
        season_end_year=season_end_year,
        season_type=season_type,
        source_prefix=source_prefix,
        source_file=source_file,
        source_column="Is_Turnover-Is_BadPass_TOV",
    )

    component_specs = [
        ("ALT_EFG", "FC_EFG_Diff", "alt_efg_non_turnover_baseline"),
        ("ALT_SQ", "FC_SQ_Diff", "alt_sq_non_turnover_baseline"),
        ("ALT_MAKE", "FC_MAKE_Diff", "alt_make_non_turnover_baseline"),
        ("ALT_FT", "FC_FT_Diff", "alt_ft_non_turnover_baseline"),
    ]
    for metric_key, source_col, baseline_stat in component_specs:
        if source_col not in df.columns:
            continue

        component_values = pd.to_numeric(df.loc[~turnover_mask, source_col], errors="coerce")
        component_mean, component_count = coerce_mean(component_values)
        emitted_metrics.append(metric_key)

        add_solver_stat(
            rows,
            metric_key=metric_key,
            stat_name=baseline_stat,
            mean_value=component_mean,
            sample_count=component_count,
            row_count=row_count,
            season_end_year=season_end_year,
            season_type=season_type,
            source_prefix=source_prefix,
            source_file=source_file,
            source_column=f"{source_col}|Is_Turnover=0",
        )
        add_solver_stat(
            rows,
            metric_key=metric_key,
            stat_name="off_diff_mean",
            mean_value=component_mean,
            sample_count=component_count,
            row_count=row_count,
            season_end_year=season_end_year,
            season_type=season_type,
            source_prefix=source_prefix,
            source_file=source_file,
            source_column=f"{source_col}|Is_Turnover=0",
        )
        add_solver_stat(
            rows,
            metric_key=metric_key,
            stat_name="def_diff_mean",
            mean_value=(-component_mean if component_mean is not None else None),
            sample_count=component_count,
            row_count=row_count,
            season_end_year=season_end_year,
            season_type=season_type,
            source_prefix=source_prefix,
            source_file=source_file,
            source_column=f"{source_col}|Is_Turnover=0",
        )
        add_solver_stat(
            rows,
            metric_key=metric_key,
            stat_name="turnover_rate",
            mean_value=turnover_rate,
            sample_count=row_count,
            row_count=row_count,
            season_end_year=season_end_year,
            season_type=season_type,
            source_prefix=source_prefix,
            source_file=source_file,
            source_column="Is_Turnover",
        )

    if "Is_FC_Transition_Possession" in df.columns:
        y = pd.to_numeric(df["Net_Diff"], errors="coerce")
        is_transition = (
            pd.to_numeric(df["Is_FC_Transition_Possession"], errors="coerce")
            .fillna(0)
            .astype(int)
            .eq(1)
        )
        transition_values = y[is_transition]
        halfcourt_values = y[~is_transition]
        transition_baseline, transition_count = coerce_mean(transition_values)
        halfcourt_baseline, halfcourt_count = coerce_mean(halfcourt_values)
        transition_share = float(is_transition.mean()) if len(is_transition) else None

        mode_metrics = [
            "FC_TRANSITION_SCORING",
            "FC_HALFCOURT_SCORING",
            "FC_MODE_MIX",
            "FC_TRANSITION_VALUE",
            "FC_HALFCOURT_VALUE",
        ]
        for metric_key in mode_metrics:
            emitted_metrics.append(metric_key)
            add_solver_stat(
                rows,
                metric_key=metric_key,
                stat_name="transition_ppp_baseline",
                mean_value=transition_baseline,
                sample_count=transition_count,
                row_count=row_count,
                season_end_year=season_end_year,
                season_type=season_type,
                source_prefix=source_prefix,
                source_file=source_file,
                source_column="Net_Diff|Is_FC_Transition_Possession=1",
            )
            add_solver_stat(
                rows,
                metric_key=metric_key,
                stat_name="halfcourt_ppp_baseline",
                mean_value=halfcourt_baseline,
                sample_count=halfcourt_count,
                row_count=row_count,
                season_end_year=season_end_year,
                season_type=season_type,
                source_prefix=source_prefix,
                source_file=source_file,
                source_column="Net_Diff|Is_FC_Transition_Possession=0",
            )
            add_solver_stat(
                rows,
                metric_key=metric_key,
                stat_name="transition_share",
                mean_value=transition_share,
                sample_count=row_count,
                row_count=row_count,
                season_end_year=season_end_year,
                season_type=season_type,
                source_prefix=source_prefix,
                source_file=source_file,
                source_column="Is_FC_Transition_Possession",
            )

        if transition_baseline is not None and halfcourt_baseline is not None:
            append_signed_metric_rows(
                rows,
                metric_key="FC_TRANSITION_SCORING",
                series=y.where(is_transition, transition_baseline),
                season_end_year=season_end_year,
                season_type=season_type,
                row_count=row_count,
                source_prefix=source_prefix,
                source_file=source_file,
                source_column="Net_Diff|transition rows; transition_ppp placeholder otherwise",
            )
            append_signed_metric_rows(
                rows,
                metric_key="FC_HALFCOURT_SCORING",
                series=y.where(~is_transition, halfcourt_baseline),
                season_end_year=season_end_year,
                season_type=season_type,
                row_count=row_count,
                source_prefix=source_prefix,
                source_file=source_file,
                source_column="Net_Diff|halfcourt rows; halfcourt_ppp placeholder otherwise",
            )
            append_signed_metric_rows(
                rows,
                metric_key="FC_MODE_MIX",
                series=pd.Series(halfcourt_baseline, index=df.index).where(~is_transition, transition_baseline),
                season_end_year=season_end_year,
                season_type=season_type,
                row_count=row_count,
                source_prefix=source_prefix,
                source_file=source_file,
                source_column="mode_ppp_baseline",
            )
            append_signed_metric_rows(
                rows,
                metric_key="FC_TRANSITION_VALUE",
                series=(y - transition_baseline).where(is_transition, 0.0),
                season_end_year=season_end_year,
                season_type=season_type,
                row_count=row_count,
                source_prefix=source_prefix,
                source_file=source_file,
                source_column="Net_Diff-transition_ppp|transition rows",
            )
            append_signed_metric_rows(
                rows,
                metric_key="FC_HALFCOURT_VALUE",
                series=(y - halfcourt_baseline).where(~is_transition, 0.0),
                season_end_year=season_end_year,
                season_type=season_type,
                row_count=row_count,
                source_prefix=source_prefix,
                source_file=source_file,
                source_column="Net_Diff-halfcourt_ppp|halfcourt rows",
            )
    return emitted_metrics


def append_solver_rows(solver_rows: list[dict], df: pd.DataFrame, *, source_prefix: str,
                       source_file: str, season_end_year: int, season_type: str,
                       row_count: int) -> list[str]:
    columns = set(df.columns)
    emitted_metrics: list[str] = []

    if source_prefix == "FIRST_CHANCE" and {"Net_Diff", "Is_Turnover"}.issubset(columns):
        return append_first_chance_rows(
            solver_rows,
            df,
            season_end_year=season_end_year,
            season_type=season_type,
            row_count=row_count,
            source_prefix=source_prefix,
            source_file=source_file,
        )

    if {"Off_Diff", "Def_Diff"}.issubset(columns):
        append_existing_diff_rows(
            solver_rows,
            df,
            metric_key=source_prefix,
            season_end_year=season_end_year,
            season_type=season_type,
            row_count=row_count,
            source_prefix=source_prefix,
            source_file=source_file,
        )
        emitted_metrics.append(source_prefix)
        if source_prefix == "RAPM" and "Net_Diff" in columns:
            append_signed_metric_rows(
                solver_rows,
                metric_key="RAPM_PURE",
                series=df["Net_Diff"],
                season_end_year=season_end_year,
                season_type=season_type,
                row_count=row_count,
                source_prefix=source_prefix,
                source_file=source_file,
                source_column="Net_Diff",
            )
            net_mean, net_count = coerce_mean(df["Net_Diff"])
            add_solver_stat(
                solver_rows,
                metric_key="RAPM_PURE",
                stat_name="net_diff_mean",
                mean_value=net_mean,
                sample_count=net_count,
                row_count=row_count,
                season_end_year=season_end_year,
                season_type=season_type,
                source_prefix=source_prefix,
                source_file=source_file,
                source_column="Net_Diff",
            )
            emitted_metrics.append("RAPM_PURE")
        return emitted_metrics

    if source_prefix == "TOV" and "Is_Turnover" in columns:
        turnover_rate, turnover_count = coerce_mean(df["Is_Turnover"])
        add_solver_stat(
            solver_rows,
            metric_key="TOV",
            stat_name="turnover_rate",
            mean_value=turnover_rate,
            sample_count=turnover_count,
            row_count=row_count,
            season_end_year=season_end_year,
            season_type=season_type,
            source_prefix=source_prefix,
            source_file=source_file,
            source_column="Is_Turnover",
        )
        append_signed_metric_rows(
            solver_rows,
            metric_key="TOV",
            series=-pd.to_numeric(df["Is_Turnover"], errors="coerce"),
            season_end_year=season_end_year,
            season_type=season_type,
            row_count=row_count,
            source_prefix=source_prefix,
            source_file=source_file,
            source_column="Is_Turnover",
        )
        emitted_metrics.append("TOV")

        if "Is_BadPass_TOV" in columns:
            badpass_rate, badpass_count = coerce_mean(df["Is_BadPass_TOV"])
            add_solver_stat(
                solver_rows,
                metric_key="BADPASS_TOV",
                stat_name="turnover_rate",
                mean_value=badpass_rate,
                sample_count=badpass_count,
                row_count=row_count,
                season_end_year=season_end_year,
                season_type=season_type,
                source_prefix=source_prefix,
                source_file=source_file,
                source_column="Is_BadPass_TOV",
            )
            append_signed_metric_rows(
                solver_rows,
                metric_key="BADPASS_TOV",
                series=-pd.to_numeric(df["Is_BadPass_TOV"], errors="coerce"),
                season_end_year=season_end_year,
                season_type=season_type,
                row_count=row_count,
                source_prefix=source_prefix,
                source_file=source_file,
                source_column="Is_BadPass_TOV",
            )
            emitted_metrics.append("BADPASS_TOV")

            scoring_series = pd.to_numeric(df["Is_Turnover"], errors="coerce") - pd.to_numeric(df["Is_BadPass_TOV"], errors="coerce")
            scoring_rate, scoring_count = coerce_mean(scoring_series)
            add_solver_stat(
                solver_rows,
                metric_key="SCORING_TOV",
                stat_name="turnover_rate",
                mean_value=scoring_rate,
                sample_count=scoring_count,
                row_count=row_count,
                season_end_year=season_end_year,
                season_type=season_type,
                source_prefix=source_prefix,
                source_file=source_file,
                source_column="Is_Turnover-Is_BadPass_TOV",
            )
            append_signed_metric_rows(
                solver_rows,
                metric_key="SCORING_TOV",
                series=-scoring_series,
                season_end_year=season_end_year,
                season_type=season_type,
                row_count=row_count,
                source_prefix=source_prefix,
                source_file=source_file,
                source_column="Is_Turnover-Is_BadPass_TOV",
            )
            emitted_metrics.append("SCORING_TOV")
        return emitted_metrics

    if source_prefix == "BLOCK_RECOVERY" and "Block_Recovered_By_Defense" in columns:
        target = pd.to_numeric(df["Block_Recovered_By_Defense"], errors="coerce")
        recovery_rate, recovery_count = coerce_mean(target)
        add_solver_stat(
            solver_rows,
            metric_key="BLOCK_RECOVERY",
            stat_name="defensive_recovery_rate",
            mean_value=recovery_rate,
            sample_count=recovery_count,
            row_count=row_count,
            season_end_year=season_end_year,
            season_type=season_type,
            source_prefix=source_prefix,
            source_file=source_file,
            source_column="Block_Recovered_By_Defense",
        )
        append_signed_metric_rows(
            solver_rows,
            metric_key="BLOCK_RECOVERY",
            series=-target,
            season_end_year=season_end_year,
            season_type=season_type,
            row_count=row_count,
            source_prefix=source_prefix,
            source_file=source_file,
            source_column="Block_Recovered_By_Defense",
        )
        emitted_metrics.append("BLOCK_RECOVERY")
        return emitted_metrics

    for column in MIRROR_COLUMNS:
        if column in columns:
            append_signed_metric_rows(
                solver_rows,
                metric_key=source_prefix,
                series=df[column],
                season_end_year=season_end_year,
                season_type=season_type,
                row_count=row_count,
                source_prefix=source_prefix,
                source_file=source_file,
                source_column=column,
            )
            emitted_metrics.append(source_prefix)
            return emitted_metrics

    if "Net_Diff" in columns:
        append_signed_metric_rows(
            solver_rows,
            metric_key=source_prefix,
            series=df["Net_Diff"],
            season_end_year=season_end_year,
            season_type=season_type,
            row_count=row_count,
            source_prefix=source_prefix,
            source_file=source_file,
            source_column="Net_Diff",
        )
        net_mean, net_count = coerce_mean(df["Net_Diff"])
        add_solver_stat(
            solver_rows,
            metric_key=source_prefix,
            stat_name="net_diff_mean",
            mean_value=net_mean,
            sample_count=net_count,
            row_count=row_count,
            season_end_year=season_end_year,
            season_type=season_type,
            source_prefix=source_prefix,
            source_file=source_file,
            source_column="Net_Diff",
        )
        emitted_metrics.append(source_prefix)

    return emitted_metrics


def build_lookup_json(solver_df: pd.DataFrame) -> dict:
    lookup: dict = {}
    for row in solver_df.itertuples(index=False):
        season_bucket = (
            lookup
            .setdefault(row.season_type, {})
            .setdefault(str(row.season_end_year), {})
            .setdefault(row.metric_key, {
                "_meta": {
                    "source_prefix": row.source_prefix,
                    "source_file": row.source_file,
                    "row_count": int(row.row_count),
                }
            })
        )
        season_bucket[row.stat_name] = row.mean_value
    return lookup


def resolve_parquet_season(df: pd.DataFrame) -> tuple[int | None, str]:
    if "Season" not in df.columns:
        return None, "filename_only"
    season_values = (
        pd.to_numeric(df["Season"], errors="coerce")
        .dropna()
        .astype(int)
        .map(normalize_season_end_year)
        .dropna()
        .unique()
        .tolist()
    )
    if not season_values:
        return None, "filename_only"
    season_values = sorted(season_values)
    if len(season_values) == 1:
        return int(season_values[0]), "parquet_single_value"
    return int(season_values[0]), "parquet_multi_value"


def build_season_intercepts(processed_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_rows: list[dict] = []
    solver_rows: list[dict] = []
    audit_rows: list[dict] = []

    files = sorted(path for path in processed_dir.glob("*.parquet") if not should_skip_file(path))
    logging.info("Scanning %d processed parquet files", len(files))

    for path in files:
        try:
            source_prefix, season_from_filename, season_type = parse_processed_filename(path)
        except ValueError:
            logging.info("Skipping non-standard parquet filename: %s", path.name)
            continue

        parquet_file = pq.ParquetFile(path)
        source_columns = parquet_file.schema.names
        selected_columns = [column for column in source_columns if column == "Season" or column not in CONTROL_COLUMNS]
        df = pd.read_parquet(path, columns=selected_columns)
        row_count = len(df)

        parquet_season, parquet_season_mode = resolve_parquet_season(df)
        resolved_season = season_from_filename
        raw_metric_columns = append_raw_rows(
            raw_rows,
            df,
            source_prefix=source_prefix,
            source_file=path.name,
            season_end_year=resolved_season,
            season_type=season_type,
            row_count=row_count,
        )
        solver_metric_keys = append_solver_rows(
            solver_rows,
            df,
            source_prefix=source_prefix,
            source_file=path.name,
            season_end_year=resolved_season,
            season_type=season_type,
            row_count=row_count,
        )
        audit_rows.append({
            "source_file": path.name,
            "source_prefix": source_prefix,
            "season_end_year": resolved_season,
            "season_type": season_type,
            "row_count": row_count,
            "season_from_filename": season_from_filename,
            "season_from_parquet": parquet_season,
            "parquet_season_mode": parquet_season_mode,
            "season_mismatch": bool(parquet_season is not None and parquet_season != season_from_filename),
            "raw_metric_columns": ",".join(raw_metric_columns),
            "solver_metric_keys": ",".join(solver_metric_keys),
        })
        logging.info(
            "Processed %s -> season=%s %s, raw_cols=%d, solver_metrics=%s",
            path.name,
            resolved_season,
            season_type,
            len(raw_metric_columns),
            ",".join(solver_metric_keys) if solver_metric_keys else "none",
        )

    raw_df = pd.DataFrame(raw_rows).sort_values(
        by=["season_end_year", "season_type", "source_prefix", "source_column"]
    ).reset_index(drop=True)
    solver_df = pd.DataFrame(solver_rows).sort_values(
        by=["season_end_year", "season_type", "metric_key", "stat_name"]
    ).reset_index(drop=True)
    audit_df = pd.DataFrame(audit_rows).sort_values(
        by=["season_end_year", "season_type", "source_prefix"]
    ).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_season_column_means.csv"
    solver_path = output_dir / "solver_season_intercepts.csv"
    audit_path = output_dir / "processed_file_audit.csv"
    lookup_path = output_dir / "season_intercepts.json"

    raw_df.to_csv(raw_path, index=False)
    solver_df.to_csv(solver_path, index=False)
    audit_df.to_csv(audit_path, index=False)
    lookup_path.write_text(json.dumps(build_lookup_json(solver_df), indent=2, sort_keys=True))

    logging.info("Wrote raw column means to %s", raw_path)
    logging.info("Wrote solver intercepts to %s", solver_path)
    logging.info("Wrote file audit to %s", audit_path)
    logging.info("Wrote JSON lookup to %s", lookup_path)
    return raw_df, solver_df, audit_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build season-level intercept artifacts from processed parquets.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory containing processed parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where season-intercept artifacts should be written.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    raw_df, solver_df, audit_df = build_season_intercepts(args.processed_dir, args.output_dir)
    print(
        f"Built season intercept harness for {audit_df['source_file'].nunique()} files: "
        f"{len(raw_df)} raw means, {len(solver_df)} solver intercept rows."
    )


if __name__ == "__main__":
    main()
