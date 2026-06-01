#!/usr/bin/env python3
"""WNBA clean first-/second-chance EFG-value decomposition.

This is the WNBA mirror of the NBA clean alt3 EFG-value workflow:
- process raw WNBA CSVs into FIRST_CHANCE and SECOND_CHANCE_CLEAN CSVs
- solve the point-valued first-chance EFG/FT/completion components
- write a weighted-factors style display file under wnba_test/Results
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nba_pipeline.scripts.process_rapm_blocks.common import (
    build_shot_flags_df,
    series_contains,
)
from wnba_test.master_wnba_rapm import build_design_matrix, transform_to_home_away_format
from wnba_test.process_rapm_wnba import (
    _base_processing,
    _finalize_df,
    case_when,
    prepare_wnba_standard_possession_df,
)
from wnba_test.wnba_stats_source import load_wnba_player_shooting_stats


WNBA_ROOT = ROOT / "wnba_test"
PROCESSED_DIR = WNBA_ROOT / "Processed"
RESULTS_DIR = WNBA_ROOT / "Results"

ALT_VALUE_COLUMNS = {
    "ALT_EFG_RIM_FREQ": "FC_RIM_EFG_FREQ_Diff",
    "ALT_EFG_RIM_FG": "FC_RIM_EFG_FG_Diff",
    "ALT_EFG_MID_FREQ": "FC_MID_EFG_FREQ_Diff",
    "ALT_EFG_MID_FG": "FC_MID_EFG_FG_Diff",
    "ALT_EFG_THREE_FREQ": "FC_THREE_EFG_FREQ_Diff",
    "ALT_EFG_THREE_FG": "FC_THREE_EFG_FG_Diff",
    "ALT_FT": "FC_FT_Diff",
    "ALT_FT_FREQ": "FC_FT_FREQ_Diff",
    "ALT_FT_SEVERITY": "FC_FT_SEVERITY_Diff",
    "ALT_EFG_VALUE": "FC_EFG_VALUE_Diff",
}

DISPLAY_PREFIXES = [
    "ALT_EFG_RIM_FREQ",
    "ALT_EFG_RIM_FG",
    "ALT_EFG_MID_FREQ",
    "ALT_EFG_MID_FG",
    "ALT_EFG_THREE_FREQ",
    "ALT_EFG_THREE_FG",
    "ALT_FT",
    "ALT_FT_FREQ",
    "ALT_FT_SEVERITY",
    "ALT_FC_COMPLETION",
    "ALT_TOV_VALUE",
    "ALT_BADPASS_TOV_VALUE",
    "ALT_SCORING_TOV_VALUE",
    "SECOND_CHANCE_CLEAN",
]

WNBA_SEASON_DATE_BOUNDS = {
    2021: ("2021-05-14", "2021-10-17"),
    2022: ("2022-05-06", "2022-09-18"),
    2023: ("2023-05-19", "2023-10-18"),
    2024: ("2024-05-14", "2024-10-20"),
    2025: ("2025-05-16", "2025-10-10"),
}


def _season_from_suffix(year_suffix: int) -> tuple[int, str]:
    end_year = 2000 + int(year_suffix) if int(year_suffix) < 100 else int(year_suffix)
    return end_year, str(end_year)


def _load_name_map() -> dict[str, str]:
    path = WNBA_ROOT / "player_index_map.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["EntityId"].astype(str), df["pbp_name"].astype(str)))


def _infer_year_from_path(path: Path) -> int:
    match = re.search(r"(\d{2,4})(?:_PS)?\.csv$", path.name)
    if not match:
        raise ValueError(f"Could not infer season year from {path}")
    suffix = int(match.group(1)[-2:])
    return 2000 + suffix


def _attach_date_proxy(df: pd.DataFrame, source_year: int) -> pd.DataFrame:
    """Attach a stable game-date proxy when WNBA schedule dates are unavailable."""
    out = df.copy()
    if "game_date" in out.columns and out["game_date"].notna().any():
        return out

    start_str, end_str = WNBA_SEASON_DATE_BOUNDS.get(
        source_year,
        (f"{source_year}-05-15", f"{source_year}-10-15"),
    )
    start = pd.Timestamp(start_str)
    end = pd.Timestamp(end_str)
    game_ids = sorted(out["game_id"].astype(str).str.replace(r"\.0$", "", regex=True).unique())
    if len(game_ids) == 1:
        mapping = {game_ids[0]: end.strftime("%Y-%m-%d")}
    else:
        dates = pd.date_range(start=start, end=end, periods=len(game_ids))
        mapping = {game_id: date.strftime("%Y-%m-%d") for game_id, date in zip(game_ids, dates)}
    out["game_date"] = out["game_id"].astype(str).str.replace(r"\.0$", "", regex=True).map(mapping)
    return out


def _attach_wnba_ft_pct(
    df: pd.DataFrame,
    season_year: int,
    is_playoffs: bool,
    label: str,
) -> pd.DataFrame:
    """Attach local WNBA player-season FT% to the shooter row when available."""
    out = df.copy()
    if "FTPerc" not in out.columns:
        out["FTPerc"] = np.nan

    shooter_col = next(
        (col for col in ["player1_id", "player1_id_numeric", "PlayerID", "player_id"] if col in out.columns),
        None,
    )
    if shooter_col is None:
        logging.warning("%s has no shooter id column; FT%% will use existing/fallback values", label)
        return out

    stats = load_wnba_player_shooting_stats(season_year, is_playoffs)
    stats = stats[["nba_id", "FTPerc"]].rename(columns={"FTPerc": "_WNBA_FTPerc"})
    if stats.empty:
        logging.warning(
            "No WNBA FT%% rows for %s %s; FT%% will use existing/fallback values",
            season_year,
            "PS" if is_playoffs else "RS",
        )
        return out

    shooter_ids = pd.to_numeric(out[shooter_col], errors="coerce").astype("Int64").astype(str)
    shooter_ids = shooter_ids.mask(shooter_ids.eq("<NA>"), np.nan)
    existing_ft = pd.to_numeric(out["FTPerc"], errors="coerce")
    merged = pd.DataFrame({"_row": np.arange(len(out)), "_shooter_nba_id": shooter_ids}).merge(
        stats,
        left_on="_shooter_nba_id",
        right_on="nba_id",
        how="left",
    )
    out["FTPerc"] = pd.to_numeric(merged["_WNBA_FTPerc"], errors="coerce").combine_first(existing_ft)
    matched_rows = int(merged["_WNBA_FTPerc"].notna().sum())
    matched_players = int(merged.loc[merged["_WNBA_FTPerc"].notna(), "_shooter_nba_id"].nunique())
    logging.info(
        "Attached WNBA FT%% for %s %s %s: %d rows, %d shooters",
        season_year,
        "PS" if is_playoffs else "RS",
        label,
        matched_rows,
        matched_players,
    )
    return out


def _attach_clean_scoring(
    df: pd.DataFrame,
    label: str,
    season_year: int,
    is_playoffs: bool,
) -> pd.DataFrame:
    """Attach expected-FT scoring and first-chance component action scores."""
    out = _attach_wnba_ft_pct(df, season_year, is_playoffs, label)
    if "ThreePerc" not in out.columns:
        out["ThreePerc"] = np.nan

    out["ExpFT"] = pd.to_numeric(out["FTPerc"], errors="coerce").fillna(0.75)
    out["Exp3PT"] = pd.to_numeric(out["ThreePerc"], errors="coerce").fillna(0.35)

    desc_all = (
        out["home_description"].fillna("").astype(str) + " " +
        out["visitor_description"].fillna("").astype(str)
    ).str.strip()

    is_ft_event = (
        series_contains(out["home_description"], "Free Throw", case=False) |
        series_contains(out["visitor_description"], "Free Throw", case=False) |
        (out["event_type"] == "FreeThrow")
    )
    is_fga = out["event_type"].isin(["MAKE", "MISS"])
    is_2pt_make = (
        series_contains(out["home_description"], "PTS", case=False) &
        ~series_contains(out["home_description"], "3PT", case=False) &
        (out["event_type"] == "MAKE")
    ) | (
        series_contains(out["visitor_description"], "PTS", case=False) &
        ~series_contains(out["visitor_description"], "3PT", case=False) &
        (out["event_type"] == "MAKE")
    )
    is_3pt_event = (
        series_contains(out["home_description"], "3PT", case=False) |
        series_contains(out["visitor_description"], "3PT", case=False)
    )
    actual_score = (out["Home_Action_Score"] + out["Away_Action_Score"]).astype(float)

    out["ActualNet"] = case_when(
        is_ft_event, out["ExpFT"],
        is_2pt_make, 2.0,
        is_3pt_event, actual_score,
        0.0,
    ).astype(float)

    fga_made = is_fga & (out["event_type"] == "MAKE")
    fga_3pt = desc_all.str.contains("3PT", case=False, na=False)
    fga_pts = np.where(fga_made & fga_3pt, 3.0, np.where(fga_made, 2.0, 0.0))
    avg_actual_pts = float(fga_pts[is_fga].mean()) if is_fga.any() else 1.09

    is_tech_ft = (
        series_contains(out["home_description"], "Technical", case=False) |
        series_contains(out["visitor_description"], "Technical", case=False)
    ) & is_ft_event
    is_1of1 = series_contains(desc_all, r"\b1 of 1\b", regex=True) & is_ft_event
    is_3shot = series_contains(desc_all, r"\b[123] of 3\b", regex=True) & is_ft_event
    is_2shot = series_contains(desc_all, r"\b[12] of 2\b", regex=True) & is_ft_event
    is_and1 = is_1of1 & ~is_tech_ft
    ft_baseline = np.where(
        is_and1 | is_tech_ft,
        0.0,
        np.where(is_3shot, avg_actual_pts / 3.0, np.where(is_2shot, avg_actual_pts / 2.0, 0.0)),
    )

    out["FC_FT_Action_Score"] = np.where(is_ft_event, out["ExpFT"] - ft_baseline, 0.0)
    out["FC_FT_SQ_BASELINE_Action_Score"] = ft_baseline
    logging.info("Calculated WNBA clean scoring for %s with avg FGA pts %.4f", label, avg_actual_pts)
    return out


def build_clean_chance_base(
    file_path: Path,
    year: int,
    season_str: str,
    label: str,
    is_playoffs: bool,
) -> pd.DataFrame | None:
    """Build WNBA clean first-/second-chance possession rows."""
    logging.info("Starting %s base processing for %s", label, season_str)
    raw = _base_processing(str(file_path), allow_partial_lineups=True)
    if raw is None or raw.empty:
        return None

    checked, group_cols = prepare_wnba_standard_possession_df(raw, label)
    checked = _attach_clean_scoring(checked, label, year, is_playoffs)

    if group_cols:
        checked["poss_group"] = checked.groupby(group_cols)["End_of_Possession"].transform(
            lambda x: x.shift(1, fill_value=False).cumsum()
        )
    else:
        checked["poss_group"] = checked["End_of_Possession"].shift(1, fill_value=False).cumsum()

    poss_group_key = group_cols + ["poss_group"] if group_cols else ["poss_group"]
    checked["poss_offense"] = checked.groupby(poss_group_key)["TeamOnOffense"].transform(
        lambda x: x.replace("", np.nan).bfill().ffill()
    )

    is_miss = checked["event_type"] == "MISS"
    home_miss = is_miss & series_contains(checked["home_description"], "MISS", case=False)
    away_miss = is_miss & series_contains(checked["visitor_description"], "MISS", case=False)
    checked["off_miss"] = (
        (home_miss & (checked["poss_offense"] == "Home")) |
        (away_miss & (checked["poss_offense"] == "Away"))
    )
    checked["after_miss"] = checked.groupby(poss_group_key)["off_miss"].transform(
        lambda x: x.shift(1, fill_value=False).cummax()
    ).astype(bool)

    shot_flags = build_shot_flags_df(checked)
    is_fga = shot_flags["is_fga"].astype(bool)
    is_rim = is_fga & shot_flags["is_rim"].astype(bool) & ~shot_flags["is_3pt"].astype(bool)
    is_three = is_fga & shot_flags["is_3pt"].astype(bool)
    is_mid = is_fga & ~is_rim & ~is_three
    first_chance_fga = is_fga & ~checked["after_miss"]

    fc_efg_avg = (
        float(checked.loc[first_chance_fga, "ActualNet"].mean())
        if first_chance_fga.any()
        else float(checked.loc[is_fga, "ActualNet"].mean()) if is_fga.any()
        else 1.09
    )
    zone_avgs = {
        "RIM": float(checked.loc[first_chance_fga & is_rim, "ActualNet"].mean())
        if (first_chance_fga & is_rim).any() else fc_efg_avg,
        "MID": float(checked.loc[first_chance_fga & is_mid, "ActualNet"].mean())
        if (first_chance_fga & is_mid).any() else fc_efg_avg,
        "THREE": float(checked.loc[first_chance_fga & is_three, "ActualNet"].mean())
        if (first_chance_fga & is_three).any() else fc_efg_avg,
    }

    desc_all = (
        checked["home_description"].fillna("").astype(str) + " " +
        checked["visitor_description"].fillna("").astype(str)
    ).str.strip()
    is_ft_event = (
        series_contains(checked["home_description"], "Free Throw", case=False) |
        series_contains(checked["visitor_description"], "Free Throw", case=False) |
        (checked["event_type"] == "FreeThrow")
    )
    is_tech_ft = (
        series_contains(checked["home_description"], "Technical", case=False) |
        series_contains(checked["visitor_description"], "Technical", case=False)
    ) & is_ft_event
    is_1of1 = series_contains(desc_all, r"\b1 of 1\b", regex=True) & is_ft_event
    is_3shot = series_contains(desc_all, r"\b[123] of 3\b", regex=True) & is_ft_event
    is_2shot = series_contains(desc_all, r"\b[12] of 2\b", regex=True) & is_ft_event
    is_and1 = is_1of1 & ~is_tech_ft
    ft_baseline = np.where(
        is_and1 | is_tech_ft,
        0.0,
        np.where(is_3shot, fc_efg_avg / 3.0, np.where(is_2shot, fc_efg_avg / 2.0, 0.0)),
    )

    checked["FC_FT_Action_Score"] = np.where(is_ft_event, checked["ExpFT"] - ft_baseline, 0.0)
    checked["FC_EFG_BASELINE_Action_Score"] = np.where(is_fga, fc_efg_avg, ft_baseline)

    first_chance_ft = is_ft_event & ~checked["after_miss"].fillna(False).astype(bool)
    checked["_fc_ft_event"] = first_chance_ft.astype(bool)
    if group_cols:
        checked["_prev_fc_ft_event"] = checked.groupby(poss_group_key)["_fc_ft_event"].shift(fill_value=False)
        checked["_fc_ft_trip_start"] = checked["_fc_ft_event"] & ~checked["_prev_fc_ft_event"].astype(bool)
        checked["_fc_ft_trip_id"] = checked.groupby(group_cols)["_fc_ft_trip_start"].cumsum()
        ft_trip_keys = group_cols + ["_fc_ft_trip_id"]
    else:
        checked["_prev_fc_ft_event"] = checked["_fc_ft_event"].shift(fill_value=False)
        checked["_fc_ft_trip_start"] = checked["_fc_ft_event"] & ~checked["_prev_fc_ft_event"].astype(bool)
        checked["_fc_ft_trip_id"] = checked["_fc_ft_trip_start"].cumsum()
        ft_trip_keys = ["_fc_ft_trip_id"]

    checked["_fc_ft_trip_value"] = 0.0
    if first_chance_ft.any():
        checked.loc[first_chance_ft, "_fc_ft_trip_value"] = (
            checked.loc[first_chance_ft]
            .groupby(ft_trip_keys)["FC_FT_Action_Score"]
            .transform("sum")
        )
    ft_trip_starts = checked["_fc_ft_trip_start"].astype(bool)
    avg_ft_trip_value = (
        float(checked.loc[ft_trip_starts, "_fc_ft_trip_value"].mean())
        if ft_trip_starts.any()
        else 0.0
    )
    checked["FC_FT_FREQ_Action_Score"] = np.where(
        ft_trip_starts,
        avg_ft_trip_value,
        0.0,
    )
    checked["FC_FT_SEVERITY_Action_Score"] = np.where(
        ft_trip_starts,
        checked["_fc_ft_trip_value"] - avg_ft_trip_value,
        0.0,
    )

    zone_masks = {"RIM": is_rim, "MID": is_mid, "THREE": is_three}
    for zone, mask in zone_masks.items():
        avg = zone_avgs[zone]
        checked[f"FC_{zone}_EFG_AA_Action_Score"] = np.where(mask, checked["ActualNet"] - fc_efg_avg, 0.0)
        checked[f"FC_{zone}_EFG_FREQ_Action_Score"] = np.where(mask, avg - fc_efg_avg, 0.0)
        checked[f"FC_{zone}_EFG_FG_Action_Score"] = np.where(mask, checked["ActualNet"] - avg, 0.0)

    checked["FC_EFG_VALUE_Action_Score"] = (
        checked["FC_RIM_EFG_AA_Action_Score"] +
        checked["FC_MID_EFG_AA_Action_Score"] +
        checked["FC_THREE_EFG_AA_Action_Score"]
    )

    checked["First_Chance_Net"] = np.where(checked["after_miss"], 0.0, checked["ActualNet"])
    checked["Second_Chance_Net"] = np.where(checked["after_miss"], checked["ActualNet"], 0.0)

    component_pairs = [
        ("FC_FT_Action_Score", "First_Chance_FT_Net"),
        ("FC_FT_FREQ_Action_Score", "First_Chance_FT_FREQ_Net"),
        ("FC_FT_SEVERITY_Action_Score", "First_Chance_FT_SEVERITY_Net"),
        ("FC_EFG_BASELINE_Action_Score", "First_Chance_EFG_BASELINE_Net"),
        ("FC_EFG_VALUE_Action_Score", "First_Chance_EFG_VALUE_Net"),
        ("FC_RIM_EFG_AA_Action_Score", "First_Chance_RIM_EFG_AA_Net"),
        ("FC_MID_EFG_AA_Action_Score", "First_Chance_MID_EFG_AA_Net"),
        ("FC_THREE_EFG_AA_Action_Score", "First_Chance_THREE_EFG_AA_Net"),
        ("FC_RIM_EFG_FREQ_Action_Score", "First_Chance_RIM_EFG_FREQ_Net"),
        ("FC_RIM_EFG_FG_Action_Score", "First_Chance_RIM_EFG_FG_Net"),
        ("FC_MID_EFG_FREQ_Action_Score", "First_Chance_MID_EFG_FREQ_Net"),
        ("FC_MID_EFG_FG_Action_Score", "First_Chance_MID_EFG_FG_Net"),
        ("FC_THREE_EFG_FREQ_Action_Score", "First_Chance_THREE_EFG_FREQ_Net"),
        ("FC_THREE_EFG_FG_Action_Score", "First_Chance_THREE_EFG_FG_Net"),
    ]
    for source, target in component_pairs:
        checked[target] = np.where(checked["after_miss"], 0.0, checked[source])

    poss_groupers = [checked[col] for col in poss_group_key]
    checked["first_chance_completion_event"] = (
        ~checked["after_miss"].fillna(False).astype(bool) &
        (is_fga | is_ft_event)
    )
    completion_seen = (
        checked["first_chance_completion_event"]
        .astype(int)
        .groupby(poss_groupers, sort=False)
        .cumsum()
    )
    checked["had_first_chance_completion_before"] = (
        completion_seen
        .groupby(poss_groupers, sort=False)
        .shift(1, fill_value=0)
        .gt(0)
    )

    cumulative_pairs = [
        ("Actual_Total", "ActualNet"),
        ("First_Chance_Total", "First_Chance_Net"),
        ("Second_Chance_Total", "Second_Chance_Net"),
        *[(target.replace("_Net", "_Total"), target) for _, target in component_pairs],
    ]
    for total, net in cumulative_pairs:
        if group_cols:
            checked[total] = checked.groupby(group_cols)[net].cumsum()
        else:
            checked[total] = checked[net].cumsum()

    filt = checked[checked["End_of_Possession"]].copy()
    if filt.empty:
        return None

    diff_pairs = [
        ("Actual_Diff", "Actual_Total"),
        ("First_Chance_Diff", "First_Chance_Total"),
        ("Second_Chance_Diff", "Second_Chance_Total"),
        ("FC_FT_Diff", "First_Chance_FT_Total"),
        ("FC_FT_FREQ_Diff", "First_Chance_FT_FREQ_Total"),
        ("FC_FT_SEVERITY_Diff", "First_Chance_FT_SEVERITY_Total"),
        ("FC_EFG_BASELINE_Diff", "First_Chance_EFG_BASELINE_Total"),
        ("FC_EFG_VALUE_Diff", "First_Chance_EFG_VALUE_Total"),
        ("FC_RIM_EFG_AA_Diff", "First_Chance_RIM_EFG_AA_Total"),
        ("FC_MID_EFG_AA_Diff", "First_Chance_MID_EFG_AA_Total"),
        ("FC_THREE_EFG_AA_Diff", "First_Chance_THREE_EFG_AA_Total"),
        ("FC_RIM_EFG_FREQ_Diff", "First_Chance_RIM_EFG_FREQ_Total"),
        ("FC_RIM_EFG_FG_Diff", "First_Chance_RIM_EFG_FG_Total"),
        ("FC_MID_EFG_FREQ_Diff", "First_Chance_MID_EFG_FREQ_Total"),
        ("FC_MID_EFG_FG_Diff", "First_Chance_MID_EFG_FG_Total"),
        ("FC_THREE_EFG_FREQ_Diff", "First_Chance_THREE_EFG_FREQ_Total"),
        ("FC_THREE_EFG_FG_Diff", "First_Chance_THREE_EFG_FG_Total"),
    ]
    for diff, total in diff_pairs:
        if group_cols:
            filt[diff] = filt[total] - filt.groupby(group_cols)[total].shift(fill_value=0)
        else:
            filt[diff] = filt[total] - filt[total].shift(fill_value=0)

    desc_eop = (
        filt["home_description"].fillna("").astype(str) + " " +
        filt["visitor_description"].fillna("").astype(str)
    )
    filt["Is_Turnover"] = (
        (filt["event_type"] == "Turnover") &
        ~filt["after_miss"].fillna(False).astype(bool) &
        ~filt["had_first_chance_completion_before"].fillna(False).astype(bool)
    ).astype(int)
    filt["Is_BadPass_TOV"] = (
        filt["Is_Turnover"].astype(bool) &
        desc_eop.str.contains("Bad Pass", case=False, na=False)
    ).astype(int)
    filt["FC_EFG_Diff"] = (
        pd.to_numeric(filt["First_Chance_Diff"], errors="coerce").fillna(0.0) -
        pd.to_numeric(filt["FC_FT_Diff"], errors="coerce").fillna(0.0)
    )

    identity_gap = (filt["First_Chance_Diff"] + filt["Second_Chance_Diff"] - filt["Actual_Diff"]).abs().max()
    value_cols = [
        "FC_RIM_EFG_FREQ_Diff",
        "FC_RIM_EFG_FG_Diff",
        "FC_MID_EFG_FREQ_Diff",
        "FC_MID_EFG_FG_Diff",
        "FC_THREE_EFG_FREQ_Diff",
        "FC_THREE_EFG_FG_Diff",
    ]
    value_gap = (filt["FC_EFG_VALUE_Diff"] - filt[value_cols].sum(axis=1)).abs().max()
    ft_gap = (
        pd.to_numeric(filt["FC_FT_Diff"], errors="coerce").fillna(0.0) -
        pd.to_numeric(filt["FC_FT_FREQ_Diff"], errors="coerce").fillna(0.0) -
        pd.to_numeric(filt["FC_FT_SEVERITY_Diff"], errors="coerce").fillna(0.0)
    ).abs().max()
    logging.info(
        "%s identity gap %.8f; FT split gap %.8f; EFG value component gap %.8f",
        label,
        identity_gap,
        ft_gap,
        value_gap,
    )
    return filt


def process_first_chance(
    file_path: Path,
    year: int,
    season_str: str,
    is_playoffs: bool,
) -> pd.DataFrame | None:
    filt = build_clean_chance_base(file_path, year, season_str, "FIRST_CHANCE", is_playoffs)
    if filt is None or filt.empty:
        return None
    filt = filt.copy()
    filt["Net_Diff"] = filt["First_Chance_Diff"]
    cols = [
        "Net_Diff",
        "Is_Turnover",
        "Is_BadPass_TOV",
        "FC_FT_Diff",
        "FC_FT_FREQ_Diff",
        "FC_FT_SEVERITY_Diff",
        "FC_EFG_Diff",
        "FC_EFG_BASELINE_Diff",
        "FC_EFG_VALUE_Diff",
        "FC_RIM_EFG_AA_Diff",
        "FC_MID_EFG_AA_Diff",
        "FC_THREE_EFG_AA_Diff",
        "FC_RIM_EFG_FREQ_Diff",
        "FC_RIM_EFG_FG_Diff",
        "FC_MID_EFG_FREQ_Diff",
        "FC_MID_EFG_FG_Diff",
        "FC_THREE_EFG_FREQ_Diff",
        "FC_THREE_EFG_FG_Diff",
    ]
    return _finalize_df(filt, cols, year)


def process_second_chance_clean(
    file_path: Path,
    year: int,
    season_str: str,
    is_playoffs: bool,
) -> pd.DataFrame | None:
    filt = build_clean_chance_base(file_path, year, season_str, "SECOND_CHANCE_CLEAN", is_playoffs)
    if filt is None or filt.empty:
        return None
    filt = filt.copy()
    filt["Net_Diff"] = filt["Second_Chance_Diff"]
    filt["Off_Diff"] = filt["Second_Chance_Diff"]
    filt["Def_Diff"] = -filt["Second_Chance_Diff"]
    return _finalize_df(filt, ["Net_Diff", "Off_Diff", "Def_Diff"], year)


def process_raw_file(raw_file: Path) -> list[Path]:
    match = re.search(r"WNBA(\d{2,4})(?:_PS)?\.csv$", raw_file.name, flags=re.I)
    if not match:
        raise ValueError(f"Could not infer WNBA season from {raw_file}")
    year_suffix = int(match.group(1)[-2:])
    year, season_str = _season_from_suffix(year_suffix)
    is_playoffs = raw_file.stem.endswith("_PS")
    ps_suffix = "_PS" if is_playoffs else ""

    outputs: list[Path] = []
    for metric, fn in [
        ("FIRST_CHANCE", process_first_chance),
        ("SECOND_CHANCE_CLEAN", process_second_chance_clean),
    ]:
        out = fn(raw_file, year, season_str, is_playoffs)
        if out is None or out.empty:
            raise RuntimeError(f"{metric} processing produced no rows for {raw_file}")
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        path = PROCESSED_DIR / f"{metric}{year_suffix:02d}{ps_suffix}.csv"
        out.to_csv(path, index=False)
        logging.info("Wrote %s (%d rows)", path, len(out))
        outputs.append(path)
    return outputs


def _files_for_metric(prefix: str, years: Iterable[int], season_type: str) -> list[Path]:
    source = "FIRST_CHANCE" if prefix.startswith("ALT_") else prefix
    files: list[Path] = []
    for year in years:
        if season_type in {"RS", "ALL"}:
            files.append(PROCESSED_DIR / f"{source}{year:02d}.csv")
        if season_type in {"PS", "ALL"}:
            files.append(PROCESSED_DIR / f"{source}{year:02d}_PS.csv")
    return files


def _load_clean_total_df(years: Iterable[int], season_type: str) -> pd.DataFrame:
    """Load FIRST_CHANCE + SECOND_CHANCE_CLEAN as the clean total target."""
    pieces: list[pd.DataFrame] = []
    key_cols = ["game_id", "event_num", *[f"O{i}" for i in range(1, 6)], *[f"D{i}" for i in range(1, 6)]]
    for year in years:
        suffixes = []
        if season_type in {"RS", "ALL"}:
            suffixes.append("")
        if season_type in {"PS", "ALL"}:
            suffixes.append("_PS")
        for suffix in suffixes:
            fc_path = PROCESSED_DIR / f"FIRST_CHANCE{year:02d}{suffix}.csv"
            sc_path = PROCESSED_DIR / f"SECOND_CHANCE_CLEAN{year:02d}{suffix}.csv"
            if not fc_path.exists() or not sc_path.exists():
                raise FileNotFoundError(f"Missing clean total inputs: {fc_path} / {sc_path}")
            source_year = _infer_year_from_path(fc_path)
            fc = _attach_date_proxy(pd.read_csv(fc_path), source_year)
            sc = _attach_date_proxy(pd.read_csv(sc_path), source_year)
            if len(fc) != len(sc):
                raise ValueError(f"Clean row mismatch for {year:02d}{suffix}: {len(fc)} vs {len(sc)}")
            present_keys = [col for col in key_cols if col in fc.columns and col in sc.columns]
            for col in present_keys:
                if not fc[col].astype(str).reset_index(drop=True).equals(sc[col].astype(str).reset_index(drop=True)):
                    raise ValueError(f"Clean row key mismatch on {col} for {year:02d}{suffix}")
            total = fc.copy()
            total["Net_Diff"] = (
                pd.to_numeric(fc["Net_Diff"], errors="coerce").fillna(0.0) +
                pd.to_numeric(sc["Net_Diff"], errors="coerce").fillna(0.0)
            )
            total["Off_Diff"] = total["Net_Diff"]
            total["Def_Diff"] = -total["Net_Diff"]
            pieces.append(total)
    return pd.concat(pieces, ignore_index=True)


def _prepare_metric_df(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.copy()
    if prefix == "RAPM":
        out["Off_Diff"] = out["Net_Diff"]
        out["Def_Diff"] = -out["Net_Diff"]
        return out
    if prefix == "SECOND_CHANCE_CLEAN":
        return out
    if not prefix.startswith("ALT_"):
        return out

    if "Is_Turnover" not in out.columns:
        raise ValueError(f"{prefix} requires FIRST_CHANCE rows")
    non_tov = out["Is_Turnover"].astype(int) == 0
    if not non_tov.any():
        raise ValueError(f"{prefix} has no non-turnover rows for baseline calculation")

    baselines = {
        alias: float(pd.to_numeric(out.loc[non_tov, col], errors="coerce").mean())
        for alias, col in ALT_VALUE_COLUMNS.items()
        if col in out.columns
    }
    fc_non_tov_avg = float(pd.to_numeric(out.loc[non_tov, "Net_Diff"], errors="coerce").mean())

    if prefix in ALT_VALUE_COLUMNS:
        col = ALT_VALUE_COLUMNS[prefix]
        target = np.where(out["Is_Turnover"].astype(int).eq(1), baselines[prefix], out[col])
    elif prefix == "ALT_FC_COMPLETION":
        efg = np.where(out["Is_Turnover"].astype(int).eq(1), baselines["ALT_EFG_VALUE"], out["FC_EFG_VALUE_Diff"])
        ft = np.where(out["Is_Turnover"].astype(int).eq(1), baselines["ALT_FT"], out["FC_FT_Diff"])
        target = pd.to_numeric(out["Net_Diff"], errors="coerce").fillna(0.0).to_numpy() - efg - ft
    elif prefix == "ALT_TOV_VALUE":
        target = np.where(out["Is_Turnover"].astype(int).eq(1), out["Net_Diff"] - fc_non_tov_avg, 0.0)
    elif prefix == "ALT_BADPASS_TOV_VALUE":
        is_bad = out["Is_BadPass_TOV"].astype(int).eq(1)
        target = np.where(is_bad, out["Net_Diff"] - fc_non_tov_avg, 0.0)
    elif prefix == "ALT_SCORING_TOV_VALUE":
        is_scoring = out["Is_Turnover"].astype(int).eq(1) & out["Is_BadPass_TOV"].astype(int).eq(0)
        target = np.where(is_scoring, out["Net_Diff"] - fc_non_tov_avg, 0.0)
    else:
        raise ValueError(f"Unsupported WNBA clean metric prefix: {prefix}")

    out["Off_Diff"] = pd.to_numeric(pd.Series(target, index=out.index), errors="coerce").fillna(0.0)
    out["Def_Diff"] = -out["Off_Diff"]
    return out


def _time_decay_weights(transformed: pd.DataFrame, half_life_days: float | None) -> np.ndarray:
    if not half_life_days:
        return np.ones(len(transformed), dtype=float)
    if "game_date" not in transformed.columns:
        raise ValueError("Time decay requested but game_date is unavailable.")
    dates = pd.to_datetime(transformed["game_date"], errors="coerce")
    if dates.isna().any():
        missing = int(dates.isna().sum())
        raise ValueError(f"Time decay requested but {missing} transformed rows have missing game_date.")
    max_date = dates.max()
    age_days = (max_date - dates).dt.days.astype(float).to_numpy()
    return np.power(0.5, age_days / float(half_life_days))


def _run_direct_ridge(
    df: pd.DataFrame,
    name_map: dict[str, str],
    off_alpha: float = 3000.0,
    def_alpha: float = 3000.0,
    time_decay_days: float | None = None,
) -> pd.DataFrame:
    """Run one direct ridge RAPM solve and keep full precision for additivity."""
    transformed = transform_to_home_away_format(df)
    if "game_date" not in transformed.columns and "game_date" in df.columns:
        game_dates = (
            df[["game_id", "game_date"]]
            .dropna(subset=["game_date"])
            .assign(gameid=lambda x: x["game_id"].astype(str))
            .drop_duplicates("gameid")
            [["gameid", "game_date"]]
        )
        transformed = transformed.merge(game_dates, on="gameid", how="left")
    all_players = set()
    for col in [f"a{i}" for i in range(1, 6)] + [f"h{i}" for i in range(1, 6)]:
        all_players.update(transformed[col].astype(int).astype(str).unique())
    all_players = sorted(player for player in all_players if str(player) != "0")

    player_to_col: dict[str, int] = {}
    idx = 0
    for player in all_players:
        player_to_col[f"{player}_off"] = idx
        idx += 1
        player_to_col[f"{player}_def"] = idx
        idx += 1

    X, y = build_design_matrix(transformed, player_to_col)
    weights = _time_decay_weights(transformed, time_decay_days)
    sqrt_weights = np.sqrt(weights)
    X_weighted = X.multiply(sqrt_weights[:, None])
    y_weighted = (y - np.average(y, weights=weights)) * sqrt_weights
    penalties = np.zeros(X.shape[1], dtype=float)
    for key, col in player_to_col.items():
        penalties[col] = off_alpha if key.endswith("_off") else def_alpha
    xtx = (X_weighted.T @ X_weighted).toarray()
    xty = X_weighted.T @ y_weighted
    beta = np.linalg.solve(xtx + np.diag(penalties), xty).astype(float)

    off_poss = {player: 0 for player in all_players}
    def_poss = {player: 0 for player in all_players}
    for _, row in transformed.iterrows():
        away = [str(int(row[f"a{i}"])) for i in range(1, 6)]
        home = [str(int(row[f"h{i}"])) for i in range(1, 6)]
        offense, defense = (home, away) if row["home_poss"] == 1 else (away, home)
        for player in offense:
            if player == "0":
                continue
            off_poss[player] = off_poss.get(player, 0) + 1
        for player in defense:
            if player == "0":
                continue
            def_poss[player] = def_poss.get(player, 0) + 1

    total_off = sum(off_poss.values())
    total_def = sum(def_poss.values())
    if total_off:
        off_offset = sum(beta[player_to_col[f"{p}_off"]] * off_poss[p] for p in all_players) / total_off
        for player in all_players:
            beta[player_to_col[f"{player}_off"]] -= off_offset
    if total_def:
        def_offset = sum(beta[player_to_col[f"{p}_def"]] * def_poss[p] for p in all_players) / total_def
        for player in all_players:
            beta[player_to_col[f"{player}_def"]] -= def_offset

    rows = []
    for player in all_players:
        off = beta[player_to_col[f"{player}_off"]] * 100.0
        defense = beta[player_to_col[f"{player}_def"]] * 100.0
        rows.append({
            "player_id": player,
            "player_name": name_map.get(player, f"ID_{player}"),
            "off": off,
            "def": defense,
            "net_rapm": off - defense,
            "off_poss": off_poss[player],
            "def_poss": def_poss[player],
        })
    return pd.DataFrame(rows).sort_values("net_rapm", ascending=False).reset_index(drop=True)


def run_metric(
    prefix: str,
    years: list[int],
    season_type: str,
    name_map: dict[str, str],
    off_alpha: float,
    def_alpha: float,
    time_decay_days: float | None,
) -> pd.DataFrame:
    if prefix == "RAPM":
        df = _load_clean_total_df(years, season_type)
    else:
        files = [path for path in _files_for_metric(prefix, years, season_type) if path.exists()]
        if not files:
            raise FileNotFoundError(f"No processed files found for {prefix}: {_files_for_metric(prefix, years, season_type)}")
        dfs = [_attach_date_proxy(pd.read_csv(path), _infer_year_from_path(path)) for path in files]
        df = _prepare_metric_df(pd.concat(dfs, ignore_index=True), prefix)
    return _run_direct_ridge(
        df,
        name_map,
        off_alpha=off_alpha,
        def_alpha=def_alpha,
        time_decay_days=time_decay_days,
    )


def _component_series(results: dict[str, pd.DataFrame], prefix: str, side: str, player_ids: pd.Series) -> pd.Series:
    df = results[prefix][["player_id", side]].copy()
    df["player_id"] = df["player_id"].astype(str)
    merged = pd.DataFrame({"player_id": player_ids.astype(str)}).merge(df, on="player_id", how="left")
    values = pd.to_numeric(merged[side], errors="coerce").fillna(0.0)
    return values


def build_clean_weighted_factors(
    start_year: int,
    end_year: int,
    season_type: str,
    off_alpha: float = 3000.0,
    def_alpha: float = 3000.0,
    time_decay_days: float | None = None,
) -> Path:
    years = list(range(start_year, end_year + 1))
    name_map = _load_name_map()
    prefixes = ["RAPM", *DISPLAY_PREFIXES]
    results = {}
    for prefix in prefixes:
        logging.info("Solving WNBA %s", prefix)
        results[prefix] = run_metric(
            prefix,
            years,
            season_type,
            name_map,
            off_alpha=off_alpha,
            def_alpha=def_alpha,
            time_decay_days=time_decay_days,
        )

    base = results["RAPM"].copy()
    base["player_id"] = base["player_id"].astype(str)
    out = base[["player_id", "player_name", "off_poss", "def_poss"]].copy()
    out["Latest_Year"] = 2000 + end_year
    out["off"] = pd.to_numeric(base["off"], errors="coerce")
    out["def"] = -pd.to_numeric(base["def"], errors="coerce")
    out["net_rapm"] = pd.to_numeric(base["net_rapm"], errors="coerce")
    out["possessions"] = out["off_poss"] + out["def_poss"]

    player_ids = out["player_id"]
    for prefix in DISPLAY_PREFIXES:
        if prefix == "SECOND_CHANCE_CLEAN":
            out["oSC"] = _component_series(results, prefix, "off", player_ids)
            out["dSC"] = -_component_series(results, prefix, "def", player_ids)
            continue
        if prefix == "ALT_FC_COMPLETION":
            col = "ALT_TOV_VALUE"
        elif prefix == "ALT_TOV_VALUE":
            col = "ALT_TOV_LOSS_VALUE"
        else:
            col = prefix
        out[f"o{col}"] = _component_series(results, prefix, "off", player_ids)
        out[f"d{col}"] = -_component_series(results, prefix, "def", player_ids)

    for side in ["o", "d"]:
        out[f"{side}ALT_EFG_RIM"] = out[f"{side}ALT_EFG_RIM_FREQ"] + out[f"{side}ALT_EFG_RIM_FG"]
        out[f"{side}ALT_EFG_MID"] = out[f"{side}ALT_EFG_MID_FREQ"] + out[f"{side}ALT_EFG_MID_FG"]
        out[f"{side}ALT_EFG_THREE"] = out[f"{side}ALT_EFG_THREE_FREQ"] + out[f"{side}ALT_EFG_THREE_FG"]
        out[f"{side}ALT_EFG_VALUE"] = (
            out[f"{side}ALT_EFG_RIM"] + out[f"{side}ALT_EFG_MID"] + out[f"{side}ALT_EFG_THREE"]
        )
        out[f"{side}FC"] = out[f"{side}ALT_EFG_VALUE"] + out[f"{side}ALT_FT"] + out[f"{side}ALT_TOV_VALUE"]
        out[f"{side}DISPLAY_SUM"] = out[f"{side}FC"] + out[f"{side}SC"]

    out["nALT_EFG_VALUE"] = out["oALT_EFG_VALUE"] + out["dALT_EFG_VALUE"]
    out["nALT_FT"] = out["oALT_FT"] + out["dALT_FT"]
    out["nALT_FT_FREQ"] = out["oALT_FT_FREQ"] + out["dALT_FT_FREQ"]
    out["nALT_FT_SEVERITY"] = out["oALT_FT_SEVERITY"] + out["dALT_FT_SEVERITY"]
    out["nALT_TOV_VALUE"] = out["oALT_TOV_VALUE"] + out["dALT_TOV_VALUE"]
    out["nSC"] = out["oSC"] + out["dSC"]
    out["nDISPLAY_SUM"] = out["oDISPLAY_SUM"] + out["dDISPLAY_SUM"]
    out["oRESID"] = out["off"] - out["oDISPLAY_SUM"]
    out["dRESID"] = out["def"] - out["dDISPLAY_SUM"]
    out["RESID"] = out["net_rapm"] - out["nDISPLAY_SUM"]
    out["oALT_TOV_VALUE_split"] = out["oALT_BADPASS_TOV_VALUE"] + out["oALT_SCORING_TOV_VALUE"]
    out["dALT_TOV_VALUE_split"] = out["dALT_BADPASS_TOV_VALUE"] + out["dALT_SCORING_TOV_VALUE"]

    value_cols = out.select_dtypes(include=[np.number]).columns.difference(
        ["Latest_Year", "off_poss", "def_poss", "possessions"]
    )
    out[value_cols] = out[value_cols].round(3)

    ordered = [
        "player_id", "player_name", "Latest_Year",
        "oALT_EFG_VALUE", "oALT_EFG_RIM", "oALT_EFG_RIM_FREQ", "oALT_EFG_RIM_FG",
        "oALT_EFG_MID", "oALT_EFG_MID_FREQ", "oALT_EFG_MID_FG",
        "oALT_EFG_THREE", "oALT_EFG_THREE_FREQ", "oALT_EFG_THREE_FG",
        "oALT_FT", "oALT_FT_FREQ", "oALT_FT_SEVERITY", "oALT_TOV_VALUE", "oSC", "oFC", "oDISPLAY_SUM", "oRESID",
        "oALT_TOV_LOSS_VALUE", "oALT_BADPASS_TOV_VALUE", "oALT_SCORING_TOV_VALUE", "oALT_TOV_VALUE_split",
        "dALT_EFG_VALUE", "dALT_EFG_RIM", "dALT_EFG_RIM_FREQ", "dALT_EFG_RIM_FG",
        "dALT_EFG_MID", "dALT_EFG_MID_FREQ", "dALT_EFG_MID_FG",
        "dALT_EFG_THREE", "dALT_EFG_THREE_FREQ", "dALT_EFG_THREE_FG",
        "dALT_FT", "dALT_FT_FREQ", "dALT_FT_SEVERITY", "dALT_TOV_VALUE", "dSC", "dFC", "dDISPLAY_SUM", "dRESID",
        "dALT_TOV_LOSS_VALUE", "dALT_BADPASS_TOV_VALUE", "dALT_SCORING_TOV_VALUE", "dALT_TOV_VALUE_split",
        "nALT_EFG_VALUE", "nALT_FT", "nALT_FT_FREQ", "nALT_FT_SEVERITY", "nALT_TOV_VALUE", "nSC", "nDISPLAY_SUM", "RESID",
        "off", "def", "net_rapm", "possessions", "off_poss", "def_poss",
    ]
    out = out[ordered].sort_values("net_rapm", ascending=False)

    year_range = f"{start_year:02d}_{end_year:02d}" if start_year != end_year else f"{end_year:02d}"
    suffix = season_type.lower()
    run_suffix_parts = [suffix]
    if time_decay_days:
        run_suffix_parts.append(f"td{int(time_decay_days)}")
    if off_alpha != 3000.0 or def_alpha != 3000.0:
        run_suffix_parts.append(f"a{int(off_alpha)}_{int(def_alpha)}")
    run_suffix = "_".join(run_suffix_parts)
    out_dir = RESULTS_DIR / f"clean_decomp_{year_range}_{run_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"weighted_factors_alt3_efg_value_{year_range}_{run_suffix}.csv"
    out.to_csv(path, index=False)
    logging.info("Wrote %s (%d players)", path, len(out))
    logging.info(
        "Display residual max: offense %.4f, defense %.4f, net %.4f",
        out["oRESID"].abs().max(),
        out["dRESID"].abs().max(),
        out["RESID"].abs().max(),
    )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    process = sub.add_parser("process", help="process raw WNBA CSVs into clean decomp CSVs")
    process.add_argument("raw_files", nargs="+", type=Path)

    run = sub.add_parser("run", help="run clean EFG-value weighted factors")
    run.add_argument("start_year", type=int)
    run.add_argument("end_year", type=int)
    run.add_argument("season_type", choices=["RS", "PS", "ALL"])
    run.add_argument("--time-decay-days", type=float)
    run.add_argument("--off-alpha", type=float, default=3000.0)
    run.add_argument("--def-alpha", type=float, default=3000.0)

    all_cmd = sub.add_parser("process-and-run", help="process raw files, then run weighted factors")
    all_cmd.add_argument("start_year", type=int)
    all_cmd.add_argument("end_year", type=int)
    all_cmd.add_argument("season_type", choices=["RS", "PS", "ALL"])
    all_cmd.add_argument("--time-decay-days", type=float)
    all_cmd.add_argument("--off-alpha", type=float, default=3000.0)
    all_cmd.add_argument("--def-alpha", type=float, default=3000.0)
    return parser.parse_args()


def raw_files_for_window(start_year: int, end_year: int, season_type: str) -> list[Path]:
    files: list[Path] = []
    for year in range(start_year, end_year + 1):
        if season_type in {"RS", "ALL"}:
            files.append(WNBA_ROOT / f"WNBA{year:02d}.csv")
        if season_type in {"PS", "ALL"}:
            files.append(WNBA_ROOT / f"WNBA{year:02d}_PS.csv")
    return files


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    if args.cmd == "process":
        for raw_file in args.raw_files:
            process_raw_file(raw_file)
        return 0
    if args.cmd == "process-and-run":
        for raw_file in raw_files_for_window(args.start_year, args.end_year, args.season_type):
            if raw_file.exists():
                process_raw_file(raw_file)
            else:
                logging.warning("Skipping missing raw file %s", raw_file)
        build_clean_weighted_factors(
            args.start_year,
            args.end_year,
            args.season_type,
            off_alpha=args.off_alpha,
            def_alpha=args.def_alpha,
            time_decay_days=args.time_decay_days,
        )
        return 0
    if args.cmd == "run":
        build_clean_weighted_factors(
            args.start_year,
            args.end_year,
            args.season_type,
            off_alpha=args.off_alpha,
            def_alpha=args.def_alpha,
            time_decay_days=args.time_decay_days,
        )
        return 0
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    sys.exit(main())
