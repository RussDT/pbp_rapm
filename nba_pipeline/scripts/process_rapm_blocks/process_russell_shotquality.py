"""
RUSSELL_SHOTQUALITY processor.

Possession-based RAPM target that substitutes repo-owned PBP shot-quality EV for
every field-goal attempt while preserving the standard possession denominator.

Default FGA value is ``shot_quality_season_ev`` from
``results/pbp_shot_quality/pbp_shot_quality_1997_2026.parquet``. That keeps the
RAPM target offense/shooter/context driven and avoids baking the defender
lineup residual into the defensive side of the same solve.
"""

from pathlib import Path
import os
import time

import numpy as np
import pandas as pd

from .common import (
    PIPELINE_ROOT,
    _base_processing,
    _finalize_df,
    case_when,
    fetch_player_stats_supabase,
    normalize_game_id_series,
    prepare_standard_possession_df,
    series_contains,
)


DEFAULT_SHOT_QUALITY_PATH = (
    PIPELINE_ROOT
    / "results"
    / "pbp_shot_quality"
    / "pbp_shot_quality_1997_2026.parquet"
)
DEFAULT_SHOT_QUALITY_COLUMN = "shot_quality_season_ev"


def _load_shot_quality_values(source_file, shot_quality_path, shot_quality_column):
    """Load per-FGA shot-quality values for a raw source file."""
    shot_quality_path = Path(shot_quality_path)
    if not shot_quality_path.exists():
        raise FileNotFoundError(f"Shot-quality parquet not found: {shot_quality_path}")

    required_cols = ["source_file", "game_id", "event_num", shot_quality_column]
    try:
        sq = pd.read_parquet(
            shot_quality_path,
            columns=required_cols,
            filters=[("source_file", "=", source_file)],
        )
    except Exception:
        sq = pd.read_parquet(shot_quality_path, columns=required_cols)
        sq = sq[sq["source_file"].eq(source_file)].copy()

    if sq.empty:
        raise ValueError(
            f"No shot-quality rows found for source_file={source_file} in {shot_quality_path}"
        )

    sq["game_id"] = normalize_game_id_series(sq["game_id"])
    sq["event_num"] = pd.to_numeric(sq["event_num"], errors="coerce").astype("Int64")
    sq[shot_quality_column] = pd.to_numeric(sq[shot_quality_column], errors="coerce")
    sq = sq.dropna(subset=["event_num", shot_quality_column])
    sq = sq.drop_duplicates(["game_id", "event_num"], keep="first")
    return sq[["game_id", "event_num", shot_quality_column]]


def process_russell_shotquality_py(
    file_path,
    year,
    season_str,
    shot_quality_path=DEFAULT_SHOT_QUALITY_PATH,
    shot_quality_column=DEFAULT_SHOT_QUALITY_COLUMN,
    metric_label="RUSSELL_SHOTQUALITY",
):
    """
    Build a standard-possession RAPM target with FGA points replaced by Russell SQ.

    Event values:
    - Free throws: shooter expected FT value, matching EV_RAPM-style processors.
    - Field-goal attempts: repo-owned shot-quality EV from the PBP SQ artifact.
    - Everything else, including turnovers: 0.
    """
    print(f"  Starting {metric_label} Processing for {season_str}...")
    nba_df = _base_processing(file_path)
    if nba_df is None:
        return None
    start_time = time.time()

    source_file = os.path.basename(file_path)
    sq = _load_shot_quality_values(source_file, shot_quality_path, shot_quality_column)
    print(
        f"      Loaded {len(sq):,} {shot_quality_column} rows for {source_file} "
        f"from {shot_quality_path}"
    )

    nba_df["event_num"] = pd.to_numeric(nba_df["event_num"], errors="coerce").astype("Int64")
    nba_df = nba_df.merge(sq, on=["game_id", "event_num"], how="left", validate="m:1")

    nba_df_checked, group_cols = prepare_standard_possession_df(
        nba_df, metric_label
    )

    is_playoffs = "_PS" in source_file.upper()
    player_stats_df = fetch_player_stats_supabase(year, is_playoffs)
    shooter_id_col = "player1_id"
    if player_stats_df is not None and shooter_id_col in nba_df_checked.columns:
        nba_df_checked["player1_id_numeric"] = pd.to_numeric(
            nba_df_checked[shooter_id_col], errors="coerce"
        )
        player_stats_df["PlayerID"] = pd.to_numeric(
            player_stats_df["PlayerID"], errors="coerce"
        ).astype("Int64")

        nba_df_checked = nba_df_checked.dropna(subset=["player1_id_numeric"])
        player_stats_df = player_stats_df.dropna(subset=["PlayerID"])
        nba_df_checked["player1_id_numeric"] = nba_df_checked[
            "player1_id_numeric"
        ].astype("Int64")

        nba_df_checked = pd.merge(
            nba_df_checked,
            player_stats_df[["PlayerID", "FTPerc"]],
            left_on="player1_id_numeric",
            right_on="PlayerID",
            how="left",
        )
        nba_df_checked = nba_df_checked.drop(
            columns=["player1_id_numeric", "PlayerID"], errors="ignore"
        )
        print(f"      Merged FT stats for {nba_df_checked['FTPerc'].notna().sum()} rows.")
    else:
        print("      Skipping player stats merge. Using default FT%.")
        if "FTPerc" not in nba_df_checked.columns:
            nba_df_checked["FTPerc"] = np.nan

    default_ft_perc = 0.75
    nba_df_checked["ExpFT"] = nba_df_checked["FTPerc"].fillna(default_ft_perc).astype(float)

    is_ft_event = (
        series_contains(nba_df_checked["home_description"], "Free Throw", case=False)
        | series_contains(nba_df_checked["visitor_description"], "Free Throw", case=False)
        | nba_df_checked["event_type"].eq("FreeThrow")
    )
    is_fga = nba_df_checked["event_type"].isin(["MAKE", "MISS"])

    fga_count = int(is_fga.sum())
    matched_fga_count = int((is_fga & nba_df_checked[shot_quality_column].notna()).sum())
    missing_fga_count = fga_count - matched_fga_count
    if missing_fga_count:
        fallback = float(nba_df_checked.loc[is_fga, shot_quality_column].mean())
        if not np.isfinite(fallback):
            fallback = 1.0
        print(
            f"      Warning: {missing_fga_count:,}/{fga_count:,} FGA rows missing "
            f"{shot_quality_column}; filling with sample mean {fallback:.4f}."
        )
    else:
        fallback = 1.0
        print(f"      Matched shot-quality values for all {fga_count:,} FGA rows.")

    nba_df_checked["ShotQuality_Event_EV"] = (
        pd.to_numeric(nba_df_checked[shot_quality_column], errors="coerce")
        .fillna(fallback)
        .astype(float)
    )
    nba_df_checked["ShotQualityNet"] = case_when(
        is_ft_event,
        nba_df_checked["ExpFT"],
        is_fga,
        nba_df_checked["ShotQuality_Event_EV"],
        0.0,
    )
    print(f"      Calculated ShotQualityNet (FTs=ExpFT, all FGAs={shot_quality_column})")

    if group_cols:
        nba_df_checked["ShotQualityTotal"] = nba_df_checked.groupby(group_cols)[
            "ShotQualityNet"
        ].cumsum()
    else:
        nba_df_checked["ShotQualityTotal"] = nba_df_checked["ShotQualityNet"].cumsum()

    if (
        "End_of_Possession" in nba_df_checked.columns
        and pd.api.types.is_bool_dtype(nba_df_checked["End_of_Possession"])
    ):
        nba_filt = nba_df_checked[nba_df_checked["End_of_Possession"]].copy()
    else:
        print("      Warning: End_of_Possession missing or not boolean.")
        nba_filt = nba_df_checked.copy()
    print(f"      Filtered to {len(nba_filt):,} {metric_label} possession rows")

    if not nba_filt.empty and "ShotQualityTotal" in nba_filt.columns:
        if group_cols:
            nba_filt["Off_Diff"] = nba_filt["ShotQualityTotal"] - nba_filt.groupby(
                group_cols
            )["ShotQualityTotal"].shift(fill_value=0)
        else:
            nba_filt["Off_Diff"] = nba_filt["ShotQualityTotal"] - nba_filt[
                "ShotQualityTotal"
            ].shift(fill_value=0)
        nba_filt["Net_Diff"] = nba_filt["Off_Diff"]
        nba_filt["Def_Diff"] = -nba_filt["Off_Diff"]
        print(f"      Calculated diffs for {metric_label}")

    output = _finalize_df(nba_filt, ["Net_Diff", "Off_Diff", "Def_Diff"], year)

    end_time = time.time()
    print(f"  Finished {metric_label} Processing. Time: {end_time - start_time:.2f} seconds.")
    return output


def process_context_shotquality_py(file_path, year, season_str):
    """Build a standard-possession RAPM target with context-only FGA EV."""
    return process_russell_shotquality_py(
        file_path,
        year,
        season_str,
        shot_quality_column="shot_context_ev",
        metric_label="CONTEXT_SHOTQUALITY",
    )
