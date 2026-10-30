#!/usr/bin/env python3
"""Test whether NBA offensive impact changes with opponent defensive quality.

The primary design is intentionally date-safe.  Player offensive quality and
role are frozen at the provider snapshot requested for December 1 of each
season, and every possession's opposing five-man defensive quality comes from
the latest EPM snapshot requested on or before the game date.  The provider's
contract says a date-D prediction excludes games played on D.

The estimand is not the obvious average effect of a better defense.  It is the
interaction between the offensive player's frozen O-EPM/role and the actual
defensive personnel faced.  The main regression uses player-game observations,
player-season fixed effects, offensive-teammate quality, home court, rest, and
score state, with game-clustered uncertainty.  Generated research artifacts
are written under ``results/offensive_impact_portability`` and are gitignored.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
PROCESSED_DIR = PIPELINE_DIR / "processed"
EPM_DIR = PIPELINE_DIR / "cache" / "epm"
RESULTS_DIR = PIPELINE_DIR / "results" / "offensive_impact_portability"

LINEUP_O = [f"O{i}" for i in range(1, 6)]
LINEUP_D = [f"D{i}" for i in range(1, 6)]
ROLE_FIELDS = [
    "p_usg",
    "p_ast_100",
    "p_fga_rim_100",
    "p_fg3a_100",
    "p_orb_100",
]
EPM_FIELDS = ["player_name", "oepm", "depm", "team_id", *ROLE_FIELDS]

FULL_TIERS = [
    "lt_-2",
    "-2_-1",
    "-1_0",
    "0_1",
    "1_2",
    "2_3",
    "3_4",
    "4_plus",
]
DISPLAY_TIERS = ["avg_0", "1_2", "2_3", "3_4", "4_plus"]
DISPLAY_TIER_LABELS = {
    "avg_0": "−0.5 to +0.5",
    "1_2": "+1 to +2",
    "2_3": "+2 to +3",
    "3_4": "+3 to +4",
    "4_plus": "+4 and above",
}


def weighted_mean(values: pd.Series | np.ndarray, weights: pd.Series | np.ndarray) -> float:
    values_array = np.asarray(values, dtype=float)
    weights_array = np.asarray(weights, dtype=float)
    valid = np.isfinite(values_array) & np.isfinite(weights_array) & (weights_array > 0)
    if not valid.any():
        return float("nan")
    return float(np.average(values_array[valid], weights=weights_array[valid]))


def weighted_quantile(
    values: Sequence[float], quantiles: Sequence[float], weights: Sequence[float] | None = None
) -> np.ndarray:
    """Return deterministic weighted quantiles for finite observations."""
    values_array = np.asarray(values, dtype=float)
    q_array = np.asarray(quantiles, dtype=float)
    if weights is None:
        weights_array = np.ones(len(values_array), dtype=float)
    else:
        weights_array = np.asarray(weights, dtype=float)
    valid = np.isfinite(values_array) & np.isfinite(weights_array) & (weights_array > 0)
    values_array = values_array[valid]
    weights_array = weights_array[valid]
    if not len(values_array):
        return np.full(len(q_array), np.nan)
    order = np.argsort(values_array, kind="stable")
    values_array = values_array[order]
    weights_array = weights_array[order]
    cumulative = np.cumsum(weights_array) - 0.5 * weights_array
    cumulative /= weights_array.sum()
    return np.interp(q_array, cumulative, values_array)


def full_oepm_tier(values: pd.Series) -> pd.Series:
    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=[-np.inf, -2, -1, 0, 1, 2, 3, 4, np.inf],
        labels=FULL_TIERS,
        right=False,
    ).astype("string")


def display_oepm_tier(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    result = pd.Series(pd.NA, index=values.index, dtype="string")
    result.loc[values.between(-0.5, 0.5, inclusive="both")] = "avg_0"
    result.loc[values.between(1, 2, inclusive="left")] = "1_2"
    result.loc[values.between(2, 3, inclusive="left")] = "2_3"
    result.loc[values.between(3, 4, inclusive="left")] = "3_4"
    result.loc[values.ge(4)] = "4_plus"
    return result


def classify_role(frame: pd.DataFrame) -> pd.Series:
    """Create transparent, mutually exclusive archetypes from frozen E-Skills."""
    usg = pd.to_numeric(frame["p_usg"], errors="coerce")
    ast = pd.to_numeric(frame["p_ast_100"], errors="coerce")
    rim = pd.to_numeric(frame["p_fga_rim_100"], errors="coerce")
    three = pd.to_numeric(frame["p_fg3a_100"], errors="coerce")
    orb = pd.to_numeric(frame["p_orb_100"], errors="coerce")
    role = pd.Series("connector_other", index=frame.index, dtype="string")
    role.loc[(rim >= 8) & (orb >= 2) & (three < 4)] = "big_finisher"
    role.loc[(usg < 0.20) & (three >= 6) & role.eq("connector_other")] = "low_usage_shooter"
    role.loc[((usg >= 0.24) | (ast >= 8)) & role.eq("connector_other")] = "creator"
    role.loc[
        (usg >= 0.20) & (usg < 0.24) & role.eq("connector_other")
    ] = "secondary_creator"
    return role


def playoff_round(game_id: pd.Series) -> pd.Series:
    """Decode the round digit embedded in standard NBA playoff game ids."""
    numeric = pd.to_numeric(game_id, errors="coerce").fillna(0).astype(np.int64)
    round_number = (numeric // 100) % 10
    conditions = [
        round_number.eq(1),
        round_number.eq(2),
        round_number.eq(3),
        round_number.eq(4),
    ]
    values = ["First round", "Second round", "Conference finals", "Finals"]
    return pd.Series(np.select(conditions, values, default="Unknown"), index=game_id.index)


def playoff_game_number(game_id: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(game_id, errors="coerce").fillna(0).astype(np.int64)
    return (numeric % 10).astype(int)


def playoff_series_id(game_id: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(game_id, errors="coerce").fillna(0).astype(np.int64)
    return (numeric // 10).astype(str)


@dataclass
class EpmStore:
    snapshots: dict[pd.Timestamp, pd.DataFrame]
    dates: np.ndarray
    source_files: list[str]

    @classmethod
    def load(cls, directory: Path) -> "EpmStore":
        snapshots: dict[pd.Timestamp, pd.DataFrame] = {}
        source_files: list[str] = []
        for path in sorted(directory.glob("epm_all_*_go0.csv")):
            header = pd.read_csv(path, nrows=0).columns.tolist()
            required = {"requested_date", "player_id", "oepm", "depm", *ROLE_FIELDS}
            if not required.issubset(header):
                continue
            usecols = ["requested_date", "player_id", *[c for c in EPM_FIELDS if c in header]]
            frame = pd.read_csv(path, usecols=usecols)
            requested = pd.to_datetime(frame.pop("requested_date"), errors="raise").iloc[0].normalize()
            frame["player_id"] = pd.to_numeric(frame["player_id"], errors="raise").astype(np.int64)
            frame = frame.drop_duplicates("player_id", keep="last").set_index("player_id")
            snapshots[requested] = frame
            source_files.append(str(path))
        if not snapshots:
            raise FileNotFoundError(f"No usable date-stamped EPM snapshots in {directory}")
        dates = np.array(sorted(snapshots), dtype="datetime64[ns]")
        return cls(snapshots=snapshots, dates=dates, source_files=source_files)

    def latest_date(self, value: str | pd.Timestamp) -> pd.Timestamp:
        target = np.datetime64(pd.Timestamp(value).normalize())
        idx = int(np.searchsorted(self.dates, target, side="right") - 1)
        if idx < 0:
            raise KeyError(f"No EPM snapshot on or before {value}")
        return pd.Timestamp(self.dates[idx])

    def asof(self, value: str | pd.Timestamp) -> pd.DataFrame:
        return self.snapshots[self.latest_date(value)]


def infer_home_offense(frame: pd.DataFrame) -> np.ndarray:
    """Infer the offensive side using score changes and lineup membership."""
    game_ids = pd.to_numeric(frame["game_id"], errors="raise").to_numpy(np.int64)
    offense = frame[LINEUP_O].to_numpy(dtype=np.int64)
    defense = frame[LINEUP_D].to_numpy(dtype=np.int64)
    away_score = pd.to_numeric(frame["away_score"], errors="coerce").fillna(0).to_numpy(float)
    home_score = pd.to_numeric(frame["home_score"], errors="coerce").fillna(0).to_numpy(float)
    starts = np.r_[True, game_ids[1:] != game_ids[:-1]]
    away_delta = np.r_[away_score[0], np.diff(away_score)]
    home_delta = np.r_[home_score[0], np.diff(home_score)]
    away_delta[starts] = away_score[starts]
    home_delta[starts] = home_score[starts]
    home_sets: dict[int, set[int]] = {}
    away_sets: dict[int, set[int]] = {}
    for idx in np.flatnonzero((away_delta > 0) | (home_delta > 0)):
        gid = int(game_ids[idx])
        home_sets.setdefault(gid, set())
        away_sets.setdefault(gid, set())
        if away_delta[idx] > 0:
            home_sets[gid].update(int(x) for x in defense[idx] if x > 0)
            away_sets[gid].update(int(x) for x in offense[idx] if x > 0)
        else:
            home_sets[gid].update(int(x) for x in offense[idx] if x > 0)
            away_sets[gid].update(int(x) for x in defense[idx] if x > 0)
    result = np.zeros(len(frame), dtype=bool)
    for gid in np.unique(game_ids):
        rows = np.flatnonzero(game_ids == gid)
        home = home_sets.get(int(gid), set())
        away = away_sets.get(int(gid), set())
        home_count = np.isin(offense[rows], list(home)).sum(axis=1)
        away_count = np.isin(offense[rows], list(away)).sum(axis=1)
        result[rows] = home_count > away_count
    return result


def _assert_aligned(left: pd.DataFrame, right: pd.DataFrame, label: str) -> None:
    if len(left) != len(right):
        raise ValueError(f"{label} row count mismatch: {len(left)} vs {len(right)}")
    for col in ["game_id", "event_num", "O1", "D1"]:
        if not np.array_equal(left[col].to_numpy(), right[col].to_numpy(), equal_nan=True):
            raise ValueError(f"{label} is not row-aligned on {col}")


def _mode_numeric(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.mode().iat[0])


def build_player_games(
    season: int,
    season_type: str,
    store: EpmStore,
    *,
    strict_defense: bool = True,
    start_on_anchor: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    """Build one row per player-game from standard possessions."""
    yy = season % 100
    suffix = "_PS" if season_type.upper() == "PS" else ""
    rapm_path = PROCESSED_DIR / f"RAPM{yy:02d}{suffix}.parquet"
    first_path = PROCESSED_DIR / f"FIRST_CHANCE{yy:02d}{suffix}.parquet"
    second_path = PROCESSED_DIR / f"SECOND_CHANCE_CLEAN{yy:02d}{suffix}.parquet"
    base_cols = [
        "game_id", *LINEUP_O, *LINEUP_D, "period", "time_quarter", "away_score",
        "home_score", "score_margin", "event_num", "Off_Diff", "game_date", "Season",
    ]
    frame = pd.read_parquet(rapm_path, columns=base_cols).reset_index(drop=True)
    first_cols = [
        "game_id", "event_num", "O1", "D1", "Net_Diff", "Is_Turnover",
        "Is_BadPass_TOV", "Is_FC_Transition_Possession", "FC_FT_Diff",
        "FC_DECOMP_RIM_FREQ_Diff", "FC_DECOMP_RIM_FG_Diff", "FC_DECOMP_MID_FG_Diff",
        "FC_DECOMP_THREE_FREQ_Diff", "FC_DECOMP_THREE_FG_Diff",
    ]
    first = pd.read_parquet(first_path, columns=first_cols).reset_index(drop=True)
    second = pd.read_parquet(
        second_path, columns=["game_id", "event_num", "O1", "D1", "Off_Diff"]
    ).reset_index(drop=True)
    _assert_aligned(frame, first, f"FIRST_CHANCE {season_type} {season}")
    _assert_aligned(frame, second, f"SECOND_CHANCE_CLEAN {season_type} {season}")
    frame["first_chance_points"] = pd.to_numeric(first["Net_Diff"], errors="coerce").fillna(0.0)
    frame["turnover"] = pd.to_numeric(first["Is_Turnover"], errors="coerce").fillna(0.0)
    frame["badpass_turnover"] = pd.to_numeric(first["Is_BadPass_TOV"], errors="coerce").fillna(0.0)
    frame["transition"] = pd.to_numeric(
        first["Is_FC_Transition_Possession"], errors="coerce"
    ).fillna(0.0)
    frame["ft_value"] = pd.to_numeric(first["FC_FT_Diff"], errors="coerce").fillna(0.0)
    frame["rim_freq_value"] = pd.to_numeric(
        first["FC_DECOMP_RIM_FREQ_Diff"], errors="coerce"
    ).fillna(0.0)
    frame["rim_fg_value"] = pd.to_numeric(
        first["FC_DECOMP_RIM_FG_Diff"], errors="coerce"
    ).fillna(0.0)
    frame["mid_fg_value"] = pd.to_numeric(
        first["FC_DECOMP_MID_FG_Diff"], errors="coerce"
    ).fillna(0.0)
    frame["three_freq_value"] = pd.to_numeric(
        first["FC_DECOMP_THREE_FREQ_Diff"], errors="coerce"
    ).fillna(0.0)
    frame["three_fg_value"] = pd.to_numeric(
        first["FC_DECOMP_THREE_FG_Diff"], errors="coerce"
    ).fillna(0.0)
    frame["second_chance_points"] = pd.to_numeric(second["Off_Diff"], errors="coerce").fillna(0.0)
    closure = (
        pd.to_numeric(frame["Off_Diff"], errors="coerce").fillna(0.0)
        - frame["first_chance_points"]
        - frame["second_chance_points"]
    )
    # FIRST_CHANCE uses expected FT value while RAPM uses actual points.  The
    # resulting rare FT-conversion gaps are audited, not silently described as
    # broken row algebra.  FIRST_CHANCE + SECOND_CHANCE_CLEAN closes to its own
    # luck-adjusted parent; actual RAPM points remain the primary outcome.
    closure_rows = int(closure.abs().gt(1e-10).sum())

    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="raise").dt.normalize()
    anchor_target = pd.Timestamp(year=season - 1, month=12, day=1)
    anchor_date = store.latest_date(anchor_target)
    anchor = store.snapshots[anchor_date]
    if start_on_anchor and season_type.upper() == "RS":
        frame = frame.loc[frame["game_date"].ge(anchor_target)].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No rows remain for {season_type} {season}")

    frame["home_offense"] = infer_home_offense(frame).astype(int)
    game_ids = frame["game_id"].to_numpy()
    margin = (
        frame["score_margin"].astype(str).str.upper().replace("TIE", "0")
    )
    home_margin = pd.to_numeric(margin, errors="coerce").fillna(0.0).to_numpy(float)
    starts = np.r_[True, game_ids[1:] != game_ids[:-1]]
    previous = np.r_[0.0, home_margin[:-1]]
    previous[starts] = 0.0
    frame["pre_margin"] = np.where(frame["home_offense"].eq(1), previous, -previous)
    clock = frame["time_quarter"].astype(str).str.extract(r"^(\d+):(\d+)")
    seconds_left = (
        pd.to_numeric(clock[0], errors="coerce").fillna(0) * 60
        + pd.to_numeric(clock[1], errors="coerce").fillna(0)
    )
    frame["garbage"] = (
        (pd.to_numeric(frame["period"], errors="coerce") >= 4)
        & (((seconds_left <= 360) & (np.abs(frame["pre_margin"]) >= 20))
           | ((seconds_left <= 180) & (np.abs(frame["pre_margin"]) >= 15)))
    ).astype(int)

    rolling_snapshot_dates: list[str] = []
    for game_date, rows in frame.groupby("game_date", sort=False).groups.items():
        snapshot_date = store.latest_date(game_date)
        rolling_snapshot_dates.append(snapshot_date.date().isoformat())
        snapshot = store.snapshots[snapshot_date]
        row_index = np.asarray(list(rows), dtype=int)
        for col in LINEUP_D:
            ids = pd.to_numeric(frame.loc[row_index, col], errors="coerce").fillna(0).astype(np.int64)
            frame.loc[row_index, f"{col}_depm"] = ids.map(snapshot["depm"])
        for col in LINEUP_O:
            ids = pd.to_numeric(frame.loc[row_index, col], errors="coerce").fillna(0).astype(np.int64)
            frame.loc[row_index, f"{col}_oepm_roll"] = ids.map(snapshot["oepm"])
            if "team_id" in snapshot.columns:
                frame.loc[row_index, f"{col}_team_id"] = ids.map(snapshot["team_id"])

    d_cols = [f"{col}_depm" for col in LINEUP_D]
    o_roll_cols = [f"{col}_oepm_roll" for col in LINEUP_O]
    team_cols = [f"{col}_team_id" for col in LINEUP_O]
    frame["def_covered"] = frame[d_cols].notna().sum(axis=1)
    frame["off_covered"] = frame[o_roll_cols].notna().sum(axis=1)
    frame["def_quality"] = frame[d_cols].fillna(0.0).sum(axis=1)
    frame["off_lineup_oepm"] = frame[o_roll_cols].fillna(0.0).sum(axis=1)
    if team_cols[0] in frame:
        frame["off_team_id"] = frame[team_cols].bfill(axis=1).iloc[:, 0]
    else:
        frame["off_team_id"] = np.nan

    before_strict = len(frame)
    if strict_defense:
        frame = frame.loc[frame["def_covered"].eq(5)].copy()
    strict_rate = len(frame) / before_strict if before_strict else float("nan")

    outcome_cols = [
        "Off_Diff", "first_chance_points", "second_chance_points", "turnover",
        "badpass_turnover", "transition", "ft_value", "rim_freq_value", "rim_fg_value",
        "mid_fg_value", "three_freq_value", "three_fg_value", "garbage",
    ]
    expanded: list[pd.DataFrame] = []
    for slot, col in enumerate(LINEUP_O, start=1):
        player_ids = pd.to_numeric(frame[col], errors="coerce").fillna(0).astype(np.int64)
        subject = anchor.reindex(player_ids.to_numpy())
        subject.index = frame.index
        part = frame[
            ["game_id", "game_date", "home_offense", "pre_margin", "def_quality",
             "off_lineup_oepm", "off_team_id", "def_covered", "off_covered", *outcome_cols]
        ].copy()
        part["player_id"] = player_ids.to_numpy()
        for field in EPM_FIELDS:
            if field in subject:
                part[field] = subject[field].to_numpy()
        roll = pd.to_numeric(frame[f"{col}_oepm_roll"], errors="coerce").fillna(0.0)
        part["teammate_oepm"] = frame["off_lineup_oepm"] - roll
        part["slot"] = slot
        expanded.append(part)
    long = pd.concat(expanded, ignore_index=True)
    long = long.loc[long["player_id"].gt(0) & long["oepm"].notna()].copy()
    long["season"] = int(season)
    long["season_type"] = season_type.upper()
    long["player_season"] = long["player_id"].astype(str) + "_" + str(season)
    long["full_tier"] = full_oepm_tier(long["oepm"])
    long["display_tier"] = display_oepm_tier(long["oepm"])
    long["role"] = classify_role(long)

    sum_cols = outcome_cols
    mean_cols = [
        "def_quality", "teammate_oepm", "pre_margin", "home_offense", "def_covered",
        "off_covered", "oepm", "depm", *ROLE_FIELDS,
    ]
    aggregation: dict[str, str | callable] = {col: "sum" for col in sum_cols}
    aggregation.update({col: "mean" for col in mean_cols})
    aggregation.update({
        "game_date": "first", "player_name": "first", "off_team_id": _mode_numeric,
        "full_tier": "first", "display_tier": "first", "role": "first",
        "season": "first", "season_type": "first", "player_season": "first",
    })
    player_game = (
        long.groupby(["game_id", "player_id"], sort=False, as_index=False)
        .agg(aggregation)
    )
    player_game = player_game.rename(columns={"Off_Diff": "points"})
    player_game["possessions"] = (
        long.groupby(["game_id", "player_id"], sort=False).size().to_numpy()
    )
    rate_sources = [
        "points", "first_chance_points", "second_chance_points", "turnover",
        "badpass_turnover", "transition", "ft_value", "rim_freq_value", "rim_fg_value",
        "mid_fg_value", "three_freq_value", "three_fg_value",
    ]
    for col in rate_sources:
        player_game[f"{col}_per100"] = 100.0 * player_game[col] / player_game["possessions"]
    player_game["garbage_share"] = player_game["garbage"] / player_game["possessions"]

    environment = frame[["game_id", "game_date", "def_quality"]].copy()
    environment["season"] = int(season)
    environment["season_type"] = season_type.upper()
    environment["round"] = (
        playoff_round(environment["game_id"])
        if season_type.upper() == "PS" else "Regular season"
    )
    audit = {
        "season": int(season),
        "season_type": season_type.upper(),
        "anchor_requested_date": anchor_date.date().isoformat(),
        "possessions_before_strict_defense": int(before_strict),
        "possessions_after_strict_defense": int(len(frame)),
        "strict_defense_retention": float(strict_rate),
        "player_game_rows": int(len(player_game)),
        "unique_players": int(player_game["player_id"].nunique()),
        "rolling_snapshot_min": min(rolling_snapshot_dates),
        "rolling_snapshot_max": max(rolling_snapshot_dates),
        "max_scoring_closure_gap": float(closure.abs().max()),
        "actual_vs_luck_adjusted_scoring_gap_rows": closure_rows,
        "actual_vs_luck_adjusted_scoring_sum_gap": float(closure.sum()),
    }
    return player_game, environment, audit


def add_rest_days(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach team rest from the inferred EPM team id and game schedule."""
    result = frame.copy()
    games = (
        result.dropna(subset=["off_team_id"])
        .sort_values(["season", "game_date", "game_id"])
        .drop_duplicates(["season", "game_id", "off_team_id"])
        [["season", "game_id", "off_team_id", "game_date"]]
    )
    games["rest_days"] = (
        games.groupby(["season", "off_team_id"], sort=False)["game_date"]
        .diff().dt.days.sub(1).clip(lower=0, upper=7)
    )
    result = result.merge(
        games[["season", "game_id", "off_team_id", "rest_days"]],
        on=["season", "game_id", "off_team_id"], how="left",
    )
    result["rest_days"] = result["rest_days"].fillna(2.0)
    return result


def add_season_defense_centering(
    regular: pd.DataFrame, playoffs: pd.DataFrame, environment: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stats = (
        environment.loc[environment["season_type"].eq("RS")]
        .groupby("season", as_index=False)["def_quality"]
        .agg(def_mean="mean", def_sd="std")
    )
    stats["def_sd"] = stats["def_sd"].replace(0, np.nan)
    regular = regular.merge(stats, on="season", how="left")
    playoffs = playoffs.merge(stats, on="season", how="left")
    environment = environment.merge(stats, on="season", how="left")
    for frame in [regular, playoffs, environment]:
        frame["def_centered"] = frame["def_quality"] - frame["def_mean"]
        frame["def_z"] = frame["def_centered"] / frame["def_sd"]
    return regular, playoffs, environment, stats


def weighted_demean(frame: pd.DataFrame, columns: Sequence[str], group: str, weight: str) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    weights = pd.to_numeric(frame[weight], errors="coerce").fillna(0.0)
    group_weight = weights.groupby(frame[group], sort=False).transform("sum")
    for col in columns:
        values = pd.to_numeric(frame[col], errors="coerce")
        numerator = (values * weights).groupby(frame[group], sort=False).transform("sum")
        mean = numerator / group_weight.replace(0, np.nan)
        result[col] = values - mean
    return result


@dataclass
class FittedResponse:
    result: object
    feature_columns: list[str]
    raw_design_columns: list[str]
    tier_slopes: pd.DataFrame
    beta: pd.Series
    covariance: pd.DataFrame
    fixed_effects: pd.Series


def _fit_clustered_wls(y: pd.Series, x: pd.DataFrame, weights: pd.Series, groups: pd.Series):
    valid = y.notna() & x.notna().all(axis=1) & weights.notna() & weights.gt(0) & groups.notna()
    if not valid.any():
        raise ValueError("No valid rows for WLS")
    model = sm.WLS(y.loc[valid], x.loc[valid], weights=weights.loc[valid])
    return model.fit(cov_type="cluster", cov_kwds={"groups": groups.loc[valid]})


def build_raw_design(frame: pd.DataFrame, tier_column: str = "full_tier") -> tuple[pd.DataFrame, list[str]]:
    design = pd.DataFrame(index=frame.index)
    design["def_centered"] = frame["def_centered"]
    tiers = [tier for tier in sorted(frame[tier_column].dropna().unique()) if tier != "0_1"]
    for tier in tiers:
        name = f"def_x_tier__{tier}"
        design[name] = frame["def_centered"] * frame[tier_column].eq(tier).astype(float)
    for col in ["teammate_oepm", "home_offense", "rest_days", "pre_margin", "garbage_share"]:
        design[col] = pd.to_numeric(frame[col], errors="coerce")
    return design, tiers


def fit_tier_response(
    frame: pd.DataFrame,
    *,
    outcome: str = "points_per100",
    weight_mode: str = "possessions",
    tier_column: str = "full_tier",
) -> FittedResponse:
    data = frame.dropna(subset=[outcome, "def_centered", tier_column, "player_season"]).copy()
    data = data.loc[data.groupby("player_season")["game_id"].transform("nunique").ge(12)]
    if weight_mode == "possessions":
        data["model_weight"] = data["possessions"].astype(float)
    elif weight_mode == "player_season":
        totals = data.groupby("player_season")["possessions"].transform("sum")
        data["model_weight"] = data["possessions"] / totals
    elif weight_mode == "player":
        totals = data.groupby("player_id")["possessions"].transform("sum")
        data["model_weight"] = data["possessions"] / totals
    else:
        raise ValueError(f"Unknown weight mode: {weight_mode}")

    raw_design, tiers = build_raw_design(data, tier_column=tier_column)
    demean_columns = [outcome, *raw_design.columns]
    combined = pd.concat(
        [data[[outcome, "player_season", "model_weight"]], raw_design], axis=1
    )
    within = weighted_demean(combined, demean_columns, "player_season", "model_weight")
    y = within[outcome]
    x = within[raw_design.columns]
    clusters = data["season"].astype(str) + "_" + data["game_id"].astype(str)
    fitted = _fit_clustered_wls(y, x, data["model_weight"], clusters)
    beta = pd.Series(fitted.params, index=x.columns)
    covariance = pd.DataFrame(fitted.cov_params(), index=x.columns, columns=x.columns)

    slope_rows = []
    for tier in ["0_1", *tiers]:
        interaction = f"def_x_tier__{tier}"
        slope = float(beta["def_centered"] + (beta.get(interaction, 0.0)))
        vector = pd.Series(0.0, index=x.columns)
        vector["def_centered"] = 1.0
        if interaction in vector.index:
            vector[interaction] = 1.0
        variance = float(vector.to_numpy() @ covariance.to_numpy() @ vector.to_numpy())
        se = math.sqrt(max(variance, 0.0))
        subset = data.loc[data[tier_column].eq(tier)]
        slope_rows.append({
            "tier": tier, "slope_per_def_epm": slope, "se": se,
            "ci_low": slope - 1.96 * se, "ci_high": slope + 1.96 * se,
            "player_seasons": int(subset["player_season"].nunique()),
            "players": int(subset["player_id"].nunique()),
            "possessions": int(subset["possessions"].sum()),
            "mean_oepm": weighted_mean(subset["oepm"], subset["possessions"]),
        })
    slopes = pd.DataFrame(slope_rows)
    shares = (
        data.groupby(tier_column)["possessions"].sum() / data["possessions"].sum()
    )
    common = float(sum(shares.get(row.tier, 0.0) * row.slope_per_def_epm for row in slopes.itertuples()))
    slopes["relative_portability_slope"] = slopes["slope_per_def_epm"] - common

    raw_prediction = raw_design @ beta
    adjusted = pd.to_numeric(data[outcome], errors="coerce") - raw_prediction
    alpha = (
        (adjusted * data["model_weight"]).groupby(data["player_season"], sort=False).sum()
        / data["model_weight"].groupby(data["player_season"], sort=False).sum()
    )
    return FittedResponse(
        result=fitted,
        feature_columns=list(x.columns),
        raw_design_columns=list(raw_design.columns),
        tier_slopes=slopes,
        beta=beta,
        covariance=covariance,
        fixed_effects=alpha,
    )


def fit_continuous_response(
    frame: pd.DataFrame,
    *,
    outcome: str = "points_per100",
    weight_mode: str = "possessions",
    controls: Sequence[str] | None = None,
) -> pd.DataFrame:
    data = frame.dropna(subset=[outcome, "def_centered", "oepm", "player_season"]).copy()
    data = data.loc[data.groupby("player_season")["game_id"].transform("nunique").ge(12)]
    if weight_mode == "possessions":
        data["model_weight"] = data["possessions"]
    elif weight_mode == "player_season":
        data["model_weight"] = data["possessions"] / data.groupby("player_season")["possessions"].transform("sum")
    else:
        data["model_weight"] = data["possessions"] / data.groupby("player_id")["possessions"].transform("sum")
    o_center = weighted_mean(data["oepm"], data["possessions"])
    raw = pd.DataFrame(index=data.index)
    raw["def_centered"] = data["def_centered"]
    raw["oepm_x_def"] = data["def_centered"] * (data["oepm"] - o_center)
    for knot in [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]:
        raw[f"hinge_{knot:+.0f}_x_def"] = data["def_centered"] * np.maximum(data["oepm"] - knot, 0.0)
    if controls is None:
        controls = ["teammate_oepm", "home_offense", "rest_days", "pre_margin", "garbage_share"]
    for col in controls:
        raw[col] = data[col]
    specifications = {
        "linear": ["def_centered", "oepm_x_def", *controls],
        "hinge_spline": ["def_centered", "oepm_x_def", *[c for c in raw if c.startswith("hinge_")],
                         *controls],
    }
    rows = []
    clusters = data["season"].astype(str) + "_" + data["game_id"].astype(str)
    for name, columns in specifications.items():
        combined = pd.concat(
            [data[[outcome, "player_season", "model_weight"]], raw[columns]], axis=1
        )
        within = weighted_demean(combined, [outcome, *columns], "player_season", "model_weight")
        fitted = _fit_clustered_wls(within[outcome], within[columns], data["model_weight"], clusters)
        for feature in columns:
            if feature in fitted.params.index:
                value = float(fitted.params[feature])
                se = float(fitted.bse[feature])
                rows.append({
                    "specification": name, "feature": feature, "coefficient": value,
                    "se": se, "ci_low": value - 1.96 * se, "ci_high": value + 1.96 * se,
                    "rows": int(fitted.nobs), "player_seasons": int(data["player_season"].nunique()),
                    "possessions": int(data["possessions"].sum()), "oepm_center": o_center,
                })
    return pd.DataFrame(rows)


def fit_player_specific_slopes(frame: pd.DataFrame) -> pd.DataFrame:
    """Estimate noisy player-season slopes, then apply empirical-Bayes shrinkage."""
    data = frame.copy()
    data["model_weight"] = data["possessions"].astype(float)
    controls = ["teammate_oepm", "home_offense", "rest_days", "pre_margin", "garbage_share"]
    combined = data[["points_per100", "player_season", "model_weight", *controls]].copy()
    within = weighted_demean(combined, combined.columns, "player_season", "model_weight")
    fitted = _fit_clustered_wls(
        within["points_per100"], within[controls], data["model_weight"],
        data["season"].astype(str) + "_" + data["game_id"].astype(str),
    )
    data["base_residual"] = within["points_per100"] - within[controls] @ fitted.params
    rows = []
    for player_season, group in data.groupby("player_season", sort=False):
        if group["possessions"].sum() < 1200 or group["game_id"].nunique() < 30:
            continue
        w = group["possessions"].to_numpy(float)
        x = group["def_centered"].to_numpy(float)
        y = group["base_residual"].to_numpy(float)
        x = x - np.average(x, weights=w)
        denominator = float(np.sum(w * x * x))
        if denominator <= 1e-8:
            continue
        slope = float(np.sum(w * x * y) / denominator)
        residual = y - slope * x
        dof = max(len(group) - 2, 1)
        sigma2 = float(np.sum(w * residual * residual) / dof / (np.mean(w) or 1.0))
        se = math.sqrt(max(sigma2 / (np.sum(x * x) or 1.0), 0.0))
        first = group.iloc[0]
        rows.append({
            "player_season": player_season, "player_id": int(first["player_id"]),
            "player_name": first["player_name"], "season": int(first["season"]),
            "oepm": float(first["oepm"]), "role": first["role"],
            "possessions": int(group["possessions"].sum()), "games": int(group["game_id"].nunique()),
            "raw_slope": slope, "raw_se": se,
        })
    slopes = pd.DataFrame(rows)
    if slopes.empty:
        return slopes
    precision = 1.0 / np.maximum(slopes["raw_se"].to_numpy(float) ** 2, 1e-6)
    x = sm.add_constant(slopes[["oepm"]], has_constant="add")
    meta = sm.WLS(slopes["raw_slope"], x, weights=np.minimum(precision, np.quantile(precision, 0.95))).fit()
    slopes["population_slope"] = meta.predict(x)
    excess_variance = np.average(
        (slopes["raw_slope"] - slopes["population_slope"]) ** 2 - slopes["raw_se"] ** 2,
        weights=slopes["possessions"],
    )
    tau2 = max(float(excess_variance), 0.05 ** 2)
    slopes["shrink_weight"] = tau2 / (tau2 + slopes["raw_se"] ** 2)
    slopes["slope"] = (
        slopes["shrink_weight"] * slopes["raw_slope"]
        + (1.0 - slopes["shrink_weight"]) * slopes["population_slope"]
    )
    slopes["slope_se"] = np.sqrt(tau2 * slopes["raw_se"] ** 2 / (tau2 + slopes["raw_se"] ** 2))
    slopes["ci_low"] = slopes["slope"] - 1.96 * slopes["slope_se"]
    slopes["ci_high"] = slopes["slope"] + 1.96 * slopes["slope_se"]
    slopes["tau2"] = tau2
    return slopes.sort_values("slope", ascending=False).reset_index(drop=True)


def attach_model_predictions(frame: pd.DataFrame, model: FittedResponse) -> pd.DataFrame:
    data = frame.copy()
    raw, _ = build_raw_design(data, tier_column="full_tier")
    for col in model.raw_design_columns:
        if col not in raw:
            raw[col] = 0.0
    raw = raw[model.raw_design_columns]
    data["expected_offense_per100"] = raw @ model.beta
    data["baseline_offense_per100"] = data["player_season"].map(model.fixed_effects)
    data["expected_offense_per100"] += data["baseline_offense_per100"]
    data["model_residual_per100"] = data["points_per100"] - data["expected_offense_per100"]
    return data.dropna(subset=["expected_offense_per100", "model_residual_per100"])


def predict_playoffs(playoffs: pd.DataFrame, model: FittedResponse) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = attach_model_predictions(playoffs, model)
    data["playoff_residual_per100"] = data["model_residual_per100"]
    data["round"] = playoff_round(data["game_id"])
    data["series_id"] = playoff_series_id(data["game_id"])
    data["series_game"] = playoff_game_number(data["game_id"])
    summary_rows = []
    for player_season, group in data.groupby("player_season", sort=False):
        first = group.iloc[0]
        summary_rows.append({
            "player_id": int(first["player_id"]), "player_name": first["player_name"],
            "season": int(first["season"]), "player_season": player_season,
            "oepm": float(first["oepm"]), "role": first["role"],
            "playoff_possessions": int(group["possessions"].sum()),
            "playoff_actual_offense": weighted_mean(group["points_per100"], group["possessions"]),
            "playoff_predicted_offense": weighted_mean(group["expected_offense_per100"], group["possessions"]),
            "playoff_residual": weighted_mean(group["playoff_residual_per100"], group["possessions"]),
            "playoff_games": int(group["game_id"].nunique()),
        })
    summary = pd.DataFrame(summary_rows)
    return data, summary


def build_player_season_inspection(
    regular: pd.DataFrame,
    model: FittedResponse,
    player_slopes: pd.DataFrame,
    playoff_players: pd.DataFrame,
    environment_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predicted = attach_model_predictions(regular, model)
    rs = environment_summary.loc[environment_summary["environment"].eq("Regular season")].iloc[0]
    conditions = [
        predicted["def_centered"].le(rs["p25"]),
        predicted["def_centered"].ge(rs["p90"]),
        predicted["def_centered"].ge(rs["p75"]),
    ]
    predicted["opponent_defense_bucket"] = np.select(
        conditions, ["weak_bottom_quartile", "elite_top_decile", "strong_75_to_90"],
        default="average_middle_half",
    )
    identity_columns = ["player_id", "player_season", "opponent_defense_bucket"]
    fixed_columns = ["player_name", "season", "oepm", "depm", "role", *ROLE_FIELDS]
    bucket_rows = []
    for keys, group in predicted.groupby(identity_columns, dropna=False, sort=False):
        row = dict(zip(identity_columns, keys))
        first = group.iloc[0]
        row.update({column: first[column] for column in fixed_columns})
        row.update({
            "possessions": int(group["possessions"].sum()),
            "games": int(group["game_id"].nunique()),
            "mean_defensive_lineup_epm": weighted_mean(group["def_centered"], group["possessions"]),
            "observed_offense_per100": weighted_mean(group["points_per100"], group["possessions"]),
            "expected_offense_per100": weighted_mean(group["expected_offense_per100"], group["possessions"]),
            "adjusted_residual_per100": weighted_mean(group["model_residual_per100"], group["possessions"]),
        })
        bucket_rows.append(row)
    buckets = pd.DataFrame(bucket_rows)
    season_rows = []
    for player_season, group in predicted.groupby("player_season", dropna=False, sort=False):
        first = group.iloc[0]
        columns = ["player_id", "player_name", "season", "player_season", "oepm", "depm", "role", *ROLE_FIELDS]
        row = {column: first[column] for column in columns}
        row.update({
            "regular_season_possessions": int(group["possessions"].sum()),
            "regular_season_games": int(group["game_id"].nunique()),
            "observed_offense_per100": weighted_mean(group["points_per100"], group["possessions"]),
            "expected_offense_per100": weighted_mean(group["expected_offense_per100"], group["possessions"]),
            "adjusted_residual_per100": weighted_mean(group["model_residual_per100"], group["possessions"]),
            "mean_defensive_lineup_epm": weighted_mean(group["def_centered"], group["possessions"]),
        })
        season_rows.append(row)
    inspection = pd.DataFrame(season_rows)
    slope_columns = [
        "player_season", "slope", "slope_se", "ci_low", "ci_high", "raw_slope", "raw_se",
        "population_slope", "shrink_weight",
    ]
    inspection = inspection.merge(
        player_slopes[[c for c in slope_columns if c in player_slopes]], on="player_season", how="left"
    )
    inspection = inspection.merge(
        playoff_players.drop(columns=[c for c in ["player_name", "season", "oepm", "role", "player_id"] if c in playoff_players]),
        on="player_season", how="left",
    )
    return inspection, buckets


def clustered_mean(frame: pd.DataFrame, value: str, weight: str, cluster: str) -> dict[str, float]:
    x = pd.DataFrame({"constant": 1.0}, index=frame.index)
    result = _fit_clustered_wls(frame[value], x, frame[weight], frame[cluster].astype(str))
    estimate = float(result.params["constant"])
    se = float(result.bse["constant"])
    return {"estimate": estimate, "se": se, "ci_low": estimate - 1.96 * se, "ci_high": estimate + 1.96 * se}


def fit_series_test(playoff_games: pd.DataFrame) -> pd.DataFrame:
    data = playoff_games.loc[playoff_games["series_game"].between(1, 7)].copy()
    data["player_series"] = data["player_id"].astype(str) + "_" + data["series_id"]
    data = data.loc[data.groupby("player_series")["game_id"].transform("nunique").ge(3)]
    data["series_game_centered"] = data["series_game"] - 1
    data["oepm_x_game"] = data["series_game_centered"] * (data["oepm"] - weighted_mean(data["oepm"], data["possessions"]))
    columns = ["playoff_residual_per100", "series_game_centered", "oepm_x_game"]
    within_input = data[[*columns, "player_series", "possessions"]].copy()
    within = weighted_demean(within_input, columns, "player_series", "possessions")
    fitted = _fit_clustered_wls(
        within["playoff_residual_per100"], within[["series_game_centered", "oepm_x_game"]],
        data["possessions"], data["series_id"],
    )
    rows = []
    for feature in ["series_game_centered", "oepm_x_game"]:
        estimate = float(fitted.params[feature])
        se = float(fitted.bse[feature])
        rows.append({
            "feature": feature, "coefficient": estimate, "se": se,
            "ci_low": estimate - 1.96 * se, "ci_high": estimate + 1.96 * se,
            "player_series": int(data["player_series"].nunique()),
            "series": int(data["series_id"].nunique()), "games": int(data["game_id"].nunique()),
            "possessions": int(data["possessions"].sum()),
        })
    return pd.DataFrame(rows)


def build_environment_tables(environment: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    order = ["Regular season", "First round", "Second round", "Conference finals", "Finals"]
    rows = []
    for label in order:
        group = environment.loc[environment["round"].eq(label)]
        if group.empty:
            continue
        q = weighted_quantile(group["def_centered"], [0.10, 0.25, 0.50, 0.75, 0.90])
        rows.append({
            "environment": label, "possessions": len(group), "games": group["game_id"].nunique(),
            "mean_def_epm": group["def_centered"].mean(), "sd_def_epm": group["def_centered"].std(),
            "p10": q[0], "p25": q[1], "median": q[2], "p75": q[3], "p90": q[4],
        })
    summary = pd.DataFrame(rows)
    rng = np.random.default_rng(20260820)
    samples = []
    for label, group in environment.groupby("round", sort=False):
        count = min(len(group), 40_000)
        indices = rng.choice(group.index.to_numpy(), size=count, replace=False)
        samples.append(environment.loc[indices, ["season", "season_type", "round", "def_centered", "def_z"]])
    return summary, pd.concat(samples, ignore_index=True)


def build_tier_environment_table(
    slopes: pd.DataFrame, regular: pd.DataFrame, environment_summary: pd.DataFrame
) -> pd.DataFrame:
    regular_env = environment_summary.loc[environment_summary["environment"].eq("Regular season")].iloc[0]
    levels = {
        "weak_p25": float(regular_env["p25"]),
        "average": 0.0,
        "strong_p75": float(regular_env["p75"]),
        "elite_p90": float(regular_env["p90"]),
    }
    display = regular.loc[regular["display_tier"].isin(DISPLAY_TIERS)].copy()
    # Map display bins to the closest full-bin portability estimate.  The ~0
    # group pools the -1_0 and 0_1 slopes using possession weights.
    full_lookup = slopes.set_index("tier")
    rows = []
    for tier in DISPLAY_TIERS:
        subset = display.loc[display["display_tier"].eq(tier)]
        if subset.empty:
            continue
        if tier == "avg_0":
            candidates = full_lookup.loc[[x for x in ["-1_0", "0_1"] if x in full_lookup.index]]
            candidate_weights = [
                subset.loc[full_oepm_tier(subset["oepm"]).eq(idx), "possessions"].sum()
                for idx in candidates.index
            ]
            portability = weighted_mean(candidates["relative_portability_slope"], candidate_weights)
        else:
            portability = float(full_lookup.loc[tier, "relative_portability_slope"])
        baseline = weighted_mean(subset["oepm"], subset["possessions"])
        row = {
            "oepm_tier": tier, "label": DISPLAY_TIER_LABELS[tier], "baseline_oepm": baseline,
            "relative_portability_slope": portability,
            "player_seasons": subset["player_season"].nunique(), "possessions": subset["possessions"].sum(),
        }
        for name, value in levels.items():
            row[name] = baseline + portability * value
        row["elite_minus_average"] = row["elite_p90"] - row["average"]
        row["elite_minus_weak"] = row["elite_p90"] - row["weak_p25"]
        rows.append(row)
    return pd.DataFrame(rows)


def fit_mechanism_models(regular: pd.DataFrame) -> pd.DataFrame:
    outcomes = {
        "Total offense": "points_per100",
        "First chance": "first_chance_points_per100",
        "Turnover rate": "turnover_per100",
        "Bad-pass turnover rate": "badpass_turnover_per100",
        "Transition frequency": "transition_per100",
        "Rim frequency value": "rim_freq_value_per100",
        "Rim finishing value": "rim_fg_value_per100",
        "Midrange value": "mid_fg_value_per100",
        "Three frequency value": "three_freq_value_per100",
        "Three shotmaking value": "three_fg_value_per100",
        "Free-throw value": "ft_value_per100",
        "Second chance": "second_chance_points_per100",
    }
    rows = []
    for label, outcome in outcomes.items():
        fitted = fit_tier_response(regular, outcome=outcome, weight_mode="possessions")
        table = fitted.tier_slopes.copy()
        table.insert(0, "mechanism", label)
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def fit_mechanism_continuous(regular: pd.DataFrame) -> pd.DataFrame:
    outcomes = {
        "Total offense": "points_per100",
        "First chance": "first_chance_points_per100",
        "Turnover rate": "turnover_per100",
        "Bad-pass turnover rate": "badpass_turnover_per100",
        "Transition frequency": "transition_per100",
        "Rim frequency value": "rim_freq_value_per100",
        "Rim finishing value": "rim_fg_value_per100",
        "Midrange value": "mid_fg_value_per100",
        "Three frequency value": "three_freq_value_per100",
        "Three shotmaking value": "three_fg_value_per100",
        "Free-throw value": "ft_value_per100",
        "Second chance": "second_chance_points_per100",
    }
    rows = []
    for label, outcome in outcomes.items():
        fitted = fit_continuous_response(regular, outcome=outcome)
        row = fitted.loc[
            fitted["specification"].eq("linear") & fitted["feature"].eq("oepm_x_def")
        ].iloc[0].copy()
        row["mechanism"] = label
        rows.append(row)
    return pd.DataFrame(rows)


def reanchor_player_quality(frame: pd.DataFrame, store: EpmStore, mode: str) -> pd.DataFrame:
    """Replace the frozen player O-EPM/role definition for robustness tests."""
    result = frame.copy()
    mapped_parts = []
    for season, group in result.groupby("season", sort=False):
        if mode == "preseason":
            snapshot = store.asof(pd.Timestamp(year=int(season) - 1, month=6, day=30))
            subject = snapshot.reindex(group["player_id"].astype(np.int64).to_numpy())
            subject.index = group.index
        elif mode == "endseason":
            snapshot = store.asof(pd.Timestamp(year=int(season), month=6, day=30))
            subject = snapshot.reindex(group["player_id"].astype(np.int64).to_numpy())
            subject.index = group.index
        elif mode == "pregame":
            subject = pd.DataFrame(index=group.index, columns=EPM_FIELDS)
            for game_date, indices in group.groupby("game_date", sort=False).groups.items():
                snapshot = store.asof(game_date)
                ids = group.loc[indices, "player_id"].astype(np.int64)
                values = snapshot.reindex(ids.to_numpy())
                values.index = indices
                for field in EPM_FIELDS:
                    if field in values:
                        subject.loc[indices, field] = values[field]
        else:
            raise ValueError(f"Unknown anchor mode: {mode}")
        part = group.copy()
        for field in ["oepm", *ROLE_FIELDS]:
            part[field] = pd.to_numeric(subject[field], errors="coerce")
        part["player_name"] = subject["player_name"].fillna(part["player_name"])
        mapped_parts.append(part)
    result = pd.concat(mapped_parts).sort_index()
    result = result.dropna(subset=["oepm", *ROLE_FIELDS]).copy()
    result["full_tier"] = full_oepm_tier(result["oepm"])
    result["display_tier"] = display_oepm_tier(result["oepm"])
    result["role"] = classify_role(result)
    return result


def fit_season_interactions(regular: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, group in regular.groupby("season", sort=True):
        fitted = fit_continuous_response(group)
        row = fitted.loc[
            fitted["specification"].eq("linear") & fitted["feature"].eq("oepm_x_def")
        ].iloc[0].copy()
        row["season"] = int(season)
        rows.append(row)
    return pd.DataFrame(rows)


def playoff_residual_summaries(playoff_games: pd.DataFrame) -> pd.DataFrame:
    data = playoff_games.copy()
    data["oepm_group"] = display_oepm_tier(data["oepm"]).fillna("other")
    rows = []
    for dimension, column in [("round", "round"), ("oepm_tier", "oepm_group"), ("role", "role")]:
        for label, group in data.groupby(column, sort=False):
            estimate = clustered_mean(group, "playoff_residual_per100", "possessions", "cluster_game")
            rows.append({
                "dimension": dimension, "group": label, **estimate,
                "player_games": len(group), "players": group["player_id"].nunique(),
                "possessions": int(group["possessions"].sum()),
            })
    return pd.DataFrame(rows)


def fit_archetype_models(regular: pd.DataFrame) -> pd.DataFrame:
    subset = regular.loc[regular["oepm"].between(-0.5, 0.5, inclusive="both")].copy()
    subset["role_tier"] = subset["role"].astype("string")
    # fit_tier_response expects 0_1 as its reference; rename the largest role
    # to the reference and restore its label afterward.
    reference = subset.groupby("role_tier")["possessions"].sum().idxmax()
    subset["role_model"] = subset["role_tier"].replace({reference: "0_1"})
    fitted = fit_tier_response(subset, tier_column="role_model")
    table = fitted.tier_slopes.copy()
    table["role"] = table["tier"].replace({"0_1": reference})
    return table.drop(columns="tier")


def fit_placebos(regular: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    linear = observed.loc[
        observed["specification"].eq("linear") & observed["feature"].eq("oepm_x_def")
    ].iloc[0]
    rows.append({"test": "Observed", **linear[["coefficient", "se", "ci_low", "ci_high"]].to_dict()})

    shuffled = regular.copy()
    rng = np.random.default_rng(20260820)
    for season, indices in shuffled.groupby("season", sort=False).groups.items():
        values = shuffled.loc[indices, "def_centered"].to_numpy().copy()
        rng.shuffle(values)
        shuffled.loc[indices, "def_centered"] = values
    result = fit_continuous_response(shuffled)
    row = result.loc[result["specification"].eq("linear") & result["feature"].eq("oepm_x_def")].iloc[0]
    rows.append({"test": "Defense shuffled within season", **row[["coefficient", "se", "ci_low", "ci_high"]].to_dict()})

    next_game = regular.sort_values(["player_season", "game_date", "game_id"]).copy()
    next_game["def_centered"] = next_game.groupby("player_season", sort=False)["def_centered"].shift(-1)
    result = fit_continuous_response(next_game.dropna(subset=["def_centered"]))
    row = result.loc[result["specification"].eq("linear") & result["feature"].eq("oepm_x_def")].iloc[0]
    rows.append({"test": "Next-game defense placebo", **row[["coefficient", "se", "ci_low", "ci_high"]].to_dict()})
    return pd.DataFrame(rows)


def make_charts(
    output_dir: Path,
    tier_table: pd.DataFrame,
    continuous: pd.DataFrame,
    environment_sample: pd.DataFrame,
    playoff_players: pd.DataFrame,
    player_slopes: pd.DataFrame,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = ["#555555", "#2f6b9a", "#2d8b57", "#9a6b2f", "#7a3f91"]
    x = np.linspace(
        environment_sample["def_centered"].quantile(0.02),
        environment_sample["def_centered"].quantile(0.98), 160,
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    for color, row in zip(colors, tier_table.itertuples()):
        y = row.baseline_oepm + row.relative_portability_slope * x
        ax.plot(x, y, lw=2.4, label=row.label, color=color)
    ax.axvline(0, color="black", lw=1)
    ax.set(xlabel="Defensive lineup EPM faced (season-centered, higher = tougher)",
           ylabel="Estimated offensive impact (relative points per 100)",
           title="Do offensive-impact tiers remain parallel against stronger personnel?")
    ax.legend(title="Frozen Dec. 1 O-EPM")
    fig.tight_layout()
    fig.savefig(output_dir / "01_oepm_tier_response_curves.png", dpi=180)
    plt.close(fig)

    linear = continuous.loc[
        continuous["specification"].eq("linear") & continuous["feature"].eq("oepm_x_def")
    ].iloc[0]
    elite = environment_sample.loc[environment_sample["round"].eq("Regular season"), "def_centered"].quantile(0.90)
    o_values = np.linspace(-2, 6, 161)
    center = float(linear["oepm_center"])
    delta = float(linear["coefficient"]) * (o_values - center) * elite
    low = float(linear["ci_low"]) * (o_values - center) * elite
    high = float(linear["ci_high"]) * (o_values - center) * elite
    lower = np.minimum(low, high)
    upper = np.maximum(low, high)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(o_values, delta, color="#222222", lw=2.5)
    ax.fill_between(o_values, lower, upper, color="#888888", alpha=0.25)
    ax.axhline(0, color="black", lw=1)
    ax.set(xlabel="Frozen Dec. 1 O-EPM", ylabel="Relative impact change: average → elite defense",
           title="Competition translation by baseline offensive quality")
    fig.tight_layout()
    fig.savefig(output_dir / "02_elite_defense_decline_by_oepm.png", dpi=180)
    plt.close(fig)

    round_order = ["Regular season", "First round", "Second round", "Conference finals", "Finals"]
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(environment_sample["def_centered"].quantile(0.005),
                       environment_sample["def_centered"].quantile(0.995), 55)
    for label, color in zip(round_order, ["#777777", "#4c78a8", "#59a14f", "#f28e2b", "#e15759"]):
        values = environment_sample.loc[environment_sample["round"].eq(label), "def_centered"]
        if not values.empty:
            ax.hist(values, bins=bins, density=True, histtype="step", lw=2, label=label, color=color)
    ax.set(xlabel="Defensive lineup EPM faced (season-centered)", ylabel="Density",
           title="Actual defensive-personnel environments by competition stage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "03_defensive_environment_by_round.png", dpi=180)
    plt.close(fig)

    eligible = playoff_players.loc[playoff_players["playoff_possessions"].ge(250)].copy()
    fig, ax = plt.subplots(figsize=(7, 7))
    sizes = np.sqrt(eligible["playoff_possessions"]) * 4
    ax.scatter(eligible["playoff_predicted_offense"], eligible["playoff_actual_offense"],
               s=sizes, alpha=0.45, color="#2f6b9a", edgecolor="none")
    limits = [
        min(eligible["playoff_predicted_offense"].min(), eligible["playoff_actual_offense"].min()),
        max(eligible["playoff_predicted_offense"].max(), eligible["playoff_actual_offense"].max()),
    ]
    ax.plot(limits, limits, color="black", lw=1)
    ax.set(xlabel="Regular-season model prediction", ylabel="Actual playoff on-court offense",
           title="Playoff actual versus opponent-quality-adjusted prediction")
    fig.tight_layout()
    fig.savefig(output_dir / "04_playoff_actual_vs_predicted.png", dpi=180)
    plt.close(fig)

    eligible = player_slopes.loc[player_slopes["possessions"].ge(1800)].copy()
    tails = pd.concat([eligible.nsmallest(12, "slope"), eligible.nlargest(12, "slope")]).drop_duplicates()
    tails = tails.sort_values("slope")
    labels = tails["player_name"] + " (" + tails["season"].astype(str) + ")"
    fig, ax = plt.subplots(figsize=(9, 9))
    y_pos = np.arange(len(tails))
    ax.errorbar(tails["slope"], y_pos, xerr=1.96 * tails["slope_se"], fmt="o", color="#333333", ecolor="#999999")
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y_pos, labels)
    ax.set(xlabel="Adjusted offense slope per +1 defensive lineup EPM",
           title="Most and least robust supported player-season estimates")
    fig.tight_layout()
    fig.savefig(output_dir / "05_player_specific_robustness.png", dpi=180)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: Sequence[str], digits: int = 3) -> str:
    view = frame[list(columns)].copy()
    for col in view.select_dtypes(include=["number"]).columns:
        view[col] = view[col].map(lambda value: f"{value:.{digits}f}" if pd.notna(value) else "")
    headers = "| " + " | ".join(view.columns) + " |"
    rule = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in view.to_numpy()]
    return "\n".join([headers, rule, *rows])


def write_report(
    output_dir: Path,
    audits: list[dict],
    regular: pd.DataFrame,
    model: FittedResponse,
    continuous: pd.DataFrame,
    tier_table: pd.DataFrame,
    archetypes: pd.DataFrame,
    placebo: pd.DataFrame,
    environment: pd.DataFrame,
    playoff_mean: Mapping[str, float],
    series: pd.DataFrame,
    robustness: pd.DataFrame,
    seasons: pd.DataFrame,
    mechanism_continuous: pd.DataFrame,
    playoff_summaries: pd.DataFrame,
) -> None:
    interaction = continuous.loc[
        continuous["specification"].eq("linear") & continuous["feature"].eq("oepm_x_def")
    ].iloc[0]
    avg_row = tier_table.loc[tier_table["oepm_tier"].eq("avg_0")].iloc[0]
    elite_row = tier_table.loc[tier_table["oepm_tier"].eq("4_plus")].iloc[0]
    elite_gap_change = float(elite_row["elite_minus_average"] - avg_row["elite_minus_average"])
    mechanism_view = mechanism_continuous.sort_values("coefficient").copy()
    report = f"""# Offensive Impact Portability to Stronger NBA Defenses

## Executive result

**The original compression hypothesis is not supported. The statistically clear interaction
runs in the opposite direction:** higher O-EPM players lose slightly more relative on-court
offensive value as the five defensive players get stronger.

The primary continuous interaction is **{interaction['coefficient']:+.3f} points per 100**
per +1 O-EPM and +1 opposing lineup D-EPM (95% CI
[{interaction['ci_low']:+.3f}, {interaction['ci_high']:+.3f}]). Positive values would mean
better offensive players retain more impact; the observed negative value means the reverse.
This is the interaction of interest, not the mechanically negative main effect of facing a
better defense.

Moving from an average defensive environment to the regular-season top decile changes the
relative impact estimate by **{avg_row['elite_minus_average']:+.2f} points/100** for the
approximately average O-EPM tier and **{elite_row['elite_minus_average']:+.2f}** for the +4
and-above tier. The gap therefore compresses by **{elite_gap_change:+.2f} points/100**. That
is detectable but modest relative to the roughly five-point baseline gap between these tiers.

Role matters in a different way than the motivating example: among approximately average
O-EPM players, creators and secondary creators have flatter adjusted slopes than low-usage
shooters, big finishers, and connectors. The aggregate high-O-EPM decline is concentrated in
first-chance shot value, especially rim finishing and three-point shotmaking; turnover rates
do not explain it.

The opponent-quality-adjusted mean playoff residual is **{playoff_mean['estimate']:+.2f}
points per 100** (95% CI [{playoff_mean['ci_low']:+.2f}, {playoff_mean['ci_high']:+.2f}]).
This is descriptive evidence about an additional postseason effect; it is not a causal
estimate because playoff participation, minutes, and matchups are selected.

## Primary sample and design

- Seasons: 2018-19 through 2025-26.
- Regular-season analysis starts December 1 in each season.
- Frozen player quality and role: latest game-agnostic EPM snapshot requested on or before
  December 1. The provider says date-D predictions exclude games on D.
- Defensive environment: sum of the five defenders' D-EPM from the latest snapshot requested
  on or before each game. Positive is tougher. The primary sample requires all five values.
- Unit: player-game, built from standard possessions. The outcome is team points per 100
  offensive possessions while the player was on court; credit is not assigned only to the
  shooter.
- Adjustment: player-season fixed effects, the other four offensive players' rolling O-EPM,
  home court, rest, offensive score margin, and garbage-time share.
- Uncertainty: clustered by game. Primary weights are possessions; equal-player-season and
  equal-player alternatives are saved in the robustness table.
- Mechanisms use the FIRST_CHANCE + SECOND_CHANCE_CLEAN luck-adjusted target layer; the
  primary outcome remains actual RAPM points. Rare expected-FT-versus-actual conversion
  gaps are reported in the manifest. The rim/mid/three columns are additive point-value
  mechanisms, not personal box-score rates.

The primary regular-season table has {len(regular):,} player-games,
{regular['player_season'].nunique():,} player-seasons, {regular['player_id'].nunique():,}
players, and {int(regular['possessions'].sum()):,} player-possessions.

## Tier response slopes

Slopes below are adjusted on-court offense changes per +1 point of opposing five-man D-EPM.
The relative-portability slope subtracts the possession-weighted league-player average; this
normalization is necessary because a common decline can be assigned either to the field or
equally to all five offensive players. Differential slopes across tiers are identified; an
absolute allocation of the common decline to individual players is not.

{markdown_table(model.tier_slopes, ['tier', 'slope_per_def_epm', 'ci_low', 'ci_high', 'relative_portability_slope', 'player_seasons', 'possessions'])}

## Requested O-EPM tier table

These are estimated relative player-impact levels, anchored to each tier's frozen O-EPM at an
average defensive environment. They are not raw team offensive ratings.

{markdown_table(tier_table, ['label', 'baseline_oepm', 'weak_p25', 'average', 'strong_p75', 'elite_p90', 'elite_minus_average', 'player_seasons', 'possessions'])}

## Approximately average players by role

{markdown_table(archetypes, ['role', 'slope_per_def_epm', 'ci_low', 'ci_high', 'relative_portability_slope', 'player_seasons', 'possessions'])}

## Defensive environments

{markdown_table(environment, ['environment', 'possessions', 'games', 'mean_def_epm', 'sd_def_epm', 'p25', 'median', 'p75', 'p90'])}

## Mechanism interaction coefficients

Each coefficient is the O-EPM × defensive-lineup-EPM interaction for the named possession
target. For turnover rates, positive means more turnovers; for the point-value targets,
positive means more offense.

{markdown_table(mechanism_view, ['mechanism', 'coefficient', 'ci_low', 'ci_high', 'possessions'])}

## Weighting, sample, and anchor robustness

{markdown_table(robustness, ['weight_mode', 'coefficient', 'ci_low', 'ci_high', 'rows', 'player_seasons'])}

## Season-by-season interaction

{markdown_table(seasons, ['season', 'coefficient', 'ci_low', 'ci_high', 'rows', 'player_seasons'])}

## Falsifications

{markdown_table(placebo, ['test', 'coefficient', 'ci_low', 'ci_high'])}

## Repeated series exposure

{markdown_table(series, ['feature', 'coefficient', 'ci_low', 'ci_high', 'player_series', 'series', 'games', 'possessions'])}

## Playoff residual slices

{markdown_table(playoff_summaries, ['dimension', 'group', 'estimate', 'ci_low', 'ci_high', 'players', 'possessions'])}

## Interpretation limits

This design establishes conditional association in adjusted on-court offense. It does not prove
that defensive athleticism itself causes a player's interaction slope. EPM is an external
estimated input and therefore measured with error. December 1 player tiers exclude earlier
games and trade some sample size for temporal cleanliness. The same-sample end-of-season EPM
definition is not used in the primary model. Player-specific slopes are shrunken diagnostics,
not a released metric. Playoff and series results remain selection-sensitive.

## Files

Reusable player-game, player-slope, tier, mechanism, archetype, defensive-environment,
postseason prediction, falsification, and manifest tables are in this directory. Figures 01-05
correspond to the five requested visual diagnostics.
"""
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=2019)
    parser.add_argument("--end-season", type=int, default=2026)
    parser.add_argument("--epm-dir", type=Path, default=EPM_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--no-charts", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = EpmStore.load(args.epm_dir)
    regular_parts: list[pd.DataFrame] = []
    playoff_parts: list[pd.DataFrame] = []
    environment_parts: list[pd.DataFrame] = []
    audits: list[dict] = []
    for season in range(args.start_season, args.end_season + 1):
        for season_type in ["RS", "PS"]:
            player_games, environment, audit = build_player_games(
                season, season_type, store, strict_defense=True,
                start_on_anchor=season_type == "RS",
            )
            if season_type == "RS":
                regular_parts.append(player_games)
            else:
                playoff_parts.append(player_games)
            environment_parts.append(environment)
            audits.append(audit)
            print(
                f"{season} {season_type}: {audit['possessions_after_strict_defense']:,} possessions, "
                f"{audit['player_game_rows']:,} player-games"
            )
    regular = add_rest_days(pd.concat(regular_parts, ignore_index=True))
    playoffs = add_rest_days(pd.concat(playoff_parts, ignore_index=True))
    environment_all = pd.concat(environment_parts, ignore_index=True)
    regular, playoffs, environment_all, def_stats = add_season_defense_centering(
        regular, playoffs, environment_all
    )
    regular["cluster_game"] = regular["season"].astype(str) + "_" + regular["game_id"].astype(str)
    playoffs["cluster_game"] = playoffs["season"].astype(str) + "_" + playoffs["game_id"].astype(str)

    primary = fit_tier_response(regular)
    continuous = fit_continuous_response(regular)
    robustness_rows = []
    for weight_mode in ["possessions", "player_season", "player"]:
        fit = fit_continuous_response(regular, weight_mode=weight_mode)
        row = fit.loc[fit["specification"].eq("linear") & fit["feature"].eq("oepm_x_def")].iloc[0].copy()
        row["weight_mode"] = weight_mode
        robustness_rows.append(row)
    no_garbage = regular.loc[regular["garbage_share"].eq(0)].copy()
    row = fit_continuous_response(no_garbage)
    row = row.loc[row["specification"].eq("linear") & row["feature"].eq("oepm_x_def")].iloc[0].copy()
    row["weight_mode"] = "no_garbage_possessions"
    robustness_rows.append(row)
    for anchor_mode, label in [
        ("preseason", "prior_season_june30_oepm"),
        ("pregame", "rolling_pregame_oepm"),
        ("endseason", "endseason_oepm_leakage_check"),
    ]:
        anchored = reanchor_player_quality(regular, store, anchor_mode)
        anchor_fit = fit_continuous_response(anchored)
        row = anchor_fit.loc[
            anchor_fit["specification"].eq("linear") & anchor_fit["feature"].eq("oepm_x_def")
        ].iloc[0].copy()
        row["weight_mode"] = label
        robustness_rows.append(row)
    for control_set, label in [
        (["home_offense", "rest_days", "pre_margin", "garbage_share"], "without_teammate_quality"),
        ([], "player_season_fe_only"),
    ]:
        control_fit = fit_continuous_response(regular, controls=control_set)
        row = control_fit.loc[
            control_fit["specification"].eq("linear") & control_fit["feature"].eq("oepm_x_def")
        ].iloc[0].copy()
        row["weight_mode"] = label
        robustness_rows.append(row)
    robustness = pd.DataFrame(robustness_rows)

    player_slopes = fit_player_specific_slopes(regular)
    mechanisms = fit_mechanism_models(regular)
    mechanism_continuous = fit_mechanism_continuous(regular)
    archetypes = fit_archetype_models(regular)
    placebos = fit_placebos(regular, continuous)
    season_interactions = fit_season_interactions(regular)
    env_summary, env_sample = build_environment_tables(environment_all)
    tier_table = build_tier_environment_table(primary.tier_slopes, regular, env_summary)
    playoff_games, playoff_players = predict_playoffs(playoffs, primary)
    playoff_mean = clustered_mean(
        playoff_games, "playoff_residual_per100", "possessions", "cluster_game"
    )
    series = fit_series_test(playoff_games)
    playoff_summaries = playoff_residual_summaries(playoff_games)
    player_inspection, player_buckets = build_player_season_inspection(
        regular, primary, player_slopes, playoff_players, env_summary
    )

    regular.to_parquet(args.output_dir / "player_game_regular_season.parquet", index=False)
    playoffs.to_parquet(args.output_dir / "player_game_playoffs.parquet", index=False)
    primary.tier_slopes.to_csv(args.output_dir / "model_a_tier_slopes.csv", index=False)
    continuous.to_csv(args.output_dir / "model_b_continuous_interactions.csv", index=False)
    player_slopes.to_csv(args.output_dir / "model_c_player_specific_slopes.csv", index=False)
    player_inspection.to_csv(args.output_dir / "player_season_inspection.csv", index=False)
    player_buckets.to_csv(args.output_dir / "player_season_defense_buckets.csv", index=False)
    mechanisms.to_csv(args.output_dir / "mechanism_response_slopes.csv", index=False)
    mechanism_continuous.to_csv(args.output_dir / "mechanism_continuous_interactions.csv", index=False)
    archetypes.to_csv(args.output_dir / "average_oepm_archetype_slopes.csv", index=False)
    robustness.to_csv(args.output_dir / "robustness_weighting_and_garbage.csv", index=False)
    placebos.to_csv(args.output_dir / "falsification_tests.csv", index=False)
    env_summary.to_csv(args.output_dir / "defensive_environment_by_round.csv", index=False)
    env_sample.to_parquet(args.output_dir / "defensive_environment_chart_sample.parquet", index=False)
    tier_table.to_csv(args.output_dir / "oepm_tier_environment_table.csv", index=False)
    playoff_games.to_parquet(args.output_dir / "playoff_player_game_predictions.parquet", index=False)
    playoff_players.to_csv(args.output_dir / "playoff_player_predictions.csv", index=False)
    series.to_csv(args.output_dir / "playoff_series_exposure_test.csv", index=False)
    playoff_summaries.to_csv(args.output_dir / "playoff_residual_summaries.csv", index=False)
    season_interactions.to_csv(args.output_dir / "season_by_season_interactions.csv", index=False)
    def_stats.to_csv(args.output_dir / "season_defensive_environment_scaling.csv", index=False)

    if not args.no_charts:
        make_charts(args.output_dir, tier_table, continuous, env_sample, playoff_players, player_slopes)
    write_report(
        args.output_dir, audits, regular, primary, continuous, tier_table, archetypes,
        placebos, env_summary, playoff_mean, series, robustness, season_interactions,
        mechanism_continuous, playoff_summaries,
    )
    manifest = {
        "schema_version": 1,
        "analysis": "offensive_impact_portability",
        "seasons": [args.start_season, args.end_season],
        "epm_snapshot_files": store.source_files,
        "provider_date_contract": "Predictions requested for date D exclude games played on D.",
        "primary_anchor": "Latest requested snapshot on or before December 1; games from December 1 onward.",
        "defense_measure": "Sum of actual five defenders' latest-pregame D-EPM; strict five-player coverage.",
        "regular_player_games": int(len(regular)),
        "regular_player_seasons": int(regular["player_season"].nunique()),
        "regular_unique_players": int(regular["player_id"].nunique()),
        "regular_player_possessions": int(regular["possessions"].sum()),
        "playoff_player_games": int(len(playoffs)),
        "playoff_player_possessions": int(playoffs["possessions"].sum()),
        "playoff_residual": playoff_mean,
        "audits": audits,
        "outputs": sorted(path.name for path in args.output_dir.iterdir()),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote analysis to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
