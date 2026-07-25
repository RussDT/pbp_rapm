#!/usr/bin/env python3
"""Fit historical exact-five offense and defense above EPM expectations.

This is a genuine reconstruction model, not an inversion of the recovered
459-row output table.  It combines:

* clean exact-five scoreboard possessions;
* season-end Dunks & Threes EPM / E-Skills snapshots;
* historical on-ball time and potential-assist player features; and
* forward-only ridge validation.

The five EPM predicted-usage values are normalized to a 100% lineup allocation
for concentration and overlap features.  Their raw sum is retained only as a
role-demand surplus/shortfall diagnostic.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
REPO_DIR = PIPELINE_DIR.parent
DEFAULT_ROLE_CACHE = (
    PIPELINE_DIR / "cache" / "five_man_fit" / "player_role_features_2014_2026.parquet"
)
DEFAULT_OUTPUT = (
    PIPELINE_DIR / "results" / "five_man_fit" / "historical_epm_fit.parquet"
)
DEFAULT_REPORT = (
    PIPELINE_DIR / "results" / "five_man_fit" / "historical_epm_fit_report.json"
)
DEFAULT_BUILDER_OUTPUT = (
    REPO_DIR / "five-man-fit-lab" / "data" / "builder_players.json"
)

OFFENSE_FEATURE_GROUPS = {
    "usage": [
        "role_demand_surplus",
        "role_demand_overload",
        "role_demand_shortfall",
        "usage_top1_share",
        "usage_top2_share",
        "usage_hhi",
        "usage_top2_overlap",
    ],
    "onball": [
        "onball_sum",
        "onball_top2",
        "onball_hhi",
        "onball_top2_overlap",
    ],
    "playmaking": [
        "potential_assist_sum",
        "potential_assist_top2",
        "potential_assist_hhi",
        "potential_assist_overlap",
        "passer_finisher_fit",
    ],
    "pressure": [
        "rim_pressure_sum",
        "ft_pressure_sum",
        "three_volume_sum",
        "turnover_burden_sum",
        "usage_weighted_tspct",
    ],
}

OFFENSE_CANDIDATE_FEATURE_GROUPS = {
    **OFFENSE_FEATURE_GROUPS,
    "usage": [
        "role_demand_overload",
        "role_demand_shortfall",
        "usage_top1_share",
        "usage_top2_share",
        "usage_hhi",
        "usage_top2_overlap",
    ],
}

OFFENSE_DISPLAY_FEATURE_GROUPS = {
    group: OFFENSE_FEATURE_GROUPS[group]
    for group in ("usage", "onball", "playmaking")
}

DEFENSE_FEATURE_GROUPS = {
    "workload": [
        "role_demand_surplus",
        "role_demand_overload",
        "usage_top1_share",
        "usage_top2_share",
        "usage_hhi",
    ],
    "onball": [
        "onball_sum",
        "onball_top2",
        "onball_hhi",
        "onball_top2_overlap",
    ],
    "defensive_activity": [
        "steal_skill_sum",
        "steal_skill_top2",
        "block_skill_sum",
        "block_skill_top2",
    ],
    "possession_finish": [
        "def_rebound_skill_sum",
        "def_rebound_skill_bottom2",
        "rim_protection_balance",
    ],
}

DEFENSE_CHALLENGER_GROUPS = ("defensive_activity", "possession_finish")
FIXED_OFFENSE_RAW_COEFFICIENTS = {
    # A lineup below 92% raw EPM role demand pays 1.5 points per 10 percentage
    # points of shortfall. This monotone capacity guardrail prevents a sparse,
    # collinear tail from learning that missing creation is beneficial.
    "role_demand_shortfall": -15.0,
}
ALPHA_GRID = [0.0, 10.0, 30.0, 100.0, 300.0, 1_000.0, 3_000.0, 10_000.0, 30_000.0]
IDENTITY_GRID = [0.0, 250.0, 500.0, 750.0, 1_000.0, 1_500.0, 2_000.0, 3_000.0, 5_000.0]
LINEUP_COLUMNS = [f"{side}{idx}" for side in ("O", "D") for idx in range(1, 6)]


@dataclass
class RidgeFit:
    coefficients: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    feature_names: list[str]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.feature_names].to_numpy(dtype=float)
        return (values - self.mean) / self.scale

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.transform(frame) @ self.coefficients

    def contributions(self, frame: pd.DataFrame) -> pd.DataFrame:
        values = self.transform(frame) * self.coefficients
        return pd.DataFrame(
            values,
            columns=self.feature_names,
            index=frame.index,
        )


def flatten(groups: dict[str, list[str]]) -> list[str]:
    return [feature for features in groups.values() for feature in features]


def season_label(year: int) -> str:
    return f"{year - 1}\u2013{str(year)[-2:]}"


def lineup_key(frame: pd.DataFrame, prefix: str) -> pd.Series:
    columns = [f"{prefix}{idx}" for idx in range(1, 6)]
    values = (
        frame[columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype("int64")
        .to_numpy()
    )
    values.sort(axis=1)
    return pd.Series(
        ["-".join(map(str, row)) for row in values],
        index=frame.index,
        dtype="string",
    )


def valid_lineups(frame: pd.DataFrame) -> pd.Series:
    valid = frame[LINEUP_COLUMNS].gt(0).all(axis=1)
    valid &= frame[[f"O{i}" for i in range(1, 6)]].nunique(axis=1).eq(5)
    valid &= frame[[f"D{i}" for i in range(1, 6)]].nunique(axis=1).eq(5)
    return valid


def fetch_role_features(first_year: int, last_year: int) -> pd.DataFrame:
    try:
        from supabase import create_client
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("supabase-py is required to refresh role features") from exc

    url = os.environ.get("SUPABASE_URL", "").strip()
    key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        or os.environ.get("SUPABASE_KEY", "").strip()
    )
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required to refresh "
            "historical role features"
        )
    client = create_client(url, key)
    columns = (
        'nba_id,year,player_name,"OffPoss","Minutes",'
        '"PASSING_on-ball-time%","PASSING_Potential Assists/100"'
    )
    rows: list[dict[str, object]] = []
    for year in range(first_year, last_year + 1):
        response = (
            client.table("player_stats_with_metrics")
            .select(columns)
            .eq("year", year)
            .eq("playoffs", 0)
            .execute()
        )
        rows.extend(response.data)
    frame = pd.DataFrame(rows)
    numeric = [
        "nba_id",
        "year",
        "OffPoss",
        "Minutes",
        "PASSING_on-ball-time%",
        "PASSING_Potential Assists/100",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values(
        ["year", "nba_id", "OffPoss", "Minutes"],
        ascending=[True, True, False, False],
        kind="stable",
    ).drop_duplicates(["year", "nba_id"], keep="first")
    frame["nba_id"] = frame["nba_id"].astype("int64")
    frame["year"] = frame["year"].astype("int64")
    return frame.reset_index(drop=True)


def load_role_features(
    path: Path,
    first_year: int,
    last_year: int,
    refresh: bool,
) -> pd.DataFrame:
    if refresh or not path.exists():
        frame = fetch_role_features(first_year, last_year)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    else:
        frame = pd.read_parquet(path)
    required = {
        "nba_id",
        "year",
        "PASSING_on-ball-time%",
        "PASSING_Potential Assists/100",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Role feature cache is missing {sorted(missing)}")
    return frame.loc[
        frame["year"].between(first_year, last_year)
    ].copy()


def load_epm(year: int) -> pd.DataFrame:
    path = PIPELINE_DIR / "cache" / "epm" / f"epm_all_{year}-06-30_go0.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}; fetch a season-end game_optimized=0 EPM snapshot"
        )
    columns = [
        "player_id",
        "player_name",
        "p_usg",
        "p_fga_rim_100",
        "p_fg3a_100",
        "p_fta_100",
        "p_tov_100",
        "p_tspct",
        "p_drb_100",
        "p_stl_100",
        "p_blk_100",
        "oepm",
        "depm",
    ]
    frame = pd.read_csv(path, usecols=columns)
    frame["player_id"] = pd.to_numeric(
        frame["player_id"], errors="raise"
    ).astype("int64")
    for column in set(columns) - {"player_id", "player_name"}:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame["player_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate player ids")
    return frame.set_index("player_id")


def hhi(values: np.ndarray) -> float:
    total = float(np.nansum(values))
    if total <= 0:
        return np.nan
    shares = values / total
    return float(np.sum(shares**2))


def top(values: np.ndarray, count: int) -> float:
    return float(np.sort(values)[-count:].sum())


def bottom(values: np.ndarray, count: int) -> float:
    return float(np.sort(values)[:count].sum())


def pair_overlap(values: np.ndarray) -> float:
    return float(
        sum(values[left] * values[right] for left in range(5) for right in range(left + 1, 5))
    )


def build_lineup_features(
    year: int,
    keys: Iterable[str],
    epm: pd.DataFrame,
    roles: pd.DataFrame,
) -> pd.DataFrame:
    role_year = (
        roles.loc[roles["year"].eq(year)]
        .drop_duplicates("nba_id")
        .set_index("nba_id")
    )
    records: list[dict[str, object]] = []
    for key in sorted(set(keys)):
        ids = np.array([int(value) for value in key.split("-")], dtype=np.int64)
        epm_rows = epm.reindex(ids)
        role_rows = role_year.reindex(ids)
        complete = bool(
            epm_rows[
                [
                    "p_usg",
                    "p_fga_rim_100",
                    "p_fg3a_100",
                    "p_fta_100",
                    "p_tov_100",
                    "p_tspct",
                    "p_drb_100",
                    "p_stl_100",
                    "p_blk_100",
                    "oepm",
                    "depm",
                ]
            ]
            .notna()
            .all()
            .all()
            and role_rows[
                ["PASSING_on-ball-time%", "PASSING_Potential Assists/100"]
            ]
            .notna()
            .all()
            .all()
        )
        record: dict[str, object] = {
            "season": year,
            "lineup_key": key,
            "feature_complete": complete,
        }
        if not complete:
            records.append(record)
            continue

        usage = epm_rows["p_usg"].to_numpy(dtype=float)
        allocation = usage / usage.sum()
        usage_sorted = np.sort(allocation)
        onball = (
            role_rows["PASSING_on-ball-time%"].to_numpy(dtype=float) / 100.0
        )
        onball_sorted = np.sort(onball)
        potential = role_rows[
            "PASSING_Potential Assists/100"
        ].to_numpy(dtype=float)
        rim = epm_rows["p_fga_rim_100"].to_numpy(dtype=float)
        ft = epm_rows["p_fta_100"].to_numpy(dtype=float)
        three = epm_rows["p_fg3a_100"].to_numpy(dtype=float)
        turnover = epm_rows["p_tov_100"].to_numpy(dtype=float)
        tspct = epm_rows["p_tspct"].to_numpy(dtype=float)
        steal = epm_rows["p_stl_100"].to_numpy(dtype=float)
        block = epm_rows["p_blk_100"].to_numpy(dtype=float)
        drb = epm_rows["p_drb_100"].to_numpy(dtype=float)
        finish_pressure = rim + 0.44 * ft

        raw_demand = float(usage.sum())
        record.update(
            {
                "epm_usage_demand_sum": raw_demand,
                "normalized_usage_sum": float(allocation.sum()),
                "role_demand_surplus": raw_demand - 1.0,
                "role_demand_overload": max(raw_demand - 1.03, 0.0),
                "role_demand_shortfall": max(0.92 - raw_demand, 0.0),
                "usage_top1_share": float(usage_sorted[-1]),
                "usage_top2_share": float(usage_sorted[-2:].sum()),
                "usage_hhi": float(np.sum(allocation**2)),
                "usage_top2_overlap": float(
                    2.0 * usage_sorted[-1] * usage_sorted[-2]
                ),
                "onball_sum": float(onball.sum()),
                "onball_top2": float(onball_sorted[-2:].sum()),
                "onball_hhi": hhi(onball),
                "onball_top2_overlap": float(
                    2.0 * onball_sorted[-1] * onball_sorted[-2]
                ),
                "potential_assist_sum": float(potential.sum()),
                "potential_assist_top2": top(potential, 2),
                "potential_assist_hhi": hhi(potential),
                "potential_assist_overlap": pair_overlap(potential),
                "passer_finisher_fit": float(
                    np.mean(
                        [
                            potential[idx]
                            * np.delete(finish_pressure, idx).mean()
                            for idx in range(5)
                        ]
                    )
                ),
                "rim_pressure_sum": float(rim.sum()),
                "ft_pressure_sum": float(ft.sum()),
                "three_volume_sum": float(three.sum()),
                "turnover_burden_sum": float(turnover.sum()),
                "usage_weighted_tspct": float(np.sum(allocation * tspct)),
                "steal_skill_sum": float(steal.sum()),
                "steal_skill_top2": top(steal, 2),
                "block_skill_sum": float(block.sum()),
                "block_skill_top2": top(block, 2),
                "def_rebound_skill_sum": float(drb.sum()),
                "def_rebound_skill_bottom2": bottom(drb, 2),
                "rim_protection_balance": hhi(block + 0.25),
                "lineup_oepm_sum": float(epm_rows["oepm"].sum()),
                "lineup_depm_sum": float(epm_rows["depm"].sum()),
                "players": [
                    {
                        "id": int(player_id),
                        "name": str(epm_rows.loc[player_id, "player_name"]),
                        "epm_predicted_usage": float(
                            epm_rows.loc[player_id, "p_usg"]
                        ),
                        "normalized_lineup_usage": float(allocation[idx]),
                        "onball_time": float(onball[idx]),
                        "potential_assists_per100": float(potential[idx]),
                        "oepm": float(epm_rows.loc[player_id, "oepm"]),
                        "depm": float(epm_rows.loc[player_id, "depm"]),
                    }
                    for idx, player_id in enumerate(ids)
                ],
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def map_sum(
    frame: pd.DataFrame,
    columns: list[str],
    values: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    mapped = pd.DataFrame(
        {column: frame[column].map(values) for column in columns},
        index=frame.index,
    )
    return mapped.sum(axis=1, min_count=len(columns)), mapped.notna().all(axis=1)


def grouped_side(
    frame: pd.DataFrame,
    *,
    side: str,
    split: str | None = None,
) -> pd.DataFrame:
    subset = frame if split is None else frame.loc[frame["split"].eq(split)]
    if side == "off":
        grouped = (
            subset.groupby(["off_team_abbr", "off_lineup_key"], dropna=False)
            .agg(
                off_points=("off_points", "sum"),
                off_possessions=("off_poss", "sum"),
                expected_offense_sum=("expected_offense_weighted", "sum"),
                expected_offense_possessions=("expected_offense_poss", "sum"),
            )
            .reset_index()
            .rename(
                columns={
                    "off_team_abbr": "team",
                    "off_lineup_key": "lineup_key",
                }
            )
        )
    else:
        grouped = (
            subset.groupby(["def_team_abbr", "def_lineup_key"], dropna=False)
            .agg(
                points_allowed=("off_points", "sum"),
                def_possessions=("off_poss", "sum"),
                expected_defense_sum=("expected_defense_weighted", "sum"),
                expected_defense_possessions=("expected_defense_poss", "sum"),
            )
            .reset_index()
            .rename(
                columns={
                    "def_team_abbr": "team",
                    "def_lineup_key": "lineup_key",
                }
            )
        )
    if split is not None:
        suffix = f"_{split}"
        grouped = grouped.rename(
            columns={
                column: f"{column}{suffix}"
                for column in grouped.columns
                if column not in {"team", "lineup_key"}
            }
        )
    return grouped


def aggregate_season(
    year: int,
    roles: pd.DataFrame,
    minimum_possessions: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    short_year = year - 2000
    path = (
        PIPELINE_DIR
        / "processed"
        / f"LINEUP_NET_RATING_POSSESSIONS{short_year}.parquet"
    )
    frame = pd.read_parquet(path)
    for column in LINEUP_COLUMNS:
        frame[column] = (
            pd.to_numeric(frame[column], errors="coerce")
            .fillna(0)
            .astype("int64")
        )
    frame = frame.loc[valid_lineups(frame)].copy()
    frame["off_lineup_key"] = lineup_key(frame, "O")
    frame["def_lineup_key"] = lineup_key(frame, "D")

    games = sorted(frame["game_id"].astype(str).unique())
    game_order = {game_id: index for index, game_id in enumerate(games)}
    frame["split"] = np.where(
        frame["game_id"].astype(str).map(game_order) < len(games) / 2,
        "early",
        "late",
    )

    epm = load_epm(year)
    oepm = epm["oepm"]
    depm = epm["depm"]
    offense_oepm, offense_oepm_ok = map_sum(
        frame, [f"O{i}" for i in range(1, 6)], oepm
    )
    defense_depm, defense_depm_ok = map_sum(
        frame, [f"D{i}" for i in range(1, 6)], depm
    )
    defense_oepm, defense_oepm_ok = map_sum(
        frame, [f"D{i}" for i in range(1, 6)], oepm
    )
    offense_depm, offense_depm_ok = map_sum(
        frame, [f"O{i}" for i in range(1, 6)], depm
    )
    expected_offense_ok = offense_oepm_ok & defense_depm_ok & frame["off_poss"].gt(0)
    expected_defense_ok = defense_depm_ok & offense_oepm_ok & frame["off_poss"].gt(0)
    frame["expected_offense_weighted"] = (
        offense_oepm - defense_depm
    ).where(expected_offense_ok, 0.0) * frame["off_poss"]
    frame["expected_offense_poss"] = expected_offense_ok.astype(float) * frame[
        "off_poss"
    ]
    frame["expected_defense_weighted"] = (
        defense_depm - offense_oepm
    ).where(expected_defense_ok, 0.0) * frame["off_poss"]
    frame["expected_defense_poss"] = expected_defense_ok.astype(float) * frame[
        "off_poss"
    ]

    league_possessions = float(frame["off_poss"].sum())
    league_points = float(frame["off_points"].sum())
    league_ortg = 100.0 * league_points / league_possessions

    output = grouped_side(frame, side="off").merge(
        grouped_side(frame, side="def"),
        on=["team", "lineup_key"],
        how="inner",
    )
    for split in ("early", "late"):
        output = output.merge(
            grouped_side(frame, side="off", split=split),
            on=["team", "lineup_key"],
            how="left",
        ).merge(
            grouped_side(frame, side="def", split=split),
            on=["team", "lineup_key"],
            how="left",
        )
    output = output.loc[
        output[["off_possessions", "def_possessions"]]
        .min(axis=1)
        .ge(minimum_possessions)
    ].copy()
    output["possessions"] = (
        (output["off_possessions"] + output["def_possessions"]) / 2.0
    )
    output["ortg"] = 100.0 * output["off_points"] / output["off_possessions"]
    output["drtg"] = 100.0 * output["points_allowed"] / output["def_possessions"]
    output["net_rating"] = output["ortg"] - output["drtg"]
    output["offense_vs_average"] = output["ortg"] - league_ortg
    output["defense_vs_average"] = league_ortg - output["drtg"]
    output["expected_offense_vs_average"] = (
        output["expected_offense_sum"] / output["expected_offense_possessions"]
    )
    output["expected_defense_vs_average"] = (
        output["expected_defense_sum"] / output["expected_defense_possessions"]
    )
    output["off_target"] = (
        output["offense_vs_average"] - output["expected_offense_vs_average"]
    )
    output["def_target"] = (
        output["defense_vs_average"] - output["expected_defense_vs_average"]
    )

    for split in ("early", "late"):
        output[f"ortg_{split}"] = (
            100.0
            * output[f"off_points_{split}"]
            / output[f"off_possessions_{split}"]
        )
        output[f"drtg_{split}"] = (
            100.0
            * output[f"points_allowed_{split}"]
            / output[f"def_possessions_{split}"]
        )
        output[f"expected_offense_vs_average_{split}"] = (
            output[f"expected_offense_sum_{split}"]
            / output[f"expected_offense_possessions_{split}"]
        )
        output[f"expected_defense_vs_average_{split}"] = (
            output[f"expected_defense_sum_{split}"]
            / output[f"expected_defense_possessions_{split}"]
        )
        output[f"off_target_{split}"] = (
            output[f"ortg_{split}"]
            - league_ortg
            - output[f"expected_offense_vs_average_{split}"]
        )
        output[f"def_target_{split}"] = (
            league_ortg
            - output[f"drtg_{split}"]
            - output[f"expected_defense_vs_average_{split}"]
        )

    features = build_lineup_features(
        year,
        output["lineup_key"],
        epm,
        roles,
    )
    output = output.merge(features, on="lineup_key", how="left")
    output["season"] = year
    output["season_label"] = season_label(year)
    output["id"] = (
        "fit-"
        + str(year)
        + "-"
        + output["team"].astype(str).str.lower()
        + "-"
        + output["lineup_key"].astype(str)
    )
    metadata = {
        "season": year,
        "season_label": season_label(year),
        "league_ortg": league_ortg,
        "source_possessions": int(league_possessions),
        "qualifying_lineups": int(len(output)),
        "feature_complete_lineups": int(output["feature_complete"].fillna(False).sum()),
        "epm_offense_possession_coverage": float(
            frame["expected_offense_poss"].sum() / league_possessions
        ),
        "epm_defense_possession_coverage": float(
            frame["expected_defense_poss"].sum() / league_possessions
        ),
    }
    return output, metadata


def fit_ridge(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    weight: str,
    alpha: float,
    fixed_raw_coefficients: dict[str, float] | None = None,
) -> RidgeFit:
    values = frame[features].to_numpy(dtype=float)
    weights = frame[weight].to_numpy(dtype=float)
    mean = np.average(values, axis=0, weights=weights)
    variance = np.average((values - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(variance)
    scale[scale < 1e-12] = 1.0
    standardized = (values - mean) / scale
    fixed_raw_coefficients = fixed_raw_coefficients or {}
    coefficients = np.zeros(len(features), dtype=float)
    fixed_indices = []
    for feature, raw_coefficient in fixed_raw_coefficients.items():
        if feature not in features:
            continue
        index = features.index(feature)
        fixed_indices.append(index)
        coefficients[index] = float(raw_coefficient) * scale[index]
    fixed_prediction = standardized @ coefficients
    free_indices = [
        index for index in range(len(features)) if index not in fixed_indices
    ]
    if free_indices:
        model = Ridge(alpha=alpha, fit_intercept=False)
        model.fit(
            standardized[:, free_indices],
            frame[target].to_numpy(dtype=float) - fixed_prediction,
            sample_weight=weights,
        )
        coefficients[free_indices] = model.coef_
    return RidgeFit(
        coefficients=coefficients,
        mean=mean,
        scale=scale,
        feature_names=features,
    )


def expand_fit(
    fit: RidgeFit,
    frame: pd.DataFrame,
    *,
    features: list[str],
    weight: str,
) -> RidgeFit:
    values = frame[features].to_numpy(dtype=float)
    weights = frame[weight].to_numpy(dtype=float)
    mean = np.average(values, axis=0, weights=weights)
    variance = np.average((values - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(variance)
    scale[scale < 1e-12] = 1.0
    coefficients = np.zeros(len(features), dtype=float)
    for source_index, feature in enumerate(fit.feature_names):
        target_index = features.index(feature)
        raw_coefficient = fit.coefficients[source_index] / fit.scale[source_index]
        coefficients[target_index] = raw_coefficient * scale[target_index]
    return RidgeFit(
        coefficients=coefficients,
        mean=mean,
        scale=scale,
        feature_names=features,
    )


def weighted_mse(actual: np.ndarray, predicted: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average((actual - predicted) ** 2, weights=weights))


def choose_alpha(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    weight: str,
    first_validation_year: int,
    fixed_raw_coefficients: dict[str, float] | None = None,
) -> tuple[float, list[dict[str, object]], pd.Series]:
    scores: list[dict[str, object]] = []
    predictions = {
        alpha: pd.Series(np.nan, index=frame.index, dtype=float)
        for alpha in ALPHA_GRID
    }
    for alpha in ALPHA_GRID:
        fold_rows = []
        for year in range(first_validation_year, int(frame["season"].max()) + 1):
            train = frame.loc[frame["season"].lt(year)]
            validation = frame.loc[frame["season"].eq(year)]
            if train.empty or validation.empty:
                continue
            fit = fit_ridge(
                train,
                features,
                target,
                weight,
                alpha,
                fixed_raw_coefficients,
            )
            pred = fit.predict(validation)
            predictions[alpha].loc[validation.index] = pred
            actual = validation[target].to_numpy(dtype=float)
            weights = validation[weight].to_numpy(dtype=float)
            fold_rows.append(
                {
                    "season": year,
                    "rows": int(len(validation)),
                    "mse": weighted_mse(actual, pred, weights),
                    "baseline_mse": weighted_mse(
                        actual, np.zeros_like(actual), weights
                    ),
                    "weight": float(weights.sum()),
                }
            )
        total_weight = sum(row["weight"] for row in fold_rows)
        mse = sum(row["mse"] * row["weight"] for row in fold_rows) / total_weight
        baseline = (
            sum(row["baseline_mse"] * row["weight"] for row in fold_rows)
            / total_weight
        )
        scores.append(
            {
                "alpha": alpha,
                "mse": mse,
                "baseline_mse": baseline,
                "mse_improvement_pct": 100.0 * (baseline - mse) / baseline,
                "folds": fold_rows,
            }
        )
    selected = min(scores, key=lambda row: row["mse"])
    return (
        float(selected["alpha"]),
        scores,
        predictions[float(selected["alpha"])],
    )


def choose_identity_shrinkage(
    frame: pd.DataFrame,
    *,
    side: str,
    forward_prediction: pd.Series,
    minimum_split_possessions: int,
) -> tuple[float, list[dict[str, float]]]:
    target_early = f"{side}_target_early"
    target_late = f"{side}_target_late"
    weight_early = "off_possessions_early" if side == "off" else "def_possessions_early"
    weight_late = "off_possessions_late" if side == "off" else "def_possessions_late"
    eligible = frame.loc[
        forward_prediction.notna()
        & frame[target_early].notna()
        & frame[target_late].notna()
        & frame[weight_early].ge(minimum_split_possessions)
        & frame[weight_late].ge(minimum_split_possessions)
    ].copy()
    mechanism = forward_prediction.loc[eligible.index].to_numpy(dtype=float)
    early = eligible[target_early].to_numpy(dtype=float)
    late = eligible[target_late].to_numpy(dtype=float)
    early_weight = eligible[weight_early].to_numpy(dtype=float)
    late_weight = eligible[weight_late].to_numpy(dtype=float)
    scores = []
    for shrinkage in IDENTITY_GRID:
        identity = (early - mechanism) * early_weight / (early_weight + shrinkage)
        predicted = mechanism + identity
        scores.append(
            {
                "shrinkage_possessions": float(shrinkage),
                "rows": float(len(eligible)),
                "mse": weighted_mse(late, predicted, late_weight),
                "mechanism_only_mse": weighted_mse(late, mechanism, late_weight),
            }
        )
    selected = min(scores, key=lambda row: row["mse"])
    return float(selected["shrinkage_possessions"]), scores


def walk_forward_predictions(
    frame: pd.DataFrame,
    *,
    features: list[str],
    target: str,
    weight: str,
    alpha: float,
    years: Iterable[int],
    fixed_raw_coefficients: dict[str, float] | None = None,
) -> pd.Series:
    predictions = pd.Series(np.nan, index=frame.index, dtype=float)
    for year in years:
        train = frame.loc[frame["season"].lt(year)]
        validation = frame.loc[frame["season"].eq(year)]
        if train.empty or validation.empty:
            continue
        fit = fit_ridge(
            train,
            features,
            target,
            weight,
            alpha,
            fixed_raw_coefficients,
        )
        predictions.loc[validation.index] = fit.predict(validation)
    return predictions


def prediction_score(
    frame: pd.DataFrame,
    *,
    prediction: pd.Series,
    target: str,
    weight: str,
) -> dict[str, float | int]:
    eligible = frame.loc[prediction.notna()]
    actual = eligible[target].to_numpy(dtype=float)
    predicted = prediction.loc[eligible.index].to_numpy(dtype=float)
    weights = eligible[weight].to_numpy(dtype=float)
    mse = weighted_mse(actual, predicted, weights)
    baseline = weighted_mse(actual, np.zeros_like(actual), weights)
    return {
        "rows": int(len(eligible)),
        "weight": float(weights.sum()),
        "mse": mse,
        "baseline_mse": baseline,
        "mse_improvement_pct": 100.0 * (baseline - mse) / baseline,
    }


def confirmation_identity_score(
    frame: pd.DataFrame,
    *,
    side: str,
    prediction: pd.Series,
    shrinkage: float,
    minimum_split_possessions: int,
) -> dict[str, float | int]:
    target_early = f"{side}_target_early"
    target_late = f"{side}_target_late"
    weight_early = "off_possessions_early" if side == "off" else "def_possessions_early"
    weight_late = "off_possessions_late" if side == "off" else "def_possessions_late"
    eligible = frame.loc[
        prediction.notna()
        & frame[target_early].notna()
        & frame[target_late].notna()
        & frame[weight_early].ge(minimum_split_possessions)
        & frame[weight_late].ge(minimum_split_possessions)
    ]
    mechanism = prediction.loc[eligible.index].to_numpy(dtype=float)
    early = eligible[target_early].to_numpy(dtype=float)
    late = eligible[target_late].to_numpy(dtype=float)
    early_weight = eligible[weight_early].to_numpy(dtype=float)
    late_weight = eligible[weight_late].to_numpy(dtype=float)
    identity = (early - mechanism) * early_weight / (early_weight + shrinkage)
    late_weight = eligible[weight_late].to_numpy(dtype=float)
    mse = weighted_mse(late, mechanism + identity, late_weight)
    mechanism_only = weighted_mse(late, mechanism, late_weight)
    return {
        "rows": int(len(eligible)),
        "mse": mse,
        "mechanism_only_mse": mechanism_only,
        "mse_improvement_pct": 100.0 * (mechanism_only - mse) / mechanism_only,
    }


def feature_group_ablation(
    frame: pd.DataFrame,
    *,
    groups: dict[str, list[str]],
    target: str,
    weight: str,
    first_validation_year: int,
    fixed_raw_coefficients: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    import itertools

    rows: list[dict[str, object]] = []
    names = list(groups)
    for group_count in range(1, len(names) + 1):
        for selected_groups in itertools.combinations(names, group_count):
            features = [
                feature
                for group in selected_groups
                for feature in groups[group]
            ]
            alpha, scores, _ = choose_alpha(
                frame,
                features,
                target,
                weight,
                first_validation_year,
                fixed_raw_coefficients,
            )
            selected_score = next(
                score for score in scores if float(score["alpha"]) == alpha
            )
            rows.append(
                {
                    "groups": list(selected_groups),
                    "selected_alpha": alpha,
                    "mse": selected_score["mse"],
                    "baseline_mse": selected_score["baseline_mse"],
                    "mse_improvement_pct": selected_score[
                        "mse_improvement_pct"
                    ],
                }
            )
    return sorted(rows, key=lambda row: float(row["mse"]))


def add_model_outputs(
    frame: pd.DataFrame,
    *,
    side: str,
    groups: dict[str, list[str]],
    target: str,
    weight: str,
    alpha: float,
    identity_shrinkage: float,
    fixed_raw_coefficients: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, RidgeFit]:
    features = flatten(groups)
    fit = fit_ridge(
        frame,
        features,
        target,
        weight,
        alpha,
        fixed_raw_coefficients,
    )
    output = frame.copy()
    output[f"{side}_mechanism_fit"] = fit.predict(output)
    contributions = fit.contributions(output)
    for feature in features:
        output[f"{side}_contribution_{feature}"] = contributions[feature]
    for group, group_features in groups.items():
        output[f"{side}_domain_{group}"] = contributions[group_features].sum(
            axis=1
        )
    domain_columns = [f"{side}_domain_{group}" for group in groups]
    gap = (
        output[f"{side}_mechanism_fit"] - output[domain_columns].sum(axis=1)
    ).abs().max()
    if float(gap) > 1e-9:
        raise ValueError(f"{side} mechanism domain identity failed: {gap}")
    raw_identity = output[target] - output[f"{side}_mechanism_fit"]
    sample = output[weight]
    output[f"{side}_raw_identity_residual"] = raw_identity
    output[f"{side}_lineup_specific_fit"] = (
        raw_identity * sample / (sample + identity_shrinkage)
    )
    output[f"{side}_total_fit"] = (
        output[f"{side}_mechanism_fit"]
        + output[f"{side}_lineup_specific_fit"]
    )
    return output, fit


def add_unselected_offense_diagnostics(
    frame: pd.DataFrame,
    selected_groups: dict[str, list[str]],
) -> pd.DataFrame:
    output = frame.copy()
    selected_features = set(flatten(selected_groups))
    for group, features in OFFENSE_DISPLAY_FEATURE_GROUPS.items():
        if group not in selected_groups:
            output[f"off_domain_{group}"] = 0.0
        for feature in features:
            if feature not in selected_features:
                output[f"off_contribution_{feature}"] = 0.0
    return output


def add_rejected_defense_outputs(
    frame: pd.DataFrame,
    *,
    target: str,
) -> pd.DataFrame:
    output = frame.copy()
    output["def_mechanism_fit"] = 0.0
    for group in DEFENSE_FEATURE_GROUPS:
        output[f"def_domain_{group}"] = 0.0
    output["def_raw_identity_residual"] = output[target]
    output["def_lineup_specific_fit"] = 0.0
    output["def_total_fit"] = 0.0
    output["def_fit_status"] = "rejected_no_confirmation_lift"
    return output


def serialize_fit(fit: RidgeFit) -> dict[str, object]:
    return {
        "features": fit.feature_names,
        "weighted_feature_means": fit.mean.tolist(),
        "weighted_feature_scales": fit.scale.tolist(),
        "standardized_coefficients": fit.coefficients.tolist(),
        "fit_intercept": False,
    }


def export_builder_payload(
    *,
    year: int,
    roles: pd.DataFrame,
    fit: RidgeFit,
    feature_groups: dict[str, list[str]],
    selected_feature_groups: Iterable[str],
    output_path: Path,
) -> None:
    epm = load_epm(year).reset_index().rename(columns={"player_id": "nba_id"})
    role_year = (
        roles.loc[roles["year"].eq(year)]
        .sort_values(["nba_id", "OffPoss", "Minutes"], ascending=[True, False, False])
        .drop_duplicates("nba_id")
    )
    players = epm.merge(
        role_year[
            [
                "nba_id",
                "OffPoss",
                "Minutes",
                "PASSING_on-ball-time%",
                "PASSING_Potential Assists/100",
            ]
        ],
        on="nba_id",
        how="inner",
        validate="one_to_one",
    )
    required = [
        "p_usg",
        "p_fga_rim_100",
        "p_fg3a_100",
        "p_fta_100",
        "p_tov_100",
        "p_tspct",
        "oepm",
        "depm",
        "PASSING_on-ball-time%",
        "PASSING_Potential Assists/100",
    ]
    players = players.dropna(subset=required).copy()
    players = players.loc[players["OffPoss"].fillna(0).ge(100)].copy()
    players = players.sort_values(
        ["OffPoss", "player_name"],
        ascending=[False, True],
    )
    player_rows = [
        {
            "id": int(row.nba_id),
            "name": str(row.player_name),
            "off_possessions": int(round(float(row.OffPoss))),
            "minutes": round(float(row.Minutes), 1),
            "p_usg": float(row.p_usg),
            "onball_time": float(row["PASSING_on-ball-time%"]) / 100.0,
            "potential_assists_per100": float(
                row["PASSING_Potential Assists/100"]
            ),
            "rim_attempts_per100": float(row.p_fga_rim_100),
            "three_attempts_per100": float(row.p_fg3a_100),
            "free_throw_attempts_per100": float(row.p_fta_100),
            "turnovers_per100": float(row.p_tov_100),
            "predicted_true_shooting": float(row.p_tspct),
            "oepm": float(row.oepm),
            "depm": float(row.depm),
        }
        for _, row in players.iterrows()
    ]
    payload = {
        "meta": {
            "season": year,
            "season_label": season_label(year),
            "player_minimum_offensive_possessions": 100,
            "player_count": len(player_rows),
            "estimate_contract": (
                "sum five O-EPM + forward-selected offensive construction "
                "fit; sum five D-EPM for defense"
            ),
            "identity_note": (
                "A hypothetical five has no exact-lineup residual; identity "
                "is zero until the group has an observed sample."
            ),
        },
        "model": {
            **serialize_fit(fit),
            "feature_groups": feature_groups,
            "selected_feature_groups": list(selected_feature_groups),
            "fixed_raw_coefficients": FIXED_OFFENSE_RAW_COEFFICIENTS,
        },
        "players": player_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-year", type=int, default=2014)
    parser.add_argument("--last-year", type=int, default=2026)
    parser.add_argument("--minimum-possessions", type=int, default=200)
    parser.add_argument("--minimum-split-possessions", type=int, default=75)
    parser.add_argument("--first-validation-year", type=int, default=2018)
    parser.add_argument("--confirmation-seasons", type=int, default=3)
    parser.add_argument("--role-cache", type=Path, default=DEFAULT_ROLE_CACHE)
    parser.add_argument("--refresh-role-cache", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--builder-output",
        type=Path,
        default=DEFAULT_BUILDER_OUTPUT,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roles = load_role_features(
        args.role_cache,
        args.first_year,
        args.last_year,
        args.refresh_role_cache,
    )
    frames = []
    seasons = []
    for year in range(args.first_year, args.last_year + 1):
        frame, metadata = aggregate_season(
            year,
            roles,
            args.minimum_possessions,
        )
        frames.append(frame)
        seasons.append(metadata)
        print(
            f"{metadata['season_label']}: {metadata['feature_complete_lineups']:,}/"
            f"{metadata['qualifying_lineups']:,} feature-complete lineups"
        )
    all_rows = pd.concat(frames, ignore_index=True)
    model_rows = all_rows.loc[all_rows["feature_complete"].fillna(False)].copy()
    model_rows = model_rows.replace([np.inf, -np.inf], np.nan)
    off_features = flatten(OFFENSE_FEATURE_GROUPS)
    def_features = flatten(DEFENSE_FEATURE_GROUPS)
    required = sorted(
        set(
            off_features
            + def_features
            + [
                "off_target",
                "def_target",
                "off_possessions",
                "def_possessions",
            ]
        )
    )
    model_rows = model_rows.dropna(subset=required).copy()

    confirmation_first_year = args.last_year - args.confirmation_seasons + 1
    development = model_rows.loc[
        model_rows["season"].lt(confirmation_first_year)
    ].copy()
    confirmation_years = range(confirmation_first_year, args.last_year + 1)

    off_development_ablation = feature_group_ablation(
        development,
        groups=OFFENSE_CANDIDATE_FEATURE_GROUPS,
        target="off_target",
        weight="off_possessions",
        first_validation_year=args.first_validation_year,
        fixed_raw_coefficients=FIXED_OFFENSE_RAW_COEFFICIENTS,
    )
    selected_offense_group_names = tuple(
        str(group) for group in off_development_ablation[0]["groups"]
    )
    selected_offense_groups = {
        group: OFFENSE_CANDIDATE_FEATURE_GROUPS[group]
        for group in selected_offense_group_names
    }
    selected_offense_features = flatten(selected_offense_groups)
    off_alpha, off_alpha_scores, off_development_forward = choose_alpha(
        development,
        selected_offense_features,
        "off_target",
        "off_possessions",
        args.first_validation_year,
        FIXED_OFFENSE_RAW_COEFFICIENTS,
    )
    off_confirmation_forward = walk_forward_predictions(
        model_rows,
        features=selected_offense_features,
        target="off_target",
        weight="off_possessions",
        alpha=off_alpha,
        years=confirmation_years,
        fixed_raw_coefficients=FIXED_OFFENSE_RAW_COEFFICIENTS,
    )
    off_identity, off_identity_scores = choose_identity_shrinkage(
        development,
        side="off",
        forward_prediction=off_development_forward,
        minimum_split_possessions=args.minimum_split_possessions,
    )

    defense_challenger_groups = {
        group: DEFENSE_FEATURE_GROUPS[group]
        for group in DEFENSE_CHALLENGER_GROUPS
    }
    defense_challenger_features = flatten(defense_challenger_groups)
    def_alpha, def_alpha_scores, def_development_forward = choose_alpha(
        development,
        defense_challenger_features,
        "def_target",
        "def_possessions",
        args.first_validation_year,
    )
    def_confirmation_forward = walk_forward_predictions(
        model_rows,
        features=defense_challenger_features,
        target="def_target",
        weight="def_possessions",
        alpha=def_alpha,
        years=confirmation_years,
    )
    def_identity, def_identity_scores = choose_identity_shrinkage(
        development,
        side="def",
        forward_prediction=def_development_forward,
        minimum_split_possessions=args.minimum_split_possessions,
    )

    modeled, off_fit = add_model_outputs(
        model_rows,
        side="off",
        groups=selected_offense_groups,
        target="off_target",
        weight="off_possessions",
        alpha=off_alpha,
        identity_shrinkage=off_identity,
        fixed_raw_coefficients=FIXED_OFFENSE_RAW_COEFFICIENTS,
    )
    modeled = add_unselected_offense_diagnostics(
        modeled,
        selected_offense_groups,
    )
    full_off_fit = expand_fit(
        off_fit,
        model_rows,
        features=flatten(OFFENSE_DISPLAY_FEATURE_GROUPS),
        weight="off_possessions",
    )
    defense_challenger_fit = fit_ridge(
        model_rows,
        defense_challenger_features,
        target="def_target",
        alpha=def_alpha,
        weight="def_possessions",
    )
    modeled = add_rejected_defense_outputs(
        modeled,
        target="def_target",
    )
    modeled["modeled_offense_vs_average"] = (
        modeled["expected_offense_vs_average"] + modeled["off_total_fit"]
    )
    modeled["modeled_defense_vs_average"] = (
        modeled["expected_defense_vs_average"] + modeled["def_total_fit"]
    )
    modeled["modeled_net_vs_average"] = (
        modeled["modeled_offense_vs_average"]
        + modeled["modeled_defense_vs_average"]
    )
    modeled["modeled_neutral_offense"] = (
        modeled["lineup_oepm_sum"] + modeled["off_total_fit"]
    )
    modeled["modeled_neutral_defense"] = modeled["lineup_depm_sum"]
    modeled["modeled_neutral_net"] = (
        modeled["modeled_neutral_offense"] + modeled["modeled_neutral_defense"]
    )
    modeled["off_fit_status"] = "validated_core_construction_plus_identity"
    modeled["players_json"] = modeled["players"].map(
        lambda value: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    )
    modeled = modeled.drop(columns=["players"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    modeled.to_parquet(args.output, index=False)
    report = {
        "schema_version": 1,
        "title": "Historical Five-Man EPM Fit",
        "scope": {
            "first_season": args.first_year,
            "last_season": args.last_year,
            "minimum_possessions_each_side": args.minimum_possessions,
            "minimum_split_possessions": args.minimum_split_possessions,
            "lineup_seasons": int(len(modeled)),
            "development_seasons": [
                args.first_validation_year,
                confirmation_first_year - 1,
            ],
            "confirmation_seasons": [confirmation_first_year, args.last_year],
            "seasons": seasons,
        },
        "usage_contract": {
            "provider_field": "Dunks & Threes p_usg",
            "display_allocation": (
                "Each player's p_usg divided by the five-player p_usg sum; "
                "the five displayed shares close to 100%."
            ),
            "role_demand_diagnostic": (
                "The unnormalized five-player p_usg sum is role demand, never "
                "lineup usage. Below 92%, shortfall receives a fixed monotone "
                "penalty of 1.5 points per 10 percentage points."
            ),
        },
        "offense": {
            "target": (
                "actual ORtg vs season minus (sum five oEPM minus "
                "possession-weighted opponent dEPM)"
            ),
            "selected_alpha": off_alpha,
            "published_feature_groups": list(selected_offense_group_names),
            "selection_rule": (
                "Choose the feature-group combination with the lowest "
                "possession-weighted forward MSE in 2018-2023 development."
            ),
            "fixed_raw_coefficients": FIXED_OFFENSE_RAW_COEFFICIENTS,
            "alpha_grid": off_alpha_scores,
            "development_score": prediction_score(
                development,
                prediction=off_development_forward,
                target="off_target",
                weight="off_possessions",
            ),
            "confirmation_score": prediction_score(
                model_rows,
                prediction=off_confirmation_forward,
                target="off_target",
                weight="off_possessions",
            ),
            "development_group_ablation": off_development_ablation,
            "selected_identity_shrinkage_possessions": off_identity,
            "identity_grid": off_identity_scores,
            "confirmation_identity_score": confirmation_identity_score(
                model_rows,
                side="off",
                prediction=off_confirmation_forward,
                shrinkage=off_identity,
                minimum_split_possessions=args.minimum_split_possessions,
            ),
            "feature_groups": OFFENSE_DISPLAY_FEATURE_GROUPS,
            "final_fit": serialize_fit(full_off_fit),
        },
        "defense": {
            "target": (
                "positive-good actual defense vs season minus (sum five dEPM "
                "minus possession-weighted opponent oEPM)"
            ),
            "status": "rejected_no_confirmation_lift",
            "published_adjustment": 0.0,
            "challenger_feature_groups": list(DEFENSE_CHALLENGER_GROUPS),
            "selected_alpha": def_alpha,
            "alpha_grid": def_alpha_scores,
            "development_score": prediction_score(
                development,
                prediction=def_development_forward,
                target="def_target",
                weight="def_possessions",
            ),
            "confirmation_score": prediction_score(
                model_rows,
                prediction=def_confirmation_forward,
                target="def_target",
                weight="def_possessions",
            ),
            "development_group_ablation": feature_group_ablation(
                development,
                groups=DEFENSE_FEATURE_GROUPS,
                target="def_target",
                weight="def_possessions",
                first_validation_year=args.first_validation_year,
            ),
            "selected_identity_shrinkage_possessions": def_identity,
            "identity_grid": def_identity_scores,
            "confirmation_identity_score": confirmation_identity_score(
                model_rows,
                side="def",
                prediction=def_confirmation_forward,
                shrinkage=def_identity,
                minimum_split_possessions=args.minimum_split_possessions,
            ),
            "feature_groups": defense_challenger_groups,
            "challenger_fit_not_published": serialize_fit(
                defense_challenger_fit
            ),
            "usage_onball_result": (
                "No workload/on-ball group combination beat the EPM-only "
                "defensive baseline in development; the strongest defensive "
                "E-Skills challenger then failed the 2024-2026 confirmation."
            ),
        },
        "mechanism_layer": (
            "Observed possession mechanisms remain a separate descriptive layer "
            "and are not predictors in either fit regression."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    export_builder_payload(
        year=args.last_year,
        roles=roles,
        fit=full_off_fit,
        feature_groups=OFFENSE_DISPLAY_FEATURE_GROUPS,
        selected_feature_groups=selected_offense_group_names,
        output_path=args.builder_output,
    )
    print(
        f"Selected offense groups={','.join(selected_offense_group_names)}, "
        f"alpha={off_alpha:g}, identity={off_identity:g}; "
        f"defense alpha={def_alpha:g}, identity={def_identity:g}"
    )
    print(f"Wrote {len(modeled):,} lineups to {args.output}")
    print(f"Wrote validation report to {args.report}")
    print(f"Wrote current player builder inputs to {args.builder_output}")


if __name__ == "__main__":
    main()
