#!/usr/bin/env python3
"""Audit the deprecated DPM-adjusted 2014+ lineup-history prototype.

This is a descriptive, season-level companion to the recovered offensive EPM
model. It uses the clean raw-scoreboard possession surface for actual ORtg and
DRtg, derives player usage from play-by-play, and uses same-season DARKO
offensive/defensive DPM only as a transparent talent-and-opponent control.
It does not recreate or extend the lost EPM training pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
DEFAULT_DPM_PATH = PIPELINE_DIR / ".cache" / "teammate_help" / "dpm_history.csv"
DEFAULT_OUTPUT = (
    PIPELINE_DIR
    / "validation"
    / "five_man_fit_lab"
    / "deprecated_dpm_history_audit.json"
)

LINEUP_COLUMNS = [f"{side}{idx}" for side in ("O", "D") for idx in range(1, 6)]


def season_end_year(short_year: int) -> int:
    return 2000 + int(short_year)


def season_label(year: int) -> str:
    return f"{year - 1}\u2013{str(year)[-2:]}"


def lineup_keys(frame: pd.DataFrame, prefix: str) -> pd.Series:
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


def weighted_correlation(x: pd.Series, y: pd.Series, weight: pd.Series) -> float:
    x_values = x.to_numpy(dtype=float)
    y_values = y.to_numpy(dtype=float)
    weights = weight.to_numpy(dtype=float)
    weight_sum = weights.sum()
    if weight_sum <= 0:
        return float("nan")
    x_mean = np.average(x_values, weights=weights)
    y_mean = np.average(y_values, weights=weights)
    covariance = np.average((x_values - x_mean) * (y_values - y_mean), weights=weights)
    x_variance = np.average((x_values - x_mean) ** 2, weights=weights)
    y_variance = np.average((y_values - y_mean) ** 2, weights=weights)
    denominator = np.sqrt(x_variance * y_variance)
    return float(covariance / denominator) if denominator > 0 else float("nan")


def weighted_slope_per_ten_usage(
    x: pd.Series, y: pd.Series, weight: pd.Series
) -> float:
    x_values = x.to_numpy(dtype=float)
    y_values = y.to_numpy(dtype=float)
    weights = weight.to_numpy(dtype=float)
    x_mean = np.average(x_values, weights=weights)
    y_mean = np.average(y_values, weights=weights)
    denominator = np.sum(weights * (x_values - x_mean) ** 2)
    if denominator <= 0:
        return float("nan")
    slope = np.sum(weights * (x_values - x_mean) * (y_values - y_mean)) / denominator
    return float(slope * 0.10)


def load_dpm(path: Path, first_year: int, last_year: int) -> pd.DataFrame:
    dpm = pd.read_csv(path)
    required = {"nba_id", "player_name", "season", "o_dpm", "d_dpm"}
    missing = required.difference(dpm.columns)
    if missing:
        raise ValueError(f"DPM history is missing columns: {sorted(missing)}")
    dpm = dpm[dpm["season"].between(first_year, last_year)].copy()
    dpm["nba_id"] = pd.to_numeric(dpm["nba_id"], errors="coerce")
    dpm = dpm.dropna(subset=["nba_id"]).copy()
    dpm["nba_id"] = dpm["nba_id"].astype("int64")
    if dpm.duplicated(["season", "nba_id"]).any():
        raise ValueError("DPM history must contain one row per player-season")
    return dpm


def usage_numerators(raw_path: Path) -> tuple[pd.Series, dict[int, str]]:
    raw = pd.read_parquet(
        raw_path,
        columns=["event_type", "player1_id", "player1_name"],
    )
    event_type = pd.to_numeric(raw["event_type"], errors="coerce")
    player_id = pd.to_numeric(raw["player1_id"], errors="coerce")
    weights = np.select(
        [event_type.isin([1, 2]), event_type.eq(3), event_type.eq(5)],
        [1.0, 0.44, 1.0],
        default=0.0,
    )
    events = pd.DataFrame({"player_id": player_id, "usage_event": weights})
    events = events[(events["player_id"] > 0) & (events["usage_event"] > 0)].copy()
    events["player_id"] = events["player_id"].astype("int64")
    numerator = events.groupby("player_id")["usage_event"].sum()

    names = raw.loc[player_id.gt(0), ["player1_id", "player1_name"]].copy()
    names["player1_id"] = pd.to_numeric(names["player1_id"], errors="coerce").astype(
        "int64"
    )
    names["player1_name"] = names["player1_name"].fillna("").astype(str).str.strip()
    names = names[names["player1_name"].ne("")]
    name_lookup = (
        names.drop_duplicates("player1_id", keep="last")
        .set_index("player1_id")["player1_name"]
        .to_dict()
    )
    return numerator, {int(key): str(value) for key, value in name_lookup.items()}


def on_court_offensive_possessions(possessions: pd.DataFrame) -> pd.Series:
    counted = possessions.loc[
        possessions["off_poss"].gt(0), ["off_poss", *[f"O{i}" for i in range(1, 6)]]
    ].copy()
    long = counted.melt(
        id_vars="off_poss",
        value_vars=[f"O{i}" for i in range(1, 6)],
        value_name="player_id",
    )
    long["player_id"] = pd.to_numeric(long["player_id"], errors="coerce")
    long = long[long["player_id"].gt(0)].copy()
    long["player_id"] = long["player_id"].astype("int64")
    return long.groupby("player_id")["off_poss"].sum()


def sum_player_values(
    frame: pd.DataFrame,
    columns: Iterable[str],
    lookup: dict[int, float],
) -> tuple[pd.Series, pd.Series]:
    total = pd.Series(0.0, index=frame.index)
    coverage = pd.Series(0, index=frame.index, dtype="int64")
    for column in columns:
        player_ids = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(
            "int64"
        )
        mapped = player_ids.map(lookup)
        total = total.add(mapped.fillna(0.0), fill_value=0.0)
        coverage = coverage.add(mapped.notna().astype("int64"), fill_value=0)
    return total, coverage


def aggregate_season(
    short_year: int,
    dpm_season: pd.DataFrame,
    minimum_possessions: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    year = season_end_year(short_year)
    possession_path = (
        PIPELINE_DIR
        / "processed"
        / f"LINEUP_NET_RATING_POSSESSIONS{short_year}.parquet"
    )
    raw_path = PIPELINE_DIR / "raw_data" / f"NBA{short_year}.parquet"
    if not possession_path.exists():
        raise FileNotFoundError(
            f"Missing {possession_path}. Run build_clean_lineup_net_rating.py first."
        )

    poss = pd.read_parquet(possession_path)
    missing = set(LINEUP_COLUMNS + ["off_points", "off_poss"]).difference(poss.columns)
    if missing:
        raise ValueError(f"{possession_path} is missing columns: {sorted(missing)}")

    for column in LINEUP_COLUMNS:
        poss[column] = pd.to_numeric(poss[column], errors="coerce").fillna(0).astype(
            "int64"
        )
    valid_lineups = poss[LINEUP_COLUMNS].gt(0).all(axis=1)
    valid_lineups &= (
        poss[[f"O{i}" for i in range(1, 6)]].nunique(axis=1).eq(5)
        & poss[[f"D{i}" for i in range(1, 6)]].nunique(axis=1).eq(5)
    )
    poss = poss[valid_lineups].copy()
    poss["off_lineup_key"] = lineup_keys(poss, "O")
    poss["def_lineup_key"] = lineup_keys(poss, "D")

    o_dpm = dpm_season.set_index("nba_id")["o_dpm"].to_dict()
    d_dpm = dpm_season.set_index("nba_id")["d_dpm"].to_dict()
    dpm_names = dpm_season.set_index("nba_id")["player_name"].astype(str).to_dict()

    offense_o_dpm, offense_o_coverage = sum_player_values(
        poss, [f"O{i}" for i in range(1, 6)], o_dpm
    )
    defense_d_dpm, defense_d_coverage = sum_player_values(
        poss, [f"D{i}" for i in range(1, 6)], d_dpm
    )
    poss["expected_offense"] = offense_o_dpm - defense_d_dpm
    poss["expected_defense"] = defense_d_dpm - offense_o_dpm
    poss["expected_coverage"] = offense_o_coverage + defense_d_coverage
    poss["expected_offense_weighted"] = poss["expected_offense"] * poss["off_poss"]
    poss["expected_defense_weighted"] = poss["expected_defense"] * poss["off_poss"]
    poss["coverage_weighted"] = poss["expected_coverage"] * poss["off_poss"]

    league_possessions = float(poss["off_poss"].sum())
    league_points = float(poss["off_points"].sum())
    league_ortg = 100.0 * league_points / league_possessions

    offense = (
        poss.groupby(["off_team_abbr", "off_lineup_key"], dropna=False)
        .agg(
            off_points=("off_points", "sum"),
            off_possessions=("off_poss", "sum"),
            expected_offense_sum=("expected_offense_weighted", "sum"),
            offense_coverage_sum=("coverage_weighted", "sum"),
        )
        .reset_index()
        .rename(
            columns={
                "off_team_abbr": "team",
                "off_lineup_key": "lineup_key",
            }
        )
    )
    defense = (
        poss.groupby(["def_team_abbr", "def_lineup_key"], dropna=False)
        .agg(
            points_allowed=("off_points", "sum"),
            def_possessions=("off_poss", "sum"),
            expected_defense_sum=("expected_defense_weighted", "sum"),
            defense_coverage_sum=("coverage_weighted", "sum"),
        )
        .reset_index()
        .rename(
            columns={
                "def_team_abbr": "team",
                "def_lineup_key": "lineup_key",
            }
        )
    )
    lineups = offense.merge(defense, on=["team", "lineup_key"], how="inner")
    lineups = lineups[
        lineups[["off_possessions", "def_possessions"]]
        .min(axis=1)
        .ge(minimum_possessions)
    ].copy()

    lineups["ortg"] = 100.0 * lineups["off_points"] / lineups["off_possessions"]
    lineups["drtg"] = (
        100.0 * lineups["points_allowed"] / lineups["def_possessions"]
    )
    lineups["net_rating"] = lineups["ortg"] - lineups["drtg"]
    lineups["offense_vs_average"] = lineups["ortg"] - league_ortg
    lineups["defense_vs_average"] = league_ortg - lineups["drtg"]
    lineups["expected_offense_vs_average"] = (
        lineups["expected_offense_sum"] / lineups["off_possessions"]
    )
    lineups["expected_defense_vs_average"] = (
        lineups["expected_defense_sum"] / lineups["def_possessions"]
    )
    lineups["offense_above_dpm"] = (
        lineups["offense_vs_average"] - lineups["expected_offense_vs_average"]
    )
    lineups["defense_above_dpm"] = (
        lineups["defense_vs_average"] - lineups["expected_defense_vs_average"]
    )
    lineups["net_above_dpm"] = (
        lineups["offense_above_dpm"] + lineups["defense_above_dpm"]
    )
    lineups["possessions"] = (
        (lineups["off_possessions"] + lineups["def_possessions"]) / 2.0
    ).round()
    lineups["dpm_coverage"] = np.minimum(
        lineups["offense_coverage_sum"] / (10.0 * lineups["off_possessions"]),
        lineups["defense_coverage_sum"] / (10.0 * lineups["def_possessions"]),
    )

    usage_numerator, raw_names = usage_numerators(raw_path)
    usage_denominator = on_court_offensive_possessions(poss)
    usage = (
        usage_numerator.reindex(usage_denominator.index, fill_value=0.0)
        .div(usage_denominator)
        .replace([np.inf, -np.inf], np.nan)
    )
    usage_lookup = usage.to_dict()
    name_lookup = {**raw_names, **{int(k): str(v) for k, v in dpm_names.items()}}

    player_lists: list[list[dict[str, object]]] = []
    lineup_names: list[str] = []
    combined_usage: list[float] = []
    usage_coverage: list[float] = []
    for key in lineups["lineup_key"]:
        player_ids = [int(value) for value in str(key).split("-")]
        player_rows = [
            {
                "id": player_id,
                "name": name_lookup.get(player_id, f"Player {player_id}"),
                "usage": usage_lookup.get(player_id),
            }
            for player_id in player_ids
        ]
        player_rows.sort(
            key=lambda player: (
                -(float(player["usage"]) if pd.notna(player["usage"]) else -1.0),
                str(player["name"]),
            )
        )
        player_lists.append(player_rows)
        lineup_names.append(" \u00b7 ".join(str(player["name"]) for player in player_rows))
        available_usage = [
            float(player["usage"])
            for player in player_rows
            if pd.notna(player["usage"])
        ]
        combined_usage.append(float(sum(available_usage)))
        usage_coverage.append(len(available_usage) / 5.0)

    lineups["players"] = player_lists
    lineups["lineup_name"] = lineup_names
    lineups["combined_usage"] = combined_usage
    lineups["usage_coverage"] = usage_coverage
    lineups["season"] = year
    lineups["season_label"] = season_label(year)
    lineups["id"] = (
        "history-"
        + lineups["season"].astype(str)
        + "-"
        + lineups["team"].astype(str).str.lower()
        + "-"
        + lineups["lineup_key"]
    )

    metadata = {
        "season": year,
        "season_label": season_label(year),
        "league_ortg": round(league_ortg, 4),
        "source_possessions": int(league_possessions),
        "source_points": round(league_points, 1),
        "qualifying_lineups": int(len(lineups)),
    }
    return lineups, metadata


def usage_defense_summary(lineups: pd.DataFrame) -> dict[str, object]:
    eligible = lineups[
        lineups["combined_usage"].notna()
        & lineups["defense_vs_average"].notna()
        & lineups["defense_above_dpm"].notna()
        & lineups["usage_coverage"].ge(1.0)
        & lineups["dpm_coverage"].ge(0.8)
    ].copy()
    weight = eligible["def_possessions"].clip(upper=1000)
    bins = [-np.inf, 0.9, 1.0, 1.1, 1.2, np.inf]
    labels = ["<90%", "90\u2013100%", "100\u2013110%", "110\u2013120%", "120%+"]
    eligible["usage_band"] = pd.cut(
        eligible["combined_usage"], bins=bins, labels=labels, right=False
    )

    band_rows: list[dict[str, object]] = []
    for label in labels:
        group = eligible[eligible["usage_band"].eq(label)]
        if group.empty:
            continue
        group_weight = group["def_possessions"].clip(upper=1000)
        band_rows.append(
            {
                "label": label,
                "lineups": int(len(group)),
                "possessions": int(group["def_possessions"].sum()),
                "defense_vs_average": round(
                    float(np.average(group["defense_vs_average"], weights=group_weight)),
                    4,
                ),
                "defense_above_dpm": round(
                    float(np.average(group["defense_above_dpm"], weights=group_weight)),
                    4,
                ),
            }
        )

    return {
        "eligible_lineups": int(len(eligible)),
        "weighting": "Defensive possessions, capped at 1,000 per lineup-season",
        "weighted_correlation_actual": round(
            weighted_correlation(
                eligible["combined_usage"], eligible["defense_vs_average"], weight
            ),
            4,
        ),
        "weighted_correlation_above_dpm": round(
            weighted_correlation(
                eligible["combined_usage"], eligible["defense_above_dpm"], weight
            ),
            4,
        ),
        "slope_actual_per_10_usage_points": round(
            weighted_slope_per_ten_usage(
                eligible["combined_usage"], eligible["defense_vs_average"], weight
            ),
            4,
        ),
        "slope_above_dpm_per_10_usage_points": round(
            weighted_slope_per_ten_usage(
                eligible["combined_usage"], eligible["defense_above_dpm"], weight
            ),
            4,
        ),
        "bands": band_rows,
    }


def round_for_export(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        frame[column] = frame[column].astype(float).round(4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-year", type=int, default=14)
    parser.add_argument("--last-year", type=int, default=26)
    parser.add_argument("--minimum-possessions", type=int, default=200)
    parser.add_argument("--dpm-path", type=Path, default=DEFAULT_DPM_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    first_year = season_end_year(args.first_year)
    last_year = season_end_year(args.last_year)
    dpm = load_dpm(args.dpm_path, first_year, last_year)

    season_frames: list[pd.DataFrame] = []
    season_metadata: list[dict[str, object]] = []
    for short_year in range(args.first_year, args.last_year + 1):
        year = season_end_year(short_year)
        frame, metadata = aggregate_season(
            short_year,
            dpm[dpm["season"].eq(year)].copy(),
            args.minimum_possessions,
        )
        season_frames.append(frame)
        season_metadata.append(metadata)
        print(
            f"{metadata['season_label']}: {metadata['qualifying_lineups']:,} "
            f"lineups from {metadata['source_possessions']:,} possessions"
        )

    lineups = pd.concat(season_frames, ignore_index=True)
    export_columns = [
        "id",
        "season",
        "season_label",
        "team",
        "lineup_name",
        "players",
        "possessions",
        "off_possessions",
        "def_possessions",
        "ortg",
        "drtg",
        "net_rating",
        "offense_vs_average",
        "defense_vs_average",
        "expected_offense_vs_average",
        "expected_defense_vs_average",
        "offense_above_dpm",
        "defense_above_dpm",
        "net_above_dpm",
        "combined_usage",
        "usage_coverage",
        "dpm_coverage",
    ]
    lineups = lineups[export_columns].copy()
    round_for_export(
        lineups,
        [
            "ortg",
            "drtg",
            "net_rating",
            "offense_vs_average",
            "defense_vs_average",
            "expected_offense_vs_average",
            "expected_defense_vs_average",
            "offense_above_dpm",
            "defense_above_dpm",
            "net_above_dpm",
            "combined_usage",
            "usage_coverage",
            "dpm_coverage",
        ],
    )
    lineups = lineups.sort_values(
        ["season", "net_rating", "possessions"], ascending=[False, False, False]
    )
    if not np.isfinite(
        lineups.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    ).all():
        raise ValueError("Historical lineup export contains non-finite numeric values")

    payload = {
        "meta": {
            "title": "Five-Man Fit Lab historical lineup performance",
            "first_season": first_year,
            "last_season": last_year,
            "season_count": last_year - first_year + 1,
            "minimum_possessions_each_side": args.minimum_possessions,
            "lineup_count": int(len(lineups)),
            "performance_source": "Clean raw-scoreboard lineup possessions",
            "talent_control": "Same-season DARKO oDPM/dDPM plus opponent quality",
            "usage_definition": (
                "(FGA + 0.44 × FTA + TOV) divided by team possessions while on court"
            ),
            "scope_note": (
                "2014+ is the feature-complete modern-lineup window. This is a "
                "descriptive DPM-adjusted companion, not a recreation of the lost "
                "EPM training pipeline."
            ),
            "seasons": season_metadata,
        },
        "usage_defense": usage_defense_summary(lineups),
        "lineups": lineups.to_dict(orient="records"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Wrote {len(lineups):,} lineups to {args.output}")


if __name__ == "__main__":
    main()
