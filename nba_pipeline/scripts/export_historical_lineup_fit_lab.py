#!/usr/bin/env python3
"""Export the 2014+ PBP mechanism history for Five-Man Fit Lab.

This product lane uses no external player rating. Actual ORtg/DRtg comes from
the clean raw-scoreboard lineup surface. The mechanism ledger comes from the
repo-owned first-chance/second-chance possession decomposition and closes
exactly in RAPM-style expected-FT scoring:

shot access + shot selection + shotmaking + free throws + second chance.
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
REPO_DIR = PIPELINE_DIR.parent
DEFAULT_OUTPUT = REPO_DIR / "five-man-fit-lab" / "data" / "historical_lineups.json"
DEFAULT_FIT = (
    PIPELINE_DIR / "results" / "five_man_fit" / "historical_epm_fit.parquet"
)
DEFAULT_FIT_REPORT = (
    PIPELINE_DIR / "results" / "five_man_fit" / "historical_epm_fit_report.json"
)
LINEUP_COLUMNS = [f"{side}{idx}" for side in ("O", "D") for idx in range(1, 6)]

PARENT_COMPONENTS = {
    "shot_access": ["FC_EFG_BASELINE_Diff"],
    "shot_selection": [
        "FC_RIM_EFG_FREQ_Diff",
        "FC_MID_EFG_FREQ_Diff",
        "FC_THREE_EFG_FREQ_Diff",
    ],
    "shotmaking": [
        "FC_RIM_EFG_FG_Diff",
        "FC_MID_EFG_FG_Diff",
        "FC_THREE_EFG_FG_Diff",
    ],
    "free_throws": ["FC_FT_Diff"],
    "second_chance": ["Second_Chance_Diff"],
}

ATOMIC_COMPONENTS = {
    "rim_frequency": "FC_RIM_EFG_FREQ_Diff",
    "rim_shotmaking": "FC_RIM_EFG_FG_Diff",
    "mid_frequency": "FC_MID_EFG_FREQ_Diff",
    "mid_shotmaking": "FC_MID_EFG_FG_Diff",
    "three_frequency": "FC_THREE_EFG_FREQ_Diff",
    "three_shotmaking": "FC_THREE_EFG_FG_Diff",
    "ft_frequency": "FC_FT_FREQ_Diff",
    "ft_conversion": "FC_FT_SEVERITY_Diff",
}


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


def valid_lineup_rows(frame: pd.DataFrame) -> pd.Series:
    valid = frame[LINEUP_COLUMNS].gt(0).all(axis=1)
    valid &= frame[[f"O{i}" for i in range(1, 6)]].nunique(axis=1).eq(5)
    valid &= frame[[f"D{i}" for i in range(1, 6)]].nunique(axis=1).eq(5)
    return valid


def weighted_correlation(x: pd.Series, y: pd.Series, weight: pd.Series) -> float:
    xv = x.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    w = weight.to_numpy(dtype=float)
    x_mean = np.average(xv, weights=w)
    y_mean = np.average(yv, weights=w)
    covariance = np.average((xv - x_mean) * (yv - y_mean), weights=w)
    denominator = np.sqrt(
        np.average((xv - x_mean) ** 2, weights=w)
        * np.average((yv - y_mean) ** 2, weights=w)
    )
    return float(covariance / denominator) if denominator > 0 else 0.0


def weighted_slope_per_ten_usage(
    x: pd.Series, y: pd.Series, weight: pd.Series
) -> float:
    xv = x.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    w = weight.to_numpy(dtype=float)
    x_mean = np.average(xv, weights=w)
    y_mean = np.average(yv, weights=w)
    denominator = np.sum(w * (xv - x_mean) ** 2)
    if denominator <= 0:
        return 0.0
    return float(
        0.10 * np.sum(w * (xv - x_mean) * (yv - y_mean)) / denominator
    )


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
    lookup = (
        names.drop_duplicates("player1_id", keep="last")
        .set_index("player1_id")["player1_name"]
        .to_dict()
    )
    return numerator, {int(key): str(value) for key, value in lookup.items()}


def on_court_offensive_possessions(possessions: pd.DataFrame) -> pd.Series:
    counted = possessions.loc[
        possessions["off_poss"].gt(0), ["off_poss", *[f"O{i}" for i in range(1, 6)]]
    ]
    long = counted.melt(
        id_vars="off_poss",
        value_vars=[f"O{i}" for i in range(1, 6)],
        value_name="player_id",
    )
    long["player_id"] = pd.to_numeric(long["player_id"], errors="coerce")
    long = long[long["player_id"].gt(0)].copy()
    long["player_id"] = long["player_id"].astype("int64")
    return long.groupby("player_id")["off_poss"].sum()


def assert_parallel_rows(parent: pd.DataFrame, child: pd.DataFrame, label: str) -> None:
    keys = ["game_id", "event_num", *LINEUP_COLUMNS]
    if len(parent) != len(child):
        raise ValueError(f"{label} row count differs from FIRST_CHANCE")
    left = parent[keys].fillna(0).astype(str).to_numpy()
    right = child[keys].fillna(0).astype(str).to_numpy()
    if not np.array_equal(left, right):
        raise ValueError(f"{label} possession keys differ from FIRST_CHANCE")


def load_mechanism_rows(short_year: int) -> pd.DataFrame:
    processed = PIPELINE_DIR / "processed"
    first = pd.read_parquet(processed / f"FIRST_CHANCE{short_year}.parquet")
    second = pd.read_parquet(
        processed / f"SECOND_CHANCE_CLEAN{short_year}.parquet"
    )
    assists = pd.read_parquet(processed / f"ASSIST_POINTS{short_year}.parquet")
    rim_assists = pd.read_parquet(processed / f"RIM_ASSIST{short_year}.parquet")
    assert_parallel_rows(first, second, "SECOND_CHANCE_CLEAN")
    assert_parallel_rows(first, assists, "ASSIST_POINTS")
    assert_parallel_rows(first, rim_assists, "RIM_ASSIST")

    rows = first.copy()
    rows["Second_Chance_Diff"] = pd.to_numeric(
        second["Off_Diff"], errors="coerce"
    ).fillna(0.0)
    rows["Assist_Points"] = pd.to_numeric(
        assists["Assist_Points"], errors="coerce"
    ).fillna(0.0)
    rows["Rim_Assists"] = pd.to_numeric(
        rim_assists["Is_Rim_Assist"], errors="coerce"
    ).fillna(0.0)
    rows["Transition"] = pd.to_numeric(
        rows["Is_FC_Transition_Possession"], errors="coerce"
    ).fillna(0.0)
    rows["Turnover"] = pd.to_numeric(rows["Is_Turnover"], errors="coerce").fillna(
        0.0
    )
    for column in set(
        [item for values in PARENT_COMPONENTS.values() for item in values]
        + list(ATOMIC_COMPONENTS.values())
    ):
        rows[column] = pd.to_numeric(rows[column], errors="coerce").fillna(0.0)
    for name, columns in PARENT_COMPONENTS.items():
        rows[name] = rows[columns].sum(axis=1)

    component_total = rows[list(PARENT_COMPONENTS)].sum(axis=1)
    scoring_total = (
        pd.to_numeric(rows["Net_Diff"], errors="coerce").fillna(0.0)
        + rows["Second_Chance_Diff"]
    )
    gap = float((component_total - scoring_total).abs().max())
    if gap > 1e-9:
        raise ValueError(f"Mechanism row identity failed for {short_year}: {gap}")

    for column in LINEUP_COLUMNS:
        rows[column] = pd.to_numeric(rows[column], errors="coerce").fillna(0).astype(
            "int64"
        )
    rows = rows[valid_lineup_rows(rows)].copy()
    rows["off_lineup_key"] = lineup_keys(rows, "O")
    rows["def_lineup_key"] = lineup_keys(rows, "D")
    return rows


def aggregate_mechanisms(rows: pd.DataFrame) -> pd.DataFrame:
    value_columns = [
        *PARENT_COMPONENTS,
        *ATOMIC_COMPONENTS.values(),
        "Turnover",
        "Transition",
        "Assist_Points",
        "Rim_Assists",
    ]
    league = rows[value_columns].mean()
    offense = rows.groupby("off_lineup_key")[value_columns].mean()
    defense = rows.groupby("def_lineup_key")[value_columns].mean()
    offense["mechanism_possessions_off"] = rows.groupby("off_lineup_key").size()
    defense["mechanism_possessions_def"] = rows.groupby("def_lineup_key").size()

    out = offense[["mechanism_possessions_off"]].join(
        defense[["mechanism_possessions_def"]], how="inner"
    )
    for name in PARENT_COMPONENTS:
        out[f"off_{name}"] = 100.0 * (offense[name] - league[name])
        out[f"def_{name}"] = 100.0 * (league[name] - defense[name])
    for name, column in ATOMIC_COMPONENTS.items():
        out[f"off_{name}"] = 100.0 * (offense[column] - league[column])
        out[f"def_{name}"] = 100.0 * (league[column] - defense[column])

    out["mechanism_offense_vs_average"] = out[
        [f"off_{name}" for name in PARENT_COMPONENTS]
    ].sum(axis=1)
    out["mechanism_defense_vs_average"] = out[
        [f"def_{name}" for name in PARENT_COMPONENTS]
    ].sum(axis=1)
    out["mechanism_net_vs_average"] = (
        out["mechanism_offense_vs_average"]
        + out["mechanism_defense_vs_average"]
    )
    out["off_turnover_rate"] = 100.0 * offense["Turnover"]
    out["def_forced_turnover_rate"] = 100.0 * defense["Turnover"]
    out["off_transition_rate"] = 100.0 * offense["Transition"]
    out["def_transition_rate_allowed"] = 100.0 * defense["Transition"]
    out["off_assist_points_per100"] = 100.0 * offense["Assist_Points"]
    out["def_assist_points_allowed_per100"] = 100.0 * defense["Assist_Points"]
    out["off_rim_assists_per100"] = 100.0 * offense["Rim_Assists"]
    out["def_rim_assists_allowed_per100"] = 100.0 * defense["Rim_Assists"]

    off_gap = (
        out["mechanism_offense_vs_average"]
        - out[[f"off_{name}" for name in PARENT_COMPONENTS]].sum(axis=1)
    ).abs().max()
    def_gap = (
        out["mechanism_defense_vs_average"]
        - out[[f"def_{name}" for name in PARENT_COMPONENTS]].sum(axis=1)
    ).abs().max()
    if max(float(off_gap), float(def_gap)) > 1e-9:
        raise ValueError("Aggregated mechanism identity failed")
    out.index.name = "lineup_key"
    return out.reset_index()


def aggregate_season(
    short_year: int,
    minimum_possessions: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    year = season_end_year(short_year)
    possession_path = (
        PIPELINE_DIR
        / "processed"
        / f"LINEUP_NET_RATING_POSSESSIONS{short_year}.parquet"
    )
    raw_path = PIPELINE_DIR / "raw_data" / f"NBA{short_year}.parquet"
    poss = pd.read_parquet(possession_path)
    for column in LINEUP_COLUMNS:
        poss[column] = pd.to_numeric(poss[column], errors="coerce").fillna(0).astype(
            "int64"
        )
    poss = poss[valid_lineup_rows(poss)].copy()
    poss["off_lineup_key"] = lineup_keys(poss, "O")
    poss["def_lineup_key"] = lineup_keys(poss, "D")

    league_possessions = float(poss["off_poss"].sum())
    league_points = float(poss["off_points"].sum())
    league_ortg = 100.0 * league_points / league_possessions
    offense = (
        poss.groupby(["off_team_abbr", "off_lineup_key"], dropna=False)
        .agg(
            off_points=("off_points", "sum"),
            off_possessions=("off_poss", "sum"),
        )
        .reset_index()
        .rename(columns={"off_team_abbr": "team", "off_lineup_key": "lineup_key"})
    )
    defense = (
        poss.groupby(["def_team_abbr", "def_lineup_key"], dropna=False)
        .agg(
            points_allowed=("off_points", "sum"),
            def_possessions=("off_poss", "sum"),
        )
        .reset_index()
        .rename(columns={"def_team_abbr": "team", "def_lineup_key": "lineup_key"})
    )
    lineups = offense.merge(defense, on=["team", "lineup_key"], how="inner")
    lineups = lineups[
        lineups[["off_possessions", "def_possessions"]]
        .min(axis=1)
        .ge(minimum_possessions)
    ].copy()
    lineups["ortg"] = 100.0 * lineups["off_points"] / lineups["off_possessions"]
    lineups["drtg"] = 100.0 * lineups["points_allowed"] / lineups["def_possessions"]
    lineups["net_rating"] = lineups["ortg"] - lineups["drtg"]
    lineups["offense_vs_average"] = lineups["ortg"] - league_ortg
    lineups["defense_vs_average"] = league_ortg - lineups["drtg"]
    lineups["possessions"] = (
        (lineups["off_possessions"] + lineups["def_possessions"]) / 2.0
    ).round()

    mechanisms = aggregate_mechanisms(load_mechanism_rows(short_year))
    lineups = lineups.merge(mechanisms, on="lineup_key", how="inner")

    usage_numerator, name_lookup = usage_numerators(raw_path)
    usage_denominator = on_court_offensive_possessions(poss)
    usage = usage_numerator.reindex(
        usage_denominator.index, fill_value=0.0
    ).div(usage_denominator)
    usage_lookup = usage.to_dict()

    players_out: list[list[dict[str, object]]] = []
    lineup_names: list[str] = []
    combined_usage: list[float] = []
    top2_usage: list[float] = []
    usage_hhi: list[float] = []
    for key in lineups["lineup_key"]:
        player_ids = [int(value) for value in str(key).split("-")]
        player_rows = [
            {
                "id": player_id,
                "name": name_lookup.get(player_id, f"Player {player_id}"),
                "usage": float(usage_lookup.get(player_id, 0.0)),
            }
            for player_id in player_ids
        ]
        player_rows.sort(key=lambda player: (-float(player["usage"]), str(player["name"])))
        usages = [float(player["usage"]) for player in player_rows]
        players_out.append(player_rows)
        lineup_names.append(" \u00b7 ".join(str(player["name"]) for player in player_rows))
        combined_usage.append(sum(usages))
        top2_usage.append(sum(sorted(usages, reverse=True)[:2]))
        usage_hhi.append(
            sum((value / max(sum(usages), 1e-9)) ** 2 for value in usages)
        )

    lineups["players"] = players_out
    lineups["lineup_name"] = lineup_names
    lineups["combined_usage"] = combined_usage
    lineups["top2_usage"] = top2_usage
    lineups["usage_hhi"] = usage_hhi
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
        "qualifying_lineups": int(len(lineups)),
    }
    return lineups, metadata


def role_defense_summary(lineups: pd.DataFrame) -> dict[str, object]:
    eligible = lineups[
        lineups["usage_top2_share"].notna()
        & lineups["def_target"].notna()
    ].copy()
    weight = eligible["def_possessions"].clip(upper=1000)
    bins = [-np.inf, 0.46, 0.50, 0.54, 0.58, np.inf]
    labels = ["<46%", "46\u201350%", "50\u201354%", "54\u201358%", "58%+"]
    eligible["top2_usage_band"] = pd.cut(
        eligible["usage_top2_share"], bins=bins, labels=labels, right=False
    )
    bands = []
    for label in labels:
        group = eligible[eligible["top2_usage_band"].eq(label)]
        if group.empty:
            continue
        group_weight = group["def_possessions"].clip(upper=1000)
        bands.append(
            {
                "label": label,
                "lineups": int(len(group)),
                "possessions": int(group["def_possessions"].sum()),
                "defense_above_epm": round(
                    float(np.average(group["def_target"], weights=group_weight)),
                    4,
                ),
                "defense_vs_average": round(
                    float(np.average(group["defense_vs_average"], weights=group_weight)),
                    4,
                ),
            }
        )
    return {
        "eligible_lineups": int(len(eligible)),
        "weighting": "Defensive possessions, capped at 1,000 per lineup-season",
        "top2_usage_share_definition": (
            "Two largest normalized Dunks & Threes p_usg shares; all five "
            "player shares sum to 100%."
        ),
        "weighted_correlation_top2_usage_vs_defense_above_epm": round(
            weighted_correlation(
                eligible["usage_top2_share"], eligible["def_target"], weight
            ),
            4,
        ),
        "slope_defense_above_epm_per_10_top2_usage_points": round(
            weighted_slope_per_ten_usage(
                eligible["usage_top2_share"], eligible["def_target"], weight
            ),
            4,
        ),
        "weighted_correlation_role_demand_vs_defense_above_epm": round(
            weighted_correlation(
                eligible["epm_usage_demand_sum"], eligible["def_target"], weight
            ),
            4,
        ),
        "bands": bands,
    }


def parse_model_players(value: str) -> list[dict[str, object]]:
    players = json.loads(value)
    output = [
        {
            "id": int(player["id"]),
            "name": str(player["name"]),
            "epm_predicted_usage": float(player["epm_predicted_usage"]),
            "usage": float(player["normalized_lineup_usage"]),
            "onball_time": float(player["onball_time"]),
            "potential_assists_per100": float(
                player["potential_assists_per100"]
            ),
            "oepm": float(player["oepm"]),
            "depm": float(player["depm"]),
        }
        for player in players
    ]
    output.sort(key=lambda player: (-float(player["usage"]), str(player["name"])))
    return output


def merge_epm_fit(lineups: pd.DataFrame, fit_path: Path) -> pd.DataFrame:
    fit = pd.read_parquet(fit_path)
    lineups = lineups.drop(
        columns=["combined_usage", "top2_usage", "usage_hhi"],
        errors="ignore",
    )
    contribution_columns = [
        column for column in fit.columns if column.startswith("off_contribution_")
    ]
    model_columns = [
        "season",
        "team",
        "lineup_key",
        "players_json",
        "epm_usage_demand_sum",
        "normalized_usage_sum",
        "role_demand_surplus",
        "role_demand_overload",
        "role_demand_shortfall",
        "usage_top1_share",
        "usage_top2_share",
        "usage_hhi",
        "usage_top2_overlap",
        "onball_sum",
        "onball_top2",
        "onball_hhi",
        "onball_top2_overlap",
        "potential_assist_sum",
        "potential_assist_top2",
        "potential_assist_hhi",
        "potential_assist_overlap",
        "passer_finisher_fit",
        "rim_pressure_sum",
        "ft_pressure_sum",
        "three_volume_sum",
        "turnover_burden_sum",
        "usage_weighted_tspct",
        "lineup_oepm_sum",
        "lineup_depm_sum",
        "expected_offense_vs_average",
        "expected_defense_vs_average",
        "off_target",
        "def_target",
        "off_mechanism_fit",
        "off_lineup_specific_fit",
        "off_total_fit",
        "off_domain_usage",
        "off_domain_onball",
        "off_domain_playmaking",
        "def_mechanism_fit",
        "def_lineup_specific_fit",
        "def_total_fit",
        "modeled_neutral_offense",
        "modeled_neutral_defense",
        "modeled_neutral_net",
        "off_fit_status",
        "def_fit_status",
        *contribution_columns,
    ]
    if fit.duplicated(["season", "team", "lineup_key"]).any():
        raise ValueError("Historical EPM fit contains duplicate lineup-season rows")
    merged = lineups.merge(
        fit[model_columns],
        on=["season", "team", "lineup_key"],
        how="inner",
        validate="one_to_one",
    )
    merged["players"] = merged["players_json"].map(parse_model_players)
    merged["lineup_name"] = merged["players"].map(
        lambda players: " \u00b7 ".join(str(player["name"]) for player in players)
    )
    merged = merged.drop(columns=["players_json"])
    return merged


def round_for_export(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        frame[column] = frame[column].astype(float).round(4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-year", type=int, default=14)
    parser.add_argument("--last-year", type=int, default=26)
    parser.add_argument("--minimum-possessions", type=int, default=200)
    parser.add_argument("--fit", type=Path, default=DEFAULT_FIT)
    parser.add_argument("--fit-report", type=Path, default=DEFAULT_FIT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = []
    season_meta = []
    for short_year in range(args.first_year, args.last_year + 1):
        frame, metadata = aggregate_season(short_year, args.minimum_possessions)
        frames.append(frame)
        season_meta.append(metadata)
        print(
            f"{metadata['season_label']}: {metadata['qualifying_lineups']:,} "
            "lineups"
        )
    lineups = merge_epm_fit(pd.concat(frames, ignore_index=True), args.fit)
    fit_report = json.loads(args.fit_report.read_text(encoding="utf-8"))
    base_columns = [
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
        "epm_usage_demand_sum",
        "normalized_usage_sum",
        "role_demand_surplus",
        "role_demand_overload",
        "role_demand_shortfall",
        "usage_top1_share",
        "usage_top2_share",
        "usage_hhi",
        "usage_top2_overlap",
        "onball_sum",
        "onball_top2",
        "onball_hhi",
        "onball_top2_overlap",
        "potential_assist_sum",
        "potential_assist_top2",
        "potential_assist_hhi",
        "potential_assist_overlap",
        "passer_finisher_fit",
        "rim_pressure_sum",
        "ft_pressure_sum",
        "three_volume_sum",
        "turnover_burden_sum",
        "usage_weighted_tspct",
        "lineup_oepm_sum",
        "lineup_depm_sum",
        "expected_offense_vs_average",
        "expected_defense_vs_average",
        "off_target",
        "def_target",
        "off_mechanism_fit",
        "off_lineup_specific_fit",
        "off_total_fit",
        "off_domain_usage",
        "off_domain_onball",
        "off_domain_playmaking",
        "def_mechanism_fit",
        "def_lineup_specific_fit",
        "def_total_fit",
        "modeled_neutral_offense",
        "modeled_neutral_defense",
        "modeled_neutral_net",
        "off_fit_status",
        "def_fit_status",
        "mechanism_possessions_off",
        "mechanism_possessions_def",
        "mechanism_offense_vs_average",
        "mechanism_defense_vs_average",
        "mechanism_net_vs_average",
        "off_turnover_rate",
        "def_forced_turnover_rate",
        "off_transition_rate",
        "def_transition_rate_allowed",
        "off_assist_points_per100",
        "def_assist_points_allowed_per100",
        "off_rim_assists_per100",
        "def_rim_assists_allowed_per100",
    ]
    mechanism_columns = [
        *[f"{side}_{name}" for side in ("off", "def") for name in PARENT_COMPONENTS],
        *[f"{side}_{name}" for side in ("off", "def") for name in ATOMIC_COMPONENTS],
    ]
    contribution_columns = sorted(
        column
        for column in lineups.columns
        if column.startswith("off_contribution_")
    )
    lineups = lineups[
        base_columns + contribution_columns + mechanism_columns
    ].copy()
    numeric = [
        column
        for column in lineups.select_dtypes(include=[np.number]).columns
        if column not in {"season"}
    ]
    round_for_export(lineups, numeric)
    lineups = lineups.sort_values(
        ["season", "net_rating", "possessions"], ascending=[False, False, False]
    )
    if not np.isfinite(
        lineups.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    ).all():
        raise ValueError("Historical export contains non-finite values")

    payload = {
        "meta": {
            "title": "Five-Man Fit Lab PBP mechanism history",
            "first_season": season_end_year(args.first_year),
            "last_season": season_end_year(args.last_year),
            "season_count": args.last_year - args.first_year + 1,
            "minimum_possessions_each_side": args.minimum_possessions,
            "lineup_count": int(len(lineups)),
            "model_source": (
                "Season-end Dunks & Threes EPM/E-Skills plus historical "
                "on-ball time and potential assists"
            ),
            "model_contract": (
                "neutral EPM expectation + forward-selected offensive usage "
                "fit + heavily shrunk exact-five evidence"
            ),
            "usage_definition": (
                "Dunks & Threes p_usg normalized within each five-player "
                "lineup; displayed player allocations sum to 100%."
            ),
            "role_demand_definition": (
                "The raw sum of five EPM p_usg predictions is retained as "
                "role demand, never lineup usage. Demand below 92% receives "
                "a monotone creation-capacity penalty."
            ),
            "offense_validation": {
                "selected_feature_groups": fit_report["offense"][
                    "published_feature_groups"
                ],
                "fixed_raw_coefficients": fit_report["offense"][
                    "fixed_raw_coefficients"
                ],
                "development": fit_report["offense"]["development_score"],
                "confirmation": fit_report["offense"]["confirmation_score"],
                "identity_confirmation": fit_report["offense"][
                    "confirmation_identity_score"
                ],
            },
            "defense_validation": {
                "status": fit_report["defense"]["status"],
                "development": fit_report["defense"]["development_score"],
                "confirmation": fit_report["defense"]["confirmation_score"],
                "result": fit_report["defense"]["usage_onball_result"],
            },
            "performance_source": "Clean raw-scoreboard lineup possessions",
            "mechanism_source": (
                "Repo-owned FIRST_CHANCE + SECOND_CHANCE_CLEAN possession decomposition"
            ),
            "mechanism_contract": (
                "shot access + shot selection + shotmaking + free throws + "
                "second chance = RAPM-style lineup scoring versus season average"
            ),
            "scope_note": (
                "2014+ is the feature-complete modern-lineup window. This is "
                "a new historical refit from source data, not coefficient "
                "recovery from the published five-year output table."
            ),
            "seasons": season_meta,
        },
        "usage_defense": role_defense_summary(lineups),
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
