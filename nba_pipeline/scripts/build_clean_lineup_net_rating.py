#!/usr/bin/env python3
"""
Build a raw-scoreboard possession parquet for lineup WOWY/net rating.

This is a duplicate path from the RAPM processed parquet surface. It keeps
actual scoreboard points and uses period boundaries as hard possession split
points so end-of-quarter heaves and final empty possessions are not merged into
the next period.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from process_rapm_blocks.common import (
    TEAM_ABBREVIATION_LABELS,
    _base_processing,
    _normalize_team_label,
    prepare_standard_possession_df,
)
from lineup_stats import infer_game_team_map


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
RAW_DATA_DIR = PIPELINE_DIR / "raw_data"
PROCESSED_DIR = PIPELINE_DIR / "processed"


def season_raw_path(year: int, season_type: str) -> Path:
    suffix = "_PS" if season_type.upper() == "PS" else ""
    return RAW_DATA_DIR / f"NBA{year}{suffix}.parquet"


def default_output_path(year: int, season_type: str) -> Path:
    suffix = "_PS" if season_type.upper() == "PS" else ""
    return PROCESSED_DIR / f"LINEUP_NET_RATING_POSSESSIONS{year}{suffix}.parquet"


def score_delta(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").ffill().fillna(0)
    return values.diff().fillna(values).astype(float)


def event_text(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["home_description"].fillna("").astype(str)
        + " "
        + frame["visitor_description"].fillna("").astype(str)
        + " "
        + frame["neutral_description"].fillna("").astype(str)
    ).str.strip()


def labels_for_abbr(abbr: object) -> set[str]:
    if pd.isna(abbr):
        return set()
    text = str(abbr).strip().upper()
    labels = {text}
    labels.update(str(label).upper() for label in TEAM_ABBREVIATION_LABELS.get(text, []))
    return {label for label in labels if label}


def text_matches_label(text: str, labels: set[str]) -> bool:
    normalized = _normalize_team_label(text)
    for label in labels:
        normalized_label = _normalize_team_label(label)
        if normalized_label and re.search(rf"\b{re.escape(normalized_label)}\b", normalized):
            return True
    return False


def infer_segment_offense(group: pd.DataFrame, home_abbr: object, away_abbr: object) -> str:
    offense_source = group["TeamOnOffense"].mask(group["TeamOnOffense"].eq(""), np.nan).dropna()
    if not offense_source.empty:
        return str(offense_source.iloc[-1])

    text = " ".join(event_text(group).tolist())
    home_match = text_matches_label(text, labels_for_abbr(home_abbr))
    away_match = text_matches_label(text, labels_for_abbr(away_abbr))
    if home_match and not away_match:
        return "Home"
    if away_match and not home_match:
        return "Away"
    return ""


def lineup_values(row: pd.Series, side: str) -> list[int]:
    prefix = "h" if side == "Home" else "a"
    values: list[int] = []
    for idx in range(1, 6):
        value = pd.to_numeric(row.get(f"{prefix}{idx}"), errors="coerce")
        values.append(int(value) if pd.notna(value) and int(value) > 0 else 0)
    return values


def action_scoring_side(row: pd.Series) -> str:
    home_value = pd.to_numeric(row.get("home_action_points"), errors="coerce")
    away_value = pd.to_numeric(row.get("away_action_points"), errors="coerce")
    home_points = float(home_value) if pd.notna(home_value) else 0.0
    away_points = float(away_value) if pd.notna(away_value) else 0.0
    if home_points > 0 and away_points == 0:
        return "Home"
    if away_points > 0 and home_points == 0:
        return "Away"
    return ""


def is_admin_free_throw_text(text: object) -> bool:
    return bool(re.search(r"\b(Technical|Flagrant|Defensive 3 Seconds|Clear Path)\b", str(text), flags=re.I))


def base_row(
    gid: object,
    year: int,
    season_type: str,
    group: pd.DataFrame,
    row_source: pd.Series,
    poss_group: object,
    offense: str,
    home_points: float,
    away_points: float,
    off_points: float,
    def_points: float,
    off_poss: int,
    def_poss: int,
    row_type: str,
) -> dict[str, object]:
    defense = "Away" if offense == "Home" else "Home"
    off_lineup = lineup_values(row_source, offense)
    def_lineup = lineup_values(row_source, defense)
    row = {
        "game_id": str(gid),
        "season": int(year),
        "season_type": season_type.upper(),
        "period": int(row_source["period"]),
        "net_poss_group": int(poss_group),
        "start_event_num": int(group["event_num"].iloc[0]),
        "end_event_num": int(group["event_num"].iloc[-1]),
        "start_time": str(group["time_quarter"].iloc[0]),
        "end_time": str(group["time_quarter"].iloc[-1]),
        "poss_offense": offense,
        "off_team_abbr": row_source["home_team_abbr"] if offense == "Home" else row_source["away_team_abbr"],
        "def_team_abbr": row_source["away_team_abbr"] if offense == "Home" else row_source["home_team_abbr"],
        "home_points": home_points,
        "away_points": away_points,
        "off_points": off_points,
        "def_points": def_points,
        "off_poss": off_poss,
        "def_poss": def_poss,
        "event_count": int(len(group)),
        "period_count": int(group["period"].nunique()),
        "terminal_event_type": str(row_source["event_type"]),
        "terminal_event_text": str(row_source["event_text"]),
        "row_type": row_type,
    }
    for idx, value in enumerate(off_lineup, start=1):
        row[f"O{idx}"] = value
    for idx, value in enumerate(def_lineup, start=1):
        row[f"D{idx}"] = value
    return row


def build_possessions(year: int, season_type: str, game_id: str | None = None) -> pd.DataFrame:
    df = _base_processing(season_raw_path(year, season_type))
    if df is None or df.empty:
        raise RuntimeError("Raw play-by-play could not be loaded")

    if game_id:
        normalized_game_id = str(game_id).strip().zfill(10)
        df = df[df["game_id"].astype(str).str.zfill(10).eq(normalized_game_id)].copy()
        if df.empty:
            raise ValueError(f"Game {normalized_game_id} not found")

    team_map = infer_game_team_map(df)
    df = df.merge(team_map, on="game_id", how="left")
    parsed, _ = prepare_standard_possession_df(df, "CLEAN_LINEUP_NET_RATING")
    parsed = parsed.reset_index(drop=True)

    parsed["home_score_delta"] = parsed.groupby("game_id")["home_score"].transform(score_delta)
    parsed["away_score_delta"] = parsed.groupby("game_id")["away_score"].transform(score_delta)
    parsed["home_action_points"] = pd.to_numeric(parsed["Home_Action_Score"], errors="coerce").fillna(0).astype(float)
    parsed["away_action_points"] = pd.to_numeric(parsed["Away_Action_Score"], errors="coerce").fillna(0).astype(float)
    parsed["action_scoring_side"] = np.select(
        [
            parsed["home_action_points"].gt(0) & parsed["away_action_points"].eq(0),
            parsed["away_action_points"].gt(0) & parsed["home_action_points"].eq(0),
        ],
        ["Home", "Away"],
        default="",
    )
    parsed["event_text"] = event_text(parsed)
    parsed["is_standard_eop"] = parsed["End_of_Possession"].fillna(False).astype(bool)
    parsed["is_period_boundary"] = parsed["event_type"].eq("EndOfPeriod")
    parsed["net_rating_eop"] = parsed["is_standard_eop"] | parsed["is_period_boundary"]
    parsed["net_poss_group"] = parsed.groupby("game_id")["net_rating_eop"].transform(
        lambda value: value.shift(1, fill_value=False).cumsum()
    )
    parsed["is_live_net_rating_event"] = (
        parsed["event_type"].isin(["MAKE", "MISS", "FreeThrow", "Turnover"])
        | parsed["event_text"].str.contains(r"\bHeave\b", case=False, regex=True, na=False)
        | (parsed["home_score_delta"] + parsed["away_score_delta"]).ne(0)
    )
    parsed["is_admin_free_throw"] = parsed["event_type"].eq("FreeThrow") & parsed["event_text"].map(
        is_admin_free_throw_text
    )

    keys = ["game_id", "net_poss_group"]
    parsed["_segment_offense"] = parsed["TeamOnOffense"].replace("", np.nan)
    parsed["_has_action_score"] = (
        parsed["home_action_points"] + parsed["away_action_points"]
    ).ne(0)

    grouped = parsed.groupby(keys, sort=False, dropna=False)
    summary = grouped.agg(
        start_event_num=("event_num", "first"),
        end_event_num=("event_num", "last"),
        start_time=("time_quarter", "first"),
        end_time=("time_quarter", "last"),
        event_count=("event_num", "size"),
        period_count=("period", "nunique"),
        has_live_event=("is_live_net_rating_event", "any"),
        has_score=("_has_action_score", "any"),
        home_points_total=("home_action_points", "sum"),
        away_points_total=("away_action_points", "sum"),
        poss_offense=("_segment_offense", "last"),
    ).reset_index()

    terminal_columns = [
        *keys,
        "period",
        "event_type",
        "event_text",
        "home_team_abbr",
        "away_team_abbr",
        *[f"h{i}" for i in range(1, 6)],
        *[f"a{i}" for i in range(1, 6)],
    ]
    terminal = grouped.tail(1)[terminal_columns].copy()
    summary = summary.merge(terminal, on=keys, how="left", validate="one_to_one")
    summary = summary[summary["has_live_event"] | summary["has_score"]].copy()

    missing_offense = ~summary["poss_offense"].isin(["Home", "Away"])
    if missing_offense.any():
        group_lookup = grouped
        for index in summary.index[missing_offense]:
            row = summary.loc[index]
            group = group_lookup.get_group((row["game_id"], row["net_poss_group"]))
            summary.at[index, "poss_offense"] = infer_segment_offense(
                group,
                row.get("home_team_abbr"),
                row.get("away_team_abbr"),
            )
    summary = summary[summary["poss_offense"].isin(["Home", "Away"])].copy()

    offense_lookup = summary[keys + ["poss_offense"]]
    parsed = parsed.merge(offense_lookup, on=keys, how="left", suffixes=("", "_resolved"))
    split_admin = parsed[
        parsed["is_admin_free_throw"]
        & (parsed["home_action_points"].gt(0) | parsed["away_action_points"].gt(0))
        & parsed["action_scoring_side"].isin(["Home", "Away"])
        & parsed["action_scoring_side"].ne(parsed["poss_offense_resolved"])
    ].copy()
    if split_admin.empty:
        admin_points = pd.DataFrame(
            columns=[*keys, "home_admin_points", "away_admin_points"]
        )
    else:
        admin_points = (
            split_admin.groupby(keys, sort=False, dropna=False)
            .agg(
                home_admin_points=("home_action_points", "sum"),
                away_admin_points=("away_action_points", "sum"),
            )
            .reset_index()
        )
    summary = summary.merge(admin_points, on=keys, how="left")
    summary[["home_admin_points", "away_admin_points"]] = summary[
        ["home_admin_points", "away_admin_points"]
    ].fillna(0.0)
    summary["home_points"] = (
        summary["home_points_total"] - summary["home_admin_points"]
    )
    summary["away_points"] = (
        summary["away_points_total"] - summary["away_admin_points"]
    )
    is_home_offense = summary["poss_offense"].eq("Home")
    summary["off_points"] = np.where(
        is_home_offense, summary["home_points"], summary["away_points"]
    )
    summary["def_points"] = np.where(
        is_home_offense, summary["away_points"], summary["home_points"]
    )

    possession_rows = pd.DataFrame(
        {
            "game_id": summary["game_id"].astype(str),
            "season": int(year),
            "season_type": season_type.upper(),
            "period": summary["period"].astype(int),
            "net_poss_group": summary["net_poss_group"].astype(int),
            "start_event_num": summary["start_event_num"].astype(int),
            "end_event_num": summary["end_event_num"].astype(int),
            "start_time": summary["start_time"].astype(str),
            "end_time": summary["end_time"].astype(str),
            "poss_offense": summary["poss_offense"].astype(str),
            "off_team_abbr": np.where(
                is_home_offense,
                summary["home_team_abbr"],
                summary["away_team_abbr"],
            ),
            "def_team_abbr": np.where(
                is_home_offense,
                summary["away_team_abbr"],
                summary["home_team_abbr"],
            ),
            "home_points": summary["home_points"].astype(float),
            "away_points": summary["away_points"].astype(float),
            "off_points": summary["off_points"].astype(float),
            "def_points": summary["def_points"].astype(float),
            "off_poss": 1,
            "def_poss": 1,
            "event_count": summary["event_count"].astype(int),
            "period_count": summary["period_count"].astype(int),
            "terminal_event_type": summary["event_type"].astype(str),
            "terminal_event_text": summary["event_text"].astype(str),
            "row_type": "possession",
        }
    )
    for idx in range(1, 6):
        possession_rows[f"O{idx}"] = np.where(
            is_home_offense, summary[f"h{idx}"], summary[f"a{idx}"]
        )
        possession_rows[f"D{idx}"] = np.where(
            is_home_offense, summary[f"a{idx}"], summary[f"h{idx}"]
        )

    admin_rows: list[dict[str, object]] = []
    for admin in split_admin.itertuples(index=False):
        admin_row = pd.Series(admin._asdict())
        admin_offense = str(admin_row["action_scoring_side"])
        if admin_offense not in {"Home", "Away"}:
            continue
        admin_home_points = float(admin_row["home_action_points"])
        admin_away_points = float(admin_row["away_action_points"])
        admin_off_points = (
            admin_home_points if admin_offense == "Home" else admin_away_points
        )
        admin_group = pd.DataFrame([admin_row])
        admin_rows.append(
            base_row(
                admin_row["game_id"],
                year,
                season_type,
                admin_group,
                admin_row,
                admin_row["net_poss_group"],
                admin_offense,
                admin_home_points,
                admin_away_points,
                admin_off_points,
                0.0,
                0,
                0,
                "admin_free_throw",
            )
        )

    if admin_rows:
        output = pd.concat(
            [possession_rows, pd.DataFrame(admin_rows)],
            ignore_index=True,
        )
        output["_row_order"] = output["row_type"].eq("admin_free_throw").astype(int)
        output = output.sort_values(
            ["game_id", "net_poss_group", "_row_order", "start_event_num"],
            kind="stable",
        ).drop(columns="_row_order")
        return output.reset_index(drop=True)
    return possession_rows.reset_index(drop=True)


def validate_possessions(poss: pd.DataFrame) -> dict[str, object]:
    if poss.empty:
        return {"rows": 0}
    return {
        "rows": int(len(poss)),
        "games": int(poss["game_id"].nunique()),
        "period_change_rows": int(poss["period_count"].gt(1).sum()),
        "zero_lineup_rows": int((poss[[f"O{i}" for i in range(1, 6)] + [f"D{i}" for i in range(1, 6)]] == 0).any(axis=1).sum()),
        "home_points": float(poss["home_points"].sum()),
        "away_points": float(poss["away_points"].sum()),
        "off_points": float(poss["off_points"].sum()),
        "def_points": float(poss["def_points"].sum()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--season-type", choices=["RS", "PS"], default="RS")
    parser.add_argument("--game-id", help="Optional single game audit/build")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    poss = build_possessions(args.year, args.season_type, args.game_id)
    print(validate_possessions(poss))

    if args.no_write:
        return

    output = args.output or default_output_path(args.year, args.season_type)
    output.parent.mkdir(parents=True, exist_ok=True)
    poss.to_parquet(output, index=False)
    print(f"Wrote {len(poss):,} rows to {output}")


if __name__ == "__main__":
    main()
