#!/usr/bin/env python3
"""
Build offensive rating by possession start type from raw NBA play-by-play.

The report uses the repo's standard RAPM possession parser, then labels each
possession by the terminal event that created the next possession. Categories
are intentionally not mutually exclusive: an at-rim miss is also a missed 2PT
and a missed FG.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lineup_stats import infer_game_team_map
from process_rapm_blocks.common import (
    RIM_ACTION_TYPES,
    _base_processing,
    prepare_standard_possession_df,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
RAW_DATA_DIR = PIPELINE_DIR / "raw_data"
RESULTS_DIR = PIPELINE_DIR / "results"

CATEGORY_ORDER = [
    "Live Ball Turnover",
    "At Rim Block",
    "At Rim Miss",
    "Missed 2PT",
    "Missed FG",
    "Short Mid Range Miss",
    "Long Mid Range Miss",
    "Above The Break 3 Miss",
    "Missed 3PT",
    "Corner 3 Miss",
    "Deadball",
    "Made FG",
    "FT Make",
    "FT Miss",
    "Timeout",
]

LIVE_ACTION_EVENTS = {"MAKE", "MISS", "FreeThrow", "Turnover", "JumpBall"}
MISS_DISTANCE_RE = re.compile(r"MISS\s+.+?\s+(\d+)'\s+", re.IGNORECASE)


def season_raw_path(year: int, season_type: str) -> Path:
    suffix = "_PS" if season_type.upper() == "PS" else ""
    return RAW_DATA_DIR / f"NBA{year}{suffix}.parquet"


def build_possession_rows(year: int, season_type: str) -> pd.DataFrame:
    raw_path = season_raw_path(year, season_type)
    base = _base_processing(raw_path)
    if base is None or base.empty:
        raise RuntimeError(f"Could not load raw play-by-play from {raw_path}")

    team_map = infer_game_team_map(base)
    parsed_input = base.merge(team_map, on="game_id", how="left")
    parsed, _ = prepare_standard_possession_df(parsed_input, "POSSESSION_START_REPORT")
    parsed = parsed.reset_index(drop=True)
    parsed["row_order"] = np.arange(len(parsed))
    parsed["is_eop"] = parsed["End_of_Possession"].fillna(False).astype(bool)
    parsed["is_timeout"] = parsed["event_type"].eq("Timeout")
    parsed["is_live_action"] = parsed["event_type"].isin(LIVE_ACTION_EVENTS)
    parsed["timeout_order"] = np.where(parsed["is_timeout"], parsed["row_order"], np.nan)
    parsed["live_order"] = np.where(parsed["is_live_action"], parsed["row_order"], np.nan)
    parsed["Prev_event_action_type"] = parsed.groupby("game_id", sort=False)["event_action_type"].shift(1)
    parsed["event_text"] = (
        parsed["home_description"].fillna("").astype(str)
        + " "
        + parsed["visitor_description"].fillna("").astype(str)
        + " "
        + parsed["neutral_description"].fillna("").astype(str)
    ).str.strip()

    key_cols = ["game_id", "poss_group"]
    group_summary = parsed.groupby(key_cols, sort=False).agg(
        period=("period", "first"),
        start_event_num=("event_num", "first"),
        start_time=("time_quarter", "first"),
        home_points=("Home_Action_Score", "sum"),
        away_points=("Away_Action_Score", "sum"),
        first_timeout_order=("timeout_order", "min"),
        first_live_order=("live_order", "min"),
    ).reset_index()

    terminal = (
        parsed.loc[parsed["is_eop"]]
        .sort_values([*key_cols, "row_order"], kind="stable")
        .drop_duplicates(key_cols, keep="last")
        .copy()
    )
    terminal = terminal[[
        "game_id",
        "poss_group",
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
        "event_num",
        "time_quarter",
        "poss_offense",
        "event_type",
        "event_text",
        "home_team_abbr",
        "away_team_abbr",
        "Prev_Event",
        "Prev_home_desc",
        "Prev_visitor_desc",
        "Prev_event_action_type",
    ]].rename(columns={
        "event_num": "end_event_num",
        "time_quarter": "end_time",
        "event_type": "terminal_event_type",
        "event_text": "terminal_event_text",
    })

    terminal = terminal.sort_values(["game_id", "poss_group"], kind="stable").reset_index(drop=True)
    previous_cols = [
        "terminal_event_type",
        "terminal_event_text",
        "Prev_Event",
        "Prev_home_desc",
        "Prev_visitor_desc",
        "Prev_event_action_type",
    ]
    for col in previous_cols:
        terminal[f"previous_{col}"] = terminal.groupby("game_id", sort=False)[col].shift(1)

    poss = group_summary.merge(terminal, on=key_cols, how="inner")
    poss = poss[poss["poss_offense"].isin(["Home", "Away"])].copy()
    poss["D5_numeric"] = pd.to_numeric(poss["D5"], errors="coerce")
    poss = poss[poss["D5_numeric"].notna() & poss["D5_numeric"].ne(0)].copy()
    poss["season"] = int(year)
    poss["season_type"] = season_type.upper()
    poss["off_team_abbr"] = np.where(
        poss["poss_offense"].eq("Home"),
        poss["home_team_abbr"],
        poss["away_team_abbr"],
    )
    poss["def_team_abbr"] = np.where(
        poss["poss_offense"].eq("Home"),
        poss["away_team_abbr"],
        poss["home_team_abbr"],
    )
    poss["off_points"] = np.where(
        poss["poss_offense"].eq("Home"),
        poss["home_points"],
        poss["away_points"],
    ).astype(float)
    poss["def_points"] = np.where(
        poss["poss_offense"].eq("Home"),
        poss["away_points"],
        poss["home_points"],
    ).astype(float)

    prev_type = poss["previous_terminal_event_type"].fillna("")
    prev_text = poss["previous_terminal_event_text"].fillna("")
    prev_text_upper = prev_text.str.upper()
    shot_text = (
        poss["previous_Prev_home_desc"].fillna("").astype(str)
        + " "
        + poss["previous_Prev_visitor_desc"].fillna("").astype(str)
    ).str.strip()
    shot_text_upper = shot_text.str.upper()
    is_rebound_after_missed_ft = (
        prev_type.eq("Rebound")
        & poss["previous_Prev_Event"].fillna("").eq("FreeThrow")
        & shot_text_upper.str.contains("FREE THROW", regex=False, na=False)
        & shot_text_upper.str.contains(r"\bMISS\b", regex=True, na=False)
    )
    distance = pd.to_numeric(shot_text.str.extract(MISS_DISTANCE_RE, expand=False), errors="coerce")
    prev_action_type = pd.to_numeric(poss["previous_Prev_event_action_type"], errors="coerce")
    is_prev_miss_rebound = prev_type.eq("Rebound") & poss["previous_Prev_Event"].fillna("").eq("MISS")
    is_missed_fg = is_prev_miss_rebound & shot_text_upper.str.contains(r"\bMISS\b", regex=True, na=False)
    is_missed_3 = is_missed_fg & shot_text_upper.str.contains("3PT", regex=False, na=False)
    is_missed_2 = is_missed_fg & ~is_missed_3
    is_rim = is_missed_2 & (
        prev_action_type.isin(RIM_ACTION_TYPES)
        | shot_text.str.contains(r"layup|dunk| tip ", case=False, regex=True, na=False)
        | distance.le(4)
    )
    is_blocked = prev_text_upper.str.contains("BLOCK", regex=False, na=False) | shot_text_upper.str.contains("BLOCK", regex=False, na=False)

    poss["Live Ball Turnover"] = prev_type.eq("Turnover") & prev_text_upper.str.contains("STEAL", regex=False, na=False)
    poss["Deadball"] = prev_type.eq("Turnover") & ~poss["Live Ball Turnover"]
    poss["Made FG"] = prev_type.eq("MAKE")
    poss["FT Make"] = prev_type.eq("FreeThrow") & ~prev_text_upper.str.contains(r"\bMISS\b", regex=True, na=False)
    poss["FT Miss"] = (
        (prev_type.eq("FreeThrow") & prev_text_upper.str.contains(r"\bMISS\b", regex=True, na=False))
        | is_rebound_after_missed_ft
    )
    poss["Missed FG"] = is_missed_fg
    poss["Missed 2PT"] = is_missed_2
    poss["Missed 3PT"] = is_missed_3
    poss["At Rim Miss"] = is_rim
    poss["At Rim Block"] = is_rim & is_blocked
    poss["Short Mid Range Miss"] = is_missed_2 & ~is_rim & distance.between(5, 14, inclusive="both")
    poss["Long Mid Range Miss"] = is_missed_2 & ~is_rim & distance.ge(15)
    poss["Corner 3 Miss"] = is_missed_3 & distance.le(24)
    poss["Above The Break 3 Miss"] = is_missed_3 & distance.ge(25)
    poss["Timeout"] = poss["first_timeout_order"].notna() & (
        poss["first_live_order"].isna() | poss["first_timeout_order"].lt(poss["first_live_order"])
    )

    output_cols = [
        "game_id",
        "season",
        "season_type",
        "period",
        "poss_group",
        "start_event_num",
        "end_event_num",
        "start_time",
        "end_time",
        "poss_offense",
        "off_team_abbr",
        "def_team_abbr",
        "home_points",
        "away_points",
        "off_points",
        "def_points",
        "terminal_event_type",
        "terminal_event_text",
        "previous_terminal_event_type",
        "previous_terminal_event_text",
        *CATEGORY_ORDER,
    ]
    return poss[output_cols].copy()


def summarize(possessions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    league_possessions = len(possessions)
    league_points = float(possessions["off_points"].sum())
    league_ortg = 100.0 * league_points / league_possessions

    league_rows: list[dict[str, object]] = []
    team_rows: list[dict[str, object]] = []

    for category in CATEGORY_ORDER:
        mask = possessions[category].astype(bool)
        points = float(possessions.loc[mask, "off_points"].sum())
        poss = int(mask.sum())
        ortg = 100.0 * points / poss if poss else np.nan
        league_rows.append({
            "scope": "League",
            "team": "League",
            "category": category,
            "possessions": poss,
            "points": points,
            "ortg": ortg,
            "league_avg_ortg": league_ortg,
            "delta_vs_league": ortg - league_ortg if poss else np.nan,
            "scope_avg_ortg": league_ortg,
            "delta_vs_scope": ortg - league_ortg if poss else np.nan,
        })

    for team, team_df in possessions.groupby("off_team_abbr", sort=True):
        if not team:
            continue
        team_avg = 100.0 * float(team_df["off_points"].sum()) / len(team_df)
        for category in CATEGORY_ORDER:
            mask = team_df[category].astype(bool)
            points = float(team_df.loc[mask, "off_points"].sum())
            poss = int(mask.sum())
            ortg = 100.0 * points / poss if poss else np.nan
            team_rows.append({
                "scope": "Team",
                "team": team,
                "category": category,
                "possessions": poss,
                "points": points,
                "ortg": ortg,
                "league_avg_ortg": league_ortg,
                "delta_vs_league": ortg - league_ortg if poss else np.nan,
                "scope_avg_ortg": team_avg,
                "delta_vs_scope": ortg - team_avg if poss else np.nan,
            })

    return pd.DataFrame(league_rows), pd.DataFrame(team_rows)


def render_chart(
    summary: pd.DataFrame,
    output_path: Path,
    title: str,
    subtitle: str,
    compare_col: str,
    min_possessions: int,
) -> None:
    plot_df = summary.copy()
    plot_df = plot_df[plot_df["possessions"].ge(min_possessions)].copy()
    plot_df["category"] = pd.Categorical(plot_df["category"], categories=CATEGORY_ORDER, ordered=True)
    plot_df = plot_df.sort_values("category", ascending=True)
    plot_df = plot_df.dropna(subset=[compare_col])
    if plot_df.empty:
        raise ValueError("No categories meet the minimum possession threshold")

    values = plot_df[compare_col].to_numpy()
    colors = np.where(values >= 0, "#52c8bd", "#ff858c")
    edge_colors = np.where(values >= 0, "#007f78", "#b83443")

    fig_height = max(7.0, 0.42 * len(plot_df) + 2.2)
    fig, ax = plt.subplots(figsize=(10.8, fig_height), dpi=180)
    fig.patch.set_facecolor("#faf7ec")
    ax.set_facecolor("#faf7ec")

    y_pos = np.arange(len(plot_df))
    ax.barh(y_pos, values, color=colors, edgecolor=edge_colors, linewidth=1.4, height=0.78)
    ax.axvline(0, color="#b83a48", linewidth=1.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["category"], fontsize=10.5)
    ax.invert_yaxis()

    max_abs = max(5.0, float(np.nanmax(np.abs(values))) + 1.0)
    max_abs = min(max_abs, 24.0)
    ax.set_xlim(-max_abs, max_abs)
    ax.grid(axis="x", color="#dedbd2", linewidth=1.0)
    ax.grid(axis="y", color="#e7e2d6", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, colors="#222")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:+.0f}" if x else "0")
    ax.set_xlabel("Points Per 100 Possessions Above/Below Comparison Average", fontsize=12)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.text(0.055, 0.965, title, fontsize=18, fontweight="bold", ha="left", va="top")
    fig.text(0.055, 0.92, subtitle, fontsize=12, color="#85827b", ha="left", va="top")

    positive_y = max(1, min(len(plot_df) - 1, 3))
    negative_y = max(positive_y + 4, min(len(plot_df) - 2, len(plot_df) - 4))
    ax.text(
        max_abs * 0.35,
        positive_y,
        "Possessions that start\nwith these events lead to\nBETTER outcomes than\nthe comparison possession",
        color="#52bfb6",
        fontsize=10.5,
        fontweight="bold",
        va="center",
        ha="center",
        bbox={"boxstyle": "square,pad=0.35", "fc": "#faf7ec", "ec": "#52bfb6", "lw": 1.0},
    )
    ax.text(
        max_abs * 0.32,
        negative_y,
        "Possessions that start\nwith these events lead to\nWORSE outcomes than\nthe comparison possession",
        color="#ee7e86",
        fontsize=10.5,
        fontweight="bold",
        va="center",
        ha="center",
        bbox={"boxstyle": "square,pad=0.35", "fc": "#faf7ec", "ec": "#ee7e86", "lw": 1.0},
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.29, right=0.97, top=0.86, bottom=0.12)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=26)
    parser.add_argument("--season-type", choices=["RS", "PS"], default="RS")
    parser.add_argument("--team", help="Optional offensive team abbreviation, e.g. OKC")
    parser.add_argument("--compare", choices=["league", "scope"], default="league")
    parser.add_argument("--min-possessions", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "possession_start_report",
    )
    parser.add_argument("--write-possession-detail", action="store_true")
    parser.add_argument("--no-chart", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    possessions = build_possession_rows(args.year, args.season_type)
    league_summary, team_summary = summarize(possessions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.year}_{args.season_type.lower()}"
    league_csv = args.output_dir / f"possession_start_league_{suffix}.csv"
    team_csv = args.output_dir / f"possession_start_teams_{suffix}.csv"
    league_summary.to_csv(league_csv, index=False)
    team_summary.to_csv(team_csv, index=False)
    print(f"Wrote league summary: {league_csv}")
    print(f"Wrote team summary: {team_csv}")

    if args.write_possession_detail:
        detail_path = args.output_dir / f"possession_start_possessions_{suffix}.parquet"
        possessions.to_parquet(detail_path, index=False)
        print(f"Wrote possession detail: {detail_path}")

    if args.no_chart:
        return

    compare_col = "delta_vs_league" if args.compare == "league" else "delta_vs_scope"
    comparison_label = "league average" if args.compare == "league" else "team average"
    season_label = f"20{args.year - 1:02d}-{args.year:02d} {args.season_type}"

    if args.team:
        team = args.team.upper()
        chart_summary = team_summary[team_summary["team"].eq(team)].copy()
        if chart_summary.empty:
            raise ValueError(f"No possessions found for team {team}")
        title = f"{team} Offensive Rating by Possession Start Type"
        chart_name = f"possession_start_{team.lower()}_{suffix}_{args.compare}.png"
        svg_name = chart_name.replace(".png", ".svg")
    else:
        chart_summary = league_summary
        title = "League Average Offensive Rating by Possession Start Type"
        chart_name = f"possession_start_league_{suffix}_{args.compare}.png"
        svg_name = chart_name.replace(".png", ".svg")

    subtitle = f"{season_label} | Categories are not mutually exclusive | Compared to {comparison_label}"
    render_chart(
        chart_summary,
        args.output_dir / chart_name,
        title,
        subtitle,
        compare_col,
        args.min_possessions,
    )
    render_chart(
        chart_summary,
        args.output_dir / svg_name,
        title,
        subtitle,
        compare_col,
        args.min_possessions,
    )
    print(f"Wrote chart: {args.output_dir / chart_name}")
    print(f"Wrote chart: {args.output_dir / svg_name}")


if __name__ == "__main__":
    main()
