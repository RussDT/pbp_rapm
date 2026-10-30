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
    "At Rim Miss (Unblocked)",
    "Missed 2PT",
    "Missed FG",
    "Short Mid Range Miss",
    "Long Mid Range Miss",
    "Above The Break 3 Miss",
    "Missed 3PT",
    "Corner 3 Miss",
    "Deadball",
    "Made FG",
    "Made Rim FG",
    "Made Mid FG",
    "Made 3FG",
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
        "event_action_type",
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
        "event_action_type",
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
    made_three = poss["Made FG"] & prev_text_upper.str.contains("3PT", regex=False, na=False)
    made_distance = pd.to_numeric(prev_text.str.extract(r"(?:^|\s)(\d+)'", expand=False), errors="coerce")
    made_action = pd.to_numeric(poss["previous_event_action_type"], errors="coerce")
    made_rim = poss["Made FG"] & ~made_three & (
        made_action.isin(RIM_ACTION_TYPES)
        | prev_text.str.contains(r"layup|dunk| tip ", case=False, regex=True, na=False)
        | made_distance.le(4)
    )
    poss["Made Rim FG"] = made_rim
    poss["Made Mid FG"] = poss["Made FG"] & ~made_three & ~made_rim
    poss["Made 3FG"] = made_three
    poss["FT Make"] = prev_type.eq("FreeThrow") & ~prev_text_upper.str.contains(r"\bMISS\b", regex=True, na=False)
    poss["FT Miss"] = (
        (prev_type.eq("FreeThrow") & prev_text_upper.str.contains(r"\bMISS\b", regex=True, na=False))
        | is_rebound_after_missed_ft
    )
    poss["Missed FG"] = is_missed_fg
    poss["Missed 2PT"] = is_missed_2
    poss["Missed 3PT"] = is_missed_3
    poss["At Rim Miss (Unblocked)"] = is_rim & ~is_blocked
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
    # Retain aggregate/audit categories in CSVs, but show one missed-three bar.
    hidden = {"Corner 3 Miss", "Above The Break 3 Miss", "Made FG"}
    plot_df = summary.loc[~summary["category"].isin(hidden)].copy()
    plot_df = plot_df.loc[plot_df["possessions"].ge(min_possessions)].dropna(subset=[compare_col])
    plot_df = plot_df.sort_values(compare_col, ascending=False).reset_index(drop=True)
    if plot_df.empty:
        raise ValueError("No categories meet the minimum possession threshold")
    values = plot_df[compare_col] / 100.0
    baseline_col = "scope_avg_ortg" if compare_col == "delta_vs_scope" else "league_avg_ortg"
    baseline = float(plot_df[baseline_col].iloc[0]) / 100.0
    bg = "#faf7ec"
    fig, ax = plt.subplots(figsize=(12, max(9, 0.43 * len(plot_df) + 2.2)), dpi=180)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.barh(range(len(plot_df)), values, color=np.where(values >= 0, "#198e83", "#ca6471"), height=.63, zorder=3)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["category"])
    ax.invert_yaxis()
    for i, row in plot_df.iterrows():
        value = row[compare_col] / 100.0
        ax.text(value + (.004 if value >= 0 else -.004), i, f"{value:+.3f}", va="center", ha="left" if value >= 0 else "right", fontsize=11, fontweight="bold")
        ax.text(1.03, i, f"{row['ortg']/100:.3f}", transform=ax.get_yaxis_transform(), va="center", ha="right", fontsize=12, fontweight="bold")
        ax.text(1.19, i, f"{row['possessions']:,.0f}", transform=ax.get_yaxis_transform(), va="center", ha="right", fontsize=11, color="#666666")
    for x, label in [(1.03, "ACTUAL PPP"), (1.19, "POSS.")]:
        ax.text(x, 1.037, label, transform=ax.transAxes, ha="right", fontsize=10, color="#555555", fontweight="bold")
    ax.axvline(0, color="#333333", linewidth=1.4, zorder=2)
    ax.set_xlim(min(-.10, values.min() - .05), max(.25, values.max() + .075))
    ax.xaxis.set_major_formatter(lambda v, _: "Avg" if abs(v) < 1e-8 else f"{v:+.2f}")
    ax.set_xlabel("PPP above / below comparison average", fontsize=12, labelpad=14)
    ax.tick_params(axis="both", labelsize=11, length=0, pad=8)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#dedbd2", linewidth=.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(.035, .955, title.replace("Offensive Rating", "PPP"), fontsize=22, fontweight="bold")
    fig.text(.035, .914, subtitle.split(" | ")[0] + f" • Full ensuing possession • Comparison average: {baseline:.3f} PPP", fontsize=12, color="#555555")
    fig.text(.035, .052, "Miss categories require a defensive rebound. Rim-miss and rim-block bars are separate; other miss aggregates overlap.", fontsize=10, color="#555555")
    fig.text(.035, .03, "Made shots are split into rim, other two-pointers (mid), and threes. Descriptive associations.", fontsize=10, color="#555555")
    fig.subplots_adjust(left=.265, right=.85, top=.835, bottom=.145)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
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
