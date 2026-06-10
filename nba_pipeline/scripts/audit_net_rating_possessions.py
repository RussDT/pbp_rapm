#!/usr/bin/env python3
"""
Audit raw play-by-play possession splits for on-court net-rating use.

This is intentionally a validation harness, not a RAPM solver input builder. It
loads a single raw game, runs the shared standard possession parser, and writes
line-by-line evidence so possession ownership can be reviewed before building a
WOWY/net-rating parquet from the same assumptions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from process_rapm_blocks.common import (
    _base_processing,
    prepare_standard_possession_df,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
RAW_DATA_DIR = PIPELINE_DIR / "raw_data"
VALIDATION_DIR = PIPELINE_DIR / "validation" / "net_rating_possession_audit"


def _season_raw_path(year: int, season_type: str) -> Path:
    suffix = "_PS" if season_type.upper() == "PS" else ""
    return RAW_DATA_DIR / f"NBA{year}{suffix}.parquet"


def _event_text(row: pd.Series) -> str:
    parts = [
        str(row.get("home_description", "") or "").strip(),
        str(row.get("visitor_description", "") or "").strip(),
        str(row.get("neutral_description", "") or "").strip(),
    ]
    return " | ".join(part for part in parts if part)


def _score_delta(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").ffill().fillna(0)
    return values.diff().fillna(values).astype(float)


def _lineup_key(row: pd.Series, side: str) -> str:
    prefix = "h" if side == "Home" else "a"
    values = []
    for idx in range(1, 6):
        value = pd.to_numeric(row.get(f"{prefix}{idx}"), errors="coerce")
        values.append(str(int(value)) if pd.notna(value) and int(value) > 0 else "0")
    return "-".join(values)


def load_game(year: int, season_type: str, game_id: str) -> pd.DataFrame:
    raw_path = _season_raw_path(year, season_type)
    df = _base_processing(raw_path)
    if df is None or df.empty:
        raise RuntimeError(f"Could not load raw file: {raw_path}")

    normalized_game_id = str(game_id).strip().zfill(10)
    game = df[df["game_id"].astype(str).str.zfill(10).eq(normalized_game_id)].copy()
    if game.empty:
        available = ", ".join(df["game_id"].drop_duplicates().head(10).astype(str))
        raise ValueError(f"Game {normalized_game_id} not found in {raw_path}; first games: {available}")
    return game.reset_index(drop=True)


def annotate_game(game: pd.DataFrame) -> pd.DataFrame:
    parsed, _ = prepare_standard_possession_df(game.copy(), "NET_RATING_AUDIT")
    parsed = parsed.reset_index(drop=True)

    parsed["home_score_delta"] = _score_delta(parsed["home_score"])
    parsed["away_score_delta"] = _score_delta(parsed["away_score"])
    parsed["action_score_sum"] = (
        pd.to_numeric(parsed["Home_Action_Score"], errors="coerce").fillna(0)
        + pd.to_numeric(parsed["Away_Action_Score"], errors="coerce").fillna(0)
    )
    parsed["score_delta_sum"] = parsed["home_score_delta"] + parsed["away_score_delta"]
    parsed["score_delta_mismatch"] = (
        parsed["action_score_sum"].round(6).ne(parsed["score_delta_sum"].round(6))
    )

    parsed["home_lineup_key"] = parsed.apply(lambda row: _lineup_key(row, "Home"), axis=1)
    parsed["away_lineup_key"] = parsed.apply(lambda row: _lineup_key(row, "Away"), axis=1)
    parsed["off_lineup_key"] = np.where(
        parsed["poss_offense"].eq("Home"),
        parsed["home_lineup_key"],
        np.where(parsed["poss_offense"].eq("Away"), parsed["away_lineup_key"], ""),
    )
    parsed["def_lineup_key"] = np.where(
        parsed["poss_offense"].eq("Home"),
        parsed["away_lineup_key"],
        np.where(parsed["poss_offense"].eq("Away"), parsed["home_lineup_key"], ""),
    )

    parsed["event_text"] = parsed.apply(_event_text, axis=1)
    parsed["row_num"] = np.arange(1, len(parsed) + 1)
    parsed["is_eop"] = parsed["End_of_Possession"].fillna(False).astype(bool)
    parsed["offense_score_delta"] = np.where(
        parsed["poss_offense"].eq("Home"),
        parsed["home_score_delta"],
        np.where(parsed["poss_offense"].eq("Away"), parsed["away_score_delta"], 0.0),
    )
    parsed["defense_score_delta"] = np.where(
        parsed["poss_offense"].eq("Home"),
        parsed["away_score_delta"],
        np.where(parsed["poss_offense"].eq("Away"), parsed["home_score_delta"], 0.0),
    )
    return parsed


def summarize_possessions(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (game_id, poss_group), group in events.groupby(["game_id", "poss_group"], sort=False):
        terminal = group[group["is_eop"]].tail(1)
        terminal_row = terminal.iloc[0] if not terminal.empty else group.iloc[-1]
        offense_values = sorted(value for value in group["poss_offense"].dropna().unique() if value)
        team_on_offense_values = sorted(value for value in group["TeamOnOffense"].dropna().unique() if value)
        home_points = float(group["home_score_delta"].sum())
        away_points = float(group["away_score_delta"].sum())
        offense = str(terminal_row.get("poss_offense", "") or "")
        offense_points = home_points if offense == "Home" else away_points if offense == "Away" else 0.0
        defense_points = away_points if offense == "Home" else home_points if offense == "Away" else 0.0
        side_scoring_count = int(home_points > 0) + int(away_points > 0)
        period_count = int(group["period"].nunique())
        end_period_not_eop = group["event_type"].eq("EndOfPeriod") & ~group["is_eop"]
        event_lines = [
            f"{int(row.event_num)} {row.event_type} {row.time_quarter}: {row.event_text}"
            for row in group.itertuples(index=False)
        ]
        flags = []
        if terminal.empty:
            flags.append("NO_EOP")
        if not offense:
            flags.append("BLANK_OFFENSE")
        if len(offense_values) > 1:
            flags.append("POSS_OFFENSE_SWITCH")
        if len(team_on_offense_values) > 1:
            flags.append("TEAM_ON_OFFENSE_SWITCH")
        if side_scoring_count > 1:
            flags.append("BOTH_TEAMS_SCORED")
        if group["score_delta_mismatch"].any():
            flags.append("SCORE_DELTA_MISMATCH")
        if defense_points > 0:
            flags.append("DEFENSE_POINTS_IN_GROUP")
        if period_count > 1:
            flags.append("PERIOD_CHANGE")
        if end_period_not_eop.any():
            flags.append("END_PERIOD_NOT_EOP")

        rows.append(
            {
                "game_id": game_id,
                "poss_group": int(poss_group),
                "period": int(terminal_row.get("period", group.iloc[0].get("period", 0)) or 0),
                "start_event_num": int(group["event_num"].iloc[0]),
                "end_event_num": int(group["event_num"].iloc[-1]),
                "start_time": str(group["time_quarter"].iloc[0]),
                "end_time": str(group["time_quarter"].iloc[-1]),
                "poss_offense": offense,
                "home_points": home_points,
                "away_points": away_points,
                "offense_points": offense_points,
                "defense_points": defense_points,
                "terminal_event_type": str(terminal_row.get("event_type", "")),
                "terminal_event_text": _event_text(terminal_row),
                "off_lineup_key": str(terminal_row.get("off_lineup_key", "")),
                "def_lineup_key": str(terminal_row.get("def_lineup_key", "")),
                "team_on_offense_values": ",".join(team_on_offense_values),
                "num_events": int(len(group)),
                "period_count": period_count,
                "flags": ",".join(flags),
                "event_lines": "\n".join(event_lines),
            }
        )
    return pd.DataFrame(rows)


def write_outputs(events: pd.DataFrame, possessions: pd.DataFrame, output_dir: Path, game_id: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / f"{game_id}_events.csv"
    possessions_path = output_dir / f"{game_id}_possessions.csv"
    markdown_path = output_dir / f"{game_id}_review.md"

    event_cols = [
        "row_num",
        "game_id",
        "poss_group",
        "period",
        "time_quarter",
        "event_num",
        "event_type",
        "event_text",
        "home_score",
        "away_score",
        "home_score_delta",
        "away_score_delta",
        "TeamOnOffense",
        "poss_offense",
        "is_eop",
        "off_lineup_key",
        "def_lineup_key",
        "score_delta_mismatch",
    ]
    events[event_cols].to_csv(events_path, index=False)
    possessions.to_csv(possessions_path, index=False)

    flagged = possessions[possessions["flags"].astype(str).ne("")]
    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# Net Rating Possession Audit: {game_id}\n\n")
        handle.write(f"- Possessions: {len(possessions)}\n")
        handle.write(f"- Events: {len(events)}\n")
        handle.write(f"- Flagged possessions: {len(flagged)}\n")
        handle.write(f"- Event CSV: `{events_path.name}`\n")
        handle.write(f"- Possession CSV: `{possessions_path.name}`\n\n")
        handle.write("## Flagged Possessions\n\n")
        if flagged.empty:
            handle.write("No flagged possessions.\n")
        else:
            for row in flagged.itertuples(index=False):
                handle.write(
                    f"### Possession {row.poss_group} "
                    f"({row.period} {row.start_time}-{row.end_time}, {row.flags})\n\n"
                )
                handle.write(f"- Offense: {row.poss_offense}; score: H {row.home_points:g}, A {row.away_points:g}\n")
                handle.write(f"- Terminal: {row.terminal_event_type} - {row.terminal_event_text}\n\n")
                handle.write("```text\n")
                handle.write(str(row.event_lines))
                handle.write("\n```\n\n")

    print(f"Wrote {events_path}")
    print(f"Wrote {possessions_path}")
    print(f"Wrote {markdown_path}")
    print(f"Possessions: {len(possessions):,}; flagged: {len(flagged):,}")
    if not flagged.empty:
        print(flagged[["poss_group", "period", "start_time", "end_time", "poss_offense", "home_points", "away_points", "flags"]].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True, help="Season end year, e.g. 26")
    parser.add_argument("--season-type", choices=["RS", "PS"], default="RS")
    parser.add_argument("--game-id", required=True, help="NBA game id")
    parser.add_argument("--output-dir", type=Path, default=VALIDATION_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    game_id = str(args.game_id).strip().zfill(10)
    game = load_game(args.year, args.season_type, game_id)
    events = annotate_game(game)
    possessions = summarize_possessions(events)
    output_dir = args.output_dir / f"{args.year}_{args.season_type.lower()}" / game_id
    write_outputs(events, possessions, output_dir, game_id)


if __name__ == "__main__":
    main()
