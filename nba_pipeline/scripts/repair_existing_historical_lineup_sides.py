#!/usr/bin/env python3
"""Repair game-local lineup side conflicts in existing historical raw parquets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_gabriel_old_pbp import (
    add_side_vote,
    build_lineup_player_side_map,
    merge_missing_side_map,
    normalize_player_id,
    repair_partial_lineups_from_stable_neighbors,
    sanitize_lineup_side_conflicts,
)


HOME_COLS = [f"home_player{i}" for i in range(1, 6)]
AWAY_COLS = [f"away_player{i}" for i in range(1, 6)]


def parse_years(value: str) -> list[int]:
    years: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            years.extend(range(int(start), int(end) + 1))
        else:
            years.append(int(part))
    return sorted(dict.fromkeys(years))


def suffix(year: int) -> str:
    return f"{int(year) % 100:02d}"


def raw_name(year: int, season_type: str) -> str:
    return f"NBA{suffix(year)}_PS.parquet" if season_type == "ps" else f"NBA{suffix(year)}.parquet"


def has_text(value) -> bool:
    return pd.notna(value) and str(value).strip() != ""


def normalize_team(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    return text or None


def infer_game_sides(game: pd.DataFrame) -> tuple[str | None, str | None]:
    votes = {"home": {}, "away": {}}
    for _, row in game.iterrows():
        team = normalize_team(row.get("player1_team_abbreviation"))
        if team is None:
            continue
        if has_text(row.get("home_description")):
            votes["home"][team] = votes["home"].get(team, 0) + 1
        if has_text(row.get("visitor_description")):
            votes["away"][team] = votes["away"].get(team, 0) + 1

    def winner(side: str) -> str | None:
        if not votes[side]:
            return None
        return sorted(votes[side].items(), key=lambda item: (-item[1], item[0]))[0][0]

    home = winner("home")
    away = winner("away")
    if home is not None and away is not None and home != away:
        return home, away

    all_teams = []
    for col in ("player1_team_abbreviation", "player2_team_abbreviation", "player3_team_abbreviation"):
        all_teams.extend(team for team in game[col].map(normalize_team).dropna().tolist() if team)
    unique = sorted(dict.fromkeys(all_teams))
    if home is None and unique:
        home = unique[0]
    if away is None:
        away_candidates = [team for team in unique if team != home]
        away = away_candidates[0] if away_candidates else None
    if home == away:
        away = None
    return home, away


def side_for_team(team: str | None, home_team: str | None, away_team: str | None) -> str | None:
    if team == home_team:
        return "home"
    if team == away_team:
        return "away"
    return None


def build_actor_side_map(game: pd.DataFrame, home_team: str | None, away_team: str | None) -> tuple[dict[int, str], dict[str, int]]:
    votes: dict[int, dict[str, int]] = {}
    for _, row in game.iterrows():
        for idx in (1, 2, 3):
            side = side_for_team(normalize_team(row.get(f"player{idx}_team_abbreviation")), home_team, away_team)
            add_side_vote(votes, row.get(f"player{idx}_id"), side, weight=3)

    side_map: dict[int, str] = {}
    ambiguous = 0
    for player_id, player_votes in votes.items():
        home_votes = player_votes.get("home", 0)
        away_votes = player_votes.get("away", 0)
        if home_votes == away_votes:
            ambiguous += 1
            continue
        side_map[player_id] = "home" if home_votes > away_votes else "away"
    return side_map, {
        "actor_side_map_players": len(side_map),
        "actor_side_map_ambiguous_players": ambiguous,
    }


def compact_side(row: dict, side: str, side_map: dict[int, str]) -> tuple[list[float], int]:
    cols = HOME_COLS if side == "home" else AWAY_COLS
    kept: list[float] = []
    removed = 0
    seen: set[int] = set()
    for col in cols:
        player_id = normalize_player_id(row.get(col))
        if pd.isna(player_id) or int(player_id) <= 0:
            continue
        player_int = int(player_id)
        if player_int in seen:
            removed += 1
            continue
        known_side = side_map.get(player_int)
        if known_side is not None and known_side != side:
            removed += 1
            continue
        seen.add(player_int)
        kept.append(float(player_int))
    return kept[:5], removed


def strip_opposite_side_players(game_rows: list[dict], side_map: dict[int, str]) -> tuple[list[dict], dict[str, int]]:
    stripped = [dict(row) for row in game_rows]
    report = {"home_opposite_side_players_removed": 0, "away_opposite_side_players_removed": 0}
    for row in stripped:
        for side, cols in (("home", HOME_COLS), ("away", AWAY_COLS)):
            kept, removed = compact_side(row, side, side_map)
            report[f"{side}_opposite_side_players_removed"] += removed
            for idx, col in enumerate(cols):
                row[col] = kept[idx] if idx < len(kept) else np.nan
    return stripped, report


def repair_game(game: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int | str | None]]:
    sort_cols = [col for col in ("game_id", "period", "event_num", "source_actionNumber") if col in game.columns]
    game = game.sort_values(sort_cols, kind="stable") if sort_cols else game.copy()
    home_team, away_team = infer_game_sides(game)

    actor_side_map, actor_summary = build_actor_side_map(game, home_team, away_team)
    game_rows = game.to_dict("records")
    lineup_side_map, lineup_summary = build_lineup_player_side_map(game_rows)
    lineup_added = merge_missing_side_map(actor_side_map, lineup_side_map)

    game_rows, strip_summary = strip_opposite_side_players(game_rows, actor_side_map)
    game_rows, pre_summary = sanitize_lineup_side_conflicts(game_rows, actor_side_map)
    game_rows, partial_summary = repair_partial_lineups_from_stable_neighbors(game_rows)

    post_lineup_side_map, post_lineup_summary = build_lineup_player_side_map(game_rows)
    post_added = merge_missing_side_map(actor_side_map, post_lineup_side_map)
    game_rows, post_summary = strip_opposite_side_players(game_rows, actor_side_map)
    game_rows, final_summary = sanitize_lineup_side_conflicts(game_rows, actor_side_map)

    out = pd.DataFrame(game_rows, columns=game.columns)
    return out, {
        "game_id": str(game["game_id"].iloc[0]),
        "home_team": home_team,
        "away_team": away_team,
        **actor_summary,
        **lineup_summary,
        "lineup_side_map_added_players": lineup_added,
        "post_lineup_side_map_players": post_lineup_summary["lineup_side_map_players"],
        "post_lineup_side_map_ambiguous_players": post_lineup_summary["lineup_side_map_ambiguous_players"],
        "post_lineup_side_map_added_players": post_added,
        **strip_summary,
        **{f"pre_{key}": value for key, value in pre_summary.items()},
        **partial_summary,
        **{f"post_strip_{key}": value for key, value in post_summary.items()},
        **{f"final_{key}": value for key, value in final_summary.items()},
    }


def count_raw_overlaps(df: pd.DataFrame) -> dict[str, int]:
    home = df[HOME_COLS].apply(pd.to_numeric, errors="coerce").fillna(0).astype("int64").to_numpy()
    away = df[AWAY_COLS].apply(pd.to_numeric, errors="coerce").fillna(0).astype("int64").to_numpy()
    overlap = ((home[:, :, None] == away[:, None, :]) & (home[:, :, None] != 0)).any(axis=(1, 2))
    incomplete_home = ((home != 0).sum(axis=1) < 4)
    incomplete_away = ((away != 0).sum(axis=1) < 4)
    return {
        "raw_same_player_home_away_overlap_rows": int(overlap.sum()),
        "raw_incomplete_home_rows": int(incomplete_home.sum()),
        "raw_incomplete_away_rows": int(incomplete_away.sum()),
    }


def repair_file(input_path: Path, output_path: Path) -> dict:
    df = pd.read_parquet(input_path)
    before = count_raw_overlaps(df)
    repaired_parts: list[pd.DataFrame] = []
    game_summaries: list[dict] = []
    for _, game in df.groupby("game_id", sort=False):
        repaired_game, game_summary = repair_game(game)
        repaired_parts.append(repaired_game)
        game_summaries.append(game_summary)
    out = pd.concat(repaired_parts, ignore_index=True)
    after = count_raw_overlaps(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows": int(len(df)),
        "games": int(df["game_id"].nunique()),
        **{f"before_{key}": value for key, value in before.items()},
        **{f"after_{key}": value for key, value in after.items()},
        "actor_side_map_players": int(sum(item.get("actor_side_map_players", 0) for item in game_summaries)),
        "actor_side_map_ambiguous_players": int(sum(item.get("actor_side_map_ambiguous_players", 0) for item in game_summaries)),
        "lineup_side_map_players": int(sum(item.get("lineup_side_map_players", 0) for item in game_summaries)),
        "lineup_side_map_added_players": int(sum(item.get("lineup_side_map_added_players", 0) for item in game_summaries)),
        "home_opposite_side_players_removed": int(sum(item.get("home_opposite_side_players_removed", 0) for item in game_summaries)),
        "away_opposite_side_players_removed": int(sum(item.get("away_opposite_side_players_removed", 0) for item in game_summaries)),
        "final_overlap_rows_after": int(sum(item.get("final_same_player_side_overlap_rows_after", 0) for item in game_summaries)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", default="1997-2013")
    parser.add_argument("--season-types", default="ps", choices=["rs", "ps", "all"])
    parser.add_argument("--raw-dir", type=Path, default=Path("nba_pipeline/raw_data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    season_types = ["rs", "ps"] if args.season_types == "all" else [args.season_types]
    summaries: list[dict] = []
    for year in parse_years(args.years):
        for season_type in season_types:
            name = raw_name(year, season_type)
            input_path = args.raw_dir / name
            output_path = args.output_dir / name
            if not input_path.exists():
                summaries.append({
                    "year": year,
                    "season_type": season_type,
                    "input_path": str(input_path),
                    "skipped": True,
                    "reason": "missing_input",
                })
                continue
            summary = repair_file(input_path, output_path)
            summary.update({"year": year, "season_type": season_type, "skipped": False})
            summaries.append(summary)
            print(
                f"{name}: overlaps {summary['before_raw_same_player_home_away_overlap_rows']} -> "
                f"{summary['after_raw_same_player_home_away_overlap_rows']}, "
                f"incomplete home/away {summary['after_raw_incomplete_home_rows']}/"
                f"{summary['after_raw_incomplete_away_rows']}"
            )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summaries, indent=2, sort_keys=True))
    pd.DataFrame(summaries).to_csv(args.report.with_suffix(".csv"), index=False)
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
