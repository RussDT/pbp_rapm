#!/usr/bin/env python3
"""Validation-only repair for one-sided historical substitution clusters."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


HOME_COLS = [f"home_player{i}" for i in range(1, 6)]
AWAY_COLS = [f"away_player{i}" for i in range(1, 6)]


def normalize_label(value) -> str:
    if pd.isna(value):
        return ""
    text = re.sub(r"[^a-z0-9 ]+", " ", str(value).lower())
    return re.sub(r"\s+", " ", text).strip()


def normalize_id(value) -> int | None:
    if pd.isna(value):
        return None
    try:
        player_id = int(float(value))
    except (TypeError, ValueError):
        return None
    return player_id if player_id > 0 else None


def has_text(value) -> bool:
    return pd.notna(value) and str(value).strip() != ""


def raw_name(year: int) -> str:
    return f"NBA{year % 100:02d}.parquet"


def side_cols(side: str) -> list[str]:
    return HOME_COLS if side == "home" else AWAY_COLS


def lineup_set(row: dict, side: str) -> set[int]:
    out: set[int] = set()
    for col in side_cols(side):
        player_id = normalize_id(row.get(col))
        if player_id is not None:
            out.add(player_id)
    return out


def lineup_count(row: dict, side: str) -> int:
    return len(lineup_set(row, side))


def set_lineup(row: dict, side: str, players: set[int]) -> None:
    ordered = sorted(players)
    cols = side_cols(side)
    for i, col in enumerate(cols):
        row[col] = float(ordered[i]) if i < len(ordered) else np.nan


def desc_text(row: dict) -> str:
    parts = []
    for col in ("home_description", "visitor_description", "neutral_description"):
        value = row.get(col)
        if has_text(value):
            parts.append(str(value))
    return " | ".join(parts)


def infer_game_sides(game: pd.DataFrame) -> dict[str, str]:
    votes: dict[str, dict[str, int]] = {"home": {}, "away": {}}
    for _, row in game.iterrows():
        team = row.get("player1_team_abbreviation")
        if pd.isna(team):
            continue
        team = str(team).strip().upper()
        if not team:
            continue
        if has_text(row.get("home_description")):
            votes["home"][team] = votes["home"].get(team, 0) + 1
        if has_text(row.get("visitor_description")):
            votes["away"][team] = votes["away"].get(team, 0) + 1

    out: dict[str, str] = {}
    for side in ("home", "away"):
        if votes[side]:
            out[side] = sorted(votes[side].items(), key=lambda item: (-item[1], item[0]))[0][0]
    return out


def build_name_lookup(frame: pd.DataFrame, team_abbr: str) -> dict[str, int]:
    lookup: dict[str, int] = {}

    def add_name(player_id, name) -> None:
        pid = normalize_id(player_id)
        label = normalize_label(name)
        if pid is None or not label:
            return
        labels = {label}
        parts = label.split()
        if parts:
            labels.add(parts[-1])
        if len(parts) >= 2:
            labels.add(f"{parts[0][0]} {parts[-1]}")
        for item in labels:
            lookup.setdefault(item, pid)

    for _, row in frame.iterrows():
        for idx in (1, 2, 3):
            team = row.get(f"player{idx}_team_abbreviation")
            if pd.notna(team) and str(team).strip().upper() == team_abbr:
                add_name(row.get(f"player{idx}_id"), row.get(f"player{idx}_name"))
    return lookup


def resolve_incoming(label: str, lookup: dict[str, int]) -> int | None:
    normalized = normalize_label(label)
    candidates = [normalized]
    parts = normalized.split()
    if parts:
        candidates.append(parts[-1])
    if len(parts) >= 2:
        candidates.append(f"{parts[0][0]} {parts[-1]}")
    for candidate in candidates:
        player_id = lookup.get(candidate)
        if player_id is not None:
            return player_id
    return None


def parse_explicit_sub(row: dict, lookup: dict[str, int]) -> tuple[int | None, int | None]:
    text = desc_text(row)
    match = re.search(r"sub:\s*(.*?)\s+for\s+(.*)$", text, flags=re.IGNORECASE)
    if not match:
        if re.search(r"\bsub\s+out\b", text, flags=re.IGNORECASE):
            return None, normalize_id(row.get("player1_id"))
        return None, None
    incoming = resolve_incoming(match.group(1), lookup)
    outgoing = normalize_id(row.get("player1_id"))
    return incoming, outgoing


def is_sub_row(row: dict) -> bool:
    return "sub" in desc_text(row).lower()


def row_team(row: dict) -> str | None:
    value = row.get("player1_team_abbreviation")
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    return text or None


def first_changed_after(rows: list[dict], start_idx: int, side: str, before: set[int], max_scan: int = 20) -> int | None:
    for j in range(start_idx + 1, min(len(rows), start_idx + 1 + max_scan)):
        current = lineup_set(rows[j], side)
        if len(current) >= 4 and current != before:
            return j
    return None


def apply_repaired_lineup_forward(rows: list[dict], start_idx: int, side: str, repaired: set[int], max_scan: int = 60) -> int:
    filled = 0
    for j in range(start_idx, min(len(rows), start_idx + max_scan)):
        if j > start_idx and is_sub_row(rows[j]) and lineup_count(rows[j], side) >= 4:
            break
        current = lineup_set(rows[j], side)
        if not current:
            continue
        if current.issubset(repaired) and len(current) < 5:
            set_lineup(rows[j], side, repaired)
            filled += 1
        elif len(current) == 5 and current != repaired:
            break
    return filled


def repair_game(
    game: pd.DataFrame,
    team_abbr: str,
    season_lookup: dict[str, int],
) -> tuple[pd.DataFrame, dict[str, int]]:
    sides = infer_game_sides(game)
    side = None
    if sides.get("home") == team_abbr:
        side = "home"
    elif sides.get("away") == team_abbr:
        side = "away"
    if side is None:
        return game.copy(), {"games_touched": 0}

    lookup = dict(season_lookup)
    lookup.update(build_name_lookup(game, team_abbr))
    rows = game.sort_values(["period", "event_num"], kind="stable").to_dict("records")
    summary = {
        "games_touched": 1,
        "clusters_seen": 0,
        "clusters_repaired": 0,
        "rows_filled": 0,
        "explicit_subs": 0,
        "implicit_subs": 0,
    }

    i = 0
    while i < len(rows):
        row = rows[i]
        if not is_sub_row(row) or row_team(row) != team_abbr:
            i += 1
            continue

        period = row.get("period")
        time_quarter = row.get("time_quarter")
        cluster_indices = []
        j = i
        while j < len(rows):
            candidate = rows[j]
            if (
                candidate.get("period") == period
                and candidate.get("time_quarter") == time_quarter
                and is_sub_row(candidate)
            ):
                cluster_indices.append(j)
                j += 1
                continue
            if cluster_indices and candidate.get("period") == period and candidate.get("time_quarter") == time_quarter:
                j += 1
                continue
            break

        team_cluster = [idx for idx in cluster_indices if row_team(rows[idx]) == team_abbr]
        if not team_cluster:
            i = max(j, i + 1)
            continue

        before_idx = team_cluster[0] - 1
        while before_idx >= 0 and lineup_count(rows[before_idx], side) < 5:
            before_idx -= 1
        if before_idx < 0:
            i = max(j, i + 1)
            continue
        before = lineup_set(rows[before_idx], side)
        if len(before) != 5:
            i = max(j, i + 1)
            continue

        after_idx = first_changed_after(rows, max(cluster_indices), side, before)
        if after_idx is None:
            i = max(j, i + 1)
            continue
        after_known = lineup_set(rows[after_idx], side)
        if len(after_known) < 4:
            i = max(j, i + 1)
            continue

        summary["clusters_seen"] += 1
        outgoing: set[int] = set()
        incoming: set[int] = set()
        for idx in team_cluster:
            inc, out = parse_explicit_sub(rows[idx], lookup)
            if out is not None:
                outgoing.add(out)
            if inc is not None:
                incoming.add(inc)
                summary["explicit_subs"] += 1
            elif out is not None:
                outgoing.add(out)

        if not outgoing:
            i = max(j, i + 1)
            continue

        holdovers = before - outgoing
        observed_new = after_known - holdovers
        inferred = observed_new - incoming
        final = holdovers | incoming | inferred
        if len(final) != 5:
            i = max(j, i + 1)
            continue
        if after_known and not after_known.issubset(final):
            i = max(j, i + 1)
            continue

        filled = apply_repaired_lineup_forward(rows, after_idx, side, final)
        if filled:
            summary["clusters_repaired"] += 1
            summary["rows_filled"] += filled
            summary["implicit_subs"] += len(inferred)

        i = max(j, i + 1)

    return pd.DataFrame(rows, columns=game.columns), summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--team", required=True, help="Team abbreviation, e.g. LAL")
    parser.add_argument("--raw-dir", type=Path, default=Path("nba_pipeline/raw_data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    team_abbr = args.team.strip().upper()
    input_path = args.raw_dir / raw_name(args.year)
    output_path = args.output_dir / raw_name(args.year)
    df = pd.read_parquet(input_path)
    parts: list[pd.DataFrame] = []
    summaries: list[dict[str, int]] = []
    season_lookup = build_name_lookup(df, team_abbr)
    for _, game in df.groupby("game_id", sort=False):
        repaired, summary = repair_game(game, team_abbr, season_lookup)
        parts.append(repaired)
        if summary.get("games_touched"):
            summaries.append({"game_id": str(game["game_id"].iloc[0]), **summary})

    out = pd.concat(parts, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    aggregate = {
        "year": args.year,
        "team": team_abbr,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "games_touched": len(summaries),
        "clusters_seen": int(sum(s.get("clusters_seen", 0) for s in summaries)),
        "clusters_repaired": int(sum(s.get("clusters_repaired", 0) for s in summaries)),
        "rows_filled": int(sum(s.get("rows_filled", 0) for s in summaries)),
        "explicit_subs": int(sum(s.get("explicit_subs", 0) for s in summaries)),
        "implicit_subs": int(sum(s.get("implicit_subs", 0) for s in summaries)),
    }
    args.report.write_text(json.dumps({"summary": aggregate, "games": summaries}, indent=2) + "\n")
    print(json.dumps(aggregate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
