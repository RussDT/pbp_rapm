#!/usr/bin/env python3
"""Evidence-driven historical lineup repair harness.

This script is a validation-first repair layer for Gabriel-era NBA raw parquets.
It uses the independent NBA archive PlayByPlay event stream as hard evidence:

- event actors must be on court for their game/team side
- compact substitutions like "SUB: Weaver FOR Mason" are replayed atomically
- first-sub contradictions can repair bad opening lineups
- raw row snapshots are weak fallback anchors, not the primary truth

The default output goes to validation and does not overwrite production files.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = Path("/Users/russellthomas/Docs/2026_NBA_PIPELINE/archive (1)")

HOME_COLS = [f"home_player{i}" for i in range(1, 6)]
AWAY_COLS = [f"away_player{i}" for i in range(1, 6)]
SIDE_TO_COLS = {"home": HOME_COLS, "away": AWAY_COLS}
WEAVER_ID = 201602


@dataclass(frozen=True)
class SubMove:
    game_id: int
    team_id: int
    side: str
    action_number: int
    period: int
    clock_seconds: int
    incoming: int
    outgoing: int
    description: str


def normalize_player_id(value) -> int | None:
    if pd.isna(value):
        return None
    try:
        player_id = int(float(value))
    except (TypeError, ValueError):
        return None
    return player_id if player_id > 0 else None


def normalize_game_id(value) -> int:
    return int(str(value).lstrip("0") or "0")


def normalize_label(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_clock_seconds(value) -> int:
    if pd.isna(value):
        return 0
    text = str(value)
    iso = re.search(r"PT(\d+)M(\d+(?:\.\d+)?)S", text)
    if iso:
        return int(iso.group(1)) * 60 + int(float(iso.group(2)))
    mmss = re.search(r"(\d+):(\d+)", text)
    if mmss:
        return int(mmss.group(1)) * 60 + int(mmss.group(2))
    return 0


def unique_ordered(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        player_id = normalize_player_id(value)
        if player_id is None or player_id in seen:
            continue
        seen.add(player_id)
        out.append(player_id)
    return out


def lineup_from_row(row: pd.Series, cols: list[str]) -> list[int]:
    return unique_ordered(row.get(col) for col in cols)


def write_lineup(df: pd.DataFrame, idx, cols: list[str], lineup: list[int]) -> None:
    padded = unique_ordered(lineup)[:5]
    padded += [0] * (5 - len(padded))
    for col, player_id in zip(cols, padded):
        df.at[idx, col] = player_id


def fill_from_raw_snapshot(lineup: list[int], row: pd.Series, cols: list[str]) -> tuple[list[int], bool]:
    filled = unique_ordered(lineup)
    if len(filled) >= 5:
        return filled[:5], False
    for player_id in lineup_from_row(row, cols):
        if player_id not in filled:
            filled.append(player_id)
        if len(filled) == 5:
            return filled, True
    return filled, len(filled) > len(lineup)


def infer_game_teams_from_raw(raw_game: pd.DataFrame) -> tuple[int | None, int | None]:
    player_ids = pd.to_numeric(raw_game.get("player1_id"), errors="coerce").fillna(0)
    team_ids = pd.to_numeric(raw_game.get("player1_team_id"), errors="coerce")
    valid = raw_game[(player_ids > 0) & team_ids.notna()].copy()
    if valid.empty:
        return None, None

    home_counts = (
        valid[valid["home_description"].notna()]
        .assign(team_id=team_ids.loc[valid[valid["home_description"].notna()].index].astype(int))
        ["team_id"]
        .value_counts()
    )
    away_counts = (
        valid[valid["visitor_description"].notna()]
        .assign(team_id=team_ids.loc[valid[valid["visitor_description"].notna()].index].astype(int))
        ["team_id"]
        .value_counts()
    )
    home = int(home_counts.index[0]) if not home_counts.empty else None
    away = int(away_counts.index[0]) if not away_counts.empty else None
    if home is not None and away is not None and home != away:
        return home, away
    return home, away


def load_player_statistics(game_ids: set[int]) -> pd.DataFrame:
    usecols = [
        "firstName",
        "lastName",
        "personId",
        "gameId",
        "playerteamId",
        "opponentteamId",
        "home",
        "numMinutes",
        "comment",
    ]
    stats = pd.read_csv(ARCHIVE_ROOT / "PlayerStatistics.csv", usecols=usecols, low_memory=False)
    stats["game_id_int"] = pd.to_numeric(stats["gameId"], errors="coerce").fillna(-1).astype(int)
    stats["person_id_int"] = pd.to_numeric(stats["personId"], errors="coerce").fillna(0).astype(int)
    stats["team_id_int"] = pd.to_numeric(stats["playerteamId"], errors="coerce").fillna(0).astype(int)
    return stats[stats["game_id_int"].isin(game_ids)].copy()


def build_name_lookup(stats: pd.DataFrame) -> dict[int, dict[int, dict[str, int]]]:
    lookup: dict[int, dict[int, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    for _, row in stats.iterrows():
        player_id = normalize_player_id(row.get("person_id_int"))
        team_id = normalize_player_id(row.get("team_id_int"))
        if player_id is None or team_id is None:
            continue
        labels = set()
        full = normalize_label(f"{row.get('firstName', '')} {row.get('lastName', '')}")
        first = normalize_label(row.get("firstName"))
        last = normalize_label(row.get("lastName"))
        if full:
            labels.add(full)
        if last:
            labels.add(last)
        if first and last:
            labels.add(f"{first[0]} {last}")
        for label in labels:
            lookup[int(row["game_id_int"])][team_id].setdefault(label, player_id)
    return lookup


def build_player_side_maps(
    stats: pd.DataFrame,
    game_team_sides: dict[int, dict[str, int]],
) -> dict[int, dict[int, str]]:
    maps: dict[int, dict[int, str]] = defaultdict(dict)
    for _, row in stats.iterrows():
        game_id = int(row["game_id_int"])
        team_id = normalize_player_id(row.get("team_id_int"))
        player_id = normalize_player_id(row.get("person_id_int"))
        if team_id is None or player_id is None:
            continue
        sides = game_team_sides.get(game_id, {})
        if team_id == sides.get("home"):
            maps[game_id][player_id] = "home"
        elif team_id == sides.get("away"):
            maps[game_id][player_id] = "away"
    return maps


def load_archive(game_ids: set[int]) -> pd.DataFrame:
    cols = [
        "gameId",
        "actionNumber",
        "period",
        "clock",
        "actionType",
        "description",
        "personId",
        "playerName",
        "playerFullName",
        "teamId",
        "teamTricode",
        "assistPersonId",
        "stealPersonId",
        "blockPersonId",
        "foulDrawnPersonId",
        "jumpBallRecoveredPersonId",
        "jumpBallWonPersonId",
        "jumpBallLostPersonId",
    ]
    archive = pd.read_parquet(
        ARCHIVE_ROOT / "PlayByPlay.parquet",
        columns=cols,
        filters=[("gameId", "in", [str(g) for g in sorted(game_ids)])],
    )
    archive["game_id_int"] = archive["gameId"].map(normalize_game_id)
    archive["_clock_seconds"] = archive["clock"].map(parse_clock_seconds)
    archive["team_id_int"] = pd.to_numeric(archive["teamId"], errors="coerce").fillna(0).astype(int)
    return archive.sort_values(
        ["game_id_int", "period", "_clock_seconds", "actionNumber"],
        ascending=[True, True, False, True],
        kind="stable",
    ).reset_index(drop=True)


def add_archive_names(lookup: dict[int, dict[int, dict[str, int]]], archive: pd.DataFrame) -> None:
    for _, row in archive.iterrows():
        game_id = normalize_game_id(row["gameId"])
        team_id = normalize_player_id(row.get("teamId"))
        player_id = normalize_player_id(row.get("personId"))
        if team_id is None or player_id is None:
            continue
        labels = {normalize_label(row.get("playerName")), normalize_label(row.get("playerFullName"))}
        for label in list(labels):
            parts = label.split()
            if parts:
                labels.add(parts[-1])
            if len(parts) >= 2:
                labels.add(f"{parts[0][0]} {parts[-1]}")
        for label in labels:
            if label:
                lookup[game_id][team_id].setdefault(label, player_id)


def side_for_team(team_id: int | None, sides: dict[str, int]) -> str | None:
    if team_id == sides.get("home"):
        return "home"
    if team_id == sides.get("away"):
        return "away"
    return None


def opposite(side: str | None) -> str | None:
    if side == "home":
        return "away"
    if side == "away":
        return "home"
    return None


def actor_side_from_maps(
    player_id: int | None,
    game_id: int,
    player_side_map: dict[int, dict[int, str]],
) -> str | None:
    if player_id is None:
        return None
    return player_side_map.get(game_id, {}).get(player_id)


def event_actor_sides(
    row: pd.Series,
    sides: dict[str, int],
    player_side_map: dict[int, dict[int, str]],
) -> dict[str, set[int]]:
    game_id = normalize_game_id(row["gameId"])
    team_id = normalize_player_id(row.get("teamId"))
    action_side = side_for_team(team_id, sides)
    defense_side = opposite(action_side)
    actors: dict[str, set[int]] = {"home": set(), "away": set()}

    def add(player_value, fallback_side: str | None) -> None:
        player_id = normalize_player_id(player_value)
        if player_id is None:
            return
        side = actor_side_from_maps(player_id, game_id, player_side_map) or fallback_side
        if side in actors:
            actors[side].add(player_id)

    for col in ["personId", "assistPersonId", "jumpBallRecoveredPersonId", "jumpBallWonPersonId", "jumpBallLostPersonId"]:
        add(row.get(col), action_side)
    for col in ["stealPersonId", "blockPersonId", "foulDrawnPersonId"]:
        add(row.get(col), defense_side)
    return actors


def parse_compact_sub(
    row: pd.Series,
    sides: dict[str, int],
    lookup: dict[int, dict[int, dict[str, int]]],
) -> SubMove | None:
    if str(row.get("actionType", "")).lower() != "substitution":
        return None
    team_id = normalize_player_id(row.get("teamId"))
    if team_id is None:
        return None
    side = side_for_team(team_id, sides)
    if side is None:
        return None
    desc = "" if pd.isna(row.get("description")) else str(row.get("description"))
    match = re.search(r"sub:\s*(.*?)\s+for\s+(.*)$", desc, flags=re.IGNORECASE)
    if not match:
        return None
    game_id = normalize_game_id(row["gameId"])
    incoming_label = normalize_label(match.group(1))
    candidates = [incoming_label]
    parts = incoming_label.split()
    if parts:
        candidates.append(parts[-1])
    if len(parts) >= 2:
        candidates.append(f"{parts[0][0]} {parts[-1]}")

    incoming = None
    for candidate in candidates:
        incoming = lookup.get(game_id, {}).get(team_id, {}).get(candidate)
        if incoming is not None:
            break
    outgoing = normalize_player_id(row.get("personId"))
    if incoming is None or outgoing is None:
        return None
    return SubMove(
        game_id=game_id,
        team_id=team_id,
        side=side,
        action_number=int(row["actionNumber"]),
        period=int(row["period"]),
        clock_seconds=parse_clock_seconds(row.get("clock")),
        incoming=int(incoming),
        outgoing=int(outgoing),
        description=desc,
    )


def first_complete_after(
    raw_game: pd.DataFrame,
    cols: list[str],
    action_number: int,
    must_include: set[int] | None = None,
    must_exclude: set[int] | None = None,
) -> list[int]:
    must_include = must_include or set()
    must_exclude = must_exclude or set()
    after = raw_game[pd.to_numeric(raw_game["source_actionNumber"], errors="coerce") > action_number]
    for _, row in after.sort_values("event_num", kind="stable").iterrows():
        lineup = lineup_from_row(row, cols)
        lineup_set = set(lineup)
        if len(lineup) == 5 and must_include.issubset(lineup_set) and not (must_exclude & lineup_set):
            return lineup
    return []


def hard_actors_before_first_sub(
    events: pd.DataFrame,
    side: str,
    first_moves: list[SubMove],
    sides: dict[str, int],
    player_side_map: dict[int, dict[int, str]],
) -> set[int]:
    if not first_moves:
        return set()
    first_period = first_moves[0].period
    first_clock = first_moves[0].clock_seconds
    actors: set[int] = set()
    for _, row in events.iterrows():
        period = int(row["period"])
        clock = int(row["_clock_seconds"])
        if period > first_period or (period == first_period and clock <= first_clock):
            break
        actors.update(event_actor_sides(row, sides, player_side_map)[side])
    return actors


def seed_lineup(
    raw_game: pd.DataFrame,
    cols: list[str],
    side: str,
    moves: list[SubMove],
    events: pd.DataFrame,
    sides: dict[str, int],
    player_side_map: dict[int, dict[int, str]],
    diagnostics: list[dict],
) -> list[int]:
    game_id = int(raw_game.iloc[0]["game_id_int"])
    first = raw_game.sort_values("event_num", kind="stable").iloc[0]
    candidate = lineup_from_row(first, cols)
    seed = unique_ordered(candidate)
    if not moves:
        return seed[:5]

    first_key = (moves[0].period, moves[0].clock_seconds)
    first_moves = [m for m in moves if (m.period, m.clock_seconds) == first_key]
    first_ins = {m.incoming for m in first_moves}
    first_outs = {m.outgoing for m in first_moves}
    first_action = min(m.action_number for m in first_moves)
    later = first_complete_after(raw_game, cols, first_action, must_include=first_ins, must_exclude=first_outs)

    if later:
        seed = unique_ordered([p for p in later if p not in first_ins] + list(first_outs))
    else:
        seed = [p for p in seed if p not in first_ins]
        seed = unique_ordered(list(first_outs) + seed)
        fallback_later = first_complete_after(raw_game, cols, first_action)
        for player_id in fallback_later:
            if player_id not in seed and player_id not in first_ins:
                seed.append(player_id)
            if len(seed) == 5:
                break

    early_actors = hard_actors_before_first_sub(events, side, first_moves, sides, player_side_map)
    for actor in sorted(early_actors):
        if actor not in seed:
            if len(seed) < 5:
                seed.append(actor)
            else:
                seed = seed[1:] + [actor]

    if len(seed) < 5:
        for player_id in candidate:
            if player_id not in seed and player_id not in first_ins:
                seed.append(player_id)
            if len(seed) == 5:
                break

    seed = unique_ordered(seed)[:5]
    if set(seed) != set(candidate):
        diagnostics.append(
            {
                "game_id": game_id,
                "side": side,
                "issue": "starter_seed_repaired_from_first_sub",
                "action_number": first_action,
                "details": json.dumps(
                    {
                        "candidate": candidate,
                        "seed": seed,
                        "first_ins": sorted(first_ins),
                        "first_outs": sorted(first_outs),
                        "later_complete": later,
                        "early_actors": sorted(early_actors),
                    }
                ),
            }
        )
    return seed


def build_next_seen(actor_sets: list[set[int]]) -> list[dict[int, int]]:
    next_seen: dict[int, int] = {}
    out = [None] * len(actor_sets)
    for pos in range(len(actor_sets) - 1, -1, -1):
        out[pos] = dict(next_seen)
        for player_id in actor_sets[pos]:
            next_seen[player_id] = pos
    return out


def evict_for_actor(
    current: list[int],
    actor: int,
    event_pos: int,
    next_sub_pos: int | None,
    next_seen_by_pos: list[dict[int, int]],
    last_seen: dict[int, int],
) -> tuple[list[int], int | None]:
    if actor in current:
        return current, None
    if len(current) < 5:
        return unique_ordered(current + [actor]), None

    next_seen = next_seen_by_pos[event_pos]
    best_player = None
    best_score = None
    for player_id in current:
        next_pos = next_seen.get(player_id)
        next_before_sub = next_pos is not None and (next_sub_pos is None or next_pos < next_sub_pos)
        score = (
            int(next_before_sub) * 1000,
            last_seen.get(player_id, -10_000),
            -player_id,
        )
        if best_score is None or score < best_score:
            best_score = score
            best_player = player_id
    repaired = [p for p in current if p != best_player] + [actor]
    return unique_ordered(repaired), best_player


def repair_side(
    raw_game: pd.DataFrame,
    events: pd.DataFrame,
    side: str,
    sides: dict[str, int],
    lookup: dict[int, dict[int, dict[str, int]]],
    player_side_map: dict[int, dict[int, str]],
    diagnostics: list[dict],
) -> tuple[dict[int, list[int]], Counter]:
    cols = SIDE_TO_COLS[side]
    moves_all = [m for _, row in events.iterrows() if (m := parse_compact_sub(row, sides, lookup)) is not None]
    moves = [m for m in moves_all if m.side == side]
    move_groups: dict[tuple[int, int], list[SubMove]] = defaultdict(list)
    for move in moves:
        move_groups[(move.period, move.clock_seconds)].append(move)
    first_action_for_group = {key: min(m.action_number for m in group) for key, group in move_groups.items()}
    moves_by_first_action: dict[int, list[SubMove]] = {
        action: move_groups[key] for key, action in first_action_for_group.items()
    }

    actor_sets = [event_actor_sides(row, sides, player_side_map)[side] for _, row in events.iterrows()]
    sub_positions = [
        pos
        for pos, row in events.iterrows()
        if int(row["actionNumber"]) in moves_by_first_action
    ]
    next_sub_pos_for_event: list[int | None] = []
    for pos in range(len(events)):
        later = [sub_pos for sub_pos in sub_positions if sub_pos > pos]
        next_sub_pos_for_event.append(later[0] if later else None)
    next_seen_by_pos = build_next_seen(actor_sets)
    current = seed_lineup(raw_game, cols, side, moves, events, sides, player_side_map, diagnostics)
    last_seen: dict[int, int] = {}
    state_by_action: dict[int, list[int]] = {}
    conflicts = Counter()
    game_id = int(raw_game.iloc[0]["game_id_int"])

    for event_pos, (_, row) in enumerate(events.iterrows()):
        action = int(row["actionNumber"])
        for actor in sorted(actor_sets[event_pos]):
            if actor not in current:
                before = list(current)
                current, evicted = evict_for_actor(
                    current,
                    actor,
                    event_pos,
                    next_sub_pos_for_event[event_pos],
                    next_seen_by_pos,
                    last_seen,
                )
                conflicts["hard_actor_inserted"] += 1
                diagnostics.append(
                    {
                        "game_id": game_id,
                        "side": side,
                        "issue": "hard_actor_inserted",
                        "action_number": action,
                        "details": json.dumps(
                            {
                                "actor": actor,
                                "evicted": evicted,
                                "before": before,
                                "after": current,
                                "description": row.get("description"),
                            }
                        ),
                    }
                )
            last_seen[actor] = event_pos

        if action in moves_by_first_action:
            before = list(current)
            group = moves_by_first_action[action]
            for move in group:
                if move.incoming in current:
                    current = [p for p in current if p != move.incoming]
                    conflicts["sub_in_already_active_removed"] += 1
                if move.outgoing not in current:
                    current, _ = evict_for_actor(
                        current,
                        move.outgoing,
                        event_pos,
                        event_pos + 1,
                        next_seen_by_pos,
                        last_seen,
                    )
                    conflicts["sub_out_missing_inserted"] += 1
            for move in group:
                current = [p for p in current if p != move.outgoing]
            for move in group:
                if move.incoming not in current:
                    current.append(move.incoming)
            current = unique_ordered(current)[:5]
            if len(current) != 5:
                conflicts["invalid_post_sub_count"] += 1
            if set(before) != set(current):
                diagnostics.append(
                    {
                        "game_id": game_id,
                        "side": side,
                        "issue": "sub_group_applied",
                        "action_number": action,
                        "details": json.dumps(
                            {
                                "before": before,
                                "after": current,
                                "moves": [
                                    {
                                        "incoming": m.incoming,
                                        "outgoing": m.outgoing,
                                        "description": m.description,
                                    }
                                    for m in group
                                ],
                            }
                        ),
                    }
                )
        state_by_action[action] = list(current)

    return state_by_action, conflicts


def sanitize_wrong_side(row: pd.Series, side: str, lineup: list[int], player_side_map: dict[int, str]) -> tuple[list[int], int]:
    kept = []
    removed = 0
    for player_id in unique_ordered(lineup):
        known_side = player_side_map.get(player_id)
        if known_side is not None and known_side != side:
            removed += 1
            continue
        kept.append(player_id)
    return kept, removed


def repair_game(
    raw_game: pd.DataFrame,
    events: pd.DataFrame,
    sides: dict[str, int],
    lookup: dict[int, dict[int, dict[str, int]]],
    player_side_maps: dict[int, dict[int, str]],
    diagnostics: list[dict],
) -> pd.DataFrame:
    game_id = int(raw_game.iloc[0]["game_id_int"])
    repaired = raw_game.copy()
    if events.empty:
        diagnostics.append({"game_id": game_id, "side": "", "issue": "missing_archive_game", "action_number": 0, "details": ""})
        return repaired

    state_by_side = {}
    conflict_summary = {}
    for side in ("home", "away"):
        state_by_action, conflicts = repair_side(
            raw_game,
            events,
            side,
            sides,
            lookup,
            player_side_maps,
            diagnostics,
        )
        state_by_side[side] = state_by_action
        conflict_summary[side] = dict(conflicts)

    sorted_actions_by_side = {side: sorted(states) for side, states in state_by_side.items()}
    action_idx_by_side = {"home": 0, "away": 0}
    current_state_by_side = {}
    game_player_side = player_side_maps.get(game_id, {})
    game_conflicts = Counter()

    for side, actions in sorted_actions_by_side.items():
        cols = SIDE_TO_COLS[side]
        current_state_by_side[side] = (
            state_by_side[side][actions[0]] if actions else lineup_from_row(repaired.iloc[0], cols)
        )

    for idx, row in repaired.sort_values("event_num", kind="stable").iterrows():
        action = int(pd.to_numeric(row.get("source_actionNumber"), errors="coerce"))
        for side in ("home", "away"):
            actions = sorted_actions_by_side[side]
            while action_idx_by_side[side] < len(actions) and actions[action_idx_by_side[side]] <= action:
                current_state_by_side[side] = state_by_side[side][actions[action_idx_by_side[side]]]
                action_idx_by_side[side] += 1
            cols = SIDE_TO_COLS[side]
            stamped, filled = fill_from_raw_snapshot(current_state_by_side[side], row, cols)
            stamped, removed = sanitize_wrong_side(row, side, stamped, game_player_side)
            if filled:
                game_conflicts[f"{side}_row_state_filled_from_raw_snapshot"] += 1
            if removed:
                game_conflicts[f"{side}_wrong_side_removed"] += removed
            write_lineup(repaired, idx, cols, stamped)

    if conflict_summary or game_conflicts:
        diagnostics.append(
            {
                "game_id": game_id,
                "side": "both",
                "issue": "game_conflict_summary",
                "action_number": 0,
                "details": json.dumps({"side_conflicts": conflict_summary, "stamp_conflicts": dict(game_conflicts)}),
            }
        )
    return repaired


def row_overlap_counts(df: pd.DataFrame) -> dict[str, int]:
    raw_overlap = 0
    home_dup = 0
    away_dup = 0
    incomplete = 0
    for _, row in df.iterrows():
        home = lineup_from_row(row, HOME_COLS)
        away = lineup_from_row(row, AWAY_COLS)
        if set(home) & set(away):
            raw_overlap += 1
        if len(home) != len(set(home)):
            home_dup += 1
        if len(away) != len(set(away)):
            away_dup += 1
        if len(home) < 5 or len(away) < 5:
            incomplete += 1
    return {
        "same_player_home_away_rows": raw_overlap,
        "home_duplicate_rows": home_dup,
        "away_duplicate_rows": away_dup,
        "incomplete_rows": incomplete,
    }


def processed_overlap_counts(df: pd.DataFrame) -> dict[str, int]:
    o_cols = [f"O{i}" for i in range(1, 6)]
    d_cols = [f"D{i}" for i in range(1, 6)]
    overlap = 0
    incomplete = 0
    rows_with_zero = 0
    for _, row in df.iterrows():
        offense = unique_ordered(row.get(col) for col in o_cols)
        defense = unique_ordered(row.get(col) for col in d_cols)
        if set(offense) & set(defense):
            overlap += 1
        if len(offense) < 5 or len(defense) < 5:
            incomplete += 1
        if any(normalize_player_id(row.get(col)) is None for col in o_cols + d_cols):
            rows_with_zero += 1
    return {
        "same_player_off_def_rows": overlap,
        "incomplete_rows": incomplete,
        "rows_with_zero": rows_with_zero,
    }


def player_oncourt_summary(rapm: pd.DataFrame, player_id: int) -> dict[str, float]:
    o_cols = [f"O{i}" for i in range(1, 6)]
    d_cols = [f"D{i}" for i in range(1, 6)]
    o_mask = rapm[o_cols].apply(pd.to_numeric, errors="coerce").eq(player_id).any(axis=1)
    d_mask = rapm[d_cols].apply(pd.to_numeric, errors="coerce").eq(player_id).any(axis=1)
    pf = float(pd.to_numeric(rapm.loc[o_mask, "Off_Diff"], errors="coerce").fillna(0).sum())
    pa = float(pd.to_numeric(rapm.loc[d_mask, "Def_Diff"], errors="coerce").fillna(0).sum())
    return {
        "off_poss": int(o_mask.sum()),
        "def_poss": int(d_mask.sum()),
        "points_for": pf,
        "points_against": pa,
        "ortg": pf / o_mask.sum() * 100 if o_mask.sum() else np.nan,
        "drtg": pa / d_mask.sum() * 100 if d_mask.sum() else np.nan,
        "netrtg": pf / o_mask.sum() * 100 - pa / d_mask.sum() * 100 if o_mask.sum() and d_mask.sum() else np.nan,
    }


@contextlib.contextmanager
def maybe_disable_player_stats(disable: bool):
    if not disable:
        yield
        return
    saved = {key: os.environ.get(key) for key in ["SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_KEY"]}
    try:
        for key in saved:
            os.environ.pop(key, None)
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def process_rapm(output_raw: Path, output_rapm: Path, year: int, season: str, disable_player_stats: bool) -> pd.DataFrame:
    sys.path.insert(0, str(PIPELINE_ROOT / "scripts"))
    from process_rapm_blocks.process_rapm import process_rapm_py

    with maybe_disable_player_stats(disable_player_stats):
        rapm, _ = process_rapm_py(
            str(output_raw),
            year,
            season,
            o_luck=0.0,
            d_luck=0.0,
            missing_ft_fallback="actual",
        )
    if rapm is None:
        raise RuntimeError("RAPM processing returned no output")
    rapm.to_parquet(output_rapm, index=False)
    return rapm


def repair_season(args: argparse.Namespace) -> None:
    year_short = args.year % 100
    season = f"{args.year - 1}-{year_short:02d}"
    input_raw = Path(args.input_raw) if args.input_raw else PIPELINE_ROOT / "raw_data" / f"NBA{year_short:02d}.parquet"
    out_dir = Path(args.output_dir) if args.output_dir else PIPELINE_ROOT / "validation" / f"historical_lineup_evidence_repair_{args.year}"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_raw = out_dir / f"NBA{year_short:02d}_evidence_repair.parquet"
    rapm_suffix = "actual_all_ft" if args.disable_player_stats else "actual_missing_ft"
    output_rapm = out_dir / f"RAPM{year_short:02d}_evidence_repair_{rapm_suffix}.parquet"
    diagnostics_csv = out_dir / f"NBA{year_short:02d}_evidence_repair_diagnostics.csv"
    summary_json = out_dir / f"NBA{year_short:02d}_evidence_repair_summary.json"

    raw = pd.read_parquet(input_raw)
    raw["game_id_int"] = raw["game_id"].map(normalize_game_id)
    game_ids = set(raw["game_id_int"].unique())

    game_team_sides: dict[int, dict[str, int]] = {}
    for game_id, game in raw.groupby("game_id_int", sort=True):
        home, away = infer_game_teams_from_raw(game)
        if home is not None and away is not None:
            game_team_sides[int(game_id)] = {"home": int(home), "away": int(away)}

    stats = load_player_statistics(game_ids)
    for game_id, group in stats.groupby("game_id_int"):
        if int(game_id) in game_team_sides:
            continue
        home_team = group[group["home"].astype(bool)]["team_id_int"].dropna().unique()
        away_team = group[~group["home"].astype(bool)]["team_id_int"].dropna().unique()
        if len(home_team) and len(away_team):
            game_team_sides[int(game_id)] = {"home": int(home_team[0]), "away": int(away_team[0])}

    player_side_maps = build_player_side_maps(stats, game_team_sides)
    name_lookup = build_name_lookup(stats)
    archive = load_archive(game_ids)
    add_archive_names(name_lookup, archive)

    diagnostics: list[dict] = []
    parts: list[pd.DataFrame] = []
    missing_side_context = 0
    for game_id, raw_game in raw.groupby("game_id_int", sort=True):
        sides = game_team_sides.get(int(game_id))
        if sides is None:
            missing_side_context += 1
            diagnostics.append(
                {"game_id": int(game_id), "side": "", "issue": "missing_game_team_sides", "action_number": 0, "details": ""}
            )
            parts.append(raw_game)
            continue
        events = archive[archive["game_id_int"].eq(int(game_id))]
        parts.append(repair_game(raw_game, events, sides, name_lookup, player_side_maps, diagnostics))

    repaired = pd.concat(parts, ignore_index=True)
    repaired = repaired.sort_values(["game_id", "event_num"], kind="stable").drop(columns=["game_id_int"], errors="ignore")
    repaired.to_parquet(output_raw, index=False)

    baseline_raw_counts = row_overlap_counts(raw.drop(columns=["game_id_int"], errors="ignore"))
    repaired_raw_counts = row_overlap_counts(repaired)
    rapm = process_rapm(output_raw, output_rapm, args.year, season, args.disable_player_stats)
    baseline_rapm_path = PIPELINE_ROOT / "processed" / f"RAPM{year_short:02d}.parquet"
    baseline_rapm = pd.read_parquet(baseline_rapm_path) if baseline_rapm_path.exists() else pd.DataFrame()
    baseline_rapm_counts = processed_overlap_counts(baseline_rapm) if not baseline_rapm.empty else {}
    repaired_rapm_counts = processed_overlap_counts(rapm)

    diag_df = pd.DataFrame(diagnostics)
    diag_df.to_csv(diagnostics_csv, index=False)
    summary = {
        "input_raw": str(input_raw),
        "output_raw": str(output_raw),
        "output_rapm": str(output_rapm),
        "player_stats_disabled": bool(args.disable_player_stats),
        "ft_scoring_mode": "all_actual_ft" if args.disable_player_stats else "player_stats_with_actual_missing_ft_fallback",
        "year": args.year,
        "season": season,
        "games": int(repaired["game_id"].nunique()),
        "missing_side_context_games": missing_side_context,
        "raw_rows": int(len(raw)),
        "repaired_raw_rows": int(len(repaired)),
        "rapm_rows": int(len(rapm)),
        "diagnostic_issue_counts": dict(Counter(item["issue"] for item in diagnostics)),
        "baseline_raw_counts": baseline_raw_counts,
        "repaired_raw_counts": repaired_raw_counts,
        "baseline_rapm_counts": baseline_rapm_counts,
        "repaired_rapm_counts": repaired_rapm_counts,
        "weaver_baseline": player_oncourt_summary(baseline_rapm, WEAVER_ID) if not baseline_rapm.empty else {},
        "weaver_repaired": player_oncourt_summary(rapm, WEAVER_ID),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.promote:
        production_raw = PIPELINE_ROOT / "raw_data" / f"NBA{year_short:02d}.parquet"
        production_rapm = PIPELINE_ROOT / "processed" / f"RAPM{year_short:02d}.parquet"
        backup_raw = out_dir / production_raw.name.replace(".parquet", "_pre_evidence_backup.parquet")
        backup_rapm = out_dir / production_rapm.name.replace(".parquet", "_pre_evidence_backup.parquet")
        shutil.copy2(production_raw, backup_raw)
        shutil.copy2(production_rapm, backup_rapm)
        shutil.copy2(output_raw, production_raw)
        shutil.copy2(output_rapm, production_rapm)
        summary["promoted"] = {
            "production_raw": str(production_raw),
            "production_rapm": str(production_rapm),
            "backup_raw": str(backup_raw),
            "backup_rapm": str(backup_rapm),
        }
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair historical NBA lineup parquets using archive event evidence.")
    parser.add_argument("--year", type=int, required=True, help="Season end year, e.g. 2009")
    parser.add_argument("--input-raw", help="Input raw NBA parquet. Defaults to nba_pipeline/raw_data/NBA{YY}.parquet")
    parser.add_argument("--output-dir", help="Validation output directory")
    parser.add_argument(
        "--disable-player-stats",
        action="store_true",
        help="Clear Supabase player stats during RAPM processing so FT values use actual makes/misses.",
    )
    parser.add_argument("--promote", action="store_true", help="After validation output, overwrite production NBA/RAPM parquets with backups")
    return parser.parse_args()


if __name__ == "__main__":
    repair_season(parse_args())
