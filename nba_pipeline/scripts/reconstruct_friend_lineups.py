#!/usr/bin/env python3
"""
Reconstruct home/away lineups from a friend's substitution stream.

The friend PBP parquet carries off/def/player-on snapshots, but those snapshots
collapse during same-clock substitution batches. For RAPM we need a stable
five-man state, so this script treats substitution rows as an event stream:

- infer each game's home/away side from an official raw parquet
- seed starters from the first stable friend lineup, repaired by first-sub outs
  and early event actors
- apply all substitutions at the same game/period/clock atomically by team
- stamp reconstructed home_player1..5 and away_player1..5 on every row

The official parquet is used for side/team inference and validation on modern
seasons. The reconstruction itself is driven by the friend substitution stream.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRIEND_RAW = PIPELINE_ROOT / "raw_data" / "BKN_2020_rs.parquet"
DEFAULT_OFFICIAL_RAW = PIPELINE_ROOT / "raw_data" / "NBA20.parquet"
DEFAULT_OUTPUT = (
    PIPELINE_ROOT
    / "validation"
    / "friend_ingest"
    / "raw"
    / "BKN_2020_rs_friend_reconstructed_lineups.parquet"
)
DEFAULT_REPORT = PIPELINE_ROOT / "validation" / "friend_ingest" / "lineup_reconstruction_report.json"
DEFAULT_ANOMALIES = PIPELINE_ROOT / "validation" / "friend_ingest" / "lineup_reconstruction_anomalies.csv"


PLAYER_COLS_HOME = [f"home_player{i}" for i in range(1, 6)]
PLAYER_COLS_AWAY = [f"away_player{i}" for i in range(1, 6)]
ACTOR_COLUMNS = ["person_id", "assister_id", "stealPersonId", "blockPersonId"]
SIDE_NAMES = ("home", "away")


@dataclass(frozen=True)
class GameSides:
    home_team: str | None
    away_team: str | None


def normalize_game_id(value) -> int:
    return int(float(value))


def normalize_player_id(value) -> int | None:
    if pd.isna(value):
        return None
    try:
        player_id = int(float(value))
    except (TypeError, ValueError):
        return None
    return player_id if player_id > 0 else None


def parse_player_pipe(value) -> list[int]:
    if pd.isna(value):
        return []
    players: list[int] = []
    for part in str(value).split("|"):
        player_id = normalize_player_id(part.strip())
        if player_id is not None:
            players.append(player_id)
    return players


def parse_clock_seconds(value) -> int:
    if pd.isna(value):
        return 0
    match = re.search(r"(\d+):(\d+)", str(value))
    if not match:
        return 0
    return int(match.group(1)) * 60 + int(match.group(2))


def is_substitution(row: pd.Series) -> bool:
    return str(row.get("actionType", "")).lower() == "substitution"


def sub_kind(row: pd.Series) -> str | None:
    desc = "" if pd.isna(row.get("description")) else str(row.get("description")).lower()
    if "sub out" in desc:
        return "out"
    if "sub in" in desc:
        return "in"
    # Friend data sometimes has a compact "SUB: Player FOR Player" row. In
    # those rows person_id is the outgoing player; the incoming player appears
    # as a separate "SUB in" row only in some feeds, so keep this as an out.
    if desc.startswith("sub:") and " for " in desc:
        return "out"
    return None


def normalize_name_label(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compact_sub_in_label(row: pd.Series) -> str | None:
    desc = "" if pd.isna(row.get("description")) else str(row.get("description"))
    match = re.search(r"sub:\s*(.*?)\s+for\s+", desc, flags=re.IGNORECASE)
    if not match:
        return None
    label = normalize_name_label(match.group(1))
    return label or None


def build_name_lookup(friend_df: pd.DataFrame) -> dict[tuple[int, str], dict[str, int]]:
    lookup: dict[tuple[int, str], dict[str, int]] = defaultdict(dict)
    for _, row in friend_df.iterrows():
        player_id = normalize_player_id(row.get("person_id"))
        if player_id is None:
            continue
        team = row.get("team")
        if pd.isna(team):
            continue
        full = normalize_name_label(row.get("playerName"))
        if not full:
            continue
        labels = {full}
        parts = full.split()
        if parts:
            labels.add(parts[-1])
        if len(parts) >= 2 and parts[-1] in {"jr", "sr", "ii", "iii", "iv"}:
            labels.add(" ".join(parts[-2:]))
        key = (normalize_game_id(row["game_id"]), str(team))
        global_key = (0, str(team))
        for label in labels:
            lookup[key].setdefault(label, player_id)
            lookup[global_key].setdefault(label, player_id)
    return lookup


def unique_ordered(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def lineup_key(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted(int(v) for v in values if pd.notna(v)))


def infer_bkn_games(official_df: pd.DataFrame, friend_game_ids: set[int], team: str) -> dict[int, GameSides]:
    rows: dict[int, GameSides] = {}
    official_df = official_df[official_df["game_id"].isin(friend_game_ids)]
    for game_id, game in official_df.groupby("game_id"):
        home_events = game[
            game["home_description"].notna() & game["player1_team_abbreviation"].notna()
        ]["player1_team_abbreviation"]
        away_events = game[
            game["visitor_description"].notna() & game["player1_team_abbreviation"].notna()
        ]["player1_team_abbreviation"]
        home = home_events.mode().iat[0] if not home_events.mode().empty else None
        away = away_events.mode().iat[0] if not away_events.mode().empty else None
        if home == team or away == team:
            rows[int(game_id)] = GameSides(home_team=home, away_team=away)
    return rows


def build_side_player_sets(official_subset: pd.DataFrame) -> dict[int, dict[str, set[int]]]:
    side_sets: dict[int, dict[str, set[int]]] = {}
    for game_id, game in official_subset.groupby("game_id"):
        side_sets[int(game_id)] = {
            "home": set(
                pd.to_numeric(game[PLAYER_COLS_HOME].stack(), errors="coerce").dropna().astype(int)
            ),
            "away": set(
                pd.to_numeric(game[PLAYER_COLS_AWAY].stack(), errors="coerce").dropna().astype(int)
            ),
        }
    return side_sets


def map_friend_lineups_to_sides(row: pd.Series, side_sets: dict[str, set[int]]) -> dict[str, list[int]]:
    off = parse_player_pipe(row.get("off_players_on"))
    deff = parse_player_pipe(row.get("def_players_on"))
    home_set = side_sets.get("home", set())
    off_home_overlap = len(set(off) & home_set)
    def_home_overlap = len(set(deff) & home_set)
    if off_home_overlap >= def_home_overlap:
        return {"home": off, "away": deff}
    return {"home": deff, "away": off}


def team_to_side(team: str | None, sides: GameSides) -> str | None:
    if team is None or pd.isna(team):
        return None
    team = str(team)
    if team == sides.home_team:
        return "home"
    if team == sides.away_team:
        return "away"
    return None


def actor_side(row: pd.Series, actor_col: str, sides: GameSides) -> str | None:
    action = str(row.get("actionType", "")).lower()
    team_side = team_to_side(row.get("team"), sides)
    if actor_col in {"person_id", "assister_id", "stealPersonId", "blockPersonId"}:
        return team_side
    if actor_col == "foulDrawnPersonId" and team_side is not None:
        return "away" if team_side == "home" else "home"
    return None


def event_sort_frame(game: pd.DataFrame) -> pd.DataFrame:
    out = game.copy()
    out["_clock_seconds"] = out["clock_display"].map(parse_clock_seconds)
    out["_is_sub"] = out["actionType"].astype(str).str.lower().eq("substitution").astype(int)
    return out.sort_values(
        ["period", "_clock_seconds", "_is_sub", "actionNumber"],
        ascending=[True, False, True, True],
        kind="stable",
    )


def first_sub_records(game_sorted: pd.DataFrame, sides: GameSides, side: str) -> pd.DataFrame:
    team = sides.home_team if side == "home" else sides.away_team
    sub_rows = game_sorted[
        game_sorted["actionType"].astype(str).str.lower().eq("substitution")
        & game_sorted["team"].astype(str).eq(str(team))
    ]
    if sub_rows.empty:
        return sub_rows
    first = sub_rows.iloc[0]
    return sub_rows[
        (sub_rows["period"] == first["period"])
        & (sub_rows["clock_display"].astype(str) == str(first["clock_display"]))
    ]


def early_actors(game_sorted: pd.DataFrame, sides: GameSides, side: str, first_sub: pd.DataFrame) -> set[int]:
    if first_sub.empty:
        cutoff_period = int(game_sorted["period"].max()) + 1
        cutoff_clock = -1
    else:
        cutoff_period = int(first_sub.iloc[0]["period"])
        cutoff_clock = parse_clock_seconds(first_sub.iloc[0]["clock_display"])

    actors: set[int] = set()
    for _, row in game_sorted.iterrows():
        period = int(row["period"])
        clock_seconds = parse_clock_seconds(row.get("clock_display"))
        if period > cutoff_period or (period == cutoff_period and clock_seconds <= cutoff_clock):
            break
        if is_substitution(row):
            continue
        for col in ACTOR_COLUMNS:
            player_id = normalize_player_id(row.get(col))
            if player_id is not None and actor_side(row, col, sides) == side:
                actors.add(player_id)
    return actors


def initial_candidate_lineup(
    game_sorted: pd.DataFrame,
    side_sets: dict[str, set[int]],
    side: str,
) -> list[int]:
    for _, row in game_sorted.iterrows():
        if is_substitution(row):
            continue
        mapped = map_friend_lineups_to_sides(row, side_sets)
        candidate = unique_ordered(mapped.get(side, []))
        if len(candidate) == 5:
            return candidate
    return []


def period_start_lineups(period_group: pd.DataFrame, side_sets: dict[str, set[int]]) -> dict[str, list[int]]:
    for _, row in period_group.iterrows():
        if is_substitution(row):
            continue
        mapped = map_friend_lineups_to_sides(row, side_sets)
        home = unique_ordered(mapped.get("home", []))
        away = unique_ordered(mapped.get("away", []))
        if len(home) == 5 and len(away) == 5:
            return {"home": home, "away": away}
    return {"home": [], "away": []}


def repair_seed_to_five(
    seed: list[int],
    candidate: list[int],
    first_sub_outs: set[int],
    first_sub_ins: set[int],
    actors: set[int],
) -> list[int]:
    seed = unique_ordered(seed)

    # First-sub ins are the clearest evidence that a player should not be part
    # of the pre-sub starter state. First-sub outs and early actors are evidence
    # that a player should be in it.
    seed = [p for p in seed if p not in first_sub_ins]
    seed = unique_ordered(list(first_sub_outs) + [p for p in seed if p not in first_sub_outs])
    for player_id in sorted(actors):
        if player_id not in seed and len(seed) < 5:
            seed.append(player_id)

    if len(seed) < 5:
        for player_id in candidate:
            if player_id not in seed and player_id not in first_sub_ins:
                seed.append(player_id)
            if len(seed) == 5:
                break

    if len(seed) > 5:
        priority = {
            player_id: (
                int(player_id in first_sub_outs),
                int(player_id in actors),
                int(player_id in candidate),
            )
            for player_id in seed
        }
        seed = sorted(seed, key=lambda p: priority[p], reverse=True)[:5]

    return unique_ordered(seed)


def seed_lineups(
    game_sorted: pd.DataFrame,
    sides: GameSides,
    side_sets: dict[str, set[int]],
    diagnostics: list[dict],
) -> dict[str, list[int]]:
    lineups: dict[str, list[int]] = {}
    game_id = normalize_game_id(game_sorted.iloc[0]["game_id"])

    for side in SIDE_NAMES:
        candidate = initial_candidate_lineup(game_sorted, side_sets, side)
        first_sub = first_sub_records(game_sorted, sides, side)
        first_sub_outs: set[int] = set()
        first_sub_ins: set[int] = set()
        for _, row in first_sub.iterrows():
            player_id = normalize_player_id(row.get("person_id"))
            if player_id is None:
                continue
            if sub_kind(row) == "out":
                first_sub_outs.add(player_id)
            elif sub_kind(row) == "in":
                first_sub_ins.add(player_id)
        actors = early_actors(game_sorted, sides, side, first_sub)
        seed = repair_seed_to_five(
            seed=unique_ordered(candidate + sorted(actors)),
            candidate=candidate,
            first_sub_outs=first_sub_outs,
            first_sub_ins=first_sub_ins,
            actors=actors,
        )
        lineups[side] = seed
        if len(seed) != 5:
            diagnostics.append(
                {
                    "game_id": game_id,
                    "period": int(game_sorted.iloc[0]["period"]),
                    "clock_display": str(game_sorted.iloc[0]["clock_display"]),
                    "side": side,
                    "issue": "invalid_seed_count",
                    "details": "|".join(str(v) for v in seed),
                }
            )
        if lineup_key(seed) != lineup_key(candidate):
            diagnostics.append(
                {
                    "game_id": game_id,
                    "period": int(game_sorted.iloc[0]["period"]),
                    "clock_display": str(game_sorted.iloc[0]["clock_display"]),
                    "side": side,
                    "issue": "seed_repaired",
                    "details": json.dumps(
                        {
                            "candidate": candidate,
                            "seed": seed,
                            "first_sub_outs": sorted(first_sub_outs),
                            "first_sub_ins": sorted(first_sub_ins),
                            "early_actors": sorted(actors),
                        }
                    ),
                }
            )

    return lineups


def substitution_groups(game_sorted: pd.DataFrame) -> dict[tuple[int, int], pd.DataFrame]:
    sub_rows = game_sorted[game_sorted["actionType"].astype(str).str.lower().eq("substitution")].copy()
    groups: dict[tuple[int, int], pd.DataFrame] = {}
    for (period, clock_seconds), group in sub_rows.groupby(["period", "_clock_seconds"], sort=False):
        groups[(int(period), int(clock_seconds))] = group
    return groups


def apply_substitution_group(
    group: pd.DataFrame,
    sides: GameSides,
    side_sets: dict[str, set[int]],
    name_lookup: dict[tuple[int, str], dict[str, int]],
    lineups: dict[str, list[int]],
    diagnostics: list[dict],
) -> None:
    game_id = normalize_game_id(group.iloc[0]["game_id"])
    period = int(group.iloc[0]["period"])
    clock_display = str(group.iloc[0]["clock_display"])

    for side in SIDE_NAMES:
        rows = group[group["team"].map(lambda value: team_to_side(value, sides) == side)]
        if rows.empty:
            continue

        outs: list[int] = []
        ins: list[int] = []
        for _, row in rows.sort_values("actionNumber", kind="stable").iterrows():
            player_id = normalize_player_id(row.get("person_id"))
            if player_id is None:
                continue
            kind = sub_kind(row)
            if kind == "out":
                outs.append(player_id)
            elif kind == "in":
                ins.append(player_id)
            label = compact_sub_in_label(row)
            team = row.get("team")
            if label and pd.notna(team):
                incoming_id = name_lookup.get((game_id, str(team)), {}).get(label)
                if incoming_id is None:
                    incoming_id = name_lookup.get((0, str(team)), {}).get(label)
                if incoming_id is not None:
                    ins.append(incoming_id)
                else:
                    diagnostics.append(
                        {
                            "game_id": game_id,
                            "period": period,
                            "clock_display": clock_display,
                            "side": side,
                            "issue": "compact_sub_in_unresolved",
                            "details": label,
                        }
                    )

        current = list(lineups.get(side, []))
        for player_id in outs:
            if player_id in current:
                current.remove(player_id)
            else:
                diagnostics.append(
                    {
                        "game_id": game_id,
                        "period": period,
                        "clock_display": clock_display,
                        "side": side,
                        "issue": "sub_out_missing",
                        "details": str(player_id),
                    }
                )
        for player_id in ins:
            if player_id in current:
                diagnostics.append(
                    {
                        "game_id": game_id,
                        "period": period,
                        "clock_display": clock_display,
                        "side": side,
                        "issue": "sub_in_already_active",
                        "details": str(player_id),
                    }
                )
            else:
                current.append(player_id)

        current = unique_ordered(current)
        if len(current) != 5:
            final_snapshot = final_snapshot_for_side(rows, side_sets, side)
            if len(final_snapshot) == 5:
                diagnostics.append(
                    {
                        "game_id": game_id,
                        "period": period,
                        "clock_display": clock_display,
                        "side": side,
                        "issue": "sub_group_repaired_from_snapshot",
                        "details": json.dumps({"computed": current, "snapshot": final_snapshot}),
                    }
                )
                current = final_snapshot
            else:
                diagnostics.append(
                    {
                        "game_id": game_id,
                        "period": period,
                        "clock_display": clock_display,
                        "side": side,
                        "issue": "invalid_post_sub_count",
                        "details": "|".join(str(v) for v in current),
                    }
                )

        lineups[side] = current[:5]


def final_snapshot_for_side(rows: pd.DataFrame, side_sets: dict[str, set[int]], side: str) -> list[int]:
    if rows.empty:
        return []
    last = rows.sort_values("actionNumber", kind="stable").iloc[-1]
    # This is only a repair path for malformed substitution groups. It trusts
    # the row's off/def side labels less than the substitution stream, so use it
    # only when the computed stream state fails to produce five.
    return map_friend_lineups_to_sides(last, side_sets).get(side, [])[:5]


def prefer_valid_row_snapshots(
    reconstructed: pd.DataFrame,
    side_sets: dict[int, dict[str, set[int]]],
    diagnostics: list[dict],
) -> tuple[pd.DataFrame, dict]:
    out = reconstructed.copy()
    preferred_rows = 0
    invalid_snapshot_rows = 0

    for idx, row in out.iterrows():
        sets = side_sets.get(normalize_game_id(row["game_id"]))
        if sets is None:
            continue
        mapped = map_friend_lineups_to_sides(row, sets)
        home = unique_ordered(mapped.get("home", []))
        away = unique_ordered(mapped.get("away", []))
        if len(home) == 5 and len(away) == 5:
            for pos, player_id in enumerate(home, start=1):
                out.at[idx, f"home_player{pos}"] = player_id
            for pos, player_id in enumerate(away, start=1):
                out.at[idx, f"away_player{pos}"] = player_id
            preferred_rows += 1
        else:
            invalid_snapshot_rows += 1

    diagnostics.append(
        {
            "game_id": 0,
            "period": np.nan,
            "clock_display": "",
            "side": "both",
            "issue": "hybrid_snapshot_preference",
            "details": json.dumps(
                {
                    "preferred_valid_snapshot_rows": preferred_rows,
                    "invalid_snapshot_rows_repaired_by_stream": invalid_snapshot_rows,
                }
            ),
        }
    )
    return out, {
        "preferred_valid_snapshot_rows": preferred_rows,
        "invalid_snapshot_rows_repaired_by_stream": invalid_snapshot_rows,
    }


def stamp_lineups(
    friend_df: pd.DataFrame,
    sides_by_game: dict[int, GameSides],
    side_sets: dict[int, dict[str, set[int]]],
    prefer_snapshots: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    out_parts: list[pd.DataFrame] = []
    diagnostics: list[dict] = []
    name_lookup = build_name_lookup(friend_df)

    for game_id, game in friend_df.groupby("game_id", sort=True):
        game_id_int = normalize_game_id(game_id)
        sides = sides_by_game.get(game_id_int)
        sets = side_sets.get(game_id_int)
        if sides is None or sets is None:
            diagnostics.append(
                {
                    "game_id": game_id_int,
                    "period": np.nan,
                    "clock_display": "",
                    "side": "",
                    "issue": "missing_game_side_context",
                    "details": "",
                }
            )
            continue

        ordered = event_sort_frame(game)
        lineups: dict[str, list[int]] | None = None
        sub_groups = substitution_groups(ordered)
        stamped_rows: list[pd.Series] = []

        for period, period_group in ordered.groupby("period", sort=False):
            if lineups is None:
                lineups = seed_lineups(period_group, sides, sets, diagnostics)
            else:
                period_seed = period_start_lineups(period_group, sets)
                if all(len(period_seed.get(side, [])) == 5 for side in SIDE_NAMES):
                    if any(lineup_key(lineups.get(side, [])) != lineup_key(period_seed[side]) for side in SIDE_NAMES):
                        diagnostics.append(
                            {
                                "game_id": game_id_int,
                                "period": int(period),
                                "clock_display": str(period_group.iloc[0].get("clock_display", "")),
                                "side": "both",
                                "issue": "period_start_reseed",
                                "details": json.dumps({"previous": lineups, "seed": period_seed}),
                            }
                        )
                    lineups = period_seed

            for (period_value, clock_seconds), clock_group in period_group.groupby(["period", "_clock_seconds"], sort=False):
                key = (int(period_value), int(clock_seconds))
                group_subs = sub_groups.get(key)
                first_sub_action = (
                    float(group_subs["actionNumber"].min()) if group_subs is not None and not group_subs.empty else None
                )
                applied_sub_group = False

                for _, row in clock_group.sort_values("actionNumber", kind="stable").iterrows():
                    if (
                        first_sub_action is not None
                        and not applied_sub_group
                        and float(row["actionNumber"]) >= first_sub_action
                    ):
                        apply_substitution_group(group_subs, sides, sets, name_lookup, lineups, diagnostics)
                        applied_sub_group = True

                    stamped = row.copy()
                    for idx, player_id in enumerate(lineups.get("home", [])[:5], start=1):
                        stamped[f"home_player{idx}"] = player_id
                    for idx, player_id in enumerate(lineups.get("away", [])[:5], start=1):
                        stamped[f"away_player{idx}"] = player_id
                    stamped_rows.append(stamped)

        stamped_game = pd.DataFrame(stamped_rows).drop(columns=["_clock_seconds", "_is_sub"], errors="ignore")
        out_parts.append(stamped_game)

    if out_parts:
        out = pd.concat(out_parts, ignore_index=True, sort=False)
        out = out.sort_values(["game_id", "period", "actionNumber"], kind="stable").reset_index(drop=True)
    else:
        out = pd.DataFrame(columns=list(friend_df.columns) + PLAYER_COLS_HOME + PLAYER_COLS_AWAY)

    hybrid_summary: dict = {}
    if prefer_snapshots and not out.empty:
        out, hybrid_summary = prefer_valid_row_snapshots(out, side_sets, diagnostics)
    return out, pd.DataFrame(diagnostics), hybrid_summary


def validation_summary(reconstructed: pd.DataFrame, official_subset: pd.DataFrame) -> dict:
    rec = reconstructed.copy()
    rec["event_num"] = pd.to_numeric(rec["actionNumber"], errors="coerce").astype("Int64")
    rec["game_id"] = pd.to_numeric(rec["game_id"], errors="coerce").astype("Int64")

    official = official_subset[["game_id", "event_num", *PLAYER_COLS_HOME, *PLAYER_COLS_AWAY]].copy()
    official["game_id"] = pd.to_numeric(official["game_id"], errors="coerce").astype("Int64")
    official["event_num"] = pd.to_numeric(official["event_num"], errors="coerce").astype("Int64")

    merged = rec.merge(official, on=["game_id", "event_num"], how="inner", suffixes=("_recon", "_official"))
    real = merged[~merged["actionType"].astype(str).str.lower().isin({"substitution", "period"})].copy()
    if real.empty:
        return {"matched_rows": 0}

    home_match = pd.Series(
        [
            lineup_key(row[f"{c}_recon"] for c in PLAYER_COLS_HOME)
            == lineup_key(row[f"{c}_official"] for c in PLAYER_COLS_HOME)
            for _, row in real.iterrows()
        ],
        index=real.index,
    )
    away_match = pd.Series(
        [
            lineup_key(row[f"{c}_recon"] for c in PLAYER_COLS_AWAY)
            == lineup_key(row[f"{c}_official"] for c in PLAYER_COLS_AWAY)
            for _, row in real.iterrows()
        ],
        index=real.index,
    )
    both_match = home_match & away_match
    return {
        "matched_non_sub_rows": int(len(real)),
        "home_lineup_match_rows": int(home_match.sum()),
        "away_lineup_match_rows": int(away_match.sum()),
        "both_lineups_match_rows": int(both_match.sum()),
        "both_lineups_match_rate": float(both_match.mean()),
        "mismatch_rows": int((~both_match).sum()),
    }


def invalid_lineup_summary(reconstructed: pd.DataFrame) -> dict:
    home_counts = reconstructed[PLAYER_COLS_HOME].notna().sum(axis=1)
    away_counts = reconstructed[PLAYER_COLS_AWAY].notna().sum(axis=1)
    real_mask = ~reconstructed["actionType"].astype(str).str.lower().isin({"substitution", "period"})
    invalid = real_mask & ((home_counts != 5) | (away_counts != 5))
    return {
        "rows": int(len(reconstructed)),
        "real_non_sub_rows": int(real_mask.sum()),
        "invalid_real_non_sub_rows": int(invalid.sum()),
        "invalid_real_non_sub_rate": float(invalid.mean()) if len(invalid) else 0.0,
    }


def compact_anomaly_counts(anomalies: pd.DataFrame) -> dict[str, int]:
    if anomalies.empty:
        return {}
    return {str(k): int(v) for k, v in Counter(anomalies["issue"]).items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct friend PBP lineups from substitutions.")
    parser.add_argument("--friend-raw", type=Path, default=DEFAULT_FRIEND_RAW)
    parser.add_argument("--official-raw", type=Path, default=DEFAULT_OFFICIAL_RAW)
    parser.add_argument("--team", default="BKN")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--anomalies", type=Path, default=DEFAULT_ANOMALIES)
    parser.add_argument(
        "--prefer-stream",
        action="store_true",
        help="Use reconstructed substitution-stream lineups on all rows instead of only repairing invalid row snapshots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    friend_df = pd.read_parquet(args.friend_raw).copy()
    friend_df["game_id"] = friend_df["game_id"].map(normalize_game_id)
    official_df = pd.read_parquet(args.official_raw).copy()
    official_df["game_id"] = official_df["game_id"].map(normalize_game_id)

    friend_game_ids = set(friend_df["game_id"].unique())
    sides_by_game = infer_bkn_games(official_df, friend_game_ids, args.team)
    official_subset = official_df[official_df["game_id"].isin(sides_by_game.keys())].copy()
    side_sets = build_side_player_sets(official_subset)

    reconstructed, anomalies, hybrid_summary = stamp_lineups(
        friend_df,
        sides_by_game,
        side_sets,
        prefer_snapshots=not args.prefer_stream,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.anomalies.parent.mkdir(parents=True, exist_ok=True)
    reconstructed.to_parquet(args.output, index=False)
    anomalies.to_csv(args.anomalies, index=False)

    report = {
        "friend_raw": str(args.friend_raw),
        "official_raw": str(args.official_raw),
        "output": str(args.output),
        "games": int(len(sides_by_game)),
        "mode": "stream" if args.prefer_stream else "hybrid_snapshot_then_stream_repair",
        "hybrid_summary": hybrid_summary,
        "invalid_lineups": invalid_lineup_summary(reconstructed),
        "validation_against_official_raw": validation_summary(reconstructed, official_subset),
        "anomaly_counts": compact_anomaly_counts(anomalies),
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
