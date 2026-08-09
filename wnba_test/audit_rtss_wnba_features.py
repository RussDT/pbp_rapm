#!/usr/bin/env python3
"""Audit the local WNBA Second Spectrum/RTSS marking export.

The audit is deliberately read-only. It profiles the 3D markings files, validates
their possession lineups, builds a provider-UUID -> NBA/WNBA player-id crosswalk
from exact play-by-play event ids, and emits a modeling-oriented feature catalog.

Signed download URLs in the portal metadata are never copied to audit outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ROOT = Path("downloads/rtss-data/wnba")
DEFAULT_OUTPUT = Path("wnba_test/Results/rtss_feature_audit")

CORE_TABLES = {
    "possessions",
    "chances",
    "shots",
    "rebounds",
    "turnovers",
    "free_throws",
    "fouls",
    "passes",
    "touches",
    "dribbles",
    "picks",
    "isolations",
    "handoffs",
    "drives",
    "posts",
    "off_ball_screens",
    "fake_handoffs",
    "closeouts",
    "chance_players",
    "rotations",
    "poss_matchups",
    "speed_load",
    "shot_contests",
    "player_stints",
    "jumps",
    "boxout_opportunities",
    "falls",
    "missed_closeouts",
    "defender_stance",
}

CATEGORICAL_FIELDS = {
    "possessions": {"startType", "outcome", "zone"},
    "chances": {"startType", "outcome", "transition", "category", "zone"},
    "shots": {
        "region",
        "shotType",
        "complexShotType",
        "complexOutcome",
        "contestLevel",
        "finishType",
        "jumperType",
    },
    "turnovers": {"turnoverType", "pbpSubtype", "region"},
    "fouls": {"foulType", "pbpSubtype", "pbpDescriptor", "region"},
    "passes": {"passType", "region", "startRegion", "endRegion", "outcome"},
    "touches": {"startRegion", "endRegion", "outcome"},
    "picks": {
        "defenseType",
        "complexDefenseType",
        "rollPop",
        "pickDirection",
        "outcome",
    },
    "isolations": {"outcome", "direction"},
    "handoffs": {"outcome", "direction"},
    "drives": {"category", "endType", "direction", "outcome"},
    "posts": {"postType", "direction", "outcome"},
    "off_ball_screens": {"screenType", "outcome"},
    "closeouts": {"outcome", "closeoutType"},
}

EVENT_ACTOR_FIELDS = {
    "shots": ("shooterId",),
}

# These are recommendation-level labels, not provider semantic claims.
FEATURE_CATALOG = [
    {
        "family": "lineup exposure",
        "grain": "possession/chance-player",
        "source_tables": "possessions; chances; chance_players; player_stints",
        "examples": "offPlayerIds, defPlayerIds, start/end clocks, offense flag",
        "model_role": "production input",
        "priority": 1,
        "notes": "Canonical 2023-25 lineup and minutes validation surface; validate against player_stints.",
    },
    {
        "family": "midrange/three shot context",
        "grain": "shot",
        "source_tables": "shots; shot_contests; touches; chances",
        "examples": "distance, region, shotClock, dribblesBefore, catchAndShoot, closestDefDist, contest kinematics",
        "model_role": "production candidate",
        "priority": 1,
        "notes": "Use only MID and 3P per project decision; train outcome out-of-season and group folds by game.",
    },
    {
        "family": "provider shot-quality benchmark",
        "grain": "shot",
        "source_tables": "shots; touches",
        "examples": "qSQ, qSP, startQSP, endQSP",
        "model_role": "benchmark / semantic audit",
        "priority": 1,
        "notes": "Do not ingest as a prior or live feature until the provider definitions and timing contract are verified.",
    },
    {
        "family": "shotmaking mechanics",
        "grain": "shot",
        "source_tables": "shots; jumps",
        "examples": "releaseTime, arcAngle, peakShotHeight, shooterSpeed, shot jump",
        "model_role": "separate research target",
        "priority": 3,
        "notes": "Not pre-shot quality. Keep separate from context SQ and exclude outcome-path fields.",
    },
    {
        "family": "post-shot outcome diagnostics",
        "grain": "shot",
        "source_tables": "shots; free_throws",
        "examples": "rimDepth, rimLeftRight, complexOutcome, bank, outcome",
        "model_role": "target / diagnostic only",
        "priority": 4,
        "notes": "Direct leakage for expected-shot models; never use as pre-shot predictors.",
    },
    {
        "family": "touch creation and pressure",
        "grain": "touch",
        "source_tables": "touches; dribbles; drives; isolations; posts",
        "examples": "touchTime, dribbles, defender distance, doubleTeam, blowby, start/end regions",
        "model_role": "production candidate",
        "priority": 2,
        "notes": "Aggregate prior-game-only to player-season rates before testing as VPM priors.",
    },
    {
        "family": "passing and advantage creation",
        "grain": "pass",
        "source_tables": "passes; touches",
        "examples": "passer/receiver defenders, ledToShot, assistOpp, reversal, pass distance, deflection",
        "model_role": "production candidate",
        "priority": 2,
        "notes": "Supports passer/receiver and creation-pressure components beyond box assists.",
    },
    {
        "family": "screening and roll gravity",
        "grain": "screen/handoff",
        "source_tables": "picks; off_ball_screens; handoffs; fake_handoffs",
        "examples": "contact, slip, roll/pop, taggerIds, spacerIds, ledToShot",
        "model_role": "production candidate",
        "priority": 2,
        "notes": "Mechanism features; estimate player effects with opportunity denominators and game-grouped validation.",
    },
    {
        "family": "defensive assignments and help",
        "grain": "possession/rotation",
        "source_tables": "poss_matchups; rotations; chance_players",
        "examples": "offPlayerId-defPlayerId pairs, matchup ids, rotations, rearview, contest",
        "model_role": "production candidate",
        "priority": 1,
        "notes": "Keep provider-specific assignments distinct from physical contest geometry.",
    },
    {
        "family": "closeout and contest quality",
        "grain": "shot/closeout",
        "source_tables": "shot_contests; closeouts; missed_closeouts; defender_stance",
        "examples": "distanceToShooter, speedTowardShooter, hand height, rearview, missed closeout",
        "model_role": "production candidate",
        "priority": 1,
        "notes": "Best first defensive mechanism family; model MID/3P suppression separately from opponent misses.",
    },
    {
        "family": "rebounding and boxouts",
        "grain": "rebound opportunity",
        "source_tables": "rebounds; boxout_opportunities; chance_players; jumps",
        "examples": "crashed, rebound opportunity, boxout pairing, jump height, rebound outcome",
        "model_role": "production candidate",
        "priority": 2,
        "notes": "Opportunity-based OREB/DREB and boxout-credit models; no shot-outcome leakage into pre-shot SQ.",
    },
    {
        "family": "physical workload",
        "grain": "player-possession",
        "source_tables": "speed_load; jumps; falls",
        "examples": "distance, speed tiers, acceleration/deceleration load, vertical load, jumps",
        "model_role": "research / availability",
        "priority": 4,
        "notes": "Potential fatigue/health context; avoid same-game target leakage and sensitive interpretation.",
    },
]


def classify_column(table: str, column: str) -> tuple[str, str, str]:
    """Assign a conservative modeling role to a provider column."""
    lower = column.lower()
    if lower in {"qsq", "qsp", "startqsp", "endqsp"}:
        return (
            "provider benchmark",
            "benchmark only",
            "Provider-derived quality/probability output; definition and timing contract not yet verified.",
        )
    if lower in {
        "outcome",
        "complexoutcome",
        "ptsscored",
        "homescore",
        "awayscore",
        "homestartscore",
        "awaystartscore",
        "pointvalue",
        "rimdepth",
        "rimleftright",
        "rebounded",
        "defensive",
        "blocked",
        "assisted",
        "secondaryassist",
        "turnover",
        "ledtoshot",
        "assistopp",
        "shooting",
        "charge",
    } or "outcome" in lower or lower.endswith(
        ("rebounded", "made", "scored", "recovered", "successful", "complete", "result")
    ):
        return (
            "target / realized outcome",
            "target or diagnostic",
            "Observed result or label; exclude from pre-event predictors and fold-build any aggregate derived from it.",
        )
    if lower in {
        "arcangle",
        "peakshotheight",
        "releasetime",
        "finishType".lower(),
        "jumpertype",
        "bank",
    } or "fallid" in lower or "jumpid" in lower:
        return (
            "post-release / pose diagnostic",
            "separate mechanics research",
            "Measured during or after the action; not eligible for context-only pre-shot quality.",
        )
    if (
        lower == "id"
        or lower.endswith("id")
        or lower.endswith("ids")
        or lower
        in {
            "season",
            "period",
            "frame",
            "startframe",
            "endframe",
            "wallclock",
            "startwallclock",
            "endwallclock",
            "pbpid",
            "outcomepbpid",
        }
    ):
        return (
            "identifier / join key",
            "join or grouping only",
            "Retain for relational integrity, player attribution, grouping, and leakage-safe folds.",
        )
    if "clock" in lower or lower in {
        "startscore",
        "endscore",
        "foulstopenalty",
        "aftertimeout",
        "transition",
        "category",
        "zone",
        "region",
        "location",
        "distance",
        "three",
        "dunk",
    }:
        return (
            "context candidate",
            "eligible with as-of timing",
            "Game/action context that can be known before the modeled outcome when joined at the correct frame.",
        )
    if table in {
        "shots",
        "shot_contests",
        "touches",
        "passes",
        "dribbles",
        "picks",
        "isolations",
        "handoffs",
        "drives",
        "posts",
        "off_ball_screens",
        "fake_handoffs",
        "closeouts",
        "chance_players",
        "rotations",
        "poss_matchups",
        "speed_load",
        "boxout_opportunities",
        "defender_stance",
    }:
        return (
            "mechanism candidate",
            "candidate after timing audit",
            "Potential player/action mechanism; define its opportunity denominator and verify it precedes the target.",
        )
    return (
        "review",
        "manual semantic review",
        "No safe automatic role assigned; inspect provider semantics before modeling.",
    )


def is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    if value == "":
        return False
    return True


def value_kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def safe_game_metadata(root: Path, markings_path: Path) -> dict[str, Any]:
    """Read only non-secret filename metadata from a portal response."""
    portal_path = root / "api" / "portal" / "gamedata" / f"{markings_path.parent.name}.json"
    payload = json.loads(portal_path.read_text())
    item = next(
        row
        for row in payload["files"]
        if row.get("file") == "basketball-markings-3d.json"
    )
    filename = item["downloadFilename"]
    game_id = filename[1:11]
    return {
        "provider_game_id": markings_path.parent.name,
        "game_id": game_id,
        "season": 2000 + int(game_id[3:5]),
        "game_type_code": game_id[:3],
        "file_bytes": markings_path.stat().st_size,
    }


def valid_player_list(value: Any) -> list[str]:
    return [str(x) for x in value] if isinstance(value, list) else []


def scan_game(args: tuple[str, str]) -> dict[str, Any]:
    root_raw, path_raw = args
    root = Path(root_raw)
    path = Path(path_raw)
    meta = safe_game_metadata(root, path)
    payload_3d = json.loads(path.read_text())
    path_2d = path.with_name("basketball-markings.json")
    payload_2d = json.loads(path_2d.read_text())
    meta["file_bytes_2d"] = path_2d.stat().st_size
    season = int(meta["season"])

    table_rows: dict[str, int] = {}
    table_rows_2d: dict[str, int] = {}
    table_rows_3d: dict[str, int] = {}
    parity: dict[str, dict[str, Any]] = {}
    columns: dict[tuple[str, str], Counter] = defaultdict(Counter)
    categories: dict[tuple[str, str, str], int] = defaultdict(int)
    actors: list[tuple[str, int, str, str]] = []
    all_players: set[str] = set()

    for table, records in payload_2d.items():
        if isinstance(records, list):
            table_rows_2d[table] = len(records)
            table_rows[table] = len(records)
    for table, records in payload_3d.items():
        if isinstance(records, list):
            table_rows_3d[table] = len(records)
            if table not in payload_2d:
                table_rows[table] = len(records)
    for table in sorted(set(table_rows_2d) & set(table_rows_3d)):
        rows_2d = payload_2d[table]
        rows_3d = payload_3d[table]
        ids_2d = {
            str(row["id"])
            for row in rows_2d
            if isinstance(row, dict) and is_present(row.get("id"))
        }
        ids_3d = {
            str(row["id"])
            for row in rows_3d
            if isinstance(row, dict) and is_present(row.get("id"))
        }
        parity[table] = {
            "rows_2d": len(rows_2d),
            "rows_3d": len(rows_3d),
            "row_count_equal": int(len(rows_2d) == len(rows_3d)),
            "ids_2d": len(ids_2d),
            "ids_3d": len(ids_3d),
            "common_ids": len(ids_2d & ids_3d),
            "id_sets_equal": int(bool(ids_2d or ids_3d) and ids_2d == ids_3d),
        }

    # Profile the richer schema. The parity audit records where these rows must
    # be joined onto (rather than substituted for) the 2D canonical event rows.
    for table, records in payload_3d.items():
        if not isinstance(records, list):
            continue
        if table not in CORE_TABLES and table not in EVENT_ACTOR_FIELDS:
            continue
        for row in records:
            if not isinstance(row, dict):
                continue
            for column, value in row.items():
                counter = columns[(table, column)]
                counter["rows"] += 1
                counter[f"type_{value_kind(value)}"] += 1
                if is_present(value):
                    counter["non_null"] += 1
            for column in CATEGORICAL_FIELDS.get(table, set()):
                value = row.get(column)
                if is_present(value):
                    categories[(table, column, str(value))] += 1

    # Identity evidence and player universe use the more complete 2D base.
    for table, records in payload_2d.items():
        if not isinstance(records, list):
            continue
        for row in records:
            if not isinstance(row, dict):
                continue
            for key, value in row.items():
                if (
                    key.endswith("PlayerId")
                    or key.endswith("playerId")
                    or key.endswith("PlayerIds")
                    or key.endswith("playerIds")
                    or key
                    in {
                        "playerId",
                        "offPlayerId",
                        "defPlayerId",
                        "shooterId",
                        "contesterId",
                        "passerId",
                        "receiverId",
                        "ballhandlerId",
                        "screenerId",
                        "defenderId",
                        "rebounderId",
                        "foulerId",
                        "fouledId",
                        "turnedOverId",
                        "stealerId",
                        "offPlayerIds",
                        "defPlayerIds",
                    }
                ):
                    if isinstance(value, list):
                        all_players.update(str(v) for v in value if is_present(v))
                    elif is_present(value):
                        all_players.add(str(value))
            pbp_id = row.get("pbpId")
            if isinstance(pbp_id, (int, float)) and not isinstance(pbp_id, bool):
                for actor_field in EVENT_ACTOR_FIELDS.get(table, ()):
                    provider_player_id = row.get(actor_field)
                    if is_present(provider_player_id):
                        actors.append(
                            (
                                meta["game_id"],
                                int(pbp_id),
                                str(provider_player_id),
                                f"{table}.{actor_field}",
                            )
                        )

    possessions = payload_2d.get("possessions", [])
    lineup = Counter()
    for row in possessions:
        off = valid_player_list(row.get("offPlayerIds"))
        deff = valid_player_list(row.get("defPlayerIds"))
        lineup["possessions"] += 1
        lineup["off_exact_5"] += int(len(off) == 5)
        lineup["def_exact_5"] += int(len(deff) == 5)
        lineup["both_exact_5"] += int(len(off) == 5 and len(deff) == 5)
        lineup["off_unique"] += int(len(off) == len(set(off)))
        lineup["def_unique"] += int(len(deff) == len(set(deff)))
        lineup["no_overlap"] += int(not (set(off) & set(deff)))
        lineup["fully_valid"] += int(
            len(off) == 5
            and len(deff) == 5
            and len(set(off)) == 5
            and len(set(deff)) == 5
            and not (set(off) & set(deff))
        )

    shot_stats = Counter()
    rich_shots = {
        str(row.get("id")): row
        for row in payload_3d.get("shots", [])
        if isinstance(row, dict) and is_present(row.get("id"))
    }
    for base_row in payload_2d.get("shots", []):
        row = dict(base_row)
        rich_row = rich_shots.get(str(base_row.get("id")))
        if rich_row is not None:
            row.update(rich_row)
        shot_stats["shots"] += 1
        shot_stats["rich3d_match_present"] += int(rich_row is not None)
        is_three = row.get("three") is True
        shot_type = str(row.get("shotType") or "")
        is_mid = (not is_three) and shot_type in {"jumper", "post", "floater"}
        shot_stats["three_attempts"] += int(is_three)
        shot_stats["mid_attempts"] += int(is_mid)
        shot_stats["impact_target_eligible"] += int(is_three or is_mid)
        for field in (
            "pbpId",
            "qSQ",
            "qSP",
            "region",
            "shotClock",
            "closestDefDist",
            "closestDefId",
            "contestLevel",
            "contesterIds",
            "dribblesBefore",
            "catchAndShoot",
            "releaseTime",
            "arcAngle",
            "peakShotHeight",
            "rimDepth",
            "rimLeftRight",
        ):
            shot_stats[f"{field}_present"] += int(is_present(row.get(field)))

    chance_player_stats = Counter()
    chance_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in payload_2d.get("chance_players", []):
        chance_id = row.get("chanceId")
        if is_present(chance_id):
            chance_groups[str(chance_id)].append(row)
    for chance_id, rows in chance_groups.items():
        del chance_id
        player_ids = [str(row.get("playerId")) for row in rows if is_present(row.get("playerId"))]
        offense_rows = [row for row in rows if row.get("offense") is True]
        defense_rows = [row for row in rows if row.get("offense") is False]
        chance_player_stats["chance_groups"] += 1
        chance_player_stats["exact_10_rows"] += int(len(rows) == 10)
        chance_player_stats["exact_10_unique_players"] += int(
            len(rows) == 10 and len(player_ids) == 10 and len(set(player_ids)) == 10
        )
        chance_player_stats["exact_5_offense"] += int(len(offense_rows) == 5)
        chance_player_stats["exact_5_defense"] += int(len(defense_rows) == 5)
        chance_player_stats["fully_valid"] += int(
            len(rows) == 10
            and len(player_ids) == 10
            and len(set(player_ids)) == 10
            and len(offense_rows) == 5
            and len(defense_rows) == 5
        )

    meta.update(
        {
            "table_rows": table_rows,
            "table_rows_2d": table_rows_2d,
            "table_rows_3d": table_rows_3d,
            "parity": parity,
            "columns": {
                f"{table}\t{column}": dict(counter)
                for (table, column), counter in columns.items()
            },
            "categories": {
                f"{table}\t{column}\t{value}": count
                for (table, column, value), count in categories.items()
            },
            "actors": actors,
            "all_players": sorted(all_players),
            "lineup": dict(lineup),
            "shot_stats": dict(shot_stats),
            "chance_player_stats": dict(chance_player_stats),
        }
    )
    return meta


def load_pbp_lookup(base: Path) -> dict[tuple[str, int], tuple[int, str]]:
    lookup: dict[tuple[str, int], tuple[int, str]] = {}
    for path in sorted(base.glob("WNBA2[3-5]*.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    game_id = str(row["game_id"]).strip()
                    event_num = int(float(row["event_num"]))
                    person_id = int(float(row.get("personid") or 0))
                except (KeyError, TypeError, ValueError):
                    continue
                if person_id <= 0:
                    continue
                name = (row.get("playername") or "").strip()
                key = (game_id, event_num)
                prior = lookup.get(key)
                if prior is None or prior[0] == person_id:
                    lookup[key] = (person_id, name)
    return lookup


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Optional deterministic smoke-test limit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = args.output_dir.resolve()
    files = sorted(root.glob("downloads/*/basketball-markings-3d.json"))
    if args.max_games:
        files = files[: args.max_games]
    if not files:
        raise SystemExit(f"No 3D markings files found under {root}")

    jobs = [(str(root), str(path)) for path in files]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        games = list(pool.map(scan_game, jobs, chunksize=1))

    inventory_rows = []
    table_agg: dict[tuple[int, str], Counter] = defaultdict(Counter)
    parity_agg: dict[tuple[int, str], Counter] = defaultdict(Counter)
    column_agg: dict[tuple[int, str, str], Counter] = defaultdict(Counter)
    category_agg: Counter = Counter()
    lineup_agg: dict[int, Counter] = defaultdict(Counter)
    shot_agg: dict[int, Counter] = defaultdict(Counter)
    chance_player_agg: dict[int, Counter] = defaultdict(Counter)
    actor_events: list[tuple[str, int, str, str]] = []
    all_players_by_season: dict[int, set[str]] = defaultdict(set)

    for game in games:
        season = int(game["season"])
        inventory_rows.append(
            {
                "game_id": game["game_id"],
                "provider_game_id": game["provider_game_id"],
                "season": season,
                "game_type_code": game["game_type_code"],
                "file_bytes": game["file_bytes"],
                "file_bytes_2d": game["file_bytes_2d"],
                "possessions": game["table_rows"].get("possessions", 0),
                "possessions_3d": game["table_rows_3d"].get("possessions", 0),
                "chances": game["table_rows"].get("chances", 0),
                "chances_3d": game["table_rows_3d"].get("chances", 0),
                "shots": game["table_rows"].get("shots", 0),
                "shots_3d": game["table_rows_3d"].get("shots", 0),
                "passes": game["table_rows"].get("passes", 0),
                "touches": game["table_rows"].get("touches", 0),
                "lineup_fully_valid": game["lineup"].get("fully_valid", 0),
                "lineup_invalid": game["lineup"].get("possessions", 0)
                - game["lineup"].get("fully_valid", 0),
            }
        )
        for table, count in game["table_rows"].items():
            table_agg[(season, table)]["games_present"] += int(count > 0)
            table_agg[(season, table)]["rows"] += count
        for table, stats in game["parity"].items():
            counter = parity_agg[(season, table)]
            counter["games"] += 1
            counter["games_2d_nonempty"] += int(stats["rows_2d"] > 0)
            counter["games_3d_nonempty"] += int(stats["rows_3d"] > 0)
            counter["row_count_equal_games"] += stats["row_count_equal"]
            counter["id_sets_equal_games"] += stats["id_sets_equal"]
            counter["rows_2d"] += stats["rows_2d"]
            counter["rows_3d"] += stats["rows_3d"]
            counter["ids_2d"] += stats["ids_2d"]
            counter["ids_3d"] += stats["ids_3d"]
            counter["common_ids"] += stats["common_ids"]
        for key, counts in game["columns"].items():
            table, column = key.split("\t", 1)
            column_agg[(season, table, column)].update(counts)
        for key, count in game["categories"].items():
            table, column, value = key.split("\t", 2)
            category_agg[(season, table, column, value)] += count
        lineup_agg[season].update(game["lineup"])
        shot_agg[season].update(game["shot_stats"])
        chance_player_agg[season].update(game["chance_player_stats"])
        actor_events.extend(game["actors"])
        all_players_by_season[season].update(game["all_players"])

    write_csv(
        output / "game_inventory.csv",
        sorted(inventory_rows, key=lambda row: row["game_id"]),
        [
            "game_id",
            "provider_game_id",
            "season",
            "game_type_code",
            "file_bytes",
            "file_bytes_2d",
            "possessions",
            "possessions_3d",
            "chances",
            "chances_3d",
            "shots",
            "shots_3d",
            "passes",
            "touches",
            "lineup_fully_valid",
            "lineup_invalid",
        ],
    )

    table_rows = []
    games_per_season = Counter(row["season"] for row in inventory_rows)
    for (season, table), counts in sorted(table_agg.items()):
        table_rows.append(
            {
                "season": season,
                "table": table,
                "games_present": counts["games_present"],
                "season_games": games_per_season[season],
                "game_coverage": counts["games_present"] / games_per_season[season],
                "rows": counts["rows"],
                "rows_per_game": counts["rows"] / counts["games_present"],
            }
        )
    write_csv(
        output / "table_coverage.csv",
        table_rows,
        [
            "season",
            "table",
            "games_present",
            "season_games",
            "game_coverage",
            "rows",
            "rows_per_game",
        ],
    )

    parity_rows = []
    for (season, table), counts in sorted(parity_agg.items()):
        games_count = counts["games"]
        parity_rows.append(
            {
                "season": season,
                "table": table,
                "games": games_count,
                "games_2d_nonempty": counts["games_2d_nonempty"],
                "games_3d_nonempty": counts["games_3d_nonempty"],
                "row_count_equal_pct": counts["row_count_equal_games"] / games_count
                if games_count
                else 0.0,
                "id_sets_equal_pct": counts["id_sets_equal_games"] / games_count
                if games_count
                else 0.0,
                "rows_2d": counts["rows_2d"],
                "rows_3d": counts["rows_3d"],
                "common_id_pct_of_2d": counts["common_ids"] / counts["ids_2d"]
                if counts["ids_2d"]
                else 0.0,
                "common_id_pct_of_3d": counts["common_ids"] / counts["ids_3d"]
                if counts["ids_3d"]
                else 0.0,
            }
        )
    write_csv(
        output / "two_d_three_d_parity.csv",
        parity_rows,
        [
            "season",
            "table",
            "games",
            "games_2d_nonempty",
            "games_3d_nonempty",
            "row_count_equal_pct",
            "id_sets_equal_pct",
            "rows_2d",
            "rows_3d",
            "common_id_pct_of_2d",
            "common_id_pct_of_3d",
        ],
    )

    column_rows = []
    for (season, table, column), counts in sorted(column_agg.items()):
        type_counts = {
            key.removeprefix("type_"): value
            for key, value in counts.items()
            if key.startswith("type_") and key != "type_null" and value
        }
        dominant_type = (
            max(type_counts, key=type_counts.get) if type_counts else "unknown"
        )
        rows = counts["rows"]
        non_null = counts["non_null"]
        column_rows.append(
            {
                "season": season,
                "table": table,
                "column": column,
                "rows": rows,
                "non_null": non_null,
                "coverage": non_null / rows if rows else 0.0,
                "dominant_type": dominant_type,
                "types_seen": "|".join(sorted(type_counts)),
            }
        )
    write_csv(
        output / "column_coverage.csv",
        column_rows,
        [
            "season",
            "table",
            "column",
            "rows",
            "non_null",
            "coverage",
            "dominant_type",
            "types_seen",
        ],
    )

    column_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in column_rows:
        column_by_pair[(row["table"], row["column"])].append(row)
    modeling_rows = []
    for (table, column), rows in sorted(column_by_pair.items()):
        role, recommended_use, reason = classify_column(table, column)
        coverages = [float(row["coverage"]) for row in rows]
        modeling_rows.append(
            {
                "table": table,
                "column": column,
                "modeling_role": role,
                "recommended_use": recommended_use,
                "min_coverage": min(coverages),
                "max_coverage": max(coverages),
                "coverage_drift": max(coverages) - min(coverages),
                "seasons_present": "|".join(
                    str(row["season"]) for row in sorted(rows, key=lambda x: x["season"])
                ),
                "reason": reason,
            }
        )
    write_csv(
        output / "column_modeling_catalog.csv",
        modeling_rows,
        [
            "table",
            "column",
            "modeling_role",
            "recommended_use",
            "min_coverage",
            "max_coverage",
            "coverage_drift",
            "seasons_present",
            "reason",
        ],
    )

    category_rows = [
        {
            "season": season,
            "table": table,
            "column": column,
            "value": value,
            "count": count,
        }
        for (season, table, column, value), count in sorted(category_agg.items())
    ]
    write_csv(
        output / "categorical_values.csv",
        category_rows,
        ["season", "table", "column", "value", "count"],
    )

    lineup_rows = []
    for season, counts in sorted(lineup_agg.items()):
        total = counts["possessions"]
        lineup_rows.append(
            {
                "season": season,
                "possessions": total,
                "off_exact_5_pct": counts["off_exact_5"] / total if total else 0.0,
                "def_exact_5_pct": counts["def_exact_5"] / total if total else 0.0,
                "both_exact_5_pct": counts["both_exact_5"] / total if total else 0.0,
                "off_unique_pct": counts["off_unique"] / total if total else 0.0,
                "def_unique_pct": counts["def_unique"] / total if total else 0.0,
                "no_overlap_pct": counts["no_overlap"] / total if total else 0.0,
                "fully_valid_pct": counts["fully_valid"] / total if total else 0.0,
                "invalid_possessions": total - counts["fully_valid"],
            }
        )
    write_csv(
        output / "lineup_quality.csv",
        lineup_rows,
        list(lineup_rows[0]),
    )

    chance_player_rows = []
    for season, counts in sorted(chance_player_agg.items()):
        total = counts["chance_groups"]
        chance_player_rows.append(
            {
                "season": season,
                "chance_groups": total,
                "exact_10_rows_pct": counts["exact_10_rows"] / total if total else 0.0,
                "exact_10_unique_players_pct": counts["exact_10_unique_players"] / total
                if total
                else 0.0,
                "exact_5_offense_pct": counts["exact_5_offense"] / total
                if total
                else 0.0,
                "exact_5_defense_pct": counts["exact_5_defense"] / total
                if total
                else 0.0,
                "fully_valid_pct": counts["fully_valid"] / total if total else 0.0,
                "invalid_chances": total - counts["fully_valid"],
            }
        )
    write_csv(
        output / "chance_player_quality.csv",
        chance_player_rows,
        list(chance_player_rows[0]),
    )

    shot_rows = []
    for season, counts in sorted(shot_agg.items()):
        total = counts["shots"]
        row = {"season": season, "shots": total}
        for key, count in sorted(counts.items()):
            if key.endswith("_present"):
                row[key.replace("_present", "_coverage")] = count / total if total else 0.0
            elif key in {"three_attempts", "mid_attempts", "impact_target_eligible"}:
                row[key] = count
                row[f"{key}_pct"] = count / total if total else 0.0
        shot_rows.append(row)
    shot_fields = ["season", "shots"] + sorted(
        {key for row in shot_rows for key in row} - {"season", "shots"}
    )
    write_csv(output / "shot_feature_coverage.csv", shot_rows, shot_fields)

    pbp_lookup = load_pbp_lookup(Path("wnba_test/GabrielNormalizedClean"))
    evidence: dict[str, Counter] = defaultdict(Counter)
    evidence_names: dict[tuple[str, int], Counter] = defaultdict(Counter)
    evidence_roles: dict[str, Counter] = defaultdict(Counter)
    matched_events = 0
    for game_id, pbp_id, provider_player_id, role in actor_events:
        player = pbp_lookup.get((game_id, pbp_id))
        if player is None:
            continue
        nba_id, name = player
        evidence[provider_player_id][nba_id] += 1
        evidence_names[(provider_player_id, nba_id)][name] += 1
        evidence_roles[provider_player_id][role] += 1
        matched_events += 1

    all_players = set().union(*all_players_by_season.values())
    crosswalk_rows = []
    mapped_players = set()
    exact_mapped_players = set()
    conflicting_players = set()
    for provider_player_id, candidates in sorted(evidence.items()):
        ordered = candidates.most_common()
        nba_id, best_count = ordered[0]
        total = sum(candidates.values())
        conflict_count = total - best_count
        name = evidence_names[(provider_player_id, nba_id)].most_common(1)[0][0]
        best_share = best_count / total if total else 0.0
        status = (
            "exact"
            if conflict_count == 0
            else "high_confidence_consensus"
            if best_count >= 3 and best_share >= 0.80
            else "ambiguous"
        )
        if status == "ambiguous":
            conflicting_players.add(provider_player_id)
        else:
            mapped_players.add(provider_player_id)
            if status == "exact":
                exact_mapped_players.add(provider_player_id)
        seasons = sorted(
            season
            for season, players in all_players_by_season.items()
            if provider_player_id in players
        )
        crosswalk_rows.append(
            {
                "provider_player_id": provider_player_id,
                "nba_id": nba_id,
                "player_name": name,
                "evidence_events": best_count,
                "all_candidate_events": total,
                "conflicting_events": conflict_count,
                "conflict": bool(conflict_count),
                "best_evidence_share": best_share,
                "status": status,
                "seasons": "|".join(map(str, seasons)),
                "evidence_roles": "|".join(
                    f"{role}:{count}"
                    for role, count in evidence_roles[provider_player_id].most_common()
                ),
            }
        )
    write_csv(
        output / "player_crosswalk.csv",
        crosswalk_rows,
        [
            "provider_player_id",
            "nba_id",
            "player_name",
            "evidence_events",
            "all_candidate_events",
            "conflicting_events",
            "conflict",
            "best_evidence_share",
            "status",
            "seasons",
            "evidence_roles",
        ],
    )

    write_csv(
        output / "feature_catalog.csv",
        FEATURE_CATALOG,
        [
            "family",
            "grain",
            "source_tables",
            "examples",
            "model_role",
            "priority",
            "notes",
        ],
    )

    identity_by_season = []
    for season, players in sorted(all_players_by_season.items()):
        identity_by_season.append(
            {
                "season": season,
                "provider_players": len(players),
                "mapped_high_confidence": len(players & mapped_players),
                "mapped_exact": len(players & exact_mapped_players),
                "ambiguous": len(players & conflicting_players),
                "unmapped": len(players - mapped_players - conflicting_players),
                "mapped_pct": len(players & mapped_players) / len(players) if players else 0.0,
            }
        )
    write_csv(
        output / "identity_coverage.csv",
        identity_by_season,
        list(identity_by_season[0]),
    )

    manifest = {
        "source": "local WNBA RTSS/Second Spectrum 3D markings export",
        "root_label": "downloads/rtss-data/wnba",
        "games": len(games),
        "seasons": dict(sorted(games_per_season.items())),
        "bytes_profiled": sum(
            row["file_bytes"] + row["file_bytes_2d"] for row in inventory_rows
        ),
        "game_type_codes": dict(
            sorted(Counter(row["game_type_code"] for row in inventory_rows).items())
        ),
        "tables": len({row["table"] for row in table_rows}),
        "columns": len({(row["table"], row["column"]) for row in column_rows}),
        "actor_events": len(actor_events),
        "actor_events_matched_to_pbp": matched_events,
        "provider_players": len(all_players),
        "players_mapped_high_confidence": len(all_players & mapped_players),
        "players_mapped_exact": len(all_players & exact_mapped_players),
        "players_ambiguous": len(all_players & conflicting_players),
        "players_unmapped": len(all_players - mapped_players - conflicting_players),
        "outputs": [
            "game_inventory.csv",
            "table_coverage.csv",
            "two_d_three_d_parity.csv",
            "column_coverage.csv",
            "column_modeling_catalog.csv",
            "categorical_values.csv",
            "lineup_quality.csv",
            "chance_player_quality.csv",
            "shot_feature_coverage.csv",
            "player_crosswalk.csv",
            "identity_coverage.csv",
            "feature_catalog.csv",
        ],
        "security_note": "Portal signed URLs were not copied into any output.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
