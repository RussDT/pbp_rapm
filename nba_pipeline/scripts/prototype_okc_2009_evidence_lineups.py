#!/usr/bin/env python3
"""Prototype evidence-driven lineup repair for 2009 OKC only.

This is intentionally a validation harness, not production conversion logic.
It starts from the current guarded OKC 2009 raw file, uses the independent
archive PlayByPlay substitution/event stream to reconstruct only the OKC side,
then processes a test RAPM parquet for Weaver-focused audits.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PIPELINE_ROOT.parent
ARCHIVE_ROOT = Path("/Users/russellthomas/Docs/2026_NBA_PIPELINE/archive (1)")

INPUT_RAW = (
    PIPELINE_ROOT
    / "validation"
    / "okc_2009_guarded_lineup_repair"
    / "NBA09_OKC_guarded_repair.parquet"
)
OUT_DIR = PIPELINE_ROOT / "validation" / "okc_2009_evidence_lineup_repair"
OUTPUT_RAW = OUT_DIR / "NBA09_OKC_evidence_repair.parquet"
OUTPUT_RAPM = OUT_DIR / "RAPM09_OKC_evidence_repair_actual_ft.parquet"
DIAGNOSTICS_CSV = OUT_DIR / "okc_2009_evidence_repair_diagnostics.csv"
SUMMARY_JSON = OUT_DIR / "okc_2009_evidence_repair_summary.json"
WEAVER_ID = 201602
OKC_TEAM_ID = 1610612760
OKC = "OKC"

HOME_COLS = [f"home_player{i}" for i in range(1, 6)]
AWAY_COLS = [f"away_player{i}" for i in range(1, 6)]


@dataclass(frozen=True)
class SubMove:
    action_number: int
    period: int
    clock_seconds: int
    clock: str
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


def infer_okc_side(raw_game: pd.DataFrame) -> str | None:
    okc = raw_game[raw_game["player1_team_abbreviation"].astype(str).eq(OKC)]
    home_votes = okc["home_description"].notna().sum()
    away_votes = okc["visitor_description"].notna().sum()
    if home_votes == away_votes == 0:
        return None
    return "home" if home_votes > away_votes else "away"


def load_player_names(game_ids: set[int]) -> dict[int, dict[str, int]]:
    usecols = ["firstName", "lastName", "personId", "gameId", "playerteamId"]
    stats = pd.read_csv(ARCHIVE_ROOT / "PlayerStatistics.csv", usecols=usecols, low_memory=False)
    stats["game_id_int"] = pd.to_numeric(stats["gameId"], errors="coerce").fillna(-1).astype(int)
    stats = stats[(stats["game_id_int"].isin(game_ids)) & (stats["playerteamId"] == OKC_TEAM_ID)].copy()

    lookup: dict[int, dict[str, int]] = defaultdict(dict)
    for _, row in stats.iterrows():
        player_id = normalize_player_id(row["personId"])
        if player_id is None:
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
            lookup[int(row["game_id_int"])].setdefault(label, player_id)
    return lookup


def add_archive_event_names(lookup: dict[int, dict[str, int]], archive: pd.DataFrame) -> None:
    okc = archive[archive["teamId"].eq(OKC_TEAM_ID)]
    for _, row in okc.iterrows():
        player_id = normalize_player_id(row.get("personId"))
        if player_id is None:
            continue
        labels = {
            normalize_label(row.get("playerName")),
            normalize_label(row.get("playerFullName")),
        }
        for label in list(labels):
            parts = label.split()
            if parts:
                labels.add(parts[-1])
            if len(parts) >= 2:
                labels.add(f"{parts[0][0]} {parts[-1]}")
        for label in labels:
            if label:
                lookup[normalize_game_id(row["gameId"])].setdefault(label, player_id)


def parse_compact_sub(row: pd.Series, lookup: dict[int, dict[str, int]]) -> SubMove | None:
    if str(row.get("actionType", "")).lower() != "substitution":
        return None
    if normalize_player_id(row.get("teamId")) != OKC_TEAM_ID:
        return None
    desc = "" if pd.isna(row.get("description")) else str(row.get("description"))
    match = re.search(r"sub:\s*(.*?)\s+for\s+(.*)$", desc, flags=re.IGNORECASE)
    if not match:
        return None
    game_id = normalize_game_id(row["gameId"])
    incoming_label = normalize_label(match.group(1))
    incoming = lookup.get(game_id, {}).get(incoming_label)
    if incoming is None:
        # Try last token for historical rows that only use last names.
        parts = incoming_label.split()
        incoming = lookup.get(game_id, {}).get(parts[-1] if parts else incoming_label)
    outgoing = normalize_player_id(row.get("personId"))
    if incoming is None or outgoing is None:
        return None
    return SubMove(
        action_number=int(row["actionNumber"]),
        period=int(row["period"]),
        clock_seconds=parse_clock_seconds(row.get("clock")),
        clock=str(row.get("clock")),
        incoming=int(incoming),
        outgoing=int(outgoing),
        description=desc,
    )


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
        "reboundTotal",
        "stealPersonId",
        "blockPersonId",
        "foulDrawnPersonId",
        "jumpBallRecoveredPersonId",
        "jumpBallWonPersonId",
        "jumpBallLostPersonId",
    ]
    filters = [("gameId", "in", [str(g) for g in sorted(game_ids)])]
    archive = pd.read_parquet(ARCHIVE_ROOT / "PlayByPlay.parquet", columns=cols, filters=filters)
    archive["game_id_int"] = archive["gameId"].map(normalize_game_id)
    archive["_clock_seconds"] = archive["clock"].map(parse_clock_seconds)
    return archive.sort_values(
        ["game_id_int", "period", "_clock_seconds", "actionNumber"],
        ascending=[True, True, False, True],
        kind="stable",
    ).reset_index(drop=True)


def hard_okc_actors(row: pd.Series) -> set[int]:
    actors: set[int] = set()
    team_id = normalize_player_id(row.get("teamId"))
    if team_id == OKC_TEAM_ID:
        for col in ["personId", "assistPersonId", "jumpBallRecoveredPersonId", "jumpBallWonPersonId", "jumpBallLostPersonId"]:
            player_id = normalize_player_id(row.get(col))
            if player_id is not None:
                actors.add(player_id)
    elif team_id is not None:
        for col in ["stealPersonId", "blockPersonId", "foulDrawnPersonId"]:
            player_id = normalize_player_id(row.get(col))
            if player_id is not None:
                actors.add(player_id)
    return actors


def build_next_seen(events: pd.DataFrame) -> list[dict[int, int]]:
    next_seen: dict[int, int] = {}
    out = [None] * len(events)
    for pos in range(len(events) - 1, -1, -1):
        out[pos] = dict(next_seen)
        for player_id in hard_okc_actors(events.iloc[pos]):
            next_seen[player_id] = pos
    return out


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
        if (
            len(lineup) == 5
            and must_include.issubset(lineup_set)
            and not (must_exclude & lineup_set)
        ):
            return lineup
    return []


def seed_lineup(raw_game: pd.DataFrame, cols: list[str], moves: list[SubMove], diagnostics: list[dict]) -> list[int]:
    first = raw_game.sort_values("event_num", kind="stable").iloc[0]
    candidate = lineup_from_row(first, cols)
    seed = unique_ordered(candidate)
    if moves:
        first_clock = (moves[0].period, moves[0].clock_seconds)
        first_moves = [m for m in moves if (m.period, m.clock_seconds) == first_clock]
        first_ins = {m.incoming for m in first_moves}
        first_outs = {m.outgoing for m in first_moves}
        seed = [p for p in seed if p not in first_ins]
        seed = unique_ordered(list(first_outs) + seed)
        later = first_complete_after(
            raw_game,
            cols,
            min(m.action_number for m in first_moves),
            must_include=first_ins,
            must_exclude=first_outs,
        )
        if not later:
            later = first_complete_after(raw_game, cols, min(m.action_number for m in first_moves))
        reversed_later = unique_ordered([p for p in later if p not in first_ins] + list(first_outs))
        for player_id in reversed_later:
            if len(seed) >= 5:
                break
            if player_id not in seed:
                seed.append(player_id)
        if set(seed[:5]) != set(candidate):
            diagnostics.append(
                {
                    "game_id": int(first["game_id"]),
                    "issue": "starter_seed_repaired_from_first_sub",
                    "action_number": min(m.action_number for m in first_moves),
                    "details": json.dumps(
                        {
                            "candidate": candidate,
                            "seed": seed[:5],
                            "first_ins": sorted(first_ins),
                            "first_outs": sorted(first_outs),
                            "later_complete": later,
                        }
                    ),
                }
            )
    return seed[:5]


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


def reconstruct_game(
    raw_game: pd.DataFrame,
    archive_game: pd.DataFrame,
    okc_side: str,
    lookup: dict[int, dict[str, int]],
    diagnostics: list[dict],
) -> pd.DataFrame:
    cols = HOME_COLS if okc_side == "home" else AWAY_COLS
    events = archive_game.sort_values(
        ["period", "_clock_seconds", "actionNumber"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    moves = [m for _, row in events.iterrows() if (m := parse_compact_sub(row, lookup)) is not None]
    moves_by_action: dict[int, list[SubMove]] = defaultdict(list)
    for move in moves:
        moves_by_action[move.action_number].append(move)

    sub_positions = [i for i, row in events.iterrows() if int(row["actionNumber"]) in moves_by_action]
    next_sub_pos_for_event: list[int | None] = []
    for pos in range(len(events)):
        later = [sub_pos for sub_pos in sub_positions if sub_pos > pos]
        next_sub_pos_for_event.append(later[0] if later else None)

    next_seen_by_pos = build_next_seen(events)
    current = seed_lineup(raw_game, cols, moves, diagnostics)
    last_seen: dict[int, int] = {}
    state_by_action: dict[int, list[int]] = {}
    conflicts = Counter()

    for event_pos, row in events.iterrows():
        action = int(row["actionNumber"])
        game_id = normalize_game_id(row["gameId"])
        for actor in sorted(hard_okc_actors(row)):
            if actor not in current:
                before = list(current)
                current, evicted = evict_for_actor(
                    current=current,
                    actor=actor,
                    event_pos=event_pos,
                    next_sub_pos=next_sub_pos_for_event[event_pos],
                    next_seen_by_pos=next_seen_by_pos,
                    last_seen=last_seen,
                )
                conflicts["hard_actor_repairs"] += 1
                diagnostics.append(
                    {
                        "game_id": game_id,
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

        if action in moves_by_action:
            before = list(current)
            group = moves_by_action[action]
            for move in group:
                if move.incoming in current:
                    current = [p for p in current if p != move.incoming]
                    conflicts["sub_in_already_active_removed"] += 1
                if move.outgoing not in current:
                    current, evicted = evict_for_actor(
                        current=current,
                        actor=move.outgoing,
                        event_pos=event_pos,
                        next_sub_pos=event_pos + 1,
                        next_seen_by_pos=next_seen_by_pos,
                        last_seen=last_seen,
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
            if set(before) == set(current):
                continue
            diagnostics.append(
                {
                    "game_id": game_id,
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

    repaired = raw_game.copy()
    action_state = list(current)
    sorted_actions = sorted(state_by_action)
    action_idx = 0
    current_state = state_by_action[sorted_actions[0]] if sorted_actions else seed_lineup(raw_game, cols, moves, diagnostics)
    for idx, row in repaired.sort_values("event_num", kind="stable").iterrows():
        action = int(pd.to_numeric(row.get("source_actionNumber"), errors="coerce"))
        while action_idx < len(sorted_actions) and sorted_actions[action_idx] <= action:
            current_state = state_by_action[sorted_actions[action_idx]]
            action_idx += 1
        stamped_state, filled = fill_from_raw_snapshot(current_state, row, cols)
        if filled:
            conflicts["row_state_filled_from_raw_snapshot"] += 1
        write_lineup(repaired, idx, cols, stamped_state)

    if conflicts:
        diagnostics.append(
            {
                "game_id": int(raw_game.iloc[0]["game_id"]),
                "issue": "game_conflict_summary",
                "action_number": 0,
                "details": json.dumps(dict(conflicts)),
            }
        )
    return repaired


def summarize_weaver(rapm: pd.DataFrame) -> dict[str, float]:
    o_cols = [f"O{i}" for i in range(1, 6)]
    d_cols = [f"D{i}" for i in range(1, 6)]
    o_mask = rapm[o_cols].apply(pd.to_numeric, errors="coerce").eq(WEAVER_ID).any(axis=1)
    d_mask = rapm[d_cols].apply(pd.to_numeric, errors="coerce").eq(WEAVER_ID).any(axis=1)
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


def overlap_rows(df: pd.DataFrame) -> int:
    count = 0
    for _, row in df.iterrows():
        home = set(lineup_from_row(row, HOME_COLS))
        away = set(lineup_from_row(row, AWAY_COLS))
        if home & away:
            count += 1
    return count


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(INPUT_RAW)
    raw["game_id_int"] = raw["game_id"].map(normalize_game_id)
    game_ids = set(raw["game_id_int"].unique())
    archive = load_archive(game_ids)
    lookup = load_player_names(game_ids)
    add_archive_event_names(lookup, archive)

    diagnostics: list[dict] = []
    parts: list[pd.DataFrame] = []
    side_counts = Counter()
    for game_id, raw_game in raw.groupby("game_id_int", sort=True):
        okc_side = infer_okc_side(raw_game)
        if okc_side is None:
            diagnostics.append(
                {"game_id": int(game_id), "issue": "missing_okc_side", "action_number": 0, "details": ""}
            )
            parts.append(raw_game)
            continue
        side_counts[okc_side] += 1
        archive_game = archive[archive["game_id_int"].eq(int(game_id))]
        if archive_game.empty:
            diagnostics.append(
                {"game_id": int(game_id), "issue": "missing_archive_game", "action_number": 0, "details": ""}
            )
            parts.append(raw_game)
            continue
        parts.append(reconstruct_game(raw_game, archive_game, okc_side, lookup, diagnostics))

    repaired = pd.concat(parts, ignore_index=True).drop(columns=["game_id_int"], errors="ignore")
    repaired = repaired.sort_values(["game_id", "event_num"], kind="stable").reset_index(drop=True)
    repaired.to_parquet(OUTPUT_RAW, index=False)

    sys.path.insert(0, str(PIPELINE_ROOT / "scripts"))
    from process_rapm_blocks.process_rapm import process_rapm_py

    rapm, _ = process_rapm_py(
        str(OUTPUT_RAW),
        2009,
        "2008-09",
        o_luck=0.0,
        d_luck=0.0,
        missing_ft_fallback="actual",
    )
    if rapm is None:
        raise RuntimeError("RAPM processing returned no output")
    rapm.to_parquet(OUTPUT_RAPM, index=False)

    baseline_rapm = pd.read_parquet(
        PIPELINE_ROOT
        / "validation"
        / "okc_2009_guarded_lineup_repair"
        / "RAPM09_OKC_guarded_repair_actual_ft.parquet"
    )
    pd.DataFrame(diagnostics).to_csv(DIAGNOSTICS_CSV, index=False)
    summary = {
        "input_raw": str(INPUT_RAW),
        "output_raw": str(OUTPUT_RAW),
        "output_rapm": str(OUTPUT_RAPM),
        "games": int(repaired["game_id"].nunique()),
        "okc_side_counts": dict(side_counts),
        "raw_overlap_rows_before": overlap_rows(raw),
        "raw_overlap_rows_after": overlap_rows(repaired),
        "diagnostic_issue_counts": dict(Counter(item["issue"] for item in diagnostics)),
        "weaver_baseline": summarize_weaver(baseline_rapm),
        "weaver_evidence_repair": summarize_weaver(rapm),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
