#!/usr/bin/env python3
"""Audit and repair modern NBA raw-PBP lineup corruption.

The NBA GameRotation endpoint occasionally returns incomplete or overlapping
stints.  The raw fetcher historically converted those stints directly into
``home_player1..5`` / ``away_player1..5``, so a bad rotation response can
produce duplicate or missing players for hundreds of events in one game.

This script repairs only games that fail lineup integrity checks.  It solves
each team/period as a constrained substitution replay using:

* substitution descriptions (the hard state transitions),
* event actors (strong positive on-court evidence),
* Gabriel Michael's archived GameRotation stints (positive evidence only),
* surviving unique players in the raw lineup slots (weak positive evidence).

Rotation *absence* is deliberately not treated as evidence that a player was
off the floor; incomplete rotation intervals are the corruption being fixed.
By default repaired season files and reports are written under ``validation``.
Production raw files are overwritten only with ``--promote`` and are backed up
first.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import shutil
import tempfile
import unicodedata
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = REPO_ROOT / "nba_pipeline" / "raw_data"
DEFAULT_VALIDATION_DIR = REPO_ROOT / "nba_pipeline" / "validation" / "modern_lineup_repairs"
ROTATION_URL = (
    "https://raw.githubusercontent.com/gabriel1200/shot_data/"
    "master/rotations/{season}.csv"
)
ROTATION_TEAM_URL = (
    "https://raw.githubusercontent.com/gabriel1200/shot_data/"
    "master/rotations/{season}/{team_id}.csv"
)

HOME_COLS = [f"home_player{i}" for i in range(1, 6)]
AWAY_COLS = [f"away_player{i}" for i in range(1, 6)]
LINEUP_COLS = AWAY_COLS + HOME_COLS
SUB_RE = re.compile(r"\bSUB:\s*(.+?)\s+FOR\s+(.+?)(?:\s*\||$)", re.IGNORECASE)


@dataclass(frozen=True)
class Substitution:
    period: int
    elapsed_tenths: float
    event_num: int
    outgoing: int
    incoming: int

    @property
    def boundary(self) -> tuple[float, int]:
        return (self.elapsed_tenths, self.event_num)


@dataclass
class PeriodSolution:
    states: list[frozenset[int]]
    boundaries: list[tuple[float, int]]
    penalty: float
    second_penalty: float
    candidates: int
    unresolved_subs: int

    @property
    def margin(self) -> float:
        return self.second_penalty - self.penalty


def normalize_game_ids(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(10)
    )


def _numeric_player(value: object) -> int | None:
    try:
        if pd.isna(value):
            return None
        player = int(float(value))
    except (TypeError, ValueError):
        return None
    return player if player > 0 else None


def elapsed_tenths(df: pd.DataFrame) -> pd.Series:
    """Return elapsed game time in tenths of seconds.

    ``minute_game`` preserves the original V3 clock precision and is preferred.
    The period-clock fallback supports older raw files.
    """
    if "minute_game" in df.columns:
        minute_game = pd.to_numeric(df["minute_game"], errors="coerce")
    else:
        minute_game = pd.Series(np.nan, index=df.index, dtype=float)

    period = pd.to_numeric(df.get("period"), errors="coerce").fillna(1).astype(int)
    minutes = pd.to_numeric(df.get("minute_remaining_quarter"), errors="coerce").fillna(0)
    seconds = pd.to_numeric(df.get("seconds_remaining_quarter"), errors="coerce").fillna(0)
    remaining = (minutes * 60.0 + seconds) * 10.0
    regulation_start = (period - 1).clip(lower=0) * 7200.0
    overtime_start = 28800.0 + (period - 5).clip(lower=0) * 3000.0
    fallback = np.where(
        period <= 4,
        regulation_start + 7200.0 - remaining,
        overtime_start + 3000.0 - remaining,
    )
    return minute_game.mul(600.0).where(minute_game.notna(), fallback).astype(float)


def sort_pbp_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    """Sort corrected V3 actions by clock first, then event number."""
    out = df.copy()
    out["_source_order"] = np.arange(len(out), dtype=np.int64)
    out["event_elapsed_tenths"] = elapsed_tenths(out)
    out["_event_num_sort"] = pd.to_numeric(out.get("event_num"), errors="coerce").fillna(0)
    cols = ["game_id", "event_elapsed_tenths", "_event_num_sort", "_source_order"]
    out = out.sort_values(cols, kind="stable").reset_index(drop=True)
    return out.drop(columns=["_event_num_sort", "_source_order"])


def lineup_issue_mask(df: pd.DataFrame) -> pd.Series:
    issues = pd.Series(False, index=df.index)
    numeric = df[LINEUP_COLS].apply(pd.to_numeric, errors="coerce")
    for cols in (AWAY_COLS, HOME_COLS):
        side = numeric[cols].to_numpy(dtype=float)
        issues |= np.isnan(side).any(axis=1) | (side <= 0).any(axis=1)
        ordered = np.sort(side, axis=1)
        issues |= np.equal(ordered[:, 1:], ordered[:, :-1]).any(axis=1)
    away = numeric[AWAY_COLS].to_numpy(dtype=float)
    home = numeric[HOME_COLS].to_numpy(dtype=float)
    issues |= np.equal(away[:, :, None], home[:, None, :]).any(axis=(1, 2))
    return issues


def _description(row: pd.Series) -> str:
    for col in ("home_description", "visitor_description", "neutral_description"):
        value = row.get(col)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def actor_requires_on_court(row: object) -> bool:
    event_type = int(getattr(row, "event_type", 0) or 0)
    if event_type in {8, 12, 13}:
        return False
    description = " | ".join(
        str(getattr(row, col))
        for col in ("home_description", "visitor_description", "neutral_description")
        if isinstance(getattr(row, col, None), str)
    )
    # Technicals can be charged to bench players or coaches.
    return re.search(r"\bT\.?\s*FOUL\b|\bTECHNICAL\b", description, re.IGNORECASE) is None


def _normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return re.sub(r"\s+", " ", text)


class NameResolver:
    def __init__(self) -> None:
        self._names: dict[str, set[int]] = defaultdict(set)
        self._exact: dict[str, set[int]] = defaultdict(set)
        self._labels: dict[int, set[str]] = defaultdict(set)
        self._canonical: dict[int, set[str]] = defaultdict(set)

    def add(self, player_id: object, name: object) -> None:
        player = _numeric_player(player_id)
        normalized = _normalize_name(name)
        if player is None or not normalized:
            return
        self._exact[normalized].add(player)
        self._labels[player].add(normalized)
        tokens = normalized.split()
        suffixes = {"jr", "sr", "ii", "iii", "iv", "v"}
        suffix = tokens[-1] if tokens and tokens[-1] in suffixes else None
        base = tokens[:-1] if suffix else tokens
        base_name = " ".join(base)
        self._canonical[player].add(base_name)
        keys = {normalized, base_name}
        if base:
            keys.add(base[-1])
        if len(base) >= 2:
            keys.add(f"{base[0][0]} {base[-1]}")
            keys.add(f"{base[0]} {base[-1]}")
            if suffix:
                keys.add(f"{base[-1]} {suffix}")
                keys.add(f"{base[0][0]} {base[-1]} {suffix}")
        for key in keys:
            self._names[key].add(player)

    def merge(self, other: "NameResolver") -> None:
        for player, names in other._labels.items():
            for name in names:
                self.add(player, name)

    def candidates(self, name: str, allowed: set[int] | None = None) -> set[int]:
        normalized = _normalize_name(name)
        tokens = normalized.split()
        universe = set(self._canonical) if allowed is None else allowed
        exact = self._exact.get(normalized, set()) & universe
        if exact:
            return exact
        keys = [normalized]
        if len(tokens) >= 2:
            keys.append(f"{tokens[0][0]} {tokens[-1]}")
        ambiguous: set[int] = set()
        for key in keys:
            matches = self._names.get(key, set()) & universe
            if len(matches) == 1:
                return matches
            ambiguous.update(matches)
        # NBA substitution text abbreviates colliding first names as "Jay."
        # and "Jal.".  Match that token as a first-name prefix within one last
        # name instead of collapsing both players to "J. Williams".
        suffixes = {"jr", "sr", "ii", "iii", "iv", "v"}
        base_tokens = tokens[:-1] if tokens and tokens[-1] in suffixes else tokens
        if len(base_tokens) >= 2:
            first, last = base_tokens[0], base_tokens[-1]
            prefix_matches = {
                player
                for player in universe
                for canonical in self._canonical.get(player, set())
                if canonical.split()[-1] == last
                and (
                    canonical.split()[0].startswith(first)
                    or first.startswith(canonical.split()[0])
                )
            }
            if prefix_matches:
                return prefix_matches
        return ambiguous

    def resolve(self, name: str, allowed: set[int] | None = None) -> int | None:
        matches = self.candidates(name, allowed)
        return next(iter(matches)) if len(matches) == 1 else None


def infer_team_ids(game: pd.DataFrame, rotations: pd.DataFrame) -> tuple[int, int]:
    home_votes: Counter[int] = Counter()
    away_votes: Counter[int] = Counter()
    team_col = pd.to_numeric(game.get("player1_team_id"), errors="coerce")
    for idx, team_value in team_col.items():
        team = _numeric_player(team_value)
        if team is None:
            continue
        if isinstance(game.at[idx, "home_description"], str):
            home_votes[team] += 1
        if isinstance(game.at[idx, "visitor_description"], str):
            away_votes[team] += 1

    teams = set(home_votes) | set(away_votes)
    if not rotations.empty:
        teams.update(int(x) for x in rotations["TEAM_ID"].dropna().unique())
    if len(teams) != 2:
        raise ValueError(f"expected two team ids, found {sorted(teams)}")

    best: tuple[int, int] | None = None
    best_score = -1
    for home, away in itertools.permutations(sorted(teams), 2):
        score = home_votes[home] + away_votes[away]
        if score > best_score:
            best, best_score = (home, away), score
    assert best is not None
    return best


def build_player_sides(
    game: pd.DataFrame,
    rotations: pd.DataFrame,
    home_team: int,
    away_team: int,
) -> dict[int, str]:
    votes: dict[int, Counter[str]] = defaultdict(Counter)
    team_to_side = {home_team: "home", away_team: "away"}
    for n in (1, 2, 3):
        pid_col, team_col = f"player{n}_id", f"player{n}_team_id"
        if pid_col not in game or team_col not in game:
            continue
        for pid, tid in zip(game[pid_col], game[team_col]):
            player, team = _numeric_player(pid), _numeric_player(tid)
            if player is not None and team in team_to_side:
                votes[player][team_to_side[team]] += 20
    if not rotations.empty:
        for row in rotations.itertuples(index=False):
            player, team = _numeric_player(row.PERSON_ID), _numeric_player(row.TEAM_ID)
            if player is not None and team in team_to_side:
                votes[player][team_to_side[team]] += 50
    for side, cols in (("home", HOME_COLS), ("away", AWAY_COLS)):
        for value in game[cols].to_numpy().ravel():
            player = _numeric_player(value)
            if player is not None:
                votes[player][side] += 1
    return {
        player: counts.most_common(1)[0][0]
        for player, counts in votes.items()
        if counts
    }


def build_name_resolver(game: pd.DataFrame, rotations: pd.DataFrame) -> NameResolver:
    resolver = NameResolver()
    for n in (1, 2, 3):
        if f"player{n}_id" in game and f"player{n}_name" in game:
            for player, name in zip(game[f"player{n}_id"], game[f"player{n}_name"]):
                resolver.add(player, name)
    if not rotations.empty:
        for row in rotations.itertuples(index=False):
            resolver.add(row.PERSON_ID, f"{row.PLAYER_FIRST} {row.PLAYER_LAST}")
    return resolver


def build_season_team_players(frame: pd.DataFrame) -> dict[int, set[int]]:
    team_players: dict[int, set[int]] = defaultdict(set)
    for n in (1, 2, 3):
        player_col, team_col = f"player{n}_id", f"player{n}_team_id"
        if player_col not in frame or team_col not in frame:
            continue
        pairs = frame[[player_col, team_col]].drop_duplicates()
        for player_value, team_value in pairs.itertuples(index=False, name=None):
            player, team = _numeric_player(player_value), _numeric_player(team_value)
            if player is not None and team is not None:
                team_players[team].add(player)
    return team_players


def parse_substitutions(
    game: pd.DataFrame,
    side: str,
    roster: set[int],
    player_sides: dict[int, str],
    resolver: NameResolver,
    preferred_candidates: set[int] | None = None,
    secondary_candidates: set[int] | None = None,
    fallback_candidates: set[int] | None = None,
    rotations: pd.DataFrame | None = None,
) -> tuple[list[Substitution], int]:
    substitutions: list[Substitution] = []
    unresolved = 0
    for row in game.itertuples(index=False):
        if int(getattr(row, "event_type", 0) or 0) != 8:
            continue
        outgoing = _numeric_player(getattr(row, "player1_id", None))
        if outgoing is None or player_sides.get(outgoing) != side:
            continue
        match = SUB_RE.search(
            " | ".join(
                str(getattr(row, col))
                for col in ("home_description", "visitor_description", "neutral_description")
                if isinstance(getattr(row, col), str)
            )
        )
        incoming = None
        if match:
            candidate_sets = [
                roster if preferred_candidates is None else roster & preferred_candidates,
                roster & secondary_candidates if secondary_candidates is not None else set(),
                roster & fallback_candidates if fallback_candidates is not None else set(),
            ]
            for allowed in candidate_sets:
                if not allowed:
                    continue
                matches = resolver.candidates(match.group(1), allowed)
                if len(matches) == 1:
                    incoming = next(iter(matches))
                    break
                if len(matches) > 1 and rotations is not None and not rotations.empty:
                    starting = rotations[
                        rotations["PERSON_ID"].isin(matches)
                        & rotations["IN_TIME_REAL"].sub(float(row.event_elapsed_tenths)).abs().le(1.0)
                    ]["PERSON_ID"].dropna().astype(int).unique()
                    if len(starting) == 1:
                        incoming = int(starting[0])
                        break
        if incoming is None:
            unresolved += 1
            print(
                f"    unresolved substitution {side} P{int(row.period)} "
                f"event {int(row.event_num)}: {_description(pd.Series(row._asdict()))}"
            )
            continue
        substitutions.append(
            Substitution(
                period=int(row.period),
                elapsed_tenths=float(row.event_elapsed_tenths),
                event_num=int(row.event_num),
                outgoing=outgoing,
                incoming=incoming,
            )
        )
    return substitutions, unresolved


def collapse_substitutions(subs: Sequence[Substitution]) -> list[tuple[tuple[float, int], set[int], set[int]]]:
    grouped: dict[tuple[float, int], tuple[set[int], set[int]]] = {}
    # All substitutions by one team at one clock are one atomic boundary.  The
    # first event number distinguishes actions that happened earlier at that clock.
    by_clock: dict[float, list[Substitution]] = defaultdict(list)
    for sub in subs:
        by_clock[sub.elapsed_tenths].append(sub)
    for clock, clock_subs in by_clock.items():
        boundary = (clock, min(sub.event_num for sub in clock_subs))
        outgoing = {sub.outgoing for sub in clock_subs}
        incoming = {sub.incoming for sub in clock_subs}
        # Corrected V3 feeds can repeat/cancel substitutions at one clock.  A
        # player present on both sides of the atomic cluster is a net no-op.
        unchanged = outgoing & incoming
        grouped[boundary] = (outgoing - unchanged, incoming - unchanged)
    return [(boundary, *grouped[boundary]) for boundary in sorted(grouped)]


def replay(start: Iterable[int], clusters: Sequence[tuple[tuple[float, int], set[int], set[int]]]) -> tuple[list[frozenset[int]], int]:
    state = set(start)
    states = [frozenset(state)]
    contradictions = 0
    for _, outgoing, incoming in clusters:
        contradictions += len(outgoing - state)
        contradictions += len(incoming & state)
        state = (state - outgoing) | incoming
        if len(state) != 5:
            contradictions += abs(len(state) - 5) + 1
        states.append(frozenset(state))
    return states, contradictions


def state_index(boundaries: Sequence[tuple[float, int]], elapsed: float, event_num: int) -> int:
    key = (elapsed, event_num)
    return sum(boundary <= key for boundary in boundaries)


def _rotation_active(rotations: pd.DataFrame, elapsed: float) -> set[int]:
    if rotations.empty:
        return set()
    mask = (rotations["IN_TIME_REAL"] <= elapsed) & (elapsed < rotations["OUT_TIME_REAL"])
    return {int(x) for x in rotations.loc[mask, "PERSON_ID"].dropna() if int(x) > 0}


def build_evidence(
    period_rows: pd.DataFrame,
    side: str,
    roster: set[int],
    player_sides: dict[int, str],
    rotations: pd.DataFrame,
    clusters: Sequence[tuple[tuple[float, int], set[int], set[int]]],
) -> tuple[list[Counter[int]], list[Counter[int]], list[set[int]]]:
    boundaries = [cluster[0] for cluster in clusters]
    outgoing_at = {boundary: outgoing for boundary, outgoing, _ in clusters}
    raw = [Counter() for _ in range(len(boundaries) + 1)]
    actors = [Counter() for _ in range(len(boundaries) + 1)]
    rotation = [set() for _ in range(len(boundaries) + 1)]
    cols = HOME_COLS if side == "home" else AWAY_COLS

    for row in period_rows.itertuples(index=False):
        event_num = int(getattr(row, "event_num", 0) or 0)
        elapsed = float(row.event_elapsed_tenths)
        index = state_index(boundaries, elapsed, event_num)
        for col in cols:
            player = _numeric_player(getattr(row, col))
            if player in roster and player_sides.get(player) == side:
                raw[index][player] += 1
        actor = _numeric_player(getattr(row, "player1_id", None))
        if actor_requires_on_court(row) and actor in roster and player_sides.get(actor) == side:
            actor_index = index
            if index > 0:
                boundary = boundaries[index - 1]
                if boundary[0] == elapsed and actor in outgoing_at[boundary]:
                    actor_index -= 1
            actors[actor_index][actor] += 1
        rotation[index].update(_rotation_active(rotations, elapsed) & roster)
    return raw, actors, rotation


def solve_period(
    period_rows: pd.DataFrame,
    side: str,
    roster: set[int],
    substitutions: Sequence[Substitution],
    player_sides: dict[int, str],
    rotations: pd.DataFrame,
    unresolved_subs: int,
) -> PeriodSolution:
    clusters = collapse_substitutions(substitutions)
    boundaries = [cluster[0] for cluster in clusters]
    raw, actors, rotation = build_evidence(
        period_rows, side, roster, player_sides, rotations, clusters
    )
    if len(roster) < 5:
        raise ValueError(f"{side} roster has only {len(roster)} resolved players")

    scored: list[tuple[float, tuple[int, ...], list[frozenset[int]]]] = []
    for candidate in itertools.combinations(sorted(roster), 5):
        states, contradictions = replay(candidate, clusters)
        penalty = contradictions * 10000.0
        for index, state in enumerate(states):
            penalty += sum(count * 30.0 for player, count in actors[index].items() if player not in state)
            penalty += sum(15.0 for player in rotation[index] if player not in state)
            penalty += sum(min(count, 10) for player, count in raw[index].items() if player not in state)
        scored.append((penalty, candidate, states))
    scored.sort(key=lambda item: (item[0], item[1]))
    best = scored[0]
    second = scored[1][0] if len(scored) > 1 else float("inf")
    return PeriodSolution(
        states=best[2],
        boundaries=boundaries,
        penalty=best[0],
        second_penalty=second,
        candidates=len(scored),
        unresolved_subs=unresolved_subs,
    )


def repair_game(
    game: pd.DataFrame,
    rotations: pd.DataFrame,
    season_resolver: NameResolver | None = None,
    season_team_players: dict[int, set[int]] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    game = sort_pbp_chronologically(game)
    home_team, away_team = infer_team_ids(game, rotations)
    player_sides = build_player_sides(game, rotations, home_team, away_team)
    resolver = build_name_resolver(game, rotations)
    if season_resolver is not None:
        resolver.merge(season_resolver)

    rosters: dict[str, set[int]] = {"home": set(), "away": set()}
    trusted_rosters: dict[str, set[int]] = {"home": set(), "away": set()}
    rotation_rosters: dict[str, set[int]] = {"home": set(), "away": set()}
    for player, side in player_sides.items():
        rosters[side].add(player)
    team_to_side = {home_team: "home", away_team: "away"}
    for n in (1, 2, 3):
        player_col, team_col = f"player{n}_id", f"player{n}_team_id"
        if player_col not in game or team_col not in game:
            continue
        for player_value, team_value in game[[player_col, team_col]].drop_duplicates().itertuples(
            index=False, name=None
        ):
            player, team = _numeric_player(player_value), _numeric_player(team_value)
            side = team_to_side.get(team)
            if player is not None and side is not None:
                trusted_rosters[side].add(player)
    if not rotations.empty:
        for row in rotations.itertuples(index=False):
            side = team_to_side.get(int(row.TEAM_ID))
            player = _numeric_player(row.PERSON_ID)
            if side and player:
                rosters[side].add(player)
                trusted_rosters[side].add(player)
                rotation_rosters[side].add(player)

    # A zero-stat bench player can appear only as an incoming substitution and
    # be absent from a corrupt rotation response.  Resolve that name against
    # the season-wide player/name map and attach it to the outgoing player's side.
    for row in game[game["event_type"].eq(8)].itertuples(index=False):
        outgoing = _numeric_player(getattr(row, "player1_id", None))
        side = player_sides.get(outgoing) if outgoing is not None else None
        if side is None:
            continue
        match = SUB_RE.search(
            " | ".join(
                str(getattr(row, col))
                for col in ("home_description", "visitor_description", "neutral_description")
                if isinstance(getattr(row, col), str)
            )
        )
        team_id = home_team if side == "home" else away_team
        allowed = season_team_players.get(team_id) if season_team_players else None
        incoming = resolver.resolve(match.group(1), rotation_rosters[side]) if match else None
        if incoming is None and match:
            incoming = resolver.resolve(match.group(1), trusted_rosters[side])
        if incoming is None and match:
            incoming = resolver.resolve(match.group(1), allowed)
        if incoming is not None:
            rosters[side].add(incoming)
            trusted_rosters[side].add(incoming)
            player_sides[incoming] = side

    all_subs: dict[str, list[Substitution]] = {}
    unresolved: dict[str, int] = {}
    for side in ("home", "away"):
        team_id = home_team if side == "home" else away_team
        fallback_candidates = season_team_players.get(team_id) if season_team_players else None
        all_subs[side], unresolved[side] = parse_substitutions(
            game,
            side,
            rosters[side],
            player_sides,
            resolver,
            rotation_rosters[side],
            trusted_rosters[side],
            fallback_candidates,
            rotations[rotations["TEAM_ID"].eq(team_id)] if not rotations.empty else rotations,
        )

    period_reports: list[dict[str, object]] = []
    for period, period_rows in game.groupby("period", sort=True):
        for side, cols, team_id in (
            ("home", HOME_COLS, home_team),
            ("away", AWAY_COLS, away_team),
        ):
            period_subs = [sub for sub in all_subs[side] if sub.period == int(period)]
            team_rotations = rotations[
                (rotations["TEAM_ID"] == team_id)
            ] if not rotations.empty else rotations
            solution = solve_period(
                period_rows,
                side,
                rosters[side],
                period_subs,
                player_sides,
                team_rotations,
                unresolved[side],
            )
            outgoing_at = {
                boundary: outgoing
                for boundary, outgoing, _ in collapse_substitutions(period_subs)
            }
            for idx, row in period_rows.iterrows():
                elapsed = float(row.event_elapsed_tenths)
                state_pos = state_index(solution.boundaries, elapsed, int(row.event_num))
                actor = _numeric_player(row.get("player1_id"))
                if state_pos > 0:
                    boundary = solution.boundaries[state_pos - 1]
                    if boundary[0] == elapsed and actor in outgoing_at.get(boundary, set()):
                        state_pos -= 1
                state = solution.states[state_pos]
                game.loc[idx, cols] = sorted(state)
            period_reports.append(
                {
                    "period": int(period),
                    "side": side,
                    "penalty": solution.penalty,
                    "second_penalty": solution.second_penalty,
                    "margin": solution.margin,
                    "candidates": solution.candidates,
                    "substitution_boundaries": len(solution.boundaries),
                    "unresolved_substitutions": solution.unresolved_subs,
                }
            )

    after_issues = int(lineup_issue_mask(game).sum())
    actor_misses = count_actor_misses(game, player_sides)
    report = {
        "game_id": str(game["game_id"].iloc[0]),
        "rows": len(game),
        "home_team_id": home_team,
        "away_team_id": away_team,
        "home_roster_players": len(rosters["home"]),
        "away_roster_players": len(rosters["away"]),
        "unresolved_substitutions": unresolved["home"] + unresolved["away"],
        "post_repair_lineup_issue_rows": after_issues,
        "post_repair_actor_misses": actor_misses,
        "periods": period_reports,
    }
    return game.drop(columns=["event_elapsed_tenths"]), report


def count_actor_misses(game: pd.DataFrame, player_sides: dict[int, str] | None = None) -> int:
    misses = 0
    for row in game.itertuples(index=False):
        if not actor_requires_on_court(row):
            continue
        player = _numeric_player(getattr(row, "player1_id", None))
        if player is None:
            continue
        side = player_sides.get(player) if player_sides else None
        if side is None:
            team = _numeric_player(getattr(row, "player1_team_id", None))
            home_ids = {_numeric_player(getattr(row, col)) for col in HOME_COLS}
            away_ids = {_numeric_player(getattr(row, col)) for col in AWAY_COLS}
            if player in home_ids:
                side = "home"
            elif player in away_ids:
                side = "away"
            elif team is None:
                continue
        lineup = HOME_COLS if side == "home" else AWAY_COLS if side == "away" else []
        if lineup and player not in {_numeric_player(getattr(row, col)) for col in lineup}:
            misses += 1
    return misses


def load_rotation_season(season: str | int, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{season}.csv"
    if not path.exists():
        url = ROTATION_URL.format(season=season)
        print(f"  downloading rotation evidence: {url}")
        try:
            with urllib.request.urlopen(url, timeout=120) as response, tempfile.NamedTemporaryFile(
                dir=cache_dir, suffix=".tmp", delete=False
            ) as temp:
                shutil.copyfileobj(response, temp)
                temp_path = Path(temp.name)
            temp_path.replace(path)
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            print(f"  rotation archive unavailable ({exc}); using PBP evidence only")
            return pd.DataFrame(
                columns=[
                    "GAME_ID", "TEAM_ID", "PERSON_ID", "PLAYER_FIRST", "PLAYER_LAST",
                    "IN_TIME_REAL", "OUT_TIME_REAL",
                ]
            )
    rotations = pd.read_csv(path, dtype={"GAME_ID": str})
    rotations["GAME_ID"] = normalize_game_ids(rotations["GAME_ID"])
    for col in ("TEAM_ID", "PERSON_ID", "IN_TIME_REAL", "OUT_TIME_REAL"):
        rotations[col] = pd.to_numeric(rotations[col], errors="coerce")
    return rotations


def load_rotation_teams(
    season: str | int,
    team_ids: Iterable[int],
    cache_dir: Path,
) -> pd.DataFrame:
    """Load per-team archive files when the aggregate season CSV is absent."""
    frames: list[pd.DataFrame] = []
    team_cache = cache_dir / str(season)
    team_cache.mkdir(parents=True, exist_ok=True)
    for team_id in sorted(set(int(team) for team in team_ids)):
        path = team_cache / f"{team_id}.csv"
        if not path.exists():
            url = ROTATION_TEAM_URL.format(season=season, team_id=team_id)
            print(f"    downloading team rotation evidence: {url}")
            try:
                with urllib.request.urlopen(url, timeout=120) as response, tempfile.NamedTemporaryFile(
                    dir=team_cache, suffix=".tmp", delete=False
                ) as temp:
                    shutil.copyfileobj(response, temp)
                    temp_path = Path(temp.name)
                temp_path.replace(path)
            except (urllib.error.HTTPError, urllib.error.URLError) as exc:
                print(f"    team rotation archive unavailable ({exc})")
                continue
        frame = pd.read_csv(path, dtype={"GAME_ID": str})
        frame["GAME_ID"] = normalize_game_ids(frame["GAME_ID"])
        for col in ("TEAM_ID", "PERSON_ID", "IN_TIME_REAL", "OUT_TIME_REAL"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=[
                "GAME_ID", "TEAM_ID", "PERSON_ID", "PLAYER_FIRST", "PLAYER_LAST",
                "IN_TIME_REAL", "OUT_TIME_REAL",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def season_file(raw_dir: Path, season: int, season_type: str) -> Path:
    suffix = "_PS" if season_type == "ps" else ""
    return raw_dir / f"NBA{season % 100:02d}{suffix}.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--season-types", choices=("rs", "ps", "all"), default="all")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_VALIDATION_DIR / "raw")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_VALIDATION_DIR / "reports")
    parser.add_argument("--rotation-cache", type=Path, default=DEFAULT_VALIDATION_DIR / "rotation_cache")
    parser.add_argument("--games", help="optional comma-separated game ids")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--promote", action="store_true", help="back up and replace production raw files")
    parser.add_argument(
        "--allow-actor-misses",
        action="store_true",
        help="allow promotion when a repaired game still has non-admin actor mismatches",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.start_year < 2014:
        raise SystemExit("This modern repair path starts at 2014; use the historical repair tools before 2014.")
    if args.end_year < args.start_year:
        raise SystemExit("--end-year must be >= --start-year")
    requested_games = {
        value.strip().zfill(10) for value in (args.games or "").split(",") if value.strip()
    }
    season_types = ("rs", "ps") if args.season_types == "all" else (args.season_types,)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    run_reports: list[dict[str, object]] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for season in range(args.start_year, args.end_year + 1):
        for season_type in season_types:
            source = season_file(args.raw_dir, season, season_type)
            if not source.exists():
                print(f"skip missing {source}")
                continue
            print(f"audit {source}")
            frame = pd.read_parquet(source)
            frame["game_id"] = normalize_game_ids(frame["game_id"])
            issue_mask = lineup_issue_mask(frame)
            all_bad_games = set(frame.loc[issue_mask, "game_id"].unique())
            bad_games = set(all_bad_games)
            if requested_games:
                bad_games &= requested_games
            file_report: dict[str, object] = {
                "season": season,
                "season_type": season_type,
                "source": str(source),
                "rows": len(frame),
                "lineup_issue_rows_before": int(issue_mask.sum()),
                "all_bad_games": sorted(all_bad_games),
                "bad_games": sorted(bad_games),
                "games": [],
            }
            if not bad_games or args.audit_only:
                run_reports.append(file_report)
                continue

            rotation_season = season if season_type == "rs" else f"{season}ps"
            rotations = load_rotation_season(rotation_season, args.rotation_cache)
            season_resolver = build_name_resolver(frame, rotations)
            season_team_players = build_season_team_players(frame)
            repaired_parts: list[pd.DataFrame] = []
            unresolved_games: list[str] = []
            for game_id, game in frame.groupby("game_id", sort=False):
                if game_id not in bad_games:
                    repaired_parts.append(game)
                    continue
                print(f"  repair {game_id}")
                game_rotations = rotations[rotations["GAME_ID"] == game_id].copy()
                try:
                    if game_rotations.empty:
                        home_team, away_team = infer_team_ids(game, game_rotations)
                        team_rotations = load_rotation_teams(
                            rotation_season,
                            (home_team, away_team),
                            args.rotation_cache,
                        )
                        season_resolver.merge(build_name_resolver(frame.iloc[0:0], team_rotations))
                        for team_id, team_rows in team_rotations.groupby("TEAM_ID"):
                            season_team_players[int(team_id)].update(
                                int(player)
                                for player in team_rows["PERSON_ID"].dropna().unique()
                                if int(player) > 0
                            )
                        game_rotations = team_rotations[
                            team_rotations["GAME_ID"] == game_id
                        ].copy()
                    repaired, game_report = repair_game(
                        game,
                        game_rotations,
                        season_resolver,
                        season_team_players,
                    )
                    game_report["lineup_issue_rows_before"] = int(lineup_issue_mask(game).sum())
                    unsafe = bool(game_report["post_repair_lineup_issue_rows"])
                    unsafe |= bool(game_report["unresolved_substitutions"])
                    unsafe |= bool(game_report["post_repair_actor_misses"] and not args.allow_actor_misses)
                    game_report["status"] = "unresolved" if unsafe else "repaired"
                    if unsafe:
                        unresolved_games.append(game_id)
                        repaired_parts.append(game)
                    else:
                        repaired_parts.append(repaired)
                except Exception as exc:
                    game_report = {
                        "game_id": game_id,
                        "lineup_issue_rows_before": int(lineup_issue_mask(game).sum()),
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    unresolved_games.append(game_id)
                    repaired_parts.append(game)
                file_report["games"].append(game_report)

            output = sort_pbp_chronologically(pd.concat(repaired_parts, ignore_index=True)).drop(
                columns=["event_elapsed_tenths"]
            )
            output_path = args.output_dir / source.name
            output.to_parquet(output_path, index=False)
            file_report["output"] = str(output_path)
            file_report["unresolved_games"] = unresolved_games
            file_report["lineup_issue_rows_after"] = int(lineup_issue_mask(output).sum())

            if args.promote:
                remaining_issue_rows = int(file_report["lineup_issue_rows_after"])
                if unresolved_games or remaining_issue_rows:
                    print(
                        f"  REFUSE promote {source.name}: unresolved games {unresolved_games}; "
                        f"remaining issue rows {remaining_issue_rows}"
                    )
                else:
                    backup_dir = args.report_dir / f"production_backup_{timestamp}"
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    backup = backup_dir / source.name
                    shutil.copy2(source, backup)
                    shutil.copy2(output_path, source)
                    file_report["production_backup"] = str(backup)
                    file_report["promoted"] = True
            run_reports.append(file_report)

    report_path = args.report_dir / f"modern_lineup_repair_{timestamp}.json"
    report_path.write_text(json.dumps(run_reports, indent=2, default=str) + "\n")
    print(f"report: {report_path}")
    unresolved = sum(len(item.get("unresolved_games", [])) for item in run_reports)
    print(f"files audited: {len(run_reports)}; unresolved games: {unresolved}")
    return 1 if unresolved else 0


if __name__ == "__main__":
    raise SystemExit(main())
