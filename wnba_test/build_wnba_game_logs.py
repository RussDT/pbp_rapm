"""Build the per-game WNBA player box/minutes panel for the betting program.

Contract:
- Input: cached CDN liveData boxscores (``common.BOXSCORE_CACHE_DIR``), plus an
  optional network probe for missing 2026 regular-season FINAL games beyond the
  max cached id (the cache can be a few games stale). Cached NON-final
  snapshots below the probe frontier (the cache dir is shared with writers
  that can cache in-progress payloads) are refetched with ``refresh=True`` on
  every build; a failed refetch falls back to the cached payload, and a game
  still non-final after refetch is skipped from the panel but stays eligible
  next run. ``--no-fetch`` disables all network access (probe and refresh)
  for fully offline rebuilds.
- Only FINAL games (``gameStatus == 3``) are emitted.
- ``game_date`` is the US/Eastern calendar date of ``gameTimeUTC`` (matching
  the ESPN-derived keys of the odds/forecast lane), falling back to the
  arena-local ``gameTimeLocal`` date only when ``gameTimeUTC`` is absent or
  unparseable.
- Outputs (atomic overwrite: temp file then rename):
  * ``common.GAMES_TABLE_PATH`` — one row per game.
  * ``common.GAME_LOG_PATH`` — one row per rostered player per game,
    INCLUDING DNPs (minutes 0, played 0).
  * ``Results/gameday/game_log_gaps.json`` — final games whose boxscore could
    not be loaded, or whose team minutes sum deviates from
    ``200 + 25 * OT_periods`` by more than ``--minutes-tolerance``.
- Market-blind: no odds/lines touch this builder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

import wnba_gameday_common as common

GAPS_REPORT_PATH = common.GAMEDAY_DIR / "game_log_gaps.json"

GAMES_COLUMNS = [
    "game_id",
    "season",
    "season_type",
    "game_date",
    "game_time_utc",
    "home_tricode",
    "away_tricode",
    "home_team_id",
    "away_team_id",
    "home_score",
    "away_score",
    "status",
    "period",
    "attendance",
]

LOG_COLUMNS = [
    "game_id",
    "season",
    "season_type",
    "game_date",
    "team_tricode",
    "team_id",
    "opp_tricode",
    "home",
    "person_id",
    "player_name",
    "position",
    "starter",
    "status",
    "played",
    "not_playing_reason",
    "minutes",
    "pts",
    "fgm",
    "fga",
    "fg3m",
    "fg3a",
    "ftm",
    "fta",
    "oreb",
    "dreb",
    "reb",
    "ast",
    "stl",
    "blk",
    "tov",
    "pf",
    "plus_minus",
    "team_score",
    "opp_score",
]

# Player stat field -> output column.
STAT_FIELDS = {
    "points": "pts",
    "fieldGoalsMade": "fgm",
    "fieldGoalsAttempted": "fga",
    "threePointersMade": "fg3m",
    "threePointersAttempted": "fg3a",
    "freeThrowsMade": "ftm",
    "freeThrowsAttempted": "fta",
    "reboundsOffensive": "oreb",
    "reboundsDefensive": "dreb",
    "reboundsTotal": "reb",
    "assists": "ast",
    "steals": "stl",
    "blocks": "blk",
    "turnovers": "tov",
    "foulsPersonal": "pf",
}


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _flag(value: Any) -> int:
    """CDN '1'/'0' string flag -> int 1/0."""
    return 1 if str(value).strip() == "1" else 0


def _clean_reason(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def probe_missing_2026_rs(max_misses: int) -> list[str]:
    """Fetch-and-cache 2026 RS FINAL boxscores beyond the max cached id.

    Walks ``1022600001`` upward starting after the max cached 2026 RS id and
    stops after ``max_misses`` consecutive fetch misses or non-final games.
    Non-final payloads fetched by the probe are evicted from the cache so a
    later offline rebuild never sees a permanently stale in-progress snapshot.
    Returns the game ids newly cached as FINAL.
    """
    cached_2026_rs = [
        gid
        for gid in common.iter_cached_game_ids()
        if gid.startswith(common.RS_PREFIX) and common.game_id_season(gid) == 2026
    ]
    start_seq = (int(max(cached_2026_rs)[5:]) + 1) if cached_2026_rs else 1
    fetched: list[str] = []
    consecutive_misses = 0
    seq = start_seq
    while consecutive_misses < max_misses:
        game_id = f"{common.RS_PREFIX}26{seq:05d}"
        was_cached = (common.BOXSCORE_CACHE_DIR / f"{game_id}.json").exists()
        payload = common.load_cached_boxscore(game_id, fetch_missing=True)
        if payload is None:
            consecutive_misses += 1
        elif common.boxscore_is_final(payload):
            consecutive_misses = 0
            if not was_cached:
                fetched.append(game_id)
                print(f"probe: fetched FINAL boxscore {game_id}")
        else:
            consecutive_misses += 1
            if not was_cached:
                # Evict the non-final snapshot the probe just cached.
                (common.BOXSCORE_CACHE_DIR / f"{game_id}.json").unlink(missing_ok=True)
        seq += 1
    return fetched


def resolve_final_payload(
    game_id: str, allow_fetch: bool
) -> tuple[dict[str, Any] | None, str]:
    """Load one cached boxscore for the panel, refreshing stale non-finals.

    A cached non-final snapshot below the probe frontier would otherwise never
    be revisited, losing the game forever. Returns ``(payload, disposition)``
    with disposition one of ``final``, ``refreshed_final``,
    ``non_final_skipped``, ``unloadable``.
    """
    payload = common.load_cached_boxscore(game_id)
    if payload is None:
        return None, "unloadable"
    if common.boxscore_is_final(payload):
        return payload, "final"
    if allow_fetch:
        payload = common.load_cached_boxscore(game_id, refresh=True)
        if payload is not None and common.boxscore_is_final(payload):
            return payload, "refreshed_final"
    return None, "non_final_skipped"


def extract_game(game_id: str, payload: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One FINAL boxscore payload -> (games row, player log rows)."""
    game = payload["game"]
    season = common.game_id_season(game_id)
    season_type = common.game_id_season_type(game_id)
    game_date = common.eastern_game_date(game.get("gameTimeUTC"))
    if game_date is None:
        game_date = str(game.get("gameTimeLocal") or "")[:10] or None
    home = game["homeTeam"]
    away = game["awayTeam"]
    game_row = {
        "game_id": game_id,
        "season": season,
        "season_type": season_type,
        "game_date": game_date,
        "game_time_utc": game.get("gameTimeUTC"),
        "home_tricode": home.get("teamTricode"),
        "away_tricode": away.get("teamTricode"),
        "home_team_id": _to_int(home.get("teamId")),
        "away_team_id": _to_int(away.get("teamId")),
        "home_score": _to_int(home.get("score")),
        "away_score": _to_int(away.get("score")),
        "status": _to_int(game.get("gameStatus")),
        "period": _to_int(game.get("period")),
        "attendance": game.get("attendance"),
    }
    player_rows: list[dict[str, Any]] = []
    for team, opp, is_home in ((home, away, 1), (away, home, 0)):
        for player in team.get("players") or []:
            stats = player.get("statistics") or {}
            row = {
                "game_id": game_id,
                "season": season,
                "season_type": season_type,
                "game_date": game_date,
                "team_tricode": team.get("teamTricode"),
                "team_id": _to_int(team.get("teamId")),
                "opp_tricode": opp.get("teamTricode"),
                "home": is_home,
                "person_id": _to_int(player.get("personId")),
                "player_name": player.get("name"),
                "position": player.get("position") or None,
                "starter": _flag(player.get("starter")),
                "status": player.get("status"),
                "played": _flag(player.get("played")),
                "not_playing_reason": _clean_reason(player.get("notPlayingReason")),
                "minutes": round(common.parse_iso_minutes(stats.get("minutes")), 3),
                "plus_minus": float(stats.get("plusMinusPoints") or 0.0),
                "team_score": _to_int(team.get("score")),
                "opp_score": _to_int(opp.get("score")),
            }
            for field, column in STAT_FIELDS.items():
                row[column] = _to_int(stats.get(field))
            player_rows.append(row)
    return game_row, player_rows


def minutes_gaps(games: pd.DataFrame, logs: pd.DataFrame, tolerance: float) -> list[dict[str, Any]]:
    """Team-games whose minutes sum deviates from 200 + 25*OT beyond tolerance."""
    period = games.set_index("game_id")["period"]
    team_minutes = logs.groupby(["game_id", "team_tricode"], sort=True)["minutes"].sum()
    gaps: list[dict[str, Any]] = []
    for (game_id, tricode), total in team_minutes.items():
        ot_periods = max(int(period.get(game_id, 4)) - 4, 0)
        expected = 200.0 + 25.0 * ot_periods
        if abs(total - expected) > tolerance:
            gaps.append(
                {
                    "issue": "team_minutes_mismatch",
                    "game_id": game_id,
                    "team_tricode": tricode,
                    "minutes_sum": round(float(total), 3),
                    "expected": expected,
                    "ot_periods": ot_periods,
                }
            )
    return gaps


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="fully offline rebuild: skip the 2026 RS probe for missing FINAL boxscores",
    )
    parser.add_argument(
        "--max-consecutive-misses",
        type=int,
        default=20,
        help="probe stops after this many consecutive fetch misses / non-final games",
    )
    parser.add_argument(
        "--minutes-tolerance",
        type=float,
        default=1.5,
        help="max allowed |team minutes sum - (200 + 25*OT)| before a gap is reported",
    )
    args = parser.parse_args()

    if not args.no_fetch:
        fetched = probe_missing_2026_rs(args.max_consecutive_misses)
        print(f"probe: {len(fetched)} new FINAL 2026 RS boxscore(s) cached")

    game_rows: list[dict[str, Any]] = []
    player_rows: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    n_not_final = 0
    n_refreshed_final = 0
    for game_id in common.iter_cached_game_ids():
        payload, disposition = resolve_final_payload(game_id, allow_fetch=not args.no_fetch)
        if disposition == "unloadable":
            gaps.append({"issue": "boxscore_unloadable", "game_id": game_id})
            continue
        if disposition == "non_final_skipped":
            n_not_final += 1
            print(f"cached non-final boxscore {game_id} skipped (eligible next run)")
            continue
        if disposition == "refreshed_final":
            n_refreshed_final += 1
            print(f"refreshed stale non-final boxscore {game_id} -> FINAL")
        try:
            game_row, rows = extract_game(game_id, payload)
        except (KeyError, TypeError, ValueError) as exc:
            gaps.append({"issue": "boxscore_parse_error", "game_id": game_id, "detail": repr(exc)})
            continue
        game_rows.append(game_row)
        player_rows.extend(rows)

    games = pd.DataFrame(game_rows, columns=GAMES_COLUMNS).sort_values("game_id")
    logs = pd.DataFrame(player_rows, columns=LOG_COLUMNS).sort_values(
        ["game_id", "home", "team_tricode", "starter", "person_id"],
        ascending=[True, False, True, False, True],
    )
    games["attendance"] = pd.array(
        [None if a is None else _to_int(a) for a in games["attendance"]], dtype="Int64"
    )
    gaps.extend(minutes_gaps(games, logs, args.minutes_tolerance))

    atomic_write_csv(games, common.GAMES_TABLE_PATH)
    atomic_write_csv(logs, common.GAME_LOG_PATH)
    atomic_write_json(gaps, GAPS_REPORT_PATH)

    by_season = games.groupby(["season", "season_type"]).size()
    print(
        f"games: {len(games)} FINAL ({n_not_final} cached non-final skipped, "
        f"{n_refreshed_final} refreshed to FINAL)"
    )
    print(by_season.to_string())
    print(f"player rows: {len(logs)} (DNP rows: {int((logs['played'] == 0).sum())})")
    print(f"gaps: {len(gaps)}")
    for gap in gaps:
        print(f"  {gap}")
    print(f"wrote {common.GAMES_TABLE_PATH}")
    print(f"wrote {common.GAME_LOG_PATH}")
    print(f"wrote {GAPS_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
