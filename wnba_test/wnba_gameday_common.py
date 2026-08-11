"""Shared contract module for the WNBA game-day forecast lane.

Every script in the Phase 0/1 betting-program lane (game logs, minutes
projection, pre-tip forecasts, odds capture, scoring, sigma calibration,
Kalman box panel) imports paths, parsers, and log primitives from here so the
lane stays consistent. Keep this module dependency-light: stdlib + pandas.

Contract facts this module encodes (do not re-derive elsewhere):
- Cached CDN boxscores live in ``nba_pipeline/external/gabriel_wnba_boxscores``
  as ``{game_id}.json`` with the liveData schema (``game.homeTeam.players[*]``).
- Player minutes are ISO-ish durations like ``PT28M29.00S``.
- Game ids: ``102{YY}{seq:05d}`` regular season, ``104{YY}{seq:05d}`` playoffs;
  ``gameStatus == 3`` means final.
- The current ratings board is positive-good on BOTH sides and
  ``net_rapm = off + def`` in points per 100 possessions.
- The board has no team column; team assignment joins through the PBP Stats
  box cache (EntityId) and ``player_index_map.csv``.
- ESPN is the odds/schedule source; its team abbreviations differ from CDN
  tricodes (see ``ESPN_TO_CDN_TRICODE``). Portland is CDN ``PDX``, Phoenix is
  ``PHO`` through 2024 and ``PHX`` from 2025.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo

import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

WNBA_TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = WNBA_TEST_DIR.parent
BOXSCORE_CACHE_DIR = REPO_ROOT / "nba_pipeline" / "external" / "gabriel_wnba_boxscores"
GAMEDAY_DIR = WNBA_TEST_DIR / "Results" / "gameday"
FORECAST_LOG_DIR = GAMEDAY_DIR / "forecast_log"
ODDS_LOG_DIR = GAMEDAY_DIR / "odds_log"
CURRENT_BOARD_PATH = WNBA_TEST_DIR / "Results" / "current" / "wnba_current_metrics.csv"
HISTORY_DIR = WNBA_TEST_DIR / "Results" / "metric_history"
PLAYER_INDEX_MAP_PATH = WNBA_TEST_DIR / "player_index_map.csv"
# PBP Stats backend cache (one row per player-season with TeamId/TeamAbbreviation).
WNBA_DATA_DIR = REPO_ROOT.parent / "wnba" / "data"

GAME_LOG_PATH = GAMEDAY_DIR / "player_game_logs.csv"
GAMES_TABLE_PATH = GAMEDAY_DIR / "games.csv"

# --------------------------------------------------------------------------
# Game ids / seasons
# --------------------------------------------------------------------------

RS_PREFIX = "102"
PS_PREFIX = "104"


def game_id_season(game_id: str) -> int:
    return 2000 + int(str(game_id)[3:5])


def game_id_season_type(game_id: str) -> str:
    prefix = str(game_id)[:3]
    if prefix == RS_PREFIX:
        return "rs"
    if prefix == PS_PREFIX:
        return "ps"
    raise ValueError(f"unrecognized WNBA game id prefix: {game_id}")


# --------------------------------------------------------------------------
# Boxscore cache
# --------------------------------------------------------------------------

BOXSCORE_URL = "https://cdn.wnba.com/static/json/liveData/boxscore/boxscore_{game_id}.json"

_MINUTES_RE = re.compile(r"^PT(?:(\d+)M)?(?:([\d.]+)S)?$")


def parse_iso_minutes(value: Any) -> float:
    """``'PT28M29.00S'`` -> 28.483 minutes. None/''/unparseable -> 0.0."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0
    match = _MINUTES_RE.match(str(value).strip())
    if not match:
        return 0.0
    minutes = float(match.group(1) or 0)
    seconds = float(match.group(2) or 0)
    return minutes + seconds / 60.0


def load_cached_boxscore(
    game_id: str,
    *,
    fetch_missing: bool = False,
    refresh: bool = False,
) -> dict[str, Any] | None:
    """Read one cached CDN boxscore; optionally fetch-and-cache when absent.

    Network access happens only when ``fetch_missing`` (cache miss) or
    ``refresh`` (forced) is set, via ``fetch_wnba_cdn.try_fetch_json`` so the
    Akamai-block detection and curl_cffi impersonation are reused.
    """
    path = BOXSCORE_CACHE_DIR / f"{game_id}.json"
    cached: dict[str, Any] | None = None
    if path.exists():
        try:
            cached = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            cached = None
    if cached is not None and not refresh:
        return cached
    if cached is None and not (fetch_missing or refresh):
        return None
    import sys

    if str(WNBA_TEST_DIR) not in sys.path:
        sys.path.insert(0, str(WNBA_TEST_DIR))
    from fetch_wnba_cdn import try_fetch_json

    payload = try_fetch_json(BOXSCORE_URL.format(game_id=game_id))
    if payload is None:
        # A failed refresh must not make a validly cached game look missing.
        return cached
    BOXSCORE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return payload


def iter_cached_game_ids() -> Iterator[str]:
    for path in sorted(BOXSCORE_CACHE_DIR.glob("*.json")):
        stem = path.stem
        if stem.isdigit() and len(stem) == 10:
            yield stem


def boxscore_is_final(payload: dict[str, Any]) -> bool:
    return int((payload.get("game") or {}).get("gameStatus") or 0) == 3


# --------------------------------------------------------------------------
# Game dates
# --------------------------------------------------------------------------

# The odds/forecast lane keys games by US/Eastern calendar date (ESPN's
# convention), so panel game_date must use the same clock — a 9pm+ Pacific
# tip crosses midnight Eastern and an arena-local date would break the
# (game_date, away_tricode, home_tricode) join.
EASTERN_TZ = ZoneInfo("America/New_York")


def eastern_game_date(game_time_utc: Any) -> str | None:
    """UTC tip timestamp (``'2026-08-11T02:00:00Z'``) -> US/Eastern date string.

    Returns None when the timestamp is absent or unparseable so callers can
    fall back to another date source. Naive timestamps are assumed UTC.
    """
    if game_time_utc is None:
        return None
    text = str(game_time_utc).strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(EASTERN_TZ).date().isoformat()


# --------------------------------------------------------------------------
# Teams
# --------------------------------------------------------------------------

# ESPN abbreviation -> CDN tricode. CDN tricodes observed in cached boxscores:
# 2022-2024: ATL CHI CON DAL IND LAS LVA MIN NYL PHO SEA WAS
# 2025+:     PHO->PHX, +GSV; 2026: +PDX (Portland Fire), +TOR (Toronto Tempo)
ESPN_TO_CDN_TRICODE = {
    "ATL": "ATL",
    "CHI": "CHI",
    "CONN": "CON",
    "CON": "CON",
    "DAL": "DAL",
    "GS": "GSV",
    "GSV": "GSV",
    "IND": "IND",
    "LA": "LAS",
    "LAS": "LAS",
    "LV": "LVA",
    "LVA": "LVA",
    "MIN": "MIN",
    "NY": "NYL",
    "NYL": "NYL",
    "PHX": "PHX",
    "PHO": "PHX",
    "POR": "PDX",
    "PDX": "PDX",
    "SEA": "SEA",
    "TOR": "TOR",
    "WAS": "WAS",
    "WSH": "WAS",
}

# Fallback: normalized team nickname -> current CDN tricode.
TEAM_NAME_TO_TRICODE = {
    "dream": "ATL",
    "sky": "CHI",
    "sun": "CON",
    "wings": "DAL",
    "valkyries": "GSV",
    "fever": "IND",
    "sparks": "LAS",
    "aces": "LVA",
    "lynx": "MIN",
    "liberty": "NYL",
    "fire": "PDX",
    "mercury": "PHX",
    "storm": "SEA",
    "tempo": "TOR",
    "mystics": "WAS",
}


def espn_to_cdn_tricode(abbreviation: str | None, team_name: str | None = None) -> str:
    """Map an ESPN team abbreviation (with nickname fallback) to a CDN tricode.

    Raises rather than guessing: an unknown team must fail closed so a bad
    join can never silently attach odds or forecasts to the wrong game.
    """
    if abbreviation:
        key = str(abbreviation).strip().upper()
        if key in ESPN_TO_CDN_TRICODE:
            return ESPN_TO_CDN_TRICODE[key]
    if team_name:
        nickname = str(team_name).strip().lower().split()[-1]
        if nickname in TEAM_NAME_TO_TRICODE:
            return TEAM_NAME_TO_TRICODE[nickname]
    raise KeyError(
        f"unknown ESPN team (abbreviation={abbreviation!r}, name={team_name!r}); "
        "extend ESPN_TO_CDN_TRICODE / TEAM_NAME_TO_TRICODE in wnba_gameday_common"
    )


# ESPN keyless public endpoints (same pattern as unified-rapm build_clv):
ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard?dates={yyyymmdd}"
)
ESPN_SUMMARY_URL = (
    "https://site.web.api.espn.com/apis/site/v2/sports/basketball/wnba/summary?event={event_id}"
)


# --------------------------------------------------------------------------
# Ratings board
# --------------------------------------------------------------------------


def load_current_board(path: Path | None = None) -> pd.DataFrame:
    """Load the live board. Both sides positive-good, net = off + def, per-100."""
    board = pd.read_csv(path or CURRENT_BOARD_PATH)
    board["player_id"] = board["player_id"].astype(int)
    return board


def load_team_assignments(season: int) -> pd.DataFrame:
    """player_id -> (team_id, team_tricode) for one season.

    Primary: PBP Stats backend cache (``{season}_pbp.csv``: EntityId, TeamId,
    TeamAbbreviation). Fallback/updates: ``player_index_map.csv`` rows for the
    season (columns year_season, player_id, team, team_id). Later sources do
    not overwrite the primary.

    Known limitation: the ``keep="first"`` dedupe keeps one team row per
    player, so a mid-season traded player can be listed under a stale team.
    Roster consumers should prefer a roster derived from each player's most
    recent game-log appearance and use this table only as a final fallback
    (e.g. a brand-new franchise with no game-log history yet).
    """
    frames: list[pd.DataFrame] = []
    cache_path = WNBA_DATA_DIR / f"{season}_pbp.csv"
    if cache_path.exists():
        cache = pd.read_csv(cache_path)
        if {"EntityId", "TeamAbbreviation"}.issubset(cache.columns):
            frame = cache[["EntityId", "TeamId", "TeamAbbreviation"]].copy()
            frame.columns = ["player_id", "team_id", "team_tricode"]
            frames.append(frame)
    if PLAYER_INDEX_MAP_PATH.exists():
        index_map = pd.read_csv(PLAYER_INDEX_MAP_PATH)
        rows = index_map[index_map["year_season"] == season]
        if not rows.empty:
            frame = rows[["player_id", "team_id", "team"]].copy()
            frame.columns = ["player_id", "team_id", "team_tricode"]
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"no team-assignment source for season {season}: "
            f"missing {cache_path} and no player_index_map rows"
        )
    merged = pd.concat(frames, ignore_index=True)
    merged["player_id"] = pd.to_numeric(merged["player_id"], errors="coerce")
    merged = merged.dropna(subset=["player_id"])
    merged["player_id"] = merged["player_id"].astype(int)
    return merged.drop_duplicates(subset=["player_id"], keep="first").reset_index(drop=True)


# --------------------------------------------------------------------------
# Append-only hash-chained JSONL logs
# --------------------------------------------------------------------------

GENESIS_HASH = "0" * 64


def _json_sanitize(value: Any) -> Any:
    """Return a JSON-native copy of ``value`` for stable canonical hashing.

    Numpy scalars are converted to their Python equivalents instead of being
    silently stringified by ``default=str``, and non-finite floats are
    rejected because ``json.dumps`` would emit non-standard ``NaN``/``Infinity``
    tokens that break ``json.loads`` on verification.
    """
    if isinstance(value, dict):
        return {str(key): _json_sanitize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(item) for item in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("log records must not contain NaN/Infinity values")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"log records must be JSON-native; got {type(value).__name__}")


def canonical_json(record: dict[str, Any]) -> str:
    return json.dumps(
        _json_sanitize(record), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _last_hash(path: Path) -> str:
    if not path.exists():
        return GENESIS_HASH
    last_line = None
    with path.open() as handle:
        for line in handle:
            if line.strip():
                last_line = line
    if last_line is None:
        return GENESIS_HASH
    return json.loads(last_line)["record_hash"]


def append_log_record(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Append one record to a hash-chained JSONL log and return the stored row.

    The stored row is ``record`` plus ``prev_hash`` and ``record_hash`` where
    ``record_hash = sha256(prev_hash + canonical_json(record))``. Any later
    edit to an earlier row breaks every subsequent hash.
    """
    if "prev_hash" in record or "record_hash" in record:
        raise ValueError("record must not pre-set prev_hash/record_hash")
    path.parent.mkdir(parents=True, exist_ok=True)
    prev_hash = _last_hash(path)
    record_hash = hashlib.sha256((prev_hash + canonical_json(record)).encode()).hexdigest()
    stored = dict(record)
    stored["prev_hash"] = prev_hash
    stored["record_hash"] = record_hash
    with path.open("a") as handle:
        handle.write(canonical_json(stored) + "\n")
    return stored


def verify_log_chain(path: Path) -> tuple[bool, int, str]:
    """Verify a hash-chained log. Returns (ok, n_records, first_bad_reason)."""
    if not path.exists():
        return True, 0, ""
    prev_hash = GENESIS_HASH
    count = 0
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            stored_prev = row.pop("prev_hash", None)
            stored_hash = row.pop("record_hash", None)
            if stored_prev != prev_hash:
                return False, count, f"line {line_no}: prev_hash mismatch"
            expected = hashlib.sha256((prev_hash + canonical_json(row)).encode()).hexdigest()
            if stored_hash != expected:
                return False, count, f"line {line_no}: record_hash mismatch"
            prev_hash = stored_hash
            count += 1
    return True, count, ""
