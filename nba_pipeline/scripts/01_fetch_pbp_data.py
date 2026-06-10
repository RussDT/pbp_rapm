"""
NBA Play-by-Play V3 → V2-Style PBP with Lineups (PARALLEL VERSION)

Usage (from root):
    python nba_pipeline/scripts/01_fetch_pbp_data.py 26        # Update 2025-26 Regular Season
    python nba_pipeline/scripts/01_fetch_pbp_data.py 26 PS     # Update 2025-26 Playoffs
    python nba_pipeline/scripts/01_fetch_pbp_data.py 26 --full # Full fetch (not update)

Uses proxy and concurrent requests to speed up fetching.
SNEAKY MODE: Rotates User-Agents, headers, and timing to avoid fingerprinting.

Output goes to: nba_pipeline/raw_data/NBAXX.csv
"""

import argparse
import os
import sys
import time
import random
import uuid
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from pathlib import Path

import numpy as np
import pandas as pd

# Route nba_api through curl_cffi (Chrome TLS fingerprint) so stats.nba.com's
# Akamai bot-mitigation doesn't silently stall our requests. No-op if curl_cffi
# is unavailable. Must precede any nba_api request.
sys.path.insert(0, str(Path(__file__).parent))
import _nba_http_curlcffi  # noqa: E402,F401

from nba_api.stats.endpoints import (
    leaguegamelog,
    playbyplayv3,
    gamerotation,
)
from nba_api.stats.static import teams as static_teams

# Get the pipeline root directory (parent of scripts/)
SCRIPT_DIR = Path(__file__).parent
PIPELINE_ROOT = SCRIPT_DIR.parent
RAW_DATA_DIR = PIPELINE_ROOT / "raw_data"

# ---------------------------------------------------------------------
# Proxy Configuration
# ---------------------------------------------------------------------
NBA_PROXY_HOST = 'geo.iproyal.com'
NBA_PROXY_PORT = '12321'
NBA_PROXY_USERNAME = 'd5lJCmpqq6SOK4LD'

def get_proxy_url(worker_id: int = 0) -> str:
    """Generate proxy URL with unique session per worker for IP rotation."""
    session_id = f"worker{worker_id}_{uuid.uuid4().hex[:8]}"
    password = f'LRYNd2qaZXVePBAF_country-us_session-{session_id}_lifetime-5m'
    return f"http://{NBA_PROXY_USERNAME}:{password}@{NBA_PROXY_HOST}:{NBA_PROXY_PORT}"

# ---------------------------------------------------------------------
# Sneaky Request Headers - Rotate to avoid fingerprinting
# ---------------------------------------------------------------------
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
]

REFERERS = [
    "https://www.nba.com/",
    "https://www.nba.com/stats",
    "https://www.nba.com/games",
    "https://stats.nba.com/",
    "https://www.google.com/",
]

def get_random_headers() -> Dict[str, str]:
    """Generate randomized headers to avoid fingerprinting."""
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Referer": random.choice(REFERERS),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": random.choice(["en-US,en;q=0.9", "en-US,en;q=0.8", "en;q=0.9"]),
        "Accept-Encoding": "gzip, deflate, br",
        "Origin": "https://www.nba.com",
        "Connection": "keep-alive",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors", 
        "Sec-Fetch-Site": "same-site",
        "x-nba-stats-origin": "stats",
        "x-nba-stats-token": "true",
    }

def random_delay(min_sec: float = 0.1, max_sec: float = 0.3) -> None:
    """Random delay to avoid timing fingerprinting."""
    time.sleep(random.uniform(min_sec, max_sec))

# Parallel config
MAX_WORKERS = 8
REQUEST_DELAY = 0.2

# Gentle serial pacing (env-tunable). NBA_INTER_GAME_DELAY adds a fixed sleep at
# the start of each game so a workers=1 run paces ~1 game / N seconds.
INTER_GAME_DELAY = float(os.environ.get("NBA_INTER_GAME_DELAY", "0"))
# Rotation retry attempts (gamerotation rate-limits; lower = fail fast + rely on
# resumable re-passes; higher = grind each game now).
ROT_ATTEMPTS = int(os.environ.get("NBA_ROT_ATTEMPTS", "8"))

# Thread-safe counter for progress
progress_lock = threading.Lock()
progress_counter = {"completed": 0, "total": 0, "failed": 0}

# Thread-local storage for worker ID
thread_local = threading.local()


def normalize_game_id_series(series: pd.Series) -> pd.Series:
    """Normalize game IDs to stable 10-character strings."""
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(10)
    )


def dedupe_raw_pbp_rows(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Drop exact duplicate raw PBP rows and keep write order stable."""
    if df.empty:
        return df

    deduped = df.drop_duplicates().copy()
    removed = len(df) - len(deduped)
    if removed:
        print(f"  Removed {removed} exact duplicate raw PBP rows from {label}.")

    sort_cols = [col for col in ["game_id", "period", "event_num"] if col in deduped.columns]
    if sort_cols:
        deduped = deduped.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    else:
        deduped = deduped.reset_index(drop=True)
    return deduped


def repair_lineup_slots(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Fill sparse lineup slots within each game before writing raw parquet."""
    player_cols = [f"{team}_player{i}" for team in ["away", "home"] for i in range(1, 6)]
    missing_cols = [col for col in player_cols if col not in df.columns]
    if missing_cols or df.empty or "game_id" not in df.columns:
        return df

    missing_rows = df[player_cols].isna().any(axis=1)
    if not missing_rows.any():
        return df

    repaired = df.copy()
    repaired[player_cols] = repaired.groupby("game_id")[player_cols].ffill().bfill()
    recovered_rows = (
        missing_rows
        & ~repaired[player_cols].isna().any(axis=1)
    ).sum()
    if recovered_rows:
        print(f"  Recovered {int(recovered_rows)} rows with missing lineup slots in {label}.")
    return repaired


# ---------------------------------------------------------------------
# Retry wrapper for NBA API calls (with sneaky proxy + headers)
# ---------------------------------------------------------------------
def retry_api_call(func, max_attempts=6, delay_seconds=2, late_delay_seconds=4,
                   use_proxy=True, worker_id=0, backoff_cap=None, **kwargs):
    """Retry an nba_api endpoint call. A 500/empty body (common when
    gamerotation is rate-limited) raises during parsing and is retried here.
    Later attempts use exponential backoff up to ``backoff_cap`` seconds so a
    throttled endpoint has time to cool down."""
    attempt = 1
    while attempt <= max_attempts:
        try:
            headers = get_random_headers()
            if use_proxy:
                proxy = get_proxy_url(worker_id)
                return func(proxy=proxy, headers=headers, timeout=30, **kwargs)
            else:
                return func(headers=headers, timeout=30, **kwargs)
        except Exception as e:
            if attempt == max_attempts:
                return None
            if backoff_cap is not None:
                # Exponential backoff with jitter, capped (for throttled endpoints).
                base = min(late_delay_seconds * (2 ** (attempt - 1)), backoff_cap)
                delay = random.uniform(base * 0.7, base)
            else:
                delay = (random.uniform(delay_seconds, delay_seconds * 1.5)
                         if attempt < 4 else
                         random.uniform(late_delay_seconds, late_delay_seconds * 1.5))
            time.sleep(delay)
            attempt += 1
    return None


# ---------------------------------------------------------------------
# Static team metadata
# ---------------------------------------------------------------------
def build_team_lookup() -> Dict[int, Dict[str, str]]:
    lookup: Dict[int, Dict[str, str]] = {}
    for t in static_teams.get_teams():
        try:
            tid = int(t["id"])
            lookup[tid] = {
                "city": t.get("city", None),
                "nickname": t.get("nickname", None),
                "abbreviation": t.get("abbreviation", None),
            }
        except Exception:
            continue
    return lookup


TEAM_LOOKUP = build_team_lookup()


# ---------------------------------------------------------------------
# Game IDs from LeagueGameLog
# ---------------------------------------------------------------------
def get_game_ids(season: str, season_type: Optional[str], use_proxy: bool = True) -> List[str]:
    print(f"  Fetching game log for season {season}, type: {season_type}...")
    
    season_type_map = {
        "Regular Season": "Regular Season",
        "Playoffs": "Playoffs",
        "Pre Season": "Pre Season",
    }
    api_season_type = season_type_map.get(season_type, "Regular Season")
    
    kwargs = {
        "season": season, 
        "league_id": "00",
        "season_type_all_star": api_season_type,
    }

    game_log = retry_api_call(
        leaguegamelog.LeagueGameLog,
        max_attempts=6,
        delay_seconds=5,
        late_delay_seconds=4,
        use_proxy=use_proxy,
        worker_id=0,
        **kwargs,
    )
    if game_log is None:
        print("  Error: failed to fetch game log")
        return []

    try:
        df = game_log.get_data_frames()[0]
        if "GAME_ID" not in df.columns:
            print("  Error: GAME_ID column not in game log")
            return []

        game_ids = df["GAME_ID"].unique().tolist()
        random.shuffle(game_ids)
        print(f"  Found {len(game_ids)} unique {season_type} game IDs (shuffled for stealth)")
        return game_ids
    except Exception as e:
        print(f"  Error processing game log: {e}")
        return []


# ---------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------
def parse_clock_iso_to_secs_remaining(clock_str: str) -> Optional[float]:
    if not isinstance(clock_str, str) or not clock_str.startswith("PT"):
        return None
    try:
        s = clock_str[2:-1]
        if "M" not in s:
            return None
        m_str, s_str = s.split("M")
        minutes = int(m_str)
        seconds = float(s_str)
        return minutes * 60.0 + seconds
    except Exception:
        return None


def compute_event_seconds_from_start(period: int, secs_remaining: float) -> float:
    total = 0.0
    for p in range(1, period):
        if p <= 4:
            total += 12 * 60.0
        else:
            total += 5 * 60.0

    if period <= 4:
        period_len = 12 * 60.0
    else:
        period_len = 5 * 60.0

    elapsed_in_period = period_len - secs_remaining
    return total + elapsed_in_period


def clock_iso_to_mmss(clock_str: str) -> Optional[str]:
    secs_rem = parse_clock_iso_to_secs_remaining(clock_str)
    if secs_rem is None:
        return None
    total_sec = int(round(secs_rem))
    m = total_sec // 60
    s = total_sec % 60
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------
# GameRotation → stints
# ---------------------------------------------------------------------
def fetch_rotation_stints(game_id: str, max_event_sec: float, use_proxy: bool = True, worker_id: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    random_delay(0.1, 0.3)
    
    rot = retry_api_call(
        gamerotation.GameRotation,
        max_attempts=ROT_ATTEMPTS,
        delay_seconds=3,
        late_delay_seconds=4,
        backoff_cap=20,  # gamerotation rate-limits hard; brief cooldown recovers it
        use_proxy=use_proxy,
        worker_id=worker_id,
        game_id=game_id,
        league_id="00",
    )
    if rot is None:
        return pd.DataFrame(), pd.DataFrame()

    dfs = rot.get_data_frames()
    if len(dfs) < 2:
        return pd.DataFrame(), pd.DataFrame()

    away_df_raw, home_df_raw = dfs[0].copy(), dfs[1].copy()

    def build_stints(df_raw: pd.DataFrame) -> pd.DataFrame:
        if df_raw.empty:
            return pd.DataFrame(columns=["person_id", "team_id", "in_secs", "out_secs"])

        cols = {c.upper(): c for c in df_raw.columns}
        team_col = cols.get("TEAM_ID")
        person_col = cols.get("PERSON_ID")
        in_col = cols.get("IN_TIME_REAL")
        out_col = cols.get("OUT_TIME_REAL")

        stints = []
        for _, row in df_raw.iterrows():
            try:
                person_id = int(row[person_col])
                team_id = int(row[team_col])
            except Exception:
                continue

            in_raw = row[in_col]
            out_raw = row[out_col]

            def parse_rot_time(val):
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    return None
                try:
                    return float(val) / 10.0
                except Exception:
                    return None

            in_secs = parse_rot_time(in_raw)
            out_secs = parse_rot_time(out_raw)

            if in_secs is None:
                in_secs = 0.0
            if out_secs is None:
                out_secs = max_event_sec + 10.0

            stints.append({
                "person_id": person_id,
                "team_id": team_id,
                "in_secs": in_secs,
                "out_secs": out_secs,
            })

        return pd.DataFrame(stints)

    away_stints = build_stints(away_df_raw)
    home_stints = build_stints(home_df_raw)
    return away_stints, home_stints


# ---------------------------------------------------------------------
# Map V3 actionType/subType → V2-style event_type
# ---------------------------------------------------------------------
def map_event_type(action_type: Optional[str], sub_type: Optional[str]) -> int:
    a = (action_type or "").strip().lower()
    s = (sub_type or "").strip().lower()

    if a == "period":
        if s == "start":
            return 12
        if s == "end":
            return 13
        return 12

    if a.startswith("made"):
        return 1
    if a.startswith("miss"):
        return 2
    if "free" in a and "throw" in a:
        return 3
    if "rebound" in a:
        return 4
    if "turnover" in a:
        return 5
    if "foul" in a:
        return 6
    if "violation" in a:
        return 7
    if "substitution" in a or a == "sub":
        return 8
    if "timeout" in a:
        return 9
    if "jump" in a and "ball" in a:
        return 10

    return 0


# ---------------------------------------------------------------------
# Fetch + process one game (called by workers)
# ---------------------------------------------------------------------
def fetch_and_process_game(game_id: str, use_proxy: bool = True, worker_id: int = 0) -> Optional[pd.DataFrame]:
    global progress_counter

    try:
        if INTER_GAME_DELAY > 0:
            time.sleep(INTER_GAME_DELAY)
        random_delay(0.1, 0.3)

        pbp_obj = retry_api_call(
            playbyplayv3.PlayByPlayV3,
            max_attempts=6,
            delay_seconds=2,
            late_delay_seconds=4,
            use_proxy=use_proxy,
            worker_id=worker_id,
            game_id=game_id,
        )
        if pbp_obj is None:
            return None

        df_raw = pbp_obj.get_data_frames()[0]
        if df_raw.empty:
            return None

        random_delay(0.1, 0.3)

        df = df_raw.copy()

        required = [
            "gameId", "actionNumber", "clock", "period", "teamId", "teamTricode",
            "personId", "playerName", "xLegacy", "yLegacy", "shotDistance",
            "shotResult", "isFieldGoal", "scoreHome", "scoreAway", "pointsTotal",
            "location", "description", "actionType", "subType", "videoAvailable", "actionId",
        ]
        for col in required:
            if col not in df.columns:
                df[col] = None

        df["secs_remaining_q"] = df["clock"].apply(parse_clock_iso_to_secs_remaining)
        df.loc[df["secs_remaining_q"].isna(), "secs_remaining_q"] = 12 * 60.0
        df["period"] = df["period"].astype(int)
        df["event_secs"] = df.apply(
            lambda r: compute_event_seconds_from_start(int(r["period"]), float(r["secs_remaining_q"])),
            axis=1,
        )

        max_event_sec = float(df["event_secs"].max())

        away_stints, home_stints = fetch_rotation_stints(game_id, max_event_sec, use_proxy, worker_id)
        if away_stints.empty or home_stints.empty:
            return None

        away_team_id = int(away_stints["team_id"].iloc[0])
        home_team_id = int(home_stints["team_id"].iloc[0])

        sub_mask = df["actionType"].fillna("").str.lower().str.contains("substitution")
        sub_events = df[sub_mask][["actionNumber", "event_secs", "teamId", "personId"]].copy()
        sub_events["actionNumber"] = sub_events["actionNumber"].astype(int)
        sub_events["event_secs"] = sub_events["event_secs"].astype(float)
        sub_events["teamId"] = pd.to_numeric(sub_events["teamId"], errors="coerce").fillna(0).astype(int)
        sub_events["personId"] = pd.to_numeric(sub_events["personId"], errors="coerce").fillna(0).astype(int)

        def get_lineup_for_time(stints: pd.DataFrame, t: float) -> List[int]:
            active = stints[(stints["in_secs"] <= t) & (t < stints["out_secs"])]
            players = active["person_id"].astype(int).tolist()
            if len(players) < 5:
                ending_now = stints[(stints["in_secs"] <= t) & (stints["out_secs"] == t)]
                ending_players = ending_now["person_id"].astype(int).tolist()
                for p in ending_players:
                    if p not in players:
                        players.append(p)
            players = sorted(players)
            if len(players) > 5:
                players = players[:5]
            while len(players) < 5:
                players.append(None)
            return players

        def get_lineup_for_event(stints: pd.DataFrame, t: float, event_num: int, team_id: int) -> List[int]:
            lineup = get_lineup_for_time(stints, t)
            TIME_TOLERANCE = 0.6
            same_time_subs = sub_events[
                (abs(sub_events["event_secs"] - t) < TIME_TOLERANCE) &
                (sub_events["actionNumber"] > event_num) &
                (sub_events["teamId"] == team_id)
            ]
            if same_time_subs.empty:
                return lineup
            for _, sub_row in same_time_subs.iterrows():
                player_out_id = int(sub_row["personId"])
                sub_time = float(sub_row["event_secs"])
                if player_out_id not in lineup:
                    player_in_id = None
                    for pid in list(lineup):
                        if pid is None:
                            continue
                        player_stints = stints[stints["person_id"] == pid]
                        for _, ps in player_stints.iterrows():
                            if abs(ps["in_secs"] - sub_time) < 0.5:
                                player_in_id = pid
                                break
                        if player_in_id:
                            break
                    if player_in_id and player_in_id in lineup:
                        idx = lineup.index(player_in_id)
                        lineup[idx] = player_out_id
            return sorted([p for p in lineup if p is not None]) + [None] * (5 - len([p for p in lineup if p is not None]))

        away_lineups: List[List[Optional[int]]] = []
        home_lineups: List[List[Optional[int]]] = []

        for _, row in df.iterrows():
            t = float(row["event_secs"])
            event_num = int(row["actionNumber"])
            away_lineups.append(get_lineup_for_event(away_stints, t, event_num, away_team_id))
            home_lineups.append(get_lineup_for_event(home_stints, t, event_num, home_team_id))

        away_lineups_arr = np.array(away_lineups, dtype=object)
        home_lineups_arr = np.array(home_lineups, dtype=object)

        for i in range(5):
            df[f"away_player{i+1}"] = away_lineups_arr[:, i]
            df[f"home_player{i+1}"] = home_lineups_arr[:, i]

        out = pd.DataFrame()
        out["game_id"] = df["gameId"]
        out["event_num"] = df["actionNumber"].astype(int)
        out["event_type"] = df.apply(lambda r: map_event_type(r["actionType"], r["subType"]), axis=1)
        out["event_action_type"] = 0
        out["period"] = df["period"].astype(int)

        total_game_min = max_event_sec / 60.0
        out["minute_game"] = df["event_secs"] / 60.0
        out["time_remaining"] = total_game_min - out["minute_game"]
        out["wc_time_string"] = None
        out["time_quarter"] = df["clock"].apply(clock_iso_to_mmss)
        out["minute_remaining_quarter"] = df["secs_remaining_q"].apply(lambda s: int(s // 60))
        out["seconds_remaining_quarter"] = df["secs_remaining_q"].apply(lambda s: int(s % 60))

        def split_descriptions(row):
            loc = (row["location"] or "").lower() if isinstance(row["location"], str) else ""
            desc = row["description"]
            if not isinstance(desc, str) or desc.strip() == "":
                return pd.Series({"home": None, "neutral": None, "visitor": None})
            if loc == "h":
                return pd.Series({"home": desc, "neutral": None, "visitor": None})
            if loc == "v":
                return pd.Series({"home": None, "neutral": None, "visitor": desc})
            return pd.Series({"home": None, "neutral": desc, "visitor": None})

        desc_split = df.apply(split_descriptions, axis=1)
        out["home_description"] = desc_split["home"]
        out["neutral_description"] = desc_split["neutral"]
        out["visitor_description"] = desc_split["visitor"]

        out["home_score"] = pd.to_numeric(df["scoreHome"], errors="coerce")
        out["away_score"] = pd.to_numeric(df["scoreAway"], errors="coerce")

        def make_score(row):
            hs, as_ = row["home_score"], row["away_score"]
            if np.isnan(hs) or np.isnan(as_):
                return None
            return f"{int(hs)} - {int(as_)}"

        out["score"] = out.apply(make_score, axis=1)

        def make_margin(row):
            hs, as_ = row["home_score"], row["away_score"]
            if np.isnan(hs) or np.isnan(as_):
                return None
            return int(hs) - int(as_)

        out["score_margin"] = out.apply(make_margin, axis=1)

        out["person1type"] = 0
        out["player1_id"] = pd.to_numeric(df["personId"], errors="coerce").astype("Int64")
        out["player1_name"] = df["playerName"]

        team_ids_series = pd.to_numeric(df["teamId"], errors="coerce").astype("Int64")
        out["player1_team_id"] = team_ids_series

        def team_field(tid, field):
            if pd.isna(tid):
                return None
            info = TEAM_LOOKUP.get(int(tid))
            if not info:
                return None
            return info.get(field)

        out["player1_team_city"] = team_ids_series.apply(lambda tid: team_field(tid, "city"))
        out["player1_team_nickname"] = team_ids_series.apply(lambda tid: team_field(tid, "nickname"))
        out["player1_team_abbreviation"] = team_ids_series.apply(lambda tid: team_field(tid, "abbreviation"))

        out["person2type"] = 0
        out["player2_id"] = pd.Series([pd.NA] * len(out), dtype="Int64")
        out["player2_name"] = None
        out["player2_team_id"] = pd.Series([pd.NA] * len(out), dtype="Int64")
        out["player2_team_city"] = None
        out["player2_team_nickname"] = None
        out["player2_team_abbreviation"] = None

        out["person3type"] = 0
        out["player3_id"] = pd.Series([pd.NA] * len(out), dtype="Int64")
        out["player3_name"] = None
        out["player3_team_id"] = pd.Series([pd.NA] * len(out), dtype="Int64")
        out["player3_team_city"] = None
        out["player3_team_nickname"] = None
        out["player3_team_abbreviation"] = None

        out["video_available_flag"] = pd.to_numeric(df["videoAvailable"], errors="coerce").fillna(0).astype(int)

        def who_leads(row):
            hs, as_ = row["home_score"], row["away_score"]
            if np.isnan(hs) or np.isnan(as_):
                return None
            if hs > as_:
                return "Home"
            if hs < as_:
                return "Away"
            return "TIE"

        out["team_leading"] = out.apply(who_leads, axis=1)

        for i in range(5):
            out[f"away_player{i+1}"] = df[f"away_player{i+1}"].astype("Int64")
        for i in range(5):
            out[f"home_player{i+1}"] = df[f"home_player{i+1}"].astype("Int64")

        # Merge type=0 rows
        type0_rows = out[out["event_type"] == 0].copy()
        if len(type0_rows) > 0:
            merged_count = 0
            rows_to_drop = []
            for idx, t0_row in type0_rows.iterrows():
                ev_num = t0_row["event_num"]
                parent_mask = (out["event_num"] == ev_num) & (out["event_type"] != 0)
                parent_rows = out[parent_mask]
                if len(parent_rows) > 0:
                    parent_idx = parent_rows.index[0]
                    for desc_col in ["home_description", "visitor_description"]:
                        t0_desc = t0_row[desc_col]
                        if pd.notna(t0_desc) and str(t0_desc).strip():
                            parent_desc = out.loc[parent_idx, desc_col]
                            if pd.notna(parent_desc) and str(parent_desc).strip():
                                out.loc[parent_idx, desc_col] = f"{parent_desc} | {t0_desc}"
                            else:
                                out.loc[parent_idx, desc_col] = t0_desc
                    rows_to_drop.append(idx)
                    merged_count += 1
            if rows_to_drop:
                out = out.drop(rows_to_drop).reset_index(drop=True)

        desired_cols = [
            "game_id", "event_num", "event_type", "event_action_type", "period",
            "minute_game", "time_remaining", "wc_time_string", "time_quarter",
            "minute_remaining_quarter", "seconds_remaining_quarter",
            "home_description", "neutral_description", "visitor_description",
            "score", "away_score", "home_score", "score_margin",
            "person1type", "player1_id", "player1_name", "player1_team_id",
            "player1_team_city", "player1_team_nickname", "player1_team_abbreviation",
            "person2type", "player2_id", "player2_name", "player2_team_id",
            "player2_team_city", "player2_team_nickname", "player2_team_abbreviation",
            "person3type", "player3_id", "player3_name", "player3_team_id",
            "player3_team_city", "player3_team_nickname", "player3_team_abbreviation",
            "video_available_flag", "team_leading",
            "away_player1", "away_player2", "away_player3", "away_player4", "away_player5",
            "home_player1", "home_player2", "home_player3", "home_player4", "home_player5",
        ]

        for col in desired_cols:
            if col not in out.columns:
                out[col] = None

        out = out[desired_cols]
        out = out.sort_values(by=["event_num"]).reset_index(drop=True)

        return out

    except Exception as e:
        return None


def process_game_wrapper(game_id: str, game_num: int, total_games: int, use_proxy: bool = True, worker_id: int = 0) -> Tuple[str, Optional[pd.DataFrame]]:
    """Wrapper for parallel processing with progress tracking."""
    global progress_counter
    
    if not hasattr(thread_local, 'worker_id'):
        thread_local.worker_id = worker_id
    
    result = fetch_and_process_game(game_id, use_proxy, worker_id)
    
    with progress_lock:
        progress_counter["completed"] += 1
        if result is None:
            progress_counter["failed"] += 1
        completed = progress_counter["completed"]
        failed = progress_counter["failed"]
        
    status = "OK" if result is not None else "FAILED"
    events = len(result) if result is not None else 0
    print(f"  [W{worker_id}][{completed}/{total_games}] {game_id}: {status} ({events} events) [Failed: {failed}]")
    
    return game_id, result


# ---------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------
def get_existing_game_ids(parquet_path: str) -> set:
    """Read existing parquet file and return set of game IDs already fetched."""
    if not os.path.exists(parquet_path):
        return set()
    df = pd.read_parquet(parquet_path, columns=["game_id"])
    return set(df["game_id"].astype(str).str.zfill(10).unique())


def main():
    parser = argparse.ArgumentParser(
        description="Fetch NBA PlayByPlayV3 + GameRotation with PARALLEL requests.",
        epilog="Examples:\n"
               "  python 01_fetch_pbp_data.py 26        # Update 2025-26 regular season\n"
               "  python 01_fetch_pbp_data.py 26 PS     # Update 2025-26 playoffs\n"
               "  python 01_fetch_pbp_data.py 25        # Update 2024-25 regular season\n"
               "  python 01_fetch_pbp_data.py 26 --full # Full fetch (not update)\n",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("year", type=str, nargs="?", default="26",
                        help="2-digit year (e.g., 26 for 2025-26 season). Default: 26")
    parser.add_argument("season_type_short", type=str, nargs="?", default="RS",
                        choices=["RS", "PS", "PRE"],
                        help="Season type: RS=Regular Season, PS=Playoffs, PRE=Pre Season. Default: RS")
    parser.add_argument("--full", action="store_true",
                        help="Full fetch instead of update mode")
    parser.add_argument("--limit", type=int, default=0, help="Limit games (0=all)")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help=f"Parallel workers (default: {MAX_WORKERS})")
    parser.add_argument("--no-proxy", action="store_true", help="Disable proxy")
    # Legacy arguments for backward compatibility
    parser.add_argument("--season", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--season-type", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--update", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Convert short year to full season string (26 -> 2025-26)
    if args.season is None:
        year_int = int(args.year)
        if year_int < 50:  # Assume 2000s
            start_year = 2000 + year_int - 1
        else:  # Assume 1900s
            start_year = 1900 + year_int - 1
        args.season = f"{start_year}-{args.year}"
    
    # Convert short season type to full name
    if args.season_type is None:
        season_type_map = {"RS": "Regular Season", "PS": "Playoffs", "PRE": "Pre Season"}
        args.season_type = season_type_map.get(args.season_type_short, "Regular Season")
    
    # Default to update mode unless --full is specified
    update_mode = not args.full and not args.update  # --update is legacy, treat as update
    if args.update:
        update_mode = True

    # Ensure raw_data directory exists
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Auto-generate output filename
    if args.output is None:
        ps_suffix = "_PS" if args.season_type != "Regular Season" else ""
        args.output = f"NBA{args.year}{ps_suffix}.parquet"

    use_proxy = not args.no_proxy
    
    # Output to raw_data directory
    out_path = RAW_DATA_DIR / args.output

    print("\n" + "=" * 60)
    print("NBA PlayByPlayV3 → V2-Style PBP (PARALLEL)")
    print(f"Season      : {args.season}")
    print(f"Season Type : {args.season_type}")
    print(f"Output File : {out_path}")
    print(f"Workers     : {args.workers}")
    print(f"Proxy       : {'Enabled' if use_proxy else 'Disabled'}")
    print("=" * 60 + "\n")

    game_ids = get_game_ids(args.season, args.season_type, use_proxy)
    if not game_ids:
        print("No games found. Exiting.")
        return

    # Filter to only new games if update mode (default)
    if update_mode:
        existing_ids = get_existing_game_ids(str(out_path))
        original_count = len(game_ids)
        game_ids = [g for g in game_ids if g not in existing_ids]
        print(f"  UPDATE MODE: Found {len(existing_ids)} existing games in {out_path}")
        print(f"  New games to fetch: {len(game_ids)} (of {original_count} total)\n")
        
        if not game_ids:
            print("Already up to date! No new games to fetch.")
            return

    if args.limit > 0:
        game_ids = game_ids[:args.limit]
        print(f"Limiting to first {len(game_ids)} games\n")

    global progress_counter
    progress_counter = {"completed": 0, "total": len(game_ids), "failed": 0}

    print(f"Processing {len(game_ids)} games with {args.workers} parallel workers (SNEAKY MODE)...\n")
    print(f"  - Rotating User-Agents: {len(USER_AGENTS)} variants")
    print(f"  - Rotating proxy sessions per worker")
    print(f"  - Randomized request timing")
    print(f"  - Staggered worker entry (3s between each)")
    print()

    all_results = []
    
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            
            print("  Staggering initial worker entry...")
            for i, gid in enumerate(game_ids, 1):
                worker_id = (i - 1) % args.workers
                futures[executor.submit(process_game_wrapper, gid, i, len(game_ids), use_proxy, worker_id)] = gid
                
                if i <= args.workers:
                    print(f"    Worker {worker_id} entering...")
                    time.sleep(3)
            
            for future in as_completed(futures):
                try:
                    game_id, result = future.result(timeout=120)
                    if result is not None and not result.empty:
                        all_results.append(result)
                except Exception as e:
                    gid = futures[future]
                    print(f"  ERROR processing {gid}: {e}")
                    continue
    except KeyboardInterrupt:
        print("\n  Interrupted by user. Saving partial results...")
    except Exception as e:
        print(f"\n  FATAL ERROR in executor: {e}")
        import traceback
        traceback.print_exc()

    success = len(all_results)
    fail = len(game_ids) - success

    if not all_results:
        print("\nNo successful games. Exiting.")
        return

    print(f"\nCombining {success} games...")
    combined = pd.concat(all_results, ignore_index=True)

    # Ensure game_id is string type to avoid ArrowInvalid errors
    combined["game_id"] = normalize_game_id_series(combined["game_id"])
    combined = dedupe_raw_pbp_rows(combined, "new fetch batch")
    combined = repair_lineup_slots(combined, "new fetch batch")

    # In update mode, append to existing parquet instead of overwriting
    if update_mode and out_path.exists():
        print(f"  Appending to existing {out_path}...")
        existing_df = pd.read_parquet(out_path)
        existing_df["game_id"] = normalize_game_id_series(existing_df["game_id"])
        combined = pd.concat([existing_df, combined], ignore_index=True)
        combined = dedupe_raw_pbp_rows(combined, out_path.name)
        combined = repair_lineup_slots(combined, out_path.name)
        total_games = len(combined["game_id"].unique())
        print(f"  Total games after merge: {total_games}")

    combined.to_parquet(out_path, index=False)

    print("\n" + "=" * 60)
    print(f"Successfully processed: {success} new games")
    print(f"Failed               : {fail} games")
    print(f"Total events         : {len(combined)}")
    print(f"Output saved to      : {out_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
