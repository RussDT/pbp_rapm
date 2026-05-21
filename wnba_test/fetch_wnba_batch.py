"""
WNBA Play-by-Play Batch Fetcher (PARALLEL VERSION)
Uses same parallel/proxy strategy as NBA fetch_pbp_v3_parallel.py
"""
import argparse
import os
import time
import random
import uuid
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
import pandas as pd

from nba_api.stats.endpoints import leaguegamelog, playbyplayv3, gamerotation

WNBA_LEAGUE_ID = "10"

# ---------------------------------------------------------------------
# Proxy Configuration (same as NBA)
# ---------------------------------------------------------------------
NBA_PROXY_HOST = 'geo.iproyal.com'
NBA_PROXY_PORT = '12321'
NBA_PROXY_USERNAME = 'd5lJCmpqq6SOK4LD'

def get_proxy_url(worker_id: int = 0) -> str:
    """Generate proxy URL with unique session per worker for IP rotation."""
    session_id = f"wnba_w{worker_id}_{uuid.uuid4().hex[:8]}"
    password = f'LRYNd2qaZXVePBAF_country-us_session-{session_id}_lifetime-5m'
    return f"http://{NBA_PROXY_USERNAME}:{password}@{NBA_PROXY_HOST}:{NBA_PROXY_PORT}"

# ---------------------------------------------------------------------
# Sneaky Request Headers
# ---------------------------------------------------------------------
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]

REFERERS = [
    "https://www.wnba.com/",
    "https://stats.wnba.com/",
    "https://www.google.com/",
]

def get_random_headers() -> Dict[str, str]:
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Referer": random.choice(REFERERS),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Origin": "https://www.wnba.com",
        "Connection": "keep-alive",
    }

def random_delay(min_sec: float = 0.1, max_sec: float = 0.3) -> None:
    time.sleep(random.uniform(min_sec, max_sec))

# Parallel config
MAX_WORKERS = 6
progress_lock = threading.Lock()
progress_counter = {"completed": 0, "total": 0, "failed": 0}

# ---------------------------------------------------------------------
# Retry wrapper with proxy support
# ---------------------------------------------------------------------
def retry_api_call(func, max_attempts=6, delay_seconds=2, use_proxy=True, worker_id=0, **kwargs):
    for attempt in range(1, max_attempts + 1):
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
            delay = random.uniform(delay_seconds, delay_seconds * 1.5)
            time.sleep(delay)
    return None

# ---------------------------------------------------------------------
# Get game IDs
# ---------------------------------------------------------------------
def get_game_ids(season: str, season_type: str = "Regular Season", use_proxy: bool = True) -> List[str]:
    print(f"  Fetching WNBA game log for {season} {season_type}...")
    log = retry_api_call(
        leaguegamelog.LeagueGameLog,
        use_proxy=use_proxy,
        worker_id=0,
        season=season,
        league_id=WNBA_LEAGUE_ID,
        season_type_all_star=season_type
    )
    if log:
        df = log.get_data_frames()[0]
        game_ids = df["GAME_ID"].unique().tolist()
        random.shuffle(game_ids)  # Shuffle for stealth
        print(f"  Found {len(game_ids)} games (shuffled)")
        return game_ids
    return []

# ---------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------
def clock_to_seconds(clock_str):
    if not clock_str or not isinstance(clock_str, str):
        return 0
    c = clock_str.replace("PT", "").replace("S", "")
    if "M" in c:
        parts = c.split("M")
        m = float(parts[0])
        s = float(parts[1]) if parts[1] else 0
        return m * 60 + s
    return float(c)

def get_lineup_at_time(stints, time_from_start):
    active = stints[(stints["IN_TIME_REAL"] <= time_from_start) & (stints["OUT_TIME_REAL"] > time_from_start)]
    return active["PERSON_ID"].tolist()

# ---------------------------------------------------------------------
# Process single game
# ---------------------------------------------------------------------
def process_single_game(game_id: str, use_proxy: bool = True, worker_id: int = 0) -> Optional[pd.DataFrame]:
    try:
        random_delay(0.1, 0.3)
        
        # Get PBP
        pbp = retry_api_call(playbyplayv3.PlayByPlayV3, use_proxy=use_proxy, worker_id=worker_id, game_id=game_id)
        if not pbp:
            return None
        df = pbp.get_data_frames()[0]
        df.columns = [c.lower() for c in df.columns]
        
        random_delay(0.1, 0.3)
        
        # Get rotation
        rot = retry_api_call(gamerotation.GameRotation, use_proxy=use_proxy, worker_id=worker_id, game_id=game_id)
        if not rot:
            return None
        dfs = rot.get_data_frames()
        rot_away, rot_home = dfs[0], dfs[1]
        
        if len(rot_away) == 0 or len(rot_home) == 0:
            return None
        
        away_id = int(rot_away.iloc[0]["TEAM_ID"])
        home_id = int(rot_home.iloc[0]["TEAM_ID"])
        
        # Add lineup columns
        for i in range(1, 6):
            df[f"away_player{i}"] = np.nan
            df[f"home_player{i}"] = np.nan
        
        for idx, row in df.iterrows():
            period = int(row["period"])
            rem_q = clock_to_seconds(row["clock"])
            # WNBA has 10-min quarters, rotation API uses 12-min scale
            elapsed_in_q = 600 - rem_q
            stretched_elapsed = elapsed_in_q * 1.2
            game_time_tenths = ((period - 1) * 7200) + (stretched_elapsed * 10)
            check_time = game_time_tenths + 5
            
            away_l = get_lineup_at_time(rot_away, check_time)
            home_l = get_lineup_at_time(rot_home, check_time)
            
            for i in range(5):
                if i < len(away_l):
                    df.at[idx, f"away_player{i+1}"] = away_l[i]
                if i < len(home_l):
                    df.at[idx, f"home_player{i+1}"] = home_l[i]
        
        # Standard normalization
        df["player1_id"] = df["personid"]
        df["player1_team_id"] = df["teamid"]
        df["home_description"] = np.where(df["teamid"] == home_id, df["description"], "")
        df["visitor_description"] = np.where(df["teamid"] == away_id, df["description"], "")
        df["seconds_remaining_quarter"] = df["clock"].apply(clock_to_seconds)
        df["score"] = df["scoreaway"].astype(str) + " - " + df["scorehome"].astype(str)
        df = df.rename(columns={
            "gameid": "game_id",
            "actionnumber": "event_num",
            "scorehome": "home_score",
            "scoreaway": "away_score"
        })
        
        # Event type mapping
        def map_ev(a, s):
            a, s = str(a).lower(), str(s).lower()
            if "period" in a:
                return 12 if "start" in s else 13
            if "made" in a:
                return 1
            if "miss" in a:
                return 2
            if "free throw" in a:
                return 3
            if "rebound" in a:
                return 4
            if "turnover" in a:
                return 5
            if "foul" in a:
                return 6
            if "substitution" in a:
                return 8
            return 0
        
        df["event_type"] = df.apply(lambda r: map_ev(r["actiontype"], r["subtype"]), axis=1)
        df["away_score"] = pd.to_numeric(df["away_score"], errors='coerce').fillna(0)
        df["home_score"] = pd.to_numeric(df["home_score"], errors='coerce').fillna(0)
        
        return df
        
    except Exception as e:
        return None

def process_game_wrapper(game_id: str, total_games: int, use_proxy: bool, worker_id: int) -> Tuple[str, Optional[pd.DataFrame]]:
    """Wrapper for parallel processing with progress tracking."""
    global progress_counter
    
    result = process_single_game(game_id, use_proxy, worker_id)
    
    with progress_lock:
        progress_counter["completed"] += 1
        if result is None:
            progress_counter["failed"] += 1
        completed = progress_counter["completed"]
        failed = progress_counter["failed"]
    
    status = "OK" if result is not None else "FAIL"
    events = len(result) if result is not None else 0
    print(f"  [W{worker_id}][{completed}/{total_games}] {game_id}: {status} ({events} events) [Failed: {failed}]")
    
    return game_id, result

# ---------------------------------------------------------------------
# Fetch season data (parallel)
# ---------------------------------------------------------------------
def fetch_season_data(season: str, season_type: str, output_dir: str = "wnba_test", 
                      use_proxy: bool = True, workers: int = MAX_WORKERS) -> Optional[str]:
    global progress_counter
    
    game_ids = get_game_ids(season, season_type, use_proxy)
    if not game_ids:
        print(f"  No games found for {season} {season_type}")
        return None
    
    print(f"  Processing {len(game_ids)} games with {workers} parallel workers...")
    
    progress_counter = {"completed": 0, "total": len(game_ids), "failed": 0}
    all_results = []
    
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            
            # Stagger worker entry
            print("  Staggering initial worker entry...")
            for i, gid in enumerate(game_ids):
                worker_id = i % workers
                futures[executor.submit(process_game_wrapper, gid, len(game_ids), use_proxy, worker_id)] = gid
                
                # Stagger first batch of workers
                if i < workers:
                    print(f"    Worker {worker_id} entering...")
                    time.sleep(2)
            
            for future in as_completed(futures):
                try:
                    game_id, result = future.result(timeout=120)
                    if result is not None and not result.empty:
                        all_results.append(result)
                except Exception as e:
                    continue
                    
    except KeyboardInterrupt:
        print("\n  Interrupted! Saving partial results...")
    
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        
        # Determine filename
        year_suffix = season[-2:]
        if season_type == "Playoffs":
            filename = f"{output_dir}/WNBA{year_suffix}_PS.csv"
        else:
            filename = f"{output_dir}/WNBA{year_suffix}.csv"
        
        combined.to_csv(filename, index=False)
        print(f"\n  Saved {len(combined)} rows ({len(all_results)} games) to {filename}")
        print(f"  Failed: {progress_counter['failed']} games")
        
        return filename
    else:
        print(f"  No data collected for {season} {season_type}")
        return None

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fetch WNBA play-by-play data (PARALLEL)")
    parser.add_argument("--seasons", type=str, nargs="+", default=["2023", "2024", "2025"],
                       help="Seasons to fetch (e.g., 2023 2024 2025)")
    parser.add_argument("--types", type=str, nargs="+", default=["Regular Season", "Playoffs"],
                       choices=["Regular Season", "Playoffs"],
                       help="Season types to fetch")
    parser.add_argument("--output-dir", type=str, default="wnba_test",
                       help="Output directory for CSV files")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                       help=f"Parallel workers (default: {MAX_WORKERS})")
    parser.add_argument("--no-proxy", action="store_true", help="Disable proxy")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    use_proxy = not args.no_proxy
    
    print("\n" + "=" * 60)
    print("WNBA Play-by-Play Fetcher (PARALLEL)")
    print(f"Seasons     : {args.seasons}")
    print(f"Types       : {args.types}")
    print(f"Workers     : {args.workers}")
    print(f"Proxy       : {'Enabled' if use_proxy else 'Disabled'}")
    print("=" * 60 + "\n")
    
    for season in args.seasons:
        for season_type in args.types:
            print(f"\n{'='*60}")
            print(f"Fetching {season} {season_type}")
            print(f"{'='*60}")
            fetch_season_data(season, season_type, args.output_dir, use_proxy, args.workers)

if __name__ == "__main__":
    main()
