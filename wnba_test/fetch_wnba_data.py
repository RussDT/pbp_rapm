import argparse
import os
import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog, playbyplayv3, gamerotation, boxscoretraditionalv2
import numpy as np

WNBA_LEAGUE_ID = "10"

def retry_api_call(func, max_attempts=6, delay_seconds=2, **kwargs):
    for attempt in range(1, max_attempts + 1):
        try:
            return func(**kwargs)
        except Exception as e:
            if attempt == max_attempts:
                print(f"      Error: {e}")
                return None
            time.sleep(delay_seconds)
    return None

def get_game_ids(season: str, season_type: str = "Regular Season"):
    print(f"  Fetching WNBA game log for {season} {season_type}...")
    log = retry_api_call(leaguegamelog.LeagueGameLog, 
                         season=season, 
                         league_id=WNBA_LEAGUE_ID,
                         season_type_all_star=season_type)
    return log.get_data_frames()[0]["GAME_ID"].unique().tolist() if log else []

def get_rotation_stints(game_id: str):
    rot = retry_api_call(gamerotation.GameRotation, game_id=game_id)
    if not rot: return None, None
    dfs = rot.get_data_frames()
    return dfs[0], dfs[1] # Away, Home stints

def clock_to_seconds(clock_str):
    if not clock_str or not isinstance(clock_str, str): return 0
    c = clock_str.replace("PT", "").replace("S", "")
    if "M" in c:
        parts = c.split("M")
        m = float(parts[0])
        s = float(parts[1]) if parts[1] else 0
        return m * 60 + s
    return float(c)

def get_lineup_at_time(stints, time_from_start):
    # GameRotation uses tenths of a second from start, but stretched to 12-min quarters (NBA style)
    # Even for WNBA 10-min quarters.
    # Q1: 0-7200, Q2: 7200-14400, etc.
    # We need to map our PBP time to this.
    active = stints[(stints["IN_TIME_REAL"] <= time_from_start) & (stints["OUT_TIME_REAL"] > time_from_start)]
    return active["PERSON_ID"].tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=str, default="2025")
    parser.add_argument("--season-type", type=str, default="Regular Season", choices=["Regular Season", "Playoffs", "Pre Season"])
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing data instead of appending")
    args = parser.parse_args()
    
    ps_suffix = "_PS" if args.season_type == "Playoffs" else ""
    output_file = f"wnba_test/WNBA{args.season[-2:]}{ps_suffix}.csv"
    existing_df = None
    existing_game_ids = set()
    
    if os.path.exists(output_file) and not args.overwrite:
        try:
            existing_df = pd.read_csv(output_file, dtype={"game_id": str})
            if "game_id" in existing_df.columns:
                existing_game_ids = set(existing_df["game_id"].unique().tolist())
                print(f"  Found {len(existing_game_ids)} existing games in {output_file}")
        except Exception as e:
            print(f"  Warning: Could not load existing file {output_file}: {e}")

    game_ids = get_game_ids(args.season, args.season_type)
    if not game_ids: return
    
    # Filter out already pulled games
    new_game_ids = [str(gid) for gid in game_ids if str(gid) not in existing_game_ids]
    print(f"  Total games found in log: {len(game_ids)}, New games to fetch: {len(new_game_ids)}")
    
    new_game_ids = new_game_ids[:args.limit]
    
    all_games = [] if existing_df is None or args.overwrite else [existing_df]
    
    for gid in new_game_ids:
        print(f"  Processing {gid}...")
        pbp = retry_api_call(playbyplayv3.PlayByPlayV3, game_id=gid)
        if not pbp: continue
        df = pbp.get_data_frames()[0]
        df.columns = [c.lower() for c in df.columns]
        
        rot_away, rot_home = get_rotation_stints(gid)
        if rot_away is None: continue
        
        away_id = int(rot_away.iloc[0]["TEAM_ID"])
        home_id = int(rot_home.iloc[0]["TEAM_ID"])
        
        # Add lineup columns
        for i in range(1, 6):
            df[f"away_player{i}"] = np.nan
            df[f"home_player{i}"] = np.nan
            
        for idx, row in df.iterrows():
            # Calculate time from start in tenths of a second, adjusted to 12-min quarter scale
            period = int(row["period"])
            rem_q = clock_to_seconds(row["clock"])
            
            # WNBA quarter is 10 mins (600s). NBA is 12 mins (720s).
            # The API rotation clock treats every quarter as 7200 tenths.
            elapsed_in_q = 600 - rem_q
            # Stretch 10 mins to 12 mins
            stretched_elapsed = elapsed_in_q * 1.2
            
            # Time from start of game in tenths
            game_time_tenths = ((period - 1) * 7200) + (stretched_elapsed * 10)
            
            # Small offset to avoid sub boundaries (0.5 seconds)
            check_time = game_time_tenths + 5
            
            away_l = get_lineup_at_time(rot_away, check_time)
            home_l = get_lineup_at_time(rot_home, check_time)
            
            for i in range(5):
                if i < len(away_l): df.at[idx, f"away_player{i+1}"] = away_l[i]
                if i < len(home_l): df.at[idx, f"home_player{i+1}"] = home_l[i]
        
        # Standard normalization for RAPM pipeline
        df["player1_id"] = df["personid"]
        df["player1_team_id"] = df["teamid"]
        df["home_description"] = np.where(df["teamid"] == home_id, df["description"], "")
        df["visitor_description"] = np.where(df["teamid"] == away_id, df["description"], "")
        df["seconds_remaining_quarter"] = df["clock"].apply(clock_to_seconds)
        df["score"] = df["scoreaway"].astype(str) + " - " + df["scorehome"].astype(str)
        df = df.rename(columns={"gameid": "game_id", "actionnumber": "event_num", "scorehome": "home_score", "scoreaway": "away_score"})
        
        # Basic event mapping
        def map_ev(a, s):
            a, s = str(a).lower(), str(s).lower()
            if "period" in a: return 12 if "start" in s else 13
            if "made" in a: return 1
            if "miss" in a: return 2
            if "free throw" in a: return 3
            if "rebound" in a: return 4
            if "turnover" in a: return 5
            if "foul" in a: return 6
            if "substitution" in a: return 8
            return 0
        df["event_type"] = df.apply(lambda r: map_ev(r["actiontype"], r["subtype"]), axis=1)
        
        all_games.append(df)
        time.sleep(1.2)
        
    if len(all_games) > (1 if existing_df is not None and not args.overwrite else 0):
        out = pd.concat(all_games, ignore_index=True)
        # Ensure away_score and home_score are numeric
        out["away_score"] = pd.to_numeric(out["away_score"], errors='coerce').fillna(0)
        out["home_score"] = pd.to_numeric(out["home_score"], errors='coerce').fillna(0)
        out.to_csv(output_file, index=False)
        print(f"Saved {len(out)} rows to {output_file}")
    else:
        print("  No new games fetched.")

if __name__ == "__main__":
    main()
