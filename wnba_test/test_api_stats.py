from nba_api.stats.endpoints import leaguedashplayerstats
import time

try:
    stats = leaguedashplayerstats.LeagueDashPlayerStats(season="2024", league_id_nullable="10")
    df = stats.get_data_frames()[0]
    print(f"Successfully fetched {len(df)} WNBA player stats for 2024")
    print(df[["PLAYER_ID", "PLAYER_NAME", "FT_PCT", "FG3_PCT"]].head())
except Exception as e:
    print(f"Error: {e}")

