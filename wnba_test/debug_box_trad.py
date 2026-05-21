from nba_api.stats.endpoints import boxscoretraditionalv2
import pandas as pd

gid = "1022400001"
box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=gid)
dfs = box.get_data_frames()
for i, df in enumerate(dfs):
    print(f"DF {i} columns: {df.columns.tolist() if not df.empty else 'EMPTY'}")
    if not df.empty and "START_POSITION" in df.columns:
        print(f"Found START_POSITION in DF {i}")
        print(df[df["START_POSITION"].notna() & (df["START_POSITION"] != "")][["PLAYER_ID", "PLAYER_NAME", "START_POSITION", "TEAM_ID"]])

