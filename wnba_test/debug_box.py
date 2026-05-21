from nba_api.stats.endpoints import boxscoresummaryv2
import pandas as pd

gid = "1022400001"
box = boxscoresummaryv2.BoxScoreSummaryV2(game_id=gid)
dfs = box.get_data_frames()
for i, df in enumerate(dfs):
    print(f"DF {i} columns: {df.columns.tolist() if not df.empty else 'EMPTY'}")
    if not df.empty and "START_POSITION" in df.columns:
        print(f"Found START_POSITION in DF {i}")
        print(df[["PLAYER_ID", "START_POSITION", "TEAM_ID"]].head())

