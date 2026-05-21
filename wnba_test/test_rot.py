from nba_api.stats.endpoints import gamerotation
import pandas as pd

gid = "1022400001"
rot = gamerotation.GameRotation(game_id=gid)
dfs = rot.get_data_frames()
for i, df in enumerate(dfs):
    print(f"DF {i} columns: {df.columns.tolist() if not df.empty else 'EMPTY'}")
    if not df.empty:
        print(df.head())

