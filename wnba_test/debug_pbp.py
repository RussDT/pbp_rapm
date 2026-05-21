from nba_api.stats.endpoints import playbyplayv3
import pandas as pd

gid = "1022400001"
pbp = playbyplayv3.PlayByPlayV3(game_id=gid)
df = pbp.get_data_frames()[0]
print(f"PBP columns: {df.columns.tolist()}")
print(df.head())

