from nba_api.stats.endpoints import playbyplayv3
import pandas as pd

gid = "0022300001" # NBA game
pbp = playbyplayv3.PlayByPlayV3(game_id=gid)
df = pbp.get_data_frames()[0]
print(f"NBA PBP V3 columns: {df.columns.tolist()}")

