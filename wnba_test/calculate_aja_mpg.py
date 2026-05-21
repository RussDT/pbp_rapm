import pandas as pd
import numpy as np

AJA_ID = 1628932
df = pd.read_csv("wnba_test/WNBA25.csv")
df = df.sort_values(['game_id', 'period', 'event_num'])

game_mins = {}

for gid in df['game_id'].unique():
    game_df = df[df['game_id'] == gid]
    game_sec = 0
    aja_in_game = False
    
    for period in game_df['period'].unique():
        p_df = game_df[game_df['period'] == period].reset_index(drop=True)
        last_time = 600.0
        for i, row in p_df.iterrows():
            curr_time = row['seconds_remaining_quarter']
            lineup = [row[f'away_player{j}'] for j in range(1, 6)] + \
                     [row[f'home_player{j}'] for j in range(1, 6)]
            if AJA_ID in lineup:
                duration = last_time - curr_time
                if duration > 0:
                    game_sec += duration
                aja_in_game = True
            last_time = curr_time
    
    if aja_in_game:
        game_mins[gid] = game_sec / 60

print("\n--- A'ja Wilson Minutes Per Game (2025) ---")
for gid, mins in sorted(game_mins.items()):
    print(f"Game {gid}: {mins:.2f} mins")

total_mins = sum(game_mins.values())
print(f"\nTotal Games: {len(game_mins)}")
print(f"Total Minutes: {total_mins:.2f}")

