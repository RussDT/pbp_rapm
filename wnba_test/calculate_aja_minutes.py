import pandas as pd
import numpy as np

AJA_ID = 1628932

# Load raw PBP data for 2025
df = pd.read_csv("wnba_test/WNBA25.csv")

# Ensure time columns are correct
# seconds_remaining_quarter was added by my fetch script
if 'seconds_remaining_quarter' not in df.columns:
    def clock_v3_to_seconds(c):
        if not c or not isinstance(c, str): return 0
        c = c.replace("PT", "").replace("S", "")
        if "M" in c:
            parts = c.split("M")
            m = float(parts[0])
            s = float(parts[1]) if parts[1] else 0
            return m * 60 + s
        return float(c)
    df['seconds_remaining_quarter'] = df['time_quarter'].apply(clock_v3_to_seconds)

# Sort by game, period, and event_num to track time intervals
df = df.sort_values(['game_id', 'period', 'event_num'])

total_seconds_on = 0

for gid in df['game_id'].unique():
    game_df = df[df['game_id'] == gid]
    for period in game_df['period'].unique():
        p_df = game_df[game_df['period'] == period].reset_index(drop=True)
        
        # Start of period is 600s (10 mins)
        last_time = 600.0
        
        for i, row in p_df.iterrows():
            curr_time = row['seconds_remaining_quarter']
            
            # Check if A'ja was on court at this event
            # Lineup columns: away_player1-5, home_player1-5
            lineup = [row[f'away_player{j}'] for j in range(1, 6)] + \
                     [row[f'home_player{j}'] for j in range(1, 6)]
            
            is_on = AJA_ID in lineup
            
            if is_on:
                # Add the time since the last event to her total
                # Note: This assumes she was on for the whole interval
                # which is generally true in PBP unless a sub happened at this exact time
                duration = last_time - curr_time
                if duration > 0:
                    total_seconds_on += duration
            
            last_time = curr_time

print(f"\n--- A'ja Wilson On-Court Time Calculation (2025) ---")
print(f"Total Seconds: {total_seconds_on:.2f}")
print(f"Total Minutes: {total_seconds_on / 60:.2f}")
print(f"BBRef Expected: 1248")
print(f"Difference:    {(total_seconds_on / 60) - 1248:.2f} minutes")

