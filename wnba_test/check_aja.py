import pandas as pd
import numpy as np

# A'ja Wilson ID and Aces Team ID (from player_index_map)
AJA_ID = 1628932
ACES_TEAM_ID = 1611661319

# Load processed possessions
df = pd.read_csv("wnba_test/Processed/RAPM25.csv")

# We need to identify Aces possessions. 
# Since we don't have team_id in RAPM25.csv, let's find games where A'ja played.
# O1-O5 and D1-D5 contain player IDs.
on_mask = df.loc[:, 'O1':'D5'].isin([AJA_ID]).any(axis=1)
aja_games = df[on_mask]['game_id'].unique()

print(f"Found {len(aja_games)} games where A'ja Wilson played.")

# Filter to only these games
df_aces_games = df[df['game_id'].isin(aja_games)].copy()

# For these games, we need to know which side is the Aces.
# In a possession where AJA is on, if she's in O1-O5, Aces are on Offense.
# If she's in D1-D5, Aces are on Defense.

def get_aja_stats(row):
    off_players = [row[f'O{i}'] for i in range(1, 6)]
    def_players = [row[f'D{i}'] for i in range(1, 6)]
    
    if AJA_ID in off_players:
        return "on_offense", row['Net_Diff']
    if AJA_ID in def_players:
        # Net_Diff is from offensive perspective, so defensive impact is -Net_Diff
        return "on_defense", -row['Net_Diff']
    
    # If Aja is off, we need to know if it's an Aces possession.
    # This is tricky because we don't have team IDs. 
    # Let's assume for these games, the Aces are one of the teams.
    # We can try to identify her teammates to confirm.
    return "off", None

# Let's find Aja's teammates in these games to help identify "Aces off" possessions
aja_teammates = set()
for _, row in df[on_mask].iterrows():
    off_players = [int(row[f'O{i}']) for i in range(1, 6)]
    def_players = [int(row[f'D{i}']) for i in range(1, 6)]
    if AJA_ID in off_players:
        aja_teammates.update(off_players)
    if AJA_ID in def_players:
        aja_teammates.update(def_players)
aja_teammates.remove(AJA_ID)

print(f"Identified {len(aja_teammates)} potential teammates for A'ja.")

stats = {
    "on_poss": 0,
    "on_net_pts": 0,
    "off_poss": 0,
    "off_net_pts": 0
}

for _, row in df_aces_games.iterrows():
    off_players = [int(row[f'O{i}']) for i in range(1, 6)]
    def_players = [int(row[f'D{i}']) for i in range(1, 6)]
    
    is_aja_on = AJA_ID in off_players or AJA_ID in def_players
    
    # Check if this is an Aces possession (either on offense or defense)
    # We define an Aces possession if at least 3 identified teammates are on one side
    aces_on_offense = len(set(off_players) & aja_teammates) >= 3 or AJA_ID in off_players
    aces_on_defense = len(set(def_players) & aja_teammates) >= 3 or AJA_ID in def_players
    
    if is_aja_on:
        stats["on_poss"] += 1
        if AJA_ID in off_players:
            stats["on_net_pts"] += row['Net_Diff']
        else:
            stats["on_net_pts"] -= row['Net_Diff']
    else:
        # Aja is off. Is it an Aces possession?
        if aces_on_offense:
            stats["off_poss"] += 1
            stats["off_net_pts"] += row['Net_Diff']
        elif aces_on_defense:
            stats["off_poss"] += 1
            stats["off_net_pts"] -= row['Net_Diff']

on_rtg = (stats["on_net_pts"] / stats["on_poss"]) * 100 if stats["on_poss"] > 0 else 0
off_rtg = (stats["off_net_pts"] / stats["off_poss"]) * 100 if stats["off_poss"] > 0 else 0

print("\n--- Manual On/Off Calculation for A'ja Wilson (2025) ---")
print(f"On Possessions:  {stats['on_poss']}")
print(f"On Net Points:   {stats['on_net_pts']}")
print(f"On Net Rating:   {on_rtg:+.2f}")
print(f"Off Possessions: {stats['off_poss']}")
print(f"Off Net Points:  {stats['off_net_pts']}")
print(f"Off Net Rating:  {off_rtg:+.2f}")
print(f"On-Off Net:      {on_rtg - off_rtg:+.2f}")

