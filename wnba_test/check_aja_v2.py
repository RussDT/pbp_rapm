import pandas as pd
import numpy as np

AJA_ID = 1628932
ACES_TEAM_ID = 1611661319

# Get all Aces player IDs from player_index_map
map_df = pd.read_csv("wnba_test/player_index_map.csv", header=None)
# Column 2 is EntityId, Column 6 is team_id
# Let's be safer and check columns
map_df = pd.read_csv("wnba_test/player_index_map.csv")
aces_players = set(map_df[map_df['team_id'] == ACES_TEAM_ID]['EntityId'].unique())

# Load processed possessions
df = pd.read_csv("wnba_test/Processed/RAPM25.csv")

# Identify Aces possessions
# A possession is an Aces possession if Aja is on, OR if at least 3 players on one side are Aces players.
def calculate_on_off(df, player_id, team_players):
    on_poss = 0
    on_net_pts = 0
    off_poss = 0
    off_net_pts = 0
    
    for _, row in df.iterrows():
        off_players = set([int(row[f'O{i}']) for i in range(1, 6)])
        def_players = set([int(row[f'D{i}']) for i in range(1, 6)])
        
        is_player_on_offense = player_id in off_players
        is_player_on_defense = player_id in def_players
        
        # Determine if team is on offense or defense
        team_on_offense = is_player_on_offense or (len(off_players & team_players) >= 3)
        team_on_defense = is_player_on_defense or (len(def_players & team_players) >= 3)
        
        if is_player_on_offense:
            on_poss += 1
            on_net_pts += row['Net_Diff']
        elif is_player_on_defense:
            on_poss += 1
            on_net_pts -= row['Net_Diff']
        else:
            # Player is OFF
            if team_on_offense:
                off_poss += 1
                off_net_pts += row['Net_Diff']
            elif team_on_defense:
                off_poss += 1
                off_net_pts -= row['Net_Diff']
                
    return on_poss, on_net_pts, off_poss, off_net_pts

on_p, on_pts, off_p, off_pts = calculate_on_off(df, AJA_ID, aces_players)

on_rtg = (on_pts / on_p) * 100 if on_p > 0 else 0
off_rtg = (off_pts / off_p) * 100 if off_p > 0 else 0

print("\n--- A'ja Wilson On/Off (2025 Processed Data) ---")
print(f"On Possessions:  {on_p}")
print(f"On Net Points:   {on_pts}")
print(f"On Net Rating:   {on_rtg:+.2f}")
print(f"Off Possessions: {off_p}")
print(f"Off Net Points:  {off_pts}")
print(f"Off Net Rating:  {off_rtg:+.2f}")
print(f"On-Off Net:      {on_rtg - off_rtg:+.2f}")

# Also check her teammates to see why RAPM might be lower
print("\n--- Teammate On/Off ---")
for p_id in sorted(aces_players):
    if p_id == AJA_ID: continue
    # Only check if they have significant possessions
    p_name = map_df[map_df['EntityId'] == p_id]['pbp_name'].iloc[0] if p_id in map_df['EntityId'].values else f"ID_{p_id}"
    tp_on_p = df.loc[:, 'O1':'D5'].isin([p_id]).any(axis=1).sum()
    if tp_on_p > 1000:
        p_on_p, p_on_pts, _, _ = calculate_on_off(df, p_id, aces_players)
        p_rtg = (p_on_pts / p_on_p) * 100 if p_on_p > 0 else 0
        print(f"{p_name:20s}: Net {p_rtg:+.2f} ({p_on_p} poss)")

