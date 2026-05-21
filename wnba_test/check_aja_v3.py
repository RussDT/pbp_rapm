import pandas as pd
import numpy as np

AJA_ID = 1628932
ACES_TEAM_ID = 1611661319

map_df = pd.read_csv("wnba_test/player_index_map.csv")
aces_players = set(map_df[map_df['team_id'] == ACES_TEAM_ID]['EntityId'].unique())

df = pd.read_csv("wnba_test/Processed/RAPM25.csv")

def calculate_rtgs(df, player_id, team_players):
    # Separate offensive and defensive possessions when player is ON
    on_off_poss = 0      # Possessions where player is on offense
    on_def_poss = 0      # Possessions where player is on defense
    on_pts_scored = 0    # Points scored when player is on offense
    on_pts_allowed = 0   # Points allowed when player is on defense
    
    # Separate offensive and defensive possessions when player is OFF
    off_off_poss = 0     # Team offensive possessions when player is off
    off_def_poss = 0     # Team defensive possessions when player is off
    off_pts_scored = 0   # Points scored when player is off
    off_pts_allowed = 0  # Points allowed when player is off
    
    for _, row in df.iterrows():
        off_players = set([int(row[f'O{i}']) for i in range(1, 6)])
        def_players = set([int(row[f'D{i}']) for i in range(1, 6)])
        
        is_player_on_off = player_id in off_players
        is_player_on_def = player_id in def_players
        
        # Determine if team is on offense or defense
        team_on_off = is_player_on_off or (len(off_players & team_players) >= 3)
        team_on_def = is_player_on_def or (len(def_players & team_players) >= 3)
        
        if is_player_on_off:
            on_off_poss += 1
            on_pts_scored += row['Net_Diff']
        elif is_player_on_def:
            on_def_poss += 1
            on_pts_allowed += row['Net_Diff']
        else:
            # Player is OFF
            if team_on_off:
                off_off_poss += 1
                off_pts_scored += row['Net_Diff']
            elif team_on_def:
                off_def_poss += 1
                off_pts_allowed += row['Net_Diff']
                
    return on_off_poss, on_def_poss, on_pts_scored, on_pts_allowed, off_off_poss, off_def_poss, off_pts_scored, off_pts_allowed

on_off_poss, on_def_poss, on_pts_scored, on_pts_allowed, off_off_poss, off_def_poss, off_pts_scored, off_pts_allowed = calculate_rtgs(df, AJA_ID, aces_players)

# Calculate ratings using SEPARATE offensive and defensive possessions
# ORtg = points scored / offensive possessions * 100
# DRtg = points allowed / defensive possessions * 100
on_off_rtg = (on_pts_scored / on_off_poss) * 100 if on_off_poss > 0 else 0
on_def_rtg = (on_pts_allowed / on_def_poss) * 100 if on_def_poss > 0 else 0
on_net = on_off_rtg - on_def_rtg

off_off_rtg = (off_pts_scored / off_off_poss) * 100 if off_off_poss > 0 else 0
off_def_rtg = (off_pts_allowed / off_def_poss) * 100 if off_def_poss > 0 else 0
off_net = off_off_rtg - off_def_rtg

print("\n--- A'ja Wilson Detailed On/Off (2025) ---")
print(f"ON  Offense: {on_off_poss} poss, {on_pts_scored:.0f} pts -> ORtg {on_off_rtg:.2f}")
print(f"ON  Defense: {on_def_poss} poss, {on_pts_allowed:.0f} pts -> DRtg {on_def_rtg:.2f}")
print(f"ON  Net: {on_net:+.2f}")
print(f"OFF Offense: {off_off_poss} poss, {off_pts_scored:.0f} pts -> ORtg {off_off_rtg:.2f}")
print(f"OFF Defense: {off_def_poss} poss, {off_pts_allowed:.0f} pts -> DRtg {off_def_rtg:.2f}")
print(f"OFF Net: {off_net:+.2f}")
print(f"\nON-OFF: ORtg {on_off_rtg-off_off_rtg:+.2f}, DRtg {on_def_rtg-off_def_rtg:+.2f}, Net {on_net-off_net:+.2f}")

