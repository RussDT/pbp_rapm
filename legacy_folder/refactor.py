###UPDATED OPENAI CODE
import pandas as pd
import numpy as np
import re
import time
# -------------------------------
# Helper functions
# -------------------------------

# Pre-compile any frequently used regex patterns to speed repeated usage.
PAT_REBOUND = re.compile(r"REBOUND", re.IGNORECASE)
PAT_REBOUND_LC = re.compile(r"rebound", re.IGNORECASE)
PAT_FREE_THROW = re.compile(r"Free Throw", re.IGNORECASE)
PAT_MISS = re.compile(r"MISS", re.IGNORECASE)
PAT_1OF2_1OF3_2OF3_TECH_FLAG = re.compile(r"(1 of 2|1 of 3|2 of 3|Technical|Flagrant)")
PAT_PTS = re.compile(r"PTS", re.IGNORECASE)
PAT_3PT = re.compile(r"3PT", re.IGNORECASE)
PAT_FOUL = re.compile(r"Foul", re.IGNORECASE)
PAT_SFOUL = re.compile(r"S\.FOUL", re.IGNORECASE)
PAT_TRANSITION = re.compile(r"Transition", re.IGNORECASE)

def drop_empty_rows(df, cols):
    df = df.dropna(subset=cols)
    for col in cols:
        df = df[df[col] != ""]
    return df

def map_event_type(x):
    mapping = {
        1: "MAKE", 2: "MISS", 3: "FreeThrow", 4: "Rebound",
        5: "Turnover", 6: "Foul", 7: "Violation", 8: "Substitution",
        9: "Timeout", 10: "JumpBall", 11: "Ejection", 12: "StartOfPeriod",
        13: "EndOfPeriod", 14: "Empty"
    }
    return mapping.get(x, str(x))

# -------------------------------
# ProcessRAPM Function
# -------------------------------
def process_rapm(file_name, year, season):
    start_time = time.time()  # Start timing the function
    
    df = pd.read_csv(file_name)
    df.columns = df.columns.str.lower()
    
    # Drop rows with missing players
    player_cols = [
        'away_player1','away_player2','away_player3','away_player4','away_player5',
        'home_player1','home_player2','home_player3','home_player4','home_player5'
    ]
    df = drop_empty_rows(df, player_cols)
    
    # Rename eventmsgtype -> event_type and map to string
    if 'eventmsgtype' in df.columns:
        df.rename(columns={'eventmsgtype': 'event_type'}, inplace=True)
    df['event_type'] = df['event_type'].apply(map_event_type)
    
    # Fill missing object columns with ""
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna("")
    
    print("Completed initial data processing")
    
    # Home_Action_Score / Away_Action_Score
    def compute_home_action(row):
        desc = row['homedescription']
        if PAT_FREE_THROW.search(desc) and not PAT_MISS.search(desc):
            return 1
        elif PAT_PTS.search(desc) and not PAT_3PT.search(desc) and row['event_type'] == "MAKE":
            return 2
        elif PAT_3PT.search(desc) and not PAT_MISS.search(desc):
            return 3
        return 0
    
    def compute_away_action(row):
        desc = row['visitordescription']
        if PAT_FREE_THROW.search(desc) and not PAT_MISS.search(desc):
            return 1
        elif PAT_PTS.search(desc) and not PAT_3PT.search(desc) and row['event_type'] == "MAKE":
            return 2
        elif PAT_3PT.search(desc) and not PAT_MISS.search(desc):
            return 3
        return 0
    
    df['Home_Action_Score'] = df.apply(compute_home_action, axis=1)
    df['Away_Action_Score'] = df.apply(compute_away_action, axis=1)
    
    print("Computed action scores")
    
    # Cumulative scores
    df['Net_Home'] = df['Home_Action_Score'].cumsum()
    df['Net_Away'] = df['Away_Action_Score'].cumsum()
    df['Net_Total'] = df['Net_Home'] + df['Net_Away']
    
    print("Computed cumulative scores")
    
    # Convert time to seconds
    def convert_time_to_seconds(time_str):
        if isinstance(time_str, int):
            return time_str
        minutes, seconds = map(int, time_str.split(':'))
        return 60 * minutes + seconds
    
    df['seconds_remaining_quarter'] = df['pctimestring'].apply(convert_time_to_seconds)
    df['prev_seconds'] = df['seconds_remaining_quarter'].shift(1)
    
    print("Converted time to seconds")
    
    # Create lag/lead columns
    df['Prev_visitor_desc']  = df['visitordescription'].shift(1, fill_value="")
    df['Prev_visitor_desc2'] = df['visitordescription'].shift(2, fill_value="")
    df['Prev_home_desc']     = df['homedescription'].shift(1, fill_value="")
    df['Prev_home_desc2']    = df['homedescription'].shift(2, fill_value="")
    df['Next_home_desc']     = df['homedescription'].shift(-1, fill_value="")
    df['Next_visitor_desc']  = df['visitordescription'].shift(-1, fill_value="")
    
    print("Created lag/lead columns")
    
    # Offensive FT Rebound
    def compute_offensive_ft_rebound(row):
        cond_away = (
            (PAT_REBOUND.search(row['visitordescription']) or PAT_REBOUND_LC.search(row['visitordescription'])) and
            PAT_MISS.search(row['Prev_visitor_desc']) and
            PAT_FREE_THROW.search(row['Prev_visitor_desc']) and
            not PAT_1OF2_1OF3_2OF3_TECH_FLAG.search(row['Prev_visitor_desc'])
        )
        cond_home = (
            (PAT_REBOUND.search(row['homedescription']) or PAT_REBOUND_LC.search(row['homedescription'])) and
            PAT_MISS.search(row['Prev_home_desc']) and
            PAT_FREE_THROW.search(row['Prev_home_desc']) and
            not PAT_1OF2_1OF3_2OF3_TECH_FLAG.search(row['Prev_home_desc'])
        )
        return cond_away or cond_home
    
    df['offensive_FT_Rebound'] = df.apply(compute_offensive_ft_rebound, axis=1)
    
    print("Computed offensive FT rebounds")
    
    # End_of_Possession
    def compute_end_of_possession(row):
        cond1 = ("EndOfPeriod" in row['event_type']) and (row['prev_seconds'] > 0 if pd.notnull(row['prev_seconds']) else False)
        cond2 = ("Flagrant 2 of 2" in row['homedescription']) or ("Flagrant 3 of 3" in row['homedescription'])
        cond3 = ("Flagrant 2 of 2" in row['visitordescription']) or ("Flagrant 3 of 3" in row['visitordescription'])
        cond4 = ("Turnover" in row['event_type'])
        cond5 = (
            PAT_REBOUND.search(row['homedescription']) and 
            PAT_MISS.search(row['Prev_visitor_desc']) and 
            not PAT_1OF2_1OF3_2OF3_TECH_FLAG.search(row['Prev_visitor_desc'])
        )
        cond6 = (
            PAT_REBOUND_LC.search(row['homedescription']) and 
            PAT_MISS.search(row['Prev_visitor_desc']) and 
            not PAT_1OF2_1OF3_2OF3_TECH_FLAG.search(row['Prev_visitor_desc'])
        )
        cond7 = (
            PAT_REBOUND.search(row['visitordescription']) and 
            PAT_MISS.search(row['Prev_home_desc']) and 
            not PAT_1OF2_1OF3_2OF3_TECH_FLAG.search(row['Prev_home_desc'])
        )
        cond8 = (
            PAT_REBOUND_LC.search(row['visitordescription']) and 
            PAT_MISS.search(row['Prev_home_desc']) and 
            not PAT_1OF2_1OF3_2OF3_TECH_FLAG.search(row['Prev_home_desc'])
        )
        cond9 = (
            PAT_PTS.search(row['homedescription']) and 
            row['event_type'] == "MAKE" and 
            not PAT_SFOUL.search(row['Next_visitor_desc'])
        )
        cond10 = (
            PAT_PTS.search(row['visitordescription']) and 
            row['event_type'] == "MAKE" and 
            not PAT_SFOUL.search(row['Next_home_desc'])
        )
        cond11 = (
            (
                "1 of 1" in row['homedescription'] or 
                "2 of 2" in row['homedescription'] or 
                "3 of 3" in row['homedescription']
            ) and 
            PAT_PTS.search(row['homedescription']) and 
            not (
                PAT_TRANSITION.search(row['Prev_visitor_desc2']) or 
                PAT_TRANSITION.search(row['Prev_visitor_desc']) or 
                "Flagrant" in row['homedescription']
            )
        )
        cond12 = (
            (
                "1 of 1" in row['visitordescription'] or 
                "2 of 2" in row['visitordescription'] or 
                "3 of 3" in row['visitordescription']
            ) and 
            PAT_PTS.search(row['visitordescription']) and 
            not (
                PAT_TRANSITION.search(row['Prev_home_desc2']) or 
                PAT_TRANSITION.search(row['Prev_home_desc']) or 
                "Flagrant" in row['visitordescription']
            )
        )
        return (cond1 or cond2 or cond3 or cond4 or cond5 or cond6 or cond7 or cond8 or cond9 or cond10 or cond11 or cond12)
    
    df['End_of_Possession'] = df.apply(compute_end_of_possession, axis=1)
    
    print("Computed end of possession")
    
    # TeamOnOffense
    def compute_team_on_offense(row):
        # Merge all logic into single checks
        # Note: Checking "Turnover" in homedescription or visitordescription
        # can be done if the raw data is consistent. This is the same logic as original, just shorter.
        if ("Flagrant 2 of 2" in row['homedescription']) or ("Flagrant 3 of 3" in row['homedescription']):
            return "Home"
        elif ("Flagrant 2 of 2" in row['visitordescription']) or ("Flagrant 3 of 3" in row['visitordescription']):
            return "Away"
        elif (PAT_REBOUND.search(row['homedescription']) or PAT_REBOUND_LC.search(row['homedescription'])) and PAT_MISS.search(row['Prev_visitor_desc']):
            return "Away"
        elif (PAT_REBOUND.search(row['visitordescription']) or PAT_REBOUND_LC.search(row['visitordescription'])) and PAT_MISS.search(row['Prev_home_desc']):
            return "Home"
        elif "Turnover" in row['homedescription']:
            return "Home"
        elif "Turnover" in row['visitordescription']:
            return "Away"
        elif PAT_PTS.search(row['homedescription']) and (row['event_type']=="MAKE") and not PAT_SFOUL.search(row['Next_visitor_desc']):
            return "Home"
        elif PAT_PTS.search(row['visitordescription']) and (row['event_type']=="MAKE") and not PAT_SFOUL.search(row['Next_home_desc']):
            return "Away"
        elif (
            (
                "1 of 1" in row['homedescription'] or 
                "2 of 2" in row['homedescription'] or 
                "3 of 3" in row['homedescription']
            ) and 
            PAT_PTS.search(row['homedescription']) and 
            not (
                PAT_TRANSITION.search(row['Prev_visitor_desc2']) or 
                PAT_TRANSITION.search(row['Prev_visitor_desc']) or 
                "Flagrant" in row['homedescription']
            )
        ):
            return "Home"
        elif (
            (
                "1 of 1" in row['visitordescription'] or 
                "2 of 2" in row['visitordescription'] or 
                "3 of 3" in row['visitordescription']
            ) and 
            PAT_PTS.search(row['visitordescription']) and 
            not (
                PAT_TRANSITION.search(row['Prev_home_desc2']) or 
                PAT_TRANSITION.search(row['Prev_home_desc']) or 
                "Flagrant" in row['visitordescription']
            )
        ):
            return "Away"
        return ""
    
    df['TeamOnOffense'] = df.apply(compute_team_on_offense, axis=1)
    
    print("Computed team on offense")
    
    # Rolling counts
    df['FTEvents'] = df['event_type'].eq("FreeThrow").rolling(window=5, min_periods=1).sum()
    df['SubEvents'] = df['event_type'].eq("Substitution").rolling(window=5, min_periods=1).sum()
    
    def compute_potential_foul(row):
        if (row['FTEvents'] != 0) and (row['SubEvents'] != 0) and ("Foul" in row['event_type']):
            # Exclude T.FOUL, FLAGRANT, Offens, Transition
            if (
                "T.FOUL" not in row['visitordescription'] and
                "FLAGRANT" not in row['visitordescription'] and
                "Offens" not in row['visitordescription'] and
                "Transition" not in row['visitordescription'] and
                "T.FOUL" not in row['homedescription'] and
                "FLAGRANT" not in row['homedescription'] and
                "Transition" not in row['homedescription']
            ):
                return True
        return False
    
    df['PotentialFoul'] = df.apply(compute_potential_foul, axis=1)
    print("Total Potential Fouls:", df['PotentialFoul'].sum())
    
    print("Computed potential fouls")
    
    # Copy original lineup columns
    df['a1'] = df['away_player1']
    df['a2'] = df['away_player2']
    df['a3'] = df['away_player3']
    df['a4'] = df['away_player4']
    df['a5'] = df['away_player5']
    df['h1'] = df['home_player1']
    df['h2'] = df['home_player2']
    df['h3'] = df['home_player3']
    df['h4'] = df['home_player4']
    df['h5'] = df['home_player5']
    
    print("Copied original lineup columns")
    
    # Gather foul rows
    foul_rows = list(df.index[df['PotentialFoul'] == True])
    # Make sure last row is included
    if df.index[-1] not in foul_rows:
        foul_rows.append(df.index[-1])

    # Single-pass propagation to avoid nested loops:
    # We walk through foul_rows one by one but track forward in a single sweep.
    n = len(df)
    i = 0
    while i < len(foul_rows):
        foul_idx = foul_rows[i]
        if df.at[foul_idx, 'PotentialFoul']:
            # Store the lineup
            a_vals = df.loc[foul_idx, ['a1','a2','a3','a4','a5']].values
            h_vals = df.loc[foul_idx, ['h1','h2','h3','h4','h5']].values
            found_free_throw = False
            j = foul_idx + 1
            while j < n:
                # If next row is a new foul, break early (we'll handle it in next iteration)
                if df.at[j, 'PotentialFoul']:
                    break
                # If we see a FreeThrow, mark that we've found one
                if df.at[j, 'event_type'] == "FreeThrow":
                    found_free_throw = True
                # If end_of_possession or offensive_FT_Rebound, fill the chunk if found_free_throw
                if df.at[j, 'End_of_Possession'] or df.at[j, 'offensive_FT_Rebound']:
                    if found_free_throw:
                        # Fill from foul_idx+1 to j (or j-1 if it's an FT rebound) 
                        fill_end = j
                        if df.at[j, 'offensive_FT_Rebound']:
                            fill_end = j - 1
                        # Assign the lineups
                        if fill_end >= foul_idx + 1:
                            df.loc[foul_idx+1:fill_end, ['a1','a2','a3','a4','a5']] = a_vals
                            df.loc[foul_idx+1:fill_end, ['h1','h2','h3','h4','h5']] = h_vals
                    break
                j += 1
            i += 1
        else:
            i += 1
    
    print("Completed single-pass propagation")
    
    # Offensive FT Rebound check to mark possession end if lineups differ:
    # We do a single pass over the foul rows again
    i = 0
    while i < len(foul_rows):
        idx = foul_rows[i]
        if df.at[idx, 'PotentialFoul']:
            old_vals = df.loc[idx, ['a1','a2','a3','a4','a5','h1','h2','h3','h4','h5']]
            j = idx + 1
            while j < n:
                if df.at[j, 'offensive_FT_Rebound']:
                    new_vals = df.loc[j, ['a1','a2','a3','a4','a5','h1','h2','h3','h4','h5']]
                    if not new_vals.equals(old_vals):
                        # Mark the prior row as End_of_Possession
                        df.at[j-1, 'End_of_Possession'] = True
                    break
                if df.at[j, 'End_of_Possession'] or df.at[j, 'PotentialFoul']:
                    break
                j += 1
        i += 1
    
    print("Completed offensive FT rebound check")
    
    # Compute offense/defense columns
    df['O1'] = np.where(df['TeamOnOffense'] == "Away", df['a1'],
                        np.where(df['TeamOnOffense'] == "Home", df['h1'], 0))
    df['O2'] = np.where(df['TeamOnOffense'] == "Away", df['a2'],
                        np.where(df['TeamOnOffense'] == "Home", df['h2'], 0))
    df['O3'] = np.where(df['TeamOnOffense'] == "Away", df['a3'],
                        np.where(df['TeamOnOffense'] == "Home", df['h3'], 0))
    df['O4'] = np.where(df['TeamOnOffense'] == "Away", df['a4'],
                        np.where(df['TeamOnOffense'] == "Home", df['h4'], 0))
    df['O5'] = np.where(df['TeamOnOffense'] == "Away", df['a5'],
                        np.where(df['TeamOnOffense'] == "Home", df['h5'], 0))
    df['D1'] = np.where(df['TeamOnOffense'] == "Away", df['h1'],
                        np.where(df['TeamOnOffense'] == "Home", df['a1'], 0))
    df['D2'] = np.where(df['TeamOnOffense'] == "Away", df['h2'],
                        np.where(df['TeamOnOffense'] == "Home", df['a2'], 0))
    df['D3'] = np.where(df['TeamOnOffense'] == "Away", df['h3'],
                        np.where(df['TeamOnOffense'] == "Home", df['a3'], 0))
    df['D4'] = np.where(df['TeamOnOffense'] == "Away", df['h4'],
                        np.where(df['TeamOnOffense'] == "Home", df['a4'], 0))
    df['D5'] = np.where(df['TeamOnOffense'] == "Away", df['h5'],
                        np.where(df['TeamOnOffense'] == "Home", df['a5'], 0))
    
    print("Computed offense/defense columns")
    
    # Merge with external stats if available
    try:
        data_stats = pd.read_csv("pbp_totals_25.csv")
    except:
        data_stats = pd.DataFrame(columns=["nba_id","FtPoints","FTA","F3GA","FG3M"])
    
    # Ensure numeric conversion and handle NaN values
    for col in ["nba_id","FTPerc","ThreePerc","FTA","ThreePA"]:
        if col in data_stats.columns:
            data_stats[col] = pd.to_numeric(data_stats[col], errors='coerce')
    
    # Fill NaN values with 0
    data_stats.fillna(0, inplace=True)
    
    # Select the required columns
    df_selected = data_stats[["nba_id","FtPoints","FTA","FG3A","FG3M"]] if not data_stats.empty else None
    
    if df_selected is not None:
        df = df.merge(df_selected, left_on="player1_id", right_on="nba_id", how="left")
        df["Exp3PT"] = df["FG3M"] / df['FG3A']
        df["ExpFT"] = df["FtPoints"] / df['FTA']
        df.drop(columns=["FtPoints","FTA","FG3A","FG3M"], errors='ignore', inplace=True)
    else:
        df["Exp3PT"] = 0
        df["ExpFT"] = 0
    
    print("Merged with external stats")
    
    # OffNet, DefNet
    def compute_offnet(row):
        if PAT_FREE_THROW.search(row['homedescription']) or PAT_FREE_THROW.search(row['visitordescription']) or ("FreeThrow" in row['event_type']):
            return row["ExpFT"]
        elif (
            (PAT_PTS.search(row['homedescription']) and not PAT_3PT.search(row['homedescription']) and row['event_type']=="MAKE") or
            (PAT_PTS.search(row['visitordescription']) and not PAT_3PT.search(row['visitordescription']) and row['event_type']=="MAKE")
        ):
            return 2
        elif PAT_3PT.search(row['homedescription']) or PAT_3PT.search(row['visitordescription']):
            # Weighted blend example from original code
            return (row["Exp3PT"] * 3) * 0.2 + 0.8 * (row['Home_Action_Score'] + row['Away_Action_Score'])
        return 0
    
    def compute_defnet(row):
        if PAT_FREE_THROW.search(row['homedescription']) or PAT_FREE_THROW.search(row['visitordescription']) or ("FreeThrow" in row['event_type']):
            return row["ExpFT"]
        elif (
            (PAT_PTS.search(row['homedescription']) and not PAT_3PT.search(row['homedescription']) and row['event_type']=="MAKE") or
            (PAT_PTS.search(row['visitordescription']) and not PAT_3PT.search(row['visitordescription']) and row['event_type']=="MAKE")
        ):
            return 2
        elif PAT_3PT.search(row['homedescription']) or PAT_3PT.search(row['visitordescription']):
            return (row["Exp3PT"] * 3) * 0.4 + 0.6 * (row['Home_Action_Score'] + row['Away_Action_Score'])
        return 0
    
    df["OffNet"] = df.apply(compute_offnet, axis=1)
    df["DefNet"] = df.apply(compute_defnet, axis=1)
    df["OffTotal"] = df["OffNet"].cumsum()
    df["DefTotal"] = df["DefNet"].cumsum()
    
    print("Computed OffNet and DefNet")
    
    # Load player dictionary if exists
    try:
        Player24 = pd.read_csv("csvs/autocomplete_map.csv")
    except:
        Player24 = pd.DataFrame(columns=["nba_id","player_name"])
    
    # Just an example subset
    Dict24 = Player24[["nba_id","player_name"]]
    
    print("Loaded player dictionary")
    
    # Filter and final columns
    NBA24Filt = df[df["End_of_Possession"] == True].copy()
    NBA24Filt["Net_Diff"] = NBA24Filt["Net_Total"] - NBA24Filt["Net_Total"].shift(1, fill_value=0)
    NBA24Filt["Off_Diff"] = NBA24Filt["OffTotal"] - NBA24Filt["OffTotal"].shift(1, fill_value=0)
    NBA24Filt["Def_Diff"] = NBA24Filt["DefTotal"] - NBA24Filt["DefTotal"].shift(1, fill_value=0)
    
    # Add Season
    NBA24Filt["Season"] = year + 1
    
    end_time = time.time()  # End timing the function
    print(f"process_rapm took {end_time - start_time:.2f} seconds")
    
    return NBA24Filt

# -------------------------------
# ProcessTOs, ProcessRebounding, ProcessTS
# -------------------------------
def process_tos(file_name, year):
    df = pd.read_csv(file_name)
    df.columns = df.columns.str.lower()
    df = drop_empty_rows(df, [
        'away_player1','away_player2','away_player3','away_player4','away_player5',
        'home_player1','home_player2','home_player3','home_player4','home_player5'
    ])
    df['event_type'] = df['event_type'].apply(map_event_type)
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna("")
    # ...similar logic for turnovers...
    return df

def process_rebounding(file_name, year):
    df = pd.read_csv(file_name)
    df.columns = df.columns.str.lower()
    df = drop_empty_rows(df, [
        'away_player1','away_player2','away_player3','away_player4','away_player5',
        'home_player1','home_player2','home_player3','home_player4','home_player5'
    ])
    df['event_type'] = df['event_type'].apply(map_event_type)
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna("")
    # ...similar logic for rebounding...
    return df

def process_ts(file_name, year):
    df = pd.read_csv(file_name)
    df.columns = df.columns.str.lower()
    df = drop_empty_rows(df, [
        'away_player1','away_player2','away_player3','away_player4','away_player5',
        'home_player1','home_player2','home_player3','home_player4','home_player5'
    ])
    df['event_type'] = df['event_type'].apply(map_event_type)
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna("")
    # ...similar logic for TS...
    return df

# -------------------------------
# Modify DataFrame Function
# -------------------------------
def modify_dataframe(df):
    df = df.iloc[:, 1:]
    df['possessions'] = 1
    return df