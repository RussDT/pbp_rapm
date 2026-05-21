import pandas as pd
import numpy as np
import re
import time
import warnings
import requests
import io
import argparse
import os
from nba_api.stats.endpoints import leaguedashplayerstats



# Ignore SettingWithCopyWarning, common in this type of step-by-step processing
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# Helper function to mimic R's grepl with ignore.case=TRUE/FALSE
def series_contains(series, pattern, case=True, na=False, regex=True):
    """Checks if a string pattern is contained within Series elements."""
    if not pd.api.types.is_string_dtype(series):
       series = series.astype(str) # Ensure Series is string type
    # Fill NA to avoid errors during str ops, consistent with R's na.rm=T implicit behavior sometimes
    return series.fillna('').str.contains(pattern, case=case, na=na, regex=regex)

# Helper function to mimic R's case_when using numpy.select
def case_when(*args):
    """Mimics R's dplyr::case_when using numpy.select.
    Expects pairs of condition (boolean Series) and result,
    with the last argument being the default result.
    """
    conditions = [args[i] for i in range(0, len(args) - 1, 2)]
    results = [args[i] for i in range(1, len(args) - 1, 2)]
    default = args[-1]
    # Ensure conditions are boolean arrays/Series
    conditions = [pd.Series(c) if not isinstance(c, pd.Series) else c for c in conditions]
    return np.select(conditions, results, default=default)

# --- Player Propagation and FT Check Functions (Implemented based on R code) ---

def propagate_player_values_py(df):
    """
    Propagates player values (a1-h5) after a PotentialFoul if FTs occur
    before the next EOP or offensive FT rebound. Mimics R's propagate_player_values.
    """
    print("      Running Player Propagation...")
    start_time = time.time()
    df_copy = df.copy() # Work on a copy to avoid modifying original during iteration

    player_cols = [f"{team}{i}" for team in ["a", "h"] for i in range(1, 6)]

    # Get indices where PotentialFoul is True
    # Ensure PotentialFoul exists and is boolean
    if 'PotentialFoul' not in df_copy.columns or not pd.api.types.is_bool_dtype(df_copy['PotentialFoul']):
         print("      Warning: 'PotentialFoul' column missing or not boolean. Skipping propagation.")
         return df # Return original df if column is missing/wrong type

    foul_indices = df_copy.index[df_copy['PotentialFoul']].tolist()
    if not foul_indices:
        print("      No PotentialFoul rows found. Skipping propagation.")
        return df # Return original if no fouls

    num_propagations = 0
    last_index = df_copy.index[-1]

    for i in foul_indices:
        # Store player values from the foul row
        player_values_at_foul = df_copy.loc[i, player_cols].to_dict()

        found_free_throw = False
        # Loop from the row *after* the foul index to the end
        # Use .loc for robust index slicing, even if indices are non-sequential
        start_loc = df_copy.index.get_loc(i) + 1
        end_loc = len(df_copy) # Go up to the end

        for current_loc in range(start_loc, end_loc):
            current_index = df_copy.index[current_loc]
            row_data = df_copy.loc[current_index]

            # R code breaks if another Foul is encountered (seems unnecessary here?)
            # if row_data.get('event_type', '') == "Foul":
            #    break

            # Check if a free throw occurred
            if row_data.get('event_type', '') == "FreeThrow":
                found_free_throw = True

            # Check for EOP or offensive FT rebound (termination conditions)
            # Ensure columns exist and handle potential NaN comparison issues
            is_eop = row_data.get('End_of_Possession', False)
            is_off_ft_reb = row_data.get('offensive_FT_Rebound', False)

            if is_eop or is_off_ft_reb:
                if found_free_throw:
                    # Propagate values from foul_row+1 up to *this* termination row (inclusive for EOP, exclusive for off_FT_reb in R?)
                    # R code seems to propagate up to j for EOP, j-1 for off_FT_reb
                    end_prop_loc = current_loc if is_eop else current_loc -1 # Index location before the rebound
                    if end_prop_loc >= start_loc: # Check if there's a range to propagate
                        prop_indices = df_copy.index[start_loc : end_prop_loc + 1]
                        if not prop_indices.empty:
                            # Use .loc for assignment
                            df_copy.loc[prop_indices, player_cols] = pd.DataFrame([player_values_at_foul] * len(prop_indices), index=prop_indices)
                            num_propagations += len(prop_indices)
                break # Stop searching for this foul event

    end_time = time.time()
    print(f"      Finished Player Propagation. Propagated values for {num_propagations} rows. Time: {end_time - start_time:.2f}s")
    return df_copy


def ft_off_check_py(df):
    """
    Checks for offensive FT rebounds after PotentialFouls where player lineups change.
    If found, marks the preceding row as End_of_Possession. Mimics R's FTOffCheck.
    """
    print("      Running FT Off Check...")
    start_time = time.time()
    df_copy = df.copy() # Work on a copy

    player_cols = [f"{team}{i}" for team in ["a", "h"] for i in range(1, 6)]

    # Get indices where PotentialFoul is True
    if 'PotentialFoul' not in df_copy.columns or not pd.api.types.is_bool_dtype(df_copy['PotentialFoul']):
         print("      Warning: 'PotentialFoul' column missing or not boolean. Skipping FT Off Check.")
         return df # Return original

    foul_indices = df_copy.index[df_copy['PotentialFoul']].tolist()
    if not foul_indices:
        print("      No PotentialFoul rows found. Skipping FT Off Check.")
        return df # Return original

    eop_corrections = 0
    last_index = df_copy.index[-1]

    for i in foul_indices:
        # Get player values at the foul index (as Series for comparison)
        values_at_foul = df_copy.loc[i, player_cols]

        # Loop from the row *after* the foul index to the end
        start_loc = df_copy.index.get_loc(i) + 1
        end_loc = len(df_copy)

        for current_loc in range(start_loc, end_loc):
            current_index = df_copy.index[current_loc]
            row_data = df_copy.loc[current_index]

            # Ensure columns exist and handle potential NaN
            is_off_ft_reb = row_data.get('offensive_FT_Rebound', False)
            is_eop = row_data.get('End_of_Possession', False)

            if is_off_ft_reb:
                values_at_reb = row_data[player_cols]
                # Check if *any* player is different (R uses `any(df[j, cols] != values)`)
                # Convert to strings for robust comparison, handle NaNs
                if not values_at_foul.astype(str).equals(values_at_reb.astype(str)):
                    # Mark the row *before* the rebound as EOP=True
                    if current_loc > 0: # Ensure there is a previous row
                       prev_index = df_copy.index[current_loc - 1]
                       # Only update if it's not already EOP
                       if not df_copy.loc[prev_index, 'End_of_Possession']:
                            df_copy.loc[prev_index, 'End_of_Possession'] = True
                            eop_corrections += 1
                    break # Stop searching for this foul event after correction

            # Check for the next End_of_Possession row (termination condition)
            if is_eop:
                break # Stop searching for this foul event

    end_time = time.time()
    print(f"      Finished FT Off Check. Made {eop_corrections} EOP corrections. Time: {end_time - start_time:.2f}s")
    return df_copy

# --- End Implemented Functions ---


# Global variable to cache game dates DF (optional, improves performance if processing multiple files)
_game_dates_cache = None

def load_game_dates(url="https://raw.githubusercontent.com/gabriel1200/shot_data/refs/heads/master/game_dates.csv"):
    """
    Loads and prepares the game dates data from the specified URL.
    Handles caching, type conversion, and deduplication.
    """
    global _game_dates_cache
    if _game_dates_cache is not None:
        # print("        Using cached game dates data.") # Uncomment for debug
        return _game_dates_cache.copy() # Return a copy to prevent modification of cache

    print(f"        Loading game dates data from {url}...")
    try:
        response = requests.get(url, timeout=20) # Added timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        csv_content = io.StringIO(response.text) # Read text content into string buffer
        game_dates_df = pd.read_csv(csv_content)
        # R Schedule date seems to be YYYY-MM-DD string, let's try that format
        # game_dates_df['date'] = pd.to_datetime(game_dates_df['date'].astype(str), format='%Y%m%d').dt.date.astype(str)
        game_dates_df['date'] = pd.to_datetime(game_dates_df['date'].astype(str), format='%Y%m%d', errors='coerce').dt.strftime('%Y-%m-%d')


        game_dates_df['season']=game_dates_df['season'].str.split('-').str[0]
        game_dates_df['season']=game_dates_df['season'].astype(int)
        print(f"        Successfully loaded {len(game_dates_df)} rows from game dates CSV.")

        # --- Preprocessing ---
        # Ensure GAME_ID is numeric, drop rows where conversion fails
        game_dates_df['GAME_ID'] = pd.to_numeric(game_dates_df['GAME_ID'], errors='coerce')
        game_dates_df = game_dates_df.dropna(subset=['GAME_ID'])
        game_dates_df['GAME_ID'] = game_dates_df['GAME_ID'].astype(int)

        # Ensure 'date' is a string (or valid object), drop rows where conversion failed
        game_dates_df = game_dates_df.dropna(subset=['date'])

        # Ensure 'season' is numeric (assuming it's the starting year like 2024)
        game_dates_df['season'] = pd.to_numeric(game_dates_df['season'], errors='coerce')
        game_dates_df = game_dates_df.dropna(subset=['season'])
        game_dates_df['season'] = game_dates_df['season'].astype(int)


        # Keep only necessary columns and remove duplicates per game
        # We need GAME_ID, date, and season (for filtering)
        game_dates_df = game_dates_df[['GAME_ID', 'date', 'season']].drop_duplicates(subset=['GAME_ID'], keep='first')
        game_dates_df = game_dates_df.rename(columns={'date': 'game_date'}) # Rename for merge


        print(f"        Processed game dates data: {len(game_dates_df)} unique games found.")
        _game_dates_cache = game_dates_df # Cache the processed DataFrame
        return game_dates_df.copy() # Return a copy

    except requests.exceptions.RequestException as e:
        print(f"        Error loading game dates CSV from URL: {e}")
        return None
    except Exception as e:
        print(f"        Error processing game dates CSV: {e}")
        return None

# --- Base Processing Function ---
def _base_processing(file_path, allow_partial_lineups=False):
    # ... (initial loading, cleaning, event_type mapping, fillna) ...
    print(f"    Running Base Processing for {file_path}...")
    try:
        # Explicitly define 'NA' as a value to be treated as NaN
        nba_df = pd.read_csv(file_path, low_memory=False, na_values=['NA'])
        print(f"      Read {len(nba_df)} rows from {file_path}")
    except FileNotFoundError:
        print(f"      Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"      Error reading {file_path}: {e}")
        return None

    # --- Ensure seconds_remaining_quarter is numeric ---
    if 'seconds_remaining_quarter' in nba_df.columns:
        nba_df['seconds_remaining_quarter'] = pd.to_numeric(nba_df['seconds_remaining_quarter'], errors='coerce')
        print("      Ensured 'seconds_remaining_quarter' is numeric.")
    else:
        print("      Warning: 'seconds_remaining_quarter' column not found for numeric conversion.")

    # --- Initial Cleaning ---
    player_cols = [f"{team}_player{i}" for team in ["away", "home"] for i in range(1, 6)]
    missing_p_cols = [p_col for p_col in player_cols if p_col not in nba_df.columns]
    if missing_p_cols:
        print(f"      Error: Missing required player columns: {missing_p_cols}")
        return None
    if allow_partial_lineups:
        away_cols = [f"away_player{i}" for i in range(1, 6)]
        home_cols = [f"home_player{i}" for i in range(1, 6)]
        away_known = nba_df[away_cols].replace('', np.nan).notna().sum(axis=1)
        home_known = nba_df[home_cols].replace('', np.nan).notna().sum(axis=1)
        initial_rows = len(nba_df)
        nba_df = nba_df[(away_known >= 4) & (home_known >= 4)].copy()
        nba_df[player_cols] = nba_df[player_cols].replace('', np.nan).fillna(0)
        nba_df['allow_partial_lineups'] = True
        print(
            "      Partial-lineup mode enabled; kept raw rows with at least 4 known home and away players "
            f"({len(nba_df)} rows, removed {initial_rows - len(nba_df)})."
        )
    else:
        nba_df = nba_df.dropna(subset=player_cols)
        nba_df = nba_df[~(nba_df[player_cols].astype(str).apply(lambda x: x.str.strip() == '')).any(axis=1)]
    print(f"      Rows after player filtering: {len(nba_df)}")
    if len(nba_df) == 0:
        print("      Error: No rows remaining after player filtering.")
        return None

    # --- BEGIN: Initialize and Forward fill score columns ---
    score_cols_to_fill = ['score', 'away_score', 'home_score']
    print(f"      Attempting to initialize and forward fill for columns: {score_cols_to_fill}")

    if 'game_id' in nba_df.columns:
        # Ensure 'away_score' and 'home_score' are numeric, converting non-parsable to NaN
        if 'away_score' in nba_df.columns:
            nba_df['away_score'] = pd.to_numeric(nba_df['away_score'], errors='coerce')
        if 'home_score' in nba_df.columns:
            nba_df['home_score'] = pd.to_numeric(nba_df['home_score'], errors='coerce')
        
        # The 'score' column can remain as object/string type
        # If 'NA' strings were not converted to np.nan by read_csv (na_values did its job),
        # this step might be needed for .isna() to work on original NAs for the 'score' column.
        # However, with na_values=['NA'] in read_csv, original 'NA's become np.nan.
        # if 'score' in nba_df.columns:
        #     nba_df['score'] = nba_df['score'].replace('NA', np.nan) # Should not be needed if na_values works

        # --- BEGIN: Initialize first row null scores to 0 or "0 - 0" ---
        print("        Initializing scores to 0 for first null rows per game_id...")
        # Identify first row of each game_id group
        # Note: Make sure game_id is sorted if cumcount is to reliably get the first chronological event.
        # The problem description implies the base file is already sorted by time.
        is_first_row_in_group = nba_df.groupby('game_id').cumcount() == 0

        if 'away_score' in nba_df.columns:
            nba_df.loc[is_first_row_in_group & nba_df['away_score'].isna(), 'away_score'] = 0
            if (is_first_row_in_group & nba_df['away_score'].isna()).any():
                 print("          Set initial null away_score to 0 for some games.")
        if 'home_score' in nba_df.columns:
            nba_df.loc[is_first_row_in_group & nba_df['home_score'].isna(), 'home_score'] = 0
            if (is_first_row_in_group & nba_df['home_score'].isna()).any():
                 print("          Set initial null home_score to 0 for some games.")
        if 'score' in nba_df.columns:
            # Ensure 'score' column can hold string "0 - 0"
            nba_df['score'] = nba_df['score'].astype(object) 
            nba_df.loc[is_first_row_in_group & nba_df['score'].isna(), 'score'] = "0 - 0"
            if (is_first_row_in_group & nba_df['score'].isna()).any():
                 print("          Set initial null score string to '0 - 0' for some games.")
        # --- END: Initialize first row null scores ---

        # Apply forward fill per game
        existing_score_cols = [col for col in score_cols_to_fill if col in nba_df.columns]
        if existing_score_cols:
            nba_df[existing_score_cols] = nba_df.groupby('game_id')[existing_score_cols].ffill()
            print(f"      Applied forward fill for: {existing_score_cols}")
        else:
            print(f"      Skipped forward fill as none of the target score columns were found: {score_cols_to_fill}")
            
        # Recalculate and/or forward-fill score_margin
        if 'score_margin' in nba_df.columns and 'away_score' in nba_df.columns and 'home_score' in nba_df.columns:
            # Ensure score_margin can hold strings like "TIE"
            nba_df['score_margin'] = nba_df['score_margin'].astype(object)

            # Mask for rows where home_score and away_score are not NaN (they should be numbers now or 0)
            mask_can_calculate_margin = nba_df['home_score'].notna() & nba_df['away_score'].notna()
            
            # Calculate numeric margin for these rows
            numeric_margin = nba_df.loc[mask_can_calculate_margin, 'home_score'] - nba_df.loc[mask_can_calculate_margin, 'away_score']
            
            # Update score_margin based on calculated numeric_margin
            # Create a temporary series for new margin values
            temp_margin_values = pd.Series(index=numeric_margin.index, dtype=object)
            temp_margin_values[numeric_margin == 0] = "TIE"
            temp_margin_values[numeric_margin != 0] = numeric_margin[numeric_margin != 0].astype(int).astype(str)
            
            nba_df.loc[mask_can_calculate_margin, 'score_margin'] = temp_margin_values
            
            # Forward fill score_margin itself to propagate initial "TIE" or first calculated margin
            nba_df['score_margin'] = nba_df.groupby('game_id')['score_margin'].ffill()
            print("      Recalculated and/or forward-filled 'score_margin' based on filled scores.")

    else:
        print("      Warning: 'game_id' column not found. Cannot perform grouped forward fill for scores.")
    # --- END: Initialize and Forward fill score columns ---

    # --- Map event_type ---
    event_type_map = {
        1: "MAKE", 2: "MISS", 3: "FreeThrow", 4: "Rebound", 5: "Turnover",
        6: "Foul", 7: "Violation", 8: "Substitution", 9: "Timeout",
        10: "JumpBall", 11: "Ejection", 12: "StartOfPeriod", 13: "EndOfPeriod",
        14: "Empty"
    }
    nba_df['event_type_num'] = pd.to_numeric(nba_df['event_type'], errors='coerce')
    nba_df['event_type'] = nba_df['event_type_num'].map(event_type_map).fillna(nba_df['event_type_num'].astype(str))
    print("      Mapped event types")

    # --- Fill NA ---
    for col in nba_df.select_dtypes(include=['object', 'string']).columns:
        if col in nba_df.columns:
            # For score columns that were ffilled and might still have initial NaNs (if a game had NO score events at all)
            # and are object type (like 'score' and 'score_margin'), this converts them to ''.
            # For 'away_score', 'home_score' (numeric), NaNs will remain NaNs.
            nba_df[col] = nba_df[col].fillna('')
    print("      Filled NA values in string columns (after score initialization and ffill)")

    # --- Calculate Base Scores --- ### CORRECTED case_when calls ###
    nba_df['home_description'] = nba_df['home_description'].astype(str)
    nba_df['visitor_description'] = nba_df['visitor_description'].astype(str)
    nba_df = nba_df.assign(
        Home_Action_Score=case_when(
            series_contains(nba_df['home_description'], "Free Throw", case=False) & ~series_contains(nba_df['home_description'], "MISS", case=False), 1,
            series_contains(nba_df['home_description'], "PTS", case=False) & ~series_contains(nba_df['home_description'], "3PT", case=False) & (nba_df['event_type'] == "MAKE"), 2,
            series_contains(nba_df['home_description'], "3PT", case=False) & ~series_contains(nba_df['home_description'], "MISS", case=False), 3,
            0 # Default value is the last argument
        ).astype(int),
        Away_Action_Score=case_when(
            series_contains(nba_df['visitor_description'], "Free Throw", case=False) & ~series_contains(nba_df['visitor_description'], "MISS", case=False), 1,
            series_contains(nba_df['visitor_description'], "PTS", case=False) & ~series_contains(nba_df['visitor_description'], "3PT", case=False) & (nba_df['event_type'] == "MAKE"), 2,
            series_contains(nba_df['visitor_description'], "3PT", case=False) & ~series_contains(nba_df['visitor_description'], "MISS", case=False), 3,
            0 # Default value is the last argument
        ).astype(int)
    )
    # ... (rest of score calculation: cumsum, Net_Total) ...
    group_cols = ['game_id'] if 'game_id' in nba_df.columns else []
    if group_cols:
        # Ensure game_id is appropriate for grouping, convert to string if mixed or problematic
        if not pd.api.types.is_integer_dtype(nba_df['game_id']) and not pd.api.types.is_string_dtype(nba_df['game_id']):
            # Check if conversion to int is possible first, if not, then string
            try:
                nba_df['game_id_temp'] = pd.to_numeric(nba_df['game_id'])
                if nba_df['game_id_temp'].notna().all() and (nba_df['game_id_temp'] == nba_df['game_id_temp'].astype(int)).all():
                     nba_df['game_id'] = nba_df['game_id_temp'].astype(int)
                else: # Contains non-integers or floats, convert to string
                    print(f"      Converting game_id from {nba_df['game_id'].dtype} to string for grouping due to mixed types.")
                    nba_df['game_id'] = nba_df['game_id'].astype(str)
                nba_df.drop(columns=['game_id_temp'], inplace=True, errors='ignore')
            except: # Fallback to string if any error
                print(f"      Converting game_id from {nba_df['game_id'].dtype} to string for grouping due to conversion error.")
                nba_df['game_id'] = nba_df['game_id'].astype(str)

        nba_df['Net_Home'] = nba_df.groupby(group_cols)['Home_Action_Score'].cumsum()
        nba_df['Net_Away'] = nba_df.groupby(group_cols)['Away_Action_Score'].cumsum()
    else:
        nba_df['Net_Home'] = nba_df['Home_Action_Score'].cumsum()
        nba_df['Net_Away'] = nba_df['Away_Action_Score'].cumsum()
    nba_df['Net_Total'] = nba_df['Net_Home'] + nba_df['Net_Away']
    print("      Calculated action scores and cumulative totals")

    # --- Create Lag/Lead Columns ---
    # ... (lag/lead calculation remains the same) ...
    cols_to_shift = {
        'seconds_remaining_quarter': ['prev_seconds'],
        'visitor_description': ['Prev_visitor_desc', 'Prev_visitor_desc2', 'Next_visitor_desc'],
        'home_description': ['Prev_home_desc', 'Prev_home_desc2', 'Next_home_desc'],
        'event_type': ['Prev_Event', 'Next_event']
    }
    shift_periods = {'prev': 1, 'Prev': 1, 'Next': -1}
    for col, new_cols in cols_to_shift.items():
        if col in nba_df.columns:
            for new_col in new_cols:
                prefix = new_col.split('_')[0]
                period = shift_periods.get(prefix, 0)
                shift_num = 1
                if '2' in new_col: shift_num = 2
                period *= shift_num
                # Determine fill value based on dtype more robustly
                if pd.api.types.is_numeric_dtype(nba_df[col]):
                    fill_val_shift = np.nan
                else:
                    fill_val_shift = "" # Default for string/object

                if period != 0:
                    if group_cols:
                        nba_df[new_col] = nba_df.groupby(group_cols)[col].shift(period)
                        # Apply fill_value after shift if NaNs are introduced
                        if pd.api.types.is_numeric_dtype(nba_df[new_col]):
                             nba_df[new_col] = nba_df[new_col].fillna(fill_val_shift)
                        else:
                             nba_df[new_col] = nba_df[new_col].fillna(fill_val_shift)
                    else:
                        nba_df[new_col] = nba_df[col].shift(period, fill_value=fill_val_shift)
        else:
             print(f"      Warning: Column '{col}' needed for lag/lead generation not found.")
    print("      Created lag/lead columns")


    # --- Alias Player Columns ---
    # Drop existing a1-h5 if they exist to avoid duplicates after rename
    player_aliases = [f"{prefix}{i}" for prefix in ["a", "h"] for i in range(1, 6)]
    nba_df = nba_df.drop(columns=[col for col in player_aliases if col in nba_df.columns])
    
    alias_map = {f"{team}_player{i}": f"{prefix}{i}" for team, prefix in [("away", "a"), ("home", "h")] for i in range(1, 6)}
    nba_df = nba_df.rename(columns=alias_map)
    print("      Aliased player columns (a1-h5)")

    print(f"    Finished Base Processing for {file_path}.")
    return nba_df


def _mode_int(series):
    values = pd.to_numeric(series, errors='coerce').dropna()
    values = values[values != 0]
    if values.empty:
        return 0
    modes = values.astype(int).mode()
    return int(modes.iloc[0]) if not modes.empty else 0


def _infer_wnba_rebound_context(nba_df):
    """Infer missed-shot side and rebound side for WNBA player/team rebounds.

    WNBA raw rows can store team rebounds only in the generic `description`
    column and blocked shots can insert a nonstandard event between the miss
    and rebound. A row-local `Prev_*_desc contains MISS` rule therefore misses
    legitimate defensive rebounds and collapses possessions.
    """
    out = nba_df.copy()
    index = out.index
    teamid = pd.to_numeric(out.get('teamid', 0), errors='coerce').fillna(0).astype(int)
    personid = pd.to_numeric(out.get('personid', 0), errors='coerce').fillna(0).astype(int)
    home_desc = out['home_description'].fillna('').astype(str)
    away_desc = out['visitor_description'].fillna('').astype(str)
    generic_desc = out.get('description', '').fillna('').astype(str)

    home_event_with_team = home_desc.str.strip().ne('') & teamid.ne(0)
    away_event_with_team = away_desc.str.strip().ne('') & teamid.ne(0)
    if 'game_id' in out.columns:
        home_team_ids = out.loc[home_event_with_team].groupby('game_id')['teamid'].agg(_mode_int)
        away_team_ids = out.loc[away_event_with_team].groupby('game_id')['teamid'].agg(_mode_int)
        home_team_id = out['game_id'].map(home_team_ids).fillna(0).astype(int)
        away_team_id = out['game_id'].map(away_team_ids).fillna(0).astype(int)
    else:
        home_team_id = pd.Series(_mode_int(out.loc[home_event_with_team, 'teamid']), index=index)
        away_team_id = pd.Series(_mode_int(out.loc[away_event_with_team, 'teamid']), index=index)

    home_rebound = series_contains(home_desc, r'\bREBOUND\b|Rebound', case=False, regex=True)
    away_rebound = series_contains(away_desc, r'\bREBOUND\b|Rebound', case=False, regex=True)
    generic_team_rebound = (
        out['event_type'].eq('Rebound') &
        home_desc.str.strip().eq('') &
        away_desc.str.strip().eq('') &
        series_contains(generic_desc, r'\bRebound\b', case=False, regex=True)
    )

    rebound_side = pd.Series('', index=index, dtype=object)
    rebound_side.loc[home_rebound] = 'Home'
    rebound_side.loc[away_rebound] = 'Away'
    rebound_side.loc[generic_team_rebound & personid.eq(home_team_id)] = 'Home'
    rebound_side.loc[generic_team_rebound & personid.eq(away_team_id)] = 'Away'

    generic_team_side = pd.Series('', index=index, dtype=object)
    generic_team_side.loc[personid.eq(home_team_id)] = 'Home'
    generic_team_side.loc[personid.eq(away_team_id)] = 'Away'

    home_miss = (
        out['event_type'].eq('MISS') &
        series_contains(home_desc, r'\bMISS\b', case=False, regex=True)
    )
    away_miss = (
        out['event_type'].eq('MISS') &
        series_contains(away_desc, r'\bMISS\b', case=False, regex=True)
    )
    home_final_ft_miss = (
        out['event_type'].eq('FreeThrow') &
        series_contains(home_desc, r'\bMISS\b', case=False, regex=True) &
        series_contains(home_desc, r'\b(1 of 1|2 of 2|3 of 3)\b', case=False, regex=True) &
        ~series_contains(home_desc, 'technical', case=False)
    )
    away_final_ft_miss = (
        out['event_type'].eq('FreeThrow') &
        series_contains(away_desc, r'\bMISS\b', case=False, regex=True) &
        series_contains(away_desc, r'\b(1 of 1|2 of 2|3 of 3)\b', case=False, regex=True) &
        ~series_contains(away_desc, 'technical', case=False)
    )

    miss_side = pd.Series('', index=index, dtype=object)
    miss_side.loc[home_miss | home_final_ft_miss] = 'Home'
    miss_side.loc[away_miss | away_final_ft_miss] = 'Away'

    pending_miss_side = pd.Series('', index=index, dtype=object)
    pending_ft_miss = pd.Series(False, index=index, dtype=bool)
    event_type = out['event_type'].fillna('').astype(str)
    game_groups = out.groupby('game_id', sort=False).groups if 'game_id' in out.columns else {None: list(index)}
    for _, group_index in game_groups.items():
        pending_side = ''
        pending_was_ft = False
        for idx in group_index:
            event = event_type.at[idx]
            if event == 'Rebound':
                pending_miss_side.at[idx] = pending_side
                pending_ft_miss.at[idx] = pending_was_ft
                pending_side = ''
                pending_was_ft = False
                continue
            side = miss_side.at[idx]
            if side:
                pending_side = side
                pending_was_ft = bool(home_final_ft_miss.at[idx] or away_final_ft_miss.at[idx])
                continue
            if event in {'MAKE', 'Turnover', 'EndOfPeriod', 'JumpBall'}:
                pending_side = ''
                pending_was_ft = False

    defensive_rebound_eop = (
        event_type.eq('Rebound') &
        rebound_side.ne('') &
        pending_miss_side.ne('') &
        rebound_side.ne(pending_miss_side)
    )
    offensive_ft_rebound = (
        event_type.eq('Rebound') &
        rebound_side.ne('') &
        pending_miss_side.ne('') &
        rebound_side.eq(pending_miss_side) &
        pending_ft_miss
    )

    return pd.DataFrame({
        'wnba_home_team_id': home_team_id,
        'wnba_away_team_id': away_team_id,
        'wnba_generic_team_side': generic_team_side,
        'wnba_rebound_side': rebound_side,
        'wnba_pending_miss_side': pending_miss_side,
        'wnba_defensive_rebound_eop': defensive_rebound_eop.astype(bool),
        'wnba_offensive_ft_rebound': offensive_ft_rebound.astype(bool),
    }, index=index)


def _opposite_side(side):
    if side == 'Home':
        return 'Away'
    if side == 'Away':
        return 'Home'
    return ''


def _infer_wnba_end_period_offense_side(nba_df, terminal_side):
    """Infer the live possession owner for generic EndOfPeriod terminal rows."""
    index = nba_df.index
    home_desc = nba_df['home_description'].fillna('').astype(str)
    away_desc = nba_df['visitor_description'].fillna('').astype(str)
    event_type = nba_df['event_type'].fillna('').astype(str)

    explicit_side = pd.Series('', index=index, dtype=object)
    explicit_side.loc[
        event_type.isin(['MAKE', 'MISS', 'FreeThrow']) &
        (series_contains(home_desc, r'\b(MISS|PTS|Free Throw)\b', case=False, regex=True))
    ] = 'Home'
    explicit_side.loc[
        event_type.isin(['MAKE', 'MISS', 'FreeThrow']) &
        (series_contains(away_desc, r'\b(MISS|PTS|Free Throw)\b', case=False, regex=True))
    ] = 'Away'
    explicit_side.loc[
        event_type.eq('Turnover') & series_contains(home_desc, 'Turnover', case=False)
    ] = 'Home'
    explicit_side.loc[
        event_type.eq('Turnover') & series_contains(away_desc, 'Turnover', case=False)
    ] = 'Away'
    explicit_side.loc[
        event_type.eq('Turnover') &
        explicit_side.eq('') &
        nba_df['wnba_generic_team_side'].isin(['Home', 'Away'])
    ] = nba_df.loc[
        event_type.eq('Turnover') &
        explicit_side.eq('') &
        nba_df['wnba_generic_team_side'].isin(['Home', 'Away']),
        'wnba_generic_team_side'
    ]

    end_period_side = pd.Series('', index=index, dtype=object)
    group_cols = ['game_id'] if 'game_id' in nba_df.columns else []
    game_groups = nba_df.groupby(group_cols, sort=False).groups if group_cols else {None: list(index)}
    for _, group_index in game_groups.items():
        current_side = ''
        for idx in group_index:
            event = event_type.at[idx]
            if event == 'StartOfPeriod':
                current_side = ''
                continue
            if event == 'EndOfPeriod':
                end_period_side.at[idx] = current_side
                current_side = ''
                continue

            row_explicit_side = explicit_side.at[idx]
            if row_explicit_side:
                current_side = row_explicit_side

            if bool(nba_df.at[idx, 'wnba_defensive_rebound_eop']):
                rebound_side = nba_df.at[idx, 'wnba_rebound_side']
                current_side = rebound_side if rebound_side in {'Home', 'Away'} else _opposite_side(current_side)
                continue

            if event == 'Rebound':
                rebound_side = nba_df.at[idx, 'wnba_rebound_side']
                if rebound_side in {'Home', 'Away'}:
                    current_side = rebound_side
                continue

            row_terminal_side = terminal_side.at[idx] if idx in terminal_side.index else ''
            if row_terminal_side in {'Home', 'Away'}:
                current_side = _opposite_side(row_terminal_side)

    return end_period_side


def prepare_wnba_standard_possession_df(nba_df, label="RAPM"):
    """Apply a WNBA-specific standard possession definition and O/D mapping."""
    rebound_ctx = _infer_wnba_rebound_context(nba_df)
    nba_df = nba_df.join(rebound_ctx)

    home_offensive_foul_no_turnover = (
        (nba_df['event_type'] == "Foul") &
        series_contains(nba_df['home_description'], r'offensive|charge', case=False, regex=True) &
        ~nba_df['Next_event'].eq("Turnover")
    )
    away_offensive_foul_no_turnover = (
        (nba_df['event_type'] == "Foul") &
        series_contains(nba_df['visitor_description'], r'offensive|charge', case=False, regex=True) &
        ~nba_df['Next_event'].eq("Turnover")
    )

    nba_df['offensive_FT_Rebound'] = nba_df['wnba_offensive_ft_rebound'].astype(bool)
    home_def_rebound = nba_df['wnba_defensive_rebound_eop'] & nba_df['wnba_pending_miss_side'].eq('Home')
    away_def_rebound = nba_df['wnba_defensive_rebound_eop'] & nba_df['wnba_pending_miss_side'].eq('Away')

    made_home = (
        series_contains(nba_df['home_description'], "PTS", case=False) &
        (nba_df['event_type'] == "MAKE") &
        ~series_contains(nba_df['Next_visitor_desc'], "S.FOUL", case=True)
    )
    made_away = (
        series_contains(nba_df['visitor_description'], "PTS", case=False) &
        (nba_df['event_type'] == "MAKE") &
        ~series_contains(nba_df['Next_home_desc'], "S.FOUL", case=True)
    )
    made_final_home_ft = (
        series_contains(nba_df['home_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False, regex=True) &
        series_contains(nba_df['home_description'], "PTS", case=False) &
        ~(series_contains(nba_df['Prev_visitor_desc2'], "Transition", case=False) |
          series_contains(nba_df['Prev_visitor_desc'], "Transition", case=False) |
          series_contains(nba_df['home_description'], "Flagrant", case=False))
    )
    made_final_away_ft = (
        series_contains(nba_df['visitor_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False, regex=True) &
        series_contains(nba_df['visitor_description'], "PTS", case=False) &
        ~(series_contains(nba_df['Prev_home_desc2'], "Transition", case=False) |
          series_contains(nba_df['Prev_home_desc'], "Transition", case=False) |
          series_contains(nba_df['visitor_description'], "Flagrant", case=False))
    )

    nba_df['End_of_Possession'] = case_when(
        (nba_df['event_type'] == "EndOfPeriod") & (nba_df['prev_seconds'] > 0), True,
        series_contains(nba_df['home_description'], r'Flagrant (2 of 2|3 of 3)', case=False, regex=True), True,
        series_contains(nba_df['visitor_description'], r'Flagrant (2 of 2|3 of 3)', case=False, regex=True), True,
        (nba_df['event_type'] == "Turnover"), True,
        nba_df['wnba_defensive_rebound_eop'], True,
        made_home, True,
        made_away, True,
        made_final_home_ft, True,
        made_final_away_ft, True,
        home_offensive_foul_no_turnover, True,
        away_offensive_foul_no_turnover, True,
        False
    ).astype(bool)

    nba_df['TeamOnOffense'] = case_when(
        series_contains(nba_df['home_description'], r'Flagrant (2 of 2|3 of 3)', case=False, regex=True), "Home",
        series_contains(nba_df['visitor_description'], r'Flagrant (2 of 2|3 of 3)', case=False, regex=True), "Away",
        home_def_rebound, "Home",
        away_def_rebound, "Away",
        series_contains(nba_df['home_description'], "Turnover", case=False), "Home",
        series_contains(nba_df['visitor_description'], "Turnover", case=False), "Away",
        made_home, "Home",
        made_away, "Away",
        made_final_home_ft, "Home",
        made_final_away_ft, "Away",
        home_offensive_foul_no_turnover, "Home",
        away_offensive_foul_no_turnover, "Away",
        ""
    )
    generic_team_turnover = (
        nba_df['event_type'].eq('Turnover') &
        nba_df['TeamOnOffense'].eq('') &
        nba_df['wnba_generic_team_side'].isin(['Home', 'Away'])
    )
    nba_df.loc[generic_team_turnover, 'TeamOnOffense'] = nba_df.loc[
        generic_team_turnover,
        'wnba_generic_team_side'
    ]
    blank_ft_side = nba_df['TeamOnOffense'].eq('') & nba_df['event_type'].eq('FreeThrow')
    nba_df.loc[
        blank_ft_side & series_contains(nba_df['home_description'], 'Free Throw', case=False),
        'TeamOnOffense'
    ] = 'Home'
    nba_df.loc[
        blank_ft_side & series_contains(nba_df['visitor_description'], 'Free Throw', case=False),
        'TeamOnOffense'
    ] = 'Away'
    end_period_side = _infer_wnba_end_period_offense_side(nba_df, nba_df['TeamOnOffense'])
    blank_end_period = (
        nba_df['TeamOnOffense'].eq('') &
        nba_df['event_type'].eq('EndOfPeriod') &
        end_period_side.isin(['Home', 'Away'])
    )
    nba_df.loc[blank_end_period, 'TeamOnOffense'] = end_period_side.loc[blank_end_period]
    print(f"      Calculated robust WNBA EOP, TeamOnOffense for {label}")

    group_cols = ['game_id'] if 'game_id' in nba_df.columns else []
    window_size = 5
    if group_cols:
        ft_events = nba_df.groupby(group_cols)['event_type'].transform(
            lambda x: (x == "FreeThrow").rolling(window=window_size, min_periods=1).sum().shift(-(window_size - 1)).fillna(0)
        )
        sub_events = nba_df.groupby(group_cols)['event_type'].transform(
            lambda x: (x == "Substitution").rolling(window=window_size, min_periods=1).sum().shift(-(window_size - 1)).fillna(0)
        )
    else:
        ft_events = (nba_df['event_type'] == "FreeThrow").rolling(window=window_size, min_periods=1).sum().shift(-(window_size - 1)).fillna(0)
        sub_events = (nba_df['event_type'] == "Substitution").rolling(window=window_size, min_periods=1).sum().shift(-(window_size - 1)).fillna(0)

    exclude_foul_pattern = r'T\.FOUL|FLAGRANT|Offens|Transition'
    nba_df['PotentialFoul'] = case_when(
        (ft_events > 0) & (sub_events > 0) & (nba_df['event_type'] == "Foul") &
        ~series_contains(nba_df['visitor_description'], exclude_foul_pattern, case=False, regex=True) &
        ~series_contains(nba_df['home_description'], exclude_foul_pattern, case=False, regex=True), True,
        False
    ).astype(bool)
    print(f"      Identified {nba_df['PotentialFoul'].sum()} potential fouls")

    nba_df_propagated = propagate_player_values_py(nba_df)
    nba_df_checked = ft_off_check_py(nba_df_propagated)

    o_mapping = {}
    d_mapping = {}
    for i in range(1, 6):
        o_mapping[f'O{i}'] = np.where(
            nba_df_checked['TeamOnOffense'] == "Away",
            nba_df_checked[f'a{i}'],
            np.where(nba_df_checked['TeamOnOffense'] == "Home", nba_df_checked[f'h{i}'], np.nan)
        )
        d_mapping[f'D{i}'] = np.where(
            nba_df_checked['TeamOnOffense'] == "Away",
            nba_df_checked[f'h{i}'],
            np.where(nba_df_checked['TeamOnOffense'] == "Home", nba_df_checked[f'a{i}'], np.nan)
        )
    nba_df_checked = nba_df_checked.assign(**o_mapping, **d_mapping)
    print("      Mapped O/D Players")
    return nba_df_checked, group_cols


# --- Process RAPM Function ---
def process_rapm_py(file_path, year, season_str):
    """ Processes data for standard RAPM calculation. """
    print(f"  Starting RAPM Processing for {season_str}...")
    nba_df = _base_processing(file_path, allow_partial_lineups=True)
    if nba_df is None: return None
    start_time = time.time()

    nba_df_checked, group_cols = prepare_wnba_standard_possession_df(nba_df, "RAPM")

    # --- Calculate Luck-Adjusted Points (RAPM Specific) --- ### CORRECTED case_when calls ###
    # ... (placeholder join for player stats remains the same) ...
# --- Join Player Season Stats (FTPerc, ThreePerc) ---
    print(f"      Fetching player stats for season {season_str} using nba_api...")
    player_stats_df = None
    try:
        # Import necessary nba_api components (ensure nba_api is installed)
        # You might want to move this import to the top of the file


        # Call the endpoint
        league_id = "10" if "WNBA" in file_path.upper() else "00"
        player_stats_endpoint = leaguedashplayerstats.LeagueDashPlayerStats(season=season_str, league_id_nullable=league_id)
        # Add a small delay to avoid hitting rate limits if called frequently
        time.sleep(0.6) # Add a small delay (e.g., 600ms)
        player_stats_list = player_stats_endpoint.get_data_frames()

        if player_stats_list:
            player_stats_df = player_stats_list[0]
            # Select and rename columns to match R code's intent (or use directly)
            # R used indices: PlayerID=1, FTPerc=20, ThreePerc=17
            # nba_api typically uses: PLAYER_ID, FT_PCT, FG3_PCT
            player_stats_df = player_stats_df[['PLAYER_ID', 'FT_PCT', 'FG3_PCT']].copy()
            # Rename for clarity or consistency if desired (optional)
            player_stats_df = player_stats_df.rename(columns={
                'PLAYER_ID': 'PlayerID',
                'FT_PCT': 'FTPerc',
                'FG3_PCT': 'ThreePerc'
            })
            print(f"      Successfully fetched {len(player_stats_df)} player stat entries.")
        else:
            print("      Warning: nba_api returned no data for player stats.")

    except ImportError:
        print("      Error: 'nba_api' library not found. Cannot fetch player stats. Please install it (`pip install nba_api`).")
    except Exception as e:
        print(f"      Error fetching or processing player stats via nba_api: {e}")

    # --- Merge Player Stats ---
    shooter_id_col = 'player1_id'
    if player_stats_df is not None and shooter_id_col in nba_df_checked.columns:
        print("      Preparing data for player stats merge...")
        # Ensure merge keys are compatible numeric types (handle potential errors)
        nba_df_checked['player1_id_numeric'] = pd.to_numeric(nba_df_checked[shooter_id_col], errors='coerce')
        # Use nullable Int64 for PlayerID in stats_df to handle potential NaNs robustly if conversion fails
        player_stats_df['PlayerID'] = pd.to_numeric(player_stats_df['PlayerID'], errors='coerce').astype('Int64')

        # Drop rows where conversion failed in either df to ensure clean merge
        rows_before_drop = len(nba_df_checked)
        nba_df_checked = nba_df_checked.dropna(subset=['player1_id_numeric'])
        if len(nba_df_checked) < rows_before_drop:
            print(f"      Warning: Dropped {rows_before_drop - len(nba_df_checked)} rows from main df due to invalid player1_id.")

        player_stats_df = player_stats_df.dropna(subset=['PlayerID'])
        # Convert player1_id to Int64 as well for consistent merging
        nba_df_checked['player1_id_numeric'] = nba_df_checked['player1_id_numeric'].astype('Int64')


        print(f"      Merging player stats onto {len(nba_df_checked)} rows...")
        original_rows = len(nba_df_checked)
        nba_df_checked = pd.merge(
            nba_df_checked,
            player_stats_df[['PlayerID', 'FTPerc', 'ThreePerc']],
            left_on='player1_id_numeric',
            right_on='PlayerID',
            how='left'  # Keep all rows from nba_df_checked
        )
        if len(nba_df_checked) != original_rows:
             print(f"      Warning: Row count changed during player stats merge ({original_rows} -> {len(nba_df_checked)}). Check for duplicate IDs?")
        # Clean up merge key columns if no longer needed
        nba_df_checked = nba_df_checked.drop(columns=['player1_id_numeric', 'PlayerID'], errors='ignore')
        merged_stats_count = nba_df_checked['FTPerc'].notna().sum()
        print(f"      Successfully merged player stats for {merged_stats_count} player instances.")

    else:
        print(f"      Skipping player stats merge (Stats data unavailable or '{shooter_id_col}' missing).")
        # Ensure columns exist for the next step, even if merge failed/skipped
        if 'FTPerc' not in nba_df_checked.columns: nba_df_checked['FTPerc'] = np.nan
        if 'ThreePerc' not in nba_df_checked.columns: nba_df_checked['ThreePerc'] = np.nan

    # --- Calculate Expected Points based on Stats (Handle Missing) ---
    default_ft_perc = 0.75
    default_3p_perc = 0.35

    # Use merged stats, fill NaN with defaults
    nba_df_checked['ExpFT'] = nba_df_checked['FTPerc'].fillna(default_ft_perc).astype(float)
    nba_df_checked['Exp3PT'] = nba_df_checked['ThreePerc'].fillna(default_3p_perc).astype(float)

    # Clean up intermediate columns
    nba_df_checked = nba_df_checked.drop(columns=['FTPerc', 'ThreePerc'], errors='ignore')

    print(f"      Assigned ExpFT (default: {default_ft_perc}) and Exp3PT (default: {default_3p_perc}).")

    # --- Calculate Luck-Adjusted Points (RAPM Specific) --- # (This line was already present, calculation follows)

    shooter_id_col = 'player1_id'
    if shooter_id_col not in nba_df_checked.columns:
         print(f"      Warning: Shooter ID column '{shooter_id_col}' not found for player stats join. Using defaults.")
         nba_df_checked['FTPerc'] = 0.75
         nba_df_checked['ThreePerc'] = 0.35
    else:
         print(f"      Found shooter ID column '{shooter_id_col}'. Using defaults for FTPerc/ThreePerc for now.")
         nba_df_checked['FTPerc'] = 0.75
         nba_df_checked['ThreePerc'] = 0.35

    nba_df_checked['ExpFT'] = nba_df_checked['FTPerc'].fillna(0.75).astype(float)
    nba_df_checked['Exp3PT'] = nba_df_checked['ThreePerc'].fillna(0.35).astype(float)

    is_ft_event = series_contains(nba_df_checked['home_description'], "Free Throw", case=False) | \
                  series_contains(nba_df_checked['visitor_description'], "Free Throw", case=False) | \
                  (nba_df_checked['event_type'] == "FreeThrow")
    is_2pt_make_event = (series_contains(nba_df_checked['home_description'], "PTS", case=False) & ~series_contains(nba_df_checked['home_description'], "3PT", case=False) & (nba_df_checked['event_type'] == "MAKE")) | \
                        (series_contains(nba_df_checked['visitor_description'], "PTS", case=False) & ~series_contains(nba_df_checked['visitor_description'], "3PT", case=False) & (nba_df_checked['event_type'] == "MAKE"))
    is_3pt_event = series_contains(nba_df_checked['home_description'], "3PT", case=False) | \
                   series_contains(nba_df_checked['visitor_description'], "3PT", case=False)
    actual_score = (nba_df_checked['Home_Action_Score'] + nba_df_checked['Away_Action_Score']).astype(float)

    nba_df_checked['OffNet'] = case_when(
        is_ft_event, nba_df_checked['ExpFT'],
        is_2pt_make_event, 2.0,
        is_3pt_event, (nba_df_checked['Exp3PT'] * 3.0) * 0.2 + 0.8 * actual_score,
        0.0 # Default
    )
    nba_df_checked['DefNet'] = case_when(
        is_ft_event, nba_df_checked['ExpFT'],
        is_2pt_make_event, 2.0,
        is_3pt_event, (nba_df_checked['Exp3PT'] * 3.0) * 0.4 + 0.6 * actual_score,
        0.0 # Default
    )

    # ... (calculate OffTotal, DefTotal remains the same) ...
    if group_cols:
      nba_df_checked['OffTotal'] = nba_df_checked.groupby(group_cols)['OffNet'].cumsum()
      nba_df_checked['DefTotal'] = nba_df_checked.groupby(group_cols)['DefNet'].cumsum()
    else:
      nba_df_checked['OffTotal'] = nba_df_checked['OffNet'].cumsum()
      nba_df_checked['DefTotal'] = nba_df_checked['DefNet'].cumsum()
    print("      Calculated OffNet/DefNet and cumulative totals")


    # --- Filter for EOP & Calculate Diffs ---
    # ... (filtering and diff calculation remains the same) ...
    if 'End_of_Possession' in nba_df_checked.columns and pd.api.types.is_bool_dtype(nba_df_checked['End_of_Possession']):
        nba_filt = nba_df_checked[nba_df_checked['End_of_Possession']].copy()
    else:
         print("      Warning: 'End_of_Possession' column missing or not boolean for filtering. Skipping.")
         nba_filt = nba_df_checked.copy() # Or return None
    print(f"      Filtered down to {len(nba_filt)} RAPM End of Possession rows")

    if not nba_filt.empty:
        cols_to_diff = ['Net_Total', 'OffTotal', 'DefTotal']
        diff_col_names = ['Net_Diff', 'Off_Diff', 'Def_Diff']
        for col, diff_col in zip(cols_to_diff, diff_col_names):
            if col in nba_filt.columns:
                if group_cols:
                    nba_filt[diff_col] = nba_filt[col] - nba_filt.groupby(group_cols)[col].shift(fill_value=0)
                else:
                    nba_filt[diff_col] = nba_filt[col] - nba_filt[col].shift(fill_value=0)
            else:
                 print(f"      Warning: Column '{col}' not found for diff calculation.")
                 nba_filt[diff_col] = 0 # Add as zero if missing
        print("      Calculated Net/Off/Def differences")
    else:
         print("      Warning: DataFrame is empty after EOP filter. Skipping diff calculation.")


    # --- Finalize ---
    # ... (finalization remains the same) ...
    rapm_final_cols = ['Net_Diff', 'Off_Diff', 'Def_Diff']
    rapm_final_cols_present = [c for c in rapm_final_cols if c in nba_filt.columns]
    if not rapm_final_cols_present:
        print("      Error: Could not find Net_Diff, Off_Diff, or Def_Diff columns for final output.")
        return None
    nba_rapm_output = _finalize_df(nba_filt, rapm_final_cols_present, year)

    end_time = time.time()
    print(f"  Finished RAPM Processing. Time: {end_time - start_time:.2f} seconds.")
    return nba_rapm_output

# --- Process TS (True Shooting) Function ---
def process_ts_py(file_path, year, season_str):
    """ Processes data for True Shooting calculation. """
    print(f"  Starting TS Processing for {season_str}...")
    nba_df = _base_processing(file_path, allow_partial_lineups=True)
    if nba_df is None: return None
    start_time = time.time()

    nba_df_checked, group_cols = prepare_wnba_standard_possession_df(nba_df, "TS")
    if 'End_of_Possession' in nba_df_checked.columns and pd.api.types.is_bool_dtype(nba_df_checked['End_of_Possession']):
        nba_filt = nba_df_checked[nba_df_checked['End_of_Possession']].copy()
    else:
        print("      Warning: 'End_of_Possession' column missing or not boolean for filtering. Skipping.")
        nba_filt = nba_df_checked.copy()
    print(f"      Filtered down to {len(nba_filt)} robust TS End of Possession rows (before TO removal)")

    initial_rows_ts = len(nba_filt)
    nba_filt = nba_filt[nba_filt['event_type'] != 'Turnover'].copy()
    print(f"      Filtered out {initial_rows_ts - len(nba_filt)} Turnover possessions for TS")

    if not nba_filt.empty and 'Net_Total' in nba_filt.columns:
        if group_cols:
            nba_filt['Net_Diff'] = nba_filt['Net_Total'] - nba_filt.groupby(group_cols)['Net_Total'].shift(fill_value=0)
        else:
            nba_filt['Net_Diff'] = nba_filt['Net_Total'] - nba_filt['Net_Total'].shift(fill_value=0)
        print("      Calculated robust Net_Diff for TS")
    else:
        print("      Warning: DataFrame empty or Net_Total missing for TS Net_Diff calculation.")
        nba_filt['Net_Diff'] = 0

    nba_ts_output = _finalize_df(nba_filt, 'Net_Diff', year)
    end_time = time.time()
    print(f"  Finished TS Processing. Time: {end_time - start_time:.2f} seconds.")
    return nba_ts_output

    # --- Define Possession Characteristics (TS Version) --- ### CORRECTED case_when calls ###
    nba_df['End_of_Possession'] = case_when(
        (nba_df['event_type'] == "EndOfPeriod") & (nba_df['prev_seconds'] > 0), True,
        series_contains(nba_df['home_description'], r'Flagrant (2 of 2|3 of 3)', case=False, regex=True), True,
        series_contains(nba_df['visitor_description'], r'Flagrant (2 of 2|3 of 3)', case=False, regex=True), True,
        (nba_df['event_type'] == "Turnover"), True,
        series_contains(nba_df['visitor_description'], "MISS", case=False) & (nba_df['event_type'] == "MISS"), True,
        series_contains(nba_df['home_description'], "MISS", case=False) & (nba_df['event_type'] == "MISS"), True,
        series_contains(nba_df['home_description'], "PTS", case=False) & (nba_df['event_type'] == "MAKE") & ~series_contains(nba_df['Next_visitor_desc'], "S.FOUL", case=True), True,
        series_contains(nba_df['visitor_description'], "PTS", case=False) & (nba_df['event_type'] == "MAKE") & ~series_contains(nba_df['Next_home_desc'], "S.FOUL", case=True), True,
        series_contains(nba_df['home_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False, regex=True) & \
             ~(series_contains(nba_df['Prev_home_desc2'], "Transition", case=False) | series_contains(nba_df['Prev_home_desc'], "Transition", case=False) | series_contains(nba_df['home_description'], "Flagrant", case=False)), True, # Adjusted R check slightly
        series_contains(nba_df['visitor_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False, regex=True) & \
             ~(series_contains(nba_df['Prev_visitor_desc2'], "Transition", case=False) | series_contains(nba_df['Prev_visitor_desc'], "Transition", case=False) | series_contains(nba_df['visitor_description'], "Flagrant", case=False)), True, # Adjusted R check slightly
        False # Default
    ).astype(bool)

    nba_df['TeamOnOffense'] = case_when(
        series_contains(nba_df['home_description'], r'Flagrant (2 of 2|3 of 3)', case=False, regex=True), "Home",
        series_contains(nba_df['visitor_description'], r'Flagrant (2 of 2|3 of 3)', case=False, regex=True), "Away",
        series_contains(nba_df['home_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_visitor_desc'], "MISS", case=False), "Away",
        series_contains(nba_df['visitor_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_home_desc'], "MISS", case=False), "Home",
        series_contains(nba_df['visitor_description'], "MISS", case=False), "Away",
        series_contains(nba_df['home_description'], "MISS", case=False), "Home",
        series_contains(nba_df['visitor_description'], "MAKE", case=False), "Away",
        series_contains(nba_df['home_description'], "MAKE", case=False), "Home",
        series_contains(nba_df['home_description'], "Turnover", case=False), "Home",
        series_contains(nba_df['visitor_description'], "Turnover", case=False), "Away",
        series_contains(nba_df['home_description'], "PTS", case=False), "Home",
        series_contains(nba_df['visitor_description'], "PTS", case=False), "Away",
        series_contains(nba_df['home_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False, regex=True), "Home",
        series_contains(nba_df['visitor_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False, regex=True), "Away",
        "" # Default
    )
    print("      Calculated EOP, TeamOnOffense for TS")

    # --- Identify Potential Fouls --- ### CORRECTED case_when call ###
    window_size = 5
    group_cols = ['game_id'] if 'game_id' in nba_df.columns else []
    if group_cols:
        ft_events = nba_df.groupby(group_cols)['event_type'].transform(lambda x: (x == "FreeThrow").rolling(window=window_size, min_periods=1).sum().shift(-(window_size - 1)).fillna(0))
        sub_events = nba_df.groupby(group_cols)['event_type'].transform(lambda x: (x == "Substitution").rolling(window=window_size, min_periods=1).sum().shift(-(window_size - 1)).fillna(0))
    else:
        ft_events = (nba_df['event_type'] == "FreeThrow").rolling(window=window_size, min_periods=1).sum().shift(-(window_size - 1)).fillna(0)
        sub_events = (nba_df['event_type'] == "Substitution").rolling(window=window_size, min_periods=1).sum().shift(-(window_size - 1)).fillna(0)

    exclude_foul_pattern = r'T\.FOUL|FLAGRANT|Offens|Transition'
    nba_df['PotentialFoul'] = case_when(
        (ft_events > 0) & (sub_events > 0) & (nba_df['event_type'] == "Foul") & \
        ~series_contains(nba_df['visitor_description'], exclude_foul_pattern, case=False, regex=True) & \
        ~series_contains(nba_df['home_description'], exclude_foul_pattern, case=False, regex=True), True,
        False # Default
    ).astype(bool)
    print(f"      Identified {nba_df['PotentialFoul'].sum()} potential fouls")

    # --- Apply Propagation & Map Players ---
    nba_df_propagated = propagate_player_values_py(nba_df)
    nba_df_checked = nba_df_propagated # Skip ft_off_check for TS

    # ... (map O/D players remains the same) ...
    o_mapping = {}
    d_mapping = {}
    for i in range(1, 6):
        o_mapping[f'O{i}'] = np.where(nba_df_checked['TeamOnOffense'] == "Away", nba_df_checked[f'a{i}'],
                                        np.where(nba_df_checked['TeamOnOffense'] == "Home", nba_df_checked[f'h{i}'], np.nan))
        d_mapping[f'D{i}'] = np.where(nba_df_checked['TeamOnOffense'] == "Away", nba_df_checked[f'h{i}'],
                                        np.where(nba_df_checked['TeamOnOffense'] == "Home", nba_df_checked[f'a{i}'], np.nan))
    nba_df_checked = nba_df_checked.assign(**o_mapping, **d_mapping)
    print("      Mapped O/D Players")

    # --- Filter for EOP & Calculate Diffs ---
    # ... (filtering, TO removal, diff calculation remains the same) ...
    if 'End_of_Possession' in nba_df_checked.columns and pd.api.types.is_bool_dtype(nba_df_checked['End_of_Possession']):
        nba_filt = nba_df_checked[nba_df_checked['End_of_Possession']].copy()
    else:
         print("      Warning: 'End_of_Possession' column missing or not boolean for filtering. Skipping.")
         nba_filt = nba_df_checked.copy()
    print(f"      Filtered down to {len(nba_filt)} TS End of Possession rows (before TO removal)")

    initial_rows_ts = len(nba_filt)
    nba_filt = nba_filt[nba_filt['event_type'] != 'Turnover'].copy()
    print(f"      Filtered out {initial_rows_ts - len(nba_filt)} Turnover possessions for TS")

    if not nba_filt.empty:
        if 'Net_Total' in nba_filt.columns:
            if group_cols:
                 nba_filt['Net_Diff'] = nba_filt['Net_Total'] - nba_filt.groupby(group_cols)['Net_Total'].shift(fill_value=0)
            else:
                nba_filt['Net_Diff'] = nba_filt['Net_Total'] - nba_filt['Net_Total'].shift(fill_value=0)
            print("      Calculated Net_Diff for TS")
        else:
            print("      Warning: 'Net_Total' column missing for TS Net_Diff calculation.")
            nba_filt['Net_Diff'] = 0 # Add as zero if missing
    else:
        print("      Warning: DataFrame empty after Turnover filter. Skipping Net_Diff calculation.")

    # --- Finalize ---
    nba_ts_output = _finalize_df(nba_filt, 'Net_Diff', year)

    end_time = time.time()
    print(f"  Finished TS Processing. Time: {end_time - start_time:.2f} seconds.")
    return nba_ts_output


# --- Process REB (Rebounding) Function ---
def process_reb_py(file_path, year, season_str):
    """ Processes data for Rebounding factor calculation. """
    print(f"  Starting REB Processing for {season_str}...")
    nba_df = _base_processing(file_path, allow_partial_lineups=True)
    if nba_df is None: return None
    start_time = time.time()

    rebound_ctx = _infer_wnba_rebound_context(nba_df)
    nba_df = nba_df.join(rebound_ctx)
    rebound_chance = (
        nba_df['event_type'].eq('Rebound') &
        nba_df['wnba_pending_miss_side'].isin(['Home', 'Away']) &
        nba_df['wnba_rebound_side'].isin(['Home', 'Away'])
    )
    nba_df['End_of_Possession'] = rebound_chance.astype(bool)
    nba_df['TeamOnOffense'] = np.where(
        rebound_chance,
        nba_df['wnba_pending_miss_side'],
        ''
    )
    nba_df['Offensive_Rebound'] = (
        rebound_chance &
        nba_df['wnba_rebound_side'].eq(nba_df['wnba_pending_miss_side'])
    ).astype(int)
    print(f"      Calculated robust REB chances ({int(rebound_chance.sum())}) and offensive rebounds ({int(nba_df['Offensive_Rebound'].sum())})")

    o_mapping = {}
    d_mapping = {}
    for i in range(1, 6):
        o_mapping[f'O{i}'] = np.where(
            nba_df['TeamOnOffense'] == "Away",
            nba_df[f'a{i}'],
            np.where(nba_df['TeamOnOffense'] == "Home", nba_df[f'h{i}'], np.nan)
        )
        d_mapping[f'D{i}'] = np.where(
            nba_df['TeamOnOffense'] == "Away",
            nba_df[f'h{i}'],
            np.where(nba_df['TeamOnOffense'] == "Home", nba_df[f'a{i}'], np.nan)
        )
    nba_df_checked = nba_df.assign(**o_mapping, **d_mapping)
    nba_filt = nba_df_checked[nba_df_checked['End_of_Possession']].copy()
    print(f"      Filtered down to {len(nba_filt)} robust REB chance rows")

    nba_reb_output = _finalize_df(nba_filt, 'Offensive_Rebound', year)
    end_time = time.time()
    print(f"  Finished REB Processing. Time: {end_time - start_time:.2f} seconds.")
    return nba_reb_output

    # --- Define Possession Characteristics (REB Version) --- ### CORRECTED case_when calls ###
    time_col_exists = 'time_quarter' in nba_df.columns
    if not time_col_exists:
        print("      Warning: 'time_quarter' column not found. Offensive Rebound logic might be inaccurate.")

    nba_df['Offensive_Rebound'] = case_when(
        series_contains(nba_df['home_description'], r'\bREBOUND\b', case=True, regex=True) & \
            series_contains(nba_df['Prev_home_desc'], "MISS", case=False) & \
            (True if not time_col_exists else ~(series_contains(nba_df['time_quarter'], "0:00", case=False))) & \
            ~(nba_df['Next_event'] == "EndOfPeriod") & \
            ~series_contains(nba_df['Prev_home_desc'], r'(1 of [23]|2 of 3|Technical|Flagrant)', case=False, regex=True), 1,
        series_contains(nba_df['home_description'], r'\bREBOUND\b', case=True, regex=True) & \
            series_contains(nba_df['Prev_home_desc'], "Putback", case=False) & \
             (True if not time_col_exists else ~(series_contains(nba_df['time_quarter'], "0:00", case=False))) & \
             ~(nba_df['Next_event'] == "EndOfPeriod") & \
            ~series_contains(nba_df['Prev_home_desc'], r'(1 of [23]|2 of 3|Technical|Flagrant)', case=False, regex=True), 1,
        series_contains(nba_df['visitor_description'], r'\bREBOUND\b', case=True, regex=True) & \
            series_contains(nba_df['Prev_visitor_desc'], "MISS", case=False) & \
            (True if not time_col_exists else ~(series_contains(nba_df['time_quarter'], "0:00", case=False))) & \
            ~(nba_df['Next_event'] == "EndOfPeriod") & \
            ~series_contains(nba_df['Prev_visitor_desc'], r'(1 of [23]|2 of 3|Technical|Flagrant)', case=False, regex=True), 1,
        series_contains(nba_df['visitor_description'], r'\bREBOUND\b', case=True, regex=True) & \
            series_contains(nba_df['Prev_visitor_desc'], "Putback", case=False) & \
             (True if not time_col_exists else ~(series_contains(nba_df['time_quarter'], "0:00", case=False))) & \
             ~(nba_df['Next_event'] == "EndOfPeriod") & \
            ~series_contains(nba_df['Prev_visitor_desc'], r'(1 of [23]|2 of 3|Technical|Flagrant)', case=False, regex=True), 1,
        0 # Default
    ).astype(int)
    print(f"      Calculated Offensive_Rebound flag (found {nba_df['Offensive_Rebound'].sum()})")

    ft_exclude_pattern = r'\b(1 of [23]|2 of 3|Technical|Flagrant)\b'
    nba_df['End_of_Possession'] = case_when(
        series_contains(nba_df['home_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_visitor_desc'], "MISS", case=False) & ~series_contains(nba_df['Prev_visitor_desc'], ft_exclude_pattern, case=False, regex=True), True,
        series_contains(nba_df['visitor_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_home_desc'], "MISS", case=False) & ~series_contains(nba_df['Prev_home_desc'], ft_exclude_pattern, case=False, regex=True), True,
        series_contains(nba_df['home_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_home_desc'], "MISS", case=False) & ~series_contains(nba_df['Prev_home_desc'], ft_exclude_pattern, case=False, regex=True), True,
        series_contains(nba_df['visitor_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_visitor_desc'], "MISS", case=False) & ~series_contains(nba_df['Prev_visitor_desc'], ft_exclude_pattern, case=False, regex=True), True,
        False # Default
    ).astype(bool)

    nba_df['TeamOnOffense'] = case_when(
        series_contains(nba_df['home_description'], r'Flagrant (2 of 2|3 of 3)', case=False, regex=True), "Home",
        series_contains(nba_df['visitor_description'], r'Flagrant (2 of 2|3 of 3)', case=False, regex=True), "Away",
        series_contains(nba_df['home_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_visitor_desc'], "MISS", case=False), "Away",
        series_contains(nba_df['visitor_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_home_desc'], "MISS", case=False), "Home",
        series_contains(nba_df['home_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_home_desc'], "MISS", case=False), "Home",
        series_contains(nba_df['visitor_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_visitor_desc'], "MISS", case=False), "Away",
        series_contains(nba_df['home_description'], "Turnover", case=False), "Home",
        series_contains(nba_df['visitor_description'], "Turnover", case=False), "Away",
        series_contains(nba_df['home_description'], "PTS", case=False) & (nba_df['event_type'] == "MAKE") & ~series_contains(nba_df['Next_visitor_desc'], "S.FOUL", case=True), "Home",
        series_contains(nba_df['visitor_description'], "PTS", case=False) & (nba_df['event_type'] == "MAKE") & ~series_contains(nba_df['Next_home_desc'], "S.FOUL", case=True), "Away",
        series_contains(nba_df['home_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False, regex=True), "Home",
        series_contains(nba_df['visitor_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False, regex=True), "Away",
        "" # Default
    )
    print("      Calculated EOP, TeamOnOffense for REB")

    # --- Propagation Check ---
    print("      Skipping Player Propagation/FT Check for REB (as per R script structure)")
    nba_df_checked = nba_df

    # --- Map Players ---
    # ... (map O/D players remains the same) ...
    o_mapping = {}
    d_mapping = {}
    for i in range(1, 6):
        o_mapping[f'O{i}'] = np.where(nba_df_checked['TeamOnOffense'] == "Away", nba_df_checked[f'a{i}'],
                                        np.where(nba_df_checked['TeamOnOffense'] == "Home", nba_df_checked[f'h{i}'], np.nan))
        d_mapping[f'D{i}'] = np.where(nba_df_checked['TeamOnOffense'] == "Away", nba_df_checked[f'h{i}'],
                                        np.where(nba_df_checked['TeamOnOffense'] == "Home", nba_df_checked[f'a{i}'], np.nan))
    nba_df_checked = nba_df_checked.assign(**o_mapping, **d_mapping)
    print("      Mapped O/D Players")

    # --- Filter for EOP ---
    # ... (filtering remains the same) ...
    if 'End_of_Possession' in nba_df_checked.columns and pd.api.types.is_bool_dtype(nba_df_checked['End_of_Possession']):
        nba_filt = nba_df_checked[nba_df_checked['End_of_Possession']].copy()
    else:
         print("      Warning: 'End_of_Possession' column missing or not boolean for filtering. Skipping.")
         nba_filt = nba_df_checked.copy()
    print(f"      Filtered down to {len(nba_filt)} REB End of Possession rows")

    # --- Finalize ---
    nba_reb_output = _finalize_df(nba_filt, 'Offensive_Rebound', year)

    end_time = time.time()
    print(f"  Finished REB Processing. Time: {end_time - start_time:.2f} seconds.")
    return nba_reb_output


# --- Process TOV (Turnover) Function ---
def process_tov_py(file_path, year, season_str):
    """ Processes data for Turnover factor calculation. """
    print(f"  Starting TOV Processing for {season_str}...")
    nba_df = _base_processing(file_path, allow_partial_lineups=True)
    if nba_df is None: return None
    start_time = time.time()

    nba_df_checked, _ = prepare_wnba_standard_possession_df(nba_df, "TOV")
    if 'End_of_Possession' in nba_df_checked.columns and pd.api.types.is_bool_dtype(nba_df_checked['End_of_Possession']):
        nba_filt = nba_df_checked[nba_df_checked['End_of_Possession']].copy()
    else:
        print("      Warning: 'End_of_Possession' column missing or not boolean for filtering. Skipping.")
        nba_filt = nba_df_checked.copy()
    print(f"      Filtered down to {len(nba_filt)} robust TOV End of Possession rows")

    nba_filt['Is_Turnover'] = (nba_filt['event_type'] == 'Turnover').astype(int)
    print(f"      Calculated robust Is_Turnover flag (found {nba_filt['Is_Turnover'].sum()})")

    nba_tov_output = _finalize_df(nba_filt, 'Is_Turnover', year)
    end_time = time.time()
    print(f"  Finished TOV Processing. Time: {end_time - start_time:.2f} seconds.")
    return nba_tov_output

    # --- Define Possession Characteristics (TOV Version) --- ### CORRECTED case_when calls ###
    ft_exclude_pattern = r'\b(1 of [23]|2 of 3|Technical|Flagrant)\b'
    nba_df['End_of_Possession'] = case_when(
        (nba_df['event_type'] == "EndOfPeriod") & (nba_df['prev_seconds'] > 0), True,
        series_contains(nba_df['home_description'], r'Flagrant (2 of 2|3 of 3)', case=False, regex=True), True,
        series_contains(nba_df['visitor_description'], r'Flagrant (2 of 2|3 of 3)', case=False, regex=True), True,
        (nba_df['event_type'] == "Turnover"), True,
        series_contains(nba_df['home_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_visitor_desc'], "MISS", case=False) & (nba_df['Prev_Event'] != "FreeThrow"), True,
        series_contains(nba_df['visitor_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_home_desc'], "MISS", case=False) & (nba_df['Prev_Event'] != "FreeThrow"), True,
        series_contains(nba_df['home_description'], "PTS", case=False) & (nba_df['event_type'] == "MAKE") & ~series_contains(nba_df['Next_visitor_desc'], "S.FOUL", case=True), True,
        series_contains(nba_df['visitor_description'], "PTS", case=False) & (nba_df['event_type'] == "MAKE") & ~series_contains(nba_df['Next_home_desc'], "S.FOUL", case=True), True,
        series_contains(nba_df['home_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False, regex=True) & series_contains(nba_df['home_description'], "PTS", case=False) & \
           ~(series_contains(nba_df['Prev_visitor_desc2'], "Transition", case=False) | series_contains(nba_df['Prev_visitor_desc'], "Transition", case=False) | series_contains(nba_df['home_description'], "Flagrant", case=False)), True,
        series_contains(nba_df['visitor_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False, regex=True) & series_contains(nba_df['visitor_description'], "PTS", case=False) & \
           ~(series_contains(nba_df['Prev_home_desc2'], "Transition", case=False) | series_contains(nba_df['Prev_home_desc'], "Transition", case=False) | series_contains(nba_df['visitor_description'], "Flagrant", case=False)), True,
        False # Default
    ).astype(bool)

    nba_df['TeamOnOffense'] = case_when(
        series_contains(nba_df['home_description'], r'Flagrant (2 of 2|3 of 3)', case=False, regex=True), "Home",
        series_contains(nba_df['visitor_description'], r'Flagrant (2 of 2|3 of 3)', case=False, regex=True), "Away",
        series_contains(nba_df['home_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_visitor_desc'], "MISS", case=False), "Away",
        series_contains(nba_df['visitor_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_home_desc'], "MISS", case=False), "Home",
        series_contains(nba_df['home_description'], "Turnover", case=False), "Home",
        series_contains(nba_df['visitor_description'], "Turnover", case=False), "Away",
        series_contains(nba_df['home_description'], "PTS", case=False), "Home",
        series_contains(nba_df['visitor_description'], "PTS", case=False), "Away",
        series_contains(nba_df['home_description'], "MISS", case=False), "Home",
        series_contains(nba_df['visitor_description'], "MISS", case=False), "Away",
        series_contains(nba_df['home_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False, regex=True), "Home",
        series_contains(nba_df['visitor_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False, regex=True), "Away",
        "" # Default
    )
    print("      Calculated EOP, TeamOnOffense for TOV")

    # --- Identify Potential Fouls --- ### CORRECTED case_when call ###
    window_size = 5
    group_cols = ['game_id'] if 'game_id' in nba_df.columns else []
    if group_cols:
        ft_events = nba_df.groupby(group_cols)['event_type'].transform(lambda x: (x == "FreeThrow").rolling(window=window_size, min_periods=1).sum().shift(-(window_size - 1)).fillna(0))
        sub_events = nba_df.groupby(group_cols)['event_type'].transform(lambda x: (x == "Substitution").rolling(window=window_size, min_periods=1).sum().shift(-(window_size - 1)).fillna(0))
    else:
        ft_events = (nba_df['event_type'] == "FreeThrow").rolling(window=window_size, min_periods=1).sum().shift(-(window_size - 1)).fillna(0)
        sub_events = (nba_df['event_type'] == "Substitution").rolling(window=window_size, min_periods=1).sum().shift(-(window_size - 1)).fillna(0)

    exclude_foul_pattern = r'T\.FOUL|FLAGRANT|Offens|Transition'
    nba_df['PotentialFoul'] = case_when(
        (ft_events > 0) & (sub_events > 0) & (nba_df['event_type'] == "Foul") & \
        ~series_contains(nba_df['visitor_description'], exclude_foul_pattern, case=False, regex=True) & \
        ~series_contains(nba_df['home_description'], exclude_foul_pattern, case=False, regex=True), True,
        False # Default
    ).astype(bool)
    print(f"      Identified {nba_df['PotentialFoul'].sum()} potential fouls")

    # --- Apply Propagation & Map Players ---
    nba_df_propagated = propagate_player_values_py(nba_df)
    nba_df_checked = ft_off_check_py(nba_df_propagated)

    # ... (map O/D players remains the same) ...
    o_mapping = {}
    d_mapping = {}
    for i in range(1, 6):
        o_mapping[f'O{i}'] = np.where(nba_df_checked['TeamOnOffense'] == "Away", nba_df_checked[f'a{i}'],
                                        np.where(nba_df_checked['TeamOnOffense'] == "Home", nba_df_checked[f'h{i}'], np.nan))
        d_mapping[f'D{i}'] = np.where(nba_df_checked['TeamOnOffense'] == "Away", nba_df_checked[f'h{i}'],
                                        np.where(nba_df_checked['TeamOnOffense'] == "Home", nba_df_checked[f'a{i}'], np.nan))
    nba_df_checked = nba_df_checked.assign(**o_mapping, **d_mapping)
    print("      Mapped O/D Players")


    # --- Filter for EOP ---
    # ... (filtering remains the same) ...
    if 'End_of_Possession' in nba_df_checked.columns and pd.api.types.is_bool_dtype(nba_df_checked['End_of_Possession']):
        nba_filt = nba_df_checked[nba_df_checked['End_of_Possession']].copy()
    else:
         print("      Warning: 'End_of_Possession' column missing or not boolean for filtering. Skipping.")
         nba_filt = nba_df_checked.copy()
    print(f"      Filtered down to {len(nba_filt)} TOV End of Possession rows")

    # --- Create Turnover Numerator ---
    nba_filt['Is_Turnover'] = (nba_filt['event_type'] == 'Turnover').astype(int)
    print(f"      Calculated Is_Turnover flag (found {nba_filt['Is_Turnover'].sum()})")


    # --- Finalize ---
    nba_tov_output = _finalize_df(nba_filt, 'Is_Turnover', year)

    end_time = time.time()
    print(f"  Finished TOV Processing. Time: {end_time - start_time:.2f} seconds.")
    return nba_tov_output


# --- Main Execution Block ---
# ... (main execution block remains the same) ...

# --- Final Output Preparation Helper ---
def _finalize_df(df, numerator_col, year, id_cols=['game_id'], o_cols=[f'O{i}' for i in range(1,6)], d_cols=[f'D{i}' for i in range(1,6)]):
    """Handles common final steps: joining external data, selecting columns, filtering."""
    print("      Running Finalization...")
    if df is None or df.empty:
        print("      Skipping finalization: Input DataFrame is empty or None.")
        return None
    allow_partial_lineups = (
        'allow_partial_lineups' in df.columns
        and df['allow_partial_lineups'].fillna(False).astype(bool).any()
    )
    if allow_partial_lineups:
        lineup_cols = [col for col in o_cols + d_cols if col in df.columns]
        df[lineup_cols] = df[lineup_cols].replace('', np.nan).fillna(0)

    # --- Load and Merge Game Dates ---
    game_dates_df = load_game_dates() # Load (or get from cache)

    if game_dates_df is not None and 'game_id' in df.columns:
        # Filter game_dates for the relevant season (using the 'year' passed in)
        filtered_dates_df = game_dates_df[game_dates_df['season'] == year]

        if not filtered_dates_df.empty:
            print(f"        Merging with {len(filtered_dates_df)} game date entries for season {year}...")

            # Ensure game_id in main df is integer for merging
            if not pd.api.types.is_integer_dtype(df['game_id']):
                print(f"        Converting main df game_id from {df['game_id'].dtype} to numeric/int...")
                df['game_id'] = pd.to_numeric(df['game_id'], errors='coerce')
                if df['game_id'].isna().any():
                    print(f"        Warning: Found {df['game_id'].isna().sum()} NaN game_ids after conversion. Filling with -1 before int conversion.")
                    df['game_id'] = df['game_id'].fillna(-1)
                df['game_id'] = df['game_id'].astype(int)


            original_rows = len(df)
            # Perform the merge
            df = pd.merge(
                df,
                filtered_dates_df[['GAME_ID', 'game_date']],    
                left_on='game_id',
                right_on='GAME_ID',
                how='left'
            )
            if 'GAME_ID' in df.columns:
                df = df.drop(columns=['GAME_ID'])

            rows_after_merge = len(df)
            if rows_after_merge != original_rows:
                 print(f"        Warning: Row count changed during game_date merge ({original_rows} -> {rows_after_merge}). Check for duplicate game_ids?")
            merged_dates_count = df['game_date'].notna().sum()
            print(f"        Successfully merged game dates for {merged_dates_count} rows.")
            missing_dates = len(df) - merged_dates_count
            if missing_dates > 0:
                 print(f"        Warning: Could not find matching game dates for {missing_dates} rows.")

        else:
             print(f"        Warning: No game dates found for season {year} in the loaded schedule data.")
             if 'game_date' not in df.columns: df['game_date'] = pd.NA # Use pandas NA for missing dates

    elif 'game_date' not in df.columns:
        print("       Warning: Game dates CSV failed to load or game_id missing. Adding empty game_date column.")
        df['game_date'] = pd.NA


    df['Season'] = year + 1 # R code used year + 1

    # --- Select Final Columns ---
    # Construct the list of columns needed
    final_cols = id_cols + o_cols + d_cols

    # Add user requested columns for score, quarter, and time
    # These columns should be present in the df if they were in the original CSV
    # and not explicitly dropped.
    game_state_cols = ['score', 'period', 'time_quarter','away_score','home_score','score_margin','event_num'] # ADDED THESE
    final_cols.extend(game_state_cols) # ADDED THESE


    # Add numerator column(s) safely
    if isinstance(numerator_col, list):
        final_cols.extend(numerator_col)
    else:
        final_cols.append(numerator_col)

    if 'game_date' in df.columns: final_cols.append('game_date')
    final_cols.append('Season')

    # Ensure columns exist before selecting
    # This will also check if 'score', 'period', 'time_quarter' are available
    final_cols_present = [col for col in final_cols if col in df.columns]
    missing_final_cols = [col for col in final_cols if col not in final_cols_present]
    if missing_final_cols:
        print(f"      Warning: Missing expected final columns: {missing_final_cols}. These columns will not be in the output.")
        # If 'score', 'period', 'time_quarter' are in missing_final_cols,
        # it means they were not available in the DataFrame 'df' passed to this function.
        # This could be due to them not being in the input CSV, or being dropped earlier.

    # Select only existing columns
    df_final = df[final_cols_present].copy()


    # --- Final Filtering ---
    if allow_partial_lineups:
        initial_rows = len(df_final)
        present_o_cols = [col for col in o_cols if col in df_final.columns]
        present_d_cols = [col for col in d_cols if col in df_final.columns]
        off_known = (df_final[present_o_cols].apply(pd.to_numeric, errors='coerce').fillna(0) > 0).sum(axis=1)
        def_known = (df_final[present_d_cols].apply(pd.to_numeric, errors='coerce').fillna(0) > 0).sum(axis=1)
        df_final = df_final[(off_known >= 4) & (def_known >= 4)].copy()
        print(
            "      Partial-lineup mode enabled; kept rows with at least 4 known O and D players "
            f"({len(df_final)} rows, removed {initial_rows - len(df_final)})."
        )
    elif 'D5' in df_final.columns:
        initial_rows = len(df_final)
        # Robustly filter out rows where D5 is considered empty (NaN, None, 0, empty string)
        df_final = df_final[~df_final['D5'].isin([0, '0', '', None, np.nan])]
        df_final = df_final.dropna(subset=['D5']) # Also remove if it became NaN during other ops

        print(f"      Rows after final D5 != 0/Null/Empty filter: {len(df_final)} (removed {initial_rows - len(df_final)})")
    else:
        print("      Warning: D5 column not found for final filtering.")


    print("      Finished Finalization.")
    return df_final



# --- Main Execution Block ---
if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Process NBA play-by-play data for RAPM, TS, REB, and TOV analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_rapm2.py NBA26.csv           # Auto-detects 2025-26 season
  python process_rapm2.py NBA23_PS.csv        # Auto-detects 2022-23 playoffs
  python process_rapm2.py NBA2024.csv         # Auto-detects 2023-24 season
  
  # Manual override (optional):
  python process_rapm2.py NBA26.csv --year 2025 --season "2025-26"
        """
    )
    
    parser.add_argument('input_file', 
                       help='Path to the input CSV file (e.g., NBA26.csv, NBA23_PS.csv)')
    parser.add_argument('--year', '-y', 
                       type=int,
                       required=False,
                       help='(Optional) Starting year of the season - auto-inferred from filename if not provided')
    parser.add_argument('--season', '-s',
                       required=False,
                       help='(Optional) Season string in format YYYY-YY - auto-inferred from filename if not provided')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.input_file):
        print(f"ERROR: Input file '{args.input_file}' not found. Please provide the correct path.")
        exit(1)
    
    # Infer year and season from filename if not provided
    if args.year is None or args.season is None:
        # Extract year from filename pattern NBAXX or NBAXXXX or WNBAXX (with optional _PS suffix)
        filename = os.path.basename(args.input_file)
        # Match NBA/WNBA followed by 2 or 4 digits, followed by anything and .csv
        year_match = re.search(r'(?:NBA|WNBA)(\d{2,4}).*\.csv', filename, re.IGNORECASE)
        
        if year_match:
            year_str = year_match.group(1)
            is_wnba = "WNBA" in filename.upper()
            
            if len(year_str) == 2:
                ending_year = 2000 + int(year_str)
            else:
                ending_year = int(year_str)
            
            if is_wnba:
                starting_year = ending_year
                season_str = str(ending_year)
            else:
                starting_year = ending_year - 1
                end_year_short = ending_year % 100
                season_str = f"{starting_year}-{end_year_short:02d}"
            
            if args.year is None:
                args.year = starting_year
            
            if args.season is None:
                args.season = season_str
            
            print(f"Auto-detected from filename: Year={args.year}, Season={args.season}")
        else:
            # Could not infer from filename
            print(f"ERROR: Could not auto-detect year/season from filename '{filename}'.")
            print(f"Expected format: NBAXX.csv or NBAXX_PS.csv (e.g., NBA26.csv, NBA23_PS.csv)")
            print(f"Or provide --year and --season arguments explicitly.")
            print(f"Example: python process_rapm2.py {args.input_file} --year 2025 --season '2025-26'")
            exit(1)
    
    file_to_process = args.input_file
    processing_year = args.year
    processing_season = args.season
    
    print(f"\n--- Starting All Processing for {processing_season} using {file_to_process} ---")
    print(f"--- Year: {processing_year}, Season: {processing_season} ---\n")

    # --- Run Each Process ---
    rapm_df = process_rapm_py(file_to_process, processing_year, processing_season)
    ts_df = process_ts_py(file_to_process, processing_year, processing_season)
    reb_df = process_reb_py(file_to_process, processing_year, processing_season)
    tov_df = process_tov_py(file_to_process, processing_year, processing_season)

    # --- Save Results ---
    # Ensure Processed directory exists
    out_dir = "wnba_test/Processed" if "WNBA" in file_to_process.upper() else "Processed"
    os.makedirs(out_dir, exist_ok=True)
    
    # Check if input file has _PS suffix
    input_basename = os.path.basename(file_to_process)
    input_name_without_ext = os.path.splitext(input_basename)[0]
    ps_suffix = "_PS" if input_name_without_ext.endswith("_PS") else ""
    
    outputs = {
        "RAPM": rapm_df,
        "TS": ts_df,
        "REB": reb_df,
        "TOV": tov_df
    }

    for name, df in outputs.items():
        if df is not None and not df.empty:
            # Create filename like RAPM24.csv from 2023-24 season
            if "-" in processing_season:
                season_suffix = processing_season.split('-')[1] # Get '24'
            else:
                season_suffix = processing_season[-2:] # Get '24' from '2024'
            output_filename = f"{out_dir}/{name}{season_suffix}{ps_suffix}.csv"
            try:
                df.to_csv(output_filename, index=False)
                print(f"--- Saved {name} results to {output_filename} ({len(df)} rows) ---")
            except Exception as e:
                print(f"--- Error saving {output_filename}: {e} ---")
        else:
            print(f"--- No results generated or DataFrame empty for {name} {processing_season}. ---")

    print(f"\n--- Finished All Processing for {processing_season} ---")
