import pandas as pd
import numpy as np
import re
import time
import warnings
import requests
import io
import argparse
import os
from pathlib import Path
# from nba_api.stats.endpoints import leaguedashplayerstats # Removed to use Supabase
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the pipeline root directory (parent of scripts/)
SCRIPT_DIR = Path(__file__).parent
PIPELINE_ROOT = SCRIPT_DIR.parent
RAW_DATA_DIR = PIPELINE_ROOT / "raw_data"
PROCESSED_DIR = PIPELINE_ROOT / "processed"



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

def fetch_player_stats_supabase(year, is_playoffs):
    """
    Fetches player stats (3P_PERC, FT_PERC) from Supabase table 'player_stats'.
    """
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    
    if not url or not key:
        print("      Warning: SUPABASE_URL or SUPABASE_KEY missing in .env. Falling back to defaults.")
        return None

    try:
        supabase: Client = create_client(url, key)
        playoff_flag = 1 if is_playoffs else 0
        
        # Table uses ending year convention (2026 for 2025-26 season), so add 1 to starting year
        query_year = year + 1
        print(f"      Querying Supabase for player stats (Year: {query_year}, Playoffs: {playoff_flag})...")
        
        # Pull only necessary columns to keep payload small
        response = supabase.table("player_stats_with_metrics") \
            .select("nba_id, year, playoffs, \"3P_PERC\", \"FT_PERC\"") \
            .eq("year", query_year) \
            .eq("playoffs", playoff_flag) \
            .execute()
        
        if response.data:
            df = pd.DataFrame(response.data)
            # Rename to match expected names in script
            df = df.rename(columns={
                '3P_PERC': 'ThreePerc',
                'FT_PERC': 'FTPerc',
                'nba_id': 'PlayerID'
            })
            
            # Deduplicate by PlayerID to avoid inflating rows during merge
            # (In case a player has multiple team entries in the stats view)
            initial_count = len(df)
            df = df.drop_duplicates(subset=['PlayerID'], keep='first')
            if len(df) < initial_count:
                print(f"      Deduplicated player stats: {initial_count} -> {len(df)} unique players.")
                
            print(f"      Successfully fetched {len(df)} player stat entries from Supabase.")
            return df
        else:
            print("      Warning: Supabase returned no data for player stats.")
            return None
            
    except Exception as e:
        print(f"      Error fetching player stats from Supabase: {e}")
        return None

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
def _base_processing(file_path):
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
    # ... (aliasing code remains the same) ...
    alias_map = {f"{team}_player{i}": f"{prefix}{i}" for team, prefix in [("away", "a"), ("home", "h")] for i in range(1, 6)}
    nba_df = nba_df.rename(columns=alias_map)
    print("      Aliased player columns (a1-h5)")

    print(f"    Finished Base Processing for {file_path}.")
    return nba_df
# --- Process RAPM Function ---
def process_rapm_py(file_path, year, season_str, o_luck=1.0, d_luck=1.0):
    """
    Processes data for standard RAPM calculation.
    
    Args:
        file_path: Path to input CSV
        year: Starting year of season (e.g., 2025 for 2025-26)
        season_str: Season string (e.g., "2025-26")
        o_luck: Offensive luck adjustment weight (0.0 = no adjustment, 1.0 = full adjustment)
        d_luck: Defensive luck adjustment weight (0.0 = no adjustment, 1.0 = full adjustment)
    """
    print(f"  Starting RAPM Processing for {season_str} (o_luck={o_luck}, d_luck={d_luck})...")
    nba_df = _base_processing(file_path)
    if nba_df is None: return None
    start_time = time.time()

    # --- Define Possession Characteristics --- ### CORRECTED case_when calls ###
    ft_exclude_pattern = r'\b(1 of [23]|2 of 3|Technical|Flagrant)\b'

    nba_df['offensive_FT_Rebound'] = case_when(
        series_contains(nba_df['visitor_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_visitor_desc'], "MISS", case=False) & series_contains(nba_df['Prev_visitor_desc'], "Free Throw", case=False) & ~series_contains(nba_df['Prev_visitor_desc'], ft_exclude_pattern, case=False, regex=True), True,
        series_contains(nba_df['home_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_home_desc'], "MISS", case=False) & series_contains(nba_df['Prev_home_desc'], "Free Throw", case=False) & ~series_contains(nba_df['Prev_home_desc'], ft_exclude_pattern, case=False, regex=True), True,
        False # Default
    ).astype(bool)

    # FIX: Added Flagrant 1 of 1 to pattern (was only matching 2 of 2|3 of 3)
    # FIX: Changed S.FOUL check from case=True to case=False for case-insensitive matching
    nba_df['End_of_Possession'] = case_when(
        (nba_df['event_type'] == "EndOfPeriod") & (nba_df['prev_seconds'] > 0), True,
        series_contains(nba_df['home_description'], r'Flagrant (1 of 1|2 of 2|3 of 3)', case=False, regex=True), True,
        series_contains(nba_df['visitor_description'], r'Flagrant (1 of 1|2 of 2|3 of 3)', case=False, regex=True), True,
        (nba_df['event_type'] == "Turnover"), True,
        series_contains(nba_df['home_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_visitor_desc'], "MISS", case=False) & ~series_contains(nba_df['Prev_visitor_desc'], ft_exclude_pattern, case=False, regex=True), True,
        series_contains(nba_df['visitor_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_home_desc'], "MISS", case=False) & ~series_contains(nba_df['Prev_home_desc'], ft_exclude_pattern, case=False, regex=True), True,
        series_contains(nba_df['home_description'], "PTS", case=False) & (nba_df['event_type'] == "MAKE") & ~series_contains(nba_df['Next_visitor_desc'], "S.FOUL", case=False), True,
        series_contains(nba_df['visitor_description'], "PTS", case=False) & (nba_df['event_type'] == "MAKE") & ~series_contains(nba_df['Next_home_desc'], "S.FOUL", case=False), True,
        series_contains(nba_df['home_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False, regex=True) & series_contains(nba_df['home_description'], "PTS", case=False) & \
            ~(series_contains(nba_df['Prev_visitor_desc2'], "Transition", case=False) | series_contains(nba_df['Prev_visitor_desc'], "Transition", case=False) | series_contains(nba_df['home_description'], "Flagrant", case=False)), True,
        series_contains(nba_df['visitor_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False, regex=True) & series_contains(nba_df['visitor_description'], "PTS", case=False) & \
            ~(series_contains(nba_df['Prev_home_desc2'], "Transition", case=False) | series_contains(nba_df['Prev_home_desc'], "Transition", case=False) | series_contains(nba_df['visitor_description'], "Flagrant", case=False)), True,
        False # Default
    ).astype(bool)

    # FIX: Same fixes applied to TeamOnOffense
    nba_df['TeamOnOffense'] = case_when(
        series_contains(nba_df['home_description'], r'Flagrant (1 of 1|2 of 2|3 of 3)', case=False, regex=True), "Home",
        series_contains(nba_df['visitor_description'], r'Flagrant (1 of 1|2 of 2|3 of 3)', case=False, regex=True), "Away",
        series_contains(nba_df['home_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_visitor_desc'], "MISS", case=False), "Away",
        series_contains(nba_df['visitor_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_home_desc'], "MISS", case=False), "Home",
        series_contains(nba_df['home_description'], "Turnover", case=False), "Home",
        series_contains(nba_df['visitor_description'], "Turnover", case=False), "Away",
        series_contains(nba_df['home_description'], "PTS", case=False) & (nba_df['event_type'] == "MAKE") & ~series_contains(nba_df['Next_visitor_desc'], "S.FOUL", case=False), "Home",
        series_contains(nba_df['visitor_description'], "PTS", case=False) & (nba_df['event_type'] == "MAKE") & ~series_contains(nba_df['Next_home_desc'], "S.FOUL", case=False), "Away",
        series_contains(nba_df['home_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False, regex=True) & series_contains(nba_df['home_description'], "PTS", case=False) & \
            ~(series_contains(nba_df['Prev_visitor_desc2'], "Transition", case=False) | series_contains(nba_df['Prev_visitor_desc'], "Transition", case=False) | series_contains(nba_df['home_description'], "Flagrant", case=False)), "Home",
        series_contains(nba_df['visitor_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False, regex=True) & series_contains(nba_df['visitor_description'], "PTS", case=False) & \
            ~(series_contains(nba_df['Prev_home_desc2'], "Transition", case=False) | series_contains(nba_df['Prev_home_desc'], "Transition", case=False) | series_contains(nba_df['visitor_description'], "Flagrant", case=False)), "Away",
        "" # Default
    )
    print("      Calculated EOP, TeamOnOffense for RAPM")

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

    # Map Offensive and Defensive players based on TeamOnOffense
    o_mapping = {}
    d_mapping = {}
    for i in range(1, 6):
        o_mapping[f'O{i}'] = np.where(nba_df_checked['TeamOnOffense'] == "Away", nba_df_checked[f'a{i}'],
                                        np.where(nba_df_checked['TeamOnOffense'] == "Home", nba_df_checked[f'h{i}'], np.nan)) # Use NaN/None if no team
        d_mapping[f'D{i}'] = np.where(nba_df_checked['TeamOnOffense'] == "Away", nba_df_checked[f'h{i}'],
                                        np.where(nba_df_checked['TeamOnOffense'] == "Home", nba_df_checked[f'a{i}'], np.nan)) # Use NaN/None if no team
    nba_df_checked = nba_df_checked.assign(**o_mapping, **d_mapping)
    print("      Mapped O/D Players")

    # --- Join Player Season Stats (FTPerc, ThreePerc) from Supabase ---
    is_playoffs = "_PS" in os.path.basename(file_path).upper()
    player_stats_df = fetch_player_stats_supabase(year, is_playoffs)

    # --- Merge Player Stats ---
    shooter_id_col = 'player1_id'
    if player_stats_df is not None and shooter_id_col in nba_df_checked.columns:
        print("      Preparing data for player stats merge...")
        nba_df_checked['player1_id_numeric'] = pd.to_numeric(nba_df_checked[shooter_id_col], errors='coerce')
        player_stats_df['PlayerID'] = pd.to_numeric(player_stats_df['PlayerID'], errors='coerce').astype('Int64')

        # Drop rows where conversion failed in either df to ensure clean merge
        nba_df_checked = nba_df_checked.dropna(subset=['player1_id_numeric'])
        player_stats_df = player_stats_df.dropna(subset=['PlayerID'])
        nba_df_checked['player1_id_numeric'] = nba_df_checked['player1_id_numeric'].astype('Int64')

        print(f"      Merging player stats onto {len(nba_df_checked)} rows...")
        nba_df_checked = pd.merge(
            nba_df_checked,
            player_stats_df[['PlayerID', 'FTPerc', 'ThreePerc']],
            left_on='player1_id_numeric',
            right_on='PlayerID',
            how='left'
        )
        nba_df_checked = nba_df_checked.drop(columns=['player1_id_numeric', 'PlayerID'], errors='ignore')
        print(f"      Successfully merged player stats for {nba_df_checked['FTPerc'].notna().sum()} rows.")
    else:
        print(f"      Skipping player stats merge (Stats data unavailable or '{shooter_id_col}' missing). Using defaults.")
        if 'FTPerc' not in nba_df_checked.columns: nba_df_checked['FTPerc'] = np.nan
        if 'ThreePerc' not in nba_df_checked.columns: nba_df_checked['ThreePerc'] = np.nan

    # --- Calculate Expected Points based on Stats (Handle Missing) ---
    default_ft_perc = 0.75
    default_3p_perc = 0.35

    nba_df_checked['ExpFT'] = nba_df_checked['FTPerc'].fillna(default_ft_perc).astype(float)
    nba_df_checked['Exp3PT'] = nba_df_checked['ThreePerc'].fillna(default_3p_perc).astype(float)

    print(f"      Assigned ExpFT (default: {default_ft_perc}) and Exp3PT (default: {default_3p_perc}).")

    is_ft_event = series_contains(nba_df_checked['home_description'], "Free Throw", case=False) | \
                  series_contains(nba_df_checked['visitor_description'], "Free Throw", case=False) | \
                  (nba_df_checked['event_type'] == "FreeThrow")
    is_2pt_make_event = (series_contains(nba_df_checked['home_description'], "PTS", case=False) & ~series_contains(nba_df_checked['home_description'], "3PT", case=False) & (nba_df_checked['event_type'] == "MAKE")) | \
                        (series_contains(nba_df_checked['visitor_description'], "PTS", case=False) & ~series_contains(nba_df_checked['visitor_description'], "3PT", case=False) & (nba_df_checked['event_type'] == "MAKE"))
    is_3pt_event = series_contains(nba_df_checked['home_description'], "3PT", case=False) | \
                   series_contains(nba_df_checked['visitor_description'], "3PT", case=False)
    actual_score = (nba_df_checked['Home_Action_Score'] + nba_df_checked['Away_Action_Score']).astype(float)

    # --- NO LUCK ADJUSTMENT VERSION (Pure Actuals) ---
    # Off_Diff and Def_Diff will just be actual points scored
    nba_df_checked['ActualNet'] = actual_score  # Same for both offense and defense
    
    # --- LUCK ADJUSTMENT VERSION ---
    # Uses o_luck and d_luck parameters to blend expected vs actual
    # o_luck/d_luck = 0.0: 100% actual (no adjustment)
    # o_luck/d_luck = 1.0: 100% expected (full adjustment)
    exp_ft_value = nba_df_checked['ExpFT']  # Expected points per FT attempt
    exp_3pt_value = nba_df_checked['Exp3PT'] * 3.0  # Expected points per 3PT attempt
    
    # Offensive luck adjustment
    nba_df_checked['LA_OffNet'] = case_when(
        is_ft_event, o_luck * exp_ft_value + (1 - o_luck) * actual_score,
        is_2pt_make_event, 2.0,  # No luck adjustment for 2PT
        is_3pt_event, o_luck * exp_3pt_value + (1 - o_luck) * actual_score,
        0.0 # Default
    )
    
    # Defensive luck adjustment
    nba_df_checked['LA_DefNet'] = case_when(
        is_ft_event, d_luck * exp_ft_value + (1 - d_luck) * actual_score,
        is_2pt_make_event, 2.0,  # No luck adjustment for 2PT
        is_3pt_event, d_luck * exp_3pt_value + (1 - d_luck) * actual_score,
        0.0 # Default
    )
    print(f"      Calculated ActualNet (no LA) and LA_OffNet/LA_DefNet (o_luck={o_luck}, d_luck={d_luck})")

    # --- Calculate cumulative totals for BOTH versions ---
    if group_cols:
        # No LA version
        nba_df_checked['ActualTotal'] = nba_df_checked.groupby(group_cols)['ActualNet'].cumsum()
        # LA version
        nba_df_checked['LA_OffTotal'] = nba_df_checked.groupby(group_cols)['LA_OffNet'].cumsum()
        nba_df_checked['LA_DefTotal'] = nba_df_checked.groupby(group_cols)['LA_DefNet'].cumsum()
    else:
        nba_df_checked['ActualTotal'] = nba_df_checked['ActualNet'].cumsum()
        nba_df_checked['LA_OffTotal'] = nba_df_checked['LA_OffNet'].cumsum()
        nba_df_checked['LA_DefTotal'] = nba_df_checked['LA_DefNet'].cumsum()
    print("      Calculated cumulative totals for both versions")


    # --- Filter for EOP & Calculate Diffs ---
    if 'End_of_Possession' in nba_df_checked.columns and pd.api.types.is_bool_dtype(nba_df_checked['End_of_Possession']):
        nba_filt = nba_df_checked[nba_df_checked['End_of_Possession']].copy()
    else:
         print("      Warning: 'End_of_Possession' column missing or not boolean for filtering. Skipping.")
         nba_filt = nba_df_checked.copy()
    print(f"      Filtered down to {len(nba_filt)} RAPM End of Possession rows")

    if not nba_filt.empty:
        # --- NO LUCK ADJUSTMENT DIFFS (Pure Actuals) ---
        # For no-LA version: Off_Diff = Def_Diff = actual points per possession
        if 'ActualTotal' in nba_filt.columns:
            if group_cols:
                nba_filt['Off_Diff'] = nba_filt['ActualTotal'] - nba_filt.groupby(group_cols)['ActualTotal'].shift(fill_value=0)
            else:
                nba_filt['Off_Diff'] = nba_filt['ActualTotal'] - nba_filt['ActualTotal'].shift(fill_value=0)
            nba_filt['Def_Diff'] = nba_filt['Off_Diff']  # Same as Off_Diff for no-LA
        
        # Also keep Net_Diff for reference (same as Off_Diff for no-LA)
        if 'Net_Total' in nba_filt.columns:
            if group_cols:
                nba_filt['Net_Diff'] = nba_filt['Net_Total'] - nba_filt.groupby(group_cols)['Net_Total'].shift(fill_value=0)
            else:
                nba_filt['Net_Diff'] = nba_filt['Net_Total'] - nba_filt['Net_Total'].shift(fill_value=0)
        
        # --- LUCK ADJUSTMENT DIFFS ---
        if 'LA_OffTotal' in nba_filt.columns and 'LA_DefTotal' in nba_filt.columns:
            if group_cols:
                nba_filt['LA_Off_Diff'] = nba_filt['LA_OffTotal'] - nba_filt.groupby(group_cols)['LA_OffTotal'].shift(fill_value=0)
                nba_filt['LA_Def_Diff'] = nba_filt['LA_DefTotal'] - nba_filt.groupby(group_cols)['LA_DefTotal'].shift(fill_value=0)
            else:
                nba_filt['LA_Off_Diff'] = nba_filt['LA_OffTotal'] - nba_filt['LA_OffTotal'].shift(fill_value=0)
                nba_filt['LA_Def_Diff'] = nba_filt['LA_DefTotal'] - nba_filt['LA_DefTotal'].shift(fill_value=0)
        
        print("      Calculated diffs for both No-LA and LA versions")
    else:
         print("      Warning: DataFrame is empty after EOP filter. Skipping diff calculation.")


    # --- Finalize BOTH versions ---
    # Version 1: No Luck Adjustment (RAPMXX.csv)
    no_la_cols = ['Net_Diff', 'Off_Diff', 'Def_Diff']
    no_la_cols_present = [c for c in no_la_cols if c in nba_filt.columns]
    if not no_la_cols_present:
        print("      Error: Could not find diff columns for No-LA output.")
        nba_rapm_no_la = None
    else:
        nba_rapm_no_la = _finalize_df(nba_filt, no_la_cols_present, year)
    
    # Version 2: With Luck Adjustment (LA_RAPMXX.csv)
    # Replace Off_Diff/Def_Diff with LA versions for finalization
    nba_filt_la = nba_filt.copy()
    if 'LA_Off_Diff' in nba_filt_la.columns:
        nba_filt_la['Off_Diff'] = nba_filt_la['LA_Off_Diff']
    if 'LA_Def_Diff' in nba_filt_la.columns:
        nba_filt_la['Def_Diff'] = nba_filt_la['LA_Def_Diff']
    
    la_cols = ['Net_Diff', 'Off_Diff', 'Def_Diff']
    la_cols_present = [c for c in la_cols if c in nba_filt_la.columns]
    if not la_cols_present:
        print("      Error: Could not find diff columns for LA output.")
        nba_rapm_la = None
    else:
        nba_rapm_la = _finalize_df(nba_filt_la, la_cols_present, year)

    end_time = time.time()
    print(f"  Finished RAPM Processing. Time: {end_time - start_time:.2f} seconds.")
    
    # Return both versions as a tuple
    return nba_rapm_no_la, nba_rapm_la

# --- Process TS (True Shooting) Function ---
def process_ts_py(file_path, year, season_str):
    """ Processes data for True Shooting calculation. """
    print(f"  Starting TS Processing for {season_str}...")
    nba_df = _base_processing(file_path)
    if nba_df is None: return None
    start_time = time.time()

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
    nba_df = _base_processing(file_path)
    if nba_df is None: return None
    start_time = time.time()

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
    nba_df = _base_processing(file_path)
    if nba_df is None: return None
    start_time = time.time()

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


    # --- Final Filtering (D5 != 0 or NaN/blank/Null) ---
    # R code checks `D5 != 0`. This likely assumes D5 contains player IDs (numeric)
    # If names are mapped, this check needs adjustment. Let's filter based on non-empty/non-zero D5.
    if 'D5' in df_final.columns:
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
  python 02_process_rapm.py NBA26.csv           # Auto-detects 2025-26 season
  python 02_process_rapm.py NBA23_PS.csv        # Auto-detects 2022-23 playoffs
  
  # With luck adjustment parameters:
  python 02_process_rapm.py NBA26.csv --o-luck 1.0 --d-luck 1.0   # Full LA for both
  python 02_process_rapm.py NBA26.csv --o-luck 0.5 --d-luck 1.0   # 50% off, 100% def
  python 02_process_rapm.py NBA26.csv --o-luck 0.0 --d-luck 0.0   # No LA (pure actuals)
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
    parser.add_argument('--o-luck',
                       type=float,
                       default=1.0,
                       help='Offensive luck adjustment (0.0=no adjustment, 1.0=full adjustment). Default: 1.0')
    parser.add_argument('--d-luck',
                       type=float,
                       default=1.0,
                       help='Defensive luck adjustment (0.0=no adjustment, 1.0=full adjustment). Default: 1.0')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.input_file):
        print(f"ERROR: Input file '{args.input_file}' not found. Please provide the correct path.")
        exit(1)
    
    # Infer year and season from filename if not provided
    if args.year is None or args.season is None:
        # Extract year from filename pattern NBAXX or NBAXXXX (with optional _PS suffix)
        # Examples: NBA26.csv, NBA26_PS.csv, NBA2024.csv
        filename = os.path.basename(args.input_file)
        # Match NBA followed by 2 or 4 digits, optionally followed by _PS
        year_match = re.search(r'NBA(\d{2,4})(?:_PS)?\.csv', filename, re.IGNORECASE)
        
        if year_match:
            year_str = year_match.group(1)
            if len(year_str) == 2:
                # Two-digit year (e.g., 26 -> ending year 2026, so season is 2025-26)
                ending_year = 2000 + int(year_str)
                starting_year = ending_year - 1
            else:
                # Four-digit year (e.g., 2024 -> ending year 2024, so season is 2023-24)
                ending_year = int(year_str)
                starting_year = ending_year - 1
            
            if args.year is None:
                args.year = starting_year
            
            if args.season is None:
                end_year_short = ending_year % 100
                args.season = f"{starting_year}-{end_year_short:02d}"
            
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
    o_luck = args.o_luck
    d_luck = args.d_luck
    
    print(f"\n--- Starting All Processing for {processing_season} using {file_to_process} ---")
    print(f"--- Year: {processing_year}, Season: {processing_season} ---")
    print(f"--- Luck Adjustment: o_luck={o_luck}, d_luck={d_luck} ---\n")

    # --- Run Each Process ---
    # RAPM now returns a tuple: (no_la_df, la_df)
    rapm_result = process_rapm_py(file_to_process, processing_year, processing_season, o_luck=o_luck, d_luck=d_luck)
    if rapm_result is not None:
        rapm_no_la_df, rapm_la_df = rapm_result
    else:
        rapm_no_la_df, rapm_la_df = None, None
    
    ts_df = process_ts_py(file_to_process, processing_year, processing_season)
    reb_df = process_reb_py(file_to_process, processing_year, processing_season)
    tov_df = process_tov_py(file_to_process, processing_year, processing_season)

    # --- Save Results ---
    # Ensure Processed directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if input file has _PS suffix
    input_basename = os.path.basename(file_to_process)
    input_name_without_ext = os.path.splitext(input_basename)[0]
    ps_suffix = "_PS" if input_name_without_ext.endswith("_PS") else ""
    
    # Get season suffix for filenames
    season_suffix = processing_season.split('-')[1]  # Get '26' from '2025-26'
    
    # Save RAPM files (both versions)
    if rapm_no_la_df is not None and not rapm_no_la_df.empty:
        output_filename = PROCESSED_DIR / f"RAPM{season_suffix}{ps_suffix}.csv"
        try:
            rapm_no_la_df.to_csv(output_filename, index=False)
            print(f"--- Saved RAPM (No LA) to {output_filename} ({len(rapm_no_la_df)} rows) ---")
        except Exception as e:
            print(f"--- Error saving {output_filename}: {e} ---")
    else:
        print(f"--- No results for RAPM (No LA) {processing_season}. ---")
    
    if rapm_la_df is not None and not rapm_la_df.empty:
        output_filename = PROCESSED_DIR / f"LA_RAPM{season_suffix}{ps_suffix}.csv"
        try:
            rapm_la_df.to_csv(output_filename, index=False)
            print(f"--- Saved LA_RAPM (With LA) to {output_filename} ({len(rapm_la_df)} rows) ---")
        except Exception as e:
            print(f"--- Error saving {output_filename}: {e} ---")
    else:
        print(f"--- No results for LA_RAPM {processing_season}. ---")
    
    # Save other outputs (TS, REB, TOV)
    other_outputs = {
        "TS": ts_df,
        "REB": reb_df,
        "TOV": tov_df
    }

    for name, df in other_outputs.items():
        if df is not None and not df.empty:
            output_filename = PROCESSED_DIR / f"{name}{season_suffix}{ps_suffix}.csv"
            try:
                df.to_csv(output_filename, index=False)
                print(f"--- Saved {name} results to {output_filename} ({len(df)} rows) ---")
            except Exception as e:
                print(f"--- Error saving {output_filename}: {e} ---")
        else:
            print(f"--- No results generated or DataFrame empty for {name} {processing_season}. ---")

    print(f"\n--- Finished All Processing for {processing_season} ---")