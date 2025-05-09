import pandas as pd
import numpy as np
import re
import time
import warnings
import requests
import io
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

# --- Player Propagation and FT Check Functions (from previous response) ---
# Assume propagate_player_values_py and ft_off_check_py are defined here
# (These are complex and copied from the previous response for brevity)
# ... (propagate_player_values_py definition) ...
# ... (ft_off_check_py definition) ...
# --- pholder for the actual function code from previous response ---
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
        game_dates_df['date'] = pd.to_datetime(game_dates_df['date'].astype(str), format='%Y%m%d').dt.date.astype(str)

    
        game_dates_df['season']=game_dates_df['season'].str.split('-').str[0]
        game_dates_df['season']=game_dates_df['season'].astype(int)
        print(f"        Successfully loaded {len(game_dates_df)} rows from game dates CSV.")

        # --- Preprocessing ---
        # Ensure GAME_ID is numeric, drop rows where conversion fails
        game_dates_df['GAME_ID'] = pd.to_numeric(game_dates_df['GAME_ID'], errors='coerce')
        game_dates_df = game_dates_df.dropna(subset=['GAME_ID'])
        game_dates_df['GAME_ID'] = game_dates_df['GAME_ID'].astype(int)

        # Ensure 'date' is datetime, drop rows where conversion fails
        game_dates_df['date'] = pd.to_datetime(game_dates_df['date'], errors='coerce')
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
def propagate_player_values_py(df):
    print("      Running Player Propagation...")
    # Placeholder implementation - replace with full code
    # For now, just returns the df to allow script structure completion
    time.sleep(0.1) # Simulate work
    print("      Finished Player Propagation (Placeholder).")
    return df

def ft_off_check_py(df):
    print("      Running FT Off Check...")
    # Placeholder implementation - replace with full code
    time.sleep(0.1) # Simulate work
    print("      Finished FT Off Check (Placeholder).")
    return df
# --- End Placeholder ---


# --- Base Processing Function ---
def _base_processing(file_path):
    """Handles common data loading and initial preparation steps."""
    print(f"    Running Base Processing for {file_path}...")
    try:
        nba_df = pd.read_csv(file_path, low_memory=False)
        print(f"      Read {len(nba_df)} rows from {file_path}")
    except FileNotFoundError:
        print(f"      Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"      Error reading {file_path}: {e}")
        return None

    # --- Ensure seconds_remaining_quarter is numeric --- ### ADDED ###
    if 'seconds_remaining_quarter' in nba_df.columns:
        nba_df['seconds_remaining_quarter'] = pd.to_numeric(nba_df['seconds_remaining_quarter'], errors='coerce')
        # Optional: Handle rows where conversion failed if necessary
        # rows_failed = nba_df['seconds_remaining_quarter'].isna().sum()
        # if rows_failed > 0:
        #    print(f"      Warning: Failed to convert 'seconds_remaining_quarter' to numeric for {rows_failed} rows.")
        print("      Ensured 'seconds_remaining_quarter' is numeric.")
    else:
        print("      Warning: 'seconds_remaining_quarter' column not found for numeric conversion.")
        # If this column is critical, you might want to return None here

    # --- Initial Cleaning ---
    # ... (rest of cleaning code remains the same) ...
    player_cols = [f"{team}_player{i}" for team in ["away", "home"] for i in range(1, 6)]
    missing_p_cols = [p_col for p_col in player_cols if p_col not in nba_df.columns]
    if missing_p_cols:
        print(f"      Error: Missing required player columns: {missing_p_cols}")
        return None

    nba_df = nba_df.dropna(subset=player_cols)
    nba_df = nba_df[~(nba_df[player_cols] == '').any(axis=1)]
    print(f"      Rows after player filtering: {len(nba_df)}")
    if len(nba_df) == 0:
        print("      Error: No rows remaining after player filtering.")
        return None

    # --- Map event_type ---
    # ... (mapping code remains the same) ...
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
    # ... (fillna code remains the same) ...
    for col in nba_df.select_dtypes(include=['object', 'string']).columns:
        if col in nba_df.columns: # Check if column exists
            nba_df[col] = nba_df[col].fillna('')
    print("      Filled NA values in string columns")


    # --- Calculate Base Scores ---
    # ... (score calculation remains the same) ...
    nba_df = nba_df.assign(
        Home_Action_Score=case_when(
            series_contains(nba_df['home_description'], "Free Throw", case=False) & ~series_contains(nba_df['home_description'], "MISS", case=False), 1,
            series_contains(nba_df['home_description'], "PTS", case=False) & ~series_contains(nba_df['home_description'], "3PT", case=False) & (nba_df['event_type'] == "MAKE"), 2,
            series_contains(nba_df['home_description'], "3PT", case=False) & ~series_contains(nba_df['home_description'], "MISS", case=False), 3,
            0 # Default
        ),
        Away_Action_Score=case_when(
            series_contains(nba_df['visitor_description'], "Free Throw", case=False) & ~series_contains(nba_df['visitor_description'], "MISS", case=False), 1,
            series_contains(nba_df['visitor_description'], "PTS", case=False) & ~series_contains(nba_df['visitor_description'], "3PT", case=False) & (nba_df['event_type'] == "MAKE"), 2,
            series_contains(nba_df['visitor_description'], "3PT", case=False) & ~series_contains(nba_df['visitor_description'], "MISS", case=False), 3,
            0 # Default
        )
    )
    group_cols = ['game_id'] if 'game_id' in nba_df.columns else None
    if group_cols:
        nba_df['Net_Home'] = nba_df.groupby(group_cols)['Home_Action_Score'].cumsum()
        nba_df['Net_Away'] = nba_df.groupby(group_cols)['Away_Action_Score'].cumsum()
    else:
        nba_df['Net_Home'] = nba_df['Home_Action_Score'].cumsum()
        nba_df['Net_Away'] = nba_df['Away_Action_Score'].cumsum()
    nba_df['Net_Total'] = nba_df['Net_Home'] + nba_df['Net_Away']
    print("      Calculated action scores and cumulative totals")


    # --- Create Lag/Lead Columns ---                ### MODIFIED ###
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

                # --- Use np.nan for numeric shifts, "" otherwise --- ### MODIFIED ###
                fill_val = np.nan if pd.api.types.is_numeric_dtype(nba_df[col]) else ""

                if period != 0:
                    if group_cols:
                        nba_df[new_col] = nba_df.groupby(group_cols)[col].shift(period, fill_value=fill_val)
                    else:
                        nba_df[new_col] = nba_df[col].shift(period, fill_value=fill_val)
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
# --- Final Output Preparation Helper ---
# --- Final Output Preparation Helper ---
def _finalize_df(df, numerator_col, year, id_cols=['game_id'], o_cols=[f'O{i}' for i in range(1,6)], d_cols=[f'D{i}' for i in range(1,6)]):
    """Handles common final steps: joining external data, selecting columns, filtering."""
    print("      Running Finalization...")
    if df is None or df.empty:
        print("      Skipping finalization: Input DataFrame is empty or None.")
        return None

    # --- Load and Merge Game Dates --- ### MODIFIED ###
    game_dates_df = load_game_dates() # Load (or get from cache)

    if game_dates_df is not None and 'game_id' in df.columns:
        # Filter game_dates for the relevant season (using the 'year' passed in)
        # Assumes 'season' column in game_dates_df is the starting year (e.g., 2024)
        filtered_dates_df = game_dates_df[game_dates_df['season'] == year]

        if not filtered_dates_df.empty:
            print(f"        Merging with {len(filtered_dates_df)} game date entries for season {year}...")

            # --- Ensure game_id in main df is integer for merging ---
            # Check current dtype
            if not pd.api.types.is_integer_dtype(df['game_id']):
                print(f"        Converting main df game_id from {df['game_id'].dtype} to numeric/int...")
                df['game_id'] = pd.to_numeric(df['game_id'], errors='coerce')
                # Handle potential NaNs introduced by coerce before converting to int
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
                how='left' # Keep all rows from the main df
            )
            # Drop the extra GAME_ID column from the merge
            if 'GAME_ID' in df.columns:
                df = df.drop(columns=['GAME_ID'])

            # Check merge results
            rows_after_merge = len(df)
            if rows_after_merge != original_rows:
                 print(f"        Warning: Row count changed during game_date merge ({original_rows} -> {rows_after_merge}). Check for duplicate game_ids in main df?")
            merged_dates_count = df['game_date'].notna().sum()
            print(f"        Successfully merged game dates for {merged_dates_count} rows.")
            missing_dates = len(df) - merged_dates_count
            if missing_dates > 0:
                 print(f"        Warning: Could not find matching game dates for {missing_dates} rows.")

        else:
             print(f"        Warning: No game dates found for season {year} in the loaded schedule data. 'game_date' column will be missing or empty.")
             if 'game_date' not in df.columns: df['game_date'] = pd.NaT # Add placeholder if column doesn't exist at all
    # Handle case where game_dates loading failed or game_id wasn't in main df
    elif 'game_date' not in df.columns:
        print("       Warning: Game dates CSV failed to load or game_id missing in main df. Adding empty game_date column.")
        df['game_date'] = pd.NaT


    # --- Placeholder for Joining External Data ---
    # Player Dictionary (ID -> Name)
    # ... (Placeholder code remains the same) ...
    print("      --- Placeholder for replacing Player IDs with Names in O1-D5 ---")

    # --- Schedule Data (Game Date) --- ### PLACEHOLDER REMOVED ###
    # (Now handled by the merge logic above)

    # --- End Placeholder ---

    df['Season'] = year + 1 # R code used year + 1

    # --- Select Final Columns ---
    final_cols = id_cols + o_cols + d_cols
    # Add numerator_col safely - it might be a list for RAPM
    if isinstance(numerator_col, list):
        final_cols.extend(numerator_col)
    else:
        final_cols.append(numerator_col)

    if 'game_date' in df.columns: final_cols.append('game_date') # Keep game_date if added
    final_cols.append('Season')

    # Ensure columns exist before selecting
    final_cols_present = [col for col in final_cols if col in df.columns]
    missing_final_cols = [col for col in final_cols if col not in final_cols_present]
    if missing_final_cols:
        print(f"      Warning: Missing expected final columns: {missing_final_cols}")

    # Select only existing columns to avoid errors
    df_final = df[final_cols_present].copy()


    # --- Final Filtering (D5 != 0 or NaN/blank after potential name mapping) ---
    if 'D5' in df_final.columns:
        initial_rows = len(df_final)
        # Filter based on type to avoid errors
        if pd.api.types.is_numeric_dtype(df_final['D5']):
             df_final = df_final[df_final['D5'] != 0]
        # Also filter out NAs and potential empty strings robustly
        df_final = df_final[df_final['D5'].astype(str).fillna('').str.strip() != '']
        df_final = df_final[~pd.isna(df_final['D5'])]

        print(f"      Rows after final D5 != 0/Null filter: {len(df_final)} (removed {initial_rows - len(df_final)})")
    else:
        print("      Warning: D5 column not found for final filtering.")


    print("      Finished Finalization.")
    return df_final

# --- Process RAPM Function ---
def process_rapm_py(file_path, year, season_str):
    """ Processes data for standard RAPM calculation. """
    print(f"  Starting RAPM Processing for {season_str}...")
    nba_df = _base_processing(file_path)
    if nba_df is None: return None
    start_time = time.time()

    # --- Define Possession Characteristics (RAPM/TOV Version) ---
    # Regex for FTs not 1 of 2, 1 of 3, 2 of 3, Tech, Flagrant
    ft_exclude_pattern = r'\b(1 of [23]|2 of 3|Technical|Flagrant)\b'

    nba_df['offensive_FT_Rebound'] = case_when(
        series_contains(nba_df['visitor_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_visitor_desc'], "MISS", case=False) & series_contains(nba_df['Prev_visitor_desc'], "Free Throw", case=False) & ~series_contains(nba_df['Prev_visitor_desc'], ft_exclude_pattern, case=False), True,
        series_contains(nba_df['home_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_home_desc'], "MISS", case=False) & series_contains(nba_df['Prev_home_desc'], "Free Throw", case=False) & ~series_contains(nba_df['Prev_home_desc'], ft_exclude_pattern, case=False), True,
        False
    ).astype(bool)

    # End_of_Possession (RAPM Version - from ProcessRAPM R code)
    nba_df['End_of_Possession'] = case_when(
        (nba_df['event_type'] == "EndOfPeriod") & (nba_df['prev_seconds'] > 0), True,
        series_contains(nba_df['home_description'], r'Flagrant (2 of 2|3 of 3)', case=False), True,
        series_contains(nba_df['visitor_description'], r'Flagrant (2 of 2|3 of 3)', case=False), True,
        (nba_df['event_type'] == "Turnover"), True,
        series_contains(nba_df['home_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_visitor_desc'], "MISS", case=False) & ~series_contains(nba_df['Prev_visitor_desc'], ft_exclude_pattern, case=False), True,
        series_contains(nba_df['visitor_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_home_desc'], "MISS", case=False) & ~series_contains(nba_df['Prev_home_desc'], ft_exclude_pattern, case=False), True,
        series_contains(nba_df['home_description'], "PTS", case=False) & (nba_df['event_type'] == "MAKE") & ~series_contains(nba_df['Next_visitor_desc'], "S.FOUL", case=True), True,
        series_contains(nba_df['visitor_description'], "PTS", case=False) & (nba_df['event_type'] == "MAKE") & ~series_contains(nba_df['Next_home_desc'], "S.FOUL", case=True), True,
        series_contains(nba_df['home_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False) & series_contains(nba_df['home_description'], "PTS", case=False) & \
            ~(series_contains(nba_df['Prev_visitor_desc2'], "Transition", case=False) | series_contains(nba_df['Prev_visitor_desc'], "Transition", case=False) | series_contains(nba_df['home_description'], "Flagrant", case=False)), True,
        series_contains(nba_df['visitor_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False) & series_contains(nba_df['visitor_description'], "PTS", case=False) & \
            ~(series_contains(nba_df['Prev_home_desc2'], "Transition", case=False) | series_contains(nba_df['Prev_home_desc'], "Transition", case=False) | series_contains(nba_df['visitor_description'], "Flagrant", case=False)), True,
        False
    ).astype(bool)

    # TeamOnOffense (RAPM Version - slightly different from others)
    nba_df['TeamOnOffense'] = case_when(
        series_contains(nba_df['home_description'], r'Flagrant (2 of 2|3 of 3)', case=False), "Home",
        series_contains(nba_df['visitor_description'], r'Flagrant (2 of 2|3 of 3)', case=False), "Away",
        series_contains(nba_df['home_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_visitor_desc'], "MISS", case=False), "Away",
        series_contains(nba_df['visitor_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_home_desc'], "MISS", case=False), "Home",
        series_contains(nba_df['home_description'], "Turnover", case=False), "Home",
        series_contains(nba_df['visitor_description'], "Turnover", case=False), "Away",
        series_contains(nba_df['home_description'], "PTS", case=False) & (nba_df['event_type'] == "MAKE") & ~series_contains(nba_df['Next_visitor_desc'], "S.FOUL", case=True), "Home",
        series_contains(nba_df['visitor_description'], "PTS", case=False) & (nba_df['event_type'] == "MAKE") & ~series_contains(nba_df['Next_home_desc'], "S.FOUL", case=True), "Away",
        series_contains(nba_df['home_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False) & series_contains(nba_df['home_description'], "PTS", case=False) & \
            ~(series_contains(nba_df['Prev_visitor_desc2'], "Transition", case=False) | series_contains(nba_df['Prev_visitor_desc'], "Transition", case=False) | series_contains(nba_df['home_description'], "Flagrant", case=False)), "Home",
        series_contains(nba_df['visitor_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False) & series_contains(nba_df['visitor_description'], "PTS", case=False) & \
            ~(series_contains(nba_df['Prev_home_desc2'], "Transition", case=False) | series_contains(nba_df['Prev_home_desc'], "Transition", case=False) | series_contains(nba_df['visitor_description'], "Flagrant", case=False)), "Away",
        ""
    )
    print("      Calculated EOP, TeamOnOffense for RAPM")

    # --- Identify Potential Fouls ---
    window_size = 5
    group_cols = ['game_id'] if 'game_id' in nba_df.columns else None # Group rolling by game
    if group_cols:
        ft_events = (nba_df['event_type'] == "FreeThrow").groupby(nba_df['game_id']).rolling(window=window_size, min_periods=1).sum().reset_index(level=0, drop=True).shift(-(window_size - 1)).fillna(0)
        sub_events = (nba_df['event_type'] == "Substitution").groupby(nba_df['game_id']).rolling(window=window_size, min_periods=1).sum().reset_index(level=0, drop=True).shift(-(window_size - 1)).fillna(0)
    else:
        ft_events = (nba_df['event_type'] == "FreeThrow").rolling(window=window_size, min_periods=1).sum().shift(-(window_size - 1)).fillna(0)
        sub_events = (nba_df['event_type'] == "Substitution").rolling(window=window_size, min_periods=1).sum().shift(-(window_size - 1)).fillna(0)

    exclude_foul_pattern = r'T\.FOUL|FLAGRANT|Offens|Transition'
    nba_df['PotentialFoul'] = case_when(
        (ft_events != 0) & (sub_events != 0) & (nba_df['event_type'] == "Foul") & \
        ~series_contains(nba_df['visitor_description'], exclude_foul_pattern, case=False) & \
        ~series_contains(nba_df['home_description'], exclude_foul_pattern, case=False), True,
        False
    ).astype(bool)
    print(f"      Identified {nba_df['PotentialFoul'].sum()} potential fouls")

    # --- Apply Propagation & Map Players ---
    nba_df_propagated = propagate_player_values_py(nba_df) # Apply propagation
    nba_df_checked = ft_off_check_py(nba_df_propagated) # Apply FT check

    for i in range(1, 6):
        nba_df_checked[f'O{i}'] = np.where(nba_df_checked['TeamOnOffense'] == "Away", nba_df_checked[f'a{i}'],
                                        np.where(nba_df_checked['TeamOnOffense'] == "Home", nba_df_checked[f'h{i}'], 0))
        nba_df_checked[f'D{i}'] = np.where(nba_df_checked['TeamOnOffense'] == "Away", nba_df_checked[f'h{i}'],
                                        np.where(nba_df_checked['TeamOnOffense'] == "Home", nba_df_checked[f'a{i}'], 0))
    print("      Mapped O/D Players")

    # --- Calculate Luck-Adjusted Points (RAPM Specific) ---
    # --- Pholder for Joining Player Stats (FTPerc, ThreePerc) ---
    print("      --- Placeholder: Joining player stats (FTPerc, ThreePerc) ---")
    if 'player1_id' not in nba_df_checked.columns: # Need a player ID column
         print("      Warning: Missing 'player1_id' column for player stats join.")
         nba_df_checked['FTPerc'] = 0.75 # Default
         nba_df_checked['ThreePerc'] = 0.35 # Default
    else:
         # Example: df_selected = fetch_player_stats(season_str)
         # nba_df_checked = pd.merge(nba_df_checked, df_selected, left_on='player1_id', right_on='PlayerID', how='left')
         nba_df_checked['FTPerc'] = nba_df_checked.get('FTPerc', pd.Series(0.75, index=nba_df_checked.index)) # Use .get with default Series
         nba_df_checked['ThreePerc'] = nba_df_checked.get('ThreePerc', pd.Series(0.35, index=nba_df_checked.index))


    nba_df_checked['ExpFT'] = nba_df_checked['FTPerc'].fillna(0.75)
    nba_df_checked['Exp3PT'] = nba_df_checked['ThreePerc'].fillna(0.35)

    is_ft_event = series_contains(nba_df_checked['home_description'], "Free Throw", case=False) | series_contains(nba_df_checked['visitor_description'], "Free Throw", case=False) | (nba_df_checked['event_type'] == "FreeThrow")
    is_2pt_make_event = (series_contains(nba_df_checked['home_description'], "PTS", case=False) & ~series_contains(nba_df_checked['home_description'], "3PT", case=False) & (nba_df_checked['event_type'] == "MAKE")) | \
                        (series_contains(nba_df_checked['visitor_description'], "PTS", case=False) & ~series_contains(nba_df_checked['visitor_description'], "3PT", case=False) & (nba_df_checked['event_type'] == "MAKE"))
    is_3pt_event = series_contains(nba_df_checked['home_description'], "3PT", case=False) | series_contains(nba_df_checked['visitor_description'], "3PT", case=False)
    actual_score = nba_df_checked['Home_Action_Score'] + nba_df_checked['Away_Action_Score']

    nba_df_checked['OffNet'] = case_when(is_ft_event, nba_df_checked['ExpFT'], is_2pt_make_event, 2.0, is_3pt_event, (nba_df_checked['Exp3PT'] * 3) * 0.2 + 0.8 * actual_score, 0.0)
    nba_df_checked['DefNet'] = case_when(is_ft_event, nba_df_checked['ExpFT'], is_2pt_make_event, 2.0, is_3pt_event, (nba_df_checked['Exp3PT'] * 3) * 0.4 + 0.6 * actual_score, 0.0)

    if group_cols:
      nba_df_checked['OffTotal'] = nba_df_checked.groupby(group_cols)['OffNet'].cumsum()
      nba_df_checked['DefTotal'] = nba_df_checked.groupby(group_cols)['DefNet'].cumsum()
    else:
      nba_df_checked['OffTotal'] = nba_df_checked['OffNet'].cumsum()
      nba_df_checked['DefTotal'] = nba_df_checked['DefNet'].cumsum()
    print("      Calculated OffNet/DefNet")

    # --- Filter for EOP & Calculate Diffs ---
    nba_filt = nba_df_checked[nba_df_checked['End_of_Possession'] == True].copy()
    print(f"      Filtered down to {len(nba_filt)} RAPM End of Possession rows")

    if group_cols:
        nba_filt['Net_Diff'] = nba_filt['Net_Total'] - nba_filt.groupby(group_cols)['Net_Total'].shift(fill_value=0)
        nba_filt['Off_Diff'] = nba_filt['OffTotal'] - nba_filt.groupby(group_cols)['OffTotal'].shift(fill_value=0)
        nba_filt['Def_Diff'] = nba_filt['DefTotal'] - nba_filt.groupby(group_cols)['DefTotal'].shift(fill_value=0)
    else:
        nba_filt['Net_Diff'] = nba_filt['Net_Total'] - nba_filt['Net_Total'].shift(fill_value=0)
        nba_filt['Off_Diff'] = nba_filt['OffTotal'] - nba_filt['OffTotal'].shift(fill_value=0)
        nba_filt['Def_Diff'] = nba_filt['DefTotal'] - nba_filt['DefTotal'].shift(fill_value=0)
    print("      Calculated Net/Off/Def differences")

    # --- Finalize ---
    # RAPM numerator uses multiple diffs, often combined later. Let's keep Net_Diff for now.
    # R code selects columns 96-98 which were Net_Diff, Off_Diff, Def_Diff. We will include all 3.
    rapm_final_cols = ['Net_Diff', 'Off_Diff', 'Def_Diff']
    # Ensure they exist before trying to select
    rapm_final_cols = [c for c in rapm_final_cols if c in nba_filt.columns]
    if not rapm_final_cols:
        print("      Error: Could not find Net_Diff, Off_Diff, or Def_Diff columns for final output.")
        return None

    # Hacky way to pass multiple numerators to finalize, just select the first for now
    # A better approach might be to handle multiple numerators in _finalize_df
    nba_rapm_output = _finalize_df(nba_filt, rapm_final_cols[0], year)

    # If multiple numerators were intended, add the others back if finalize removed them
    for col in rapm_final_cols[1:]:
         if col in nba_filt.columns and col not in nba_rapm_output.columns:
              nba_rapm_output[col] = nba_filt[col]


    end_time = time.time()
    print(f"  Finished RAPM Processing. Time: {end_time - start_time:.2f} seconds.")
    return nba_rapm_output

# --- Process TS (True Shooting) Function ---
def process_ts_py(file_path, year, season_str):
    """ Processes data for True Shooting calculation. """
    print(f"  Starting TS Processing for {season_str}...")
    nba_df = _base_processing(file_path)
    if nba_df is None: return None
    start_time = time.time()

    # --- Define Possession Characteristics (TS Version) ---
    # End_of_Possession (TS Version - from ProcessTS R code)
    # Focuses on scoring attempts (MAKE, MISS, final FTs)
    nba_df['End_of_Possession'] = case_when(
        (nba_df['event_type'] == "EndOfPeriod") & (nba_df['prev_seconds'] > 0), True,
        series_contains(nba_df['home_description'], r'Flagrant (2 of 2|3 of 3)', case=False), True, # Flagrant FTs end poss
        series_contains(nba_df['visitor_description'], r'Flagrant (2 of 2|3 of 3)', case=False), True,
        (nba_df['event_type'] == "Turnover"), True, # Include TO initially, filter later
        series_contains(nba_df['visitor_description'], "MISS", case=False) & (nba_df['event_type'] == "MISS"), True, # Field Goal Miss
        series_contains(nba_df['home_description'], "MISS", case=False) & (nba_df['event_type'] == "MISS"), True, # Field Goal Miss
        series_contains(nba_df['home_description'], "PTS", case=False) & (nba_df['event_type'] == "MAKE") & ~series_contains(nba_df['Next_visitor_desc'], "S.FOUL", case=True), True, # Field Goal Make
        series_contains(nba_df['visitor_description'], "PTS", case=False) & (nba_df['event_type'] == "MAKE") & ~series_contains(nba_df['Next_home_desc'], "S.FOUL", case=True), True, # Field Goal Make
        # Final FTs (non-transition/flagrant)
        series_contains(nba_df['home_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False) & \
             ~(series_contains(nba_df['Prev_home_desc2'], "Transition", case=False) | series_contains(nba_df['Prev_home_desc'], "Transition", case=False) | series_contains(nba_df['visitor_description'], "Flagrant", case=False)), True,
        series_contains(nba_df['visitor_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False) & \
             ~(series_contains(nba_df['Prev_home_desc2'], "Transition", case=False) | series_contains(nba_df['Prev_home_desc'], "Transition", case=False) | series_contains(nba_df['visitor_description'], "Flagrant", case=False)), True,
        False
    ).astype(bool)

    # TeamOnOffense (TS Version - from ProcessTS R code)
    nba_df['TeamOnOffense'] = case_when(
        series_contains(nba_df['home_description'], r'Flagrant (2 of 2|3 of 3)', case=False), "Home",
        series_contains(nba_df['visitor_description'], r'Flagrant (2 of 2|3 of 3)', case=False), "Away",
        series_contains(nba_df['home_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_visitor_desc'], "MISS", case=False), "Away", # Def Reb -> Other team was Offense
        series_contains(nba_df['visitor_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_home_desc'], "MISS", case=False), "Home",
        series_contains(nba_df['visitor_description'], "MISS", case=False), "Away", # Who missed/made
        series_contains(nba_df['home_description'], "MISS", case=False), "Home",
        series_contains(nba_df['visitor_description'], "MAKE", case=False), "Away", # Includes FTs? Check R logic more closely if needed
        series_contains(nba_df['home_description'], "MAKE", case=False), "Home",
        series_contains(nba_df['home_description'], "Turnover", case=False), "Home", # Who turned over
        series_contains(nba_df['visitor_description'], "Turnover", case=False), "Away",
        series_contains(nba_df['home_description'], "PTS", case=False), "Home", # Catch-all for scoring descriptions
        series_contains(nba_df['visitor_description'], "PTS", case=False), "Away",
        series_contains(nba_df['home_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False), "Home", # Final FTs
        series_contains(nba_df['visitor_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False), "Away",
        ""
    )
    print("      Calculated EOP, TeamOnOffense for TS")

    # --- Identify Potential Fouls (Same logic as RAPM needed for propagation) ---
    window_size = 5
    group_cols = ['game_id'] if 'game_id' in nba_df.columns else None
    if group_cols:
        ft_events = (nba_df['event_type'] == "FreeThrow").groupby(nba_df['game_id']).rolling(window=window_size, min_periods=1).sum().reset_index(level=0, drop=True).shift(-(window_size - 1)).fillna(0)
        sub_events = (nba_df['event_type'] == "Substitution").groupby(nba_df['game_id']).rolling(window=window_size, min_periods=1).sum().reset_index(level=0, drop=True).shift(-(window_size - 1)).fillna(0)
    else:
        ft_events = (nba_df['event_type'] == "FreeThrow").rolling(window=window_size, min_periods=1).sum().shift(-(window_size - 1)).fillna(0)
        sub_events = (nba_df['event_type'] == "Substitution").rolling(window=window_size, min_periods=1).sum().shift(-(window_size - 1)).fillna(0)

    exclude_foul_pattern = r'T\.FOUL|FLAGRANT|Offens|Transition'
    nba_df['PotentialFoul'] = case_when(
        (ft_events != 0) & (sub_events != 0) & (nba_df['event_type'] == "Foul") & \
        ~series_contains(nba_df['visitor_description'], exclude_foul_pattern, case=False) & \
        ~series_contains(nba_df['home_description'], exclude_foul_pattern, case=False), True,
        False
    ).astype(bool)
    print(f"      Identified {nba_df['PotentialFoul'].sum()} potential fouls")


    # --- Apply Propagation & Map Players ---
    nba_df_propagated = propagate_player_values_py(nba_df)
    nba_df_checked = ft_off_check_py(nba_df_propagated)

    for i in range(1, 6):
        nba_df_checked[f'O{i}'] = np.where(nba_df_checked['TeamOnOffense'] == "Away", nba_df_checked[f'a{i}'],
                                        np.where(nba_df_checked['TeamOnOffense'] == "Home", nba_df_checked[f'h{i}'], 0))
        nba_df_checked[f'D{i}'] = np.where(nba_df_checked['TeamOnOffense'] == "Away", nba_df_checked[f'h{i}'],
                                        np.where(nba_df_checked['TeamOnOffense'] == "Home", nba_df_checked[f'a{i}'], 0))
    print("      Mapped O/D Players")

    # --- Filter for EOP & Calculate Diffs ---
    nba_filt = nba_df_checked[nba_df_checked['End_of_Possession'] == True].copy()
    print(f"      Filtered down to {len(nba_filt)} TS End of Possession rows (before TO removal)")

    # --- TS Specific: Filter out Turnovers ---
    initial_rows_ts = len(nba_filt)
    nba_filt = nba_filt[nba_filt['event_type'] != 'Turnover'].copy()
    print(f"      Filtered out {initial_rows_ts - len(nba_filt)} Turnover possessions for TS")

    # Calculate Net_Diff (Points scored on the possession/scoring attempt)
    group_cols = ['game_id'] if 'game_id' in nba_filt.columns else None
    if group_cols:
        nba_filt['Net_Diff'] = nba_filt['Net_Total'] - nba_filt.groupby(group_cols)['Net_Total'].shift(fill_value=0)
    else:
        nba_filt['Net_Diff'] = nba_filt['Net_Total'] - nba_filt['Net_Total'].shift(fill_value=0)
    print("      Calculated Net_Diff for TS")

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

    # --- Define Possession Characteristics (REB Version) ---
    # Calculate Offensive_Rebound flag (Numerator) - based on ProcessRebounding R code
    # Identifies ORBs after non-FT misses, not at very end of quarter
    nba_df['Offensive_Rebound'] = case_when(
        series_contains(nba_df['home_description'], r'\bREBOUND\b', case=True) & \
            series_contains(nba_df['Prev_home_desc'], "MISS", case=False) & \
            ~(series_contains(nba_df['time_quarter'], "0:00")) & ~(nba_df['Next_event'] == "EndOfPeriod") & \
            ~series_contains(nba_df['Prev_home_desc'], r'(1 of [23]|2 of 3|Technical|Flagrant)', case=False), 1,
        series_contains(nba_df['home_description'], r'\bREBOUND\b', case=True) & \
            series_contains(nba_df['Prev_home_desc'], "Putback", case=False) & \
            ~series_contains(nba_df['Prev_home_desc'], r'(1 of [23]|2 of 3|Technical|Flagrant)', case=False), 1,
        series_contains(nba_df['visitor_description'], r'\bREBOUND\b', case=True) & \
            series_contains(nba_df['Prev_visitor_desc'], "MISS", case=False) & \
            ~(series_contains(nba_df['time_quarter'], "0:00")) & ~(nba_df['Next_event'] == "EndOfPeriod") & \
            ~series_contains(nba_df['Prev_visitor_desc'], r'(1 of [23]|2 of 3|Technical|Flagrant)', case=False), 1,
        series_contains(nba_df['visitor_description'], r'\bREBOUND\b', case=True) & \
            series_contains(nba_df['Prev_visitor_desc'], "Putback", case=False) & \
            ~series_contains(nba_df['Prev_visitor_desc'], r'(1 of [23]|2 of 3|Technical|Flagrant)', case=False), 1,
        0 # Default is 0 (not an offensive rebound by this definition)
    ).astype(int)
    print(f"      Calculated Offensive_Rebound flag (found {nba_df['Offensive_Rebound'].sum()})")


    # End_of_Possession (REB Version - from ProcessRebounding R code)
    # Focuses on *any* rebound following a non-FT miss.
    ft_exclude_pattern = r'\b(1 of [23]|2 of 3|Technical|Flagrant)\b' # Reusable pattern
    nba_df['End_of_Possession'] = case_when(
        # Def Reb (Home gets reb after Away miss)
        series_contains(nba_df['home_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_visitor_desc'], "MISS", case=False) & ~series_contains(nba_df['Prev_visitor_desc'], ft_exclude_pattern, case=False), True,
        # Def Reb (Away gets reb after Home miss)
        series_contains(nba_df['visitor_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_home_desc'], "MISS", case=False) & ~series_contains(nba_df['Prev_home_desc'], ft_exclude_pattern, case=False), True,
        # Off Reb (Home gets reb after Home miss)
        series_contains(nba_df['home_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_home_desc'], "MISS", case=False) & ~series_contains(nba_df['Prev_home_desc'], ft_exclude_pattern, case=False), True,
         # Off Reb (Away gets reb after Away miss)
        series_contains(nba_df['visitor_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_visitor_desc'], "MISS", case=False) & ~series_contains(nba_df['Prev_visitor_desc'], ft_exclude_pattern, case=False), True,
        False # Default
    ).astype(bool)

    # TeamOnOffense (REB Version - from ProcessRebounding R code)
    nba_df['TeamOnOffense'] = case_when(
        series_contains(nba_df['home_description'], r'Flagrant (2 of 2|3 of 3)', case=False), "Home",
        series_contains(nba_df['visitor_description'], r'Flagrant (2 of 2|3 of 3)', case=False), "Away",
        # Def Reb -> Other team was Offense
        series_contains(nba_df['home_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_visitor_desc'], "MISS", case=False), "Away",
        series_contains(nba_df['visitor_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_home_desc'], "MISS", case=False), "Home",
         # Off Reb -> Same team was Offense
        series_contains(nba_df['home_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_home_desc'], "MISS", case=False), "Home",
        series_contains(nba_df['visitor_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_visitor_desc'], "MISS", case=False), "Away",
        # Other EOP causes from ProcessRebounding R code's TeamOnOffense section (Turnovers, Makes, Final FTs)
        series_contains(nba_df['home_description'], "Turnover", case=False), "Home",
        series_contains(nba_df['visitor_description'], "Turnover", case=False), "Away",
        series_contains(nba_df['home_description'], "PTS", case=False) & (nba_df['event_type'] == "MAKE") & ~series_contains(nba_df['Next_visitor_desc'], "S.FOUL", case=True), "Home",
        series_contains(nba_df['visitor_description'], "PTS", case=False) & (nba_df['event_type'] == "MAKE") & ~series_contains(nba_df['Next_home_desc'], "S.FOUL", case=True), "Away",
        series_contains(nba_df['home_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False), "Home",
        series_contains(nba_df['visitor_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False), "Away",
        ""
    )
    print("      Calculated EOP, TeamOnOffense for REB")

    # --- Propagation Check ---
    # NOTE: The provided R code for ProcessRebounding defines PotentialFoul but does *not* seem to call propagate_player_values or FTOffCheck.
    # We will skip these steps here to match the R script's apparent behavior for this specific function.
    print("      Skipping Player Propagation/FT Check for REB (as per R script structure)")
    nba_df_checked = nba_df # No propagation applied


    # --- Map Players ---
    for i in range(1, 6):
        nba_df_checked[f'O{i}'] = np.where(nba_df_checked['TeamOnOffense'] == "Away", nba_df_checked[f'a{i}'],
                                        np.where(nba_df_checked['TeamOnOffense'] == "Home", nba_df_checked[f'h{i}'], 0))
        nba_df_checked[f'D{i}'] = np.where(nba_df_checked['TeamOnOffense'] == "Away", nba_df_checked[f'h{i}'],
                                        np.where(nba_df_checked['TeamOnOffense'] == "Home", nba_df_checked[f'a{i}'], 0))
    print("      Mapped O/D Players")


    # --- Filter for EOP ---
    nba_filt = nba_df_checked[nba_df_checked['End_of_Possession'] == True].copy()
    print(f"      Filtered down to {len(nba_filt)} REB End of Possession rows")
    # Note: The numerator 'Offensive_Rebound' was calculated *before* filtering EOP.

    # --- Finalize ---
    # Numerator is the 'Offensive_Rebound' flag calculated earlier
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

    # --- Define Possession Characteristics (TOV Version) ---
    # Calculate TurnoverCounter (used for TurnoverTotal in R, but we need Is_Turnover flag)
    nba_df['TurnoverCounter'] = (nba_df['event_type'] == 'Turnover').astype(int)

    # End_of_Possession (TOV Version - from ProcessTOs R code)
    # Very similar to RAPM EOP definition
    ft_exclude_pattern = r'\b(1 of [23]|2 of 3|Technical|Flagrant)\b' # Defined in ProcessTOs as well? Let's assume yes. R code uses `(!str_detect(Prev_Event, "FreeThrow"))` for rebounds here.
    nba_df['End_of_Possession'] = case_when(
        (nba_df['event_type'] == "EndOfPeriod") & (nba_df['prev_seconds'] > 0), True,
        series_contains(nba_df['home_description'], r'Flagrant (2 of 2|3 of 3)', case=False), True,
        series_contains(nba_df['visitor_description'], r'Flagrant (2 of 2|3 of 3)', case=False), True,
        (nba_df['event_type'] == "Turnover"), True,
         # Def Reb - *excluding* those immediately following Free Throws (using Prev_Event check from R)
        series_contains(nba_df['home_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_visitor_desc'], "MISS", case=False) & (nba_df['Prev_Event'] != "FreeThrow"), True,
        series_contains(nba_df['visitor_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_home_desc'], "MISS", case=False) & (nba_df['Prev_Event'] != "FreeThrow"), True,
        # Made shots (not followed by S.FOUL)
        series_contains(nba_df['home_description'], "PTS", case=False) & (nba_df['event_type'] == "MAKE") & ~series_contains(nba_df['Next_visitor_desc'], "S.FOUL", case=True), True,
        series_contains(nba_df['visitor_description'], "PTS", case=False) & (nba_df['event_type'] == "MAKE") & ~series_contains(nba_df['Next_home_desc'], "S.FOUL", case=True), True,
        # Made final FT (non-transition/flagrant) - R code check looks slightly different here
        series_contains(nba_df['home_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False) & \
           ~(series_contains(nba_df['Prev_home_desc2'], "Transition", case=False) | series_contains(nba_df['Prev_home_desc'], "Transition", case=False) | series_contains(nba_df['visitor_description'], "Flagrant", case=False)), True, # Simpler check in ProcessTOs R code?
        series_contains(nba_df['visitor_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False) & \
           ~(series_contains(nba_df['Prev_home_desc2'], "Transition", case=False) | series_contains(nba_df['Prev_home_desc'], "Transition", case=False) | series_contains(nba_df['visitor_description'], "Flagrant", case=False)), True,
        False
    ).astype(bool)

     # TeamOnOffense (TOV Version - from ProcessTOs R code)
    nba_df['TeamOnOffense'] = case_when(
        series_contains(nba_df['home_description'], r'Flagrant (2 of 2|3 of 3)', case=False), "Home",
        series_contains(nba_df['visitor_description'], r'Flagrant (2 of 2|3 of 3)', case=False), "Away",
        series_contains(nba_df['home_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_visitor_desc'], "MISS", case=False), "Away",
        series_contains(nba_df['visitor_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_home_desc'], "MISS", case=False), "Home",
        series_contains(nba_df['home_description'], "Turnover", case=False), "Home",
        series_contains(nba_df['visitor_description'], "Turnover", case=False), "Away",
        series_contains(nba_df['home_description'], "PTS", case=False), "Home", # Broader than RAPM version
        series_contains(nba_df['visitor_description'], "PTS", case=False), "Away",
        series_contains(nba_df['home_description'], "MISS", case=False), "Home", # Added in TOV R code
        series_contains(nba_df['visitor_description'], "MISS", case=False), "Away",
        series_contains(nba_df['home_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False), "Home",
        series_contains(nba_df['visitor_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False), "Away",
        ""
    )
    print("      Calculated EOP, TeamOnOffense for TOV")


    # --- Identify Potential Fouls (Same logic as RAPM) ---
    window_size = 5
    group_cols = ['game_id'] if 'game_id' in nba_df.columns else None
    if group_cols:
        ft_events = (nba_df['event_type'] == "FreeThrow").groupby(nba_df['game_id']).rolling(window=window_size, min_periods=1).sum().reset_index(level=0, drop=True).shift(-(window_size - 1)).fillna(0)
        sub_events = (nba_df['event_type'] == "Substitution").groupby(nba_df['game_id']).rolling(window=window_size, min_periods=1).sum().reset_index(level=0, drop=True).shift(-(window_size - 1)).fillna(0)
    else:
        ft_events = (nba_df['event_type'] == "FreeThrow").rolling(window=window_size, min_periods=1).sum().shift(-(window_size - 1)).fillna(0)
        sub_events = (nba_df['event_type'] == "Substitution").rolling(window=window_size, min_periods=1).sum().shift(-(window_size - 1)).fillna(0)

    exclude_foul_pattern = r'T\.FOUL|FLAGRANT|Offens|Transition'
    nba_df['PotentialFoul'] = case_when(
        (ft_events != 0) & (sub_events != 0) & (nba_df['event_type'] == "Foul") & \
        ~series_contains(nba_df['visitor_description'], exclude_foul_pattern, case=False) & \
        ~series_contains(nba_df['home_description'], exclude_foul_pattern, case=False), True,
        False
    ).astype(bool)
    print(f"      Identified {nba_df['PotentialFoul'].sum()} potential fouls")

    # --- Apply Propagation & Map Players ---
    nba_df_propagated = propagate_player_values_py(nba_df)
    nba_df_checked = ft_off_check_py(nba_df_propagated)

    for i in range(1, 6):
        nba_df_checked[f'O{i}'] = np.where(nba_df_checked['TeamOnOffense'] == "Away", nba_df_checked[f'a{i}'],
                                        np.where(nba_df_checked['TeamOnOffense'] == "Home", nba_df_checked[f'h{i}'], 0))
        nba_df_checked[f'D{i}'] = np.where(nba_df_checked['TeamOnOffense'] == "Away", nba_df_checked[f'h{i}'],
                                        np.where(nba_df_checked['TeamOnOffense'] == "Home", nba_df_checked[f'a{i}'], 0))
    print("      Mapped O/D Players")

    # --- Filter for EOP ---
    nba_filt = nba_df_checked[nba_df_checked['End_of_Possession'] == True].copy()
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
if __name__ == '__main__':
    # Example for processing the file fetched earlier
    file_to_process = "NBA25.csv" # Assumes this file exists from the R script run
    processing_year = 2024       # Starting year for the 2024-25 season
    processing_season = "2024-25"

    print(f"\n--- Starting All Processing for {processing_season} ---")

    # --- Run Each Process ---
    rapm_df = process_rapm_py(file_to_process, processing_year, processing_season)
    ts_df = process_ts_py(file_to_process, processing_year, processing_season)
    reb_df = process_reb_py(file_to_process, processing_year, processing_season)
    tov_df = process_tov_py(file_to_process, processing_year, processing_season)

    # --- Save Results ---
    outputs = {
        "RAPM": rapm_df,
        "TS": ts_df,
        "REB": reb_df,
        "TOV": tov_df
    }

    for name, df in outputs.items():
        if df is not None and not df.empty:
            output_filename = f"{name}{processing_season[-2:]}.csv" # e.g., RAPM25.csv
            try:
                df.to_csv(output_filename, index=False)
                print(f"--- Saved {name} results to {output_filename} ({len(df)} rows) ---")
            except Exception as e:
                print(f"--- Error saving {output_filename}: {e} ---")
        else:
            print(f"--- No results generated or DataFrame empty for {name} {processing_season}. ---")

    print(f"\n--- Finished All Processing for {processing_season} ---")