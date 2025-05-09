import pandas as pd
import numpy as np
import re # Used for regex if needed, though str.contains often suffices
import time
# Helper function to mimic R's grepl with ignore.case=TRUE
def series_contains(series, pattern, case=False, na=False):
    """Checks if a string pattern is contained within Series elements."""
    # Ensure Series is string type, fill NA to avoid errors during str ops
    return series.astype(str).str.contains(pattern, case=case, na=na, regex=True)

# Helper function to mimic R's case_when using numpy.select
def case_when(*args):
    """Mimics R's dplyr::case_when using numpy.select.
    Expects pairs of condition (boolean Series) and result,
    with the last argument being the default result.
    """
    conditions, results = args[0:-1:2], args[1:-1:2]
    default = args[-1]
    return np.select(conditions, results, default=default)

# --- Translation of R function: propagate_player_values ---
# This function is complex due to its row-by-row logic and loops.
# Direct translation might be slow in Python. Vectorization is hard here.
# This translated version follows the R logic closely using loops.
def propagate_player_values_py(df):
    print("Starting player propagation...")
    # Ensure index is sequential if it's not already
    df = df.reset_index(drop=True)

    # Get indices where PotentialFoul is True
    foul_indices = df.index[df['PotentialFoul']].tolist()
    if not foul_indices:
        print("No potential fouls found for propagation.")
        return df # No fouls, return original df

    df_copy = df.copy() # Work on a copy to avoid SettingWithCopyWarning

    player_cols = ['a1', 'a2', 'a3', 'a4', 'a5', 'h1', 'h2', 'h3', 'h4', 'h5']

    processed_foul_indices = set() # Keep track of fouls already processed

    # Iterate through the identified foul rows
    for i in foul_indices:
        if i in processed_foul_indices:
            continue # Skip if this foul was handled by a previous iteration's lookahead

        # print(f"Processing potential foul at index {i}")
        if df_copy.loc[i, 'PotentialFoul']:
            # Store player values from the foul row
            a_values = df_copy.loc[i, ['a1', 'a2', 'a3', 'a4', 'a5']].values
            h_values = df_copy.loc[i, ['h1', 'h2', 'h3', 'h4', 'h5']].values
            found_free_throw = False

            # Loop through subsequent rows
            # Limit search range reasonably, e.g., next 50 rows or until next foul/end of data
            search_limit = min(i + 50, len(df_copy))
            next_foul_idx_in_range = -1
            try:
              next_foul_idx_in_range = min(f_idx for f_idx in foul_indices if f_idx > i)
              search_limit = min(search_limit, next_foul_idx_in_range + 1) # Search up to the next foul
            except ValueError:
                pass # No subsequent foul found


            j = i + 1
            while j < search_limit:
                # Check conditions to stop propagation for this foul event
                is_end_of_possession = df_copy.loc[j, 'End_of_Possession']
                is_offensive_ft_rebound = df_copy.loc[j, 'offensive_FT_Rebound']
                is_another_potential_foul = df_copy.loc[j, 'PotentialFoul'] # Stop if another *potential* foul happens

                if df_copy.loc[j, 'event_type'] == "FreeThrow":
                    found_free_throw = True

                # Check if the loop should break
                if is_end_of_possession or is_offensive_ft_rebound or is_another_potential_foul:
                    if found_free_throw:
                         # Apply propagation up to the row *before* the break condition (or up to j if EOP)
                        propagate_end_idx = j if is_end_of_possession else j -1
                        for k in range(i + 1, propagate_end_idx + 1):
                             if k < len(df_copy): # Boundary check
                                df_copy.loc[k, ['a1', 'a2', 'a3', 'a4', 'a5']] = a_values
                                df_copy.loc[k, ['h1', 'h2', 'h3', 'h4', 'h5']] = h_values
                                # Mark subsequent foul indices within propagation range as processed
                                if k in foul_indices:
                                     processed_foul_indices.add(k)

                    # print(f"  Propagation ended at index {j} due to: {'EOP' if is_end_of_Possession else ('OffFTReb' if is_offensive_FT_Rebound else 'NextFoul')}")
                    # Mark the current foul index as processed
                    processed_foul_indices.add(i)
                    break # Stop searching for this foul event (i)

                j += 1
            else: # If the while loop finishes without break (reached search limit)
                 if found_free_throw:
                     # Propagate up to the last checked row (j-1)
                     for k in range(i + 1, j):
                          if k < len(df_copy): # Boundary check
                             df_copy.loc[k, ['a1', 'a2', 'a3', 'a4', 'a5']] = a_values
                             df_copy.loc[k, ['h1', 'h2', 'h3', 'h4', 'h5']] = h_values
                             if k in foul_indices:
                                 processed_foul_indices.add(k)
                 # print(f"  Propagation search limit reached for foul at index {i}")
                 processed_foul_indices.add(i)


    print("Finished player propagation.")
    return df_copy


# --- Translation of R function: FTOffCheck ---
# This seems to check for lineup changes during FT sequences involving offensive rebounds
# and marks the previous row as End_of_Possession if a change occurs.
def ft_off_check_py(df):
    print("Starting FT Off Check...")
    df = df.reset_index(drop=True) # Ensure sequential index
    foul_indices = df.index[df['PotentialFoul']].tolist()
    if not foul_indices:
      print("No potential fouls found for FT Off Check.")
      return df

    df_copy = df.copy()
    player_cols = ['a1', 'a2', 'a3', 'a4', 'a5', 'h1', 'h2', 'h3', 'h4', 'h5']
    processed_foul_indices = set()

    for i in foul_indices:
         if i in processed_foul_indices:
             continue

         # print(f"Checking FT Off consistency for foul at index {i}")
         if df_copy.loc[i, 'PotentialFoul']:
            # Store player values from the foul row
            original_values = df_copy.loc[i, player_cols].values

            # Loop through subsequent rows
            search_limit = min(i + 50, len(df_copy))
            next_foul_idx_in_range = -1
            try:
              next_foul_idx_in_range = min(f_idx for f_idx in foul_indices if f_idx > i)
              search_limit = min(search_limit, next_foul_idx_in_range + 1)
            except ValueError:
                pass

            j = i + 1
            while j < search_limit:
                # Check conditions
                is_end_of_possession = df_copy.loc[j, 'End_of_Possession']
                is_offensive_ft_rebound = df_copy.loc[j, 'offensive_FT_Rebound']
                is_another_potential_foul = df_copy.loc[j, 'PotentialFoul']

                if is_offensive_ft_rebound:
                    # Check if player values at row j differ from original foul row i
                    current_values = df_copy.loc[j, player_cols].values
                    if not np.array_equal(original_values, current_values):
                        # If players changed before the offensive rebound after a potential foul,
                        # mark the row *before* the rebound as the end of the possession.
                        if (j - 1) > i: # Make sure we don't mark the foul row itself
                            print(f"  Player mismatch found at index {j} after foul {i}. Marking EOP at {j-1}.")
                            df_copy.loc[j - 1, 'End_of_Possession'] = True
                            processed_foul_indices.add(i)
                            # This break assumes one such check is needed per foul event
                            break
                        else:
                             print(f"  Player mismatch found immediately after foul {i} at {j}. No prior row to mark.")

                # Stop checking for this foul if EOP or another potential foul is encountered
                if is_end_of_possession or is_another_potential_foul:
                    processed_foul_indices.add(i)
                    # print(f"  FT Off Check for foul {i} ended at index {j} due to: {'EOP' if is_end_of_Possession else 'NextFoul'}")
                    break
                j += 1
            else: # Reached search limit
                 processed_foul_indices.add(i)


    print("Finished FT Off Check.")
    return df_copy


# --- Main Processing Function (ProcessRAPM equivalent) ---
def process_rapm_py(file_path, year, season_str):
    """
    Processes NBA PBP data to calculate RAPM-relevant possession stats.

    Args:
        file_path (str): Path to the input CSV file (e.g., "NBA24.csv").
        year (int): The starting year of the season (e.g., 2023 for "2023-24").
        season_str (str): The season string (e.g., "2023-24").

    Returns:
        pandas.DataFrame: Processed DataFrame with one row per possession.
    """
    print(f"Processing RAPM for: {file_path}, Year: {year}, Season: {season_str}")
    start_time = time.time()

    try:
        nba_df = pd.read_csv(file_path, low_memory=False)
        print(f"Read {len(nba_df)} rows from {file_path}")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    # === Initial Cleaning and Preparation ===
    # Filter out rows with missing player data (R: filter(if_all(...)))
    player_cols = [f"{team}_player{i}" for team in ["away", "home"] for i in range(1, 6)]
    nba_df = nba_df.dropna(subset=player_cols)
    nba_df = nba_df[~(nba_df[player_cols] == '').any(axis=1)]
    print(f"Rows after player filtering: {len(nba_df)}")


    # Map event_type (R: mutate(event_type = case_when(...)))
    event_type_map = {
        1: "MAKE", 2: "MISS", 3: "FreeThrow", 4: "Rebound", 5: "Turnover",
        6: "Foul", 7: "Violation", 8: "Substitution", 9: "Timeout",
        10: "JumpBall", 11: "Ejection", 12: "StartOfPeriod", 13: "EndOfPeriod",
        14: "Empty"
    }
    # Ensure event_type is numeric before mapping, handle potential non-numeric values
    nba_df['event_type_num'] = pd.to_numeric(nba_df['event_type'], errors='coerce')
    nba_df['event_type'] = nba_df['event_type_num'].map(event_type_map).fillna(nba_df['event_type_num'].astype(str)) # Keep original if no map
    print("Mapped event types")

    # Replace NA with empty strings in character columns (R: mutate(across(where(is.character), ~replace_na(., ""))))
    # Pandas usually handles NaN, but explicit fillna('') matches R logic
    for col in nba_df.select_dtypes(include=['object', 'string']).columns:
        nba_df[col] = nba_df[col].fillna('')
    print("Filled NA values in string columns")


    # === Calculate Scores and Base Metrics ===
    # (R: mutate(Home_Action_Score = case_when(...), ... Net_Total = ...))
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
    # Calculate cumulative scores (R: cumsum()) - Group by game_id if data spans multiple games
    if 'game_id' in nba_df.columns:
      nba_df['Net_Home'] = nba_df.groupby('game_id')['Home_Action_Score'].cumsum()
      nba_df['Net_Away'] = nba_df.groupby('game_id')['Away_Action_Score'].cumsum()
    else: # Assume single game if no game_id
      nba_df['Net_Home'] = nba_df['Home_Action_Score'].cumsum()
      nba_df['Net_Away'] = nba_df['Away_Action_Score'].cumsum()

    nba_df['Net_Total'] = nba_df['Net_Home'] + nba_df['Net_Away']
    print("Calculated action scores and cumulative totals")


    # === Create Lag/Lead Columns ===
    # (R: mutate(prev_seconds = lag(...), Prev_visitor_desc = lag(...), ...))
    # Need to group by game_id if processing multiple games in one file
    group_cols = ['game_id'] if 'game_id' in nba_df.columns else None

    if group_cols:
      nba_df['prev_seconds'] = nba_df.groupby(group_cols)['seconds_remaining_quarter'].shift(1)
      nba_df['Prev_visitor_desc'] = nba_df.groupby(group_cols)['visitor_description'].shift(1, fill_value="")
      nba_df['Prev_visitor_desc2'] = nba_df.groupby(group_cols)['visitor_description'].shift(2, fill_value="")
      nba_df['Prev_home_desc'] = nba_df.groupby(group_cols)['home_description'].shift(1, fill_value="")
      nba_df['Prev_home_desc2'] = nba_df.groupby(group_cols)['home_description'].shift(2, fill_value="")
      nba_df['Next_home_desc'] = nba_df.groupby(group_cols)['home_description'].shift(-1, fill_value="")
      nba_df['Next_visitor_desc'] = nba_df.groupby(group_cols)['visitor_description'].shift(-1, fill_value="")
    else: # Single game assumption
      nba_df['prev_seconds'] = nba_df['seconds_remaining_quarter'].shift(1)
      nba_df['Prev_visitor_desc'] = nba_df['visitor_description'].shift(1, fill_value="")
      nba_df['Prev_visitor_desc2'] = nba_df['visitor_description'].shift(2, fill_value="")
      nba_df['Prev_home_desc'] = nba_df['home_description'].shift(1, fill_value="")
      nba_df['Prev_home_desc2'] = nba_df['home_description'].shift(2, fill_value="")
      nba_df['Next_home_desc'] = nba_df['home_description'].shift(-1, fill_value="")
      nba_df['Next_visitor_desc'] = nba_df['visitor_description'].shift(-1, fill_value="")

    print("Created lag/lead columns")

    # === Define Possession Characteristics ===
    # (R: mutate(offensive_FT_Rebound = case_when(...), End_of_Possession = case_when(...), TeamOnOffense = case_when(...)))

    # Note: R uses str_detect which is case-sensitive by default. Pandas str.contains is also case-sensitive by default.
    # R's grepl(..., ignore.case=TRUE) is equivalent to pandas str.contains(..., case=False)
    # The R code uses a mix (str_detect implies case=True, grepl implies case=False).
    # We'll try to match based on the function used in R. Using helper function for clarity.

    # Regex for FTs not 1 of 2, 1 of 3, 2 of 3, Tech, Flagrant
    ft_exclude_pattern = r'\b(1 of [23]|2 of 3|Technical|Flagrant)\b' # \b for word boundaries

    nba_df['offensive_FT_Rebound'] = case_when(
        series_contains(nba_df['visitor_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_visitor_desc'], "MISS", case=False) & series_contains(nba_df['Prev_visitor_desc'], "Free Throw", case=False) & ~series_contains(nba_df['Prev_visitor_desc'], ft_exclude_pattern, case=False), True,
        # R code had duplicate rebound condition (lowercase) - combining here.
        series_contains(nba_df['home_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_home_desc'], "MISS", case=False) & series_contains(nba_df['Prev_home_desc'], "Free Throw", case=False) & ~series_contains(nba_df['Prev_home_desc'], ft_exclude_pattern, case=False), True,
        False # Default
    ).astype(bool)


    # End_of_Possession - This is complex and needs careful translation
    nba_df['End_of_Possession'] = case_when(
        (nba_df['event_type'] == "EndOfPeriod") & (nba_df['prev_seconds'] > 0), True,
        series_contains(nba_df['home_description'], r'Flagrant (2 of 2|3 of 3)', case=False), True,
        series_contains(nba_df['visitor_description'], r'Flagrant (2 of 2|3 of 3)', case=False), True,
        (nba_df['event_type'] == "Turnover"), True,
        # Defensive Rebound (Home gets rebound after Away Miss) - excluding certain FTs
        series_contains(nba_df['home_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_visitor_desc'], "MISS", case=False) & ~series_contains(nba_df['Prev_visitor_desc'], ft_exclude_pattern, case=False), True,
        # Defensive Rebound (Away gets rebound after Home Miss) - excluding certain FTs
        series_contains(nba_df['visitor_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_home_desc'], "MISS", case=False) & ~series_contains(nba_df['Prev_home_desc'], ft_exclude_pattern, case=False), True,
         # Made shot (Home) not followed by shooting foul on Visitor
        series_contains(nba_df['home_description'], "PTS", case=False) & (nba_df['event_type'] == "MAKE") & ~series_contains(nba_df['Next_visitor_desc'], "S.FOUL", case=True), True, # S.FOUL seems case-sensitive in context
        # Made shot (Away) not followed by shooting foul on Home
        series_contains(nba_df['visitor_description'], "PTS", case=False) & (nba_df['event_type'] == "MAKE") & ~series_contains(nba_df['Next_home_desc'], "S.FOUL", case=True), True,
        # Made final FT (Home) - excluding Transition/Flagrant fouls in previous 2 events
        series_contains(nba_df['home_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False) & series_contains(nba_df['home_description'], "PTS", case=False) & \
            ~(series_contains(nba_df['Prev_visitor_desc2'], "Transition", case=False) | series_contains(nba_df['Prev_visitor_desc'], "Transition", case=False) | series_contains(nba_df['home_description'], "Flagrant", case=False)), True,
        # Made final FT (Away) - excluding Transition/Flagrant fouls in previous 2 events
        series_contains(nba_df['visitor_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False) & series_contains(nba_df['visitor_description'], "PTS", case=False) & \
            ~(series_contains(nba_df['Prev_home_desc2'], "Transition", case=False) | series_contains(nba_df['Prev_home_desc'], "Transition", case=False) | series_contains(nba_df['visitor_description'], "Flagrant", case=False)), True,
        False # Default
    ).astype(bool)

    # TeamOnOffense
    # NOTE: This logic might need refinement based on precise PBP data format & rules.
    # R code seems to have some potentially overlapping/redundant conditions.
    nba_df['TeamOnOffense'] = case_when(
        series_contains(nba_df['home_description'], r'Flagrant (2 of 2|3 of 3)', case=False), "Home",
        series_contains(nba_df['visitor_description'], r'Flagrant (2 of 2|3 of 3)', case=False), "Away",
        # Defensive Rebound (Home gets rebound after Away Miss) -> Away was on offense
        series_contains(nba_df['home_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_visitor_desc'], "MISS", case=False), "Away",
         # Defensive Rebound (Away gets rebound after Home Miss) -> Home was on offense
        series_contains(nba_df['visitor_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_home_desc'], "MISS", case=False), "Home",
        # Turnover by Home -> Home was on offense
        series_contains(nba_df['home_description'], "Turnover", case=False), "Home",
        # Turnover by Away -> Away was on offense
        series_contains(nba_df['visitor_description'], "Turnover", case=False), "Away",
         # Home scores (and not followed by S.FOUL on Visitor) -> Home was on offense
        series_contains(nba_df['home_description'], "PTS", case=False) & (nba_df['event_type'] == "MAKE") & ~series_contains(nba_df['Next_visitor_desc'], "S.FOUL", case=True), "Home",
        # Away scores (and not followed by S.FOUL on Home) -> Away was on offense
        series_contains(nba_df['visitor_description'], "PTS", case=False) & (nba_df['event_type'] == "MAKE") & ~series_contains(nba_df['Next_home_desc'], "S.FOUL", case=True), "Away",
        # Home makes final FT (non-transition/flagrant) -> Home was on offense
         series_contains(nba_df['home_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False) & series_contains(nba_df['home_description'], "PTS", case=False) & \
            ~(series_contains(nba_df['Prev_visitor_desc2'], "Transition", case=False) | series_contains(nba_df['Prev_visitor_desc'], "Transition", case=False) | series_contains(nba_df['home_description'], "Flagrant", case=False)), "Home",
        # Away makes final FT (non-transition/flagrant) -> Away was on offense
        series_contains(nba_df['visitor_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False) & series_contains(nba_df['visitor_description'], "PTS", case=False) & \
            ~(series_contains(nba_df['Prev_home_desc2'], "Transition", case=False) | series_contains(nba_df['Prev_home_desc'], "Transition", case=False) | series_contains(nba_df['visitor_description'], "Flagrant", case=False)), "Away",
        "" # Default empty string
    )
    print("Calculated offensive_FT_Rebound, End_of_Possession, TeamOnOffense")


    # === Identify Potential Fouls for Propagation ===
    # (R: mutate(FTEvents = rollapply(...), SubEvents = rollapply(...), PotentialFoul = case_when(...)))
    # Using pandas rolling window. `align='left'` in R means the window includes the current row and the next N-1 rows.
    # Pandas default is right-aligned. To match R's align='left', we can shift the result of a right-aligned window.
    # R's `rollapply(..., width = 5, ..., align = "left", fill = 0)` means sum over current row + next 4 rows.
    # Pandas equivalent for align='left': .rolling(window=5, min_periods=1).sum().shift(-(5-1))
    window_size = 5
    ft_events = (nba_df['event_type'] == "FreeThrow").rolling(window=window_size, min_periods=1).sum().shift(-(window_size - 1)).fillna(0)
    sub_events = (nba_df['event_type'] == "Substitution").rolling(window=window_size, min_periods=1).sum().shift(-(window_size - 1)).fillna(0)

    # Note: R code checks for various foul types NOT to be present.
    # Define patterns for fouls to exclude from 'PotentialFoul'
    exclude_foul_pattern = r'T\.FOUL|FLAGRANT|Offens|Transition' # Case-insensitive check needed? R code seems mixed. Assuming case-insensitive.

    nba_df['PotentialFoul'] = case_when(
        (ft_events != 0) & (sub_events != 0) & (nba_df['event_type'] == "Foul") & \
        ~series_contains(nba_df['visitor_description'], exclude_foul_pattern, case=False) & \
        ~series_contains(nba_df['home_description'], exclude_foul_pattern, case=False), True,
        False # Default
    ).astype(bool)

    potential_foul_count = nba_df['PotentialFoul'].sum()
    print(f"Identified {potential_foul_count} potential fouls for propagation")


    # === Alias Player Columns ===
    # (R: mutate(a1 = away_player1, ... h5 = home_player5))
    alias_map = {f"{team}_player{i}": f"{prefix}{i}" for team, prefix in [("away", "a"), ("home", "h")] for i in range(1, 6)}
    nba_df = nba_df.rename(columns=alias_map)
    print("Aliased player columns (a1-h5)")


    # === Apply Player Propagation and FT Check Logic ===
    # (R: propagate_player_values(NBA24,1), FTOffCheck(NBA24,1))
    # These custom R functions are complex. Apply Python versions.
    nba_df_propagated = propagate_player_values_py(nba_df)
    nba_df_checked = ft_off_check_py(nba_df_propagated)
    print("Applied player propagation and FT off check logic")


    # === Map Offense/Defense Players ===
    # (R: mutate(O1 = case_when(...), ..., D5 = case_when(...)))
    for i in range(1, 6):
        nba_df_checked[f'O{i}'] = np.where(nba_df_checked['TeamOnOffense'] == "Away", nba_df_checked[f'a{i}'],
                                        np.where(nba_df_checked['TeamOnOffense'] == "Home", nba_df_checked[f'h{i}'],
                                                 0)) # R default was 0
        nba_df_checked[f'D{i}'] = np.where(nba_df_checked['TeamOnOffense'] == "Away", nba_df_checked[f'h{i}'],
                                        np.where(nba_df_checked['TeamOnOffense'] == "Home", nba_df_checked[f'a{i}'],
                                                 0)) # R default was 0
    print("Mapped offense (O1-O5) and defense (D1-D5) players")


    # === Incorporate Player Stats (FT%/3P%) for Luck Adjustment ===
    # (R: left_join(df_selected, ...), mutate(Exp3PT = ..., ExpFT = ...))
    # --- PLACEHOLDER ---
    # This requires fetching player season stats (FTA, FTPerc, 3PA, ThreePerc)
    # e.g., using nba_api or reading a pre-saved stats file.
    # Example: df_selected = fetch_player_stats(season_str) # Needs implementation
    # nba_df_checked = pd.merge(nba_df_checked, df_selected, left_on='player1_id', right_on='PlayerID', how='left')
    # For now, create placeholder columns if they don't exist from merge
    if 'FTPerc' not in nba_df_checked.columns: nba_df_checked['FTPerc'] = 0.75 # Example default
    if 'ThreePerc' not in nba_df_checked.columns: nba_df_checked['ThreePerc'] = 0.35 # Example default
    print("--- Placeholder for joining player stats (FTPerc, ThreePerc) ---")
    # --- END PLACEHOLDER ---

    nba_df_checked = nba_df_checked.assign(
        ExpFT = nba_df_checked['FTPerc'].fillna(0), # Use default if merge failed/no data
        Exp3PT = nba_df_checked['ThreePerc'].fillna(0) # Use default if merge failed/no data
    )

    # Calculate OffNet/DefNet using the formulas from R
    # Combine home/away descriptions check for FT/3PT events
    is_ft_event = series_contains(nba_df_checked['home_description'], "Free Throw", case=False) | \
                  series_contains(nba_df_checked['visitor_description'], "Free Throw", case=False) | \
                  (nba_df_checked['event_type'] == "FreeThrow")

    is_2pt_make_event = (series_contains(nba_df_checked['home_description'], "PTS", case=False) & ~series_contains(nba_df_checked['home_description'], "3PT", case=False) & (nba_df_checked['event_type'] == "MAKE")) | \
                        (series_contains(nba_df_checked['visitor_description'], "PTS", case=False) & ~series_contains(nba_df_checked['visitor_description'], "3PT", case=False) & (nba_df_checked['event_type'] == "MAKE"))

    is_3pt_event = series_contains(nba_df_checked['home_description'], "3PT", case=False) | \
                   series_contains(nba_df_checked['visitor_description'], "3PT", case=False)

    actual_score = nba_df_checked['Home_Action_Score'] + nba_df_checked['Away_Action_Score']

    nba_df_checked['OffNet'] = case_when(
        is_ft_event, nba_df_checked['ExpFT'], # Expected points for FT is FT% * 1
        is_2pt_make_event, 2.0,               # Actual points for 2pt make
        is_3pt_event, (nba_df_checked['Exp3PT'] * 3) * 0.2 + 0.8 * actual_score, # Luck adj 3PT
        0.0 # Default
    )
    nba_df_checked['DefNet'] = case_when(
        is_ft_event, nba_df_checked['ExpFT'], # Expected points for FT
        is_2pt_make_event, 2.0,               # Actual points for 2pt make
        is_3pt_event, (nba_df_checked['Exp3PT'] * 3) * 0.4 + 0.6 * actual_score, # Luck adj 3PT (different weights?)
        0.0 # Default
    )

    # Calculate cumulative OffNet/DefNet (R: cumsum())
    if group_cols:
      nba_df_checked['OffTotal'] = nba_df_checked.groupby(group_cols)['OffNet'].cumsum()
      nba_df_checked['DefTotal'] = nba_df_checked.groupby(group_cols)['DefNet'].cumsum()
    else: # Single game assumption
      nba_df_checked['OffTotal'] = nba_df_checked['OffNet'].cumsum()
      nba_df_checked['DefTotal'] = nba_df_checked['DefNet'].cumsum()

    print("Calculated luck-adjusted OffNet/DefNet and totals")


    # === Final Filtering and Selection ===
    # (R: Player24 <- hoopR::nba_commonallplayers(...), Dict24 <- ..., NBA24Filt <- NBA24[NBA24$End_of_Possession == TRUE, ])
    nba_filt = nba_df_checked[nba_df_checked['End_of_Possession'] == True].copy() # Filter for EOP rows
    print(f"Filtered down to {len(nba_filt)} End of Possession rows")

    # Calculate Net_Diff, Off_Diff, Def_Diff (R: mutate(Net_Diff = Net_Total - lag(Net_Total, ...)))
    # Need lag within game if multiple games present
    if group_cols:
        nba_filt['Net_Diff'] = nba_filt['Net_Total'] - nba_filt.groupby(group_cols)['Net_Total'].shift(fill_value=0)
        nba_filt['Off_Diff'] = nba_filt['OffTotal'] - nba_filt.groupby(group_cols)['OffTotal'].shift(fill_value=0)
        nba_filt['Def_Diff'] = nba_filt['DefTotal'] - nba_filt.groupby(group_cols)['DefTotal'].shift(fill_value=0)
    else: # Single game assumption
        nba_filt['Net_Diff'] = nba_filt['Net_Total'] - nba_filt['Net_Total'].shift(fill_value=0)
        nba_filt['Off_Diff'] = nba_filt['OffTotal'] - nba_filt['OffTotal'].shift(fill_value=0)
        nba_filt['Def_Diff'] = nba_filt['DefTotal'] - nba_filt['DefTotal'].shift(fill_value=0)
    print("Calculated Net/Off/Def differences per possession")


    # Replace Player IDs with Names in O1-D5 columns
    # (R: loop with Dict24)
    # --- PLACEHOLDER ---
    # Requires fetching player dictionary (PlayerID -> PlayerName)
    # Example: player_dict = fetch_player_dict(season_str) # Needs implementation {player_id: player_name}
    # for col in [f'O{i}' for i in range(1, 6)] + [f'D{i}' for i in range(1, 6)]:
    #     nba_filt[col] = nba_filt[col].map(player_dict).fillna(nba_filt[col]) # Keep ID if name not found
    print("--- Placeholder for replacing Player IDs with Names in O1-D5 ---")
    # --- END PLACEHOLDER ---


    # Select final columns (R: select(1, 78:87, 96:98))
    # Column indices (78:87, 96:98) are unreliable. Select by name.
    # Need 'game_id' (col 1 in R), O1-D5 (cols 78-87 in R ex), Net_Diff, Off_Diff, Def_Diff (cols 96-98 in R ex)
    final_cols = ['game_id'] + \
                 [f'O{i}' for i in range(1, 6)] + \
                 [f'D{i}' for i in range(1, 6)] + \
                 ['Net_Diff', 'Off_Diff', 'Def_Diff']

    # Ensure all expected columns exist before selecting
    missing_cols = [col for col in final_cols if col not in nba_filt.columns]
    if missing_cols:
      print(f"Warning: Missing expected final columns: {missing_cols}")
      # Add missing columns with default value (e.g., None or 0) if needed
      for col in missing_cols:
          nba_filt[col] = None # Or 0, depending on expected type

    # Add game_date and Season
    # --- PLACEHOLDER ---
    # Requires fetching schedule data
    # Example: schedule_df = fetch_schedule(year) # Needs implementation, ensure game_id matches type
    # nba_filt['game_id'] = nba_filt['game_id'].astype(int) # Match type for merge
    # nba_filt = pd.merge(nba_filt, schedule_df[['game_id', 'game_date']], on='game_id', how='left')
    print("--- Placeholder for joining schedule data (game_date) ---")
    if 'game_date' not in nba_filt.columns: nba_filt['game_date'] = pd.NaT # Add placeholder if no merge
    # --- END PLACEHOLDER ---

    nba_filt['Season'] = year + 1 # R used year+1 for Season value

    # Final filter (R: NBA24Filt = NBA24Filt[NBA24Filt$D5 != 0,]) - applied after potential name mapping
    # Assuming 0 means player slot was empty or unmapped, filter these out
    # Check D5 specifically, assuming it must be populated
    nba_filt = nba_filt[nba_filt['D5'] != 0]
    nba_filt = nba_filt[~pd.isna(nba_filt['D5'])] # Also check for NaN/NaT if names were mapped
    print(f"Rows after final D5 != 0 filter: {len(nba_filt)}")


    # Ensure final columns exist before selecting
    final_cols_present = [col for col in final_cols if col in nba_filt.columns]
    if 'game_date' in nba_filt.columns: final_cols_present.append('game_date')
    final_cols_present.append('Season')

    nba_rapm_output = nba_filt[final_cols_present]

    end_time = time.time()
    print(f"Finished processing RAPM for {file_path}. Time: {end_time - start_time:.2f} seconds.")

    return nba_rapm_output

# --- Example Usage (mimicking process_factors.R calls) ---
if __name__ == '__main__':
    # Example for one season/file
    file_to_process = "NBA25.csv" # Make sure this file exists in the same directory or provide full path
    processing_year = 2024
    processing_season = "2024-25"

    print(f"\n--- Starting RAPM processing for {processing_season} ---")
    rapm_df = process_rapm_py(file_to_process, processing_year, processing_season)

    if rapm_df is not None:
        print(f"\nRAPM DataFrame ({processing_season}) Head:")
        print(rapm_df.head())
        output_filename = f"RAPM{processing_season[-2:]}.csv"
        try:
            rapm_df.to_csv(output_filename, index=False)
            print(f"Saved results to {output_filename}")
        except Exception as e:
            print(f"Error saving {output_filename}: {e}")

