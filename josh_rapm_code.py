import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.sparse import lil_matrix, csr_matrix, vstack
from sklearn.linear_model import Ridge
from itertools import product
from collections import defaultdict

# We assume you already have `playoff_dates` and `is_playoff(date, playoff_dates)` defined elsewhere
# e.g.:
# from playoffs_config import playoff_dates, is_playoff
# For demonstration we assume they're in scope.

# If you like, pick your playoff weight factor:
PLAYOFF_WEIGHT_FACTOR = 1.0

# Standardized cross-era fraction of game possessions an average player plays.
f_crossperiod = 0.45

###############################################################################
# 1) GARBAGE TIME HELPER
###############################################################################
def is_garbage_time(row):
    season = row['season']
    period = int(row['period'])
    score_diff = row['score_diff']
    poss_from_end = row['poss_count_from_end']

    # EXACT logic from the original snippet
    if season in range(1999, 2001):
         if ((period in [1, 2, 3] and score_diff >= 25) or
            (period == 4 and score_diff >= 20) or
            (period == 4 and score_diff > 10 and poss_from_end<=4) or
            (period == 4 and score_diff > 8 and poss_from_end<=2) or
            (period == 4 and score_diff > 15 and poss_from_end<=6)):
            return True
    elif season in [1997, 1998, 2001, 2002, 2003, 2004, 2005, 2006]:
         if ((period in [1, 2, 3] and score_diff >= 25) or
            (period == 4 and score_diff >= 20) or
            (period == 4 and score_diff > 10 and poss_from_end<=4) or
            (period == 4 and score_diff > 8 and poss_from_end<=2) or
            (period == 4 and score_diff > 15 and poss_from_end<=6)):
            return True
    elif season in range(2007, 2017):
        if ((period in [1, 2, 3] and score_diff >= 25) or
            (period == 4 and score_diff >= 20) or
            (period == 4 and score_diff > 10 and poss_from_end<=4) or
            (period == 4 and score_diff > 8 and poss_from_end<=2) or
            (period == 4 and score_diff > 15 and poss_from_end<=6)):
            return True
    elif season in range(2018, 2023):
        if ((period in [1, 2, 3] and score_diff >= 25) or
            (period == 4 and score_diff >= 20) or
            (period == 4 and score_diff > 10 and poss_from_end<=4) or
            (period == 4 and score_diff > 8 and poss_from_end<=2) or
            (period == 4 and score_diff > 15 and poss_from_end<=6)):
            return True
    elif season in range(2023, 2026):
        # covers 2023..2025
        if ((period in [1, 2, 3] and score_diff >= 25) or
            (period == 4 and score_diff >= 20) or
            (period == 4 and score_diff >= 12 and poss_from_end<=4) or
            (period == 4 and score_diff > 8  and poss_from_end<=2) or
            (period == 4 and score_diff > 15 and poss_from_end<=6)):
            return True
    return False

###############################################################################
# 2) PROCESS DATA
###############################################################################
def process_data(data):
    """
    Convert the raw data -> DataFrame, then:
      - Enforce data types
      - Build 'poss_count_from_end'
      - Build scoreboard-based 'score_diff'
      - Filter out garbage time
    """
    columns = [
        'home_poss','pts','a1','a2','a3','a4','a5',
        'h1','h2','h3','h4','h5','season','date','period','gameid'
    ]
    df = pd.DataFrame(data, columns=columns)

    # Convert data types
    df = df.astype({
        'home_poss': 'int64',
        'pts': 'int64',
        'a1': 'int64',
        'a2': 'int64',
        'a3': 'int64',
        'a4': 'int64',
        'a5': 'int64',
        'h1': 'int64',
        'h2': 'int64',
        'h3': 'int64',
        'h4': 'int64',
        'h5': 'int64',
        'season': 'int64',
        'date': 'object',
        'period': 'int64',
        'gameid': 'object'
    })

    # Count possessions from end of game
    df['poss_count_from_end'] = df.groupby('gameid').cumcount(ascending=False)

    # Build scoreboard -> 'score_diff'
    df['score_diff'] = 0
    scores = {}
    for gid in df['gameid'].unique():
        scores[gid] = {'home': 0, 'away': 0}

    for i, row in df.iterrows():
        gid = row['gameid']
        home_score = scores[gid]['home']
        away_score = scores[gid]['away']
        if row['home_poss'] == 1:
            cdiff = home_score - away_score
        else:
            cdiff = away_score - home_score
        df.at[i, 'score_diff'] = cdiff

        if row['home_poss'] == 1:
            scores[gid]['home'] += row['pts']
        else:
            scores[gid]['away'] += row['pts']

    # Filter out garbage time
    df = df[~df.apply(is_garbage_time, axis=1)]
    df.reset_index(drop=True, inplace=True)
    return df

###############################################################################
# 3) BUILD B2B MAP
###############################################################################
def build_b2b_map(df):
    """
    Build a map: (player, date_str) -> 1 if back-to-back, else 0
    """
    df['date'] = pd.to_datetime(df['date'])
    player_dates = defaultdict(list)

    for i, row in df.iterrows():
        date_ = row['date']
        players = [row[f'a{k}'] for k in range(1,6)] + [row[f'h{k}'] for k in range(1,6)]
        for p in players:
            p_str = str(p)
            player_dates[p_str].append(date_)

    for p in player_dates:
        player_dates[p] = sorted(list(set(player_dates[p])))

    player_b2b_map = {}
    for p, dates in player_dates.items():
        for idx, d in enumerate(dates):
            d_str = d.strftime('%Y-%m-%d')
            if idx == 0:
                player_b2b_map[(p, d_str)] = 0
            else:
                prev_d = dates[idx-1]
                if (d - prev_d).days == 1:
                    player_b2b_map[(p, d_str)] = 1
                else:
                    player_b2b_map[(p, d_str)] = 0
    return player_b2b_map

###############################################################################
# 4) BUILD 6-DAY POSSESSION MAP
###############################################################################
def build_6day_possessions_map(df):
    """
    For each player p and date d, count how many possessions p had
    in the preceding 6 days (i.e., [d-6, d) ).
    """
    df['date'] = pd.to_datetime(df['date'])

    player_day_count = defaultdict(int)
    for i, row_ in df.iterrows():
        day_str = row_['date'].strftime('%Y-%m-%d')
        away_pl = [row_[f'a{k}'] for k in range(1,6)]
        home_pl = [row_[f'h{k}'] for k in range(1,6)]
        for p in away_pl + home_pl:
            player_day_count[(str(p), day_str)] += 1

    all_dates = sorted({ r_['date'].strftime('%Y-%m-%d') for _, r_ in df.iterrows() })
    day_to_obj = { ds: datetime.strptime(ds, '%Y-%m-%d') for ds in all_dates }

    player_day_counts_map = defaultdict(dict)
    for (p_str, ds), c_ in player_day_count.items():
        player_day_counts_map[p_str][ds] = c_

    past6_map = {}
    def get_daycount(pstr, dstr):
        return player_day_counts_map[pstr].get(dstr, 0)

    for ds in all_dates:
        day_obj = day_to_obj[ds]
        day_obj_start = day_obj - timedelta(days=6)
        for p_str in player_day_counts_map.keys():
            total_6 = 0
            for ds2 in all_dates:
                d2_obj = day_to_obj[ds2]
                if (day_obj_start <= d2_obj) and (d2_obj < day_obj):
                    total_6 += get_daycount(p_str, ds2)
            past6_map[(p_str, ds)] = total_6

    return past6_map

###############################################################################
# 5) ADD CONSECUTIVE POSSESSIONS
###############################################################################
def add_consecutive_possessions(df):
    """
    For each row, set 'off_consecutive_prop' and 'def_consecutive_prop'
    to the average consecutive possessions so far for the offense
    and defense lineups in that game.
    """
    df['off_consecutive_prop'] = 0.0
    df['def_consecutive_prop'] = 0.0
    df['date'] = pd.to_datetime(df['date'])

    for gid in df['gameid'].unique():
        game_df = df[df['gameid'] == gid].sort_values(by=['date','period']).copy()
        game_consec = {}
        prev_period = None

        for i, row_ in game_df.iterrows():
            cur_period = row_['period']
            # Example: reset counters if we jump from period<3 to period>=3
            if prev_period is not None and (prev_period < 3 and cur_period >= 3):
                game_consec = {}

            away_pl = [row_[f'a{k}'] for k in range(1, 6)]
            home_pl = [row_[f'h{k}'] for k in range(1, 6)]

            if row_['home_poss'] == 1:
                off_players = home_pl
                def_players = away_pl
            else:
                off_players = away_pl
                def_players = home_pl

            # increment consecutive count
            for p in off_players + def_players:
                p_str = str(p)
                game_consec[p_str] = game_consec.get(p_str, 0) + 1

            off_counts = [game_consec[str(p)] for p in off_players]
            def_counts = [game_consec[str(p)] for p in def_players]

            df.at[i, 'off_consecutive_prop'] = np.mean(off_counts)
            df.at[i, 'def_consecutive_prop'] = np.mean(def_counts)

            prev_period = cur_period

    return df

###############################################################################
# 6) BUILD DESIGN MATRIX + sample_weight
###############################################################################
def build_design_matrix(df, player_to_col, player_b2b_map, past6_map, max_poly=3):
    """
    Build X (sparse) and y from df:
      - Off/Def indicators
      - polynomial expansions of score_diff
      - b2b, consecutive, 6-day features
      - home offense/defense indicators

    In addition, we create `sample_weight` to emphasize playoff possessions.
    """
    off_b2b_prop_idx = player_to_col.get('off_b2b_prop', None)
    def_b2b_prop_idx = player_to_col.get('def_b2b_prop', None)
    off_consec_idx   = player_to_col.get('off_consecutive_prop', None)
    def_consec_idx   = player_to_col.get('def_consecutive_prop', None)
    off_6day_idx     = player_to_col.get('off_6day_poss', None)
    def_6day_idx     = player_to_col.get('def_6day_poss', None)
    home_off_idx     = player_to_col.get('home_off', None)
    home_def_idx     = player_to_col.get('home_def', None)

    n_samples = len(df)
    n_features= len(player_to_col)
    X = lil_matrix((n_samples, n_features), dtype=np.float64)
    y = np.zeros(n_samples, dtype=np.float64)

    # Build sample_weight array: 2.0 if it's playoffs, else 1.0
    sample_weight = np.ones(n_samples, dtype=np.float64)

    for i, row_ in enumerate(df.itertuples()):
        # row_ is a Pandas namedtuple with fields from df
        # Access as row_.date, row_.pts, etc.
        home_poss = row_.home_poss
        pts = float(row_.pts)
        period_ = int(row_.period)
        score_diff = float(row_.score_diff)
        date_obj = pd.to_datetime(row_.date).date()

        away_players = [getattr(row_, f'a{k}') for k in range(1,6)]
        home_players = [getattr(row_, f'h{k}') for k in range(1,6)]

        # Apply playoff weighting
        # (We assume `is_playoff(date_obj, playoff_dates)` is defined externally)
        if is_playoff(date_obj, playoff_dates):
            sample_weight[i] = PLAYOFF_WEIGHT_FACTOR

        # OFF/DEF indicator
        if home_poss == 1:
            roles = ['def']*5 + ['off']*5
            offense_players = home_players
            defense_players = away_players
            if home_off_idx is not None:
                X[i, home_off_idx] = 1.0
        else:
            roles = ['off']*5 + ['def']*5
            offense_players = away_players
            defense_players = home_players
            if home_def_idx is not None:
                X[i, home_def_idx] = 1.0

        # Score_diff polynomial expansions
        for p_ in range(1, max_poly+1):
            col_name = f"sd_poly{p_}_period{period_}"
            c_idx = player_to_col.get(col_name, None)
            if c_idx is not None:
                val = (score_diff**p_)
                X[i, c_idx] = val

        # b2b
        date_str = date_obj.strftime('%Y-%m-%d')
        off_b2b_count = sum(player_b2b_map.get((str(p), date_str), 0) for p in offense_players)
        def_b2b_count = sum(player_b2b_map.get((str(p), date_str), 0) for p in defense_players)
        off_b2b_prop  = off_b2b_count / 5.0
        def_b2b_prop  = def_b2b_count / 5.0

        if off_b2b_prop_idx is not None:
            X[i, off_b2b_prop_idx] = off_b2b_prop
        if def_b2b_prop_idx is not None:
            X[i, def_b2b_prop_idx] = def_b2b_prop

        # consecutive
        if off_consec_idx is not None:
            X[i, off_consec_idx] = row_.off_consecutive_prop
        if def_consec_idx is not None:
            X[i, def_consec_idx] = row_.def_consecutive_prop

        # 6-day
        if off_6day_idx is not None:
            off_vals = [past6_map.get((str(p), date_str),0) for p in offense_players]
            X[i, off_6day_idx] = np.mean(off_vals) if off_vals else 0.0
        if def_6day_idx is not None:
            def_vals = [past6_map.get((str(p), date_str),0) for p in defense_players]
            X[i, def_6day_idx] = np.mean(def_vals) if def_vals else 0.0

        # Mark each player's OFF/DEF
        players = away_players + home_players
        for p_, r_ in zip(players, roles):
            key = f"{p_}_{r_}"
            c_idx = player_to_col.get(key, None)
            if c_idx is not None:
                X[i, c_idx] = 1.0

        y[i] = pts

    return X.tocsr(), y, sample_weight

###############################################################################
# 7) ALTERNATING MINIMIZATION RAPM WITH SAMPLE WEIGHTS
###############################################################################
def alternating_minimization_RAPM(
        X, y, sample_weight,
        player_to_col, all_players,
        alpha_offense, alpha_defense, alpha_diff,
        max_iter=200, tol=1e-4
    ):
    """
    Solve RAPM with partial decoupling: offense, defense,
    polynomial expansions, b2b, consecutive, 6day, etc.
    We pass sample_weight to each Ridge fit to emphasize playoffs.
    """
    n_samples, n_features = X.shape
    beta = np.zeros(n_features, dtype=np.float64)

    offense_indices = []
    defense_indices = []
    b2b_indices     = []
    consec_indices  = []
    day6_indices    = []
    home_indices    = []
    poly_indices    = []

    for k,v in player_to_col.items():
        if k in ['off_b2b_prop','def_b2b_prop']:
            b2b_indices.append(v)
        elif k in ['off_consecutive_prop','def_consecutive_prop']:
            consec_indices.append(v)
        elif k in ['off_6day_poss','def_6day_poss']:
            day6_indices.append(v)
        elif k in ['home_off','home_def']:
            home_indices.append(v)
        elif k.endswith('_off'):
            offense_indices.append(v)
        elif k.endswith('_def'):
            defense_indices.append(v)
        elif k.startswith('sd_poly'):
            poly_indices.append(v)

    offense_indices = np.array(offense_indices, dtype=int)
    defense_indices = np.array(defense_indices, dtype=int)
    b2b_indices     = np.array(b2b_indices,     dtype=int)
    consec_indices  = np.array(consec_indices,  dtype=int)
    day6_indices    = np.array(day6_indices,    dtype=int)
    home_indices    = np.array(home_indices,    dtype=int)
    poly_indices    = np.array(poly_indices,    dtype=int)

    X_off  = X[:, offense_indices]
    X_def  = X[:, defense_indices]
    X_b2b  = X[:, b2b_indices]
    X_con  = X[:, consec_indices]
    X_day6 = X[:, day6_indices]
    X_home = X[:, home_indices]
    X_poly = X[:, poly_indices]

    ridge_off   = Ridge(alpha=alpha_offense, fit_intercept=False)
    ridge_def   = Ridge(alpha=alpha_defense, fit_intercept=False)
    ridge_poly  = Ridge(alpha=alpha_diff,    fit_intercept=False)
    ridge_b2b   = Ridge(alpha=100,           fit_intercept=False)
    ridge_con   = Ridge(alpha=100,           fit_intercept=False)
    ridge_day6  = Ridge(alpha=100,           fit_intercept=False)
    ridge_home  = Ridge(alpha=50,            fit_intercept=False)

    residual = y.copy()

    for iteration in range(max_iter):
        beta_prev = beta.copy()

        # Off
        residual += X_off @ beta[offense_indices]
        ridge_off.fit(X_off, residual, sample_weight=sample_weight)
        beta_off = ridge_off.coef_
        beta[offense_indices] = beta_off
        residual -= X_off @ beta_off

        # Def
        residual += X_def @ beta[defense_indices]
        ridge_def.fit(X_def, residual, sample_weight=sample_weight)
        beta_def = ridge_def.coef_
        beta[defense_indices] = beta_def
        residual -= X_def @ beta_def

        # poly
        if poly_indices.size>0:
            residual += X_poly @ beta[poly_indices]
            ridge_poly.fit(X_poly, residual, sample_weight=sample_weight)
            beta_poly = ridge_poly.coef_
            beta[poly_indices] = beta_poly
            residual -= X_poly @ beta_poly

        # b2b
        if b2b_indices.size>0:
            residual += X_b2b @ beta[b2b_indices]
            ridge_b2b.fit(X_b2b, residual, sample_weight=sample_weight)
            beta_b2b = ridge_b2b.coef_
            beta[b2b_indices] = beta_b2b
            residual -= X_b2b @ beta_b2b

        # consecutive
        if consec_indices.size>0:
            residual += X_con @ beta[consec_indices]
            ridge_con.fit(X_con, residual, sample_weight=sample_weight)
            beta_con = ridge_con.coef_
            beta[consec_indices] = beta_con
            residual -= X_con @ beta_con

        # day6
        if day6_indices.size>0:
            residual += X_day6 @ beta[day6_indices]
            ridge_day6.fit(X_day6, residual, sample_weight=sample_weight)
            beta_day6 = ridge_day6.coef_
            beta[day6_indices] = beta_day6
            residual -= X_day6 @ beta_day6

        # home
        if home_indices.size>0:
            residual += X_home @ beta[home_indices]
            ridge_home.fit(X_home, residual, sample_weight=sample_weight)
            beta_home = ridge_home.coef_
            beta[home_indices] = beta_home
            residual -= X_home @ beta_home

        delta_beta = np.linalg.norm(beta - beta_prev)
        if (iteration+1) % 10 == 0:
            logging.info(f"Iteration {iteration+1}, delta_beta={delta_beta:.6f}")
        if delta_beta < tol:
            logging.info(f"Converged after {iteration+1} iterations")
            break
    else:
        logging.info("Max iterations reached without full convergence")

    return beta

###############################################################################
# LOGGING SETUP
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("RAPM with polynomial expansions, plus B2B, consecutive, 6-day, and playoff weighting...")

###############################################################################
# HELPER: CREATE Z-SCORES FOR TOP K
###############################################################################
def add_group_zscore(df, metric, top_k, zcolname):
    # Sort by metric descending, pick top_k
    df_sorted = df.sort_values(metric, ascending=False)
    group = df_sorted.head(top_k)

    # compute mean, std
    mean_ = group[metric].mean()
    std_  = group[metric].std()

    # create new col, default = NaN
    df[zcolname] = np.nan

    if pd.isna(std_) or std_ == 0:
        # If std=0, all players in group had the same value => zscore=0
        for idx in group.index:
            df.at[idx, zcolname] = 0.0
    else:
        for idx in group.index:
            val_ = df.at[idx, metric]
            z_   = (val_ - mean_) / std_
            df.at[idx, zcolname] = z_

###############################################################################
# MAIN LOGIC: SINGLE MULTI‐YEAR FIT WITH PLAYOFF WEIGHTING
###############################################################################
def run_rapm_for_multi_year(start_year, end_year,
                            alpha_offense=2300,
                            alpha_defense=2700,
                            alpha_diff=50):
    """
    Gathers data for all seasons in [start_year..end_year],
    merges them into a single DataFrame, then runs one RAPM fit across
    the entire multi-year window, giving extra weight to playoff possessions.
    """

    # We'll keep reading external name maps just once
    try:
        df_nba_main = pd.read_csv("nbarapm3_2.csv")
    except FileNotFoundError:
        logging.warning("nbarapm3_2.csv not found.")
        df_nba_main = pd.DataFrame(columns=['nba_id','player_name'])

    try:
        df_unmapped = pd.read_csv("unmapped_players.csv")
    except FileNotFoundError:
        logging.warning("unmapped_players.csv not found.")
        df_unmapped = pd.DataFrame(columns=['player_name','nba_id'])

    name_map = {}
    for i, row_ in df_nba_main.iterrows():
        id_str = str(row_['nba_id'])
        name_map[id_str] = row_['player_name']

    for i, row_ in df_unmapped.iterrows():
        id_str = str(row_['nba_id'])
        if id_str not in name_map:
            name_map[id_str] = row_['player_name']

    logging.info(f"==== Multi‐Year RAPM for {start_year}..{end_year} with playoff weighting ====")

    # Collect all data from each season in one DataFrame
    df_all_years = pd.DataFrame()

    for season_year in range(start_year, end_year+1):
        logging.info(f"Loading data for season {season_year} ...")

        # EXACT special handling for 2024 and 2025, as in the prior code
        if season_year in [2024, 2025]:
            df_db = pd.DataFrame()  # from DB might be empty for 2024, 2025
            try:
                temp_csv = pd.read_csv(f"final_rapm_{season_year}.csv")
                temp_csv = temp_csv.astype({
                    'home_poss': 'int64',
                    'pts': 'int64',
                    'a1': 'int64','a2': 'int64','a3': 'int64','a4': 'int64','a5': 'int64',
                    'h1': 'int64','h2': 'int64','h3': 'int64','h4': 'int64','h5': 'int64',
                    'season': 'int64',
                    'date': 'object',
                    'period': 'int64',
                    'gameid': 'object'
                })
                # Keep only that season's data
                temp_csv = temp_csv[temp_csv['season'] == season_year]
                df_this_year = pd.concat([df_db, temp_csv], ignore_index=True)
            except FileNotFoundError:
                logging.warning(f"No final_rapm_{season_year}.csv found.")
                df_this_year = df_db
        else:
            # Standard DB fetch (like the prior code)
            query = f"""
                SELECT home_poss, pts, a1, a2, a3, a4, a5,
                       h1, h2, h3, h4, h5, season, date, period, gameid
                FROM matchups
                WHERE season = {season_year}
            """
            # PLEASE ASSUME fetch_data(query) EXISTS EXACTLY AS IN PRIOR CODE
            data = fetch_data(query)
            if not data:
                logging.info(f"No DB data for {season_year}")
                df_this_year = pd.DataFrame()
            else:
                df_this_year = pd.DataFrame(data, columns=[
                    'home_poss','pts','a1','a2','a3','a4','a5',
                    'h1','h2','h3','h4','h5','season','date','period','gameid'
                ])

        if df_this_year.empty:
            logging.info(f"Empty data for season {season_year}, skipping.")
        else:
            df_all_years = pd.concat([df_all_years, df_this_year], ignore_index=True)

    if df_all_years.empty:
        logging.info(f"No data loaded for {start_year}..{end_year}. Exiting.")
        return pd.DataFrame()

    # -------------------------------------------------------------------------
    # Now we run the pipeline ONCE over all those seasons combined
    # -------------------------------------------------------------------------
    # 1) Process & filter garbage time
    df_processed = process_data(df_all_years)
    if df_processed.empty:
        logging.info(f"After garbage time filter, no data in {start_year}..{end_year}.")
        return pd.DataFrame()

    # 2) Add consecutive possessions
    df_processed = add_consecutive_possessions(df_processed)
    df_processed.reset_index(drop=True, inplace=True)

    # 3) Build global player->column mapping
    wplayers = set(map(str, df_processed[[
        'a1','a2','a3','a4','a5','h1','h2','h3','h4','h5'
    ]].values.flatten()))
    global_player_list = sorted(wplayers)
    player_to_col = {}
    idx = 0

    # Off/Def for each player
    for p_ in global_player_list:
        player_to_col[f"{p_}_off"] = idx; idx+=1
        player_to_col[f"{p_}_def"] = idx; idx+=1

    # sd_poly expansions
    for p_ in range(1,4):
        for period_ in [1,2,3,4]:
            col_name = f"sd_poly{p_}_period{period_}"
            player_to_col[col_name] = idx
            idx += 1

    # home
    player_to_col['home_off'] = idx; idx+=1
    player_to_col['home_def'] = idx; idx+=1

    # b2b
    player_to_col['off_b2b_prop'] = idx; idx+=1
    player_to_col['def_b2b_prop'] = idx; idx+=1

    # consecutive
    player_to_col['off_consecutive_prop'] = idx; idx+=1
    player_to_col['def_consecutive_prop'] = idx; idx+=1

    # 6-day
    player_to_col['off_6day_poss'] = idx; idx+=1
    player_to_col['def_6day_poss'] = idx; idx+=1

    # 4) Build b2b map, 6-day map
    player_b2b_map = build_b2b_map(df_processed)
    past6_map      = build_6day_possessions_map(df_processed)

    # 5) Build design matrix + sample weights
    X_blk, y_blk, w_blk = build_design_matrix(
        df_processed, player_to_col,
        player_b2b_map, past6_map,
        max_poly=3
    )
    # Center the outcome
    y_blk_centered = y_blk - np.mean(y_blk)

    # 6) Fit with alternating minimization, using sample_weight = w_blk
    beta_ = alternating_minimization_RAPM(
        X_blk, y_blk_centered, w_blk,
        player_to_col, global_player_list,
        alpha_offense, alpha_defense, alpha_diff,
        max_iter=200, tol=1e-4
    )

    # 7) Re-center net RAPM so that weighted average = 0
    player_possessions = defaultdict(int)
    for idx2, row_ in df_processed.iterrows():
        away_pl = [row_[f'a{k}'] for k in range(1,6)]
        home_pl = [row_[f'h{k}'] for k in range(1,6)]
        for p_ in away_pl + home_pl:
            player_possessions[str(p_)] += 1

    sumNet = 0.0
    sumPoss= 0.0
    for p_ in global_player_list:
        off_key = f"{p_}_off"
        def_key = f"{p_}_def"
        off_val= beta_[ player_to_col[off_key] ]
        def_val= beta_[ player_to_col[def_key] ]
        net_val= off_val - def_val
        poss   = player_possessions[p_]
        sumNet += net_val * poss
        sumPoss+= poss

    if sumPoss > 0:
        offset = sumNet / sumPoss
        logging.info(f"Re-centering netRAPM by offset={offset:.4f} so weighted avg=0.")
        for p_ in global_player_list:
            off_key = f"{p_}_off"
            def_key = f"{p_}_def"
            beta_[ player_to_col[off_key] ] -= 0.5 * offset
            beta_[ player_to_col[def_key] ] += 0.5 * offset
    else:
        logging.warning("No possessions to re-center (sumPoss=0).")

    # 8) Build final records (per-player)
    gameid_to_totalpos = defaultdict(int)
    for gid_ in df_processed['gameid'].unique():
        game_df_ = df_processed[df_processed['gameid'] == gid_]
        gameid_to_totalpos[gid_] = len(game_df_)

    pl_gm_map = defaultdict(int)
    for i2, row_ in df_processed.iterrows():
        gid_ = row_['gameid']
        away_pl = [row_[f'a{k}'] for k in range(1,6)]
        home_pl = [row_[f'h{k}'] for k in range(1,6)]
        for pp_ in away_pl + home_pl:
            pl_gm_map[(str(pp_), gid_)] += 1

    player_sum_poss    = defaultdict(int)
    player_sum_teampos = defaultdict(int)
    for (p_str, gid_), poss_count_ in pl_gm_map.items():
        player_sum_poss[p_str] += poss_count_
        player_sum_teampos[p_str] += gameid_to_totalpos[gid_]

    fraction_of_teampos = {}
    for p_ in global_player_list:
        p_str = str(p_)
        if p_str not in player_sum_poss or player_sum_teampos[p_str] == 0:
            fraction_of_teampos[p_str] = 0.0
        else:
            fraction_of_teampos[p_str] = player_sum_poss[p_str] / float(player_sum_teampos[p_str])

    # If you want "era fraction" logic, define it or skip. We'll skip it here.
    def compute_f_period(p_str):
        return 0.0  # or your logic if you have fraction_era_avg

    final_records = []
    for p_ in global_player_list:
        p_str  = str(p_)
        off_key = f"{p_}_off"
        def_key = f"{p_}_def"
        ov = beta_[ player_to_col[off_key] ]
        dv = beta_[ player_to_col[def_key] ]
        net_val = ov - dv

        f_player = fraction_of_teampos[p_str]
        f_period = compute_f_period(p_str)

        # "vanilla" per-game
        vanilla_pg = net_val * f_player * 100.0
        # "nuanced" per-game
        if f_period <= 0:
            nuanced_pg = vanilla_pg
        else:
            nuanced_pg = net_val * 100.0 * (f_player / f_period) * f_crossperiod

        final_records.append({
            'player_id': p_str,
            'off': ov,
            'def': dv,
            'total_coefficient': net_val,
            'fraction_player': f_player,
            'f_period': f_period,
            'vanilla_per_game': vanilla_pg,
            'nuanced_per_game': nuanced_pg
        })

    df_finalplayers = pd.DataFrame(final_records)
    df_finalplayers['player_name'] = df_finalplayers['player_id'].apply(
        lambda pid: name_map.get(pid, f"ID={pid}")
    )

    # Reorder columns
    df_finalplayers = df_finalplayers[
        [
            'player_id',
            'player_name',
            'off',
            'def',
            'total_coefficient',
            'fraction_player',
            'f_period',
            'vanilla_per_game',
            'nuanced_per_game'
        ]
    ]

    # Add top-K z-scores
    add_group_zscore(df_finalplayers, 'vanilla_per_game', 10,  'zscore_top10_vanilla')
    add_group_zscore(df_finalplayers, 'vanilla_per_game', 25,  'zscore_top25_vanilla')
    add_group_zscore(df_finalplayers, 'vanilla_per_game', 50,  'zscore_top50_vanilla')
    add_group_zscore(df_finalplayers, 'nuanced_per_game', 10,  'zscore_top10_nuanced')
    add_group_zscore(df_finalplayers, 'nuanced_per_game', 25,  'zscore_top25_nuanced')
    add_group_zscore(df_finalplayers, 'nuanced_per_game', 50,  'zscore_top50_nuanced')

    df_finalplayers_sorted = df_finalplayers.sort_values(
        'nuanced_per_game', ascending=False
    )

    logging.info(f"TOP 20 by 'nuanced_per_game' for {start_year}..{end_year}:")
    for _, row_ in df_finalplayers_sorted.head(20).iterrows():
        logging.info(
            f"ID={row_['player_id']}, "
            f"Name={row_['player_name']}, "
            f"TotalCoeff={row_['total_coefficient']:.3f}, "
            f"VanillaPG={row_['vanilla_per_game']:.3f}, "
            f"NuancedPG={row_['nuanced_per_game']:.3f}, "
            f"zscore_top10_nuanced={row_['zscore_top10_nuanced']}"
        )

    # Example CSV save:
    df_finalplayers.to_csv(f"rapm_{start_year}_{end_year}_playoff_weighted.csv", index=False)

    return df_finalplayers


###############################################################################
# EXAMPLE USAGE
###############################################################################
if __name__ == "__main__":
    # Example: run multi-year range from 2015..2017,
    # weighting playoffs more heavily:
    result_2015_17 = run_rapm_for_multi_year(2009, 2010)
    logging.info(f"Done. Final shape: {result_2015_17.shape}")