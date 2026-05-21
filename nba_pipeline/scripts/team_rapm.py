#!/usr/bin/env python3
"""
Team RAPM Analysis

Runs RAPM at the team level instead of player level.
Uses the same processed parquet files as rapm.py but builds a 30-team design matrix.

Usage:
    python team_rapm.py RAPM 26 26 ALL
    python team_rapm.py RAPM 23 26 ALL --pure
    python team_rapm.py TS 24 26 ALL
    python team_rapm.py TOV 23 26 ALL
"""

import argparse
import logging
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_ROOT.parent
PROCESSED_DIR = PIPELINE_ROOT / "processed"
RAW_DIR = PIPELINE_ROOT / "raw_data"
RESULTS_DIR = PIPELINE_ROOT / "results" / "teams"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import shared utilities from rapm.py
from rapm import (
    detect_file_type_and_prepare,
    _load_files_sequential,
    _load_files_parallel,
    half_life_to_decay_base,
    compute_time_decay_weights,
    N_CORES,
    RUBBERBAND_BINS,
)


def build_game_team_map(years, season_type):
    """
    Build game_id -> (home_team, away_team) mapping from raw PBP.
    Also builds a set of home player IDs per game to determine offense/defense.

    Returns:
        game_teams: dict of game_id -> {'home': abbr, 'away': abbr}
        game_home_players: dict of game_id -> set of home player IDs
    """
    logging.info("Building game-to-team mapping from raw PBP...")
    game_teams = {}
    game_home_players = {}

    for year in years:
        files = []
        if season_type in ['RS', 'ALL']:
            files.append(RAW_DIR / f"NBA{year:02d}.parquet")
        if season_type in ['PS', 'ALL']:
            files.append(RAW_DIR / f"NBA{year:02d}_PS.parquet")

        for f in files:
            if not f.exists():
                logging.warning(f"File not found: {f}")
                continue

            raw = pd.read_parquet(f, columns=[
                'game_id', 'player1_id', 'player1_team_abbreviation',
                'home_player1', 'home_player2', 'home_player3', 'home_player4', 'home_player5',
                'away_player1', 'away_player2', 'away_player3', 'away_player4', 'away_player5',
            ])

            for gid, grp in raw.groupby('game_id'):
                # Normalize game_id: strip leading zeros to match processed parquet format
                gid_str = str(gid).lstrip('0') or '0'

                # Collect ALL home/away player IDs across entire game
                home_pids = set()
                away_pids = set()
                for i in range(1, 6):
                    home_pids.update(grp[f'home_player{i}'].dropna().astype(int).unique())
                    away_pids.update(grp[f'away_player{i}'].dropna().astype(int).unique())

                game_home_players[gid_str] = home_pids

                # Find team abbreviations from events
                events = grp[grp['player1_team_abbreviation'].notna() & grp['player1_id'].notna()]
                home_team = None
                away_team = None
                for _, evt in events.iterrows():
                    pid = int(evt['player1_id'])
                    team = evt['player1_team_abbreviation']
                    if pid in home_pids and home_team is None:
                        home_team = team
                    elif pid in away_pids and away_team is None:
                        away_team = team
                    if home_team and away_team:
                        break

                if home_team and away_team:
                    game_teams[gid_str] = {'home': home_team, 'away': away_team}

            logging.info(f"  Processed {f.name}: {raw['game_id'].nunique() if len(raw) > 0 else 0} games")

    logging.info(f"Mapped {len(game_teams):,} games to teams")
    return game_teams, game_home_players


def transform_to_team_format(df, game_teams, game_home_players):
    """
    Transform possession data to team-level format.
    Determines offensive/defensive team by checking if O1 is a home player.
    """
    logging.info("Transforming to team-level format...")

    rows = []
    unmapped = 0

    for _, row in df.iterrows():
        gid = str(row['game_id'])
        teams = game_teams.get(gid)
        home_pids = game_home_players.get(gid)

        if teams is None or home_pids is None:
            unmapped += 1
            continue

        # O1 tells us which team is on offense
        o1 = int(row['O1'])
        if o1 in home_pids:
            off_team = teams['home']
            def_team = teams['away']
        else:
            off_team = teams['away']
            def_team = teams['home']

        rows.append({
            'off_team': off_team,
            'def_team': def_team,
            'off_diff': float(row['Off_Diff']),
            'game_date': row['game_date'],
            'period': int(row['period']),
            'game_id': gid,
            'score_margin': row.get('score_margin', 0),
        })

    if unmapped > 0:
        logging.warning(f"  {unmapped:,} possessions unmapped (game not found in raw PBP)")

    result = pd.DataFrame(rows)
    logging.info(f"  {len(result):,} team-level possessions")
    return result


def run_team_rapm(input_files, pure=False, prefix=None, timedecay=False, half_life=None,
                  alpha=100, years=None, season_type='ALL'):
    """
    Run team-level RAPM analysis.

    Args:
        alpha: Regularization strength (default 100, much lower than player RAPM's 3000
               since 30 teams are well-identified)
    """
    # Load processed data
    if len(input_files) > 1:
        all_dfs = _load_files_parallel(input_files, pure, prefix=prefix)
    else:
        all_dfs = _load_files_sequential(input_files, pure, prefix=prefix)

    if not all_dfs:
        logging.error("No files loaded.")
        return None

    df = pd.concat(all_dfs, ignore_index=True)
    logging.info(f"Combined: {len(df):,} possessions")

    # Build game->team mapping
    game_teams, game_home_players = build_game_team_map(years, season_type)

    # Transform to team format
    team_df = transform_to_team_format(df, game_teams, game_home_players)

    if len(team_df) == 0:
        logging.error("No possessions after team mapping.")
        return None

    # Get all unique teams
    all_teams = sorted(set(team_df['off_team'].unique()) | set(team_df['def_team'].unique()))
    n_teams = len(all_teams)
    logging.info(f"Found {n_teams} unique teams")

    # Build team-to-column mapping
    team_to_col = {}
    for i, team in enumerate(all_teams):
        team_to_col[f"{team}_off"] = i
        team_to_col[f"{team}_def"] = n_teams + i

    n_features = 2 * n_teams

    # Build design matrix (dense is fine for 60 columns)
    n_samples = len(team_df)
    X = np.zeros((n_samples, n_features), dtype=np.float64)
    y = team_df['off_diff'].values.astype(np.float64)

    for i, (_, row) in enumerate(team_df.iterrows()):
        off_col = team_to_col[f"{row['off_team']}_off"]
        def_col = team_to_col[f"{row['def_team']}_def"]
        X[i, off_col] = 1.0
        X[i, def_col] = 1.0

    logging.info(f"Design matrix: {X.shape}")

    # Time decay weights
    weights = None
    if timedecay:
        decay_half_life = half_life if half_life else 700
        decay_base = half_life_to_decay_base(decay_half_life)
        dates = pd.to_datetime(team_df['game_date'], errors='coerce')
        ref_date = dates.max()
        age_days = (ref_date - dates).dt.total_seconds() / 86400.0
        age_days = age_days.fillna(0)
        weights = np.power(decay_base, age_days.values)
        weights = np.clip(weights, 0.0, 1.0)
        logging.info(f"Time decay: half-life={decay_half_life}, weights range [{weights.min():.4f}, {weights.max():.4f}]")

    # Center outcome
    if weights is not None:
        y_mean = np.average(y, weights=weights)
    else:
        y_mean = np.mean(y)
    y_centered = y - y_mean
    logging.info(f"Centered outcome, mean={y_mean:.4f}")

    # Apply weights
    if weights is not None:
        sqrt_w = np.sqrt(weights)
        X = X * sqrt_w[:, np.newaxis]
        y_centered = y_centered * sqrt_w

    # Alternating ridge: offense block, then defense block
    off_indices = [team_to_col[f"{t}_off"] for t in all_teams]
    def_indices = [team_to_col[f"{t}_def"] for t in all_teams]

    beta = np.zeros(n_features)
    residual = y_centered.copy()

    ridge_off = Ridge(alpha=alpha, fit_intercept=False)
    ridge_def = Ridge(alpha=alpha, fit_intercept=False)

    X_off = X[:, off_indices]
    X_def = X[:, def_indices]

    for iteration in range(200):
        beta_prev = beta.copy()

        # Offense
        residual += X_off @ beta[off_indices]
        ridge_off.fit(X_off, residual)
        beta[off_indices] = ridge_off.coef_
        residual -= X_off @ ridge_off.coef_

        # Defense
        residual += X_def @ beta[def_indices]
        ridge_def.fit(X_def, residual)
        beta[def_indices] = ridge_def.coef_
        residual -= X_def @ ridge_def.coef_

        delta = np.linalg.norm(beta - beta_prev)
        if delta < 1e-6:
            logging.info(f"Converged after {iteration+1} iterations")
            break

    # Count possessions per team
    team_off_poss = team_df['off_team'].value_counts().to_dict()
    team_def_poss = team_df['def_team'].value_counts().to_dict()

    # Re-center
    total_poss = sum(team_off_poss.get(t, 0) + team_def_poss.get(t, 0) for t in all_teams)
    off_mean = sum(beta[team_to_col[f"{t}_off"]] * (team_off_poss.get(t, 0) + team_def_poss.get(t, 0))
                   for t in all_teams) / total_poss
    def_mean = sum(beta[team_to_col[f"{t}_def"]] * (team_off_poss.get(t, 0) + team_def_poss.get(t, 0))
                   for t in all_teams) / total_poss

    for t in all_teams:
        beta[team_to_col[f"{t}_off"]] -= off_mean
        beta[team_to_col[f"{t}_def"]] -= def_mean

    # Build results
    results = []
    for t in all_teams:
        off_val = beta[team_to_col[f"{t}_off"]] * 100
        def_val = beta[team_to_col[f"{t}_def"]] * 100
        net = off_val - def_val
        results.append({
            'team': t,
            'off': round(off_val, 2),
            'def': round(def_val, 2),
            'net_rapm': round(net, 2),
            'off_poss': team_off_poss.get(t, 0),
            'def_poss': team_def_poss.get(t, 0),
            'total_poss': team_off_poss.get(t, 0) + team_def_poss.get(t, 0),
        })

    results_df = pd.DataFrame(results).sort_values('net_rapm', ascending=False).reset_index(drop=True)
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description='Run team-level RAPM analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python team_rapm.py RAPM 26 26 ALL --pure
  python team_rapm.py RAPM 23 26 ALL --pure
  python team_rapm.py TS 24 26 ALL
  python team_rapm.py TOV 23 26 ALL
  python team_rapm.py RAPM 21 26 ALL --pure --timedecay --half-life 700
        """
    )

    parser.add_argument('prefix',
                        choices=['RAPM', 'TOV', 'REB', 'TS', 'RIM_FREQ', 'RIM_FG_PCT',
                                 'MIDRANGE_FREQ', 'MIDRANGE_FG_PCT', 'TRANSITION_FREQ', 'TRANSITION_RIM',
                                 'INITIAL_EV', 'SPECIAL_RAPM', 'SQ_POSS', 'FT_PREMIUM',
                                 'CONTEST', 'SECOND_CHANCE', 'BADPASS_TOV', 'SCORING_TOV'],
                        help='Metric to analyze')
    parser.add_argument('start_year', type=int, help='Start year (2-digit)')
    parser.add_argument('end_year', type=int, help='End year (2-digit)')
    parser.add_argument('season_type', choices=['RS', 'PS', 'ALL'], help='Season type')
    parser.add_argument('--pure', action='store_true', help='Pure RAPM mode')
    parser.add_argument('--timedecay', '-td', action='store_true', help='Time decay weighting')
    parser.add_argument('--half-life', type=float, default=None, help='Half-life in days (default: 700)')
    parser.add_argument('--alpha', type=float, default=100, help='Ridge alpha (default: 100)')

    args = parser.parse_args()

    if args.half_life is not None:
        args.timedecay = True

    # Prefix aliasing
    PREFIX_FILE_MAP = {'BADPASS_TOV': 'TOV', 'SCORING_TOV': 'TOV'}
    file_prefix = PREFIX_FILE_MAP.get(args.prefix, args.prefix)

    # Build file list
    input_files = []
    years = list(range(args.start_year, args.end_year + 1))
    for year in years:
        if args.season_type in ['RS', 'ALL']:
            input_files.append(str(PROCESSED_DIR / f"{file_prefix}{year:02d}.parquet"))
        if args.season_type in ['PS', 'ALL']:
            input_files.append(str(PROCESSED_DIR / f"{file_prefix}{year:02d}_PS.parquet"))

    logging.info(f"Team RAPM: {args.prefix} {args.start_year}-{args.end_year} {args.season_type}")

    results_df = run_team_rapm(
        input_files, pure=args.pure, prefix=args.prefix,
        timedecay=args.timedecay, half_life=args.half_life,
        alpha=args.alpha, years=years, season_type=args.season_type,
    )

    if results_df is None:
        logging.error("Analysis failed.")
        sys.exit(1)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    year_range = f"{args.start_year:02d}_{args.end_year:02d}" if args.start_year != args.end_year else f"{args.start_year:02d}"
    season = args.season_type.lower()
    suffix_parts = [season]
    if args.pure:
        suffix_parts.append("pure")
    if args.timedecay:
        td_half = int(args.half_life) if args.half_life else 700
        suffix_parts.append(f"td{td_half}")
    suffix = "_".join(suffix_parts)

    output_file = RESULTS_DIR / f"team_{args.prefix.lower()}_{year_range}_{suffix}_results.csv"
    results_df.to_csv(output_file, index=False)
    logging.info(f"Saved to {output_file}")

    # Display results
    logging.info("\n" + "=" * 70)
    logging.info(f"TEAM {args.prefix} RANKINGS")
    logging.info("=" * 70)
    for i, row in results_df.iterrows():
        logging.info(
            f"{i+1:2d}. {row['team']:4s} | "
            f"Net: {row['net_rapm']:+6.2f} | "
            f"Off: {row['off']:+6.2f} | "
            f"Def: {row['def']:+6.2f} | "
            f"Poss: {row['total_poss']:,}"
        )

    return 0


if __name__ == '__main__':
    sys.exit(main())
