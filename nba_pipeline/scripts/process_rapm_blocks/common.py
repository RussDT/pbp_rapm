"""
Common utilities and shared functions for NBA RAPM data processing.

This module contains:
- Helper functions (series_contains, case_when)
- Constants (RIM_ACTION_TYPES, paths)
- Base processing functions (_base_processing, _finalize_df)
- External data loading (fetch_player_stats_supabase, load_game_dates)
- Player propagation functions
"""

import pandas as pd
import numpy as np
import re
import time
import warnings
import requests
import io
import os
from pathlib import Path
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the pipeline root directory (parent of scripts/)
SCRIPT_DIR = Path(__file__).parent.parent  # Go up from process_rapm_blocks to scripts
PIPELINE_ROOT = SCRIPT_DIR.parent
RAW_DATA_DIR = PIPELINE_ROOT / "raw_data"
PROCESSED_DIR = PIPELINE_ROOT / "processed"
BREF_FT_PCT_LOOKUP = PIPELINE_ROOT / "external" / "bref_ft_pct_1997_2000.csv"

# Ignore SettingWithCopyWarning, common in this type of step-by-step processing
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings(
    'ignore',
    message='This pattern is interpreted as a regular expression, and has match groups.*',
    category=UserWarning,
)

ASSIST_TAG_PATTERN = r'\([^()]+?\s+\d+\s+AST\)'
RIM_DESC_PATTERN = r'layup|dunk| tip '
TEAM_ABBREVIATION_LABELS = {
    "ATL": ["Atlanta", "Hawks"],
    "BKN": ["Brooklyn", "Nets"],
    "BOS": ["Boston", "Celtics"],
    "CHA": ["Charlotte", "Bobcats", "Hornets"],
    "CHI": ["Chicago", "Bulls"],
    "CLE": ["Cleveland", "Cavaliers"],
    "DAL": ["Dallas", "Mavericks"],
    "DEN": ["Denver", "Nuggets"],
    "DET": ["Detroit", "Pistons"],
    "GSW": ["Golden State", "Warriors"],
    "HOU": ["Houston", "Rockets"],
    "IND": ["Indiana", "Pacers"],
    "LAC": ["LA Clippers", "Los Angeles Clippers", "Clippers"],
    "LAL": ["Los Angeles Lakers", "Lakers"],
    "MEM": ["Memphis", "Grizzlies"],
    "MIA": ["Miami", "Heat"],
    "MIL": ["Milwaukee", "Bucks"],
    "MIN": ["Minnesota", "Timberwolves"],
    "NJN": ["New Jersey", "Nets"],
    "NOH": ["New Orleans", "Hornets"],
    "NOK": ["New Orleans Oklahoma City", "Hornets"],
    "NOP": ["New Orleans", "Hornets", "Pelicans"],
    "NYK": ["New York", "Knicks"],
    "OKC": ["Oklahoma City", "Thunder"],
    "ORL": ["Orlando", "Magic"],
    "PHI": ["Philadelphia", "76ers", "Sixers", "Seventy Sixers"],
    "PHX": ["Phoenix", "Suns"],
    "POR": ["Portland", "Trail Blazers", "Blazers"],
    "SAC": ["Sacramento", "Kings"],
    "SAS": ["San Antonio", "Spurs"],
    "SEA": ["Seattle", "SuperSonics", "Sonics"],
    "TOR": ["Toronto", "Raptors"],
    "UTA": ["Utah", "Jazz"],
    "WAS": ["Washington", "Wizards"],
}


# --- Helper Functions ---

def series_contains(series, pattern, case=True, na=False, regex=True):
    """Checks if a string pattern is contained within Series elements."""
    if not pd.api.types.is_string_dtype(series):
        series = series.astype(str)
    return series.fillna('').str.contains(pattern, case=case, na=na, regex=regex)


def case_when(*args):
    """Mimics R's dplyr::case_when using numpy.select.
    Expects pairs of condition (boolean Series) and result,
    with the last argument being the default result.
    """
    conditions = [args[i] for i in range(0, len(args) - 1, 2)]
    results = [args[i] for i in range(1, len(args) - 1, 2)]
    default = args[-1]
    conditions = [pd.Series(c) if not isinstance(c, pd.Series) else c for c in conditions]
    return np.select(conditions, results, default=default)


def normalize_game_id_series(series):
    """Normalize game IDs to stable 10-character strings."""
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(10)
    )


def normalize_season_end_year(year):
    """Normalize a season end year to a 4-digit integer."""
    year = int(year)
    return 2000 + year if year < 100 else year


def build_season_label_from_end_year(year):
    """Build the canonical season label from an end year."""
    end_year = normalize_season_end_year(year)
    start_year = end_year - 1
    return f"{start_year}-{end_year % 100:02d}"


def dedupe_exact_raw_pbp_rows(nba_df, label):
    """Remove byte-for-byte duplicate raw PBP rows before downstream parsing."""
    if nba_df.empty:
        return nba_df

    deduped = nba_df.drop_duplicates().copy()
    removed = len(nba_df) - len(deduped)
    if removed:
        print(f"      Removed {removed} exact duplicate raw PBP rows from {label}.")
    return deduped


# --- Rim Attempt Detection ---
RIM_ACTION_TYPES = {
    # Layups
    5,    # Layup (generic)
    6,    # Driving Layup
    41,   # Running Layup
    43,   # Alley Oop Layup
    75,   # Driving Finger Roll Layup
    98,   # Cutting Layup Shot
    99,   # Cutting Finger Roll Layup Shot
    100,  # Running Alley Oop Layup Shot
    # Dunks
    7,    # Dunk
    9,    # Driving Dunk
    50,   # Running Dunk
    52,   # Alley Oop Dunk
    87,   # Putback Dunk
    107,  # Tip Dunk Shot
    108,  # Cutting Dunk Shot
    # Tips
    97,   # Tip Layup Shot
}


def is_rim_attempt_check(event_action_type, home_desc, visitor_desc):
    """
    Check if an event is a rim attempt based on action type codes and description patterns.
    Returns True if the shot is a rim attempt (layup, dunk, or tip).
    """
    try:
        if int(event_action_type) in RIM_ACTION_TYPES:
            return True
    except (ValueError, TypeError):
        pass

    desc = str(home_desc).lower() + str(visitor_desc).lower()
    return any(kw in desc for kw in ['layup', 'dunk', ' tip '])


def build_shot_flags_df(nba_df):
    """Build shared shot-classification flags for shot and assist metrics."""
    shot_text = (
        nba_df['home_description'].fillna('').astype(str) + ' ' +
        nba_df['visitor_description'].fillna('').astype(str)
    ).str.strip()
    is_fga = nba_df['event_type'].isin(['MAKE', 'MISS'])
    is_3pt = shot_text.str.contains('3PT', case=False, regex=False, na=False)

    if 'event_action_type' in nba_df.columns:
        event_action_type_num = pd.to_numeric(nba_df['event_action_type'], errors='coerce')
        is_rim = event_action_type_num.isin(RIM_ACTION_TYPES)
    else:
        is_rim = pd.Series(False, index=nba_df.index)

    is_rim = is_rim | shot_text.str.contains(RIM_DESC_PATTERN, case=False, regex=True, na=False)
    is_assisted_make = (
        (nba_df['event_type'] == 'MAKE') &
        shot_text.str.contains(ASSIST_TAG_PATTERN, case=False, regex=True, na=False)
    )
    shot_points = np.select(
        [(nba_df['event_type'] == 'MAKE') & is_3pt, nba_df['event_type'] == 'MAKE'],
        [3, 2],
        default=0,
    ).astype(int)

    return pd.DataFrame({
        'shot_text': shot_text,
        'is_fga': is_fga,
        'is_3pt': is_3pt,
        'is_rim': is_rim,
        'is_assisted_make': is_assisted_make,
        'shot_points': shot_points,
    }, index=nba_df.index)


def map_players_from_team_on_offense(nba_df):
    """Map O1-O5 / D1-D5 from TeamOnOffense."""
    offense_col = 'poss_offense' if 'poss_offense' in nba_df.columns else 'TeamOnOffense'
    o_mapping = {}
    d_mapping = {}
    for i in range(1, 6):
        o_mapping[f'O{i}'] = np.where(
            nba_df[offense_col] == "Away",
            nba_df[f'a{i}'],
            np.where(nba_df[offense_col] == "Home", nba_df[f'h{i}'], np.nan)
        )
        d_mapping[f'D{i}'] = np.where(
            nba_df[offense_col] == "Away",
            nba_df[f'h{i}'],
            np.where(nba_df[offense_col] == "Home", nba_df[f'a{i}'], np.nan)
        )
    return nba_df.assign(**o_mapping, **d_mapping)


def dedupe_player_slot_groups(df, slot_groups, fill_value=0):
    """Remove duplicate positive player ids within each ordered lineup slot group."""
    out = df.copy()
    for cols in slot_groups:
        present_cols = [col for col in cols if col in out.columns]
        if not present_cols:
            continue
        values = out[present_cols].apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy()
        deduped = np.zeros(values.shape, dtype=float)
        for row_idx, row in enumerate(values):
            seen = set()
            write_idx = 0
            for value in row:
                player_id = int(value) if pd.notna(value) else 0
                if player_id <= 0 or player_id in seen:
                    continue
                seen.add(player_id)
                if write_idx < len(present_cols):
                    deduped[row_idx, write_idx] = float(player_id)
                    write_idx += 1
        out[present_cols] = deduped if fill_value == 0 else np.where(deduped > 0, deduped, fill_value)
    return out


def _normalize_team_label(value):
    if pd.isna(value):
        return ""
    text = re.sub(r"[^a-z0-9]+", " ", str(value).lower())
    return re.sub(r"\s+", " ", text).strip()


def _team_labels_for_side(game_df, side):
    desc_col = "home_description" if side == "Home" else "visitor_description"
    labels = set()
    side_rows = game_df[game_df[desc_col].fillna("").astype(str).str.strip().ne("")]
    for col in ["player1_team_abbreviation", "player1_team_city", "player1_team_nickname"]:
        if col in side_rows.columns:
            labels.update(_normalize_team_label(value) for value in side_rows[col].dropna().unique())
    if "player1_team_abbreviation" in side_rows.columns:
        for abbrev in side_rows["player1_team_abbreviation"].dropna().astype(str).str.upper().unique():
            labels.update(_normalize_team_label(value) for value in TEAM_ABBREVIATION_LABELS.get(abbrev, []))
    return {label for label in labels if label}


def infer_neutral_turnover_offense(nba_df):
    """Infer offense for neutral-description team turnovers using game-local team labels."""
    inferred = pd.Series("", index=nba_df.index, dtype=object)
    if "neutral_description" not in nba_df.columns or "game_id" not in nba_df.columns:
        return inferred

    neutral_turnover = (
        nba_df["event_type"].eq("Turnover")
        & nba_df["neutral_description"].fillna("").astype(str).str.contains("Turnover", case=False, regex=False)
    )
    if not neutral_turnover.any():
        return inferred

    for _, game in nba_df.loc[neutral_turnover | nba_df["home_description"].notna() | nba_df["visitor_description"].notna()].groupby("game_id", sort=False):
        home_labels = _team_labels_for_side(game, "Home")
        away_labels = _team_labels_for_side(game, "Away")
        for idx, row in game[neutral_turnover.reindex(game.index, fill_value=False)].iterrows():
            desc = _normalize_team_label(row.get("neutral_description"))
            home_match = any(re.search(rf"\b{re.escape(label)}\b", desc) for label in home_labels)
            away_match = any(re.search(rf"\b{re.escape(label)}\b", desc) for label in away_labels)
            if home_match and not away_match:
                inferred.at[idx] = "Home"
            elif away_match and not home_match:
                inferred.at[idx] = "Away"
    return inferred


def fill_terminal_lineups_from_previous_row(nba_df):
    """Fill blank terminal possession lineup slots from the prior in-game row."""
    player_cols = [f"{team}{i}" for team in ["h", "a"] for i in range(1, 6)]
    present_cols = [col for col in player_cols if col in nba_df.columns]
    if not present_cols or "End_of_Possession" not in nba_df.columns:
        return nba_df

    known_home = (nba_df[[f"h{i}" for i in range(1, 6)]].apply(pd.to_numeric, errors="coerce").fillna(0) > 0).sum(axis=1)
    known_away = (nba_df[[f"a{i}" for i in range(1, 6)]].apply(pd.to_numeric, errors="coerce").fillna(0) > 0).sum(axis=1)
    needs_fill = nba_df["End_of_Possession"].fillna(False).astype(bool) & ((known_home < 4) | (known_away < 4))
    if not needs_fill.any():
        return nba_df

    fill_source = nba_df[present_cols].replace(0, np.nan)
    if "game_id" in nba_df.columns:
        fill_values = fill_source.groupby(nba_df["game_id"], sort=False).ffill()
    else:
        fill_values = fill_source.ffill()
    current = nba_df.loc[needs_fill, present_cols].replace(0, np.nan)
    nba_df.loc[needs_fill, present_cols] = current.combine_first(fill_values.loc[needs_fill, present_cols]).fillna(0)
    return nba_df


def fill_blank_terminal_offense_from_previous_terminal(nba_df, group_cols):
    """Infer one-row terminal possessions from the prior completed possession."""
    if 'poss_offense' not in nba_df.columns or 'End_of_Possession' not in nba_df.columns:
        return nba_df

    blank_terminal = (
        nba_df['End_of_Possession'].fillna(False).astype(bool)
        & nba_df['poss_offense'].eq("")
    )
    if not blank_terminal.any():
        return nba_df

    group_key = [col for col in [*group_cols, 'period'] if col in nba_df.columns]
    terminal_offense = nba_df['poss_offense'].where(
        nba_df['End_of_Possession'].fillna(False).astype(bool) & nba_df['poss_offense'].isin(["Home", "Away"])
    )
    if group_key:
        previous_terminal_offense = terminal_offense.groupby([nba_df[col] for col in group_key], sort=False).ffill()
    else:
        previous_terminal_offense = terminal_offense.ffill()

    inferred = previous_terminal_offense.map({"Home": "Away", "Away": "Home"}).fillna("")
    nba_df.loc[blank_terminal & inferred.isin(["Home", "Away"]), 'poss_offense'] = inferred
    return nba_df


def prepare_standard_possession_df(nba_df, label):
    """
    Apply the standard RAPM possession definition, foul handling, and O/D mapping.
    Used by RAPM-style possession metrics that share the normal possession denominator.
    """
    ft_exclude_pattern = r'\b(1 of [23]|2 of 3|Technical|Flagrant)\b'

    nba_df['offensive_FT_Rebound'] = case_when(
        series_contains(nba_df['visitor_description'], "REBOUND", case=False) &
        series_contains(nba_df['Prev_visitor_desc'], "MISS", case=False) &
        series_contains(nba_df['Prev_visitor_desc'], "Free Throw", case=False) &
        ~series_contains(nba_df['Prev_visitor_desc'], ft_exclude_pattern, case=False, regex=True), True,
        series_contains(nba_df['home_description'], "REBOUND", case=False) &
        series_contains(nba_df['Prev_home_desc'], "MISS", case=False) &
        series_contains(nba_df['Prev_home_desc'], "Free Throw", case=False) &
        ~series_contains(nba_df['Prev_home_desc'], ft_exclude_pattern, case=False, regex=True), True,
        False
    ).astype(bool)

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
    next_home_rebound = nba_df['Next_event'].eq("Rebound") & series_contains(
        nba_df['Next_home_desc'], "REBOUND|Rebound", case=False, regex=True
    )
    next_away_rebound = nba_df['Next_event'].eq("Rebound") & series_contains(
        nba_df['Next_visitor_desc'], "REBOUND|Rebound", case=False, regex=True
    )
    next_home_deadball_rebound = next_home_rebound & series_contains(
        nba_df['Next_home_desc'], "deadball", case=False
    )
    next_away_deadball_rebound = next_away_rebound & series_contains(
        nba_df['Next_visitor_desc'], "deadball", case=False
    )
    next_home_team_off_rebound = next_home_rebound & series_contains(
        nba_df['Next_home_desc'], "team offensive rebound", case=False
    )
    next_home_team_def_rebound = next_home_rebound & series_contains(
        nba_df['Next_home_desc'], "team defensive rebound", case=False
    )
    next_away_team_off_rebound = next_away_rebound & series_contains(
        nba_df['Next_visitor_desc'], "team offensive rebound", case=False
    )
    next_away_team_def_rebound = next_away_rebound & series_contains(
        nba_df['Next_visitor_desc'], "team defensive rebound", case=False
    )
    home_missed_last_ft = (
        (nba_df['event_type'] == "FreeThrow") &
        series_contains(nba_df['home_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False, regex=True) &
        series_contains(nba_df['home_description'], r'\bMISS\b', case=False, regex=True) &
        ~series_contains(nba_df['home_description'], "technical", case=False)
    )
    away_missed_last_ft = (
        (nba_df['event_type'] == "FreeThrow") &
        series_contains(nba_df['visitor_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False, regex=True) &
        series_contains(nba_df['visitor_description'], r'\bMISS\b', case=False, regex=True) &
        ~series_contains(nba_df['visitor_description'], "technical", case=False)
    )
    home_live_same_team_off_reb = next_home_rebound & ~next_home_deadball_rebound & ~next_home_team_def_rebound
    away_live_same_team_off_reb = next_away_rebound & ~next_away_deadball_rebound & ~next_away_team_def_rebound
    home_live_other_team_def_reb = next_away_rebound & ~next_away_deadball_rebound & ~next_away_team_off_rebound
    away_live_other_team_def_reb = next_home_rebound & ~next_home_deadball_rebound & ~next_home_team_off_rebound
    home_missed_last_ft_eop = home_missed_last_ft & ~(home_live_same_team_off_reb | home_live_other_team_def_reb)
    away_missed_last_ft_eop = away_missed_last_ft & ~(away_live_same_team_off_reb | away_live_other_team_def_reb)
    home_shot_attempt = (
        nba_df['event_type'].isin(["MAKE", "MISS"]) &
        (
            series_contains(nba_df['home_description'], "PTS", case=False) |
            series_contains(nba_df['home_description'], "MISS", case=False)
        )
    )
    away_shot_attempt = (
        nba_df['event_type'].isin(["MAKE", "MISS"]) &
        (
            series_contains(nba_df['visitor_description'], "PTS", case=False) |
            series_contains(nba_df['visitor_description'], "MISS", case=False)
        )
    )

    nba_df['End_of_Possession'] = case_when(
        (nba_df['event_type'] == "EndOfPeriod") & (nba_df['prev_seconds'] > 0), True,
        series_contains(nba_df['home_description'], r'Flagrant (2 of 2|3 of 3)', case=False, regex=True), True,
        series_contains(nba_df['visitor_description'], r'Flagrant (2 of 2|3 of 3)', case=False, regex=True), True,
        (nba_df['event_type'] == "Turnover"), True,
        series_contains(nba_df['home_description'], "REBOUND", case=False) &
        series_contains(nba_df['Prev_visitor_desc'], "MISS", case=False) &
        ~series_contains(nba_df['Prev_visitor_desc'], ft_exclude_pattern, case=False, regex=True), True,
        series_contains(nba_df['visitor_description'], "REBOUND", case=False) &
        series_contains(nba_df['Prev_home_desc'], "MISS", case=False) &
        ~series_contains(nba_df['Prev_home_desc'], ft_exclude_pattern, case=False, regex=True), True,
        series_contains(nba_df['home_description'], "PTS", case=False) &
        (nba_df['event_type'] == "MAKE") &
        ~series_contains(nba_df['Next_visitor_desc'], "S.FOUL", case=True), True,
        series_contains(nba_df['visitor_description'], "PTS", case=False) &
        (nba_df['event_type'] == "MAKE") &
        ~series_contains(nba_df['Next_home_desc'], "S.FOUL", case=True), True,
        series_contains(nba_df['home_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False, regex=True) &
        series_contains(nba_df['home_description'], "PTS", case=False) &
        ~(series_contains(nba_df['Prev_visitor_desc2'], "Transition", case=False) |
          series_contains(nba_df['Prev_visitor_desc'], "Transition", case=False) |
          series_contains(nba_df['home_description'], "Flagrant", case=False)), True,
        series_contains(nba_df['visitor_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False, regex=True) &
        series_contains(nba_df['visitor_description'], "PTS", case=False) &
        ~(series_contains(nba_df['Prev_home_desc2'], "Transition", case=False) |
          series_contains(nba_df['Prev_home_desc'], "Transition", case=False) |
          series_contains(nba_df['visitor_description'], "Flagrant", case=False)), True,
        home_offensive_foul_no_turnover, True,
        away_offensive_foul_no_turnover, True,
        home_missed_last_ft_eop, True,
        away_missed_last_ft_eop, True,
        False
    ).astype(bool)

    nba_df['TeamOnOffense'] = case_when(
        series_contains(nba_df['home_description'], r'Flagrant (2 of 2|3 of 3)', case=False, regex=True), "Home",
        series_contains(nba_df['visitor_description'], r'Flagrant (2 of 2|3 of 3)', case=False, regex=True), "Away",
        series_contains(nba_df['home_description'], "REBOUND", case=False) &
        series_contains(nba_df['Prev_visitor_desc'], "MISS", case=False), "Away",
        series_contains(nba_df['visitor_description'], "REBOUND", case=False) &
        series_contains(nba_df['Prev_home_desc'], "MISS", case=False), "Home",
        series_contains(nba_df['home_description'], "Turnover", case=False), "Home",
        series_contains(nba_df['visitor_description'], "Turnover", case=False), "Away",
        series_contains(nba_df['home_description'], "PTS", case=False) &
        (nba_df['event_type'] == "MAKE") &
        ~series_contains(nba_df['Next_visitor_desc'], "S.FOUL", case=True), "Home",
        series_contains(nba_df['visitor_description'], "PTS", case=False) &
        (nba_df['event_type'] == "MAKE") &
        ~series_contains(nba_df['Next_home_desc'], "S.FOUL", case=True), "Away",
        series_contains(nba_df['home_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False, regex=True) &
        series_contains(nba_df['home_description'], "PTS", case=False) &
        ~(series_contains(nba_df['Prev_visitor_desc2'], "Transition", case=False) |
          series_contains(nba_df['Prev_visitor_desc'], "Transition", case=False) |
          series_contains(nba_df['home_description'], "Flagrant", case=False)), "Home",
        series_contains(nba_df['visitor_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False, regex=True) &
        series_contains(nba_df['visitor_description'], "PTS", case=False) &
        ~(series_contains(nba_df['Prev_home_desc2'], "Transition", case=False) |
          series_contains(nba_df['Prev_home_desc'], "Transition", case=False) |
          series_contains(nba_df['visitor_description'], "Flagrant", case=False)), "Away",
        home_offensive_foul_no_turnover, "Home",
        away_offensive_foul_no_turnover, "Away",
        home_missed_last_ft_eop, "Home",
        away_missed_last_ft_eop, "Away",
        home_shot_attempt, "Home",
        away_shot_attempt, "Away",
        ""
    )
    home_ft_row = (nba_df['event_type'] == "FreeThrow") & series_contains(
        nba_df['home_description'], "Free Throw", case=False
    )
    away_ft_row = (nba_df['event_type'] == "FreeThrow") & series_contains(
        nba_df['visitor_description'], "Free Throw", case=False
    )
    nba_df.loc[nba_df['TeamOnOffense'].eq("") & home_ft_row, 'TeamOnOffense'] = "Home"
    nba_df.loc[nba_df['TeamOnOffense'].eq("") & away_ft_row, 'TeamOnOffense'] = "Away"
    neutral_offense = infer_neutral_turnover_offense(nba_df)
    nba_df.loc[nba_df['TeamOnOffense'].eq("") & neutral_offense.isin(["Home", "Away"]), 'TeamOnOffense'] = neutral_offense
    print(f"      Calculated EOP, TeamOnOffense for {label}")

    window_size = 5
    group_cols = ['game_id'] if 'game_id' in nba_df.columns else []
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
    nba_df_checked = fill_terminal_lineups_from_previous_row(nba_df_checked)
    nba_df_checked = dedupe_player_slot_groups(
        nba_df_checked,
        [[f"a{i}" for i in range(1, 6)], [f"h{i}" for i in range(1, 6)]],
    )
    if group_cols:
        game_groupers = [nba_df_checked[col] for col in group_cols]
        shifted_eop = nba_df_checked.groupby(group_cols)['End_of_Possession'].shift(1, fill_value=False)
        nba_df_checked['poss_group'] = shifted_eop.groupby(game_groupers, sort=False).cumsum()
    else:
        nba_df_checked['poss_group'] = nba_df_checked['End_of_Possession'].shift(1, fill_value=False).cumsum()

    poss_group_key = group_cols + ['poss_group'] if group_cols else ['poss_group']
    poss_groupers = [nba_df_checked[col] for col in poss_group_key]
    offense_source = nba_df_checked['TeamOnOffense'].mask(nba_df_checked['TeamOnOffense'].eq(''), np.nan)
    nba_df_checked['poss_offense'] = (
        offense_source
        .groupby(poss_groupers, sort=False)
        .bfill()
        .groupby(poss_groupers, sort=False)
        .ffill()
        .fillna("")
    )
    nba_df_checked.loc[nba_df_checked['poss_offense'].eq("") & home_ft_row.reindex(nba_df_checked.index, fill_value=False), 'poss_offense'] = "Home"
    nba_df_checked.loc[nba_df_checked['poss_offense'].eq("") & away_ft_row.reindex(nba_df_checked.index, fill_value=False), 'poss_offense'] = "Away"
    nba_df_checked = fill_blank_terminal_offense_from_previous_terminal(nba_df_checked, group_cols)
    nba_df_checked = map_players_from_team_on_offense(nba_df_checked)
    nba_df_checked = dedupe_player_slot_groups(
        nba_df_checked,
        [[f"O{i}" for i in range(1, 6)], [f"D{i}" for i in range(1, 6)]],
    )
    print("      Mapped O/D Players")
    return nba_df_checked, group_cols


# --- Player Propagation Functions ---

def propagate_player_values_py(df):
    """
    Propagates player values (a1-h5) after a PotentialFoul if FTs occur
    before the next EOP or offensive FT rebound.
    """
    print("      Running Player Propagation...")
    start_time = time.time()
    df_copy = df.copy()

    player_cols = [f"{team}{i}" for team in ["a", "h"] for i in range(1, 6)]

    if 'PotentialFoul' not in df_copy.columns or not pd.api.types.is_bool_dtype(df_copy['PotentialFoul']):
        print("      Warning: 'PotentialFoul' column missing or not boolean. Skipping propagation.")
        return df

    foul_indices = df_copy.index[df_copy['PotentialFoul']].tolist()
    if not foul_indices:
        print("      No PotentialFoul rows found. Skipping propagation.")
        return df

    num_propagations = 0

    for i in foul_indices:
        player_values_at_foul = df_copy.loc[i, player_cols].to_dict()
        found_free_throw = False
        start_loc = df_copy.index.get_loc(i) + 1
        end_loc = len(df_copy)

        for current_loc in range(start_loc, end_loc):
            current_index = df_copy.index[current_loc]
            row_data = df_copy.loc[current_index]

            if row_data.get('event_type', '') == "FreeThrow":
                found_free_throw = True

            is_eop = row_data.get('End_of_Possession', False)
            is_off_ft_reb = row_data.get('offensive_FT_Rebound', False)

            if is_eop or is_off_ft_reb:
                if found_free_throw:
                    end_prop_loc = current_loc if is_eop else current_loc - 1
                    if end_prop_loc >= start_loc:
                        prop_indices = df_copy.index[start_loc : end_prop_loc + 1]
                        if not prop_indices.empty:
                            df_copy.loc[prop_indices, player_cols] = pd.DataFrame(
                                [player_values_at_foul] * len(prop_indices),
                                index=prop_indices
                            )
                            num_propagations += len(prop_indices)
                break

    end_time = time.time()
    print(f"      Finished Player Propagation. Propagated values for {num_propagations} rows. Time: {end_time - start_time:.2f}s")
    return df_copy


def ft_off_check_py(df):
    """
    Checks for offensive FT rebounds after PotentialFouls where player lineups change.
    If found, marks the preceding row as End_of_Possession.
    """
    print("      Running FT Off Check...")
    start_time = time.time()
    df_copy = df.copy()

    player_cols = [f"{team}{i}" for team in ["a", "h"] for i in range(1, 6)]

    if 'PotentialFoul' not in df_copy.columns or not pd.api.types.is_bool_dtype(df_copy['PotentialFoul']):
        print("      Warning: 'PotentialFoul' column missing or not boolean. Skipping FT Off Check.")
        return df

    foul_indices = df_copy.index[df_copy['PotentialFoul']].tolist()
    if not foul_indices:
        print("      No PotentialFoul rows found. Skipping FT Off Check.")
        return df

    eop_corrections = 0

    for i in foul_indices:
        values_at_foul = df_copy.loc[i, player_cols]
        start_loc = df_copy.index.get_loc(i) + 1
        end_loc = len(df_copy)

        for current_loc in range(start_loc, end_loc):
            current_index = df_copy.index[current_loc]
            row_data = df_copy.loc[current_index]

            is_off_ft_reb = row_data.get('offensive_FT_Rebound', False)
            is_eop = row_data.get('End_of_Possession', False)

            if is_off_ft_reb:
                values_at_reb = row_data[player_cols]
                if not values_at_foul.astype(str).equals(values_at_reb.astype(str)):
                    if current_loc > 0:
                        prev_index = df_copy.index[current_loc - 1]
                        if not df_copy.loc[prev_index, 'End_of_Possession']:
                            df_copy.loc[prev_index, 'End_of_Possession'] = True
                            eop_corrections += 1
                    break

            if is_eop:
                break

    end_time = time.time()
    print(f"      Finished FT Off Check. Made {eop_corrections} EOP corrections. Time: {end_time - start_time:.2f}s")
    return df_copy


# --- External Data Loading ---

_game_dates_cache = None


def fetch_player_stats_supabase(year, is_playoffs):
    """
    Fetches player stats (3P_PERC, FT_PERC) from Supabase table 'player_stats'.
    `year` is the season end year (e.g. 26 or 2026 for the 2025-26 season).
    """
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")
    key_name = "SUPABASE_SERVICE_ROLE_KEY" if os.environ.get("SUPABASE_SERVICE_ROLE_KEY") else "SUPABASE_KEY"

    if not url or not key:
        print("      Warning: SUPABASE_URL and Supabase key missing in .env. Falling back to defaults.")
        return load_bref_ft_pct_lookup(year, is_playoffs)

    try:
        supabase: Client = create_client(url, key)
        playoff_flag = 1 if is_playoffs else 0
        query_year = normalize_season_end_year(year)
        print(f"      Querying Supabase for player stats with {key_name} (Year: {query_year}, Playoffs: {playoff_flag})...")

        response = supabase.table("player_stats_with_metrics") \
            .select("nba_id, year, playoffs, \"3P_PERC\", \"FT_PERC\"") \
            .eq("year", query_year) \
            .eq("playoffs", playoff_flag) \
            .execute()

        if response.data:
            df = pd.DataFrame(response.data)
            df = df.rename(columns={
                '3P_PERC': 'ThreePerc',
                'FT_PERC': 'FTPerc',
                'nba_id': 'PlayerID'
            })

            initial_count = len(df)
            df = df.drop_duplicates(subset=['PlayerID'], keep='first')
            if len(df) < initial_count:
                print(f"      Deduplicated player stats: {initial_count} -> {len(df)} unique players.")

            print(f"      Successfully fetched {len(df)} player stat entries from Supabase.")
            return df
        else:
            print("      Warning: Supabase returned no data for player stats.")
            return load_bref_ft_pct_lookup(year, is_playoffs)

    except Exception as e:
        print(f"      Error fetching player stats from Supabase: {e}")
        return load_bref_ft_pct_lookup(year, is_playoffs)


def normalize_historical_season_end_year(year):
    year = int(year)
    if year < 100:
        return 1900 + year if year >= 97 else 2000 + year
    return year


def load_bref_ft_pct_lookup(year, is_playoffs, path=BREF_FT_PCT_LOOKUP):
    """Load local BRef RS FT% fallback rows for 1997-2000 RS/PS processing."""
    query_year = normalize_historical_season_end_year(year)
    if query_year < 1997 or query_year > 2000:
        return None
    if not path.exists():
        print(f"      BRef FT% fallback not found: {path}")
        return None

    try:
        df = pd.read_csv(path, usecols=["nba_id", "year", "FTPerc"])
    except Exception as exc:
        print(f"      Error loading BRef FT% fallback: {exc}")
        return None

    df = df[df["year"].eq(query_year)].copy()
    if df.empty:
        return None

    df = df.rename(columns={"nba_id": "PlayerID"})
    df["PlayerID"] = pd.to_numeric(df["PlayerID"], errors="coerce")
    df["FTPerc"] = pd.to_numeric(df["FTPerc"], errors="coerce")
    df["ThreePerc"] = np.nan
    df = df.dropna(subset=["PlayerID"]).drop_duplicates("PlayerID", keep="first")
    phase = "PS" if is_playoffs else "RS"
    print(f"      Loaded {len(df)} BRef RS FT% fallback rows for {query_year} {phase}.")
    return df[["PlayerID", "FTPerc", "ThreePerc"]]


def load_game_dates(url="https://raw.githubusercontent.com/gabriel1200/shot_data/refs/heads/master/game_dates.csv"):
    """
    Loads and prepares the game dates data from the specified URL.
    Handles caching, type conversion, and deduplication.
    """
    global _game_dates_cache
    if _game_dates_cache is not None:
        return _game_dates_cache.copy()

    print(f"        Loading game dates data from {url}...")
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        csv_content = io.StringIO(response.text)
        game_dates_df = pd.read_csv(csv_content)
        game_dates_df['date'] = pd.to_datetime(
            game_dates_df['date'].astype(str),
            format='%Y%m%d',
            errors='coerce'
        ).dt.strftime('%Y-%m-%d')

        # Keep the schedule season in canonical label form, e.g. "2025-26".
        # Finalization filters with build_season_label_from_end_year(), so
        # coercing this to the start year integer would make every merge miss.
        game_dates_df['season'] = game_dates_df['season'].astype(str)
        print(f"        Successfully loaded {len(game_dates_df)} rows from game dates CSV.")

        game_dates_df['GAME_ID'] = pd.to_numeric(game_dates_df['GAME_ID'], errors='coerce')
        game_dates_df = game_dates_df.dropna(subset=['GAME_ID'])
        game_dates_df['GAME_ID'] = game_dates_df['GAME_ID'].astype(int)

        game_dates_df = game_dates_df.dropna(subset=['date'])

        game_dates_df = game_dates_df.dropna(subset=['season'])

        game_dates_df = game_dates_df[['GAME_ID', 'date', 'season']].drop_duplicates(
            subset=['GAME_ID'], keep='first'
        )
        game_dates_df = game_dates_df.rename(columns={'date': 'game_date'})

        print(f"        Processed game dates data: {len(game_dates_df)} unique games found.")
        _game_dates_cache = game_dates_df
        return game_dates_df.copy()

    except requests.exceptions.RequestException as e:
        print(f"        Error loading game dates CSV from URL: {e}")
        return None
    except Exception as e:
        print(f"        Error processing game dates CSV: {e}")
        return None


# --- Base Processing Function ---

def _base_processing(file_path):
    """Handles common preprocessing steps for all RAPM types."""
    print(f"    Running Base Processing for {file_path}...")
    try:
        nba_df = pd.read_parquet(file_path)
        print(f"      Read {len(nba_df)} rows from {file_path}")
    except FileNotFoundError:
        print(f"      Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"      Error reading {file_path}: {e}")
        return None

    if 'game_id' in nba_df.columns:
        nba_df['game_id'] = normalize_game_id_series(nba_df['game_id'])
        print("      Normalized 'game_id' to 10-character strings.")

    allow_partial_lineups = (
        'allow_partial_lineups' in nba_df.columns
        and nba_df['allow_partial_lineups'].fillna(False).astype(bool).any()
    )

    nba_df = dedupe_exact_raw_pbp_rows(nba_df, file_path)

    sort_cols = [col for col in ['game_id', 'period', 'event_num'] if col in nba_df.columns]
    if sort_cols:
        nba_df = nba_df.sort_values(sort_cols, kind='stable').reset_index(drop=True)
        print(f"      Sorted raw rows by {sort_cols}.")

    # Ensure seconds_remaining_quarter is numeric
    if 'seconds_remaining_quarter' in nba_df.columns:
        nba_df['seconds_remaining_quarter'] = pd.to_numeric(
            nba_df['seconds_remaining_quarter'], errors='coerce'
        )
        print("      Ensured 'seconds_remaining_quarter' is numeric.")

    # Initial Cleaning
    player_cols = [f"{team}_player{i}" for team in ["away", "home"] for i in range(1, 6)]
    missing_p_cols = [p_col for p_col in player_cols if p_col not in nba_df.columns]
    if missing_p_cols:
        print(f"      Error: Missing required player columns: {missing_p_cols}")
        return None

    if allow_partial_lineups:
        nba_df[player_cols] = nba_df[player_cols].replace('', np.nan).fillna(0)
        print("      Partial-lineup mode enabled; preserving known player slots and filling missing slots with 0.")
    elif 'game_id' in nba_df.columns:
        missing_player_rows = nba_df[player_cols].isna().any(axis=1)
        if missing_player_rows.any():
            nba_df[player_cols] = nba_df.groupby('game_id')[player_cols].ffill().bfill()
            recovered_rows = (
                missing_player_rows
                & ~nba_df[player_cols].isna().any(axis=1)
            ).sum()
            if recovered_rows:
                print(f"      Recovered {int(recovered_rows)} rows with missing lineup slots via in-game fill.")

    if not allow_partial_lineups:
        nba_df = nba_df.dropna(subset=player_cols)
        nba_df = nba_df[~(nba_df[player_cols].astype(str).apply(lambda x: x.str.strip() == '')).any(axis=1)]
    nba_df = dedupe_player_slot_groups(
        nba_df,
        [[f"away_player{i}" for i in range(1, 6)], [f"home_player{i}" for i in range(1, 6)]],
    )
    print(f"      Rows after player filtering: {len(nba_df)}")
    if len(nba_df) == 0:
        print("      Error: No rows remaining after player filtering.")
        return None

    # Initialize and Forward fill score columns
    score_cols_to_fill = ['score', 'away_score', 'home_score']
    print(f"      Attempting to initialize and forward fill for columns: {score_cols_to_fill}")

    if 'game_id' in nba_df.columns:
        if 'away_score' in nba_df.columns:
            nba_df['away_score'] = pd.to_numeric(nba_df['away_score'], errors='coerce')
        if 'home_score' in nba_df.columns:
            nba_df['home_score'] = pd.to_numeric(nba_df['home_score'], errors='coerce')

        print("        Initializing scores to 0 for first null rows per game_id...")
        is_first_row_in_group = nba_df.groupby('game_id').cumcount() == 0

        if 'away_score' in nba_df.columns:
            nba_df.loc[is_first_row_in_group & nba_df['away_score'].isna(), 'away_score'] = 0
        if 'home_score' in nba_df.columns:
            nba_df.loc[is_first_row_in_group & nba_df['home_score'].isna(), 'home_score'] = 0
        if 'score' in nba_df.columns:
            nba_df['score'] = nba_df['score'].astype(object)
            nba_df.loc[is_first_row_in_group & nba_df['score'].isna(), 'score'] = "0 - 0"

        existing_score_cols = [col for col in score_cols_to_fill if col in nba_df.columns]
        if existing_score_cols:
            nba_df[existing_score_cols] = nba_df.groupby('game_id')[existing_score_cols].ffill()
            print(f"      Applied forward fill for: {existing_score_cols}")

        # Recalculate score_margin
        if 'score_margin' in nba_df.columns and 'away_score' in nba_df.columns and 'home_score' in nba_df.columns:
            nba_df['score_margin'] = nba_df['score_margin'].astype(object)
            mask_can_calculate_margin = nba_df['home_score'].notna() & nba_df['away_score'].notna()
            numeric_margin = nba_df.loc[mask_can_calculate_margin, 'home_score'] - nba_df.loc[mask_can_calculate_margin, 'away_score']

            temp_margin_values = pd.Series(index=numeric_margin.index, dtype=object)
            temp_margin_values[numeric_margin == 0] = "TIE"
            temp_margin_values[numeric_margin != 0] = numeric_margin[numeric_margin != 0].astype(int).astype(str)

            nba_df.loc[mask_can_calculate_margin, 'score_margin'] = temp_margin_values
            nba_df['score_margin'] = nba_df.groupby('game_id')['score_margin'].ffill()
            print("      Recalculated and/or forward-filled 'score_margin' based on filled scores.")

    # Map event_type
    event_type_map = {
        1: "MAKE", 2: "MISS", 3: "FreeThrow", 4: "Rebound", 5: "Turnover",
        6: "Foul", 7: "Violation", 8: "Substitution", 9: "Timeout",
        10: "JumpBall", 11: "Ejection", 12: "StartOfPeriod", 13: "EndOfPeriod",
        14: "Empty"
    }
    nba_df['event_type_num'] = pd.to_numeric(nba_df['event_type'], errors='coerce')
    nba_df['event_type'] = nba_df['event_type_num'].map(event_type_map).fillna(nba_df['event_type_num'].astype(str))
    print("      Mapped event types")

    # Fill NA in string columns
    for col in nba_df.select_dtypes(include=['object', 'string']).columns:
        if col in nba_df.columns:
            nba_df[col] = nba_df[col].fillna('')
    print("      Filled NA values in string columns")

    # Calculate Base Scores
    nba_df['home_description'] = nba_df['home_description'].astype(str)
    nba_df['visitor_description'] = nba_df['visitor_description'].astype(str)
    nba_df = nba_df.assign(
        Home_Action_Score=case_when(
            series_contains(nba_df['home_description'], "Free Throw", case=False) & ~series_contains(nba_df['home_description'], r'\bMISS\b', case=False, regex=True), 1,
            series_contains(nba_df['home_description'], "PTS", case=False) & ~series_contains(nba_df['home_description'], "3PT", case=False) & (nba_df['event_type'] == "MAKE"), 2,
            series_contains(nba_df['home_description'], "3PT", case=False) & ~series_contains(nba_df['home_description'], r'\bMISS\b', case=False, regex=True), 3,
            0
        ).astype(int),
        Away_Action_Score=case_when(
            series_contains(nba_df['visitor_description'], "Free Throw", case=False) & ~series_contains(nba_df['visitor_description'], r'\bMISS\b', case=False, regex=True), 1,
            series_contains(nba_df['visitor_description'], "PTS", case=False) & ~series_contains(nba_df['visitor_description'], "3PT", case=False) & (nba_df['event_type'] == "MAKE"), 2,
            series_contains(nba_df['visitor_description'], "3PT", case=False) & ~series_contains(nba_df['visitor_description'], r'\bMISS\b', case=False, regex=True), 3,
            0
        ).astype(int)
    )

    group_cols = ['game_id'] if 'game_id' in nba_df.columns else []
    if group_cols:
        if not pd.api.types.is_integer_dtype(nba_df['game_id']) and not pd.api.types.is_string_dtype(nba_df['game_id']):
            try:
                nba_df['game_id_temp'] = pd.to_numeric(nba_df['game_id'])
                if nba_df['game_id_temp'].notna().all() and (nba_df['game_id_temp'] == nba_df['game_id_temp'].astype(int)).all():
                    nba_df['game_id'] = nba_df['game_id_temp'].astype(int)
                else:
                    nba_df['game_id'] = nba_df['game_id'].astype(str)
                nba_df.drop(columns=['game_id_temp'], inplace=True, errors='ignore')
            except:
                nba_df['game_id'] = nba_df['game_id'].astype(str)

        nba_df['Net_Home'] = nba_df.groupby(group_cols)['Home_Action_Score'].cumsum()
        nba_df['Net_Away'] = nba_df.groupby(group_cols)['Away_Action_Score'].cumsum()
    else:
        nba_df['Net_Home'] = nba_df['Home_Action_Score'].cumsum()
        nba_df['Net_Away'] = nba_df['Away_Action_Score'].cumsum()
    nba_df['Net_Total'] = nba_df['Net_Home'] + nba_df['Net_Away']
    print("      Calculated action scores and cumulative totals")

    # Create Lag/Lead Columns
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
                if '2' in new_col:
                    shift_num = 2
                period *= shift_num

                if pd.api.types.is_numeric_dtype(nba_df[col]):
                    fill_val_shift = np.nan
                else:
                    fill_val_shift = ""

                if period != 0:
                    if group_cols:
                        nba_df[new_col] = nba_df.groupby(group_cols)[col].shift(period)
                        if pd.api.types.is_numeric_dtype(nba_df[new_col]):
                            nba_df[new_col] = nba_df[new_col].fillna(fill_val_shift)
                        else:
                            nba_df[new_col] = nba_df[new_col].fillna(fill_val_shift)
                    else:
                        nba_df[new_col] = nba_df[col].shift(period, fill_value=fill_val_shift)
    print("      Created lag/lead columns")

    # Alias Player Columns
    alias_map = {f"{team}_player{i}": f"{prefix}{i}" for team, prefix in [("away", "a"), ("home", "h")] for i in range(1, 6)}
    nba_df = nba_df.rename(columns=alias_map)
    nba_df = dedupe_player_slot_groups(
        nba_df,
        [[f"a{i}" for i in range(1, 6)], [f"h{i}" for i in range(1, 6)]],
    )
    print("      Aliased player columns (a1-h5)")

    print(f"    Finished Base Processing for {file_path}.")
    return nba_df


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
        df = dedupe_player_slot_groups(df, [o_cols, d_cols])

    # Load and Merge Game Dates
    game_dates_df = load_game_dates()

    if game_dates_df is not None and 'game_id' in df.columns:
        season_label = build_season_label_from_end_year(year)
        filtered_dates_df = game_dates_df[game_dates_df['season'] == season_label]

        if not filtered_dates_df.empty:
            print(f"        Merging with {len(filtered_dates_df)} game date entries for season {season_label}...")

            if not pd.api.types.is_integer_dtype(df['game_id']):
                print(f"        Converting main df game_id from {df['game_id'].dtype} to numeric/int...")
                df['game_id'] = pd.to_numeric(df['game_id'], errors='coerce')
                if df['game_id'].isna().any():
                    print(f"        Warning: Found {df['game_id'].isna().sum()} NaN game_ids after conversion.")
                    df['game_id'] = df['game_id'].fillna(-1)
                df['game_id'] = df['game_id'].astype(int)

            original_rows = len(df)
            df = pd.merge(
                df,
                filtered_dates_df[['GAME_ID', 'game_date']],
                left_on='game_id',
                right_on='GAME_ID',
                how='left'
            )
            if 'game_date_x' in df.columns or 'game_date_y' in df.columns:
                existing_dates = df['game_date_x'] if 'game_date_x' in df.columns else pd.Series(pd.NA, index=df.index)
                schedule_dates = df['game_date_y'] if 'game_date_y' in df.columns else pd.Series(pd.NA, index=df.index)
                df['game_date'] = existing_dates.combine_first(schedule_dates)
                df = df.drop(columns=[col for col in ['game_date_x', 'game_date_y'] if col in df.columns])
            if 'GAME_ID' in df.columns:
                df = df.drop(columns=['GAME_ID'])

            rows_after_merge = len(df)
            if rows_after_merge != original_rows:
                print(f"        Warning: Row count changed during game_date merge ({original_rows} -> {rows_after_merge}).")
            merged_dates_count = df['game_date'].notna().sum()
            print(f"        Successfully merged game dates for {merged_dates_count} rows.")
        else:
            print(f"        Warning: No game dates found for season {season_label} in the loaded schedule data.")
            if 'game_date' not in df.columns:
                df['game_date'] = pd.NA
    elif 'game_date' not in df.columns:
        print("       Warning: Game dates CSV failed to load or game_id missing. Adding empty game_date column.")
        df['game_date'] = pd.NA

    df['Season'] = normalize_season_end_year(year)

    # Select Final Columns
    final_cols = id_cols + o_cols + d_cols

    # Add game state columns
    game_state_cols = ['score', 'period', 'time_quarter', 'away_score', 'home_score', 'score_margin', 'event_num']
    final_cols.extend(game_state_cols)

    # Add numerator column(s)
    if isinstance(numerator_col, list):
        final_cols.extend(numerator_col)
    else:
        final_cols.append(numerator_col)

    if 'game_date' in df.columns:
        final_cols.append('game_date')
    final_cols.append('Season')

    final_cols_present = [col for col in final_cols if col in df.columns]
    missing_final_cols = [col for col in final_cols if col not in final_cols_present]
    if missing_final_cols:
        print(f"      Warning: Missing expected final columns: {missing_final_cols}.")

    df_final = df[final_cols_present].copy()

    # Final Filtering (D5 != 0 or NaN/blank/Null)
    if allow_partial_lineups:
        initial_rows = len(df_final)
        present_o_cols = [col for col in o_cols if col in df_final.columns]
        present_d_cols = [col for col in d_cols if col in df_final.columns]
        df_final = dedupe_player_slot_groups(df_final, [present_o_cols, present_d_cols])
        off_known = (df_final[present_o_cols].apply(pd.to_numeric, errors='coerce').fillna(0) > 0).sum(axis=1)
        def_known = (df_final[present_d_cols].apply(pd.to_numeric, errors='coerce').fillna(0) > 0).sum(axis=1)
        df_final = df_final[(off_known >= 4) & (def_known >= 4)].copy()
        print(
            "      Partial-lineup mode enabled; kept rows with at least 4 known O and D players "
            f"({len(df_final)} rows, removed {initial_rows - len(df_final)})."
        )
    elif 'D5' in df_final.columns:
        initial_rows = len(df_final)
        df_final = df_final[~df_final['D5'].isin([0, '0', '', None, np.nan])]
        df_final = df_final.dropna(subset=['D5'])
        print(f"      Rows after final D5 != 0/Null/Empty filter: {len(df_final)} (removed {initial_rows - len(df_final)})")
    else:
        print("      Warning: D5 column not found for final filtering.")

    print("      Finished Finalization.")
    return df_final
