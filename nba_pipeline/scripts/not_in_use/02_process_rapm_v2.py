"""
RAPM Possession Processor v2 - State Machine Architecture

Fixes from v1/backup:
1. Score-lineup desync: Lock lineup at action-initiating event (foul/shot)
2. Possession undercounting: Proper terminal event detection
3. Regex bugs: All patterns case-insensitive
4. Flagrant 1-of-1: Now terminates possession (was only 2/3-of-2/3)
5. Offensive FT rebound: Continues possession (was terminating)
6. Luck adjustment: Correct dimensional math (FT% * 1pt, 3P% * 3pt)
"""

from __future__ import annotations

import argparse
import io
import os
import re
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Paths ---
SCRIPT_DIR = Path(__file__).parent
PIPELINE_ROOT = SCRIPT_DIR.parent
RAW_DATA_DIR = PIPELINE_ROOT / "raw_data"
PROCESSED_DIR = PIPELINE_ROOT / "processed"

# --- Constants ---
EVENT_TYPE_MAP = {
    1: "MAKE",
    2: "MISS",
    3: "FreeThrow",
    4: "Rebound",
    5: "Turnover",
    6: "Foul",
    7: "Violation",
    8: "Substitution",
    9: "Timeout",
    10: "JumpBall",
    11: "Ejection",
    12: "StartOfPeriod",
    13: "EndOfPeriod",
    14: "Empty",
}

PLAYER_COLS_RAW = [f"{side}_player{i}" for side in ("away", "home") for i in range(1, 6)]
PLAYER_COLS_ALIASES = {f"away_player{i}": f"a{i}" for i in range(1, 6)}
PLAYER_COLS_ALIASES.update({f"home_player{i}": f"h{i}" for i in range(1, 6)})

# Default shooting percentages for luck adjustment
DEFAULT_FT_PCT = 0.75
DEFAULT_3PT_PCT = 0.35

# Known player IDs for validation logging
YVES_MISSI_ID = 1642273  # Update if needed


# --- Compiled Regex Patterns (All Case-Insensitive) ---
class Patterns:
    """Centralized regex patterns for event classification."""

    # Foul patterns
    SHOOTING_FOUL = re.compile(r'\bS\.FOUL\b', re.IGNORECASE)
    TECHNICAL_FOUL = re.compile(r'\bT\.FOUL\b', re.IGNORECASE)
    FLAGRANT_FOUL = re.compile(r'\bFLAGRANT\b', re.IGNORECASE)
    OFFENSIVE_FOUL = re.compile(r'\bOFFENS', re.IGNORECASE)
    TRANSITION_FOUL = re.compile(r'\bTRANSITION\b', re.IGNORECASE)

    # FT sequence patterns
    FT_1_OF_1 = re.compile(r'\b1 OF 1\b', re.IGNORECASE)
    FT_1_OF_2 = re.compile(r'\b1 OF 2\b', re.IGNORECASE)
    FT_2_OF_2 = re.compile(r'\b2 OF 2\b', re.IGNORECASE)
    FT_1_OF_3 = re.compile(r'\b1 OF 3\b', re.IGNORECASE)
    FT_2_OF_3 = re.compile(r'\b2 OF 3\b', re.IGNORECASE)
    FT_3_OF_3 = re.compile(r'\b3 OF 3\b', re.IGNORECASE)
    FT_FINAL = re.compile(r'\b(1 OF 1|2 OF 2|3 OF 3)\b', re.IGNORECASE)
    FT_MID_SEQ = re.compile(r'\b(1 OF [23]|2 OF 3)\b', re.IGNORECASE)

    # Scoring patterns
    PTS = re.compile(r'\(?\d+\s*PTS?\)?', re.IGNORECASE)
    THREE_PT = re.compile(r'\b3PT\b', re.IGNORECASE)
    FREE_THROW = re.compile(r'\bFREE THROW\b', re.IGNORECASE)

    # Other patterns
    REBOUND = re.compile(r'\bREBOUND\b', re.IGNORECASE)
    MISS = re.compile(r'\bMISS\b', re.IGNORECASE)
    TURNOVER = re.compile(r'\bTURNOVER\b', re.IGNORECASE)
    GOALTENDING = re.compile(r'\bGOALTEND', re.IGNORECASE)
    TEAM_REBOUND = re.compile(r'\bTEAM REBOUND\b', re.IGNORECASE)


# --- Data Classes ---
@dataclass
class FTSequence:
    """Groups FTs from same foul event with locked lineup."""
    foul_idx: int
    foul_type: str  # "shooting", "flagrant", "technical", "transition"
    shooter_id: Optional[int]
    ft_indices: List[int] = field(default_factory=list)
    total_fts: int = 0
    made_fts: int = 0
    is_and_one: bool = False
    locked_offense: List[int] = field(default_factory=list)
    locked_defense: List[int] = field(default_factory=list)


@dataclass
class Possession:
    """Represents a completed possession."""
    game_id: int
    poss_id: int
    offense_team: str
    defense_team: str
    off_players: List[int]
    def_players: List[int]
    off_points: float
    la_off_points: float
    la_def_points: float
    end_row: pd.Series
    terminal_reason: str


@dataclass
class ScoringEvent:
    """Individual scoring event within a possession."""
    idx: int
    shot_type: str  # "FT", "2PT", "3PT"
    points: float
    shooter_id: Optional[int]


# --- Helper Functions ---

def series_contains(series: pd.Series, pattern: re.Pattern) -> pd.Series:
    """Apply compiled regex pattern to series."""
    if not pd.api.types.is_string_dtype(series):
        series = series.astype(str)
    return series.fillna("").str.contains(pattern, case=False, regex=True)


def get_combined_desc(row: pd.Series) -> str:
    """Get combined description from both columns."""
    home = str(row.get("home_description", "") or "")
    away = str(row.get("visitor_description", "") or "")
    return f"{home} {away}".upper()


def is_shooting_foul(row: pd.Series) -> bool:
    """Check if event is a shooting foul (case-insensitive)."""
    desc = get_combined_desc(row)
    return bool(Patterns.SHOOTING_FOUL.search(desc))


def is_technical_foul(row: pd.Series) -> bool:
    """Check if event is a technical foul."""
    desc = get_combined_desc(row)
    return bool(Patterns.TECHNICAL_FOUL.search(desc))


def is_flagrant_foul(row: pd.Series) -> bool:
    """Check if event is a flagrant foul."""
    desc = get_combined_desc(row)
    return bool(Patterns.FLAGRANT_FOUL.search(desc))


def is_offensive_foul(row: pd.Series) -> bool:
    """Check if event is an offensive foul."""
    desc = get_combined_desc(row)
    return bool(Patterns.OFFENSIVE_FOUL.search(desc))


def is_final_ft(row: pd.Series) -> bool:
    """Check if this is a final FT in a sequence (1-of-1, 2-of-2, 3-of-3)."""
    desc = get_combined_desc(row)
    return bool(Patterns.FT_FINAL.search(desc))


def is_mid_sequence_ft(row: pd.Series) -> bool:
    """Check if this is a mid-sequence FT (1-of-2, 1-of-3, 2-of-3)."""
    desc = get_combined_desc(row)
    return bool(Patterns.FT_MID_SEQ.search(desc))


def classify_shot_type(row: pd.Series, is_ft: bool = False) -> str:
    """Classify shot as FT, 2PT, or 3PT."""
    if is_ft:
        return "FT"
    desc = get_combined_desc(row)
    if Patterns.THREE_PT.search(desc):
        return "3PT"
    return "2PT"


def get_rebound_team(row: pd.Series) -> Optional[str]:
    """Determine which team got the rebound."""
    home_desc = str(row.get("home_description", "") or "").upper()
    away_desc = str(row.get("visitor_description", "") or "").upper()

    if Patterns.REBOUND.search(home_desc):
        return "Home"
    elif Patterns.REBOUND.search(away_desc):
        return "Away"
    return None


def get_miss_team(row: pd.Series) -> Optional[str]:
    """Determine which team missed."""
    home_desc = str(row.get("home_description", "") or "").upper()
    away_desc = str(row.get("visitor_description", "") or "").upper()

    if Patterns.MISS.search(home_desc):
        return "Home"
    elif Patterns.MISS.search(away_desc):
        return "Away"
    return None


def is_ft_event(row: pd.Series) -> bool:
    """Check if event is a free throw."""
    if row.get("event_type") == "FreeThrow":
        return True
    desc = get_combined_desc(row)
    return bool(Patterns.FREE_THROW.search(desc))


def load_game_dates(
    url: str = "https://raw.githubusercontent.com/gabriel1200/shot_data/refs/heads/master/game_dates.csv"
) -> Optional[pd.DataFrame]:
    """Load game dates from external source."""
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df["date"] = pd.to_datetime(
            df["date"].astype(str), format="%Y%m%d", errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        df["season"] = pd.to_numeric(
            df["season"].str.split("-").str[0], errors="coerce"
        ).astype("Int64")
        df["GAME_ID"] = pd.to_numeric(df["GAME_ID"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["date", "season", "GAME_ID"])
        return df[["GAME_ID", "date", "season"]].rename(columns={"date": "game_date"})
    except Exception as e:
        logger.warning(f"Failed to load game dates: {e}")
        return None


def fetch_player_stats() -> Optional[pd.DataFrame]:
    """Fetch player shooting stats from Supabase."""
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")

        if not supabase_url or not supabase_key:
            logger.warning("Supabase credentials not found, using default shooting percentages")
            return None

        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}"
        }

        response = requests.get(
            f"{supabase_url}/rest/v1/player_stats_with_metrics?select=PlayerID,FTPerc,ThreePerc",
            headers=headers,
            timeout=30
        )
        response.raise_for_status()

        df = pd.DataFrame(response.json())
        df["PlayerID"] = pd.to_numeric(df["PlayerID"], errors="coerce").astype("Int64")
        df["FTPerc"] = pd.to_numeric(df["FTPerc"], errors="coerce")
        df["ThreePerc"] = pd.to_numeric(df["ThreePerc"], errors="coerce")

        return df.dropna(subset=["PlayerID"])

    except Exception as e:
        logger.warning(f"Failed to fetch player stats: {e}")
        return None


# --- Data Preparation ---

def prepare_pbp(file_path: Path) -> pd.DataFrame:
    """Load and prepare PBP data for processing."""
    logger.info(f"Loading {file_path}")
    df = pd.read_csv(file_path, low_memory=False, na_values=["NA"])

    if "game_id" not in df.columns:
        raise ValueError("game_id column missing")

    # Basic ordering
    sort_cols = [c for c in ["game_id", "period", "event_num"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # Map event types
    df["event_type_num"] = pd.to_numeric(df["event_type"], errors="coerce")
    df["event_type"] = df["event_type_num"].map(EVENT_TYPE_MAP).fillna(
        df["event_type_num"].astype(str)
    )

    # Forward/backward fill player columns
    missing_cols = [c for c in PLAYER_COLS_RAW if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing player columns: {missing_cols}")
    df[PLAYER_COLS_RAW] = df.groupby("game_id")[PLAYER_COLS_RAW].ffill().bfill()

    # Ensure numeric scores and set opening 0-0
    for col in ("home_score", "away_score"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    first_row_mask = df.groupby("game_id").cumcount() == 0
    df.loc[first_row_mask & df["home_score"].isna(), "home_score"] = 0
    df.loc[first_row_mask & df["away_score"].isna(), "away_score"] = 0
    df[["home_score", "away_score"]] = df.groupby("game_id")[
        ["home_score", "away_score"]
    ].ffill()

    # Recompute score margin
    df["score_margin"] = (df["home_score"] - df["away_score"]).round().astype("Int64")
    df.loc[df["score_margin"] == 0, "score_margin"] = pd.NA
    df["score"] = (
        df["away_score"].fillna(0).astype(int).astype(str)
        + " - "
        + df["home_score"].fillna(0).astype(int).astype(str)
    )

    # Alias players to a1-h5
    df = df.rename(columns=PLAYER_COLS_ALIASES)

    # Uppercase descriptions for pattern matching
    df["home_desc_u"] = df["home_description"].fillna("").str.upper()
    df["away_desc_u"] = df["visitor_description"].fillna("").str.upper()

    return df


# --- Score Attribution ---

def find_scoring_anchor(df: pd.DataFrame, idx: int, team: str, window: int = 6) -> int:
    """Move scoreboard bumps back to the real scoring action."""
    loc = df.index.get_loc(idx)
    start = max(0, loc - window)
    window_df = df.iloc[start : loc + 1]

    if team == "Home":
        cand = window_df[
            (window_df["home_scoring_candidate"])
            | (
                (window_df["event_team"] == "Home")
                & ~window_df["event_type"].isin(["Substitution", "Timeout", "JumpBall"])
            )
        ]
    else:
        cand = window_df[
            (window_df["away_scoring_candidate"])
            | (
                (window_df["event_team"] == "Away")
                & ~window_df["event_type"].isin(["Substitution", "Timeout", "JumpBall"])
            )
        ]

    if not cand.empty:
        return cand.index[-1]
    return idx


def attach_action_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate per-event scores using score diffs with backward anchoring."""
    df = df.copy()

    # Determine event team
    df["event_team"] = np.where(
        df["home_desc_u"] != "",
        "Home",
        np.where(df["away_desc_u"] != "", "Away", None),
    )

    # Identify scoring candidates
    df["home_scoring_candidate"] = (
        df["event_type"].isin(["MAKE", "FreeThrow"])
        | df["home_desc_u"].str.contains("PTS", case=False, na=False)
        | df["home_desc_u"].str.contains("FREE THROW", case=False, na=False)
    )
    df["away_scoring_candidate"] = (
        df["event_type"].isin(["MAKE", "FreeThrow"])
        | df["away_desc_u"].str.contains("PTS", case=False, na=False)
        | df["away_desc_u"].str.contains("FREE THROW", case=False, na=False)
    )

    # Initialize action scores
    df["Home_Action_Score"] = 0
    df["Away_Action_Score"] = 0

    # Calculate score deltas
    home_delta = df.groupby("game_id")["home_score"].diff().fillna(df["home_score"]).clip(lower=0)
    away_delta = df.groupby("game_id")["away_score"].diff().fillna(df["away_score"]).clip(lower=0)

    # Anchor scores back to actual scoring events
    for idx, pts in home_delta[home_delta > 0].items():
        anchor = find_scoring_anchor(df, idx, "Home")
        df.at[anchor, "Home_Action_Score"] += int(pts)

    for idx, pts in away_delta[away_delta > 0].items():
        anchor = find_scoring_anchor(df, idx, "Away")
        df.at[anchor, "Away_Action_Score"] += int(pts)

    # Track which team scored
    df["scoring_team"] = np.where(
        df["Home_Action_Score"] > 0,
        "Home",
        np.where(df["Away_Action_Score"] > 0, "Away", None),
    )

    return df


# --- FT Sequence Grouping ---

def group_ft_sequences(game_df: pd.DataFrame) -> Dict[int, FTSequence]:
    """
    Group FTs by their originating foul and lock lineups.

    This is critical for proper lineup attribution during FT sequences
    with substitutions.
    """
    sequences = {}

    # Find foul events
    foul_mask = game_df["event_type"] == "Foul"
    foul_indices = game_df[foul_mask].index.tolist()

    for foul_idx in foul_indices:
        foul_row = game_df.loc[foul_idx]
        foul_loc = game_df.index.get_loc(foul_idx)

        # Skip technical fouls (dead ball)
        if is_technical_foul(foul_row):
            continue

        # Skip offensive fouls (no FTs)
        if is_offensive_foul(foul_row):
            continue

        # Determine foul type
        if is_flagrant_foul(foul_row):
            foul_type = "flagrant"
        elif is_shooting_foul(foul_row):
            foul_type = "shooting"
        else:
            foul_type = "other"

        # Search forward for associated FTs (expanded window)
        ft_indices = []
        search_end = min(foul_loc + 15, len(game_df))

        for search_loc in range(foul_loc + 1, search_end):
            search_idx = game_df.index[search_loc]
            search_row = game_df.loc[search_idx]

            # Stop at end of period or another foul
            if search_row["event_type"] == "EndOfPeriod":
                break
            if search_row["event_type"] == "Foul":
                break

            # Collect FT events
            if search_row["event_type"] == "FreeThrow" or is_ft_event(search_row):
                ft_indices.append(search_idx)

                # Check for final FT
                if is_final_ft(search_row):
                    break

        if not ft_indices:
            continue

        # Check for And-1 (made basket before foul)
        is_and_one = False
        if foul_loc > 0:
            prev_row = game_df.iloc[foul_loc - 1]
            if prev_row["event_type"] == "MAKE" or prev_row.get("scoring_team"):
                if is_shooting_foul(foul_row):
                    is_and_one = True

        # Determine offense team from foul
        if foul_row["away_desc_u"] != "" and "FOUL" in foul_row["away_desc_u"]:
            # Away team fouled, so Home is offense
            offense_team = "Home"
        else:
            offense_team = "Away"

        # Lock lineup at foul row
        if offense_team == "Home":
            off_cols = [f"h{i}" for i in range(1, 6)]
            def_cols = [f"a{i}" for i in range(1, 6)]
        else:
            off_cols = [f"a{i}" for i in range(1, 6)]
            def_cols = [f"h{i}" for i in range(1, 6)]

        locked_offense = [int(foul_row[c]) if pd.notna(foul_row[c]) else 0 for c in off_cols]
        locked_defense = [int(foul_row[c]) if pd.notna(foul_row[c]) else 0 for c in def_cols]

        # Count made FTs
        made_fts = 0
        for ft_idx in ft_indices:
            ft_row = game_df.loc[ft_idx]
            desc = get_combined_desc(ft_row)
            if not Patterns.MISS.search(desc):
                made_fts += 1

        sequences[foul_idx] = FTSequence(
            foul_idx=foul_idx,
            foul_type=foul_type,
            shooter_id=foul_row.get("player1_id"),
            ft_indices=ft_indices,
            total_fts=len(ft_indices),
            made_fts=made_fts,
            is_and_one=is_and_one,
            locked_offense=locked_offense,
            locked_defense=locked_defense,
        )

    return sequences


# --- Luck Adjustment ---

def calculate_luck_adjusted_value(
    shot_type: str,
    actual_pts: float,
    ft_pct: float,
    three_pct: float,
    luck_weight: float = 1.0,
) -> float:
    """
    Calculate luck-adjusted expected value for a scoring event.

    CRITICAL FIX: Previous version mixed percentages with point totals.
    This version ensures dimensional correctness:
    - FT: expected = FT% * 1 point
    - 3PT: expected = 3P% * 3 points
    - 2PT: no adjustment (binary outcome)
    """
    if shot_type == "FT":
        expected_pts = ft_pct * 1.0  # FT worth 1 point
    elif shot_type == "3PT":
        expected_pts = three_pct * 3.0  # 3PT worth 3 points
    elif shot_type == "2PT":
        return actual_pts  # No adjustment for 2PT
    else:
        return actual_pts

    # Both terms now in same units (points)
    return luck_weight * expected_pts + (1 - luck_weight) * actual_pts


# --- Possession State Machine ---

def build_possessions(
    df: pd.DataFrame,
    player_stats: Optional[pd.DataFrame] = None,
    o_luck: float = 1.0,
    d_luck: float = 1.0,
) -> List[Possession]:
    """
    Build possessions using state machine with proper terminal detection.

    Key improvements:
    1. Lock lineups at action initiation (not score update)
    2. Handle flagrant 1-of-1 as possession ender
    3. Continue possession on offensive FT rebound
    4. Apply dimensionally correct luck adjustment
    """
    possessions: List[Possession] = []

    # Build player stats lookup
    player_ft_pct = {}
    player_3pt_pct = {}
    if player_stats is not None:
        for _, row in player_stats.iterrows():
            pid = row["PlayerID"]
            player_ft_pct[pid] = row.get("FTPerc", DEFAULT_FT_PCT) or DEFAULT_FT_PCT
            player_3pt_pct[pid] = row.get("ThreePerc", DEFAULT_3PT_PCT) or DEFAULT_3PT_PCT

    for game_id, gdf in df.groupby("game_id"):
        # Pre-compute FT sequences for this game
        ft_sequences = group_ft_sequences(gdf)
        ft_idx_to_sequence = {}
        for seq in ft_sequences.values():
            for ft_idx in seq.ft_indices:
                ft_idx_to_sequence[ft_idx] = seq

        offense: Optional[str] = None
        defense: Optional[str] = None
        poss_id = 0
        current_events: List[int] = []
        current_scoring_events: List[ScoringEvent] = []
        locked_off_lineup: Optional[List[int]] = None
        locked_def_lineup: Optional[List[int]] = None
        in_ft_sequence = False
        current_ft_sequence: Optional[FTSequence] = None

        gdf_list = list(gdf.iterrows())

        for i, (idx, row) in enumerate(gdf_list):
            event_team = row.get("event_team")
            scoring_team = row.get("scoring_team")
            rebound_team = get_rebound_team(row)
            miss_team = get_miss_team(row)
            is_turnover = (
                row["event_type"] == "Turnover"
                or Patterns.TURNOVER.search(row["home_desc_u"])
                or Patterns.TURNOVER.search(row["away_desc_u"])
            )
            is_violation = row["event_type"] == "Violation"
            is_ft = is_ft_event(row)
            is_final = is_final_ft(row)

            # Get next row for lookahead
            next_row = gdf_list[i + 1][1] if i + 1 < len(gdf_list) else None

            # Check for And-1 incoming
            is_and_one_incoming = False
            if scoring_team and not is_ft and next_row is not None:
                if is_shooting_foul(next_row):
                    is_and_one_incoming = True

            # Bootstrap offense if unknown
            if offense is None:
                if scoring_team:
                    offense = scoring_team
                elif is_turnover and event_team:
                    offense = event_team
                elif miss_team:
                    offense = miss_team
                elif rebound_team:
                    # Defensive rebound means rebounder gets new possession
                    offense = rebound_team

            if offense:
                defense = "Home" if offense == "Away" else "Away"

            # Track events in current possession
            if offense is not None:
                if not current_events:
                    current_events.append(idx)
                    # Lock lineup at possession start
                    if offense == "Home":
                        off_cols = [f"h{i}" for i in range(1, 6)]
                        def_cols = [f"a{i}" for i in range(1, 6)]
                    else:
                        off_cols = [f"a{i}" for i in range(1, 6)]
                        def_cols = [f"h{i}" for i in range(1, 6)]
                    locked_off_lineup = [
                        int(row[c]) if pd.notna(row[c]) else 0 for c in off_cols
                    ]
                    locked_def_lineup = [
                        int(row[c]) if pd.notna(row[c]) else 0 for c in def_cols
                    ]
                else:
                    current_events.append(idx)

            # Track scoring events
            if scoring_team and scoring_team == offense:
                pts = (
                    row["Home_Action_Score"]
                    if offense == "Home"
                    else row["Away_Action_Score"]
                )
                if pts > 0:
                    shot_type = classify_shot_type(row, is_ft=is_ft)
                    shooter_id = row.get("player1_id")
                    current_scoring_events.append(
                        ScoringEvent(
                            idx=idx,
                            shot_type=shot_type,
                            points=pts,
                            shooter_id=shooter_id,
                        )
                    )

            # Check if entering FT sequence (for lineup locking)
            if idx in ft_idx_to_sequence:
                seq = ft_idx_to_sequence[idx]
                if seq.ft_indices[0] == idx:  # First FT of sequence
                    in_ft_sequence = True
                    current_ft_sequence = seq
                    # Override locked lineup with foul-locked lineup
                    locked_off_lineup = seq.locked_offense
                    locked_def_lineup = seq.locked_defense

            # --- Terminal Event Detection ---
            possession_end = False
            next_offense: Optional[str] = offense
            terminal_reason = ""

            # 1. End of Period
            if row["event_type"] == "EndOfPeriod":
                possession_end = True
                next_offense = None
                terminal_reason = "end_of_period"

            # 2. Turnover or violation
            elif is_turnover or is_violation:
                possession_end = True
                if offense:
                    next_offense = "Home" if offense == "Away" else "Away"
                terminal_reason = "turnover" if is_turnover else "violation"

            # 3. Defensive rebound
            elif rebound_team and offense and rebound_team != offense:
                # Check if this is an offensive FT rebound (continue possession)
                if is_ft and current_ft_sequence:
                    # If rebound is by offense, continue
                    pass
                else:
                    possession_end = True
                    next_offense = rebound_team
                    terminal_reason = "defensive_rebound"

            # 4. Made field goal (unless And-1)
            elif scoring_team and not is_ft:
                if is_and_one_incoming:
                    possession_end = False  # Continue for And-1
                else:
                    possession_end = True
                    next_offense = "Home" if scoring_team == "Away" else "Away"
                    terminal_reason = "made_fg"

            # 5. Final FT (1-of-1, 2-of-2, 3-of-3)
            elif is_ft and is_final:
                # Check for offensive FT rebound possibility
                is_off_ft_reb = False
                if next_row is not None:
                    next_reb_team = get_rebound_team(next_row)
                    if next_reb_team == offense:
                        is_off_ft_reb = True

                if is_off_ft_reb:
                    possession_end = False  # Continue on offensive FT rebound
                else:
                    possession_end = True
                    if current_ft_sequence and current_ft_sequence.foul_type == "flagrant":
                        # Flagrant: Fouled team keeps possession
                        next_offense = offense
                        terminal_reason = "flagrant_ft"
                    else:
                        next_offense = "Home" if offense == "Away" else "Away"
                        terminal_reason = "final_ft"

                in_ft_sequence = False
                current_ft_sequence = None

            # 6. Team rebound (silent possession change)
            elif Patterns.TEAM_REBOUND.search(get_combined_desc(row)):
                # Team rebounds can indicate possession change
                if rebound_team and rebound_team != offense:
                    possession_end = True
                    next_offense = rebound_team
                    terminal_reason = "team_rebound"

            # --- Finalize Possession ---
            if possession_end and offense and current_events:
                # Calculate points
                off_points = sum(se.points for se in current_scoring_events)

                # Calculate luck-adjusted points
                la_off_points = 0.0
                la_def_points = 0.0
                for se in current_scoring_events:
                    ft_pct = player_ft_pct.get(se.shooter_id, DEFAULT_FT_PCT)
                    three_pct = player_3pt_pct.get(se.shooter_id, DEFAULT_3PT_PCT)

                    la_off_points += calculate_luck_adjusted_value(
                        se.shot_type, se.points, ft_pct, three_pct, o_luck
                    )
                    la_def_points += calculate_luck_adjusted_value(
                        se.shot_type, se.points, ft_pct, three_pct, d_luck
                    )

                # Use locked lineup (from possession start or foul event)
                off_players = locked_off_lineup or [0] * 5
                def_players = locked_def_lineup or [0] * 5

                possessions.append(
                    Possession(
                        game_id=int(game_id),
                        poss_id=poss_id,
                        offense_team=offense,
                        defense_team=defense,
                        off_players=off_players,
                        def_players=def_players,
                        off_points=off_points,
                        la_off_points=la_off_points,
                        la_def_points=la_def_points,
                        end_row=row,
                        terminal_reason=terminal_reason,
                    )
                )
                poss_id += 1

                # Reset state
                current_events = []
                current_scoring_events = []
                locked_off_lineup = None
                locked_def_lineup = None
                offense = next_offense
                defense = "Home" if offense == "Away" else "Away" if offense else None
                in_ft_sequence = False
                current_ft_sequence = None
            else:
                offense = next_offense

    return possessions


# --- Output Assembly ---

def possessions_to_frame(
    possessions: List[Possession],
    year: int,
    include_la: bool = True,
) -> pd.DataFrame:
    """Convert possessions to DataFrame for RAPM analysis."""
    rows: List[Dict[str, Any]] = []

    for poss in possessions:
        end = poss.end_row
        row = {
            "game_id": poss.game_id,
            "O1": poss.off_players[0],
            "O2": poss.off_players[1],
            "O3": poss.off_players[2],
            "O4": poss.off_players[3],
            "O5": poss.off_players[4],
            "D1": poss.def_players[0],
            "D2": poss.def_players[1],
            "D3": poss.def_players[2],
            "D4": poss.def_players[3],
            "D5": poss.def_players[4],
            "score": end.get("score", ""),
            "period": end.get("period", ""),
            "time_quarter": end.get("time_quarter", ""),
            "away_score": end.get("away_score", ""),
            "home_score": end.get("home_score", ""),
            "score_margin": end.get("score_margin", ""),
            "event_num": end.get("event_num", ""),
            "Net_Diff": poss.off_points,
            "Off_Diff": poss.off_points,
            "Def_Diff": poss.off_points,
            "Season": year + 1,
            "terminal_reason": poss.terminal_reason,
        }

        if include_la:
            row["LA_Off_Diff"] = poss.la_off_points
            row["LA_Def_Diff"] = poss.la_def_points

        rows.append(row)

    df = pd.DataFrame(rows)

    # Merge game dates
    game_dates = load_game_dates()
    if game_dates is not None and not df.empty:
        df = df.merge(
            game_dates[game_dates["season"] == year][["GAME_ID", "game_date"]],
            how="left",
            left_on="game_id",
            right_on="GAME_ID",
        ).drop(columns=["GAME_ID"], errors="ignore")

    return df


# --- Validation ---

def validate_output(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate processed RAPM output."""
    issues = []

    # Required columns
    required = (
        ["game_id"]
        + [f"O{i}" for i in range(1, 6)]
        + [f"D{i}" for i in range(1, 6)]
        + ["Off_Diff", "Def_Diff", "Net_Diff", "Season", "period"]
    )

    missing = [c for c in required if c not in df.columns]
    if missing:
        issues.append(f"ERROR: Missing required columns: {missing}")
        return False, issues

    # D5 must not be null
    null_d5 = df["D5"].isnull().sum() + (df["D5"] == 0).sum()
    if null_d5 > 0:
        issues.append(f"WARNING: {null_d5} rows with null/zero D5")

    # Check possession counts
    if len(df) < 1000:
        issues.append(f"WARNING: Low possession count ({len(df)})")

    # Check score diffs
    max_diff = df["Net_Diff"].abs().max()
    if max_diff > 10:
        issues.append(f"WARNING: Max Net_Diff is {max_diff}")

    # Per-game check
    poss_per_game = df.groupby("game_id").size()
    low_games = poss_per_game[poss_per_game < 100]
    if len(low_games) > 0:
        issues.append(f"WARNING: {len(low_games)} games with <100 possessions")

    is_valid = not any(issue.startswith("ERROR") for issue in issues)
    return is_valid, issues


# --- Comparison Logging ---

def log_comparison(new_df: pd.DataFrame, backup_path: Optional[Path], season: str) -> None:
    """Compare new processor output to backup script output."""
    print(f"\n{'='*60}")
    print(f"RAPM PROCESSOR V2 COMPARISON: Season {season}")
    print(f"{'='*60}")

    print(f"\nNew Processor Results:")
    print(f"  Total possessions: {len(new_df):,}")
    print(f"  Total games: {new_df['game_id'].nunique():,}")
    print(f"  Mean poss/game: {len(new_df) / new_df['game_id'].nunique():.1f}")

    # Total points
    total_pts = new_df["Off_Diff"].sum()
    print(f"  Total points: {total_pts:,.0f}")

    # ORtg estimate
    ortg = (total_pts / len(new_df)) * 100
    print(f"  Estimated ORtg: {ortg:.1f} (target: ~114.8)")

    # Terminal reason breakdown
    print(f"\nTerminal Reason Breakdown:")
    for reason, count in new_df["terminal_reason"].value_counts().items():
        print(f"  {reason}: {count:,} ({count/len(new_df)*100:.1f}%)")

    # Yves Missi check
    player_cols = ["O1", "O2", "O3", "O4", "O5"]
    missi_mask = new_df[player_cols].apply(
        lambda x: x.astype(str).str.contains(str(YVES_MISSI_ID), na=False)
    ).any(axis=1)
    missi_df = new_df[missi_mask]

    if len(missi_df) > 0:
        print(f"\nYves Missi (ID: {YVES_MISSI_ID}):")
        print(f"  Offensive possessions: {len(missi_df)}")
        print(f"  Total points: {missi_df['Off_Diff'].sum():.0f}")
        print(f"  PPS: {missi_df['Off_Diff'].sum() / len(missi_df):.3f}")

    # Compare to backup if available
    if backup_path and backup_path.exists():
        try:
            backup_df = pd.read_csv(backup_path)
            print(f"\nComparison to Backup ({backup_path.name}):")
            print(f"  Backup possessions: {len(backup_df):,}")
            print(f"  Difference: {len(new_df) - len(backup_df):+,} ({(len(new_df)/len(backup_df)-1)*100:+.2f}%)")

            backup_pts = backup_df["Off_Diff"].sum()
            backup_ortg = (backup_pts / len(backup_df)) * 100
            print(f"  Backup ORtg: {backup_ortg:.1f}")
            print(f"  ORtg improvement: {ortg - backup_ortg:+.2f}")
        except Exception as e:
            print(f"\nCould not load backup file: {e}")


# --- CLI ---

def process_file(
    file_path: Path,
    year: int,
    season_str: str,
    player_stats: Optional[pd.DataFrame] = None,
    o_luck: float = 1.0,
    d_luck: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process a single season file and return both regular and LA versions."""
    start = time.time()
    logger.info(f"Processing {file_path} for {season_str}")

    pbp = prepare_pbp(file_path)
    pbp = attach_action_scores(pbp)

    # Build possessions with luck adjustment
    possessions = build_possessions(pbp, player_stats, o_luck, d_luck)

    # Create output frames
    df_regular = possessions_to_frame(possessions, year, include_la=False)
    df_la = possessions_to_frame(possessions, year, include_la=True)

    # Validate
    is_valid, issues = validate_output(df_regular)
    for issue in issues:
        if issue.startswith("ERROR"):
            logger.error(issue)
        else:
            logger.warning(issue)

    elapsed = time.time() - start
    logger.info(f"Processed {len(df_regular):,} possessions in {elapsed:.1f}s")

    return df_regular, df_la


def infer_year_and_season(filename: str) -> Tuple[int, str]:
    """Infer season from filename."""
    match = re.search(r"NBA(\d{2,4})", filename, flags=re.IGNORECASE)
    if not match:
        raise ValueError("Could not infer season from filename (expected NBAXX.csv)")
    year_str = match.group(1)
    if len(year_str) == 2:
        ending_year = 2000 + int(year_str)
    else:
        ending_year = int(year_str)
    starting_year = ending_year - 1
    return starting_year, f"{starting_year}-{ending_year % 100:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAPM possession processor v2 (state machine with fixes)"
    )
    parser.add_argument("input_file", help="Path to raw NBA pbp CSV (e.g., NBA26.csv)")
    parser.add_argument("--year", type=int, help="Starting season year (e.g., 2025 for 2025-26)")
    parser.add_argument("--season", help="Season label (e.g., 2025-26)")
    parser.add_argument("--o-luck", type=float, default=1.0, help="Offensive luck weight (0-1)")
    parser.add_argument("--d-luck", type=float, default=1.0, help="Defensive luck weight (0-1)")
    parser.add_argument("--compare", action="store_true", help="Compare to backup output")
    args = parser.parse_args()

    file_path = Path(args.input_file)
    if not file_path.exists():
        raise SystemExit(f"Input file {file_path} not found")

    year = args.year
    season_str = args.season
    if year is None or season_str is None:
        inferred_year, inferred_season = infer_year_and_season(file_path.name)
        year = year or inferred_year
        season_str = season_str or inferred_season
        logger.info(f"Inferred season: {season_str}")

    # Fetch player stats for luck adjustment
    player_stats = fetch_player_stats()
    if player_stats is not None:
        logger.info(f"Loaded shooting stats for {len(player_stats)} players")

    # Process
    df_regular, df_la = process_file(
        file_path, year, season_str, player_stats, args.o_luck, args.d_luck
    )

    if df_regular.empty:
        logger.error("No possessions produced; exiting without write")
        return

    # Save outputs
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    season_suffix = season_str.split("-")[1]

    regular_path = PROCESSED_DIR / f"RAPM{season_suffix}.csv"
    la_path = PROCESSED_DIR / f"LA_RAPM{season_suffix}.csv"

    df_regular.to_csv(regular_path, index=False)
    df_la.to_csv(la_path, index=False)

    logger.info(f"Wrote {len(df_regular):,} possessions to {regular_path}")
    logger.info(f"Wrote {len(df_la):,} LA possessions to {la_path}")

    # Comparison logging
    if args.compare:
        backup_path = PROCESSED_DIR / f"RAPM{season_suffix}_backup.csv"
        log_comparison(df_regular, backup_path, season_str)
    else:
        log_comparison(df_regular, None, season_str)


if __name__ == "__main__":
    main()
