"""
Experimental RAPM processor with possession state machine.

Goals:
- Trust the official scoreboard for point magnitude.
- Re-anchor scores to the real basketball action (not the row where NBA.com applies the diff).
- Freeze lineups at possession start so substitutions between the foul and the score do not steal credit.
- Produce one row per possession with offense/defense players and the points scored.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

SCRIPT_DIR = Path(__file__).parent
PIPELINE_ROOT = SCRIPT_DIR.parent
RAW_DATA_DIR = PIPELINE_ROOT / "raw_data"
PROCESSED_DIR = PIPELINE_ROOT / "processed"


# --- Helpers -----------------------------------------------------------------

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


def series_contains(series: pd.Series, pattern: str, *, regex: bool = True) -> pd.Series:
    """Case-insensitive contains that copes with nulls."""
    if not pd.api.types.is_string_dtype(series):
        series = series.astype(str)
    return series.fillna("").str.contains(pattern, case=False, regex=regex)


def load_game_dates(url: str = "https://raw.githubusercontent.com/gabriel1200/shot_data/refs/heads/master/game_dates.csv") -> Optional[pd.DataFrame]:
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
        df["season"] = pd.to_numeric(df["season"].str.split("-").str[0], errors="coerce").astype("Int64")
        df["GAME_ID"] = pd.to_numeric(df["GAME_ID"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["date", "season", "GAME_ID"])
        return df[["GAME_ID", "date", "season"]].rename(columns={"date": "game_date"})
    except Exception:
        return None


# --- Base prep ----------------------------------------------------------------


def prepare_pbp(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, low_memory=False, na_values=["NA"])
    if "game_id" not in df.columns:
        raise ValueError("game_id column missing")

    # Basic ordering: game, period, event_num
    sort_cols = [c for c in ["game_id", "period", "event_num"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # Map event types
    df["event_type_num"] = pd.to_numeric(df["event_type"], errors="coerce")
    df["event_type"] = df["event_type_num"].map(EVENT_TYPE_MAP).fillna(df["event_type_num"].astype(str))

    # Forward/backward fill player columns instead of dropping rows (keeps fouls/timeouts)
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
    df[["home_score", "away_score"]] = df.groupby("game_id")[["home_score", "away_score"]].ffill()

    # Recompute score_margin/score string for consistency
    df["score_margin"] = (df["home_score"] - df["away_score"]).round().astype("Int64")
    df.loc[df["score_margin"] == 0, "score_margin"] = pd.NA
    df["score"] = (df["away_score"].fillna(0).astype(int).astype(str) + " - " + df["home_score"].fillna(0).astype(int).astype(str))

    # Alias players to a1-h5 for easier mapping
    df = df.rename(columns=PLAYER_COLS_ALIASES)

    # Upper-cased descriptions for fast substring checks
    df["home_desc_u"] = df["home_description"].fillna("").str.upper()
    df["away_desc_u"] = df["visitor_description"].fillna("").str.upper()
    return df


# --- Scoring attribution ------------------------------------------------------


def find_scoring_anchor(df: pd.DataFrame, idx: int, team: str, window: int = 6) -> int:
    """Move scoreboard bumps back to the real scoring action."""
    loc = df.index.get_loc(idx)
    start = max(0, loc - window)
    window_df = df.iloc[start : loc + 1]

    if team == "Home":
        cand = window_df[
            (window_df["home_scoring_candidate"])
            | (window_df["event_team"] == "Home")
            & ~window_df["event_type"].isin(["Substitution", "Timeout", "JumpBall"])
        ]
    else:
        cand = window_df[
            (window_df["away_scoring_candidate"])
            | (window_df["event_team"] == "Away")
            & ~window_df["event_type"].isin(["Substitution", "Timeout", "JumpBall"])
        ]

    if not cand.empty:
        return cand.index[-1]
    return idx


def attach_action_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["event_team"] = np.where(df["home_desc_u"] != "", "Home", np.where(df["away_desc_u"] != "", "Away", None))
    df["home_scoring_candidate"] = (
        df["event_type"].isin(["MAKE", "FreeThrow"])
        | series_contains(df["home_description"], "PTS")
        | series_contains(df["home_description"], "FREE THROW")
    )
    df["away_scoring_candidate"] = (
        df["event_type"].isin(["MAKE", "FreeThrow"])
        | series_contains(df["visitor_description"], "PTS")
        | series_contains(df["visitor_description"], "FREE THROW")
    )

    df["Home_Action_Score"] = 0
    df["Away_Action_Score"] = 0

    home_delta = df.groupby("game_id")["home_score"].diff().fillna(df["home_score"]).clip(lower=0)
    away_delta = df.groupby("game_id")["away_score"].diff().fillna(df["away_score"]).clip(lower=0)

    for idx, pts in home_delta[home_delta > 0].items():
        anchor = find_scoring_anchor(df, idx, "Home")
        df.at[anchor, "Home_Action_Score"] += int(pts)

    for idx, pts in away_delta[away_delta > 0].items():
        anchor = find_scoring_anchor(df, idx, "Away")
        df.at[anchor, "Away_Action_Score"] += int(pts)

    df["scoring_team"] = np.where(
        df["Home_Action_Score"] > 0, "Home", np.where(df["Away_Action_Score"] > 0, "Away", None)
    )
    return df


# --- Possession reconstruction ------------------------------------------------

@dataclass
class Possession:
    game_id: int
    poss_id: int
    offense_team: str
    defense_team: str
    off_players: List
    def_players: List
    off_points: float
    end_row: pd.Series


def detect_final_ft(desc_upper: str) -> bool:
    return bool(re.search(r"\b(1 OF 1|2 OF 2|3 OF 3)\b", desc_upper))


def build_possessions(df: pd.DataFrame) -> List[Possession]:
    possessions: List[Possession] = []
    for game_id, gdf in df.groupby("game_id"):
        offense: Optional[str] = None
        defense: Optional[str] = None
        poss_id = 0
        current_events: List[int] = []
        last_shot_team: Optional[str] = None

        for idx, row in gdf.iterrows():
            event_team = row["event_team"]
            scoring_team = row["scoring_team"]
            rebound_team = "Home" if "REBOUND" in row["home_desc_u"] else "Away" if "REBOUND" in row["away_desc_u"] else None
            miss_team = "Home" if "MISS" in row["home_desc_u"] else "Away" if "MISS" in row["away_desc_u"] else None
            is_turnover = (row["event_type"] == "Turnover") or ("TURNOVER" in row["home_desc_u"]) or ("TURNOVER" in row["away_desc_u"])
            is_violation = row["event_type"] == "Violation"
            is_ft = ("FREE THROW" in row["home_desc_u"]) or ("FREE THROW" in row["away_desc_u"])
            is_final_ft = is_ft and (detect_final_ft(row["home_desc_u"]) or detect_final_ft(row["away_desc_u"]))
            is_and_one = False
            if scoring_team and not is_ft:
                next_idx = gdf.index.get_loc(idx) + 1
                if next_idx < len(gdf):
                    next_row = gdf.iloc[next_idx]
                    if ("S.FOUL" in next_row["home_desc_u"]) or ("S.FOUL" in next_row["away_desc_u"]):
                        is_and_one = True

            # Bootstrap offense if unknown
            if offense is None:
                if scoring_team:
                    offense = scoring_team
                elif is_turnover and event_team:
                    offense = event_team
                elif miss_team:
                    offense = miss_team
                elif rebound_team:
                    offense = rebound_team
            if offense:
                defense = "Home" if offense == "Away" else "Away"

            if offense is not None and not current_events:
                current_events.append(idx)
            elif offense is not None:
                current_events.append(idx)

            possession_end = False
            next_offense: Optional[str] = offense

            if row["event_type"] == "EndOfPeriod":
                possession_end = True
                next_offense = None
            elif is_turnover or is_violation:
                possession_end = True
                if offense:
                    next_offense = "Home" if offense == "Away" else "Away"
            elif rebound_team and offense and rebound_team != offense:
                possession_end = True
                next_offense = rebound_team
            elif scoring_team:
                if is_ft and not is_final_ft:
                    possession_end = False
                elif is_and_one:
                    possession_end = False
                else:
                    possession_end = True
                    next_offense = "Home" if scoring_team == "Away" else "Away"

            # Update last shot marker for rebound logic
            if scoring_team or miss_team:
                last_shot_team = scoring_team or miss_team

            if possession_end and offense:
                poss_slice = gdf.loc[current_events]
                off_points = (
                    poss_slice["Home_Action_Score"].sum() if offense == "Home" else poss_slice["Away_Action_Score"].sum()
                )
                start_row = poss_slice.iloc[0]
                end_row = row
                off_cols = [f"{'a' if offense == 'Away' else 'h'}{i}" for i in range(1, 6)]
                def_cols = [f"{'h' if offense == 'Away' else 'a'}{i}" for i in range(1, 6)]

                possessions.append(
                    Possession(
                        game_id=int(game_id),
                        poss_id=poss_id,
                        offense_team=offense,
                        defense_team="Home" if offense == "Away" else "Away",
                        off_players=[start_row[c] for c in off_cols],
                        def_players=[start_row[c] for c in def_cols],
                        off_points=float(off_points),
                        end_row=end_row,
                    )
                )
                poss_id += 1
                current_events = []
                offense = next_offense
                defense = "Home" if offense == "Away" else "Away" if offense else None
                last_shot_team = None
            else:
                offense = next_offense

    return possessions


# --- Output assembly ----------------------------------------------------------


def possessions_to_frame(possessions: List[Possession], year: int) -> pd.DataFrame:
    rows: List[Dict] = []
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
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    game_dates = load_game_dates()
    if game_dates is not None and not df.empty:
        df = df.merge(
            game_dates[game_dates["season"] == year][["GAME_ID", "game_date"]],
            how="left",
            left_on="game_id",
            right_on="GAME_ID",
        ).drop(columns=["GAME_ID"])
    return df


# --- CLI ----------------------------------------------------------------------


def process_file(file_path: Path, year: int, season_str: str) -> pd.DataFrame:
    start = time.time()
    print(f"Processing {file_path} for {season_str} with state-machine possessions...")
    pbp = prepare_pbp(file_path)
    pbp = attach_action_scores(pbp)
    possessions = build_possessions(pbp)
    out_df = possessions_to_frame(possessions, year)
    print(f"Finished in {time.time() - start:.1f}s | Possessions: {len(out_df)}")
    return out_df


def infer_year_and_season(filename: str) -> Tuple[int, str]:
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
    parser = argparse.ArgumentParser(description="RAPM possession processor (state machine edition)")
    parser.add_argument("input_file", help="Path to raw NBA pbp CSV (e.g., NBA26.csv)")
    parser.add_argument("--year", type=int, help="Starting season year (e.g., 2025 for 2025-26)")
    parser.add_argument("--season", help="Season label (e.g., 2025-26)")
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
        print(f"Inferred season: {season_str}")

    df = process_file(file_path, year, season_str)
    if df.empty:
        print("No possessions produced; exiting without write.")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    season_suffix = season_str.split("-")[1]
    out_path = PROCESSED_DIR / f"RAPM_state_{season_suffix}.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} possession rows to {out_path}")


if __name__ == "__main__":
    main()
