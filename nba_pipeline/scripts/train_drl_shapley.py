#!/usr/bin/env python3
"""
Paper-faithful DRL/Shapley rebuild for NBA play-by-play.

This script implements a practical end-to-end version of the Sloan paper stack:
- event-level transition dataset from raw play-by-play
- paper-style 37-feature numeric state + 10-player lineup encoder
- distributional TD value model over 81 margin bins
- Monte Carlo Shapley target generation with position-conditioned replacements
- neural attributor with exact efficiency correction
- hybrid action allocation and 2026-only reporting outputs

The implementation keeps the existing XGBoost MVP script untouched and treats
that file as the baseline/debug path.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import math
import os
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, mean_squared_error
from xgboost import XGBRegressor

try:
    from supabase import create_client
except Exception:  # pragma: no cover - optional import guard
    create_client = None


SCRIPT_DIR = Path(__file__).parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_ROOT.parent
RAW_DATA_DIR = PIPELINE_ROOT / "raw_data"
RESULTS_ROOT = PIPELINE_ROOT / "results" / "drl_shapley"
CACHE_ROOT = PIPELINE_ROOT / "processed" / "drl_shapley"
AUTOCOMPLETE_MAP = PROJECT_ROOT / "autocomplete_map.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger("drl_shapley")


ACTION_CLASSES = [
    "make_2",
    "make_3",
    "miss_2",
    "miss_3",
    "make_ft",
    "miss_ft",
    "off_rebound",
    "def_rebound",
    "turnover_bad_pass",
    "turnover_other",
    "steal",
    "block",
    "shooting_foul",
    "offensive_foul",
    "loose_ball_foul",
    "jump_ball",
    "violation",
    "timeout",
    "substitution",
    "end_period",
    "transition_make",
    "transition_miss",
    "other",
]
ACTION_TO_ID = {name: idx for idx, name in enumerate(ACTION_CLASSES)}

OFFENSIVE_ACTIONS = {
    "make_2",
    "make_3",
    "miss_2",
    "miss_3",
    "make_ft",
    "miss_ft",
    "off_rebound",
    "turnover_bad_pass",
    "turnover_other",
    "transition_make",
    "transition_miss",
}
DEFENSIVE_ACTIONS = {
    "def_rebound",
    "steal",
    "block",
    "shooting_foul",
    "offensive_foul",
    "loose_ball_foul",
    "violation",
    "jump_ball",
    "timeout",
    "substitution",
    "end_period",
    "other",
}
SCORING_ACTIONS = {
    "make_2",
    "make_3",
    "miss_2",
    "miss_3",
    "make_ft",
    "miss_ft",
    "transition_make",
    "transition_miss",
}

POSITION_GROUPS = ["PG", "SG", "SF", "PF", "C", "UNK"]
POSITION_TO_ID = {name: idx for idx, name in enumerate(POSITION_GROUPS)}
THREE_POINT_MARKERS = ("3PT", "3-PT", "THREE POINT")

LINEUP_COLS = [f"home_player{i}" for i in range(1, 6)] + [f"away_player{i}" for i in range(1, 6)]
SUPPORT = np.arange(-40, 41, dtype=np.float32)
SUPPORT_TENSOR = torch.tensor(SUPPORT, dtype=torch.float32)

NUMERIC_FEATURES = [
    "pre_margin_home",
    "abs_pre_margin_home",
    "home_possession_flag",
    "period",
    "seconds_remaining_game",
    "seconds_remaining_quarter",
    "normalized_game_progress",
    "clutch_flag",
    "ew_home_scoring_30",
    "ew_home_scoring_60",
    "ew_home_scoring_120",
    "ew_away_scoring_30",
    "ew_away_scoring_60",
    "ew_away_scoring_120",
    "ew_margin_swing_30",
    "ew_margin_swing_60",
    "ew_margin_swing_120",
    "ew_event_rate_30",
    "ew_event_rate_60",
    "ew_event_rate_120",
    "ew_transition_rate_30",
    "ew_transition_rate_60",
    "ew_transition_rate_120",
    "seconds_since_last_event",
    "seconds_since_possession_start",
    "event_count_in_possession",
    "prev_event_make_flag",
    "prev_event_miss_flag",
    "prev_event_turnover_flag",
    "prev_event_rebound_flag",
    "prev_event_foul_flag",
    "prev_event_transition_flag",
    "prev_initial_ev",
    "prev_dynamic_ev",
    "prev_offense_changed_flag",
    "season_year_normalized",
    "is_playoffs_flag",
]


@dataclass
class ArrayBundle:
    numeric: np.ndarray
    numeric_scaled: np.ndarray
    player_ids: np.ndarray
    side_ids: np.ndarray
    pos_ids: np.ndarray
    rewards: np.ndarray
    gamma: np.ndarray
    next_index: np.ndarray
    terminal: np.ndarray
    pre_margin_home: np.ndarray
    terminal_final_margin_home: np.ndarray
    terminal_remaining_margin: np.ndarray
    home_win: np.ndarray
    action_id: np.ndarray
    primary_player_id: np.ndarray
    secondary_player_id: np.ndarray
    tertiary_player_id: np.ndarray
    season_end_year: np.ndarray
    game_date_ord: np.ndarray
    game_id: np.ndarray
    event_num: np.ndarray
    team_id_home: np.ndarray
    team_id_away: np.ndarray
    output_meta: pd.DataFrame
    player_vocab: np.ndarray
    player_name_map: Dict[int, str]
    player_team_map: Dict[int, int]
    player_pos_group: Dict[int, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the paper-style DRL/Shapley model.")
    parser.add_argument("--train-start", type=int, default=21, help="First season suffix, e.g. 21")
    parser.add_argument("--train-end", type=int, default=26, help="Last season suffix, e.g. 26")
    parser.add_argument("--output-year", type=int, default=26, help="Season suffix to emit outputs for")
    parser.add_argument("--season-type", type=str, default="RS", choices=["RS"], help="Regular season only")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--attributor-epochs", type=int, default=6)
    parser.add_argument("--allocator-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=7e-4)
    parser.add_argument("--validation-frac", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--lineup-dim", type=int, default=20)
    parser.add_argument("--num-attn-heads", type=int, default=8)
    parser.add_argument("--max-games-per-season", type=int, default=0, help="0 = full season")
    parser.add_argument("--max-shapley-states", type=int, default=2500)
    parser.add_argument("--shapley-permutations", type=int, default=100)
    parser.add_argument("--permutation-tests", type=int, default=10000)
    parser.add_argument("--min-pair-possessions", type=int, default=500)
    parser.add_argument("--leaderboard-min-possessions", type=int, default=500)
    parser.add_argument("--report-shrinkage-k", type=float, default=1500.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_game_id(value: object) -> int:
    text = str(value).strip()
    if not text:
        return -1
    return int(text.lstrip("0") or "0")


def normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace(".", "").replace("'", "").replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def infer_pos_group(pos2: object) -> str:
    text = str(pos2 or "").upper()
    for pos in POSITION_GROUPS[:-1]:
        if pos in text:
            return pos
    return "UNK"


def parse_pos_weights(pos2: object) -> Dict[str, float]:
    text = str(pos2 or "").upper().replace("/", "-").replace(",", "-")
    hits = [pos for pos in POSITION_GROUPS[:-1] if pos in text]
    if not hits:
        return {"UNK": 1.0}
    weight = 1.0 / len(hits)
    return {pos: weight for pos in hits}


def determine_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_game_dates() -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/gabriel1200/shot_data/refs/heads/master/game_dates.csv"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    df["season"] = df["season"].astype(str).str.split("-").str[0].astype(int)
    df["GAME_ID"] = pd.to_numeric(df["GAME_ID"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["GAME_ID", "date", "season"]).copy()
    df["GAME_ID"] = df["GAME_ID"].astype(int)
    return df[["GAME_ID", "date", "season"]].drop_duplicates("GAME_ID", keep="first")


def fetch_player_metadata(train_start: int, train_end: int) -> pd.DataFrame:
    load_dotenv(PROJECT_ROOT / ".env")
    rows: List[dict] = []
    if create_client is not None and os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_KEY"):
        client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
        for end_year in range(2000 + train_start, 2000 + train_end + 1):
            try:
                response = (
                    client.table("player_stats")
                    .select('nba_id, player_name, "Name", "ShortName", "Pos2", "TeamId", "TeamAbbreviation", year, playoffs')
                    .eq("year", end_year)
                    .eq("playoffs", 0)
                    .execute()
                )
                rows.extend(response.data or [])
            except Exception as exc:
                LOGGER.warning("Supabase player metadata query failed for %s: %s", end_year, exc)
    if not rows and AUTOCOMPLETE_MAP.exists():
        auto = pd.read_csv(AUTOCOMPLETE_MAP)
        auto["player_name"] = auto["player_name"].astype(str)
        auto["Pos2"] = "UNK"
        auto["TeamId"] = pd.NA
        auto["TeamAbbreviation"] = pd.NA
        auto["year"] = 2000 + train_end
        auto = auto.rename(columns={"nba_id": "nba_id"})
        return auto[["nba_id", "player_name", "Pos2", "TeamId", "TeamAbbreviation", "year"]].drop_duplicates("nba_id")

    meta = pd.DataFrame(rows)
    if meta.empty:
        raise RuntimeError("Unable to build player metadata from Supabase or local fallback.")

    if "player_name" not in meta.columns:
        meta["player_name"] = meta.get("Name", meta.get("ShortName", pd.Series("", index=meta.index)))
    meta["player_name"] = meta["player_name"].fillna(meta.get("Name", "")).fillna(meta.get("ShortName", "")).astype(str)
    meta["nba_id"] = pd.to_numeric(meta["nba_id"], errors="coerce").astype("Int64")
    meta = meta.dropna(subset=["nba_id"]).copy()
    meta["nba_id"] = meta["nba_id"].astype(int)
    meta["Pos2"] = meta.get("Pos2", "UNK").fillna("UNK")
    meta["TeamId"] = pd.to_numeric(meta.get("TeamId"), errors="coerce").astype("Int64")
    meta["year"] = pd.to_numeric(meta.get("year"), errors="coerce").astype("Int64")
    meta = meta.sort_values(["nba_id", "year"]).drop_duplicates(["nba_id", "year", "TeamId"], keep="first")
    return meta


def build_player_maps(meta: pd.DataFrame) -> Tuple[Dict[int, str], Dict[int, int], Dict[int, str], Dict[int, Dict[str, float]], Dict[int, List[str]]]:
    name_map: Dict[int, str] = {}
    team_map: Dict[int, int] = {}
    pos_map: Dict[int, str] = {}
    pos_weights_map: Dict[int, Dict[str, float]] = {}
    aliases: Dict[int, List[str]] = defaultdict(list)

    for player_id, group in meta.groupby("nba_id", sort=False):
        latest = group.sort_values("year").iloc[-1]
        name = str(latest.get("player_name", "") or latest.get("Name", "") or latest.get("ShortName", ""))
        if not name:
            continue
        name_map[player_id] = name
        team_val = latest.get("TeamId")
        if pd.notna(team_val):
            team_map[player_id] = int(team_val)
        pos = infer_pos_group(latest.get("Pos2"))
        pos_map[player_id] = pos
        pos_weights_map[player_id] = parse_pos_weights(latest.get("Pos2"))

        alias_candidates = {
            normalize_name(name),
            normalize_name(latest.get("Name", "")),
            normalize_name(latest.get("ShortName", "")),
        }
        split = normalize_name(name).split()
        if split:
            alias_candidates.add(split[-1])
            alias_candidates.add(" ".join(split[-2:]))
        aliases[player_id] = [alias for alias in alias_candidates if alias]
    return name_map, team_map, pos_map, pos_weights_map, aliases


def is_valid_player_id(value: object) -> bool:
    try:
        return int(value) >= 1000
    except Exception:
        return False


def seconds_remaining_game(period: int, seconds_remaining_quarter: float) -> float:
    if period <= 4:
        return max(0.0, (4 - period) * 720.0 + seconds_remaining_quarter)
    return max(0.0, seconds_remaining_quarter)


def elapsed_seconds(period: int, seconds_remaining_quarter: float) -> float:
    total = 0.0
    for p in range(1, period):
        total += 720.0 if p <= 4 else 300.0
    current_len = 720.0 if period <= 4 else 300.0
    return total + max(0.0, current_len - seconds_remaining_quarter)


def classify_primary_side(
    row: pd.Series,
    home_lineup: Sequence[int],
    away_lineup: Sequence[int],
) -> str:
    player_id = row.get("player1_id")
    if is_valid_player_id(player_id):
        player_id = int(player_id)
        if player_id in home_lineup:
            return "home"
        if player_id in away_lineup:
            return "away"
    home_desc = str(row.get("home_description") or "")
    visitor_desc = str(row.get("visitor_description") or "")
    if home_desc and not visitor_desc:
        return "home"
    if visitor_desc and not home_desc:
        return "away"
    return "neutral"


def infer_home_away_team_ids(game_df: pd.DataFrame) -> Tuple[int, int]:
    home_team = -1
    away_team = -1
    for _, row in game_df.iterrows():
        team_id = row.get("player1_team_id")
        if not pd.notna(team_id):
            continue
        team_id = int(team_id)
        if team_id <= 0:
            continue
        home_desc = str(row.get("home_description") or "")
        visitor_desc = str(row.get("visitor_description") or "")
        if home_desc and home_team < 0:
            home_team = team_id
        if visitor_desc and away_team < 0:
            away_team = team_id
        if home_team > 0 and away_team > 0:
            break
    return home_team, away_team


def parse_assist_name(text: str) -> Optional[str]:
    match = re.search(r"\(([^()]+?)\s+\d+\s+AST\)", text)
    return match.group(1).strip() if match else None


def parse_tip_recipient(text: str) -> Optional[str]:
    match = re.search(r"Tip to ([A-Za-z.\-'\s]+)", text)
    return match.group(1).strip() if match else None


def is_three_point_event_text(text: str) -> bool:
    upper = str(text or "").upper()
    return any(marker in upper for marker in THREE_POINT_MARKERS)


def match_name_to_lineup(
    fragment: Optional[str],
    candidate_ids: Sequence[int],
    aliases: Dict[int, List[str]],
) -> Optional[int]:
    if not fragment:
        return None
    needle = normalize_name(fragment)
    if not needle:
        return None
    tokens = needle.split()
    candidates = []
    for player_id in candidate_ids:
        if not is_valid_player_id(player_id):
            continue
        for alias in aliases.get(int(player_id), []):
            if alias == needle or alias.endswith(needle) or needle.endswith(alias):
                return int(player_id)
            if tokens and alias.endswith(tokens[-1]):
                candidates.append(int(player_id))
    return candidates[0] if len(candidates) == 1 else None


def parse_rebound_type(text: str) -> Optional[str]:
    if "Off:" in text and "Def:" in text:
        off_match = re.search(r"Off:(\d+)", text)
        def_match = re.search(r"Def:(\d+)", text)
        if off_match and def_match:
            if int(off_match.group(1)) > 0:
                return "off"
            if int(def_match.group(1)) > 0:
                return "def"
    return None


def is_final_free_throw(text: str) -> bool:
    if "Technical" in text or "Flagrant" in text or "Clear Path" in text:
        return False
    match = re.search(r"(\d+)\s+of\s+(\d+)", text)
    if not match:
        return True
    return match.group(1) == match.group(2)


def classify_event(
    row: pd.Series,
    offense_side: str,
    aliases: Dict[int, List[str]],
) -> dict:
    home_lineup = [int(x) for x in row[LINEUP_COLS[:5]].tolist()]
    away_lineup = [int(x) for x in row[LINEUP_COLS[5:]].tolist()]
    home_desc = str(row.get("home_description") or "")
    visitor_desc = str(row.get("visitor_description") or "")
    neutral_desc = str(row.get("neutral_description") or "")
    text = " ".join(part for part in [home_desc, visitor_desc, neutral_desc] if part).strip()
    lower = text.lower()
    event_type = int(row["event_type"])
    transition_flag = bool(row.get("is_transition", False))
    player1_id = int(row["player1_id"]) if is_valid_player_id(row.get("player1_id")) else -1
    primary_side = classify_primary_side(row, home_lineup, away_lineup)
    offense_ids = home_lineup if offense_side == "home" else away_lineup
    defense_ids = away_lineup if offense_side == "home" else home_lineup

    primary = player1_id if player1_id > 0 else -1
    secondary = -1
    tertiary = -1
    event_side = primary_side

    if event_type == 1:
        secondary = match_name_to_lineup(parse_assist_name(text), offense_ids, aliases) or -1
        if transition_flag:
            action_class = "transition_make"
        else:
            action_class = "make_3" if is_three_point_event_text(text) else "make_2"
        event_side = primary_side
    elif event_type == 2:
        blocker_side = "home" if "BLOCK" in home_desc else ("away" if "BLOCK" in visitor_desc else "neutral")
        if blocker_side != "neutral":
            tertiary = match_name_to_lineup(text.split("BLOCK")[0].split()[-1] if False else None, defense_ids, aliases) or -1
        if transition_flag:
            action_class = "transition_miss"
        else:
            action_class = "miss_3" if is_three_point_event_text(text) else "miss_2"
        if "BLOCK" in home_desc:
            tertiary = match_name_to_lineup(home_desc.split(" BLOCK")[0].split()[-1], home_lineup, aliases) or -1
        if "BLOCK" in visitor_desc:
            tertiary = match_name_to_lineup(visitor_desc.split(" BLOCK")[0].split()[-1], away_lineup, aliases) or -1
        event_side = primary_side
    elif event_type == 3:
        action_class = "miss_ft" if "MISS" in text.upper() else "make_ft"
        event_side = primary_side
    elif event_type == 4:
        reb_type = parse_rebound_type(text)
        if reb_type == "off":
            action_class = "off_rebound"
            event_side = primary_side
        elif reb_type == "def":
            action_class = "def_rebound"
            event_side = primary_side
        else:
            event_side = primary_side if primary_side != "neutral" else offense_side
            action_class = "off_rebound" if event_side == offense_side else "def_rebound"
    elif event_type == 5:
        steal_side = "home" if "STEAL" in home_desc else ("away" if "STEAL" in visitor_desc else "neutral")
        if steal_side != "neutral":
            action_class = "steal"
            event_side = steal_side
            primary = -1
            stealer_fragment = home_desc.split(" STEAL")[0] if steal_side == "home" else visitor_desc.split(" STEAL")[0]
            tertiary = match_name_to_lineup(stealer_fragment, home_lineup if steal_side == "home" else away_lineup, aliases) or -1
        else:
            action_class = "turnover_bad_pass" if "BAD PASS" in text.upper() else "turnover_other"
            event_side = primary_side if primary_side != "neutral" else offense_side
    elif event_type == 6:
        upper = text.upper()
        if "OFF.FOUL" in upper or "OFFENSIVE" in upper or "CHARGE" in upper:
            action_class = "offensive_foul"
        elif "L.B.FOUL" in upper or "LOOSE BALL FOUL" in upper:
            action_class = "loose_ball_foul"
        elif "S.FOUL" in upper:
            action_class = "shooting_foul"
        else:
            action_class = "other"
        event_side = primary_side
    elif event_type == 7:
        action_class = "violation"
        event_side = primary_side if primary_side != "neutral" else offense_side
    elif event_type == 8:
        action_class = "substitution"
        event_side = primary_side
    elif event_type == 9:
        action_class = "timeout"
        event_side = primary_side
    elif event_type == 10:
        action_class = "jump_ball"
        tip_name = parse_tip_recipient(text)
        tip_id = match_name_to_lineup(tip_name, home_lineup + away_lineup, aliases)
        tertiary = tip_id or -1
        if tip_id in home_lineup:
            event_side = "home"
        elif tip_id in away_lineup:
            event_side = "away"
        else:
            event_side = offense_side
    elif event_type == 13:
        action_class = "end_period"
        event_side = offense_side
    else:
        action_class = "other"
        event_side = primary_side if primary_side != "neutral" else offense_side

    return {
        "action_class": action_class,
        "event_side": event_side,
        "primary_player_id": primary,
        "secondary_player_id": secondary,
        "tertiary_player_id": tertiary,
        "text": text,
        "transition_flag": int(transition_flag),
    }


def infer_initial_offense_side(game_df: pd.DataFrame, aliases: Dict[int, List[str]]) -> str:
    for _, row in game_df.iterrows():
        home_lineup = [int(x) for x in row[LINEUP_COLS[:5]].tolist()]
        away_lineup = [int(x) for x in row[LINEUP_COLS[5:]].tolist()]
        event_type = int(row["event_type"])
        home_desc = str(row.get("home_description") or "")
        visitor_desc = str(row.get("visitor_description") or "")
        text = " ".join(part for part in [home_desc, visitor_desc, str(row.get("neutral_description") or "")] if part)
        primary_side = classify_primary_side(row, home_lineup, away_lineup)
        if event_type in (1, 2, 3):
            return primary_side if primary_side in {"home", "away"} else "home"
        if event_type == 5:
            if "STEAL" in home_desc:
                return "away"
            if "STEAL" in visitor_desc:
                return "home"
            if primary_side in {"home", "away"}:
                return primary_side
        if event_type == 4:
            reb_type = parse_rebound_type(text)
            if reb_type == "off" and primary_side in {"home", "away"}:
                return primary_side
            if reb_type == "def" and primary_side == "home":
                return "away"
            if reb_type == "def" and primary_side == "away":
                return "home"
        if event_type == 10:
            tip_name = parse_tip_recipient(text)
            tip_id = match_name_to_lineup(tip_name, home_lineup + away_lineup, aliases)
            if tip_id in home_lineup:
                return "home"
            if tip_id in away_lineup:
                return "away"
    return "home"


def next_offense_side(
    offense_side: str,
    event: dict,
    row: pd.Series,
) -> str:
    action = event["action_class"]
    event_side = event["event_side"]
    text = event["text"]
    if action in {"make_2", "make_3", "transition_make"}:
        return "away" if event_side == "home" else "home"
    if action == "make_ft":
        return offense_side if not is_final_free_throw(text) else ("away" if event_side == "home" else "home")
    if action == "miss_ft":
        return offense_side
    if action == "off_rebound":
        return event_side
    if action == "def_rebound":
        return event_side
    if action in {"turnover_bad_pass", "turnover_other", "offensive_foul"}:
        return "away" if event_side == "home" else "home"
    if action == "steal":
        return event_side
    if action == "jump_ball":
        return event_side if event_side in {"home", "away"} else offense_side
    if action == "violation":
        upper = text.upper()
        if "KICKED BALL" in upper or "DEFENSIVE GOALTENDING" in upper:
            return offense_side
        if "SHOT CLOCK" in upper or "TRAVELING" in upper or "DOUBLE DRIBBLE" in upper:
            return "away" if offense_side == "home" else "home"
    return offense_side


def game_progress(period: int, seconds_remaining_q: float) -> float:
    elapsed = elapsed_seconds(period, seconds_remaining_q)
    total = 2880.0
    if period > 4:
        total += (period - 4) * 300.0
    return elapsed / max(total, 1.0)


def make_score_buckets(series: pd.Series) -> pd.Series:
    return pd.cut(
        series,
        bins=[-1e9, -10, -5, 5, 10, 1e9],
        labels=["down_10_plus", "down_5_9", "within_5", "up_5_9", "up_10_plus"],
    ).astype(str)


def make_time_buckets(series: pd.Series) -> pd.Series:
    return pd.cut(
        series,
        bins=[-1, 180, 360, 720, 1e9],
        labels=["0_3_min", "3_6_min", "6_12_min", "12_plus_min"],
    ).astype(str)


def build_transition_dataframe(
    train_start: int,
    train_end: int,
    max_games_per_season: int,
    aliases: Dict[int, List[str]],
) -> pd.DataFrame:
    schedule = load_game_dates()
    all_frames: List[pd.DataFrame] = []

    for suffix in range(train_start, train_end + 1):
        season_end_year = 2000 + suffix
        path = RAW_DATA_DIR / f"NBA{suffix}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        LOGGER.info("Loading raw events from %s", path)
        use_cols = [
            "game_id",
            "event_num",
            "event_type",
            "event_action_type",
            "period",
            "time_quarter",
            "seconds_remaining_quarter",
            "home_description",
            "visitor_description",
            "neutral_description",
            "home_score",
            "away_score",
            "score_margin",
            "player1_id",
            "player1_name",
            "player1_team_id",
            "player2_id",
            "player2_name",
            "player2_team_id",
            "player3_id",
            "player3_name",
            "player3_team_id",
            *LINEUP_COLS,
        ]
        raw_cols = pd.read_parquet(path, engine="pyarrow").columns.tolist()
        for optional in ["is_transition", "initial_ev", "dynamic_ev"]:
            if optional in raw_cols:
                use_cols.append(optional)
        df = pd.read_parquet(path, columns=use_cols)
        df = df.rename(columns={"game_id": "game_id_raw"})
        df["game_id"] = df["game_id_raw"].map(normalize_game_id)
        df["event_type"] = pd.to_numeric(df["event_type"], errors="coerce").fillna(0).astype(int)
        df["event_num"] = pd.to_numeric(df["event_num"], errors="coerce").fillna(0).astype(int)
        df["period"] = pd.to_numeric(df["period"], errors="coerce").fillna(1).astype(int)
        df["seconds_remaining_quarter"] = pd.to_numeric(df["seconds_remaining_quarter"], errors="coerce").fillna(0.0)
        df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
        df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
        if "is_transition" not in df.columns:
            df["is_transition"] = False
        if "initial_ev" not in df.columns:
            df["initial_ev"] = 0.0
        if "dynamic_ev" not in df.columns:
            df["dynamic_ev"] = 0.0
        df["initial_ev"] = pd.to_numeric(df["initial_ev"], errors="coerce").fillna(0.0)
        df["dynamic_ev"] = pd.to_numeric(df["dynamic_ev"], errors="coerce").fillna(0.0)
        for col in LINEUP_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)

        season_schedule = schedule[schedule["season"] == season_end_year - 1].copy()
        df = df.merge(season_schedule.rename(columns={"GAME_ID": "game_id", "date": "game_date"}), on="game_id", how="left")
        if df["game_date"].isna().all():
            raise RuntimeError(f"No game dates matched for season {season_end_year}")
        if max_games_per_season > 0:
            keep_games = sorted(df["game_id"].dropna().unique())[:max_games_per_season]
            df = df[df["game_id"].isin(keep_games)].copy()
        df = df.sort_values(["game_date", "game_id", "event_num"]).reset_index(drop=True)

        records: List[dict] = []
        for game_id, game_df in df.groupby("game_id", sort=False):
            game_df = game_df.sort_values("event_num").reset_index(drop=True)
            game_date = pd.to_datetime(game_df["game_date"].iloc[0])
            home_team_id, away_team_id = infer_home_away_team_ids(game_df)

            home_post = game_df["home_score"].ffill().fillna(0.0).astype(float).to_numpy()
            away_post = game_df["away_score"].ffill().fillna(0.0).astype(float).to_numpy()
            pre_home = np.roll(home_post, 1)
            pre_away = np.roll(away_post, 1)
            pre_home[0] = 0.0
            pre_away[0] = 0.0
            reward_home = (home_post - pre_home) - (away_post - pre_away)
            final_margin = float(home_post[-1] - away_post[-1])

            elapsed = np.array(
                [
                    elapsed_seconds(int(period), float(seconds_remaining))
                    for period, seconds_remaining in zip(
                        game_df["period"].to_numpy(),
                        game_df["seconds_remaining_quarter"].to_numpy(),
                    )
                ],
                dtype=np.float32,
            )
            next_elapsed = np.roll(elapsed, -1)
            next_elapsed[-1] = elapsed[-1]
            delta_seconds = np.maximum(0.0, next_elapsed - elapsed)

            offense_side = infer_initial_offense_side(game_df, aliases)
            last_elapsed = elapsed[0]
            possession_start_elapsed = elapsed[0]
            events_in_possession = 0.0
            prev_action = "other"
            prev_transition = 0.0
            prev_initial_ev = 0.0
            prev_dynamic_ev = 0.0
            prev_offense_changed = 0.0
            ew_state = {
                "home_scoring": {30: 0.0, 60: 0.0, 120: 0.0},
                "away_scoring": {30: 0.0, 60: 0.0, 120: 0.0},
                "margin_swing": {30: 0.0, 60: 0.0, 120: 0.0},
                "event_rate": {30: 0.0, 60: 0.0, 120: 0.0},
                "transition_rate": {30: 0.0, 60: 0.0, 120: 0.0},
            }

            for i, row in game_df.iterrows():
                current_elapsed = elapsed[i]
                dt = 0.0 if i == 0 else max(0.0, current_elapsed - last_elapsed)
                for tau in (30, 60, 120):
                    decay = math.exp(-dt / tau) if dt > 0 else 1.0
                    for metric_map in ew_state.values():
                        metric_map[tau] *= decay

                event = classify_event(row, offense_side, aliases)
                pre_margin = float(pre_home[i] - pre_away[i])
                secs_rem_game = float(seconds_remaining_game(int(row["period"]), float(row["seconds_remaining_quarter"])))
                abs_margin = abs(pre_margin)
                clutch_flag = float(secs_rem_game <= 180 and abs_margin <= 5)

                record = {
                    "game_id": int(game_id),
                    "game_date": game_date,
                    "season_end_year": season_end_year,
                    "event_num": int(row["event_num"]),
                    "action_class": event["action_class"],
                    "action_id": ACTION_TO_ID[event["action_class"]],
                    "reward_home": float(reward_home[i]),
                    "delta_seconds": float(delta_seconds[i]),
                    "gamma": float(0.997 ** delta_seconds[i]),
                    "terminal": int(i == len(game_df) - 1),
                    "next_local_index": int(i + 1 if i < len(game_df) - 1 else -1),
                    "terminal_final_margin_home": final_margin,
                    "terminal_remaining_margin": float(final_margin - pre_margin),
                    "home_win": int(final_margin > 0),
                    "primary_player_id": int(event["primary_player_id"]),
                    "secondary_player_id": int(event["secondary_player_id"]),
                    "tertiary_player_id": int(event["tertiary_player_id"]),
                    "team_id_home": int(home_team_id),
                    "team_id_away": int(away_team_id),
                    "pre_margin_home": pre_margin,
                    "abs_pre_margin_home": abs_margin,
                    "home_possession_flag": float(offense_side == "home"),
                    "period": float(row["period"]),
                    "seconds_remaining_game": secs_rem_game,
                    "seconds_remaining_quarter": float(row["seconds_remaining_quarter"]),
                    "normalized_game_progress": float(game_progress(int(row["period"]), float(row["seconds_remaining_quarter"]))),
                    "clutch_flag": clutch_flag,
                    "ew_home_scoring_30": ew_state["home_scoring"][30] / 30.0,
                    "ew_home_scoring_60": ew_state["home_scoring"][60] / 60.0,
                    "ew_home_scoring_120": ew_state["home_scoring"][120] / 120.0,
                    "ew_away_scoring_30": ew_state["away_scoring"][30] / 30.0,
                    "ew_away_scoring_60": ew_state["away_scoring"][60] / 60.0,
                    "ew_away_scoring_120": ew_state["away_scoring"][120] / 120.0,
                    "ew_margin_swing_30": ew_state["margin_swing"][30] / 30.0,
                    "ew_margin_swing_60": ew_state["margin_swing"][60] / 60.0,
                    "ew_margin_swing_120": ew_state["margin_swing"][120] / 120.0,
                    "ew_event_rate_30": ew_state["event_rate"][30] / 30.0,
                    "ew_event_rate_60": ew_state["event_rate"][60] / 60.0,
                    "ew_event_rate_120": ew_state["event_rate"][120] / 120.0,
                    "ew_transition_rate_30": ew_state["transition_rate"][30] / 30.0,
                    "ew_transition_rate_60": ew_state["transition_rate"][60] / 60.0,
                    "ew_transition_rate_120": ew_state["transition_rate"][120] / 120.0,
                    "seconds_since_last_event": float(0.0 if i == 0 else dt),
                    "seconds_since_possession_start": float(max(0.0, current_elapsed - possession_start_elapsed)),
                    "event_count_in_possession": float(events_in_possession),
                    "prev_event_make_flag": float(prev_action in {"make_2", "make_3", "make_ft", "transition_make"}),
                    "prev_event_miss_flag": float(prev_action in {"miss_2", "miss_3", "miss_ft", "transition_miss"}),
                    "prev_event_turnover_flag": float(prev_action in {"turnover_bad_pass", "turnover_other", "steal"}),
                    "prev_event_rebound_flag": float(prev_action in {"off_rebound", "def_rebound"}),
                    "prev_event_foul_flag": float(prev_action in {"shooting_foul", "offensive_foul", "loose_ball_foul"}),
                    "prev_event_transition_flag": float(prev_transition),
                    "prev_initial_ev": float(prev_initial_ev),
                    "prev_dynamic_ev": float(prev_dynamic_ev),
                    "prev_offense_changed_flag": float(prev_offense_changed),
                    "season_year_normalized": float((season_end_year - 2021) / max(train_end - train_start, 1)),
                    "is_playoffs_flag": 0.0,
                    "is_transition_event": int(bool(row.get("is_transition", False))),
                    "initial_ev": float(row.get("initial_ev", 0.0) or 0.0),
                    "dynamic_ev": float(row.get("dynamic_ev", 0.0) or 0.0),
                }
                for idx_col, col in enumerate(LINEUP_COLS, start=1):
                    record[col] = int(row[col])
                records.append(record)

                home_points = float(home_post[i] - pre_home[i])
                away_points = float(away_post[i] - pre_away[i])
                for tau in (30, 60, 120):
                    ew_state["home_scoring"][tau] += home_points
                    ew_state["away_scoring"][tau] += away_points
                    ew_state["margin_swing"][tau] += float(reward_home[i])
                    ew_state["event_rate"][tau] += 1.0
                    ew_state["transition_rate"][tau] += float(bool(row.get("is_transition", False)))

                next_offense = next_offense_side(offense_side, event, row)
                offense_changed = float(next_offense != offense_side)
                if offense_changed:
                    possession_start_elapsed = current_elapsed
                    events_in_possession = 0.0
                else:
                    events_in_possession += 1.0

                prev_action = event["action_class"]
                prev_transition = float(bool(row.get("is_transition", False)))
                prev_initial_ev = float(row.get("initial_ev", 0.0) or 0.0)
                prev_dynamic_ev = float(row.get("dynamic_ev", 0.0) or 0.0)
                prev_offense_changed = offense_changed
                offense_side = next_offense
                last_elapsed = current_elapsed

        season_df = pd.DataFrame.from_records(records)
        season_df["season_suffix"] = suffix
        all_frames.append(season_df)
        LOGGER.info(
            "Built %s transitions across %s games for %s",
            len(season_df),
            season_df["game_id"].nunique(),
            season_end_year,
        )

    df = pd.concat(all_frames, ignore_index=True)
    df = df.sort_values(["game_date", "game_id", "event_num"]).reset_index(drop=True)
    df["global_index"] = np.arange(len(df), dtype=np.int64)
    df["next_index"] = (
        df.groupby("game_id", sort=False)["global_index"].shift(-1).fillna(-1).astype(np.int64)
    )
    df["terminal"] = (df["next_index"] < 0).astype(int)
    return df


def save_dataset_cache(df: pd.DataFrame, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    LOGGER.info("Saved dataset cache to %s", cache_path)


def load_or_build_dataset(args: argparse.Namespace, aliases: Dict[int, List[str]]) -> pd.DataFrame:
    games_tag = "all" if args.max_games_per_season <= 0 else f"games{args.max_games_per_season}"
    cache_path = CACHE_ROOT / f"transitions_{args.train_start}_{args.train_end}_{args.season_type}_{games_tag}.parquet"
    if cache_path.exists() and not args.rebuild_dataset:
        LOGGER.info("Loading cached transitions from %s", cache_path)
        return pd.read_parquet(cache_path)
    df = build_transition_dataframe(
        train_start=args.train_start,
        train_end=args.train_end,
        max_games_per_season=args.max_games_per_season,
        aliases=aliases,
    )
    save_dataset_cache(df, cache_path)
    return df


def materialize_arrays(
    df: pd.DataFrame,
    output_year: int,
    player_name_map: Dict[int, str],
    player_team_map: Dict[int, int],
    player_pos_group: Dict[int, str],
) -> ArrayBundle:
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["game_date_ord"] = df["game_date"].map(pd.Timestamp.toordinal).astype(np.int32)

    numeric = df[NUMERIC_FEATURES].astype(np.float32).to_numpy()
    player_ids = df[LINEUP_COLS].astype(np.int64).to_numpy()
    offense_home = df["home_possession_flag"].to_numpy(np.int64).reshape(-1, 1)
    side_ids = np.zeros_like(player_ids, dtype=np.int64)
    side_ids[:, :5] = offense_home
    side_ids[:, 5:] = 1 - offense_home
    pos_ids = np.vectorize(lambda pid: POSITION_TO_ID.get(player_pos_group.get(int(pid), "UNK"), POSITION_TO_ID["UNK"]))(
        player_ids
    ).astype(np.int64)

    output_mask = df["season_end_year"].eq(2000 + output_year).to_numpy()
    output_meta_cols = [
        "game_id",
        "game_date",
        "event_num",
        "season_end_year",
        "action_class",
        "reward_home",
        "gamma",
        "pre_margin_home",
        "terminal_final_margin_home",
        "primary_player_id",
        "secondary_player_id",
        "tertiary_player_id",
        "team_id_home",
        "team_id_away",
        *LINEUP_COLS,
        *NUMERIC_FEATURES,
    ]
    output_meta_cols = list(dict.fromkeys(output_meta_cols))
    output_meta = df.loc[output_mask, output_meta_cols].reset_index(drop=True)
    output_meta["global_index"] = df.loc[output_mask, "global_index"].to_numpy(np.int64)
    output_meta["score_bucket"] = make_score_buckets(output_meta["pre_margin_home"])
    output_meta["time_bucket"] = make_time_buckets(output_meta["seconds_remaining_game"])

    player_vocab = np.array(sorted({int(pid) for pid in np.unique(player_ids) if int(pid) > 0}), dtype=np.int64)
    return ArrayBundle(
        numeric=numeric,
        numeric_scaled=numeric.copy(),
        player_ids=player_ids,
        side_ids=side_ids,
        pos_ids=pos_ids,
        rewards=df["reward_home"].astype(np.float32).to_numpy(),
        gamma=df["gamma"].astype(np.float32).to_numpy(),
        next_index=df["next_index"].astype(np.int64).to_numpy(),
        terminal=df["terminal"].astype(bool).to_numpy(),
        pre_margin_home=df["pre_margin_home"].astype(np.float32).to_numpy(),
        terminal_final_margin_home=df["terminal_final_margin_home"].astype(np.float32).to_numpy(),
        terminal_remaining_margin=df["terminal_remaining_margin"].astype(np.float32).to_numpy(),
        home_win=df["home_win"].astype(np.int64).to_numpy(),
        action_id=df["action_id"].astype(np.int64).to_numpy(),
        primary_player_id=df["primary_player_id"].astype(np.int64).to_numpy(),
        secondary_player_id=df["secondary_player_id"].astype(np.int64).to_numpy(),
        tertiary_player_id=df["tertiary_player_id"].astype(np.int64).to_numpy(),
        season_end_year=df["season_end_year"].astype(np.int64).to_numpy(),
        game_date_ord=df["game_date_ord"].astype(np.int32).to_numpy(),
        game_id=df["game_id"].astype(np.int64).to_numpy(),
        event_num=df["event_num"].astype(np.int64).to_numpy(),
        team_id_home=df["team_id_home"].astype(np.int64).to_numpy(),
        team_id_away=df["team_id_away"].astype(np.int64).to_numpy(),
        output_meta=output_meta,
        player_vocab=player_vocab,
        player_name_map=player_name_map,
        player_team_map=player_team_map,
        player_pos_group=player_pos_group,
    )


def forward_chaining_split(bundle: ArrayBundle, output_year: int, validation_frac: float) -> Tuple[np.ndarray, np.ndarray]:
    output_season = 2000 + output_year
    output_mask = bundle.season_end_year == output_season
    output_dates = np.unique(bundle.game_date_ord[output_mask])
    output_dates.sort()
    holdout_count = max(1, int(math.ceil(len(output_dates) * validation_frac)))
    valid_dates = set(output_dates[-holdout_count:].tolist())
    valid_mask = output_mask & np.isin(bundle.game_date_ord, list(valid_dates))
    train_mask = ~valid_mask
    train_idx = np.flatnonzero(train_mask)
    valid_idx = np.flatnonzero(valid_mask)
    if len(train_idx) == 0 or len(valid_idx) == 0:
        output_games = np.unique(bundle.game_id[output_mask])
        output_games.sort()
        holdout_games = max(1, int(math.ceil(len(output_games) * validation_frac)))
        valid_game_ids = set(output_games[-holdout_games:].tolist())
        valid_mask = output_mask & np.isin(bundle.game_id, list(valid_game_ids))
        train_mask = ~valid_mask
        train_idx = np.flatnonzero(train_mask)
        valid_idx = np.flatnonzero(valid_mask)
    return train_idx, valid_idx


def scale_numeric_features(bundle: ArrayBundle, train_idx: np.ndarray) -> None:
    train_numeric = bundle.numeric[train_idx]
    mean = train_numeric.mean(axis=0, keepdims=True)
    std = train_numeric.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    bundle.numeric_scaled = ((bundle.numeric - mean) / std).astype(np.float32)


def batch_indices(indices: np.ndarray, batch_size: int, shuffle: bool, rng: np.random.Generator) -> Iterable[np.ndarray]:
    work = indices.copy()
    if shuffle:
        rng.shuffle(work)
    for start in range(0, len(work), batch_size):
        yield work[start : start + batch_size]


def to_device_batch(array: np.ndarray, batch: np.ndarray, device: torch.device, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    tensor = torch.as_tensor(array[batch], device=device)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def project_distribution(
    next_probs: torch.Tensor,
    rewards: torch.Tensor,
    gamma: torch.Tensor,
    support: torch.Tensor,
    terminal: torch.Tensor,
) -> torch.Tensor:
    batch_size = next_probs.shape[0]
    n_atoms = support.numel()
    target = torch.zeros_like(next_probs)

    tz = rewards.unsqueeze(1) + gamma.unsqueeze(1) * support.unsqueeze(0)
    tz = torch.where(terminal.unsqueeze(1), rewards.unsqueeze(1).expand_as(tz), tz)
    tz = tz.clamp(float(support[0]), float(support[-1]))

    b = (tz - support[0]) / (support[1] - support[0])
    l = b.floor().long()
    u = b.ceil().long()

    offset = torch.arange(batch_size, device=next_probs.device).unsqueeze(1) * n_atoms
    target_flat = target.view(-1)
    probs = torch.where(terminal.unsqueeze(1), torch.zeros_like(next_probs), next_probs)

    same = l == u
    target_flat.index_add_(
        0,
        (l + offset).view(-1),
        torch.where(same, torch.ones_like(b), (u.float() - b)).view(-1) * probs.view(-1),
    )
    target_flat.index_add_(
        0,
        (u + offset).view(-1),
        torch.where(same, torch.zeros_like(b), (b - l.float())).view(-1) * probs.view(-1),
    )
    if terminal.any():
        terminal_idx = torch.where(terminal)[0]
        rewards_terminal = rewards[terminal_idx].clamp(float(support[0]), float(support[-1]))
        bt = (rewards_terminal - support[0]) / (support[1] - support[0])
        lt = bt.floor().long()
        ut = bt.ceil().long()
        target[terminal_idx] = 0.0
        same_t = lt == ut
        for idx_local, idx_batch in enumerate(terminal_idx):
            if same_t[idx_local]:
                target[idx_batch, lt[idx_local]] = 1.0
            else:
                target[idx_batch, lt[idx_local]] += float(ut[idx_local] - bt[idx_local])
                target[idx_batch, ut[idx_local]] += float(bt[idx_local] - lt[idx_local])
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return target


class StateValueModel(nn.Module):
    def __init__(
        self,
        num_players: int,
        embedding_dim: int,
        hidden_size: int,
        lineup_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.player_embedding = nn.Embedding(num_players + 1, embedding_dim, padding_idx=0)
        self.side_embedding = nn.Embedding(2, 8)
        self.pos_embedding = nn.Embedding(len(POSITION_GROUPS), 8)
        self.token_proj = nn.Linear(embedding_dim + 8 + 8, embedding_dim)
        self.lineup_attn = nn.MultiheadAttention(embedding_dim, num_heads=num_heads, batch_first=True)
        self.lineup_norm = nn.LayerNorm(embedding_dim)
        self.lineup_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, lineup_dim),
        )
        self.numeric_head = nn.Sequential(
            nn.Linear(len(NUMERIC_FEATURES), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size + lineup_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, SUPPORT.shape[0]),
        )

    def embed_players(
        self,
        player_ids: torch.Tensor,
        side_ids: torch.Tensor,
        pos_ids: torch.Tensor,
        replacement_mask: Optional[torch.Tensor] = None,
        replacement_embed_by_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        player_emb = self.player_embedding(player_ids)
        if replacement_mask is not None and replacement_embed_by_pos is not None:
            replacement_tokens = replacement_embed_by_pos[pos_ids]
            player_emb = torch.where(replacement_mask.unsqueeze(-1), player_emb, replacement_tokens)
        token = torch.cat(
            [
                player_emb,
                self.side_embedding(side_ids),
                self.pos_embedding(pos_ids),
            ],
            dim=-1,
        )
        return self.token_proj(token)

    def encode_state(
        self,
        numeric: torch.Tensor,
        player_ids: torch.Tensor,
        side_ids: torch.Tensor,
        pos_ids: torch.Tensor,
        replacement_mask: Optional[torch.Tensor] = None,
        replacement_embed_by_pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.embed_players(
            player_ids,
            side_ids,
            pos_ids,
            replacement_mask=replacement_mask,
            replacement_embed_by_pos=replacement_embed_by_pos,
        )
        attn_out, _ = self.lineup_attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.lineup_norm(tokens + attn_out)
        lineup_summary = self.lineup_head(tokens.mean(dim=1))
        numeric_summary = self.numeric_head(numeric)
        encoded = torch.cat([numeric_summary, lineup_summary], dim=-1)
        return encoded, tokens, lineup_summary

    def forward(
        self,
        numeric: torch.Tensor,
        player_ids: torch.Tensor,
        side_ids: torch.Tensor,
        pos_ids: torch.Tensor,
        replacement_mask: Optional[torch.Tensor] = None,
        replacement_embed_by_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoded, _, _ = self.encode_state(
            numeric,
            player_ids,
            side_ids,
            pos_ids,
            replacement_mask=replacement_mask,
            replacement_embed_by_pos=replacement_embed_by_pos,
        )
        return self.output_head(encoded)


class ShapleyAttributor(nn.Module):
    def __init__(self, num_players: int, embedding_dim: int, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.player_embedding = nn.Embedding(num_players + 1, embedding_dim, padding_idx=0)
        self.side_embedding = nn.Embedding(2, 8)
        self.pos_embedding = nn.Embedding(len(POSITION_GROUPS), 8)
        self.token_proj = nn.Linear(embedding_dim + 8 + 8 + hidden_size, embedding_dim)
        self.attn = nn.MultiheadAttention(embedding_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embedding_dim)
        self.context_head = nn.Sequential(
            nn.Linear(len(NUMERIC_FEATURES), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden_size, hidden_size)
        self.output_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        numeric: torch.Tensor,
        player_ids: torch.Tensor,
        side_ids: torch.Tensor,
        pos_ids: torch.Tensor,
        target_value: torch.Tensor,
    ) -> torch.Tensor:
        context = self.context_head(numeric)
        context_token = self.value_head(context).unsqueeze(1).expand(-1, player_ids.size(1), -1)
        tokens = torch.cat(
            [
                self.player_embedding(player_ids),
                self.side_embedding(side_ids),
                self.pos_embedding(pos_ids),
                context_token,
            ],
            dim=-1,
        )
        tokens = self.token_proj(tokens)
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm(tokens + attn_out)
        raw = self.output_head(tokens).squeeze(-1)
        residual = target_value.unsqueeze(1) - raw.sum(dim=1, keepdim=True)
        return raw + residual / raw.size(1)


class OffensiveAllocator(nn.Module):
    def __init__(self, num_players: int, embedding_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.player_embedding = nn.Embedding(num_players + 1, embedding_dim, padding_idx=0)
        self.side_embedding = nn.Embedding(2, 8)
        self.pos_embedding = nn.Embedding(len(POSITION_GROUPS), 8)
        self.action_embedding = nn.Embedding(len(ACTION_CLASSES), 8)
        self.context_head = nn.Sequential(
            nn.Linear(len(NUMERIC_FEATURES), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.player_head = nn.Sequential(
            nn.Linear(embedding_dim + 8 + 8 + 8 + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        numeric: torch.Tensor,
        candidate_player_ids: torch.Tensor,
        candidate_side_ids: torch.Tensor,
        candidate_pos_ids: torch.Tensor,
        action_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        context = self.context_head(numeric).unsqueeze(1).expand(-1, candidate_player_ids.size(1), -1)
        tokens = torch.cat(
            [
                self.player_embedding(candidate_player_ids),
                self.side_embedding(candidate_side_ids),
                self.pos_embedding(candidate_pos_ids),
                self.action_embedding(action_ids).unsqueeze(1).expand(-1, candidate_player_ids.size(1), -1),
                context,
            ],
            dim=-1,
        )
        logits = self.player_head(tokens).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e9)
        return F.softmax(logits, dim=1)


def expected_remaining_margin(probs: torch.Tensor) -> torch.Tensor:
    return (probs * SUPPORT_TENSOR.to(probs.device)).sum(dim=1)


def sanitize_probability_tensor(probs: torch.Tensor) -> torch.Tensor:
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = probs.clamp_min(0.0)
    row_sums = probs.sum(dim=1, keepdim=True)
    safe = probs / row_sums.clamp_min(1e-8)
    zero_rows = row_sums.squeeze(1) <= 1e-8
    if zero_rows.any():
        safe[zero_rows] = 1.0 / safe.size(1)
    return safe


def margin_to_win_prob(probs: torch.Tensor, pre_margin_home: torch.Tensor) -> torch.Tensor:
    probs = sanitize_probability_tensor(probs)
    final_support = SUPPORT_TENSOR.to(probs.device).unsqueeze(0) + pre_margin_home.unsqueeze(1)
    greater = (final_support > 0).float()
    tie = (final_support == 0).float() * 0.5
    return (probs * (greater + tie)).sum(dim=1).clamp(0.0, 1.0)


def gather_batch(bundle: ArrayBundle, batch: np.ndarray, device: torch.device) -> dict:
    numeric = torch.as_tensor(bundle.numeric_scaled[batch], dtype=torch.float32, device=device)
    player_ids = torch.as_tensor(bundle.player_ids[batch], dtype=torch.long, device=device)
    side_ids = torch.as_tensor(bundle.side_ids[batch], dtype=torch.long, device=device)
    pos_ids = torch.as_tensor(bundle.pos_ids[batch], dtype=torch.long, device=device)
    rewards = torch.as_tensor(bundle.rewards[batch], dtype=torch.float32, device=device)
    gamma = torch.as_tensor(bundle.gamma[batch], dtype=torch.float32, device=device)
    terminal = torch.as_tensor(bundle.terminal[batch], dtype=torch.bool, device=device)
    pre_margin = torch.as_tensor(bundle.pre_margin_home[batch], dtype=torch.float32, device=device)
    next_idx = bundle.next_index[batch]
    nonterminal_mask = next_idx >= 0
    safe_next = next_idx.copy()
    safe_next[~nonterminal_mask] = batch[~nonterminal_mask]
    next_numeric = torch.as_tensor(bundle.numeric_scaled[safe_next], dtype=torch.float32, device=device)
    next_player_ids = torch.as_tensor(bundle.player_ids[safe_next], dtype=torch.long, device=device)
    next_side_ids = torch.as_tensor(bundle.side_ids[safe_next], dtype=torch.long, device=device)
    next_pos_ids = torch.as_tensor(bundle.pos_ids[safe_next], dtype=torch.long, device=device)
    return {
        "numeric": numeric,
        "player_ids": player_ids,
        "side_ids": side_ids,
        "pos_ids": pos_ids,
        "rewards": rewards,
        "gamma": gamma,
        "terminal": terminal,
        "pre_margin": pre_margin,
        "next_numeric": next_numeric,
        "next_player_ids": next_player_ids,
        "next_side_ids": next_side_ids,
        "next_pos_ids": next_pos_ids,
    }


def build_player_lookup(bundle: ArrayBundle) -> Tuple[Dict[int, int], np.ndarray]:
    vocab = np.unique(bundle.player_vocab)
    lookup = {int(pid): idx + 1 for idx, pid in enumerate(vocab.tolist())}
    return lookup, vocab


def remap_player_arrays(bundle: ArrayBundle, lookup: Dict[int, int]) -> None:
    remap = np.vectorize(lambda pid: lookup.get(int(pid), 0))
    bundle.player_ids = remap(bundle.player_ids).astype(np.int64)
    bundle.primary_player_id = remap(bundle.primary_player_id).astype(np.int64)
    bundle.secondary_player_id = remap(bundle.secondary_player_id).astype(np.int64)
    bundle.tertiary_player_id = remap(bundle.tertiary_player_id).astype(np.int64)
    for col in LINEUP_COLS:
        bundle.output_meta[col] = bundle.output_meta[col].map(lambda pid: lookup.get(int(pid), 0) if pd.notna(pid) else 0).astype(int)
    for col in ["primary_player_id", "secondary_player_id", "tertiary_player_id"]:
        bundle.output_meta[col] = bundle.output_meta[col].map(lambda pid: lookup.get(int(pid), 0) if pd.notna(pid) else 0).astype(int)


def train_value_model(
    bundle: ArrayBundle,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[StateValueModel, Dict[str, float]]:
    num_players = int(bundle.player_ids.max())
    model = StateValueModel(
        num_players=num_players,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        lineup_dim=args.lineup_dim,
        num_heads=args.num_attn_heads,
    ).to(device)
    target_model = StateValueModel(
        num_players=num_players,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        lineup_dim=args.lineup_dim,
        num_heads=args.num_attn_heads,
    ).to(device)
    target_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    best_state = None
    best_rmse = float("inf")
    epochs_without_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: List[float] = []
        for batch in batch_indices(train_idx, args.batch_size, shuffle=True, rng=rng):
            batch_data = gather_batch(bundle, batch, device)
            logits = model(batch_data["numeric"], batch_data["player_ids"], batch_data["side_ids"], batch_data["pos_ids"])
            log_probs = F.log_softmax(logits, dim=1)
            with torch.no_grad():
                next_logits = target_model(
                    batch_data["next_numeric"],
                    batch_data["next_player_ids"],
                    batch_data["next_side_ids"],
                    batch_data["next_pos_ids"],
                )
                next_probs = sanitize_probability_tensor(F.softmax(next_logits, dim=1))
                target_probs = project_distribution(
                    next_probs=next_probs,
                    rewards=batch_data["rewards"],
                    gamma=batch_data["gamma"],
                    support=SUPPORT_TENSOR.to(device),
                    terminal=batch_data["terminal"],
                )
            loss = -(target_probs * log_probs).sum(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        target_model.load_state_dict(model.state_dict())

        metrics = evaluate_value_model(model, bundle, valid_idx, device)
        LOGGER.info(
            "Value epoch %s | train_loss=%.4f valid_rmse=%.4f valid_brier=%.4f valid_logloss=%.4f",
            epoch,
            float(np.mean(losses) if losses else 0.0),
            metrics["rmse_final_margin"],
            metrics["brier"],
            metrics["logloss"],
        )
        if metrics["rmse_final_margin"] < best_rmse:
            best_rmse = metrics["rmse_final_margin"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = evaluate_value_model(model, bundle, valid_idx, device)
    return model, metrics


def predict_value_distribution(
    model: StateValueModel,
    bundle: ArrayBundle,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs_list: List[np.ndarray] = []
    values: List[np.ndarray] = []
    entropy: List[np.ndarray] = []
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for batch in batch_indices(indices, batch_size, shuffle=False, rng=rng):
            numeric = torch.as_tensor(bundle.numeric_scaled[batch], dtype=torch.float32, device=device)
            player_ids = torch.as_tensor(bundle.player_ids[batch], dtype=torch.long, device=device)
            side_ids = torch.as_tensor(bundle.side_ids[batch], dtype=torch.long, device=device)
            pos_ids = torch.as_tensor(bundle.pos_ids[batch], dtype=torch.long, device=device)
            logits = model(numeric, player_ids, side_ids, pos_ids)
            probs = sanitize_probability_tensor(F.softmax(logits, dim=1))
            value = expected_remaining_margin(probs)
            ent = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1)
            probs_list.append(probs.cpu().numpy())
            values.append(value.cpu().numpy())
            entropy.append(ent.cpu().numpy())
    return np.vstack(probs_list), np.concatenate(values), np.concatenate(entropy)


def evaluate_value_model(model: StateValueModel, bundle: ArrayBundle, indices: np.ndarray, device: torch.device) -> Dict[str, float]:
    probs, remaining_margin, entropy = predict_value_distribution(model, bundle, indices, device)
    pred_final_margin = bundle.pre_margin_home[indices] + remaining_margin
    observed_final_margin = bundle.terminal_final_margin_home[indices]
    pre_margin_tensor = torch.as_tensor(bundle.pre_margin_home[indices], dtype=torch.float32)
    win_probs = margin_to_win_prob(torch.as_tensor(probs, dtype=torch.float32), pre_margin_tensor).numpy()
    observed_home_win = bundle.home_win[indices]

    entropy_corr = np.corrcoef(entropy, (observed_final_margin - pred_final_margin) ** 2)[0, 1]
    return {
        "rmse_final_margin": float(math.sqrt(mean_squared_error(observed_final_margin, pred_final_margin))),
        "brier": float(brier_score_loss(observed_home_win, win_probs)),
        "logloss": float(log_loss(observed_home_win, np.clip(win_probs, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "entropy_variance_corr": float(entropy_corr if np.isfinite(entropy_corr) else 0.0),
    }


def evaluate_baselines(bundle: ArrayBundle, train_idx: np.ndarray, valid_idx: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float], pd.DataFrame]:
    feature_cols = [
        NUMERIC_FEATURES.index("pre_margin_home"),
        NUMERIC_FEATURES.index("abs_pre_margin_home"),
        NUMERIC_FEATURES.index("home_possession_flag"),
        NUMERIC_FEATURES.index("period"),
        NUMERIC_FEATURES.index("seconds_remaining_game"),
        NUMERIC_FEATURES.index("seconds_remaining_quarter"),
        NUMERIC_FEATURES.index("clutch_flag"),
    ]
    x_train_small = bundle.numeric_scaled[train_idx][:, feature_cols]
    x_valid_small = bundle.numeric_scaled[valid_idx][:, feature_cols]
    y_train_win = bundle.home_win[train_idx]
    y_valid_win = bundle.home_win[valid_idx]
    y_train_margin = bundle.terminal_final_margin_home[train_idx]
    y_valid_margin = bundle.terminal_final_margin_home[valid_idx]

    if len(np.unique(y_train_win)) >= 2 and len(train_idx) >= 10:
        logistic = LogisticRegression(max_iter=1000, n_jobs=1)
        logistic.fit(x_train_small, y_train_win)
        logistic_probs = logistic.predict_proba(x_valid_small)[:, 1]
    else:
        logistic_probs = np.full(len(valid_idx), float(np.mean(y_train_win) if len(y_train_win) else 0.5))
    logistic_metrics = {
        "brier": float(brier_score_loss(y_valid_win, logistic_probs)),
        "logloss": float(log_loss(y_valid_win, np.clip(logistic_probs, 1e-6, 1 - 1e-6), labels=[0, 1])),
    }

    if len(train_idx) >= 20:
        tree = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=7,
            tree_method="hist",
        )
        tree.fit(bundle.numeric[train_idx], y_train_margin)
        tree_margin = tree.predict(bundle.numeric[valid_idx])
    else:
        tree_margin = np.full(len(valid_idx), float(np.mean(y_train_margin) if len(y_train_margin) else 0.0))
    tree_probs = 1.0 / (1.0 + np.exp(-(tree_margin / 6.0)))
    tree_metrics = {
        "rmse_final_margin": float(math.sqrt(mean_squared_error(y_valid_margin, tree_margin))),
        "brier": float(brier_score_loss(y_valid_win, np.clip(tree_probs, 1e-6, 1 - 1e-6))),
        "logloss": float(log_loss(y_valid_win, np.clip(tree_probs, 1e-6, 1 - 1e-6), labels=[0, 1])),
    }

    eval_df = pd.DataFrame(
        {
            "game_id": bundle.game_id[valid_idx],
            "event_num": bundle.event_num[valid_idx],
            "game_date_ord": bundle.game_date_ord[valid_idx],
            "actual_home_win": y_valid_win,
            "actual_final_margin": y_valid_margin,
            "baseline_logistic_home_win_prob": logistic_probs,
            "baseline_tree_final_margin": tree_margin,
            "baseline_tree_home_win_prob": tree_probs,
        }
    )
    return logistic_metrics, tree_metrics, eval_df


def compute_replacement_embeddings(
    model: StateValueModel,
    remapped_player_ids: np.ndarray,
    player_pos_group: Dict[int, str],
    device: torch.device,
) -> torch.Tensor:
    emb = model.player_embedding.weight.detach().cpu()
    buckets: Dict[str, List[torch.Tensor]] = {pos: [] for pos in POSITION_GROUPS}
    for remapped in remapped_player_ids.tolist():
        group = player_pos_group.get(int(remapped), "UNK")
        buckets[group].append(emb[remapped])
    out = []
    for pos in POSITION_GROUPS:
        if buckets[pos]:
            out.append(torch.stack(buckets[pos]).mean(dim=0))
        else:
            out.append(torch.zeros_like(emb[0]))
    return torch.stack(out).to(device)


def generate_shapley_targets(
    model: StateValueModel,
    bundle: ArrayBundle,
    candidate_indices: np.ndarray,
    permutations: int,
    max_states: int,
    device: torch.device,
    cache_path: Path,
) -> pd.DataFrame:
    if cache_path.exists():
        LOGGER.info("Loading cached Shapley targets from %s", cache_path)
        return pd.read_parquet(cache_path)

    rng = np.random.default_rng(7)
    sample_indices = candidate_indices.copy()
    if len(sample_indices) > max_states:
        rng.shuffle(sample_indices)
        sample_indices = np.sort(sample_indices[:max_states])

    remapped_player_ids = np.unique(bundle.player_ids[bundle.player_ids > 0])
    replacement_embeddings = compute_replacement_embeddings(
        model=model,
        remapped_player_ids=remapped_player_ids,
        player_pos_group=bundle.player_pos_group,
        device=device,
    )

    rows = []
    model.eval()
    with torch.no_grad():
        for index in sample_indices.tolist():
            numeric = torch.as_tensor(bundle.numeric_scaled[index : index + 1], dtype=torch.float32, device=device)
            player_ids = torch.as_tensor(bundle.player_ids[index : index + 1], dtype=torch.long, device=device)
            side_ids = torch.as_tensor(bundle.side_ids[index : index + 1], dtype=torch.long, device=device)
            pos_ids = torch.as_tensor(bundle.pos_ids[index : index + 1], dtype=torch.long, device=device)
            perms = np.vstack([rng.permutation(10) for _ in range(permutations)])
            shap = np.zeros(10, dtype=np.float32)

            mask = torch.zeros((permutations, 10), dtype=torch.bool, device=device)
            numeric_rep = numeric.repeat(permutations, 1)
            player_rep = player_ids.repeat(permutations, 1)
            side_rep = side_ids.repeat(permutations, 1)
            pos_rep = pos_ids.repeat(permutations, 1)
            prev_value = None
            for step in range(10):
                step_players = perms[:, step]
                mask[torch.arange(permutations), torch.as_tensor(step_players, device=device)] = True
                logits = model(
                    numeric_rep,
                    player_rep,
                    side_rep,
                    pos_rep,
                    replacement_mask=mask,
                    replacement_embed_by_pos=replacement_embeddings,
                )
                probs = sanitize_probability_tensor(F.softmax(logits, dim=1))
                values = expected_remaining_margin(probs).cpu().numpy()
                if prev_value is None:
                    marginal = values
                else:
                    marginal = values - prev_value
                prev_value = values
                for perm_idx in range(permutations):
                    shap[perms[perm_idx, step]] += marginal[perm_idx]

            shap /= max(permutations, 1)
            rows.append({"global_index": index, **{f"phi_{i}": float(shap[i]) for i in range(10)}})
    shapley_df = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    shapley_df.to_parquet(cache_path, index=False)
    return shapley_df


def train_attributor(
    bundle: ArrayBundle,
    model: StateValueModel,
    shapley_targets: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
) -> ShapleyAttributor:
    train_indices = shapley_targets["global_index"].to_numpy(np.int64)
    target_matrix = shapley_targets[[f"phi_{i}" for i in range(10)]].astype(np.float32).to_numpy()
    num_players = int(bundle.player_ids.max())
    attributor = ShapleyAttributor(
        num_players=num_players,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_heads=args.num_attn_heads,
    ).to(device)
    with torch.no_grad():
        attributor.player_embedding.weight.copy_(model.player_embedding.weight)
    optimizer = torch.optim.AdamW(attributor.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed + 1)

    for epoch in range(1, args.attributor_epochs + 1):
        attributor.train()
        losses: List[float] = []
        for batch in batch_indices(train_indices, args.batch_size, shuffle=True, rng=rng):
            batch_pos = np.searchsorted(train_indices, batch)
            numeric = torch.as_tensor(bundle.numeric_scaled[batch], dtype=torch.float32, device=device)
            player_ids = torch.as_tensor(bundle.player_ids[batch], dtype=torch.long, device=device)
            side_ids = torch.as_tensor(bundle.side_ids[batch], dtype=torch.long, device=device)
            pos_ids = torch.as_tensor(bundle.pos_ids[batch], dtype=torch.long, device=device)
            target_phi = torch.as_tensor(target_matrix[batch_pos], dtype=torch.float32, device=device)
            with torch.no_grad():
                probs = sanitize_probability_tensor(F.softmax(model(numeric, player_ids, side_ids, pos_ids), dim=1))
                target_value = expected_remaining_margin(probs)
            pred_phi = attributor(numeric, player_ids, side_ids, pos_ids, target_value)
            mse = F.mse_loss(pred_phi, target_phi)
            eff = (pred_phi.sum(dim=1) - target_value).pow(2).mean()
            null_reg = pred_phi.abs().mean()
            loss = mse + 5.0 * eff + 0.01 * null_reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        LOGGER.info("Attributor epoch %s | loss=%.4f", epoch, float(np.mean(losses) if losses else 0.0))
    return attributor


def predict_attributions(
    attributor: ShapleyAttributor,
    model: StateValueModel,
    bundle: ArrayBundle,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    preds: List[np.ndarray] = []
    attributor.eval()
    model.eval()
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for batch in batch_indices(indices, batch_size, shuffle=False, rng=rng):
            numeric = torch.as_tensor(bundle.numeric_scaled[batch], dtype=torch.float32, device=device)
            player_ids = torch.as_tensor(bundle.player_ids[batch], dtype=torch.long, device=device)
            side_ids = torch.as_tensor(bundle.side_ids[batch], dtype=torch.long, device=device)
            pos_ids = torch.as_tensor(bundle.pos_ids[batch], dtype=torch.long, device=device)
            probs = sanitize_probability_tensor(F.softmax(model(numeric, player_ids, side_ids, pos_ids), dim=1))
            target_value = expected_remaining_margin(probs)
            phi = attributor(numeric, player_ids, side_ids, pos_ids, target_value)
            preds.append(phi.cpu().numpy())
    return np.vstack(preds)


def build_offensive_allocator_targets(
    bundle: ArrayBundle,
    output_indices: np.ndarray,
    phi_current: np.ndarray,
    phi_next: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidate_ids = np.zeros((len(output_indices), 3), dtype=np.int64)
    candidate_side = np.zeros_like(candidate_ids)
    candidate_pos = np.zeros_like(candidate_ids)
    candidate_mask = np.zeros_like(candidate_ids, dtype=bool)
    targets = np.zeros_like(candidate_ids, dtype=np.float32)
    keep_rows = np.zeros(len(output_indices), dtype=bool)

    for out_row, global_idx in enumerate(output_indices.tolist()):
        action_name = ACTION_CLASSES[int(bundle.action_id[global_idx])]
        if action_name not in OFFENSIVE_ACTIONS:
            continue
        lineup = bundle.player_ids[global_idx]
        sides = bundle.side_ids[global_idx]
        positions = bundle.pos_ids[global_idx]
        candidates = [int(bundle.primary_player_id[global_idx]), int(bundle.secondary_player_id[global_idx]), int(bundle.tertiary_player_id[global_idx])]
        unique_candidates: List[int] = []
        for pid in candidates:
            if pid > 0 and pid in lineup.tolist() and pid not in unique_candidates:
                token_pos = lineup.tolist().index(pid)
                if sides[token_pos] == 1:
                    unique_candidates.append(pid)
        if not unique_candidates:
            continue
        keep_rows[out_row] = True
        delta_phi = phi_next[out_row] - phi_current[out_row]
        score = []
        for slot, pid in enumerate(unique_candidates[:3]):
            token_pos = lineup.tolist().index(pid)
            candidate_ids[out_row, slot] = pid
            candidate_side[out_row, slot] = sides[token_pos]
            candidate_pos[out_row, slot] = positions[token_pos]
            candidate_mask[out_row, slot] = True
            score.append(delta_phi[token_pos])
        score_arr = np.asarray(score, dtype=np.float32)
        if np.allclose(score_arr, 0.0):
            score_arr = np.ones_like(score_arr) / len(score_arr)
        else:
            score_arr = np.exp(score_arr - score_arr.max())
            score_arr = score_arr / score_arr.sum()
        targets[out_row, : len(score_arr)] = score_arr

    return candidate_ids, candidate_side, candidate_pos, candidate_mask, targets, keep_rows


def train_offensive_allocator(
    bundle: ArrayBundle,
    output_indices: np.ndarray,
    phi_current: np.ndarray,
    phi_next: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> OffensiveAllocator:
    candidate_ids, candidate_side, candidate_pos, candidate_mask, targets, keep_rows = build_offensive_allocator_targets(
        bundle=bundle,
        output_indices=output_indices,
        phi_current=phi_current,
        phi_next=phi_next,
    )
    train_rows = np.flatnonzero(keep_rows)
    allocator = OffensiveAllocator(int(bundle.player_ids.max()), args.embedding_dim, args.hidden_size).to(device)
    optimizer = torch.optim.AdamW(allocator.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed + 2)

    if len(train_rows) == 0:
        return allocator

    for epoch in range(1, args.allocator_epochs + 1):
        allocator.train()
        losses: List[float] = []
        for batch in batch_indices(train_rows, args.batch_size, shuffle=True, rng=rng):
            global_batch = output_indices[batch]
            numeric = torch.as_tensor(bundle.numeric_scaled[global_batch], dtype=torch.float32, device=device)
            action_ids = torch.as_tensor(bundle.action_id[global_batch], dtype=torch.long, device=device)
            cand_ids = torch.as_tensor(candidate_ids[batch], dtype=torch.long, device=device)
            cand_side = torch.as_tensor(candidate_side[batch], dtype=torch.long, device=device)
            cand_pos = torch.as_tensor(candidate_pos[batch], dtype=torch.long, device=device)
            mask = torch.as_tensor(candidate_mask[batch], dtype=torch.bool, device=device)
            target = torch.as_tensor(targets[batch], dtype=torch.float32, device=device)
            pred = allocator(numeric, cand_ids, cand_side, cand_pos, action_ids, mask)
            loss = -(target * pred.clamp_min(1e-8).log()).sum(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        LOGGER.info("Allocator epoch %s | loss=%.4f", epoch, float(np.mean(losses) if losses else 0.0))
    return allocator


def compute_delta_v(
    bundle: ArrayBundle,
    output_indices: np.ndarray,
    remaining_current: np.ndarray,
    remaining_next: np.ndarray,
) -> np.ndarray:
    return bundle.rewards[output_indices] + bundle.gamma[output_indices] * remaining_next - remaining_current


def stabilize_delta_v(delta_v: np.ndarray, entropy: np.ndarray, action_ids: np.ndarray) -> np.ndarray:
    entropy_scale = 1.0 / (1.0 + entropy / math.log(len(SUPPORT)))
    raw = delta_v * entropy_scale
    out = raw.copy()
    for action_id in np.unique(action_ids):
        mask = action_ids == action_id
        std = np.std(raw[mask])
        if std > 1e-6:
            out[mask] = raw[mask] / std * np.median(np.abs(raw))
    return out


def allocate_action_credits(
    bundle: ArrayBundle,
    output_indices: np.ndarray,
    phi_current: np.ndarray,
    phi_next: np.ndarray,
    delta_v_raw: np.ndarray,
    delta_v_stabilized: np.ndarray,
    allocator: OffensiveAllocator,
    args: argparse.Namespace,
    device: torch.device,
) -> pd.DataFrame:
    rows: List[dict] = []
    allocator.eval()

    candidate_ids, candidate_side, candidate_pos, candidate_mask, _, keep_rows = build_offensive_allocator_targets(
        bundle=bundle,
        output_indices=output_indices,
        phi_current=phi_current,
        phi_next=phi_next,
    )

    pred_weights_map: Dict[int, np.ndarray] = {}
    with torch.no_grad():
        for batch in batch_indices(np.flatnonzero(keep_rows), args.batch_size, shuffle=False, rng=np.random.default_rng(0)):
            global_batch = output_indices[batch]
            numeric = torch.as_tensor(bundle.numeric_scaled[global_batch], dtype=torch.float32, device=device)
            action_ids = torch.as_tensor(bundle.action_id[global_batch], dtype=torch.long, device=device)
            cand_ids = torch.as_tensor(candidate_ids[batch], dtype=torch.long, device=device)
            cand_side = torch.as_tensor(candidate_side[batch], dtype=torch.long, device=device)
            cand_pos = torch.as_tensor(candidate_pos[batch], dtype=torch.long, device=device)
            mask = torch.as_tensor(candidate_mask[batch], dtype=torch.bool, device=device)
            weights = allocator(numeric, cand_ids, cand_side, cand_pos, action_ids, mask).cpu().numpy()
            for local_row, out_row in enumerate(batch.tolist()):
                pred_weights_map[out_row] = weights[local_row]

    for out_pos, global_idx in enumerate(output_indices.tolist()):
        lineup = bundle.player_ids[global_idx]
        raw_credit = np.zeros(10, dtype=np.float32)
        stab_credit = np.zeros(10, dtype=np.float32)
        action_name = ACTION_CLASSES[int(bundle.action_id[global_idx])]
        primary_pid = int(bundle.primary_player_id[global_idx])
        secondary_pid = int(bundle.secondary_player_id[global_idx])
        tertiary_pid = int(bundle.tertiary_player_id[global_idx])

        if action_name in OFFENSIVE_ACTIONS and out_pos in pred_weights_map:
            weights = pred_weights_map[out_pos]
            for slot in range(3):
                pid = candidate_ids[out_pos, slot]
                if pid <= 0:
                    continue
                token = lineup.tolist().index(pid)
                raw_credit[token] = delta_v_raw[out_pos] * weights[slot]
                stab_credit[token] = delta_v_stabilized[out_pos] * weights[slot]
        else:
            diffs = phi_next[out_pos] - phi_current[out_pos]
            total = diffs.sum()
            if abs(total) < 1e-6:
                active = np.ones(10, dtype=np.float32) / 10.0
            else:
                active = diffs / total
            raw_credit = delta_v_raw[out_pos] * active
            stab_credit = delta_v_stabilized[out_pos] * active

        for token_idx in range(10):
            pid = int(lineup[token_idx])
            if pid <= 0:
                continue
            team_id = int(bundle.team_id_home[global_idx] if token_idx < 5 else bundle.team_id_away[global_idx])
            court_side = "home" if token_idx < 5 else "away"
            phase_side = "offense" if int(bundle.side_ids[global_idx, token_idx]) == 1 else "defense"
            if pid == primary_pid:
                role = "primary"
            elif pid == secondary_pid:
                role = "secondary"
            elif pid == tertiary_pid:
                role = "tertiary"
            else:
                role = "lineup"
            rows.append(
                {
                    "global_index": global_idx,
                    "game_id": int(bundle.game_id[global_idx]),
                    "event_num": int(bundle.event_num[global_idx]),
                    "game_date_ord": int(bundle.game_date_ord[global_idx]),
                    "player_id": pid,
                    "player_name": bundle.player_name_map.get(pid, str(pid)),
                    "team_id": team_id,
                    "pos_group": bundle.player_pos_group.get(pid, "UNK"),
                    "token_idx": token_idx,
                    "court_side": court_side,
                    "phase_side": phase_side,
                    "action_class": action_name,
                    "role": role,
                    "credit_raw": float(raw_credit[token_idx]),
                    "credit_stabilized": float(stab_credit[token_idx]),
                }
            )
    return pd.DataFrame(rows)


def build_state_value_table(
    bundle: ArrayBundle,
    output_indices: np.ndarray,
    remaining_current: np.ndarray,
    remaining_next: np.ndarray,
    delta_v_raw: np.ndarray,
    delta_v_stabilized: np.ndarray,
    entropy: np.ndarray,
    phi_current: np.ndarray,
    phi_next: np.ndarray,
) -> pd.DataFrame:
    meta = bundle.output_meta.set_index("global_index").loc[output_indices]
    next_indices = bundle.next_index[output_indices].copy()
    next_indices[next_indices < 0] = output_indices[next_indices < 0]
    phi_sum_current = phi_current.sum(axis=1)
    phi_sum_next = phi_next.sum(axis=1)
    return pd.DataFrame(
        {
            "global_index": output_indices,
            "game_id": bundle.game_id[output_indices],
            "event_num": bundle.event_num[output_indices],
            "game_date_ord": bundle.game_date_ord[output_indices],
            "season_end_year": bundle.season_end_year[output_indices],
            "action_class": [ACTION_CLASSES[idx] for idx in bundle.action_id[output_indices]],
            "score_bucket": meta["score_bucket"].to_numpy(),
            "time_bucket": meta["time_bucket"].to_numpy(),
            "reward_home": bundle.rewards[output_indices],
            "gamma": bundle.gamma[output_indices],
            "terminal": bundle.terminal[output_indices].astype(np.int64),
            "pre_margin_home": bundle.pre_margin_home[output_indices],
            "next_pre_margin_home": bundle.pre_margin_home[next_indices],
            "terminal_final_margin_home": bundle.terminal_final_margin_home[output_indices],
            "value_current": remaining_current,
            "value_next": remaining_next,
            "pred_final_margin_current": bundle.pre_margin_home[output_indices] + remaining_current,
            "pred_final_margin_next": bundle.pre_margin_home[next_indices] + remaining_next,
            "delta_v_raw": delta_v_raw,
            "delta_v_stabilized": delta_v_stabilized,
            "pred_entropy": entropy,
            "phi_sum_current": phi_sum_current,
            "phi_sum_next": phi_sum_next,
            "state_efficiency_gap": phi_sum_current - remaining_current,
            "next_state_efficiency_gap": phi_sum_next - remaining_next,
        }
    )


def attach_event_credit_sums(state_values: pd.DataFrame, credits: pd.DataFrame) -> pd.DataFrame:
    if credits.empty:
        out = state_values.copy()
        out["event_credit_raw_sum"] = 0.0
        out["event_credit_stabilized_sum"] = 0.0
    else:
        grouped = (
            credits.groupby("global_index", as_index=False)
            .agg(
                event_credit_raw_sum=("credit_raw", "sum"),
                event_credit_stabilized_sum=("credit_stabilized", "sum"),
            )
        )
        out = state_values.merge(grouped, on="global_index", how="left")
        out[["event_credit_raw_sum", "event_credit_stabilized_sum"]] = out[
            ["event_credit_raw_sum", "event_credit_stabilized_sum"]
        ].fillna(0.0)
    out["event_conservation_gap_raw"] = out["event_credit_raw_sum"] - out["delta_v_raw"]
    out["event_conservation_gap_stabilized"] = out["event_credit_stabilized_sum"] - out["delta_v_stabilized"]
    return out


def build_state_player_phi_table(
    bundle: ArrayBundle,
    output_indices: np.ndarray,
    phi_current: np.ndarray,
    phi_next: np.ndarray,
) -> pd.DataFrame:
    rows: List[dict] = []
    for row_pos, global_idx in enumerate(output_indices.tolist()):
        lineup = bundle.player_ids[global_idx]
        sides = bundle.side_ids[global_idx]
        for token_idx, pid in enumerate(lineup.tolist()):
            if pid <= 0:
                continue
            team_id = int(bundle.team_id_home[global_idx] if token_idx < 5 else bundle.team_id_away[global_idx])
            rows.append(
                {
                    "global_index": global_idx,
                    "game_id": int(bundle.game_id[global_idx]),
                    "event_num": int(bundle.event_num[global_idx]),
                    "game_date_ord": int(bundle.game_date_ord[global_idx]),
                    "player_id": pid,
                    "player_name": bundle.player_name_map.get(pid, str(pid)),
                    "team_id": team_id,
                    "pos_group": bundle.player_pos_group.get(pid, "UNK"),
                    "token_idx": token_idx,
                    "court_side": "home" if token_idx < 5 else "away",
                    "phase_side": "offense" if int(sides[token_idx]) == 1 else "defense",
                    # Sign-correct: away players' phi must be negated for player-centric view
                    "phi_current": (1.0 if token_idx < 5 else -1.0) * float(phi_current[row_pos, token_idx]),
                    "phi_next": (1.0 if token_idx < 5 else -1.0) * float(phi_next[row_pos, token_idx]),
                    "phi_delta": (1.0 if token_idx < 5 else -1.0) * float(phi_next[row_pos, token_idx] - phi_current[row_pos, token_idx]),
                }
            )
    return pd.DataFrame(rows)


def build_player_ratings(
    bundle: ArrayBundle,
    output_indices: np.ndarray,
    phi_current: np.ndarray,
    args: argparse.Namespace,
) -> pd.DataFrame:
    totals = defaultdict(
        lambda: {
            "state_value_total": 0.0,
            "state_value_off": 0.0,
            "state_value_def": 0.0,
            "states_total": 0,
            "states_off": 0,
            "states_def": 0,
        }
    )
    for row_pos, global_idx in enumerate(output_indices.tolist()):
        lineup = bundle.player_ids[global_idx]
        sides = bundle.side_ids[global_idx]
        for token_idx, pid in enumerate(lineup.tolist()):
            if pid <= 0:
                continue
            rec = totals[pid]
            # phi is home-centric (V(s) = expected home margin).
            # Away players who help their team get negative phi; negate to get player-centric value.
            sign = 1.0 if token_idx < 5 else -1.0
            val = sign * float(phi_current[row_pos, token_idx])
            rec["state_value_total"] += val
            rec["states_total"] += 1
            if sides[token_idx] == 1:
                rec["state_value_off"] += val
                rec["states_off"] += 1
            else:
                rec["state_value_def"] += val
                rec["states_def"] += 1

    rows = []
    for pid, rec in totals.items():
        total_per100 = 100.0 * rec["state_value_total"] / max(rec["states_total"], 1)
        off_per100 = 100.0 * rec["state_value_off"] / max(rec["states_off"], 1)
        def_per100 = 100.0 * rec["state_value_def"] / max(rec["states_def"], 1)
        shrink = rec["states_total"] / (rec["states_total"] + args.report_shrinkage_k)
        rows.append(
            {
                "player_id": pid,
                "player_name": bundle.player_name_map.get(pid, str(pid)),
                "pos_group": bundle.player_pos_group.get(pid, "UNK"),
                "team_id": bundle.player_team_map.get(pid),
                "state_value_total": rec["state_value_total"],
                "state_value_off": rec["state_value_off"],
                "state_value_def": rec["state_value_def"],
                "states_total": rec["states_total"],
                "states_off": rec["states_off"],
                "states_def": rec["states_def"],
                "state_total_per100": total_per100,
                "state_off_per100": off_per100,
                "state_def_per100": def_per100,
                "total_per100": total_per100,
                "off_per100": off_per100,
                "def_per100": def_per100,
                "shrunk_state_total_per100": total_per100 * shrink,
                "shrunk_state_off_per100": off_per100 * shrink,
                "shrunk_state_def_per100": def_per100 * shrink,
                "shrunk_total_per100": total_per100 * shrink,
                "shrunk_off_per100": off_per100 * shrink,
                "shrunk_def_per100": def_per100 * shrink,
            }
        )
    ratings = pd.DataFrame(rows).sort_values("shrunk_total_per100", ascending=False).reset_index(drop=True)
    return ratings


def build_action_outputs(
    bundle: ArrayBundle,
    output_indices: np.ndarray,
    delta_v_raw: np.ndarray,
    delta_v_stabilized: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    meta = bundle.output_meta.copy()
    meta["delta_v_raw"] = delta_v_raw
    meta["delta_v_stabilized"] = delta_v_stabilized
    by_class = (
        meta.groupby("action_class", as_index=False)
        .agg(
            events=("action_class", "size"),
            mean_raw=("delta_v_raw", "mean"),
            std_raw=("delta_v_raw", "std"),
            mean_stabilized=("delta_v_stabilized", "mean"),
            std_stabilized=("delta_v_stabilized", "std"),
        )
        .sort_values("mean_stabilized", ascending=False)
    )
    grid = (
        meta.groupby(["action_class", "score_bucket", "time_bucket"], as_index=False)
        .agg(
            events=("action_class", "size"),
            mean_raw=("delta_v_raw", "mean"),
            mean_stabilized=("delta_v_stabilized", "mean"),
        )
        .sort_values(["action_class", "score_bucket", "time_bucket"])
    )
    return by_class, grid


def build_player_value_decomposition(
    bundle: ArrayBundle,
    ratings: pd.DataFrame,
    credits: pd.DataFrame,
) -> pd.DataFrame:
    base_cols = [
        "player_id",
        "player_name",
        "team_id",
        "pos_group",
        "states_total",
        "states_off",
        "states_def",
        "state_value_total",
        "state_value_off",
        "state_value_def",
        "state_total_per100",
        "state_off_per100",
        "state_def_per100",
        "shrunk_state_total_per100",
        "shrunk_state_off_per100",
        "shrunk_state_def_per100",
        "total_per100",
        "off_per100",
        "def_per100",
        "shrunk_total_per100",
        "shrunk_off_per100",
        "shrunk_def_per100",
    ]
    out = ratings[base_cols].copy()

    if credits.empty:
        out["event_rows"] = 0
        out["event_credit_raw_total"] = 0.0
        out["event_credit_stabilized_total"] = 0.0
        out["scoring"] = 0.0
        out["playmaking"] = 0.0
        out["offensive_rebounding"] = 0.0
        out["defensive_actions"] = 0.0
        out["turnovers"] = 0.0
        out["defensive_presence"] = 0.0
        out["bucket_total_stabilized"] = 0.0
        out["bucket_gap_stabilized"] = 0.0
        return out.sort_values("shrunk_total_per100", ascending=False)

    categorized = categorize_credit_rows(credits)
    # Sign-correct: away players' credits are home-centric; negate for player-centric aggregation
    if "court_side" in categorized.columns:
        away_mask = categorized["court_side"] == "away"
        categorized.loc[away_mask, "credit_raw"] *= -1
        categorized.loc[away_mask, "credit_stabilized"] *= -1
    event_totals = (
        categorized.groupby("player_id", as_index=False)
        .agg(
            event_rows=("global_index", "size"),
            event_credit_raw_total=("credit_raw", "sum"),
            event_credit_stabilized_total=("credit_stabilized", "sum"),
        )
    )
    grouped = categorized.groupby(["player_id", "category"], as_index=False)["credit_stabilized"].sum()
    pivot = grouped.pivot(index="player_id", columns="category", values="credit_stabilized").fillna(0.0).reset_index()
    out = out.merge(event_totals, on="player_id", how="left").merge(pivot, on="player_id", how="left").fillna(0.0)
    bucket_cols = ["scoring", "playmaking", "offensive_rebounding", "defensive_actions", "turnovers", "defensive_presence"]
    for col in bucket_cols:
        if col not in out.columns:
            out[col] = 0.0
    out["bucket_total_stabilized"] = out[bucket_cols].sum(axis=1)
    out["bucket_gap_stabilized"] = out["event_credit_stabilized_total"] - out["bucket_total_stabilized"]
    return out.sort_values("shrunk_total_per100", ascending=False)


def categorize_credit_rows(credits: pd.DataFrame) -> pd.DataFrame:
    if credits.empty:
        out = credits.copy()
        out["category"] = pd.Series(dtype=object)
        return out
    out = credits.copy()
    # Players directly involved in discrete actions get named buckets;
    # lineup bystanders get "defensive_presence" (off-ball / diffuse influence).
    out["category"] = np.where(
        out["role"] == "lineup", "defensive_presence", "defensive_actions"
    )
    scoring_mask = out["action_class"].isin(SCORING_ACTIONS)
    out.loc[scoring_mask & out["role"].eq("primary"), "category"] = "scoring"
    out.loc[scoring_mask & out["role"].isin({"secondary", "tertiary"}), "category"] = "playmaking"
    out.loc[out["action_class"].eq("off_rebound"), "category"] = "offensive_rebounding"
    out.loc[out["action_class"].isin({"turnover_bad_pass", "turnover_other"}), "category"] = "turnovers"
    return out


def build_player_bucket_totals(bundle: ArrayBundle, credits: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "player_id",
        "player_name",
        "team_id",
        "pos_group",
        "category",
        "event_rows",
        "credit_raw_total",
        "credit_stabilized_total",
    ]
    if credits.empty:
        return pd.DataFrame(columns=columns)
    categorized = categorize_credit_rows(credits)
    # Sign-correct: away players' credits are home-centric; negate for player-centric aggregation
    if "court_side" in categorized.columns:
        away_mask = categorized["court_side"] == "away"
        categorized.loc[away_mask, "credit_raw"] *= -1
        categorized.loc[away_mask, "credit_stabilized"] *= -1
    grouped = (
        categorized.groupby(["player_id", "team_id", "pos_group", "category"], as_index=False)
        .agg(
            event_rows=("global_index", "size"),
            credit_raw_total=("credit_raw", "sum"),
            credit_stabilized_total=("credit_stabilized", "sum"),
        )
    )
    grouped["player_name"] = grouped["player_id"].map(bundle.player_name_map).fillna(grouped["player_id"].astype(str))
    return grouped[columns].sort_values(["player_id", "category"]).reset_index(drop=True)


def build_player_totals(ratings: pd.DataFrame, decomposition: pd.DataFrame) -> pd.DataFrame:
    out = ratings[
        [
            "player_id",
            "player_name",
            "team_id",
            "pos_group",
            "states_total",
            "states_off",
            "states_def",
            "state_value_total",
            "state_value_off",
            "state_value_def",
            "state_total_per100",
            "state_off_per100",
            "state_def_per100",
            "shrunk_state_total_per100",
            "shrunk_state_off_per100",
            "shrunk_state_def_per100",
        ]
    ].copy()
    if decomposition.empty:
        out["event_rows"] = 0
        out["event_credit_raw_total"] = 0.0
        out["event_credit_stabilized_total"] = 0.0
        out["bucket_total_stabilized"] = 0.0
        out["bucket_gap_stabilized"] = 0.0
        return out.sort_values("shrunk_state_total_per100", ascending=False).reset_index(drop=True)
    out = out.merge(
        decomposition[
            [
                "player_id",
                "event_rows",
                "event_credit_raw_total",
                "event_credit_stabilized_total",
                "bucket_total_stabilized",
                "bucket_gap_stabilized",
            ]
        ],
        on="player_id",
        how="left",
    ).fillna(0.0)
    return out.sort_values("shrunk_state_total_per100", ascending=False).reset_index(drop=True)


def build_reconciliation_report(state_values: pd.DataFrame, player_totals: pd.DataFrame) -> dict:
    def summarize_abs(series: pd.Series, tolerance: float) -> dict:
        if series.empty:
            return {
                "count": 0,
                "max_abs": 0.0,
                "mean_abs": 0.0,
                "p95_abs": 0.0,
                "within_tolerance_rate": 1.0,
            }
        abs_values = series.abs().to_numpy(dtype=np.float64)
        return {
            "count": int(abs_values.size),
            "max_abs": float(abs_values.max(initial=0.0)),
            "mean_abs": float(abs_values.mean()),
            "p95_abs": float(np.quantile(abs_values, 0.95)),
            "within_tolerance_rate": float(np.mean(abs_values <= tolerance)),
        }

    tolerance_state = 1e-4
    tolerance_event = 1e-5
    tolerance_player = 1e-6

    state_current = summarize_abs(state_values["state_efficiency_gap"], tolerance_state)
    state_next = summarize_abs(state_values["next_state_efficiency_gap"], tolerance_state)
    event_raw = summarize_abs(state_values["event_conservation_gap_raw"], tolerance_event)
    event_stabilized = summarize_abs(state_values["event_conservation_gap_stabilized"], tolerance_event)
    player_bucket = summarize_abs(player_totals["bucket_gap_stabilized"], tolerance_player)

    return {
        "tolerances": {
            "state_efficiency": tolerance_state,
            "event_conservation": tolerance_event,
            "player_bucket": tolerance_player,
        },
        "state_current_efficiency": state_current,
        "state_next_efficiency": state_next,
        "event_conservation_raw": event_raw,
        "event_conservation_stabilized": event_stabilized,
        "player_bucket_reconciliation": player_bucket,
        "all_contracts_pass": bool(
            state_current["max_abs"] <= tolerance_state
            and state_next["max_abs"] <= tolerance_state
            and event_raw["max_abs"] <= tolerance_event
            and event_stabilized["max_abs"] <= tolerance_event
            and player_bucket["max_abs"] <= tolerance_player
        ),
    }


def benjamini_hochberg(p_values: np.ndarray, q: float = 0.05) -> np.ndarray:
    order = np.argsort(p_values)
    ranked = p_values[order]
    n = len(ranked)
    thresholds = q * (np.arange(1, n + 1) / n)
    passed = ranked <= thresholds
    if not passed.any():
        return np.zeros(n, dtype=bool)
    cutoff = np.max(np.where(passed)[0])
    sig = np.zeros(n, dtype=bool)
    sig[order[: cutoff + 1]] = True
    return sig


def permutation_test_three_group(
    together: np.ndarray,
    apart_i: np.ndarray,
    apart_j: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> float:
    observed = together.mean() - apart_i.mean() - apart_j.mean()
    pooled = np.concatenate([together, apart_i, apart_j])
    n1, n2 = len(together), len(apart_i)
    exceed = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled.size)
        g1 = pooled[perm[:n1]]
        g2 = pooled[perm[n1 : n1 + n2]]
        g3 = pooled[perm[n1 + n2 :]]
        stat = g1.mean() - g2.mean() - g3.mean()
        if abs(stat) >= abs(observed):
            exceed += 1
    return (exceed + 1) / (n_perm + 1)


def build_synergy_outputs(
    bundle: ArrayBundle,
    output_indices: np.ndarray,
    phi_current: np.ndarray,
    ratings: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    player_state_values = defaultdict(lambda: {"total": {}, "offense": {}, "defense": {}})
    pair_state_values = defaultdict(lambda: {"total": {}, "offense": {}, "defense": {}})
    lineup_counts = Counter()
    team_wins = Counter()

    seen_games = set()
    for row_pos, global_idx in enumerate(output_indices.tolist()):
        lineup = bundle.player_ids[global_idx]
        sides = bundle.side_ids[global_idx]
        values = phi_current[row_pos]
        home_players = lineup[:5].tolist()
        away_players = lineup[5:].tolist()
        home_team_id = int(bundle.team_id_home[global_idx])
        away_team_id = int(bundle.team_id_away[global_idx])

        lineup_counts[(home_team_id, tuple(sorted(pid for pid in home_players if pid > 0)))] += 1
        lineup_counts[(away_team_id, tuple(sorted(pid for pid in away_players if pid > 0)))] += 1

        game_key = int(bundle.game_id[global_idx])
        if game_key not in seen_games:
            seen_games.add(game_key)
            final_margin = float(bundle.terminal_final_margin_home[global_idx])
            if home_team_id > 0:
                team_wins[home_team_id] += int(final_margin > 0)
            if away_team_id > 0:
                team_wins[away_team_id] += int(final_margin < 0)

        for team_players, team_id in ((home_players, home_team_id), (away_players, away_team_id)):
            active = [pid for pid in team_players if pid > 0]
            if not active:
                continue
            token_positions = {pid: lineup.tolist().index(pid) for pid in active}
            team_is_offense = int(sides[token_positions[active[0]]]) == 1
            side_key = "offense" if team_is_offense else "defense"
            for pid in active:
                pos = token_positions[pid]
                player_rec = player_state_values[(pid, team_id)]
                player_rec["total"][global_idx] = float(values[pos])
                player_rec[side_key][global_idx] = float(values[pos])
            for i in range(len(active)):
                for j in range(i + 1, len(active)):
                    pid_i, pid_j = active[i], active[j]
                    pair = (min(pid_i, pid_j), max(pid_i, pid_j), team_id)
                    pos_i = token_positions[pid_i]
                    pos_j = token_positions[pid_j]
                    pair_val = float(values[pos_i] + values[pos_j])
                    pair_rec = pair_state_values[pair]
                    pair_rec["total"][global_idx] = pair_val
                    pair_rec[side_key][global_idx] = pair_val

    pair_rows = []
    rng = np.random.default_rng(args.seed + 3)
    for (pid_i, pid_j, team_id), rec in pair_state_values.items():
        player_i_states = player_state_values[(pid_i, team_id)]
        player_j_states = player_state_values[(pid_j, team_id)]

        together_total_rows = set(rec["total"].keys())
        together_total = np.asarray(list(rec["total"].values()), dtype=np.float32)
        apart_i_total = np.asarray(
            [value for row_id, value in player_i_states["total"].items() if row_id not in together_total_rows],
            dtype=np.float32,
        )
        apart_j_total = np.asarray(
            [value for row_id, value in player_j_states["total"].items() if row_id not in together_total_rows],
            dtype=np.float32,
        )

        together_count = int(together_total.size)
        apart_i_count = int(apart_i_total.size)
        apart_j_count = int(apart_j_total.size)
        if together_count < args.min_pair_possessions or apart_i_count < args.min_pair_possessions or apart_j_count < args.min_pair_possessions:
            continue

        together_mean = float(together_total.mean())
        apart_i_mean = float(apart_i_total.mean())
        apart_j_mean = float(apart_j_total.mean())
        synergy_total = together_mean - apart_i_mean - apart_j_mean

        together_off_rows = set(rec["offense"].keys())
        together_off = np.asarray(list(rec["offense"].values()), dtype=np.float32)
        apart_i_off = np.asarray(
            [value for row_id, value in player_i_states["offense"].items() if row_id not in together_off_rows],
            dtype=np.float32,
        )
        apart_j_off = np.asarray(
            [value for row_id, value in player_j_states["offense"].items() if row_id not in together_off_rows],
            dtype=np.float32,
        )
        synergy_off = (
            float(together_off.mean()) - float(apart_i_off.mean()) - float(apart_j_off.mean())
            if together_off.size > 0 and apart_i_off.size > 0 and apart_j_off.size > 0
            else 0.0
        )

        together_def_rows = set(rec["defense"].keys())
        together_def = np.asarray(list(rec["defense"].values()), dtype=np.float32)
        apart_i_def = np.asarray(
            [value for row_id, value in player_i_states["defense"].items() if row_id not in together_def_rows],
            dtype=np.float32,
        )
        apart_j_def = np.asarray(
            [value for row_id, value in player_j_states["defense"].items() if row_id not in together_def_rows],
            dtype=np.float32,
        )
        synergy_def = (
            float(together_def.mean()) - float(apart_i_def.mean()) - float(apart_j_def.mean())
            if together_def.size > 0 and apart_i_def.size > 0 and apart_j_def.size > 0
            else 0.0
        )

        p_value = permutation_test_three_group(
            together=together_total,
            apart_i=apart_i_total,
            apart_j=apart_j_total,
            n_perm=args.permutation_tests,
            rng=rng,
        )
        pair_rows.append(
            {
                "team_id": team_id,
                "player_i": pid_i,
                "player_j": pid_j,
                "player_i_name": bundle.player_name_map.get(pid_i, str(pid_i)),
                "player_j_name": bundle.player_name_map.get(pid_j, str(pid_j)),
                "possessions_together": together_count,
                "possessions_apart_i": apart_i_count,
                "possessions_apart_j": apart_j_count,
                "synergy_total": synergy_total,
                "synergy_offense": synergy_off,
                "synergy_defense": synergy_def,
                "p_value": p_value,
            }
        )

    pair_df = pd.DataFrame(pair_rows)
    if not pair_df.empty:
        sig = benjamini_hochberg(pair_df["p_value"].to_numpy(), q=0.05)
        pair_df["fdr_significant"] = sig
        pair_df = pair_df.sort_values("synergy_total", ascending=False).reset_index(drop=True)
    else:
        pair_df = pd.DataFrame(
            columns=[
                "team_id",
                "player_i",
                "player_j",
                "player_i_name",
                "player_j_name",
                "possessions_together",
                "possessions_apart_i",
                "possessions_apart_j",
                "synergy_total",
                "synergy_offense",
                "synergy_defense",
                "p_value",
                "fdr_significant",
            ]
        )

    team_rows = []
    for team_id, win_count in team_wins.items():
        lineup_items = [(lineup, count) for (tid, lineup), count in lineup_counts.items() if tid == team_id]
        lineup_items.sort(key=lambda item: item[1], reverse=True)
        top_lineups = lineup_items[:5]
        synergy_sum = 0.0
        weight_sum = 0.0
        for lineup, count in top_lineups:
            lineup_synergy = 0.0
            for i in range(len(lineup)):
                for j in range(i + 1, len(lineup)):
                    mask = (
                        (pair_df["team_id"] == team_id)
                        & (pair_df["player_i"] == min(lineup[i], lineup[j]))
                        & (pair_df["player_j"] == max(lineup[i], lineup[j]))
                    )
                    if mask.any():
                        lineup_synergy += float(pair_df.loc[mask, "synergy_total"].iloc[0])
            synergy_sum += lineup_synergy * count
            weight_sum += count
        avg_synergy = synergy_sum / max(weight_sum, 1.0)
        team_ratings = [
            row["shrunk_total_per100"]
            for _, row in ratings.iterrows()
            if row.get("team_id") == team_id
        ]
        projected_team_rating = float(np.mean(team_ratings)) if team_ratings else 0.0
        projected_wins = 41.0 + 2.7 * projected_team_rating / 5.0
        team_rows.append(
            {
                "team_id": team_id,
                "actual_wins": win_count,
                "projected_wins_from_individual_ratings": projected_wins,
                "top5_lineup_synergy": avg_synergy,
            }
        )
    team_df = pd.DataFrame(team_rows).sort_values("actual_wins", ascending=False)
    return pair_df, team_df


def run_sanity_checks(
    bundle: ArrayBundle,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    probs_valid: np.ndarray,
    remaining_valid: np.ndarray,
) -> Dict[str, float]:
    checks: Dict[str, float] = {}
    checks["next_state_same_game_rate"] = float(
        np.mean(
            np.where(
                bundle.next_index[valid_idx] >= 0,
                bundle.game_id[valid_idx] == bundle.game_id[bundle.next_index[valid_idx]],
                True,
            )
        )
    )
    checks["player_lineup_valid_rate"] = float(np.mean((bundle.player_ids[:, :5] > 0).all(axis=1) & (bundle.player_ids[:, 5:] > 0).all(axis=1)))
    checks["no_validation_leak"] = float(np.intersect1d(train_idx, valid_idx).size == 0)

    pre_margin = bundle.pre_margin_home[valid_idx]
    secs = bundle.numeric[valid_idx, NUMERIC_FEATURES.index("seconds_remaining_game")]
    final_margin_pred = pre_margin + remaining_valid
    win_probs = margin_to_win_prob(torch.as_tensor(probs_valid, dtype=torch.float32), torch.as_tensor(pre_margin, dtype=torch.float32)).numpy()

    pos_slice = (pre_margin >= 5) & (secs <= 180)
    neg_slice = (pre_margin <= -5) & (secs <= 180)
    checks["monotonic_positive_clutch_home_win_prob"] = float(np.mean(win_probs[pos_slice])) if pos_slice.any() else 0.0
    checks["monotonic_negative_clutch_home_win_prob"] = float(np.mean(win_probs[neg_slice])) if neg_slice.any() else 0.0
    checks["pred_margin_rmse"] = float(math.sqrt(mean_squared_error(bundle.terminal_final_margin_home[valid_idx], final_margin_pred)))
    return checks


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, default=float)


def model_state_fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode("utf-8"))
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()[:12]


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = determine_device(args.device)
    torch.set_float32_matmul_precision("high")

    output_dir = RESULTS_ROOT / f"{args.output_year:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    player_meta = fetch_player_metadata(args.train_start, args.train_end)
    player_name_map, player_team_map, player_pos_group, _pos_weights_map, aliases = build_player_maps(player_meta)
    transition_df = load_or_build_dataset(args, aliases)
    bundle = materialize_arrays(
        transition_df,
        output_year=args.output_year,
        player_name_map=player_name_map,
        player_team_map=player_team_map,
        player_pos_group=player_pos_group,
    )
    lookup, vocab = build_player_lookup(bundle)
    original_lookup = {pid: lookup[pid] for pid in vocab.tolist()}
    remap_player_arrays(bundle, lookup)
    bundle.player_vocab = vocab
    bundle.player_name_map = {lookup.get(pid, 0): name for pid, name in player_name_map.items() if pid in lookup}
    bundle.player_team_map = {lookup.get(pid, 0): team for pid, team in player_team_map.items() if pid in lookup}
    bundle.player_pos_group = {lookup.get(pid, 0): pos for pid, pos in player_pos_group.items() if pid in lookup}

    train_idx, valid_idx = forward_chaining_split(bundle, output_year=args.output_year, validation_frac=args.validation_frac)
    scale_numeric_features(bundle, train_idx)
    save_json(
        output_dir / "summary.json",
        {
            "config": vars(args),
            "n_transitions": int(len(bundle.numeric)),
            "n_games": int(np.unique(bundle.game_id).size),
            "n_players": int(bundle.player_ids.max()),
            "train_rows": int(len(train_idx)),
            "valid_rows": int(len(valid_idx)),
            "device": str(device),
        },
    )

    if args.skip_baselines:
        baseline_logistic = {}
        baseline_tree = {}
        baseline_eval = pd.DataFrame()
    else:
        baseline_logistic, baseline_tree, baseline_eval = evaluate_baselines(bundle, train_idx, valid_idx)
        save_json(output_dir / "baseline_logistic_metrics.json", baseline_logistic)
        save_json(output_dir / "baseline_tree_metrics.json", baseline_tree)

    value_model, value_metrics = train_value_model(bundle, train_idx, valid_idx, args, device)
    torch.save(value_model.state_dict(), output_dir / "value_model.pt")
    probs_valid, remaining_valid, entropy_valid = predict_value_distribution(value_model, bundle, valid_idx, device)
    sanity = run_sanity_checks(bundle, train_idx, valid_idx, probs_valid, remaining_valid)
    value_metrics.update(sanity)
    save_json(output_dir / "value_model_metrics.json", value_metrics)

    output_indices = bundle.output_meta["global_index"].to_numpy(np.int64)
    probs_output, remaining_output, entropy_output = predict_value_distribution(value_model, bundle, output_indices, device)
    next_output = bundle.next_index[output_indices].copy()
    next_output[next_output < 0] = output_indices[next_output < 0]
    probs_next_output, remaining_next_output, _ = predict_value_distribution(value_model, bundle, next_output, device)
    remaining_next_output = np.where(bundle.terminal[output_indices], 0.0, remaining_next_output)

    games_tag = "all" if args.max_games_per_season <= 0 else f"games{args.max_games_per_season}"
    model_fingerprint = model_state_fingerprint(value_model)
    shapley_cache = output_dir / (
        f"shapley_targets_{args.train_start}_{args.train_end}_{games_tag}"
        f"_model{model_fingerprint}_states{args.max_shapley_states}_perm{args.shapley_permutations}.parquet"
    )
    shapley_targets = generate_shapley_targets(
        model=value_model,
        bundle=bundle,
        candidate_indices=train_idx,
        permutations=args.shapley_permutations,
        max_states=args.max_shapley_states,
        device=device,
        cache_path=shapley_cache,
    )
    attributor = train_attributor(bundle, value_model, shapley_targets, args, device)
    torch.save(attributor.state_dict(), output_dir / "attributor.pt")

    phi_output = predict_attributions(attributor, value_model, bundle, output_indices, device)
    phi_next_output = predict_attributions(attributor, value_model, bundle, next_output, device)
    phi_next_output = np.where(bundle.terminal[output_indices, None], 0.0, phi_next_output)
    delta_v_raw = compute_delta_v(bundle, output_indices, remaining_output, remaining_next_output)
    delta_v_stabilized = stabilize_delta_v(delta_v_raw, entropy_output, bundle.action_id[output_indices])
    state_values = build_state_value_table(
        bundle=bundle,
        output_indices=output_indices,
        remaining_current=remaining_output,
        remaining_next=remaining_next_output,
        delta_v_raw=delta_v_raw,
        delta_v_stabilized=delta_v_stabilized,
        entropy=entropy_output,
        phi_current=phi_output,
        phi_next=phi_next_output,
    )
    state_player_phi = build_state_player_phi_table(bundle, output_indices, phi_output, phi_next_output)

    allocator = train_offensive_allocator(bundle, output_indices, phi_output, phi_next_output, args, device)
    credits = allocate_action_credits(
        bundle=bundle,
        output_indices=output_indices,
        phi_current=phi_output,
        phi_next=phi_next_output,
        delta_v_raw=delta_v_raw,
        delta_v_stabilized=delta_v_stabilized,
        allocator=allocator,
        args=args,
        device=device,
    )
    state_values = attach_event_credit_sums(state_values, credits)
    state_values.to_parquet(output_dir / "state_values.parquet", index=False)
    state_player_phi.to_parquet(output_dir / "state_player_phi.parquet", index=False)
    credits.to_parquet(output_dir / "event_player_credit.parquet", index=False)

    ratings = build_player_ratings(bundle, output_indices, phi_output, args)
    ratings.to_csv(output_dir / "player_ratings.csv", index=False)
    ratings[ratings["states_total"] >= args.leaderboard_min_possessions].to_csv(
        output_dir / "player_ratings_min_possessions.csv",
        index=False,
    )

    action_values_by_class, action_grid = build_action_outputs(bundle, output_indices, delta_v_raw, delta_v_stabilized)
    action_values_by_class.to_csv(output_dir / "action_values_by_class.csv", index=False)
    action_grid.to_csv(output_dir / "action_values_context_grid.csv", index=False)

    player_bucket_totals = build_player_bucket_totals(bundle, credits)
    player_bucket_totals.to_csv(output_dir / "player_bucket_totals.csv", index=False)
    decomposition = build_player_value_decomposition(bundle, ratings, credits)
    decomposition.to_csv(output_dir / "player_value_decomposition.csv", index=False)
    player_totals = build_player_totals(ratings, decomposition)
    player_totals.to_csv(output_dir / "player_totals.csv", index=False)
    reconciliation = build_reconciliation_report(state_values, player_totals)
    save_json(output_dir / "reconciliation_report.json", reconciliation)

    pair_synergy, team_synergy = build_synergy_outputs(bundle, output_indices, phi_output, ratings, args)
    pair_synergy.to_csv(output_dir / "pair_synergy.csv", index=False)
    team_synergy.to_csv(output_dir / "team_synergy.csv", index=False)

    pred_final_margin_valid = bundle.pre_margin_home[valid_idx] + remaining_valid
    pred_home_win_valid = margin_to_win_prob(
        torch.as_tensor(probs_valid, dtype=torch.float32),
        torch.as_tensor(bundle.pre_margin_home[valid_idx], dtype=torch.float32),
    ).numpy()
    value_eval_df = pd.DataFrame(
        {
            "game_id": bundle.game_id[valid_idx],
            "event_num": bundle.event_num[valid_idx],
            "game_date_ord": bundle.game_date_ord[valid_idx],
            "actual_final_margin": bundle.terminal_final_margin_home[valid_idx],
            "pred_final_margin": pred_final_margin_valid,
            "actual_home_win": bundle.home_win[valid_idx],
            "pred_home_win_prob": pred_home_win_valid,
            "pred_entropy": entropy_valid,
        }
    )
    if not baseline_eval.empty:
        value_eval_df = value_eval_df.merge(
            baseline_eval,
            on=["game_id", "event_num", "game_date_ord", "actual_home_win", "actual_final_margin"],
            how="left",
        )
    value_eval_df.to_csv(output_dir / "game_prediction_eval.csv", index=False)

    summary = {
        "config": vars(args),
        "n_transitions": int(len(bundle.numeric)),
        "n_games": int(np.unique(bundle.game_id).size),
        "n_players": int(bundle.player_ids.max()),
        "train_rows": int(len(train_idx)),
        "valid_rows": int(len(valid_idx)),
        "value_metrics": value_metrics,
        "baseline_logistic": baseline_logistic,
        "baseline_tree": baseline_tree,
        "output_rows": int(len(output_indices)),
        "state_rows": int(len(state_values)),
        "state_phi_rows": int(len(state_player_phi)),
        "event_credit_rows": int(len(credits)),
        "player_bucket_rows": int(len(player_bucket_totals)),
        "pair_synergies": int(len(pair_synergy)),
        "reconciliation": reconciliation,
    }
    save_json(output_dir / "summary.json", summary)
    LOGGER.info("Finished DRL/Shapley run. Outputs written to %s", output_dir)


if __name__ == "__main__":
    main()
