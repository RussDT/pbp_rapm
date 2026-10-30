#!/usr/bin/env python3
"""Build an exact LUMA-compatible historical WNBA RAPM decomposition.

This bridge intentionally does not impersonate the event-sequenced public WNBA
DECOMP.  LUMA aggregate stints contain shot zones, free throws, turnovers, and
offensive rebounds, but not the event order needed to distinguish first chances
from direct second chances.  The eight targets below therefore decompose
``points - league_ppp * Oliver_possessions`` exactly, with the rebound term
named ``OREB_OPPORTUNITY_VALUE``.

The annual prior pipeline is retrospective: the exact stabilized PBP Stats
feature architecture used by the 2022-2026 WNBA component-prior build is mapped
to low-alpha component targets using labels from earlier seasons only. Generated
CSV and JSON artifacts belong under the gitignored Results tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


COMPONENTS = (
    "RIM_FREQ",
    "RIM_FG",
    "MID_BASE",
    "THREE_FREQ",
    "THREE_FG",
    "FT_VALUE",
    "TOV_VALUE",
    "OREB_OPPORTUNITY_VALUE",
)
SHOT_COMPONENTS = COMPONENTS[:6]
PARENT_COMPONENTS = {
    "TS": SHOT_COMPONENTS,
    "TOV": ("TOV_VALUE",),
    "REB": ("OREB_OPPORTUNITY_VALUE",),
    "TOTAL": COMPONENTS,
}

# Exact component/side feature contract from build_wnba_box_priors.py.  The
# LUMA MID_BASE target is the aggregate-compatible counterpart of active
# DECOMP_MID_FG; OREB_OPPORTUNITY_VALUE uses the same mechanisms as the modern
# second-chance prior while retaining its distinct target name.
OFFENSE_FEATURES = {
    "RIM_FREQ": ("rim_freq", "rim_fga100", "fga100", "usage_proxy", "rim_assists100", "two_assisted_share", "unassisted_two_points100"),
    "RIM_FG": ("rim_pct", "rim_fga100", "rim_assists100", "assist_points100", "assisted_two_points100", "unassisted_two_points100", "two_assisted_share"),
    "MID_BASE": ("mid_pct", "mid_fga100", "usage_proxy", "assisted_two_points100", "unassisted_two_points100", "two_assisted_share"),
    "THREE_FREQ": ("three_freq", "three_fga100", "fga100", "assist_points100", "three_assisted_share", "unassisted_three_points100"),
    "THREE_FG": ("three_pct", "three_fga100", "assist_points100", "assisted_three_points100", "unassisted_three_points100", "three_assisted_share"),
    "FT_VALUE": ("fta100", "ft_pct", "shooting_fouls_drawn100", "fouls_drawn100", "usage_proxy"),
    "TOV_VALUE": ("turnovers100", "badpass_tov100", "lostball_tov100", "liveball_tov100", "assists100", "assist_tov_ratio", "usage_proxy"),
    "OREB_OPPORTUNITY_VALUE": ("oreb100", "self_oreb100", "rim_fga100", "mid_fga100"),
}
DEFENSE_FEATURES = {
    "RIM_FREQ": ("blocks100", "blocked2s100", "shooting_fouls100", "fouls100", "defreb100"),
    "RIM_FG": ("blocks100", "blocked2s100", "shooting_fouls100", "defreb100"),
    "MID_BASE": ("blocks100", "steals100", "fouls100", "defreb100"),
    "THREE_FREQ": ("steals100", "blocked3s100", "fouls100", "defreb100"),
    "THREE_FG": ("steals100", "blocked3s100", "fouls100", "defreb100"),
    "FT_VALUE": ("shooting_fouls100", "fouls100", "blocks100"),
    "TOV_VALUE": ("steals100", "badpass_steals100", "lostball_steals100"),
    "OREB_OPPORTUNITY_VALUE": ("defreb100", "blocks100", "fouls100"),
}

MAPPER_ALPHA_GRID = (0.1, 1.0, 10.0, 100.0)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORICAL_FEATURE_ROOT = REPO_ROOT / "nba_pipeline" / "external" / "gabriel_wnba" / "data"
DEFAULT_MODERN_FEATURE_ROOT = REPO_ROOT.parent / "wnba" / "data"

TALLY_LENGTH = 16
DEFAULT_EVENT_HISTORY = (
    Path(__file__).resolve().parent
    / "Results"
    / "current"
    / "publish"
    / "wnba_vpm_boxprior_history_2022_2026.csv"
)


@dataclass
class SeasonData:
    season: int
    observations: pd.DataFrame
    audit: dict[str, Any]
    baselines: dict[str, float]


@dataclass
class SolveResult:
    players: list[str]
    beta: np.ndarray
    prior: np.ndarray
    off_exposure: np.ndarray
    def_exposure: np.ndarray
    weighted_means: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_float_array(value: Any) -> np.ndarray | None:
    if not isinstance(value, list) or len(value) != TALLY_LENGTH:
        return None
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(result)):
        return None
    return result


def _valid_lineups(home: Any, away: Any) -> bool:
    if not isinstance(home, list) or not isinstance(away, list):
        return False
    if len(home) != 5 or len(away) != 5:
        return False
    home_ids = [str(value) for value in home]
    away_ids = [str(value) for value in away]
    return len(set(home_ids)) == 5 and len(set(away_ids)) == 5 and not (set(home_ids) & set(away_ids))


def _side_record(
    *,
    season: int,
    season_type: str,
    game_id: str,
    date: str | None,
    seconds: float,
    offense: list[str],
    defense: list[str],
    points: float,
    tally: np.ndarray,
) -> dict[str, Any]:
    fga = float(tally[0] + tally[2] + tally[4])
    possessions = fga + 0.44 * float(tally[6]) + float(tally[8]) - float(tally[9])
    return {
        "season": season,
        "season_type": season_type,
        "game_id": str(game_id),
        "date": date,
        "seconds": seconds,
        "offense": tuple(str(value) for value in offense),
        "defense": tuple(str(value) for value in defense),
        "points": points,
        "possessions": possessions,
        "rim_fga": float(tally[0]),
        "rim_points": float(tally[1]),
        "mid_fga": float(tally[2]),
        "mid_points": float(tally[3]),
        "three_fga": float(tally[4]),
        "three_points": float(tally[5]),
        "fta": float(tally[6]),
        "ft_points": float(tally[7]),
        "turnovers": float(tally[8]),
        "oreb": float(tally[9]),
    }


def load_luma_season(
    luma_root: Path,
    season: int,
    season_types: Iterable[str] = ("rs", "po"),
    *,
    allow_zero_seconds: bool = False,
) -> SeasonData:
    """Load and strictly validate one LUMA season into offense-side rows."""

    rows: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    files: list[dict[str, Any]] = []
    game_ids: set[str] = set()
    duplicate_game_ids: set[str] = set()
    raw_games = 0
    raw_stints = 0
    candidate_sides = 0

    for season_type in season_types:
        source = luma_root / "data" / "stints" / f"{season_type}_{season}.json"
        if not source.exists():
            reasons[f"missing_{season_type}_file"] += 1
            continue
        with source.open() as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected object at {source}, got {type(payload).__name__}")
        files.append({"path": str(source), "sha256": _sha256(source), "games": len(payload)})
        raw_games += len(payload)

        for game_id, game in payload.items():
            if game_id in game_ids:
                duplicate_game_ids.add(str(game_id))
            game_ids.add(str(game_id))
            if not isinstance(game, dict) or not isinstance(game.get("stints"), list):
                reasons["invalid_game_payload"] += 1
                continue
            date = game.get("date")
            for stint in game["stints"]:
                raw_stints += 1
                candidate_sides += 2
                if not isinstance(stint, list) or len(stint) != 7:
                    reasons["invalid_stint_shape"] += 2
                    continue
                home, away, seconds_raw, home_points_raw, away_points_raw, home_raw, away_raw = stint
                if not _valid_lineups(home, away):
                    reasons["invalid_lineups"] += 2
                    continue
                try:
                    seconds = float(seconds_raw)
                    home_points = float(home_points_raw)
                    away_points = float(away_points_raw)
                except (TypeError, ValueError):
                    reasons["invalid_numeric_header"] += 2
                    continue
                if not np.isfinite(seconds) or not np.isfinite(home_points) or not np.isfinite(away_points):
                    reasons["invalid_numeric_header"] += 2
                    continue
                if seconds < 0 or (seconds == 0 and not allow_zero_seconds):
                    reasons["nonpositive_seconds"] += 2
                    continue

                for offense, defense, points, raw_tally in (
                    (home, away, home_points, home_raw),
                    (away, home, away_points, away_raw),
                ):
                    tally = _as_float_array(raw_tally)
                    if tally is None:
                        reasons["invalid_tally"] += 1
                        continue
                    if np.any(tally < 0):
                        reasons["negative_tally"] += 1
                        continue
                    if not np.isclose(points, tally[1] + tally[3] + tally[5] + tally[7], atol=1e-9):
                        reasons["score_tally_mismatch"] += 1
                        continue
                    if not np.isclose(tally[8], tally[13] + tally[14] + tally[15], atol=1e-9):
                        reasons["turnover_partition_mismatch"] += 1
                        continue
                    record = _side_record(
                        season=season,
                        season_type=season_type,
                        game_id=str(game_id),
                        date=None if date is None else str(date),
                        seconds=seconds,
                        offense=offense,
                        defense=defense,
                        points=points,
                        tally=tally,
                    )
                    if record["possessions"] <= 0:
                        reasons["nonpositive_possessions"] += 1
                        continue
                    rows.append(record)

    observations = pd.DataFrame(rows)
    if observations.empty:
        raise ValueError(f"No valid LUMA observations for season {season}")
    baselines = add_component_targets(observations)
    audit = {
        "season": season,
        "files": files,
        "raw_games": raw_games,
        "unique_games": len(game_ids),
        "duplicate_game_ids": sorted(duplicate_game_ids),
        "raw_stints": raw_stints,
        "candidate_sides": candidate_sides,
        "kept_sides": len(observations),
        "excluded_sides": candidate_sides - len(observations),
        "exclusion_reasons": dict(sorted(reasons.items())),
        "allow_zero_seconds": allow_zero_seconds,
    }
    return SeasonData(season=season, observations=observations, audit=audit, baselines=baselines)


def add_component_targets(frame: pd.DataFrame) -> dict[str, float]:
    """Add exact point-numerator targets and return league baselines."""

    total_possessions = float(frame["possessions"].sum())
    fga = frame["rim_fga"] + frame["mid_fga"] + frame["three_fga"]
    rim_fga = float(frame["rim_fga"].sum())
    mid_fga = float(frame["mid_fga"].sum())
    three_fga = float(frame["three_fga"].sum())
    if min(total_possessions, rim_fga, mid_fga, three_fga) <= 0:
        raise ValueError("A season needs positive possessions and attempts in every shot zone")

    league_ppp = float(frame["points"].sum()) / total_possessions
    rim_pps = float(frame["rim_points"].sum()) / rim_fga
    mid_pps = float(frame["mid_points"].sum()) / mid_fga
    three_pps = float(frame["three_points"].sum()) / three_fga

    frame["RIM_FREQ"] = (rim_pps - mid_pps) * frame["rim_fga"]
    frame["RIM_FG"] = frame["rim_points"] - rim_pps * frame["rim_fga"]
    frame["MID_BASE"] = (
        mid_pps * (frame["rim_fga"] + frame["three_fga"])
        + frame["mid_points"]
        - league_ppp * fga
    )
    frame["THREE_FREQ"] = (three_pps - mid_pps) * frame["three_fga"]
    frame["THREE_FG"] = frame["three_points"] - three_pps * frame["three_fga"]
    frame["FT_VALUE"] = frame["ft_points"] - 0.44 * league_ppp * frame["fta"]
    frame["TOV_VALUE"] = -league_ppp * frame["turnovers"]
    frame["OREB_OPPORTUNITY_VALUE"] = league_ppp * frame["oreb"]
    frame["TOTAL_CENTERED"] = frame["points"] - league_ppp * frame["possessions"]

    component_sum = frame[list(COMPONENTS)].sum(axis=1)
    error = component_sum - frame["TOTAL_CENTERED"]
    max_error = float(np.max(np.abs(error)))
    if max_error > 1e-9:
        raise AssertionError(f"LUMA row algebra failed: max error {max_error:.3e}")

    return {
        "league_ppp": league_ppp,
        "rim_points_per_attempt": rim_pps,
        "mid_points_per_attempt": mid_pps,
        "three_points_per_attempt": three_pps,
        "total_possessions": total_possessions,
        "max_row_algebra_error": max_error,
    }


def build_design(frame: pd.DataFrame) -> tuple[sparse.csr_matrix, list[str], np.ndarray, np.ndarray]:
    players = sorted(
        {player for lineup in frame["offense"] for player in lineup}
        | {player for lineup in frame["defense"] for player in lineup}
    )
    index = {player: idx for idx, player in enumerate(players)}
    row_idx: list[int] = []
    col_idx: list[int] = []
    n_players = len(players)
    for row_number, (offense, defense) in enumerate(zip(frame["offense"], frame["defense"])):
        row_idx.extend([row_number] * 10)
        col_idx.extend([index[player] for player in offense])
        col_idx.extend([n_players + index[player] for player in defense])
    values = np.ones(len(row_idx), dtype=float)
    design = sparse.csr_matrix((values, (row_idx, col_idx)), shape=(len(frame), 2 * n_players))
    weights = frame["possessions"].to_numpy(dtype=float)
    seconds = frame["seconds"].to_numpy(dtype=float)
    return design, players, weights, seconds


def _center_sides(values: np.ndarray, off_exposure: np.ndarray, def_exposure: np.ndarray) -> np.ndarray:
    centered = np.asarray(values, dtype=float).copy()
    if centered.ndim == 1:
        centered = centered[:, None]
    n_players = len(off_exposure)
    for start, exposure in ((0, off_exposure), (n_players, def_exposure)):
        block = centered[start : start + n_players]
        active = exposure > 0
        if np.any(active):
            mean = np.average(block[active], axis=0, weights=exposure[active])
            block[active] -= mean
    return centered


def solve_block(
    design: sparse.csr_matrix,
    players: list[str],
    weights: np.ndarray,
    targets: np.ndarray,
    *,
    off_alpha: float,
    def_alpha: float,
    prior: np.ndarray | None = None,
) -> SolveResult:
    """Solve multiple prior-centred targets with one shared ridge system."""

    target_matrix = np.asarray(targets, dtype=float)
    if target_matrix.ndim == 1:
        target_matrix = target_matrix[:, None]
    if target_matrix.shape[0] != design.shape[0]:
        raise ValueError("Target rows do not match the design matrix")
    n_players = len(players)
    weighted_means = np.average(target_matrix, axis=0, weights=weights)
    centered_targets = target_matrix - weighted_means

    exposures = np.asarray(design.T @ weights).reshape(-1)
    off_exposure = exposures[:n_players]
    def_exposure = exposures[n_players:]
    if prior is None:
        prior_matrix = np.zeros((2 * n_players, target_matrix.shape[1]), dtype=float)
    else:
        prior_matrix = np.asarray(prior, dtype=float)
        if prior_matrix.ndim == 1:
            prior_matrix = prior_matrix[:, None]
        if prior_matrix.shape != (2 * n_players, target_matrix.shape[1]):
            raise ValueError("Prior shape does not match the ridge problem")
    prior_matrix = _center_sides(prior_matrix, off_exposure, def_exposure)

    penalties = np.concatenate(
        [np.full(n_players, float(off_alpha)), np.full(n_players, float(def_alpha))]
    )
    weighted_design = design.multiply(weights[:, None])
    lhs = (design.T @ weighted_design).toarray()
    lhs.flat[:: lhs.shape[0] + 1] += penalties
    rhs = np.asarray(design.T @ (weights[:, None] * centered_targets))
    rhs += penalties[:, None] * prior_matrix
    beta = np.linalg.solve(lhs, rhs)
    beta = _center_sides(beta, off_exposure, def_exposure)
    return SolveResult(
        players=players,
        beta=beta,
        prior=prior_matrix,
        off_exposure=off_exposure,
        def_exposure=def_exposure,
        weighted_means=weighted_means,
    )


def _target_matrix(frame: pd.DataFrame) -> np.ndarray:
    numerators = frame[list(COMPONENTS) + ["TOTAL_CENTERED"]].to_numpy(dtype=float)
    return numerators / frame["possessions"].to_numpy(dtype=float)[:, None]


def load_crosswalk(luma_root: Path) -> pd.DataFrame:
    path = luma_root / "data" / "crosswalk.csv"
    crosswalk = pd.read_csv(path, dtype=str).fillna("")
    if "luma_id" not in crosswalk or crosswalk["luma_id"].duplicated().any():
        raise ValueError(f"Expected a unique luma_id in {path}")
    return crosswalk


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _smoothed_rate(
    count: pd.Series,
    exposure: pd.Series,
    prior_exposure: float,
) -> pd.Series:
    count = pd.to_numeric(count, errors="coerce").fillna(0.0)
    exposure = pd.to_numeric(exposure, errors="coerce").fillna(0.0)
    league_rate = 100.0 * count.sum() / max(float(exposure.sum()), 1.0)
    return (100.0 * count + prior_exposure * league_rate) / (exposure + prior_exposure)


def _smoothed_share(
    numerator: pd.Series,
    denominator: pd.Series,
    prior_attempts: float,
) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce").fillna(0.0)
    denominator = pd.to_numeric(denominator, errors="coerce").fillna(0.0)
    league_share = numerator.sum() / max(float(denominator.sum()), 1.0)
    return (numerator + prior_attempts * league_share) / (denominator + prior_attempts)


def _normalize_name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = text.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]", "", ascii_text)


def load_pbpstats_features(path: Path, season: int) -> pd.DataFrame:
    """Build the exact stabilized feature surface used by the modern WNBA prior."""

    raw = pd.read_csv(path, low_memory=False)
    if not {"EntityId", "Name"}.issubset(raw.columns):
        raise ValueError(f"Historical feature source lacks EntityId/Name: {path}")
    out = pd.DataFrame(
        {
            "source_player_id": (
                pd.to_numeric(raw["EntityId"], errors="coerce").astype("Int64").astype(str)
            ),
            "source_player_name": raw["Name"].fillna(raw.get("ShortName", "")).astype(str),
            "season": int(season),
        }
    )
    out["minutes"] = _numeric(raw, "Minutes").clip(lower=0.0)
    out["off_poss"] = _numeric(raw, "OffPoss")
    out["def_poss"] = _numeric(raw, "DefPoss")

    fg2a = _numeric(raw, "FG2A")
    fg3a = _numeric(raw, "FG3A")
    fga = fg2a + fg3a
    fta = _numeric(raw, "FTA")
    rim_fga = _numeric(raw, "AtRimFGA")
    rim_fgm = _numeric(raw, "AtRimFGM")
    mid_fga = _numeric(raw, "ShortMidRangeFGA") + _numeric(raw, "LongMidRangeFGA")
    mid_fgm = _numeric(raw, "ShortMidRangeFGM") + _numeric(raw, "LongMidRangeFGM")
    three_fga = fg3a
    three_fgm = _numeric(raw, "FG3M")
    assisted_two_points = _numeric(raw, "PtsAssisted2s")
    unassisted_two_points = _numeric(raw, "PtsUnassisted2s")
    assisted_three_points = _numeric(raw, "PtsAssisted3s")
    unassisted_three_points = _numeric(raw, "PtsUnassisted3s")

    offense_counts = {
        "points100": _numeric(raw, "Points"),
        "fga100": fga,
        "tsa100": fga + 0.44 * fta,
        "rim_fga100": rim_fga,
        "mid_fga100": mid_fga,
        "three_fga100": three_fga,
        "fta100": fta,
        "assists100": _numeric(raw, "Assists"),
        "assist_points100": _numeric(raw, "AssistPoints"),
        "rim_assists100": _numeric(raw, "AtRimAssists"),
        "assisted_two_points100": assisted_two_points,
        "unassisted_two_points100": unassisted_two_points,
        "assisted_three_points100": assisted_three_points,
        "unassisted_three_points100": unassisted_three_points,
        "turnovers100": _numeric(raw, "Turnovers"),
        "badpass_tov100": _numeric(raw, "BadPassTurnovers"),
        "lostball_tov100": _numeric(raw, "LostBallTurnovers"),
        "liveball_tov100": _numeric(raw, "LiveBallTurnovers"),
        "oreb100": _numeric(raw, "OffRebounds"),
        "self_oreb100": _numeric(raw, "SelfOReb"),
        "fouls_drawn100": _numeric(raw, "FoulsDrawn"),
        "shooting_fouls_drawn100": (
            _numeric(raw, "TwoPtShootingFoulsDrawn")
            + _numeric(raw, "ThreePtShootingFoulsDrawn")
        ),
    }
    for name, count in offense_counts.items():
        out[name] = _smoothed_rate(count, out["off_poss"], prior_exposure=250.0)

    defense_counts = {
        "steals100": _numeric(raw, "Steals"),
        "badpass_steals100": _numeric(raw, "BadPassSteals"),
        "lostball_steals100": _numeric(raw, "LostBallSteals"),
        "blocks100": _numeric(raw, "Blocks"),
        "blocked2s100": _numeric(raw, "Blocked2s"),
        "blocked3s100": _numeric(raw, "Blocked3s"),
        "defreb100": _numeric(raw, "DefRebounds"),
        "shooting_fouls100": _numeric(raw, "ShootingFouls"),
        "fouls100": _numeric(raw, "Fouls"),
    }
    for name, count in defense_counts.items():
        out[name] = _smoothed_rate(count, out["def_poss"], prior_exposure=400.0)

    out["rim_freq"] = _smoothed_share(rim_fga, fga, prior_attempts=75.0)
    out["mid_freq"] = _smoothed_share(mid_fga, fga, prior_attempts=75.0)
    out["three_freq"] = _smoothed_share(three_fga, fga, prior_attempts=75.0)
    out["rim_pct"] = _smoothed_share(rim_fgm, rim_fga, prior_attempts=50.0)
    out["mid_pct"] = _smoothed_share(mid_fgm, mid_fga, prior_attempts=50.0)
    out["three_pct"] = _smoothed_share(three_fgm, three_fga, prior_attempts=75.0)
    # Made-shot assist splits are a directional role proxy, not a full zTS
    # adjustment: misses do not carry an assisted/self-created label here.
    out["two_assisted_share"] = _smoothed_share(
        assisted_two_points,
        assisted_two_points + unassisted_two_points,
        prior_attempts=100.0,
    )
    out["three_assisted_share"] = _smoothed_share(
        assisted_three_points,
        assisted_three_points + unassisted_three_points,
        prior_attempts=100.0,
    )
    out["ft_pct"] = _smoothed_share(_numeric(raw, "FtPoints"), fta, prior_attempts=75.0)
    out["usage_proxy"] = _smoothed_rate(
        fga + 0.44 * fta + _numeric(raw, "Turnovers"),
        out["off_poss"],
        prior_exposure=250.0,
    )
    out["assist_tov_ratio"] = (
        _numeric(raw, "Assists") + 1.0
    ) / (_numeric(raw, "Turnovers") + 1.0)
    out["normalized_name"] = out["source_player_name"].map(_normalize_name)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def map_features_to_luma(
    features: pd.DataFrame,
    crosswalk: pd.DataFrame,
    season: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Map PBP Stats ids/names to stable season-specific LUMA ids."""

    active = crosswalk.copy()
    first = pd.to_numeric(active["first_season"], errors="coerce")
    last = pd.to_numeric(active["last_season"], errors="coerce")
    active = active.loc[(first <= season) & (last >= season)].copy()
    active["normalized_name"] = active["display_name"].map(_normalize_name)
    name_counts = active["normalized_name"].value_counts()
    unique_names = set(name_counts.loc[name_counts == 1].index) - {""}
    name_map = active.loc[active["normalized_name"].isin(unique_names)].set_index(
        "normalized_name"
    )["luma_id"].to_dict()
    id_rows = active.loc[active["wnba_stats_id"].astype(str).str.len() > 0].copy()
    id_rows["wnba_stats_id"] = id_rows["wnba_stats_id"].str.replace(r"\.0$", "", regex=True)
    id_map = id_rows.set_index("wnba_stats_id")["luma_id"].to_dict()

    mapped = features.copy()
    mapped["id_match"] = mapped["source_player_id"].astype(str).map(id_map)
    mapped["name_match"] = mapped["normalized_name"].map(name_map)
    conflict = (
        mapped["id_match"].notna()
        & mapped["name_match"].notna()
        & (mapped["id_match"] != mapped["name_match"])
    )
    mapped["player_id"] = mapped["id_match"].fillna(mapped["name_match"])
    mapped["match_method"] = np.select(
        [conflict, mapped["id_match"].notna(), mapped["name_match"].notna()],
        ["conflict", "wnba_stats_id", "normalized_name"],
        default="unmatched",
    )
    mapped.loc[conflict, "player_id"] = np.nan
    matched = mapped.loc[mapped["player_id"].notna()].copy()
    duplicates = matched["player_id"].duplicated(keep=False)
    if duplicates.any():
        duplicate_ids = sorted(matched.loc[duplicates, "player_id"].astype(str).unique())
        raise ValueError(f"Duplicate PBP Stats to LUMA mappings for {season}: {duplicate_ids}")
    audit = {
        "season": season,
        "raw_feature_rows": len(features),
        "matched_feature_rows": len(matched),
        "match_methods": dict(sorted(mapped["match_method"].value_counts().to_dict().items())),
        "active_crosswalk_rows": len(active),
    }
    drop_columns = ["id_match", "name_match"]
    return matched.drop(columns=drop_columns), audit


def load_historical_feature_panel(
    seasons: Iterable[int],
    *,
    historical_root: Path,
    modern_root: Path,
    crosswalk: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    panels: list[pd.DataFrame] = []
    audits: dict[str, Any] = {}
    for season in seasons:
        root = modern_root if season >= 2022 else historical_root
        path = root / f"{season}_pbp.csv"
        if not path.exists():
            audits[str(season)] = {"season": season, "status": "missing", "path": str(path)}
            continue
        raw_features = load_pbpstats_features(path, season)
        mapped, audit = map_features_to_luma(raw_features, crosswalk, season)
        audit.update({"status": "loaded", "path": str(path.resolve()), "sha256": _sha256(path)})
        audits[str(season)] = audit
        panels.append(mapped)
    if not panels:
        return pd.DataFrame(columns=["player_id", "season"]), audits
    panel = pd.concat(panels, ignore_index=True, sort=False)
    if panel.duplicated(["season", "player_id"]).any():
        raise ValueError("Historical PBP Stats panel has duplicate season/player keys")
    return panel, audits


def result_frame(result: SolveResult, target_names: Iterable[str]) -> pd.DataFrame:
    names = list(target_names)
    n_players = len(result.players)
    output = pd.DataFrame(
        {
            "player_id": result.players,
            "off_possessions": result.off_exposure,
            "def_possessions": result.def_exposure,
        }
    )
    for column, name in enumerate(names):
        off = 100.0 * result.beta[:n_players, column]
        defense = -100.0 * result.beta[n_players:, column]
        prior_off = 100.0 * result.prior[:n_players, column]
        prior_def = -100.0 * result.prior[n_players:, column]
        output[f"{name}_off"] = off
        output[f"{name}_def"] = defense
        output[f"{name}_net"] = off + defense
        output[f"prior_{name}_off"] = prior_off
        output[f"prior_{name}_def"] = prior_def
        output[f"prior_{name}_net"] = prior_off + prior_def
    return output


def build_target_labels(
    season_data: SeasonData,
    *,
    off_alpha: float,
    def_alpha: float,
) -> pd.DataFrame:
    frame = season_data.observations
    design, players, weights, _ = build_design(frame)
    result = solve_block(
        design,
        players,
        weights,
        _target_matrix(frame)[:, : len(COMPONENTS)],
        off_alpha=off_alpha,
        def_alpha=def_alpha,
    )
    labels = result_frame(result, COMPONENTS)
    labels.insert(0, "season", season_data.season)
    return labels


def _fit_predict_prior(
    training: pd.DataFrame,
    current_box: pd.DataFrame,
    *,
    target: str,
    feature_columns: tuple[str, ...],
    weight_column: str,
    alpha_grid: tuple[float, ...],
) -> tuple[dict[str, float], dict[str, Any]]:
    needed = ["player_id", "season", target, weight_column, *feature_columns]
    fit = training[needed].copy()
    fit[target] = pd.to_numeric(fit[target], errors="coerce")
    fit[weight_column] = pd.to_numeric(fit[weight_column], errors="coerce")
    fit = fit.loc[fit[target].notna() & (fit[weight_column] > 0)].copy()
    if len(fit) < max(20, 3 * len(feature_columns)) or current_box.empty:
        return {}, {"status": "insufficient_history", "training_rows": len(fit)}

    fit[list(feature_columns)] = (
        fit[list(feature_columns)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    )
    current = current_box.copy()
    current[list(feature_columns)] = (
        current[list(feature_columns)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    )

    def fit_model(rows: pd.DataFrame, alpha: float) -> tuple[StandardScaler, Ridge]:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(rows[list(feature_columns)])
        model = Ridge(alpha=float(alpha))
        weights = np.sqrt(rows[weight_column].clip(lower=1.0, upper=2500.0))
        weights = weights / weights.mean()
        model.fit(
            x_scaled,
            rows[target].to_numpy(dtype=float),
            sample_weight=weights.to_numpy(dtype=float),
        )
        return scaler, model

    fold_years = sorted(int(value) for value in fit["season"].unique())[1:]
    alpha_scores: list[dict[str, Any]] = []
    for alpha in alpha_grid:
        errors: list[float] = []
        weights: list[float] = []
        predictions: list[float] = []
        actuals: list[float] = []
        for validation_year in fold_years:
            train_rows = fit.loc[fit["season"] < validation_year]
            validation_rows = fit.loc[fit["season"] == validation_year]
            if len(train_rows) < max(20, 3 * len(feature_columns)) or validation_rows.empty:
                continue
            fold_scaler, fold_model = fit_model(train_rows, alpha)
            fold_prediction = fold_model.predict(
                fold_scaler.transform(validation_rows[list(feature_columns)])
            )
            fold_weight = np.sqrt(
                validation_rows[weight_column].clip(lower=1.0, upper=2500.0)
            ).to_numpy(dtype=float)
            errors.extend((fold_prediction - validation_rows[target].to_numpy(dtype=float)).tolist())
            weights.extend(fold_weight.tolist())
            predictions.extend(fold_prediction.tolist())
            actuals.extend(validation_rows[target].astype(float).tolist())
        alpha_scores.append(
            {
                "alpha": float(alpha),
                "forward_rmse": (
                    float(np.sqrt(np.average(np.square(errors), weights=weights)))
                    if errors
                    else None
                ),
                "forward_correlation": (
                    float(pd.Series(predictions).corr(pd.Series(actuals)))
                    if len(predictions) >= 2
                    else None
                ),
                "validation_rows": len(errors),
            }
        )
    scored = pd.DataFrame(alpha_scores)
    usable = scored.loc[scored["forward_rmse"].notna()].copy()
    if usable.empty:
        selected_alpha = 10.0
    else:
        usable["forward_correlation"] = usable["forward_correlation"].fillna(-np.inf)
        selected_alpha = float(
            usable.sort_values(
                ["forward_rmse", "forward_correlation", "alpha"],
                ascending=[True, False, False],
            ).iloc[0]["alpha"]
        )
    scaler, model = fit_model(fit, selected_alpha)
    prediction = model.predict(scaler.transform(current[list(feature_columns)]))
    lower = float(fit[target].quantile(0.01))
    upper = float(fit[target].quantile(0.99))
    prediction = np.clip(prediction, lower, upper)
    return (
        dict(zip(current["player_id"].astype(str), prediction.astype(float))),
        {
            "status": "fit",
            "training_rows": len(fit),
            "training_seasons": sorted(int(value) for value in fit["season"].unique()),
            "features": list(feature_columns),
            "selected_alpha": selected_alpha,
            "alpha_scores": alpha_scores,
            "prediction_clip": [lower, upper],
        },
    )


def build_prior_matrix(
    *,
    season: int,
    players: list[str],
    target_panel: pd.DataFrame,
    box_panel: pd.DataFrame,
    mapper_alpha_grid: tuple[float, ...],
    prior_scale: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    n_players = len(players)
    prior = np.zeros((2 * n_players, len(COMPONENTS) + 1), dtype=float)
    earlier_targets = target_panel.loc[target_panel["season"] < season].copy()
    earlier_box = box_panel.loc[box_panel["season"] < season].copy()
    current_box = box_panel.loc[box_panel["season"] == season].copy()
    training = earlier_targets.merge(earlier_box, on=["season", "player_id"], how="inner")
    player_index = {player: index for index, player in enumerate(players)}
    metadata: dict[str, Any] = {"season": season, "components": {}}

    for component_index, component in enumerate(COMPONENTS):
        metadata["components"][component] = {}
        for side, offset, features, weight_column in (
            ("off", 0, OFFENSE_FEATURES[component], "off_possessions"),
            ("def", n_players, DEFENSE_FEATURES[component], "def_possessions"),
        ):
            predictions, fit_metadata = _fit_predict_prior(
                training,
                current_box,
                target=f"{component}_{side}",
                feature_columns=features,
                weight_column=weight_column,
                alpha_grid=mapper_alpha_grid,
            )
            metadata["components"][component][side] = fit_metadata
            for player_id, value in predictions.items():
                if player_id not in player_index:
                    continue
                # Reader-facing defense is positive-good; the design's defense
                # block is points allowed, so it enters the solve with a minus.
                sign = 1.0 if side == "off" else -1.0
                prior[offset + player_index[player_id], component_index] = (
                    sign * prior_scale * float(value) / 100.0
                )

    prior[:, -1] = prior[:, : len(COMPONENTS)].sum(axis=1)
    metadata["training_seasons"] = sorted(int(value) for value in training["season"].unique())
    metadata["prior_scale"] = float(prior_scale)
    return prior, metadata


def _player_seconds(
    frame: pd.DataFrame,
    players: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    index = {player: idx for idx, player in enumerate(players)}
    offense = np.zeros(len(players), dtype=float)
    defense = np.zeros(len(players), dtype=float)
    for row in frame.itertuples(index=False):
        for player in row.offense:
            offense[index[player]] += float(row.seconds)
        for player in row.defense:
            defense[index[player]] += float(row.seconds)
    return offense, defense


def _add_parent_columns(board: pd.DataFrame) -> pd.DataFrame:
    for parent, children in PARENT_COMPONENTS.items():
        for side in ("off", "def", "net"):
            board[f"{parent}_{side}"] = board[[f"{child}_{side}" for child in children]].sum(axis=1)
            board[f"prior_{parent}_{side}"] = board[
                [f"prior_{child}_{side}" for child in children]
            ].sum(axis=1)
    return board


def build_season_board(
    season_data: SeasonData,
    *,
    prior: np.ndarray,
    off_alpha: float,
    def_alpha: float,
    pure_alpha: float,
    crosswalk: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    frame = season_data.observations
    design, players, weights, _ = build_design(frame)
    targets = _target_matrix(frame)
    fitted = solve_block(
        design,
        players,
        weights,
        targets,
        off_alpha=off_alpha,
        def_alpha=def_alpha,
        prior=prior,
    )
    board = result_frame(fitted, (*COMPONENTS, "FITTED_TOTAL"))
    board = _add_parent_columns(board)

    pure = solve_block(
        design,
        players,
        weights,
        targets[:, -1],
        off_alpha=pure_alpha,
        def_alpha=pure_alpha,
    )
    pure_frame = result_frame(pure, ("PURE_TOTAL",))
    board = board.merge(
        pure_frame[["player_id", "PURE_TOTAL_off", "PURE_TOTAL_def", "PURE_TOTAL_net"]],
        on="player_id",
        how="left",
        validate="one_to_one",
    )

    off_seconds, def_seconds = _player_seconds(frame, players)
    seconds_frame = pd.DataFrame(
        {
            "player_id": players,
            "minutes": (off_seconds + def_seconds) / 120.0,
        }
    )
    board = board.merge(seconds_frame, on="player_id", how="left", validate="one_to_one")
    board["possessions"] = (board["off_possessions"] + board["def_possessions"]) / 2.0
    board.insert(0, "season", season_data.season)
    board = board.merge(
        crosswalk.rename(columns={"luma_id": "player_id"}),
        on="player_id",
        how="left",
        validate="one_to_one",
    )

    # Reader-facing aliases. LUMA_VPM is the true-prior compatible total;
    # PURE_RAPM is the conventional equal-alpha no-prior total.
    board["LUMA_ORAPM"] = board["TOTAL_off"]
    board["LUMA_DRAPM"] = board["TOTAL_def"]
    board["LUMA_VPM"] = board["TOTAL_net"]
    board["PURE_ORAPM"] = board["PURE_TOTAL_off"]
    board["PURE_DRAPM"] = board["PURE_TOTAL_def"]
    board["PURE_RAPM"] = board["PURE_TOTAL_net"]
    board["qualified_1000"] = board["possessions"] >= 1000.0
    board["LUMA_VPM_rank"] = board["LUMA_VPM"].rank(method="min", ascending=False).astype(int)
    qualified_rank = board.loc[board["qualified_1000"], "LUMA_VPM"].rank(method="min", ascending=False)
    board["LUMA_VPM_rank_qualified"] = pd.Series(pd.NA, index=board.index, dtype="Int64")
    board.loc[qualified_rank.index, "LUMA_VPM_rank_qualified"] = qualified_rank.astype("Int64")

    fitted_component_sum = board[[f"{component}_net" for component in COMPONENTS]].sum(axis=1)
    diagnostics = {
        "max_player_net_closure_error": float(
            np.max(np.abs(fitted_component_sum - board["FITTED_TOTAL_net"]))
        ),
        "max_parent_total_alias_error": float(np.max(np.abs(board["TOTAL_net"] - board["LUMA_VPM"]))),
        "weighted_total_target_mean": float(fitted.weighted_means[-1]),
    }
    return board.sort_values(["LUMA_VPM", "player_id"], ascending=[False, True]), diagnostics


def compare_to_event_history(
    luma_history: pd.DataFrame,
    event_history_path: Path,
) -> pd.DataFrame:
    """Describe overlap with the event-sequenced board without merging families."""

    event = pd.read_csv(event_history_path, dtype={"player_id": str})
    luma = luma_history.copy()
    luma["join_id"] = (
        luma["wnba_stats_id"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    )
    event["join_id"] = event["player_id"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    event["season"] = pd.to_numeric(event["season"], errors="coerce").astype("Int64")
    common = luma.merge(event, on=["season", "join_id"], suffixes=("_luma", "_event"), how="inner")
    common["event_TS_off"] = common["oALT_EFG_VALUE"] + common["oALT_FT"]
    common["event_TS_def"] = common["dALT_EFG_VALUE"] + common["dALT_FT"]
    metric_pairs = {
        "TOTAL_net": ("LUMA_VPM", "net_rapm"),
        "TOTAL_off": ("LUMA_ORAPM", "off"),
        "TOTAL_def": ("LUMA_DRAPM", "def"),
        "TS_off": ("TS_off", "event_TS_off"),
        "TS_def": ("TS_def", "event_TS_def"),
        "TOV_off": ("TOV_off", "oALT_TOV_VALUE"),
        "TOV_def": ("TOV_def", "dALT_TOV_VALUE"),
        "REB_vs_SECOND_CHANCE_off": ("REB_off", "oSC"),
        "REB_vs_SECOND_CHANCE_def": ("REB_def", "dSC"),
    }
    rows: list[dict[str, Any]] = []
    for season in sorted(int(value) for value in common["season"].dropna().unique()):
        season_rows = common.loc[common["season"] == season].copy()
        scopes = {
            "all_overlap": season_rows,
            "common_1000_possessions": season_rows.loc[
                (season_rows["possessions_luma"] >= 1000)
                & (season_rows["possessions_event"] >= 1000)
            ],
        }
        for scope, scoped in scopes.items():
            for metric, (luma_column, event_column) in metric_pairs.items():
                pair = scoped[[luma_column, event_column]].apply(pd.to_numeric, errors="coerce").dropna()
                rows.append(
                    {
                        "season": season,
                        "scope": scope,
                        "metric": metric,
                        "n_players": len(pair),
                        "pearson": pair[luma_column].corr(pair[event_column]) if len(pair) >= 2 else np.nan,
                        "mean_luma": pair[luma_column].mean(),
                        "mean_event": pair[event_column].mean(),
                        "mean_absolute_difference": (pair[luma_column] - pair[event_column]).abs().mean(),
                    }
                )
    return pd.DataFrame(rows)


def _parse_season_types(value: str) -> tuple[str, ...]:
    mapping = {"RS": ("rs",), "PO": ("po",), "ALL": ("rs", "po")}
    try:
        return mapping[value.upper()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError("season type must be RS, PO, or ALL") from exc


def _parse_float_grid(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a comma-separated numeric grid") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("alpha grid must contain positive values")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--luma-root",
        type=Path,
        default=Path(os.environ["LUMA_WNBA_ROOT"]) if os.environ.get("LUMA_WNBA_ROOT") else None,
        help="Root of the LUMA WNBA repository (or set LUMA_WNBA_ROOT).",
    )
    parser.add_argument("--start-year", type=int, default=2003)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--season-type", type=_parse_season_types, default=("rs", "po"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("wnba_test/Results/luma_historical_our_priors"),
    )
    parser.add_argument("--off-alpha", type=float, default=5000.0)
    parser.add_argument("--def-alpha", type=float, default=40000.0)
    parser.add_argument("--pure-alpha", type=float, default=3000.0)
    parser.add_argument("--target-off-alpha", type=float, default=750.0)
    parser.add_argument("--target-def-alpha", type=float, default=1000.0)
    parser.add_argument(
        "--mapper-alphas",
        type=_parse_float_grid,
        default=MAPPER_ALPHA_GRID,
        help="Expanding-season mapper-alpha grid.",
    )
    parser.add_argument("--prior-scale", type=float, default=1.5)
    parser.add_argument(
        "--historical-feature-root",
        type=Path,
        default=DEFAULT_HISTORICAL_FEATURE_ROOT,
        help="Directory holding archived 2009-2021 YEAR_pbp.csv player totals.",
    )
    parser.add_argument(
        "--modern-feature-root",
        type=Path,
        default=DEFAULT_MODERN_FEATURE_ROOT,
        help="Directory holding canonical 2022+ YEAR_pbp.csv player totals.",
    )
    parser.add_argument(
        "--event-history",
        type=Path,
        default=DEFAULT_EVENT_HISTORY,
        help="Optional current event-DECOMP history used only for a descriptive overlap audit.",
    )
    parser.add_argument(
        "--allow-zero-seconds",
        action="store_true",
        help="Include exact positive-possession rows whose aggregate stint duration is zero.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.luma_root is None:
        raise SystemExit("Pass --luma-root or set LUMA_WNBA_ROOT")
    luma_root = args.luma_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    annual_dir = output_dir / "annual"
    annual_dir.mkdir(parents=True, exist_ok=True)
    seasons = list(range(args.start_year, args.end_year + 1))
    crosswalk = load_crosswalk(luma_root)
    box_panel, feature_audits = load_historical_feature_panel(
        seasons,
        historical_root=args.historical_feature_root.expanduser().resolve(),
        modern_root=args.modern_feature_root.expanduser().resolve(),
        crosswalk=crosswalk,
    )

    print("Pass 1/2: building historical-only component targets")
    target_frames: list[pd.DataFrame] = []
    audits: dict[str, Any] = {}
    baselines: dict[str, Any] = {}
    for season in seasons:
        data = load_luma_season(
            luma_root,
            season,
            args.season_type,
            allow_zero_seconds=args.allow_zero_seconds,
        )
        target_frames.append(
            build_target_labels(
                data,
                off_alpha=args.target_off_alpha,
                def_alpha=args.target_def_alpha,
            )
        )
        audits[str(season)] = data.audit
        baselines[str(season)] = data.baselines
        print(
            f"  {season}: {data.audit['unique_games']} games, "
            f"{data.audit['kept_sides']:,} valid side rows"
        )
    target_panel = pd.concat(target_frames, ignore_index=True, sort=False)
    target_path = output_dir / "annual_component_targets.csv"
    target_panel.to_csv(target_path, index=False)
    for season in seasons:
        target_season = target_panel.loc[target_panel["season"] == season].copy()
        feature_ids = set(
            box_panel.loc[box_panel["season"] == season, "player_id"].astype(str)
        )
        covered = target_season["player_id"].astype(str).isin(feature_ids)
        possessions = (
            target_season["off_possessions"] + target_season["def_possessions"]
        ) / 2.0
        audit = feature_audits.setdefault(
            str(season),
            {"season": season, "status": "missing"},
        )
        audit["luma_player_rows"] = len(target_season)
        audit["covered_luma_player_rows"] = int(covered.sum())
        audit["luma_player_row_coverage"] = float(covered.mean()) if len(covered) else 0.0
        audit["luma_possession_coverage"] = (
            float(possessions.loc[covered].sum() / possessions.sum())
            if float(possessions.sum()) > 0
            else 0.0
        )

    print("Pass 2/2: fitting earlier-season priors and annual boards")
    boards: list[pd.DataFrame] = []
    prior_metadata: dict[str, Any] = {}
    solve_diagnostics: dict[str, Any] = {}
    for season in seasons:
        data = load_luma_season(
            luma_root,
            season,
            args.season_type,
            allow_zero_seconds=args.allow_zero_seconds,
        )
        design, players, _, _ = build_design(data.observations)
        del design
        prior, prior_meta = build_prior_matrix(
            season=season,
            players=players,
            target_panel=target_panel,
            box_panel=box_panel,
            mapper_alpha_grid=args.mapper_alphas,
            prior_scale=args.prior_scale,
        )
        board, diagnostics = build_season_board(
            data,
            prior=prior,
            off_alpha=args.off_alpha,
            def_alpha=args.def_alpha,
            pure_alpha=args.pure_alpha,
            crosswalk=crosswalk,
        )
        fit_active = any(
            side_metadata.get("status") == "fit"
            for component_metadata in prior_meta["components"].values()
            for side_metadata in component_metadata.values()
        )
        current_feature_ids = set(
            box_panel.loc[box_panel["season"] == season, "player_id"].astype(str)
        )
        board["has_prior_features"] = (
            board["player_id"].astype(str).isin(current_feature_ids) & fit_active
        )
        board["model_variant"] = np.where(
            board["has_prior_features"],
            "pbpstats_component_true_prior_assisted_role_v1",
            "zero_or_centered_missing_feature_prior",
        )
        board_path = annual_dir / f"luma_historical_metrics_{season}.csv"
        board.to_csv(board_path, index=False)
        boards.append(board)
        prior_metadata[str(season)] = prior_meta
        solve_diagnostics[str(season)] = diagnostics
        print(
            f"  {season}: {len(board)} players, closure "
            f"{diagnostics['max_player_net_closure_error']:.3e}"
        )

    history = pd.concat(boards, ignore_index=True, sort=False)
    history_path = output_dir / f"luma_historical_metrics_{args.start_year}_{args.end_year}.csv"
    history.to_csv(history_path, index=False)
    comparison_path: Path | None = None
    event_history_metadata: dict[str, Any] | None = None
    if args.event_history and args.event_history.exists():
        comparison = compare_to_event_history(history, args.event_history)
        comparison_path = output_dir / "comparison_to_event_decomp.csv"
        comparison.to_csv(comparison_path, index=False)
        event_history_metadata = {
            "path": str(args.event_history.resolve()),
            "sha256": _sha256(args.event_history.resolve()),
            "comparison_rows": len(comparison),
        }
    max_row_error = max(float(value["max_row_algebra_error"]) for value in baselines.values())
    max_player_error = max(
        float(value["max_player_net_closure_error"]) for value in solve_diagnostics.values()
    )
    manifest = {
        "schema_version": 3,
        "series": "LUMA historical compatible WNBA RAPM decomposition",
        "source_root": str(luma_root),
        "season_range": [args.start_year, args.end_year],
        "season_types": list(args.season_type),
        "components": list(COMPONENTS),
        "possessions": "FGA + 0.44*FTA + TOV - OREB",
        "warnings": [
            "Retrospective descriptive series; same-season stabilized PBP Stats features inform the prior mean.",
            "Prior mappers use labels from earlier seasons only.",
            "2003-2008 have no matching PBP Stats feature files; 2009 is the pure feature warm-up and fitted priors begin in 2010.",
            "OREB_OPPORTUNITY_VALUE is an opportunity term, not event-sequenced SECOND_CHANCE_CLEAN.",
            "ShotQuality EV/residual and net-continuation metrics are not identified by aggregate stints.",
            "Assisted/self-created scoring inputs label made points only and are used as smoothed directional role proxies, not as a full zTS adjustment.",
            "5000/40000 and prior scale 1.5 are borrowed bridge defaults, not LUMA-target-selected hyperparameters.",
        ],
        "parameters": {
            "off_alpha": args.off_alpha,
            "def_alpha": args.def_alpha,
            "pure_alpha": args.pure_alpha,
            "target_off_alpha": args.target_off_alpha,
            "target_def_alpha": args.target_def_alpha,
            "mapper_alphas": list(args.mapper_alphas),
            "prior_scale": args.prior_scale,
            "allow_zero_seconds": args.allow_zero_seconds,
        },
        "feature_contract": {
            "source": "assisted-role-v1 stabilized feature architecture from build_wnba_box_priors.py",
            "historical_root": str(args.historical_feature_root.expanduser().resolve()),
            "modern_root": str(args.modern_feature_root.expanduser().resolve()),
            "offense_features": {key: list(value) for key, value in OFFENSE_FEATURES.items()},
            "defense_features": {key: list(value) for key, value in DEFENSE_FEATURES.items()},
            "offense_rate_prior_possessions": 250.0,
            "defense_rate_prior_possessions": 400.0,
            "assisted_role_share_prior_points": 100.0,
            "assisted_role_semantics": "made-point assisted share plus assisted/unassisted scoring rates; directional role proxy, not all-attempt shot difficulty",
        },
        "feature_audits": feature_audits,
        "audits": audits,
        "baselines": baselines,
        "prior_metadata": prior_metadata,
        "solve_diagnostics": solve_diagnostics,
        "validation": {
            "max_row_algebra_error": max_row_error,
            "max_player_component_closure_error": max_player_error,
        },
        "artifacts": {
            "history": str(history_path),
            "annual_directory": str(annual_dir),
            "target_panel": str(target_path),
            "event_overlap_comparison": None if comparison_path is None else str(comparison_path),
        },
        "event_history_source": event_history_metadata,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote {history_path}")
    if comparison_path is not None:
        print(f"Wrote {comparison_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
