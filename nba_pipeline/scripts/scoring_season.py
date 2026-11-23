#!/usr/bin/env python3
"""Season span, fit freeze, and partial-season rules for the scoring / rZTOV chain.

One place answers three questions that used to be the literal ``2026`` in five
builders:

* which seasons a build covers (``--last-season``; the daily runner takes it
  from ``NBA_RAPM_SEASON_YEAR`` like ``daily_rapm_update.sh``),
* which seasons may enter a fit (``FROZEN_FIT_SEASON``: break-evens, lane point
  prices, estimator labels and CV windows stop there, so a later season is
  scored with the last frozen window and never refit), and
* how a season that is still being played is floored and labelled.

The partial-season floors are a declared rule, not a tuned one: the full-season
possession floors are multiplied by the share of the 1,230-game schedule that
has been played, with a small absolute minimum.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


FROZEN_FIT_SEASON = 2026
SEASON_ENV = "NBA_RAPM_SEASON_YEAR"
FALLBACK_SEASON = 2027
FULL_SEASON_GAMES = 1230

ROW_MIN_POSS = 500.0
QUAL_MIN_POSS = 1000.0
PARTIAL_ROW_MIN_POSS = 100.0
PARTIAL_QUAL_MIN_POSS = 200.0

STATUS_COMPLETE = "complete"
STATUS_IN_PROGRESS = "in_progress"


def current_season() -> int:
    """Season-ending year the daily jobs are on (``2027`` = 2026-27)."""
    value = os.environ.get(SEASON_ENV, "").strip()
    return int(value) if value else FALLBACK_SEASON


def span_tag(first: int, last: int) -> str:
    return f"{int(first)}_{int(last)}"


def season_label(year: int) -> str:
    """``2027`` -> ``2026-27``."""
    return f"{int(year) - 1}-{int(year) % 100:02d}"


def fit_cap(last_season: int, fit_through: int = FROZEN_FIT_SEASON) -> int:
    """Last season allowed into any fit for a build that ends at ``last_season``."""
    return min(int(last_season), int(fit_through))


def rs_games_played(raw_dir: Path, season: int) -> int:
    """Regular-season (``002``) games present in the season's raw PBP file."""
    path = Path(raw_dir) / f"NBA{int(season) % 100:02d}.parquet"
    if not path.exists():
        return 0
    game = pd.read_parquet(path, columns=["game_id"])["game_id"]
    game = game.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)
    return int(game[game.str.startswith("002")].nunique())


def season_progress(games: int, full_games: int = FULL_SEASON_GAMES) -> float:
    return float(min(1.0, max(0.0, games / full_games)))


def season_state(
    season: int,
    games: int,
    fit_through: int = FROZEN_FIT_SEASON,
    full_games: int = FULL_SEASON_GAMES,
) -> dict[str, object]:
    """Status, schedule progress and possession floors for one season.

    Only a season after the frozen fit can be in progress; every season inside
    the fit keeps the full-season floors whatever its game count (lockout and
    2020 seasons are complete seasons).
    """
    progress = season_progress(games, full_games)
    in_progress = int(season) > int(fit_through) and games < full_games
    scale = progress if in_progress else 1.0
    return {
        "season": int(season),
        "games": int(games),
        "progress": progress if in_progress else 1.0,
        "status": STATUS_IN_PROGRESS if in_progress else STATUS_COMPLETE,
        "row_min_poss": max(PARTIAL_ROW_MIN_POSS, ROW_MIN_POSS * scale) if in_progress else ROW_MIN_POSS,
        "qual_min_poss": max(PARTIAL_QUAL_MIN_POSS, QUAL_MIN_POSS * scale) if in_progress else QUAL_MIN_POSS,
    }


def season_states(
    raw_dir: Path,
    first_season: int,
    last_season: int,
    fit_through: int = FROZEN_FIT_SEASON,
) -> dict[int, dict[str, object]]:
    """State for every season in a span; only post-freeze seasons read the raw file."""
    states: dict[int, dict[str, object]] = {}
    for season in range(int(first_season), int(last_season) + 1):
        games = rs_games_played(raw_dir, season) if season > fit_through else FULL_SEASON_GAMES
        states[season] = season_state(season, games, fit_through)
    return states


def floors_by_year(states: dict[int, dict[str, object]], key: str) -> dict[int, float]:
    return {int(season): float(state[key]) for season, state in states.items()}
