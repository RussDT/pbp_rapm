"""Shared WNBA player-season shooting stats source for RAPM processing."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
WNBA_STATS_PATH = ROOT.parent / "wnba" / "wnba-stats.csv"
WNBA_BACKEND_DATA_DIR = ROOT.parent / "wnba" / "data"
PBPSTATS_TOTALS_URL = "https://api.pbpstats.com/get-totals/wnba"

_CACHE: dict[tuple[int, bool, str], pd.DataFrame] = {}


def _normalize_pct(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.where(values <= 1.0, values / 100.0)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["PlayerID", "nba_id", "FTPerc", "ThreePerc", "source"])


def _numeric_col(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[name], errors="coerce")


def _backend_pbp_path(season_year: int, is_playoffs: bool) -> Path:
    year_label = f"{season_year}ps" if is_playoffs else str(season_year)
    return WNBA_BACKEND_DATA_DIR / f"{year_label}_pbp.csv"


def _normalize_pbpstats_totals(stats: pd.DataFrame, source: str) -> pd.DataFrame:
    if stats.empty or "EntityId" not in stats.columns:
        return _empty()

    out = stats.copy()
    out["PlayerID"] = pd.to_numeric(out["EntityId"], errors="coerce").astype("Int64")
    out["nba_id"] = out["PlayerID"].astype(str).mask(out["PlayerID"].isna(), np.nan)
    fta = _numeric_col(out, "FTA")
    ftm = _numeric_col(out, "FtPoints")
    out["FTPerc"] = np.where(fta > 0, ftm / fta, np.nan)

    if "Fg3Pct" in out.columns:
        out["ThreePerc"] = _normalize_pct(out["Fg3Pct"])
    else:
        fg3a = _numeric_col(out, "FG3A")
        fg3m = _numeric_col(out, "FG3M")
        out["ThreePerc"] = np.where(fg3a > 0, fg3m / fg3a, np.nan)

    out["source"] = source
    out = out.dropna(subset=["PlayerID"]).drop_duplicates("PlayerID", keep="last")
    return out[["PlayerID", "nba_id", "FTPerc", "ThreePerc", "source"]]


def _from_local_stats(season_year: int, is_playoffs: bool, path: Path) -> pd.DataFrame:
    if not path.exists():
        return _empty()

    usecols = ["year", "is_playoffs", "nba_id", "FT_PERC", "3P_PERC"]
    stats = pd.read_csv(path, usecols=usecols)
    stats["year"] = pd.to_numeric(stats["year"], errors="coerce")
    stats["is_playoffs"] = stats["is_playoffs"].astype(str).str.lower().isin({"true", "1", "yes"})
    stats = stats[
        (stats["year"] == int(season_year)) &
        (stats["is_playoffs"] == bool(is_playoffs))
    ].copy()
    if stats.empty:
        return _empty()

    stats["PlayerID"] = pd.to_numeric(stats["nba_id"], errors="coerce").astype("Int64")
    stats["nba_id"] = stats["PlayerID"].astype(str).mask(stats["PlayerID"].isna(), np.nan)
    stats["FTPerc"] = _normalize_pct(stats["FT_PERC"])
    stats["ThreePerc"] = _normalize_pct(stats["3P_PERC"])
    stats["source"] = str(path)
    stats = stats.dropna(subset=["PlayerID"]).drop_duplicates("PlayerID", keep="last")
    return stats[["PlayerID", "nba_id", "FTPerc", "ThreePerc", "source"]]


def _from_backend_pbp_csv(season_year: int, is_playoffs: bool) -> pd.DataFrame:
    path = _backend_pbp_path(season_year, is_playoffs)
    if not path.exists():
        return _empty()

    stats = pd.read_csv(path)
    return _normalize_pbpstats_totals(stats, str(path))


def _from_pbpstats(season_year: int, is_playoffs: bool) -> pd.DataFrame:
    season_type = "Playoffs" if is_playoffs else "Regular Season"
    params = {"Season": str(season_year), "SeasonType": season_type, "Type": "Player"}
    response = None
    last_exc: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = requests.get(
                PBPSTATS_TOTALS_URL,
                params=params,
                headers={"User-Agent": "pbp-rapm-wnba-stats/1.0"},
                timeout=60,
            )
            if response.status_code < 500:
                break
        except requests.RequestException as exc:
            last_exc = exc
            response = None
        if attempt < 3:
            time.sleep(3 * attempt)
    if response is None:
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("PBP Stats request did not return a response")
    response.raise_for_status()
    rows = response.json().get("multi_row_table_data", [])
    if not rows:
        return _empty()

    stats = pd.DataFrame(rows)
    stats["year"] = f"{season_year}ps" if is_playoffs else str(season_year)
    path = _backend_pbp_path(season_year, is_playoffs)
    path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(path, index=False)
    logging.info("Saved WNBA PBP Stats player totals cache to %s", path)
    return _normalize_pbpstats_totals(stats, "pbpstats")


def load_wnba_player_shooting_stats(
    season_year: int,
    is_playoffs: bool,
    local_path: Path = WNBA_STATS_PATH,
) -> pd.DataFrame:
    """Return WNBA player-season FT% and 3P% from the canonical stats pipeline.

    Historical rows come from `/Users/russellthomas/Docs/wnba/wnba-stats.csv`.
    If that final file has not been rebuilt for the requested season, use the
    WNBA backend's PBP Stats player totals cache under
    `/Users/russellthomas/Docs/wnba/data/`. If the cache is missing, pull the
    same totals from PBP Stats and write the backend-style cache file. Raw
    play-by-play is not used to estimate player shooting priors.
    """
    cache_key = (int(season_year), bool(is_playoffs), str(local_path))
    if cache_key in _CACHE:
        return _CACHE[cache_key].copy()

    stats = _from_local_stats(season_year, is_playoffs, local_path)
    if not stats.empty:
        logging.info(
            "Loaded %d WNBA shooting-stat rows for %s %s from %s",
            len(stats),
            season_year,
            "PS" if is_playoffs else "RS",
            local_path,
        )
        _CACHE[cache_key] = stats
        return stats.copy()

    stats = _from_backend_pbp_csv(season_year, is_playoffs)
    if not stats.empty:
        path = _backend_pbp_path(season_year, is_playoffs)
        logging.info(
            "Loaded %d WNBA shooting-stat rows for %s %s from %s",
            len(stats),
            season_year,
            "PS" if is_playoffs else "RS",
            path,
        )
        _CACHE[cache_key] = stats
        return stats.copy()

    try:
        stats = _from_pbpstats(season_year, is_playoffs)
    except Exception as exc:
        logging.warning("Could not fetch WNBA shooting stats from PBP Stats: %s", exc)
        stats = _empty()

    if not stats.empty:
        logging.info(
            "Fetched %d WNBA shooting-stat rows for %s %s from PBP Stats",
            len(stats),
            season_year,
            "PS" if is_playoffs else "RS",
        )
    else:
        logging.warning(
            "No WNBA shooting-stat rows for %s %s from local stats or PBP Stats",
            season_year,
            "PS" if is_playoffs else "RS",
        )
    _CACHE[cache_key] = stats
    return stats.copy()
