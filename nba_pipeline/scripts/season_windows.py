"""Season-window token helpers shared by the Supabase uploaders.

Season year N means season (N-1)-N, so 2027 is the 2026-27 season. Every
two-digit token in a result folder/file name is derived from the season year,
exactly as `nba_pipeline/daily_rapm_update.sh` derives its `yy`/`win` tokens:

    two_digit(2027)          -> "27"
    window_token(2027, 1)    -> "27"
    window_token(2027, 3)    -> "25_27"

Keep this module in sync with the shell helpers; the nightly and the uploaders
must agree on the filenames for the same season.
"""

import os

DEFAULT_SEASON_YEAR_ENV = "NBA_RAPM_SEASON_YEAR"
FALLBACK_SEASON_YEAR = 2027


def default_season_year() -> int:
    """Season year from NBA_RAPM_SEASON_YEAR, else the current default."""
    raw = os.environ.get(DEFAULT_SEASON_YEAR_ENV)
    if not raw:
        return FALLBACK_SEASON_YEAR
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{DEFAULT_SEASON_YEAR_ENV} must be a four-digit season end year, got {raw!r}"
        ) from exc


def two_digit(season_year: int) -> str:
    """Two-digit season-end form used in result folder and file names."""
    return f"{int(season_year) % 100:02d}"


def window_token(season_year: int, length: int) -> str:
    """`27` for a single season, `25_27` for a 3-season window, etc."""
    end_yy = two_digit(season_year)
    if length <= 1:
        return end_yy
    return f"{two_digit(int(season_year) - length + 1)}_{end_yy}"
