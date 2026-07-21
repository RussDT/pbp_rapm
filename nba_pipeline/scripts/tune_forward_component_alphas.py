#!/usr/bin/env python3
"""Shared data/target helpers for stabilized DECOMP and VPM research.

The private research repo imports this module from the production
`pbp_rapm/nba_pipeline/scripts` tree. It deliberately contains no experiment
selection logic; it only loads aligned processed families and constructs the
same targets used by `rapm.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import rapm
except ImportError:  # Private research scripts import this as a top-level module.
    import rapm


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PIPELINE_ROOT / "processed"

FIRST_CHANCE_COMPONENTS = {
    "DECOMP_EFG",
    "DECOMP_RIM_FREQ",
    "DECOMP_RIM_FG",
    "DECOMP_MID_FG",
    "DECOMP_THREE_FREQ",
    "DECOMP_THREE_FG",
    "DECOMP_MID_VALUE",
    "ALT_FT",
    "ALT_FT_FREQ",
    "ALT_FT_SEVERITY",
    "ALT_TOV_VALUE",
    "ALT_BADPASS_TOV_VALUE",
    "ALT_SCORING_TOV_VALUE",
}


def family_of(component: str) -> str:
    """Return the physical processed-file family for a research component."""
    component = component.upper()
    if component in FIRST_CHANCE_COMPONENTS:
        return "FIRST_CHANCE"
    if component == "SECOND_CHANCE_CLEAN":
        return "SECOND_CHANCE_CLEAN"
    if component == "RAPM":
        return "RAPM"
    raise ValueError(f"Unsupported stabilized-RAPM component: {component}")


def _year_suffix(year: int) -> int:
    return int(year) % 100


def load_family_raw(family: str, years) -> pd.DataFrame:
    """Load RS+PS processed parquets for a family and annotate source season."""
    family = family.upper()
    frames = []
    missing = []
    for year in years:
        year = int(year)
        suffix = _year_suffix(year)
        found_for_year = False
        for phase, phase_suffix in [("RS", ""), ("PS", "_PS")]:
            path = Path(PROCESSED_DIR) / f"{family}{suffix:02d}{phase_suffix}.parquet"
            if not path.exists():
                continue
            frame = pd.read_parquet(path)
            frame["season_phase"] = phase
            frame["source_season_end_year"] = rapm.normalize_season_end_year(year)
            frames.append(frame)
            found_for_year = True
        if not found_for_year:
            missing.append(year)
    if not frames:
        raise FileNotFoundError(
            f"No {family} parquets found in {PROCESSED_DIR} for years {list(years)}"
        )
    if missing:
        raise FileNotFoundError(
            f"Missing every {family} phase for season-end years: {missing} in {PROCESSED_DIR}"
        )
    return pd.concat(frames, ignore_index=True)


def train_baselines(
    raw_df: pd.DataFrame,
    component: str,
    half_life: float | None,
) -> dict[str, float] | None:
    """Compute the exact training-split turnover placeholder for a component."""
    if family_of(component) != "FIRST_CHANCE":
        return None
    return rapm.compute_alt_first_chance_baselines(
        raw_df,
        timedecay=half_life is not None,
        half_life=half_life,
        requested_prefixes=[component.upper()],
    )


def prepare_frame(
    raw_df: pd.DataFrame,
    component: str,
    family: str | None = None,
    baselines: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Attach Off_Diff/Def_Diff for one component without changing row order."""
    component = component.upper()
    family = family.upper() if family is not None else family_of(component)
    expected_family = family_of(component)
    if family != expected_family:
        raise ValueError(
            f"{component} belongs to {expected_family}, not requested family {family}"
        )
    if family == "FIRST_CHANCE":
        if baselines is None:
            raise ValueError(f"{component} requires training-split FIRST_CHANCE baselines")
        return rapm.detect_file_type_and_prepare(
            raw_df,
            pure=False,
            prefix=component,
            alt_baselines=baselines,
        )
    return rapm.detect_file_type_and_prepare(
        raw_df,
        pure=False,
        prefix=component,
    )


def prepare_target(
    raw_df: pd.DataFrame,
    component: str,
    family: str | None = None,
    baselines: dict[str, float] | None = None,
) -> np.ndarray:
    """Return the offense-oriented possession target used by the RAPM solver."""
    prepared = prepare_frame(raw_df, component, family, baselines)
    target = pd.to_numeric(prepared["Off_Diff"], errors="coerce")
    if target.isna().any():
        raise ValueError(
            f"{component} target contains {int(target.isna().sum())} missing values"
        )
    return target.to_numpy(dtype=np.float64)


__all__ = [
    "PROCESSED_DIR",
    "family_of",
    "load_family_raw",
    "prepare_frame",
    "prepare_target",
    "train_baselines",
]
