#!/usr/bin/env python3
"""Walk-forward NBA playoff translation, archetype, and rotation diagnostics.

This is a research-only analysis.  It starts from ratings frozen before each
postseason, estimates persistent player playoff translations using only earlier
postseasons, and tests whether entering-playoffs factor and role profiles add
information on later playoff possessions.  It never replaces the canonical
RAPM/VPM board.

The primary historical stress test is deliberately temporal:

* development: 2007-2018 postseasons;
* stress: 2019-2026 postseasons;
* bubble sensitivity: stress excluding 2020.

The primary factor and role panels use target season ``Y`` regular-season data
only for postseason ``Y``.  A five-season panel ending in ``Y-1`` remains a
separately emitted sensitivity rather than being blended after seeing stress.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from nba_pipeline.scripts.playoff_epm_prior import (
        DEFENSE_COLUMNS,
        OFFENSE_COLUMNS,
        _epm_possession_expectation,
        build_walk_forward_playoff_epm_ratings,
        load_playoff_possessions,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from playoff_epm_prior import (  # type: ignore[no-redef]
        DEFENSE_COLUMNS,
        OFFENSE_COLUMNS,
        _epm_possession_expectation,
        build_walk_forward_playoff_epm_ratings,
        load_playoff_possessions,
    )


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = DEFAULT_ROOT / "results" / "playoff_droppers"
DEFAULT_FACTOR_DIR = (
    DEFAULT_ROOT / "validation" / "rapms_reference" / "master_results"
)
DEFAULT_CURRENT_FACTOR_PROFILES = (
    DEFAULT_ROOT / "results" / "playoff_droppers" / "current_rs_factor_profiles.csv"
)
DEFAULT_RATINGS = (
    DEFAULT_ROOT / "results" / "championship_help" / "playoff_epm_prior_ratings.csv"
)
DEFAULT_PORTABILITY_DIR = DEFAULT_ROOT / "results" / "offensive_impact_portability"
DEFAULT_BOXSCORE = DEFAULT_ROOT.parents[1] / "vpm" / "boxscore.csv"

DEVELOPMENT_START = 2007
DEVELOPMENT_END = 2018
STRESS_START = 2019
STRESS_END = 2026
PLAYOFF_ALPHA_OFFENSE = 24_000.0
PLAYOFF_ALPHA_DEFENSE = 24_000.0
PLAYOFF_HISTORY_WINDOW = 8

FACTOR_COLUMNS = (
    "oTS",
    "oTOV",
    "oREB",
    "oTHREE_FREQ",
    "dTS",
    "dTOV",
    "dREB",
)

BASE_FACTOR_FEATURES = (
    "off_ots_z",
    "off_otov_z",
    "off_oreb_z",
    "off_three_freq_z",
    "def_dts_z",
    "def_dtov_z",
    "def_dreb_z",
)
OFFENSIVE_ROLE_FEATURES = (
    "off_creation_load",
    "off_assisted_dependence",
    "off_transition_dependence",
)
PROFILE_ROLE_COLUMNS = (
    "creation_load",
    "assisted_dependence",
    "transition_dependence",
)
DEFENSIVE_EXPLOIT_FEATURES = (
    "def_rebound_only",
    "rebound_only_x_spacing",
)
RIDGE_ALPHA_GRID = (
    0.0,
    100.0,
    300.0,
    1_000.0,
    3_000.0,
    10_000.0,
    30_000.0,
    100_000.0,
    300_000.0,
    1_000_000.0,
    1_000_000_000.0,
)

ROLE_RAW_COLUMNS = (
    "final_creation_TSA100",
    "PASSING_on-ball-time%",
    "PASSING_Potential Assists/100",
    "v2_pct_assisted_2pt",
    "v2_pct_assisted_3pt",
    "v2_pct_pts_fast_break",
)


@dataclass(frozen=True)
class LinearAdjustment:
    """Small transparent residual model fitted without an intercept."""

    features: tuple[str, ...]
    coefficients: np.ndarray
    target: str
    ridge_alpha: float = 0.0

    def coefficient_table(self, label: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "model": label,
                "target": self.target,
                "feature": self.features,
                "coefficient_points_per100": self.coefficients * 100.0,
                "ridge_alpha": self.ridge_alpha,
            }
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _season_window_name(target_season: int, window: int = 5) -> str:
    start = int(target_season) - int(window)
    end = int(target_season) - 1
    return f"{start % 100:02d}_{end % 100:02d}"


def _weighted_location_scale(
    values: pd.Series,
    weights: pd.Series,
) -> tuple[float, float]:
    valid = values.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return 0.0, 1.0
    x = values.loc[valid].to_numpy(dtype=float)
    w = weights.loc[valid].to_numpy(dtype=float)
    mean = float(np.average(x, weights=w))
    variance = float(np.average((x - mean) ** 2, weights=w))
    scale = float(np.sqrt(max(variance, 1e-12)))
    return mean, scale


def load_lagged_factor_profiles(
    factor_dir: Path,
    seasons: Iterable[int],
    *,
    window: int = 5,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Load and season-standardize factors ending before each postseason."""
    frames: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    required = {"player_id", "player_name", *FACTOR_COLUMNS, "off_poss", "def_poss"}
    for season in sorted(set(int(value) for value in seasons)):
        label = _season_window_name(season, window=window)
        path = factor_dir / f"weighted_factors_{label}_all_rb_se_a2000_4000.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing lagged factor profile for {season}: {path}")
        frame = pd.read_csv(path)
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Lagged factor file {path} is missing {sorted(missing)}")
        frame = frame[list(required)].copy()
        frame["player_id"] = pd.to_numeric(frame["player_id"], errors="raise").astype(int)
        if frame["player_id"].duplicated().any():
            raise ValueError(f"Lagged factor file is not unique by player: {path}")
        for column in (*FACTOR_COLUMNS, "off_poss", "def_poss"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        for column in FACTOR_COLUMNS:
            weight_column = "off_poss" if column.startswith("o") else "def_poss"
            mean, scale = _weighted_location_scale(frame[column], frame[weight_column])
            frame[f"{column}_z"] = ((frame[column] - mean) / scale).clip(-4.0, 4.0)
        # Positive-good dTS: negative z is the hypothesized shot-defense weakness.
        frame["low_dts"] = (-frame["dTS_z"]).clip(lower=0.0)
        frame["high_dreb"] = frame["dREB_z"].clip(lower=0.0)
        frame["rebound_only"] = frame["low_dts"] * frame["high_dreb"]
        frame["spacing_positive"] = frame["oTHREE_FREQ_z"].clip(lower=0.0)
        frame.insert(0, "season", season)
        frame["source_window"] = label
        frame["source_path"] = str(path)
        frames.append(frame)
        audits.append(
            {
                "season": season,
                "source_window": label,
                "path": str(path),
                "sha256": _sha256(path),
                "players": int(len(frame)),
                "max_source_year": season - 1,
            }
        )
    return pd.concat(frames, ignore_index=True), audits


def load_current_rs_factor_profiles(
    path: Path,
    seasons: Iterable[int],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Load factors built only from each target season's completed RS."""
    frame = pd.read_csv(path)
    required = {
        "season",
        "player_id",
        "player_name",
        *FACTOR_COLUMNS,
        "off_poss",
        "def_poss",
        "source_kind",
        "source_max_season",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Current-RS factor panel is missing {sorted(missing)}")
    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
    frame["player_id"] = pd.to_numeric(frame["player_id"], errors="raise").astype(int)
    requested = sorted(set(int(value) for value in seasons))
    frame = frame.loc[frame["season"].isin(requested)].copy()
    if frame.duplicated(["season", "player_id"]).any():
        raise ValueError("Current-RS factor panel is not unique by season/player")
    for column in (*FACTOR_COLUMNS, "off_poss", "def_poss", "source_max_season"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if (frame["source_max_season"] > frame["season"]).any():
        raise ValueError("Current-RS factor panel contains a future-season source")
    parts = []
    audits = []
    checksum = _sha256(path)
    for season, group in frame.groupby("season", sort=True):
        part = group.copy()
        for column in FACTOR_COLUMNS:
            weight_column = "off_poss" if column.startswith("o") else "def_poss"
            mean, scale = _weighted_location_scale(part[column], part[weight_column])
            part[f"{column}_z"] = ((part[column] - mean) / scale).clip(-4.0, 4.0)
        part["low_dts"] = (-part["dTS_z"]).clip(lower=0.0)
        part["high_dreb"] = part["dREB_z"].clip(lower=0.0)
        part["rebound_only"] = part["low_dts"] * part["high_dreb"]
        part["spacing_positive"] = part["oTHREE_FREQ_z"].clip(lower=0.0)
        part["source_window"] = f"{int(season)}_RS"
        part["source_path"] = str(path)
        parts.append(part)
        audits.append(
            {
                "season": int(season),
                "source_window": f"{int(season)}_RS",
                "path": str(path),
                "sha256": checksum,
                "players": int(len(part)),
                "max_source_year": int(part["source_max_season"].max()),
            }
        )
    missing_seasons = sorted(set(requested).difference(frame["season"].unique()))
    # The translation history can begin before the archetype panel.  Missing
    # seasons are allowed, but every development/stress season must be present.
    required_model_seasons = set(range(DEVELOPMENT_START, STRESS_END + 1))
    absent_required = sorted(required_model_seasons.intersection(missing_seasons))
    if absent_required:
        raise ValueError(f"Current-RS factor panel misses model seasons {absent_required}")
    return pd.concat(parts, ignore_index=True), audits


def load_offensive_role_profiles(
    path: Path,
    seasons: Iterable[int],
    *,
    source_offset: int = 0,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Build season-relative offensive archetypes from regular-season box data.

    ``source_offset=0`` is the primary entering-playoffs contract.  ``-1`` is
    used only by the explicitly lagged sensitivity.  The composite definitions
    are fixed basketball hypotheses, not estimates selected on playoff outcomes.
    """
    required = {"nba_id", "year", "Name", "Minutes", *ROLE_RAW_COLUMNS}
    frame = pd.read_csv(path, usecols=lambda column: column in required, low_memory=False)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Offensive-role panel is missing {sorted(missing)}")
    frame = frame.rename(
        columns={"nba_id": "player_id", "year": "source_year", "Name": "player_name"}
    )
    frame["player_id"] = pd.to_numeric(frame["player_id"], errors="raise").astype(int)
    frame["source_year"] = pd.to_numeric(frame["source_year"], errors="raise").astype(int)
    frame["Minutes"] = pd.to_numeric(frame["Minutes"], errors="coerce")
    for column in ROLE_RAW_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    requested = sorted(set(int(value) for value in seasons))
    parts: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    checksum = _sha256(path)
    for season in requested:
        source_year = season + int(source_offset)
        part = frame.loc[frame["source_year"].eq(source_year)].copy()
        if part.empty:
            continue
        if part["player_id"].duplicated().any():
            raise ValueError(f"Box profile is not unique by player in {source_year}")
        for column in ROLE_RAW_COLUMNS:
            mean, scale = _weighted_location_scale(part[column], part["Minutes"])
            part[f"{column}_z"] = ((part[column] - mean) / scale).clip(-4.0, 4.0)

        creation_inputs = part[
            [
                "final_creation_TSA100_z",
                "PASSING_on-ball-time%_z",
                "PASSING_Potential Assists/100_z",
            ]
        ]
        # Two near-complete handling/passing fields keep the composite available;
        # final creation volume contributes where the historical panel has it.
        part["creation_load"] = creation_inputs.mean(axis=1, skipna=True).where(
            creation_inputs.notna().sum(axis=1).ge(2)
        )
        assisted_inputs = pd.concat(
            [
                part[["v2_pct_assisted_2pt_z", "v2_pct_assisted_3pt_z"]],
                -part[["PASSING_on-ball-time%_z"]].rename(
                    columns={"PASSING_on-ball-time%_z": "low_on_ball_z"}
                ),
            ],
            axis=1,
        )
        part["assisted_dependence"] = assisted_inputs.mean(axis=1, skipna=True).where(
            assisted_inputs.notna().sum(axis=1).ge(2)
        )
        part["transition_dependence"] = part["v2_pct_pts_fast_break_z"]
        part["season"] = season
        part["role_source_year"] = source_year
        parts.append(
            part[
                [
                    "season",
                    "player_id",
                    "creation_load",
                    "assisted_dependence",
                    "transition_dependence",
                    "role_source_year",
                ]
            ]
        )
        audits.append(
            {
                "season": season,
                "source_year": source_year,
                "path": str(path),
                "sha256": checksum,
                "players": int(len(part)),
                "complete_creation": int(part["creation_load"].notna().sum()),
                "complete_assisted": int(part["assisted_dependence"].notna().sum()),
                "complete_transition": int(part["transition_dependence"].notna().sum()),
            }
        )
    if not parts:
        raise ValueError("No offensive-role seasons overlap the requested analysis")
    return pd.concat(parts, ignore_index=True), audits


def merge_factor_and_role_profiles(
    factors: pd.DataFrame,
    roles: pd.DataFrame,
) -> pd.DataFrame:
    role_columns = [
        "season",
        "player_id",
        "creation_load",
        "assisted_dependence",
        "transition_dependence",
        "role_source_year",
    ]
    return factors.merge(
        roles[role_columns],
        on=["season", "player_id"],
        how="left",
        validate="one_to_one",
    )


def attach_lineup_archetypes(
    possessions: pd.DataFrame,
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    """Attach lineup factor/role means and the rebound-only/spacing interaction."""
    required = {
        "season",
        "player_id",
        "dTS_z",
        "dTOV_z",
        "dREB_z",
        "oTS_z",
        "oTOV_z",
        "oREB_z",
        "oTHREE_FREQ_z",
        "low_dts",
        "high_dreb",
        "rebound_only",
        "spacing_positive",
        "creation_load",
        "assisted_dependence",
        "transition_dependence",
    }
    missing = required.difference(profiles.columns)
    if missing:
        raise ValueError(f"Factor profiles are missing {sorted(missing)}")
    frame = possessions.copy().reset_index(drop=True)
    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
    output_parts: list[pd.DataFrame] = []
    profile_columns = (
        "dTS_z",
        "dTOV_z",
        "dREB_z",
        "oTS_z",
        "oTOV_z",
        "oREB_z",
        "oTHREE_FREQ_z",
        "low_dts",
        "high_dreb",
        "rebound_only",
        "spacing_positive",
        "creation_load",
        "assisted_dependence",
        "transition_dependence",
    )
    for season, group in frame.groupby("season", sort=True):
        season_profiles = profiles.loc[profiles["season"].eq(int(season))]
        lookup = season_profiles.set_index("player_id")
        part = group.copy()
        off_coverage = np.zeros(len(part), dtype=np.int8)
        def_coverage = np.zeros(len(part), dtype=np.int8)
        off_role_coverage = np.zeros(len(part), dtype=np.int8)
        slot_values: dict[tuple[str, str], list[np.ndarray]] = {}
        for side, slots in (("off", OFFENSE_COLUMNS), ("def", DEFENSE_COLUMNS)):
            coverage = off_coverage if side == "off" else def_coverage
            for source_column in slots:
                ids = pd.to_numeric(part[source_column], errors="coerce").fillna(0).astype(int)
                mapped_marker = ids.map(pd.Series(1, index=lookup.index)).fillna(0).to_numpy(dtype=np.int8)
                coverage += mapped_marker
                if side == "off":
                    role_complete = lookup[list(PROFILE_ROLE_COLUMNS)].notna().all(axis=1)
                    role_marker = ids.map(role_complete).eq(True).to_numpy(dtype=np.int8)
                    off_role_coverage += role_marker
                for profile_column in profile_columns:
                    values = ids.map(lookup[profile_column]).to_numpy(dtype=float)
                    slot_values.setdefault((side, profile_column), []).append(values)
        part["off_factor_coverage"] = off_coverage
        part["def_factor_coverage"] = def_coverage
        part["off_role_coverage"] = off_role_coverage

        def mean_slots(side: str, column: str) -> np.ndarray:
            values = np.column_stack(slot_values[(side, column)])
            counts = np.isfinite(values).sum(axis=1)
            return np.divide(
                np.nansum(values, axis=1),
                counts,
                out=np.full(len(values), np.nan, dtype=float),
                where=counts > 0,
            )

        part["off_ots_z"] = mean_slots("off", "oTS_z")
        part["off_otov_z"] = mean_slots("off", "oTOV_z")
        part["off_oreb_z"] = mean_slots("off", "oREB_z")
        part["off_three_freq_z"] = mean_slots("off", "oTHREE_FREQ_z")
        part["def_dts_z"] = mean_slots("def", "dTS_z")
        part["def_dtov_z"] = mean_slots("def", "dTOV_z")
        part["def_dreb_z"] = mean_slots("def", "dREB_z")
        part["def_low_dts"] = mean_slots("def", "low_dts")
        part["def_high_dreb"] = mean_slots("def", "high_dreb")
        part["def_rebound_only"] = mean_slots("def", "rebound_only")
        part["off_spacing_z"] = mean_slots("off", "spacing_positive")
        part["off_creation_load"] = mean_slots("off", "creation_load")
        part["off_assisted_dependence"] = mean_slots("off", "assisted_dependence")
        part["off_transition_dependence"] = mean_slots("off", "transition_dependence")
        part["rebound_only_x_spacing"] = (
            part["def_rebound_only"] * part["off_spacing_z"]
        )
        output_parts.append(part)
    return pd.concat(output_parts, ignore_index=True)


def _center_within_season(values: np.ndarray, seasons: pd.Series) -> np.ndarray:
    series = pd.Series(values, index=seasons.index, dtype=float)
    means = series.groupby(seasons).transform("mean")
    return series.to_numpy(dtype=float) - means.to_numpy(dtype=float)


def build_prediction_frame(
    possessions: pd.DataFrame,
    priors: pd.DataFrame,
    adjusted: pd.DataFrame,
) -> pd.DataFrame:
    """Create level-neutral base, O-only, D-only, and O+D predictions."""
    frame = possessions.copy().reset_index(drop=True)
    seasons = pd.to_numeric(frame["season"], errors="raise").astype(int)
    actual = pd.to_numeric(frame["points"], errors="raise").to_numpy(dtype=float)
    base = _epm_possession_expectation(frame, priors)

    columns = [
        "season",
        "player_id",
        "playoff_adjusted_o_dpm",
        "playoff_adjusted_d_dpm",
        "playoff_adjusted_dpm",
    ]
    adjusted_priors = adjusted[columns].rename(
        columns={
            "playoff_adjusted_o_dpm": "o_dpm",
            "playoff_adjusted_d_dpm": "d_dpm",
            "playoff_adjusted_dpm": "dpm",
        }
    )
    offense_only = adjusted_priors[["season", "player_id", "o_dpm"]].merge(
        priors[["season", "player_id", "d_dpm"]],
        on=["season", "player_id"],
        how="left",
        validate="one_to_one",
    )
    offense_only["dpm"] = offense_only["o_dpm"] + offense_only["d_dpm"]
    defense_only = priors[["season", "player_id", "o_dpm"]].merge(
        adjusted_priors[["season", "player_id", "d_dpm"]],
        on=["season", "player_id"],
        how="left",
        validate="one_to_one",
    )
    defense_only["dpm"] = defense_only["o_dpm"] + defense_only["d_dpm"]

    predictions = {
        "prediction_base": base,
        "prediction_history_offense": _epm_possession_expectation(frame, offense_only),
        "prediction_history_defense": _epm_possession_expectation(frame, defense_only),
        "prediction_history_both": _epm_possession_expectation(frame, adjusted_priors),
    }
    frame["actual_centered"] = _center_within_season(actual, seasons)
    for column, values in predictions.items():
        frame[column] = _center_within_season(values, seasons)
    frame["game_key"] = seasons.astype(str) + "_" + frame["game_id"].astype(str)
    return frame


def center_features_by_season(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    result = frame.copy()
    for column in feature_columns:
        values = pd.to_numeric(result[column], errors="coerce")
        result[f"{column}_centered"] = values - values.groupby(result["season"]).transform("mean")
    return result


def fit_linear_adjustment(
    frame: pd.DataFrame,
    *,
    target_prediction: str,
    features: Sequence[str],
    ridge_alpha: float = 0.0,
) -> LinearAdjustment:
    centered = center_features_by_season(frame, features)
    feature_columns = [f"{column}_centered" for column in features]
    valid = centered[["actual_centered", target_prediction, *feature_columns]].notna().all(axis=1)
    if int(valid.sum()) <= len(feature_columns):
        raise ValueError("Not enough complete rows to fit the archetype adjustment")
    x = centered.loc[valid, feature_columns].to_numpy(dtype=float)
    y = (
        centered.loc[valid, "actual_centered"].to_numpy(dtype=float)
        - centered.loc[valid, target_prediction].to_numpy(dtype=float)
    )
    if ridge_alpha < 0:
        raise ValueError("ridge_alpha must be nonnegative")
    if ridge_alpha >= 1_000_000_000.0:
        # Explicit null candidate in the one-SE grid.  Keeping it exactly zero
        # avoids presenting numerical dust as a validated archetype effect.
        coefficients = np.zeros(x.shape[1], dtype=float)
    elif ridge_alpha == 0:
        coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    else:
        gram = x.T @ x
        coefficients = np.linalg.solve(
            gram + float(ridge_alpha) * np.eye(gram.shape[0]),
            x.T @ y,
        )
    return LinearAdjustment(
        tuple(features), coefficients, target_prediction, float(ridge_alpha)
    )


def select_ridge_alpha(
    frame: pd.DataFrame,
    *,
    target_prediction: str,
    features: Sequence[str],
    alphas: Sequence[float] = RIDGE_ALPHA_GRID,
) -> tuple[float, pd.DataFrame]:
    """Select a nested adjustment by leave-one-postseason-out development MSE.

    The one-standard-error rule chooses the largest (most shrunk) alpha whose
    mean season-held-out MSE is within one standard error of the minimum.
    """
    complete = frame.dropna(
        subset=["actual_centered", target_prediction, *features]
    ).copy()
    seasons = sorted(int(value) for value in complete["season"].unique())
    if len(seasons) < 3:
        raise ValueError("Ridge selection requires at least three complete seasons")
    rows: list[dict[str, float | int]] = []
    for alpha in alphas:
        fold_delta: list[float] = []
        for season in seasons:
            training = complete.loc[complete["season"].ne(season)]
            held_out = complete.loc[complete["season"].eq(season)]
            model = fit_linear_adjustment(
                training,
                target_prediction=target_prediction,
                features=features,
                ridge_alpha=float(alpha),
            )
            predicted = apply_linear_adjustment(
                held_out, model, output_column="_ridge_candidate"
            )
            candidate_residual = (
                predicted["actual_centered"] - predicted["_ridge_candidate"]
            )
            baseline_residual = (
                predicted["actual_centered"] - predicted[target_prediction]
            )
            fold_delta.append(
                float(np.mean(candidate_residual**2 - baseline_residual**2))
            )
        values = np.asarray(fold_delta, dtype=float)
        rows.append(
            {
                "ridge_alpha": float(alpha),
                "cv_seasons": int(len(values)),
                "mean_heldout_mse_delta": float(values.mean()),
                "paired_season_standard_error": float(
                    values.std(ddof=1) / np.sqrt(len(values))
                ),
            }
        )
    report = pd.DataFrame(rows)
    best_index = report["mean_heldout_mse_delta"].idxmin()
    threshold = float(
        report.loc[best_index, "mean_heldout_mse_delta"]
        + report.loc[best_index, "paired_season_standard_error"]
    )
    eligible = report.loc[report["mean_heldout_mse_delta"].le(threshold)]
    selected = float(eligible["ridge_alpha"].max())
    report["one_se_threshold"] = threshold
    report["selected"] = report["ridge_alpha"].eq(selected)
    return selected, report


def apply_linear_adjustment(
    frame: pd.DataFrame,
    model: LinearAdjustment,
    *,
    output_column: str,
) -> pd.DataFrame:
    result = center_features_by_season(frame, model.features)
    x = result[[f"{column}_centered" for column in model.features]].to_numpy(dtype=float)
    result[output_column] = result[model.target].to_numpy(dtype=float) + x @ model.coefficients
    return result


def _equal_season_delta(frame: pd.DataFrame, candidate: str, baseline: str) -> float:
    delta = (
        (frame["actual_centered"] - frame[candidate]) ** 2
        - (frame["actual_centered"] - frame[baseline]) ** 2
    )
    return float(delta.groupby(frame["season"]).mean().mean())


def evaluate_prediction(
    frame: pd.DataFrame,
    *,
    candidate: str,
    baseline: str = "prediction_base",
    bootstrap_samples: int = 1000,
    seed: int = 20260829,
) -> dict[str, object]:
    required = ["actual_centered", candidate, baseline, "season", "game_key"]
    data = frame.dropna(subset=required).copy()
    if data.empty:
        raise ValueError(f"No complete evaluation rows for {candidate}")
    data["squared_error_delta"] = (
        (data["actual_centered"] - data[candidate]) ** 2
        - (data["actual_centered"] - data[baseline]) ** 2
    )
    game = (
        data.groupby(["season", "game_key"], as_index=False)
        .agg(
            squared_error_delta=("squared_error_delta", "sum"),
            possessions=("squared_error_delta", "size"),
        )
    )
    point = _equal_season_delta(data, candidate, baseline)
    rng = np.random.default_rng(seed)
    draws = np.empty(int(bootstrap_samples), dtype=float)
    season_groups = [group.reset_index(drop=True) for _, group in game.groupby("season", sort=True)]
    for draw_index in range(int(bootstrap_samples)):
        season_values = []
        for season_game in season_groups:
            sampled = season_game.iloc[rng.integers(0, len(season_game), size=len(season_game))]
            season_values.append(
                sampled["squared_error_delta"].sum() / sampled["possessions"].sum()
            )
        draws[draw_index] = float(np.mean(season_values))
    pooled = float(data["squared_error_delta"].mean())
    return {
        "candidate": candidate,
        "baseline": baseline,
        "seasons": int(data["season"].nunique()),
        "games": int(data["game_key"].nunique()),
        "possessions": int(len(data)),
        "equal_season_mse_delta": point,
        "equal_season_ci_low": float(np.quantile(draws, 0.025)),
        "equal_season_ci_high": float(np.quantile(draws, 0.975)),
        "pooled_mse_delta": pooled,
        "interpretation": "negative favors candidate",
    }


def evaluate_prediction_by_season(
    frame: pd.DataFrame,
    *,
    candidate: str,
    baseline: str = "prediction_base",
) -> pd.DataFrame:
    """Return transparent season-level MSE deltas on identical rows."""
    required = ["actual_centered", candidate, baseline, "season", "game_key"]
    data = frame.dropna(subset=required).copy()
    data["squared_error_delta"] = (
        (data["actual_centered"] - data[candidate]) ** 2
        - (data["actual_centered"] - data[baseline]) ** 2
    )
    return (
        data.groupby("season", as_index=False)
        .agg(
            games=("game_key", "nunique"),
            possessions=("squared_error_delta", "size"),
            mse_delta=("squared_error_delta", "mean"),
        )
        .assign(candidate=candidate, baseline=baseline)
    )


def build_rotation_translation(
    regular_path: Path,
    playoff_path: Path,
    profiles: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build observed-player rotation changes on the common EPM-covered sample."""
    required = ["season", "player_id", "player_name", "off_team_id", "possessions", "game_id"]
    phases = []
    for label, path in (("RS", regular_path), ("PS", playoff_path)):
        data = pd.read_parquet(path, columns=required)
        data["season"] = pd.to_numeric(data["season"], errors="raise").astype(int)
        data["player_id"] = pd.to_numeric(data["player_id"], errors="raise").astype(int)
        data["off_team_id"] = pd.to_numeric(data["off_team_id"], errors="coerce")
        data = data.dropna(subset=["off_team_id"]).copy()
        aggregate = (
            data.groupby(
                ["season", "player_id", "player_name", "off_team_id"], as_index=False
            )
            .agg(
                player_possessions=("possessions", "sum"),
                player_games=("game_id", "nunique"),
            )
        )
        team = (
            aggregate.groupby(["season", "off_team_id"], as_index=False)["player_possessions"]
            .sum()
            .rename(columns={"player_possessions": "five_player_possessions"})
        )
        aggregate = aggregate.merge(team, on=["season", "off_team_id"], validate="many_to_one")
        aggregate["rotation_share"] = (
            5.0 * aggregate["player_possessions"] / aggregate["five_player_possessions"]
        )
        aggregate["possessions_per_game"] = (
            aggregate["player_possessions"] / aggregate["player_games"]
        )
        phases.append(aggregate.rename(columns={
            "player_possessions": f"{label.lower()}_possessions",
            "player_games": f"{label.lower()}_games",
            "five_player_possessions": f"{label.lower()}_five_player_possessions",
            "rotation_share": f"{label.lower()}_rotation_share",
            "possessions_per_game": f"{label.lower()}_possessions_per_game",
        }))
    keys = ["season", "player_id", "player_name", "off_team_id"]
    merged = phases[0].merge(phases[1], on=keys, how="inner", validate="one_to_one")
    merged["rotation_share_delta"] = merged["ps_rotation_share"] - merged["rs_rotation_share"]
    merged["possessions_per_game_delta"] = (
        merged["ps_possessions_per_game"] - merged["rs_possessions_per_game"]
    )
    profile_columns = [
        "season",
        "player_id",
        "dTS_z",
        "dREB_z",
        "low_dts",
        "high_dreb",
        "rebound_only",
        "source_window",
    ]
    merged = merged.merge(
        profiles[profile_columns],
        on=["season", "player_id"],
        how="left",
        validate="many_to_one",
    )
    positive = merged["rebound_only"].dropna()
    threshold = float(positive.loc[positive.gt(0)].median()) if positive.gt(0).any() else np.inf
    merged["rebound_only_archetype"] = merged["rebound_only"].ge(threshold)
    summary = (
        merged.groupby("rebound_only_archetype", dropna=False, as_index=False)
        .agg(
            player_seasons=("player_id", "size"),
            players=("player_id", "nunique"),
            mean_rotation_share_delta=("rotation_share_delta", "mean"),
            median_rotation_share_delta=("rotation_share_delta", "median"),
            mean_possessions_per_game_delta=("possessions_per_game_delta", "mean"),
            median_possessions_per_game_delta=("possessions_per_game_delta", "median"),
        )
    )
    summary["positive_rebound_only_median_threshold"] = threshold
    return merged, summary


def build_environment_change(
    regular_path: Path,
    playoff_path: Path,
) -> pd.DataFrame:
    """Compare mechanisms for the same player-seasons in RS and playoffs."""
    metrics = (
        "points_per100",
        "first_chance_points_per100",
        "second_chance_points_per100",
        "turnover_per100",
        "badpass_turnover_per100",
        "transition_per100",
        "ft_value_per100",
        "rim_freq_value_per100",
        "rim_fg_value_per100",
        "mid_fg_value_per100",
        "three_freq_value_per100",
        "three_fg_value_per100",
    )
    phase_tables = {}
    for label, path in (("rs", regular_path), ("ps", playoff_path)):
        columns = ["season", "player_id", "possessions", *metrics]
        frame = pd.read_parquet(path, columns=columns)
        rows = []
        for (season, player_id), group in frame.groupby(["season", "player_id"], sort=False):
            weights = group["possessions"].to_numpy(dtype=float)
            row = {
                "season": int(season),
                "player_id": int(player_id),
                f"{label}_possessions": float(weights.sum()),
            }
            for metric in metrics:
                row[f"{label}_{metric}"] = float(
                    np.average(group[metric].to_numpy(dtype=float), weights=weights)
                )
            rows.append(row)
        phase_tables[label] = pd.DataFrame(rows)
    paired = phase_tables["rs"].merge(
        phase_tables["ps"], on=["season", "player_id"], how="inner", validate="one_to_one"
    )
    weights = np.minimum(paired["rs_possessions"], paired["ps_possessions"]).to_numpy(dtype=float)
    rows = []
    for metric in metrics:
        rs_values = paired[f"rs_{metric}"].to_numpy(dtype=float)
        ps_values = paired[f"ps_{metric}"].to_numpy(dtype=float)
        rows.append(
            {
                "metric": metric,
                "regular_season": float(np.average(rs_values, weights=weights)),
                "playoffs": float(np.average(ps_values, weights=weights)),
                "playoff_minus_regular": float(np.average(ps_values - rs_values, weights=weights)),
                "player_seasons": int(len(paired)),
                "players": int(paired["player_id"].nunique()),
                "paired_weight": float(weights.sum()),
            }
        )
    return pd.DataFrame(rows)


def _latest_translation_table(
    adjusted: pd.DataFrame,
    profiles: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    names = profiles.loc[profiles["season"].eq(season), ["player_id", "player_name"]]
    table = adjusted.loc[adjusted["season"].eq(season)].merge(
        names, on="player_id", how="left", validate="one_to_one"
    )
    columns = [
        "season",
        "player_id",
        "player_name",
        "o_dpm",
        "d_dpm",
        "dpm",
        "playoff_translation_o_dpm",
        "playoff_translation_d_dpm",
        "playoff_translation_dpm",
        "playoff_adjusted_o_dpm",
        "playoff_adjusted_d_dpm",
        "playoff_adjusted_dpm",
        "playoff_prior_seasons",
        "playoff_prior_possessions",
    ]
    return table[columns].sort_values("playoff_translation_dpm", ascending=True)


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str], digits: int = 4) -> str:
    view = frame[list(columns)].copy()
    for column in view.select_dtypes(include=["number"]).columns:
        view[column] = view[column].map(
            lambda value: f"{value:.{digits}f}" if pd.notna(value) else ""
        )
    header = "| " + " | ".join(view.columns) + " |"
    rule = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in view.to_numpy()]
    return "\n".join([header, rule, *rows])


def write_report(
    output_dir: Path,
    evaluations: pd.DataFrame,
    incremental: pd.DataFrame,
    coefficients: pd.DataFrame,
    alpha_cv: pd.DataFrame,
    environment: pd.DataFrame,
    rotation_summary: pd.DataFrame,
    latest: pd.DataFrame,
    coverage: dict[str, object],
) -> None:
    current_profiles = coverage.get("profile_source") == "current_rs"
    profile_contract = (
        "Archetype features come from target season `Y` regular-season-only factor "
        "solves; no target playoff possession enters those features."
        if current_profiles
        else "Archetype features come from the five-season factor board ending in `Y-1`; "
        "no target postseason or current regular season enters those features."
    )
    profile_limit = (
        "The current-RS factors are exact entering-playoffs measurements but noisier than "
        "the lagged five-year profile. The lagged and current candidates remain separate "
        "rather than being blended after seeing stress."
        if current_profiles
        else "The lagged factor boards contain previous playoff evidence, but end before "
        "the target season and are therefore date-safe. A current-RS-only factor panel is "
        "the preferred next challenger because it is fresher and avoids mixing earlier "
        "playoffs into the archetype measurement."
    )
    first_improvement = (
        "Preserve the current-RS and lagged-five-year candidates separately; any future "
        "stabilized blend must be declared and selected on development seasons only."
        if current_profiles
        else "Build current-regular-season-only factor priors for every postseason, with "
        "fixed factor alphas chosen outside playoff outcomes."
    )
    stress = evaluations.loc[evaluations["sample"].eq("stress")].copy()
    stress["delta_e6"] = stress["equal_season_mse_delta"] * 1_000_000
    stress["low_e6"] = stress["equal_season_ci_low"] * 1_000_000
    stress["high_e6"] = stress["equal_season_ci_high"] * 1_000_000
    incremental_stress = incremental.loc[incremental["sample"].eq("stress")].copy()
    incremental_stress["delta_e6"] = (
        incremental_stress["equal_season_mse_delta"] * 1_000_000
    )
    incremental_stress["low_e6"] = (
        incremental_stress["equal_season_ci_low"] * 1_000_000
    )
    incremental_stress["high_e6"] = (
        incremental_stress["equal_season_ci_high"] * 1_000_000
    )
    alpha_summary_rows = []
    for model_label, group in alpha_cv.groupby("model", sort=False):
        minimum = group.loc[group["mean_heldout_mse_delta"].idxmin()]
        selected = group.loc[group["selected"]].iloc[0]
        alpha_summary_rows.append(
            {
                "model": model_label,
                "cv_min_alpha": minimum["ridge_alpha"],
                "cv_min_delta_e6": minimum["mean_heldout_mse_delta"] * 1_000_000,
                "cv_min_se_e6": minimum["paired_season_standard_error"] * 1_000_000,
                "selected_alpha": selected["ridge_alpha"],
                "cv_seasons": selected["cv_seasons"],
            }
        )
    alpha_summary = pd.DataFrame(alpha_summary_rows)
    eligible = latest.loc[latest["playoff_prior_possessions"].ge(1000)].copy()
    offense_droppers = eligible.nsmallest(10, "playoff_translation_o_dpm")
    defense_droppers = eligible.nsmallest(10, "playoff_translation_d_dpm")
    report = f"""# NBA Playoff Dropper Translation — Baseline Research Receipt

## Status

Research-only. No rating, prior, public leaderboard, or production config is changed.
The primary question is whether earlier-postseason player translations and entering
archetypes improve later-playoff possession prediction beyond ratings frozen before the
postseason.

## Temporal contract

- Development: {DEVELOPMENT_START}-{DEVELOPMENT_END} playoffs.
- Historical stress: {STRESS_START}-{STRESS_END} playoffs.
- Bubble sensitivity: stress excluding 2020.
- Player-history translations use only earlier postseasons, at alpha
  {PLAYOFF_ALPHA_OFFENSE:.0f}/{PLAYOFF_ALPHA_DEFENSE:.0f} and an
  {PLAYOFF_HISTORY_WINDOW}-postseason window.
- {profile_contract}
- Scoring is future actual playoff possession points. Season levels are centered, and
  uncertainty is clustered by game with equal postseason weight.

## Historical stress gate

Negative MSE delta favors the candidate relative to pre-playoff EPM.

{_markdown_table(stress, ['model', 'delta_e6', 'low_e6', 'high_e6', 'seasons', 'games', 'possessions'], digits=2)}

The player-history rows are the direct test of the "real individual playoff
dropper" claim. A favorable point estimate is not enough; the translation must
improve later postseasons. The stress result does not currently validate those
individual labels, so the inspection tails below must not be called established
82-game/16-game player effects.

## Entering-playoffs rebound-only hypothesis

`dTS` is positive-good. The predeclared rebound-only score is therefore
`max(-z(dTS), 0) * max(z(dREB), 0)`. The spacing-exploitation term multiplies
the defensive lineup's mean rebound-only score by the offensive lineup's positive
three-frequency score. Main offensive and defensive factor terms remain in the model,
so the interaction is not allowed to stand in for ordinary defensive quality.

### Direct incremental gate

These comparisons isolate what each additional idea contributes. Negative favors the
candidate named on the left.

{_markdown_table(incremental_stress, ['comparison', 'delta_e6', 'low_e6', 'high_e6', 'seasons', 'games', 'possessions'], digits=2)}

The key distinction is between the ordinary factor update and each specific archetype
hypothesis. Only a negative held-out incremental delta supports the candidate; an interval
crossing zero keeps it exploratory, and a positive interval rejects it in this form.

Each nested update selects ridge strength by leave-one-postseason-out development MSE and
the one-standard-error most-shrunk rule. An alpha of 1,000,000,000 is the explicit null
candidate: the development evidence did not justify carrying that update forward.

{_markdown_table(alpha_summary, ['model', 'cv_min_alpha', 'cv_min_delta_e6', 'cv_min_se_e6', 'selected_alpha', 'cv_seasons'], digits=2)}

{_markdown_table(coefficients, ['model', 'target', 'feature', 'coefficient_points_per100', 'ridge_alpha'], digits=4)}

## How the offensive environment changes

These are paired regular-season/playoff changes for the same 2019-2026 player-seasons,
weighted by their common phase exposure. They describe selected playoff participants;
they are not causal player effects.

{_markdown_table(environment, ['metric', 'regular_season', 'playoffs', 'playoff_minus_regular'], digits=3)}

## Rotation survival

This first pass uses players observed in both phase tables. It therefore measures role
compression among players who appear and does **not** yet include rostered players reduced
to zero playoff minutes. Transaction-aware roster reconstruction is required before the
selection model is complete.

{_markdown_table(rotation_summary, ['rebound_only_archetype', 'player_seasons', 'players', 'mean_rotation_share_delta', 'mean_possessions_per_game_delta'], digits=4)}

## Entering-{int(latest['season'].max())} inspection tails

These are shrinkage estimates from earlier postseasons, not retrospective grades of the
current postseason. A 1,000-possession history floor is used only for this inspection.

### Offensive translation — most negative

{_markdown_table(offense_droppers, ['player_name', 'playoff_translation_o_dpm', 'playoff_translation_d_dpm', 'playoff_translation_dpm', 'playoff_prior_possessions'], digits=3)}

### Defensive translation — most negative

{_markdown_table(defense_droppers, ['player_name', 'playoff_translation_o_dpm', 'playoff_translation_d_dpm', 'playoff_translation_dpm', 'playoff_prior_possessions'], digits=3)}

## Coverage and interpretation limits

- Strict factor prediction requires all ten players to have an entering factor profile.
- Offensive-role candidates additionally require all five offensive players to have
  complete creation, assisted-dependence, and transition profiles.
- Strict rows: {coverage['strict_rows']:,} of {coverage['all_rows']:,}
  ({coverage['strict_share']:.1%}).
- {profile_limit}
- Player translation remains partly team, coach, health, and matchup translation. Stable
  year-ahead possession prediction is the admission test; a famous bad series is not.
- The stress period is historical, not untouched. Any selected improvement must be frozen
  before a genuinely future postseason.

## Improvement ledger

1. {first_improvement}
2. Add transaction-aware playoff roster membership so zero-minute benchings enter the
   rotation-survival target.
3. Separate player translation from team/coach postseason random effects and health.
4. Add opponent spacing, screening involvement, matchup assignment burden, and lineup
   size as interactions rather than generic player-credit priors.
5. Tune playoff-update precision only on early development seasons, use a one-standard-
   error most-shrunk rule, then freeze separate offense/defense penalties.
6. Add split-postseason and leave-one-series-out reliability before publishing any player
   label.
7. Decompose the translation into actual point-valued TS, TOV, and second-chance targets
   on one shared possession universe with exact row closure.
"""
    (output_dir / "REPORT.md").write_text(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_ROOT / "processed")
    parser.add_argument("--ratings", type=Path, default=DEFAULT_RATINGS)
    parser.add_argument("--factor-dir", type=Path, default=DEFAULT_FACTOR_DIR)
    parser.add_argument(
        "--current-factor-profiles", type=Path, default=DEFAULT_CURRENT_FACTOR_PROFILES
    )
    parser.add_argument(
        "--profile-source",
        choices=("current_rs", "lagged"),
        default="current_rs",
    )
    parser.add_argument("--portability-dir", type=Path, default=DEFAULT_PORTABILITY_DIR)
    parser.add_argument("--boxscore", type=Path, default=DEFAULT_BOXSCORE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-season", type=int, default=2002)
    parser.add_argument("--end-season", type=int, default=2026)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260829)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_ratings = pd.read_csv(args.ratings)
    priors = source_ratings[["season", "player_id", "o_dpm", "d_dpm", "dpm"]].copy()
    possessions = load_playoff_possessions(
        args.processed_dir, args.start_season, args.end_season
    )
    adjusted, translation_report = build_walk_forward_playoff_epm_ratings(
        possessions,
        priors,
        alpha_offense=PLAYOFF_ALPHA_OFFENSE,
        alpha_defense=PLAYOFF_ALPHA_DEFENSE,
        history_window=PLAYOFF_HISTORY_WINDOW,
    )
    if args.profile_source == "current_rs":
        profiles, factor_audits = load_current_rs_factor_profiles(
            args.current_factor_profiles,
            range(args.start_season, args.end_season + 1),
        )
    else:
        profiles, factor_audits = load_lagged_factor_profiles(
            args.factor_dir,
            range(args.start_season, args.end_season + 1),
        )
    role_offset = 0 if args.profile_source == "current_rs" else -1
    roles, role_audits = load_offensive_role_profiles(
        args.boxscore,
        range(max(args.start_season, 2014 - role_offset), args.end_season + 1),
        source_offset=role_offset,
    )
    profiles = merge_factor_and_role_profiles(profiles, roles)
    frame = build_prediction_frame(possessions, priors, adjusted)
    frame = attach_lineup_archetypes(frame, profiles)
    strict = frame.loc[
        frame["off_factor_coverage"].eq(5) & frame["def_factor_coverage"].eq(5)
    ].copy()
    development = strict.loc[
        strict["season"].between(DEVELOPMENT_START, DEVELOPMENT_END)
    ].copy()
    stress = strict.loc[strict["season"].between(STRESS_START, STRESS_END)].copy()
    stress_no_bubble = stress.loc[stress["season"].ne(2020)].copy()

    coefficient_parts = []
    alpha_cv_parts = []
    model_columns: dict[str, str] = {
        "history_offense": "prediction_history_offense",
        "history_defense": "prediction_history_defense",
        "history_both": "prediction_history_both",
    }
    nested_specs = (
        ("factor_main", "prediction_base", BASE_FACTOR_FEATURES),
        ("rebound_only", "prediction_factor_main", ("def_rebound_only",)),
        (
            "spacing_exploitation",
            "prediction_rebound_only",
            ("rebound_only_x_spacing",),
        ),
        (
            "offensive_archetypes",
            "prediction_factor_main",
            OFFENSIVE_ROLE_FEATURES,
        ),
        (
            "joint_archetypes",
            "prediction_offensive_archetypes",
            DEFENSIVE_EXPLOIT_FEATURES,
        ),
    )
    for label, target_prediction, features in nested_specs:
        selected_alpha, alpha_cv = select_ridge_alpha(
            development,
            target_prediction=target_prediction,
            features=features,
        )
        alpha_cv.insert(0, "model", label)
        alpha_cv.insert(1, "target", target_prediction)
        alpha_cv_parts.append(alpha_cv)
        model = fit_linear_adjustment(
            development,
            target_prediction=target_prediction,
            features=features,
            ridge_alpha=selected_alpha,
        )
        coefficient_parts.append(model.coefficient_table(label))
        development = apply_linear_adjustment(
            development, model, output_column=f"prediction_{label}"
        )
        stress = apply_linear_adjustment(stress, model, output_column=f"prediction_{label}")
        stress_no_bubble = apply_linear_adjustment(
            stress_no_bubble, model, output_column=f"prediction_{label}"
        )
        model_columns[label] = f"prediction_{label}"

    for sample in (development, stress, stress_no_bubble):
        sample["prediction_history_plus_joint_archetypes"] = (
            sample["prediction_history_both"]
            + sample["prediction_joint_archetypes"]
            - sample["prediction_base"]
        )
    model_columns["history_plus_joint_archetypes"] = (
        "prediction_history_plus_joint_archetypes"
    )

    evaluation_rows = []
    incremental_rows = []
    season_rows = []
    direct_comparisons = (
        ("rebound_only_vs_factor_main", "rebound_only", "factor_main"),
        (
            "spacing_exploitation_vs_factor_main",
            "spacing_exploitation",
            "factor_main",
        ),
        (
            "spacing_exploitation_vs_rebound_only",
            "spacing_exploitation",
            "rebound_only",
        ),
        (
            "offensive_archetypes_vs_factor_main",
            "offensive_archetypes",
            "factor_main",
        ),
        (
            "joint_archetypes_vs_offensive_archetypes",
            "joint_archetypes",
            "offensive_archetypes",
        ),
        (
            "joint_archetypes_vs_spacing_exploitation",
            "joint_archetypes",
            "spacing_exploitation",
        ),
        (
            "history_plus_archetype_vs_history",
            "history_plus_joint_archetypes",
            "history_both",
        ),
        (
            "history_plus_archetype_vs_archetype",
            "history_plus_joint_archetypes",
            "joint_archetypes",
        ),
    )
    for sample_label, sample in (
        ("development_in_sample", development),
        ("stress", stress),
        ("stress_excluding_2020", stress_no_bubble),
    ):
        for model_label, prediction_column in model_columns.items():
            result = evaluate_prediction(
                sample,
                candidate=prediction_column,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
            )
            result["sample"] = sample_label
            result["model"] = model_label
            evaluation_rows.append(result)
            season = evaluate_prediction_by_season(
                sample,
                candidate=prediction_column,
            )
            season["sample"] = sample_label
            season["model"] = model_label
            season_rows.append(season)
        for comparison, candidate_label, baseline_label in direct_comparisons:
            result = evaluate_prediction(
                sample,
                candidate=model_columns[candidate_label],
                baseline=model_columns[baseline_label],
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
            )
            result["sample"] = sample_label
            result["comparison"] = comparison
            incremental_rows.append(result)
    evaluations = pd.DataFrame(evaluation_rows)
    incremental = pd.DataFrame(incremental_rows)
    evaluation_seasons = pd.concat(season_rows, ignore_index=True)
    coefficients = pd.concat(coefficient_parts, ignore_index=True)
    alpha_cv = pd.concat(alpha_cv_parts, ignore_index=True)

    regular_game = args.portability_dir / "player_game_regular_season.parquet"
    playoff_game = args.portability_dir / "player_game_playoffs.parquet"
    rotation, rotation_summary = build_rotation_translation(
        regular_game, playoff_game, profiles
    )
    environment = build_environment_change(regular_game, playoff_game)
    latest = _latest_translation_table(adjusted, profiles, args.end_season)

    adjusted.to_csv(args.output_dir / "walk_forward_player_translations.csv", index=False)
    latest.to_csv(args.output_dir / "latest_entering_playoffs_translations.csv", index=False)
    evaluations.to_csv(args.output_dir / "prediction_gate.csv", index=False)
    incremental.to_csv(args.output_dir / "incremental_prediction_gate.csv", index=False)
    evaluation_seasons.to_csv(args.output_dir / "prediction_gate_by_season.csv", index=False)
    coefficients.to_csv(args.output_dir / "archetype_coefficients.csv", index=False)
    alpha_cv.to_csv(args.output_dir / "archetype_alpha_cv.csv", index=False)
    environment.to_csv(args.output_dir / "environment_changes.csv", index=False)
    rotation.to_csv(args.output_dir / "rotation_translation.csv", index=False)
    rotation_summary.to_csv(args.output_dir / "rotation_archetype_summary.csv", index=False)
    pd.DataFrame(factor_audits).to_csv(args.output_dir / "factor_provenance.csv", index=False)

    coverage_frame = frame.loc[frame["season"].ge(DEVELOPMENT_START)]
    coverage_strict = strict.loc[strict["season"].ge(DEVELOPMENT_START)]
    coverage = {
        "all_rows": int(len(coverage_frame)),
        "strict_rows": int(len(coverage_strict)),
        "strict_share": float(len(coverage_strict) / len(coverage_frame)),
        "profile_source": args.profile_source,
        "strict_offensive_role_rows": int(
            coverage_strict["off_role_coverage"].eq(5).sum()
        ),
    }
    manifest = {
        "status": "research_only",
        "timing": (
            "walk-forward player history; target-season regular-season factor/role profiles"
            if args.profile_source == "current_rs"
            else "walk-forward player history; factor/role profiles end before target season"
        ),
        "development": [DEVELOPMENT_START, DEVELOPMENT_END],
        "stress": [STRESS_START, STRESS_END],
        "translation": translation_report,
        "coverage": coverage,
        "ratings_source": {"path": str(args.ratings), "sha256": _sha256(args.ratings)},
        "factor_sources": factor_audits,
        "offensive_role_sources": role_audits,
        "bootstrap_samples": int(args.bootstrap_samples),
        "seed": int(args.seed),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    write_report(
        args.output_dir,
        evaluations,
        incremental,
        coefficients,
        alpha_cv,
        environment,
        rotation_summary,
        latest,
        coverage,
    )
    print(f"Wrote playoff-dropper research receipt to {args.output_dir}")


if __name__ == "__main__":
    main()
