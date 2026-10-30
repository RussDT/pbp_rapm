#!/usr/bin/env python3
"""WNBA time-decay, prior-informed, predictive net-DECOMP VPM research lane.

Kalman predictive box/skill panel states become factor-specific prior centers
inside a multi-year time-decay prior-centered ridge on the exact ten-target
WNBA net-continuation DECOMP family, gated on forward actual-possession
prediction with the pure tuner's scoring.

Protocol:

- Targets, possession loading, fold construction, and scoring reuse
  ``tune_wnba_daily_net_hyperparams`` machinery (ten-target structural family
  closing exactly to actual points; rolling-origin completed-season folds;
  game-weighted future possession MSE).
- Labels are per-season single-season PURE zero-centered solves of the same
  ten-target family at the tuner-selected 2000/5000 penalties,
  exposure-centered per side, per component side.
- Kalman as-of prior features: each player's latest ``post_rate`` strictly
  before the target season's first game date (14 box stats + 8 estimated
  skills). 2022 is panel warm-up and never receives a Kalman prior.
- Factor-specific compact ridge prior models are trained strictly
  year-forward (states as-of season T start -> season T labels, T < fold
  season); ridge strength is chosen inside the year-forward training folds,
  never on the fold's target season.
- The solve inserts priors exactly the way ``clean_decomp_wnba`` does
  (prior_scale multiplies the prior, exposure-centering per side, then
  ``rhs += penalties * prior``), generalized to the multi-target time-decay
  gram system with weighted exposure centering.
- Grid/selection: base off/def alphas in {2000/5000, 5000/40000}, half-life
  in {none, 1100d, 550d}, prior scale in {1.0, 1.5}; selection ONLY on the
  2024 and 2025 forward folds with the tuner's paired one-SE rule.
- Fold 2026 is SPENT report-only evidence in this lane (the pure tuner's
  alpha selection already saw 2026); the untouched confirmation season is
  2027.

Research-only. The live published WNBA VPM (box-prior gross lane) is not
touched and remains canonical. This lane is the sanctioned research
exception to the "Kalman panels are never rating priors" rule; any promotion
requires the untouched 2027 forward gate plus explicit owner sign-off. No
proprietary ratings (DPM/PI/NPI/EPM) appear anywhere as features, targets,
or tuning inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from wnba_test.build_wnba_box_priors import feature_columns, load_box_features
from wnba_test.clean_decomp_wnba import _load_name_map
from wnba_test.master_wnba_rapm import (
    build_design_matrix,
    transform_to_home_away_format,
)
from wnba_test.tune_wnba_daily_net_hyperparams import (
    DEFAULT_PROCESSED_DIR,
    DEFAULT_SQ_DIR,
    _file_hash,
    _player_mapping,
    build_component_targets,
    load_target_rows,
    paired_one_se,
)

WNBA_ROOT = ROOT / "wnba_test"
DEFAULT_OUTPUT_DIR = WNBA_ROOT / "Results" / "predictive_net_vpm"
KALMAN_BOX_PATH = (
    WNBA_ROOT / "Results" / "gameday" / "kalman_box" / "predictions.parquet"
)
SKILLS_PATH = (
    WNBA_ROOT / "Results" / "gameday" / "estimated_skills" / "predictions.parquet"
)

# Label solve = the pure tuner's one-SE selection (2024-2026 folds; that
# selection makes 2026 spent evidence for this lane as well).
LABEL_OFF_ALPHA = 2000.0
LABEL_DEF_ALPHA = 5000.0

# Predeclared solve grid. Selection only on the 2024/2025 forward folds.
BASE_ALPHA_GRID: List[Tuple[float, float]] = [(2000.0, 5000.0), (5000.0, 40000.0)]
HALF_LIFE_GRID: List[Optional[float]] = [None, 1100.0, 550.0]
PRIOR_SCALE_GRID: List[float] = [1.0, 1.5]
DEV_FOLDS = [2024, 2025]
REPORT_ONLY_FOLD = 2026

# The existing single-season box-prior lane's frozen identity, embedded in
# this machinery as the structural comparator.
SINGLE_SEASON_OFF_ALPHA = 5000.0
SINGLE_SEASON_DEF_ALPHA = 40000.0
SINGLE_SEASON_PRIOR_SCALE = 1.5

# Predeclared compact prior-model regularization. Selected inside the
# year-forward training folds; when a fold has no interior year (predicting
# 2024 trains on 2023 alone) the strong predeclared fallback applies.
PRIOR_MODEL_ALPHAS = [1.0, 10.0, 100.0, 1000.0]
PRIOR_MODEL_FALLBACK_ALPHA = 100.0

BOOTSTRAP_DRAWS = 4000
SEED = 20260826

KB_STATS = [
    "ast", "blk", "dreb", "fg3a", "fg3m", "fga", "fgm", "fta", "ftm",
    "oreb", "pf", "pts", "stl", "tov",
]
SK_SKILLS = [
    "ft_pct", "fta_rate", "mid_freq", "mid_pct", "rim_freq", "rim_pct",
    "three_freq", "three_pct",
]
KALMAN_FEATURE_COLUMNS = [f"kb_{s}" for s in KB_STATS] + [f"sk_{s}" for s in SK_SKILLS]

# Factor-specific Kalman state maps, mirroring the box-prior lane's
# component-relevant feature philosophy on the panel's available states.
KALMAN_FEATURES: Dict[Tuple[str, str], List[str]] = {
    ("RIM_FREQ", "off"): ["sk_rim_freq", "kb_fga", "kb_pts", "kb_ast"],
    ("RIM_FG", "off"): ["sk_rim_pct", "sk_rim_freq", "kb_fgm", "kb_ast"],
    ("MID_FREQ", "off"): ["sk_mid_freq", "kb_fga", "kb_pts"],
    ("MID_FG", "off"): ["sk_mid_pct", "sk_mid_freq", "kb_ast"],
    ("THREE_FREQ", "off"): ["sk_three_freq", "kb_fg3a", "kb_fga", "kb_ast"],
    ("THREE_FG", "off"): ["sk_three_pct", "kb_fg3a", "kb_ast"],
    ("FT", "off"): ["sk_fta_rate", "sk_ft_pct", "kb_fta", "kb_pts"],
    ("TOV", "off"): ["kb_tov", "kb_ast", "kb_fga", "kb_pts"],
    ("NET_BASE", "off"): ["kb_pts", "kb_fga", "kb_ast", "kb_tov"],
    ("SECOND_CHANCE", "off"): ["kb_oreb", "sk_rim_freq", "kb_fga"],
    ("RIM_FREQ", "def"): ["kb_blk", "kb_pf", "kb_dreb"],
    ("RIM_FG", "def"): ["kb_blk", "kb_dreb", "kb_pf"],
    ("MID_FREQ", "def"): ["kb_blk", "kb_stl", "kb_pf", "kb_dreb"],
    ("MID_FG", "def"): ["kb_blk", "kb_stl", "kb_dreb"],
    ("THREE_FREQ", "def"): ["kb_stl", "kb_pf", "kb_dreb"],
    ("THREE_FG", "def"): ["kb_stl", "kb_dreb", "kb_blk"],
    ("FT", "def"): ["kb_pf", "kb_blk", "kb_stl"],
    ("TOV", "def"): ["kb_stl", "kb_blk", "kb_pf"],
    ("NET_BASE", "def"): ["kb_stl", "kb_blk", "kb_dreb", "kb_pf"],
    ("SECOND_CHANCE", "def"): ["kb_dreb", "kb_blk", "kb_pf"],
}

# The box-snapshot arm reuses the existing lane's feature mapping through the
# corresponding gross prefixes.
BOX_PREFIX_FOR_TARGET = {
    "RIM_FREQ": "ALT_EFG_RIM_FREQ",
    "RIM_FG": "ALT_EFG_RIM_FG",
    "MID_FREQ": "ALT_EFG_MID_FREQ",
    "MID_FG": "ALT_EFG_MID_FG",
    "THREE_FREQ": "ALT_EFG_THREE_FREQ",
    "THREE_FG": "ALT_EFG_THREE_FG",
    "FT": "ALT_FT",
    "TOV": "ALT_TOV_VALUE",
    "NET_BASE": "ALT_FC_COMPLETION",
    "SECOND_CHANCE": "SECOND_CHANCE_CLEAN",
}


def _box_feature_map(target: str, side: str) -> List[str]:
    return feature_columns(BOX_PREFIX_FOR_TARGET[target], side)


def _kalman_feature_map(target: str, side: str) -> List[str]:
    return KALMAN_FEATURES[(target, side)]


# ---------------------------------------------------------------------------
# Fold machinery (lifted from tune_wnba_daily_net_hyperparams.prepare_folds,
# extended to retain the player mapping so priors can be inserted).
# ---------------------------------------------------------------------------


@dataclass
class PriorFold:
    label: str
    validation_year: int
    train_years: List[int]
    cutoff_date: pd.Timestamp
    train_dates: np.ndarray
    train_x: object
    train_targets: np.ndarray
    validation_x: object
    validation_actual: np.ndarray
    validation_games: np.ndarray
    mapping: Dict[str, int]
    off_indices: np.ndarray
    def_indices: np.ndarray
    off_players: List[str]
    def_players: List[str]
    cache: dict = field(default_factory=dict)

    def weighted_system(self, half_life: Optional[float]):
        if half_life not in self.cache:
            if half_life is None:
                weights = np.ones(len(self.train_dates), dtype=float)
            else:
                age_days = (
                    self.cutoff_date - pd.to_datetime(self.train_dates)
                ).days.astype(float)
                weights = np.asarray(
                    np.power(0.5, age_days / float(half_life)), dtype=float
                )
            target_means = np.average(self.train_targets, axis=0, weights=weights)
            centered = self.train_targets - target_means
            weighted_x = self.train_x.multiply(np.sqrt(weights)[:, None])
            gram = (weighted_x.T @ weighted_x).toarray()
            rhs = np.asarray(self.train_x.T @ (weights[:, None] * centered))
            exposure = np.asarray(self.train_x.T @ weights).reshape(-1)
            self.cache[half_life] = (gram, rhs, target_means, exposure)
        return self.cache[half_life]

    def raw_exposure(self) -> np.ndarray:
        return np.asarray(self.train_x.sum(axis=0)).reshape(-1)


def _mapping_sides(mapping: Dict[str, int]):
    off_keys = [key for key in mapping if key.endswith("_off")]
    def_keys = [key for key in mapping if key.endswith("_def")]
    off_indices = np.asarray([mapping[key] for key in off_keys], dtype=int)
    def_indices = np.asarray([mapping[key] for key in def_keys], dtype=int)
    off_players = [key.rsplit("_", 1)[0] for key in off_keys]
    def_players = [key.rsplit("_", 1)[0] for key in def_keys]
    return off_indices, def_indices, off_players, def_players


def prepare_ordered(rows: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ordered = rows.sort_values(["game_id", "event_num"], kind="stable").reset_index(
        drop=True
    )
    solver_frame = ordered.copy()
    solver_frame["Off_Diff"] = solver_frame["_actual_points"]
    solver_frame["Def_Diff"] = -solver_frame["_actual_points"]
    transformed = transform_to_home_away_format(solver_frame)
    source_games = ordered["game_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    transformed_games = (
        transformed["gameid"].astype(str).str.replace(r"\.0$", "", regex=True)
    )
    if not source_games.reset_index(drop=True).equals(
        transformed_games.reset_index(drop=True)
    ):
        raise ValueError("home/away transform changed row order; alignment is unsafe")
    return ordered, transformed


def build_fold(
    ordered: pd.DataFrame,
    transformed: pd.DataFrame,
    train_years: Iterable[int],
    validation_year: int,
    label: str,
) -> Tuple[PriorFold, List[str]]:
    train_years = sorted(int(year) for year in train_years)
    train_mask = ordered["season_end_year"].isin(train_years).to_numpy()
    validation_mask = ordered["season_end_year"].eq(validation_year).to_numpy()
    if not train_mask.any() or not validation_mask.any():
        raise ValueError(f"fold {label} has empty train or validation rows")
    train_transformed = transformed.loc[train_mask].reset_index(drop=True)
    validation_transformed = transformed.loc[validation_mask].reset_index(drop=True)
    train_prepared, target_names = build_component_targets(
        ordered.loc[train_mask].reset_index(drop=True), "net"
    )
    mapping = _player_mapping(train_transformed)
    train_x, _ = build_design_matrix(train_transformed, mapping)
    validation_x, _ = build_design_matrix(validation_transformed, mapping)
    train_dates = ordered.loc[train_mask, "game_date"].to_numpy()
    validation_dates = ordered.loc[validation_mask, "game_date"]
    if pd.Timestamp(pd.to_datetime(train_dates).max()) >= validation_dates.min():
        raise ValueError(f"fold {label} training overlaps validation dates")
    off_indices, def_indices, off_players, def_players = _mapping_sides(mapping)
    fold = PriorFold(
        label=label,
        validation_year=int(validation_year),
        train_years=train_years,
        cutoff_date=pd.Timestamp(validation_dates.min()),
        train_dates=train_dates,
        train_x=train_x,
        train_targets=train_prepared[target_names].to_numpy(float),
        validation_x=validation_x,
        validation_actual=ordered.loc[validation_mask, "_actual_points"].to_numpy(
            float
        ),
        validation_games=(
            ordered.loc[validation_mask, "game_id"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .to_numpy()
        ),
        mapping=mapping,
        off_indices=off_indices,
        def_indices=def_indices,
        off_players=off_players,
        def_players=def_players,
    )
    return fold, target_names


# ---------------------------------------------------------------------------
# Labels: per-season single-season pure zero-centered ten-target solves.
# ---------------------------------------------------------------------------


def solve_season_labels(
    ordered: pd.DataFrame,
    transformed: pd.DataFrame,
    year: int,
) -> Tuple[pd.DataFrame, List[str]]:
    mask = ordered["season_end_year"].eq(year).to_numpy()
    if not mask.any():
        raise ValueError(f"no rows for label season {year}")
    season_rows = ordered.loc[mask].reset_index(drop=True)
    season_transformed = transformed.loc[mask].reset_index(drop=True)
    prepared, target_names = build_component_targets(season_rows, "net")
    mapping = _player_mapping(season_transformed)
    design, _ = build_design_matrix(season_transformed, mapping)
    targets = prepared[target_names].to_numpy(float)
    means = targets.mean(axis=0)
    centered = targets - means
    gram = (design.T @ design).toarray()
    rhs = np.asarray(design.T @ centered)
    penalties = np.zeros(gram.shape[0], dtype=float)
    off_indices, def_indices, off_players, def_players = _mapping_sides(mapping)
    penalties[off_indices] = LABEL_OFF_ALPHA
    penalties[def_indices] = LABEL_DEF_ALPHA
    beta = np.linalg.solve(gram + np.diag(penalties), rhs)
    exposure = np.asarray(design.sum(axis=0)).reshape(-1)
    for indices in (off_indices, def_indices):
        weight = exposure[indices]
        total = float(weight.sum())
        if total > 0:
            mean = (beta[indices, :] * weight[:, None]).sum(axis=0) / total
            beta[indices, :] = beta[indices, :] - mean[None, :]
    short_names = [name.replace("target_", "") for name in target_names]
    records = []
    for indices, players, side in (
        (off_indices, off_players, "off"),
        (def_indices, def_players, "def"),
    ):
        for row_index, player in zip(indices, players):
            record = {
                "player_id": str(player),
                "side": side,
                "season": int(year),
                "poss": float(exposure[row_index]),
            }
            for t, name in enumerate(short_names):
                record[f"label_{name}"] = float(beta[row_index, t])
            records.append(record)
    return pd.DataFrame(records), target_names


# ---------------------------------------------------------------------------
# Kalman as-of feature snapshots.
# ---------------------------------------------------------------------------


def build_kalman_snapshots(
    boundaries: Dict[int, pd.Timestamp],
    kalman_box_path: Path,
    skills_path: Path,
) -> Tuple[Dict[int, pd.DataFrame], List[dict]]:
    kb = pd.read_parquet(kalman_box_path)
    sk = pd.read_parquet(skills_path)
    variants = sorted(sk["variant"].astype(str).unique())
    if variants != ["uncentered"]:
        raise ValueError(f"unexpected estimated-skills variants: {variants}")
    missing_stats = sorted(set(KB_STATS) - set(kb["stat"].unique()))
    missing_skills = sorted(set(SK_SKILLS) - set(sk["skill"].unique()))
    if missing_stats or missing_skills:
        raise ValueError(
            f"panel is missing stats {missing_stats} / skills {missing_skills}"
        )
    snapshots: Dict[int, pd.DataFrame] = {}
    audit: List[dict] = []
    for season in sorted(boundaries):
        boundary = pd.Timestamp(boundaries[season])
        kb_prior = kb.loc[kb["game_date"] < boundary]
        sk_prior = sk.loc[sk["game_date"] < boundary]
        if kb_prior.empty:
            raise ValueError(f"no Kalman box states strictly before {boundary}")
        latest_kb = kb_prior.sort_values("game_date", kind="stable").drop_duplicates(
            ["person_id", "stat"], keep="last"
        )
        wide_kb = latest_kb.pivot(index="person_id", columns="stat", values="post_rate")
        wide_kb.columns = [f"kb_{column}" for column in wide_kb.columns]
        var_kb = latest_kb.pivot(index="person_id", columns="stat", values="post_var")
        latest_sk = sk_prior.sort_values("game_date", kind="stable").drop_duplicates(
            ["person_id", "skill"], keep="last"
        )
        wide_sk = latest_sk.pivot(index="person_id", columns="skill", values="post_rate")
        wide_sk.columns = [f"sk_{column}" for column in wide_sk.columns]
        snap = wide_kb.join(wide_sk, how="outer")
        for column in KALMAN_FEATURE_COLUMNS:
            if column not in snap.columns:
                snap[column] = np.nan
        n_games = kb_prior.groupby("person_id")["game_id"].nunique()
        last_date = kb_prior.groupby("person_id")["game_date"].max()
        snap["asof_n_games"] = n_games.reindex(snap.index).fillna(0.0)
        snap["asof_last_game_date"] = last_date.reindex(snap.index)
        snap["asof_mean_post_var"] = var_kb.mean(axis=1).reindex(snap.index)
        missing_before = int(snap[KALMAN_FEATURE_COLUMNS].isna().sum().sum())
        column_means = snap[KALMAN_FEATURE_COLUMNS].mean()
        snap[KALMAN_FEATURE_COLUMNS] = snap[KALMAN_FEATURE_COLUMNS].fillna(column_means)
        snap.index = pd.Index(
            [str(int(value)) for value in snap.index], name="player_id"
        )
        snapshots[season] = snap
        audit.append(
            {
                "season": int(season),
                "boundary": str(boundary.date()),
                "players": int(len(snap)),
                "imputed_feature_cells": missing_before,
                "total_feature_cells": int(
                    len(snap) * len(KALMAN_FEATURE_COLUMNS)
                ),
                "max_state_date": str(pd.Timestamp(kb_prior["game_date"].max()).date()),
            }
        )
    return snapshots, audit


def load_box_snapshot(year: int) -> pd.DataFrame:
    frame = load_box_features(year)
    frame = frame.drop_duplicates("player_id", keep="first")
    return frame.set_index("player_id")


# ---------------------------------------------------------------------------
# Factor-specific compact prior models (year-forward alpha selection).
# ---------------------------------------------------------------------------


def _training_rows(
    features_by_year: Dict[int, pd.DataFrame],
    labels_by_year: Dict[int, pd.DataFrame],
    years: Iterable[int],
    target: str,
    side: str,
    features: List[str],
) -> pd.DataFrame:
    pieces = []
    for year in years:
        feats = features_by_year[year]
        labels = labels_by_year[year]
        side_labels = labels.loc[labels["side"].eq(side), ["player_id", "poss", f"label_{target}"]]
        side_labels = side_labels.rename(columns={f"label_{target}": "target"})
        merged = side_labels.merge(
            feats[features].reset_index(), on="player_id", how="inner"
        )
        merged["pair_year"] = int(year)
        pieces.append(merged)
    if not pieces:
        return pd.DataFrame(columns=["player_id", "poss", "target", *features])
    return pd.concat(pieces, ignore_index=True).dropna(subset=["target"])


def _fit_prior_model(rows: pd.DataFrame, features: List[str], alpha: float) -> Pipeline:
    model = Pipeline(
        [("scale", StandardScaler()), ("ridge", Ridge(alpha=float(alpha)))]
    )
    weights = np.sqrt(rows["poss"].clip(lower=1.0, upper=2500.0))
    weights = weights / weights.mean()
    model.fit(rows[features], rows["target"], ridge__sample_weight=weights)
    return model


def build_prior_predictions(
    arm: str,
    fold_year: int,
    pair_years: List[int],
    features_by_year: Dict[int, pd.DataFrame],
    prediction_features: pd.DataFrame,
    labels_by_year: Dict[int, pd.DataFrame],
    short_names: List[str],
    feature_map: Callable[[str, str], List[str]],
) -> Tuple[pd.DataFrame, List[dict]]:
    """Fit factor-specific models on year-forward pairs and predict the fold."""
    prior = pd.DataFrame(index=prediction_features.index.astype(str))
    prior.index.name = "player_id"
    metadata: List[dict] = []
    for target in short_names:
        for side in ("off", "def"):
            features = feature_map(target, side)
            missing = [f for f in features if f not in prediction_features.columns]
            if missing:
                raise ValueError(
                    f"{arm} fold {fold_year} {target}/{side} missing features {missing}"
                )
            fold_years = pair_years[1:]
            alpha_scores = []
            for alpha in PRIOR_MODEL_ALPHAS:
                errors: List[float] = []
                weights: List[float] = []
                for validation_year in fold_years:
                    fit_years = [y for y in pair_years if y < validation_year]
                    fit_rows = _training_rows(
                        features_by_year, labels_by_year, fit_years, target, side, features
                    )
                    validation = _training_rows(
                        features_by_year, labels_by_year, [validation_year], target, side, features
                    )
                    if fit_rows.empty or validation.empty:
                        continue
                    model = _fit_prior_model(fit_rows, features, alpha)
                    prediction = model.predict(validation[features])
                    weight = np.sqrt(
                        validation["poss"].clip(lower=1.0, upper=2500.0)
                    ).to_numpy()
                    errors.extend((prediction - validation["target"]).tolist())
                    weights.extend(weight.tolist())
                if errors:
                    rmse = float(
                        np.sqrt(np.average(np.square(errors), weights=weights))
                    )
                else:
                    rmse = float("nan")
                alpha_scores.append({"alpha": float(alpha), "forward_rmse": rmse})
            scored = pd.DataFrame(alpha_scores)
            if scored["forward_rmse"].notna().any():
                selected_alpha = float(
                    scored.sort_values("forward_rmse").iloc[0]["alpha"]
                )
            else:
                selected_alpha = PRIOR_MODEL_FALLBACK_ALPHA
            rows = _training_rows(
                features_by_year, labels_by_year, pair_years, target, side, features
            )
            if rows.empty:
                raise ValueError(
                    f"{arm} fold {fold_year} {target}/{side} has no training rows"
                )
            model = _fit_prior_model(rows, features, selected_alpha)
            lower = float(rows["target"].quantile(0.01))
            upper = float(rows["target"].quantile(0.99))
            values = np.clip(
                model.predict(prediction_features[features]), lower, upper
            )
            prior[f"{side}_{target}"] = values
            metadata.append(
                {
                    "arm": arm,
                    "fold": int(fold_year),
                    "target": target,
                    "side": side,
                    "features": features,
                    "pair_years": pair_years,
                    "selected_alpha": selected_alpha,
                    "alpha_scores": alpha_scores,
                    "training_rows": int(len(rows)),
                    "prediction_clip": [lower, upper],
                }
            )
    return prior, metadata


# ---------------------------------------------------------------------------
# Prior-centered multi-target ridge solve and scoring (tuner-exact scoring).
# ---------------------------------------------------------------------------


def build_prior_matrix(
    fold: PriorFold,
    prior_wide: Optional[pd.DataFrame],
    short_names: List[str],
) -> Tuple[np.ndarray, int]:
    matrix = np.zeros((len(fold.mapping), len(short_names)), dtype=float)
    if prior_wide is None:
        return matrix, 0
    lookup = prior_wide.to_dict(orient="index")
    matched_players = set()
    for key, index in fold.mapping.items():
        player, side = key.rsplit("_", 1)
        row = lookup.get(player)
        if row is None:
            continue
        found = False
        for t, name in enumerate(short_names):
            value = row.get(f"{side}_{name}")
            if value is not None and not pd.isna(value):
                matrix[index, t] = float(value)
                found = True
        if found:
            matched_players.add(player)
    return matrix, len(matched_players)


def solve_fold(
    fold: PriorFold,
    off_alpha: float,
    def_alpha: float,
    half_life: Optional[float],
    prior_matrix: Optional[np.ndarray] = None,
    prior_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    gram, rhs, means, exposure = fold.weighted_system(half_life)
    penalties = np.zeros(gram.shape[0], dtype=float)
    penalties[fold.off_indices] = float(off_alpha)
    penalties[fold.def_indices] = float(def_alpha)
    centered_prior = None
    if prior_matrix is not None:
        centered_prior = prior_matrix * float(prior_scale)
        for indices in (fold.off_indices, fold.def_indices):
            weight = exposure[indices]
            total = float(weight.sum())
            if total > 0:
                mean = (
                    centered_prior[indices, :] * weight[:, None]
                ).sum(axis=0) / total
                centered_prior[indices, :] = centered_prior[indices, :] - mean[None, :]
        target_rhs = rhs + penalties[:, None] * centered_prior
    else:
        target_rhs = rhs
    coefficients = np.linalg.solve(gram + np.diag(penalties), target_rhs)
    return coefficients, means, centered_prior


def score_fold(
    fold: PriorFold, coefficients: np.ndarray, means: np.ndarray
) -> pd.DataFrame:
    prediction = means.sum() + fold.validation_x @ coefficients.sum(axis=1)
    errors = np.square(fold.validation_actual - np.asarray(prediction).reshape(-1))
    frame = pd.DataFrame(
        {
            "validation_year": fold.validation_year,
            "game_id": fold.validation_games,
            "squared_error": errors,
        }
    )
    return frame.groupby(["validation_year", "game_id"], as_index=False)[
        "squared_error"
    ].mean()


def mean_of_year_mse(by_game: pd.DataFrame) -> float:
    return float(by_game.groupby("validation_year")["squared_error"].mean().mean())


def bootstrap_paired(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    draws: int,
    seed: int,
) -> dict:
    merged = candidate.merge(
        baseline,
        on=["validation_year", "game_id"],
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    merged["difference"] = (
        merged["squared_error_candidate"] - merged["squared_error_baseline"]
    )
    year_groups = {
        int(year): group["difference"].to_numpy()
        for year, group in merged.groupby("validation_year")
    }
    rng = np.random.default_rng(seed)
    stats = np.empty(draws, dtype=float)
    for draw in range(draws):
        year_means = []
        for values in year_groups.values():
            indices = rng.integers(0, len(values), len(values))
            year_means.append(float(values[indices].mean()))
        stats[draw] = float(np.mean(year_means))
    point = float(np.mean([values.mean() for values in year_groups.values()]))
    low, high = np.percentile(stats, [2.5, 97.5])
    return {
        "delta_mse": point,
        "delta_mse_x1e6": point * 1e6,
        "ci_low_x1e6": float(low) * 1e6,
        "ci_high_x1e6": float(high) * 1e6,
        "games": int(len(merged)),
        "years": sorted(year_groups),
        "wins_by_year": {
            str(year): bool(values.mean() < 0) for year, values in year_groups.items()
        },
    }


# ---------------------------------------------------------------------------
# Mechanisms (no bare nulls): label-space fit, solve coupling, error slopes.
# ---------------------------------------------------------------------------


def _weighted_corr_slope(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    total = float(w.sum())
    if total <= 0 or len(x) < 3:
        return float("nan"), float("nan")
    w = w / total
    mean_x = float((w * x).sum())
    mean_y = float((w * y).sum())
    cov = float((w * (x - mean_x) * (y - mean_y)).sum())
    var_x = float((w * (x - mean_x) ** 2).sum())
    var_y = float((w * (y - mean_y) ** 2).sum())
    corr = cov / np.sqrt(var_x * var_y) if var_x > 0 and var_y > 0 else float("nan")
    slope = cov / var_x if var_x > 0 else float("nan")
    return corr, slope


def compute_mechanisms(
    arm: str,
    fold: PriorFold,
    prior_wide: pd.DataFrame,
    raw_prior_matrix: np.ndarray,
    arm_coefficients: np.ndarray,
    pure_coefficients: np.ndarray,
    labels: Optional[pd.DataFrame],
    short_names: List[str],
) -> List[dict]:
    rows: List[dict] = []
    label_lookup: Dict[Tuple[str, str], pd.Series] = {}
    if labels is not None:
        for side in ("off", "def"):
            subset = labels.loc[labels["side"].eq(side)].set_index("player_id")
            label_lookup[side] = subset
    for t, name in enumerate(short_names):
        for side, indices, players in (
            ("off", fold.off_indices, fold.off_players),
            ("def", fold.def_indices, fold.def_players),
        ):
            record = {
                "arm": arm,
                "fold": fold.validation_year,
                "target": name,
                "side": side,
            }
            prior_column = f"{side}_{name}"
            if labels is not None and prior_column in prior_wide.columns:
                side_labels = label_lookup[side]
                joined = side_labels[[f"label_{name}", "poss"]].join(
                    prior_wide[prior_column].rename("prior"), how="left"
                )
                joined["prior"] = joined["prior"].fillna(0.0)
                weights = np.sqrt(joined["poss"].clip(lower=1.0, upper=2500.0)).to_numpy()
                label_values = joined[f"label_{name}"].to_numpy(float)
                prior_values = joined["prior"].to_numpy(float)
                rmse_prior = float(
                    np.sqrt(
                        np.average(np.square(prior_values - label_values), weights=weights)
                    )
                )
                rmse_zero = float(
                    np.sqrt(np.average(np.square(label_values), weights=weights))
                )
                corr, slope = _weighted_corr_slope(prior_values, label_values, weights)
                record.update(
                    {
                        "label_players": int(len(joined)),
                        "label_rmse_prior": rmse_prior,
                        "label_rmse_zero": rmse_zero,
                        "label_rmse_ratio": rmse_prior / rmse_zero if rmse_zero > 0 else float("nan"),
                        "label_corr": corr,
                        "label_slope_on_prior": slope,
                    }
                )
            prior_side = raw_prior_matrix[indices, t]
            arm_side = arm_coefficients[indices, t]
            pure_side = pure_coefficients[indices, t]
            matched = np.abs(prior_side) > 0
            if matched.sum() >= 3:
                solve_corr = float(
                    np.corrcoef(prior_side[matched], arm_side[matched])[0, 1]
                )
            else:
                solve_corr = float("nan")
            record.update(
                {
                    "matched_coefficients": int(matched.sum()),
                    "solve_corr_prior_coef": solve_corr,
                    "sd_prior": float(np.std(prior_side[matched])) if matched.any() else 0.0,
                    "sd_coef_arm": float(np.std(arm_side)),
                    "sd_coef_pure": float(np.std(pure_side)),
                    "sd_arm_minus_pure": float(np.std(arm_side - pure_side)),
                }
            )
            if labels is not None:
                side_labels = label_lookup[side]
                x_values, y_values, w_values = [], [], []
                for row_index, player in zip(indices, players):
                    if player not in side_labels.index:
                        continue
                    label_value = float(side_labels.at[player, f"label_{name}"])
                    poss = float(side_labels.at[player, "poss"])
                    x_values.append(raw_prior_matrix[row_index, t] - pure_coefficients[row_index, t])
                    y_values.append(label_value - pure_coefficients[row_index, t])
                    w_values.append(np.sqrt(min(max(poss, 1.0), 2500.0)))
                if len(x_values) >= 3:
                    corr_fw, slope_fw = _weighted_corr_slope(
                        np.asarray(x_values), np.asarray(y_values), np.asarray(w_values)
                    )
                else:
                    corr_fw, slope_fw = float("nan"), float("nan")
                record.update(
                    {
                        "fwd_error_slope": slope_fw,
                        "fwd_error_corr": corr_fw,
                        "fwd_error_players": int(len(x_values)),
                    }
                )
            rows.append(record)
    return rows


# ---------------------------------------------------------------------------
# Grid selection (tuner-style paired one-SE on the dev folds only).
# ---------------------------------------------------------------------------


def select_arm_config(
    arm: str,
    folds: Dict[int, PriorFold],
    prior_matrices: Dict[int, np.ndarray],
) -> Tuple[pd.DataFrame, dict]:
    records = []
    game_frames = {}
    for off_alpha, def_alpha in BASE_ALPHA_GRID:
        for half_life in HALF_LIFE_GRID:
            for prior_scale in PRIOR_SCALE_GRID:
                frames = []
                for fold_year in DEV_FOLDS:
                    fold = folds[fold_year]
                    coefficients, means, _ = solve_fold(
                        fold,
                        off_alpha,
                        def_alpha,
                        half_life,
                        prior_matrices[fold_year],
                        prior_scale,
                    )
                    frames.append(score_fold(fold, coefficients, means))
                by_game = pd.concat(frames, ignore_index=True)
                key = (off_alpha, def_alpha, half_life, prior_scale)
                game_frames[key] = by_game
                records.append(
                    {
                        "arm": arm,
                        "off_alpha": off_alpha,
                        "def_alpha": def_alpha,
                        "half_life_days": half_life,
                        "prior_scale": prior_scale,
                        "dev_mean_game_weighted_possession_mse": mean_of_year_mse(
                            by_game
                        ),
                    }
                )
    grid = (
        pd.DataFrame(records)
        .sort_values("dev_mean_game_weighted_possession_mse")
        .reset_index(drop=True)
    )
    best_row = grid.iloc[0]
    best_key = (
        float(best_row["off_alpha"]),
        float(best_row["def_alpha"]),
        None
        if pd.isna(best_row["half_life_days"])
        else float(best_row["half_life_days"]),
        float(best_row["prior_scale"]),
    )
    best_games = game_frames[best_key]
    one_se_values = []
    for row in grid.itertuples(index=False):
        key = (
            float(row.off_alpha),
            float(row.def_alpha),
            None if pd.isna(row.half_life_days) else float(row.half_life_days),
            float(row.prior_scale),
        )
        one_se_values.append(paired_one_se(game_frames[key], best_games))
    grid["paired_one_se"] = one_se_values
    grid["within_one_se"] = (
        grid["dev_mean_game_weighted_possession_mse"]
        <= float(best_row["dev_mean_game_weighted_possession_mse"])
        + grid["paired_one_se"]
    )
    eligible = grid.loc[grid["within_one_se"]].copy()
    eligible["shrinkage"] = np.sqrt(eligible["off_alpha"] * eligible["def_alpha"])
    eligible["half_life_preference"] = eligible["half_life_days"].fillna(np.inf)
    selected_row = eligible.sort_values(
        ["shrinkage", "half_life_preference", "prior_scale"],
        ascending=[False, False, True],
    ).iloc[0]
    selected = {
        "arm": arm,
        "off_alpha": float(selected_row["off_alpha"]),
        "def_alpha": float(selected_row["def_alpha"]),
        "half_life_days": (
            None
            if pd.isna(selected_row["half_life_days"])
            else float(selected_row["half_life_days"])
        ),
        "prior_scale": float(selected_row["prior_scale"]),
        "dev_mean_game_weighted_possession_mse": float(
            selected_row["dev_mean_game_weighted_possession_mse"]
        ),
        "best_unconstrained_dev_mse": float(
            best_row["dev_mean_game_weighted_possession_mse"]
        ),
        "selection_rule": (
            "most-shrunk, then longest-history, then smallest-prior-scale "
            "candidate within paired one-SE of the dev best; dev folds 2024+2025 only"
        ),
    }
    return grid, selected


# ---------------------------------------------------------------------------
# Cache helpers for the expensive home/away transform.
# ---------------------------------------------------------------------------


def _load_or_build_ordered(
    output_dir: Path, sources: List[Path], rows: pd.DataFrame, use_cache: bool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    digest = hashlib.sha256()
    for path in sorted({str(path) for path in sources}):
        digest.update(path.encode())
        digest.update(_file_hash(Path(path)).encode())
    key = digest.hexdigest()[:16]
    cache_dir = output_dir / "_cache" / key
    ordered_path = cache_dir / "ordered.pkl"
    transformed_path = cache_dir / "transformed.pkl"
    if use_cache and ordered_path.exists() and transformed_path.exists():
        logging.info("Loading cached transform (%s)", key)
        ordered = pd.read_pickle(ordered_path)
        transformed = pd.read_pickle(transformed_path)
        if len(ordered) == len(rows) and len(transformed) == len(rows):
            return ordered, transformed
        logging.warning("Cache row-count mismatch; rebuilding transform")
    started = time.time()
    ordered, transformed = prepare_ordered(rows)
    logging.info("Home/away transform took %.1fs", time.time() - started)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ordered.to_pickle(ordered_path)
    transformed.to_pickle(transformed_path)
    return ordered, transformed


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--kalman-box", type=Path, default=KALMAN_BOX_PATH)
    parser.add_argument("--skills", type=Path, default=SKILLS_PATH)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    years = list(range(2022, 2027))
    rows, sources = load_target_rows(
        years,
        target_family="net",
        processed_dir=args.processed_dir,
        sq_dir=DEFAULT_SQ_DIR,
        include_playoffs=True,
    )
    if int(rows["season_end_year"].max()) > 2026:
        raise ValueError("2027 is embargoed everywhere in this lane")
    ordered, transformed = _load_or_build_ordered(
        output_dir, sources, rows, use_cache=not args.no_cache
    )

    # Labels: single-season pure zero-centered ten-target solves.
    labels_by_year: Dict[int, pd.DataFrame] = {}
    target_names: Optional[List[str]] = None
    for year in years:
        logging.info("Solving single-season labels for %s", year)
        labels, names = solve_season_labels(ordered, transformed, year)
        labels_by_year[year] = labels
        if target_names is None:
            target_names = names
        elif target_names != names:
            raise ValueError("label target names changed across seasons")
    assert target_names is not None
    short_names = [name.replace("target_", "") for name in target_names]
    labels_all = pd.concat(labels_by_year.values(), ignore_index=True)
    labels_all.to_csv(output_dir / "labels_by_season.csv", index=False)

    # Kalman as-of snapshots (strictly before each season's first game date).
    boundaries = {
        season: pd.Timestamp(
            ordered.loc[ordered["season_end_year"].eq(season), "game_date"].min()
        )
        for season in range(2023, 2027)
    }
    kalman_snapshots, kalman_audit = build_kalman_snapshots(
        boundaries, args.kalman_box, args.skills
    )
    for season, snap in kalman_snapshots.items():
        snap.reset_index().to_csv(
            output_dir / f"kalman_asof_features_{season}.csv", index=False
        )

    box_snapshots = {year: load_box_snapshot(year) for year in range(2022, 2026)}

    # Folds: multi-year (train = all completed seasons before the fold) and
    # single-season (train = the fold's previous season only).
    folds: Dict[int, PriorFold] = {}
    ss_folds: Dict[int, PriorFold] = {}
    for fold_year in (2024, 2025, 2026):
        fold, fold_names = build_fold(
            ordered,
            transformed,
            range(2022, fold_year),
            fold_year,
            label=f"multi_{fold_year}",
        )
        if fold_names != target_names:
            raise ValueError("fold target names diverged from label target names")
        folds[fold_year] = fold
        ss_fold, _ = build_fold(
            ordered,
            transformed,
            [fold_year - 1],
            fold_year,
            label=f"single_{fold_year}",
        )
        ss_folds[fold_year] = ss_fold

    # Prior predictions per arm per fold.
    prior_model_metadata: List[dict] = []
    kalman_priors: Dict[int, pd.DataFrame] = {}
    box_priors: Dict[int, pd.DataFrame] = {}
    ss_priors: Dict[int, pd.DataFrame] = {}
    for fold_year in (2024, 2025, 2026):
        pair_years = list(range(2023, fold_year))
        logging.info("Fitting Kalman prior models for fold %s", fold_year)
        prior, metadata = build_prior_predictions(
            "kalman",
            fold_year,
            pair_years,
            {year: kalman_snapshots[year] for year in pair_years},
            kalman_snapshots[fold_year],
            labels_by_year,
            short_names,
            _kalman_feature_map,
        )
        kalman_priors[fold_year] = prior
        prior_model_metadata.extend(metadata)
        prior.reset_index().to_csv(
            output_dir / f"prior_predictions_kalman_{fold_year}.csv", index=False
        )

        logging.info("Fitting box-snapshot prior models for fold %s", fold_year)
        prior, metadata = build_prior_predictions(
            "box",
            fold_year,
            pair_years,
            {year: box_snapshots[year - 1] for year in pair_years},
            box_snapshots[fold_year - 1],
            labels_by_year,
            short_names,
            _box_feature_map,
        )
        box_priors[fold_year] = prior
        prior_model_metadata.extend(metadata)
        prior.reset_index().to_csv(
            output_dir / f"prior_predictions_box_{fold_year}.csv", index=False
        )

        # Existing single-season lane: same-season-trained box models
        # predicting the fold's previous-season board (T <= fold-2 pairs).
        ss_pair_years = list(range(2022, fold_year - 1))
        logging.info("Fitting single-season-lane prior models for fold %s", fold_year)
        prior, metadata = build_prior_predictions(
            "single_season_box",
            fold_year,
            ss_pair_years,
            {year: box_snapshots[year] for year in ss_pair_years},
            box_snapshots[fold_year - 1],
            labels_by_year,
            short_names,
            _box_feature_map,
        )
        ss_priors[fold_year] = prior
        prior_model_metadata.extend(metadata)

    (output_dir / "prior_model_metadata.json").write_text(
        json.dumps(prior_model_metadata, indent=2, default=float) + "\n"
    )

    # Prior matrices per fold per arm.
    kalman_matrices: Dict[int, np.ndarray] = {}
    box_matrices: Dict[int, np.ndarray] = {}
    ss_matrices: Dict[int, np.ndarray] = {}
    prior_match_counts: Dict[str, Dict[int, int]] = {"kalman": {}, "box": {}, "single_season_box": {}}
    for fold_year in (2024, 2025, 2026):
        matrix, matched = build_prior_matrix(
            folds[fold_year], kalman_priors[fold_year], short_names
        )
        kalman_matrices[fold_year] = matrix
        prior_match_counts["kalman"][fold_year] = matched
        matrix, matched = build_prior_matrix(
            folds[fold_year], box_priors[fold_year], short_names
        )
        box_matrices[fold_year] = matrix
        prior_match_counts["box"][fold_year] = matched
        matrix, matched = build_prior_matrix(
            ss_folds[fold_year], ss_priors[fold_year], short_names
        )
        ss_matrices[fold_year] = matrix
        prior_match_counts["single_season_box"][fold_year] = matched

    # Selection on the dev folds only.
    logging.info("Selecting Kalman-arm configuration on dev folds %s", DEV_FOLDS)
    kalman_grid, kalman_selected = select_arm_config("kalman", folds, kalman_matrices)
    kalman_grid.to_csv(output_dir / "selection_grid_kalman.csv", index=False)
    logging.info("Selecting box-arm configuration on dev folds %s", DEV_FOLDS)
    box_grid, box_selected = select_arm_config("box", folds, box_matrices)
    box_grid.to_csv(output_dir / "selection_grid_box.csv", index=False)
    logging.info("Kalman selected: %s", kalman_selected)
    logging.info("Box selected: %s", box_selected)

    k_config = (
        kalman_selected["off_alpha"],
        kalman_selected["def_alpha"],
        kalman_selected["half_life_days"],
    )
    k_scale = kalman_selected["prior_scale"]
    b_config = (
        box_selected["off_alpha"],
        box_selected["def_alpha"],
        box_selected["half_life_days"],
    )
    b_scale = box_selected["prior_scale"]

    # Final arm evaluation across all three folds.
    all_folds = (2024, 2025, 2026)
    arm_games: Dict[str, pd.DataFrame] = {}
    arm_coefficients: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}

    def run_arm(
        name: str,
        fold_set: Dict[int, PriorFold],
        config: Tuple[float, float, Optional[float]],
        matrices: Optional[Dict[int, np.ndarray]],
        scale: float,
    ) -> None:
        frames = []
        coefficient_map: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for fold_year in all_folds:
            fold = fold_set[fold_year]
            matrix = matrices[fold_year] if matrices is not None else None
            coefficients, means, _ = solve_fold(
                fold, config[0], config[1], config[2], matrix, scale
            )
            coefficient_map[fold_year] = (coefficients, means)
            frames.append(score_fold(fold, coefficients, means))
        arm_games[name] = pd.concat(frames, ignore_index=True)
        arm_coefficients[name] = coefficient_map

    run_arm("pure_at_kalman_config", folds, k_config, None, 1.0)
    run_arm("kalman_selected", folds, k_config, kalman_matrices, k_scale)
    run_arm("box_at_kalman_config", folds, k_config, box_matrices, k_scale)
    incumbent_config = (2000.0, 5000.0, None)
    if k_config != incumbent_config:
        run_arm("pure_tuner_incumbent", folds, incumbent_config, None, 1.0)
    else:
        arm_games["pure_tuner_incumbent"] = arm_games["pure_at_kalman_config"]
        arm_coefficients["pure_tuner_incumbent"] = arm_coefficients[
            "pure_at_kalman_config"
        ]
    if b_config != k_config or b_scale != k_scale:
        run_arm("box_selected", folds, b_config, box_matrices, b_scale)
        if b_config != k_config:
            run_arm("pure_at_box_config", folds, b_config, None, 1.0)
        else:
            arm_games["pure_at_box_config"] = arm_games["pure_at_kalman_config"]
            arm_coefficients["pure_at_box_config"] = arm_coefficients[
                "pure_at_kalman_config"
            ]
    else:
        arm_games["box_selected"] = arm_games["box_at_kalman_config"]
        arm_coefficients["box_selected"] = arm_coefficients["box_at_kalman_config"]
        arm_games["pure_at_box_config"] = arm_games["pure_at_kalman_config"]
        arm_coefficients["pure_at_box_config"] = arm_coefficients[
            "pure_at_kalman_config"
        ]
    ss_config = (SINGLE_SEASON_OFF_ALPHA, SINGLE_SEASON_DEF_ALPHA, None)
    run_arm(
        "single_season_box_lane", ss_folds, ss_config, ss_matrices, SINGLE_SEASON_PRIOR_SCALE
    )

    fold_scores = []
    for name, by_game in arm_games.items():
        for year, group in by_game.groupby("validation_year"):
            fold_scores.append(
                {
                    "arm": name,
                    "fold": int(year),
                    "spent": bool(int(year) == REPORT_ONLY_FOLD),
                    "game_weighted_possession_mse": float(group["squared_error"].mean()),
                    "games": int(len(group)),
                }
            )
    fold_scores_frame = pd.DataFrame(fold_scores)
    fold_scores_frame.to_csv(output_dir / "fold_scores.csv", index=False)

    # Paired comparisons with game-clustered bootstrap CIs.
    comparisons = [
        ("kalman_selected", "pure_at_kalman_config"),
        ("box_at_kalman_config", "pure_at_kalman_config"),
        ("kalman_selected", "box_at_kalman_config"),
        ("box_selected", "pure_at_box_config"),
        ("kalman_selected", "single_season_box_lane"),
        ("single_season_box_lane", "pure_at_kalman_config"),
        ("kalman_selected", "pure_tuner_incumbent"),
    ]
    comparison_rows = []
    seed_offset = 0
    for candidate, baseline in comparisons:
        cand = arm_games[candidate]
        base = arm_games[baseline]
        for scope, year_filter in (
            ("fold_2024", [2024]),
            ("fold_2025", [2025]),
            ("dev_pooled_2024_2025", [2024, 2025]),
            ("fold_2026_spent", [2026]),
        ):
            cand_subset = cand.loc[cand["validation_year"].isin(year_filter)]
            base_subset = base.loc[base["validation_year"].isin(year_filter)]
            result = bootstrap_paired(
                cand_subset,
                base_subset,
                args.bootstrap_draws,
                SEED + seed_offset,
            )
            seed_offset += 1
            comparison_rows.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    "scope": scope,
                    "spent": scope == "fold_2026_spent",
                    **result,
                }
            )
    comparison_frame = pd.DataFrame(comparison_rows)
    comparison_frame.to_csv(output_dir / "paired_comparisons.csv", index=False)

    # Per-component attribution via single-target swaps at the shared config.
    attribution_rows = []
    for arm_name, matrices, scale in (
        ("kalman_selected", kalman_matrices, k_scale),
        ("box_at_kalman_config", box_matrices, k_scale),
    ):
        for fold_year in all_folds:
            fold = folds[fold_year]
            pure_coeff, means = arm_coefficients["pure_at_kalman_config"][fold_year]
            arm_coeff, _ = arm_coefficients[arm_name][fold_year]
            base_mse = float(
                score_fold(fold, pure_coeff, means)["squared_error"].mean()
            )
            full_mse = float(
                score_fold(fold, arm_coeff, means)["squared_error"].mean()
            )
            for t, name in enumerate(short_names):
                mixed = pure_coeff.copy()
                mixed[:, t] = arm_coeff[:, t]
                swap_mse = float(
                    score_fold(fold, mixed, means)["squared_error"].mean()
                )
                attribution_rows.append(
                    {
                        "arm": arm_name,
                        "fold": fold_year,
                        "spent": fold_year == REPORT_ONLY_FOLD,
                        "target": name,
                        "swap_delta_mse_x1e6": (swap_mse - base_mse) * 1e6,
                        "full_delta_mse_x1e6": (full_mse - base_mse) * 1e6,
                    }
                )
    attribution_frame = pd.DataFrame(attribution_rows)
    attribution_frame.to_csv(output_dir / "component_attribution.csv", index=False)

    # Mechanisms.
    mechanism_rows: List[dict] = []
    for arm_name, priors_by_fold, matrices, scale in (
        ("kalman", kalman_priors, kalman_matrices, k_scale),
        ("box", box_priors, box_matrices, k_scale),
    ):
        arm_key = "kalman_selected" if arm_name == "kalman" else "box_at_kalman_config"
        for fold_year in all_folds:
            fold = folds[fold_year]
            arm_coeff, _ = arm_coefficients[arm_key][fold_year]
            pure_coeff, _ = arm_coefficients["pure_at_kalman_config"][fold_year]
            mechanism_rows.extend(
                compute_mechanisms(
                    arm_name,
                    fold,
                    priors_by_fold[fold_year],
                    matrices[fold_year],
                    arm_coeff,
                    pure_coeff,
                    labels_by_year.get(fold_year),
                    short_names,
                )
            )
    mechanisms_frame = pd.DataFrame(mechanism_rows)
    mechanisms_frame.to_csv(output_dir / "mechanisms.csv", index=False)

    # Research board for the 2026 fold (trained on <=2025; report-only).
    name_map = _load_name_map()
    fold = folds[2026]
    coefficients, means = arm_coefficients["kalman_selected"][2026]
    pure_coeff, _ = arm_coefficients["pure_at_kalman_config"][2026]
    prior_matrix = kalman_matrices[2026] * k_scale
    exposure = fold.raw_exposure()
    board_rows = []
    for off_row, player in zip(fold.off_indices, fold.off_players):
        def_row = fold.mapping[f"{player}_def"]
        off_total = float(coefficients[off_row, :].sum()) * 100.0
        def_total = float(coefficients[def_row, :].sum()) * 100.0
        board_rows.append(
            {
                "player_id": player,
                "player_name": name_map.get(player, f"ID_{player}"),
                "off_poss_window": float(exposure[off_row]),
                "def_poss_window": float(exposure[def_row]),
                "off100": off_total,
                "def100_raw": def_total,
                "net100": off_total - def_total,
                "prior_off100": float(prior_matrix[off_row, :].sum()) * 100.0,
                "prior_def100_raw": float(prior_matrix[def_row, :].sum()) * 100.0,
                "pure_off100": float(pure_coeff[off_row, :].sum()) * 100.0,
                "pure_def100_raw": float(pure_coeff[def_row, :].sum()) * 100.0,
                "has_prior": int(np.abs(kalman_matrices[2026][off_row, :]).sum() > 0),
            }
        )
    board = pd.DataFrame(board_rows).sort_values("net100", ascending=False)
    board.to_csv(output_dir / "research_board_fold2026.csv", index=False, float_format="%.4f")

    source_hashes = {str(path): _file_hash(path) for path in sorted(set(sources))}
    source_hashes[str(args.kalman_box)] = _file_hash(args.kalman_box)
    source_hashes[str(args.skills)] = _file_hash(args.skills)

    fold_audit = []
    for fold_year in all_folds:
        fold = folds[fold_year]
        fold_audit.append(
            {
                "fold": fold_year,
                "train_years": fold.train_years,
                "train_rows": int(fold.train_x.shape[0]),
                "validation_rows": int(len(fold.validation_actual)),
                "validation_games": int(pd.Series(fold.validation_games).nunique()),
                "train_players": len(fold.off_players),
                "kalman_prior_players": prior_match_counts["kalman"][fold_year],
                "box_prior_players": prior_match_counts["box"][fold_year],
                "cutoff_date": str(fold.cutoff_date.date()),
            }
        )

    report = {
        "status": "development_only_research_lane",
        "lane": "wnba_predictive_net_vpm",
        "warning": (
            "2026 is SPENT evidence in this lane: the pure tuner's alpha "
            "selection already used the 2026 fold, and every 2026 number here "
            "is report-only. The untouched confirmation season is 2027. The "
            "live published WNBA VPM (box-prior gross lane) is unaffected."
        ),
        "kalman_prior_exception": (
            "This lane is the owner-sanctioned research exception to the "
            "'Kalman panels are never rating priors' rule. Promotion requires "
            "the untouched 2027 forward gate plus explicit owner sign-off."
        ),
        "no_proprietary_inputs": True,
        "target_family": "net (ten-target structural family, tuner-exact)",
        "label_protocol": {
            "solve": "single-season pure zero-centered ten-target ridge",
            "off_alpha": LABEL_OFF_ALPHA,
            "def_alpha": LABEL_DEF_ALPHA,
            "exposure_centered_per_side": True,
            "note_2026": "2026 labels are partial-season (RS to date), mechanism-reporting only",
        },
        "prior_feature_protocol": {
            "kalman": (
                "latest post_rate per stat/skill strictly before the target "
                "season's first game date; 2022 is warm-up and receives no prior"
            ),
            "box": "previous-season stabilized box row (load_box_features(S-1))",
            "training_pairs": "states as-of season T -> season T labels, T < fold season",
            "single_season_box_lane": (
                "same-season pairs T <= fold-2 predicting the fold's previous-"
                "season board, alphas 5000/40000, prior scale 1.5, matching the "
                "existing lane's frozen identity"
            ),
            "prior_model_alphas": PRIOR_MODEL_ALPHAS,
            "prior_model_fallback_alpha": PRIOR_MODEL_FALLBACK_ALPHA,
        },
        "solve_grid": {
            "base_alphas": BASE_ALPHA_GRID,
            "half_life_days": HALF_LIFE_GRID,
            "prior_scale": PRIOR_SCALE_GRID,
            "selection_folds": DEV_FOLDS,
            "report_only_fold": REPORT_ONLY_FOLD,
        },
        "selected": {"kalman": kalman_selected, "box": box_selected},
        "fold_audit": fold_audit,
        "kalman_snapshot_audit": kalman_audit,
        "prior_match_counts": prior_match_counts,
        "fold_scores": fold_scores,
        "paired_comparisons": comparison_rows,
        "source_hashes": source_hashes,
        "artifacts": {
            "labels": str(output_dir / "labels_by_season.csv"),
            "selection_grids": [
                str(output_dir / "selection_grid_kalman.csv"),
                str(output_dir / "selection_grid_box.csv"),
            ],
            "fold_scores": str(output_dir / "fold_scores.csv"),
            "paired_comparisons": str(output_dir / "paired_comparisons.csv"),
            "component_attribution": str(output_dir / "component_attribution.csv"),
            "mechanisms": str(output_dir / "mechanisms.csv"),
            "research_board_fold2026": str(output_dir / "research_board_fold2026.csv"),
            "prior_model_metadata": str(output_dir / "prior_model_metadata.json"),
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=float) + "\n"
    )
    logging.info("Wrote %s", output_dir / "report.json")

    headline = comparison_frame.loc[
        comparison_frame["scope"].isin(["dev_pooled_2024_2025", "fold_2026_spent"])
    ][
        ["candidate", "baseline", "scope", "delta_mse_x1e6", "ci_low_x1e6", "ci_high_x1e6"]
    ]
    print(headline.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
