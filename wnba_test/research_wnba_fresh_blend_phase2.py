#!/usr/bin/env python3
"""Pre-registered fresh-blend phase 2 for the WNBA predictive net-DECOMP VPM lane.

Question: within a season, does blending in-season Kalman movement into the
frozen season-start kalman prior predict future possessions better, and at
what weight? The prior at cut date t is

    prior_Y(t) = mu_start_Y + w * (mu_fresh_Y(t) - mu_start_Y)

where ``mu_start_Y`` is the lane's phase-1 season-start kalman prior for
season Y and ``mu_fresh_Y(t)`` applies the SAME frozen year-forward prior
models (trained only on seasons < Y, refit deterministically through the
lane's own ``build_prior_predictions`` code path) to Kalman feature
snapshots as of cut date t (``build_kalman_snapshots`` with boundary t).

Pre-registered protocol (``--phase select``; seasons 2024 and 2025 ONLY):

- cuts at the 0.30/0.50/0.70 nearest-index quantiles of each season's
  distinct regular-season game dates (game_id prefix ``102``); games on the
  cut date itself score as validation;
- cut folds mirror ``lane.build_fold``: train = seasons 2022..Y-1 plus
  season-Y rows strictly before t, validation = season-Y RS+PS rows on or
  after t, player mapping from training rows, date-overlap asserted;
- coverage rule: players in both mu_start and mu_fresh blend normally;
  players with a fresh prior only (in-season debutants) get ``w * mu_fresh``
  (their whole prior is in-season movement); players in mu_start missing
  from the fresh snapshot carry mu_start unchanged;
- solves at w=0 and w=1 with the frozen selected config (off/def 2000/5000,
  no decay, prior scale 1.0); the solve is linear in the prior center, so
  predictions at any w are exact interpolations and cell MSE(w) is exactly
  quadratic in w;
- pooled selection rule (do not deviate): pooled MSE(w) = mean over the 6
  cells of the cell-level game-mean MSE(w); w* = the pooled-quadratic vertex
  clamped to [0, 1] and rounded to the nearest 0.05; game-clustered
  bootstrap (resampling games within each cell) gives the 95% CI of the
  unclamped pooled vertex;
- alpha-retune arm (report-only): the f=0.50 cut of each selection season,
  off_alpha {1000, 2000, 4000} x def_alpha {2500, 5000, 10000}, solving w=1
  and w=w*, each paired against the incumbent (w=0 at 2000/5000);
- regression gates: reproducing the lane's own season-start priors must
  match ``prior_predictions_kalman_{Y}.csv`` within 1e-9 per cell, and one
  direct w=0.5 solve must match the interpolation within 1e-9, else stop.

``--phase report2026`` (requires ``fresh_blend/selection.json``; refuses to
run without it and never rewrites it) repeats the cut construction on the
2026 regular-season dates played so far with the same frozen <2026 models,
scoring forward to the current end of data, and reports w*(frozen) / 0.5 /
1.0 vs w=0 paired deltas explicitly labeled
"2026 spent/descriptive - not confirmation".

Fences: selection never loads 2026 rows (years 2022-2025 only, asserted,
and all snapshot boundaries < 2026); seasons after 2026 hard-fail
everywhere; artifacts under
``wnba_test/Results/predictive_net_vpm/fresh_blend/`` are generated files
and stay out of git; the lane scripts, live published WNBA VPM surfaces,
and the current research boards are untouched. The headline/current boards
keep the season-start prior unless an explicit owner decision adopts w*,
with 2027 the untouched confirmation season for any adoption.

Run from the repo root:
``.venv/bin/python wnba_test/research_wnba_fresh_blend_phase2.py --phase select``
``.venv/bin/python wnba_test/research_wnba_fresh_blend_phase2.py --phase report2026``
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wnba_test import build_wnba_predictive_net_vpm as lane  # noqa: E402

LANE_DIR = lane.DEFAULT_OUTPUT_DIR
OUTPUT_DIR = LANE_DIR / "fresh_blend"
SELECTION_PATH = OUTPUT_DIR / "selection.json"

FROZEN_OFF_ALPHA = 2000.0
FROZEN_DEF_ALPHA = 5000.0
FROZEN_HALF_LIFE: Optional[float] = None
FROZEN_PRIOR_SCALE = 1.0
FROZEN_PRIOR_FAMILY = "kalman"

SELECTION_SEASONS = (2024, 2025)
REPORT_SEASON = 2026
CUT_FRACTIONS = (0.30, 0.50, 0.70)
W_FRAME_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
ALPHA_GRID_OFF = (1000.0, 2000.0, 4000.0)
ALPHA_GRID_DEF = (2500.0, 5000.0, 10000.0)
BOOTSTRAP_DRAWS = 4000
SEED = 20260829
RS_GAME_PREFIX = "102"
ALLOWED_GAME_PREFIXES = {"102", "104"}
REPRODUCTION_TOL = 1e-9
LINEARITY_TOL = 1e-9
SPENT_2026_LABEL = "2026 spent/descriptive - not confirmation"

SELECTION_RULE = (
    "pooled MSE(w) = mean over the 6 selection cells (seasons 2024/2025 x cut "
    "fractions 0.30/0.50/0.70) of the cell-level game-mean MSE(w); w* is the "
    "vertex of the pooled quadratic, clamped to [0,1], rounded to the nearest "
    "0.05. Selection seasons are 2024 and 2025 only; 2026 is spent/descriptive "
    "and 2027 is the untouched confirmation for any adoption."
)
COVERAGE_RULE = (
    "players present in both mu_start and mu_fresh blend normally; players "
    "with no season-start prior but a fresh prior (in-season debutants) get "
    "w * mu_fresh; players in mu_start but absent from the fresh snapshot "
    "keep mu_start unchanged (carry-forward)."
)
CUT_RULE = (
    "cut date t = the f-quantile (nearest-index: floor(f*(n-1)+0.5) over the "
    "n sorted distinct regular-season game dates, game_id prefix 102) of the "
    "season's RS dates; training takes game_date < t, validation game_date "
    ">= t, so games on the cut date score as validation."
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_ids(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True)


def load_frozen_report() -> dict:
    report = json.loads((LANE_DIR / "report.json").read_text())
    selected = report["selected"][FROZEN_PRIOR_FAMILY]
    expectations = {
        "off_alpha": FROZEN_OFF_ALPHA,
        "def_alpha": FROZEN_DEF_ALPHA,
        "half_life_days": FROZEN_HALF_LIFE,
        "prior_scale": FROZEN_PRIOR_SCALE,
    }
    for key, expected in expectations.items():
        if selected[key] != expected:
            raise ValueError(
                f"lane report.json selected {key}={selected[key]!r} but this "
                f"experiment froze {expected!r}; update deliberately"
            )
    return report


def load_rows(years: List[int], use_cache: bool):
    rows, sources = lane.load_target_rows(
        years,
        target_family="net",
        processed_dir=lane.DEFAULT_PROCESSED_DIR,
        sq_dir=lane.DEFAULT_SQ_DIR,
        include_playoffs=True,
    )
    max_year = int(rows["season_end_year"].max())
    if max_year > 2026:
        raise ValueError("seasons after 2026 are embargoed everywhere in this lane")
    if max_year > max(years):
        raise ValueError(f"loaded season {max_year} beyond requested {max(years)}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ordered, transformed = lane._load_or_build_ordered(
        OUTPUT_DIR, sources, rows, use_cache=use_cache
    )
    return ordered, transformed, sources


def season_boundaries(ordered: pd.DataFrame, seasons) -> Dict[int, pd.Timestamp]:
    return {
        int(season): pd.Timestamp(
            ordered.loc[ordered["season_end_year"].eq(int(season)), "game_date"].min()
        )
        for season in seasons
    }


def assert_lane_boundaries(boundaries: Dict[int, pd.Timestamp], report: dict) -> None:
    audit = {int(a["season"]): a["boundary"] for a in report["kalman_snapshot_audit"]}
    for season, stamp in boundaries.items():
        if str(stamp.date()) != audit[season]:
            raise ValueError(
                f"season {season} first-game boundary {stamp.date()} does not "
                f"match the lane audit {audit[season]}"
            )


def choose_cut_date(
    ordered: pd.DataFrame, season: int, fraction: float
) -> Tuple[pd.Timestamp, int, int]:
    season_rows = ordered.loc[ordered["season_end_year"].eq(int(season))]
    ids = _clean_ids(season_rows["game_id"])
    prefixes = set(ids.str[:3].unique())
    if not prefixes <= ALLOWED_GAME_PREFIXES:
        raise ValueError(f"season {season} has unexpected game_id prefixes {prefixes}")
    rs_dates = np.sort(
        season_rows.loc[ids.str.startswith(RS_GAME_PREFIX), "game_date"].unique()
    )
    if len(rs_dates) < 20:
        raise ValueError(f"season {season} has only {len(rs_dates)} RS dates")
    index = int(np.floor(fraction * (len(rs_dates) - 1) + 0.5))
    return pd.Timestamp(rs_dates[index]), int(len(rs_dates)), index


def build_cut_fold(
    ordered: pd.DataFrame,
    transformed: pd.DataFrame,
    season: int,
    cut: pd.Timestamp,
    cell_key: int,
    label: str,
) -> Tuple[lane.PriorFold, List[str]]:
    """Mid-season fold modeled directly on ``lane.build_fold``.

    Training = seasons 2022..season-1 (RS+PS) plus season rows strictly
    before the cut; validation = season rows (RS+PS) on/after the cut.
    """
    season = int(season)
    seasons = ordered["season_end_year"]
    dates = ordered["game_date"]
    train_mask = (
        seasons.between(2022, season - 1) | (seasons.eq(season) & dates.lt(cut))
    ).to_numpy()
    validation_mask = (seasons.eq(season) & dates.ge(cut)).to_numpy()
    if not train_mask.any() or not validation_mask.any():
        raise ValueError(f"cut fold {label} has empty train or validation rows")
    train_transformed = transformed.loc[train_mask].reset_index(drop=True)
    validation_transformed = transformed.loc[validation_mask].reset_index(drop=True)
    train_prepared, target_names = lane.build_component_targets(
        ordered.loc[train_mask].reset_index(drop=True), "net"
    )
    mapping = lane._player_mapping(train_transformed)
    train_x, _ = lane.build_design_matrix(train_transformed, mapping)
    validation_x, _ = lane.build_design_matrix(validation_transformed, mapping)
    train_dates = ordered.loc[train_mask, "game_date"].to_numpy()
    validation_dates = ordered.loc[validation_mask, "game_date"]
    if pd.Timestamp(pd.to_datetime(train_dates).max()) >= validation_dates.min():
        raise ValueError(f"cut fold {label} training overlaps validation dates")
    off_indices, def_indices, off_players, def_players = lane._mapping_sides(mapping)
    fold = lane.PriorFold(
        label=label,
        validation_year=int(cell_key),
        train_years=sorted(int(v) for v in seasons[train_mask].unique()),
        cutoff_date=pd.Timestamp(validation_dates.min()),
        train_dates=train_dates,
        train_x=train_x,
        train_targets=train_prepared[target_names].to_numpy(float),
        validation_x=validation_x,
        validation_actual=ordered.loc[validation_mask, "_actual_points"].to_numpy(
            float
        ),
        validation_games=_clean_ids(
            ordered.loc[validation_mask, "game_id"]
        ).to_numpy(),
        mapping=mapping,
        off_indices=off_indices,
        def_indices=def_indices,
        off_players=off_players,
        def_players=def_players,
    )
    return fold, target_names


def solve_labels(
    ordered: pd.DataFrame, transformed: pd.DataFrame, years
) -> Tuple[Dict[int, pd.DataFrame], List[str]]:
    labels_by_year: Dict[int, pd.DataFrame] = {}
    target_names: Optional[List[str]] = None
    for year in years:
        logging.info("Solving single-season labels for %s", year)
        labels, names = lane.solve_season_labels(ordered, transformed, int(year))
        labels_by_year[int(year)] = labels
        if target_names is None:
            target_names = names
        elif target_names != names:
            raise ValueError("label target names changed across seasons")
    assert target_names is not None
    return labels_by_year, target_names


def fit_season_prior(
    season: int,
    prediction_features: pd.DataFrame,
    start_snapshots: Dict[int, pd.DataFrame],
    labels_by_year: Dict[int, pd.DataFrame],
    short_names: List[str],
    arm: str,
) -> Tuple[pd.DataFrame, List[dict]]:
    """Fit the lane's frozen year-forward kalman models for ``season`` and
    predict ``prediction_features`` (the fit never sees prediction rows)."""
    pair_years = list(range(2023, int(season)))
    return lane.build_prior_predictions(
        arm,
        int(season),
        pair_years,
        {year: start_snapshots[year] for year in pair_years},
        prediction_features,
        labels_by_year,
        short_names,
        lane._kalman_feature_map,
    )


def check_start_reproduction(season: int, reproduced: pd.DataFrame) -> float:
    stored = pd.read_csv(
        LANE_DIR / f"prior_predictions_kalman_{season}.csv",
        dtype={"player_id": str},
    ).set_index("player_id")
    if set(stored.index) != set(reproduced.index):
        raise ValueError(
            f"season {season} season-start prior player set differs from the "
            f"lane CSV ({len(stored)} vs {len(reproduced)})"
        )
    if list(stored.columns) != list(reproduced.columns):
        raise ValueError(f"season {season} prior columns diverge from the lane CSV")
    aligned = stored.loc[reproduced.index, reproduced.columns]
    max_diff = float((reproduced - aligned).abs().to_numpy().max())
    if max_diff > REPRODUCTION_TOL:
        raise ValueError(
            f"REGRESSION CHECK FAILED: season {season} season-start prior "
            f"reproduction max abs diff {max_diff:.3e} > {REPRODUCTION_TOL}"
        )
    logging.info(
        "Regression check season %s: season-start prior reproduced, max abs diff %.3e",
        season,
        max_diff,
    )
    return max_diff


def assert_same_models(start_meta: List[dict], fresh_meta: List[dict]) -> None:
    """The fresh call must have refit byte-identical models (same selected
    alpha, training rows, and clip) — only prediction features differ."""
    key = lambda m: (m["target"], m["side"])  # noqa: E731
    start_map = {key(m): m for m in start_meta}
    for meta in fresh_meta:
        ref = start_map[key(meta)]
        if (
            meta["selected_alpha"] != ref["selected_alpha"]
            or meta["training_rows"] != ref["training_rows"]
            or meta["prediction_clip"] != ref["prediction_clip"]
        ):
            raise ValueError(
                f"fresh prior model for {key(meta)} diverged from the frozen "
                "season-start model"
            )


def build_blend_endpoints(
    mu_start: pd.DataFrame, mu_fresh: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    columns = list(mu_start.columns)
    if list(mu_fresh.columns) != columns:
        raise ValueError("mu_start / mu_fresh column mismatch")
    union = mu_start.index.union(mu_fresh.index)
    in_start = union.isin(mu_start.index)
    in_fresh = union.isin(mu_fresh.index)
    b0 = mu_start.reindex(union)
    b1 = mu_fresh.reindex(union)
    if b0.loc[in_start].isna().any().any() or b1.loc[in_fresh].isna().any().any():
        raise ValueError("prior predictions contain NaN cells")
    # in-season debutants: season-start endpoint is zero -> prior = w * fresh
    b0.loc[~in_start] = 0.0
    # carry-forward: players missing from the fresh snapshot keep mu_start
    b1.loc[~in_fresh] = b0.loc[~in_fresh].to_numpy()
    if b0.isna().any().any() or b1.isna().any().any():
        raise ValueError("blend endpoints contain NaN cells")
    groups = {
        "n_union": int(len(union)),
        "n_both_union": int((in_start & in_fresh).sum()),
        "n_fresh_only_union": int((~in_start & in_fresh).sum()),
        "n_start_only_union": int((in_start & ~in_fresh).sum()),
    }
    return b0, b1, groups


def predict_rows(
    fold: lane.PriorFold, coefficients: np.ndarray, means: np.ndarray
) -> np.ndarray:
    prediction = means.sum() + fold.validation_x @ coefficients.sum(axis=1)
    return np.asarray(prediction).reshape(-1)


def error_frame(
    cell_key: int, games: np.ndarray, r0: np.ndarray, d: np.ndarray, w: float
) -> pd.DataFrame:
    errors = np.square(r0 - float(w) * d)
    frame = pd.DataFrame(
        {"validation_year": int(cell_key), "game_id": games, "squared_error": errors}
    )
    return frame.groupby(["validation_year", "game_id"], as_index=False)[
        "squared_error"
    ].mean()


def _no_intercept_weighted_slope(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray
) -> float:
    return float(np.sum(weights * x * y) / np.sum(weights * x * x))


def _weighted_centered_corr(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    weights = weights / weights.sum()
    mean_x = float((weights * x).sum())
    mean_y = float((weights * y).sum())
    cov = float((weights * (x - mean_x) * (y - mean_y)).sum())
    var_x = float((weights * (x - mean_x) ** 2).sum())
    var_y = float((weights * (y - mean_y) ** 2).sum())
    if var_x <= 0 or var_y <= 0:
        return float("nan")
    return cov / np.sqrt(var_x * var_y)


class Cell:
    """One (season, cut-fraction) evaluation cell."""

    def __init__(
        self,
        season: int,
        fraction: float,
        cut: pd.Timestamp,
        n_rs_dates: int,
        cut_index: int,
        fold: lane.PriorFold,
        fresh_audit: dict,
        groups_union: dict,
        groups_mapped: dict,
        movement: dict,
        matched_prior_players: int,
        m0: np.ndarray,
        m1: np.ndarray,
        pred0: np.ndarray,
        pred1: np.ndarray,
    ) -> None:
        self.season = int(season)
        self.fraction = float(fraction)
        self.cut = cut
        self.n_rs_dates = int(n_rs_dates)
        self.cut_index = int(cut_index)
        self.fold = fold
        self.fresh_audit = fresh_audit
        self.groups_union = groups_union
        self.groups_mapped = groups_mapped
        self.movement = movement
        self.matched_prior_players = int(matched_prior_players)
        self.m0 = m0
        self.m1 = m1
        self.cell_key = int(season) * 100 + int(round(fraction * 100))
        self.games = fold.validation_games
        self.r0 = fold.validation_actual - pred0
        self.d = pred1 - pred0
        stats = pd.DataFrame(
            {
                "validation_year": self.cell_key,
                "game_id": self.games,
                "e00": self.r0 * self.r0,
                "e0d": self.r0 * self.d,
                "edd": self.d * self.d,
            }
        )
        self.game_stats = stats.groupby(
            ["validation_year", "game_id"], as_index=False
        )[["e00", "e0d", "edd"]].mean()
        self.quad_a = float(self.game_stats["e00"].mean())
        self.quad_b = float(self.game_stats["e0d"].mean())
        self.quad_c = float(self.game_stats["edd"].mean())
        self.vertex = self.quad_b / self.quad_c
        counts = stats.groupby("game_id")["e00"].transform("size").to_numpy(float)
        self.row_weights = 1.0 / counts
        self.slope_rowreg = _no_intercept_weighted_slope(
            self.d, self.r0, self.row_weights
        )
        if not np.isclose(self.slope_rowreg, self.vertex, rtol=1e-9, atol=1e-12):
            raise ValueError(
                f"cell {self.cell_key}: quadratic vertex {self.vertex} and "
                f"row-regression slope {self.slope_rowreg} disagree"
            )
        self.corr = _weighted_centered_corr(self.d, self.r0, self.row_weights)
        self.rms_row_movement = float(np.sqrt(np.mean(self.d * self.d)))
        self.rms_game_movement = float(np.sqrt(self.quad_c))
        self._frames: Dict[float, pd.DataFrame] = {}

    def mse_at(self, w: float) -> float:
        return self.quad_a - 2.0 * float(w) * self.quad_b + float(w) ** 2 * self.quad_c

    def frame_at(self, w: float) -> pd.DataFrame:
        w = float(w)
        if w not in self._frames:
            self._frames[w] = error_frame(self.cell_key, self.games, self.r0, self.d, w)
        return self._frames[w]

    def crosscheck(self, coeff0: np.ndarray, means: np.ndarray) -> float:
        """The w=0 interpolation frame must equal ``lane.score_fold``."""
        mine = self.frame_at(0.0)
        reference = lane.score_fold(self.fold, coeff0, means)
        merged = mine.merge(
            reference,
            on=["validation_year", "game_id"],
            suffixes=("_mine", "_ref"),
            validate="one_to_one",
        )
        gap = float(
            (merged["squared_error_mine"] - merged["squared_error_ref"]).abs().max()
        )
        if gap > 1e-9:
            raise ValueError(f"cell {self.cell_key}: score_fold crosscheck gap {gap}")
        quad_gap = abs(
            self.mse_at(0.5) - float(self.frame_at(0.5)["squared_error"].mean())
        )
        if quad_gap > 1e-9:
            raise ValueError(
                f"cell {self.cell_key}: quadratic vs frame gap {quad_gap}"
            )
        return gap

    def matrix_at(self, w: float) -> np.ndarray:
        return self.m0 + float(w) * (self.m1 - self.m0)


def build_cell(
    ordered: pd.DataFrame,
    transformed: pd.DataFrame,
    season: int,
    fraction: float,
    mu_start: pd.DataFrame,
    start_meta: List[dict],
    start_snapshots: Dict[int, pd.DataFrame],
    labels_by_year: Dict[int, pd.DataFrame],
    short_names: List[str],
    target_names: List[str],
) -> Cell:
    cut, n_rs_dates, cut_index = choose_cut_date(ordered, season, fraction)
    cell_key = int(season) * 100 + int(round(fraction * 100))
    label = f"cut_{season}_f{int(round(fraction * 100)):02d}"
    logging.info(
        "Building cell %s: cut %s (RS date %s of %s)",
        label,
        cut.date(),
        cut_index + 1,
        n_rs_dates,
    )
    fold, fold_names = build_cut_fold(ordered, transformed, season, cut, cell_key, label)
    if fold_names != target_names:
        raise ValueError(f"cell {label} target names diverged")
    fresh_snapshots, fresh_audit = lane.build_kalman_snapshots(
        {int(season): cut}, lane.KALMAN_BOX_PATH, lane.SKILLS_PATH
    )
    mu_fresh, fresh_meta = fit_season_prior(
        season,
        fresh_snapshots[int(season)],
        start_snapshots,
        labels_by_year,
        short_names,
        arm=f"kalman_fresh_{label}",
    )
    assert_same_models(start_meta, fresh_meta)
    b0, b1, groups_union = build_blend_endpoints(mu_start, mu_fresh)
    m0, matched0 = lane.build_prior_matrix(fold, b0, short_names)
    m1, matched1 = lane.build_prior_matrix(fold, b1, short_names)
    if matched0 != matched1:
        raise ValueError(f"cell {label}: matched-player mismatch between endpoints")

    mapped = set(fold.off_players)
    start_ids = set(mu_start.index)
    fresh_ids = set(mu_fresh.index)
    groups_mapped = {
        "n_both_mapped": len(mapped & start_ids & fresh_ids),
        "n_fresh_only_mapped": len((mapped & fresh_ids) - start_ids),
        "n_start_only_mapped": len((mapped & start_ids) - fresh_ids),
        "n_no_prior_mapped": len(mapped - (start_ids | fresh_ids)),
    }
    common = sorted(mapped & start_ids & fresh_ids)
    delta = mu_fresh.loc[common] - mu_start.loc[common]
    off_columns = [c for c in delta.columns if c.startswith("off_")]
    def_columns = [c for c in delta.columns if c.startswith("def_")]
    off_delta100 = (delta[off_columns].sum(axis=1) * 100.0).abs()
    def_delta100 = (delta[def_columns].sum(axis=1) * 100.0).abs()
    movement = {
        "mean_abs_component_delta": float(delta.abs().to_numpy().mean()),
        "mean_abs_off_delta100": float(off_delta100.mean()),
        "mean_abs_def_delta100": float(def_delta100.mean()),
        "median_abs_off_delta100": float(off_delta100.median()),
        "max_abs_off_delta100": float(off_delta100.max()),
        "max_abs_def_delta100": float(def_delta100.max()),
    }

    coeff0, means, _ = lane.solve_fold(
        fold, FROZEN_OFF_ALPHA, FROZEN_DEF_ALPHA, FROZEN_HALF_LIFE, m0, FROZEN_PRIOR_SCALE
    )
    coeff1, _, _ = lane.solve_fold(
        fold, FROZEN_OFF_ALPHA, FROZEN_DEF_ALPHA, FROZEN_HALF_LIFE, m1, FROZEN_PRIOR_SCALE
    )
    pred0 = predict_rows(fold, coeff0, means)
    pred1 = predict_rows(fold, coeff1, means)
    cell = Cell(
        season,
        fraction,
        cut,
        n_rs_dates,
        cut_index,
        fold,
        fresh_audit[0],
        groups_union,
        groups_mapped,
        movement,
        matched0,
        m0,
        m1,
        pred0,
        pred1,
    )
    cell.crosscheck(coeff0, means)
    return cell


def linearity_check(cell: Cell) -> float:
    coeff_mid, means, _ = lane.solve_fold(
        cell.fold,
        FROZEN_OFF_ALPHA,
        FROZEN_DEF_ALPHA,
        FROZEN_HALF_LIFE,
        cell.matrix_at(0.5),
        FROZEN_PRIOR_SCALE,
    )
    pred_mid = predict_rows(cell.fold, coeff_mid, means)
    pred0 = cell.fold.validation_actual - cell.r0
    interpolated = pred0 + 0.5 * cell.d
    gap = float(np.abs(pred_mid - interpolated).max())
    if gap > LINEARITY_TOL:
        raise ValueError(
            f"LINEARITY CHECK FAILED on cell {cell.cell_key}: max abs "
            f"prediction gap {gap:.3e} > {LINEARITY_TOL}"
        )
    logging.info(
        "Linearity check cell %s: direct w=0.5 solve matches interpolation, "
        "max abs gap %.3e",
        cell.cell_key,
        gap,
    )
    return gap


def bootstrap_pooled_vertex(
    cells: List[Cell], draws: int, seed: int
) -> Tuple[float, np.ndarray]:
    arrays = [
        (
            cell.game_stats["e0d"].to_numpy(float),
            cell.game_stats["edd"].to_numpy(float),
        )
        for cell in cells
    ]
    point = float(
        np.mean([pair[0].mean() for pair in arrays])
        / np.mean([pair[1].mean() for pair in arrays])
    )
    rng = np.random.default_rng(seed)
    stats = np.empty(draws, dtype=float)
    for draw in range(draws):
        numerators = []
        denominators = []
        for e0d, edd in arrays:
            indices = rng.integers(0, len(e0d), len(e0d))
            numerators.append(float(e0d[indices].mean()))
            denominators.append(float(edd[indices].mean()))
        stats[draw] = float(np.mean(numerators) / np.mean(denominators))
    return point, stats


def paired_delta_row(
    label: str,
    scope: str,
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    draws: int,
    seed: int,
    extra: Optional[dict] = None,
) -> dict:
    result = lane.bootstrap_paired(candidate, baseline, draws, seed)
    row = {
        "comparison": label,
        "scope": scope,
        "delta_mse_x1e6": result["delta_mse_x1e6"],
        "ci_low_x1e6": result["ci_low_x1e6"],
        "ci_high_x1e6": result["ci_high_x1e6"],
        "games": result["games"],
        "wins_by_cell": json.dumps(result["wins_by_year"]),
        "draws": draws,
        "seed": seed,
    }
    if extra:
        row.update(extra)
    return row


def cell_records(cells: List[Cell], w_star: Optional[float], phase: str) -> pd.DataFrame:
    records = []
    for cell in cells:
        record = {
            "phase": phase,
            "cell_key": cell.cell_key,
            "season": cell.season,
            "cut_fraction": cell.fraction,
            "cut_date": str(cell.cut.date()),
            "n_rs_dates_observed": cell.n_rs_dates,
            "cut_date_index0": cell.cut_index,
            "train_rows": int(cell.fold.train_x.shape[0]),
            "validation_rows": int(len(cell.fold.validation_actual)),
            "validation_games": int(pd.Series(cell.fold.validation_games).nunique()),
            "mapped_players": len(cell.fold.off_players),
            "matched_prior_players": cell.matched_prior_players,
            **cell.groups_mapped,
            **cell.groups_union,
            "fresh_snapshot_players": cell.fresh_audit["players"],
            "fresh_snapshot_imputed_cells": cell.fresh_audit["imputed_feature_cells"],
            "fresh_snapshot_max_state_date": cell.fresh_audit["max_state_date"],
            "quad_A": cell.quad_a,
            "quad_B": cell.quad_b,
            "quad_C": cell.quad_c,
            "vertex_w_opt": cell.vertex,
            "slope_check_rowreg": cell.slope_rowreg,
            "corr_residual_movement": cell.corr,
            "rms_row_movement": cell.rms_row_movement,
            "rms_game_movement": cell.rms_game_movement,
            **{f"mse_w{int(round(w * 100)):03d}": cell.mse_at(w) for w in W_FRAME_GRID},
            **cell.movement,
        }
        if w_star is not None:
            record["mse_wstar"] = cell.mse_at(w_star)
        records.append(record)
    return pd.DataFrame(records)


def mechanism_records(cells: List[Cell], phase: str) -> pd.DataFrame:
    rows = []
    for cell in cells:
        rows.append(
            {
                "phase": phase,
                "scope": f"cell_{cell.cell_key}",
                "season": cell.season,
                "cut_fraction": cell.fraction,
                "cut_date": str(cell.cut.date()),
                "residual_on_movement_slope": cell.vertex,
                "slope_check_rowreg": cell.slope_rowreg,
                "corr_residual_movement": cell.corr,
                "rms_row_movement": cell.rms_row_movement,
                "rms_game_movement": cell.rms_game_movement,
                **cell.movement,
                **cell.groups_mapped,
            }
        )
    pooled = {
        "phase": phase,
        "scope": "pooled_mean_over_cells",
        "season": np.nan,
        "cut_fraction": np.nan,
        "cut_date": "",
        "residual_on_movement_slope": float(
            np.mean([cell.quad_b for cell in cells])
            / np.mean([cell.quad_c for cell in cells])
        ),
        "slope_check_rowreg": np.nan,
        "corr_residual_movement": float(np.mean([cell.corr for cell in cells])),
        "rms_row_movement": float(
            np.mean([cell.rms_row_movement for cell in cells])
        ),
        "rms_game_movement": float(
            np.mean([cell.rms_game_movement for cell in cells])
        ),
        "mean_abs_component_delta": float(
            np.mean([cell.movement["mean_abs_component_delta"] for cell in cells])
        ),
        "mean_abs_off_delta100": float(
            np.mean([cell.movement["mean_abs_off_delta100"] for cell in cells])
        ),
        "mean_abs_def_delta100": float(
            np.mean([cell.movement["mean_abs_def_delta100"] for cell in cells])
        ),
        "median_abs_off_delta100": np.nan,
        "max_abs_off_delta100": float(
            np.max([cell.movement["max_abs_off_delta100"] for cell in cells])
        ),
        "max_abs_def_delta100": float(
            np.max([cell.movement["max_abs_def_delta100"] for cell in cells])
        ),
        "n_both_mapped": int(np.sum([cell.groups_mapped["n_both_mapped"] for cell in cells])),
        "n_fresh_only_mapped": int(
            np.sum([cell.groups_mapped["n_fresh_only_mapped"] for cell in cells])
        ),
        "n_start_only_mapped": int(
            np.sum([cell.groups_mapped["n_start_only_mapped"] for cell in cells])
        ),
        "n_no_prior_mapped": int(
            np.sum([cell.groups_mapped["n_no_prior_mapped"] for cell in cells])
        ),
    }
    rows.append(pooled)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Phase: select (2024 + 2025 only).
# ---------------------------------------------------------------------------


def run_select(args: argparse.Namespace) -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if SELECTION_PATH.exists():
        raise SystemExit(
            f"{SELECTION_PATH} already exists; selection is write-once. Refusing "
            "to re-select (2026 may already have been observed)."
        )
    if args.bootstrap_draws < 2000:
        raise SystemExit("bootstrap draws must be >= 2000 (pre-registered)")
    report = load_frozen_report()

    years = list(range(2022, 2026))  # 2022..2025 — phase select never loads 2026
    ordered, transformed, sources = load_rows(years, use_cache=not args.no_cache)
    if int(ordered["season_end_year"].max()) > 2025:
        raise ValueError("phase select must not see any 2026 rows")
    if pd.Timestamp(ordered["game_date"].max()) >= pd.Timestamp("2026-01-01"):
        raise ValueError("phase select must not see any 2026 game dates")

    labels_by_year, target_names = solve_labels(ordered, transformed, (2023, 2024))
    short_names = [name.replace("target_", "") for name in target_names]

    starts = season_boundaries(ordered, (2023, 2024, 2025))
    assert_lane_boundaries(starts, report)
    for stamp in starts.values():
        if stamp >= pd.Timestamp("2026-01-01"):
            raise ValueError("select-phase snapshot boundary reached 2026")
    start_snapshots, start_audit = lane.build_kalman_snapshots(
        starts, lane.KALMAN_BOX_PATH, lane.SKILLS_PATH
    )

    mu_start: Dict[int, pd.DataFrame] = {}
    start_meta: Dict[int, List[dict]] = {}
    reproduction: Dict[str, float] = {}
    for season in SELECTION_SEASONS:
        logging.info("Fitting frozen season-start prior models for %s", season)
        prior, metadata = fit_season_prior(
            season,
            start_snapshots[season],
            start_snapshots,
            labels_by_year,
            short_names,
            arm="kalman_start",
        )
        reproduction[str(season)] = check_start_reproduction(season, prior)
        mu_start[season] = prior
        start_meta[season] = metadata

    cells: List[Cell] = []
    for season in SELECTION_SEASONS:
        for fraction in CUT_FRACTIONS:
            cells.append(
                build_cell(
                    ordered,
                    transformed,
                    season,
                    fraction,
                    mu_start[season],
                    start_meta[season],
                    start_snapshots,
                    labels_by_year,
                    short_names,
                    target_names,
                )
            )
    linearity_gap = linearity_check(cells[0])

    # Pooled quadratic, vertex, frozen w*.
    pooled_a = float(np.mean([cell.quad_a for cell in cells]))
    pooled_b = float(np.mean([cell.quad_b for cell in cells]))
    pooled_c = float(np.mean([cell.quad_c for cell in cells]))
    vertex_unclamped = pooled_b / pooled_c
    vertex_clamped = min(max(vertex_unclamped, 0.0), 1.0)
    w_star = float(np.floor(vertex_clamped * 20.0 + 0.5) / 20.0)
    point, draws_stats = bootstrap_pooled_vertex(cells, args.bootstrap_draws, SEED)
    ci_low, ci_high = (float(v) for v in np.percentile(draws_stats, [2.5, 97.5]))
    contains = {
        "0.0": bool(ci_low <= 0.0 <= ci_high),
        "0.5": bool(ci_low <= 0.5 <= ci_high),
        "1.0": bool(ci_low <= 1.0 <= ci_high),
    }
    logging.info(
        "Pooled vertex %.4f (clamped+rounded w* = %.2f), 95%% CI [%.4f, %.4f]",
        vertex_unclamped,
        w_star,
        ci_low,
        ci_high,
    )

    # Paired deltas (lane bootstrap machinery; strata = cells).
    seed_counter = SEED + 1000
    delta_rows: List[dict] = []
    pooled_baseline = pd.concat(
        [cell.frame_at(0.0) for cell in cells], ignore_index=True
    )
    w_values = sorted(set(list(W_FRAME_GRID[1:]) + [w_star]))
    for w_value in w_values:
        candidate = pd.concat(
            [cell.frame_at(w_value) for cell in cells], ignore_index=True
        )
        label = f"w{w_value:.2f}_vs_w0.00" + (
            " (w* in-sample-selected)" if np.isclose(w_value, w_star) else ""
        )
        delta_rows.append(
            paired_delta_row(
                label,
                "pooled_6cells",
                candidate,
                pooled_baseline,
                args.bootstrap_draws,
                seed_counter,
            )
        )
        seed_counter += 1
    for w_value in (0.5, 1.0):
        for cell in cells:
            delta_rows.append(
                paired_delta_row(
                    f"w{w_value:.2f}_vs_w0.00",
                    f"cell_{cell.cell_key}",
                    cell.frame_at(w_value),
                    cell.frame_at(0.0),
                    args.bootstrap_draws,
                    seed_counter,
                )
            )
            seed_counter += 1
    paired_frame = pd.DataFrame(delta_rows)

    # Alpha-retune arm (report-only): f=0.50 cells, w=1 and w=w*.
    half_cells = [cell for cell in cells if np.isclose(cell.fraction, 0.50)]
    if len(half_cells) != 2:
        raise ValueError("expected exactly two f=0.50 cells")
    incumbent_frames = {cell.cell_key: cell.frame_at(0.0) for cell in half_cells}
    incumbent_pooled = pd.concat(incumbent_frames.values(), ignore_index=True)
    blend_ref_frames = {cell.cell_key: cell.frame_at(w_star) for cell in half_cells}
    blend_ref_pooled = pd.concat(blend_ref_frames.values(), ignore_index=True)
    blend_ref_mse = lane.mean_of_year_mse(blend_ref_pooled)
    alpha_rows: List[dict] = []
    alpha_rows.append(
        {
            "off_alpha": FROZEN_OFF_ALPHA,
            "def_alpha": FROZEN_DEF_ALPHA,
            "arm": "incumbent_w0_frozen_alphas",
            "w": 0.0,
            "scope": "pooled_2cells_f050",
            "mse": lane.mean_of_year_mse(incumbent_pooled),
            "delta_vs_incumbent_x1e6": 0.0,
            "ci_low_x1e6": np.nan,
            "ci_high_x1e6": np.nan,
            "beats_blend_ref": np.nan,
        }
    )
    blend_ref_result = lane.bootstrap_paired(
        blend_ref_pooled, incumbent_pooled, args.bootstrap_draws, seed_counter
    )
    seed_counter += 1
    alpha_rows.append(
        {
            "off_alpha": FROZEN_OFF_ALPHA,
            "def_alpha": FROZEN_DEF_ALPHA,
            "arm": "blend_ref_wstar_frozen_alphas",
            "w": w_star,
            "scope": "pooled_2cells_f050",
            "mse": blend_ref_mse,
            "delta_vs_incumbent_x1e6": blend_ref_result["delta_mse_x1e6"],
            "ci_low_x1e6": blend_ref_result["ci_low_x1e6"],
            "ci_high_x1e6": blend_ref_result["ci_high_x1e6"],
            "beats_blend_ref": np.nan,
        }
    )
    retuned_w1_frames: Dict[Tuple[float, float], pd.DataFrame] = {}
    for off_alpha in ALPHA_GRID_OFF:
        for def_alpha in ALPHA_GRID_DEF:
            for w_value, arm in ((1.0, "retuned_w1"), (w_star, "retuned_wstar")):
                frames = []
                for cell in half_cells:
                    coefficients, means, _ = lane.solve_fold(
                        cell.fold,
                        off_alpha,
                        def_alpha,
                        FROZEN_HALF_LIFE,
                        cell.matrix_at(w_value),
                        FROZEN_PRIOR_SCALE,
                    )
                    frames.append(lane.score_fold(cell.fold, coefficients, means))
                pooled = pd.concat(frames, ignore_index=True)
                if arm == "retuned_w1":
                    retuned_w1_frames[(off_alpha, def_alpha)] = pooled
                result = lane.bootstrap_paired(
                    pooled, incumbent_pooled, args.bootstrap_draws, seed_counter
                )
                seed_counter += 1
                mse = lane.mean_of_year_mse(pooled)
                alpha_rows.append(
                    {
                        "off_alpha": off_alpha,
                        "def_alpha": def_alpha,
                        "arm": arm,
                        "w": w_value,
                        "scope": "pooled_2cells_f050",
                        "mse": mse,
                        "delta_vs_incumbent_x1e6": result["delta_mse_x1e6"],
                        "ci_low_x1e6": result["ci_low_x1e6"],
                        "ci_high_x1e6": result["ci_high_x1e6"],
                        "beats_blend_ref": bool(mse < blend_ref_mse)
                        if arm == "retuned_w1"
                        else np.nan,
                    }
                )
    alpha_frame = pd.DataFrame(alpha_rows)
    w1_rows = alpha_frame.loc[alpha_frame["arm"].eq("retuned_w1")]
    best_w1 = w1_rows.sort_values("mse").iloc[0]
    best_w1_key = (float(best_w1["off_alpha"]), float(best_w1["def_alpha"]))
    best_w1_vs_blend = lane.bootstrap_paired(
        retuned_w1_frames[best_w1_key],
        blend_ref_pooled,
        args.bootstrap_draws,
        seed_counter,
    )
    seed_counter += 1
    any_w1_beats_blend = bool((w1_rows["beats_blend_ref"] == True).any())  # noqa: E712
    alpha_verdict = {
        "any_retuned_w1_beats_best_blend": any_w1_beats_blend,
        "best_w1_config": {
            "off_alpha": best_w1_key[0],
            "def_alpha": best_w1_key[1],
            "mse": float(best_w1["mse"]),
        },
        "blend_ref_wstar_frozen_mse": float(blend_ref_mse),
        "best_w1_vs_blend_ref_delta_x1e6": best_w1_vs_blend["delta_mse_x1e6"],
        "best_w1_vs_blend_ref_ci_x1e6": [
            best_w1_vs_blend["ci_low_x1e6"],
            best_w1_vs_blend["ci_high_x1e6"],
        ],
    }

    cells_frame = cell_records(cells, w_star, "select")
    mechanisms_frame = mechanism_records(cells, "select")

    cells_frame.to_csv(OUTPUT_DIR / "cells.csv", index=False)
    paired_frame.to_csv(OUTPUT_DIR / "paired_deltas.csv", index=False)
    alpha_frame.to_csv(OUTPUT_DIR / "alpha_grid.csv", index=False)
    mechanisms_frame.to_csv(OUTPUT_DIR / "mechanisms.csv", index=False)

    selection = {
        "phase": "select",
        "lane": "wnba_predictive_net_vpm/fresh_blend_phase2",
        "created_utc": _utc_now(),
        "frozen_w_star": w_star,
        "pooled_vertex_unclamped": vertex_unclamped,
        "pooled_vertex_bootstrap_point": point,
        "pooled_vertex_ci95": [ci_low, ci_high],
        "ci_contains": contains,
        "pooled_quadratic": {"A": pooled_a, "B": pooled_b, "C": pooled_c},
        "selection_rule": SELECTION_RULE,
        "coverage_rule": COVERAGE_RULE,
        "cut_rule": CUT_RULE,
        "selection_seasons": list(SELECTION_SEASONS),
        "cut_fractions": list(CUT_FRACTIONS),
        "frozen_config": {
            "off_alpha": FROZEN_OFF_ALPHA,
            "def_alpha": FROZEN_DEF_ALPHA,
            "half_life_days": FROZEN_HALF_LIFE,
            "prior_scale": FROZEN_PRIOR_SCALE,
            "prior_family": FROZEN_PRIOR_FAMILY,
        },
        "seeds": {"vertex_bootstrap": SEED, "paired_bootstrap_first": SEED + 1000},
        "bootstrap_draws": args.bootstrap_draws,
        "cells": [
            {
                "cell_key": cell.cell_key,
                "season": cell.season,
                "cut_fraction": cell.fraction,
                "cut_date": str(cell.cut.date()),
                "n_rs_dates": cell.n_rs_dates,
                "cut_date_index0": cell.cut_index,
                "vertex_w_opt": cell.vertex,
                **cell.groups_mapped,
            }
            for cell in cells
        ],
        "regression_check_max_abs_diff": reproduction,
        "linearity_check_max_abs_gap": linearity_gap,
        "alpha_retune_verdict": alpha_verdict,
        "kalman_season_start_audit": start_audit,
        "embargo": (
            "phase select loaded seasons 2022-2025 only; no 2026 possession "
            "row or post-2025 panel state was read; seasons after 2026 are "
            "embargoed everywhere"
        ),
    }
    SELECTION_PATH.write_text(json.dumps(selection, indent=2, default=float) + "\n")
    logging.info("Wrote %s", SELECTION_PATH)

    print(json.dumps({k: selection[k] for k in (
        "frozen_w_star",
        "pooled_vertex_unclamped",
        "pooled_vertex_ci95",
        "ci_contains",
        "regression_check_max_abs_diff",
        "linearity_check_max_abs_gap",
        "alpha_retune_verdict",
    )}, indent=2, default=float))
    print(paired_frame.loc[paired_frame["scope"].eq("pooled_6cells")].to_string(index=False))
    return 0


# ---------------------------------------------------------------------------
# Phase: report2026 (spent/descriptive only; requires selection.json).
# ---------------------------------------------------------------------------


def run_report2026(args: argparse.Namespace) -> int:
    if not SELECTION_PATH.exists():
        raise SystemExit(
            "refusing --phase report2026: fresh_blend/selection.json is missing. "
            "Run --phase select first; selection must be frozen before any 2026 "
            "row is loaded."
        )
    selection = json.loads(SELECTION_PATH.read_text())
    w_star = float(selection["frozen_w_star"])
    report = load_frozen_report()

    years = list(range(2022, 2027))
    ordered, transformed, sources = load_rows(years, use_cache=not args.no_cache)
    if int(ordered["season_end_year"].max()) != 2026:
        raise ValueError("report2026 expected live 2026 rows on disk")

    labels_by_year, target_names = solve_labels(ordered, transformed, (2023, 2024, 2025))
    short_names = [name.replace("target_", "") for name in target_names]

    starts = season_boundaries(ordered, (2023, 2024, 2025, 2026))
    assert_lane_boundaries(starts, report)
    start_snapshots, _ = lane.build_kalman_snapshots(
        starts, lane.KALMAN_BOX_PATH, lane.SKILLS_PATH
    )

    logging.info("Fitting frozen (<2026) season-start prior models for 2026")
    mu_start_2026, start_meta = fit_season_prior(
        REPORT_SEASON,
        start_snapshots[REPORT_SEASON],
        start_snapshots,
        labels_by_year,
        short_names,
        arm="kalman_start",
    )
    reproduction = check_start_reproduction(REPORT_SEASON, mu_start_2026)

    cells: List[Cell] = []
    for fraction in CUT_FRACTIONS:
        cells.append(
            build_cell(
                ordered,
                transformed,
                REPORT_SEASON,
                fraction,
                mu_start_2026,
                start_meta,
                start_snapshots,
                labels_by_year,
                short_names,
                target_names,
            )
        )
    linearity_gap = linearity_check(cells[0])

    pooled_b = float(np.mean([cell.quad_b for cell in cells]))
    pooled_c = float(np.mean([cell.quad_c for cell in cells]))
    vertex_unclamped = pooled_b / pooled_c
    point, draws_stats = bootstrap_pooled_vertex(cells, args.bootstrap_draws, SEED + 77)
    ci_low, ci_high = (float(v) for v in np.percentile(draws_stats, [2.5, 97.5]))

    seed_counter = SEED + 5000
    rows: List[dict] = []
    for record in cell_records(cells, w_star, "report2026").to_dict(orient="records"):
        rows.append({"record_type": "cell_summary", "label": SPENT_2026_LABEL, **record})
    pooled_baseline = pd.concat(
        [cell.frame_at(0.0) for cell in cells], ignore_index=True
    )
    w_values = sorted({0.5, 1.0, w_star})
    for w_value in w_values:
        name = f"w{w_value:.2f}_vs_w0.00" + (
            " (frozen w*)" if np.isclose(w_value, w_star) else ""
        )
        candidate = pd.concat(
            [cell.frame_at(w_value) for cell in cells], ignore_index=True
        )
        rows.append(
            {
                "record_type": "paired_delta",
                "label": SPENT_2026_LABEL,
                **paired_delta_row(
                    name,
                    "pooled_3cells_2026",
                    candidate,
                    pooled_baseline,
                    args.bootstrap_draws,
                    seed_counter,
                ),
            }
        )
        seed_counter += 1
        for cell in cells:
            rows.append(
                {
                    "record_type": "paired_delta",
                    "label": SPENT_2026_LABEL,
                    **paired_delta_row(
                        name,
                        f"cell_{cell.cell_key}",
                        cell.frame_at(w_value),
                        cell.frame_at(0.0),
                        args.bootstrap_draws,
                        seed_counter,
                    ),
                }
            )
            seed_counter += 1
    rows.append(
        {
            "record_type": "pooled_vertex",
            "label": SPENT_2026_LABEL,
            "scope": "pooled_3cells_2026",
            "vertex_w_opt": vertex_unclamped,
            "ci_low_vertex": ci_low,
            "ci_high_vertex": ci_high,
            "frozen_w_star": w_star,
            "regression_check_max_abs_diff": reproduction,
            "linearity_check_max_abs_gap": linearity_gap,
            "draws": args.bootstrap_draws,
        }
    )
    report_frame = pd.DataFrame(rows)
    report_path = OUTPUT_DIR / "report2026.csv"
    report_frame.to_csv(report_path, index=False)
    logging.info("Wrote %s (selection.json untouched)", report_path)

    summary = report_frame.loc[
        report_frame["record_type"].eq("paired_delta")
        & report_frame["scope"].eq("pooled_3cells_2026"),
        ["comparison", "delta_mse_x1e6", "ci_low_x1e6", "ci_high_x1e6", "wins_by_cell"],
    ]
    print(f"\n{SPENT_2026_LABEL}")
    print(
        f"2026 pooled vertex {vertex_unclamped:.4f} [{ci_low:.4f}, {ci_high:.4f}] "
        f"(frozen w* = {w_star:.2f})"
    )
    print(summary.to_string(index=False))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["select", "report2026"], required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    if args.phase == "select":
        return run_select(args)
    return run_report2026(args)


if __name__ == "__main__":
    raise SystemExit(main())
