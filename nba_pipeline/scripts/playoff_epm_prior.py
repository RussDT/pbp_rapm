#!/usr/bin/env python3
"""Date-correct playoff impact updates centered on pre-playoff EPM.

The public function in this module estimates a persistent player playoff
translation from *earlier* postseasons only.  The likelihood is playoff
possession scoring; pre-playoff O-EPM and D-EPM supply the possession-level
expectation.  Ridge shrinkage keeps the learned translation centered at zero.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, diags, vstack
from scipy.sparse.linalg import lsqr


OFFENSE_COLUMNS = tuple(f"O{slot}" for slot in range(1, 6))
DEFENSE_COLUMNS = tuple(f"D{slot}" for slot in range(1, 6))


def load_playoff_possessions(
    processed_dir,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Load the canonical processed playoff possession targets."""
    frames: list[pd.DataFrame] = []
    columns = ["game_id", *OFFENSE_COLUMNS, *DEFENSE_COLUMNS, "Off_Diff"]
    for season in range(int(start_year), int(end_year) + 1):
        path = processed_dir / f"RAPM{season % 100:02d}_PS.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing playoff possession file: {path}")
        frame = pd.read_parquet(path, columns=columns).rename(
            columns={"Off_Diff": "points"}
        )
        frame.insert(0, "season", int(season))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _numeric_player_ids(values: Iterable[object]) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").fillna(0).astype(int).to_numpy()


def _build_player_design(
    possessions: pd.DataFrame,
    player_ids: list[int],
) -> tuple[object, dict[int, int]]:
    player_index = {player_id: index for index, player_id in enumerate(player_ids)}
    rows: list[int] = []
    columns: list[int] = []
    n_players = len(player_ids)
    for source_column in OFFENSE_COLUMNS:
        for row_index, player_id in enumerate(_numeric_player_ids(possessions[source_column])):
            if player_id in player_index:
                rows.append(row_index)
                columns.append(player_index[player_id])
    for source_column in DEFENSE_COLUMNS:
        for row_index, player_id in enumerate(_numeric_player_ids(possessions[source_column])):
            if player_id in player_index:
                rows.append(row_index)
                columns.append(n_players + player_index[player_id])
    matrix = coo_matrix(
        (
            np.ones(len(rows), dtype=np.float64),
            (np.asarray(rows, dtype=np.int32), np.asarray(columns, dtype=np.int32)),
        ),
        shape=(len(possessions), 2 * n_players),
        dtype=np.float64,
    ).tocsr()
    return matrix, player_index


def _build_net_player_design(
    possessions: pd.DataFrame,
    player_ids: list[int],
) -> tuple[object, dict[int, int]]:
    """Build one positive-good player column shared across offense and defense."""
    player_index = {player_id: index for index, player_id in enumerate(player_ids)}
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for source_column in OFFENSE_COLUMNS:
        for row_index, player_id in enumerate(_numeric_player_ids(possessions[source_column])):
            if player_id in player_index:
                rows.append(row_index)
                columns.append(player_index[player_id])
                values.append(1.0)
    for source_column in DEFENSE_COLUMNS:
        for row_index, player_id in enumerate(_numeric_player_ids(possessions[source_column])):
            if player_id in player_index:
                rows.append(row_index)
                columns.append(player_index[player_id])
                values.append(-1.0)
    matrix = coo_matrix(
        (
            np.asarray(values, dtype=np.float64),
            (np.asarray(rows, dtype=np.int32), np.asarray(columns, dtype=np.int32)),
        ),
        shape=(len(possessions), len(player_ids)),
        dtype=np.float64,
    ).tocsr()
    return matrix, player_index


def _epm_possession_expectation(
    possessions: pd.DataFrame,
    priors: pd.DataFrame,
) -> np.ndarray:
    lookup = {
        (int(row.season), int(row.player_id)): (float(row.o_dpm), float(row.d_dpm))
        for row in priors.itertuples(index=False)
    }
    seasons = pd.to_numeric(possessions["season"], errors="coerce").astype(int).to_numpy()
    expectation = np.zeros(len(possessions), dtype=np.float64)
    for column in OFFENSE_COLUMNS:
        ids = _numeric_player_ids(possessions[column])
        expectation += np.fromiter(
            (lookup.get((season, player_id), (0.0, 0.0))[0] for season, player_id in zip(seasons, ids)),
            dtype=np.float64,
            count=len(possessions),
        ) / 100.0
    for column in DEFENSE_COLUMNS:
        ids = _numeric_player_ids(possessions[column])
        expectation -= np.fromiter(
            (lookup.get((season, player_id), (0.0, 0.0))[1] for season, player_id in zip(seasons, ids)),
            dtype=np.float64,
            count=len(possessions),
        ) / 100.0
    return expectation


def _fit_translation_coefficients(
    design,
    target: np.ndarray,
    row_mask: np.ndarray,
    alpha_offense: float,
    alpha_defense: float,
    n_players: int,
) -> np.ndarray:
    penalties = np.concatenate(
        [
            np.repeat(float(alpha_offense), n_players),
            np.repeat(float(alpha_defense), n_players),
        ]
    )
    penalty = diags(np.sqrt(penalties), offsets=0, format="csr")
    augmented_design = vstack([design[row_mask], penalty], format="csr")
    augmented_target = np.concatenate(
        [target[row_mask], np.zeros(2 * n_players, dtype=np.float64)]
    )
    solution = lsqr(
        augmented_design,
        augmented_target,
        atol=1e-7,
        btol=1e-7,
        iter_lim=1000,
        show=False,
    )
    if int(solution[1]) == 7:
        raise RuntimeError("Playoff translation ridge reached the LSQR iteration limit")
    return np.asarray(solution[0], dtype=np.float64)


def _fit_prior_centered_coefficients(
    design,
    target: np.ndarray,
    row_mask: np.ndarray,
    alpha: float,
    prior_mean: np.ndarray,
) -> np.ndarray:
    """Fit a ridge update centered on the entering-postseason player state."""
    penalty_scale = np.sqrt(float(alpha))
    penalty = diags(
        np.repeat(penalty_scale, design.shape[1]), offsets=0, format="csr"
    )
    augmented_design = vstack([design[row_mask], penalty], format="csr")
    augmented_target = np.concatenate(
        [target[row_mask], penalty_scale * np.asarray(prior_mean, dtype=np.float64)]
    )
    solution = lsqr(
        augmented_design,
        augmented_target,
        atol=1e-7,
        btol=1e-7,
        iter_lim=1000,
        show=False,
    )
    if int(solution[1]) == 7:
        raise RuntimeError(
            "Dynamic playoff translation ridge reached the LSQR iteration limit"
        )
    return np.asarray(solution[0], dtype=np.float64)


def build_dynamic_playoff_epm_ratings(
    possessions: pd.DataFrame,
    priors: pd.DataFrame,
    *,
    alpha: float,
    persistence: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return entering-playoffs EPM with a sequential net residual prior.

    Each season's exported ratings are frozen before that postseason.  The
    completed postseason then updates a single positive-good net residual used
    as the following season's prior.  The net state is split evenly across the
    O-EPM and D-EPM communication fields so their sum closes exactly.
    """
    possession_required = {
        "season",
        "game_id",
        "points",
        *OFFENSE_COLUMNS,
        *DEFENSE_COLUMNS,
    }
    prior_required = {"season", "player_id", "o_dpm", "d_dpm", "dpm"}
    for label, frame, required in (
        ("playoff possessions", possessions, possession_required),
        ("pre-playoff EPM priors", priors, prior_required),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{label} are missing columns: {sorted(missing)}")
    if alpha <= 0:
        raise ValueError("Dynamic playoff translation alpha must be positive")
    if not 0.0 <= persistence <= 1.0:
        raise ValueError("Dynamic playoff persistence must be between zero and one")

    frame = possessions.copy().reset_index(drop=True)
    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
    frame["points"] = pd.to_numeric(frame["points"], errors="raise").astype(float)
    prior_frame = priors.copy()
    for column in ("season", "player_id"):
        prior_frame[column] = pd.to_numeric(prior_frame[column], errors="raise").astype(int)
    for column in ("o_dpm", "d_dpm", "dpm"):
        prior_frame[column] = pd.to_numeric(
            prior_frame[column], errors="raise"
        ).astype(float)
    if prior_frame.duplicated(["season", "player_id"]).any():
        raise ValueError("Pre-playoff EPM priors must be unique by season and player")

    observed_ids: set[int] = set(prior_frame["player_id"].tolist())
    for column in (*OFFENSE_COLUMNS, *DEFENSE_COLUMNS):
        observed_ids.update(
            player_id for player_id in _numeric_player_ids(frame[column]) if player_id > 0
        )
    player_ids = sorted(observed_ids)
    design, player_index = _build_net_player_design(frame, player_ids)
    base_expectation = _epm_possession_expectation(frame, prior_frame)
    raw_residual = frame["points"].to_numpy(dtype=float) - base_expectation
    season_residual_mean = pd.Series(raw_residual).groupby(frame["season"]).transform("mean")
    centered_residual = raw_residual - season_residual_mean.to_numpy(dtype=float)
    season_values = frame["season"].to_numpy(dtype=int)

    state = np.zeros(len(player_ids), dtype=np.float64)
    history_counts = np.zeros(len(player_ids), dtype=np.int64)
    completed_seasons = 0
    output_parts: list[pd.DataFrame] = []
    for target_season in sorted(prior_frame["season"].unique()):
        entering_state = float(persistence) * state
        season_priors = prior_frame.loc[prior_frame["season"].eq(target_season)].copy()
        indices = season_priors["player_id"].map(player_index)
        net_delta = indices.map(
            lambda index: entering_state[int(index)] * 200.0 if pd.notna(index) else 0.0
        ).to_numpy(dtype=float)
        prior_possessions = indices.map(
            lambda index: int(history_counts[int(index)]) if pd.notna(index) else 0
        ).to_numpy(dtype=int)
        season_priors["playoff_translation_o_dpm"] = net_delta / 2.0
        season_priors["playoff_translation_d_dpm"] = net_delta / 2.0
        season_priors["playoff_translation_dpm"] = net_delta
        season_priors["playoff_adjusted_o_dpm"] = season_priors["o_dpm"] + net_delta / 2.0
        season_priors["playoff_adjusted_d_dpm"] = season_priors["d_dpm"] + net_delta / 2.0
        season_priors["playoff_adjusted_dpm"] = (
            season_priors["playoff_adjusted_o_dpm"]
            + season_priors["playoff_adjusted_d_dpm"]
        )
        season_priors["playoff_prior_seasons"] = completed_seasons
        season_priors["playoff_prior_possessions"] = prior_possessions
        output_parts.append(season_priors)

        current_mask = season_values == int(target_season)
        state = entering_state
        if current_mask.any():
            state = _fit_prior_centered_coefficients(
                design,
                centered_residual,
                current_mask,
                alpha,
                entering_state,
            )
            history_counts += np.asarray(np.abs(design[current_mask]).sum(axis=0)).ravel().astype(
                np.int64
            )
            completed_seasons += 1

    output = pd.concat(output_parts, ignore_index=True).sort_values(
        ["season", "player_id"], kind="mergesort"
    ).reset_index(drop=True)
    report: dict[str, object] = {
        "method": "sequential prior-centered net playoff residual ridge",
        "timing": (
            "ratings frozen before each postseason; completed postseason updates next season"
        ),
        "alpha": float(alpha),
        "annual_persistence": float(persistence),
        "possession_rows": int(len(frame)),
        "player_seasons": int(len(output)),
    }
    return output, report


def build_walk_forward_playoff_epm_ratings(
    possessions: pd.DataFrame,
    priors: pd.DataFrame,
    *,
    alpha_offense: float,
    alpha_defense: float,
    history_window: int | None = 8,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return entering-playoffs ratings using only earlier postseason evidence.

    ``points`` is the realized offensive possession value.  D-EPM is
    positive-good in ``priors``; the internal defensive coefficient is
    points allowed, so its learned translation is sign-flipped on export.
    """
    possession_required = {"season", "game_id", "points", *OFFENSE_COLUMNS, *DEFENSE_COLUMNS}
    prior_required = {"season", "player_id", "o_dpm", "d_dpm", "dpm"}
    for label, frame, required in (
        ("playoff possessions", possessions, possession_required),
        ("pre-playoff EPM priors", priors, prior_required),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{label} are missing columns: {sorted(missing)}")
    if alpha_offense <= 0 or alpha_defense <= 0:
        raise ValueError("Playoff translation ridge alphas must be positive")
    if history_window is not None and history_window < 1:
        raise ValueError("Playoff translation history window must be positive")

    frame = possessions.copy().reset_index(drop=True)
    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
    frame["points"] = pd.to_numeric(frame["points"], errors="raise").astype(float)
    prior_frame = priors.copy()
    for column in ("season", "player_id"):
        prior_frame[column] = pd.to_numeric(prior_frame[column], errors="raise").astype(int)
    for column in ("o_dpm", "d_dpm", "dpm"):
        prior_frame[column] = pd.to_numeric(prior_frame[column], errors="raise").astype(float)
    if prior_frame.duplicated(["season", "player_id"]).any():
        raise ValueError("Pre-playoff EPM priors must be unique by season and player")

    observed_ids: set[int] = set(prior_frame["player_id"].tolist())
    for column in (*OFFENSE_COLUMNS, *DEFENSE_COLUMNS):
        observed_ids.update(player_id for player_id in _numeric_player_ids(frame[column]) if player_id > 0)
    player_ids = sorted(observed_ids)
    design, player_index = _build_player_design(frame, player_ids)
    n_players = len(player_ids)
    base_expectation = _epm_possession_expectation(frame, prior_frame)
    raw_residual = frame["points"].to_numpy(dtype=float) - base_expectation
    season_residual_mean = pd.Series(raw_residual).groupby(frame["season"]).transform("mean")
    centered_residual = raw_residual - season_residual_mean.to_numpy(dtype=float)

    season_values = frame["season"].to_numpy(dtype=int)
    target_seasons = sorted(prior_frame["season"].unique())
    output_parts: list[pd.DataFrame] = []
    for target_season in target_seasons:
        train_mask = season_values < int(target_season)
        if history_window is not None:
            train_mask &= season_values >= int(target_season) - int(history_window)
        train_seasons = sorted(np.unique(season_values[train_mask]).tolist())
        coefficients = np.zeros(2 * n_players, dtype=np.float64)
        if train_mask.any():
            coefficients = _fit_translation_coefficients(
                design,
                centered_residual,
                train_mask,
                alpha_offense,
                alpha_defense,
                n_players,
            )
            offense_counts = np.asarray(design[train_mask, :n_players].sum(axis=0)).ravel()
            defense_counts = np.asarray(design[train_mask, n_players:].sum(axis=0)).ravel()
            if offense_counts.sum() > 0:
                coefficients[:n_players] -= np.average(
                    coefficients[:n_players], weights=offense_counts
                )
            if defense_counts.sum() > 0:
                coefficients[n_players:] -= np.average(
                    coefficients[n_players:], weights=defense_counts
                )
        history_counts = np.asarray(design[train_mask].sum(axis=0)).ravel()
        season_priors = prior_frame.loc[prior_frame["season"].eq(target_season)].copy()
        indices = season_priors["player_id"].map(player_index)
        player_history = indices.map(
            lambda index: int(
                history_counts[int(index)] + history_counts[n_players + int(index)]
            )
            if pd.notna(index)
            else 0
        ).to_numpy(dtype=int)
        off_delta = indices.map(
            lambda index: coefficients[int(index)] * 100.0 if pd.notna(index) else 0.0
        ).to_numpy(dtype=float)
        def_delta = indices.map(
            lambda index: -coefficients[n_players + int(index)] * 100.0
            if pd.notna(index)
            else 0.0
        ).to_numpy(dtype=float)
        off_delta = np.where(player_history > 0, off_delta, 0.0)
        def_delta = np.where(player_history > 0, def_delta, 0.0)
        season_priors["playoff_translation_o_dpm"] = off_delta
        season_priors["playoff_translation_d_dpm"] = def_delta
        season_priors["playoff_translation_dpm"] = off_delta + def_delta
        season_priors["playoff_adjusted_o_dpm"] = season_priors["o_dpm"] + off_delta
        season_priors["playoff_adjusted_d_dpm"] = season_priors["d_dpm"] + def_delta
        season_priors["playoff_adjusted_dpm"] = (
            season_priors["playoff_adjusted_o_dpm"]
            + season_priors["playoff_adjusted_d_dpm"]
        )
        season_priors["playoff_prior_seasons"] = len(train_seasons)
        season_priors["playoff_prior_possessions"] = player_history
        output_parts.append(season_priors)

    output = pd.concat(output_parts, ignore_index=True).sort_values(
        ["season", "player_id"], kind="mergesort"
    ).reset_index(drop=True)
    report: dict[str, object] = {
        "method": "persistent playoff translation ridge centered at zero over pre-playoff EPM",
        "timing": "earlier postseasons only",
        "alpha_offense": float(alpha_offense),
        "alpha_defense": float(alpha_defense),
        "history_window": history_window,
        "possession_rows": int(len(frame)),
        "player_seasons": int(len(output)),
    }
    return output, report


def evaluate_walk_forward_playoff_ratings(
    possessions: pd.DataFrame,
    priors: pd.DataFrame,
    adjusted_ratings: pd.DataFrame,
    *,
    start_season: int,
    end_season: int,
    bootstrap_samples: int = 2000,
    seed: int = 20260828,
) -> dict[str, object]:
    """Score adjusted ratings against pre-playoff EPM on identical PS rows."""
    adjusted_required = {
        "season",
        "player_id",
        "playoff_adjusted_o_dpm",
        "playoff_adjusted_d_dpm",
        "playoff_adjusted_dpm",
    }
    missing = adjusted_required.difference(adjusted_ratings.columns)
    if missing:
        raise ValueError(f"Adjusted playoff ratings are missing columns: {sorted(missing)}")
    if bootstrap_samples < 1:
        raise ValueError("Playoff rating gate requires at least one bootstrap sample")
    frame = possessions.copy().reset_index(drop=True)
    season = pd.to_numeric(frame["season"], errors="raise").astype(int)
    evaluation_mask = season.between(int(start_season), int(end_season)).to_numpy()
    if not evaluation_mask.any():
        raise ValueError("Playoff rating gate has no possessions in the requested window")

    adjusted_priors = adjusted_ratings[
        [
            "season",
            "player_id",
            "playoff_adjusted_o_dpm",
            "playoff_adjusted_d_dpm",
            "playoff_adjusted_dpm",
        ]
    ].rename(
        columns={
            "playoff_adjusted_o_dpm": "o_dpm",
            "playoff_adjusted_d_dpm": "d_dpm",
            "playoff_adjusted_dpm": "dpm",
        }
    )
    actual = pd.to_numeric(frame["points"], errors="raise").to_numpy(dtype=float)
    base_prediction = _epm_possession_expectation(frame, priors)
    adjusted_prediction = _epm_possession_expectation(frame, adjusted_priors)

    def center_within_season(values: np.ndarray) -> np.ndarray:
        series = pd.Series(values)
        means = series.groupby(season).transform("mean")
        return values - means.to_numpy(dtype=float)

    actual_centered = center_within_season(actual)
    base_centered = center_within_season(base_prediction)
    adjusted_centered = center_within_season(adjusted_prediction)
    row_delta = (
        (actual_centered - adjusted_centered) ** 2
        - (actual_centered - base_centered) ** 2
    )
    game_keys = season.astype(str) + "_" + frame["game_id"].astype(str)
    game = (
        pd.DataFrame(
            {
                "game": game_keys[evaluation_mask].to_numpy(),
                "squared_error_delta": row_delta[evaluation_mask],
            }
        )
        .groupby("game", as_index=False)
        .agg(
            squared_error_delta=("squared_error_delta", "sum"),
            possessions=("squared_error_delta", "size"),
        )
    )
    point = float(game["squared_error_delta"].sum() / game["possessions"].sum())
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(game), size=(int(bootstrap_samples), len(game)))
    bootstrap = (
        game["squared_error_delta"].to_numpy(dtype=float)[draws].sum(axis=1)
        / game["possessions"].to_numpy(dtype=float)[draws].sum(axis=1)
    )
    return {
        "start_season": int(start_season),
        "end_season": int(end_season),
        "games": int(len(game)),
        "possessions": int(evaluation_mask.sum()),
        "mse_delta_vs_preplayoff_epm": point,
        "mse_delta_ci_low": float(np.quantile(bootstrap, 0.025)),
        "mse_delta_ci_high": float(np.quantile(bootstrap, 0.975)),
        "interpretation": "negative favors the playoff-prior adjustment",
    }
