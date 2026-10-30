#!/usr/bin/env python3
"""Research-only primary/nonprimary defender split of net shot DECOMP.

The model is fit on a strict common sample of first-chance FGA possessions
with exactly one unambiguous NBA matchup assignment and unchanged defensive
personnel from the shot event to the terminal possession row.  For defender i:

    role_i = is_primary_i - primary_share_i

The centered interaction preserves the assignment-sample defensive parent.
Raw defensive coefficients are points allowed, so every reader-facing result
is flipped once to positive-good defense.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

try:
    from .compare_sq_defender_providers import (
        DEFAULT_MATCHUP_ROOT,
        load_nba_assignments,
    )
    from .process_rapm_blocks.clean_chance_utils import series_contains
    from .process_rapm_blocks.common import (
        _base_processing,
        build_shot_flags_df,
        prepare_standard_possession_df,
    )
    from .rapm import (
        compute_alt_first_chance_baselines,
        detect_file_type_and_prepare,
    )
    from .run_assist_points_hierarchy import (
        build_player_name_map,
        exposure_counts,
        weighted_center,
    )
    from .run_joint_ts_shooter_interaction import select_role_alpha_one_se
    from .train_sq_walk_forward_challenger import normalize_game_id
except ImportError:
    from compare_sq_defender_providers import (
        DEFAULT_MATCHUP_ROOT,
        load_nba_assignments,
    )
    from process_rapm_blocks.clean_chance_utils import series_contains
    from process_rapm_blocks.common import (
        _base_processing,
        build_shot_flags_df,
        prepare_standard_possession_df,
    )
    from rapm import (
        compute_alt_first_chance_baselines,
        detect_file_type_and_prepare,
    )
    from run_assist_points_hierarchy import (
        build_player_name_map,
        exposure_counts,
        weighted_center,
    )
    from run_joint_ts_shooter_interaction import select_role_alpha_one_se
    from train_sq_walk_forward_challenger import normalize_game_id


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PIPELINE_ROOT / "raw_data"
NETCONT_DIR = PIPELINE_ROOT / "processed_netcont_lane"
RESULTS_DIR = PIPELINE_ROOT / "results" / "primary_defender_net_decomp"

OFFENSE_COLS = [f"O{i}" for i in range(1, 6)]
DEFENSE_COLS = [f"D{i}" for i in range(1, 6)]
ROLE_COLS = [f"primary_role_D{i}" for i in range(1, 6)]

COMPONENT_PREFIXES = {
    "rim_frequency": "ALT_EFG_RIM_FREQ_NET",
    "rim_conversion": "ALT_EFG_RIM_FG_NET",
    "mid_frequency": "ALT_EFG_MID_FREQ_NET",
    "mid_conversion": "ALT_EFG_MID_FG_NET",
    "three_frequency": "ALT_EFG_THREE_FREQ_NET",
    "three_conversion": "ALT_EFG_THREE_FG_NET",
}
PARENT_KEY = "net_shot_total"
PARENT_PREFIX = "DECOMP_NET_EFG"
DEFAULT_ROLE_ALPHA_GRID = [
    100.0,
    300.0,
    1000.0,
    3000.0,
    10000.0,
    30000.0,
    100000.0,
    1000000.0,
    1.0e12,
]
READER_MIN_EXPOSURES = 3000
READER_MIN_SEASONS = 2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_matchup_player_name_map(matchup_root: Path) -> dict[int, str]:
    """Load the matchup archive's full NBA player names, newest first."""
    sources = (
        (matchup_root / "master_record.csv", "PLAYER_ID", "PLAYER_NAME"),
        (matchup_root / "index_master.csv", "nba_id", "player"),
    )
    names: dict[int, str] = {}
    for path, id_column, name_column in sources:
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path, low_memory=False)
        except (pd.errors.EmptyDataError, ValueError):
            continue
        if id_column not in frame or name_column not in frame:
            continue
        frame = frame[[id_column, name_column] + (["year"] if "year" in frame else [])]
        frame[id_column] = pd.to_numeric(frame[id_column], errors="coerce")
        frame[name_column] = frame[name_column].astype("string").str.strip()
        frame = frame[
            frame[id_column].notna()
            & frame[name_column].notna()
            & frame[name_column].ne("")
        ]
        if "year" in frame:
            frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
            frame = frame.sort_values("year", ascending=False, na_position="last")
        for player_id, player_name in frame[[id_column, name_column]].itertuples(
            index=False, name=None
        ):
            names.setdefault(int(player_id), str(player_name))
    return names


def apply_player_name_map(
    frame: pd.DataFrame | None,
    player_names: Mapping[int, str],
) -> pd.DataFrame | None:
    """Replace abbreviated PBP names with provider full names when available."""
    if frame is None:
        return None
    result = frame.copy()
    if "player_id" not in result:
        return result
    full_names = result["player_id"].map(player_names)
    if "player_name" in result:
        result["player_name"] = full_names.fillna(result["player_name"])
    else:
        result.insert(1, "player_name", full_names)
    return result


def build_reader_summary(
    summary: pd.DataFrame,
    min_exposures: int = READER_MIN_EXPOSURES,
    min_seasons: int = READER_MIN_SEASONS,
) -> pd.DataFrame:
    """Build the compact, rounded player-facing primary-defender table."""
    columns = [
        "player_id",
        "player_name",
        "def_parent_impact",
        "primary_conditional_impact",
        "nonprimary_conditional_impact",
        "primary_vs_nonprimary_contrast",
        "primary_share",
        "primary_assignments",
        "assignment_exposures",
        "assignment_seasons",
    ]
    reader = summary.loc[
        summary["assignment_exposures"].ge(min_exposures)
        & summary["assignment_seasons"].ge(min_seasons),
        columns,
    ].copy()
    reader = reader.sort_values(
        ["primary_vs_nonprimary_contrast", "assignment_exposures"],
        ascending=[False, False],
    ).reset_index(drop=True)
    reader.insert(0, "rank", np.arange(1, len(reader) + 1, dtype=int))
    reader = reader.rename(
        columns={
            "player_name": "player",
            "def_parent_impact": "overall_net_shot_defense",
            "primary_conditional_impact": "primary_defense",
            "nonprimary_conditional_impact": "nonprimary_defense",
            "primary_vs_nonprimary_contrast": "primary_advantage",
            "primary_share": "primary_share_pct",
            "assignment_seasons": "seasons",
        }
    )
    reader["primary_share_pct"] = reader["primary_share_pct"] * 100.0
    impact_columns = [
        "overall_net_shot_defense",
        "primary_defense",
        "nonprimary_defense",
        "primary_advantage",
    ]
    reader[impact_columns] = reader[impact_columns].round(2)
    reader["primary_share_pct"] = reader["primary_share_pct"].round(1)
    integer_columns = [
        "rank",
        "player_id",
        "primary_assignments",
        "assignment_exposures",
        "seasons",
    ]
    reader[integer_columns] = reader[integer_columns].astype(int)
    return reader


def _valid_five(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    return (
        values.gt(0).sum(axis=1).eq(5)
        & values.nunique(axis=1, dropna=True).eq(5)
    )


def attach_first_chance_possession_keys(
    raw: pd.DataFrame,
    group_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recreate the clean-chance possession key and return FGA/terminal rows."""
    frame = raw.copy()
    if group_cols:
        groupers = [frame[column] for column in group_cols]
        shifted = frame.groupby(group_cols)["End_of_Possession"].shift(
            1, fill_value=False
        )
        frame["poss_group"] = shifted.groupby(groupers, sort=False).cumsum()
    else:
        frame["poss_group"] = frame["End_of_Possession"].shift(
            1, fill_value=False
        ).cumsum()

    possession_keys = [*group_cols, "poss_group"]
    possession_groupers = [frame[column] for column in possession_keys]
    offense_source = frame["TeamOnOffense"].mask(
        frame["TeamOnOffense"].eq(""), np.nan
    )
    frame["poss_offense"] = (
        offense_source.groupby(possession_groupers, sort=False)
        .bfill()
        .groupby(possession_groupers, sort=False)
        .ffill()
    )

    is_miss = frame["event_type"].eq("MISS")
    home_miss = is_miss & series_contains(
        frame["home_description"], "MISS", case=False
    )
    away_miss = is_miss & series_contains(
        frame["visitor_description"], "MISS", case=False
    )
    frame["off_miss"] = (
        (home_miss & frame["poss_offense"].eq("Home"))
        | (away_miss & frame["poss_offense"].eq("Away"))
    )
    off_miss_seen = (
        frame["off_miss"]
        .astype(int)
        .groupby(possession_groupers, sort=False)
        .cumsum()
    )
    frame["after_miss"] = (
        off_miss_seen.groupby(possession_groupers, sort=False)
        .shift(1, fill_value=0)
        .gt(0)
    )
    shot_flags = build_shot_flags_df(frame)
    first_chance_fga = (
        shot_flags["is_fga"].astype(bool) & ~frame["after_miss"]
    )

    shot_columns = [
        "game_id",
        "event_num",
        "poss_group",
        "player1_id",
        *DEFENSE_COLS,
    ]
    shots = frame.loc[first_chance_fga, shot_columns].copy()
    shots["game_id_normalized"] = normalize_game_id(shots["game_id"])
    shots["GAME_EVENT_ID"] = pd.to_numeric(
        shots["event_num"], errors="coerce"
    ).astype("Int64")
    shots["shooter_id"] = pd.to_numeric(
        shots["player1_id"], errors="coerce"
    ).astype("Int64")

    terminals = frame.loc[
        frame["End_of_Possession"].fillna(False).astype(bool),
        ["game_id", "event_num", "poss_group"],
    ].copy()
    terminals["game_id_normalized"] = normalize_game_id(terminals["game_id"])
    terminals["terminal_event_id"] = pd.to_numeric(
        terminals["event_num"], errors="coerce"
    ).astype("Int64")
    terminals = terminals[
        ["game_id_normalized", "poss_group", "terminal_event_id"]
    ]
    if terminals.duplicated(["game_id_normalized", "poss_group"]).any():
        raise ValueError("Terminal possession keys are not unique.")
    return shots.reset_index(drop=True), terminals.reset_index(drop=True)


def build_centered_primary_roles(
    eligible: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Build player-centered role values on the exact assignment sample."""
    long_frames = []
    for column in DEFENSE_COLS:
        slot = int(column[1:])
        long_frames.append(
            pd.DataFrame(
                {
                    "_processed_row_id": eligible["_processed_row_id"],
                    "game_id": eligible["game_id"],
                    "player_id": pd.to_numeric(
                        eligible[column], errors="coerce"
                    ).astype(np.int64),
                    "slot": slot,
                    "is_primary": pd.to_numeric(
                        eligible[column], errors="coerce"
                    ).eq(eligible["primary_defender_id"]).astype(float),
                }
            )
        )
    role_long = pd.concat(long_frames, ignore_index=True)
    shares = (
        role_long.groupby("player_id", as_index=False)
        .agg(
            primary_assignments=("is_primary", "sum"),
            assignment_exposures=("is_primary", "size"),
        )
    )
    shares["primary_share"] = (
        shares["primary_assignments"] / shares["assignment_exposures"]
    )
    role_long = role_long.merge(
        shares[["player_id", "primary_share"]],
        on="player_id",
        how="left",
        validate="many_to_one",
    )
    role_long["primary_role"] = (
        role_long["is_primary"] - role_long["primary_share"]
    )
    center_gap = (
        role_long.groupby("player_id")["primary_role"].sum().abs()
    )
    max_gap = float(center_gap.max()) if len(center_gap) else 0.0
    if max_gap > 1e-8:
        raise ValueError(
            f"Primary-defender interactions do not center exactly: {max_gap}"
        )
    return role_long, shares, max_gap


def build_model_sample(
    raw_path: Path,
    processed_path: Path,
    assignments: pd.DataFrame,
    season: int,
) -> tuple[pd.DataFrame, pd.DataFrame, Mapping[str, object]]:
    """Build the strict single-assignment first-chance FGA model sample."""
    processed = pd.read_parquet(processed_path).reset_index(drop=True)
    required = {
        "game_id",
        "event_num",
        "Season",
        "Is_Turnover",
        *OFFENSE_COLS,
        *DEFENSE_COLS,
    }
    missing = sorted(required - set(processed.columns))
    if missing:
        raise ValueError(f"Processed net-continuation file is missing {missing}")
    processed["_processed_row_id"] = np.arange(len(processed), dtype=np.int64)
    processed["game_id_normalized"] = normalize_game_id(processed["game_id"])
    processed["terminal_event_id"] = pd.to_numeric(
        processed["event_num"], errors="coerce"
    ).astype("Int64")
    for column in [*OFFENSE_COLS, *DEFENSE_COLS]:
        processed[column] = pd.to_numeric(
            processed[column], errors="coerce"
        ).astype("Int64")

    raw = _base_processing(raw_path)
    if raw is None:
        raise ValueError(f"Could not process {raw_path}")
    raw, group_cols = prepare_standard_possession_df(
        raw, f"primary-defender net DECOMP {season}"
    )
    shots, terminals = attach_first_chance_possession_keys(raw, group_cols)
    shots = shots.merge(
        terminals,
        on=["game_id_normalized", "poss_group"],
        how="left",
        validate="many_to_one",
    )
    shots = shots.merge(
        assignments[
            [
                "game_id_normalized",
                "GAME_EVENT_ID",
                "PLAYER_ID",
                "nba_defender_ids",
                "nba_assignment_count",
            ]
        ],
        on=["game_id_normalized", "GAME_EVENT_ID"],
        how="left",
        validate="one_to_one",
    )
    shots = shots.merge(
        processed[
            [
                "_processed_row_id",
                "game_id_normalized",
                "terminal_event_id",
                *[f"D{i}" for i in range(1, 6)],
            ]
        ].rename(columns={column: f"terminal_{column}" for column in DEFENSE_COLS}),
        on=["game_id_normalized", "terminal_event_id"],
        how="left",
        validate="many_to_one",
    )

    shots["provider_shooter_id"] = pd.to_numeric(
        shots["PLAYER_ID"], errors="coerce"
    ).astype("Int64")
    shots["primary_defender_id"] = shots["nba_defender_ids"].map(
        lambda values: (
            values[0]
            if isinstance(values, tuple) and len(values) == 1
            else np.nan
        )
    )
    shot_valid = _valid_five(shots, DEFENSE_COLS)
    terminal_defense_cols = [f"terminal_{column}" for column in DEFENSE_COLS]
    terminal_valid = _valid_five(shots, terminal_defense_cols)
    same_five = pd.Series(False, index=shots.index)
    comparison_rows = shot_valid & terminal_valid
    if comparison_rows.any():
        shot_sorted = np.sort(
            shots.loc[comparison_rows, DEFENSE_COLS].to_numpy(dtype=np.int64),
            axis=1,
        )
        terminal_sorted = np.sort(
            shots.loc[comparison_rows, terminal_defense_cols].to_numpy(
                dtype=np.int64
            ),
            axis=1,
        )
        same_five.loc[comparison_rows] = np.all(
            shot_sorted == terminal_sorted, axis=1
        )
    primary_on_lineup = (
        shots[DEFENSE_COLS]
        .apply(pd.to_numeric, errors="coerce")
        .eq(shots["primary_defender_id"], axis=0)
        .sum(axis=1)
        .eq(1)
    )
    single_assignment = shots["nba_assignment_count"].eq(1)
    shooter_matches = shots["shooter_id"].eq(shots["provider_shooter_id"])
    one_shot_possession = (
        shots.groupby(["game_id_normalized", "poss_group"])["event_num"]
        .transform("size")
        .eq(1)
    )
    eligible_mask = (
        single_assignment
        & shooter_matches
        & primary_on_lineup
        & same_five
        & one_shot_possession
        & shots["_processed_row_id"].notna()
    )
    eligible_shots = shots.loc[eligible_mask].copy()
    eligible_shots["_processed_row_id"] = eligible_shots[
        "_processed_row_id"
    ].astype(np.int64)
    if eligible_shots.duplicated("_processed_row_id").any():
        raise ValueError("Eligible assignments are not possession-unique.")

    model = processed.merge(
        eligible_shots[
            ["_processed_row_id", "primary_defender_id", "GAME_EVENT_ID"]
        ],
        on="_processed_row_id",
        how="inner",
        validate="one_to_one",
    )
    model["Season"] = int(season)
    model["game_id"] = model["game_id_normalized"]
    role_long, shares, center_gap = build_centered_primary_roles(model)
    role_wide = role_long.pivot(
        index="_processed_row_id", columns="slot", values="primary_role"
    ).rename(columns={slot: f"primary_role_D{slot}" for slot in range(1, 6)})
    model = model.merge(
        role_wide.reset_index(),
        on="_processed_row_id",
        how="left",
        validate="one_to_one",
    )
    if model[ROLE_COLS].isna().any().any():
        raise ValueError("Missing centered role values on the model sample.")

    audit = {
        "processed_possessions": int(len(processed)),
        "raw_first_chance_fga": int(len(shots)),
        "matched_assignment_events": int(
            shots["nba_assignment_count"].notna().sum()
        ),
        "single_assignment_events": int(single_assignment.sum()),
        "provider_shooter_matches": int(shooter_matches.sum()),
        "primary_on_shot_lineup": int(primary_on_lineup.sum()),
        "same_shot_terminal_defensive_five": int(same_five.sum()),
        "single_first_chance_fga_possessions": int(one_shot_possession.sum()),
        "eligible_model_rows": int(len(model)),
        "eligible_share_of_raw_first_chance_fga": float(
            len(model) / len(shots) if len(shots) else 0.0
        ),
        "players_with_assignment_exposure": int(len(shares)),
        "max_centered_role_sum_gap": center_gap,
    }
    return model.reset_index(drop=True), shares, audit


def prepare_targets(
    model: pd.DataFrame,
) -> tuple[np.ndarray, list[str], pd.DataFrame, Mapping[str, float]]:
    """Create exact canonical net-shot targets and center by season."""
    keys = [*COMPONENT_PREFIXES, PARENT_KEY]
    prefixes = [*COMPONENT_PREFIXES.values(), PARENT_PREFIX]
    baselines = compute_alt_first_chance_baselines(
        model, requested_prefixes=prefixes
    )
    raw_targets = []
    baseline_rows = []
    for key, prefix in zip(keys, prefixes):
        prepared = detect_file_type_and_prepare(
            model, prefix=prefix, alt_baselines=baselines
        )
        raw = prepared["Off_Diff"].to_numpy(dtype=np.float64)
        season_means = prepared.assign(_target=raw).groupby("Season")[
            "_target"
        ].mean()
        centered = raw - prepared["Season"].map(season_means).to_numpy(float)
        raw_targets.append(centered)
        for season, value in season_means.items():
            baseline_rows.append(
                {
                    "component": key,
                    "season": int(season),
                    "baseline_per_assignment_row": float(value),
                }
            )
    matrix = np.column_stack(raw_targets)
    child_gap = matrix[:, :-1].sum(axis=1) - matrix[:, -1]
    max_gap = float(np.max(np.abs(child_gap)))
    if max_gap > 1e-9:
        raise ValueError(f"Net shot targets do not close: {max_gap}")
    return matrix, keys, pd.DataFrame(baseline_rows), {
        "max_centered_target_closure_gap": max_gap,
        **{f"baseline::{key}": float(baselines[prefix]) for key, prefix in zip(keys, prefixes)},
    }


def collect_player_ids(df: pd.DataFrame) -> np.ndarray:
    values = pd.to_numeric(
        pd.Series(df[[*OFFENSE_COLS, *DEFENSE_COLS]].to_numpy().ravel()),
        errors="coerce",
    ).dropna().astype(np.int64)
    return np.sort(values[values > 0].unique())


def _append_block(
    rows: list[np.ndarray],
    columns: list[np.ndarray],
    values: list[np.ndarray],
    players: np.ndarray,
    features: np.ndarray,
    player_ids: np.ndarray,
    offset: int,
) -> None:
    flat_players = players.reshape(-1).astype(np.int64)
    flat_features = features.reshape(-1).astype(np.float64)
    flat_rows = np.repeat(np.arange(len(players), dtype=np.int64), players.shape[1])
    valid = (flat_players > 0) & np.isfinite(flat_features)
    positions = np.searchsorted(player_ids, flat_players[valid])
    matched = (
        (positions < len(player_ids))
        & (player_ids[positions] == flat_players[valid])
    )
    rows.append(flat_rows[valid][matched])
    columns.append(offset + positions[matched])
    values.append(flat_features[valid][matched])


def build_design_matrix(
    df: pd.DataFrame,
    player_ids: np.ndarray,
    off_alpha: float,
    def_alpha: float,
    role_alpha: float,
):
    offense = df[OFFENSE_COLS].to_numpy(dtype=np.int64)
    defense = df[DEFENSE_COLS].to_numpy(dtype=np.int64)
    role = df[ROLE_COLS].to_numpy(dtype=np.float64)
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    values: list[np.ndarray] = []
    n_players = len(player_ids)
    _append_block(
        rows,
        columns,
        values,
        offense,
        np.full_like(offense, 1.0 / np.sqrt(off_alpha), dtype=float),
        player_ids,
        0,
    )
    _append_block(
        rows,
        columns,
        values,
        defense,
        np.full_like(defense, 1.0 / np.sqrt(def_alpha), dtype=float),
        player_ids,
        n_players,
    )
    _append_block(
        rows,
        columns,
        values,
        defense,
        role / np.sqrt(role_alpha),
        player_ids,
        2 * n_players,
    )
    return coo_matrix(
        (
            np.concatenate(values),
            (np.concatenate(rows), np.concatenate(columns)),
        ),
        shape=(len(df), 3 * n_players),
        dtype=np.float64,
    ).tocsr()


def fit_coefficients(X, y: np.ndarray) -> np.ndarray:
    model = Ridge(alpha=1.0, fit_intercept=False, solver="lsqr")
    model.fit(X, y)
    return np.asarray(model.coef_, dtype=np.float64)


def parse_alpha_grid(value: str | None) -> list[float]:
    if value is None:
        return list(DEFAULT_ROLE_ALPHA_GRID)
    parsed = sorted(
        {float(part.strip()) for part in value.split(",") if part.strip()}
    )
    if not parsed or any(alpha <= 0 for alpha in parsed):
        raise ValueError("Role-alpha grid must contain positive values.")
    return parsed


def cross_validate_role_alpha(
    df: pd.DataFrame,
    y: np.ndarray,
    player_ids: np.ndarray,
    off_alpha: float,
    def_alpha: float,
    candidates: Iterable[float],
    folds: int,
    cv_mode: str = "game",
) -> pd.DataFrame:
    if cv_mode == "season":
        groups = df["Season"].to_numpy(dtype=np.int64)
        expected_folds = int(df["Season"].nunique())
        if expected_folds < 2:
            raise ValueError("Season-held-out CV requires at least two seasons.")
        if int(folds) != expected_folds:
            raise ValueError(
                f"Season-held-out CV requires {expected_folds} folds, got {folds}."
            )
    elif cv_mode == "game":
        groups = df["game_id"].astype(str).to_numpy()
    else:
        raise ValueError(f"Unsupported CV mode: {cv_mode}")
    splits = list(
        GroupKFold(n_splits=int(folds)).split(
            np.zeros(len(df)), groups=groups
        )
    )
    rows = []
    for role_alpha in candidates:
        logging.info("Cross-validating primary-role alpha %.0f", role_alpha)
        X = build_design_matrix(
            df, player_ids, off_alpha, def_alpha, role_alpha
        )
        fold_mse = []
        for fold_index, (train, valid) in enumerate(splits, start=1):
            beta = fit_coefficients(X[train], y[train])
            prediction = X[valid] @ beta
            mse = float(np.mean((y[valid] - prediction) ** 2))
            fold_mse.append(mse)
            logging.info(
                "  alpha %.0f fold %d/%d MSE %.9f",
                role_alpha,
                fold_index,
                len(splits),
                mse,
            )
        if cv_mode == "season":
            fold_groups = {
                f"fold_{index + 1}_heldout": "|".join(
                    sorted({str(value) for value in groups[valid]})
                )
                for index, (_, valid) in enumerate(splits)
            }
        else:
            fold_groups = {
                f"fold_{index + 1}_heldout": (
                    f"{len(np.unique(groups[valid]))} games"
                )
                for index, (_, valid) in enumerate(splits)
            }
        rows.append(
            {
                "role_alpha": float(role_alpha),
                "cv_mode": cv_mode,
                "mean_mse": float(np.mean(fold_mse)),
                "std_mse": float(np.std(fold_mse)),
                **{
                    f"fold_{index + 1}_mse": value
                    for index, value in enumerate(fold_mse)
                },
                **fold_groups,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mean_mse", "role_alpha"], ascending=[True, False]
    )


def select_paired_role_alpha_one_se(
    cv_results: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    """Select the most-shrunk model within one paired SE of the minimum.

    Every alpha is scored on the same held-out folds. Comparing foldwise error
    differences removes unrelated season/game difficulty from the uncertainty
    estimate, which is essential when season outcome variance differs.
    """
    if cv_results.empty:
        raise ValueError("Cannot select a role alpha from empty CV results.")
    result = cv_results.copy()
    fold_columns = sorted(
        [
            column
            for column in result.columns
            if column.startswith("fold_") and column.endswith("_mse")
        ],
        key=lambda column: int(column.split("_")[1]),
    )
    if len(fold_columns) < 2:
        raise ValueError("Paired one-SE selection requires at least two folds.")
    best = result.sort_values(
        ["mean_mse", "role_alpha"], ascending=[True, False]
    ).iloc[0]
    best_errors = best[fold_columns].to_numpy(dtype=float)
    mean_excess = []
    excess_se = []
    for _, row in result.iterrows():
        differences = row[fold_columns].to_numpy(dtype=float) - best_errors
        mean_excess.append(float(differences.mean()))
        excess_se.append(
            float(differences.std(ddof=1) / np.sqrt(len(differences)))
        )
    result["paired_mean_excess_to_minimum"] = mean_excess
    result["paired_excess_standard_error"] = excess_se
    result["within_paired_one_se"] = result[
        "paired_mean_excess_to_minimum"
    ].le(result["paired_excess_standard_error"] + 1e-15)
    selected = float(
        result.loc[result["within_paired_one_se"], "role_alpha"].max()
    )
    result["selected_role_alpha"] = selected
    return selected, result


def decompose_defensive_role(
    parent_allowed: float,
    contrast_allowed: float,
    primary_share: float,
) -> Mapping[str, float]:
    """Flip defense once and return exact primary/nonprimary contributions."""
    parent_good = -float(parent_allowed)
    primary_good = -(
        float(parent_allowed) + (1.0 - primary_share) * float(contrast_allowed)
    )
    nonprimary_good = -(
        float(parent_allowed) - primary_share * float(contrast_allowed)
    )
    primary_contribution = primary_share * primary_good
    nonprimary_contribution = (1.0 - primary_share) * nonprimary_good
    return {
        "def_parent_impact": parent_good,
        "primary_vs_nonprimary_contrast": primary_good - nonprimary_good,
        "primary_conditional_impact": primary_good,
        "nonprimary_conditional_impact": nonprimary_good,
        "primary_contribution": primary_contribution,
        "nonprimary_contribution": nonprimary_contribution,
        "decomposition_gap": (
            primary_contribution + nonprimary_contribution - parent_good
        ),
    }


def build_results(
    df: pd.DataFrame,
    shares: pd.DataFrame,
    player_ids: np.ndarray,
    beta: np.ndarray,
    target_keys: list[str],
    off_alpha: float,
    def_alpha: float,
    role_alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, Mapping[str, float]]:
    n_players = len(player_ids)
    coefficients = np.atleast_2d(beta).copy()
    direct_parent = coefficients[-1].copy()
    additive_parent = coefficients[:-1].sum(axis=0)
    direct_defense_gap = float(
        np.max(
            np.abs(
                direct_parent[n_players : 2 * n_players]
                - additive_parent[n_players : 2 * n_players]
            )
        )
        * 100.0
        / np.sqrt(def_alpha)
    )
    direct_role_gap = float(
        np.max(
            np.abs(
                direct_parent[2 * n_players :]
                - additive_parent[2 * n_players :]
            )
        )
        * 100.0
        / np.sqrt(role_alpha)
    )
    # Ridge is linear in y, but independent LSQR convergence can leave tiny
    # numerical differences. Publish the algebraic child sum and retain the
    # separately solved parent only as an audit.
    coefficients[-1] = additive_parent
    defense_counts = exposure_counts(df, DEFENSE_COLS)
    share_lookup = shares.set_index("player_id")
    names = build_player_name_map(set(int(value) for value in player_ids))
    centering_offsets = {}
    rows = []
    for target_index, component in enumerate(target_keys):
        physical = coefficients[target_index]
        defense = physical[n_players : 2 * n_players] / np.sqrt(def_alpha)
        role = physical[2 * n_players :] / np.sqrt(role_alpha)
        centering_offsets[component] = weighted_center(
            defense, player_ids, defense_counts
        )
        for player_index, player_id_value in enumerate(player_ids):
            player_id = int(player_id_value)
            if player_id not in share_lookup.index:
                continue
            primary_share = float(share_lookup.loc[player_id, "primary_share"])
            values = decompose_defensive_role(
                100.0 * defense[player_index],
                100.0 * role[player_index],
                primary_share,
            )
            rows.append(
                {
                    "player_id": player_id,
                    "player_name": names.get(player_id, f"ID_{player_id}"),
                    "component": component,
                    "primary_share": primary_share,
                    "primary_assignments": int(
                        share_lookup.loc[player_id, "primary_assignments"]
                    ),
                    "assignment_exposures": int(
                        share_lookup.loc[player_id, "assignment_exposures"]
                    ),
                    "assignment_seasons": int(
                        share_lookup.loc[player_id, "assignment_seasons"]
                    )
                    if "assignment_seasons" in share_lookup.columns
                    else 1,
                    **values,
                }
            )
    long = pd.DataFrame(rows)
    max_decomposition_gap = float(long["decomposition_gap"].abs().max())
    parent = long[long["component"].eq(PARENT_KEY)].copy()
    child = (
        long[~long["component"].eq(PARENT_KEY)]
        .groupby("player_id", as_index=False)[
            [
                "def_parent_impact",
                "primary_contribution",
                "nonprimary_contribution",
            ]
        ]
        .sum()
        .rename(
            columns={
                "def_parent_impact": "children_def_parent_impact",
                "primary_contribution": "children_primary_contribution",
                "nonprimary_contribution": "children_nonprimary_contribution",
            }
        )
    )
    summary = parent.merge(child, on="player_id", how="left", validate="one_to_one")
    summary["component_parent_gap"] = (
        summary["children_def_parent_impact"] - summary["def_parent_impact"]
    )
    summary["component_primary_gap"] = (
        summary["children_primary_contribution"]
        - summary["primary_contribution"]
    )
    summary["component_nonprimary_gap"] = (
        summary["children_nonprimary_contribution"]
        - summary["nonprimary_contribution"]
    )
    for component in COMPONENT_PREFIXES:
        block = long[long["component"].eq(component)][
            [
                "player_id",
                "def_parent_impact",
                "primary_contribution",
                "nonprimary_contribution",
            ]
        ].rename(
            columns={
                column: f"{component}_{column}"
                for column in [
                    "def_parent_impact",
                    "primary_contribution",
                    "nonprimary_contribution",
                ]
            }
        )
        summary = summary.merge(
            block, on="player_id", how="left", validate="one_to_one"
        )
    summary = summary.sort_values("def_parent_impact", ascending=False)
    metadata = {
        "max_player_decomposition_gap": max_decomposition_gap,
        "max_component_parent_gap": float(summary["component_parent_gap"].abs().max()),
        "max_component_primary_gap": float(summary["component_primary_gap"].abs().max()),
        "max_component_nonprimary_gap": float(summary["component_nonprimary_gap"].abs().max()),
        "max_direct_solve_parent_defense_gap": direct_defense_gap,
        "max_direct_solve_parent_role_gap": direct_role_gap,
        "defense_centering_offsets": centering_offsets,
    }
    return long, summary, metadata


def build_split_reliability(
    model: pd.DataFrame,
    targets: np.ndarray,
    seasons: list[int],
    player_ids: np.ndarray,
    off_alpha: float,
    def_alpha: float,
    role_alpha: float,
    min_exposures: int,
) -> tuple[pd.DataFrame | None, Mapping[str, object] | None]:
    """Compare primary-role contrasts in early and late season blocks."""
    if len(seasons) < 4:
        return None, None
    split_index = (len(seasons) + 1) // 2
    early_seasons = seasons[:split_index]
    late_seasons = seasons[split_index:]
    blocks = {}
    n_players = len(player_ids)
    for label, block_seasons in (
        ("early", early_seasons),
        ("late", late_seasons),
    ):
        mask = model["Season"].isin(block_seasons).to_numpy()
        block = model.loc[mask]
        X = build_design_matrix(
            block, player_ids, off_alpha, def_alpha, role_alpha
        )
        beta = np.atleast_2d(fit_coefficients(X, targets[mask]))
        additive_parent = beta[:-1].sum(axis=0)
        role_contrast = -(
            additive_parent[2 * n_players :] / np.sqrt(role_alpha)
        ) * 100.0
        counts = exposure_counts(block, DEFENSE_COLS)
        blocks[label] = pd.DataFrame(
            {
                "player_id": player_ids.astype(int),
                f"{label}_primary_vs_nonprimary_contrast": role_contrast,
                f"{label}_assignment_exposures": [
                    int(counts.get(int(player_id), 0))
                    for player_id in player_ids
                ],
            }
        )
    comparison = blocks["early"].merge(
        blocks["late"], on="player_id", how="inner", validate="one_to_one"
    )
    names = build_player_name_map(set(int(value) for value in player_ids))
    comparison.insert(
        1,
        "player_name",
        comparison["player_id"].map(names).fillna(
            comparison["player_id"].map(lambda value: f"ID_{value}")
        ),
    )
    qualified = comparison[
        comparison["early_assignment_exposures"].ge(min_exposures)
        & comparison["late_assignment_exposures"].ge(min_exposures)
    ]
    early = qualified["early_primary_vs_nonprimary_contrast"]
    late = qualified["late_primary_vs_nonprimary_contrast"]
    audit = {
        "early_seasons": early_seasons,
        "late_seasons": late_seasons,
        "min_exposures_per_split": int(min_exposures),
        "qualified_players": int(len(qualified)),
        "pearson_role_contrast": (
            None if len(qualified) < 2 else float(early.corr(late))
        ),
        "spearman_role_contrast": (
            None
            if len(qualified) < 2
            else float(early.corr(late, method="spearman"))
        ),
        "sign_agreement": (
            None
            if qualified.empty
            else float(np.mean(np.sign(early) == np.sign(late)))
        ),
    }
    return comparison, audit


def run(args: argparse.Namespace) -> Path:
    end_season = int(args.season)
    start_season = (
        int(args.start_season) if args.start_season is not None else end_season
    )
    seasons = list(range(start_season, end_season + 1))
    if len(seasons) > 1 and (args.raw_path or args.processed_path):
        raise ValueError(
            "--raw-path/--processed-path are supported only for a single season."
        )
    assignments, assignment_audit = load_nba_assignments(
        Path(args.matchup_root), seasons=seasons
    )
    model_frames = []
    share_frames = []
    sample_audit_by_season = {}
    input_paths = []
    for season in seasons:
        suffix = f"{season % 100:02d}"
        raw_path = (
            Path(args.raw_path)
            if args.raw_path
            else RAW_DIR / f"NBA{suffix}.parquet"
        )
        processed_path = (
            Path(args.processed_path)
            if args.processed_path
            else NETCONT_DIR / f"FIRST_CHANCE{suffix}.parquet"
        )
        logging.info("Building strict assignment sample for season %d", season)
        season_model, season_shares, season_audit = build_model_sample(
            raw_path, processed_path, assignments, season
        )
        season_shares["Season"] = season
        model_frames.append(season_model)
        share_frames.append(season_shares)
        sample_audit_by_season[str(season)] = season_audit
        input_paths.append(
            {
                "season": season,
                "raw_path": str(raw_path),
                "processed_path": str(processed_path),
            }
        )
    model = pd.concat(model_frames, ignore_index=True)
    season_share_rows = pd.concat(share_frames, ignore_index=True)
    shares = (
        season_share_rows.groupby("player_id", as_index=False)
        .agg(
            primary_assignments=("primary_assignments", "sum"),
            assignment_exposures=("assignment_exposures", "sum"),
            assignment_seasons=("Season", "nunique"),
        )
    )
    shares["primary_share"] = (
        shares["primary_assignments"] / shares["assignment_exposures"]
    )
    role_audit_frames = []
    for slot, column in enumerate(DEFENSE_COLS):
        role_audit_frames.append(
            pd.DataFrame(
                {
                    "Season": model["Season"],
                    "player_id": model[column],
                    "primary_role": model[ROLE_COLS[slot]],
                }
            )
        )
    pooled_role_gap = float(
        pd.concat(role_audit_frames, ignore_index=True)
        .groupby(["Season", "player_id"])["primary_role"]
        .sum()
        .abs()
        .max()
    )
    if pooled_role_gap > 1e-8:
        raise ValueError(
            f"Pooled player-season roles do not center exactly: {pooled_role_gap}"
        )
    sample_audit = {
        "by_season": sample_audit_by_season,
        "eligible_model_rows": int(len(model)),
        "players_with_assignment_exposure": int(len(shares)),
        "max_player_season_centered_role_sum_gap": pooled_role_gap,
    }
    targets, target_keys, baselines, target_audit = prepare_targets(model)
    player_ids = collect_player_ids(model)

    cv_mode = args.cv_mode
    if cv_mode == "auto":
        cv_mode = "season" if len(seasons) > 1 else "game"
    cv_folds = len(seasons) if cv_mode == "season" else args.cv_folds

    if args.role_alpha is None:
        cv = cross_validate_role_alpha(
            model,
            targets[:, -1],
            player_ids,
            args.off_alpha,
            args.def_alpha,
            parse_alpha_grid(args.role_alpha_grid),
            cv_folds,
            cv_mode,
        )
        absolute_selected_alpha, cv = select_role_alpha_one_se(cv, cv_folds)
        selected_alpha, cv = select_paired_role_alpha_one_se(cv)
        minimum_alpha = float(
            cv.sort_values(["mean_mse", "role_alpha"], ascending=[True, False])
            .iloc[0]["role_alpha"]
        )
    else:
        selected_alpha = float(args.role_alpha)
        minimum_alpha = selected_alpha
        absolute_selected_alpha = selected_alpha
        cv = None
    logging.info(
        "Selected primary-role alpha %.0f (minimum-MSE alpha %.0f)",
        selected_alpha,
        minimum_alpha,
    )

    X = build_design_matrix(
        model, player_ids, args.off_alpha, args.def_alpha, selected_alpha
    )
    beta = fit_coefficients(X, targets)
    long, summary, result_audit = build_results(
        model,
        shares,
        player_ids,
        beta,
        target_keys,
        args.off_alpha,
        args.def_alpha,
        selected_alpha,
    )
    sensitivity_long = None
    sensitivity_summary = None
    sensitivity_audit = None
    if minimum_alpha != selected_alpha:
        sensitivity_X = build_design_matrix(
            model,
            player_ids,
            args.off_alpha,
            args.def_alpha,
            minimum_alpha,
        )
        sensitivity_beta = fit_coefficients(sensitivity_X, targets)
        sensitivity_long, sensitivity_summary, sensitivity_audit = build_results(
            model,
            shares,
            player_ids,
            sensitivity_beta,
            target_keys,
            args.off_alpha,
            args.def_alpha,
            minimum_alpha,
        )
    reliability, reliability_audit = build_split_reliability(
        model,
        targets,
        seasons,
        player_ids,
        args.off_alpha,
        args.def_alpha,
        selected_alpha,
        args.reliability_min_exposures,
    )
    player_names = load_matchup_player_name_map(Path(args.matchup_root))
    long = apply_player_name_map(long, player_names)
    summary = apply_player_name_map(summary, player_names)
    reliability = apply_player_name_map(reliability, player_names)
    sensitivity_long = apply_player_name_map(sensitivity_long, player_names)
    sensitivity_summary = apply_player_name_map(
        sensitivity_summary, player_names
    )
    reader = build_reader_summary(
        summary,
        min_exposures=args.reader_min_exposures,
        min_seasons=args.reader_min_seasons,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    range_suffix = (
        f"{start_season % 100:02d}"
        if start_season == end_season
        else f"{start_season % 100:02d}_{end_season % 100:02d}"
    )
    stem = (
        f"nba_primary_defender_net_decomp_{range_suffix}_rs_"
        f"a{int(args.off_alpha)}_{int(args.def_alpha)}_role{int(selected_alpha)}"
    )
    reader_path = output_dir / f"{stem}_reader.csv"
    summary_path = output_dir / f"{stem}_summary.csv"
    long_path = output_dir / f"{stem}_components.csv"
    baseline_path = output_dir / f"{stem}_baselines.csv"
    manifest_path = output_dir / f"{stem}_manifest.json"
    reader.to_csv(reader_path, index=False)
    summary.to_csv(summary_path, index=False, float_format="%.4f")
    long.to_csv(long_path, index=False, float_format="%.4f")
    baselines.to_csv(baseline_path, index=False, float_format="%.4f")
    reliability_path = None
    if reliability is not None:
        reliability_path = output_dir / f"{stem}_split_reliability.csv"
        reliability.to_csv(reliability_path, index=False, float_format="%.4f")
    sensitivity_outputs = None
    if sensitivity_summary is not None and sensitivity_long is not None:
        sensitivity_summary_path = output_dir / f"{stem}_minimum_mse_sensitivity_summary.csv"
        sensitivity_long_path = output_dir / f"{stem}_minimum_mse_sensitivity_components.csv"
        sensitivity_summary.to_csv(
            sensitivity_summary_path, index=False, float_format="%.4f"
        )
        sensitivity_long.to_csv(
            sensitivity_long_path, index=False, float_format="%.4f"
        )
        sensitivity_outputs = {
            "summary": str(sensitivity_summary_path),
            "components": str(sensitivity_long_path),
        }
    if cv is not None:
        cv_path = output_dir / f"{stem}_role_alpha_cv.csv"
        cv.to_csv(cv_path, index=False, float_format="%.8f")
    manifest = {
        "status": "research_only",
        "provider": "nba_matchup",
        "seasons": seasons,
        "sample": "regular-season first-chance FGA possessions with one exact assignment",
        "interpretation": (
            "Primary versus nonprimary assignment association on the common shot sample; "
            "nonprimary is not a pure help-defense measure and frequency is not deterrence."
        ),
        "input_paths": input_paths,
        "matchup_root": str(args.matchup_root),
        "cv_mode": cv_mode,
        "cv_folds": cv_folds,
        "off_alpha": float(args.off_alpha),
        "def_alpha": float(args.def_alpha),
        "selected_role_alpha_one_se": selected_alpha,
        "absolute_error_one_se_role_alpha": absolute_selected_alpha,
        "minimum_mse_role_alpha": minimum_alpha,
        "assignment_audit": assignment_audit,
        "sample_audit": sample_audit,
        "target_audit": target_audit,
        "result_audit": result_audit,
        "minimum_mse_sensitivity_audit": sensitivity_audit,
        "split_reliability_audit": reliability_audit,
        "reader_contract": {
            "sort": "primary_advantage_descending",
            "minimum_assignment_exposures": int(args.reader_min_exposures),
            "minimum_assignment_seasons": int(args.reader_min_seasons),
            "impact_decimals": 2,
            "primary_share_percentage_decimals": 1,
        },
        "outputs": {
            "reader": str(reader_path),
            "summary": str(summary_path),
            "components": str(long_path),
            "baselines": str(baseline_path),
            "minimum_mse_sensitivity": sensitivity_outputs,
            "split_reliability": (
                None if reliability_path is None else str(reliability_path)
            ),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    logging.info("Wrote %s", reader_path)
    logging.info("Wrote %s", summary_path)
    logging.info("Wrote %s", long_path)
    logging.info("Wrote %s", manifest_path)
    display = summary[
        [
            "player_name",
            "def_parent_impact",
            "primary_contribution",
            "nonprimary_contribution",
            "primary_vs_nonprimary_contrast",
            "primary_share",
            "primary_assignments",
        ]
    ]
    print("\nTop assigned-shot net defense parent (positive = good)")
    print(display.head(args.display).to_string(index=False))
    print("\nLargest primary-over-nonprimary contrasts")
    print(
        display.sort_values(
            "primary_vs_nonprimary_contrast", ascending=False
        ).head(args.display).to_string(index=False)
    )
    return reader_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a research-only centered primary/nonprimary defender split "
            "of first-chance net shot DECOMP."
        )
    )
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument(
        "--start-season",
        type=int,
        help="First season-ending year. Omit for the single --season year.",
    )
    parser.add_argument("--raw-path")
    parser.add_argument("--processed-path")
    parser.add_argument("--matchup-root", default=str(DEFAULT_MATCHUP_ROOT))
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    parser.add_argument("--off-alpha", type=float, default=3000.0)
    parser.add_argument("--def-alpha", type=float, default=5000.0)
    parser.add_argument("--role-alpha", type=float)
    parser.add_argument("--role-alpha-grid")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--cv-mode",
        choices=["auto", "game", "season"],
        default="auto",
        help="Use season-held-out CV automatically for multi-season runs.",
    )
    parser.add_argument("--display", type=int, default=20)
    parser.add_argument(
        "--reliability-min-exposures",
        type=int,
        default=1000,
        help="Minimum assignment exposures in both early/late reliability blocks.",
    )
    parser.add_argument(
        "--reader-min-exposures",
        type=int,
        default=READER_MIN_EXPOSURES,
        help="Minimum assignment exposures in the compact reader CSV.",
    )
    parser.add_argument(
        "--reader-min-seasons",
        type=int,
        default=READER_MIN_SEASONS,
        help="Minimum assignment seasons in the compact reader CSV.",
    )
    args = parser.parse_args()
    if args.cv_folds < 2:
        parser.error("--cv-folds must be at least 2.")
    if args.start_season is not None and args.start_season > args.season:
        parser.error("--start-season cannot be later than --season.")
    if min(args.off_alpha, args.def_alpha) <= 0:
        parser.error("Base alphas must be positive.")
    if args.role_alpha is not None and args.role_alpha <= 0:
        parser.error("--role-alpha must be positive.")
    if args.reliability_min_exposures < 1:
        parser.error("--reliability-min-exposures must be positive.")
    if args.reader_min_exposures < 1:
        parser.error("--reader-min-exposures must be positive.")
    if args.reader_min_seasons < 1:
        parser.error("--reader-min-seasons must be positive.")
    return args


if __name__ == "__main__":
    run(parse_args())
