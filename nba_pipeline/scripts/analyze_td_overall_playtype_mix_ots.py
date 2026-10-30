#!/usr/bin/env python3
"""Build TD overall playtype-mix RAPM and test it against future oTS.

The event target is lag-priced and outcome-invariant in the feature season.
Positive difficulty means a harder-than-reference PBP playtype.  One row is
used per FGA, so the offensive parent answers whether the *overall* lineup
shot diet becomes harder while a player is on the floor.  A player-season
shot-share-centered shooter interaction separates own from teammate mix while
preserving the exact identity

    parent = shot_share * own + (1 - shot_share) * teammate.

This is research-only.  Historical transition labels are taken from the
frozen historical-safe transition archive; pre-2025 labels are retrospective
measurement and therefore do not create a pristine historical as-of test.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

try:
    from .analyze_pbp_playtype_difficulty_ots import (
        CREATION_CONTROL_FEATURES,
        CUT_PATTERN,
        DEFAULT_BOX,
        FLOATING_SCORING_FEATURES,
        RIM_PATTERN,
        SELF_CREATED_PATTERN,
        compare_models,
        joint_season_forward_ridge,
        per_target_metrics,
        _select_alpha,
    )
    from .process_rapm_blocks.common import _base_processing, build_shot_flags_df
    from .run_joint_ts_shooter_interaction import (
        DEFENSE_COLS,
        OFFENSE_COLS,
        build_joint_design_matrix,
        collect_player_ids,
        fit_coefficients,
    )
except ImportError:  # Direct script execution.
    from analyze_pbp_playtype_difficulty_ots import (
        CREATION_CONTROL_FEATURES,
        CUT_PATTERN,
        DEFAULT_BOX,
        FLOATING_SCORING_FEATURES,
        RIM_PATTERN,
        SELF_CREATED_PATTERN,
        compare_models,
        joint_season_forward_ridge,
        per_target_metrics,
        _select_alpha,
    )
    from process_rapm_blocks.common import _base_processing, build_shot_flags_df
    from run_joint_ts_shooter_interaction import (
        DEFENSE_COLS,
        OFFENSE_COLS,
        build_joint_design_matrix,
        collect_player_ids,
        fit_coefficients,
    )


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PIPELINE_ROOT / "raw_data"
DEFAULT_TRANSITION_DIR = PIPELINE_ROOT / "results" / "sq_transition_backfill"
DEFAULT_FORWARD_ROWS = (
    PIPELINE_ROOT
    / "results"
    / "pbp_playtype_difficulty"
    / "2023_2026"
    / "pbp_playtype_future_ots_analysis.csv"
)
DEFAULT_OUTPUT = PIPELINE_ROOT / "results" / "td_overall_playtype_mix"
DEFAULT_GAME_DATES = Path(
    "/Users/russ/Documents/Codex/basketball/nba/data-sources/"
    "matchup-fga/game_dates.csv"
)

MIX_CATEGORIES = [
    "TRANSITION",
    "CUT_2",
    "SELF_CREATED_3",
    "OTHER_3",
    "SELF_CREATED_2",
    "OTHER_RIM_2",
    "OTHER_2",
]
PRICE_SHRINK_FGA = 100.0

LOGGER = logging.getLogger("td_overall_playtype_mix")


def normalize_game_id(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype("Int64")
    return numeric.astype(str).str.replace("<NA>", "", regex=False).str.zfill(10)


def classify_mix_shots(
    frame: pd.DataFrame,
    is_three: pd.Series,
    is_transition: pd.Series,
) -> pd.Series:
    """Assign transition first, then the frozen six PBP shot families."""
    description = (
        frame.get("home_description", pd.Series("", index=frame.index))
        .fillna("")
        .astype(str)
        .str.cat(
            frame.get("visitor_description", pd.Series("", index=frame.index))
            .fillna("")
            .astype(str),
            sep=" ",
        )
        .str.lower()
    )
    transition = is_transition.fillna(False).astype(bool)
    cut = description.str.contains(CUT_PATTERN, regex=True)
    self_created = description.str.contains(SELF_CREATED_PATTERN, regex=True)
    rim = description.str.contains(RIM_PATTERN, regex=True)
    return pd.Series(
        np.select(
            [
                transition,
                (~transition) & cut & ~is_three,
                (~transition) & self_created & is_three,
                (~transition) & is_three,
                (~transition) & self_created & ~is_three,
                (~transition) & rim & ~is_three,
            ],
            [
                "TRANSITION",
                "CUT_2",
                "SELF_CREATED_3",
                "OTHER_3",
                "SELF_CREATED_2",
                "OTHER_RIM_2",
            ],
            default="OTHER_2",
        ),
        index=frame.index,
        dtype="object",
    )


def estimate_lag_prices(
    shots: pd.DataFrame,
    *,
    pricing_season: int,
    feature_season: int,
    shrink_fga: float = PRICE_SHRINK_FGA,
) -> pd.DataFrame:
    global_mean = float(shots["actual_points"].mean())
    stats = (
        shots.groupby("mix_category", sort=False)["actual_points"]
        .agg(["mean", "count"])
        .reindex(MIX_CATEGORIES)
    )
    stats["mean"] = stats["mean"].fillna(global_mean)
    stats["count"] = stats["count"].fillna(0.0)
    stats["category_price"] = (
        stats["count"] * stats["mean"] + shrink_fga * global_mean
    ) / (stats["count"] + shrink_fga)
    stats["difficulty_pp"] = 50.0 * (global_mean - stats["category_price"])
    stats = stats.reset_index().rename(columns={"index": "mix_category"})
    stats.insert(0, "pricing_season", int(pricing_season))
    stats.insert(1, "feature_season", int(feature_season))
    stats["pricing_global_points_per_fga"] = global_mean
    return stats


def load_game_dates(path: Path) -> pd.DataFrame:
    dates = pd.read_csv(path, usecols=["GAME_ID", "date", "playoffs"])
    dates = dates[~dates["playoffs"].fillna(False).astype(bool)].copy()
    dates["game_id"] = normalize_game_id(dates["GAME_ID"])
    dates["game_date"] = pd.to_datetime(
        dates["date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    return dates[["game_id", "game_date"]].dropna().drop_duplicates("game_id")


def load_processed_date_fallback(season: int) -> pd.DataFrame:
    path = PIPELINE_ROOT / "processed" / f"TS_SELF_SHOT{str(season)[-2:]}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["game_id", "game_date"])
    dates = pd.read_parquet(path, columns=["game_id", "game_date"])
    dates["game_id"] = normalize_game_id(dates["game_id"])
    dates["game_date"] = pd.to_datetime(dates["game_date"], errors="coerce")
    return dates.dropna().drop_duplicates("game_id")


def load_transition_labels(path: Path) -> pd.DataFrame:
    transition = pd.read_parquet(
        path, columns=["game_id", "event_num", "transition_label_operating"]
    )
    transition["game_id"] = normalize_game_id(transition["game_id"])
    transition["event_num"] = pd.to_numeric(
        transition["event_num"], errors="coerce"
    ).astype("Int64")
    transition = transition.dropna(subset=["event_num"]).drop_duplicates(
        ["game_id", "event_num"], keep=False
    )
    return transition.rename(
        columns={"transition_label_operating": "is_transition_frozen"}
    )


def _lineup_payload(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    shooter = pd.to_numeric(frame["player1_id"], errors="coerce").to_numpy()
    away = frame[[f"a{i}" for i in range(1, 6)]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy()
    home = frame[[f"h{i}" for i in range(1, 6)]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy()
    in_away = (away == shooter[:, None]).sum(axis=1)
    in_home = (home == shooter[:, None]).sum(axis=1)
    valid = (
        np.isfinite(shooter)
        & ((in_away == 1) ^ (in_home == 1))
        & np.isfinite(away).all(axis=1)
        & np.isfinite(home).all(axis=1)
        & (away > 0).all(axis=1)
        & (home > 0).all(axis=1)
    )
    kept = frame.loc[valid].copy()
    shooter = shooter[valid].astype(np.int64)
    away = away[valid].astype(np.int64)
    home = home[valid].astype(np.int64)
    home_offense = in_home[valid] == 1
    offense = np.where(home_offense[:, None], home, away)
    defense = np.where(home_offense[:, None], away, home)
    shooter_mask = offense == shooter[:, None]
    unique_off = np.array([len(set(row)) == 5 for row in offense])
    unique_def = np.array([len(set(row)) == 5 for row in defense])
    no_overlap = np.array(
        [not bool(set(o) & set(d)) for o, d in zip(offense, defense)]
    )
    second_valid = (
        (shooter_mask.sum(axis=1) == 1) & unique_off & unique_def & no_overlap
    )
    kept = kept.loc[second_valid].copy()
    shooter = shooter[second_valid]
    offense = offense[second_valid]
    defense = defense[second_valid]
    shooter_mask = offense == shooter[:, None]
    context = offense[~shooter_mask].reshape(-1, 4)
    kept["Focal_Player"] = shooter
    for idx in range(4):
        kept[f"O{idx + 1}"] = context[:, idx]
    for idx in range(5):
        kept[f"D{idx + 1}"] = defense[:, idx]
    return kept, offense


def load_season_shots(
    season: int,
    *,
    raw_dir: Path,
    transition_dir: Path,
    game_dates: pd.DataFrame,
) -> pd.DataFrame:
    raw_path = raw_dir / f"NBA{str(season)[-2:]}.parquet"
    raw = _base_processing(raw_path)
    if raw is None:
        raise RuntimeError(f"Could not process {raw_path}")
    flags = build_shot_flags_df(raw)
    shots = raw.loc[flags["is_fga"]].copy()
    shot_flags = flags.loc[shots.index]
    shots["game_id"] = normalize_game_id(shots["game_id"])
    shots["event_num"] = pd.to_numeric(shots["event_num"], errors="coerce").astype(
        "Int64"
    )
    transition = load_transition_labels(
        transition_dir / f"transition_backfill_{season}_rs.parquet"
    )
    shots = shots.merge(
        transition,
        on=["game_id", "event_num"],
        how="left",
        validate="one_to_one",
    )
    if shots["is_transition_frozen"].isna().any():
        raise ValueError(
            f"Season {season} has {int(shots['is_transition_frozen'].isna().sum())} "
            "FGA without a unique frozen transition label"
        )
    # The merge resets the index, so recover shot flags from the merged text.
    merged_flags = build_shot_flags_df(shots)
    shots["mix_category"] = classify_mix_shots(
        shots,
        merged_flags["is_3pt"].astype(bool),
        shots["is_transition_frozen"].astype(bool),
    )
    shots["actual_points"] = merged_flags["shot_points"].astype(float)
    shots["Season"] = int(season)
    shots, _ = _lineup_payload(shots)

    dates = game_dates
    if season >= 2026:
        dates = pd.concat(
            [game_dates, load_processed_date_fallback(season)], ignore_index=True
        ).drop_duplicates("game_id", keep="last")
    # Gabriel 1997-2013 raw files carry their own game date; keep it only as a
    # fallback behind the shared schedule file.
    raw_dates = None
    if "game_date" in shots.columns:
        raw_dates = pd.to_datetime(shots.pop("game_date"), errors="coerce").to_numpy()
    shots = shots.merge(dates, on="game_id", how="left", validate="many_to_one")
    if raw_dates is not None:
        shots["game_date"] = shots["game_date"].fillna(pd.Series(raw_dates, index=shots.index))
    if shots["game_date"].isna().any():
        raise ValueError(
            f"Season {season} has {int(shots['game_date'].isna().sum())} "
            "complete-lineup FGA without game dates"
        )
    keep = [
        "Season",
        "game_id",
        "event_num",
        "game_date",
        "Focal_Player",
        *OFFENSE_COLS[1:],
        *DEFENSE_COLS,
        "mix_category",
        "actual_points",
    ]
    return shots[keep].reset_index(drop=True)


def build_attempt_panel(
    *,
    pricing_start_season: int,
    end_season: int,
    raw_dir: Path,
    transition_dir: Path,
    game_dates_path: Path,
    price_first_season_on_itself: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lag-priced attempt rows for every season after ``pricing_start_season``.

    With ``price_first_season_on_itself`` the first season is also emitted,
    priced on its own shots, for the earliest raw season that has no prior
    season to price from (1997).  Its price rows carry
    ``pricing_season == feature_season`` so the fallback stays auditable.
    """
    dates = load_game_dates(game_dates_path)
    attempt_parts = []
    price_parts = []
    prior_shots = None
    for season in range(pricing_start_season, end_season + 1):
        LOGGER.info("Building playtype attempt rows for %d", season)
        shots = load_season_shots(
            season,
            raw_dir=raw_dir,
            transition_dir=transition_dir,
            game_dates=dates,
        )
        if prior_shots is None and price_first_season_on_itself:
            prior_shots = shots[["mix_category", "actual_points"]].copy()
            pricing_season = season
        else:
            pricing_season = season - 1
        if prior_shots is not None:
            prices = estimate_lag_prices(
                prior_shots,
                pricing_season=pricing_season,
                feature_season=season,
            )
            price_parts.append(prices)
            lookup = prices.set_index("mix_category")["difficulty_pp"]
            shots["lag_difficulty_pp"] = shots["mix_category"].map(lookup)
            shots["lag_difficulty_rate"] = shots["lag_difficulty_pp"] / 100.0
            attempt_parts.append(shots)
        prior_shots = shots[["mix_category", "actual_points"]].copy()
    return (
        pd.concat(attempt_parts, ignore_index=True),
        pd.concat(price_parts, ignore_index=True),
    )


def time_decay_weights(dates: pd.Series, half_life_days: float) -> np.ndarray:
    parsed = pd.to_datetime(dates, errors="coerce")
    if parsed.isna().any():
        raise ValueError("Time-decay input contains missing game dates")
    age_days = (parsed.max() - parsed).dt.days.to_numpy(dtype=float)
    return np.power(2.0, -age_days / float(half_life_days))


def compute_weighted_player_season_shares(
    frame: pd.DataFrame, weights: np.ndarray
) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "Season": frame["Season"].to_numpy(dtype=int),
            "player_id": frame["Focal_Player"].to_numpy(dtype=int),
            "weight": weights,
        }
    )
    numerator = base.groupby(["Season", "player_id"])["weight"].sum()
    denominator_parts = []
    for column in OFFENSE_COLS:
        denominator_parts.append(
            pd.DataFrame(
                {
                    "Season": frame["Season"].to_numpy(dtype=int),
                    "player_id": frame[column].to_numpy(dtype=int),
                    "weight": weights,
                }
            )
        )
    denominator = (
        pd.concat(denominator_parts, ignore_index=True)
        .groupby(["Season", "player_id"])["weight"]
        .sum()
    )
    shares = pd.DataFrame(
        {"self_weight": numerator, "offensive_weight": denominator}
    ).fillna(0.0)
    shares["shot_share"] = shares["self_weight"] / shares["offensive_weight"]
    return shares.reset_index()


def weighted_exposures(
    frame: pd.DataFrame, columns: list[str], weights: np.ndarray
) -> pd.Series:
    parts = [
        pd.DataFrame(
            {"player_id": frame[column].to_numpy(dtype=int), "weight": weights}
        )
        for column in columns
    ]
    return pd.concat(parts, ignore_index=True).groupby("player_id")["weight"].sum()


def build_td_results(
    frame: pd.DataFrame,
    *,
    feature_season: int,
    half_life_days: float,
    off_alpha: float,
    def_alpha: float,
    role_alpha: float,
    names: pd.DataFrame,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    weights = time_decay_weights(frame["game_date"], half_life_days)
    shares = compute_weighted_player_season_shares(frame, weights)
    players = collect_player_ids(frame)
    baseline = frame.groupby("Season")["lag_difficulty_rate"].mean()
    y = (
        frame["lag_difficulty_rate"].to_numpy(dtype=float)
        - frame["Season"].map(baseline).to_numpy(dtype=float)
    )
    X, interaction = build_joint_design_matrix(
        frame, players, shares, off_alpha, def_alpha, role_alpha
    )
    sqrt_weights = np.sqrt(weights)
    X_weighted = X.multiply(sqrt_weights[:, None]).tocsr()
    beta = fit_coefficients(X_weighted, y * sqrt_weights)

    interaction_parts = []
    for index, column in enumerate(OFFENSE_COLS):
        interaction_parts.append(
            pd.DataFrame(
                {
                    "Season": frame["Season"].to_numpy(dtype=int),
                    "player_id": frame[column].to_numpy(dtype=int),
                    "weighted_interaction": interaction[:, index] * weights,
                }
            )
        )
    interaction_gap = (
        pd.concat(interaction_parts, ignore_index=True)
        .groupby(["Season", "player_id"])["weighted_interaction"]
        .sum()
        .abs()
        .max()
    )

    count = len(players)
    offense = beta[:count] / np.sqrt(off_alpha)
    defense = beta[count : 2 * count] / np.sqrt(def_alpha)
    role = beta[2 * count :] / np.sqrt(role_alpha)
    off_exp = weighted_exposures(frame, OFFENSE_COLS, weights).reindex(players).fillna(0)
    def_exp = weighted_exposures(frame, DEFENSE_COLS, weights).reindex(players).fillna(0)
    offense_center = float(np.average(offense, weights=off_exp.to_numpy()))
    defense_center = float(np.average(defense, weights=def_exp.to_numpy()))
    offense = offense - offense_center
    defense = defense - defense_center

    self_exp = (
        pd.DataFrame(
            {"player_id": frame["Focal_Player"].to_numpy(dtype=int), "weight": weights}
        )
        .groupby("player_id")["weight"]
        .sum()
        .reindex(players)
        .fillna(0)
    )
    usage = self_exp.to_numpy() / off_exp.to_numpy()
    parent = offense * 100.0
    contrast = role * 100.0
    own = parent + (1.0 - usage) * contrast
    teammate = parent - usage * contrast
    own_contribution = usage * own
    teammate_contribution = (1.0 - usage) * teammate
    result = pd.DataFrame(
        {
            "feature_season": int(feature_season),
            "player_id": players,
            "td_overall_playtype_difficulty_rapm": parent,
            "own_vs_teammate_difficulty_contrast": contrast,
            "conditional_own_playtype_difficulty": own,
            "conditional_teammate_playtype_difficulty": teammate,
            "own_mix_contribution": own_contribution,
            "teammate_mix_contribution": teammate_contribution,
            "td_shot_share": usage,
            "weighted_offensive_fga_exposure": off_exp.to_numpy(),
            "def_playtype_difficulty_rapm": defense * 100.0,
        }
    )
    result["parent_decomposition_gap"] = (
        result["own_mix_contribution"]
        + result["teammate_mix_contribution"]
        - result["td_overall_playtype_difficulty_rapm"]
    )
    result = result.merge(names, on="player_id", how="left", validate="one_to_one")
    metadata = {
        "feature_season": int(feature_season),
        "rows": int(len(frame)),
        "players": int(len(players)),
        "reference_date": str(pd.to_datetime(frame["game_date"]).max().date()),
        "max_weighted_interaction_gap": float(interaction_gap),
        "max_parent_decomposition_gap": float(
            result["parent_decomposition_gap"].abs().max()
        ),
        "offense_centering_offset_rate": offense_center,
        "defense_centering_offset_rate": defense_center,
    }
    baselines = baseline.rename("season_mean_lag_difficulty_rate").reset_index()
    baselines.insert(0, "feature_season", int(feature_season))
    return result, metadata, baselines


def model_features() -> dict[str, list[str]]:
    base = FLOATING_SCORING_FEATURES + CREATION_CONTROL_FEATURES
    own = "pbp_role_added_100"
    overall = "td_overall_playtype_difficulty_rapm"
    synergy = "synergy_role_added_100"
    return {
        "box_scoring_baseline": base,
        "own_pbp_role": base + [own],
        "td_overall_mix": base + [overall],
        "own_plus_overall_mix": base + [own, overall],
        "synergy_role": base + [synergy],
        "synergy_plus_overall_mix": base + [synergy, overall],
    }


def coefficient_audit(
    frame: pd.DataFrame,
    features: dict[str, list[str]],
    *,
    first_test_target_season: int = 2025,
) -> pd.DataFrame:
    """Export fold-level standardized and physical coefficients."""
    required = sorted(
        set().union(*(set(columns) for columns in features.values()))
        | {"future_oTS", "future_off_poss", "target_season"}
    )
    complete = frame.dropna(subset=required).copy()
    rows = []
    for target_season in sorted(complete["target_season"].unique()):
        target_season = int(target_season)
        if target_season < first_test_target_season:
            continue
        train = complete[complete["target_season"].lt(target_season)]
        test = complete[complete["target_season"].eq(target_season)]
        baseline_features = features["box_scoring_baseline"]
        baseline_scaler = StandardScaler().fit(train[baseline_features])
        alpha = _select_alpha(
            baseline_scaler.transform(train[baseline_features]),
            train["future_oTS"].to_numpy(dtype=float),
            train["future_off_poss"].to_numpy(dtype=float),
        )
        for model_name, columns in features.items():
            scaler = StandardScaler().fit(train[columns])
            model = Ridge(alpha=alpha).fit(
                scaler.transform(train[columns]),
                train["future_oTS"].to_numpy(dtype=float),
                sample_weight=train["future_off_poss"].to_numpy(dtype=float),
            )
            for column, standardized, scale in zip(
                columns, model.coef_, scaler.scale_
            ):
                rows.append(
                    {
                        "target_season": target_season,
                        "model": model_name,
                        "feature": column,
                        "train_rows": int(len(train)),
                        "test_rows": int(len(test)),
                        "selected_alpha": float(alpha),
                        "standardized_coefficient": float(standardized),
                        "physical_coefficient": float(standardized / scale),
                    }
                )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pricing-start-season", type=int, default=2018)
    parser.add_argument("--first-feature-season", type=int, default=2023)
    parser.add_argument("--last-feature-season", type=int, default=2026)
    parser.add_argument("--half-life", type=float, default=700.0)
    parser.add_argument("--off-alpha", type=float, default=3000.0)
    parser.add_argument("--def-alpha", type=float, default=5000.0)
    parser.add_argument("--role-alpha", type=float, default=10000.0)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--transition-dir", type=Path, default=DEFAULT_TRANSITION_DIR)
    parser.add_argument("--game-dates", type=Path, default=DEFAULT_GAME_DATES)
    parser.add_argument("--box", type=Path, default=DEFAULT_BOX)
    parser.add_argument("--forward-rows", type=Path, default=DEFAULT_FORWARD_ROWS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rebuild-attempts", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    attempts_path = args.output_dir / (
        f"attempt_panel_{args.pricing_start_season + 1}_{args.last_feature_season}.parquet"
    )
    prices_path = args.output_dir / "lag_category_prices.csv"
    if attempts_path.exists() and not args.rebuild_attempts:
        attempts = pd.read_parquet(attempts_path)
        prices = pd.read_csv(prices_path)
    else:
        attempts, prices = build_attempt_panel(
            pricing_start_season=args.pricing_start_season,
            end_season=args.last_feature_season,
            raw_dir=args.raw_dir,
            transition_dir=args.transition_dir,
            game_dates_path=args.game_dates,
        )
        attempts.to_parquet(attempts_path, index=False)
        prices.to_csv(prices_path, index=False)

    names = pd.read_csv(args.box, usecols=["nba_id", "ShortName", "year"])
    names = (
        names.sort_values("year")
        .drop_duplicates("nba_id", keep="last")
        .rename(columns={"nba_id": "player_id", "ShortName": "player_name"})
        [["player_id", "player_name"]]
    )
    result_parts = []
    baseline_parts = []
    solve_metadata = []
    for feature_season in range(
        args.first_feature_season, args.last_feature_season + 1
    ):
        fit = attempts[attempts["Season"].le(feature_season)].copy()
        result, metadata, baselines = build_td_results(
            fit,
            feature_season=feature_season,
            half_life_days=args.half_life,
            off_alpha=args.off_alpha,
            def_alpha=args.def_alpha,
            role_alpha=args.role_alpha,
            names=names,
        )
        result_parts.append(result)
        baseline_parts.append(baselines)
        solve_metadata.append(metadata)
    annual = pd.concat(result_parts, ignore_index=True)
    annual.to_csv(args.output_dir / "annual_td_overall_playtype_mix.csv", index=False)
    pd.concat(baseline_parts, ignore_index=True).to_csv(
        args.output_dir / "annual_season_target_baselines.csv", index=False
    )

    forward = pd.read_csv(args.forward_rows)
    evaluation = forward.merge(
        annual[
            [
                "feature_season",
                "player_id",
                "td_overall_playtype_difficulty_rapm",
            ]
        ],
        on=["feature_season", "player_id"],
        how="inner",
        validate="one_to_one",
    )
    features = model_features()
    metrics, predictions = joint_season_forward_ridge(
        evaluation, features, chronological=True
    )
    comparisons = compare_models(
        predictions,
        [
            ("own_pbp_vs_box", "box_scoring_baseline", "own_pbp_role"),
            ("overall_mix_vs_box", "box_scoring_baseline", "td_overall_mix"),
            ("overall_mix_after_own", "own_pbp_role", "own_plus_overall_mix"),
            ("own_after_overall_mix", "td_overall_mix", "own_plus_overall_mix"),
            ("own_plus_mix_vs_synergy", "synergy_role", "own_plus_overall_mix"),
            (
                "overall_mix_after_synergy",
                "synergy_role",
                "synergy_plus_overall_mix",
            ),
        ],
        evaluation="chronological_td_overall_playtype_mix",
    )
    evaluation.to_csv(args.output_dir / "future_ots_analysis_rows.csv", index=False)
    metrics.to_csv(args.output_dir / "future_ots_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "future_ots_predictions.csv", index=False)
    comparisons.to_csv(args.output_dir / "future_ots_comparisons.csv", index=False)
    coefficient_audit(evaluation, features).to_csv(
        args.output_dir / "future_ots_coefficients.csv", index=False
    )
    per_target_metrics(predictions).to_csv(
        args.output_dir / "future_ots_per_target_metrics.csv", index=False
    )

    manifest = {
        "status": "development_only",
        "pricing_rule": "immediately prior regular season category points/FGA, shrunk by 100 FGA",
        "feature_assignment": "outcome-invariant PBP descriptions plus frozen historical-safe transition operating label",
        "transition_caveat": "pre-2025 transition labels are retrospectively measured with a classifier frozen from the 2025 labeled era; this is not a pristine historical as-of feature extraction test",
        "categories": MIX_CATEGORIES,
        "difficulty_definition": "50 * (prior-season league points/FGA - prior-season shrunk category points/FGA)",
        "parent_definition": "TD overall lineup FGA-mix difficulty RAPM; positive offense means harder overall playtypes",
        "interaction_definition": "time-decay-weighted (is_shooter - player-season shot share) centered interaction",
        "identity": "parent = shot_share*conditional_own + (1-shot_share)*conditional_teammate",
        "half_life_days": args.half_life,
        "off_alpha": args.off_alpha,
        "def_alpha": args.def_alpha,
        "role_alpha": args.role_alpha,
        "pricing_start_season": args.pricing_start_season,
        "feature_seasons": [args.first_feature_season, args.last_feature_season],
        "creation_controls": CREATION_CONTROL_FEATURES,
        "forward_targets_are_spent": [2025, 2026],
        "solve_metadata": solve_metadata,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("\nFuture-oTS metrics")
    print(metrics.sort_values("rmse").to_string(index=False))
    print("\nIncremental comparisons")
    print(comparisons.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
