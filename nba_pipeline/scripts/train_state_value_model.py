#!/usr/bin/env python3
"""
Train a 2026 baseline_mvp state-value model on possession snapshots.

This is a practical baseline_mvp implementation of the DRL/Shapley direction:
- state model trained from possession states to future net rating
- on-court player features included directly in the value model
- season player attributions extracted from XGBoost feature contributions

It is intentionally simpler than the paper:
- possession-level rather than event-level
- boosted trees rather than a neural distributional value network
- TreeSHAP on player indicators rather than coalition Shapley over embeddings

Usage:
    python nba_pipeline/scripts/train_state_value_model.py 26
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import coo_matrix, csr_matrix, hstack

from rapm import detect_file_type_and_prepare, transform_to_home_away_format


SCRIPT_DIR = Path(__file__).parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_ROOT.parent
PROCESSED_DIR = PIPELINE_ROOT / "processed"
RESULTS_DIR = PIPELINE_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "state_value_model"
AUTOCOMPLETE_MAP = PROJECT_ROOT / "autocomplete_map.csv"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


STATE_FEATURES = [
    "pre_margin_home",
    "abs_pre_margin_home",
    "seconds_remaining_game",
    "poss_remaining",
    "period",
    "home_poss",
    "clutch_flag",
]

SHRINKAGE_K = 2000
LEADERBOARD_MIN_STATES = 1500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an MVP state-value model from processed possession states."
    )
    parser.add_argument("year", type=int, help="Two-digit ending year, e.g. 26 for 2025-26")
    parser.add_argument(
        "--prefix",
        default="RAPM",
        help="Processed parquet prefix to use for possession states (default: RAPM)",
    )
    parser.add_argument(
        "--validation-frac",
        type=float,
        default=0.2,
        help="Fraction of dates to hold out at the end of the season for validation.",
    )
    parser.add_argument(
        "--num-boost-round",
        type=int,
        default=500,
        help="Maximum XGBoost boosting rounds.",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=40,
        help="Early stopping patience on validation RMSE.",
    )
    parser.add_argument(
        "--shap-batch-size",
        type=int,
        default=4000,
        help="Batch size for contribution extraction.",
    )
    return parser.parse_args()


def parse_time_quarter_to_seconds(time_str: str) -> float:
    if not isinstance(time_str, str) or ":" not in time_str:
        return 0.0
    mins, secs = time_str.split(":", 1)
    try:
        return float(int(mins) * 60 + int(secs))
    except ValueError:
        return 0.0


def compute_seconds_remaining_game(period: int, time_quarter: str) -> float:
    secs_in_period = parse_time_quarter_to_seconds(time_quarter)
    if period <= 4:
        return max(0.0, (4 - period) * 720 + secs_in_period)
    return max(0.0, secs_in_period)


def load_processed_states(prefix: str, year: int) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{prefix}{year:02d}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Processed parquet not found: {path}")
    logging.info("Loading %s", path)
    return pd.read_parquet(path)


def prepare_state_frame(prefix: str, year: int) -> pd.DataFrame:
    raw_df = load_processed_states(prefix, year)
    prepared = detect_file_type_and_prepare(raw_df, pure=True, prefix=prefix)
    transformed = transform_to_home_away_format(prepared)

    working = prepared.reset_index(drop=True).copy()
    transformed = transformed.reset_index(drop=True).copy()

    game_starts = working["game_id"].ne(working["game_id"].shift(1)).to_numpy()
    post_margin_home = (working["home_score"] - working["away_score"]).astype(float).to_numpy()
    pre_margin_home = np.roll(post_margin_home, 1)
    pre_margin_home[0] = 0.0
    pre_margin_home[game_starts] = 0.0

    home_points = working["home_score"].astype(float).to_numpy() - np.where(game_starts, 0.0, working["home_score"].shift(1).fillna(0.0).astype(float).to_numpy())
    away_points = working["away_score"].astype(float).to_numpy() - np.where(game_starts, 0.0, working["away_score"].shift(1).fillna(0.0).astype(float).to_numpy())
    reward_home = home_points - away_points

    final_scores = working.groupby("game_id", sort=False)[["home_score", "away_score"]].last()
    final_margin_home = (final_scores["home_score"] - final_scores["away_score"]).astype(float).rename("final_margin_home")
    working = working.join(final_margin_home, on="game_id")

    game_sizes = working.groupby("game_id")["game_id"].transform("size")
    game_index = working.groupby("game_id").cumcount()
    poss_remaining = (game_sizes - game_index).astype(int)

    seconds_remaining_game = [
        compute_seconds_remaining_game(period, time_quarter)
        for period, time_quarter in zip(working["period"], working["time_quarter"])
    ]

    target_remaining_margin = working["final_margin_home"].astype(float).to_numpy() - pre_margin_home
    target_future_netrt = 100.0 * target_remaining_margin / poss_remaining.clip(lower=1).to_numpy()
    home_win = (working["final_margin_home"].astype(float) > 0).astype(int).to_numpy()

    state_df = transformed.copy()
    state_df["date"] = pd.to_datetime(working["game_date"])
    state_df["pre_margin_home"] = pre_margin_home
    state_df["abs_pre_margin_home"] = np.abs(pre_margin_home)
    state_df["reward_home"] = reward_home
    state_df["final_margin_home"] = working["final_margin_home"].astype(float).to_numpy()
    state_df["poss_remaining"] = poss_remaining.to_numpy()
    state_df["seconds_remaining_game"] = np.asarray(seconds_remaining_game, dtype=float)
    state_df["target_future_netrt"] = target_future_netrt
    state_df["home_win"] = home_win
    state_df["clutch_flag"] = (
        (state_df["seconds_remaining_game"] <= 180)
        & (state_df["abs_pre_margin_home"] <= 5)
    ).astype(int)

    keep_cols = [
        "gameid",
        "date",
        "season",
        "period",
        "home_poss",
        "pre_margin_home",
        "abs_pre_margin_home",
        "reward_home",
        "final_margin_home",
        "poss_remaining",
        "seconds_remaining_game",
        "target_future_netrt",
        "home_win",
        "clutch_flag",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
    ]
    state_df = state_df[keep_cols].copy()
    logging.info(
        "Prepared %s possession states across %s games",
        len(state_df),
        state_df["gameid"].nunique(),
    )
    return state_df


def build_player_matrix(df: pd.DataFrame) -> Tuple[csr_matrix, List[int], Dict[int, int]]:
    player_cols = [f"h{i}" for i in range(1, 6)] + [f"a{i}" for i in range(1, 6)]
    unique_players = sorted(
        {
            int(pid)
            for pid in pd.unique(df[player_cols].to_numpy().ravel())
            if pd.notna(pid) and int(pid) > 0
        }
    )
    player_to_col = {pid: idx for idx, pid in enumerate(unique_players)}

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    n_rows = len(df)

    for row_idx, row in enumerate(df.itertuples(index=False)):
        for pid in (row.h1, row.h2, row.h3, row.h4, row.h5):
            pid_int = int(pid)
            rows.append(row_idx)
            cols.append(player_to_col[pid_int])
            data.append(1.0)
        for pid in (row.a1, row.a2, row.a3, row.a4, row.a5):
            pid_int = int(pid)
            rows.append(row_idx)
            cols.append(player_to_col[pid_int])
            data.append(-1.0)

    matrix = coo_matrix(
        (np.asarray(data, dtype=np.float32), (rows, cols)),
        shape=(n_rows, len(unique_players)),
        dtype=np.float32,
    ).tocsr()
    logging.info("Player matrix: %s rows x %s player features", n_rows, len(unique_players))
    return matrix, unique_players, player_to_col


def build_feature_matrix(df: pd.DataFrame) -> Tuple[csr_matrix, csr_matrix, List[int], List[str]]:
    player_matrix, player_ids, _ = build_player_matrix(df)
    state_dense = df[STATE_FEATURES].astype(np.float32).to_numpy()
    state_matrix = csr_matrix(state_dense)
    feature_names = [f"player_{pid}" for pid in player_ids] + STATE_FEATURES
    full_matrix = hstack([player_matrix, state_matrix], format="csr")
    logging.info("Full feature matrix shape: %s", full_matrix.shape)
    return full_matrix, player_matrix, player_ids, feature_names


def make_train_validation_split(df: pd.DataFrame, validation_frac: float) -> Tuple[np.ndarray, np.ndarray]:
    unique_dates = np.array(sorted(df["date"].dt.normalize().unique()))
    if len(unique_dates) < 4:
        raise ValueError("Need at least 4 unique dates for a train/validation split")

    n_val_dates = max(1, int(math.ceil(len(unique_dates) * validation_frac)))
    split_idx = max(1, len(unique_dates) - n_val_dates)
    train_dates = set(unique_dates[:split_idx])
    valid_dates = set(unique_dates[split_idx:])

    train_mask = df["date"].dt.normalize().isin(train_dates).to_numpy()
    valid_mask = df["date"].dt.normalize().isin(valid_dates).to_numpy()

    if not train_mask.any() or not valid_mask.any():
        raise ValueError("Train/validation split failed to produce non-empty partitions")

    logging.info(
        "Train/validation split: %s train states, %s validation states, %s validation dates",
        int(train_mask.sum()),
        int(valid_mask.sum()),
        len(valid_dates),
    )
    return train_mask, valid_mask


def train_value_model(
    X: csr_matrix,
    y: np.ndarray,
    feature_names: List[str],
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> Tuple[xgb.Booster, Dict[str, float], np.ndarray]:
    X_train = X[train_mask]
    X_valid = X[valid_mask]
    y_train = y[train_mask]
    y_valid = y[valid_mask]

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dvalid = xgb.DMatrix(X_valid, label=y_valid, feature_names=feature_names)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.05,
        "max_depth": 7,
        "min_child_weight": 50,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "lambda": 1.0,
        "alpha": 0.0,
        "tree_method": "hist",
        "max_bin": 256,
        "nthread": 8,
    }

    evals_result: Dict[str, Dict[str, List[float]]] = {}
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=25,
    )

    valid_pred = booster.predict(dvalid, iteration_range=(0, booster.best_iteration + 1))
    rmse = float(np.sqrt(np.mean((valid_pred - y_valid) ** 2)))
    corr = float(np.corrcoef(valid_pred, y_valid)[0, 1])
    baseline_pred = np.full_like(y_valid, y_train.mean())
    baseline_rmse = float(np.sqrt(np.mean((baseline_pred - y_valid) ** 2)))

    metrics = {
        "rmse_valid": rmse,
        "corr_valid": corr,
        "rmse_valid_baseline_mean": baseline_rmse,
        "best_iteration": int(booster.best_iteration),
    }
    logging.info("Validation RMSE: %.4f", rmse)
    logging.info("Validation correlation: %.4f", corr)
    logging.info("Mean baseline RMSE: %.4f", baseline_rmse)
    return booster, metrics, valid_pred


def load_name_map() -> Dict[int, str]:
    if not AUTOCOMPLETE_MAP.exists():
        return {}
    name_map = pd.read_csv(AUTOCOMPLETE_MAP)
    return {
        int(row.nba_id): row.player_name
        for row in name_map.itertuples(index=False)
        if pd.notna(row.nba_id)
    }


def compute_player_ratings(
    booster: xgb.Booster,
    X: csr_matrix,
    player_matrix: csr_matrix,
    df: pd.DataFrame,
    player_ids: List[int],
    feature_names: List[str],
    batch_size: int,
) -> pd.DataFrame:
    n_rows = X.shape[0]
    n_players = len(player_ids)

    total_sum = np.zeros(n_players, dtype=np.float64)
    total_count = np.zeros(n_players, dtype=np.int64)
    off_sum = np.zeros(n_players, dtype=np.float64)
    off_count = np.zeros(n_players, dtype=np.int64)
    def_sum = np.zeros(n_players, dtype=np.float64)
    def_count = np.zeros(n_players, dtype=np.int64)

    home_poss = df["home_poss"].to_numpy(dtype=np.int8)

    logging.info("Extracting player contributions in batches of %s", batch_size)
    for start in range(0, n_rows, batch_size):
        end = min(n_rows, start + batch_size)
        x_batch = X[start:end]
        dmatrix = xgb.DMatrix(x_batch, feature_names=feature_names)
        contribs = booster.predict(
            dmatrix,
            pred_contribs=True,
            iteration_range=(0, booster.best_iteration + 1),
            validate_features=False,
        )[:, :n_players]

        sign_batch = player_matrix[start:end].toarray()
        signed_contribs = contribs * sign_batch
        on_mask = sign_batch != 0

        home_poss_batch = home_poss[start:end][:, None]
        off_mask = ((sign_batch > 0) & (home_poss_batch == 1)) | ((sign_batch < 0) & (home_poss_batch == 0))
        def_mask = on_mask & ~off_mask

        total_sum += np.where(on_mask, signed_contribs, 0.0).sum(axis=0)
        total_count += on_mask.sum(axis=0)
        off_sum += np.where(off_mask, signed_contribs, 0.0).sum(axis=0)
        off_count += off_mask.sum(axis=0)
        def_sum += np.where(def_mask, signed_contribs, 0.0).sum(axis=0)
        def_count += def_mask.sum(axis=0)

    name_map = load_name_map()
    ratings = pd.DataFrame(
        {
            "player_id": player_ids,
            "player_name": [name_map.get(pid, str(pid)) for pid in player_ids],
            "states": total_count,
            "off_states": off_count,
            "def_states": def_count,
            "net_impact": np.divide(total_sum, total_count, out=np.zeros_like(total_sum), where=total_count > 0),
            "off_impact": np.divide(off_sum, off_count, out=np.zeros_like(off_sum), where=off_count > 0),
            "def_impact": np.divide(def_sum, def_count, out=np.zeros_like(def_sum), where=def_count > 0),
        }
    )
    shrink = ratings["states"] / (ratings["states"] + SHRINKAGE_K)
    ratings["shrunk_net_impact"] = ratings["net_impact"] * shrink
    ratings["shrunk_off_impact"] = ratings["off_impact"] * (ratings["off_states"] / (ratings["off_states"] + SHRINKAGE_K))
    ratings["shrunk_def_impact"] = ratings["def_impact"] * (ratings["def_states"] / (ratings["def_states"] + SHRINKAGE_K))
    ratings = ratings.sort_values(["shrunk_net_impact", "states"], ascending=[False, False]).reset_index(drop=True)
    return ratings


def save_outputs(
    year: int,
    prefix: str,
    state_df: pd.DataFrame,
    booster: xgb.Booster,
    metrics: Dict[str, float],
    ratings: pd.DataFrame,
    valid_mask: np.ndarray,
    valid_pred: np.ndarray,
) -> None:
    model_dir = OUTPUT_DIR / f"{prefix.lower()}_{year:02d}"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "state_value_model.json"
    booster.save_model(model_path)

    summary_path = model_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(
            {
                "year": year,
                "prefix": prefix,
                "n_states": int(len(state_df)),
                "n_games": int(state_df["gameid"].nunique()),
                "n_players": int(len(ratings)),
                **metrics,
            },
            f,
            indent=2,
        )

    ratings_path = model_dir / "player_ratings.csv"
    ratings.to_csv(ratings_path, index=False)

    filtered_ratings = ratings[ratings["states"] >= LEADERBOARD_MIN_STATES].copy()
    filtered_ratings_path = model_dir / "player_ratings_min_states.csv"
    filtered_ratings.to_csv(filtered_ratings_path, index=False)

    predictions = state_df.loc[valid_mask, [
        "gameid",
        "date",
        "period",
        "home_poss",
        "pre_margin_home",
        "poss_remaining",
        "target_future_netrt",
        "final_margin_home",
    ]].copy()
    predictions["pred_future_netrt"] = valid_pred
    predictions["residual"] = predictions["target_future_netrt"] - predictions["pred_future_netrt"]
    predictions_path = model_dir / "validation_predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)

    top_players_path = model_dir / "top_players.txt"
    with top_players_path.open("w") as f:
        f.write(filtered_ratings.head(30).to_string(index=False))
        f.write("\n")

    logging.info("Saved model to %s", model_path)
    logging.info("Saved ratings to %s", ratings_path)
    logging.info("Saved filtered ratings to %s", filtered_ratings_path)
    logging.info("Saved validation predictions to %s", predictions_path)


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    state_df = prepare_state_frame(args.prefix, args.year)
    X, player_matrix, player_ids, feature_names = build_feature_matrix(state_df)
    y = state_df["target_future_netrt"].to_numpy(dtype=np.float32)
    train_mask, valid_mask = make_train_validation_split(state_df, args.validation_frac)

    booster, metrics, valid_pred = train_value_model(
        X=X,
        y=y,
        feature_names=feature_names,
        train_mask=train_mask,
        valid_mask=valid_mask,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    ratings = compute_player_ratings(
        booster=booster,
        X=X,
        player_matrix=player_matrix,
        df=state_df,
        player_ids=player_ids,
        feature_names=feature_names,
        batch_size=args.shap_batch_size,
    )
    save_outputs(
        year=args.year,
        prefix=args.prefix,
        state_df=state_df,
        booster=booster,
        metrics=metrics,
        ratings=ratings,
        valid_mask=valid_mask,
        valid_pred=valid_pred,
    )


if __name__ == "__main__":
    main()
