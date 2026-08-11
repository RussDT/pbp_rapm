"""DARKO-style predictive WNBA box-score panel via per-player per-stat Kalman filters.

Contract:
- Input: ``common.GAME_LOG_PATH`` rows with ``played == 1`` and ``minutes > 0``.
- For each stat in the pinned 14-stat list, track a latent per-minute rate
  ``theta`` per player. Observation: ``obs_rate = stat / minutes`` with
  observation variance ``r / minutes``. Evolution: random walk adding
  ``q_game`` per game played, ``q_day * days_elapsed`` since the player's
  previous game, and ``q_offseason`` once when crossing a season boundary.
  New players initialize at the pre-2026 league-mean per-minute rate with
  variance ``v0``.
- Hyperparameters ``(q_game, q_day, q_offseason, r, v0)`` are tuned per stat
  by ``scipy.optimize.differential_evolution`` (seed 20260810, polish=False)
  on the one-step-ahead minutes-weighted squared error in rate space over
  eligible observations: season in 2023-2025 AND the player has at least one
  prior observation. 2022 rows warm the filter but never enter the objective.
- STANDING FENCE: 2026 rows never enter the objective, the bounds, or the
  league-mean initialization (all computed from pre-2026 rows). Because the
  filter is strictly chronological, 2026 rows cannot influence any pre-2026
  prediction; after tuning, the filter runs through 2026 with frozen
  hyperparameters and 2026 error is reported as untouched holdout evidence.
- Baselines evaluated on the identical eligible sets with identical minutes
  weighting: trailing-10-game, season-to-date (falls back to career-to-date
  at a player's first game of a season), and career-to-date per-minute rates.
- Outputs under ``wnba_test/Results/gameday/kalman_box/``: hyperparams.json,
  predictions.parquet, current_states.csv (with per-40 and pct-hat columns),
  eval_report.json. Generated artifacts stay out of git.

Run: ``.venv/bin/python wnba_test/build_wnba_kalman_box.py [--quick] [--stats pts,ast]``
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

import wnba_gameday_common as common

STATS = [
    "pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
    "oreb", "dreb", "ast", "stl", "blk", "tov", "pf",
]
PARAM_NAMES = ["q_game", "q_day", "q_offseason", "r", "v0"]
DEV_SEASONS = (2023, 2024, 2025)
HOLDOUT_SEASON = 2026
SEED = 20260810
OUT_DIR = common.GAMEDAY_DIR / "kalman_box"


# --------------------------------------------------------------------------
# Panel + vectorized filter layout
# --------------------------------------------------------------------------


def load_panel() -> pd.DataFrame:
    """Played rows with minutes > 0, chronological within player, with gaps."""
    df = pd.read_csv(common.GAME_LOG_PATH)
    df = df[(df["played"] == 1) & (df["minutes"] > 0)].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(
        ["person_id", "game_date", "game_id"], kind="mergesort"
    ).reset_index(drop=True)
    df["player_idx"] = pd.factorize(df["person_id"])[0]
    grp = df.groupby("player_idx", sort=False)
    df["step_k"] = grp.cumcount()
    df["days_gap"] = grp["game_date"].diff().dt.days.fillna(0).astype(float)
    df["season_cross"] = (grp["season"].diff().fillna(0) > 0).astype(float)
    return df


class FilterLayout:
    """Precomputed step-parallel layout: all players' k-th games process together.

    The scalar Kalman recursion is sequential within a player but independent
    across players, so one pass over within-player step index k (~max games by
    any one player) with vectorized numpy ops per step replaces a per-row loop.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.n_rows = len(df)
        self.n_players = int(df["player_idx"].max()) + 1
        self.player_idx = df["player_idx"].to_numpy()
        self.minutes = df["minutes"].to_numpy(dtype=float)
        self.days_gap = df["days_gap"].to_numpy(dtype=float)
        self.season_cross = df["season_cross"].to_numpy(dtype=float)
        step = df["step_k"].to_numpy()
        order = np.argsort(step, kind="stable")
        step_sorted = step[order]
        max_step = int(step_sorted[-1])
        bounds = np.searchsorted(step_sorted, np.arange(max_step + 2))
        self.step_rows = [
            order[bounds[k] : bounds[k + 1]] for k in range(max_step + 1)
        ]


def run_filter(
    layout: FilterLayout,
    obs: np.ndarray,
    params: Sequence[float],
    mean_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One chronological pass; returns (pred_rate, post_rate, post_var) per row.

    ``pred_rate`` is the one-step-ahead prediction BEFORE the update;
    ``post_rate`` / ``post_var`` are the posterior AFTER the update.
    """
    q_game, q_day, q_off, r, v0 = params
    theta = np.empty(layout.n_players)
    var = np.empty(layout.n_players)
    pred = np.empty(layout.n_rows)
    post = np.empty(layout.n_rows)
    post_var_out = np.empty(layout.n_rows)
    for k, rows in enumerate(layout.step_rows):
        players = layout.player_idx[rows]
        if k == 0:
            prior_t = np.full(rows.shape[0], mean_rate)
            prior_v = np.full(rows.shape[0], v0)
        else:
            prior_t = theta[players]
            prior_v = (
                var[players]
                + q_game
                + q_day * layout.days_gap[rows]
                + q_off * layout.season_cross[rows]
            )
        obs_var = r / layout.minutes[rows]
        gain = prior_v / (prior_v + obs_var)
        post_t = prior_t + gain * (obs[rows] - prior_t)
        post_v = (1.0 - gain) * prior_v
        theta[players] = post_t
        var[players] = post_v
        pred[rows] = prior_t
        post[rows] = post_t
        post_var_out[rows] = post_v
    return pred, post, post_var_out


def weighted_mse(
    pred: np.ndarray, obs: np.ndarray, minutes: np.ndarray, mask: np.ndarray
) -> float:
    diff = obs[mask] - pred[mask]
    weights = minutes[mask]
    return float(np.sum(weights * diff * diff) / np.sum(weights))


# --------------------------------------------------------------------------
# Baselines (identical eligible set, identical minutes weighting)
# --------------------------------------------------------------------------


def build_baselines(df: pd.DataFrame, stats: Sequence[str]) -> dict[str, dict[str, np.ndarray]]:
    """Trailing-10 / season-to-date / career-to-date per-minute rate predictions.

    All use strictly-prior games only (shift by one). Season-to-date falls back
    to career-to-date at a player's first game of a season so every eligible
    row (>= 1 prior career observation) has a defined baseline.
    """
    pid = df["player_idx"]
    minutes = df["minutes"].astype(float)
    shifted_min = minutes.groupby(pid).shift()
    trail_min = (
        shifted_min.groupby(pid).rolling(10, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    career_min = minutes.groupby(pid).cumsum() - minutes
    season_min = minutes.groupby([pid, df["season"]]).cumsum() - minutes
    out: dict[str, dict[str, np.ndarray]] = {}
    for stat in stats:
        values = df[stat].astype(float)
        shifted = values.groupby(pid).shift()
        trail_sum = (
            shifted.groupby(pid).rolling(10, min_periods=1).sum().reset_index(level=0, drop=True)
        )
        career_sum = values.groupby(pid).cumsum() - values
        season_sum = values.groupby([pid, df["season"]]).cumsum() - values
        career = (career_sum / career_min.replace(0.0, np.nan)).to_numpy()
        season = (season_sum / season_min.replace(0.0, np.nan)).to_numpy()
        season = np.where(np.isnan(season), career, season)
        out[stat] = {
            "trail10": (trail_sum / trail_min).to_numpy(),
            "season": season,
            "career": career,
        }
    return out


# --------------------------------------------------------------------------
# Hyperparameter bounds (justified from pre-2026 data scale only)
# --------------------------------------------------------------------------


def stat_bounds(
    mean_rate: float, base_var: float, mean_minutes: float
) -> list[tuple[float, float]]:
    """Bounds in the order of PARAM_NAMES, anchored to pre-2026 scale.

    ``base_var`` is the minutes-weighted variance of pre-2026 obs rates around
    the league mean (cross-player signal + game noise), so it upper-bounds
    both the plausible prior spread (v0) and any single drift increment.
    ``r`` has units rate^2 * minutes; a typical game's noise variance is
    ``r / mean_minutes``, so ``r ~ base_var * mean_minutes`` is the natural
    scale and the bounds bracket it by ~2 orders of magnitude each way.
    ``q_day`` is capped so even a ~300-day offseason gap adds at most a few
    multiples of ``base_var``; ``q_offseason`` may add up to ``base_var`` once.
    """
    del mean_rate  # bounds are variance-scaled; mean kept for signature clarity
    r_scale = base_var * mean_minutes
    return [
        (1e-9, 0.25 * base_var),   # q_game
        (0.0, 0.01 * base_var),    # q_day
        (0.0, 1.0 * base_var),     # q_offseason
        (0.01 * r_scale, 30.0 * r_scale),  # r
        (0.005 * base_var, 4.0 * base_var),  # v0
    ]


# --------------------------------------------------------------------------
# Main build
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--quick", action="store_true", help="smoke mode: DE maxiter 5, popsize 6"
    )
    parser.add_argument(
        "--stats", type=str, default=None,
        help="comma-separated subset of stats to tune (default: all 14)",
    )
    args = parser.parse_args()
    stats = STATS if args.stats is None else [
        s.strip() for s in args.stats.split(",") if s.strip()
    ]
    unknown = [s for s in stats if s not in STATS]
    if unknown:
        raise SystemExit(f"unknown stats {unknown}; choose from {STATS}")
    maxiter, popsize = (5, 6) if args.quick else (30, 12)

    df = load_panel()
    layout = FilterLayout(df)
    minutes = layout.minutes
    seasons = df["season"].to_numpy()
    has_prior = (df["step_k"] > 0).to_numpy()
    dev_mask = np.isin(seasons, DEV_SEASONS) & has_prior
    holdout_mask = (seasons == HOLDOUT_SEASON) & has_prior
    pre2026 = seasons < HOLDOUT_SEASON
    mean_minutes_pre = float(minutes[pre2026].mean())
    print(
        f"panel: {len(df)} played rows, {layout.n_players} players, "
        f"dev eligible {int(dev_mask.sum())}, holdout eligible {int(holdout_mask.sum())}"
    )

    baselines = build_baselines(df, stats)
    obs_by_stat = {s: (df[s].astype(float) / df["minutes"]).to_numpy() for s in stats}

    hyperparams: dict[str, dict] = {}
    eval_report: dict[str, dict] = {}
    pred_frames: list[pd.DataFrame] = []
    state_cols: dict[str, np.ndarray] = {}
    de_total = 0.0

    for stat in stats:
        obs = obs_by_stat[stat]
        # Pre-2026-only initialization and scale (2026 fence).
        mean_rate = float(df.loc[pre2026, stat].sum() / df.loc[pre2026, "minutes"].sum())
        weights_pre = minutes[pre2026]
        base_var = float(
            np.sum(weights_pre * (obs[pre2026] - mean_rate) ** 2) / np.sum(weights_pre)
        )
        bounds = stat_bounds(mean_rate, base_var, mean_minutes_pre)

        def objective(params: np.ndarray) -> float:
            pred, _, _ = run_filter(layout, obs, params, mean_rate)
            return weighted_mse(pred, obs, minutes, dev_mask)

        start = time.time()
        result = differential_evolution(
            objective,
            bounds=bounds,
            seed=SEED,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-7,
            polish=False,
        )
        elapsed = time.time() - start
        de_total += elapsed
        params = [float(v) for v in result.x]
        pred, post, post_var = run_filter(layout, obs, params, mean_rate)

        stat_eval: dict[str, dict] = {}
        for window, mask in (("dev", dev_mask), ("holdout_2026", holdout_mask)):
            kalman_mse = weighted_mse(pred, obs, minutes, mask)
            entry: dict[str, float] = {
                "n_obs": int(mask.sum()),
                "kalman_wmse": kalman_mse,
            }
            for name, base_pred in baselines[stat].items():
                assert not np.isnan(base_pred[mask]).any(), (
                    f"{stat} {name} baseline has NaN on eligible rows"
                )
                base_mse = weighted_mse(base_pred, obs, minutes, mask)
                entry[f"{name}_wmse"] = base_mse
                entry[f"pct_impr_vs_{name}"] = 100.0 * (base_mse - kalman_mse) / base_mse
            stat_eval[window] = entry
        eval_report[stat] = stat_eval
        hyperparams[stat] = {
            "params": dict(zip(PARAM_NAMES, params)),
            "bounds": dict(zip(PARAM_NAMES, [list(b) for b in bounds])),
            "league_mean_rate_pre2026": mean_rate,
            "base_var_pre2026": base_var,
            "dev_objective": float(result.fun),
            "de": {
                "success": bool(result.success),
                "message": str(result.message),
                "nit": int(result.nit),
                "nfev": int(result.nfev),
                "seconds": round(elapsed, 2),
                "maxiter": maxiter,
                "popsize": popsize,
            },
        }
        pred_frames.append(
            pd.DataFrame(
                {
                    "person_id": df["person_id"].to_numpy(),
                    "game_id": df["game_id"].to_numpy(),
                    "game_date": df["game_date"].to_numpy(),
                    "season": seasons,
                    "stat": stat,
                    "minutes": minutes,
                    "obs_rate": obs,
                    "pred_rate": pred,
                    "post_rate": post,
                    "post_var": post_var,
                }
            )
        )
        state_cols[f"{stat}_rate"] = post
        state_cols[f"{stat}_var"] = post_var
        dev_t10 = stat_eval["dev"]["pct_impr_vs_trail10"]
        hold_t10 = stat_eval["holdout_2026"]["pct_impr_vs_trail10"]
        print(
            f"{stat:>5}: DE {elapsed:5.1f}s fun={result.fun:.6g} | "
            f"dev vs t10 {dev_t10:+.2f}% | 2026 vs t10 {hold_t10:+.2f}%"
        )

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "hyperparams.json").write_text(json.dumps(hyperparams, indent=2))

    predictions = pd.concat(pred_frames, ignore_index=True)
    predictions.to_parquet(OUT_DIR / "predictions.parquet", index=False)

    last_idx = df.groupby("player_idx")["step_k"].idxmax()
    states = df.loc[
        last_idx, ["person_id", "player_name", "game_date", "season"]
    ].rename(columns={"game_date": "last_game_date", "season": "last_season"})
    states["n_games"] = df.groupby("player_idx")["step_k"].max().to_numpy() + 1
    states["career_minutes"] = df.groupby("player_idx")["minutes"].sum().to_numpy()
    for stat in stats:
        states[f"{stat}_rate"] = state_cols[f"{stat}_rate"][last_idx.to_numpy()]
        states[f"{stat}_var"] = state_cols[f"{stat}_var"][last_idx.to_numpy()]
        states[f"{stat}_per40"] = states[f"{stat}_rate"] * 40.0
    for pct, num, den in (
        ("fg_pct_hat", "fgm", "fga"),
        ("fg3_pct_hat", "fg3m", "fg3a"),
        ("ft_pct_hat", "ftm", "fta"),
    ):
        if num in stats and den in stats:
            states[pct] = states[f"{num}_rate"] / states[f"{den}_rate"]
    states = states.sort_values("person_id").reset_index(drop=True)
    states.to_csv(OUT_DIR / "current_states.csv", index=False)

    summary = {
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "quick": bool(args.quick),
        "stats": stats,
        "n_panel_rows": int(len(df)),
        "n_players": int(layout.n_players),
        "n_dev_eligible": int(dev_mask.sum()),
        "n_holdout_eligible": int(holdout_mask.sum()),
        "de_total_seconds": round(de_total, 1),
        "objective": (
            "one-step-ahead minutes-weighted squared error in per-minute rate "
            "space; eligible = season in 2023-2025 with >= 1 prior observation; "
            "2022 warms the filter only; 2026 is frozen-parameter holdout"
        ),
        "season_baseline_note": (
            "season-to-date baseline falls back to career-to-date at a "
            "player's first game of a season"
        ),
        "n_stats_beating_trail10_dev": sum(
            1 for s in stats if eval_report[s]["dev"]["pct_impr_vs_trail10"] > 0
        ),
    }
    (OUT_DIR / "eval_report.json").write_text(
        json.dumps({"summary": summary, "per_stat": eval_report}, indent=2)
    )

    if "pts" in stats:
        current = states[
            (states["last_season"] == int(states["last_season"].max()))
            & (states["career_minutes"] >= 300)
        ]
        top = current.nlargest(10, "pts_per40")[
            ["player_name", "pts_per40", "fga_per40" if "fga" in stats else "pts_rate", "n_games"]
        ]
        print("\ntop-10 per-40 pts (current-season players, >=300 career minutes):")
        print(top.to_string(index=False))

    beat = summary["n_stats_beating_trail10_dev"]
    print(
        f"\nwrote {OUT_DIR}/(hyperparams.json, predictions.parquet, "
        f"current_states.csv, eval_report.json); DE total {de_total:.1f}s; "
        f"kalman beats trailing-10 on dev for {beat}/{len(stats)} stats"
    )


if __name__ == "__main__":
    main()
