"""Sigma/HCA/pace calibration and retrospective 2026 pregame replay.

Contract
--------
Two subcommands over the game-day panel (``common.GAMES_TABLE_PATH`` /
``common.GAME_LOG_PATH``), the pregame minutes engine
(``project_wnba_minutes.project_team_minutes``), and the frozen per-season
history boards (predict season S with season S-1's completed board):

``calibrate`` (dev seasons 2023-2025 ONLY): for every final game, build the
pregame forecast — prior-season board + pregame projected 200-minute
allocation — with the pinned algebra ``team_per100 = sum(net_rapm_i *
proj_minutes_i) / 40`` and ``margin_pred = (home_per100 - away_per100) *
pace / 100 + HCA`` where pace is the PRIOR season's per-team possessions per
game counted from ``Processed/RAPM{yy}.csv`` (regular-season file; sanity
70-90). Fit ``HCA = mean(actual home margin - rating margin)`` pooled over
dev seasons; ``sigma = std`` of residuals; verify 90% normal-interval
empirical coverage and report a PIT summary; when coverage is off, widen
sigma with a dev-season empirical-quantile (conformal-style) correction and
record both raw and corrected sigma. Writes
``Results/gameday/forecast_params.json`` and
``Results/gameday/calibration/{calibration_report.json, game_level.csv}``.

``replay`` (2026, frozen): apply ``forecast_params.json`` as-is + the 2025
history board + pregame projected minutes to every completed 2026 game.
Score margin RMSE/MAE, win-prob log-loss and Brier (``p = Phi(margin /
sigma)``), 90% coverage, PIT KS. Baselines scored identically: HCA-only
margin and the dev-season home base rate. A secondary conditional arm uses
ACTUAL minutes on the same board (``conditional_actual_minutes``) to isolate
minutes-projection error. Writes
``Results/gameday/replay_2026/{report.json, game_level.csv}``. Nothing is
refit on 2026.

Board sign conventions are verified empirically per board (off/def/net
column identities), never assumed. Players absent from a board score 0
(league average); rated-minutes coverage share is recorded per team-game.

Fences: market-blind (no odds inputs anywhere in this script); 2026 is the
confirmation season — scored only, with parameters frozen on 2023-2025; no
proprietary ratings; generated JSON/CSV artifacts stay out of git.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import kstest, norm

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import wnba_gameday_common as common
from project_wnba_minutes import project_team_minutes

# --------------------------------------------------------------------------
# Frozen configuration
# --------------------------------------------------------------------------

DEV_SEASONS = (2023, 2024, 2025)
REPLAY_SEASON = 2026

# Season S is forecast with season S-1's completed history board.
HISTORY_BOARD_PATHS = {
    2023: (
        "clean_decomp_22_all_a750_1000_history_warmup_pure_v1/"
        "weighted_factors_alt3_efg_value_22_all_a750_1000_history_warmup_pure_v1.csv"
    ),
    2024: (
        "clean_decomp_23_all_a5000_40000_prior1.5_box_trueprior_history_v1/"
        "weighted_factors_alt3_efg_value_23_all_a5000_40000_prior1.5_box_trueprior_history_v1.csv"
    ),
    2025: (
        "clean_decomp_24_all_a5000_40000_prior1.5_box_trueprior_history_v1/"
        "weighted_factors_alt3_efg_value_24_all_a5000_40000_prior1.5_box_trueprior_history_v1.csv"
    ),
    2026: (
        "clean_decomp_25_all_a5000_40000_prior1.5_box_trueprior_history_v1/"
        "weighted_factors_alt3_efg_value_25_all_a5000_40000_prior1.5_box_trueprior_history_v1.csv"
    ),
}

RESULTS_DIR = common.WNBA_TEST_DIR / "Results"
PROCESSED_DIR = common.WNBA_TEST_DIR / "Processed"
PARAMS_PATH = common.GAMEDAY_DIR / "forecast_params.json"
CALIBRATION_DIR = common.GAMEDAY_DIR / "calibration"
REPLAY_DIR = common.GAMEDAY_DIR / "replay_2026"

TEAM_MINUTES = 200.0
SLOT_MINUTES = TEAM_MINUTES / 5.0  # 40: expected minutes per lineup slot
Z90 = float(norm.ppf(0.95))  # 1.6449: two-sided 90% normal interval
PACE_SANITY = (70.0, 90.0)
HCA_SANITY = (0.5, 4.5)
SIGMA_SANITY = (10.0, 16.0)
COVERAGE_TOL = 0.01  # apply the conformal widening when |cov90 - 0.90| exceeds this
# Board columns are rounded to 3 decimals, so the true identity gap is <= ~0.0015.
BOARD_IDENTITY_TOL = 0.005
PROB_EPS = 1e-12

# Known conditional-on-observed-lineup benchmark: current-season board +
# actual lineups. Conditioning differs from this script's pregame arms.
CONDITIONAL_LINEUP_BENCHMARK_RMSE = 13.118


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def load_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    games = pd.read_csv(common.GAMES_TABLE_PATH, dtype={"game_id": str})
    logs = pd.read_csv(common.GAME_LOG_PATH, dtype={"game_id": str})
    logs["person_id"] = logs["person_id"].astype(int)
    logs["minutes"] = logs["minutes"].astype(float)
    return games, logs


def load_history_board(season: int) -> tuple[pd.Series, dict[str, Any]]:
    """Load the frozen board used to forecast ``season`` and verify its signs.

    The sign convention is checked empirically on the loaded frame: whichever
    of ``net_rapm = off + def`` (defense positive-good) or
    ``net_rapm = off - def`` (defense on a positive-bad allowed scale) holds
    as a column identity is recorded; neither holding is a hard failure. In
    both conventions ``net_rapm`` is the positive-good per-100 total, which
    is what the margin model consumes.
    """
    path = RESULTS_DIR / HISTORY_BOARD_PATHS[season]
    board = pd.read_csv(path)
    off = board["off"].to_numpy(dtype=float)
    dfn = board["def"].to_numpy(dtype=float)
    net = board["net_rapm"].to_numpy(dtype=float)
    add_gap = float(np.max(np.abs(off + dfn - net)))
    sub_gap = float(np.max(np.abs(off - dfn - net)))
    if add_gap <= BOARD_IDENTITY_TOL and add_gap < sub_gap:
        convention = "net_rapm = off + def (defense positive-good)"
    elif sub_gap <= BOARD_IDENTITY_TOL and sub_gap < add_gap:
        convention = "net_rapm = off - def (defense positive-bad)"
    else:
        raise AssertionError(
            f"board {path.name}: neither sign identity holds "
            f"(|off+def-net| max {add_gap:.4f}, |off-def-net| max {sub_gap:.4f})"
        )
    board["player_id"] = board["player_id"].astype(int)
    dupes = int(board["player_id"].duplicated().sum())
    if dupes:
        raise AssertionError(f"board {path.name}: {dupes} duplicate player_id rows")
    net_by_pid = board.set_index("player_id")["net_rapm"].astype(float)
    verification = {
        "forecast_season": season,
        "board_path": str(path.relative_to(common.WNBA_TEST_DIR)),
        "n_players": int(len(board)),
        "max_abs_gap_off_plus_def_minus_net": round(add_gap, 6),
        "max_abs_gap_off_minus_def_minus_net": round(sub_gap, 6),
        "sign_convention": convention,
    }
    return net_by_pid, verification


def compute_pace(season: int) -> float:
    """Per-team possessions per game from the season's Processed RAPM rows.

    Possession rows per ``game_id`` are both teams' possessions combined, so
    per-team pace is half the per-game mean. Regular-season file only.
    """
    path = PROCESSED_DIR / f"RAPM{season % 100}.csv"
    per_game = pd.read_csv(path, usecols=["game_id"]).groupby("game_id").size()
    pace = float(per_game.mean()) / 2.0
    if not (PACE_SANITY[0] <= pace <= PACE_SANITY[1]):
        raise AssertionError(f"pace {pace:.2f} for season {season} outside {PACE_SANITY}")
    return pace


# --------------------------------------------------------------------------
# Forecast construction
# --------------------------------------------------------------------------


def _rated_per100(minutes: pd.Series, person_ids: pd.Series, net_by_pid: pd.Series) -> tuple[float, float]:
    """(per-100 rating, rated-minutes coverage share) for one team-game.

    ``per100 = sum(net_rapm_i * minutes_i) / (total_minutes / 5)``; players
    absent from the board score 0 (league average). With the engine's exact
    200-minute allocation the divisor is the pinned 40.
    """
    total = float(minutes.sum())
    if total <= 0.0:
        raise AssertionError("team-game with nonpositive total minutes")
    net = person_ids.map(net_by_pid)
    per100 = float((net.fillna(0.0) * minutes).sum()) / (total / 5.0)
    rated_share = float(minutes[net.notna()].sum()) / total
    return per100, rated_share


def build_forecast_rows(
    games: pd.DataFrame,
    logs: pd.DataFrame,
    seasons: tuple[int, ...],
    boards: dict[int, pd.Series],
    pace_by_season: dict[int, float],
    with_actual_arm: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """One pregame forecast row per final game in ``seasons``.

    Season S uses ``boards[S]`` (the S-1 history board loaded upstream) and
    ``pace_by_season[S - 1]`` (prior-season pace, pregame-honest). Games the
    minutes engine cannot project pregame (brand-new franchise with zero
    panel history) are skipped and reported, never guessed.
    """
    finals = games[(games["status"] == 3) & games["season"].isin(list(seasons))]
    finals = finals.sort_values(["game_date", "game_id"], kind="mergesort")
    played = logs[logs["played"] == 1]
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for game in finals.itertuples(index=False):
        season = int(game.season)
        net_by_pid = boards[season]
        pace = pace_by_season[season - 1]
        try:
            home_proj = project_team_minutes(logs, game.home_tricode, game.game_date)
            away_proj = project_team_minutes(logs, game.away_tricode, game.game_date)
        except ValueError as exc:
            skipped.append(
                {
                    "game_id": game.game_id,
                    "season": season,
                    "game_date": game.game_date,
                    "home_tricode": game.home_tricode,
                    "away_tricode": game.away_tricode,
                    "reason": str(exc),
                }
            )
            continue
        home_per100, home_share = _rated_per100(
            home_proj["proj_minutes"], home_proj["person_id"], net_by_pid
        )
        away_per100, away_share = _rated_per100(
            away_proj["proj_minutes"], away_proj["person_id"], net_by_pid
        )
        actual_margin = float(game.home_score) - float(game.away_score)
        row: dict[str, Any] = {
            "game_id": game.game_id,
            "season": season,
            "season_type": game.season_type,
            "game_date": game.game_date,
            "home_tricode": game.home_tricode,
            "away_tricode": game.away_tricode,
            "home_score": int(game.home_score),
            "away_score": int(game.away_score),
            "actual_margin": actual_margin,
            "pace": round(pace, 4),
            "home_per100": round(home_per100, 4),
            "away_per100": round(away_per100, 4),
            "rating_margin_no_hca": (home_per100 - away_per100) * pace / 100.0,
            "home_rated_min_share": round(home_share, 4),
            "away_rated_min_share": round(away_share, 4),
        }
        if with_actual_arm:
            for side, tricode in (("home", game.home_tricode), ("away", game.away_tricode)):
                actual = played[
                    (played["game_id"] == game.game_id)
                    & (played["team_tricode"] == tricode)
                ]
                total = float(actual["minutes"].sum())
                if not (150.0 <= total <= 300.0):
                    raise AssertionError(
                        f"implausible actual team minutes {total:.1f} for "
                        f"{tricode} in {game.game_id}"
                    )
                per100, share = _rated_per100(
                    actual["minutes"], actual["person_id"], net_by_pid
                )
                row[f"{side}_per100_actual_min"] = round(per100, 4)
                row[f"{side}_rated_min_share_actual"] = round(share, 4)
            row["rating_margin_no_hca_actual_min"] = (
                (row["home_per100_actual_min"] - row["away_per100_actual_min"]) * pace / 100.0
            )
        rows.append(row)
    return pd.DataFrame(rows), skipped


# --------------------------------------------------------------------------
# Scoring helpers
# --------------------------------------------------------------------------


def _margin_metrics(residuals: np.ndarray) -> dict[str, float]:
    return {
        "rmse": round(float(np.sqrt(np.mean(residuals**2))), 4),
        "mae": round(float(np.mean(np.abs(residuals))), 4),
    }


def _prob_metrics(p_home: np.ndarray, home_win: np.ndarray) -> dict[str, float]:
    p = np.clip(p_home, PROB_EPS, 1.0 - PROB_EPS)
    log_loss = -np.mean(home_win * np.log(p) + (1.0 - home_win) * np.log(1.0 - p))
    brier = np.mean((p - home_win) ** 2)
    return {"log_loss": round(float(log_loss), 5), "brier": round(float(brier), 5)}


def _pit_summary(residuals: np.ndarray, sigma: float) -> dict[str, float]:
    pit = norm.cdf(residuals / sigma)
    ks = kstest(pit, "uniform")
    return {
        "pit_mean": round(float(pit.mean()), 4),
        "pit_std": round(float(pit.std(ddof=1)), 4),
        "ks_statistic": round(float(ks.statistic), 4),
        "ks_pvalue": round(float(ks.pvalue), 4),
        "coverage90": round(float(np.mean(np.abs(residuals) <= Z90 * sigma)), 4),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


# --------------------------------------------------------------------------
# calibrate
# --------------------------------------------------------------------------


def run_calibrate() -> dict[str, Any]:
    games, logs = load_panel()
    boards: dict[int, pd.Series] = {}
    verifications = []
    for season in DEV_SEASONS:
        net_by_pid, verification = load_history_board(season)
        boards[season] = net_by_pid
        verifications.append(verification)
    pace_seasons = sorted({s - 1 for s in DEV_SEASONS} | {max(DEV_SEASONS)})
    pace_by_season = {s: compute_pace(s) for s in pace_seasons}

    frame, skipped = build_forecast_rows(games, logs, DEV_SEASONS, boards, pace_by_season)
    if frame.empty:
        raise AssertionError("no dev-season games produced forecasts")

    rating = frame["rating_margin_no_hca"].to_numpy(dtype=float)
    actual = frame["actual_margin"].to_numpy(dtype=float)
    hca = float(np.mean(actual - rating))
    frame["margin_pred"] = rating + hca
    residuals = actual - rating - hca
    frame["residual"] = residuals
    sigma_raw = float(np.std(residuals, ddof=1))

    coverage90_raw = float(np.mean(np.abs(residuals) <= Z90 * sigma_raw))
    q90_emp = float(np.quantile(np.abs(residuals) / sigma_raw, 0.90))
    sigma_conformal = sigma_raw * q90_emp / Z90
    if abs(coverage90_raw - 0.90) > COVERAGE_TOL:
        sigma_margin = sigma_conformal
        sigma_method = "dev_empirical_q90_conformal"
    else:
        sigma_margin = sigma_raw
        sigma_method = "normal_std"

    if not (HCA_SANITY[0] <= hca <= HCA_SANITY[1]):
        raise AssertionError(f"fitted HCA {hca:.3f} outside sanity range {HCA_SANITY}")
    if not (SIGMA_SANITY[0] <= sigma_margin <= SIGMA_SANITY[1]):
        raise AssertionError(f"sigma {sigma_margin:.3f} outside sanity range {SIGMA_SANITY}")

    rating_rmse = float(np.sqrt(np.mean(residuals**2)))
    hca_only_rmse = float(np.sqrt(np.mean((actual - hca) ** 2)))
    if rating_rmse >= hca_only_rmse:
        raise AssertionError(
            f"rating arm dev RMSE {rating_rmse:.3f} does not beat HCA-only "
            f"{hca_only_rmse:.3f}; check join/sign/pace before trusting output"
        )

    per_season = {}
    for season in DEV_SEASONS:
        sub = frame[frame["season"] == season]
        sub_actual = sub["actual_margin"].to_numpy(dtype=float)
        sub_rating = sub["rating_margin_no_hca"].to_numpy(dtype=float)
        sub_resid = sub["residual"].to_numpy(dtype=float)
        per_season[str(season)] = {
            "n_games": int(len(sub)),
            "hca_season": round(float(np.mean(sub_actual - sub_rating)), 4),
            "resid_std_season": round(float(np.std(sub_resid, ddof=1)), 4),
            "coverage90_pooled_sigma": round(
                float(np.mean(np.abs(sub_resid) <= Z90 * sigma_margin)), 4
            ),
            "mean_rated_min_share": round(
                float(
                    pd.concat(
                        [sub["home_rated_min_share"], sub["away_rated_min_share"]]
                    ).mean()
                ),
                4,
            ),
        }

    home_base_rate = float(np.mean(actual > 0.0))
    fitted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    params = {
        "hca_pts": round(hca, 4),
        "sigma_margin": round(sigma_margin, 4),
        "sigma_method": sigma_method,
        "sigma_margin_raw": round(sigma_raw, 4),
        "sigma_margin_conformal": round(sigma_conformal, 4),
        "pace_by_season": {str(s): round(p, 4) for s, p in pace_by_season.items()},
        "fitted_on_seasons": list(DEV_SEASONS),
        "fitted_at_utc": fitted_at,
        "home_base_rate_dev": round(home_base_rate, 4),
        "margin_model": (
            "margin = (home_per100 - away_per100) * prior_season_pace / 100 + hca_pts; "
            "team_per100 = sum(net_rapm_i * proj_minutes_i) / 40 over the projected "
            "200-minute allocation"
        ),
    }
    _write_json(PARAMS_PATH, params)

    report = {
        "generated_at_utc": fitted_at,
        "fitted_on_seasons": list(DEV_SEASONS),
        "n_games": int(len(frame)),
        "n_skipped": len(skipped),
        "skipped_games": skipped,
        "board_verification": verifications,
        "pace_by_season": params["pace_by_season"],
        "hca_pts": params["hca_pts"],
        "sigma_margin_raw": params["sigma_margin_raw"],
        "sigma_margin_conformal": params["sigma_margin_conformal"],
        "sigma_margin_selected": params["sigma_margin"],
        "sigma_method": sigma_method,
        "conformal_q90_abs_standardized_residual": round(q90_emp, 4),
        "coverage90_raw_sigma": round(coverage90_raw, 4),
        "coverage90_selected_sigma": round(
            float(np.mean(np.abs(residuals) <= Z90 * sigma_margin)), 4
        ),
        "pit_raw_sigma": _pit_summary(residuals, sigma_raw),
        "pit_selected_sigma": _pit_summary(residuals, sigma_margin),
        "dev_margin_rmse_rating_arm": round(rating_rmse, 4),
        "dev_margin_rmse_hca_only": round(hca_only_rmse, 4),
        "home_base_rate_dev": params["home_base_rate_dev"],
        "mean_rated_min_share": round(
            float(
                pd.concat(
                    [frame["home_rated_min_share"], frame["away_rated_min_share"]]
                ).mean()
            ),
            4,
        ),
        "per_season": per_season,
    }
    _write_json(CALIBRATION_DIR / "calibration_report.json", report)
    frame.to_csv(CALIBRATION_DIR / "game_level.csv", index=False)
    return report


# --------------------------------------------------------------------------
# replay (2026, frozen)
# --------------------------------------------------------------------------


def run_replay() -> dict[str, Any]:
    if not PARAMS_PATH.exists():
        raise FileNotFoundError(f"{PARAMS_PATH} missing; run the calibrate subcommand first")
    params = json.loads(PARAMS_PATH.read_text())
    hca = float(params["hca_pts"])
    sigma = float(params["sigma_margin"])
    prior_season = REPLAY_SEASON - 1
    pace_key = str(prior_season)
    if pace_key not in params["pace_by_season"]:
        raise AssertionError(f"forecast_params.json lacks pace for season {pace_key}")
    pace = float(params["pace_by_season"][pace_key])

    games, logs = load_panel()
    net_by_pid, verification = load_history_board(REPLAY_SEASON)
    frame, skipped = build_forecast_rows(
        games,
        logs,
        (REPLAY_SEASON,),
        {REPLAY_SEASON: net_by_pid},
        {prior_season: pace},
        with_actual_arm=True,
    )
    if frame.empty:
        raise AssertionError("no completed 2026 games produced forecasts")

    actual = frame["actual_margin"].to_numpy(dtype=float)
    home_win = (actual > 0.0).astype(float)

    # Primary pregame arm.
    frame["margin_pred"] = frame["rating_margin_no_hca"] + hca
    resid = actual - frame["margin_pred"].to_numpy(dtype=float)
    frame["residual"] = resid
    frame["p_home"] = norm.cdf(frame["margin_pred"].to_numpy(dtype=float) / sigma)
    primary = _margin_metrics(resid)
    primary.update(_prob_metrics(frame["p_home"].to_numpy(dtype=float), home_win))
    primary.update(_pit_summary(resid, sigma))

    # Conditional arm: actual minutes, same frozen board and params.
    frame["margin_pred_actual_min"] = frame["rating_margin_no_hca_actual_min"] + hca
    resid_cond = actual - frame["margin_pred_actual_min"].to_numpy(dtype=float)
    frame["residual_actual_min"] = resid_cond
    frame["p_home_actual_min"] = norm.cdf(
        frame["margin_pred_actual_min"].to_numpy(dtype=float) / sigma
    )
    conditional = _margin_metrics(resid_cond)
    conditional.update(
        _prob_metrics(frame["p_home_actual_min"].to_numpy(dtype=float), home_win)
    )
    conditional.update(_pit_summary(resid_cond, sigma))

    # Baseline 1: HCA-only margin, same sigma.
    resid_hca = actual - hca
    p_hca = np.full(len(frame), float(norm.cdf(hca / sigma)))
    hca_only = _margin_metrics(resid_hca)
    hca_only.update(_prob_metrics(p_hca, home_win))
    hca_only.update(_pit_summary(resid_hca, sigma))

    # Baseline 2: dev-season home base rate (probability-only).
    base_rate = float(params["home_base_rate_dev"])
    base = _prob_metrics(np.full(len(frame), base_rate), home_win)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "season": REPLAY_SEASON,
        "frozen_params": {
            "hca_pts": hca,
            "sigma_margin": sigma,
            "sigma_method": params["sigma_method"],
            "pace": pace,
            "fitted_on_seasons": params["fitted_on_seasons"],
            "fitted_at_utc": params["fitted_at_utc"],
        },
        "board_verification": verification,
        "n_games": int(len(frame)),
        "n_skipped": len(skipped),
        "skipped_games": skipped,
        "home_win_rate_2026": round(float(home_win.mean()), 4),
        "mean_rated_min_share_pregame": round(
            float(
                pd.concat(
                    [frame["home_rated_min_share"], frame["away_rated_min_share"]]
                ).mean()
            ),
            4,
        ),
        "mean_rated_min_share_actual": round(
            float(
                pd.concat(
                    [
                        frame["home_rated_min_share_actual"],
                        frame["away_rated_min_share_actual"],
                    ]
                ).mean()
            ),
            4,
        ),
        "arms": {
            "pregame_rating": primary,
            "conditional_actual_minutes": conditional,
            "baseline_hca_only": hca_only,
            "baseline_home_base_rate_dev": dict(base, base_rate=round(base_rate, 4)),
        },
        "vs_conditional_lineup_benchmark": {
            "replay_pregame_rmse": primary["rmse"],
            "replay_conditional_actual_minutes_rmse": conditional["rmse"],
            "benchmark_rmse": CONDITIONAL_LINEUP_BENCHMARK_RMSE,
            "note": (
                "benchmark 13.118 conditions on the CURRENT-season board and actual "
                "lineups (postgame information on both axes); the replay arms use the "
                "frozen prior-season board with pregame projected or actual minutes — "
                "conditioning differs, so this is context, not a like-for-like race"
            ),
        },
        "fence_note": "2026 scored only; hca/sigma/pace frozen from forecast_params.json",
    }
    _write_json(REPLAY_DIR / "report.json", report)
    frame.to_csv(REPLAY_DIR / "game_level.csv", index=False)
    return report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("calibrate", help="fit HCA/sigma/pace on dev seasons 2023-2025")
    sub.add_parser("replay", help="score 2026 pregame replay with frozen params")
    args = parser.parse_args()
    if args.command == "calibrate":
        report = run_calibrate()
        print(
            f"calibrate: n={report['n_games']} skipped={report['n_skipped']} "
            f"hca={report['hca_pts']} sigma_raw={report['sigma_margin_raw']} "
            f"sigma_selected={report['sigma_margin_selected']} ({report['sigma_method']}) "
            f"cov90_raw={report['coverage90_raw_sigma']} "
            f"cov90_sel={report['coverage90_selected_sigma']} "
            f"rmse_rating={report['dev_margin_rmse_rating_arm']} "
            f"rmse_hca_only={report['dev_margin_rmse_hca_only']}"
        )
    else:
        report = run_replay()
        arms = report["arms"]
        print(
            f"replay 2026: n={report['n_games']} skipped={report['n_skipped']} "
            f"rmse={arms['pregame_rating']['rmse']} mae={arms['pregame_rating']['mae']} "
            f"log_loss={arms['pregame_rating']['log_loss']} "
            f"brier={arms['pregame_rating']['brier']} "
            f"cov90={arms['pregame_rating']['coverage90']} "
            f"| cond_actual_min rmse={arms['conditional_actual_minutes']['rmse']} "
            f"| hca_only rmse={arms['baseline_hca_only']['rmse']} "
            f"log_loss={arms['baseline_hca_only']['log_loss']} "
            f"| base_rate log_loss={arms['baseline_home_base_rate_dev']['log_loss']} "
            f"| benchmark(cond. lineups)={CONDITIONAL_LINEUP_BENCHMARK_RMSE}"
        )


if __name__ == "__main__":
    main()
