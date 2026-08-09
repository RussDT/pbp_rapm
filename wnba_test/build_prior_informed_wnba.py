#!/usr/bin/env python3
"""Build and walk-forward validate prior-informed WNBA RAPM/decomposition ratings.

The model is a dynamic, prior-centered RAPM:

    beta_y ~ N(persistence * beta_{y-1}, alpha^-1)

The current season's possession likelihood updates that prior. Hyperparameters are
selected only on season-forward 2023-2025 game-margin results. The 2026 season is
evaluated once after selection, then the frozen specification is used to build the
current 2026 clean decomposition.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wnba_test.benchmark_wnba_metrics import (
    load_holdout,
    metric_summary,
    resolve_offense_side,
    static_predictions,
)
from wnba_test.clean_decomp_wnba import (
    DISPLAY_PREFIXES,
    RESULTS_DIR,
    _load_clean_total_df,
    _load_name_map,
    _prepare_direct_ridge_problem,
    _solve_direct_ridge_problem,
    build_clean_weighted_factors,
)


PROCESSED_DIR = ROOT / "wnba_test" / "Processed"
RAW_DIR = ROOT / "wnba_test" / "GabrielNormalizedClean"
DEFAULT_OUTPUT = RESULTS_DIR / "prior_informed"
DEFAULT_CLEAN_PRIOR = (
    RESULTS_DIR
    / "clean_decomp_22_25_all_a1750_4000"
    / "weighted_factors_alt3_efg_value_22_25_all_a1750_4000.csv"
)
EXTERNAL_RATINGS_DIR = ROOT / "nba_pipeline" / "external" / "wnba_darko_daily"
PBP_STATS_2026 = ROOT.parent / "wnba" / "data" / "2026_pbp.csv"


def _normalize_ids(series: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .astype("Int64")
        .astype(str)
    )


def _age_state(
    state: pd.DataFrame | None,
    target_year: int,
    persistence: float,
) -> pd.DataFrame | None:
    if state is None or state.empty:
        return None
    prior = state.copy()
    gap = target_year - pd.to_numeric(prior["rating_year"], errors="raise").astype(int)
    if (gap <= 0).any():
        raise ValueError("Prior state must predate the target season")
    decay = np.power(float(persistence), gap.to_numpy(dtype=float))
    prior["off"] = pd.to_numeric(prior["off"], errors="coerce") * decay
    prior["def"] = pd.to_numeric(prior["def"], errors="coerce") * decay
    return prior


def _update_state(
    state: pd.DataFrame | None,
    current: pd.DataFrame,
    rating_year: int,
) -> pd.DataFrame:
    columns = ["player_id", "player_name", "off", "def"]
    update = current[columns].copy()
    update["player_id"] = _normalize_ids(update["player_id"])
    update["rating_year"] = int(rating_year)
    if state is None or state.empty:
        return update
    old = state.copy()
    old["player_id"] = _normalize_ids(old["player_id"])
    old = old.loc[~old["player_id"].isin(update["player_id"])]
    return pd.concat([old, update], ignore_index=True)


def _prepare_holdout(year: int) -> pd.DataFrame:
    holdout = load_holdout(PROCESSED_DIR / f"RAPM{year % 100:02d}.csv")
    holdout["offense_side"] = resolve_offense_side(
        holdout,
        RAW_DIR / f"WNBA{year % 100:02d}.csv",
    )
    return holdout


def _evaluate(
    rating: pd.DataFrame,
    holdout: pd.DataFrame,
    intercept: float,
) -> dict[str, Any]:
    frame = rating[["player_id", "off", "def"]].copy()
    frame["player_id"] = _normalize_ids(frame["player_id"])
    offense = dict(zip(frame["player_id"], pd.to_numeric(frame["off"], errors="coerce")))
    # Solver defense is lower-is-better; benchmark predictions subtract a
    # positive-is-good public defensive rating.
    defense = dict(zip(frame["player_id"], -pd.to_numeric(frame["def"], errors="coerce")))
    prediction, rated_slots, complete_rows = static_predictions(
        holdout,
        offense,
        defense,
        intercept,
    )
    scored = holdout.copy()
    scored["prediction"] = prediction
    return metric_summary(scored, "prediction", rated_slots, complete_rows)


def _solve_sequence(
    problems: dict[int, dict[str, object]],
    years: list[int],
    name_map: dict[str, str],
    off_alpha: float,
    def_alpha: float,
    persistence: float,
) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    state: pd.DataFrame | None = None
    annual: dict[int, pd.DataFrame] = {}
    states: dict[int, pd.DataFrame] = {}
    for year in years:
        prior = _age_state(state, year, persistence)
        result = _solve_direct_ridge_problem(
            problems[year],
            name_map,
            off_alpha=off_alpha,
            def_alpha=def_alpha,
            prior_results=prior,
        )
        annual[year] = result
        state = _update_state(state, result, year)
        states[year] = state.copy()
    if state is None:
        raise AssertionError("No annual ratings were solved")
    return annual, states


def _clean_prior_frames(path: Path) -> dict[str, pd.DataFrame]:
    """Convert the frozen clean display file back to raw per-component priors."""
    frame = pd.read_csv(path)
    frame["player_id"] = _normalize_ids(frame["player_id"])
    priors: dict[str, pd.DataFrame] = {}
    for prefix in DISPLAY_PREFIXES:
        if prefix == "SECOND_CHANCE_CLEAN":
            off_col, def_col = "oSC", "dSC"
        elif prefix == "ALT_FC_COMPLETION":
            off_col, def_col = "oALT_TOV_VALUE", "dALT_TOV_VALUE"
        elif prefix == "ALT_TOV_VALUE":
            off_col, def_col = "oALT_TOV_LOSS_VALUE", "dALT_TOV_LOSS_VALUE"
        else:
            off_col, def_col = f"o{prefix}", f"d{prefix}"
        missing = {off_col, def_col}.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing clean prior columns: {sorted(missing)}")
        prior = frame[["player_id", "player_name", off_col, def_col]].copy()
        prior = prior.rename(columns={off_col: "off", def_col: "def"})
        prior["def"] = -pd.to_numeric(prior["def"], errors="coerce")
        priors[prefix] = prior
    return priors


def _reconcile_display_priors(
    priors: dict[str, pd.DataFrame],
    total_prior: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Make atomic display priors sum exactly to the selected total prior.

    The clean target identity is

        total = six EFG atoms + FT + first-chance completion + second chance.

    The historical clean component file was fit around a different total than
    the selected dynamic state. Put only that prior-level difference into the
    existing completion/baseline bucket so linear ridge additivity is retained.
    """
    atomic = [
        "ALT_EFG_RIM_FREQ",
        "ALT_EFG_RIM_FG",
        "ALT_EFG_MID_FREQ",
        "ALT_EFG_MID_FG",
        "ALT_EFG_THREE_FREQ",
        "ALT_EFG_THREE_FG",
        "ALT_FT",
        "SECOND_CHANCE_CLEAN",
    ]
    total = total_prior[["player_id", "player_name", "off", "def"]].copy()
    total["player_id"] = _normalize_ids(total["player_id"])
    total = total.drop_duplicates("player_id", keep="last").set_index("player_id")
    reconciled = total[["player_name"]].copy()
    for side in ("off", "def"):
        component_sum = pd.Series(0.0, index=total.index)
        for prefix in atomic:
            component = priors[prefix][["player_id", side]].copy()
            component["player_id"] = _normalize_ids(component["player_id"])
            values = component.drop_duplicates("player_id", keep="last").set_index("player_id")[side]
            component_sum = component_sum.add(
                pd.to_numeric(values, errors="coerce").reindex(total.index).fillna(0.0),
                fill_value=0.0,
            )
        reconciled[side] = pd.to_numeric(total[side], errors="coerce") - component_sum
    reconciled = reconciled.reset_index()
    out = dict(priors)
    out["ALT_FC_COMPLETION"] = reconciled
    return out


def _parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _current_output_audit(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    audit: dict[str, Any] = {
        "players": int(len(frame)),
        "duplicate_player_ids": int(frame["player_id"].duplicated().sum()),
        "missing_values": int(frame.isna().sum().sum()),
        "max_offense_display_gap": float(
            (frame["oDISPLAY_SUM"] - frame["off"]).abs().max()
        ),
        "max_defense_display_gap": float(
            (frame["dDISPLAY_SUM"] - frame["def"]).abs().max()
        ),
        "max_net_display_gap": float(
            (frame["nDISPLAY_SUM"] - frame["net_rapm"]).abs().max()
        ),
    }
    if not PBP_STATS_2026.exists() or not EXTERNAL_RATINGS_DIR.exists():
        return audit

    minutes = (
        pd.read_csv(PBP_STATS_2026, usecols=["EntityId", "Minutes"])
        .groupby("EntityId", as_index=False)["Minutes"]
        .sum()
        .rename(columns={"EntityId": "player_id"})
    )
    comparison = frame.merge(minutes, on="player_id", how="left")
    cutoff = pd.Timestamp("2026-07-24")
    sources = [
        ("dpm_cols.parq", "dpm", "o_dpm", "d_dpm"),
        ("pi_rapm_ratings.parq", "pi", "pi_orapm", "pi_drapm"),
        ("npi_rapm_ratings.parq", "npi", "npi_orapm", "npi_drapm"),
    ]
    for filename, prefix, off_col, def_col in sources:
        ratings = pd.read_parquet(EXTERNAL_RATINGS_DIR / filename)
        ratings["date"] = pd.to_datetime(ratings["date"], errors="coerce")
        ratings = (
            ratings.loc[ratings["date"] < cutoff]
            .sort_values("date")
            .groupby("nba_id", as_index=False)
            .tail(1)
            .rename(
                columns={
                    "nba_id": "player_id",
                    off_col: f"{prefix}_off",
                    def_col: f"{prefix}_def",
                }
            )
        )
        ratings[f"{prefix}_net"] = (
            pd.to_numeric(ratings[f"{prefix}_off"], errors="coerce")
            + pd.to_numeric(ratings[f"{prefix}_def"], errors="coerce")
        )
        comparison = comparison.merge(
            ratings[
                [
                    "player_id",
                    f"{prefix}_off",
                    f"{prefix}_def",
                    f"{prefix}_net",
                ]
            ],
            on="player_id",
            how="left",
        )

    established = comparison.loc[comparison["Minutes"] >= 200].copy()
    scale: dict[str, Any] = {
        "minimum_minutes": 200,
        "players": int(len(established)),
        "snapshot_rule": "latest proprietary rating date strictly before 2026-07-24",
    }
    local_columns = {"off": "off", "def": "def", "net": "net_rapm"}
    scale["local_prior_informed"] = {
        f"{side}_sd": float(established[column].std())
        for side, column in local_columns.items()
    }
    for prefix in ("dpm", "pi", "npi"):
        scale[prefix] = {
            f"{side}_sd": float(established[f"{prefix}_{side}"].std())
            for side in ("off", "def", "net")
        }
        scale[prefix]["net_correlation_with_local"] = float(
            established["net_rapm"].corr(established[f"{prefix}_net"])
        )
    audit["external_scale_audit"] = scale
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off-alphas", default="500,1000,1750,3000")
    parser.add_argument("--def-alphas", default="1000,2000,4000,6000")
    parser.add_argument("--persistences", default="0.75,0.9,1.0")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--clean-prior", type=Path, default=DEFAULT_CLEAN_PRIOR)
    parser.add_argument(
        "--reuse-selection-dir",
        type=Path,
        help="reuse a completed selection directory and only build current components",
    )
    parser.add_argument(
        "--skip-current-components",
        action="store_true",
        help="stop after total-RAPM selection, 2026 test, and frozen-state export",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name_map = _load_name_map()

    if args.reuse_selection_dir is not None:
        report_path = args.reuse_selection_dir / "report.json"
        preseason_path = args.reuse_selection_dir / "preseason_2026_prior_state.csv"
        report = json.loads(report_path.read_text())
        selected_spec = report["selected_spec"]
        if args.skip_current_components:
            current_path = Path(report["current_2026_output"])
        else:
            component_priors = _clean_prior_frames(args.clean_prior)
            total_prior = pd.read_csv(preseason_path)
            component_priors = _reconcile_display_priors(component_priors, total_prior)
            component_priors["RAPM"] = total_prior
            current_path = build_clean_weighted_factors(
                26,
                26,
                "RS",
                off_alpha=float(selected_spec["off_alpha"]),
                def_alpha=float(selected_spec["def_alpha"]),
                prior_results_by_prefix=component_priors,
                prior_scale=1.0,
                output_label="walkforward_selected",
            )
        report["current_2026_output"] = str(current_path)
        report["current_2026_output_audit"] = _current_output_audit(current_path)
        (args.output_dir / "report.json").write_text(
            json.dumps(report, indent=2, default=float)
        )
        logging.info("Current prior-informed clean decomposition: %s", current_path)
        return 0

    logging.info("Preparing annual clean-total design matrices once")
    problems: dict[int, dict[str, object]] = {}
    annual_means: dict[int, float] = {}
    total_sum = 0.0
    total_rows = 0
    for year in range(2022, 2027):
        season_type = "ALL" if year <= 2025 else "RS"
        frame = _load_clean_total_df([year % 100], season_type)
        problems[year] = _prepare_direct_ridge_problem(frame)
        values = pd.to_numeric(frame["Net_Diff"], errors="coerce").fillna(0.0)
        total_sum += float(values.sum())
        total_rows += len(values)
        annual_means[year] = total_sum / total_rows
        logging.info("Prepared %s: %d possessions", year, len(frame))

    holdouts = {year: _prepare_holdout(year) for year in range(2023, 2027)}
    candidate_rows: list[dict[str, Any]] = []
    off_alphas = _parse_float_list(args.off_alphas)
    def_alphas = _parse_float_list(args.def_alphas)
    persistences = _parse_float_list(args.persistences)
    total_candidates = len(off_alphas) * len(def_alphas) * len(persistences)
    candidate_number = 0

    # Hyperparameter selection is sealed to 2023-2025. Do not inspect 2026 here.
    for off_alpha in off_alphas:
        for def_alpha in def_alphas:
            for persistence in persistences:
                candidate_number += 1
                annual, states = _solve_sequence(
                    problems,
                    [2022, 2023, 2024],
                    name_map,
                    off_alpha,
                    def_alpha,
                    persistence,
                )
                fold_metrics = {}
                for holdout_year in (2023, 2024, 2025):
                    rating_year = holdout_year - 1
                    forecast_state = _age_state(
                        states[rating_year],
                        holdout_year,
                        persistence,
                    )
                    fold_metrics[holdout_year] = _evaluate(
                        forecast_state,
                        holdouts[holdout_year],
                        annual_means[rating_year],
                    )
                row: dict[str, Any] = {
                    "off_alpha": off_alpha,
                    "def_alpha": def_alpha,
                    "persistence": persistence,
                    "mean_game_margin_rmse": float(
                        np.mean(
                            [
                                fold_metrics[year]["game_margin_rmse"]
                                for year in (2023, 2024, 2025)
                            ]
                        )
                    ),
                    "mean_game_margin_correlation": float(
                        np.mean(
                            [
                                fold_metrics[year]["game_margin_correlation"]
                                for year in (2023, 2024, 2025)
                            ]
                        )
                    ),
                }
                for year, metrics in fold_metrics.items():
                    row[f"rmse_{year}"] = metrics["game_margin_rmse"]
                    row[f"corr_{year}"] = metrics["game_margin_correlation"]
                    row[f"slot_coverage_{year}"] = metrics["rated_player_slot_pct"]
                candidate_rows.append(row)
                if candidate_number % 8 == 0 or candidate_number == total_candidates:
                    logging.info("Scored %d/%d candidates", candidate_number, total_candidates)

    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["mean_game_margin_rmse", "mean_game_margin_correlation"],
        ascending=[True, False],
    )
    candidates.to_csv(args.output_dir / "walk_forward_grid.csv", index=False)
    rmse_tolerance = 0.005
    near_best = candidates.loc[
        candidates["mean_game_margin_rmse"]
        <= candidates["mean_game_margin_rmse"].min() + rmse_tolerance
    ]
    selected = near_best.sort_values(
        ["mean_game_margin_correlation", "off_alpha", "def_alpha"],
        ascending=[False, False, False],
    ).iloc[0]
    selected_spec = {
        "off_alpha": float(selected["off_alpha"]),
        "def_alpha": float(selected["def_alpha"]),
        "persistence": float(selected["persistence"]),
        "selection_folds": [2023, 2024, 2025],
        "selection_metric": "mean next-season game-margin RMSE",
        "tie_break": (
            f"highest mean game-margin correlation within {rmse_tolerance:.3f} "
            "RMSE of the minimum"
        ),
    }
    logging.info("Selected specification: %s", selected_spec)

    # Only now extend through 2025 and unlock the untouched 2026 benchmark.
    annual, states = _solve_sequence(
        problems,
        [2022, 2023, 2024, 2025],
        name_map,
        selected_spec["off_alpha"],
        selected_spec["def_alpha"],
        selected_spec["persistence"],
    )
    frozen_2025 = annual[2025].copy()
    state_2025 = states[2025]
    preseason_2026 = _age_state(
        state_2025,
        2026,
        selected_spec["persistence"],
    )
    frozen_2025.to_csv(args.output_dir / "frozen_2025_prior_informed_total.csv", index=False)
    state_2025.to_csv(args.output_dir / "frozen_2025_rating_state.csv", index=False)
    preseason_2026.to_csv(args.output_dir / "preseason_2026_prior_state.csv", index=False)
    final_2026 = _evaluate(preseason_2026, holdouts[2026], annual_means[2025])

    baseline_frame = pd.read_csv(args.clean_prior)
    baseline_raw = baseline_frame[["player_id", "player_name", "off", "def"]].copy()
    baseline_raw["def"] = -pd.to_numeric(baseline_raw["def"], errors="coerce")
    baseline_2026 = _evaluate(baseline_raw, holdouts[2026], annual_means[2025])

    report: dict[str, Any] = {
        "selected_spec": selected_spec,
        "selected_pre_2026": selected.to_dict(),
        "held_out_2026": {
            "prior_informed_preseason_state": final_2026,
            "frozen_2022_2025_zero_centered_clean": baseline_2026,
        },
        "methodology": {
            "prior": "previous annual same-component RAPM, regressed toward zero by persistence",
            "likelihood": "one season of clean WNBA possession outcomes",
            "objective": "ridge centered on the dated prior with separate offense/defense alphas",
            "leakage_rule": "all selection ended with 2025 outcomes; 2026 opened once after selection",
            "proprietary_metrics_in_training": False,
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, default=float))

    if args.skip_current_components:
        logging.info("Skipped current clean-component build")
        return 0

    component_priors = _clean_prior_frames(args.clean_prior)
    component_priors = _reconcile_display_priors(component_priors, preseason_2026)
    component_priors["RAPM"] = preseason_2026
    current_path = build_clean_weighted_factors(
        26,
        26,
        "RS",
        off_alpha=selected_spec["off_alpha"],
        def_alpha=selected_spec["def_alpha"],
        prior_results_by_prefix=component_priors,
        prior_scale=1.0,
        output_label="walkforward_selected",
    )
    report["current_2026_output"] = str(current_path)
    report["current_2026_output_audit"] = _current_output_audit(current_path)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, default=float))
    logging.info("Current prior-informed clean decomposition: %s", current_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
