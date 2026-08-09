#!/usr/bin/env python3
"""Build factor-specific WNBA box priors and single-season true-prior RAPM.

The box layer is trained only on seasons before the season being scored. The
final lineup likelihood is one season only, with the box predictions inserted
as the ridge centers for the total and every clean decomposition component.
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
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wnba_test.build_prior_informed_wnba import (
    _evaluate,
    _prepare_holdout,
)
from wnba_test.clean_decomp_wnba import (
    DISPLAY_PREFIXES,
    RESULTS_DIR,
    _load_clean_total_df,
    _load_name_map,
    _prepare_direct_ridge_problem,
    _solve_direct_ridge_problem,
    build_clean_weighted_factors,
    run_metric,
)


BOX_DATA_DIR = ROOT.parent / "wnba" / "data"
DEFAULT_OUTPUT = RESULTS_DIR / "box_prior_true_prior"
TARGET_PREFIXES = [
    "ALT_EFG_RIM_FREQ",
    "ALT_EFG_RIM_FG",
    "ALT_EFG_MID_FREQ",
    "ALT_EFG_MID_FG",
    "ALT_EFG_THREE_FREQ",
    "ALT_EFG_THREE_FG",
    "ALT_FT",
    "ALT_FT_FREQ",
    "ALT_FC_COMPLETION",
    "ALT_TOV_VALUE",
    "ALT_BADPASS_TOV_VALUE",
    "SECOND_CHANCE_CLEAN",
]
ADDITIVE_PREFIXES = [
    "ALT_EFG_RIM_FREQ",
    "ALT_EFG_RIM_FG",
    "ALT_EFG_MID_FREQ",
    "ALT_EFG_MID_FG",
    "ALT_EFG_THREE_FREQ",
    "ALT_EFG_THREE_FG",
    "ALT_FT",
    "ALT_FC_COMPLETION",
    "SECOND_CHANCE_CLEAN",
]
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _smoothed_rate(
    count: pd.Series,
    exposure: pd.Series,
    prior_exposure: float,
) -> pd.Series:
    count = pd.to_numeric(count, errors="coerce").fillna(0.0)
    exposure = pd.to_numeric(exposure, errors="coerce").fillna(0.0)
    league_rate = 100.0 * count.sum() / max(float(exposure.sum()), 1.0)
    return (100.0 * count + prior_exposure * league_rate) / (
        exposure + prior_exposure
    )


def _smoothed_share(
    numerator: pd.Series,
    denominator: pd.Series,
    prior_attempts: float,
) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce").fillna(0.0)
    denominator = pd.to_numeric(denominator, errors="coerce").fillna(0.0)
    league_share = numerator.sum() / max(float(denominator.sum()), 1.0)
    return (numerator + prior_attempts * league_share) / (
        denominator + prior_attempts
    )


def load_box_features(year: int) -> pd.DataFrame:
    """Create stabilized player-season box/derived features."""
    path = BOX_DATA_DIR / f"{year}_pbp.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    raw = pd.read_csv(path)
    raw["player_id"] = (
        pd.to_numeric(raw["EntityId"], errors="coerce")
        .astype("Int64")
        .astype(str)
    )
    raw["player_name"] = raw["Name"].fillna(raw.get("ShortName", "")).astype(str)
    out = raw[["player_id", "player_name"]].copy()
    out["year"] = int(year)
    # Administrative/stat-feed corrections can occasionally produce negative
    # minute deltas. They are not meaningful exposure and must not qualify a
    # player for a current-season prior.
    out["minutes"] = _numeric(raw, "Minutes").clip(lower=0.0)
    out["off_poss"] = _numeric(raw, "OffPoss")
    out["def_poss"] = _numeric(raw, "DefPoss")

    fg2a = _numeric(raw, "FG2A")
    fg3a = _numeric(raw, "FG3A")
    fga = fg2a + fg3a
    fta = _numeric(raw, "FTA")
    rim_fga = _numeric(raw, "AtRimFGA")
    rim_fgm = _numeric(raw, "AtRimFGM")
    mid_fga = _numeric(raw, "ShortMidRangeFGA") + _numeric(raw, "LongMidRangeFGA")
    mid_fgm = _numeric(raw, "ShortMidRangeFGM") + _numeric(raw, "LongMidRangeFGM")
    three_fga = fg3a
    three_fgm = _numeric(raw, "FG3M")

    offense_counts = {
        "points100": _numeric(raw, "Points"),
        "fga100": fga,
        "tsa100": fga + 0.44 * fta,
        "rim_fga100": rim_fga,
        "mid_fga100": mid_fga,
        "three_fga100": three_fga,
        "fta100": fta,
        "assists100": _numeric(raw, "Assists"),
        "assist_points100": _numeric(raw, "AssistPoints"),
        "rim_assists100": _numeric(raw, "AtRimAssists"),
        "turnovers100": _numeric(raw, "Turnovers"),
        "badpass_tov100": _numeric(raw, "BadPassTurnovers"),
        "lostball_tov100": _numeric(raw, "LostBallTurnovers"),
        "liveball_tov100": _numeric(raw, "LiveBallTurnovers"),
        "oreb100": _numeric(raw, "OffRebounds"),
        "self_oreb100": _numeric(raw, "SelfOReb"),
        "fouls_drawn100": _numeric(raw, "FoulsDrawn"),
        "shooting_fouls_drawn100": (
            _numeric(raw, "TwoPtShootingFoulsDrawn")
            + _numeric(raw, "ThreePtShootingFoulsDrawn")
        ),
    }
    for name, count in offense_counts.items():
        out[name] = _smoothed_rate(count, out["off_poss"], prior_exposure=250.0)

    defense_counts = {
        "steals100": _numeric(raw, "Steals"),
        "badpass_steals100": _numeric(raw, "BadPassSteals"),
        "lostball_steals100": _numeric(raw, "LostBallSteals"),
        "blocks100": _numeric(raw, "Blocks"),
        "blocked2s100": _numeric(raw, "Blocked2s"),
        "blocked3s100": _numeric(raw, "Blocked3s"),
        "defreb100": _numeric(raw, "DefRebounds"),
        "shooting_fouls100": _numeric(raw, "ShootingFouls"),
        "fouls100": _numeric(raw, "Fouls"),
    }
    for name, count in defense_counts.items():
        out[name] = _smoothed_rate(count, out["def_poss"], prior_exposure=400.0)

    out["rim_freq"] = _smoothed_share(rim_fga, fga, prior_attempts=75.0)
    out["mid_freq"] = _smoothed_share(mid_fga, fga, prior_attempts=75.0)
    out["three_freq"] = _smoothed_share(three_fga, fga, prior_attempts=75.0)
    out["rim_pct"] = _smoothed_share(rim_fgm, rim_fga, prior_attempts=50.0)
    out["mid_pct"] = _smoothed_share(mid_fgm, mid_fga, prior_attempts=50.0)
    out["three_pct"] = _smoothed_share(three_fgm, three_fga, prior_attempts=75.0)
    out["ft_pct"] = _smoothed_share(
        _numeric(raw, "FtPoints"),
        fta,
        prior_attempts=75.0,
    )
    out["usage_proxy"] = _smoothed_rate(
        fga + 0.44 * fta + _numeric(raw, "Turnovers"),
        out["off_poss"],
        prior_exposure=250.0,
    )
    out["assist_tov_ratio"] = (
        _numeric(raw, "Assists") + 1.0
    ) / (_numeric(raw, "Turnovers") + 1.0)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def feature_columns(prefix: str, side: str) -> list[str]:
    if side == "off":
        mapping = {
            "ALT_EFG_RIM_FREQ": ["rim_freq", "rim_fga100", "fga100", "usage_proxy", "rim_assists100"],
            "ALT_EFG_RIM_FG": ["rim_pct", "rim_fga100", "rim_assists100", "assist_points100"],
            "ALT_EFG_MID_FREQ": ["mid_freq", "mid_fga100", "fga100", "usage_proxy"],
            "ALT_EFG_MID_FG": ["mid_pct", "mid_fga100", "usage_proxy"],
            "ALT_EFG_THREE_FREQ": ["three_freq", "three_fga100", "fga100", "assist_points100"],
            "ALT_EFG_THREE_FG": ["three_pct", "three_fga100", "assist_points100"],
            "ALT_FT": ["fta100", "ft_pct", "shooting_fouls_drawn100", "fouls_drawn100", "usage_proxy"],
            "ALT_FT_FREQ": ["fta100", "shooting_fouls_drawn100", "fouls_drawn100", "usage_proxy"],
            "ALT_FC_COMPLETION": ["turnovers100", "badpass_tov100", "lostball_tov100", "liveball_tov100", "assists100", "assist_tov_ratio", "usage_proxy"],
            "ALT_TOV_VALUE": ["turnovers100", "badpass_tov100", "lostball_tov100", "liveball_tov100", "assists100", "assist_tov_ratio", "usage_proxy"],
            "ALT_BADPASS_TOV_VALUE": ["badpass_tov100", "assists100", "assist_tov_ratio", "usage_proxy"],
            "SECOND_CHANCE_CLEAN": ["oreb100", "self_oreb100", "rim_fga100", "mid_fga100"],
        }
    else:
        mapping = {
            "ALT_EFG_RIM_FREQ": ["blocks100", "blocked2s100", "shooting_fouls100", "fouls100", "defreb100"],
            "ALT_EFG_RIM_FG": ["blocks100", "blocked2s100", "shooting_fouls100", "defreb100"],
            "ALT_EFG_MID_FREQ": ["blocks100", "steals100", "fouls100", "defreb100"],
            "ALT_EFG_MID_FG": ["blocks100", "steals100", "fouls100", "defreb100"],
            "ALT_EFG_THREE_FREQ": ["steals100", "blocked3s100", "fouls100", "defreb100"],
            "ALT_EFG_THREE_FG": ["steals100", "blocked3s100", "fouls100", "defreb100"],
            "ALT_FT": ["shooting_fouls100", "fouls100", "blocks100"],
            "ALT_FT_FREQ": ["shooting_fouls100", "fouls100", "blocks100"],
            "ALT_FC_COMPLETION": ["steals100", "badpass_steals100", "lostball_steals100", "blocks100"],
            "ALT_TOV_VALUE": ["steals100", "badpass_steals100", "lostball_steals100"],
            "ALT_BADPASS_TOV_VALUE": ["badpass_steals100", "steals100", "lostball_steals100"],
            "SECOND_CHANCE_CLEAN": ["defreb100", "blocks100", "fouls100"],
        }
    return mapping[prefix]


def target_path(output_dir: Path, year: int, prefix: str) -> Path:
    return output_dir / "annual_targets" / str(year) / f"{prefix}.csv"


def build_annual_targets(
    output_dir: Path,
    rebuild: bool,
    target_off_alpha: float,
    target_def_alpha: float,
) -> None:
    name_map = _load_name_map()
    for year in range(2022, 2026):
        for prefix in TARGET_PREFIXES:
            path = target_path(output_dir, year, prefix)
            if path.exists() and not rebuild:
                continue
            logging.info("Annual target %s %s", year, prefix)
            result = run_metric(
                prefix,
                [year % 100],
                "ALL",
                name_map,
                off_alpha=target_off_alpha,
                def_alpha=target_def_alpha,
                time_decay_days=None,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(path, index=False)


def load_target(output_dir: Path, year: int, prefix: str) -> pd.DataFrame:
    frame = pd.read_csv(target_path(output_dir, year, prefix))
    frame["player_id"] = (
        pd.to_numeric(frame["player_id"], errors="coerce")
        .astype("Int64")
        .astype(str)
    )
    frame["year"] = int(year)
    return frame


def _training_rows(
    output_dir: Path,
    box_by_year: dict[int, pd.DataFrame],
    years: list[int],
    prefix: str,
    side: str,
) -> pd.DataFrame:
    pieces = []
    for year in years:
        target = load_target(output_dir, year, prefix)
        target = target[["player_id", side, f"{side}_poss"]].rename(
            columns={side: "target", f"{side}_poss": "target_poss"}
        )
        merged = box_by_year[year].merge(
            target,
            on="player_id",
            how="inner",
        )
        merged["target"] = pd.to_numeric(merged["target"], errors="coerce")
        merged["target_poss"] = pd.to_numeric(merged["target_poss"], errors="coerce")
        pieces.append(merged)
    return pd.concat(pieces, ignore_index=True).dropna(
        subset=["target", "target_poss"]
    )


def _fit_model(
    rows: pd.DataFrame,
    features: list[str],
    alpha: float,
) -> Pipeline:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )
    weights = np.sqrt(rows["target_poss"].clip(lower=1.0, upper=2500.0))
    weights = weights / weights.mean()
    model.fit(
        rows[features],
        rows["target"],
        ridge__sample_weight=weights,
    )
    return model


def select_box_model(
    output_dir: Path,
    box_by_year: dict[int, pd.DataFrame],
    train_years: list[int],
    prefix: str,
    side: str,
) -> tuple[Pipeline, dict[str, Any]]:
    features = feature_columns(prefix, side)
    fold_years = train_years[1:]
    alpha_scores = []
    for alpha in RIDGE_ALPHAS:
        errors = []
        weights = []
        predictions = []
        actuals = []
        for validation_year in fold_years:
            fit_years = [year for year in train_years if year < validation_year]
            if not fit_years:
                continue
            fit_rows = _training_rows(
                output_dir, box_by_year, fit_years, prefix, side
            )
            validation = _training_rows(
                output_dir, box_by_year, [validation_year], prefix, side
            )
            model = _fit_model(fit_rows, features, alpha)
            prediction = model.predict(validation[features])
            weight = np.sqrt(
                validation["target_poss"].clip(lower=1.0, upper=2500.0)
            ).to_numpy()
            errors.extend((prediction - validation["target"]).tolist())
            weights.extend(weight.tolist())
            predictions.extend(prediction.tolist())
            actuals.extend(validation["target"].tolist())
        if errors:
            rmse = float(
                np.sqrt(np.average(np.square(errors), weights=weights))
            )
            correlation = float(pd.Series(predictions).corr(pd.Series(actuals)))
        else:
            rmse = np.nan
            correlation = np.nan
        alpha_scores.append(
            {"alpha": alpha, "forward_rmse": rmse, "forward_correlation": correlation}
        )
    scored = pd.DataFrame(alpha_scores)
    if scored["forward_rmse"].notna().any():
        selected_alpha = float(
            scored.sort_values(["forward_rmse", "forward_correlation"], ascending=[True, False])
            .iloc[0]["alpha"]
        )
    else:
        selected_alpha = 10.0
    rows = _training_rows(output_dir, box_by_year, train_years, prefix, side)
    model = _fit_model(rows, features, selected_alpha)
    lower = float(rows["target"].quantile(0.01))
    upper = float(rows["target"].quantile(0.99))
    metadata = {
        "prefix": prefix,
        "side": side,
        "features": features,
        "selected_alpha": selected_alpha,
        "training_years": train_years,
        "training_rows": int(len(rows)),
        "prediction_clip": [lower, upper],
        "alpha_scores": alpha_scores,
    }
    return model, metadata


def _derive_additive_priors(
    base_priors: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Add exact derived children and the additive total prior."""
    priors = {prefix: frame.copy() for prefix, frame in base_priors.items()}

    ft_severity = priors["ALT_FT"].copy()
    ft_severity["off"] = priors["ALT_FT"]["off"] - priors["ALT_FT_FREQ"]["off"]
    ft_severity["def"] = priors["ALT_FT"]["def"] - priors["ALT_FT_FREQ"]["def"]
    priors["ALT_FT_SEVERITY"] = ft_severity

    scoring_tov = priors["ALT_TOV_VALUE"].copy()
    scoring_tov["off"] = (
        priors["ALT_TOV_VALUE"]["off"] - priors["ALT_BADPASS_TOV_VALUE"]["off"]
    )
    scoring_tov["def"] = (
        priors["ALT_TOV_VALUE"]["def"] - priors["ALT_BADPASS_TOV_VALUE"]["def"]
    )
    priors["ALT_SCORING_TOV_VALUE"] = scoring_tov

    total = priors[ADDITIVE_PREFIXES[0]].copy()
    total[["off", "def"]] = 0.0
    for prefix in ADDITIVE_PREFIXES:
        total["off"] += priors[prefix]["off"]
        total["def"] += priors[prefix]["def"]
    priors["RAPM"] = total
    return priors


def classify_box_prior_sources(
    current: pd.DataFrame,
    previous: pd.DataFrame | None,
    universe: list[str],
) -> pd.Series:
    """Classify the player-level box source from possession exposure."""
    current_ids = set(
        current.loc[
            (current["off_poss"] > 0) & (current["def_poss"] > 0),
            "player_id",
        ].astype(str)
    )
    previous_ids = (
        set()
        if previous is None
        else set(previous["player_id"].astype(str))
    )
    player_ids = pd.Index(list(map(str, universe)), name="player_id")
    values = np.where(
        player_ids.isin(current_ids),
        "current_single_season",
        np.where(
            player_ids.isin(previous_ids),
            "previous_season_fallback",
            "imputed_zero",
        ),
    )
    return pd.Series(values, index=player_ids, name="box_prior_source")


def audit_box_exposure_conflicts(current: pd.DataFrame) -> dict[str, Any]:
    """Report rows whose possession exposure conflicts with reported minutes."""
    mask = (
        current["minutes"].le(0)
        & current["off_poss"].gt(0)
        & current["def_poss"].gt(0)
    )
    players = []
    for _, row in current.loc[
        mask,
        ["player_id", "player_name", "minutes", "off_poss", "def_poss"],
    ].iterrows():
        players.append(
            {
                "player_id": str(row["player_id"]),
                "player_name": str(row["player_name"]),
                "minutes": float(row["minutes"]),
                "off_poss": float(row["off_poss"]),
                "def_poss": float(row["def_poss"]),
            }
        )
    return {"count": len(players), "players": players}


def build_box_priors_for_year(
    output_dir: Path,
    box_by_year: dict[int, pd.DataFrame],
    target_year: int,
    train_years: list[int],
    universe: list[str],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, list[dict[str, Any]]]:
    current = box_by_year[target_year].copy()
    previous = box_by_year.get(target_year - 1)
    universe_frame = pd.DataFrame({"player_id": list(map(str, universe))})
    source = classify_box_prior_sources(
        current,
        previous,
        universe_frame["player_id"].tolist(),
    ).to_numpy()
    readable = universe_frame.copy()
    readable["box_prior_source"] = source
    current_context = current[
        ["player_id", "player_name", "minutes", "off_poss", "def_poss"]
    ].copy()
    readable = readable.merge(current_context, on="player_id", how="left")
    priors: dict[str, pd.DataFrame] = {}
    metadata: list[dict[str, Any]] = []

    for prefix in TARGET_PREFIXES:
        result = universe_frame.copy()
        result["player_name"] = result["player_id"].map(
            dict(zip(current["player_id"], current["player_name"]))
        ).fillna(result["player_id"].map(_load_name_map())).fillna(
            "ID_" + result["player_id"]
        )
        for side in ("off", "def"):
            model, model_metadata = select_box_model(
                output_dir,
                box_by_year,
                train_years,
                prefix,
                side,
            )
            metadata.append(model_metadata)
            features = model_metadata["features"]
            current_prediction = pd.Series(
                np.clip(
                    model.predict(current[features]),
                    *model_metadata["prediction_clip"],
                ),
                index=current["player_id"],
            )
            if previous is not None:
                previous_prediction = pd.Series(
                    np.clip(
                        model.predict(previous[features]),
                        *model_metadata["prediction_clip"],
                    )
                    * 0.90,
                    index=previous["player_id"],
                )
            else:
                previous_prediction = pd.Series(dtype=float)
            values = []
            for player_id, prior_source in zip(
                universe_frame["player_id"], source
            ):
                if prior_source == "current_single_season":
                    values.append(float(current_prediction.get(player_id, 0.0)))
                elif prior_source == "previous_season_fallback":
                    values.append(float(previous_prediction.get(player_id, 0.0)))
                else:
                    values.append(0.0)
            result[side] = values
        priors[prefix] = result

    # Preserve exact child identities instead of fitting redundant child models.
    priors = _derive_additive_priors(priors)

    for prefix, frame in priors.items():
        if prefix == "RAPM":
            readable["box_off"] = frame["off"]
            readable["box_def"] = -frame["def"]
            readable["box_net"] = readable["box_off"] + readable["box_def"]
        else:
            readable[f"box_o{prefix}"] = frame["off"]
            readable[f"box_d{prefix}"] = -frame["def"]
    return priors, readable, metadata


def _parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def load_frozen_spec(report_path: Path) -> dict[str, Any]:
    if not report_path.exists():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text())
    spec = report.get("selected_spec")
    if not isinstance(spec, dict):
        raise RuntimeError(f"missing selected_spec in {report_path}")
    required = {
        "off_alpha",
        "def_alpha",
        "prior_scale",
        "box_target_off_alpha",
        "box_target_def_alpha",
        "selection_folds",
    }
    missing = sorted(required.difference(spec))
    if missing:
        raise RuntimeError(f"frozen selected_spec is missing fields: {missing}")
    selection_folds = [int(year) for year in spec["selection_folds"]]
    if any(year >= 2026 for year in selection_folds):
        raise RuntimeError(
            f"frozen selected_spec includes 2026-or-later selection folds: "
            f"{selection_folds}"
        )
    return {
        **spec,
        "off_alpha": float(spec["off_alpha"]),
        "def_alpha": float(spec["def_alpha"]),
        "prior_scale": float(spec["prior_scale"]),
        "box_target_off_alpha": float(spec["box_target_off_alpha"]),
        "box_target_def_alpha": float(spec["box_target_def_alpha"]),
        "selection_folds": selection_folds,
    }


def build_current_board(
    output_dir: Path,
    selection_report: Path,
    output_label: str,
    target_year: int = 2026,
) -> tuple[Path, dict[str, Any]]:
    """Refresh only the current board while preserving frozen model selection."""
    if target_year != 2026:
        raise RuntimeError("the frozen daily WNBA lane currently supports 2026 only")
    spec = load_frozen_spec(selection_report)
    output_dir.mkdir(parents=True, exist_ok=True)
    build_annual_targets(
        output_dir,
        rebuild=False,
        target_off_alpha=spec["box_target_off_alpha"],
        target_def_alpha=spec["box_target_def_alpha"],
    )
    box_by_year = {
        year: load_box_features(year) for year in range(2022, target_year + 1)
    }
    box_exposure_audit = audit_box_exposure_conflicts(box_by_year[target_year])
    current_frame = _load_clean_total_df([target_year % 100], "RS")
    current_problem = _prepare_direct_ridge_problem(current_frame)
    priors, readable, metadata = build_box_priors_for_year(
        output_dir,
        box_by_year,
        target_year,
        list(range(2022, target_year)),
        list(current_problem["all_players"]),
    )
    prior_path = output_dir / f"box_priors_{target_year}.csv"
    readable.to_csv(prior_path, index=False)
    board_path = build_clean_weighted_factors(
        target_year % 100,
        target_year % 100,
        "RS",
        off_alpha=spec["off_alpha"],
        def_alpha=spec["def_alpha"],
        prior_results_by_prefix=priors,
        prior_scale=spec["prior_scale"],
        output_label=output_label,
    )
    current_report = {
        "target_year": target_year,
        "model_policy": "reuse_frozen_pre_2026_selection",
        "selection_report": str(selection_report),
        "selected_spec": spec,
        "box_prior_training_years": list(range(2022, target_year)),
        "box_prior_path": str(prior_path),
        "board_path": str(board_path),
        "box_prior_source_counts": readable["box_prior_source"]
        .value_counts(dropna=False)
        .to_dict(),
        "box_exposure_audit": box_exposure_audit,
        "model_metadata": metadata,
    }
    current_report_path = output_dir / "current_build_report.json"
    current_report_path.write_text(
        json.dumps(current_report, indent=2, default=float) + "\n"
    )
    return board_path, current_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rebuild-targets", action="store_true")
    parser.add_argument("--target-off-alpha", type=float, default=750.0)
    parser.add_argument("--target-def-alpha", type=float, default=1000.0)
    parser.add_argument("--off-alphas", default="500,1000,1750,3000,5000")
    parser.add_argument("--def-alphas", default="500,1000,2000,4000,6000")
    parser.add_argument("--prior-scales", default="0.5,0.75,1.0,1.25")
    parser.add_argument("--skip-current-components", action="store_true")
    parser.add_argument(
        "--current-only",
        action="store_true",
        help="Rebuild only the 2026 board from the frozen pre-2026 selection",
    )
    parser.add_argument(
        "--selection-report",
        type=Path,
        help="Frozen selection report; defaults to OUTPUT_DIR/report.json",
    )
    parser.add_argument("--output-label", default="box_trueprior_daily")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.current_only:
        selection_report = args.selection_report or (args.output_dir / "report.json")
        board_path, _ = build_current_board(
            args.output_dir,
            selection_report,
            args.output_label,
        )
        logging.info("Wrote current frozen-spec board to %s", board_path)
        return 0
    build_annual_targets(
        args.output_dir,
        args.rebuild_targets,
        args.target_off_alpha,
        args.target_def_alpha,
    )
    box_by_year = {year: load_box_features(year) for year in range(2022, 2027)}

    logging.info("Preparing annual clean-total lineup problems")
    problems = {}
    means = {}
    running_sum = 0.0
    running_rows = 0
    for year in range(2022, 2027):
        season_type = "ALL" if year <= 2025 else "RS"
        frame = _load_clean_total_df([year % 100], season_type)
        problems[year] = _prepare_direct_ridge_problem(frame)
        target = pd.to_numeric(frame["Net_Diff"], errors="coerce").fillna(0.0)
        running_sum += float(target.sum())
        running_rows += len(target)
        means[year] = running_sum / running_rows

    priors_by_year = {}
    readable_by_year = {}
    model_metadata = {}
    for year in range(2023, 2027):
        logging.info("Building leakage-safe %s box priors", year)
        train_years = list(range(2022, year))
        universe = list(problems[year]["all_players"])
        priors, readable, metadata = build_box_priors_for_year(
            args.output_dir,
            box_by_year,
            year,
            train_years,
            universe,
        )
        priors_by_year[year] = priors
        readable_by_year[year] = readable
        model_metadata[year] = metadata
        readable.to_csv(args.output_dir / f"box_priors_{year}.csv", index=False)

    holdouts = {year: _prepare_holdout(year) for year in (2024, 2025, 2026)}
    grid_rows = []
    for off_alpha in _parse_float_list(args.off_alphas):
        for def_alpha in _parse_float_list(args.def_alphas):
            for prior_scale in _parse_float_list(args.prior_scales):
                folds = {}
                for rating_year in (2023, 2024):
                    rating = _solve_direct_ridge_problem(
                        problems[rating_year],
                        _load_name_map(),
                        off_alpha=off_alpha,
                        def_alpha=def_alpha,
                        prior_results=priors_by_year[rating_year]["RAPM"],
                        prior_scale=prior_scale,
                    )
                    folds[rating_year + 1] = _evaluate(
                        rating,
                        holdouts[rating_year + 1],
                        means[rating_year],
                    )
                grid_rows.append(
                    {
                        "off_alpha": off_alpha,
                        "def_alpha": def_alpha,
                        "prior_scale": prior_scale,
                        "mean_game_margin_rmse": float(
                            np.mean(
                                [
                                    folds[2024]["game_margin_rmse"],
                                    folds[2025]["game_margin_rmse"],
                                ]
                            )
                        ),
                        "mean_game_margin_correlation": float(
                            np.mean(
                                [
                                    folds[2024]["game_margin_correlation"],
                                    folds[2025]["game_margin_correlation"],
                                ]
                            )
                        ),
                        "rmse_2024": folds[2024]["game_margin_rmse"],
                        "corr_2024": folds[2024]["game_margin_correlation"],
                        "rmse_2025": folds[2025]["game_margin_rmse"],
                        "corr_2025": folds[2025]["game_margin_correlation"],
                    }
                )
    grid = pd.DataFrame(grid_rows).sort_values(
        ["mean_game_margin_rmse", "mean_game_margin_correlation"],
        ascending=[True, False],
    )
    grid.to_csv(args.output_dir / "true_prior_grid.csv", index=False)
    near = grid.loc[
        grid["mean_game_margin_rmse"]
        <= grid["mean_game_margin_rmse"].min() + 0.005
    ]
    selected = near.sort_values(
        ["mean_game_margin_correlation", "off_alpha", "def_alpha"],
        ascending=[False, False, False],
    ).iloc[0]
    spec = {
        "off_alpha": float(selected["off_alpha"]),
        "def_alpha": float(selected["def_alpha"]),
        "prior_scale": float(selected["prior_scale"]),
        "box_target_off_alpha": float(args.target_off_alpha),
        "box_target_def_alpha": float(args.target_def_alpha),
        "selection_folds": [2024, 2025],
        "selection_metric": "mean next-season game-margin RMSE",
        "tie_break": "highest correlation within 0.005 RMSE of the minimum",
    }

    frozen_2025 = _solve_direct_ridge_problem(
        problems[2025],
        _load_name_map(),
        off_alpha=spec["off_alpha"],
        def_alpha=spec["def_alpha"],
        prior_results=priors_by_year[2025]["RAPM"],
        prior_scale=spec["prior_scale"],
    )
    frozen_2025.to_csv(args.output_dir / "frozen_2025_box_true_prior_total.csv", index=False)
    heldout_2026 = _evaluate(frozen_2025, holdouts[2026], means[2025])
    report: dict[str, Any] = {
        "selected_spec": spec,
        "selected_pre_2026": selected.to_dict(),
        "held_out_2026": heldout_2026,
        "box_prior_training": {
            "training_seasons": [2022, 2023, 2024, 2025],
            "current_prior_source": "stabilized current-season WNBA PBP Stats box row",
            "fallback": "90% of previous-season box prediction, then zero",
            "proprietary_metrics_in_training": False,
        },
        "model_metadata": model_metadata[2026],
    }

    if not args.skip_current_components:
        current_path = build_clean_weighted_factors(
            26,
            26,
            "RS",
            off_alpha=spec["off_alpha"],
            def_alpha=spec["def_alpha"],
            prior_results_by_prefix=priors_by_year[2026],
            prior_scale=spec["prior_scale"],
            output_label="box_trueprior_selected",
        )
        report["current_2026_output"] = str(current_path)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=float)
    )
    logging.info("Selected %s", spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
