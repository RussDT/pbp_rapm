"""PROTOTYPE: compare recency kernels on one-step-ahead NBA skill prediction.

Question: does a fixed daily exponential remain competitive with simple
alternatives when league centering, exposure weighting, shrinkage, training
dates, and scored observations are held constant?

This is deliberately not a production estimated-skills builder. It prints a
research table and writes nothing. The 2026 column is descriptive because that
season has already been inspected elsewhere in this repository.

Run:
    .venv/bin/python nba_pipeline/scripts/prototype_recency_kernels.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

import build_nba_estimated_skills as estimated


DEFAULT_SKILLS = "three_pct,ast_rate,blk_rate"
TUNE_SEASONS = (2020, 2021, 2022, 2023)
VALIDATION_SEASONS = (2024, 2025)
STRESS_SEASONS = (2026,)
PRIOR_GAME_EQUIVALENTS = (0.0, 1.0, 5.0, 20.0, 50.0)
EXP_HALF_LIVES = (30.0, 90.0, 180.0, 365.0, 550.0, 730.0, 1100.0, 1500.0, 3000.0)
ROLLING_WINDOWS = (30, 90, 180, 365, 730, 1460)
OFFSEASON_HALF_LIVES = (180.0, 365.0, 730.0, 1500.0)
OFFSEASON_RETENTIONS = (0.25, 0.50, 0.75)
FAST_HALF_LIVES = (30.0, 90.0, 180.0, 365.0)
SLOW_HALF_LIVES = (730.0, 1500.0, 3000.0)
FAST_MIXES = (0.25, 0.50, 0.75)
BOOTSTRAP_DRAWS = 2000
SEED = 20260903


@dataclass
class SelectedModel:
    family: str
    config: str
    pred: np.ndarray
    tune_wmse: float
    validation_wmse: float
    stress_wmse: float


def wmse(pred: np.ndarray, obs: np.ndarray, exposure: np.ndarray, mask: np.ndarray) -> float:
    err = obs[mask] - pred[mask]
    weight = exposure[mask]
    return float(np.sum(weight * err * err) / np.sum(weight))


def history_exponential(
    sub: pd.DataFrame,
    centered_contribution: np.ndarray,
    half_lives: Iterable[float],
    offseason_retentions: Iterable[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Return strictly-prior effective numerator/exposure for each kernel."""
    hls = np.asarray(list(half_lives), dtype=float)
    if offseason_retentions is None:
        specs = [(float(hl), 1.0) for hl in hls]
    else:
        specs = [
            (float(hl), float(retention))
            for hl in hls
            for retention in offseason_retentions
        ]
    n_specs = len(specs)
    numer = np.zeros((n_specs, len(sub)), dtype=float)
    denom = np.zeros((n_specs, len(sub)), dtype=float)
    spec_hl = np.asarray([s[0] for s in specs])
    spec_retention = np.asarray([s[1] for s in specs])
    days_gap = sub["days_gap"].to_numpy(dtype=float)
    season_cross = sub["season_cross"].to_numpy(dtype=float)
    exposure = sub["exposure"].to_numpy(dtype=float)

    for rows in sub.groupby("player_idx", sort=False).indices.values():
        state_n = np.zeros(n_specs, dtype=float)
        state_d = np.zeros(n_specs, dtype=float)
        for row in rows:
            factor = np.exp2(-days_gap[row] / spec_hl)
            if season_cross[row]:
                factor *= spec_retention
            state_n *= factor
            state_d *= factor
            numer[:, row] = state_n
            denom[:, row] = state_d
            state_n += centered_contribution[row]
            state_d += exposure[row]

    configs = [
        {"half_life": hl, "offseason_retention": retention}
        for hl, retention in specs
    ]
    return numer, denom, configs


def history_no_decay(
    sub: pd.DataFrame, centered_contribution: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    numer = np.zeros((1, len(sub)), dtype=float)
    denom = np.zeros((1, len(sub)), dtype=float)
    exposure = sub["exposure"].to_numpy(dtype=float)
    for rows in sub.groupby("player_idx", sort=False).indices.values():
        state_n = 0.0
        state_d = 0.0
        for row in rows:
            numer[0, row] = state_n
            denom[0, row] = state_d
            state_n += centered_contribution[row]
            state_d += exposure[row]
    return numer, denom, [{"decay": "none"}]


def history_rolling(
    sub: pd.DataFrame, centered_contribution: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[dict[str, int]]]:
    numer = np.zeros((len(ROLLING_WINDOWS), len(sub)), dtype=float)
    denom = np.zeros((len(ROLLING_WINDOWS), len(sub)), dtype=float)
    exposure = sub["exposure"].to_numpy(dtype=float)
    dates = sub["game_date"].to_numpy(dtype="datetime64[D]").astype(np.int64)

    for rows in sub.groupby("player_idx", sort=False).indices.values():
        player_dates = dates[rows]
        player_n = centered_contribution[rows]
        player_d = exposure[rows]
        prefix_n = np.concatenate(([0.0], np.cumsum(player_n)))
        prefix_d = np.concatenate(([0.0], np.cumsum(player_d)))
        right = np.arange(len(rows))
        for kernel_idx, window in enumerate(ROLLING_WINDOWS):
            left = np.searchsorted(player_dates, player_dates - window, side="left")
            numer[kernel_idx, rows] = prefix_n[right] - prefix_n[left]
            denom[kernel_idx, rows] = prefix_d[right] - prefix_d[left]

    return numer, denom, [{"window_days": int(w)} for w in ROLLING_WINDOWS]


def history_two_speed(
    exp_numer: np.ndarray,
    exp_denom: np.ndarray,
    exp_configs: list[dict[str, float]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    by_hl = {float(c["half_life"]): idx for idx, c in enumerate(exp_configs)}
    specs = [
        (fast, slow, mix)
        for fast in FAST_HALF_LIVES
        for slow in SLOW_HALF_LIVES
        for mix in FAST_MIXES
    ]
    numer = np.empty((len(specs), exp_numer.shape[1]), dtype=float)
    denom = np.empty_like(numer)
    for idx, (fast, avert_slow, mix) in enumerate(specs):
        fast_idx = by_hl[fast]
        slow_idx = by_hl[avert_slow]
        numer[idx] = mix * exp_numer[fast_idx] + (1.0 - mix) * exp_numer[slow_idx]
        denom[idx] = mix * exp_denom[fast_idx] + (1.0 - mix) * exp_denom[slow_idx]
    configs = [
        {"fast_half_life": f, "slow_half_life": s, "fast_mix": m}
        for f, s, m in specs
    ]
    return numer, denom, configs


def select_kernel_family(
    family: str,
    numer: np.ndarray,
    denom: np.ndarray,
    configs: list[dict],
    centers: np.ndarray,
    obs: np.ndarray,
    exposure: np.ndarray,
    tune_mask: np.ndarray,
    validation_mask: np.ndarray,
    stress_mask: np.ndarray,
    mean_game_exposure: float,
) -> SelectedModel:
    best: SelectedModel | None = None
    for kernel_idx, kernel_config in enumerate(configs):
        for prior_games in PRIOR_GAME_EQUIVALENTS:
            prior_exposure = prior_games * mean_game_exposure
            effective_denom = denom[kernel_idx] + prior_exposure
            residual = np.divide(
                numer[kernel_idx],
                effective_denom,
                out=np.zeros_like(centers),
                where=effective_denom > 0,
            )
            pred = centers + residual
            tune_score = wmse(pred, obs, exposure, tune_mask)
            config = {**kernel_config, "prior_game_equivalents": prior_games}
            candidate = SelectedModel(
                family=family,
                config=json.dumps(config, sort_keys=True),
                pred=pred,
                tune_wmse=tune_score,
                validation_wmse=wmse(pred, obs, exposure, validation_mask),
                stress_wmse=wmse(pred, obs, exposure, stress_mask),
            )
            if best is None or candidate.tune_wmse < best.tune_wmse:
                best = candidate
    assert best is not None
    return best


def fit_centered_kalman(
    sub: pd.DataFrame,
    centers: np.ndarray,
    obs: np.ndarray,
    tune_mask: np.ndarray,
    validation_mask: np.ndarray,
    stress_mask: np.ndarray,
    maxiter: int,
    popsize: int,
) -> SelectedModel:
    layout = estimated.FilterLayout(sub)
    exposure = layout.exposure
    pre_validation = sub["season"].to_numpy() <= max(TUNE_SEASONS)
    residual = obs - centers
    base_var = float(
        np.sum(exposure[pre_validation] * residual[pre_validation] ** 2)
        / np.sum(exposure[pre_validation])
    )
    mean_exposure = float(exposure[pre_validation].mean())
    fit = estimated.tune_variant(
        layout=layout,
        obs_fit=residual,
        centers=centers,
        obs_raw=obs,
        mean_rate=0.0,
        base_var=base_var,
        mean_exposure=mean_exposure,
        dev_mask=tune_mask,
        maxiter=maxiter,
        popsize=popsize,
    )
    pred = fit["pred_raw"]
    config = json.dumps(
        dict(zip(estimated.PARAM_NAMES, fit["params"])), sort_keys=True
    )
    return SelectedModel(
        family="kalman_centered",
        config=config,
        pred=pred,
        tune_wmse=wmse(pred, obs, exposure, tune_mask),
        validation_wmse=wmse(pred, obs, exposure, validation_mask),
        stress_wmse=wmse(pred, obs, exposure, stress_mask),
    )


def clustered_interval(
    model: SelectedModel,
    baseline: SelectedModel,
    sub: pd.DataFrame,
    obs: np.ndarray,
    exposure: np.ndarray,
    mask: np.ndarray,
    seed_offset: int,
) -> tuple[float, float, float]:
    scored = sub.loc[mask, ["season", "game_id"]].copy()
    scored["cluster"] = scored["season"].astype(str) + ":" + scored["game_id"].astype(str)
    model_err = obs[mask] - model.pred[mask]
    base_err = obs[mask] - baseline.pred[mask]
    scored["delta_sse"] = exposure[mask] * (model_err**2 - base_err**2)
    scored["weight"] = exposure[mask]
    games = scored.groupby("cluster", sort=False)[["delta_sse", "weight"]].sum()
    point = float(games["delta_sse"].sum() / games["weight"].sum())
    values = games[["delta_sse", "weight"]].to_numpy()
    rng = np.random.default_rng(SEED + seed_offset)
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for draw in range(BOOTSTRAP_DRAWS):
        take = rng.integers(0, len(values), size=len(values))
        sample = values[take].sum(axis=0)
        draws[draw] = sample[0] / sample[1]
    low, high = np.quantile(draws, [0.025, 0.975])
    return point, float(low), float(high)


def config_short(model: SelectedModel) -> str:
    config = json.loads(model.config)
    if model.family == "exponential":
        return f"hl={config['half_life']:g},k={config['prior_game_equivalents']:g}g"
    if model.family == "no_decay":
        return f"k={config['prior_game_equivalents']:g}g"
    if model.family == "rolling_window":
        return f"win={config['window_days']}d,k={config['prior_game_equivalents']:g}g"
    if model.family == "offseason_exponential":
        return (
            f"hl={config['half_life']:g},off={config['offseason_retention']:.2f},"
            f"k={config['prior_game_equivalents']:g}g"
        )
    if model.family == "two_speed":
        return (
            f"fast={config['fast_half_life']:g},slow={config['slow_half_life']:g},"
            f"mix={config['fast_mix']:.2f},k={config['prior_game_equivalents']:g}g"
        )
    return "adaptive q_game/q_day/q_offseason"


def run_skill(
    panel: pd.DataFrame,
    spec: estimated.SkillSpec,
    maxiter: int,
    popsize: int,
    skill_offset: int,
) -> list[dict[str, object]]:
    sub = estimated.skill_subpanel(panel, spec)
    centers, _ = estimated.league_center(sub, spec)
    obs = sub["obs_rate"].to_numpy(dtype=float)
    exposure = sub["exposure"].to_numpy(dtype=float)
    seasons = sub["season"].to_numpy()
    has_prior = sub["step_k"].to_numpy() > 0
    tune_mask = np.isin(seasons, TUNE_SEASONS) & has_prior
    validation_mask = np.isin(seasons, VALIDATION_SEASONS) & has_prior
    stress_mask = np.isin(seasons, STRESS_SEASONS) & has_prior
    mean_game_exposure = float(exposure[tune_mask].mean())
    centered_contribution = exposure * (obs - centers)

    no_n, no_d, no_configs = history_no_decay(sub, centered_contribution)
    exp_n, exp_d, exp_configs = history_exponential(
        sub, centered_contribution, EXP_HALF_LIVES
    )
    rolling_n, rolling_d, rolling_configs = history_rolling(
        sub, centered_contribution
    )
    offseason_n, offseason_d, offseason_configs = history_exponential(
        sub,
        centered_contribution,
        OFFSEASON_HALF_LIVES,
        OFFSEASON_RETENTIONS,
    )
    two_n, two_d, two_configs = history_two_speed(exp_n, exp_d, exp_configs)

    families = [
        select_kernel_family(
            "no_decay", no_n, no_d, no_configs, centers, obs, exposure,
            tune_mask, validation_mask, stress_mask, mean_game_exposure,
        ),
        select_kernel_family(
            "exponential", exp_n, exp_d, exp_configs, centers, obs, exposure,
            tune_mask, validation_mask, stress_mask, mean_game_exposure,
        ),
        select_kernel_family(
            "rolling_window", rolling_n, rolling_d, rolling_configs, centers,
            obs, exposure, tune_mask, validation_mask, stress_mask,
            mean_game_exposure,
        ),
        select_kernel_family(
            "offseason_exponential", offseason_n, offseason_d,
            offseason_configs, centers, obs, exposure, tune_mask,
            validation_mask, stress_mask, mean_game_exposure,
        ),
        select_kernel_family(
            "two_speed", two_n, two_d, two_configs, centers, obs, exposure,
            tune_mask, validation_mask, stress_mask, mean_game_exposure,
        ),
        fit_centered_kalman(
            sub, centers, obs, tune_mask, validation_mask, stress_mask,
            maxiter, popsize,
        ),
    ]
    exponential = next(model for model in families if model.family == "exponential")
    output: list[dict[str, object]] = []
    for model_idx, model in enumerate(families):
        point, low, high = clustered_interval(
            model,
            exponential,
            sub,
            obs,
            exposure,
            validation_mask,
            seed_offset=100 * skill_offset + model_idx,
        )
        output.append(
            {
                "skill": spec.name,
                "family": model.family,
                "config": config_short(model),
                "tune_wmse": model.tune_wmse,
                "validation_wmse": model.validation_wmse,
                "stress_2026_wmse": model.stress_wmse,
                "validation_delta_vs_exponential_x1e6": point * 1e6,
                "validation_ci_low_x1e6": low * 1e6,
                "validation_ci_high_x1e6": high * 1e6,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skills", default=DEFAULT_SKILLS)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a small Kalman optimizer budget; kernel grids are unchanged.",
    )
    args = parser.parse_args()
    requested = [name.strip() for name in args.skills.split(",") if name.strip()]
    unknown = sorted(set(requested) - set(estimated.SKILL_NAMES))
    if unknown:
        raise SystemExit(f"unknown skills: {unknown}")
    specs = [spec for spec in estimated.SKILLS if spec.name in requested]
    maxiter, popsize = (5, 6) if args.quick else (20, 10)

    panel = estimated.load_panel()
    print(
        "PROTOTYPE — recency-kernel comparison; no production artifacts written\n"
        f"panel={len(panel):,} rows, players={panel['player_id'].nunique():,}; "
        f"tune={TUNE_SEASONS}; validation={VALIDATION_SEASONS}; "
        f"descriptive stress={STRESS_SEASONS}\n"
    )
    rows: list[dict[str, object]] = []
    for skill_idx, spec in enumerate(specs):
        print(f"fitting {spec.name} ...", flush=True)
        rows.extend(run_skill(panel, spec, maxiter, popsize, skill_idx))

    result = pd.DataFrame(rows)
    pd.set_option("display.max_colwidth", 80)
    pd.set_option("display.width", 220)
    for skill in requested:
        view = result[result["skill"] == skill].copy()
        view = view.sort_values("validation_wmse")
        print(f"\n{skill} (selected inside each family on 2020-2023 only)")
        print(
            view[
                [
                    "family",
                    "config",
                    "tune_wmse",
                    "validation_wmse",
                    "validation_delta_vs_exponential_x1e6",
                    "validation_ci_low_x1e6",
                    "validation_ci_high_x1e6",
                    "stress_2026_wmse",
                ]
            ].to_string(index=False, float_format=lambda value: f"{value:.8f}")
        )
    print(
        "\nInterpretation fence: 2024-2025 is the kernel-family comparison window. "
        "2026 was already viewed by the estimated-skills project and is descriptive "
        "stress evidence only; none of these results promotes a production model."
    )


if __name__ == "__main__":
    main()
