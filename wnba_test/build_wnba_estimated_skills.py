"""WNBA estimated skills: zone shooting + volume Kalman panel with DE tuning.

Extends the box-panel architecture (``build_wnba_kalman_box.py``) to
skill-space targets with a generalized exposure column:

- Percentage skills (exposure = attempts, obs var ``r / attempts``):
  ``rim_pct``, ``mid_pct`` (PBP zone lines), ``three_pct``, ``ft_pct`` (box).
  Each is fit twice — raw, and centered on the as-of league mean (prior
  season totals pooled with league season-to-date strictly before the game
  date) — and the variant with the lower development WMSE is selected.
- Volume skills (exposure = minutes, obs var ``r / minutes``):
  ``rim_freq``, ``mid_freq`` (PBP, coverage-gated), ``three_freq``,
  ``fta_rate`` (box). Uncentered, matching the box panel.

Evaluation discipline is inherited verbatim from the box panel: one-step-ahead
exposure-weighted squared error in raw rate space; eligible = season 2023-2025
with >= 1 prior observation of that skill; 2022 warms the filters only; DE
bounds/means/centers-fallback use pre-2026 rows only; **2026 is a
frozen-parameter holdout** never touched by the objective, the bounds, or
variant selection. Baselines (trailing-10 / season-to-date with career
fallback / career-to-date) are exposure-weighted ratios on the identical
eligible sets.

Inputs: ``Results/gameday/player_game_logs.csv`` plus
``Results/gameday/estimated_skills/player_game_zone_lines.csv``
(``build_wnba_zone_game_lines.py``). Outputs under
``Results/gameday/estimated_skills/``: hyperparams.json,
predictions.parquet, current_states.csv, eval_report.json. Generated
artifacts stay out of git. The panel is a props/forecast-lane input, never a
rating prior; see WNBA_ESTIMATED_SKILLS.md.

Run: ``.venv/bin/python wnba_test/build_wnba_estimated_skills.py [--quick] [--skills rim_pct,...]``
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

import build_wnba_pbp_shot_quality as sq
import wnba_gameday_common as common

OUT_DIR = common.GAMEDAY_DIR / "estimated_skills"
ZONE_LINES_PATH = OUT_DIR / "player_game_zone_lines.csv"

PARAM_NAMES = ["q_game", "q_day", "q_offseason", "r", "v0"]
DEV_SEASONS = (2023, 2024, 2025)
HOLDOUT_SEASON = 2026
SEED = 20260826


@dataclass(frozen=True)
class SkillSpec:
    name: str
    numer: str           # per-game numerator column
    denom: str           # per-game exposure column ("minutes" or an attempts col)
    kind: str            # "pct" (attempts exposure) or "freq" (minutes exposure)
    needs_pbp: bool      # requires PBP zone coverage for the game


SKILLS: list[SkillSpec] = [
    SkillSpec("rim_pct", "rim_fgm", "rim_fga", "pct", True),
    SkillSpec("mid_pct", "mid_fgm", "mid_fga", "pct", True),
    SkillSpec("three_pct", "fg3m", "fg3a", "pct", False),
    SkillSpec("ft_pct", "ftm", "fta", "pct", False),
    SkillSpec("rim_freq", "rim_fga", "minutes", "freq", True),
    SkillSpec("mid_freq", "mid_fga", "minutes", "freq", True),
    SkillSpec("three_freq", "fg3a", "minutes", "freq", False),
    SkillSpec("fta_rate", "fta", "minutes", "freq", False),
]
SKILL_NAMES = [s.name for s in SKILLS]


# --------------------------------------------------------------------------
# Panel assembly
# --------------------------------------------------------------------------


def load_panel() -> pd.DataFrame:
    """Played game-log rows joined with PBP zone lines and coverage flag."""
    df = pd.read_csv(common.GAME_LOG_PATH)
    df = df[(df["played"] == 1) & (df["minutes"] > 0)].copy()
    df["game_id"] = sq.normalize_game_id(df["game_id"])
    df["game_date"] = pd.to_datetime(df["game_date"])

    lines = pd.read_csv(ZONE_LINES_PATH, dtype={"game_id": str})
    lines["game_id"] = sq.normalize_game_id(lines["game_id"])
    zone_cols = ["rim_fga", "rim_fgm", "mid_fga", "mid_fgm"]
    df = df.merge(
        lines[["game_id", "person_id", *zone_cols]],
        on=["game_id", "person_id"],
        how="left",
    )
    covered = set(lines["game_id"])
    df["pbp_covered"] = df["game_id"].isin(covered)
    for col in zone_cols:
        df[col] = np.where(
            df["pbp_covered"], df[col].fillna(0.0), np.nan
        )
    return df.sort_values(
        ["person_id", "game_date", "game_id"], kind="mergesort"
    ).reset_index(drop=True)


def skill_subpanel(df: pd.DataFrame, spec: SkillSpec) -> pd.DataFrame:
    """Rows with a defined observation for this skill, with per-skill gaps."""
    if spec.kind == "pct":
        mask = df[spec.denom].notna() & (df[spec.denom] > 0)
    else:
        mask = df[spec.numer].notna() & (df["minutes"] > 0)
        if spec.needs_pbp:
            mask &= df["pbp_covered"]
    sub = df.loc[mask].copy()
    sub["exposure"] = sub[spec.denom].astype(float)
    sub["obs_rate"] = sub[spec.numer].astype(float) / sub["exposure"]
    sub["player_idx"] = pd.factorize(sub["person_id"])[0]
    grp = sub.groupby("player_idx", sort=False)
    sub["step_k"] = grp.cumcount()
    sub["days_gap"] = grp["game_date"].diff().dt.days.fillna(0).astype(float)
    sub["season_cross"] = (grp["season"].diff().fillna(0) > 0).astype(float)
    return sub.reset_index(drop=True)


def league_center(sub: pd.DataFrame, spec: SkillSpec) -> tuple[np.ndarray, dict]:
    """As-of league mean per row: prior-season totals + season-to-date (< date).

    Fallback (first dates of the first season only) is that season's full
    league mean; those rows are warm-up-only and flagged in the meta dict.
    """
    daily = (
        sub.groupby(["season", "game_date"])
        .agg(numer=(spec.numer, "sum"), denom=("exposure", "sum"))
        .reset_index()
        .sort_values(["season", "game_date"])
    )
    grp = daily.groupby("season")
    daily["numer_std"] = grp["numer"].cumsum() - daily["numer"]
    daily["denom_std"] = grp["denom"].cumsum() - daily["denom"]
    season_totals = (
        sub.groupby("season").agg(numer=(spec.numer, "sum"), denom=("exposure", "sum"))
    )
    daily["numer_prev"] = daily["season"].map(season_totals["numer"].shift())
    daily["denom_prev"] = daily["season"].map(season_totals["denom"].shift())
    daily[["numer_prev", "denom_prev"]] = daily[["numer_prev", "denom_prev"]].fillna(0.0)
    pool_n = daily["numer_std"] + daily["numer_prev"]
    pool_d = daily["denom_std"] + daily["denom_prev"]
    first_season = int(daily["season"].min())
    fallback = float(
        season_totals.loc[first_season, "numer"] / season_totals.loc[first_season, "denom"]
    )
    daily["center"] = np.where(pool_d > 0, pool_n / pool_d.replace(0, np.nan), fallback)
    n_fallback = int((pool_d <= 0).sum())
    centers = sub[["season", "game_date"]].merge(
        daily[["season", "game_date", "center"]], on=["season", "game_date"], how="left"
    )["center"].to_numpy()
    assert not np.isnan(centers).any()
    meta = {"n_fallback_dates": n_fallback, "fallback_value": fallback}
    return centers, meta


# --------------------------------------------------------------------------
# Generalized-exposure Kalman engine (box-panel engine with exposure input)
# --------------------------------------------------------------------------


class FilterLayout:
    """Step-parallel layout over a skill sub-panel (see box panel docstring)."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.n_rows = len(df)
        self.n_players = int(df["player_idx"].max()) + 1
        self.player_idx = df["player_idx"].to_numpy()
        self.exposure = df["exposure"].to_numpy(dtype=float)
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (pred_rate, pred_var, post_rate, post_var) per row."""
    q_game, q_day, q_off, r, v0 = params
    theta = np.empty(layout.n_players)
    var = np.empty(layout.n_players)
    pred = np.empty(layout.n_rows)
    pred_var = np.empty(layout.n_rows)
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
        obs_var = r / layout.exposure[rows]
        gain = prior_v / (prior_v + obs_var)
        post_t = prior_t + gain * (obs[rows] - prior_t)
        post_v = (1.0 - gain) * prior_v
        theta[players] = post_t
        var[players] = post_v
        pred[rows] = prior_t
        pred_var[rows] = prior_v
        post[rows] = post_t
        post_var_out[rows] = post_v
    return pred, pred_var, post, post_var_out


def weighted_mse(
    pred: np.ndarray, obs: np.ndarray, weights: np.ndarray, mask: np.ndarray
) -> float:
    diff = obs[mask] - pred[mask]
    w = weights[mask]
    return float(np.sum(w * diff * diff) / np.sum(w))


def stat_bounds(base_var: float, mean_exposure: float) -> list[tuple[float, float]]:
    """Bounds in PARAM_NAMES order, anchored to pre-2026 scale (box-panel rule)."""
    r_scale = base_var * mean_exposure
    return [
        (1e-9, 0.25 * base_var),            # q_game
        (0.0, 0.01 * base_var),             # q_day
        (0.0, 1.0 * base_var),              # q_offseason
        (0.01 * r_scale, 30.0 * r_scale),   # r
        (0.005 * base_var, 4.0 * base_var), # v0
    ]


# --------------------------------------------------------------------------
# Baselines (exposure-weighted ratios; identical eligible sets)
# --------------------------------------------------------------------------


def build_baselines(sub: pd.DataFrame, spec: SkillSpec) -> dict[str, np.ndarray]:
    pid = sub["player_idx"]
    numer = sub[spec.numer].astype(float)
    expo = sub["exposure"].astype(float)
    sh_numer = numer.groupby(pid).shift()
    sh_expo = expo.groupby(pid).shift()
    trail_n = sh_numer.groupby(pid).rolling(10, min_periods=1).sum().reset_index(level=0, drop=True)
    trail_d = sh_expo.groupby(pid).rolling(10, min_periods=1).sum().reset_index(level=0, drop=True)
    career_n = numer.groupby(pid).cumsum() - numer
    career_d = expo.groupby(pid).cumsum() - expo
    season_n = numer.groupby([pid, sub["season"]]).cumsum() - numer
    season_d = expo.groupby([pid, sub["season"]]).cumsum() - expo
    career = (career_n / career_d.replace(0.0, np.nan)).to_numpy()
    season = (season_n / season_d.replace(0.0, np.nan)).to_numpy()
    season = np.where(np.isnan(season), career, season)
    return {
        "trail10": (trail_n / trail_d.replace(0.0, np.nan)).to_numpy(),
        "season": season,
        "career": career,
    }


# --------------------------------------------------------------------------
# Per-skill tuning
# --------------------------------------------------------------------------


def tune_variant(
    layout: FilterLayout,
    obs_fit: np.ndarray,
    centers: np.ndarray,
    obs_raw: np.ndarray,
    mean_rate: float,
    base_var: float,
    mean_exposure: float,
    dev_mask: np.ndarray,
    maxiter: int,
    popsize: int,
) -> dict:
    """DE-tune one (skill, centering) variant; metrics in raw rate space."""
    bounds = stat_bounds(base_var, mean_exposure)

    def objective(params: np.ndarray) -> float:
        pred, _, _, _ = run_filter(layout, obs_fit, params, mean_rate)
        return weighted_mse(pred + centers, obs_raw, layout.exposure, dev_mask)

    start = time.time()
    result = differential_evolution(
        objective, bounds=bounds, seed=SEED, maxiter=maxiter, popsize=popsize,
        tol=1e-7, polish=False,
    )
    elapsed = time.time() - start
    params = [float(v) for v in result.x]
    pred, pred_var, post, post_var = run_filter(layout, obs_fit, params, mean_rate)
    return {
        "params": params,
        "bounds": bounds,
        "dev_objective": float(result.fun),
        "de": {
            "success": bool(result.success),
            "message": str(result.message),
            "nit": int(result.nit),
            "nfev": int(result.nfev),
            "seconds": round(elapsed, 2),
        },
        "pred_raw": pred + centers,
        "pred_var": pred_var,
        "post_raw": post + centers,
        "post_var": post_var,
        "post_centered": post,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="DE maxiter 5, popsize 6")
    parser.add_argument("--skills", type=str, default=None,
                        help="comma-separated subset (default: all 8)")
    args = parser.parse_args()
    names = SKILL_NAMES if args.skills is None else [
        s.strip() for s in args.skills.split(",") if s.strip()
    ]
    unknown = [n for n in names if n not in SKILL_NAMES]
    if unknown:
        raise SystemExit(f"unknown skills {unknown}; choose from {SKILL_NAMES}")
    specs = [s for s in SKILLS if s.name in names]
    maxiter, popsize = (5, 6) if args.quick else (30, 12)

    df = load_panel()
    print(f"panel: {len(df)} played rows, {df['person_id'].nunique()} players")

    hyperparams: dict[str, dict] = {}
    eval_report: dict[str, dict] = {}
    pred_frames: list[pd.DataFrame] = []
    state_rows: dict[str, pd.DataFrame] = {}
    de_total = 0.0

    for spec in specs:
        sub = skill_subpanel(df, spec)
        layout = FilterLayout(sub)
        obs_raw = sub["obs_rate"].to_numpy()
        seasons = sub["season"].to_numpy()
        has_prior = (sub["step_k"] > 0).to_numpy()
        dev_mask = np.isin(seasons, DEV_SEASONS) & has_prior
        holdout_mask = (seasons == HOLDOUT_SEASON) & has_prior
        pre2026 = seasons < HOLDOUT_SEASON

        expo_pre = layout.exposure[pre2026]
        mean_rate = float(
            sub.loc[pre2026, spec.numer].sum() / sub.loc[pre2026, "exposure"].sum()
        )
        base_var = float(
            np.sum(expo_pre * (obs_raw[pre2026] - mean_rate) ** 2) / np.sum(expo_pre)
        )
        mean_exposure_pre = float(expo_pre.mean())

        variants: dict[str, dict] = {}
        zeros = np.zeros(len(sub))
        variants["uncentered"] = tune_variant(
            layout, obs_raw, zeros, obs_raw, mean_rate, base_var,
            mean_exposure_pre, dev_mask, maxiter, popsize,
        )
        center_meta: dict = {}
        if spec.kind == "pct":
            centers, center_meta = league_center(sub, spec)
            obs_centered = obs_raw - centers
            base_var_c = float(
                np.sum(expo_pre * obs_centered[pre2026] ** 2) / np.sum(expo_pre)
            )
            variants["centered"] = tune_variant(
                layout, obs_centered, centers, obs_raw, 0.0, base_var_c,
                mean_exposure_pre, dev_mask, maxiter, popsize,
            )
        selected = min(variants, key=lambda k: variants[k]["dev_objective"])
        chosen = variants[selected]
        de_total += sum(v["de"]["seconds"] for v in variants.values())

        baselines = build_baselines(sub, spec)
        skill_eval: dict[str, dict] = {"selected_variant": selected}
        for window, mask in (("dev", dev_mask), ("holdout_2026", holdout_mask)):
            entry: dict[str, object] = {"n_obs": int(mask.sum())}
            for vname, v in variants.items():
                entry[f"kalman_{vname}_wmse"] = weighted_mse(
                    v["pred_raw"], obs_raw, layout.exposure, mask
                )
            kalman_mse = entry[f"kalman_{selected}_wmse"]
            for bname, bpred in baselines.items():
                assert not np.isnan(bpred[mask]).any(), (
                    f"{spec.name} {bname} baseline NaN on eligible rows"
                )
                bmse = weighted_mse(bpred, obs_raw, layout.exposure, mask)
                entry[f"{bname}_wmse"] = bmse
                entry[f"pct_impr_vs_{bname}"] = 100.0 * (bmse - kalman_mse) / bmse
            skill_eval[window] = entry
        eval_report[spec.name] = skill_eval

        hyperparams[spec.name] = {
            "kind": spec.kind,
            "numer": spec.numer,
            "denom": spec.denom,
            "selected_variant": selected,
            "league_mean_rate_pre2026": mean_rate,
            "base_var_pre2026": base_var,
            "mean_exposure_pre2026": mean_exposure_pre,
            "center_meta": center_meta,
            "variants": {
                vname: {
                    "params": dict(zip(PARAM_NAMES, v["params"])),
                    "bounds": dict(zip(PARAM_NAMES, [list(b) for b in v["bounds"]])),
                    "dev_objective": v["dev_objective"],
                    "de": {**v["de"], "maxiter": maxiter, "popsize": popsize},
                }
                for vname, v in variants.items()
            },
        }

        pred_frames.append(
            pd.DataFrame(
                {
                    "person_id": sub["person_id"].to_numpy(),
                    "game_id": sub["game_id"].to_numpy(),
                    "game_date": sub["game_date"].to_numpy(),
                    "season": seasons,
                    "skill": spec.name,
                    "variant": selected,
                    "exposure": layout.exposure,
                    "obs_rate": obs_raw,
                    "pred_rate": chosen["pred_raw"],
                    "pred_var": chosen["pred_var"],
                    "post_rate": chosen["post_raw"],
                    "post_var": chosen["post_var"],
                }
            )
        )

        last_idx = sub.groupby("player_idx")["step_k"].idxmax()
        st = sub.loc[last_idx, ["person_id", "player_name", "game_date", "season"]].copy()
        st[f"p_{spec.name}"] = chosen["post_raw"][last_idx.to_numpy()]
        st[f"p_{spec.name}_var"] = chosen["post_var"][last_idx.to_numpy()]
        st[f"{spec.name}_n_obs"] = sub.groupby("player_idx")["step_k"].max().to_numpy() + 1
        st[f"{spec.name}_exposure"] = sub.groupby("player_idx")["exposure"].sum().to_numpy()
        state_rows[spec.name] = st.rename(
            columns={"game_date": f"{spec.name}_last_date", "season": f"{spec.name}_last_season"}
        )

        d = skill_eval["dev"]
        h = skill_eval["holdout_2026"]
        print(
            f"{spec.name:>10} [{selected:>10}]: dev vs t10 {d['pct_impr_vs_trail10']:+.2f}% "
            f"vs season {d['pct_impr_vs_season']:+.2f}% | "
            f"2026 vs t10 {h['pct_impr_vs_trail10']:+.2f}% "
            f"vs season {h['pct_impr_vs_season']:+.2f}%"
        )

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "hyperparams.json").write_text(json.dumps(hyperparams, indent=2))
    predictions = pd.concat(pred_frames, ignore_index=True)
    predictions.to_parquet(OUT_DIR / "predictions.parquet", index=False)

    states: pd.DataFrame | None = None
    for name, st in state_rows.items():
        st = st.drop(columns=["player_name"]) if states is not None else st
        states = st if states is None else states.merge(st, on="person_id", how="outer")
    for spec in specs:
        if spec.kind == "freq" and f"p_{spec.name}" in states:
            states[f"p_{spec.name}_per40"] = states[f"p_{spec.name}"] * 40.0
    name_map = df.groupby("person_id")["player_name"].last()
    states["player_name"] = states["person_id"].map(name_map)
    states = states.sort_values("person_id").reset_index(drop=True)
    states.to_csv(OUT_DIR / "current_states.csv", index=False)

    summary = {
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "quick": bool(args.quick),
        "skills": [s.name for s in specs],
        "n_panel_rows": int(len(df)),
        "de_total_seconds": round(de_total, 1),
        "objective": (
            "one-step-ahead exposure-weighted squared error in raw rate space; "
            "pct skills weighted by attempts, freq skills by minutes; eligible "
            "= season 2023-2025 with >= 1 prior skill observation; 2022 warms "
            "only; 2026 is frozen-parameter holdout; centering variant chosen "
            "on dev WMSE only"
        ),
    }
    (OUT_DIR / "eval_report.json").write_text(
        json.dumps({"summary": summary, "per_skill": eval_report}, indent=2)
    )
    print(
        f"\nwrote {OUT_DIR}/(hyperparams.json, predictions.parquet, "
        f"current_states.csv, eval_report.json); DE total {de_total:.1f}s"
    )


if __name__ == "__main__":
    main()
