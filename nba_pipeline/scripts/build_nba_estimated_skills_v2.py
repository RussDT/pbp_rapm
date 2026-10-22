"""NBA estimated skills v2: mean-reverting (OU) two-component state.

Research challenger to ``build_nba_estimated_skills.py`` (v1). v1's state is a
pure random walk; diagnostics on v1's own dev residuals show it systematically
over-extrapolates on **all 17 skills** (slope of residual on state-minus-league-
mean is negative everywhere, implied per-game phi 0.914-0.985), and all four
``pct`` skills pin ``q_game`` at its lower bound -- DE straining toward "no
drift" because the random-walk family cannot express reversion.

v2 state per player is two components observed additively:

    obs = mu + delta + noise

- ``mu``   slow player baseline: random walk, per-DAY variance ``q_mu``, plus a
           season-crossing bump ``q_mu_off``.
- ``delta`` short-run form: mean-zero Ornstein-Uhlenbeck with continuous-time
           decay ``a = exp(-dt/tau)`` in DAYS and stationary variance ``q_d``.

This fixes three v1 defects at once: (1) form decays instead of accumulating
permanently into the level; (2) elapsed time is handled in days by one decay
constant, so the ~130-day offseason no longer double-counts through both
``q_day*days_gap`` and ``q_offseason`` (v1's ``days_gap`` is a raw player-level
date diff never reset at season boundaries); (3) genuine long-run change still
accumulates, in ``mu``.

Params per skill: ``(q_mu, q_mu_off, tau, q_d, r, v0_mu, v0_d)``.

Protocol is v1's verbatim: 2019 warms only, dev = 2020-2025, **2026 is a frozen
holdout** this script refuses to read (``--allow-holdout`` is required and is
not used during development). Bounds/means/constants come from pre-2026 rows.

Fences: descriptive/forecast research only; never a rating, never a solve prior;
no proprietary ratings as features, targets, or tuning inputs. Age/draft/size
covariates (stage B) come from repo-owned ``player_sheets/perc/*.csv``.

Run: ``.venv/bin/python nba_pipeline/scripts/build_nba_estimated_skills_v2.py``
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_nba_estimated_skills import (  # noqa: E402
    SKILLS, SkillSpec, load_panel, skill_subpanel, league_center,
    build_baselines, weighted_mse, FilterLayout, run_filter as run_filter_v1,
    OUT_DIR, WARMUP_SEASON, DEV_SEASONS, HOLDOUT_SEASON, SEED,
)

V2_PARAMS = ["tau_mu", "s_mu", "tau_d", "s_d", "q_off", "r"]


# --------------------------------------------------------------------------
# v2 engine: two-component (baseline + OU form) scalar-observation filter
# --------------------------------------------------------------------------


class LayoutV2(FilterLayout):
    """v1 layout plus elapsed-day handling for continuous-time OU decay."""

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__(df)
        # v1's days_gap is a raw player date-diff (never reset at seasons); v2
        # uses it directly as elapsed days, floored at 1, which is exactly what
        # makes the separate q_day term unnecessary.
        self.dt = np.maximum(self.days_gap, 1.0)


def run_filter_v2(
    layout: LayoutV2,
    obs: np.ndarray,
    params,
    mean_rate: float,
    mu0: np.ndarray | None = None,
    transient: np.ndarray | None = None,
    level_shift: np.ndarray | None = None,
    binom: bool = False,
):
    """Returns (pred, pred_var, post_level, post_var) per row.

    Both components are Ornstein-Uhlenbeck with a COMMON POPULATION ATTRACTOR:

        mu_t    = m_pop + exp(-dt/tau_mu) * (mu_{t-1} - m_pop) + w_mu
        delta_t =         exp(-dt/tau_d)  * delta_{t-1}        + w_d
        obs_t   = mu_t + delta_t + eps_t

    ``s_mu`` / ``s_d`` are STATIONARY variances, so the filter initializes at
    the stationary distribution (``mu = m_pop``, ``P_mm = s_mu``) rather than a
    free ``v0``. That makes ``s_mu`` literally the between-player talent
    variance and the update literally hierarchical shrinkage toward the
    population -- which is what v1 cannot express and what the residual
    diagnostic demands. Large ``tau_mu`` recovers v1's random walk, so v2
    nests v1 and DE can reject reversion if it does not help.

    ``mu0``          per-row debut baseline (stage B; defaults to ``m_pop``)
    ``transient``    per-row additive prediction offset NOT absorbed into state
    ``level_shift``  per-row additive shift applied to ``mu`` at season crossings
    """
    tau_mu, s_mu, tau_d, s_d, q_off, r = params
    n = layout.n_players
    mu = np.zeros(n); d = np.zeros(n)
    Pmm = np.zeros(n); Pmd = np.zeros(n); Pdd = np.zeros(n)

    pred = np.empty(layout.n_rows); pred_var = np.empty(layout.n_rows)
    post = np.empty(layout.n_rows); post_var = np.empty(layout.n_rows)

    for k, rows in enumerate(layout.step_rows):
        pl = layout.player_idx[rows]
        if k == 0:
            pm = (mean_rate * np.ones(rows.shape[0])) if mu0 is None else mu0[rows]
            pd_ = np.zeros(rows.shape[0])
            amm = np.full(rows.shape[0], s_mu)
            amd = np.zeros(rows.shape[0])
            add = np.full(rows.shape[0], s_d)
        else:
            dt = layout.dt[rows]
            sc = layout.season_cross[rows]
            am = np.exp(-dt / tau_mu)
            ad = np.exp(-dt / tau_d)
            anchor = mean_rate + (0.0 if level_shift is None else level_shift[rows] * sc)
            pm = anchor + am * (mu[pl] - anchor)
            pd_ = ad * d[pl]
            amm = am * am * Pmm[pl] + s_mu * (1.0 - am * am) + q_off * sc
            amd = am * ad * Pmd[pl]
            add = ad * ad * Pdd[pl] + s_d * (1.0 - ad * ad)

        base = pm + pd_
        p_row = base if transient is None else base + transient[rows]
        hph = amm + 2.0 * amd + add
        if binom:
            pc = np.clip(p_row, 0.02, 0.98)
            obs_var = r * pc * (1.0 - pc) / layout.exposure[rows]
        else:
            obs_var = r / layout.exposure[rows]
        S = hph + obs_var
        c0 = amm + amd
        c1 = amd + add
        Km = c0 / S
        Kd = c1 / S
        innov = obs[rows] - p_row

        mu[pl] = pm + Km * innov
        d[pl] = pd_ + Kd * innov
        Pmm[pl] = amm - Km * c0
        Pmd[pl] = amd - Km * c1
        Pdd[pl] = add - Kd * c1

        pred[rows] = p_row
        pred_var[rows] = hph
        post[rows] = mu[pl] + d[pl]
        post_var[rows] = Pmm[pl] + 2.0 * Pmd[pl] + Pdd[pl]
    return pred, pred_var, post, post_var


def bounds_v2(base_var: float, mean_exposure: float) -> list[tuple[float, float]]:
    """Wider than v1 deliberately: 8/85 of v1's selected params sat on a bound.
    tau_mu reaches 200 years so the random-walk limit stays inside the box."""
    r_scale = base_var * mean_exposure
    return [
        (5.0, 73000.0),                     # tau_mu (days; >=200y == random walk)
        (1e-9, 4.0 * base_var),             # s_mu   (between-player variance)
        (0.2, 400.0),                       # tau_d  (days)
        (1e-12, 2.0 * base_var),            # s_d    (form variance)
        (0.0, 2.0 * base_var),              # q_off
        (0.001 * r_scale, 200.0 * r_scale), # r
    ]


# --------------------------------------------------------------------------
# Stage A driver: v1 (random walk) vs v2 (OU) at an IDENTICAL DE budget
# --------------------------------------------------------------------------


from build_nba_estimated_skills import stat_bounds  # noqa: E402


def tune(engine, layout, obs_fit, centers, obs_raw, mean_rate, bounds,
         dev_mask, maxiter, popsize, binom=False):
    def objective(params):
        if engine == "v1":
            pred, *_ = run_filter_v1(layout, obs_fit, params, mean_rate)
        else:
            pred, *_ = run_filter_v2(layout, obs_fit, params, mean_rate, binom=binom)
        return weighted_mse(pred + centers, obs_raw, layout.exposure, dev_mask)

    t0 = time.time()
    res = differential_evolution(objective, bounds=bounds, seed=SEED,
                                 maxiter=maxiter, popsize=popsize, tol=1e-7,
                                 polish=False)
    params = [float(v) for v in res.x]
    if engine == "v1":
        pred, *_ = run_filter_v1(layout, obs_fit, params, mean_rate)
    else:
        pred, *_ = run_filter_v2(layout, obs_fit, params, mean_rate, binom=binom)
    return {"params": params, "pred": pred + centers, "obj": float(res.fun),
            "seconds": round(time.time() - t0, 1), "converged": bool(res.success),
            "nit": int(res.nit)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skills", default="")
    ap.add_argument("--maxiter", type=int, default=45)
    ap.add_argument("--popsize", type=int, default=12)
    ap.add_argument("--binom", action="store_true",
                    help="pct skills use binomial observation variance in v2")
    ap.add_argument("--allow-holdout", action="store_true",
                    help="REQUIRED to let 2026 rows enter at all; unused in dev")
    ap.add_argument("--out", default=str(OUT_DIR / "v2_stage_a.json"))
    args = ap.parse_args()

    specs = [s for s in SKILLS if not args.skills or s.name in args.skills.split(",")]
    df = load_panel()
    if not args.allow_holdout:
        df = df[df["season"] < HOLDOUT_SEASON].copy()   # 2026 physically removed
    print(f"panel: {len(df):,} rows, {df['player_id'].nunique()} players, "
          f"seasons {df.season.min()}-{df.season.max()}")

    report = {}
    print(f"\n{'skill':<19}{'v1 dev':>12}{'v2 dev':>12}{'impr %':>9}"
          f"{'seasons won':>13}{'tau_d':>8}{'sec':>7}")
    for spec in specs:
        sub = skill_subpanel(df, spec)
        obs_raw = sub["obs_rate"].to_numpy()
        seasons = sub["season"].to_numpy()
        has_prior = (sub["step_k"] > 0).to_numpy()
        dev_mask = np.isin(seasons, DEV_SEASONS) & has_prior
        pre = seasons < HOLDOUT_SEASON
        l1 = FilterLayout(sub)
        l2 = LayoutV2(sub)
        expo = l1.exposure
        mean_rate = float(sub.loc[pre, spec.numer].sum() / sub.loc[pre, "exposure"].sum())
        base_var = float(np.sum(expo[pre] * (obs_raw[pre] - mean_rate) ** 2) / np.sum(expo[pre]))
        mean_expo = float(expo[pre].mean())

        # inherit v1's centering choice so stage A isolates the state model
        prev = json.load(open(OUT_DIR / "hyperparams.json"))
        use_centered = prev[spec.name]["selected_variant"] == "centered"
        if use_centered:
            centers, _ = league_center(sub, spec)
            obs_fit = obs_raw - centers
            mr = 0.0
            bv = float(np.sum(expo[pre] * obs_fit[pre] ** 2) / np.sum(expo[pre]))
        else:
            centers = np.zeros(len(sub)); obs_fit = obs_raw; mr = mean_rate; bv = base_var

        binom = args.binom and spec.kind == "pct"
        a = tune("v1", l1, obs_fit, centers, obs_raw, mr, stat_bounds(bv, mean_expo),
                 dev_mask, args.maxiter, args.popsize)
        b = tune("v2", l2, obs_fit, centers, obs_raw, mr, bounds_v2(bv, mean_expo),
                 dev_mask, args.maxiter, args.popsize, binom=binom)

        per_season, won = {}, 0
        for s in DEV_SEASONS:
            m = (seasons == s) & has_prior
            e1 = weighted_mse(a["pred"], obs_raw, expo, m)
            e2 = weighted_mse(b["pred"], obs_raw, expo, m)
            per_season[int(s)] = {"v1": e1, "v2": e2, "impr_pct": 100 * (e1 - e2) / e1}
            won += int(e2 < e1)
        impr = 100 * (a["obj"] - b["obj"]) / a["obj"]
        report[spec.name] = {
            "centered": use_centered, "binom": binom,
            "v1": {k: a[k] for k in ("params", "obj", "seconds", "converged", "nit")},
            "v2": {k: b[k] for k in ("params", "obj", "seconds", "converged", "nit")},
            "v2_params_named": dict(zip(V2_PARAMS, b["params"])),
            "dev_impr_pct": impr, "seasons_won": won, "per_season": per_season,
        }
        print(f"{spec.name:<19}{a['obj']:>12.6g}{b['obj']:>12.6g}{impr:>9.3f}"
              f"{f'{won}/6':>13}{b['params'][2]:>8.1f}{a['seconds']+b['seconds']:>7.0f}")
        Path(args.out).write_text(json.dumps(report, indent=1))

    tot = sum(1 for v in report.values() if v["dev_impr_pct"] > 0)
    print(f"\nv2 beats v1 on dev WMSE for {tot}/{len(report)} skills")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
