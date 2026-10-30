"""v3: OU state + FORWARD-TARGET objective + prior-season corrections + variance rescale.

Three independent changes over v1, each measurable on its own:

1. STATE (from v2): both components Ornstein-Uhlenbeck sharing a common
   population attractor, stationary-variance parameterised. v1's random walk
   is nested at ``tau_mu -> inf``. Fixes the mean-reversion bias that all 17
   skills showed.

2. OBJECTIVE (new, and the largest effect): tune against the player's next-K
   game exposure-weighted rate rather than the next single game. Game-level
   WMSE is dominated by irreducible within-game noise -- a rim_pct row averages
   ~3.5 attempts -- so it barely discriminates between hyperparameters that
   differ a lot in what deployment actually consumes. The forward target costs
   the same to evaluate (precomputed per row by reverse cumsum).

3. CORRECTIONS (from stage B): debut prior from biographical covariates, age
   level shift at season crossings, early-season transient on the prediction
   only. Every table is fit on seasons strictly before the target season.

Plus a one-parameter variance rescale, also fit on prior seasons only, because
WMSE tuning leaves the absolute variance scale unidentified (only q/r is
identified by the gain) and v1's intervals cover 99-100% at nominal 68%.

No health/injury mean term: the within-season absence diagnostic is a null
(-0.003 SD out to 60 days), so elapsed time belongs in the variance, which the
OU decay already handles.

2026 is the frozen holdout and is dropped unless --allow-holdout is passed.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import differential_evolution

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_nba_estimated_skills import (
    SKILLS, load_panel, skill_subpanel, league_center, weighted_mse,
    OUT_DIR, DEV_SEASONS, HOLDOUT_SEASON, SEED)
from build_nba_estimated_skills_v2 import LayoutV2, run_filter_v2, bounds_v2, V2_PARAMS
from build_nba_estimated_skills_v2_stageb import build_corrections, AGE_PATH
from eval_estimated_skills_forward import forward_cells, wcorr


def forward_target(sub, spec, K):
    """Per-row exposure-weighted rate over that player's next K in-season games.

    K=None means rest-of-season. Returns (target, weight); weight is the
    forward exposure so a 2-attempt tail does not outvote a 200-attempt one.
    """
    n = sub[spec.numer].to_numpy(float); e = sub["exposure"].to_numpy(float)
    tn = np.zeros(len(sub)); td = np.zeros(len(sub))
    for _, ii in sub.groupby(["player_id", "season"], sort=False).indices.items():
        ii = np.sort(ii); m = len(ii)
        cn = np.concatenate([np.cumsum(n[ii]), [np.nan]])
        ce = np.concatenate([np.cumsum(e[ii]), [np.nan]])
        hi = (m - 1) if K is None else np.minimum(np.arange(m) + K, m - 1)
        tn[ii] = cn[hi] - cn[np.arange(m)]
        td[ii] = ce[hi] - ce[np.arange(m)]
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(td > 0, tn / td, np.nan)
    return t, np.where(np.isfinite(t), td, 0.0)


def variance_scale(pred, pred_var, obs, expo, spec, mask):
    """One parameter c so that sd((obs-pred)/sqrt(c*pred_var + obs_var)) = 1.

    Closed form: c solves E[resid^2/(c*pv+ov)] = 1, found by bisection because
    the expectation is monotone decreasing in c.
    """
    r2 = (obs[mask] - pred[mask]) ** 2
    pv = np.maximum(pred_var[mask], 1e-18)
    ov = np.zeros_like(pv)  # observation noise already inside pred_var here
    lo, hi = 1e-4, 1e6
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        if np.mean(r2 / (mid * pv + ov)) > 1.0:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills", default="")
    ap.add_argument("--maxiter", type=int, default=60)
    ap.add_argument("--popsize", type=int, default=15)
    ap.add_argument("--fwd-k", type=int, default=41, help="0 = rest of season")
    ap.add_argument("--no-corrections", action="store_true")
    ap.add_argument("--allow-holdout", action="store_true")
    ap.add_argument("--stage-a", default=str(OUT_DIR / "v2_stage_a.json"))
    ap.add_argument("--out", default=str(OUT_DIR / "v3.json"))
    args = ap.parse_args()
    K = None if args.fwd_k == 0 else args.fwd_k

    stage_a = json.load(open(args.stage_a)) if Path(args.stage_a).exists() else {}
    specs = [s for s in SKILLS if not args.skills or s.name in args.skills.split(",")]
    df = load_panel()
    if not args.allow_holdout:
        df = df[df["season"] < HOLDOUT_SEASON].copy()
    age = pd.read_parquet(AGE_PATH)
    seasons_all = sorted(df["season"].unique())

    report = {}
    print(f"{'skill':<19}{'fwdMSE v1':>12}{'fwdMSE v3':>12}{'gain %':>9}"
          f"{'r v1':>8}{'r v3':>8}{'dates':>8}{'sd(z)':>8}{'sec':>7}")
    for spec in specs:
        sub = skill_subpanel(df, spec)
        obs_raw = sub["obs_rate"].to_numpy(); seasons = sub["season"].to_numpy()
        dev_mask = np.isin(seasons, DEV_SEASONS) & (sub["step_k"] > 0).to_numpy()
        pre = seasons < HOLDOUT_SEASON
        lay = LayoutV2(sub); expo = lay.exposure
        mean_rate = float(sub.loc[pre, spec.numer].sum() / sub.loc[pre, "exposure"].sum())
        centered = stage_a.get(spec.name, {}).get("centered", True)
        if centered:
            centers, _ = league_center(sub, spec)
            obs_fit = obs_raw - centers; mr = 0.0
            bv = float(np.sum(expo[pre] * obs_fit[pre] ** 2) / np.sum(expo[pre]))
        else:
            centers = np.zeros(len(sub)); obs_fit = obs_raw; mr = mean_rate
            bv = float(np.sum(expo[pre] * (obs_raw[pre] - mean_rate) ** 2) / np.sum(expo[pre]))

        if args.no_corrections:
            mu0 = shift = trans = None; meta = {}
        else:
            mu0_raw, shift, trans, meta = build_corrections(
                sub, spec, age, mean_rate, seasons_all)
            mu0 = mu0_raw - centers

        ft, fw = forward_target(sub, spec, K)
        fmask = dev_mask & np.isfinite(ft) & (fw > 0)
        min_expo = 20.0 if spec.kind == "pct" else 200.0

        def run(p):
            return run_filter_v2(lay, obs_fit, p, mr, mu0=mu0,
                                 transient=trans, level_shift=shift)

        def obj(p):
            pred = run(p)[0] + centers
            return float(np.sum(fw[fmask] * (pred[fmask] - ft[fmask]) ** 2)
                         / np.sum(fw[fmask]))

        t0 = time.time()
        res = differential_evolution(obj, bounds=bounds_v2(bv, float(expo[pre].mean())),
                                     seed=SEED, maxiter=args.maxiter,
                                     popsize=args.popsize, tol=1e-7, polish=False)
        secs = time.time() - t0
        p = [float(v) for v in res.x]
        pred, pvar, post, postvar = run(p)
        pred = pred + centers; post = post + centers

        cscale = variance_scale(pred, pvar, obs_raw, expo, spec, dev_mask)
        z = (obs_raw - pred) / np.sqrt(np.maximum(cscale * pvar, 1e-18))
        sdz = float(np.std(z[dev_mask]))

        cells = forward_cells(sub, spec, post, min_expo)
        fmse = float(np.average((cells.state - cells.target) ** 2, weights=cells.w))
        fr = wcorr(cells.state.values, cells.target.values, cells.w.values)

        # v1 reference on the identical cell set
        v1p = stage_a.get(spec.name, {}).get("v1", {}).get("params")
        if v1p:
            from build_nba_estimated_skills import run_filter, FilterLayout
            l1 = FilterLayout(sub)
            _, _, p1, _ = run_filter(l1, obs_fit, v1p, mr)
            c1 = forward_cells(sub, spec, p1 + centers, min_expo)
            m = c1.merge(cells, on=["season", "asof", "player_id"], suffixes=("_1", "_3"))
            f1 = float(np.average((m.state_1 - m.target_1) ** 2, weights=m.w_1))
            f3 = float(np.average((m.state_3 - m.target_1) ** 2, weights=m.w_1))
            r1 = wcorr(m.state_1.values, m.target_1.values, m.w_1.values)
            won = tot = 0
            for _, g in m.groupby(["season", "asof"]):
                e1 = np.average((g.state_1 - g.target_1) ** 2, weights=g.w_1)
                e3 = np.average((g.state_3 - g.target_1) ** 2, weights=g.w_1)
                tot += 1; won += int(e3 < e1)
            gain = (f1 - f3) / f1 * 100.0
        else:
            f1 = r1 = np.nan; gain = np.nan; won = tot = 0

        report[spec.name] = dict(
            params=p, params_named=dict(zip(V2_PARAMS, p)), centered=centered,
            fwd_k=args.fwd_k, corrections=not args.no_corrections,
            obj=float(res.fun), seconds=round(secs, 1), converged=bool(res.success),
            fwd_mse_v1=f1, fwd_mse_v3=fmse, fwd_gain_pct=gain,
            wcorr_v1=r1, wcorr_v3=fr, dates_won=won, dates_total=tot,
            game_mse=weighted_mse(pred, obs_raw, expo, dev_mask),
            var_scale=cscale, sd_z=sdz, correction_meta=meta)
        print(f"{spec.name:<19}{f1:>12.6g}{fmse:>12.6g}{gain:>8.2f}%"
              f"{r1:>8.4f}{fr:>8.4f}{f'{won}/{tot}':>8}{sdz:>8.3f}{secs:>7.1f}")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
