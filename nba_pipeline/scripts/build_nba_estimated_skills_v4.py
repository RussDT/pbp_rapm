"""v4 next-game candidate: OU state + prior-season corrections + next-game context.

Target (owner decision 2026-09-13): one-step-ahead exposure-weighted game MSE on
dev 2020-2025. Forward / rest-of-season objectives (v3) are not used here.

Pipeline per skill:
1. base filter = stage-A OU params, no corrections -> one-step residuals that feed
   the date-safe context terms (see ``estimated_skills_context``);
2. context design: home/back-to-back/3+ rest days, running opponent residual,
   previous-season opponent residual, running own-team residual, and for
   ``kind == "rate"`` skills pre-tip teammate absence: teammates missing from the
   team's most recent completed game, weighted by a prior-seasons
   estimate of the chance each one sits tonight given the absence streak (tonight's roster
   is never read; see the fairness rule in ``estimated_skills_context``).
   Coefficients are expanding-window (seasons < S);
3. shrinkage ``c`` is picked on dev with the base filter (a hyperparameter like
   the DE params);
4. stage-B corrections (debut / age / early-season) as prior-season tables;
5. DE re-tunes the six OU params plus a context multiplier ``lam`` on next-game
   WMSE, with ``transient = early_season + lam * context`` so context is both
   added to the prediction and removed from the innovation.

Variance: one scale ``k`` on the filter's own predictive variance
``pred_var + r/exposure`` (WMSE leaves the absolute scale unidentified), fit
expanding on seasons < S, reported as per-season sd(z).
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
import estimated_skills_context as X

CTX_PATH = OUT_DIR / "game_context.parquet"
C_GRID = (2.0, 5.0, 10.0, 20.0, 40.0)


def game_bootstrap(err_a, err_b, w, games, mask, reps, rng):
    """Pooled % gain of b over a with a game-clustered bootstrap 95% CI."""
    codes, _ = pd.factorize(games[mask])
    sa = np.bincount(codes, w[mask] * err_a[mask] ** 2)
    sb = np.bincount(codes, w[mask] * err_b[mask] ** 2)
    gain = 100 * (sa.sum() - sb.sum()) / sa.sum()
    draws = np.empty(reps)
    for i in range(reps):
        cnt = np.bincount(rng.integers(0, len(sa), len(sa)), minlength=len(sa))
        draws[i] = 100 * (cnt @ sa - cnt @ sb) / (cnt @ sa)
    return float(gain), [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def variance_scale(r2, v):
    """k such that mean(r2 / (k v)) = 1 (closed form)."""
    return float(np.mean(r2 / np.maximum(v, 1e-18)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills", default="")
    ap.add_argument("--maxiter", type=int, default=60)
    ap.add_argument("--popsize", type=int, default=15)
    ap.add_argument("--no-corrections", action="store_true")
    ap.add_argument("--no-context", action="store_true")
    ap.add_argument("--no-absence", action="store_true")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--allow-holdout", action="store_true")
    ap.add_argument("--stage-a", default=str(OUT_DIR / "v2_stage_a.json"))
    ap.add_argument("--stage-b", default=str(OUT_DIR / "v2_stage_b.json"))
    ap.add_argument("--out", default=str(OUT_DIR / "v4_nextgame.json"))
    args = ap.parse_args()

    stage_a = json.load(open(args.stage_a))
    specs = [s for s in SKILLS if not args.skills or s.name in args.skills.split(",")]
    df = load_panel()
    if not args.allow_holdout:
        df = df[df["season"] < HOLDOUT_SEASON].copy()
    df["game_id"] = df["game_id"].astype(str).str.zfill(10)
    ctx_tab = pd.read_parquet(CTX_PATH)
    df = df.merge(ctx_tab, on=["game_id", "player_id"], how="left")
    age = pd.read_parquet(AGE_PATH)
    seasons_all = sorted(df["season"].unique())
    absent = None if (args.no_context or args.no_absence) else X.absent_teammates(df, ctx_tab)
    rng = np.random.default_rng(SEED)

    report = {}
    print(f"{'skill':<19}{'c':>4}{'lam':>6}{'vs A %':>9}{'CI':>18}{'won':>5}"
          f"{'vs B %':>9}{'won':>5}{'vs v1 %':>9}{'sd(z)':>7}{'sec':>6}", flush=True)
    for spec in specs:
        sub = skill_subpanel(df, spec)
        obs = sub["obs_rate"].to_numpy(); se = sub["season"].to_numpy()
        hp = (sub["step_k"] > 0).to_numpy(); dev = np.isin(se, DEV_SEASONS) & hp
        pre = se < HOLDOUT_SEASON
        lay = LayoutV2(sub); expo = lay.exposure
        mean_rate = float(sub.loc[pre, spec.numer].sum() / sub.loc[pre, "exposure"].sum())
        sa = stage_a[spec.name]
        cen_asof, _ = league_center(sub, spec)
        if sa["centered"]:
            centers = cen_asof; obs_fit = obs - centers; mr = 0.0
            bv = float(np.sum(expo[pre] * obs_fit[pre] ** 2) / np.sum(expo[pre]))
        else:
            centers = np.zeros(len(sub)); obs_fit = obs; mr = mean_rate
            bv = float(np.sum(expo[pre] * (obs[pre] - mean_rate) ** 2) / np.sum(expo[pre]))
        t0 = time.time()

        pA, *_ = run_filter_v2(lay, obs_fit, sa["v2"]["params"], mr); pA = pA + centers
        resid0 = np.where(hp, obs - pA, np.nan)
        ctx = np.zeros(len(sub)); c_sel = None; betas = {}
        if not args.no_context:
            sub_team = sub.drop(columns="opp_team_id").assign(opp_team_id=sub["team_id"].values)
            absence = ([] if absent is None or spec.kind != "rate" else
                       X.teammate_absence_features(sub, spec, expo, cen_asof, pA, absent))
            best = None
            for c in C_GRID:
                cols = X.schedule_features(sub) + [
                    X.running_group_residual(sub, resid0, expo, c),
                    X.previous_season_group_residual(sub, resid0, expo, c),
                    X.running_group_residual(sub_team, resid0, expo, c)] + absence
                cx, bt = X.expanding_linear_context(np.column_stack(cols), resid0, expo, se, hp,
                                                    seasons_all[1:])
                pc, *_ = run_filter_v2(lay, obs_fit, sa["v2"]["params"], mr, transient=cx)
                e = weighted_mse(pc + centers, obs, expo, dev)
                if best is None or e < best[0]:
                    best = (e, c, cx, bt)
            _, c_sel, ctx, betas = best

        if args.no_corrections:
            mu0 = None; shift = None; trans = np.zeros(len(sub)); meta = {}
        else:
            mu0_raw, shift, trans, meta = build_corrections(sub, spec, age, mean_rate, seasons_all)
            mu0 = mu0_raw - centers

        def run(p):
            lam = p[6] if not args.no_context else 0.0
            return run_filter_v2(lay, obs_fit, p[:6], mr, mu0=mu0,
                                 transient=trans + lam * ctx, level_shift=shift)

        def obj(p):
            return weighted_mse(run(p)[0] + centers, obs, expo, dev)

        bounds = bounds_v2(bv, float(expo[pre].mean()))
        if not args.no_context:
            bounds = bounds + [(0.0, 2.0)]
        res = differential_evolution(obj, bounds=bounds, seed=SEED, maxiter=args.maxiter,
                                     popsize=args.popsize, tol=1e-7, polish=False)
        p = [float(v) for v in res.x]
        pred, pvar, _, _ = run(p); pred = pred + centers
        e4 = weighted_mse(pred, obs, expo, dev)

        # references on identical rows
        refs = {"A": pA}
        sbj = json.load(open(args.stage_b)) if Path(args.stage_b).exists() else {}
        if spec.name in sbj:
            m0b, shb, trb, _ = build_corrections(sub, spec, age, mean_rate, seasons_all) \
                if args.no_corrections else (mu0_raw, shift, trans, None)
            pB, *_ = run_filter_v2(lay, obs_fit, sbj[spec.name]["params"], mr, mu0=m0b - centers,
                                   transient=trb, level_shift=shb)
            refs["B"] = pB + centers
        out = {}
        for k, pr in refs.items():
            gain, ci = game_bootstrap(obs - pr, obs - pred, expo, sub["game_id"].to_numpy(), dev,
                                      args.boot, rng)
            per = {int(S): 100 * (weighted_mse(pr, obs, expo, dev & (se == S)) -
                                  weighted_mse(pred, obs, expo, dev & (se == S))) /
                   weighted_mse(pr, obs, expo, dev & (se == S)) for S in DEV_SEASONS}
            out[k] = dict(gain_pct=gain, ci95=ci, seasons_won=sum(v > 0 for v in per.values()),
                          per_season_pct=per)
        e1 = sa["v1"]["obj"]

        # variance calibration, expanding
        tot_var = pvar + p[5] / expo
        r2 = (obs - pred) ** 2
        sdz = {}
        for S in DEV_SEASONS:
            k = variance_scale(r2[(se < S) & hp], tot_var[(se < S) & hp])
            m = (se == S) & hp
            sdz[int(S)] = float(np.sqrt(np.average(r2[m] / (k * tot_var[m]))))

        report[spec.name] = dict(
            params=p, params_named=dict(zip(V2_PARAMS + ["ctx_lambda"], p)),
            centered=sa["centered"], context_c=c_sel, context_betas=betas,
            context_features=(["home", "b2b", "rest3", "opp_run", "opp_prev", "team_run"] +
                              (["eout_fresh", "eout_x_fresh", "eout_long", "eout_x_long", "eout_x_focal"]
                               if (absent is not None and spec.kind == "rate") else [])),
            corrections=not args.no_corrections, obj_v4=e4, obj_v1=e1,
            vs=out, vs_v1_pct=100 * (e1 - e4) / e1, sd_z_per_season=sdz,
            converged=bool(res.success), seconds=round(time.time() - t0, 1))
        oa = out["A"]; ob = out.get("B", {"gain_pct": np.nan, "seasons_won": -1})
        lam = p[6] if not args.no_context else 0.0
        print(f"{spec.name:<19}{(c_sel or 0):>4.0f}{lam:>6.2f}{oa['gain_pct']:>9.3f}"
              f"{'[%.3f,%.3f]' % tuple(oa['ci95']):>18}{oa['seasons_won']:>3}/6"
              f"{ob['gain_pct']:>9.3f}{ob['seasons_won']:>3}/6{100*(e1-e4)/e1:>9.3f}"
              f"{np.median(list(sdz.values())):>7.3f}{time.time()-t0:>6.0f}", flush=True)
        Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
