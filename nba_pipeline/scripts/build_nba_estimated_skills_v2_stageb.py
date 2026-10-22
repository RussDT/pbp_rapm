"""Stage B: v2 OU state + three deterministic corrections fit on PRIOR seasons.

Each correction targets a bias measured in v1's own dev residuals:

- debut prior  (v1 bias -0.076 SD, RMSE +34% on first-appearance rows): a
  ridge on repo-owned biographical covariates (age, years since draft, draft
  number, undrafted flag, height, weight) replaces "everyone starts at the
  league mean".
- age level shift (33+ at -0.028 SD vs <=22 at +0.002 in post-offseason rows):
  mean season-over-season rate change by age bin, applied to ``mu`` at season
  crossings.
- early-season transient (-0.020 SD over a player's first ~6 games): league
  within-season game-index effect, applied to the PREDICTION only so the state
  does not absorb a calendar artifact.

Expanding-window discipline: for target season S every table is fit on seasons
< S only. 2026 is never read (``--allow-holdout`` gates it and is unused here).

No health/injury term is included, deliberately: the within-season absence
diagnostic is a null (-0.003 SD across every gap bucket out to 60 days), so
elapsed time belongs in the variance, which the OU decay already handles.
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
import estimated_skills_corrections as C

AGE_PATH = str(OUT_DIR / "age_lookup.parquet")


def build_corrections(sub, spec, age, mean_rate, seasons_all):
    """Per-row (mu0_raw, level_shift, transient); tables fit on prior seasons."""
    n = len(sub)
    mu0 = np.full(n, mean_rate); shift = np.zeros(n); trans = np.zeros(n)
    ps = C.player_season_rates(sub, spec)
    d = sub.merge(age, on=["player_id", "season"], how="left")
    d["gidx"] = d.groupby(["player_id", "season"]).cumcount()
    d["gb"] = pd.cut(d.gidx, C.GIDX_BINS).astype(str)
    d["ab"] = pd.cut(d.age, C.AGE_BINS).astype(str)
    meta = {}
    for S in sorted(seasons_all):
        train = [s for s in seasons_all if s < S]
        idx = np.flatnonzero((sub["season"] == S).to_numpy())
        if not len(idx) or not train:
            continue
        dm, dfb = C.fit_debut(ps, age, train, mean_rate)
        ad = C.fit_age_delta(ps, age, train)
        es = C.fit_early_season(sub, spec, train)
        rows = d.iloc[idx]
        mu0[idx] = C.apply_debut(dm, dfb, rows)
        shift[idx] = rows["ab"].map(ad).fillna(0.0).to_numpy()
        trans[idx] = rows["gb"].map(es).fillna(0.0).to_numpy()
        meta[int(S)] = {"debut_fallback": dfb, "n_train_seasons": len(train),
                        "age_delta": ad, "early_season": es}
    return mu0, shift, trans, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills", default="")
    ap.add_argument("--maxiter", type=int, default=60)
    ap.add_argument("--popsize", type=int, default=15)
    ap.add_argument("--allow-holdout", action="store_true")
    ap.add_argument("--stage-a", default=str(OUT_DIR / "v2_stage_a.json"))
    ap.add_argument("--out", default=str(OUT_DIR / "v2_stage_b.json"))
    args = ap.parse_args()

    stage_a = json.load(open(args.stage_a))
    specs = [s for s in SKILLS if not args.skills or s.name in args.skills.split(",")]
    df = load_panel()
    if not args.allow_holdout:
        df = df[df["season"] < HOLDOUT_SEASON].copy()
    age = pd.read_parquet(AGE_PATH)
    seasons_all = sorted(df["season"].unique())

    report = {}
    print(f"{'skill':<19}{'v1':>12}{'v2 A':>12}{'v2 B':>12}"
          f"{'B vs A %':>10}{'B vs v1 %':>11}{'won':>6}{'sec':>6}")
    for spec in specs:
        if spec.name not in stage_a:
            continue
        sub = skill_subpanel(df, spec)
        obs_raw = sub["obs_rate"].to_numpy()
        seasons = sub["season"].to_numpy()
        has_prior = (sub["step_k"] > 0).to_numpy()
        dev_mask = np.isin(seasons, DEV_SEASONS) & has_prior
        pre = seasons < HOLDOUT_SEASON
        lay = LayoutV2(sub)
        expo = lay.exposure
        mean_rate = float(sub.loc[pre, spec.numer].sum() / sub.loc[pre, "exposure"].sum())
        sa = stage_a[spec.name]
        if sa["centered"]:
            centers, _ = league_center(sub, spec)
            obs_fit = obs_raw - centers; mr = 0.0
            bv = float(np.sum(expo[pre] * obs_fit[pre] ** 2) / np.sum(expo[pre]))
        else:
            centers = np.zeros(len(sub)); obs_fit = obs_raw; mr = mean_rate
            bv = float(np.sum(expo[pre] * (obs_raw[pre] - mean_rate) ** 2) / np.sum(expo[pre]))

        mu0_raw, shift, trans, meta = build_corrections(
            sub, spec, age, mean_rate, seasons_all)
        mu0 = mu0_raw - centers            # level -> fit space; deltas invariant

        def obj(p):
            pred, *_ = run_filter_v2(lay, obs_fit, p, mr, mu0=mu0,
                                     transient=trans, level_shift=shift)
            return weighted_mse(pred + centers, obs_raw, expo, dev_mask)

        t0 = time.time()
        res = differential_evolution(obj, bounds=bounds_v2(bv, float(expo[pre].mean())),
                                     seed=SEED, maxiter=args.maxiter,
                                     popsize=args.popsize, tol=1e-7, polish=False)
        pb, *_ = run_filter_v2(lay, obs_fit, [float(v) for v in res.x], mr, mu0=mu0,
                               transient=trans, level_shift=shift)
        pb = pb + centers
        pa_, *_ = run_filter_v2(lay, obs_fit, sa["v2"]["params"], mr)
        pa_ = pa_ + centers

        eB = weighted_mse(pb, obs_raw, expo, dev_mask)
        eA = sa["v2"]["obj"]; e1 = sa["v1"]["obj"]
        won = 0; per = {}
        for s in DEV_SEASONS:
            m = (seasons == s) & has_prior
            a_ = weighted_mse(pa_, obs_raw, expo, m)
            b_ = weighted_mse(pb, obs_raw, expo, m)
            per[int(s)] = {"A": a_, "B": b_, "impr_pct": 100 * (a_ - b_) / a_}
            won += int(b_ < a_)
        report[spec.name] = {
            "params": [float(v) for v in res.x],
            "params_named": dict(zip(V2_PARAMS, [float(v) for v in res.x])),
            "obj_v1": e1, "obj_v2a": eA, "obj_v2b": eB,
            "b_vs_a_pct": 100 * (eA - eB) / eA, "b_vs_v1_pct": 100 * (e1 - eB) / e1,
            "seasons_won_vs_a": won, "per_season": per,
            "converged": bool(res.success), "corrections": meta,
        }
        r = report[spec.name]
        print(f"{spec.name:<19}{e1:>12.6g}{eA:>12.6g}{eB:>12.6g}"
              f"{r['b_vs_a_pct']:>10.3f}{r['b_vs_v1_pct']:>11.3f}"
              f"{f'{won}/6':>6}{time.time()-t0:>6.0f}")
        Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
