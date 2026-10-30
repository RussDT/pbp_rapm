"""ONE-SHOT frozen 2026 confirmation: v1 incumbent vs v3 selected candidate.

Both models use hyperparameters FROZEN on dev (2020-2025) and are re-run over
a panel that now includes 2026 so the filter can reach it. Nothing is tuned
here: no DE call is made. The only 2026 quantity either model sees is the
league centering series, which is v1's own pre-existing convention and is
applied identically to both arms, so it cancels in the paired comparison.

Scoring is the forward (rest-of-season) aggregate on 2026 as-of dates only,
on the common cell set, with a player-clustered bootstrap CI.

Run once. Re-running with different candidates spends the holdout.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_nba_estimated_skills import (
    SKILLS, load_panel, skill_subpanel, league_center, FilterLayout, run_filter,
    OUT_DIR, HOLDOUT_SEASON)
from build_nba_estimated_skills_v2 import LayoutV2, run_filter_v2
from build_nba_estimated_skills_v2_stageb import build_corrections, AGE_PATH
from eval_estimated_skills_forward import forward_cells, wcorr

SEED = 20260904


def main():
    stage_a = json.load(open(OUT_DIR / "v2_stage_a.json"))
    v3 = json.load(open(OUT_DIR / "v3.json"))
    df = load_panel()                      # includes 2026
    age = pd.read_parquet(AGE_PATH)
    seasons_all = sorted(df["season"].unique())
    assert HOLDOUT_SEASON in seasons_all, "2026 missing from panel"
    rng = np.random.default_rng(SEED)

    rows = []
    print(f"{'skill':<20}{'v1 fwdMSE':>12}{'v3 fwdMSE':>12}{'gain %':>9}"
          f"{'95% CI':>18}{'dates':>7}{'r v1':>8}{'r v3':>8}{'sd(z)':>7}")
    for spec in SKILLS:
        sub = skill_subpanel(df, spec)
        obs_raw = sub["obs_rate"].to_numpy(); seasons = sub["season"].to_numpy()
        pre = seasons < HOLDOUT_SEASON
        mean_rate = float(sub.loc[pre, spec.numer].sum() / sub.loc[pre, "exposure"].sum())
        centered = stage_a[spec.name]["centered"]
        if centered:
            centers, _ = league_center(sub, spec); obs_fit = obs_raw - centers; mr = 0.0
        else:
            centers = np.zeros(len(sub)); obs_fit = obs_raw; mr = mean_rate

        # --- incumbent v1, frozen dev params -------------------------------
        _, _, p1, _ = run_filter(FilterLayout(sub), obs_fit,
                                 stage_a[spec.name]["v1"]["params"], mr)
        # --- candidate v3, frozen dev params -------------------------------
        mu0_raw, shift, trans, _ = build_corrections(sub, spec, age, mean_rate, seasons_all)
        pred3, pvar3, p3, _ = run_filter_v2(
            LayoutV2(sub), obs_fit, v3[spec.name]["params"], mr,
            mu0=mu0_raw - centers, transient=trans, level_shift=shift)

        min_expo = 20.0 if spec.kind == "pct" else 200.0
        c1 = forward_cells(sub, spec, p1 + centers, min_expo, seasons=[HOLDOUT_SEASON])
        c3 = forward_cells(sub, spec, p3 + centers, min_expo, seasons=[HOLDOUT_SEASON])
        m = c1.merge(c3, on=["season", "asof", "player_id"], suffixes=("_1", "_3"))
        m = m[m["season"] == HOLDOUT_SEASON]
        if not len(m):
            continue
        e1 = (m.state_1 - m.target_1) ** 2; e3 = (m.state_3 - m.target_1) ** 2
        w = m.w_1.to_numpy()
        f1 = float(np.average(e1, weights=w)); f3 = float(np.average(e3, weights=w))
        gain = (f1 - f3) / f1 * 100.0

        # player-clustered bootstrap on the gain
        pl = m.player_id.to_numpy(); uq = np.unique(pl)
        idx = {p: np.flatnonzero(pl == p) for p in uq}
        gs = []
        for _ in range(2000):
            take = np.concatenate([idx[p] for p in rng.choice(uq, len(uq), replace=True)])
            a = np.average(e1.values[take], weights=w[take])
            b = np.average(e3.values[take], weights=w[take])
            gs.append((a - b) / a * 100.0)
        lo, hi = np.percentile(gs, [2.5, 97.5])

        won = tot = 0
        for _, g in m.groupby("asof"):
            tot += 1
            won += int(np.average((g.state_3 - g.target_1) ** 2, weights=g.w_1)
                       < np.average((g.state_1 - g.target_1) ** 2, weights=g.w_1))
        h26 = (seasons == HOLDOUT_SEASON) & (sub["step_k"] > 0).to_numpy()
        cs = v3[spec.name]["var_scale"]
        z = (obs_raw - (pred3 + centers)) / np.sqrt(np.maximum(cs * pvar3, 1e-18))
        sdz = float(np.std(z[h26]))
        r1 = wcorr(m.state_1.values, m.target_1.values, w)
        r3 = wcorr(m.state_3.values, m.target_1.values, w)
        rows.append(dict(skill=spec.name, fwd_mse_v1=f1, fwd_mse_v3=f3,
                         gain_pct=gain, ci_lo=float(lo), ci_hi=float(hi),
                         dates_won=won, dates_total=tot, wcorr_v1=r1,
                         wcorr_v3=r3, sd_z=sdz, n_cells=int(len(m))))
        print(f"{spec.name:<20}{f1:>12.6g}{f3:>12.6g}{gain:>8.2f}%"
              f"{f'[{lo:+.2f},{hi:+.2f}]':>18}{f'{won}/{tot}':>7}{r1:>8.4f}{r3:>8.4f}{sdz:>7.3f}")

    out = OUT_DIR / "confirmation_2026.json"
    json.dump(rows, open(out, "w"), indent=2)
    g = [r["gain_pct"] for r in rows]
    print(f"\nskills improved {sum(1 for x in g if x > 0)}/{len(g)}   "
          f"median {np.median(g):+.2f}%   range {min(g):+.2f}% .. {max(g):+.2f}%")
    print(f"CI excludes zero: {sum(1 for r in rows if r['ci_lo'] > 0)}/{len(rows)}")
    W = sum(r["dates_won"] for r in rows); T = sum(r["dates_total"] for r in rows)
    print(f"as-of dates won {W}/{T} = {W/T*100:.1f}%")
    sd = [r["sd_z"] for r in rows]
    print(f"sd(z) on 2026: {min(sd):.3f} .. {max(sd):.3f}  median {np.median(sd):.3f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
