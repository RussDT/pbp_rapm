"""Forward-aggregate evaluation: the metric that matches deployment.

Game-level WMSE is dominated by irreducible within-game noise (a rim_pct row is
~3.5 attempts), so large state-quality gains show up as ~0.05% WMSE moves and
the metric cannot resolve them. Props/forecast use asks a different question:
standing at date D, how well does the state predict the player's REST-OF-SEASON
rate? That is the same design as the v1 doc's EPM E-Skills head-to-head.

For each as-of date D in a development season: take each model's latest state
strictly before D, and score it against that player's exposure-weighted actual
rate over all remaining games of that season, weighted by remaining exposure.
Reported: exposure-weighted MSE, weighted correlation, and per-date win records
versus v1 and versus the three naive baselines.

Development seasons only (2020-2025). 2026 is never read.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_nba_estimated_skills import (
    SKILLS, load_panel, skill_subpanel, league_center, FilterLayout,
    run_filter as run_v1, build_baselines, OUT_DIR, DEV_SEASONS, HOLDOUT_SEASON)
from build_nba_estimated_skills_v2 import LayoutV2, run_filter_v2

ASOF = [(12, 1), (1, 1), (2, 1), (3, 1)]


def wcorr(x, y, w):
    mx = np.average(x, weights=w); my = np.average(y, weights=w)
    cxy = np.average((x - mx) * (y - my), weights=w)
    cxx = np.average((x - mx) ** 2, weights=w); cyy = np.average((y - my) ** 2, weights=w)
    return float(cxy / np.sqrt(cxx * cyy)) if cxx > 0 and cyy > 0 else np.nan


def forward_cells(sub, spec, state, min_expo, seasons=None):
    """One row per (season, as-of date, player) with state and forward target.

    ``seasons`` defaults to DEV_SEASONS so no dev tooling can read the frozen
    holdout by accident. The one-shot 2026 confirmation passes it explicitly;
    that is the only sanctioned way to score the holdout.
    """
    seasons = DEV_SEASONS if seasons is None else seasons
    d = sub[["player_id", "season", "game_date", spec.numer, "exposure"]].copy()
    d["state"] = state
    out = []
    for s in sorted(set(d.season) & set(seasons)):
        ds = d[d.season == s]
        for mo, day in ASOF:
            yr = s - 1 if mo == 12 else s
            D = pd.Timestamp(year=yr, month=mo, day=day)
            past, fut = ds[ds.game_date < D], ds[ds.game_date >= D]
            if not len(past) or not len(fut):
                continue
            st = past.sort_values("game_date").groupby("player_id")["state"].last()
            ag = fut.groupby("player_id").agg(num=(spec.numer, "sum"), den=("exposure", "sum"))
            ag = ag[ag.den >= min_expo]
            j = ag.join(st, how="inner").dropna()
            if len(j) < 25:
                continue
            out.append(pd.DataFrame({"season": s, "asof": D, "player_id": j.index,
                                     "state": j.state.values,
                                     "target": (j.num / j.den).values, "w": j.den.values}))
    return pd.concat(out, ignore_index=True) if out else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-a", default=str(OUT_DIR / "v2_stage_a.json"))
    ap.add_argument("--stage-b", default="")
    ap.add_argument("--skills", default="")
    ap.add_argument("--out", default=str(OUT_DIR / "v2_forward_eval.json"))
    args = ap.parse_args()

    sa = json.load(open(args.stage_a))
    sb = json.load(open(args.stage_b)) if args.stage_b else {}
    df = load_panel()
    df = df[df["season"] < HOLDOUT_SEASON].copy()      # 2026 never read
    specs = [s for s in SKILLS if s.name in sa and (not args.skills or s.name in args.skills.split(","))]

    rep = {}
    hdr = f"{'skill':<19}{'v1 mse':>11}{'v2A mse':>11}{'v2B mse':>11}{'A vs v1':>9}{'B vs v1':>9}{'r v1':>7}{'r best':>8}{'cells':>7}"
    print(hdr)
    for spec in specs:
        sub = skill_subpanel(df, spec)
        obs = sub["obs_rate"].to_numpy()
        pre = (sub["season"] < HOLDOUT_SEASON).to_numpy()
        expo = sub["exposure"].to_numpy()
        mean_rate = float(sub.loc[pre, spec.numer].sum() / sub.loc[pre, "exposure"].sum())
        cfg = sa[spec.name]
        if cfg["centered"]:
            centers, _ = league_center(sub, spec); fit = obs - centers; mr = 0.0
        else:
            centers = np.zeros(len(sub)); fit = obs; mr = mean_rate
        min_expo = 20.0 if spec.kind == "pct" else 200.0

        _, _, p1, _ = run_v1(FilterLayout(sub), fit, cfg["v1"]["params"], mr)
        _, _, p2, _ = run_filter_v2(LayoutV2(sub), fit, cfg["v2"]["params"], mr)
        states = {"v1": p1 + centers, "v2A": p2 + centers}
        if spec.name in sb:
            _, _, p3, _ = run_filter_v2(LayoutV2(sub), fit, sb[spec.name]["params"], mr)
            states["v2B"] = p3 + centers

        res, cells = {}, None
        for k, st in states.items():
            c = forward_cells(sub, spec, st, min_expo)
            if c is None:
                break
            cells = c
            e = float(np.average((c.state - c.target) ** 2, weights=c.w))
            res[k] = {"mse": e, "r": wcorr(c.state.values, c.target.values, c.w.values),
                      "n_cells": int(len(c))}
        if not res or "v1" not in res:
            continue
        # per-as-of-date win record vs v1, on the COMMON cell set
        cv = forward_cells(sub, spec, states["v1"], min_expo)
        for k in [x for x in states if x != "v1"]:
            ck = forward_cells(sub, spec, states[k], min_expo)
            m = cv.merge(ck, on=["season", "asof", "player_id"], suffixes=("_1", "_k"))
            wins = tot = 0
            for _, g in m.groupby(["season", "asof"]):
                if not len(g):
                    continue
                e1 = np.average((g.state_1 - g.target_1) ** 2, weights=g.w_1)
                e2 = np.average((g.state_k - g.target_1) ** 2, weights=g.w_1)
                tot += 1; wins += int(e2 < e1)
            res[k]["dates_won_vs_v1"] = f"{wins}/{tot}"
            res[k]["impr_pct_vs_v1"] = 100 * (res["v1"]["mse"] - res[k]["mse"]) / res["v1"]["mse"]
        rep[spec.name] = res
        b = res.get("v2B", {})
        print(f"{spec.name:<19}{res['v1']['mse']:>11.5g}{res['v2A']['mse']:>11.5g}"
              f"{b.get('mse', float('nan')):>11.5g}{res['v2A']['impr_pct_vs_v1']:>9.3f}"
              f"{b.get('impr_pct_vs_v1', float('nan')):>9.3f}{res['v1']['r']:>7.3f}"
              f"{max(v['r'] for v in res.values()):>8.3f}{res['v1']['n_cells']:>7}")
        Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
