"""Estimated deflections: prepare -> tune/freeze -> historical stress/export."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from build_nba_estimated_skills import SkillSpec, skill_subpanel, FilterLayout, run_filter, stat_bounds
from build_nba_estimated_skills_v2 import LayoutV2, run_filter_v2, bounds_v2
from eval_estimated_skills_forward import ASOF, wcorr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"nba_pipeline/results/estimated_skills/deflections"
SPEC = SkillSpec("deflections", "deflections", "def_poss", "rate")
SEED = 20260912
MODELS = ["random_walk", "ou"]
BASELINES = ["trail10", "season_to_date", "career"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(out):
    rows, inputs = [], []
    for year in range(2019, 2027):
        for suffix in ("", "ps"):
            path = ROOT/f"player_sheets/game_report/year_files/{year}{suffix}_games.csv"
            cols = ["PLAYER_ID", "PLAYER_NAME", "date", "MIN", "AST", "FGA", "hustle_DEFLECTIONS", "hustle_MIN"]
            a = pd.read_csv(path, usecols=lambda c: c in cols).reindex(columns=cols)
            a = a.rename(columns={"PLAYER_ID": "player_id", "PLAYER_NAME": "player_name",
                                  "MIN": "source_minutes", "hustle_DEFLECTIONS": "deflections"})
            a["season"], a["season_type"] = year, "ps" if suffix else "rs"
            rows.append(a[a.source_minutes.gt(0)])
            inputs.append({"path": str(path), "sha256": sha(path)})
    d = pd.concat(rows, ignore_index=True)
    keys = ["player_id", "date", "season", "season_type"]
    if d.duplicated(keys).any():
        raise ValueError("Duplicate source identity")
    path = ROOT/"nba_pipeline/results/estimated_skills/player_game_lines.parquet"
    p = pd.read_parquet(path)
    p["date"] = pd.to_datetime(p.game_date).dt.strftime("%Y%m%d").astype(int)
    p = p[p.def_poss.gt(0)]
    ambiguous = p.duplicated(keys, keep=False)
    ambiguity_count = int(ambiguous.sum())
    p = p.loc[~ambiguous, keys+["game_id", "minutes", "ast", "fga", "def_poss"]]
    d = d.merge(p, on=keys, how="left", validate="one_to_one")
    d["game_date"] = pd.to_datetime(d.date.astype(int).astype(str), format="%Y%m%d")
    match = d.AST.eq(d.ast) & d.FGA.eq(d.fga) & (d.source_minutes-d.minutes).abs().le(.5)
    # Optional endpoint minute check catches partial hustle coverage where supplied.
    hustle_minutes_ok = d.hustle_MIN.isna() | d.hustle_MIN.ge(.95*d.source_minutes-.1)
    d["valid"] = match & hustle_minutes_ok & d.def_poss.gt(0) & d.deflections.ge(0) & np.isfinite(d.deflections)
    d["reason"] = np.select([d.deflections.isna(), d.def_poss.isna(), ~match, ~hustle_minutes_ok],
        ["missing_hustle", "missing_or_ambiguous_exposure", "box_or_minutes_mismatch", "partial_hustle_minutes"], default="observed")
    d = d.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    d["full_row"] = np.arange(len(d))
    d.to_parquet(out/"panel.parquet", index=False)
    coverage = []
    for s, g in d.groupby("season"):
        coverage.append({"season": int(s), "rows": len(g), "valid": int(g.valid.sum()),
            "covered_minutes_share": float(g.loc[g.valid, "source_minutes"].sum()/g.source_minutes.sum()),
            "reasons": g.reason.value_counts().to_dict()})
    inputs.append({"path": str(path), "sha256": sha(path)})
    report = {"inputs": inputs, "ambiguous_exposure_rows_excluded": ambiguity_count, "coverage": coverage}
    (out/"source_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["coverage"], indent=2))


def subpanel(d):
    return skill_subpanel(d[d.valid].copy(), SPEC)


def forward_targets(d, k=41):
    """Postgame -> next k played games, only within season; coverage gated."""
    target = np.full(len(d), np.nan)
    weight = np.zeros(len(d))
    for _, ii in d.groupby(["player_id", "season"], sort=False).indices.items():
        g = d.iloc[ii]
        n = np.where(g.valid, g.deflections, 0.)
        e = np.where(g.valid, g.def_poss, 0.)
        m = g.source_minutes.to_numpy()
        cm = np.where(g.valid, m, 0.)
        start = np.arange(len(g))+1
        stop = np.minimum(start+k, len(g))
        sums = []
        for v in (n, e, m, cm):
            cs = np.r_[0., np.cumsum(v)]
            sums.append(cs[stop]-cs[start])
        nn, ee, mm, covered = sums
        ok = (stop-start >= 10) & (ee >= 200) & (covered >= .95*mm)
        target[ii[ok]] = nn[ok]/ee[ok]
        weight[ii[ok]] = ee[ok]
    return target, weight


def baseline_states(sub):
    pid = sub.player_id
    n, e = sub.deflections, sub.exposure
    out = {}
    out["career"] = n.groupby(pid).cumsum()/e.groupby(pid).cumsum()
    out["season_to_date"] = n.groupby([pid, sub.season]).cumsum()/e.groupby([pid, sub.season]).cumsum()
    out["trail10"] = n.groupby(pid).transform(lambda x: x.rolling(10, min_periods=1).sum())/e.groupby(pid).transform(lambda x: x.rolling(10, min_periods=1).sum())
    return {k: v.to_numpy() for k, v in out.items()}


def cutoff_cells(full, sub, states, seasons):
    out = []
    past_all = sub[["player_id", "game_date", "season"]].copy()
    for name, values in states.items():
        past_all[name] = values
    for season in seasons:
        ds = full[full.season.eq(season)]
        for month, day in ASOF:
            cutoff = pd.Timestamp(season-1 if month == 12 else season, month, day)
            if ds.empty or cutoff <= ds.game_date.min():
                continue
            future = ds[ds.game_date.ge(cutoff)].copy()
            if future.empty:
                continue
            future["n"] = future.deflections.where(future.valid, 0)
            future["e"] = future.def_poss.where(future.valid, 0)
            future["covered_minutes"] = future.source_minutes.where(future.valid, 0)
            ag = future.groupby("player_id").agg(n=("n", "sum"), e=("e", "sum"),
                covered=("covered_minutes", "sum"), total=("source_minutes", "sum"))
            ag = ag[ag.e.ge(200) & ag.covered.ge(.95*ag.total)]
            # Most recent measurement strictly before D, with history across seasons.
            past = past_all[past_all.game_date.lt(cutoff)].groupby("player_id", sort=False).tail(1).set_index("player_id")
            joined = ag.join(past[list(states)], how="inner").dropna()
            if len(joined) < 25:
                continue
            joined["target"] = 100*joined.n/joined.e
            for name in states:
                joined[name] *= 100
            joined["asof"], joined["season"] = cutoff, season
            out.append(joined.reset_index())
    if not out:
        raise ValueError("No eligible forward cells")
    return pd.concat(out, ignore_index=True)


def paired(cells, candidate, baseline, draws=2000):
    """Equal-cutoff MSE, paired player-cluster bootstrap across repeated cuts."""
    cut = pd.factorize(cells["asof"])[0]
    player = pd.factorize(cells.player_id)[0]
    np_ = player.max()+1
    w = cells.e.to_numpy()
    a = (cells[candidate]-cells.target).to_numpy()**2
    b = (cells[baseline]-cells.target).to_numpy()**2
    def score(weights):
        den = np.bincount(cut, weights=weights)
        return np.mean(np.bincount(cut, weights=weights*a)/den), np.mean(np.bincount(cut, weights=weights*b)/den)
    ma, mb = score(w)
    rng = np.random.default_rng(SEED)
    delta, gains = [], []
    for _ in range(draws):
        counts = np.bincount(rng.integers(0, np_, np_), minlength=np_)
        aa, bb = score(w*counts[player])
        delta.append(bb-aa)
        gains.append(100*(bb-aa)/bb)
    wins = sum(np.average(a[cut==k], weights=w[cut==k]) < np.average(b[cut==k], weights=w[cut==k]) for k in np.unique(cut))
    return {"candidate": candidate, "baseline": baseline, "mse": float(ma), "baseline_mse": float(mb),
        "gain_pct": float(100*(mb-ma)/mb), "gain_ci95": np.percentile(gains, [2.5, 97.5]).tolist(),
        "delta_mse": float(mb-ma), "delta_se": float(np.std(delta, ddof=1)),
        "cutoffs_won": int(wins), "cutoffs": int(cut.max()+1), "cells": len(cells)}


def engine(name, sub, params, mean):
    if name == "ou":
        return run_filter_v2(LayoutV2(sub), sub.obs_rate.to_numpy(), params, mean)
    return run_filter(FilterLayout(sub), sub.obs_rate.to_numpy(), params, mean)


def tune(out, maxiter=60, popsize=15):
    full = pd.read_parquet(out/"panel.parquet")
    full = full[full.season.le(2022)].reset_index(drop=True)
    full["full_row"] = np.arange(len(full))
    sub = subpanel(full)
    warm = sub.season.eq(2019)
    mean = float(sub.loc[warm, "deflections"].sum()/sub.loc[warm, "exposure"].sum())
    var = float(np.average((sub.loc[warm, "obs_rate"]-mean)**2, weights=sub.loc[warm, "exposure"]))
    exposure = float(sub.loc[warm, "exposure"].mean())
    ft, fw = forward_targets(full)
    ft, fw = ft[sub.full_row], fw[sub.full_row]
    masks = [sub.season.eq(s).to_numpy() & np.isfinite(ft) & (fw>0) for s in (2020, 2021, 2022)]
    configs, states = {}, baseline_states(sub)
    for name in MODELS:
        layout = LayoutV2(sub) if name == "ou" else FilterLayout(sub)
        fn = run_filter_v2 if name == "ou" else run_filter
        bounds = bounds_v2(var, exposure) if name == "ou" else stat_bounds(var, exposure)
        logbounds = [(np.log(max(a, 1e-12)), np.log(b)) for a, b in bounds]
        obs = sub.obs_rate.to_numpy()
        def objective(logparams):
            post = np.maximum(fn(layout, obs, np.exp(logparams), mean)[2], 0)
            return np.mean([np.average((post[m]-ft[m])**2, weights=fw[m]) for m in masks])
        start = time.time()
        iteration = [0]
        def callback(xk, convergence):
            iteration[0] += 1
            if iteration[0] % 10 == 0:
                print(f"{name}: iteration {iteration[0]}, {time.time()-start:.0f}s", flush=True)
        result = differential_evolution(objective, logbounds, seed=SEED, maxiter=maxiter,
            popsize=popsize, tol=1e-7, polish=False, callback=callback)
        params = np.exp(result.x).tolist()
        configs[name] = {"params": params, "objective": float(result.fun), "bounds": bounds,
            "converged": bool(result.success), "iterations": result.nit, "evaluations": result.nfev,
            "seconds": time.time()-start}
        states[name] = np.maximum(fn(layout, obs, params, mean)[2], 0)
        print(name, json.dumps(configs[name]), flush=True)
        (out/"tuning_progress.json").write_text(json.dumps(configs, indent=2))
    cells = cutoff_cells(full, sub, states, [2020, 2021, 2022])
    comparison = paired(cells, "ou", "random_walk")
    selected = "ou" if comparison["delta_mse"] > comparison["delta_se"] else "random_walk"
    metrics = [paired(cells, selected, b) for b in [*BASELINES, *MODELS] if b != selected]
    frozen = {"mean_rate": mean, "models": configs, "selected": selected,
        "selection_rule": "OU only if dev improvement exceeds one paired bootstrap SE",
        "selection_comparison": comparison, "dev_metrics": metrics, "maxiter": maxiter, "popsize": popsize,
        "panel_sha256": sha(out/"panel.parquet"), "script_sha256": sha(Path(__file__)),
        "tune_seasons": [2020, 2021, 2022], "stress_seasons": [2023, 2024, 2025], "seed": SEED}
    (out/"frozen.json").write_text(json.dumps(frozen, indent=2))
    cells.to_parquet(out/"development_cells.parquet", index=False)
    print(json.dumps({"selected": selected, "dev_metrics": metrics}, indent=2))


def stress(out):
    cfg = json.loads((out/"frozen.json").read_text())
    if cfg["panel_sha256"] != sha(out/"panel.parquet"):
        raise ValueError("Panel changed since freeze")
    if cfg["script_sha256"] != sha(Path(__file__)):
        raise ValueError("Harness changed since freeze; preserve the original receipt")
    full = pd.read_parquet(out/"panel.parquet")
    sub = subpanel(full)
    states = baseline_states(sub)
    for name in MODELS:
        pred, pv, post, postvar = engine(name, sub, cfg["models"][name]["params"], cfg["mean_rate"])
        states[name] = np.maximum(post, 0)
        sub[f"{name}_pred100"] = 100*np.maximum(pred, 0)
        sub[f"{name}_post100"] = 100*states[name]
        sub[f"{name}_latent_var100"] = 10000*postvar
    selected = cfg["selected"]
    # Explicit fixed season list: 2026 cannot enter accuracy scoring.
    cells = cutoff_cells(full, sub, states, [2023, 2024, 2025])
    baselines = [b for b in [*BASELINES, *MODELS] if b != selected]
    metrics = [paired(cells, selected, b) for b in baselines]
    yearly = []
    for season, g in cells.groupby("season"):
        for b in baselines:
            r = paired(g, selected, b)
            r["season"] = int(season)
            yearly.append(r)
    game = []
    for season in [2023, 2024, 2025]:
        g = sub[sub.season.eq(season) & sub.step_k.gt(0)]
        mse = float(np.average((g[f"{selected}_pred100"]-100*g.obs_rate)**2, weights=g.exposure))
        game.append({"season": season, "next_game_mse_per100": mse})
    normalized_weights = cells.e/cells.groupby("asof").e.transform("sum")
    report = {"selected": selected, "frozen_sha256": sha(out/"frozen.json"),
        "metrics": metrics, "yearly": yearly, "next_game_diagnostic": game,
        "weighted_correlation": wcorr(cells[selected].to_numpy(), cells.target.to_numpy(), normalized_weights.to_numpy()),
        "status": "historical research stress; 2026 export only, 2027 confirmation protected"}
    (out/"stress_report.json").write_text(json.dumps(report, indent=2))
    cells.to_parquet(out/"stress_cells.parquet", index=False)
    sub.to_parquet(out/"predictions.parquet", index=False)
    board = sub.groupby("player_id", sort=False).tail(1).copy()
    board["estimated_deflections_per100"] = board[f"{selected}_post100"]
    board["model"] = selected
    board["observed_games"] = board.step_k+1
    board["career_def_poss"] = sub.groupby("player_id").exposure.sum().reindex(board.player_id).to_numpy()
    cols = ["player_id", "player_name", "game_date", "season", "estimated_deflections_per100", "model", "observed_games", "career_def_poss", f"{selected}_latent_var100"]
    board[cols].sort_values("estimated_deflections_per100", ascending=False).to_csv(out/"current_states.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "tune", "stress"])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--maxiter", type=int, default=60)
    parser.add_argument("--popsize", type=int, default=15)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        prepare(args.out)
    elif args.command == "tune":
        tune(args.out, args.maxiter, args.popsize)
    else:
        stress(args.out)
