"""Next-game per-100 deflection tuning; preserves the forward-horizon run."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

import build_estimated_deflections as prior
from build_nba_estimated_skills import FilterLayout, run_filter, build_baselines, stat_bounds
from build_nba_estimated_skills_v2 import LayoutV2, run_filter_v2, bounds_v2

OUT = prior.OUT.parent/"deflections_next_game"
SEED = prior.SEED
BASELINES = ["trail10", "season_to_date", "career"]


def next_game_cells(sub, predictions, seasons):
    mask = sub.season.isin(seasons) & sub.step_k.gt(0)
    d = sub.loc[mask, ["player_id", "player_name", "game_id", "game_date", "season", "exposure", "obs_rate"]].copy()
    d["target100"] = 100*d.obs_rate
    d["prior_exposure"] = (sub.groupby("player_id").exposure.cumsum()-sub.exposure)[mask]
    baselines = build_baselines(sub, prior.SPEC)
    for k, values in baselines.items():
        d["season_to_date" if k == "season" else k] = 100*values[mask]
    for k, values in predictions.items():
        d[k] = 100*np.maximum(values[mask], 0)
    columns = [*BASELINES, *predictions]
    if d[columns].isna().any().any():
        raise ValueError("Unequal baseline coverage on common evaluation rows")
    return d.reset_index(drop=True)


def compare(d, candidate, baseline, cluster="game", draws=2000):
    """Exposure-weighted within season; paired cluster bootstrap; equal seasons."""
    w = d.exposure.to_numpy()
    a = (d[candidate]-d.target100).to_numpy()**2*w
    b = (d[baseline]-d.target100).to_numpy()**2*w
    s, labels = pd.factorize(d.season)
    ns = len(labels)
    means_a = np.bincount(s, weights=a)/np.bincount(s, weights=w)
    means_b = np.bincount(s, weights=b)/np.bincount(s, weights=w)
    ma, mb = float(means_a.mean()), float(means_b.mean())
    rng = np.random.default_rng(SEED)
    boot_a, boot_b = np.zeros(draws), np.zeros(draws)
    if cluster == "game":
        for season in labels:
            take = d.season.eq(season).to_numpy()
            ids = pd.factorize(d.loc[take, "game_id"])[0]
            n = int(ids.max()+1)
            aa = np.bincount(ids, weights=a[take])
            bb = np.bincount(ids, weights=b[take])
            ww = np.bincount(ids, weights=w[take])
            for lo in range(0, draws, 100):
                hi = min(lo+100, draws)
                counts = rng.multinomial(n, np.full(n, 1/n), size=hi-lo)
                den = counts @ ww
                boot_a[lo:hi] += (counts @ aa)/den/ns
                boot_b[lo:hi] += (counts @ bb)/den/ns
    elif cluster == "player":
        ids = pd.factorize(d.player_id)[0]
        n = int(ids.max()+1)
        aa, bb, ww = (np.zeros((n, ns)) for _ in range(3))
        np.add.at(aa, (ids, s), a)
        np.add.at(bb, (ids, s), b)
        np.add.at(ww, (ids, s), w)
        for lo in range(0, draws, 100):
            hi = min(lo+100, draws)
            counts = rng.multinomial(n, np.full(n, 1/n), size=hi-lo)
            den = counts @ ww
            boot_a[lo:hi] = np.mean((counts @ aa)/den, axis=1)
            boot_b[lo:hi] = np.mean((counts @ bb)/den, axis=1)
    else:
        raise ValueError(cluster)
    gain = 100*(boot_b-boot_a)/boot_b
    return {"candidate": candidate, "baseline": baseline, "cluster": cluster,
        "mse_per100": ma, "baseline_mse_per100": mb, "rmse_per100": float(np.sqrt(ma)),
        "gain_pct": 100*(mb-ma)/mb, "gain_ci95": np.percentile(gain, [2.5, 97.5]).tolist(),
        "delta_mse": mb-ma, "delta_se": float(np.std(boot_b-boot_a, ddof=1)),
        "rows": len(d), "games": int(d.game_id.nunique()),
        "seasons": [int(x) for x in labels], "season_wins": int((means_a < means_b).sum())}


def choose(comparison, incumbent_comparison):
    challenger = "next_ou" if comparison["delta_mse"] > comparison["delta_se"] else "next_random_walk"
    selected = challenger if incumbent_comparison["delta_mse"] > incumbent_comparison["delta_se"] else "forward_incumbent"
    return challenger, selected


def run_models(sub, cfg):
    predictions, states = {}, {}
    for name, model in cfg["models"].items():
        pred, _, post, _ = prior.engine(model["engine"], sub, model["params"], cfg["mean_rate"])
        predictions[name] = np.maximum(pred, 0)
        states[name] = np.maximum(post, 0)
    return predictions, states


def dependencies():
    scripts = Path(__file__).parent
    return {str(p): prior.sha(p) for p in [Path(__file__), scripts/"build_estimated_deflections.py",
        scripts/"build_nba_estimated_skills.py", scripts/"build_nba_estimated_skills_v2.py",
        prior.OUT/"panel.parquet", prior.OUT/"frozen.json"]}


def tune(out):
    full = pd.read_parquet(prior.OUT/"panel.parquet")
    full = full[full.season.le(2022)].copy()
    sub = prior.subpanel(full)
    old = json.loads((prior.OUT/"frozen.json").read_text())
    mean = old["mean_rate"]
    warm = sub.season.eq(2019)
    var = float(np.average((sub.loc[warm, "obs_rate"]-mean)**2, weights=sub.loc[warm, "exposure"]))
    exposure = float(sub.loc[warm, "exposure"].mean())
    obs = sub.obs_rate.to_numpy()
    masks = [(sub.season.eq(s) & sub.step_k.gt(0)).to_numpy() for s in [2020, 2021, 2022]]
    cfg = {"mean_rate": mean, "models": {"forward_incumbent": {
        "engine": old["selected"], "params": old["models"][old["selected"]]["params"]}},
        "objective": "next-game exposure-weighted per-100 MSE; equal seasons", "seed": SEED,
        "development": [2020, 2021, 2022], "stress": [2023, 2024, 2025],
        "maxiter": 60, "popsize": 15}
    for name in ["random_walk", "ou"]:
        layout = LayoutV2(sub) if name == "ou" else FilterLayout(sub)
        fn = run_filter_v2 if name == "ou" else run_filter
        bounds = bounds_v2(var, exposure) if name == "ou" else stat_bounds(var, exposure)
        logbounds = [(np.log(max(a, 1e-12)), np.log(b)) for a, b in bounds]
        def objective(logparams):
            pred = np.maximum(fn(layout, obs, np.exp(logparams), mean)[0], 0)
            return np.mean([np.average((100*(pred[m]-obs[m]))**2, weights=layout.exposure[m]) for m in masks])
        start = time.time()
        iteration = [0]
        def callback(xk, convergence):
            iteration[0] += 1
            if iteration[0] % 10 == 0:
                print(f"next_{name}: {iteration[0]} iterations, {time.time()-start:.0f}s", flush=True)
        result = differential_evolution(objective, logbounds, seed=SEED, maxiter=60,
            popsize=15, tol=1e-7, polish=False, callback=callback)
        cfg["models"][f"next_{name}"] = {"engine": name, "params": np.exp(result.x).tolist(),
            "dev_mse_per100": float(result.fun), "converged": bool(result.success),
            "iterations": result.nit, "evaluations": result.nfev, "seconds": time.time()-start,
            "bounds": bounds}
        (out/"tuning_progress.json").write_text(json.dumps(cfg, indent=2))
    predictions, states = run_models(sub, cfg)
    cells = next_game_cells(sub, predictions, [2020, 2021, 2022])
    comparison = compare(cells, "next_ou", "next_random_walk")
    challenger = "next_ou" if comparison["delta_mse"] > comparison["delta_se"] else "next_random_walk"
    incumbent_comparison = compare(cells, challenger, "forward_incumbent")
    _, selected = choose(comparison, incumbent_comparison)
    cfg.update(selected=selected, challenger=challenger, engine_selection=comparison,
        incumbent_selection=incumbent_comparison, dependencies=dependencies())
    cfg["dev_metrics"] = [compare(cells, selected, b) for b in [*BASELINES, *cfg["models"]] if b != selected]
    cells.to_parquet(out/"development_cells.parquet", index=False)
    (out/"frozen.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps({k: cfg[k] for k in ["selected", "challenger", "engine_selection", "incumbent_selection"]}, indent=2))


def evaluate(out):
    cfg = json.loads((out/"frozen.json").read_text())
    if cfg["dependencies"] != dependencies():
        raise ValueError("Frozen input or harness dependency changed")
    full = pd.read_parquet(prior.OUT/"panel.parquet")
    sub = prior.subpanel(full)
    predictions, states = run_models(sub, cfg)
    cells = next_game_cells(sub, predictions, [2023, 2024, 2025])
    selected = cfg["selected"]
    baselines = [b for b in [*BASELINES, *cfg["models"]] if b != selected]
    report = {"selected": selected, "frozen_sha256": prior.sha(out/"frozen.json"),
        "metrics": [compare(cells, selected, b) for b in baselines],
        "player_cluster_sensitivity": [compare(cells, selected, b, cluster="player") for b in baselines],
        "established_player_sensitivity": [compare(cells[cells.prior_exposure.ge(200)], selected, b) for b in baselines],
        "yearly": [compare(g, selected, b) for _, g in cells.groupby("season") for b in baselines],
        "status": "next-game primary; already-spent historical stress; 2026 export only; 2027 protected"}
    # Secondary horizon, never used for next-game tuning/selection.
    secondary = prior.cutoff_cells(full, sub, states, [2023, 2024, 2025])
    report["remaining_season_secondary"] = [prior.paired(secondary, selected, b) for b in cfg["models"] if b != selected]
    (out/"stress_report.json").write_text(json.dumps(report, indent=2))
    cells.to_parquet(out/"stress_cells.parquet", index=False)
    secondary.to_parquet(out/"remaining_season_cells.parquet", index=False)
    for name in cfg["models"]:
        sub[name+"_pregame100"] = 100*predictions[name]
        sub[name+"_postgame100"] = 100*states[name]
    sub.to_parquet(out/"predictions.parquet", index=False)
    board = sub.groupby("player_id").tail(1).copy()
    board["estimated_deflections_per100"] = board[selected+"_postgame100"]
    board["model"] = selected
    board["observed_games"] = board.step_k+1
    board["career_def_poss"] = sub.groupby("player_id").exposure.sum().reindex(board.player_id).to_numpy()
    cols = ["player_id", "player_name", "game_date", "season", "estimated_deflections_per100", "model", "observed_games", "career_def_poss"]
    board = board[cols].sort_values("estimated_deflections_per100", ascending=False)
    board.to_csv(out/"current_states.csv", index=False)
    board[board.season.eq(2026)].to_csv(out/"current_2026.csv", index=False)
    print(json.dumps({"selected": selected, "metrics": report["metrics"], "players_2026": int(board.season.eq(2026).sum())}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["tune", "evaluate"])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "tune":
        tune(args.out)
    else:
        evaluate(args.out)
