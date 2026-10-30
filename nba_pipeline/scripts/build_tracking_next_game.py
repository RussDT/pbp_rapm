"""Next-game per-100 filters for the six unmodeled tracking stats and the fouls-drawn family.

Generalizes ``build_deflections_next_game.py`` over a stat registry. The
``tracking`` family is the Gabriel tracking game-log panel: potential assists,
drives and possession minutes per offensive possession; contested shots,
defended field-goal attempts and contested defensive rebounds per defensive
possession. Measured observations only (PA/DFGA repairs and ``*_filled`` values
are never observations); missing games are unscored, never zero. 2019 warm-up,
tune and select on 2020-2022, frozen 2023-2025 historical stress, 2026
filtering and export only, 2027 protected.

The ``fouls_drawn`` family (``build_fouls_drawn_game_lines.py``, from the CDN
foul ledger, which names the fouled player) runs the same protocol under its
own results root: offensive fouls drawn and charges drawn per 100 defensive
possessions, personal fouls drawn per 100 offensive possessions; only games
reconciled to the repo PBP and box PFD are observations. Its data start in
2019-20, so it warms up on 2020 and selects on 2021-2022; stress and export
seasons are shared.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from build_nba_estimated_skills import SkillSpec, skill_subpanel, FilterLayout, run_filter, build_baselines, stat_bounds
from build_nba_estimated_skills_v2 import LayoutV2, run_filter_v2, bounds_v2
from build_deflections_next_game import compare

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT/"nba_pipeline/results/estimated_skills"
OUT = RESULTS/"tracking_next_game"
TRACKING = RESULTS/"tracking_backfill/tracking_backfill.parquet"
FOULS = RESULTS/"fouls_drawn/fouls_drawn_game_lines.parquet"
LINES = RESULTS/"player_game_lines.parquet"
SEED = 20260912
WARMUP = 2019
DEVELOPMENT = [2020, 2021, 2022]
STRESS = [2023, 2024, 2025]
EXPORT_ONLY = 2026
FAMILY_SOURCE = {"tracking": TRACKING, "fouls_drawn": FOULS}
FAMILY_OUT = {"tracking": OUT, "fouls_drawn": RESULTS/"fouls_drawn_next_game"}
BASELINES = ["trail10", "season_to_date", "career"]
MODELS = {"next_random_walk": "random_walk", "next_ou": "ou"}


@dataclass(frozen=True)
class TrackingStat:
    name: str
    source: str            # tracking game-log column
    measured: str | None   # measured flag; repaired PA/DFGA values are never observations
    exposure: str          # off_poss or def_poss from the resolved PBP lines
    label: str
    family: str = "tracking"                      # panel source; each family has its own results root
    warmup: int = WARMUP                          # first season with data; states only, never scored
    development: tuple = tuple(DEVELOPMENT)       # tune-and-select seasons; stress and export are shared


STATS = {s.name: s for s in [
    TrackingStat("potential_assists", "potential_ast_observed", "potential_ast_is_measured", "off_poss",
                 "potential assists per 100 offensive possessions"),
    TrackingStat("drives", "DRIVES", None, "off_poss", "drives per 100 offensive possessions"),
    TrackingStat("possession_minutes", "TIME_OF_POSS", None, "off_poss",
                 "minutes of possession per 100 offensive possessions"),
    TrackingStat("contested_shots", "hustle_CONTESTED_SHOTS", None, "def_poss",
                 "contested shots per 100 defensive possessions"),
    TrackingStat("dfga", "dfga_observed", "dfga_is_measured", "def_poss",
                 "defended field-goal attempts per 100 defensive possessions"),
    TrackingStat("contested_dreb", "DREB_CONTEST", None, "def_poss",
                 "contested defensive rebounds per 100 defensive possessions"),
    TrackingStat("off_fouls_drawn", "off_fouls_drawn", "fouls_drawn_is_measured", "def_poss",
                 "offensive fouls drawn (charges plus non-charge) per 100 defensive possessions",
                 family="fouls_drawn", warmup=2020, development=(2021, 2022)),
    TrackingStat("charges_drawn", "charges_drawn", "fouls_drawn_is_measured", "def_poss",
                 "charges drawn per 100 defensive possessions", family="fouls_drawn", warmup=2020, development=(2021, 2022)),
    TrackingStat("pfd", "pfd", "fouls_drawn_is_measured", "off_poss",
                 "personal fouls drawn per 100 offensive possessions", family="fouls_drawn", warmup=2020, development=(2021, 2022)),
]}
KEYS = ["player_id", "season", "season_type", "game_date"]
GAME_KEYS = ["player_id", "game_id"]


def family_stats(family):
    return {n: s for n, s in STATS.items() if s.family == family}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spec_for(stat):
    return SkillSpec(stat.name, stat.name+"_value", stat.exposure, "rate")


def build_panel(tracking, lines):
    """Measured tracking counts joined to resolved PBP exposure; one row per player-game."""
    t = tracking[tracking.season.between(WARMUP, EXPORT_ONLY)].rename(columns={
        "PLAYER_ID": "player_id", "PLAYER_NAME": "player_name", "MIN": "source_minutes"}).copy()
    t["season_type"] = np.where(t.playoffs, "ps", "rs")
    t["game_date"] = pd.to_datetime(t.date.astype(int).astype(str), format="%Y%m%d")
    if t.duplicated(KEYS).any():
        raise ValueError("Duplicate tracking identity")
    p = lines.copy()
    p["game_date"] = pd.to_datetime(p.game_date)
    ambiguous = p.duplicated(KEYS, keep=False)
    p = p.loc[~ambiguous, KEYS+["game_id", "minutes", "ast", "fga", "off_poss", "def_poss"]]
    d = t.merge(p, on=KEYS, how="left", validate="one_to_one")
    match = d.source_minutes.sub(d.minutes).abs().le(.5) & d.AST.eq(d.ast) & d.FGA.eq(d.fga)
    d["exposure_match"] = match.fillna(False).astype(bool)
    d.loc[~d.exposure_match, ["off_poss", "def_poss"]] = np.nan
    keep = ["player_id", "player_name", "TEAM_ID", "game_id", "game_date", "season", "season_type",
            "source_minutes", "off_poss", "def_poss", "exposure_match"]
    out = d[keep].copy()
    for stat in family_stats("tracking").values():
        value = d[stat.source].astype(float)
        if stat.measured is not None:
            value = value.where(d[stat.measured].astype(bool))
        expo = d[stat.exposure]
        out[stat.name+"_value"] = value
        out[stat.name+"_valid"] = value.notna() & value.ge(0) & expo.notna() & expo.gt(0)
        out[stat.name+"_reason"] = np.select(
            [value.isna(), value.lt(0), d.minutes.isna(), ~d.exposure_match, expo.le(0)],
            ["unmeasured", "negative_value", "missing_or_ambiguous_exposure", "box_or_minutes_mismatch", "zero_exposure"],
            default="observed")
    out = out.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    return out, int(ambiguous.sum())


def build_fouls_panel(fouls, lines):
    """Reconciled fouls-drawn counts joined to resolved PBP exposure on player and game id.

    Unmeasured games (unfetched, or failing the ledger/PBP/box reconciliation) carry NaN and
    their reconciliation reason; players in a reconciled game who drew nothing are genuine zeros.
    """
    stats = family_stats("fouls_drawn")
    first = min(s.warmup for s in stats.values())
    f = fouls[fouls.season.between(first, EXPORT_ONLY)].rename(columns={"minutes": "source_minutes"}).copy()
    f["game_id"] = f.game_id.astype(str)
    f["game_date"] = pd.to_datetime(f.game_date)
    if f.duplicated(GAME_KEYS).any():
        raise ValueError("Duplicate fouls-drawn identity")
    p = lines.copy()
    p["game_id"] = p.game_id.astype(str)
    ambiguous = p.duplicated(GAME_KEYS, keep=False)
    p = p.loc[~ambiguous, GAME_KEYS+["minutes", "off_poss", "def_poss"]]
    d = f.merge(p, on=GAME_KEYS, how="left", validate="one_to_one")
    d["exposure_match"] = d.minutes.notna() & d.source_minutes.sub(d.minutes).abs().le(.5)
    d.loc[~d.exposure_match, ["off_poss", "def_poss"]] = np.nan
    keep = ["player_id", "player_name", "game_id", "game_date", "season", "season_type", "source_minutes",
            "off_poss", "def_poss", "exposure_match"]
    out = d[keep].copy()
    for stat in stats.values():
        value = d[stat.source].astype(float).where(d[stat.measured].astype(bool))
        expo = d[stat.exposure]
        out[stat.name+"_value"] = value
        out[stat.name+"_valid"] = value.notna() & value.ge(0) & expo.notna() & expo.gt(0)
        out[stat.name+"_reason"] = np.select(
            [value.isna(), value.lt(0), d.minutes.isna(), ~d.exposure_match, expo.le(0)],
            [d.fouls_drawn_reason.fillna("unmeasured"), "negative_value", "missing_or_ambiguous_exposure",
             "box_or_minutes_mismatch", "zero_exposure"], default="observed")
    out = out.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    return out, int(ambiguous.sum())


def prepare(root, family="tracking"):
    root.mkdir(parents=True, exist_ok=True)
    source = FAMILY_SOURCE[family]
    lines = pd.read_parquet(LINES)
    builder = build_panel if family == "tracking" else build_fouls_panel
    panel, ambiguous = builder(pd.read_parquet(source), lines)
    panel.to_parquet(root/"panel.parquet", index=False)
    coverage = []
    for stat in family_stats(family).values():
        for season, g in panel.groupby("season"):
            valid = g[stat.name+"_valid"]
            coverage.append({"stat": stat.name, "season": int(season), "rows": len(g), "valid": int(valid.sum()),
                "covered_minutes_share": float(g.loc[valid, "source_minutes"].sum()/g.source_minutes.sum()),
                "reasons": g[stat.name+"_reason"].value_counts().to_dict()})
    report = {"family": family, "inputs": {str(p): sha(p) for p in [source, LINES]}, "ambiguous_exposure_rows_excluded": ambiguous,
        "rows": len(panel), "seasons": [int(s) for s in sorted(panel.season.unique())], "coverage": coverage}
    (root/"source_audit.json").write_text(json.dumps(report, indent=2))
    print(pd.DataFrame(coverage).pivot(index="season", columns="stat", values="valid").to_string())


def subpanel(panel, stat):
    return skill_subpanel(panel[panel[stat.name+"_valid"]].copy(), spec_for(stat))


def engine(name, sub, params, mean):
    obs = sub.obs_rate.to_numpy()
    if name == "ou":
        return run_filter_v2(LayoutV2(sub), obs, params, mean)
    return run_filter(FilterLayout(sub), obs, params, mean)


def run_models(sub, cfg):
    predictions, states = {}, {}
    for name, model in cfg["models"].items():
        pred, _, post, _ = engine(model["engine"], sub, model["params"], cfg["mean_rate"])
        predictions[name] = np.maximum(pred, 0)
        states[name] = np.maximum(post, 0)
    return predictions, states


def next_game_cells(sub, stat, predictions, seasons):
    seasons = [int(s) for s in seasons]
    if max(seasons) >= EXPORT_ONLY:
        raise ValueError("2026 is filtering/export only and 2027 is protected; no next-game scores")
    mask = (sub.season.isin(seasons) & sub.step_k.gt(0)).to_numpy()
    d = sub.loc[mask, ["player_id", "player_name", "game_id", "game_date", "season", "exposure", "obs_rate"]].copy()
    d["target100"] = 100*d.obs_rate
    d["prior_exposure"] = (sub.groupby("player_id").exposure.cumsum()-sub.exposure).to_numpy()[mask]
    baselines = build_baselines(sub, spec_for(stat))
    for k, values in baselines.items():
        d["season_to_date" if k == "season" else k] = 100*values[mask]
    for k, values in predictions.items():
        d[k] = 100*np.maximum(values[mask], 0)
    if d[[*BASELINES, *predictions]].isna().any().any():
        raise ValueError("Unequal baseline coverage on common evaluation rows")
    return d.reset_index(drop=True)


def select(comparison):
    """OU only if it beats the next-game random walk by more than one paired bootstrap SE."""
    return "next_ou" if comparison["delta_mse"] > comparison["delta_se"] else "next_random_walk"


def dependencies(root):
    scripts = Path(__file__).parent
    return {str(p): sha(p) for p in [Path(__file__), scripts/"build_deflections_next_game.py",
        scripts/"build_nba_estimated_skills.py", scripts/"build_nba_estimated_skills_v2.py", root/"panel.parquet"]}


def tune(root, name, maxiter=60, popsize=15):
    stat = STATS[name]
    out = root/name
    out.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(root/"panel.parquet")
    development = [int(s) for s in stat.development]
    panel = panel[panel.season.le(max(development))]
    sub = subpanel(panel, stat)
    spec = spec_for(stat)
    warm = sub.season.eq(stat.warmup)
    if not warm.any():
        raise ValueError(f"No warm-up rows for {name} in {stat.warmup}")
    mean = float(sub.loc[warm, spec.numer].sum()/sub.loc[warm, "exposure"].sum())
    var = float(np.average((sub.loc[warm, "obs_rate"]-mean)**2, weights=sub.loc[warm, "exposure"]))
    exposure = float(sub.loc[warm, "exposure"].mean())
    obs = sub.obs_rate.to_numpy()
    masks = [(sub.season.eq(s) & sub.step_k.gt(0)).to_numpy() for s in development]
    cfg = {"stat": name, "family": stat.family, "label": stat.label, "source": stat.source, "measured_flag": stat.measured,
        "exposure": stat.exposure, "mean_rate": mean, "warmup_var": var, "warmup_mean_exposure": exposure,
        "models": {}, "objective": "next-game exposure-weighted per-100 MSE; equal seasons", "seed": SEED,
        "warmup": stat.warmup, "development": development, "stress": STRESS, "export_only": EXPORT_ONLY,
        "maxiter": maxiter, "popsize": popsize, "dev_rows": int(sum(m.sum() for m in masks))}
    for label, eng in MODELS.items():
        layout = LayoutV2(sub) if eng == "ou" else FilterLayout(sub)
        fn = run_filter_v2 if eng == "ou" else run_filter
        bounds = bounds_v2(var, exposure) if eng == "ou" else stat_bounds(var, exposure)
        logbounds = [(np.log(max(a, 1e-12)), np.log(b)) for a, b in bounds]
        def objective(logparams):
            pred = np.maximum(fn(layout, obs, np.exp(logparams), mean)[0], 0)
            return np.mean([np.average((100*(pred[m]-obs[m]))**2, weights=layout.exposure[m]) for m in masks])
        start = time.time()
        iteration = [0]
        def callback(xk, convergence):
            iteration[0] += 1
            if iteration[0] % 10 == 0:
                print(f"{name} {label}: {iteration[0]} iterations, {time.time()-start:.0f}s", flush=True)
        result = differential_evolution(objective, logbounds, seed=SEED, maxiter=maxiter,
            popsize=popsize, tol=1e-7, polish=False, callback=callback)
        cfg["models"][label] = {"engine": eng, "params": np.exp(result.x).tolist(),
            "dev_mse_per100": float(result.fun), "converged": bool(result.success),
            "iterations": int(result.nit), "evaluations": int(result.nfev), "seconds": time.time()-start,
            "bounds": bounds}
        (out/"tuning_progress.json").write_text(json.dumps(cfg, indent=2))
    predictions, _ = run_models(sub, cfg)
    cells = next_game_cells(sub, stat, predictions, development)
    comparison = compare(cells, "next_ou", "next_random_walk")
    selected = select(comparison)
    cfg.update(selected=selected, engine_selection=comparison, dependencies=dependencies(root),
        selection_rule="next_ou only if its development gain over next_random_walk exceeds one paired game-cluster bootstrap SE; no stress-based switching")
    cfg["dev_metrics"] = [compare(cells, selected, b) for b in [*BASELINES, *cfg["models"]] if b != selected]
    cells.to_parquet(out/"development_cells.parquet", index=False)
    (out/"frozen.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps({k: cfg[k] for k in ["stat", "selected", "engine_selection"]}, indent=2))


def evaluate(root, name):
    stat = STATS[name]
    out = root/name
    cfg = json.loads((out/"frozen.json").read_text())
    if cfg["dependencies"] != dependencies(root):
        raise ValueError("Frozen input or harness dependency changed")
    panel = pd.read_parquet(root/"panel.parquet")
    if int(panel.season.max()) > EXPORT_ONLY:
        raise ValueError("Panel contains protected seasons beyond 2026")
    sub = subpanel(panel, stat)
    predictions, states = run_models(sub, cfg)
    cells = next_game_cells(sub, stat, predictions, STRESS)
    selected = cfg["selected"]
    baselines = [b for b in [*BASELINES, *cfg["models"]] if b != selected]
    report = {"stat": name, "label": stat.label, "selected": selected, "frozen_sha256": sha(out/"frozen.json"),
        "metrics": [compare(cells, selected, b) for b in baselines],
        "player_cluster_sensitivity": [compare(cells, selected, b, cluster="player") for b in baselines],
        "established_player_sensitivity": [compare(cells[cells.prior_exposure.ge(200)], selected, b) for b in baselines],
        "yearly": [compare(g, selected, b) for _, g in cells.groupby("season") for b in baselines],
        "status": "next-game primary; frozen 2023-2025 historical stress; 2026 export only; 2027 protected"}
    (out/"stress_report.json").write_text(json.dumps(report, indent=2))
    cells.to_parquet(out/"stress_cells.parquet", index=False)
    for model in cfg["models"]:
        sub[model+"_pregame100"] = 100*predictions[model]
        sub[model+"_postgame100"] = 100*states[model]
    sub.to_parquet(out/"predictions.parquet", index=False)
    board = sub.groupby("player_id", sort=False).tail(1).copy()
    field = f"estimated_{name}_per100"
    board[field] = board[selected+"_postgame100"]
    board["model"] = selected
    board["observed_games"] = board.step_k+1
    board["career_"+stat.exposure] = sub.groupby("player_id").exposure.sum().reindex(board.player_id).to_numpy()
    cols = ["player_id", "player_name", "game_date", "season", field, "model", "observed_games", "career_"+stat.exposure]
    board = board[cols].sort_values(field, ascending=False)
    board.to_csv(out/"current_states.csv", index=False)
    board[board.season.eq(EXPORT_ONLY)].to_csv(out/"current_2026.csv", index=False)
    print(json.dumps({"stat": name, "selected": selected, "metrics": report["metrics"],
        "players_2026": int(board.season.eq(EXPORT_ONLY).sum())}, indent=2))


def _fmt(r):
    lo, hi = r["gain_ci95"]
    return f"{r['gain_pct']:+.3f}% [{lo:+.3f}, {hi:+.3f}] {r['season_wins']}/{len(r['seasons'])}"


def summary(root, family="tracking"):
    stats = family_stats(family)
    dev = sorted({s for st in stats.values() for s in st.development})
    title = "Tracking next-game filters" if family == "tracking" else "Fouls-drawn next-game filters"
    rows, lines = [], [f"# {title}: summary", "",
        f"Development {min(dev)}-{max(dev)} selects between next-game OU and next-game random walk (one paired game-cluster SE).",
        "Stress 2023-2025 is frozen historical evidence; 2026 is export only; 2027 untouched.", "",
        "| Stat | Selected | Dev OU vs RW | Stress rows / games | MSE | RMSE | vs RW | vs trailing-10 | vs season-to-date | vs career | Player-cluster vs RW | >=200 prior vs RW | Yearly vs RW wins | 2026 players |",
        "|---|---|---|---:|---:|---:|---|---|---|---|---|---|---:|---:|"]
    for name, stat in stats.items():
        out = root/name
        if not (out/"stress_report.json").exists():
            continue
        cfg = json.loads((out/"frozen.json").read_text())
        rep = json.loads((out/"stress_report.json").read_text())
        by = {m["baseline"]: m for m in rep["metrics"]}
        pc = {m["baseline"]: m for m in rep["player_cluster_sensitivity"]}
        est = {m["baseline"]: m for m in rep["established_player_sensitivity"]}
        other = "next_random_walk" if cfg["selected"] == "next_ou" else "next_ou"
        yearly = [m for m in rep["yearly"] if m["baseline"] == other]
        players = int(len(pd.read_csv(out/"current_2026.csv")))
        primary = by["trail10"]
        row = {"stat": name, "label": stat.label, "selected": cfg["selected"],
            "dev_ou_vs_rw": cfg["engine_selection"], "stress_rows": primary["rows"], "stress_games": primary["games"],
            "mse_per100": primary["mse_per100"], "rmse_per100": primary["rmse_per100"],
            "vs": {k: by[k] for k in by}, "player_cluster_vs_other": pc[other], "established_vs_other": est[other],
            "yearly_vs_other": yearly, "players_2026": players,
            "params": {m: cfg["models"][m]["params"] for m in cfg["models"]}}
        rows.append(row)
        lines.append(f"| {name} | {cfg['selected']} | {_fmt(cfg['engine_selection'])} | {primary['rows']:,} / {primary['games']:,} | "
            f"{primary['mse_per100']:.4f} | {primary['rmse_per100']:.4f} | {_fmt(by[other])} | {_fmt(by['trail10'])} | "
            f"{_fmt(by['season_to_date'])} | {_fmt(by['career'])} | {_fmt(pc[other])} | {_fmt(est[other])} | "
            f"{sum(m['season_wins'] for m in yearly)}/{len(yearly)} | {players} |")
    lines += ["", "Gains are percent next-game MSE improvements of the selected filter over each comparator on identical rows and exposure weights, with 95% paired bootstrap intervals and per-season wins.", ""]
    for row in rows:
        board = pd.read_csv(root/row["stat"]/"current_2026.csv")
        expo = "career_"+STATS[row["stat"]].exposure
        top = board[board[expo].ge(1000)].head(5)
        field = f"estimated_{row['stat']}_per100"
        lines.append(f"- {row['stat']} ({row['label']}), top 2026 states with >=1,000 career possessions: "
            + "; ".join(f"{r.player_name} {getattr(r, field):.2f}" for r in top.itertuples()))
    (root/"summary.json").write_text(json.dumps(rows, indent=2))
    (root/"summary.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "tune", "evaluate", "summary"])
    parser.add_argument("--stat", choices=[*STATS, "all"], default="all")
    parser.add_argument("--family", choices=list(FAMILY_SOURCE), default="tracking",
                        help="stat family; a named --stat overrides this with its own family")
    parser.add_argument("--out", type=Path, default=None, help="results root; defaults to the family root")
    parser.add_argument("--maxiter", type=int, default=60)
    parser.add_argument("--popsize", type=int, default=15)
    args = parser.parse_args()
    family = STATS[args.stat].family if args.stat != "all" else args.family
    if args.out is None:
        args.out = FAMILY_OUT[family]
    args.out.mkdir(parents=True, exist_ok=True)
    names = list(family_stats(family)) if args.stat == "all" else [args.stat]
    if args.command == "prepare":
        prepare(args.out, family)
    elif args.command == "tune":
        for n in names:
            tune(args.out, n, args.maxiter, args.popsize)
    elif args.command == "evaluate":
        for n in names:
            evaluate(args.out, n)
    else:
        summary(args.out, family)
