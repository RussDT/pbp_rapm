"""Refit the registered annual neutral OU recipe on an admitted clean source.

This is a separate source/estimator control, not a timing or architecture gain.
Bounds, seed, context grid and finite DE budget match the legacy annual recipe.
Every annual fit excludes its deployment period; 2026 never trains parameters.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(key, "1")

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

import build_monthly_own_skills as C
import build_nba_estimated_skills as B
from build_nba_estimated_skills_v2 import LayoutV2, bounds_v2, run_filter_v2
from build_nba_estimated_skills_v2_stageb import build_corrections
from monthly_ou_filter import FastOU, load_kernel
from build_pregame_neutral_skills import replay, compile_kernel, query_panel, centers_at_queries
import estimated_skills_context as X

ROOT = C.ROOT
OUT = ROOT / "results/estimated_skills/pregame_source_clean_v1"
SOURCE = ROOT / "results/estimated_skills/pregame_source_validated_v2"
SEED = 20260928


def measurement_panel(frame, spec):
    """Masked/malformed counts cannot enter fits as zero measurements."""
    num = frame[spec.numer].to_numpy(float)
    den = frame[spec.denom].to_numpy(float)
    valid = np.isfinite(num) & np.isfinite(den) & (num >= 0) & (den > 0)
    if spec.kind == "pct":
        valid &= num <= den
    return B.skill_subpanel(frame.loc[valid].copy(), spec)


def load_panel(source, end):
    source = Path(source)
    frame = pd.read_parquet(source / "player_game_lines.parquet")
    frame.game_date = pd.to_datetime(frame.game_date)
    frame = frame.loc[frame.game_date.lt(end)].copy()
    if frame.season.gt(2026).any():
        raise ValueError("Protected outcomes present")
    frame["tot_poss"] = frame.off_poss + frame.def_poss
    frame["usg_events"] = frame.fga + .44 * frame.fta + frame.tov
    context = pd.read_parquet(source / "game_context.parquet")
    context["is_home"] = pd.to_numeric(context.is_home, errors="coerce").astype(float)
    context = context.merge(frame[["player_id", "game_id"]], on=["player_id", "game_id"], validate="one_to_one")
    frame = frame.merge(context, on=["player_id", "game_id"], how="left", validate="one_to_one")
    return frame.sort_values(["player_id", "game_date", "game_id"], kind="stable").reset_index(drop=True), context


def prepare(out, source, maxiter=60, popsize=15):
    out, source = Path(out), Path(source).resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    artifacts = manifest["artifacts"]
    for name, digest in artifacts.items():
        if C.sha(source / name) != digest:
            raise ValueError(f"Clean-source artifact changed: {name}")
    donors = [__file__, C.__file__, B.__file__, C.AGE]
    donors += [Path(__file__).with_name(name) for name in (
        "build_nba_estimated_skills_v2.py", "build_nba_estimated_skills_v2_stageb.py",
        "estimated_skills_context.py", "estimated_skills_corrections.py",
        "monthly_ou_filter.py", "monthly_ou_kernel.cpp")]
    recipe = dict(recipe="source_clean_annual_neutral_ou_v1", source_directory=str(source),
        source_manifest_sha256=C.sha(source / "manifest.json"), source_artifacts=artifacts,
        inputs={str(Path(p).resolve()): C.sha(p) for p in donors},
        maxiter=maxiter, popsize=popsize, seed=SEED, tol=1e-7, polish=False,
        context_grid=list(C.C_GRID), annual_vintages=list(range(2015, 2026)),
        source_clean_refit=True, parameter_transport_not_refit=False,
        selected_vpm_changed=False, outcome_seasons_max=2026,
        optimization_claim="fixed_budget_search_not_certified_optimum")
    out.mkdir(parents=True, exist_ok=True)
    lock = out / "source_lock.json"
    if lock.exists() and json.loads(lock.read_text()) != recipe:
        raise ValueError("Refit recipe changed; use a new package")
    C.write_json(lock, recipe)
    load_kernel(out / "kernel")
    return recipe


def fit_one(task):
    out, year, name = task
    out = Path(out)
    lock = json.loads((out / "source_lock.json").read_text())
    lock_sha = C.sha(out / "source_lock.json")
    if year not in lock["annual_vintages"]:
        raise ValueError("Unregistered annual fit")
    target = out / "models" / str(year) / (name + ".json")
    if target.exists():
        saved = json.loads(target.read_text())
        if saved["source_lock_sha256"] != lock_sha:
            raise ValueError("Cached model recipe changed")
        return dict(year=year, skill=name, status="cached")
    started = time.monotonic()
    spec = next(s for s in B.SKILLS if s.name == name)
    cut = C.vintage_cut(year)
    frame, ctx = load_panel(lock["source_directory"], cut)
    sub = measurement_panel(frame, spec)
    if sub.empty or sub.duplicated(["player_id", "game_date"]).any():
        raise ValueError("Empty or unresolved same-date measurement source")
    obs, w = sub.obs_rate.to_numpy(), sub.exposure.to_numpy()
    seasons, hp = sub.season.to_numpy(), sub.step_k.to_numpy() > 0
    fit = hp & (seasons > seasons.min())
    if fit.sum() < 100:
        raise ValueError("Insufficient chronological skill training")
    B.WARMUP_SEASON = int(seasons.min())
    league, _ = B.league_center(sub, spec)
    mean = float(np.average(obs, weights=w))
    engine = FastOU(sub, out / "kernel")
    search = dict(seed=lock["seed"], maxiter=lock["maxiter"], popsize=lock["popsize"],
                  tol=lock["tol"], polish=lock["polish"])
    def objective(pred):
        return float(np.average((pred[fit] - obs[fit]) ** 2, weights=w[fit]))
    choices = []
    for centered in ([False, True] if spec.kind == "pct" else [False]):
        centers = league if centered else np.zeros(len(sub))
        mr = 0. if centered else mean
        response = obs - centers
        variance = float(np.average((response - mr) ** 2, weights=w))
        result = differential_evolution(lambda p: objective(engine.run(response, p, mr)[0] + centers),
                                       bounds_v2(variance, float(w.mean())), **search)
        choices.append(dict(centered=centered, params=result.x.tolist(), loss=float(result.fun),
                            converged=bool(result.success), iterations=int(result.nit), nfev=int(result.nfev)))
    stage_a = min(choices, key=lambda value: value["loss"])
    centers = league if stage_a["centered"] else np.zeros(len(sub))
    mr = 0. if stage_a["centered"] else mean
    response = obs - centers
    variance = float(np.average((response - mr) ** 2, weights=w))
    slow = run_filter_v2(LayoutV2(sub), response, stage_a["params"], mr)
    fast = engine.run(response, stage_a["params"], mr)
    gap = max(float(np.max(np.abs(a - b))) for a, b in zip(slow, fast))
    if gap > 1e-10:
        raise ValueError(f"Independent OU algebra mismatch: {gap}")
    base = fast[0] + centers
    residual = np.where(hp, obs - base, np.nan)
    absent = X.absent_teammates(frame, ctx) if spec.kind == "rate" else None
    reference_year = int(seasons.max())
    best = None
    context_scores = []
    for c in C.C_GRID:
        cols = C.context_columns(sub, spec, base, residual, league, absent, c, reference_year)
        context, _ = X.expanding_linear_context(cols, residual, w, seasons, hp, sorted(set(seasons))[1:])
        value = objective(engine.run(response, stage_a["params"], mr, trans=context)[0] + centers)
        context_scores.append(dict(c=c, objective=value))
        if best is None or value < best[0]:
            best = value, c, context
    _, c, context = best
    initial, shift, trans, _ = build_corrections(sub, spec, pd.read_parquet(C.AGE), mean, sorted(set(seasons)))
    result = differential_evolution(lambda p: objective(engine.run(response, p[:6], mr,
        initial - centers, shift, trans, context, p[6])[0] + centers),
        bounds_v2(variance, float(w.mean())) + [(0., 2.)], **search)
    payload = dict(skill=name, vintage=year, fit_cut=str(cut.date()), fit_end=str(sub.game_date.max().date()),
        training_rows=len(sub), training_seasons=sorted(map(int, set(seasons))), objective_rows=int(fit.sum()),
        mean_rate=mean, centered=stage_a["centered"], stage_a=stage_a, centering_candidates=choices,
        context_c=c, context_grid_scores=context_scores, params=result.x.tolist(), objective=float(result.fun),
        converged=bool(result.success), iterations=int(result.nit), nfev=int(result.nfev),
        maxiter=lock["maxiter"], popsize=lock["popsize"], seed=lock["seed"], reference_year=reference_year,
        fast_slow_max_gap=gap, seconds=round(time.monotonic()-started, 3), source_lock_sha256=lock_sha,
        source_clean_refit=True, parameter_transport_not_refit=False, status="chronological_source_clean_fit_research")
    if pd.Timestamp(payload["fit_end"]) >= cut or max(payload["training_seasons"]) > year:
        raise ValueError("Annual parameter chronology violated")
    C.write_json(target, payload)
    return {key: payload[key] for key in ("vintage", "skill", "training_rows", "objective", "converged", "seconds")}



# Replay semantics match the frozen neutral-query bridge; models come from this
# source-clean package, with a separate export lock after all fits are complete.
def export_one(task):
    out, year, name = task
    out = Path(out)
    target = out / "families" / str(year) / (name + ".parquet")
    receipt = target.with_suffix(".json")
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old["run_lock_sha256"] != C.sha(out / "export_lock.json") or old["output_sha256"] != C.sha(target):
            raise ValueError("Cached pregame family differs from lock")
        return dict(year=year, skill=name, status="cached", rows=old["rows"])
    start = time.monotonic()
    spec = next(s for s in B.SKILLS if s.name == name)
    model_path = out / "models" / str(year) / (name + ".json")
    model = json.loads(model_path.read_text())
    cut, end = C.vintage_cut(year), C.vintage_cut(year + 1)
    if year == 2025:
        end = pd.Timestamp("2026-09-01")
    if pd.Timestamp(model["fit_end"]) >= cut:
        raise ValueError("Parameter vintage reaches deployment")
    source_directory = json.loads((out / "export_lock.json").read_text())["source_directory"]
    if source_directory is None:
        frame, context_source = C.load_panel(C.OUT, end)
    else:
        source_directory = Path(source_directory)
        frame = pd.read_parquet(source_directory / "player_game_lines.parquet")
        frame.game_date = pd.to_datetime(frame.game_date)
        frame = frame.loc[frame.game_date.lt(end)].copy()
        frame["tot_poss"] = frame.off_poss + frame.def_poss
        frame["usg_events"] = frame.fga + .44 * frame.fta + frame.tov
        context_source = pd.read_parquet(source_directory / "game_context.parquet")
        context_source["is_home"] = pd.to_numeric(context_source.is_home, errors="coerce").astype(float)
        frame = frame.merge(context_source, on=["player_id", "game_id"], how="left", validate="one_to_one")
        frame = frame.sort_values(["player_id", "game_date", "game_id"], kind="stable").reset_index(drop=True)
    measured = frame[spec.numer].notna() & frame[spec.numer].ge(0) & frame[spec.denom].gt(0)
    if spec.kind == "pct":
        measured &= frame[spec.numer].le(frame[spec.denom])
    sub = measurement_panel(frame, spec)
    if sub.obs_rate.isna().any() or not sub.obs_rate.ge(0).all() or (spec.kind == "pct" and not sub.obs_rate.le(1).all()):
        raise ValueError("Legacy measurement panel contains inadmissible counts")
    B.WARMUP_SEASON = int(sub.season.min())
    league, _ = B.league_center(sub, spec)
    centers = league if model["centered"] else np.zeros(len(sub))
    mean = 0. if model["centered"] else model["mean_rate"]
    observation = sub.obs_rate.to_numpy() - centers
    engine = FastOU(sub, out / "kernel")
    base = engine.run(observation, model["stage_a"]["params"], mean)[0] + centers
    has_prior = sub.step_k.to_numpy() > 0
    residual = np.where(has_prior, sub.obs_rate.to_numpy() - base, np.nan)
    absent = X.absent_teammates(frame, context_source) if spec.kind == "rate" else None
    columns = C.context_columns(sub, spec, base, residual, league, absent, model["context_c"], model["reference_year"])
    context, _ = X.expanding_linear_context(columns, residual, sub.exposure.to_numpy(), sub.season.to_numpy(),
                  has_prior, sorted(sub.season.unique())[1:])
    age = pd.read_parquet(C.AGE)
    initial, shifts, transient, _ = build_corrections(sub, spec, age, model["mean_rate"], sorted(sub.season.unique()))
    parameters = model["params"]
    expected = engine.run(observation, parameters[:6], mean, initial - centers, shifts, transient, context, parameters[6])
    queries = query_panel(frame, spec)
    query_centers = centers_at_queries(queries, spec) if model["centered"] else np.zeros(len(queries))
    query_initial, query_shift, _, _ = build_corrections(queries, spec, age, model["mean_rate"], sorted(queries.season.unique()))
    # Known observation offsets preserve the original measurement innovation.
    offsets = sub[["player_id", "game_id"]].assign(offset=transient + parameters[6] * context)
    queries = queries.merge(offsets, on=["player_id", "game_id"], how="left", validate="one_to_one")
    values = replay(queries.assign(obs_rate=queries.obs_rate - query_centers), parameters[:6], mean,
                    query_initial - query_centers, query_shift, queries.offset.fillna(0).to_numpy(), compile_kernel(out))
    aligned = queries[["player_id", "game_id"]].assign(neutral_post=values[2], post_var=values[3], center=query_centers)
    original = sub[["player_id", "game_id"]].merge(aligned, on=["player_id", "game_id"], validate="one_to_one")
    same_day = sub.duplicated(["player_id", "game_date"], keep=False).any()
    gap = max(float(np.max(np.abs(original.neutral_post.to_numpy() - expected[2]))),
              float(np.max(np.abs(original.post_var.to_numpy() - expected[3]))))
    center_gap = float(np.max(np.abs(original.center.to_numpy() - centers)))
    if same_day:
        raise ValueError("Same-date legacy source must be reconciled with validated identities before admission")
    if max(gap, center_gap) > 1e-9:
        raise ValueError(f"Neutral query bridge changed legacy observations: {gap}, center={center_gap}")
    take = queries.game_date.ge(cut) & queries.game_date.lt(end)
    scale = 1. if spec.kind == "pct" else 100.
    neutral = values[0] + query_centers
    neutral = np.clip(neutral, 0, 1) if spec.kind == "pct" else np.maximum(neutral, 0)
    result = queries.loc[take, ["player_id", "game_id", "game_date", "season"]].copy()
    result["feature"] = "own_" + name
    result["value"] = neutral[take] * scale
    result["state_variance"] = values[1][take] * scale ** 2
    result["prior_exposure"] = values[4][take]
    result["last_observation_day"] = values[5][take].astype(np.int32)
    result["observed_this_game"] = queries.exposure[take].to_numpy() > 0
    result["parameter_vintage"] = year
    result["parameter_fit_end"] = pd.Timestamp(model["fit_end"])
    days = (result.game_date - pd.Timestamp("1970-01-01")).dt.days
    if not ((result.last_observation_day == -1) | (result.last_observation_day < days)).all():
        raise ValueError("Pregame query includes same-day/future observation")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    result.to_parquet(temporary, index=False)
    temporary.replace(target)
    meta = dict(status="passed_neutral_core_family", year=year, skill=name, rows=len(result),
                query_only_rows=int((~result.observed_this_game).sum()), model_sha256=C.sha(model_path),
                run_lock_sha256=C.sha(out / "export_lock.json"), output_sha256=C.sha(target),
                legacy_post_state_max_gap=gap, center_max_gap=center_gap, same_day_source=bool(same_day),
                first_date=str(result.game_date.min().date()), last_date=str(result.game_date.max().date()),
                seconds=round(time.monotonic() - start, 3), context="neutral_pregame_not_contextual_next_game",
                source_directory=str(source_directory) if source_directory else None,
                parameter_transport_not_refit=False, source_clean_refit=True)
    C.write_json(receipt, meta)
    return meta



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--stage", choices=["prepare", "train", "export"], required=True)
    parser.add_argument("--years", default=",".join(map(str, range(2015, 2026))))
    parser.add_argument("--skills", default=",".join(s.name for s in B.SKILLS))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=60)
    parser.add_argument("--popsize", type=int, default=15)
    args = parser.parse_args()
    prepare(args.out, args.source, args.maxiter, args.popsize)
    if args.stage == "prepare":
        return
    years, skills = list(map(int, args.years.split(","))), args.skills.split(",")
    if not set(skills).issubset({s.name for s in B.SKILLS}):
        raise ValueError("Unregistered skill")
    tasks = [(str(args.out), year, name) for year in years for name in skills]
    if args.stage == "export":
        models = [args.out / "models" / str(year) / (name + ".json") for _, year, name in tasks]
        export_lock = dict(source_directory=str(args.source.resolve()),
            source_lock_sha256=C.sha(args.out / "source_lock.json"),
            model_hashes={str(p): C.sha(p) for p in models}, source_clean_refit=True,
            parameter_transport_not_refit=False, selected_vpm_changed=False)
        path = args.out / "export_lock.json"
        if path.exists() and json.loads(path.read_text()) != export_lock:
            raise ValueError("Export models changed; use a new package")
        C.write_json(path, export_lock)
        compile_kernel(args.out)
    worker = export_one if args.stage == "export" else fit_one
    status_path = args.out / ("export_status.json" if args.stage == "export" else "run_status.json")
    status = dict(status="running", stage=args.stage, tasks=len(tasks), completed=0, failures=0)
    C.write_json(status_path, status)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(worker, task) for task in tasks]):
            try:
                result = future.result()
            except Exception as exc:
                status.update(status="failed", failures=status["failures"]+1, error=str(exc))
                C.write_json(status_path, status)
                raise
            status["completed"] += 1
            C.write_json(status_path, status)
            print(json.dumps(result), flush=True)
    status["status"] = "complete"
    C.write_json(status_path, status)


if __name__ == "__main__":
    main()
