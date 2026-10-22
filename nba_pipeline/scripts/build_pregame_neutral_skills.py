"""Game-date neutral queries of the existing legally dated annual OU recipes.

This bridge preserves the selected measurement/context/age/debut conventions.
Query-only games never update filter state. The separate full-history context-
free library is not silently substituted for the neutral timing control.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import ctypes
import json
import os
from pathlib import Path
import subprocess
import time

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(key, "1")

import numpy as np
import pandas as pd

import build_monthly_own_skills as C
import build_nba_estimated_skills as B
from build_nba_estimated_skills_v2_stageb import build_corrections
from build_full_skill_history import write_json
from monthly_ou_filter import FastOU
import estimated_skills_context as X

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/estimated_skills/pregame_neutral_v4"
KERNEL = Path(__file__).with_name("pregame_neutral_kernel.cpp")


def compile_kernel(out):
    folder = Path(out) / "kernel"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / ("pregame_neutral_" + C.sha(KERNEL)[:16] + ".so")
    if not target.exists():
        subprocess.run(["c++", "-O3", "-std=c++11", "-shared", "-fPIC", str(KERNEL), "-o", str(target)], check=True)
    return target


def replay(frame, parameters, mean, initial, shifts, transient, library):
    """Return pre/post neutral residual states plus earlier measurement exposure."""
    required = ["player_id", "day", "season", "exposure", "obs_rate"]
    if frame[required[:3]].isna().any().any():
        raise ValueError("Missing query identity/date")
    order = frame.sort_values(["player_id", "day"], kind="stable").index
    if not np.array_equal(order, frame.index):
        raise ValueError("Queries must be chronological within player")
    ids = np.ascontiguousarray(pd.factorize(frame.player_id)[0], dtype=np.int64)
    lib = ctypes.CDLL(str(Path(library).resolve()))
    fn = lib.pregame_neutral
    double = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    integer = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS")
    fn.argtypes = [ctypes.c_long, ctypes.c_long, integer, *([double] * 8), ctypes.c_double, *([double] * 6)]
    fn.restype = None
    arrays = [np.ascontiguousarray(value, dtype=float) for value in
              [frame.day, frame.season, frame.exposure, frame.obs_rate, initial, shifts, transient, parameters]]
    output = [np.empty(len(frame)) for _ in range(6)]
    fn(len(frame), int(ids.max()) + 1, ids, *arrays, float(mean), *output)
    return output


def query_panel(frame, spec):
    result = frame.copy().reset_index(drop=True)
    num, den = result[spec.numer].to_numpy(float), result[spec.denom].to_numpy(float)
    valid = np.isfinite(num) & np.isfinite(den) & (den > 0) & (num >= 0)
    if spec.kind == "pct":
        valid &= num <= den
    result["exposure"] = np.where(valid, den, 0.)
    result["obs_rate"] = np.divide(num, den, out=np.full(len(result), np.nan), where=valid)
    # Corrections fitted on earlier seasons must not count an inadmissible
    # numerator whose denominator is unavailable.
    result[spec.numer] = np.where(valid, num, 0.)
    result["day"] = (result.game_date - pd.Timestamp("1970-01-01")).dt.days
    result["player_idx"] = pd.factorize(result.player_id)[0]
    group = result.groupby("player_id", sort=False)
    result["step_k"] = group.cumcount()
    result["days_gap"] = group.game_date.diff().dt.days.fillna(0)
    result["season_cross"] = group.season.diff().fillna(0).gt(0).astype(float)
    return result


def centers_at_queries(queries, spec):
    """The existing prior-season+strictly-earlier-date center on every query."""
    daily = queries.groupby(["season", "game_date"], as_index=False).agg(num=(spec.numer, "sum"), den=("exposure", "sum"))
    daily = daily.sort_values(["season", "game_date"])
    totals = daily.groupby("season")[["num", "den"]].sum()
    previous = totals.shift().fillna(0)
    n = daily.groupby("season").num.cumsum() - daily.num + daily.season.map(previous.num)
    d = daily.groupby("season").den.cumsum() - daily.den + daily.season.map(previous.den)
    first = int(totals.index.min())
    fallback = totals.loc[first, "num"] / totals.loc[first, "den"]
    if not np.isfinite(fallback):
        raise ValueError("No warm-up center")
    if not daily.loc[d.le(0), "season"].eq(first).all():
        raise ValueError("Whole-warmup center reached a later season")
    daily["center"] = np.divide(n, d, out=np.full(len(daily), fallback), where=d > 0)
    return queries[["season", "game_date"]].merge(daily[["season", "game_date", "center"]],
             on=["season", "game_date"], how="left", validate="many_to_one").center.to_numpy()


def one(task):
    out, year, name = task
    out = Path(out)
    target = out / "families" / str(year) / (name + ".parquet")
    receipt = target.with_suffix(".json")
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old["run_lock_sha256"] != C.sha(out / "run_lock.json") or old["output_sha256"] != C.sha(target):
            raise ValueError("Cached pregame family differs from lock")
        return dict(year=year, skill=name, status="cached", rows=old["rows"])
    start = time.monotonic()
    spec = next(s for s in B.SKILLS if s.name == name)
    model_path = C.OUT / "models" / str(year) / (name + ".json")
    model = json.loads(model_path.read_text())
    cut, end = C.vintage_cut(year), C.vintage_cut(year + 1)
    if year == 2025:
        end = pd.Timestamp("2026-09-01")
    if pd.Timestamp(model["fit_end"]) >= cut:
        raise ValueError("Parameter vintage reaches deployment")
    source_directory = json.loads((out / "run_lock.json").read_text())["source_directory"]
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
    sub = B.skill_subpanel(frame.loc[measured].copy(), spec)
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
                run_lock_sha256=C.sha(out / "run_lock.json"), output_sha256=C.sha(target),
                legacy_post_state_max_gap=gap, center_max_gap=center_gap, same_day_source=bool(same_day),
                first_date=str(result.game_date.min().date()), last_date=str(result.game_date.max().date()),
                seconds=round(time.monotonic() - start, 3), context="neutral_pregame_not_contextual_next_game",
                source_directory=str(source_directory) if source_directory else None,
                parameter_transport_not_refit=source_directory is not None)
    write_json(receipt, meta)
    return meta


def prepare(out, source=None):
    out = Path(out)
    files = [Path(__file__), KERNEL, C.SOURCE / "player_game_lines.parquet", C.OUT / "game_context.parquet", C.AGE]
    files += list((C.OUT / "models").glob("*/*.json"))
    files += [Path(__file__).with_name(name) for name in ["build_monthly_own_skills.py", "monthly_ou_filter.py",
                "monthly_ou_kernel.cpp", "build_nba_estimated_skills.py", "build_nba_estimated_skills_v2_stageb.py",
                "estimated_skills_context.py", "estimated_skills_corrections.py"]]
    if source:
        source = Path(source).resolve()
        manifest = json.loads((source / "manifest.json").read_text())
        for name, expected in manifest["artifacts"].items():
            if C.sha(source / name) != expected:
                raise ValueError("Validated neutral source changed")
            files.append(source / name)
        files.append(source / "manifest.json")
    lock = dict(inputs={str(p): C.sha(p) for p in files}, recipe="original_neutral_annual_ou_pure_query_v1",
                source_directory=str(source) if source else None, parameter_transport_not_refit=bool(source),
                time_precision="date_strictly_earlier_observations", source_publication_vintage_certified=False,
                selected_vpm_changed=False, outcome_seasons_max=2026)
    path = out / "run_lock.json"
    if path.exists() and json.loads(path.read_text()) != lock:
        raise ValueError("Neutral bridge recipe changed; create a new package")
    write_json(path, lock)
    compile_kernel(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--years", default="2019")
    parser.add_argument("--skills", default="ft_pct,three_pct,ast_rate")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--source", type=Path, help="Separately validated source; old annual parameters are transported, not refitted")
    args = parser.parse_args()
    years, skills = list(map(int, args.years.split(","))), args.skills.split(",")
    if not all(2015 <= y <= 2025 for y in years) or not set(skills) <= set(B.SKILL_NAMES):
        parser.error("Only legal 2015–2025 neutral recipes and registered core skills")
    prepare(args.out, args.source)
    write_json(args.out / "run_status.json", dict(status="running", years=years, skills=skills, pid=os.getpid()))
    try:
        tasks = [(str(args.out), year, skill) for year in years for skill in skills]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = [pool.submit(one, task) for task in tasks]
            for future in as_completed(jobs):
                print(json.dumps(future.result()), flush=True)
        write_json(args.out / "run_status.json", dict(status="completed_requested_core_families", years=years, skills=skills,
                                                     all_vpm_families_ready=False, dynamic_mapper_fitted=False))
    except Exception as error:
        write_json(args.out / "run_status.json", dict(status="failed", error=str(error)))
        raise


if __name__ == "__main__":
    main()
