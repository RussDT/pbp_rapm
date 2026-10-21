"""Chronological independent OU baseline for the complete historical library.

Annual parameters use strictly earlier ending seasons. 1997 and each source's
first available year are explicitly labeled bootstrap, never fitted backward.
All past valid measurements feed each replay. New games are query rows even
when that skill has zero attempts or no measured tracking observation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

for _variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(_variable, "1")

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from full_skill_history_specs import SKILLS
from prepare_full_skill_history import ROOT, sha

KERNEL = Path(__file__).with_name("full_skill_history_kernel.cpp")
PARAMETERS = ["tau_mu_days", "variance_mu", "tau_form_days", "variance_form", "offseason_variance", "observation_noise"]
PANEL = None


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def compile_kernel(out):
    directory = Path(out) / "kernel"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("full_skill_" + sha(KERNEL)[:16] + ".so")
    if not path.exists():
        subprocess.run(["c++", "-O3", "-std=c++11", "-shared", "-fPIC", str(KERNEL), "-o", str(path)], check=True)
    return path


class Engine:
    def __init__(self, panel, spec, library):
        self.n = len(panel)
        self.ids = np.ascontiguousarray(pd.factorize(panel.player_id)[0], dtype=np.int64)
        self.nplayers = int(self.ids.max()) + 1
        self.day = np.ascontiguousarray(panel.day, dtype=float)
        self.season = np.ascontiguousarray(panel.season, dtype=float)
        numerator = panel[spec.numerator].to_numpy(dtype=float)
        denominator = panel[spec.denominator].to_numpy(dtype=float)
        valid = np.isfinite(numerator) & np.isfinite(denominator) & (numerator >= 0) & (denominator > 0)
        valid &= panel.season.between(spec.first_season, spec.last_season).to_numpy()
        if spec.kind == "pct":
            valid &= numerator <= denominator
        self.exposure = np.ascontiguousarray(np.where(valid, denominator, 0.0))
        self.y = np.ascontiguousarray(np.divide(numerator, denominator, out=np.full(self.n, np.nan), where=valid))
        self.pct = int(spec.kind == "pct")
        self.library = ctypes.CDLL(str(library))
        self.fn = self.library.skill_history
        double = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
        integer = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS")
        self.fn.argtypes = [ctypes.c_long, ctypes.c_long, integer, double, double, double, double,
                            double, ctypes.c_double, ctypes.c_int, *([double] * 6)]
        self.fn.restype = None

    def run(self, parameters, mean):
        arrays = [np.empty(self.n, dtype=float) for _ in range(6)]
        self.fn(self.n, self.nplayers, self.ids, self.day, self.season,
                self.exposure, self.y, np.ascontiguousarray(parameters, dtype=float), float(mean), self.pct, *arrays)
        return arrays


def initialize(path):
    global PANEL
    PANEL = pd.read_parquet(Path(path) / "observations.parquet")


def fit_one(task):
    out, name, year, maxiter, popsize = task
    out = Path(out)
    spec = next(s for s in SKILLS if s.name == name)
    destination = out / "models" / str(year) / (name + ".json")
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved.get("run_lock_sha256") != sha(out / "run_lock.json"):
            raise ValueError("Cached annual model belongs to a different run recipe")
        return dict(skill=name, year=year, status="cached")
    start = time.monotonic()
    global PANEL
    if PANEL is None:
        initialize(out)
    train = PANEL.loc[PANEL.season.lt(year)].reset_index(drop=True)
    target = PANEL.loc[PANEL.season.eq(year)]
    library = compile_kernel(out)
    bootstrap = [1500., .01, 30., .002, .002, .25 if spec.kind == "pct" else 1.]
    payload = dict(skill=name, year=year, kind=spec.kind, parameter_names=PARAMETERS,
                   training_end_season=year - 1, target_first_date=str(target.game_date.min().date()),
                   seed=20261021, maxiter=maxiter, popsize=popsize, kernel_sha256=sha(KERNEL),
                   run_lock_sha256=sha(out / "run_lock.json"),
                   status="bootstrap", parameters=bootstrap, mean=spec.initial, objective_rows=0)
    if len(train):
        engine = Engine(train, spec, library)
        valid = engine.exposure > 0
        if valid.any():
            mean = float(np.average(engine.y[valid], weights=engine.exposure[valid]))
            payload["mean"] = mean
            payload["last_training_observation"] = str(train.loc[valid, "game_date"].max().date())
            if train.loc[valid, "game_date"].max() >= target.game_date.min():
                raise ValueError("Parameters cross the target-season date fence")
            initial = engine.run(bootstrap, mean)
            # All completed earlier seasons may train parameters. Historical
            # outer-season forecasts remain strictly out of time; these are
            # internal parameter-fit rows, not exported backdated forecasts.
            fit = valid & (initial[4] > 0)
            payload["objective_rows"] = int(fit.sum())
            if fit.sum() >= 500:
                y, w = engine.y[fit], engine.exposure[fit]
                variance = max(float(np.average((y - mean) ** 2, weights=w)), 1e-8)
                noise = max(variance * float(w.mean()), 1e-8)
                bounds = [(5., 73000.), (1e-10, 4 * variance), (.2, 400.),
                          (1e-12, 2 * variance), (0., 2 * variance), (.001 * noise, 200 * noise)]
                def loss(parameters):
                    prediction = engine.run(parameters, mean)[0][fit]
                    return float(np.average((prediction - y) ** 2, weights=w))
                result = differential_evolution(loss, bounds, seed=20261021, maxiter=maxiter,
                                                popsize=popsize, tol=1e-7, polish=False)
                payload.update(status="annual_past_only_fit", parameters=result.x.tolist(), objective=float(result.fun),
                               converged=bool(result.success), evaluations=int(result.nfev), iterations=int(result.nit),
                               training_rows=len(train), training_players=int(train.player_id.nunique()))
    payload["seconds"] = round(time.monotonic() - start, 3)
    write_json(destination, payload)
    return dict(skill=name, year=year, status=payload["status"], seconds=payload["seconds"], rows=payload.get("training_rows", 0))


def replay_year(out, year, skills=None):
    out = Path(out)
    global PANEL
    if PANEL is None:
        initialize(out)
    full = PANEL.loc[PANEL.season.le(year)].reset_index(drop=True)
    take = full.season.eq(year).to_numpy()
    result = full.loc[take, ["player_id", "game_id", "game_date", "season", "playoffs", "team_id", "opponent_id", "is_home", "seconds_pbp"]].copy().reset_index(drop=True)
    library = compile_kernel(out)
    columns = {}
    for spec in SKILLS:
        if skills and spec.name not in skills:
            continue
        payload = json.loads((out / "models" / str(year) / (spec.name + ".json")).read_text())
        engine = Engine(full, spec, library)
        if not engine.exposure.any() and spec.numerator.startswith("report."):
            n = int(take.sum())
            columns[spec.name] = np.full(n, np.nan, dtype=np.float32)
            columns[spec.name + "__post"] = np.full(n, np.nan, dtype=np.float32)
            columns[spec.name + "__prior_exposure"] = np.zeros(n, dtype=np.float32)
            columns[spec.name + "__last_day"] = np.full(n, -1, dtype=np.int32)
            columns[spec.name + "__observed"] = np.zeros(n, dtype=bool)
            continue
        values = engine.run(payload["parameters"], payload["mean"])
        # Report availability must itself be pregame information. No current-day
        # observation can switch on its own forecast. Core schema has explicit
        # expert bootstrap priors; report families require an earlier source day.
        observed_days = engine.day[engine.exposure > 0]
        first_day = observed_days.min() if len(observed_days) else np.inf
        available = np.ones(engine.n, dtype=bool) if spec.numerator.startswith("core.") else engine.day > first_day
        available &= full.season.between(spec.first_season, spec.last_season).to_numpy()
        scale = 1. if spec.kind == "pct" else 100.
        columns[spec.name] = np.where(available[take], values[0][take] * scale, np.nan).astype(np.float32)
        columns[spec.name + "__post"] = np.where(available[take], values[1][take] * scale, np.nan).astype(np.float32)
        columns[spec.name + "__prior_exposure"] = values[3][take].astype(np.float32)
        columns[spec.name + "__last_day"] = values[5][take].astype(np.int32)
        columns[spec.name + "__observed"] = (engine.exposure[take] > 0)
    result = pd.concat([result, pd.DataFrame(columns)], axis=1)
    path = out / "baseline" / (str(year) + ".parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(path, index=False)
    return dict(year=year, rows=len(result), columns=len(result.columns), seconds=None)


def ensure_run_lock(out, maxiter, popsize):
    out = Path(out)
    lock = dict(observations_sha256=sha(out / "observations.parquet"), registry_sha256=sha(out / "registry.json"),
                code_sha256=sha(__file__), kernel_sha256=sha(KERNEL), maxiter=maxiter, popsize=popsize,
                seed=20261021, seasons=list(range(1997, 2027)), target="next_game_exposure_weighted_mse",
                context="context_free", selected_vpm_changed=False, protected_2027_outcomes=0)
    path = out / "run_lock.json"
    if path.exists() and json.loads(path.read_text()) != lock:
        raise ValueError("Run recipe changed: create a new immutable output directory")
    write_json(path, lock)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT)
    parser.add_argument("--stage", choices=["fit", "replay", "all", "benchmark"], default="all")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--maxiter", type=int, default=30)
    parser.add_argument("--popsize", type=int, default=10)
    parser.add_argument("--skills", default="")
    parser.add_argument("--years", default="")
    args = parser.parse_args()
    skills = [s for s in SKILLS if not args.skills or s.name in args.skills.split(",")]
    years = [int(x) for x in args.years.split(",")] if args.years else list(range(1997, 2027))
    if any(y < 1997 or y > 2026 for y in years):
        parser.error("Only ending seasons 1997 through 2026 are admitted")
    ensure_run_lock(args.out, args.maxiter, args.popsize)
    compile_kernel(args.out)
    if args.stage in {"fit", "all", "benchmark"}:
        tasks = [(str(args.out), s.name, y, args.maxiter, args.popsize) for y in years for s in skills]
        if args.stage == "benchmark":
            tasks = [(str(args.out), s.name, y, args.maxiter, args.popsize) for y, s in [(2005, skills[0]), (2026, skills[0]), (2026, next(x for x in SKILLS if x.name == "screen_assists100"))]]
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(str(args.out),)) as pool:
            jobs = {pool.submit(fit_one, task): task for task in tasks}
            for count, future in enumerate(as_completed(jobs), 1):
                result = future.result()
                print(json.dumps(dict(completed=count, total=len(tasks), **result)), flush=True)
    if args.stage in {"replay", "all"}:
        initialize(args.out)
        for year in years:
            print(json.dumps(replay_year(args.out, year, {s.name for s in skills})), flush=True)


if __name__ == "__main__":
    main()
