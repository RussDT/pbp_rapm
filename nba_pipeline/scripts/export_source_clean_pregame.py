"""Locked state exports after the immutable source-clean annual fits finish.

Keep export dependencies separate from the already running fit recipe. The
fit models are read through a checked symlink, never copied over or refitted.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(key, "1")

import pandas as pd

import build_pregame_source_clean_skills as clean
import build_pregame_neutral_skills as neutral
import build_monthly_own_skills as C

ROOT = clean.ROOT
FIT = clean.OUT
OUT = ROOT / "results/estimated_skills/pregame_source_clean_states_v1"


def verify_inputs(lock):
    for name, expected in lock["inputs"].items():
        if C.sha(name) != expected:
            raise ValueError(f"Source-clean export input changed: {name}")


def prepare(fit_root=FIT, out=OUT):
    fit_root, out = Path(fit_root).resolve(), Path(out).resolve()
    fit_lock = json.loads((fit_root / "source_lock.json").read_text())
    source = Path(fit_lock["source_directory"])
    status = json.loads((fit_root / "run_status.json").read_text())
    if status["status"] != "complete" or status["completed"] != 187 or status["tasks"] != 187:
        raise ValueError("All 187 registered source-clean fits must finish first")
    files = list(fit_lock["inputs"])
    files += [str(source / name) for name in fit_lock["source_artifacts"]]
    files += [str(source / "manifest.json"), str(fit_root / "source_lock.json"),
              str(Path(__file__).resolve()), str(Path(neutral.__file__).resolve()), str(neutral.KERNEL)]
    expected = {(year, spec.name) for year in range(2015, 2026) for spec in clean.B.SKILLS}
    models = {}
    for year, name in sorted(expected):
        path = fit_root / "models" / str(year) / (name + ".json")
        model = json.loads(path.read_text())
        if model["source_lock_sha256"] != C.sha(fit_root / "source_lock.json") or not model["source_clean_refit"]:
            raise ValueError("Model belongs to another fit/source recipe")
        if pd.Timestamp(model["fit_end"]) >= C.vintage_cut(year) or model["vintage"] != year:
            raise ValueError("Source-clean annual model chronology failed")
        files.append(str(path))
        models[str(out / "models" / str(year) / (name + ".json"))] = C.sha(path)
    # Verify against the fit's frozen hash values, not only today's file bytes.
    verify_inputs(dict(inputs=fit_lock["inputs"]))
    for name, digest in fit_lock["source_artifacts"].items():
        if C.sha(source / name) != digest:
            raise ValueError("Fitted source changed before state export")
    if C.sha(source / "manifest.json") != fit_lock["source_manifest_sha256"]:
        raise ValueError("Fitted source manifest changed")
    out.mkdir(parents=True, exist_ok=True)
    models_link = out / "models"
    if models_link.is_symlink():
        if models_link.resolve() != fit_root / "models":
            raise ValueError("State model link points at another fit")
    elif models_link.exists():
        raise ValueError("State package cannot own or overwrite fitted models")
    else:
        models_link.symlink_to(fit_root / "models", target_is_directory=True)
    lock = dict(source_directory=str(source), fit_directory=str(fit_root),
        source_lock_sha256=C.sha(fit_root / "source_lock.json"), model_hashes=models,
        inputs={str(Path(name).resolve()): C.sha(name) for name in dict.fromkeys(files)},
        source_clean_refit=True, parameter_transport_not_refit=False, selected_vpm_changed=False,
        protected_2027_outcomes=0, outcome_seasons_max=2026)
    target = out / "export_lock.json"
    if target.exists() and json.loads(target.read_text()) != lock:
        raise ValueError("State export recipe changed; use a new package")
    C.write_json(target, lock)
    neutral.compile_kernel(out)
    return lock


def one(task):
    out, year, name = task
    out = Path(out)
    lock = json.loads((out / "export_lock.json").read_text())
    model = out / "models" / str(year) / (name + ".json")
    if C.sha(model) != lock["model_hashes"][str(model)]:
        raise ValueError("Source-clean model changed before query replay")
    result = clean.export_one((str(out), year, name))
    if C.sha(model) != lock["model_hashes"][str(model)]:
        raise ValueError("Source-clean model changed during query replay")
    receipt = json.loads((out / "families" / str(year) / (name + ".json")).read_text())
    if receipt["model_sha256"] != C.sha(model) or receipt["parameter_transport_not_refit"] or not receipt["source_clean_refit"]:
        raise ValueError("Source-clean replay provenance differs")
    return {key: result[key] for key in ["year", "skill", "status", "rows"] if key in result}


def run(fit_root=FIT, out=OUT, workers=3):
    fit_root, out = Path(fit_root).resolve(), Path(out).resolve()
    lock = prepare(fit_root, out)
    verify_inputs(lock)
    tasks = [(str(out), year, spec.name) for year in range(2015, 2026) for spec in clean.B.SKILLS]
    status = dict(status="running", tasks=len(tasks), completed=0)
    C.write_json(out / "run_status.json", status)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(one, task) for task in tasks]):
            try:
                result = future.result()
            except Exception as exc:
                status.update(status="failed", error=str(exc))
                C.write_json(out / "run_status.json", status)
                raise
            status["completed"] += 1
            C.write_json(out / "run_status.json", status)
            print(json.dumps(result), flush=True)
    verify_inputs(lock)
    status["status"] = "complete"
    C.write_json(out / "run_status.json", status)
    files = list((out / "families").glob("*/*.json"))
    gaps, queries, convergence = [], {}, 0
    for receipt_path in files:
        receipt = json.loads(receipt_path.read_text())
        if receipt["output_sha256"] != C.sha(receipt_path.with_suffix(".parquet")):
            raise ValueError("Source-clean state artifact changed")
        if receipt["run_lock_sha256"] != C.sha(out / "export_lock.json"):
            raise ValueError("Source-clean state lock differs")
        queries.setdefault(receipt["year"], set()).add(receipt["rows"])
        gaps.append(receipt["legacy_post_state_max_gap"])
        model = json.loads((fit_root / "models" / str(receipt["year"]) / (receipt["skill"] + ".json")).read_text())
        convergence += int(model["converged"])
    if len(files) != 187 or len(queries) != 11 or any(len(v) != 1 for v in queries.values()):
        raise ValueError("Incomplete source-clean state family/vintage census")
    summary = dict(status="passed_source_clean_core_exports_pending_independent_prefix_audit",
        family_vintage_exports=len(files), annual_vintages=len(queries),
        player_game_queries=sum(next(iter(v)) for v in queries.values()),
        reference_post_state_max_gap=max(gaps), final_optimizer_converged=convergence,
        final_optimizer_budget_exhausted=187-convergence,
        optimization_claim="fixed_budget_search_not_certified_optimum",
        export_lock_sha256=C.sha(out / "export_lock.json"), source_clean_refit=True,
        dynamic_mapper_fitted=False, selected_vpm_changed=False, protected_2027_outcomes=0)
    C.write_json(out / "completion_receipt.json", summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-root", type=Path, default=FIT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--wait-for-fits", action="store_true")
    args = parser.parse_args()
    if args.wait_for_fits:
        status_path = args.fit_root / "run_status.json"
        while True:
            status = json.loads(status_path.read_text())
            if status["status"] == "complete":
                break
            if status["status"] != "running":
                raise ValueError(f"Annual fit stage stopped: {status}")
            # A dead/stalled process must not leave a supervisor waiting forever.
            if time.time() - status_path.stat().st_mtime > 3600:
                raise ValueError("Annual fit status stale for over an hour")
            time.sleep(20)
    run(args.fit_root, args.out, args.workers)


if __name__ == "__main__":
    main()
