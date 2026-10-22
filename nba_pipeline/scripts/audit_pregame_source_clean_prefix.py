"""Independent complete-adapter and annual-fit prefix audits for clean OU.

All counterfactuals live in new diagnostic packages. Neither original source
files nor admitted fits/states are modified. Fixed-model replay changes only
measured target/future counts; the annual refit audit mutates only observations
and context on/after the annual fit cutoff, preserving masked NaNs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

import build_monthly_own_skills as C
import build_pregame_neutral_skills as neutral
import build_pregame_source_clean_skills as clean

ROOT = clean.ROOT
OUT = ROOT / "results/estimated_skills/pregame_source_clean_prefix_v1"
SKILLS = ["ft_pct", "three_pct", "ast_rate"]


def verify_pins(pins):
    for name, expected in pins.items():
        if C.sha(name) != expected:
            raise ValueError("Independent clean-prefix input changed: " + name)


def clean_model_pins(fit_root, source, years_skills):
    """Require source-clean registered fits, never transported donor models."""
    fit_root, source = Path(fit_root).resolve(), Path(source).resolve()
    fit_lock_path = fit_root / "source_lock.json"
    fit_lock = json.loads(fit_lock_path.read_text())
    if Path(fit_lock["source_directory"]).resolve() != source:
        raise ValueError("Model fit source differs from audited source")
    if not fit_lock["source_clean_refit"] or fit_lock["parameter_transport_not_refit"]:
        raise ValueError("Transported parameters cannot pass a source-clean audit")
    if (fit_lock["maxiter"], fit_lock["popsize"], fit_lock["seed"], fit_lock["tol"], fit_lock["polish"]) != (60, 15, 20260928, 1e-7, False):
        raise ValueError("Audit requires the registered 60-iteration/15-population donor recipe")
    pins = dict(fit_lock["inputs"])
    pins.update({str(source / name): digest for name, digest in fit_lock["source_artifacts"].items()})
    pins[str(source / "manifest.json")] = fit_lock["source_manifest_sha256"]
    pins[str(fit_lock_path)] = C.sha(fit_lock_path)
    for path in [Path(__file__).resolve(), Path(neutral.__file__).resolve(), neutral.KERNEL]:
        pins[str(path)] = C.sha(path)
    for year, skill in years_skills:
        path = fit_root / "models" / str(year) / (skill + ".json")
        model = json.loads(path.read_text())
        if model["source_lock_sha256"] != pins[str(fit_lock_path)] or not model["source_clean_refit"] or model["parameter_transport_not_refit"]:
            raise ValueError("Audit model is not an original source-clean annual fit")
        if pd.Timestamp(model["fit_end"]) >= C.vintage_cut(year) or model["vintage"] != year:
            raise ValueError("Audit model reaches its deployment cutoff")
        if (model["maxiter"], model["popsize"], model["seed"]) != (60, 15, 20260928):
            raise ValueError("Audit model used another search budget")
        pins[str(path)] = C.sha(path)
    verify_pins(pins)
    return pins


def mutate_future(frame, cutoff, fields=("ftm", "three_fgm", "ast")):
    """No masked row is turned into a new observation by a counterfactual."""
    result = frame.copy()
    later = pd.to_datetime(result.game_date).ge(pd.Timestamp(cutoff))
    for name in fields:
        result.loc[later & result[name].notna(), name] = 0.
    return result


@contextmanager
def counterfactual_reads(source, cutoff, active):
    """Override only the diagnostic adapter read; disk bytes stay pinned."""
    source = Path(source).resolve()
    original = pd.read_parquet
    def read(path, *args, **kwargs):
        frame = original(path, *args, **kwargs)
        if active and Path(path).resolve() == source / "player_game_lines.parquet":
            return mutate_future(frame, cutoff)
        return frame
    pd.read_parquet = read
    try:
        yield original
    finally:
        pd.read_parquet = original


def compare_export_prefix(reference, altered, cutoff):
    keys = ["player_id", "game_id", "game_date", "feature"]
    pd.testing.assert_frame_equal(reference[keys], altered[keys], check_exact=True)
    earlier = reference.game_date.le(pd.Timestamp(cutoff))
    if not earlier.any():
        raise ValueError("No earlier/same-day queries exercised")
    fields = ["value", "state_variance", "prior_exposure", "last_observation_day"]
    differences = {}
    for name in fields:
        a, b = reference.loc[earlier, name].to_numpy(), altered.loc[earlier, name].to_numpy()
        if not np.array_equal(a, b, equal_nan=True):
            raise ValueError("Future outcomes changed pregame prefix " + name)
        differences[name] = 0.
    later = reference.game_date.gt(pd.Timestamp(cutoff))
    future_gap = float((reference.loc[later, "value"] - altered.loc[later, "value"]).abs().max())
    if not np.isfinite(future_gap) or future_gap <= 0:
        raise ValueError("Perturbation failed to exercise later state updates")
    return dict(unchanged_rows=int(earlier.sum()), prefix_max_gaps=differences,
                later_state_max_change=future_gap)


def export_audit(fit_root, source, out, year=2019, cutoff="2020-01-22"):
    fit_root, source, out = Path(fit_root).resolve(), Path(source).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError("Use a new immutable diagnostic export-audit directory")
    pins = clean_model_pins(fit_root, source, [(year, skill) for skill in SKILLS])
    records = []
    for mode in ["reference", "counterfactual"]:
        folder = out / mode
        folder.mkdir(parents=True)
        (folder / "models").symlink_to(fit_root / "models", target_is_directory=True)
        lock = dict(source_directory=str(source), fit_directory=str(fit_root),
            source_lock_sha256=C.sha(fit_root / "source_lock.json"),
            model_hashes={str(folder / "models" / str(year) / (skill + ".json")):
                pins[str(fit_root / "models" / str(year) / (skill + ".json"))] for skill in SKILLS},
            inputs=pins, source_clean_refit=True, parameter_transport_not_refit=False,
            diagnostic_only=True, counterfactual_read_override=mode == "counterfactual",
            override_rule="nonmissing target/future FT makes, three makes, assists set to zero",
            cutoff=str(pd.Timestamp(cutoff).date()), outputs_admitted=False)
        C.write_json(folder / "export_lock.json", lock)
        for skill in SKILLS:
            verify_pins(pins)
            with counterfactual_reads(source, cutoff, mode == "counterfactual"):
                clean.export_one((str(folder), year, skill))
            verify_pins(pins)
            path = folder / "families" / str(year) / (skill + ".json")
            receipt = json.loads(path.read_text())
            if receipt["parameter_transport_not_refit"] or not receipt["source_clean_refit"]:
                raise ValueError("Diagnostic replay substituted transported parameters")
            if receipt["model_sha256"] != lock["model_hashes"][str(folder / "models" / str(year) / (skill + ".json"))]:
                raise ValueError("Diagnostic replay used another model")
    for skill in SKILLS:
        frames = [pd.read_parquet(out / mode / "families" / str(year) / (skill + ".parquet"))
                  for mode in ["reference", "counterfactual"]]
        record = compare_export_prefix(*frames, cutoff)
        record.update(skill=skill, model_sha256=pins[str(fit_root / "models" / str(year) / (skill + ".json"))])
        records.append(record)
    verify_pins(pins)
    result = dict(status="passed_complete_source_clean_adapter_prefix", annual_vintage=year,
        cutoff=str(pd.Timestamp(cutoff).date()), source_clean_refit=True,
        model_hashes={skill: pins[str(fit_root / "models" / str(year) / (skill + ".json"))] for skill in SKILLS},
        input_hashes=pins, records=records, original_sources_modified=False,
        counterfactual_exports_admitted=False, protected_2027_outcomes=0)
    C.write_json(out / "receipt.json", result)
    print(json.dumps({k:v for k,v in result.items() if k != "input_hashes"}), flush=True)
    return result


def copy_counterfactual_source(source, out, cutoff):
    """A new separately hashed source copy; the original manifest stays intact."""
    source, out = Path(source).resolve(), Path(out).resolve()
    original = json.loads((source / "manifest.json").read_text())
    if out.exists():
        raise FileExistsError("Counterfactual source must be new")
    out.mkdir(parents=True)
    for name in original["artifacts"]:
        shutil.copyfile(source / name, out / name)
    frame = pd.read_parquet(out / "player_game_lines.parquet")
    altered = mutate_future(frame, cutoff)
    # Change future opportunities too, while retaining original missingness.
    later = pd.to_datetime(altered.game_date).ge(pd.Timestamp(cutoff))
    altered.loc[later & altered.fta.notna(), "fta"] = 2*altered.loc[later & altered.fta.notna(), "fta"] + 1.
    altered.to_parquet(out / "player_game_lines.parquet", index=False)
    context = pd.read_parquet(out / "game_context.parquet")
    dates = frame[["player_id", "game_id", "game_date"]]
    joined = context.merge(dates, on=["player_id", "game_id"], validate="one_to_one")
    later_context = pd.to_datetime(joined.game_date).ge(pd.Timestamp(cutoff))
    if not joined[["player_id", "game_id"]].equals(context[["player_id", "game_id"]]):
        raise ValueError("Context join changed row order")
    known = later_context & context.is_home.notna()
    context.loc[known, "is_home"] = ~context.loc[known, "is_home"].astype(bool)
    context.to_parquet(out / "game_context.parquet", index=False)
    original.update(diagnostic_only=True, counterfactual_source_admitted=False,
        parent_source_directory=str(source), parent_manifest_sha256=C.sha(source / "manifest.json"),
        mutation_cutoff=str(pd.Timestamp(cutoff).date()),
        artifacts={name:C.sha(out/name) for name in original["artifacts"]})
    C.write_json(out / "manifest.json", original)
    return out


def compare_fit_models(reference, altered):
    # Provenance paths/locks/timing naturally differ; every learned quantity,
    # preprocessing choice, objective and finite-search diagnostic must match.
    ignored = {"seconds", "source_lock_sha256"}
    left = {k:v for k,v in reference.items() if k not in ignored}
    right = {k:v for k,v in altered.items() if k not in ignored}
    if left != right:
        changed = sorted(k for k in set(left) | set(right) if left.get(k) != right.get(k))
        raise ValueError("Future observations changed annual fit: " + ", ".join(changed))
    return dict(parameters_identical=True, objective_identical=True,
                stage_a_center_context_corrections_recipe_identical=True,
                objective=float(reference["objective"]), training_rows=int(reference["training_rows"]))


def parameter_audit(fit_root, source, out, year=2015, skill="ft_pct"):
    fit_root, source, out = Path(fit_root).resolve(), Path(source).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError("Use a new immutable diagnostic parameter-audit directory")
    pins = clean_model_pins(fit_root, source, [(year, skill)])
    cutoff = C.vintage_cut(year)
    diagnostic_source = copy_counterfactual_source(source, out / "source", cutoff)
    a, ac = clean.load_panel(source, cutoff)
    b, bc = clean.load_panel(diagnostic_source, cutoff)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    pd.testing.assert_frame_equal(ac, bc, check_exact=True)
    diagnostic_fit = out / "fit"
    clean.prepare(diagnostic_fit, diagnostic_source, maxiter=60, popsize=15)
    verify_pins(pins)
    clean.fit_one((str(diagnostic_fit), year, skill))
    verify_pins(pins)
    original_path = fit_root / "models" / str(year) / (skill + ".json")
    counter_path = diagnostic_fit / "models" / str(year) / (skill + ".json")
    comparison = compare_fit_models(json.loads(original_path.read_text()), json.loads(counter_path.read_text()))
    result = dict(status="passed_annual_clean_parameter_prefix_refit", annual_vintage=year,
        skill=skill, cutoff=str(cutoff.date()), maxiter=60, popsize=15, seed=20260928,
        comparison=comparison, original_model_sha256=C.sha(original_path),
        counterfactual_model_sha256=C.sha(counter_path), input_hashes=pins,
        counterfactual_source_manifest_sha256=C.sha(diagnostic_source / "manifest.json"),
        original_sources_modified=False, counterfactual_fits_admitted=False,
        protected_2027_outcomes=0)
    C.write_json(out / "receipt.json", result)
    print(json.dumps({k:v for k,v in result.items() if k != "input_hashes"}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-root", type=Path, default=clean.OUT)
    parser.add_argument("--source", type=Path, default=clean.SOURCE)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--stage", choices=["export", "parameters", "all"], default="all")
    args = parser.parse_args()
    if args.stage in ["export", "all"]:
        export_audit(args.fit_root, args.source, args.out / "export")
    if args.stage in ["parameters", "all"]:
        parameter_audit(args.fit_root, args.source, args.out / "parameters")
