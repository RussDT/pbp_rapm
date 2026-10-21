"""Past-only cross-stat residual ridge, entered through the next-game lab.

Every input is an annual-vintage pregame baseline forecast, never the current
game's box score. Candidate coefficients and their selection use older seasons.
This is a new research library, not a replacement of existing exported skills.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

for _variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(_variable, "1")

import numpy as np
import pandas as pd

from build_full_skill_history import write_json
from full_skill_history_specs import SKILLS
from prepare_full_skill_history import ROOT, sha

ALPHAS = (.1, 1., 10.)
HALF_LIFE = 1460.


def solve_moments(gram, rhs, alpha):
    """Weighted training-only standardization and ridge with a free intercept."""
    n = gram.shape[0] - 1
    if gram[0, 0] <= 0:
        return np.zeros(n + 1)
    mass = gram[0, 0]
    mean = gram[0, 1:] / mass
    variance = np.maximum(np.diag(gram)[1:] / mass - mean ** 2, 0)
    active = variance > 1e-12
    index = np.flatnonzero(active)
    output = np.zeros(n + 1)
    ymean = rhs[0] / mass
    output[0] = ymean
    if len(index):
        scale = np.sqrt(variance[index])
        covariance = (gram[1:, 1:][np.ix_(index, index)] / mass - np.outer(mean[index], mean[index])) / np.outer(scale, scale)
        target = (rhs[1:][index] / mass - mean[index] * ymean) / scale
        coefficients = np.linalg.solve(covariance + alpha * np.eye(len(index)), target) / scale
        output[1 + index] = coefficients
        output[0] -= mean[index] @ coefficients
    return output


def design(frame):
    values = frame[[s.name for s in SKILLS]].to_numpy(dtype=float)
    return np.column_stack([np.ones(len(frame)), np.nan_to_num(values, nan=0.), np.isnan(values).astype(float)])


def choose_alpha(prior_scores, year):
    prior = [x for x in prior_scores if x["year"] < year]
    if len(prior) < 3:
        return None
    weights = np.exp2(-(year - np.array([x["year"] for x in prior])) / 4.)
    labels = ["baseline", *map(str, ALPHAS)]
    losses = {label: np.average([x[label] for x in prior], weights=weights) for label in labels}
    # Null wins exact ties. Selected alpha is recorded before current labels.
    winner = min(labels, key=lambda label: (losses[label], labels.index(label)))
    return None if winner == "baseline" else float(winner)


def run(out=ROOT, workers=6):
    out = Path(out)
    destination = out / "cross_stat"
    destination.mkdir(parents=True, exist_ok=True)
    feature_names = ["intercept", *[s.name for s in SKILLS], *[s.name + "__missing" for s in SKILLS]]
    lock = dict(baseline_lock_sha256=sha(out / "run_lock.json"), code_sha256=sha(__file__),
                feature_names=feature_names, alphas=ALPHAS, training_half_life_days=HALF_LIFE,
                selection="prior_out_of_time_equal_season_loss_with_four_year_decay_minimum_three_seasons",
                score_through=2025, export_through=2026, selected_vpm_changed=False)
    lock_path = destination / "run_lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != json.loads(json.dumps(lock)):
        raise ValueError("Cross-stat recipe changed; use a new immutable package")
    write_json(lock_path, lock)
    p = len(feature_names)
    gram = np.zeros((len(SKILLS), p, p))
    rhs = np.zeros((len(SKILLS), p))
    history = {s.name: [] for s in SKILLS}
    observations = pd.read_parquet(out / "observations.parquet")
    summary = []
    for year in range(1997, 2027):
        file = out / "baseline" / f"{year}.parquet"
        baseline = pd.read_parquet(file, columns=["player_id", "game_id", "season", "game_date", *[s.name for s in SKILLS]])
        source = observations.loc[observations.season.eq(year)].copy().reset_index(drop=True)
        if not np.array_equal(baseline[["player_id", "game_id"]].to_numpy(), source[["player_id", "game_id"]].to_numpy()):
            raise ValueError("Pregame features and observation identity/order differ")
        x = design(baseline)
        output = baseline[["player_id", "game_id", "season", "game_date"]].copy()
        output_values = {}
        chosen = {s.name: choose_alpha(history[s.name], year) for s in SKILLS}
        write_json(destination / "selection" / f"{year}.json", dict(year=year, choices=chosen, outcome_labels_used_through=year - 1))
        def one(index):
            spec = SKILLS[index]
            base = baseline[spec.name].to_numpy(dtype=float)
            predictions = {}
            models = {}
            for alpha in ALPHAS:
                beta = solve_moments(gram[index], rhs[index], alpha)
                value = base + x @ beta
                predictions[str(alpha)] = np.clip(value, 0, 1) if spec.kind == "pct" else np.maximum(value, 0)
                models[str(alpha)] = beta.tolist()
            selected = base if chosen[spec.name] is None else predictions[str(chosen[spec.name])]
            # 2026 is filtering/export only. Its target values never enter cross
            # fitting, hyperparameter choice or error summaries.
            score = None
            update_g = update_r = None
            if year <= 2025:
                numerator = source[spec.numerator].to_numpy(dtype=float)
                denominator = source[spec.denominator].to_numpy(dtype=float)
                valid = np.isfinite(base) & np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0) & (numerator >= 0)
                valid &= source.season.between(spec.first_season, spec.last_season).to_numpy()
                if spec.kind == "pct":
                    valid &= numerator <= denominator
                if valid.any():
                    scale = 1. if spec.kind == "pct" else 100.
                    y = scale * numerator[valid] / denominator[valid]
                    days = source.loc[valid, "day"].to_numpy(dtype=float)
                    w = denominator[valid] * np.exp2((days - 21000.) / HALF_LIFE)
                    a = x[valid]
                    update_g = a.T @ (w[:, None] * a)
                    update_r = a.T @ (w * (y - base[valid]))
                    score = {"year": year, "rows": int(valid.sum()), "baseline": float(np.average((base[valid] - y) ** 2, weights=denominator[valid])),
                             "selected": float(np.average((selected[valid] - y) ** 2, weights=denominator[valid]))}
                    for label, values in predictions.items():
                        score[label] = float(np.average((values[valid] - y) ** 2, weights=denominator[valid]))
            return index, selected, models, score, update_g, update_r
        all_models = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, selected, models, score, g, r in pool.map(one, range(len(SKILLS))):
                name = SKILLS[index].name
                output_values[name] = selected.astype(np.float32)
                all_models[name] = models
                if score is not None:
                    history[name].append(score)
                    summary.append(dict(skill=name, **score))
                if g is not None:
                    gram[index] += g
                    rhs[index] += r
        prediction_path = destination / "predictions" / f"{year}.parquet"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        output = pd.concat([output, pd.DataFrame(output_values)], axis=1)
        output.to_parquet(prediction_path, index=False)
        write_json(destination / "models" / f"{year}.json", dict(year=year, trained_through=year - 1,
                                                                  feature_names=feature_names, coefficients=all_models))
        print(json.dumps(dict(stage="cross_stat", year=year, rows=len(output), cross_selected=sum(a is not None for a in chosen.values()))), flush=True)
    pd.DataFrame(summary).to_csv(destination / "historical_scores.csv", index=False)
    write_json(destination / "receipt.json", dict(status="completed_research_cross_stat_library", seasons=list(range(1997, 2027)),
                                                    skills=len(SKILLS), scored_through=2025, export_only_2026=True,
                                                    causal_effects_identified=False, calibrated_uncertainty=False,
                                                    joint_makes_gate_passed=False, vpm_admission=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    run(args.out, args.workers)


if __name__ == "__main__":
    main()
