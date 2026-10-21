"""Summarize frozen predictive-stat outputs; never choose a new production model.

The evaluation contract is fixed before full-run scores are inspected. Historical
2020–2025 comparisons are spent research evidence. Game bootstrap intervals
condition on the saved annual filters, coefficients and selection decisions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_full_skill_history import write_json
from full_skill_cross_talk import HALF_LIFE, solve_moments
from full_skill_history_specs import SKILLS
from prepare_full_skill_history import ROOT, sha

YEARS = tuple(range(2020, 2026))
SEED = 20261021
BOOTSTRAPS = 1000
FT_ALPHA = .1
PRIMARY = {
    "ft_pct": ("core.ftm", "core.fta", 1.),
    "arc3_pct": ("core.arc3_fgm", "core.arc3_fga", 1.),
    "corner3_pct": ("core.corner3_fgm", "core.corner3_fga", 1.),
    "tov100": ("core.tov", "off_poss", 100.),
    "oreb100": ("core.oreb", "off_poss", 100.),
    "dreb100": ("core.dreb", "def_poss", 100.),
}


def paired_game_totals(game, truth, denominator, baseline, candidate):
    """Retain paired admissible rows and sufficient loss totals per game."""
    arrays = [np.asarray(x, dtype=float) for x in [truth, denominator, baseline, candidate]]
    truth, denominator, baseline, candidate = arrays
    valid = np.logical_and.reduce([np.isfinite(x) for x in arrays]) & (denominator > 0)
    values = pd.DataFrame({"game_id": np.asarray(game)[valid], "weight": denominator[valid],
                           "baseline": denominator[valid] * (baseline[valid] - truth[valid]) ** 2,
                           "candidate": denominator[valid] * (candidate[valid] - truth[valid]) ** 2})
    return values.groupby("game_id", sort=True)[["weight", "baseline", "candidate"]].sum()


def compare_seasons(seasons, draws=BOOTSTRAPS):
    """Equal-season WMSE, paired game resampling independently within a season."""
    rng = np.random.default_rng(SEED)
    baseline, candidate, boot = [], [], []
    rows = []
    for year, totals in sorted(seasons.items()):
        if totals.empty:
            continue
        weight, left, right = totals[["weight", "baseline", "candidate"]].to_numpy().T
        mass = weight.sum()
        baseline.append(float(left.sum() / mass))
        candidate.append(float(right.sum() / mass))
        # Each draw samples whole games, preserving all paired player rows.
        resamples = rng.multinomial(len(totals), np.full(len(totals), 1. / len(totals)), size=draws)
        boot.append((resamples @ (left - right)) / (resamples @ weight))
        rows.append(dict(year=int(year), games=len(totals), exposure=float(mass),
                         baseline_mse=baseline[-1], candidate_mse=candidate[-1]))
    if not rows:
        return dict(status="no_common_observations")
    lo, hi = np.quantile(np.mean(boot, axis=0), [.025, .975])
    a, b = float(np.mean(baseline)), float(np.mean(candidate))
    return dict(status="scored", seasons=rows, baseline_mse=a, candidate_mse=b,
                mse_gain=a - b, relative_mse_gain=(a - b) / a if a else None,
                conditional_game_bootstrap_95=[float(lo), float(hi)],
                games=sum(x["games"] for x in rows), bootstrap_draws=draws,
                aggregation="equal_season_exposure_weighted_mse",
                uncertainty="paired_games_within_season_conditional_on_saved_models")


def read_source(out, year, columns):
    """Parquet predicate excludes 2026 target rows before materialization."""
    if year > 2025:
        raise ValueError("2026 targets are export-only")
    return pd.read_parquet(Path(out) / "observations.parquet", columns=sorted(set(columns)),
                           filters=[("season", "=", year)]).reset_index(drop=True)


def require_identity(left, right):
    if not np.array_equal(left[["player_id", "game_id"]].to_numpy(), right[["player_id", "game_id"]].to_numpy()):
        raise ValueError("Prediction/source identity order differs")


def forecast_comparisons(out):
    out = Path(out)
    names = [*PRIMARY, "three_fgm100", "ftm100", "points100", "ast100"]
    result = {name: {} for name in names}
    columns = ["player_id", "game_id", "season", *[x[0] for x in PRIMARY.values()],
               "off_poss", "def_poss", "core.fta", "core.ftm"]
    zones = ["at_rim", "short_midrange", "long_midrange", "corner3", "arc3"]
    columns += ["core." + z + suffix for z in zones for suffix in ["_fgm"]]
    columns += ["core.ast_" + z for z in zones]
    for year in YEARS:
        source = read_source(out, year, columns + [x[1] for x in PRIMARY.values()])
        frames = [pd.read_parquet(out / "library" / profile / f"{year}.parquet",
                                  columns=["player_id", "game_id", *names])
                  for profile in ["independent", "cross_stat_research"]]
        for frame in frames:
            require_identity(frame, source)
        targets = {name: (source[num] * scale / source[den], source[den])
                   for name, (num, den, scale) in PRIMARY.items()}
        actual_three = source["core.corner3_fgm"] + source["core.arc3_fgm"]
        actual_two = sum(source["core." + z + "_fgm"] for z in zones[:3])
        derived = {
            "three_fgm100": actual_three,
            "ftm100": source["core.ftm"],
            "points100": 2 * actual_two + 3 * actual_three + source["core.ftm"],
            "ast100": sum(source["core.ast_" + z] for z in zones),
        }
        targets.update({name: (100 * num / source.off_poss, source.off_poss) for name, num in derived.items()})
        for name, (truth, den) in targets.items():
            result[name][year] = paired_game_totals(source.game_id, truth, den, frames[0][name], frames[1][name])
    return {name: compare_seasons(seasons) for name, seasons in result.items()}


def ft_design(own, ft, include_ft):
    own, ft = np.asarray(own), np.asarray(ft)
    values = [np.ones(len(own)), own]
    if include_ft:
        values += [np.nan_to_num(ft, nan=0.), np.isnan(ft).astype(float)]
    return np.column_stack(values)


def ft_association(out):
    """Matched fixed-penalty calibration with/without pregame FT shooting.

    Both arms predict residuals on the independent zone filter. Their only
    difference is adding pregame FT% and its missing flag. Neither arm selects
    its recipe from these reported outcomes; 2026 targets are never loaded.
    """
    out = Path(out)
    destination = out / "diagnostics"
    destination.mkdir(exist_ok=True)
    recipes, comparisons = {}, {}
    for target in ["arc3_pct", "corner3_pct"]:
        spec = next(s for s in SKILLS if s.name == target)
        moments = {arm: (np.zeros((p, p)), np.zeros(p)) for arm, p in [("own_only", 2), ("with_ft", 4)]}
        scores, coefficients, predictions = {}, [], []
        for year in range(1997, 2027):
            baseline = pd.read_parquet(out / "baseline" / f"{year}.parquet",
                                       columns=["player_id", "game_id", "season", target, "ft_pct"])
            own = baseline[target].to_numpy(dtype=float)
            ft = baseline.ft_pct.to_numpy(dtype=float)
            fitted = {}
            betas = {}
            for arm, include in [("own_only", False), ("with_ft", True)]:
                x = ft_design(own, ft, include)
                g, r = moments[arm]
                beta = solve_moments(g, r, FT_ALPHA)
                fitted[arm] = np.clip(own + x @ beta, 0, 1)
                betas[arm] = beta.tolist()
            coefficients.append(dict(year=year, outcome_labels_used_through=min(year - 1, 2025), coefficients=betas))
            if year == 2026:
                predictions.append(baseline[["player_id", "game_id", "season"]].assign(
                    target=target, own_only=fitted["own_only"], with_ft=fitted["with_ft"]))
                continue
            source = read_source(out, year, ["player_id", "game_id", "season", "day", spec.numerator, spec.denominator])
            require_identity(baseline, source)
            num, den = source[spec.numerator].to_numpy(dtype=float), source[spec.denominator].to_numpy(dtype=float)
            valid = np.isfinite(own) & np.isfinite(num) & np.isfinite(den) & (den > 0) & (num >= 0) & (num <= den)
            truth = np.divide(num, den, out=np.full(len(num), np.nan), where=valid)
            if year in YEARS:
                scores[year] = paired_game_totals(source.game_id, truth, den, fitted["own_only"], fitted["with_ft"])
            w = den[valid] * np.exp2((source.day.to_numpy()[valid] - 21000.) / HALF_LIFE)
            for arm, include in [("own_only", False), ("with_ft", True)]:
                x = ft_design(own, ft, include)[valid]
                g, r = moments[arm]
                g += x.T @ (w[:, None] * x)
                r += x.T @ (w * (truth[valid] - own[valid]))
        pd.concat(predictions, ignore_index=True).to_parquet(destination / f"ft_association_{target}_2026.parquet", index=False)
        comparisons[target] = compare_seasons(scores)
        recipes[target] = coefficients
    write_json(destination / "ft_association_coefficients.json", dict(alpha=FT_ALPHA, half_life_days=HALF_LIFE,
                purpose="fixed_recipe_predictive_association_diagnostic_not_production_selection", recipes=recipes))
    return comparisons


def head_summary(out):
    data = pd.read_csv(Path(out) / "cross_stat/historical_scores.csv")
    if data.year.max() > 2025:
        raise ValueError("Historical scores contain protected export-only labels")
    recent = data.loc[data.year.isin(YEARS)].copy()
    recent["relative_mse_gain"] = (recent.baseline - recent.selected) / recent.baseline.replace(0, np.nan)
    families = {s.name: s.family for s in SKILLS}
    heads = recent.groupby("skill", sort=True).agg(seasons=("year", "nunique"),
                baseline_mse=("baseline", "mean"), candidate_mse=("selected", "mean"),
                mean_season_relative_gain=("relative_mse_gain", "mean")).reset_index()
    heads["family"] = heads.skill.map(families)
    heads["mse_gain"] = heads.baseline_mse - heads.candidate_mse
    heads.to_csv(Path(out) / "diagnostics/head_comparisons_2020_2025.csv", index=False)
    # Different units never enter a pooled raw MSE across heads.
    families = heads.groupby("family", sort=True).agg(heads=("skill", "size"),
                positive_gain_heads=("mse_gain", lambda x: int((x > 0).sum())),
                equal_gain_heads=("mse_gain", lambda x: int((x == 0).sum())),
                negative_gain_heads=("mse_gain", lambda x: int((x < 0).sum())),
                median_head_mean_season_relative_gain=("mean_season_relative_gain", "median")).reset_index()
    families.to_csv(Path(out) / "diagnostics/family_comparisons_2020_2025.csv", index=False)
    return json.loads(families.to_json(orient="records"))


def run(out=ROOT):
    out = Path(out)
    destination = out / "diagnostics"
    destination.mkdir(exist_ok=True)
    contract = dict(code_sha256=sha(__file__), baseline_lock_sha256=sha(out / "run_lock.json"),
                    years=list(YEARS), bootstrap_draws=BOOTSTRAPS, seed=SEED, ft_alpha=FT_ALPHA,
                    ft_half_life_days=HALF_LIFE, production_models_changed=False, outcome_labels_through=2025)
    lock = destination / "run_lock.json"
    if lock.exists() and json.loads(lock.read_text()) != contract:
        raise ValueError("Diagnostic recipe changed; preserve the old diagnostic package")
    write_json(lock, contract)
    comparisons = forecast_comparisons(out)
    ft = ft_association(out)
    families = head_summary(out)
    fits = [json.loads(p.read_text()) for p in sorted((out / "models").glob("*/*.json"))]
    annual = [m for m in fits if m["status"] == "annual_past_only_fit"]
    summary = dict(status="completed_research_diagnostics", contract=contract,
                   forecasts=comparisons, ft_predictive_association=ft, families=families,
                   model_slots=len(fits), fitted_slots=len(annual), bootstrap_slots=len(fits) - len(annual),
                   optimizer_converged_slots=sum(m.get("converged", False) for m in annual),
                   optimizer_budget_exhausted_slots=sum(not m.get("converged", False) for m in annual),
                   causal_effects_identified=False, vpm_admission=False, protected_2027_outcomes=0)
    write_json(destination / "research_summary.json", summary)
    lines = ["# Full predictive-stat library: historical results", "",
             "These are retrospective 2020–2025 research comparisons. Both library profiles remain available. "
             "2026 supplies exports only. Positive MSE gain favors the candidate. Intervals resample paired "
             "games within season and condition on saved models and their selection; they do not include "
             "parameter/selection uncertainty. Neither these results nor a nonzero coefficient identifies causation or admits an input to VPM.", "",
             "| Forecast | Independent MSE | Cross-stat MSE | MSE gain [conditional 95% interval] |", "|---|---:|---:|---:|"]
    for name, result in comparisons.items():
        lo, hi = result["conditional_game_bootstrap_95"]
        lines.append(f"| {name} | {result['baseline_mse']:.6g} | {result['candidate_mse']:.6g} | {result['mse_gain']:.6g} [{lo:.6g}, {hi:.6g}] |")
    lines += ["", "Percentage targets use fractions; rate targets use counts/points per 100 side-specific possessions. "
              "Each reported MSE averages the six season-specific exposure-weighted MSEs equally. Raw MSEs across different units are never pooled.", "",
              "## Does pregame FT% help predict threes?", "",
              "The matched diagnostic adds pregame FT% and its missing flag to a calibrated own-zone forecast. "
              "Both arms use fixed ridge alpha 0.1, older annual out-of-time features and a 1460-day calibration half-life; "
              "this does not isolate FT's increment within the full 166-head cross-stat model.", "",
              "| Target | Own-only calibrated MSE | With FT MSE | MSE gain [conditional 95% interval] |", "|---|---:|---:|---:|"]
    for name, result in ft.items():
        lo, hi = result["conditional_game_bootstrap_95"]
        lines.append(f"| {name} | {result['baseline_mse']:.6g} | {result['candidate_mse']:.6g} | {result['mse_gain']:.6g} [{lo:.6g}, {hi:.6g}] |")
    lines += ["", f"Optimizer budget: {len(annual)} fitted annual recipes; "
              f"{summary['optimizer_converged_slots']} reported convergence and "
              f"{summary['optimizer_budget_exhausted_slots']} exhausted the fixed budget without convergence. "
              "The latter remain usable fixed-budget forecasts, without an optimum claim. "
              f"The remaining {summary['bootstrap_slots']} annual slots use explicit bootstrap recipes.", "",
              "Machine-readable detail: research_summary.json, head_comparisons_2020_2025.csv, "
              "family_comparisons_2020_2025.csv and ft_association_coefficients.json in this directory."]
    (destination / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(status=summary["status"], model_slots=len(fits), comparisons=len(comparisons))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT)
    run(parser.parse_args().out)
