#!/usr/bin/env python3
"""
Run a historical core six-factor weighted-factors build.

This is intentionally separate from 03_run_rapm_analysis.py so the daily
pipeline keeps its modern auxiliary metric expectations. It uses only:
RAPM, TS, TOV, REB, BADPASS_TOV, and SCORING_TOV.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

import rapm as rapm_solver


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PIPELINE_ROOT / "results"
MASTER_RESULTS_DIR = PIPELINE_ROOT / "master_results"


def expand_cli_year_range(start_year: int, end_year: int) -> list[int]:
    start = int(start_year)
    end = int(end_year)
    if start <= end:
        return list(range(start, end + 1))
    return list(range(start, 100)) + list(range(0, end + 1))


def normalize_season_end_year(year_value: int) -> int:
    year_value = int(year_value)
    if year_value >= 100:
        return year_value
    return 2000 + year_value if year_value <= 50 else 1900 + year_value


def year_range_label(start_year: int, end_year: int) -> str:
    return f"{int(start_year):02d}" if int(start_year) == int(end_year) else f"{int(start_year):02d}_{int(end_year):02d}"


def season_suffix(season_type: str) -> str:
    return season_type.lower()


def build_run_suffix(
    season_type: str,
    rubberband: bool,
    season_effects: bool,
    fixed_season_effects: bool,
    age_poly_degree: int | None,
) -> str:
    parts = [season_suffix(season_type)]
    if rubberband:
        parts.append("rb")
    if season_effects:
        parts.append("se")
    if fixed_season_effects:
        parts.append("fse")
    if age_poly_degree is not None:
        parts.append(f"agepoly{age_poly_degree}")
    return "_".join(parts)


def input_files_for(prefix: str, years: list[int], season_type: str, *, existing_only: bool = False) -> list[str]:
    file_prefix = {
        "BADPASS_TOV": "TOV",
        "SCORING_TOV": "TOV",
    }.get(prefix, prefix)
    files: list[str] = []
    for year in years:
        yy = f"{year % 100:02d}"
        if season_type in {"RS", "ALL"}:
            filename = f"{file_prefix}{yy}.parquet"
            if not existing_only or (rapm_solver.PROCESSED_DIR / filename).exists():
                files.append(filename)
        if season_type in {"PS", "ALL"}:
            filename = f"{file_prefix}{yy}_PS.parquet"
            if not existing_only or (rapm_solver.PROCESSED_DIR / filename).exists():
                files.append(filename)
    return files


def validate_inputs(prefixes: list[str], years: list[int], season_type: str) -> None:
    missing: list[str] = []
    skipped_playoffs: set[str] = set()
    for prefix in prefixes:
        for filename in input_files_for(prefix, years, season_type):
            path = rapm_solver.PROCESSED_DIR / filename
            if not path.exists():
                if season_type == "ALL" and filename.endswith("_PS.parquet"):
                    skipped_playoffs.add(filename)
                    continue
                missing.append(str(path))
    if missing:
        sample = "\n".join(missing[:40])
        extra = "" if len(missing) <= 40 else f"\n... and {len(missing) - 40} more"
        raise FileNotFoundError(f"Missing required processed parquets:\n{sample}{extra}")
    if skipped_playoffs:
        sample = ", ".join(sorted(skipped_playoffs)[:12])
        extra = "" if len(skipped_playoffs) <= 12 else f", ... and {len(skipped_playoffs) - 12} more"
        logging.warning("Skipping missing playoff files in ALL run: %s%s", sample, extra)


def expected_solver_output(
    prefix: str,
    years: list[int],
    season_type: str,
    pure: bool,
    rubberband: bool,
    season_effects: bool,
    fixed_season_effects: bool,
    age_poly_degree: int | None,
) -> Path:
    file_type = prefix.lower()
    year_tokens = [f"{year % 100:02d}" for year in years]
    years_sorted = sorted(set(year_tokens))
    year_range = years_sorted[0] if len(years_sorted) == 1 else f"{years_sorted[0]}_{years_sorted[-1]}"
    suffix_parts = [season_suffix(season_type)]
    if pure:
        suffix_parts.append("pure")
    if rubberband:
        suffix_parts.append("rb")
    if season_effects:
        suffix_parts.append("se")
    if fixed_season_effects:
        suffix_parts.append("fse")
    if age_poly_degree is not None:
        suffix_parts.append(f"agepoly{age_poly_degree}")
    return RESULTS_DIR / f"{file_type}_{year_range}_{'_'.join(suffix_parts)}_results.csv"


def run_metric(
    prefix: str,
    years: list[int],
    season_type: str,
    label: str,
    output_dir: Path,
    cores: int,
    *,
    rubberband: bool,
    season_effects: bool,
    fixed_season_baselines: dict[int, float] | None,
    age_poly_coefficients: Path | None,
    age_poly_degree: int | None,
    darko_history_path: Path,
) -> pd.DataFrame:
    pure = prefix == "RAPM"
    rapm_solver.N_CORES = int(cores)
    files = input_files_for(prefix, years, season_type, existing_only=(season_type == "ALL"))
    logging.info("Running %s on %d files", prefix, len(files))
    result = rapm_solver.run_simplified_rapm(
        files,
        pure=pure,
        rubberband=rubberband,
        season_effects=season_effects,
        fixed_season_baselines=fixed_season_baselines,
        age_poly_coefficients=age_poly_coefficients,
        age_poly_reference_age=rapm_solver.AGE_DUMMY_REFERENCE,
        darko_history_path=darko_history_path,
        start_year=years[0],
        end_year=years[-1],
        prefix=prefix,
    )
    if result is None or result.empty:
        raise RuntimeError(f"{prefix} solve produced no results")

    correct_suffix = f"{label}_{season_suffix(season_type)}"
    if prefix == "RAPM":
        correct_suffix = f"{label}_{season_suffix(season_type)}_pure"
    if rubberband:
        correct_suffix = f"{correct_suffix}_rb"
    if season_effects:
        correct_suffix = f"{correct_suffix}_se"
    if fixed_season_baselines is not None:
        correct_suffix = f"{correct_suffix}_fse"
    if age_poly_degree is not None:
        correct_suffix = f"{correct_suffix}_agepoly{age_poly_degree}"
    correct_path = output_dir / f"{prefix.lower()}_{correct_suffix}_results.csv"
    result.to_csv(correct_path, index=False)
    logging.info("Saved %s", correct_path)

    generated = expected_solver_output(
        prefix,
        years,
        season_type,
        pure=pure,
        rubberband=rubberband,
        season_effects=season_effects,
        fixed_season_effects=fixed_season_baselines is not None,
        age_poly_degree=age_poly_degree,
    )
    if generated.exists() and generated.resolve() != correct_path.resolve():
        generated.unlink()
    generated_se = generated.with_name(generated.stem.replace("_results", "_season_effects") + ".csv")
    if generated_se.exists():
        se_dest = output_dir / f"{prefix.lower()}_{correct_suffix}_season_effects.csv"
        shutil.move(str(generated_se), se_dest)
        logging.info("Moved %s", se_dest)
    generated_rb_coeffs = generated.with_name(generated.stem.replace("_results", "_rubberband_coefficients") + ".csv")
    if generated_rb_coeffs.exists():
        rb_dest = output_dir / f"{prefix.lower()}_{correct_suffix}_rubberband_coefficients.csv"
        shutil.move(str(generated_rb_coeffs), rb_dest)
        logging.info("Moved %s", rb_dest)
    generated_rb_effects = generated.with_name(generated.stem.replace("_results", "_rubberband_effects") + ".csv")
    if generated_rb_effects.exists():
        rb_dest = output_dir / f"{prefix.lower()}_{correct_suffix}_rubberband_effects.csv"
        shutil.move(str(generated_rb_effects), rb_dest)
        logging.info("Moved %s", rb_dest)
    generated_fixed_offsets = generated.with_name(generated.stem.replace("_results", "_fixed_offsets") + ".csv")
    if generated_fixed_offsets.exists():
        fixed_dest = output_dir / f"{prefix.lower()}_{correct_suffix}_fixed_offsets.csv"
        shutil.move(str(generated_fixed_offsets), fixed_dest)
        logging.info("Moved %s", fixed_dest)
    return result


def parse_year_from_processed_filename(filename: str) -> int:
    stem = Path(filename).stem.replace("_PS", "")
    digits = "".join(ch for ch in stem if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not parse year from {filename}")
    return int(digits[-2:])


def load_age_curve_coefficients(path: Path, prefixes: list[str]) -> tuple[dict[tuple[str, str], np.poly1d], int]:
    coeffs = pd.read_csv(path)
    required = {"metric", "side", "degree"}
    missing = required - set(coeffs.columns)
    if missing:
        raise ValueError(f"Age polynomial coefficient file missing columns: {sorted(missing)}")

    curves: dict[tuple[str, str], np.poly1d] = {}
    degrees = set()
    for _, row in coeffs.iterrows():
        metric = str(row["metric"]).upper()
        side = str(row["side"]).lower()
        if metric not in prefixes or side not in {"off", "def"}:
            continue
        degree = int(row["degree"])
        degrees.add(degree)
        values = [float(row[f"coef_age_pow_{power}"]) for power in range(degree, -1, -1)]
        curves[(metric, side)] = np.poly1d(values)

    expected = {(prefix, side) for prefix in prefixes for side in ("off", "def")}
    missing_curves = expected - set(curves)
    if missing_curves:
        raise ValueError(f"Age polynomial coefficient file missing curves: {sorted(missing_curves)}")
    if len(degrees) != 1:
        raise ValueError(f"Expected one polynomial degree, found {sorted(degrees)}")
    return curves, degrees.pop()


def compute_fixed_rs_season_baselines(prefixes: list[str], years: list[int]) -> tuple[dict[str, dict[int, float]], pd.DataFrame]:
    """
    Compute fixed season constants from regular-season processed parquets.

    Aliased prefixes such as BADPASS_TOV and SCORING_TOV are computed from the
    TOV parquet after rapm.py's prefix-specific target preparation.
    """
    baselines: dict[str, dict[int, float]] = {prefix: {} for prefix in prefixes}
    rows = []
    for prefix in prefixes:
        pure = prefix == "RAPM"
        for year in years:
            filename = input_files_for(prefix, [year], "RS", existing_only=True)
            if not filename:
                continue
            path = rapm_solver.PROCESSED_DIR / filename[0]
            frame = pd.read_parquet(path)
            frame["season_phase"] = "RS"
            frame["source_season_end_year"] = normalize_season_end_year(year)
            prepared = rapm_solver.detect_file_type_and_prepare(frame, pure=pure, prefix=prefix)
            values = pd.to_numeric(prepared["Off_Diff"], errors="coerce")
            baseline = float(values.mean())
            season_end_year = normalize_season_end_year(year)
            baselines[prefix][season_end_year] = baseline
            rows.append({
                "metric": prefix,
                "season": season_end_year,
                "source_file": filename[0],
                "rows": int(values.notna().sum()),
                "baseline_per_possession": baseline,
                "baseline_per_100": baseline * 100.0,
            })
    return baselines, pd.DataFrame(rows).sort_values(["metric", "season"]).reset_index(drop=True)


def load_age_lookup(darko_history_path: Path, max_year: int) -> dict[str, dict[int, float]]:
    age_source = rapm_solver.fetch_age_source_darko(max_year, darko_history_path)
    lookup: dict[str, dict[int, float]] = defaultdict(dict)
    for row in age_source.itertuples(index=False):
        player_id = str(int(row.nba_id))
        year = int(row.year)
        age = float(row.Age)
        lookup[player_id][year] = age
    return dict(lookup)


def lookup_player_age(age_lookup: dict[str, dict[int, float]], player_id: str, season_end_year: int) -> float | None:
    player_ages = age_lookup.get(str(player_id))
    if not player_ages:
        return None
    if season_end_year in player_ages:
        return player_ages[season_end_year]
    fallback_years = [year for year in player_ages if year <= season_end_year]
    if not fallback_years:
        return None
    return player_ages[max(fallback_years)]


def build_metric_exposures(prefix: str, years: list[int], season_type: str) -> dict[str, dict[str, dict[int, int]]]:
    exposures: dict[str, dict[str, dict[int, int]]] = {
        "off": defaultdict(lambda: defaultdict(int)),
        "def": defaultdict(lambda: defaultdict(int)),
    }
    columns = [f"O{i}" for i in range(1, 6)] + [f"D{i}" for i in range(1, 6)]
    for filename in input_files_for(prefix, years, season_type, existing_only=(season_type == "ALL")):
        path = rapm_solver.PROCESSED_DIR / filename
        season_end_year = normalize_season_end_year(parse_year_from_processed_filename(filename))
        frame = pd.read_parquet(path, columns=columns)
        for side, side_cols in [("off", [f"O{i}" for i in range(1, 6)]), ("def", [f"D{i}" for i in range(1, 6)])]:
            present_cols = [col for col in side_cols if col in frame.columns]
            if not present_cols:
                continue
            values = pd.Series(frame[present_cols].to_numpy().ravel())
            values = pd.to_numeric(values, errors="coerce").dropna().astype(int)
            values = values[values > 0]
            counts = values.astype(str).value_counts()
            for player_id, count in counts.items():
                exposures[side][player_id][season_end_year] += int(count)
    return {
        side: {player_id: dict(season_counts) for player_id, season_counts in side_lookup.items()}
        for side, side_lookup in exposures.items()
    }


def evaluate_age_adjustment(
    curve: np.poly1d,
    player_season_counts: dict[int, int],
    age_lookup: dict[str, dict[int, float]],
    player_id: str,
    *,
    min_age: float = 18.0,
    max_age: float = 38.0,
) -> tuple[float, int, int]:
    weighted_values = []
    weights = []
    missing_weight = 0
    for season_end_year, count in player_season_counts.items():
        age = lookup_player_age(age_lookup, player_id, season_end_year)
        if age is None:
            missing_weight += int(count)
            continue
        clipped_age = min(max(float(age), min_age), max_age)
        weighted_values.append(float(curve(clipped_age)))
        weights.append(int(count))
    if not weights:
        return 0.0, 0, missing_weight
    return float(np.average(weighted_values, weights=weights)), int(sum(weights)), int(missing_weight)


def apply_polynomial_age_curves(
    results: dict[str, pd.DataFrame],
    years: list[int],
    season_type: str,
    output_dir: Path,
    age_poly_coefficients: Path,
    darko_history_path: Path,
) -> tuple[dict[str, pd.DataFrame], int]:
    prefixes = list(results)
    curves, degree = load_age_curve_coefficients(age_poly_coefficients, prefixes)
    age_lookup = load_age_lookup(darko_history_path, max(normalize_season_end_year(year) for year in years))
    rows = []
    adjusted_results: dict[str, pd.DataFrame] = {}

    for prefix, result in results.items():
        exposures = build_metric_exposures(prefix, years, season_type)
        adjusted = result.copy()
        adjusted["player_id"] = adjusted["player_id"].astype(str)

        side_adjustments: dict[str, dict[str, float]] = {"off": {}, "def": {}}
        side_weights: dict[str, dict[str, int]] = {"off": {}, "def": {}}
        side_missing: dict[str, dict[str, int]] = {"off": {}, "def": {}}

        for side in ("off", "def"):
            curve = curves[(prefix, side)]
            for player_id in adjusted["player_id"]:
                value, used_weight, missing_weight = evaluate_age_adjustment(
                    curve,
                    exposures[side].get(player_id, {}),
                    age_lookup,
                    player_id,
                )
                side_adjustments[side][player_id] = value
                side_weights[side][player_id] = used_weight
                side_missing[side][player_id] = missing_weight

            weights = np.array([side_weights[side][pid] for pid in adjusted["player_id"]], dtype=float)
            values = np.array([side_adjustments[side][pid] for pid in adjusted["player_id"]], dtype=float)
            valid = weights > 0
            center = float(np.average(values[valid], weights=weights[valid])) if valid.any() else 0.0
            for player_id in adjusted["player_id"]:
                side_adjustments[side][player_id] -= center

        adjusted["off_age_poly"] = adjusted["player_id"].map(side_adjustments["off"]).fillna(0.0)
        adjusted["def_age_poly"] = adjusted["player_id"].map(side_adjustments["def"]).fillna(0.0)
        adjusted["net_age_poly"] = adjusted["off_age_poly"] - adjusted["def_age_poly"]
        adjusted["off"] = pd.to_numeric(adjusted["off"], errors="coerce") + adjusted["off_age_poly"]
        adjusted["def"] = pd.to_numeric(adjusted["def"], errors="coerce") + adjusted["def_age_poly"]
        adjusted["net_rapm"] = adjusted["off"] - adjusted["def"]
        for col in ["off", "def", "net_rapm", "off_age_poly", "def_age_poly", "net_age_poly"]:
            adjusted[col] = adjusted[col].round(2)
        adjusted_results[prefix] = adjusted

        for player_id in adjusted["player_id"]:
            rows.append(
                {
                    "metric": prefix,
                    "player_id": player_id,
                    "player_name": adjusted.loc[adjusted["player_id"] == player_id, "player_name"].iloc[0],
                    "off_age_poly": side_adjustments["off"].get(player_id, 0.0),
                    "def_age_poly": side_adjustments["def"].get(player_id, 0.0),
                    "net_age_poly": side_adjustments["off"].get(player_id, 0.0) - side_adjustments["def"].get(player_id, 0.0),
                    "off_age_poly_weight": side_weights["off"].get(player_id, 0),
                    "def_age_poly_weight": side_weights["def"].get(player_id, 0),
                    "off_missing_age_weight": side_missing["off"].get(player_id, 0),
                    "def_missing_age_weight": side_missing["def"].get(player_id, 0),
                }
            )

    adjustments = pd.DataFrame(rows)
    adjustments.to_csv(output_dir / f"age_poly{degree}_player_adjustments.csv", index=False)
    return adjusted_results, degree


def save_regression_coefficients(results, output_path: Path) -> None:
    rows = [
        {
            "variable": "R_squared",
            "coefficient": results.rsquared,
            "std_error": np.nan,
            "p_value": np.nan,
            "conf_int_lower": np.nan,
            "conf_int_upper": np.nan,
        },
        {
            "variable": "Adjusted_R_squared",
            "coefficient": results.rsquared_adj,
            "std_error": np.nan,
            "p_value": np.nan,
            "conf_int_lower": np.nan,
            "conf_int_upper": np.nan,
        },
        {
            "variable": "Intercept",
            "coefficient": results.params["const"],
            "std_error": results.bse["const"],
            "p_value": results.pvalues["const"],
            "conf_int_lower": results.conf_int().loc["const", 0],
            "conf_int_upper": results.conf_int().loc["const", 1],
        },
    ]
    for col in [idx for idx in results.params.index if idx != "const"]:
        rows.append(
            {
                "variable": col,
                "coefficient": results.params[col],
                "std_error": results.bse[col],
                "p_value": results.pvalues[col],
                "conf_int_lower": results.conf_int().loc[col, 0],
                "conf_int_upper": results.conf_int().loc[col, 1],
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def build_weighted_factors(
    results: dict[str, pd.DataFrame],
    output_dir: Path,
    label: str,
    end_year: int,
    season_type: str,
    *,
    rubberband: bool,
    season_effects: bool,
    fixed_season_effects: bool,
    age_poly_degree: int | None,
) -> Path:
    rapm_df = results["RAPM"]
    merge_cols = ["player_id", "player_name", "off", "def", "net_rapm", "possessions"]
    if "off_poss" in rapm_df.columns:
        merge_cols.append("off_poss")
    if "def_poss" in rapm_df.columns:
        merge_cols.append("def_poss")

    merged = rapm_df[merge_cols].copy()
    merged.rename(columns={"off": "off_rapm", "def": "def_rapm"}, inplace=True)
    for prefix, off_name, def_name in [
        ("TS", "off_ts", "def_ts"),
        ("TOV", "off_tov", "def_tov"),
        ("REB", "off_reb", "def_reb"),
        ("BADPASS_TOV", "off_badpass_tov", "def_badpass_tov"),
        ("SCORING_TOV", "off_scoring_tov", "def_scoring_tov"),
    ]:
        subset = results[prefix][["player_id", "off", "def"]].rename(columns={"off": off_name, "def": def_name})
        merged = merged.merge(subset, on="player_id", how="inner")
        logging.info("Merged %s: %d players remain", prefix, len(merged))

    feature_cols = ["off_ts", "def_ts", "off_tov", "def_tov", "off_reb", "def_reb"]
    regression_results = {}
    for target_col, target_name in [("net_rapm", "net"), ("off_rapm", "off"), ("def_rapm", "def")]:
        model = sm.WLS(merged[target_col], sm.add_constant(merged[feature_cols]), weights=merged["possessions"]).fit()
        regression_results[target_col] = model
        logging.info("%s core regression R^2 = %.6f", target_name, model.rsquared)
        save_regression_coefficients(model, output_dir / f"regression_{target_name}_coefficients.csv")

    tov_decomp = {}
    for side, tov_col, bp_col, sc_col in [
        ("off", "off_tov", "off_badpass_tov", "off_scoring_tov"),
        ("def", "def_tov", "def_badpass_tov", "def_scoring_tov"),
    ]:
        model = sm.WLS(merged[tov_col], sm.add_constant(merged[[bp_col, sc_col]]), weights=merged["possessions"]).fit()
        tov_decomp[side] = model
        logging.info("%s TOV decomposition R^2 = %.6f", side, model.rsquared)
    tov_rows = []
    for side, model in tov_decomp.items():
        for var in model.params.index:
            tov_rows.append({"side": side, "variable": var, "coefficient": model.params[var], "std_error": model.bse[var], "p_value": model.pvalues[var]})
    pd.DataFrame(tov_rows).to_csv(output_dir / "tov_decomposition_coefficients.csv", index=False)

    net_model = regression_results["net_rapm"]
    weighted = merged[["player_id", "player_name"]].copy()
    weighted["Latest_Year"] = normalize_season_end_year(end_year)
    weighted["oTS"] = (merged["off_ts"] * net_model.params["off_ts"]).round(2)
    weighted["oFT"] = pd.NA
    weighted["oTOV"] = (merged["off_tov"] * net_model.params["off_tov"]).round(2)
    weighted["oTOV_bp"] = (merged["off_badpass_tov"] * tov_decomp["off"].params["off_badpass_tov"] * net_model.params["off_tov"]).round(2)
    weighted["oTOV_sc"] = (merged["off_scoring_tov"] * tov_decomp["off"].params["off_scoring_tov"] * net_model.params["off_tov"]).round(2)
    weighted["oREB"] = (merged["off_reb"] * net_model.params["off_reb"]).round(2)
    weighted["dTS"] = (merged["def_ts"] * net_model.params["def_ts"]).round(2)
    weighted["dFT"] = pd.NA
    weighted["dTOV"] = (merged["def_tov"] * net_model.params["def_tov"]).round(2)
    weighted["dTOV_bp"] = (merged["def_badpass_tov"] * tov_decomp["def"].params["def_badpass_tov"] * net_model.params["def_tov"]).round(2)
    weighted["dTOV_sc"] = (merged["def_scoring_tov"] * tov_decomp["def"].params["def_scoring_tov"] * net_model.params["def_tov"]).round(2)
    weighted["dREB"] = (merged["def_reb"] * net_model.params["def_reb"]).round(2)
    weighted["off"] = merged["off_rapm"].round(2)
    weighted["def"] = (-merged["def_rapm"]).round(2)
    weighted["net_rapm"] = merged["net_rapm"].round(2)
    weighted["RESID"] = (weighted["net_rapm"] - weighted[["oTS", "oTOV", "oREB", "dTS", "dTOV", "dREB"]].sum(axis=1)).round(2)
    weighted["o_sc"] = pd.NA
    weighted["o_fc"] = pd.NA
    weighted["d_sc"] = pd.NA
    weighted["d_fc"] = pd.NA
    weighted["o_pval"] = (weighted["oTOV"] + weighted["oREB"]).round(2)
    weighted["d_pval"] = (weighted["dTOV"] + weighted["dREB"]).round(2)
    weighted["possessions"] = merged["possessions"]
    if "off_poss" in merged.columns:
        weighted["off_poss"] = merged["off_poss"]
    if "def_poss" in merged.columns:
        weighted["def_poss"] = merged["def_poss"]

    column_order = [
        "player_id", "player_name", "Latest_Year", "oTS", "oFT", "oTOV", "oTOV_bp", "oTOV_sc", "oREB",
        "dTS", "dFT", "dTOV", "dTOV_bp", "dTOV_sc", "dREB", "off", "def", "net_rapm", "RESID",
        "o_sc", "o_fc", "d_sc", "d_fc", "o_pval", "d_pval", "possessions",
    ]
    if "off_poss" in weighted.columns:
        column_order.append("off_poss")
    if "def_poss" in weighted.columns:
        column_order.append("def_poss")

    weighted = weighted[column_order].sort_values("net_rapm", ascending=False)
    filename = f"weighted_factors_core_{label}_{build_run_suffix(season_type, rubberband, season_effects, fixed_season_effects, age_poly_degree)}.csv"
    output_path = output_dir / filename
    weighted.to_csv(output_path, index=False)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    weighted.to_csv(RESULTS_DIR / filename, index=False)
    MASTER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_path, MASTER_RESULTS_DIR / filename)
    logging.info("Saved weighted factors to %s", output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run historical core weighted factors.")
    parser.add_argument("start_year", type=int)
    parser.add_argument("end_year", type=int)
    parser.add_argument("season_type", choices=["RS", "PS", "ALL"], default="RS")
    parser.add_argument("--cores", type=int, default=14)
    parser.add_argument("--rubberband", "-rb", action="store_true", help="Enable rubberband adjustment in each core metric solve.")
    parser.add_argument("--season-effects", "-se", action="store_true", default=False, help="Enable estimated season dummy effects.")
    parser.add_argument("--no-season-effects", dest="season_effects", action="store_false", help="Disable season fixed effects.")
    parser.add_argument(
        "--fixed-season-effects",
        action="store_true",
        help="Subtract fixed RS raw metric means by season before solving. The RS value is also used for playoff rows.",
    )
    parser.add_argument(
        "--age-poly-coefficients",
        type=Path,
        default=None,
        help="Optional polynomial coefficient CSV from fit_age_effect_curves.py to apply fixed age offsets before solving.",
    )
    parser.add_argument(
        "--darko-history",
        type=Path,
        default=rapm_solver.DEFAULT_DARKO_HISTORY_PATH,
        help="DARKO dpm_history.csv path used only for player-season ages when --age-poly-coefficients is set.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    years = expand_cli_year_range(args.start_year, args.end_year)
    label = year_range_label(args.start_year, args.end_year)
    prefixes = ["RAPM", "TS", "TOV", "REB", "BADPASS_TOV", "SCORING_TOV"]
    validate_inputs(prefixes, years, args.season_type)

    age_poly_degree = None
    if args.age_poly_coefficients:
        _, age_poly_degree = load_age_curve_coefficients(args.age_poly_coefficients, prefixes)

    fixed_season_baselines = None
    baseline_df = None
    if args.fixed_season_effects:
        fixed_season_baselines, baseline_df = compute_fixed_rs_season_baselines(prefixes, years)

    output_dir_suffix = build_run_suffix(
        args.season_type,
        args.rubberband,
        args.season_effects,
        args.fixed_season_effects,
        age_poly_degree,
    )
    output_dir = RESULTS_DIR / f"historical_core_{label}_{output_dir_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if baseline_df is not None:
        baseline_path = output_dir / "fixed_rs_season_baselines.csv"
        baseline_df.to_csv(baseline_path, index=False)
        logging.info("Saved fixed RS season baselines to %s", baseline_path)

    results = {}
    for prefix in prefixes:
        results[prefix] = run_metric(
            prefix,
            years,
            args.season_type,
            label,
            output_dir,
            args.cores,
            rubberband=args.rubberband,
            season_effects=args.season_effects,
            fixed_season_baselines=fixed_season_baselines[prefix] if fixed_season_baselines is not None else None,
            age_poly_coefficients=args.age_poly_coefficients,
            age_poly_degree=age_poly_degree,
            darko_history_path=args.darko_history,
        )

    output_path = build_weighted_factors(
        results,
        output_dir,
        label,
        args.end_year,
        args.season_type,
        rubberband=args.rubberband,
        season_effects=args.season_effects,
        fixed_season_effects=args.fixed_season_effects,
        age_poly_degree=age_poly_degree,
    )
    logging.info("Complete: %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
