#!/usr/bin/env python3
"""Build validated adaptive ridge penalties for one-season young players.

The age-27 career parent has a real uncertainty seam: a player observed in one
young season receives the full fixed population age translation even though
their player coefficient is weakly identified. This script keeps the prior
mean and age curve unchanged and adjusts only prior precision.

Penalty multipliers are selected separately for offense and defense on
historical first-to-second-season transitions. Development uses feature
seasons through 2021; 2022-2025 feature seasons are report-only stress. The
annual diagnostic approximates the incremental generalized-ridge attenuation
as ``(possessions + alpha) / (possessions + multiplier * alpha)``. The final
career solve applies the selected multipliers exactly inside generalized ridge.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PANEL_COLUMNS = {
    "player_id",
    "player_name",
    "season",
    "orapm",
    "drapm",
    "off_poss",
    "def_poss",
    "age",
}
CURVE_COLUMNS = {"age", "orapm_age_effect", "drapm_age_effect"}


def _string_ids(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.astype("Int64").astype(str)


def prepare_age27_panel(panel: pd.DataFrame, curve: pd.DataFrame) -> pd.DataFrame:
    missing_panel = PANEL_COLUMNS - set(panel.columns)
    missing_curve = CURVE_COLUMNS - set(curve.columns)
    if missing_panel:
        raise ValueError(f"Annual panel missing columns: {sorted(missing_panel)}")
    if missing_curve:
        raise ValueError(f"Age curve missing columns: {sorted(missing_curve)}")

    frame = panel.copy()
    frame["player_id"] = _string_ids(frame["player_id"])
    for column in ["season", "orapm", "drapm", "off_poss", "def_poss", "age"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(
        subset=["player_id", "season", "orapm", "drapm", "off_poss", "def_poss", "age"]
    )
    frame = frame[(frame["off_poss"] > 0) & (frame["def_poss"] > 0)].copy()

    age_curve = curve.copy()
    for column in CURVE_COLUMNS:
        age_curve[column] = pd.to_numeric(age_curve[column], errors="coerce")
    age_curve = age_curve.dropna(subset=list(CURVE_COLUMNS)).sort_values("age")
    if age_curve.empty or age_curve["age"].duplicated().any():
        raise ValueError("Age curve must contain unique numeric ages.")

    ages = age_curve["age"].to_numpy(dtype=float)
    clipped = frame["age"].clip(float(ages.min()), float(ages.max()))
    frame["orapm_age27"] = frame["orapm"] - np.interp(
        clipped,
        ages,
        age_curve["orapm_age_effect"].to_numpy(dtype=float),
    )
    frame["drapm_age27"] = frame["drapm"] - np.interp(
        clipped,
        ages,
        age_curve["drapm_age_effect"].to_numpy(dtype=float),
    )
    return frame.sort_values(["player_id", "season"]).reset_index(drop=True)


def build_first_to_second_transitions(
    panel: pd.DataFrame,
    *,
    start_year: int = 1997,
    young_age_max: float = 22.5,
) -> pd.DataFrame:
    frame = panel.sort_values(["player_id", "season"]).copy()
    frame["appearance"] = frame.groupby("player_id").cumcount() + 1
    next_rows = frame.groupby("player_id").shift(-1)
    eligible = (
        frame["appearance"].eq(1)
        & frame["season"].gt(int(start_year))
        & frame["age"].le(float(young_age_max))
        & next_rows["season"].eq(frame["season"] + 1)
    )
    first = frame.loc[eligible].copy()
    second = next_rows.loc[eligible].copy()
    rows = pd.DataFrame(
        {
            "player_id": first["player_id"].to_numpy(),
            "player_name": first["player_name"].to_numpy(),
            "feature_season": first["season"].astype(int).to_numpy(),
            "target_season": second["season"].astype(int).to_numpy(),
            "feature_age": first["age"].to_numpy(dtype=float),
            "feature_orapm_age27": first["orapm_age27"].to_numpy(dtype=float),
            "feature_drapm_age27": first["drapm_age27"].to_numpy(dtype=float),
            "feature_off_poss": first["off_poss"].to_numpy(dtype=float),
            "feature_def_poss": first["def_poss"].to_numpy(dtype=float),
            "target_orapm_age27": second["orapm_age27"].to_numpy(dtype=float),
            "target_drapm_age27": second["drapm_age27"].to_numpy(dtype=float),
            "target_off_poss": second["off_poss"].to_numpy(dtype=float),
            "target_def_poss": second["def_poss"].to_numpy(dtype=float),
        }
    )
    if rows.empty:
        raise ValueError("No eligible first-to-second-season transitions.")
    return rows


def evaluate_multiplier_grid(
    transitions: pd.DataFrame,
    multipliers: list[float],
    *,
    off_alpha: float = 2000.0,
    def_alpha: float = 3000.0,
    development_end_year: int = 2021,
) -> pd.DataFrame:
    results: list[dict[str, Any]] = []
    split_masks = {
        "development": transitions["feature_season"] <= int(development_end_year),
        "stress": transitions["feature_season"] > int(development_end_year),
    }
    side_specs = {
        "off": ("orapm", float(off_alpha)),
        "def": ("drapm", float(def_alpha)),
    }
    for split, split_mask in split_masks.items():
        subset = transitions.loc[split_mask].copy()
        if subset.empty:
            continue
        for side, (rating_name, alpha) in side_specs.items():
            feature = subset[f"feature_{rating_name}_age27"].to_numpy(dtype=float)
            target = subset[f"target_{rating_name}_age27"].to_numpy(dtype=float)
            feature_poss = subset[f"feature_{side}_poss"].to_numpy(dtype=float)
            target_poss = subset[f"target_{side}_poss"].to_numpy(dtype=float)
            for multiplier in multipliers:
                attenuation = (feature_poss + alpha) / (feature_poss + float(multiplier) * alpha)
                squared_error = (attenuation * feature - target) ** 2
                season_scores = []
                for feature_season in sorted(subset["feature_season"].unique()):
                    season_mask = subset["feature_season"].to_numpy() == feature_season
                    season_scores.append(
                        float(np.average(squared_error[season_mask], weights=target_poss[season_mask]))
                    )
                results.append(
                    {
                        "split": split,
                        "side": side,
                        "multiplier": float(multiplier),
                        "transitions": int(len(subset)),
                        "feature_seasons": int(subset["feature_season"].nunique()),
                        "mse": float(np.mean(season_scores)),
                        "season_mse_se": (
                            float(np.std(season_scores, ddof=1) / np.sqrt(len(season_scores)))
                            if len(season_scores) > 1
                            else float("nan")
                        ),
                    }
                )
    output = pd.DataFrame(results)
    for (split, side), indices in output.groupby(["split", "side"]).groups.items():
        group = output.loc[indices]
        baseline = float(group.loc[group["multiplier"] == 1.0, "mse"].iloc[0])
        output.loc[indices, "mse_gain_vs_multiplier1"] = baseline - group["mse"]
    return output.sort_values(["split", "side", "multiplier"]).reset_index(drop=True)


def select_development_multipliers(grid: pd.DataFrame) -> tuple[float, float]:
    selected: dict[str, float] = {}
    for side in ["off", "def"]:
        rows = grid[(grid["split"] == "development") & (grid["side"] == side)]
        if rows.empty:
            raise ValueError(f"No development rows for {side} multiplier selection.")
        best = rows.sort_values(["mse", "multiplier"], ascending=[True, False]).iloc[0]
        selected[side] = float(best["multiplier"])
    return selected["off"], selected["def"]


def build_current_multiplier_audit(
    panel: pd.DataFrame,
    *,
    off_multiplier: float,
    def_multiplier: float,
    reference_age: float = 27.0,
    young_age_max: float = 22.5,
    min_reference_distance: float = 4.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = panel.groupby("player_id", as_index=False).agg(
        player_name=("player_name", "last"),
        seasons=("season", "nunique"),
        first_season=("season", "min"),
        last_season=("season", "max"),
        first_age=("age", "first"),
        last_age=("age", "last"),
        nearest_age_to_reference=("age", lambda values: float((values - reference_age).abs().min())),
        career_off_poss=("off_poss", "sum"),
        career_def_poss=("def_poss", "sum"),
    )
    grouped["eligible"] = (
        grouped["seasons"].eq(1)
        & grouped["first_age"].le(float(young_age_max))
        & grouped["nearest_age_to_reference"].gt(float(min_reference_distance))
    )
    grouped["off_multiplier"] = np.where(grouped["eligible"], off_multiplier, 1.0)
    grouped["def_multiplier"] = np.where(grouped["eligible"], def_multiplier, 1.0)
    grouped["base_off_alpha"] = 2000.0
    grouped["base_def_alpha"] = 3000.0
    grouped["effective_off_alpha"] = grouped["base_off_alpha"] * grouped["off_multiplier"]
    grouped["effective_def_alpha"] = grouped["base_def_alpha"] * grouped["def_multiplier"]
    multipliers = grouped.loc[
        grouped["eligible"], ["player_id", "off_multiplier", "def_multiplier"]
    ].copy()
    return multipliers, grouped.sort_values(
        ["eligible", "last_season", "player_name"], ascending=[False, False, True]
    )


def _multiplier_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annual-panel", type=Path, required=True)
    parser.add_argument("--age-curve", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-year", type=int, default=1997)
    parser.add_argument("--development-end-year", type=int, default=2021)
    parser.add_argument("--young-age-max", type=float, default=22.5)
    parser.add_argument("--min-reference-distance", type=float, default=4.0)
    parser.add_argument("--off-alpha", type=float, default=2000.0)
    parser.add_argument("--def-alpha", type=float, default=3000.0)
    parser.add_argument("--grid", default="1,1.5,2,3,4,6,8,12")
    args = parser.parse_args()

    multipliers = [float(value) for value in args.grid.split(",")]
    if 1.0 not in multipliers or any(value < 1.0 for value in multipliers):
        raise ValueError("Multiplier grid must contain 1.0 and no value below 1.0.")
    panel = prepare_age27_panel(pd.read_csv(args.annual_panel), pd.read_csv(args.age_curve))
    transitions = build_first_to_second_transitions(
        panel,
        start_year=args.start_year,
        young_age_max=args.young_age_max,
    )
    validation = evaluate_multiplier_grid(
        transitions,
        multipliers,
        off_alpha=args.off_alpha,
        def_alpha=args.def_alpha,
        development_end_year=args.development_end_year,
    )
    selected_off, selected_def = select_development_multipliers(validation)
    multiplier_file, audit = build_current_multiplier_audit(
        panel,
        off_multiplier=selected_off,
        def_multiplier=selected_def,
        young_age_max=args.young_age_max,
        min_reference_distance=args.min_reference_distance,
    )

    label = f"young1s_o{_multiplier_label(selected_off)}_d{_multiplier_label(selected_def)}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    multiplier_path = args.output_dir / f"adaptive_penalties_{label}.csv"
    audit_path = args.output_dir / f"adaptive_penalties_{label}_audit.csv"
    validation_path = args.output_dir / "adaptive_young_penalty_validation.csv"
    receipt_path = args.output_dir / f"adaptive_penalties_{label}_receipt.json"
    multiplier_file.to_csv(multiplier_path, index=False)
    audit.to_csv(audit_path, index=False)
    validation.to_csv(validation_path, index=False)

    def score(split: str, side: str, multiplier: float) -> float:
        row = validation[
            (validation["split"] == split)
            & (validation["side"] == side)
            & (validation["multiplier"] == multiplier)
        ]
        return float(row["mse"].iloc[0]) if not row.empty else float("nan")

    receipt = {
        "model": "adaptive_young_player_generalized_ridge",
        "eligibility": "exactly one observed season, age <= young_age_max, and nearest observed age more than min_reference_distance from 27",
        "prior_mean": "unchanged; negative-LAKER prior when available, otherwise zero",
        "selected_off_multiplier": selected_off,
        "selected_def_multiplier": selected_def,
        "base_off_alpha": float(args.off_alpha),
        "base_def_alpha": float(args.def_alpha),
        "effective_off_alpha": float(args.off_alpha * selected_off),
        "effective_def_alpha": float(args.def_alpha * selected_def),
        "young_age_max": float(args.young_age_max),
        "min_reference_distance": float(args.min_reference_distance),
        "development_feature_seasons": f"{args.start_year + 1}-{args.development_end_year}",
        "stress_feature_seasons": f"{args.development_end_year + 1}-{int(transitions['feature_season'].max())}",
        "transitions": int(len(transitions)),
        "development_transitions": int((transitions["feature_season"] <= args.development_end_year).sum()),
        "stress_transitions": int((transitions["feature_season"] > args.development_end_year).sum()),
        "eligible_current_players": int(audit["eligible"].sum()),
        "development_off_mse_gain_vs_base": score("development", "off", 1.0) - score("development", "off", selected_off),
        "development_def_mse_gain_vs_base": score("development", "def", 1.0) - score("development", "def", selected_def),
        "stress_off_mse_gain_vs_base": score("stress", "off", 1.0) - score("stress", "off", selected_off),
        "stress_def_mse_gain_vs_base": score("stress", "def", 1.0) - score("stress", "def", selected_def),
        "validation_scope": "annual age-27 RAPM attenuation proxy; final career solve applies exact generalized-ridge penalties",
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({
        "multiplier_file": str(multiplier_path),
        "audit": str(audit_path),
        "validation": str(validation_path),
        "receipt": str(receipt_path),
        **receipt,
    }, indent=2))


if __name__ == "__main__":
    main()
