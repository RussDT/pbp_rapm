#!/usr/bin/env python3
"""Build and forward-test compact box-feature priors for defensive STOP RAPM.

For each outer possession fold, a nested player-season ridge maps trailing
three-year box features to next-season relative personal STOP%.  The mapped
expectation becomes the defensive center in the STOP lineup solve.  Mapper
training targets are restricted to seasons available before the outer
validation season.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import tune_stop_personal_prior as personal


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = PIPELINE_ROOT.parent.parent
DEFAULT_BOXSCORE = RESEARCH_ROOT / "vpm" / "boxscore.csv"
RESULTS_DIR = PIPELINE_ROOT / "results"

RATE_COLUMNS = [
    "STOPS_100",
    "dfga/100",
    "rimdfga/100",
    "v2_defended_at_rim_fga_per100",
    "d_Steals_Per100",
    "d_OFFD_Per100",
    "d_Blocks_Per100",
    "block_recov_percent",
    "d_DEFLECTIONS_Per100",
    "v2_deflections_per100",
    "v2_contested_shots_2pt_per100",
    "v2_contested_shots_3pt_per100",
]

PERSONAL_FEATURES = [
    "source_relative_stop_pct",
    "source_relative_stop_rel500",
    "source_relative_stop_rel1500",
    "source_relative_stop_rel3000",
    "log_source_def_poss",
]
ASSIGNMENT_FEATURES = [
    "rel_dfga/100",
    "rel_rimdfga/100",
    "rel_v2_defended_at_rim_fga_per100",
]
MECHANISM_FEATURES = [
    "rel_d_Steals_Per100",
    "rel_d_OFFD_Per100",
    "rel_d_Blocks_Per100",
    "rel_block_recov_percent",
]
ACTIVITY_FEATURES = [
    "rel_d_DEFLECTIONS_Per100",
    "rel_v2_deflections_per100",
    "rel_v2_contested_shots_2pt_per100",
    "rel_v2_contested_shots_3pt_per100",
]

FEATURE_SPECS = {
    "personal": (PERSONAL_FEATURES, []),
    "personal_position": (PERSONAL_FEATURES, ["Pos2_filled"]),
    "personal_role": (PERSONAL_FEATURES, ["Defensive_Role_filled"]),
    "personal_context": (
        PERSONAL_FEATURES,
        ["Pos2_filled", "Defensive_Role_filled"],
    ),
    "personal_context_assignment": (
        PERSONAL_FEATURES + ASSIGNMENT_FEATURES,
        ["Pos2_filled", "Defensive_Role_filled"],
    ),
    "personal_context_assignment_mechanism": (
        PERSONAL_FEATURES + ASSIGNMENT_FEATURES + MECHANISM_FEATURES,
        ["Pos2_filled", "Defensive_Role_filled"],
    ),
    "full_activity": (
        PERSONAL_FEATURES + ASSIGNMENT_FEATURES + MECHANISM_FEATURES + ACTIVITY_FEATURES,
        ["Pos2_filled", "Defensive_Role_filled"],
    ),
}


def load_box_history(path: Path) -> pd.DataFrame:
    usecols = [
        "nba_id",
        "Name",
        "year",
        "playoffs",
        "DefPoss",
        "Pos2",
        "Defensive Role",
        *RATE_COLUMNS,
    ]
    box = pd.read_csv(path, usecols=lambda column: column in set(usecols), low_memory=False)
    required = {"nba_id", "Name", "year", "DefPoss", "STOPS_100", "Pos2", "Defensive Role"}
    missing = required - set(box.columns)
    if missing:
        raise ValueError(f"boxscore source missing columns: {sorted(missing)}")
    if "playoffs" in box.columns:
        box = box[pd.to_numeric(box["playoffs"], errors="coerce").fillna(0).eq(0)].copy()
    for column in ["nba_id", "year", "DefPoss", *RATE_COLUMNS]:
        if column in box.columns:
            box[column] = pd.to_numeric(box[column], errors="coerce")
    box = box.dropna(subset=["nba_id", "year", "DefPoss", "STOPS_100"])
    box = box[box["DefPoss"] > 0].copy()
    box["nba_id"] = box["nba_id"].astype(int)
    box["year"] = box["year"].astype(int)
    box["credited_stops"] = box["STOPS_100"] * box["DefPoss"] / 100.0
    if box.duplicated(["nba_id", "year"]).any():
        raise ValueError("boxscore source has duplicate regular-season player-years")
    return box.sort_values(["nba_id", "year"]).reset_index(drop=True)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return np.nan
    return float(np.average(values[mask].astype(float), weights=weights[mask].astype(float)))


def build_trailing_feature_panel(box: pd.DataFrame, source_end: int) -> pd.DataFrame:
    source_start = int(source_end) - 2
    selected = box[box["year"].between(source_start, int(source_end))].copy()
    if selected.empty:
        raise ValueError(f"no box rows for {source_start}-{source_end}")

    league_stop = _weighted_mean(selected["STOPS_100"], selected["DefPoss"])
    rows = []
    for player_id, group in selected.groupby("nba_id", sort=False):
        ordered = group.sort_values("year")
        row = {
            "nba_id": int(player_id),
            "player_name": str(ordered.iloc[-1]["Name"]),
            "source_start": source_start,
            "source_end": int(source_end),
            "source_def_poss": float(ordered["DefPoss"].sum()),
        }
        for column in RATE_COLUMNS:
            if column not in ordered.columns:
                row[column] = np.nan
                continue
            if column == "block_recov_percent" and "d_Blocks_Per100" in ordered.columns:
                weights = ordered["DefPoss"] * ordered["d_Blocks_Per100"].fillna(0) / 100.0
            else:
                weights = ordered["DefPoss"]
            row[column] = _weighted_mean(ordered[column], weights)
        for source, output in [("Pos2", "Pos2_filled"), ("Defensive Role", "Defensive_Role_filled")]:
            known = ordered[source].dropna().astype(str)
            row[output] = known.iloc[-1] if not known.empty else "Unknown"
        rows.append(row)

    panel = pd.DataFrame(rows)
    panel["source_relative_stop_pct"] = panel["STOPS_100"] - league_stop
    panel["log_source_def_poss"] = np.log1p(panel["source_def_poss"])
    for k in [500.0, 1500.0, 3000.0]:
        panel[f"source_relative_stop_rel{int(k)}"] = (
            panel["source_relative_stop_pct"] * panel["source_def_poss"]
            / (panel["source_def_poss"] + k)
        )

    for column in RATE_COLUMNS:
        if column == "STOPS_100" or column not in panel.columns:
            continue
        mean = _weighted_mean(panel[column], panel["source_def_poss"])
        panel[f"rel_{column}"] = panel[column] - mean
    return panel


def build_mapper_examples(box: pd.DataFrame, first_source_end=2016, last_target_year=2026) -> pd.DataFrame:
    annual = box.copy()
    annual["target_league_stop_pct"] = annual["year"].map(
        annual.groupby("year").apply(
            lambda group: _weighted_mean(group["STOPS_100"], group["DefPoss"]),
            include_groups=False,
        )
    )
    annual["target_relative_stop_pct"] = annual["STOPS_100"] - annual["target_league_stop_pct"]
    target = annual[["nba_id", "year", "target_relative_stop_pct", "DefPoss"]].rename(
        columns={"year": "target_year", "DefPoss": "target_def_poss"}
    )
    examples = []
    for source_end in range(int(first_source_end), int(last_target_year)):
        panel = build_trailing_feature_panel(box, source_end)
        panel["target_year"] = source_end + 1
        examples.append(panel.merge(target, on=["nba_id", "target_year"], how="inner", validate="one_to_one"))
    return pd.concat(examples, ignore_index=True)


def make_mapper(numeric_features, categorical_features, alpha):
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    try:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse=False)
    categorical = Pipeline(
        [("impute", SimpleImputer(strategy="most_frequent")), ("onehot", one_hot)]
    )
    transform = ColumnTransformer(
        [("numeric", numeric, numeric_features), ("categorical", categorical, categorical_features)],
        remainder="drop",
    )
    return Pipeline([("features", transform), ("ridge", Ridge(alpha=float(alpha)))])


def select_mapper_alpha(train: pd.DataFrame, numeric, categorical, grid=(0.1, 1.0, 10.0, 100.0, 1000.0)):
    groups = train["target_year"].astype(int).to_numpy()
    splits = min(5, len(np.unique(groups)))
    if splits < 2:
        return float(grid[0]), pd.DataFrame()
    rows = []
    for alpha in grid:
        fold_mse = []
        for fold, (fit_idx, val_idx) in enumerate(GroupKFold(n_splits=splits).split(train, groups=groups), 1):
            fit = train.iloc[fit_idx]
            val = train.iloc[val_idx]
            model = make_mapper(numeric, categorical, alpha)
            weights = np.minimum(fit["target_def_poss"].to_numpy(float), 3000.0)
            model.fit(
                fit[numeric + categorical],
                fit["target_relative_stop_pct"],
                ridge__sample_weight=weights,
            )
            prediction = model.predict(val[numeric + categorical])
            score = mean_squared_error(
                val["target_relative_stop_pct"],
                prediction,
                sample_weight=np.minimum(val["target_def_poss"].to_numpy(float), 3000.0),
            )
            fold_mse.append(float(score))
            rows.append({"alpha": alpha, "fold": fold, "mse": score})
        rows[-1]["mean_mse_for_alpha"] = float(np.mean(fold_mse))
    receipt = pd.DataFrame(rows)
    means = receipt.groupby("alpha")["mse"].mean()
    return float(means.idxmin()), receipt


def fit_feature_prior(examples, prediction_panel, max_target_year, spec_name):
    numeric, categorical = FEATURE_SPECS[spec_name]
    train = examples[examples["target_year"] <= int(max_target_year)].copy()
    alpha, cv = select_mapper_alpha(train, numeric, categorical)
    model = make_mapper(numeric, categorical, alpha)
    weights = np.minimum(train["target_def_poss"].to_numpy(float), 3000.0)
    model.fit(
        train[numeric + categorical],
        train["target_relative_stop_pct"],
        ridge__sample_weight=weights,
    )
    output = prediction_panel[["nba_id", "player_name", "source_def_poss"]].copy()
    output["predicted_relative_stop_pct"] = model.predict(prediction_panel[numeric + categorical])
    output["prior_value"] = -output["predicted_relative_stop_pct"]
    output["mapper_alpha"] = alpha
    output["spec"] = spec_name
    return output, cv


def attach_prior(base_fold, prior_frame, train_frame, scale=1.0):
    all_players = base_fold["players"]
    raw = dict(zip(prior_frame["nba_id"].astype(str), prior_frame["prior_value"] * float(scale) / 100.0))
    shell = pd.DataFrame({"nba_id": [int(player) for player in raw], "prior_value": [value * 100 for value in raw.values()]})
    centered, offset, coverage = personal.center_prior_on_training_defense(shell, train_frame, all_players)
    fold = dict(base_fold)
    fold["def_prior"] = centered
    fold["prior_offset"] = offset
    fold["coverage"] = coverage
    return fold


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boxscore", type=Path, default=DEFAULT_BOXSCORE)
    parser.add_argument("--alpha-grid", default=personal.DEFAULT_GRID)
    parser.add_argument("--scale-grid", default="0.25,0.5,0.75,1.0,1.25")
    parser.add_argument("--receipt", type=Path, default=RESULTS_DIR / "stop_feature_prior_forward_cv.csv")
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=RESULTS_DIR / "stop_feature_prior_model_comparison.csv",
    )
    parser.add_argument("--prior-output", type=Path, default=RESULTS_DIR / "stop_feature_prior_24_26.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.getLogger().setLevel(logging.WARNING)
    box = load_box_history(args.boxscore)
    examples = build_mapper_examples(box)
    fold_specs = [(2021, 2023, 2024), (2022, 2024, 2025), (2023, 2025, 2026)]
    base_folds = []
    train_frames = []
    panels = []
    for start, end, validation in fold_specs:
        base = personal.prepare_fold(box, start, end, validation)
        train_frame = personal.load_stop_window(start, end)
        panel = build_trailing_feature_panel(box, end)
        base_folds.append(base)
        train_frames.append(train_frame)
        panels.append(panel)

    receipt_rows = []
    spec_predictions = {}
    for spec_name in FEATURE_SPECS:
        folds = []
        spec_predictions[spec_name] = []
        for (start, end, validation), base, train_frame, panel in zip(
            fold_specs, base_folds, train_frames, panels
        ):
            prior, mapper_cv = fit_feature_prior(examples, panel, end, spec_name)
            spec_predictions[spec_name].append(prior)
            fold = attach_prior(base, prior, train_frame, scale=1.0)
            folds.append(fold)
            for row in personal.evaluate_pair([fold], 3000.0, 5000.0, use_prior=True):
                row.update(
                    {
                        "stage": "feature",
                        "spec": spec_name,
                        "prior_scale": 1.0,
                        "mapper_alpha": float(prior["mapper_alpha"].iloc[0]),
                    }
                )
                receipt_rows.append(row)
        mean_mse = np.mean([row["mse"] for row in receipt_rows if row.get("stage") == "feature" and row.get("spec") == spec_name])
        print(f"feature {spec_name}: mean future MSE={mean_mse:.10f}")

    feature_receipt = pd.DataFrame([row for row in receipt_rows if row["stage"] == "feature"])
    best_spec = feature_receipt.groupby("spec")["mse"].mean().idxmin()
    print(f"selected feature spec: {best_spec}")

    scale_grid = [float(value) for value in args.scale_grid.split(",") if value.strip()]
    scale_scores = {}
    for scale in scale_grid:
        folds = [
            attach_prior(base, prior, train, scale=scale)
            for base, prior, train in zip(base_folds, spec_predictions[best_spec], train_frames)
        ]
        rows = personal.evaluate_pair(folds, 3000.0, 5000.0, use_prior=True)
        for row in rows:
            row.update({"stage": "scale", "spec": best_spec, "prior_scale": scale})
        receipt_rows.extend(rows)
        scale_scores[scale] = float(np.mean([row["mse"] for row in rows]))
        print(f"scale {scale:g}: mean future MSE={scale_scores[scale]:.10f}")
    best_scale = min(scale_scores, key=scale_scores.get)

    selected_folds = [
        attach_prior(base, prior, train, scale=best_scale)
        for base, prior, train in zip(base_folds, spec_predictions[best_spec], train_frames)
    ]
    best_off, best_def, alpha_receipt = personal.tune_alphas(
        selected_folds, personal.parse_grid(args.alpha_grid), passes=2, seed_off=3000.0, seed_def=5000.0
    )
    alpha_receipt["stage"] = "alpha"
    alpha_receipt["spec"] = best_spec
    alpha_receipt["prior_scale"] = best_scale
    receipt_rows.extend(alpha_receipt.to_dict("records"))
    benchmark = personal.evaluate_pair(selected_folds, best_off, best_def, use_prior=False)
    for row in benchmark:
        row.update({"stage": "same_alpha_benchmark", "spec": "zero_prior", "prior_scale": 0.0})
    receipt_rows.extend(benchmark)

    zero_folds = []
    for base in base_folds:
        fold = dict(base)
        fold["def_prior"] = {player: 0.0 for player in base["players"]}
        zero_folds.append(fold)
    zero_off, zero_def, zero_alpha_receipt = personal.tune_alphas(
        zero_folds,
        personal.parse_grid(args.alpha_grid),
        passes=2,
        seed_off=3000.0,
        seed_def=5000.0,
    )
    zero_alpha_receipt["stage"] = "zero_alpha"
    zero_alpha_receipt["spec"] = "zero_prior"
    zero_alpha_receipt["prior_scale"] = 0.0
    receipt_rows.extend(zero_alpha_receipt.to_dict("records"))
    optimized_zero = personal.evaluate_pair(zero_folds, zero_off, zero_def, use_prior=True)

    personal_priors = spec_predictions["personal"]
    personal_scale_scores = {}
    for scale in scale_grid:
        folds = [
            attach_prior(base, prior, train, scale=scale)
            for base, prior, train in zip(base_folds, personal_priors, train_frames)
        ]
        rows = personal.evaluate_pair(folds, 3000.0, 5000.0, use_prior=True)
        personal_scale_scores[scale] = float(np.mean([row["mse"] for row in rows]))
    personal_scale = min(personal_scale_scores, key=personal_scale_scores.get)
    personal_folds = [
        attach_prior(base, prior, train, scale=personal_scale)
        for base, prior, train in zip(base_folds, personal_priors, train_frames)
    ]
    personal_off, personal_def, personal_alpha_receipt = personal.tune_alphas(
        personal_folds,
        personal.parse_grid(args.alpha_grid),
        passes=2,
        seed_off=3000.0,
        seed_def=5000.0,
    )
    personal_alpha_receipt["stage"] = "personal_alpha"
    personal_alpha_receipt["spec"] = "personal"
    personal_alpha_receipt["prior_scale"] = personal_scale
    receipt_rows.extend(personal_alpha_receipt.to_dict("records"))
    optimized_personal = personal.evaluate_pair(
        personal_folds, personal_off, personal_def, use_prior=True
    )

    selected_rows = personal.evaluate_pair(selected_folds, best_off, best_def, use_prior=True)
    comparison_rows = []
    for model, spec, scale, off_alpha, def_alpha, rows in [
        ("optimized_zero_prior", "zero_prior", 0.0, zero_off, zero_def, optimized_zero),
        (
            "optimized_personal_mapper",
            "personal",
            personal_scale,
            personal_off,
            personal_def,
            optimized_personal,
        ),
        ("selected_feature_prior", best_spec, best_scale, best_off, best_def, selected_rows),
    ]:
        by_year = {int(row["validation_year"]): float(row["mse"]) for row in rows}
        comparison_rows.append(
            {
                "model": model,
                "spec": spec,
                "prior_scale": scale,
                "off_alpha": off_alpha,
                "def_alpha": def_alpha,
                "mse_2024": by_year[2024],
                "mse_2025": by_year[2025],
                "mse_2026": by_year[2026],
                "mean_mse": float(np.mean(list(by_year.values()))),
            }
        )
    args.comparison_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparison_rows).to_csv(args.comparison_output, index=False)

    receipt = pd.DataFrame(receipt_rows)
    receipt["selected_spec"] = best_spec
    receipt["selected_scale"] = best_scale
    receipt["selected_off_alpha"] = best_off
    receipt["selected_def_alpha"] = best_def
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.to_csv(args.receipt, index=False)

    final_panel = build_trailing_feature_panel(box, 2026)
    final_prior, _ = fit_feature_prior(examples, final_panel, 2026, best_spec)
    final_frame = personal.load_stop_window(2024, 2026)
    final_players = personal._all_players(final_frame)
    raw = dict(zip(final_prior["nba_id"].astype(str), final_prior["prior_value"] * best_scale / 100.0))
    shell = pd.DataFrame({"nba_id": [int(player) for player in raw], "prior_value": [value * 100 for value in raw.values()]})
    centered, offset, coverage = personal.center_prior_on_training_defense(shell, final_frame, final_players)
    source = final_prior.set_index(final_prior["nba_id"].astype(str))
    output = []
    for player in final_players:
        row = source.loc[player] if player in source.index else None
        output.append(
            {
                "player_id": player,
                "player_name": row["player_name"] if row is not None else f"ID_{player}",
                "prior_value": centered[player] * 100.0,
                "mapped_relative_stop_pct": row["predicted_relative_stop_pct"] if row is not None else np.nan,
                "source_def_poss": row["source_def_poss"] if row is not None else 0.0,
                "spec": best_spec,
                "prior_scale": best_scale,
                "mapper_alpha": row["mapper_alpha"] if row is not None else np.nan,
                "prior_center_offset_per_100": offset * 100.0,
                "selected_off_alpha": best_off,
                "selected_def_alpha": best_def,
                "source_start_year": 2024,
                "source_end_year": 2026,
            }
        )
    pd.DataFrame(output).to_csv(args.prior_output, index=False)

    print(f"selected scale={best_scale:g}; alphas={best_off:g}/{best_def:g}")
    print(f"selected prior mean future MSE={np.mean([row['mse'] for row in selected_rows]):.10f}")
    print(f"same-alpha zero-prior mean future MSE={np.mean([row['mse'] for row in benchmark]):.10f}")
    print(f"final prior coverage={coverage:.3%}")
    print(f"wrote {args.receipt}")
    print(f"wrote {args.comparison_output}")
    print(f"wrote {args.prior_output}")


if __name__ == "__main__":
    main()
